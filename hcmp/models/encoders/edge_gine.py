"""Edge-updating GINE-style encoder for HCMP."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.models requires torch.") from exc

from hcmp.models.pooling import NodeEdgeAttentionPooling


@dataclass
class EncoderOutput:
    """Standard HCMP encoder output."""

    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    graph_embedding: torch.Tensor


class EdgeGINEEncoder(nn.Module):
    """Minimal edge-updating message-passing encoder.

    Edges are represented as one tensor row per RDKit bond. Each layer updates
    node states from endpoint-aware edge messages, then updates edge states from
    both endpoint node states and the previous edge state.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        pooling: nn.Module | None = None,
        pooling_impl: str = "loop",
    ) -> None:
        super().__init__()
        self.node_in = nn.Linear(node_input_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [EdgeGINELayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.pooling = pooling or NodeEdgeAttentionPooling(
            hidden_dim,
            hidden_dim,
            pooling_impl=pooling_impl,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_batch: torch.Tensor | None = None,
        edge_batch: torch.Tensor | None = None,
    ) -> EncoderOutput:
        node_embeddings = self.node_in(node_features)
        edge_embeddings = self.edge_in(edge_features)
        for layer in self.layers:
            node_embeddings, edge_embeddings = layer(
                node_embeddings,
                edge_index,
                edge_embeddings,
            )
        graph_embedding = self.pooling(
            node_embeddings,
            edge_embeddings,
            node_batch=node_batch,
            edge_batch=edge_batch,
        )
        return EncoderOutput(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            graph_embedding=graph_embedding,
        )


class EdgeGINELayer(nn.Module):
    """One node-update plus edge-update block."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.message_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.node_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.edge_mlp = _mlp(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin = edge_index[0]
        end = edge_index[1]

        msg_begin_to_end = self.message_mlp(
            torch.cat([node_embeddings[begin], edge_embeddings], dim=-1)
        )
        msg_end_to_begin = self.message_mlp(
            torch.cat([node_embeddings[end], edge_embeddings], dim=-1)
        )
        aggregated = torch.zeros_like(node_embeddings)
        aggregated.index_add_(0, end, msg_begin_to_end)
        aggregated.index_add_(0, begin, msg_end_to_begin)

        node_update = self.node_mlp(torch.cat([node_embeddings, aggregated], dim=-1))
        new_nodes = self.node_norm(node_embeddings + node_update)

        edge_update = self.edge_mlp(
            torch.cat([new_nodes[begin], new_nodes[end], edge_embeddings], dim=-1)
        )
        new_edges = self.edge_norm(edge_embeddings + edge_update)
        return new_nodes, new_edges


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )
