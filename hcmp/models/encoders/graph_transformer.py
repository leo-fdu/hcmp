"""Edge-aware graph transformer encoder for HCMP."""

from __future__ import annotations

try:
    import math
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.models requires torch.") from exc

from hcmp.models.encoders.edge_gine import EncoderOutput
from hcmp.models.grouped_ops import grouped_softmax
from hcmp.models.pooling import NodeEdgeAttentionPooling


class GraphTransformerEncoder(nn.Module):
    """Dependency-light graph transformer that updates nodes and edges.

    The encoder keeps one edge row per RDKit bond. Each layer builds two
    directed attention messages per bond internally, but returns the same
    undirected edge row layout used by the graph builder and cut head.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        pooling: nn.Module | None = None,
        attention_impl: str = "loop",
        pooling_impl: str = "loop",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if attention_impl not in {"loop", "vectorized"}:
            raise ValueError("attention_impl must be either 'loop' or 'vectorized'.")
        self.node_in = nn.Linear(node_input_dim, hidden_dim)
        self.edge_in = nn.Linear(edge_input_dim, hidden_dim)
        self.attention_impl = attention_impl
        self.layers = nn.ModuleList(
            [
                EdgeAwareTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_impl=attention_impl,
                )
                for _ in range(num_layers)
            ]
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


class EdgeAwareTransformerLayer(nn.Module):
    """Sparse neighbor attention plus explicit edge update."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        attention_impl: str = "loop",
    ) -> None:
        super().__init__()
        if attention_impl not in {"loop", "vectorized"}:
            raise ValueError("attention_impl must be either 'loop' or 'vectorized'.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_impl = attention_impl
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_attention_bias = nn.Linear(hidden_dim, num_heads)
        self.node_out = nn.Linear(hidden_dim, hidden_dim)
        self.node_norm_1 = nn.LayerNorm(hidden_dim)
        self.node_norm_2 = nn.LayerNorm(hidden_dim)
        self.edge_norm_1 = nn.LayerNorm(hidden_dim)
        self.edge_norm_2 = nn.LayerNorm(hidden_dim)
        self.node_ffn = _ffn(hidden_dim, dropout)
        self.edge_ffn = _ffn(hidden_dim, dropout)
        self.edge_update = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_embeddings.shape[0]
        num_edges = edge_embeddings.shape[0]
        if num_edges == 0:
            node_context = torch.zeros_like(node_embeddings)
            node_hidden = self.node_norm_1(node_embeddings + self.dropout(node_context))
            node_hidden = self.node_norm_2(node_hidden + self.dropout(self.node_ffn(node_hidden)))
            return node_hidden, edge_embeddings

        begin = edge_index[0]
        end = edge_index[1]
        directed_src = torch.cat([begin, end], dim=0)
        directed_dst = torch.cat([end, begin], dim=0)
        directed_edge = torch.cat(
            [
                torch.arange(num_edges, device=edge_embeddings.device),
                torch.arange(num_edges, device=edge_embeddings.device),
            ],
            dim=0,
        )

        q = self.query(node_embeddings).view(num_nodes, self.num_heads, self.head_dim)
        k = self.key(node_embeddings).view(num_nodes, self.num_heads, self.head_dim)
        v = self.value(node_embeddings).view(num_nodes, self.num_heads, self.head_dim)
        edge_v = self.edge_value(edge_embeddings).view(num_edges, self.num_heads, self.head_dim)
        edge_bias = self.edge_attention_bias(edge_embeddings)

        logits = (
            (q[directed_dst] * k[directed_src]).sum(dim=-1) / math.sqrt(self.head_dim)
            + edge_bias[directed_edge]
        )
        directed_values = v[directed_src] + edge_v[directed_edge]
        if self.attention_impl == "vectorized":
            directed_messages = (
                grouped_softmax(logits, directed_dst, num_groups=num_nodes).unsqueeze(-1)
                * directed_values
            )
        else:
            directed_messages = torch.zeros_like(directed_values)
            for node_idx in range(num_nodes):
                mask = directed_dst == node_idx
                if bool(mask.any()):
                    weights = torch.softmax(logits[mask], dim=0).unsqueeze(-1)
                    directed_messages[mask] = weights * directed_values[mask]

        node_context_heads = torch.zeros(
            (num_nodes, self.num_heads, self.head_dim),
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        node_context_heads.index_add_(0, directed_dst, directed_messages)
        node_context = self.node_out(node_context_heads.reshape(num_nodes, self.hidden_dim))
        node_hidden = self.node_norm_1(node_embeddings + self.dropout(node_context))
        node_hidden = self.node_norm_2(node_hidden + self.dropout(self.node_ffn(node_hidden)))

        forward_context = directed_messages[:num_edges].reshape(num_edges, self.hidden_dim)
        backward_context = directed_messages[num_edges:].reshape(num_edges, self.hidden_dim)
        edge_delta = self.edge_update(
            torch.cat(
                [
                    node_hidden[begin],
                    node_hidden[end],
                    edge_embeddings,
                    forward_context,
                    backward_context,
                ],
                dim=-1,
            )
        )
        edge_hidden = self.edge_norm_1(edge_embeddings + self.dropout(edge_delta))
        edge_hidden = self.edge_norm_2(edge_hidden + self.dropout(self.edge_ffn(edge_hidden)))
        return node_hidden, edge_hidden


def _ffn(hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, 4 * hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(4 * hidden_dim, hidden_dim),
    )
