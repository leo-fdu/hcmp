"""Graph pooling modules."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.models requires torch.") from exc

from hcmp.models.grouped_ops import grouped_softmax


class NodeEdgeAttentionPooling(nn.Module):
    """Separate node and edge attention pooling followed by concat fusion."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int | None = None,
        pooling_impl: str = "loop",
    ) -> None:
        super().__init__()
        if pooling_impl not in {"loop", "vectorized"}:
            raise ValueError("pooling_impl must be either 'loop' or 'vectorized'.")
        output_dim = output_dim or hidden_dim
        self.pooling_impl = pooling_impl
        self.node_attention = nn.Linear(hidden_dim, 1)
        self.edge_attention = nn.Linear(hidden_dim, 1)
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        node_batch: torch.Tensor | None = None,
        edge_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_batch = _default_batch(node_embeddings, node_batch)
        edge_batch = _default_batch(edge_embeddings, edge_batch)
        num_graphs = _num_graphs(node_batch, edge_batch)
        pooled_nodes = _attention_pool(
            node_embeddings,
            node_batch,
            self.node_attention,
            num_graphs,
            pooling_impl=self.pooling_impl,
        )
        pooled_edges = _attention_pool(
            edge_embeddings,
            edge_batch,
            self.edge_attention,
            num_graphs,
            pooling_impl=self.pooling_impl,
        )
        return self.fuse(torch.cat([pooled_nodes, pooled_edges], dim=-1))


class NodeEdgeMeanPooling(nn.Module):
    """Mean pooling baseline over final node and edge states."""

    def __init__(self, hidden_dim: int, output_dim: int | None = None) -> None:
        super().__init__()
        output_dim = output_dim or hidden_dim
        self.fuse = nn.Linear(2 * hidden_dim, output_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        node_batch: torch.Tensor | None = None,
        edge_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_batch = _default_batch(node_embeddings, node_batch)
        edge_batch = _default_batch(edge_embeddings, edge_batch)
        num_graphs = _num_graphs(node_batch, edge_batch)
        nodes = _mean_pool(node_embeddings, node_batch, num_graphs)
        edges = _mean_pool(edge_embeddings, edge_batch, num_graphs)
        return self.fuse(torch.cat([nodes, edges], dim=-1))


def _default_batch(values: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
    if batch is not None:
        return batch
    return torch.zeros(values.shape[0], dtype=torch.long, device=values.device)


def _attention_pool(
    values: torch.Tensor,
    batch: torch.Tensor,
    scorer: nn.Module,
    num_graphs: int,
    pooling_impl: str = "loop",
) -> torch.Tensor:
    output = torch.zeros((num_graphs, values.shape[-1]), dtype=values.dtype, device=values.device)
    if values.shape[0] == 0:
        return output
    if pooling_impl == "vectorized":
        scores = scorer(values).squeeze(-1)
        weights = grouped_softmax(scores, batch, num_groups=num_graphs).unsqueeze(-1)
        output.index_add_(0, batch, weights * values)
        return output
    for graph_idx in range(num_graphs):
        mask = batch == graph_idx
        if not bool(mask.any()):
            continue
        graph_values = values[mask]
        weights = torch.softmax(scorer(graph_values).squeeze(-1), dim=0).unsqueeze(-1)
        output[graph_idx] = (weights * graph_values).sum(dim=0)
    return output


def _mean_pool(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    output = torch.zeros((num_graphs, values.shape[-1]), dtype=values.dtype, device=values.device)
    if values.shape[0] == 0:
        return output
    counts = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    output.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=values.dtype))
    return output / counts.clamp_min(1.0).unsqueeze(-1)


def _num_graphs(node_batch: torch.Tensor, edge_batch: torch.Tensor) -> int:
    max_values = []
    if node_batch.numel() > 0:
        max_values.append(int(node_batch.max().item()))
    if edge_batch.numel() > 0:
        max_values.append(int(edge_batch.max().item()))
    return (max(max_values) + 1) if max_values else 1
