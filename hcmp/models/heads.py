"""Prediction heads for HCMP objectives."""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.models requires torch.") from exc


class AtomBERTHead(nn.Module):
    """Predict masked atomic number only."""

    def __init__(self, hidden_dim: int, num_atom_types: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_atom_types)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(node_embeddings)


class BondBERTHead(nn.Module):
    """Predict masked bond type only."""

    def __init__(self, hidden_dim: int, num_bond_types: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_bond_types)

    def forward(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(edge_embeddings)


class GraphBERTAtomHead(nn.Module):
    """Predict all conventional atom attributes for the Graph-BERT baseline."""

    def __init__(self, hidden_dim: int, feature_spec) -> None:
        super().__init__()
        self.classifiers = nn.ModuleDict(
            {
                "atomic_number": nn.Linear(hidden_dim, feature_spec.atom_type_dim - 1),
                "formal_charge": nn.Linear(hidden_dim, feature_spec.formal_charge_dim - 1),
                "degree": nn.Linear(hidden_dim, feature_spec.degree_dim - 1),
                "hybridization": nn.Linear(hidden_dim, feature_spec.hybridization_dim - 1),
                "aromaticity": nn.Linear(hidden_dim, feature_spec.boolean_dim - 1),
                "num_hydrogens": nn.Linear(hidden_dim, feature_spec.num_hydrogens_dim - 1),
                "ring_membership": nn.Linear(hidden_dim, feature_spec.boolean_dim - 1),
                "chirality": nn.Linear(hidden_dim, feature_spec.chirality_dim - 1),
            }
        )

    def forward(self, node_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        return {field: classifier(node_embeddings) for field, classifier in self.classifiers.items()}


class GraphBERTBondHead(nn.Module):
    """Predict all conventional bond attributes for the Graph-BERT baseline."""

    def __init__(self, hidden_dim: int, feature_spec) -> None:
        super().__init__()
        self.classifiers = nn.ModuleDict(
            {
                "bond_type": nn.Linear(hidden_dim, feature_spec.bond_type_dim - 1),
                "conjugation": nn.Linear(hidden_dim, feature_spec.boolean_dim - 1),
                "aromaticity": nn.Linear(hidden_dim, feature_spec.boolean_dim - 1),
                "ring_membership": nn.Linear(hidden_dim, feature_spec.boolean_dim - 1),
                "stereo": nn.Linear(hidden_dim, feature_spec.bond_stereo_dim - 1),
            }
        )

    def forward(self, edge_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        return {field: classifier(edge_embeddings) for field, classifier in self.classifiers.items()}


class CutSegHead(nn.Module):
    """Cut-bond head consuming final edge embeddings directly."""

    input_representation = "final_edge_embedding"

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_embeddings).squeeze(-1)


class DescriptorHead(nn.Module):
    """Shared multi-output descriptor scoring head."""

    def __init__(self, graph_dim: int, hidden_dim: int, num_descriptors: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_descriptors),
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(graph_embedding)


class ScaffoldProjectionHead(nn.Module):
    """Projection head for scaffold triplet ranking."""

    def __init__(
        self,
        graph_dim: int,
        hidden_dim: int,
        projection_dim: int | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        projection_dim = projection_dim or hidden_dim
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        projection = self.net(graph_embedding)
        if self.normalize:
            projection = F.normalize(projection, p=2, dim=-1)
        return projection
