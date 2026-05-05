"""Top-level HCMP model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.models requires torch.") from exc

from hcmp.data.descriptors import DESCRIPTOR_NAMES
from hcmp.data.graph_builder import FeatureSpec
from hcmp.models.encoders import EdgeGINEEncoder, GraphTransformerEncoder
from hcmp.models.heads import (
    AtomBERTHead,
    BondBERTHead,
    CutSegHead,
    DescriptorHead,
    GraphBERTAtomHead,
    GraphBERTBondHead,
    ScaffoldProjectionHead,
)


@dataclass
class HCMPForwardOutput:
    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    atom_logits: torch.Tensor | dict[str, torch.Tensor] | None
    bond_logits: torch.Tensor | dict[str, torch.Tensor] | None
    cut_logits: torch.Tensor | None
    descriptor_scores: torch.Tensor | None
    scaffold_projection: torch.Tensor | None


class HCMPModel(nn.Module):
    """Composable HCMP model with enabled objective heads."""

    def __init__(
        self,
        feature_spec: FeatureSpec,
        hidden_dim: int = 128,
        num_layers: int = 4,
        encoder_type: str = "graph_transformer",
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_impl: str = "loop",
        pooling_impl: str = "loop",
        enabled_heads: dict[str, bool] | None = None,
        model_family: str = "hcmp",
    ) -> None:
        super().__init__()
        enabled_heads = enabled_heads or {}
        if model_family not in {"hcmp", "graph_bert"}:
            raise ValueError("model_family must be either 'hcmp' or 'graph_bert'.")
        self.model_family = model_family
        self.encoder_type = encoder_type
        self.encoder = self._build_encoder(
            encoder_type=encoder_type,
            node_input_dim=feature_spec.node_feature_dim,
            edge_input_dim=feature_spec.edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            attention_impl=attention_impl,
            pooling_impl=pooling_impl,
        )
        if model_family == "graph_bert":
            self.atom_head = GraphBERTAtomHead(hidden_dim, feature_spec)
            self.bond_head = GraphBERTBondHead(hidden_dim, feature_spec)
        else:
            self.atom_head = AtomBERTHead(hidden_dim, feature_spec.atom_type_dim - 1)
            self.bond_head = BondBERTHead(hidden_dim, feature_spec.bond_type_dim - 1)
        self.cut_head = CutSegHead(hidden_dim)
        self.descriptor_head = DescriptorHead(hidden_dim, hidden_dim, len(DESCRIPTOR_NAMES))
        self.scaffold_head = ScaffoldProjectionHead(hidden_dim, hidden_dim)
        self.enabled_heads = {
            "bert": enabled_heads.get("bert", True),
            "cut_seg": enabled_heads.get("cut_seg", True),
            "prop_rank": enabled_heads.get("prop_rank", True),
            "scaf_triplet": enabled_heads.get("scaf_triplet", True),
        }

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        node_batch: torch.Tensor | None = None,
        edge_batch: torch.Tensor | None = None,
    ) -> HCMPForwardOutput:
        encoded = self.encoder(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_batch=node_batch,
            edge_batch=edge_batch,
        )
        return HCMPForwardOutput(
            node_embeddings=encoded.node_embeddings,
            edge_embeddings=encoded.edge_embeddings,
            graph_embedding=encoded.graph_embedding,
            atom_logits=self.atom_head(encoded.node_embeddings) if self.enabled_heads["bert"] else None,
            bond_logits=self.bond_head(encoded.edge_embeddings) if self.enabled_heads["bert"] else None,
            cut_logits=self.cut_head(encoded.edge_embeddings) if self.enabled_heads["cut_seg"] else None,
            descriptor_scores=(
                self.descriptor_head(encoded.graph_embedding)
                if self.enabled_heads["prop_rank"]
                else None
            ),
            scaffold_projection=(
                self.scaffold_head(encoded.graph_embedding)
                if self.enabled_heads["scaf_triplet"]
                else None
            ),
        )

    def _build_encoder(
        self,
        encoder_type: str,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        attention_impl: str,
        pooling_impl: str,
    ) -> nn.Module:
        if encoder_type == "edge_gine":
            return EdgeGINEEncoder(
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                pooling_impl=pooling_impl,
            )
        if encoder_type == "graph_transformer":
            return GraphTransformerEncoder(
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                attention_impl=attention_impl,
                pooling_impl=pooling_impl,
            )
        raise ValueError(
            "encoder_type must be either 'graph_transformer' or 'edge_gine', "
            f"got {encoder_type!r}."
        )
