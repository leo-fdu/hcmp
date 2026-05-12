"""Unified-batch multi-task training loop for HCMP."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.training requires torch.") from exc

from hcmp.data.graph_builder import FeatureSpec, GraphBatch, MaskedGraphData
from hcmp.models.hcmp_model import HCMPModel
from hcmp.training.loss_balancing import LossBalancer
from hcmp.training.losses import (
    bert_loss,
    compute_auto_pos_weight,
    cut_seg_loss,
    descriptor_pairwise_logistic_loss,
    graph_bert_loss,
    sampled_in_batch_scaffold_triplet_loss,
)


class HCMPTrainer:
    """Trainer where all objectives are computed from the same graph batch."""

    def __init__(
        self,
        model: HCMPModel,
        optimizer: torch.optim.Optimizer,
        loss_balancer: LossBalancer,
        feature_spec: FeatureSpec,
        cut_pos_weight: str | float = "auto",
        atom_mask_ratio: float = 0.15,
        bond_mask_ratio: float = 0.15,
        descriptor_thresholds: torch.Tensor | None = None,
        scaffold_distance_matrix: torch.Tensor | None = None,
        scaffold_distance_backend: Any | None = None,
        partners_per_anchor: int = 10,
        min_distance_gap: float = 0.15,
        triplet_margin: float = 0.15,
        candidate_pairs_per_anchor: int = 10,
        device: str | torch.device = "cpu",
        scheduler: Any | None = None,
        use_amp: bool = False,
        gradient_clip_norm: float | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_balancer = loss_balancer
        self.feature_spec = feature_spec
        self.cut_pos_weight = cut_pos_weight
        self.atom_mask_ratio = atom_mask_ratio
        self.bond_mask_ratio = bond_mask_ratio
        self.descriptor_thresholds = (
            descriptor_thresholds.to(self.device) if descriptor_thresholds is not None else None
        )
        self.scaffold_distance_matrix = (
            scaffold_distance_matrix.detach().cpu() if scaffold_distance_matrix is not None else None
        )
        self.scaffold_distance_backend = (
            scaffold_distance_backend
            if scaffold_distance_backend is not None
            else self.scaffold_distance_matrix
        )
        self.partners_per_anchor = int(partners_per_anchor)
        self.min_distance_gap = min_distance_gap
        self.triplet_margin = triplet_margin
        self.candidate_pairs_per_anchor = int(candidate_pairs_per_anchor)
        self.scheduler = scheduler
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.gradient_clip_norm = gradient_clip_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.model_family = str(getattr(model, "model_family", "hcmp"))
        self.last_mining_metrics: dict[str, float] = {}

    def train_step(self, masked_graph: MaskedGraphData) -> dict[str, torch.Tensor]:
        """Backward-compatible single-graph step."""

        from hcmp.data.graph_builder import collate_graphs

        batch = collate_graphs([masked_graph.graph])
        losses = self.train_graph_batch(batch)
        return {key: torch.tensor(value) for key, value in losses.items()}

    def train_graph_batch(
        self,
        batch: GraphBatch,
        active_losses: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        """Run one forward pass, compute all available losses, and step once."""

        active_loss_set = set(active_losses) if active_losses is not None else None
        self.model.train()
        batch = move_graph_batch_to_device(batch, self.device)
        masked = _mask_graph_batch(
            batch,
            self.feature_spec,
            atom_mask_ratio=self.atom_mask_ratio,
            bond_mask_ratio=self.bond_mask_ratio,
        )

        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.model(
                masked.node_features,
                masked.edge_index,
                masked.edge_features,
                node_batch=masked.node_batch,
                edge_batch=masked.edge_batch,
            )

        raw_losses: dict[str, torch.Tensor] = {}
        reported = {
            "bert_loss": 0.0,
            "cut_loss": 0.0,
            "cut_seg_loss": 0.0,
            "prop_loss": 0.0,
            "triplet_loss": 0.0,
            "prop_rank_loss": 0.0,
            "scaf_triplet_loss": 0.0,
            "total_loss": 0.0,
        }
        mining_metrics = {
            "num_sampled_pairs": 0.0,
            "num_valid_descriptor_pairs": 0.0,
            "num_valid_descriptors": 0.0,
            "num_sampled_candidate_pairs": 0.0,
            "num_valid_triplets": 0.0,
            "cache_hits": 0.0,
            "cache_misses": 0.0,
            "scaffold_distance_failures": 0.0,
        }

        if (
            _loss_is_active("bert", active_loss_set)
            and output.atom_logits is not None
            and output.bond_logits is not None
        ):
            if self.model_family == "graph_bert":
                raw_losses["bert"] = graph_bert_loss(
                    output.atom_logits,
                    output.bond_logits,
                    masked.atom_mask_indices,
                    masked.bond_mask_indices,
                    masked.graph_bert_atom_targets,
                    masked.graph_bert_bond_targets,
                )
            else:
                raw_losses["bert"] = bert_loss(
                    output.atom_logits,
                    output.bond_logits,
                    masked.atom_mask_indices,
                    masked.bond_mask_indices,
                    masked.atom_type_targets,
                    masked.bond_type_targets,
                )
            reported["bert_loss"] = float(raw_losses["bert"].detach().cpu())

        if (
            _loss_is_active("cut_seg", active_loss_set)
            and
            output.cut_logits is not None
            and masked.cut_labels is not None
            and masked.cut_labels.numel() > 0
        ):
            pos_weight = (
                compute_auto_pos_weight(masked.cut_labels)
                if self.cut_pos_weight == "auto"
                else float(self.cut_pos_weight)
            )
            raw_losses["cut_seg"] = cut_seg_loss(output.cut_logits, masked.cut_labels, pos_weight)
            reported["cut_loss"] = float(raw_losses["cut_seg"].detach().cpu())
            reported["cut_seg_loss"] = reported["cut_loss"]

        if (
            _loss_is_active("prop_rank", active_loss_set)
            and
            output.descriptor_scores is not None
            and masked.descriptor_values is not None
            and self.descriptor_thresholds is not None
        ):
            raw_losses["prop_rank"], prop_metrics = descriptor_pairwise_logistic_loss(
                output.descriptor_scores,
                masked.descriptor_values,
                self.descriptor_thresholds,
                partners_per_anchor=self.partners_per_anchor,
                return_metrics=True,
            )
            reported["prop_loss"] = float(raw_losses["prop_rank"].detach().cpu())
            reported["prop_rank_loss"] = reported["prop_loss"]
            mining_metrics.update(prop_metrics)

        if (
            _loss_is_active("scaf_triplet", active_loss_set)
            and
            output.scaffold_projection is not None
            and self.scaffold_distance_backend is not None
            and masked.global_indices.numel() > 0
        ):
            raw_losses["scaf_triplet"], triplet_metrics = sampled_in_batch_scaffold_triplet_loss(
                output.scaffold_projection,
                masked.global_indices,
                self.scaffold_distance_backend,
                min_distance_gap=self.min_distance_gap,
                margin=self.triplet_margin,
                candidate_pairs_per_anchor=self.candidate_pairs_per_anchor,
            )
            reported["triplet_loss"] = float(raw_losses["scaf_triplet"].detach().cpu())
            reported["scaf_triplet_loss"] = reported["triplet_loss"]
            mining_metrics.update(triplet_metrics)

        if raw_losses:
            raw_total = sum(raw_losses.values())
            weighted_total = self.loss_balancer.combine(raw_losses)
            reported["raw_total_loss"] = float(raw_total.detach().cpu())
            reported["weighted_total_loss"] = float(weighted_total.detach().cpu())
            reported["total_loss"] = reported["weighted_total_loss"]
            if self.use_amp:
                self.scaler.scale(weighted_total).backward()
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.gradient_clip_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                weighted_total.backward()
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.gradient_clip_norm))
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        reported.update(mining_metrics)
        self.last_mining_metrics = mining_metrics
        return reported

    def train_one_epoch(
        self,
        multitask_loader: Any,
        active_losses: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        """Train one epoch over graph batches only."""

        totals: dict[str, float] = {}
        count = 0
        for batch in multitask_loader.iterate_graph():
            losses = self.train_graph_batch(batch, active_losses=active_losses)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
        return {key: (value / count if count else 0.0) for key, value in totals.items()}


def _loss_is_active(name: str, active_losses: set[str] | None) -> bool:
    return active_losses is None or name in active_losses


def move_graph_batch_to_device(batch: GraphBatch, device: str | torch.device) -> GraphBatch:
    """Move tensor fields in a graph batch while preserving Python metadata."""

    device = _resolve_device(device)
    return replace(
        batch,
        node_features=batch.node_features.to(device),
        edge_features=batch.edge_features.to(device),
        edge_index=batch.edge_index.to(device),
        node_batch=batch.node_batch.to(device),
        edge_batch=batch.edge_batch.to(device),
        atom_type_targets=batch.atom_type_targets.to(device),
        bond_type_targets=batch.bond_type_targets.to(device),
        graph_bert_atom_targets=(
            {key: value.to(device) for key, value in batch.graph_bert_atom_targets.items()}
            if batch.graph_bert_atom_targets is not None
            else None
        ),
        graph_bert_bond_targets=(
            {key: value.to(device) for key, value in batch.graph_bert_bond_targets.items()}
            if batch.graph_bert_bond_targets is not None
            else None
        ),
        cut_labels=batch.cut_labels.to(device) if batch.cut_labels is not None else None,
        descriptor_values=(
            batch.descriptor_values.to(device) if batch.descriptor_values is not None else None
        ),
        global_indices=batch.global_indices.to(device),
        source_row_indices=batch.source_row_indices.to(device),
        atom_mask_indices=batch.atom_mask_indices.to(device) if batch.atom_mask_indices is not None else None,
        bond_mask_indices=batch.bond_mask_indices.to(device) if batch.bond_mask_indices is not None else None,
    )


def _move_graph_batch(batch: GraphBatch, device: torch.device) -> GraphBatch:
    return move_graph_batch_to_device(batch, device)


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return resolved


def _mask_graph_batch(
    batch: GraphBatch,
    feature_spec: FeatureSpec,
    atom_mask_ratio: float,
    bond_mask_ratio: float,
) -> GraphBatch:
    node_features = batch.node_features.clone()
    edge_features = batch.edge_features.clone()
    atom_mask_indices = _sample_mask_indices(node_features.shape[0], atom_mask_ratio, node_features.device)
    bond_mask_indices = _sample_mask_indices(edge_features.shape[0], bond_mask_ratio, edge_features.device)
    if atom_mask_indices.numel() > 0:
        if feature_spec.feature_mode == "graph_bert":
            _mask_graph_bert_atom_features(node_features, atom_mask_indices, feature_spec)
        else:
            _set_one_hot_mask(node_features, atom_mask_indices, feature_spec.atom_type_slice, feature_spec.atom_type_mask_index)
            _set_one_hot_mask(node_features, atom_mask_indices, feature_spec.formal_charge_slice, feature_spec.formal_charge_mask_index)
            _set_one_hot_mask(node_features, atom_mask_indices, feature_spec.chirality_slice, feature_spec.chirality_mask_index)
    if bond_mask_indices.numel() > 0:
        if feature_spec.feature_mode == "graph_bert":
            _mask_graph_bert_bond_features(edge_features, bond_mask_indices, feature_spec)
        else:
            _set_one_hot_mask(edge_features, bond_mask_indices, feature_spec.bond_type_slice, feature_spec.bond_type_mask_index)
            _set_one_hot_mask(edge_features, bond_mask_indices, feature_spec.bond_stereo_slice, feature_spec.bond_stereo_mask_index)
    return replace(
        batch,
        node_features=node_features,
        edge_features=edge_features,
        atom_mask_indices=atom_mask_indices,
        bond_mask_indices=bond_mask_indices,
        atom_type_targets=batch.atom_type_targets[atom_mask_indices],
        bond_type_targets=batch.bond_type_targets[bond_mask_indices],
    )


def _sample_mask_indices(num_items: int, ratio: float, device: torch.device) -> torch.Tensor:
    if num_items == 0 or ratio <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    num_mask = max(1, int(round(num_items * ratio)))
    num_mask = min(num_items, num_mask)
    return torch.randperm(num_items, device=device)[:num_mask].sort().values


def _set_one_hot_mask(
    features: torch.Tensor,
    indices: torch.Tensor,
    feature_slice: slice,
    mask_index: int,
) -> None:
    features[indices, feature_slice] = 0.0
    features[indices, feature_slice.start + mask_index] = 1.0


def _mask_graph_bert_atom_features(
    node_features: torch.Tensor,
    indices: torch.Tensor,
    feature_spec: FeatureSpec,
) -> None:
    for feature_slice, mask_index in [
        (feature_spec.atom_type_slice, feature_spec.atom_type_mask_index),
        (feature_spec.formal_charge_slice, feature_spec.formal_charge_mask_index),
        (feature_spec.degree_slice, feature_spec.degree_mask_index),
        (feature_spec.hybridization_slice, feature_spec.hybridization_mask_index),
        (feature_spec.atom_aromaticity_slice, feature_spec.boolean_mask_index),
        (feature_spec.num_hydrogens_slice, feature_spec.num_hydrogens_mask_index),
        (feature_spec.atom_ring_slice, feature_spec.boolean_mask_index),
        (feature_spec.chirality_slice, feature_spec.chirality_mask_index),
    ]:
        _set_one_hot_mask(node_features, indices, feature_slice, mask_index)


def _mask_graph_bert_bond_features(
    edge_features: torch.Tensor,
    indices: torch.Tensor,
    feature_spec: FeatureSpec,
) -> None:
    for feature_slice, mask_index in [
        (feature_spec.bond_type_slice, feature_spec.bond_type_mask_index),
        (feature_spec.bond_conjugation_slice, feature_spec.boolean_mask_index),
        (feature_spec.bond_aromaticity_slice, feature_spec.boolean_mask_index),
        (feature_spec.bond_ring_slice, feature_spec.boolean_mask_index),
        (feature_spec.bond_stereo_slice, feature_spec.bond_stereo_mask_index),
    ]:
        _set_one_hot_mask(edge_features, indices, feature_slice, mask_index)
