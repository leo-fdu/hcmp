"""Objective losses for HCMP."""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.training requires torch.") from exc


def bert_loss(
    atom_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    atom_mask_indices: torch.Tensor,
    bond_mask_indices: torch.Tensor,
    atom_type_targets: torch.Tensor,
    bond_type_targets: torch.Tensor,
) -> torch.Tensor:
    """Minimal BERT loss: atom type plus bond type only."""

    losses = []
    if atom_mask_indices.numel() > 0:
        losses.append(F.cross_entropy(atom_logits[atom_mask_indices], atom_type_targets))
    if bond_mask_indices.numel() > 0:
        losses.append(F.cross_entropy(bond_logits[bond_mask_indices], bond_type_targets))
    if not losses:
        return atom_logits.sum() * 0.0
    return sum(losses) / len(losses)


def graph_bert_loss(
    atom_logits: dict[str, torch.Tensor],
    bond_logits: dict[str, torch.Tensor],
    atom_mask_indices: torch.Tensor,
    bond_mask_indices: torch.Tensor,
    atom_targets: dict[str, torch.Tensor] | None,
    bond_targets: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Multi-attribute Graph-BERT reconstruction loss.

    This is separate from HCMP BERT and intentionally averages per attribute
    before adding atom and bond components.
    """

    atom_losses = []
    bond_losses = []
    if atom_targets is not None and atom_mask_indices.numel() > 0:
        for field, logits in atom_logits.items():
            if field in atom_targets:
                atom_losses.append(F.cross_entropy(logits[atom_mask_indices], atom_targets[field][atom_mask_indices]))
    if bond_targets is not None and bond_mask_indices.numel() > 0:
        for field, logits in bond_logits.items():
            if field in bond_targets:
                bond_losses.append(F.cross_entropy(logits[bond_mask_indices], bond_targets[field][bond_mask_indices]))
    zero_source = next(iter(atom_logits.values())) if atom_logits else next(iter(bond_logits.values()))
    if not atom_losses and not bond_losses:
        return zero_source.sum() * 0.0
    atom_loss = torch.stack(atom_losses).mean() if atom_losses else zero_source.sum() * 0.0
    bond_loss = torch.stack(bond_losses).mean() if bond_losses else zero_source.sum() * 0.0
    return atom_loss + bond_loss


def cut_seg_loss(
    cut_logits: torch.Tensor,
    cut_labels: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Binary cross entropy over final-edge cut logits."""

    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(float(pos_weight), dtype=cut_logits.dtype, device=cut_logits.device)
    return F.binary_cross_entropy_with_logits(cut_logits, cut_labels.float(), pos_weight=pos_weight)


def compute_auto_pos_weight(cut_labels: torch.Tensor) -> torch.Tensor:
    """Compute N_non_cut / N_cut with a zero-positive guard."""

    positives = cut_labels.float().sum()
    negatives = cut_labels.numel() - positives
    if positives <= 0:
        return torch.tensor(1.0, dtype=torch.float32, device=cut_labels.device)
    return negatives / positives


def descriptor_margin_ranking_loss(
    descriptor_scores_i: torch.Tensor,
    descriptor_scores_j: torch.Tensor,
    descriptor_indices: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    descriptor_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balanced descriptor ranking loss averaged per descriptor first."""

    losses = []
    for descriptor_idx in descriptor_indices.unique(sorted=True):
        mask = descriptor_indices == descriptor_idx
        diff = descriptor_scores_i[mask, descriptor_idx] - descriptor_scores_j[mask, descriptor_idx]
        desc_loss = F.relu(margin - labels[mask].float() * diff).mean()
        if descriptor_weights is not None:
            desc_loss = desc_loss * descriptor_weights[descriptor_idx]
        losses.append(desc_loss)
    if not losses:
        return descriptor_scores_i.sum() * 0.0
    return sum(losses) / len(losses)


def descriptor_pairwise_logistic_loss(
    descriptor_scores: torch.Tensor,
    descriptor_values: torch.Tensor,
    thresholds: torch.Tensor,
    partners_per_anchor: int = 10,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Sample descriptor pairs inside one batch and average per descriptor first."""

    metrics = {
        "num_sampled_pairs": 0.0,
        "num_valid_descriptor_pairs": 0.0,
        "num_valid_descriptors": 0.0,
    }
    num_molecules, num_descriptors = descriptor_values.shape
    zero = descriptor_scores.sum() * 0.0
    if num_molecules < 2 or partners_per_anchor <= 0:
        return (zero, metrics) if return_metrics else zero

    anchor_indices, partner_indices = _sample_anchor_partners(
        num_molecules,
        partners_per_anchor,
        descriptor_scores.device,
    )
    metrics["num_sampled_pairs"] = float(anchor_indices.numel())
    if anchor_indices.numel() == 0:
        return (zero, metrics) if return_metrics else zero

    thresholds = thresholds.to(device=descriptor_scores.device, dtype=descriptor_scores.dtype)
    values = descriptor_values.to(device=descriptor_scores.device, dtype=descriptor_scores.dtype)
    deltas = values[anchor_indices] - values[partner_indices]
    score_gaps = descriptor_scores[anchor_indices] - descriptor_scores[partner_indices]
    finite = torch.isfinite(deltas) & torch.isfinite(thresholds).unsqueeze(0)
    valid = finite & (deltas.abs() >= thresholds.unsqueeze(0)) & (deltas != 0)
    pair_losses = F.softplus(-torch.sign(deltas) * score_gaps)

    losses = []
    for descriptor_idx in range(num_descriptors):
        descriptor_valid = valid[:, descriptor_idx]
        if bool(descriptor_valid.any()):
            losses.append(pair_losses[descriptor_valid, descriptor_idx].mean())
            metrics["num_valid_descriptor_pairs"] += float(descriptor_valid.sum().detach().cpu())
    metrics["num_valid_descriptors"] = float(len(losses))
    if not losses:
        return (zero, metrics) if return_metrics else zero
    loss = torch.stack(losses).mean()
    return (loss, metrics) if return_metrics else loss


def scaffold_triplet_loss(
    anchor_projection: torch.Tensor,
    positive_projection: torch.Tensor,
    negative_projection: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Triplet loss over scaffold projection distances."""

    d_ap = torch.linalg.vector_norm(anchor_projection - positive_projection, ord=2, dim=-1)
    d_an = torch.linalg.vector_norm(anchor_projection - negative_projection, ord=2, dim=-1)
    return F.relu(margin + d_ap - d_an).mean()


def in_batch_scaffold_triplet_loss(
    scaffold_projection: torch.Tensor,
    global_indices: torch.Tensor,
    scaffold_distance_backend,
    min_distance_gap: float = 0.15,
    margin: float = 0.15,
) -> torch.Tensor:
    """Backward-compatible sampled scaffold triplet loss."""

    loss, _metrics = sampled_in_batch_scaffold_triplet_loss(
        scaffold_projection,
        global_indices,
        scaffold_distance_backend,
        min_distance_gap=min_distance_gap,
        margin=margin,
    )
    return loss


def sampled_in_batch_scaffold_triplet_loss(
    scaffold_projection: torch.Tensor,
    global_indices: torch.Tensor,
    scaffold_distance_backend,
    min_distance_gap: float = 0.15,
    margin: float = 0.15,
    candidate_pairs_per_anchor: int = 10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sample in-batch scaffold triplets using data-side scaffold distances."""

    num_molecules = scaffold_projection.shape[0]
    zero = scaffold_projection.sum() * 0.0
    metrics = {
        "num_sampled_candidate_pairs": 0.0,
        "num_valid_triplets": 0.0,
        "cache_hits": 0.0,
        "cache_misses": 0.0,
    }
    if num_molecules < 3 or candidate_pairs_per_anchor <= 0:
        return zero, metrics

    backend = _coerce_scaffold_distance_backend(scaffold_distance_backend)
    before_stats = _backend_stats(backend)
    global_list = [int(idx) for idx in global_indices.detach().cpu().tolist()]
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    if hasattr(backend, "begin_batch"):
        backend.begin_batch()
    try:
        for anchor_idx in range(num_molecules):
            candidates = _sample_candidate_pairs(
                num_molecules,
                anchor_idx,
                candidate_pairs_per_anchor,
                scaffold_projection.device,
            )
            metrics["num_sampled_candidate_pairs"] += float(len(candidates))
            for first_idx, second_idx in candidates:
                d_first = float(backend.get_distance(global_list[anchor_idx], global_list[first_idx]))
                d_second = float(backend.get_distance(global_list[anchor_idx], global_list[second_idx]))
                if d_first + min_distance_gap < d_second:
                    positive_idx, negative_idx = first_idx, second_idx
                elif d_second + min_distance_gap < d_first:
                    positive_idx, negative_idx = second_idx, first_idx
                else:
                    continue
                anchors.append(anchor_idx)
                positives.append(positive_idx)
                negatives.append(negative_idx)
    finally:
        if hasattr(backend, "end_batch"):
            backend.end_batch()

    after_stats = _backend_stats(backend)
    metrics["cache_hits"] = float(after_stats.get("cache_hits", 0) - before_stats.get("cache_hits", 0))
    metrics["cache_misses"] = float(after_stats.get("cache_misses", 0) - before_stats.get("cache_misses", 0))
    metrics["scaffold_distance_failures"] = float(
        after_stats.get("scaffold_distance_failures", 0)
        - before_stats.get("scaffold_distance_failures", 0)
    )
    metrics["num_valid_triplets"] = float(len(anchors))
    if not anchors:
        return zero, metrics

    anchor_tensor = torch.tensor(anchors, dtype=torch.long, device=scaffold_projection.device)
    positive_tensor = torch.tensor(positives, dtype=torch.long, device=scaffold_projection.device)
    negative_tensor = torch.tensor(negatives, dtype=torch.long, device=scaffold_projection.device)
    model_d_ap = torch.linalg.vector_norm(
        scaffold_projection[anchor_tensor] - scaffold_projection[positive_tensor],
        ord=2,
        dim=-1,
    )
    model_d_an = torch.linalg.vector_norm(
        scaffold_projection[anchor_tensor] - scaffold_projection[negative_tensor],
        ord=2,
        dim=-1,
    )
    return F.relu(margin + model_d_ap - model_d_an).mean(), metrics


def _sample_anchor_partners(
    num_molecules: int,
    partners_per_anchor: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchors = []
    partners = []
    num_partners = min(num_molecules - 1, int(partners_per_anchor))
    for anchor_idx in range(num_molecules):
        available = torch.cat(
            [
                torch.arange(0, anchor_idx, dtype=torch.long, device=device),
                torch.arange(anchor_idx + 1, num_molecules, dtype=torch.long, device=device),
            ]
        )
        chosen = available[torch.randperm(available.numel(), device=device)[:num_partners]]
        anchors.append(torch.full((chosen.numel(),), anchor_idx, dtype=torch.long, device=device))
        partners.append(chosen)
    return torch.cat(anchors), torch.cat(partners)


def _sample_candidate_pairs(
    num_molecules: int,
    anchor_idx: int,
    candidate_pairs_per_anchor: int,
    device: torch.device,
) -> list[tuple[int, int]]:
    available = [idx for idx in range(num_molecules) if idx != anchor_idx]
    possible_pairs = len(available) * (len(available) - 1) // 2
    sample_count = min(possible_pairs, int(candidate_pairs_per_anchor))
    if sample_count <= 0:
        return []
    if sample_count == possible_pairs:
        return [
            (available[i], available[j])
            for i in range(len(available))
            for j in range(i + 1, len(available))
        ]

    pairs: set[tuple[int, int]] = set()
    max_attempts = max(100, sample_count * 20)
    attempts = 0
    while len(pairs) < sample_count and attempts < max_attempts:
        order = torch.randperm(len(available), device=device)[:2].detach().cpu().tolist()
        first = available[int(order[0])]
        second = available[int(order[1])]
        pairs.add((first, second) if first < second else (second, first))
        attempts += 1
    if len(pairs) < sample_count:
        for first_pos in range(len(available)):
            for second_pos in range(first_pos + 1, len(available)):
                pairs.add((available[first_pos], available[second_pos]))
                if len(pairs) >= sample_count:
                    return list(pairs)
    return list(pairs)


class _TensorScaffoldDistanceBackend:
    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix.detach().cpu()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_distance(self, global_i: int, global_j: int) -> float:
        self.cache_hits += 1
        return float(self.matrix[int(global_i), int(global_j)])

    def stats(self) -> dict[str, int]:
        return {"cache_hits": self.cache_hits, "cache_misses": self.cache_misses}


def _coerce_scaffold_distance_backend(scaffold_distance_backend):
    if hasattr(scaffold_distance_backend, "get_distance"):
        return scaffold_distance_backend
    if isinstance(scaffold_distance_backend, torch.Tensor):
        return _TensorScaffoldDistanceBackend(scaffold_distance_backend)
    raise TypeError(
        "scaffold_distance_backend must expose get_distance(i, j) or be a torch.Tensor matrix."
    )


def _backend_stats(backend) -> dict[str, int]:
    if hasattr(backend, "stats"):
        return dict(backend.stats())
    return {
        "cache_hits": int(getattr(backend, "cache_hits", 0)),
        "cache_misses": int(getattr(backend, "cache_misses", 0)),
    }
