"""Small grouped tensor operations used by optional vectorized model paths."""

from __future__ import annotations

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.models requires torch.") from exc


def grouped_softmax(
    scores: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: int | None = None,
) -> torch.Tensor:
    """Softmax over rows that share the same group id.

    ``scores`` may be shaped ``[N]`` or ``[N, ...]``. The softmax is always over
    the first dimension within each group, matching the loop implementation
    used by the graph transformer attention and graph pooling modules.
    """

    if scores.shape[0] != group_index.shape[0]:
        raise ValueError("scores and group_index must agree on the first dimension.")
    if scores.numel() == 0:
        return torch.zeros_like(scores)
    if num_groups is None:
        num_groups = int(group_index.max().item()) + 1 if group_index.numel() else 0
    if num_groups <= 0:
        return torch.zeros_like(scores)

    max_values = torch.full(
        (num_groups, *scores.shape[1:]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    if not hasattr(max_values, "scatter_reduce_"):
        return _grouped_softmax_loop(scores, group_index, num_groups)
    scatter_index = _expand_group_index(group_index, scores)
    max_values.scatter_reduce_(0, scatter_index, scores, reduce="amax", include_self=True)

    shifted = scores - max_values[group_index]
    exp_scores = shifted.exp()
    denominators = torch.zeros_like(max_values)
    denominators.index_add_(0, group_index, exp_scores)
    return exp_scores / denominators[group_index].clamp_min(torch.finfo(scores.dtype).tiny)


def _expand_group_index(group_index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return group_index
    shape = (group_index.shape[0],) + (1,) * (values.ndim - 1)
    return group_index.view(shape).expand_as(values)


def _grouped_softmax_loop(
    scores: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    output = torch.zeros_like(scores)
    for group_id in range(num_groups):
        mask = group_index == group_id
        if bool(mask.any()):
            output[mask] = torch.softmax(scores[mask], dim=0)
    return output
