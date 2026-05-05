"""Loss balancing strategies."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.training requires torch.") from exc


@dataclass
class LossBalancer:
    """Combine enabled loss terms."""

    method: str = "fixed"
    weights: dict[str, float] | None = None

    def combine(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.method != "fixed":
            raise NotImplementedError(f"Loss balancing method {self.method!r} is not implemented.")
        weights = self.weights or {}
        total: torch.Tensor | None = None
        for name, loss in losses.items():
            weighted = float(weights.get(name, 1.0)) * loss
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No losses provided to combine.")
        return total


def build_loss_balancer(config: dict) -> LossBalancer:
    """Build the configured loss balancer."""

    return LossBalancer(
        method=config.get("method", "fixed"),
        weights=dict(config.get("weights", {})),
    )
