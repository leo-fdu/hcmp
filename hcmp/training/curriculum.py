"""Progressive loss activation for HCMP training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


LOSS_TO_REPORTED_KEY = {
    "bert": "bert_loss",
    "cut_seg": "cut_loss",
    "prop_rank": "prop_rank_loss",
    "scaf_triplet": "scaf_triplet_loss",
}


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    active_losses: tuple[str, ...]
    monitor: str
    min_steps: int = 0
    max_steps: int | None = None


@dataclass(frozen=True)
class CurriculumTransition:
    old_phase: str
    new_phase: str
    previous_mean: float
    recent_mean: float
    relative_improvement: float
    relative_worsening: float
    threshold: float
    max_relative_worsening: float
    decision: str


@dataclass(frozen=True)
class ConvergenceDecision:
    previous_mean: float
    recent_mean: float
    relative_improvement: float
    relative_worsening: float
    min_relative_improvement: float
    max_relative_worsening: float
    decision: str

    @property
    def converged(self) -> bool:
        return self.decision == "plateau converged"


class ConvergenceCurriculum:
    """Advance phases when the monitored loss stops improving."""

    def __init__(
        self,
        phases: list[CurriculumPhase],
        window_epochs: int = 20,
        window_points: int | None = None,
        unit: str = "epoch",
        min_phase_steps: int = 0,
        max_phase_steps: int | None = None,
        min_relative_improvement: float = 0.01,
        max_relative_worsening: float = 0.02,
        eps: float = 1.0e-8,
    ) -> None:
        if not phases:
            raise ValueError("Curriculum requires at least one phase.")
        if unit not in {"epoch", "step"}:
            raise ValueError("curriculum.unit must be either 'epoch' or 'step'.")
        points = int(window_points if window_points is not None else window_epochs)
        if points <= 0:
            raise ValueError("curriculum.window_points/window_epochs must be positive.")
        self.phases = [
            CurriculumPhase(
                name=phase.name,
                active_losses=phase.active_losses,
                monitor=phase.monitor,
                min_steps=int(phase.min_steps or min_phase_steps),
                max_steps=phase.max_steps if phase.max_steps is not None else max_phase_steps,
            )
            for phase in phases
        ]
        self.window_epochs = int(window_epochs)
        self.window_points = points
        self.unit = unit
        self.min_phase_steps = int(min_phase_steps or 0)
        self.max_phase_steps = None if max_phase_steps is None else int(max_phase_steps)
        self.min_relative_improvement = float(min_relative_improvement)
        self.max_relative_worsening = float(max_relative_worsening)
        self.eps = float(eps)
        self.phase_index = 0
        self.phase_history: list[float] = []
        self.phase_start_step = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self.phase_index]

    @property
    def epochs_in_phase(self) -> int:
        return len(self.phase_history)

    @property
    def is_final_phase(self) -> bool:
        return self.phase_index >= len(self.phases) - 1

    def observe_epoch(
        self,
        epoch_losses: dict[str, float],
    ) -> CurriculumTransition | None:
        """Record one epoch and return a transition when convergence is reached."""
        return self.observe_point(epoch_losses)

    def observe_point(
        self,
        losses: dict[str, float],
        global_step: int | None = None,
    ) -> CurriculumTransition | None:
        """Record one monitoring point and return a transition when convergence is reached."""

        phase = self.current_phase
        value = _monitor_value(losses, phase.monitor)
        if value is None or not math.isfinite(value):
            print(
                "Curriculum warning: monitored loss is missing or non-finite; "
                f"phase={phase.name}, monitor={phase.monitor}, value={value}."
            )
            return None
        if phase.monitor == "prop_rank" and float(losses.get("num_valid_descriptor_pairs", 0.0)) <= 0.0:
            print(
                "Curriculum warning: prop_rank monitor has no valid descriptor pairs this epoch; "
                "not using it for convergence."
            )
            return None
        if phase.monitor == "scaf_triplet" and float(losses.get("num_valid_triplets", 0.0)) <= 0.0:
            print(
                "Curriculum warning: scaf_triplet monitor has no valid triplets this epoch; "
                "not using it for convergence."
            )
            return None

        self.phase_history.append(float(value))
        if len(self.phase_history) < 2 * self.window_points:
            return None
        if global_step is not None:
            min_steps = int(phase.min_steps or 0)
            if int(global_step) - int(self.phase_start_step) < min_steps:
                return None

        previous = self.phase_history[-2 * self.window_points : -self.window_points]
        recent = self.phase_history[-self.window_points :]
        previous_mean = sum(previous) / len(previous)
        recent_mean = sum(recent) / len(recent)
        decision = evaluate_convergence(
            previous_mean=previous_mean,
            recent_mean=recent_mean,
            min_relative_improvement=self.min_relative_improvement,
            max_relative_worsening=self.max_relative_worsening,
            eps=self.eps,
        )
        print(
            "curriculum_convergence_check "
            f"phase={phase.name} monitor={phase.monitor} "
            f"previous_mean={decision.previous_mean:.6g} "
            f"recent_mean={decision.recent_mean:.6g} "
            f"relative_improvement={decision.relative_improvement:.6g} "
            f"relative_worsening={decision.relative_worsening:.6g} "
            f"min_relative_improvement={decision.min_relative_improvement:.6g} "
            f"max_relative_worsening={decision.max_relative_worsening:.6g} "
            f"decision={decision.decision}"
        )
        if decision.decision == "worsened too much":
            print(
                "Curriculum phase not advanced because monitored loss worsened beyond tolerance."
            )
            return None
        if not decision.converged:
            return None

        old_phase = phase.name
        if self.is_final_phase:
            new_phase = old_phase
        else:
            self.phase_index += 1
            new_phase = self.current_phase.name
            self.phase_history = []
            if global_step is not None:
                self.phase_start_step = int(global_step)
        return CurriculumTransition(
            old_phase=old_phase,
            new_phase=new_phase,
            previous_mean=previous_mean,
            recent_mean=recent_mean,
            relative_improvement=decision.relative_improvement,
            relative_worsening=decision.relative_worsening,
            threshold=self.min_relative_improvement,
            max_relative_worsening=self.max_relative_worsening,
            decision=decision.decision,
        )


class StaticCurriculum:
    """Compatibility wrapper for non-curriculum training."""

    def __init__(self, active_losses: list[str]) -> None:
        self.phase = CurriculumPhase(
            name="static",
            active_losses=tuple(active_losses),
            monitor="none",
        )

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phase

    @property
    def epochs_in_phase(self) -> int:
        return 0

    def observe_epoch(self, epoch_losses: dict[str, float]) -> None:
        return None

    def observe_point(self, losses: dict[str, float], global_step: int | None = None) -> None:
        return None


def build_curriculum(config: dict[str, Any], enabled_losses: dict[str, bool]):
    curriculum_config = config.get("curriculum", {})
    if not enabled_losses.get("bert", False):
        raise ValueError("HCMP pretraining variants require bert=true.")
    phases = _auto_phases(enabled_losses, curriculum_config)
    _validate_phases(phases, enabled_losses)
    print("generated_curriculum=" + _phase_summary(phases))
    if not bool(curriculum_config.get("enabled", False)):
        return StaticCurriculum(
            [name for name in ["bert", "cut_seg", "prop_rank", "scaf_triplet"] if enabled_losses.get(name, False)]
        )
    if curriculum_config.get("mode", "convergence") != "convergence":
        raise ValueError("Only curriculum.mode='convergence' is supported.")
    return ConvergenceCurriculum(
        phases=phases,
        window_epochs=int(curriculum_config.get("window_epochs", 20)),
        window_points=int(curriculum_config.get("window_points", curriculum_config.get("window_epochs", 20))),
        unit=str(curriculum_config.get("unit", "epoch")),
        min_phase_steps=int(curriculum_config.get("min_phase_steps", 0) or 0),
        max_phase_steps=(
            int(curriculum_config["max_phase_steps"])
            if curriculum_config.get("max_phase_steps") is not None
            else None
        ),
        min_relative_improvement=float(curriculum_config.get("min_relative_improvement", 0.01)),
        max_relative_worsening=float(curriculum_config.get("max_relative_worsening", 0.02)),
        eps=float(curriculum_config.get("eps", 1.0e-8)),
    )


def curriculum_to_metadata(curriculum) -> list[dict[str, Any]]:
    return [
        {
            "name": phase.name,
            "active_losses": list(phase.active_losses),
            "monitor": phase.monitor,
            "min_steps": int(getattr(phase, "min_steps", 0) or 0),
            "max_steps": getattr(phase, "max_steps", None),
        }
        for phase in getattr(curriculum, "phases", [curriculum.current_phase])
    ]


def _auto_phases(enabled_losses: dict[str, bool], curriculum_config: dict[str, Any]) -> list[CurriculumPhase]:
    active = ["bert"]
    phases = [CurriculumPhase(name="bert", active_losses=("bert",), monitor="bert")]
    for loss_name, label in [
        ("cut_seg", "cut"),
        ("prop_rank", "prop"),
        ("scaf_triplet", "scaf"),
    ]:
        if enabled_losses.get(loss_name, False):
            active.append(loss_name)
            phases.append(
                CurriculumPhase(
                    name="bert_" + "_".join(_phase_label(loss) for loss in active[1:]),
                    active_losses=tuple(active),
                    monitor=loss_name,
                )
            )
    return phases


def _phase_label(loss_name: str) -> str:
    return {
        "cut_seg": "cut",
        "prop_rank": "prop",
        "scaf_triplet": "scaf",
    }.get(loss_name, loss_name)


def _validate_phases(phases: list[CurriculumPhase], enabled_losses: dict[str, bool]) -> None:
    for phase in phases:
        if phase.monitor not in phase.active_losses:
            raise ValueError(
                f"Curriculum phase {phase.name!r} monitors {phase.monitor!r}, "
                "but that loss is not active in the phase."
            )
        for loss_name in phase.active_losses:
            if not enabled_losses.get(loss_name, False):
                raise ValueError(
                    f"Curriculum phase {phase.name!r} includes disabled loss {loss_name!r}."
                )


def _phase_summary(phases: list[CurriculumPhase]) -> str:
    return json_like(
        [
            {"name": phase.name, "active_losses": list(phase.active_losses), "monitor": phase.monitor}
            for phase in phases
        ]
    )


def json_like(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def evaluate_convergence(
    previous_mean: float,
    recent_mean: float,
    min_relative_improvement: float,
    max_relative_worsening: float,
    eps: float = 1.0e-8,
) -> ConvergenceDecision:
    denom = max(abs(float(previous_mean)), float(eps))
    relative_improvement = (float(previous_mean) - float(recent_mean)) / denom
    relative_worsening = (float(recent_mean) - float(previous_mean)) / denom
    if relative_worsening > max_relative_worsening:
        decision = "worsened too much"
    elif relative_improvement < min_relative_improvement:
        decision = "plateau converged"
    else:
        decision = "still improving"
    return ConvergenceDecision(
        previous_mean=float(previous_mean),
        recent_mean=float(recent_mean),
        relative_improvement=relative_improvement,
        relative_worsening=relative_worsening,
        min_relative_improvement=float(min_relative_improvement),
        max_relative_worsening=float(max_relative_worsening),
        decision=decision,
    )


def _monitor_value(epoch_losses: dict[str, float], monitor: str) -> float | None:
    key = LOSS_TO_REPORTED_KEY.get(monitor, f"{monitor}_loss")
    value = epoch_losses.get(key)
    if value is None and monitor == "prop_rank":
        value = epoch_losses.get("prop_loss")
    if value is None and monitor == "scaf_triplet":
        value = epoch_losses.get("triplet_loss")
    return None if value is None else float(value)


def _default_phases() -> list[dict[str, Any]]:
    return [
        {"name": "bert", "active_losses": ["bert"], "monitor": "bert"},
        {"name": "segmentation", "active_losses": ["bert", "cut_seg"], "monitor": "cut_seg"},
        {
            "name": "property",
            "active_losses": ["bert", "cut_seg", "prop_rank"],
            "monitor": "prop_rank",
        },
        {
            "name": "distance",
            "active_losses": ["bert", "cut_seg", "prop_rank", "scaf_triplet"],
            "monitor": "scaf_triplet",
        },
    ]
