"""Descriptor threshold estimation and balanced pair generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hcmp.data.descriptors import DESCRIPTOR_NAMES


def estimate_descriptor_thresholds(
    descriptor_values: pd.DataFrame,
    quantile: float = 0.70,
    max_pairs_per_descriptor: int | None = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Estimate one absolute-delta threshold per descriptor."""

    rows = []
    for descriptor_name in DESCRIPTOR_NAMES:
        pairs = _all_descriptor_pairs(
            descriptor_values,
            descriptor_name,
            threshold=None,
            max_pairs=max_pairs_per_descriptor,
            random_seed=random_seed,
        )
        abs_deltas = pairs["abs_delta"].to_numpy(dtype=float)
        threshold = float(np.quantile(abs_deltas, quantile)) if abs_deltas.size else 0.0
        rows.append(
            {
                "descriptor_name": descriptor_name,
                "threshold_quantile": quantile,
                "threshold_value": threshold,
                "num_pairs_sampled": int(len(pairs)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "descriptor_name",
            "threshold_quantile",
            "threshold_value",
            "num_pairs_sampled",
        ],
    )


def build_balanced_descriptor_pairs(
    descriptor_values: pd.DataFrame,
    thresholds: pd.DataFrame,
    max_pairs_per_descriptor: int | None = None,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Build top-difference pairs independently for each descriptor."""

    threshold_by_name = dict(
        zip(thresholds["descriptor_name"], thresholds["threshold_value"], strict=False)
    )
    frames = []
    for descriptor_name in DESCRIPTOR_NAMES:
        threshold = float(threshold_by_name[descriptor_name])
        pairs = _all_descriptor_pairs(
            descriptor_values,
            descriptor_name,
            threshold=threshold,
            max_pairs=max_pairs_per_descriptor,
            random_seed=random_seed,
        )
        frames.append(pairs)
    if not frames:
        return pd.DataFrame(columns=_pair_columns())
    return pd.concat(frames, ignore_index=True)[_pair_columns()]


def _all_descriptor_pairs(
    descriptor_values: pd.DataFrame,
    descriptor_name: str,
    threshold: float | None,
    max_pairs: int | None,
    random_seed: int,
) -> pd.DataFrame:
    valid = descriptor_values[descriptor_values["status"] == "success"][
        ["mol_id", descriptor_name]
    ].dropna()
    mol_ids = valid["mol_id"].to_numpy()
    values = valid[descriptor_name].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for i in range(len(values)):
        deltas = values[i] - values[i + 1 :]
        for offset, delta in enumerate(deltas, start=i + 1):
            abs_delta = abs(float(delta))
            if threshold is not None and abs_delta < threshold:
                continue
            rows.append(
                {
                    "descriptor_name": descriptor_name,
                    "mol_i": mol_ids[i],
                    "mol_j": mol_ids[offset],
                    "delta": float(delta),
                    "abs_delta": abs_delta,
                    "threshold": 0.0 if threshold is None else float(threshold),
                    "label": 1 if delta > 0 else -1,
                }
            )
    frame = pd.DataFrame(rows, columns=_pair_columns())
    if max_pairs is not None and len(frame) > max_pairs:
        frame = frame.sample(n=max_pairs, random_state=random_seed).sort_values(
            ["descriptor_name", "mol_i", "mol_j"]
        )
    return frame.reset_index(drop=True)


def _pair_columns() -> list[str]:
    return [
        "descriptor_name",
        "mol_i",
        "mol_j",
        "delta",
        "abs_delta",
        "threshold",
        "label",
    ]
