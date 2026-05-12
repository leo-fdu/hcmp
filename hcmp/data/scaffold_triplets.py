"""Expanded-scaffold triplet generation for HCMP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcmp.data.scaffold_distance import (
    extract_expanded_scaffold,
    load_molecule_table,
    load_or_compute_scaffold_distance_matrix,
    molecule_identity_metadata,
)
from hcmp.data.scaffold_distance.config import DEFAULT_MAX_MCS_ROUNDS
from hcmp.data.scaffold_distance.io_utils import save_json


def build_scaffold_triplets_from_csv(
    input_csv: str | Path,
    smiles_column: str = "smiles",
    output_path: str | Path | None = None,
    distance_cache_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    min_distance_gap: float = 0.15,
    max_mcs_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
    max_triplets: int | None = None,
    invalid_smiles: str = "drop",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load molecules, compute/cache scaffold distances, and sample triplets."""

    molecule_table = load_molecule_table(
        input_csv,
        smiles_column=smiles_column,
        invalid_smiles=invalid_smiles,
        sort_by_canonical_smiles=True,
    )
    scaffolds = []
    extraction_failures: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    for idx, mol in enumerate(molecule_table.mols):
        try:
            scaffolds.append(extract_expanded_scaffold(mol))
            valid_indices.append(idx)
        except Exception as exc:
            extraction_failures.append({"index": idx, "message": str(exc)})

    distance_result = load_or_compute_scaffold_distance_matrix(
        scaffolds,
        cache_path=distance_cache_path,
        metadata_path=_distance_metadata_path(distance_cache_path),
        max_rounds=max_mcs_rounds,
    )
    distance_metadata_path = _distance_metadata_path(distance_cache_path)
    if distance_metadata_path is not None:
        table_identity = molecule_identity_metadata(molecule_table)
        distance_metadata = dict(distance_result.metadata)
        distance_metadata["molecules"] = [table_identity[idx] for idx in valid_indices]
        save_json(distance_metadata, distance_metadata_path)
    triplets = sample_scaffold_triplets(
        distance_result.matrix,
        min_distance_gap=min_distance_gap,
        max_triplets=max_triplets,
    )
    triplets = _attach_triplet_identity_columns(
        triplets,
        molecule_table.dataframe,
        valid_indices,
    )
    failure_count = len(extraction_failures) + len(distance_result.failures)
    metadata = {
        "input_csv": str(input_csv),
        "num_molecules": molecule_table.size,
        "num_scaffolds": len(scaffolds),
        "num_valid_triplets": int(len(triplets)),
        "min_distance_gap": float(min_distance_gap),
        "max_mcs_rounds": int(max_mcs_rounds),
        "failures": int(failure_count),
        "dropped_invalid_indices": list(molecule_table.dropped_invalid_indices),
        "extraction_failures": extraction_failures,
        "distance_failures": [
            {
                "i": failure.i,
                "j": failure.j,
                "status": failure.status,
                "message": failure.message,
            }
            for failure in distance_result.failures
        ],
        "valid_source_indices": valid_indices,
    }
    if output_path is not None:
        _write_parquet_or_csv(triplets, Path(output_path))
    if metadata_path is not None:
        save_json(metadata, metadata_path)
    return triplets, metadata


def sample_scaffold_triplets(
    distance_matrix: np.ndarray,
    min_distance_gap: float = 0.15,
    max_triplets: int | None = None,
) -> pd.DataFrame:
    """Sample deterministic triplets satisfying D_ap + min_gap < D_an."""

    matrix = np.asarray(distance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be square.")
    rows: list[dict[str, Any]] = []
    num_items = matrix.shape[0]
    for anchor_idx in range(num_items):
        positives = sorted(
            [idx for idx in range(num_items) if idx != anchor_idx],
            key=lambda idx: (matrix[anchor_idx, idx], idx),
        )
        negatives = sorted(
            [idx for idx in range(num_items) if idx != anchor_idx],
            key=lambda idx: (matrix[anchor_idx, idx], idx),
        )
        for positive_idx in positives:
            d_ap = float(matrix[anchor_idx, positive_idx])
            for negative_idx in negatives:
                if negative_idx == positive_idx:
                    continue
                d_an = float(matrix[anchor_idx, negative_idx])
                if d_ap + min_distance_gap < d_an:
                    rows.append(
                        {
                            "anchor_idx": anchor_idx,
                            "positive_idx": positive_idx,
                            "negative_idx": negative_idx,
                            "d_ap": d_ap,
                            "d_an": d_an,
                            "distance_gap": d_an - d_ap,
                        }
                    )
                    if max_triplets is not None and len(rows) >= max_triplets:
                        return pd.DataFrame(rows, columns=_triplet_columns())
    return pd.DataFrame(rows, columns=_triplet_columns())


def filter_scaffold_triplets(
    candidates: pd.DataFrame,
    min_distance_gap: float = 0.15,
) -> pd.DataFrame:
    """Compatibility helper: keep triplets satisfying D_ap + gap < D_an."""

    if {"D_ap", "D_an"}.issubset(candidates.columns):
        d_ap_col, d_an_col = "D_ap", "D_an"
    else:
        d_ap_col, d_an_col = "d_ap", "d_an"
    required = {d_ap_col, d_an_col}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing triplet columns: {sorted(missing)}")
    frame = candidates.copy()
    frame["distance_gap"] = frame[d_an_col] - frame[d_ap_col]
    return frame[frame[d_ap_col] + min_distance_gap < frame[d_an_col]].reset_index(drop=True)


def _triplet_columns() -> list[str]:
    return [
        "anchor_idx",
        "positive_idx",
        "negative_idx",
        "anchor_smiles",
        "positive_smiles",
        "negative_smiles",
        "anchor_source_row_index",
        "positive_source_row_index",
        "negative_source_row_index",
        "d_ap",
        "d_an",
        "distance_gap",
    ]


def _attach_triplet_identity_columns(
    compact_triplets: pd.DataFrame,
    molecule_frame: pd.DataFrame,
    valid_indices: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in compact_triplets.to_dict(orient="records"):
        anchor_idx = valid_indices[int(row["anchor_idx"])]
        positive_idx = valid_indices[int(row["positive_idx"])]
        negative_idx = valid_indices[int(row["negative_idx"])]
        anchor = molecule_frame.iloc[anchor_idx]
        positive = molecule_frame.iloc[positive_idx]
        negative = molecule_frame.iloc[negative_idx]
        rows.append(
            {
                "anchor_idx": anchor_idx,
                "positive_idx": positive_idx,
                "negative_idx": negative_idx,
                "anchor_smiles": anchor["canonical_smiles"],
                "positive_smiles": positive["canonical_smiles"],
                "negative_smiles": negative["canonical_smiles"],
                "anchor_source_row_index": int(anchor["source_row_index"]),
                "positive_source_row_index": int(positive["source_row_index"]),
                "negative_source_row_index": int(negative["source_row_index"]),
                "d_ap": float(row["d_ap"]),
                "d_an": float(row["d_an"]),
                "distance_gap": float(row["distance_gap"]),
            }
        )
    return pd.DataFrame(rows, columns=_triplet_columns())


def _distance_metadata_path(distance_cache_path: str | Path | None) -> Path | None:
    if distance_cache_path is None:
        return None
    return Path(str(distance_cache_path) + ".metadata.json")


def _write_parquet_or_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        frame.to_json(path, orient="records", lines=True)
        return path
    if path.suffix != ".csv":
        path = path.with_suffix(path.suffix + ".csv")
    frame.to_csv(path, index=False)
    return path
