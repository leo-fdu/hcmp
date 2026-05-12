"""Pairwise expanded-scaffold distance matrix utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem

from hcmp.data.scaffold_distance.config import (
    DEFAULT_MAX_MCS_ROUNDS,
    DEFAULT_NUMPY_DTYPE,
    DEFAULT_SCAFFOLD_FAILURE_DISTANCE,
)
from hcmp.data.scaffold_distance.data_types import (
    PairwiseComputationFailure,
    PairwiseDistanceMatrixResult,
    ScaffoldExtractionResult,
)
from hcmp.data.scaffold_distance.io_utils import load_distance_cache, save_distance_cache
from hcmp.data.scaffold_distance.scaffold import compute_scaffold_similarity


LOGGER = logging.getLogger(__name__)


def compute_pairwise_scaffold_distance_matrix(
    scaffolds: Sequence[Chem.Mol | ScaffoldExtractionResult],
    max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
    dtype: np.dtype = DEFAULT_NUMPY_DTYPE,
    logger: logging.Logger | None = None,
) -> PairwiseDistanceMatrixResult:
    """Compute the full pairwise iterative scaffold distance matrix."""

    active_logger = logger or LOGGER
    scaffold_mols = [_coerce_scaffold_mol(item) for item in scaffolds]
    num_items = len(scaffold_mols)
    matrix = np.zeros((num_items, num_items), dtype=dtype)
    failures: list[PairwiseComputationFailure] = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            try:
                match_result = compute_scaffold_similarity(
                    scaffold_mols[i],
                    scaffold_mols[j],
                    max_rounds=max_rounds,
                )
                if match_result.status.startswith("failed"):
                    raise RuntimeError(
                        f"Scaffold similarity returned failure status: {match_result.status}"
                    )
                distance = match_result.distance
            except Exception as exc:
                active_logger.warning("Scaffold distance failed for pair (%d, %d): %s", i, j, exc)
                distance = DEFAULT_SCAFFOLD_FAILURE_DISTANCE
                failures.append(PairwiseComputationFailure(i=i, j=j, status="failed", message=str(exc)))
            matrix[i, j] = distance
            matrix[j, i] = distance
    np.fill_diagonal(matrix, 0.0)
    return PairwiseDistanceMatrixResult(
        matrix=matrix,
        metric_name="scaffold",
        failures=tuple(failures),
        metadata={
            "num_items": num_items,
            "max_rounds": max_rounds,
            "dtype": str(np.dtype(dtype)),
            "num_failures": len(failures),
        },
    )


def load_or_compute_scaffold_distance_matrix(
    scaffolds: Sequence[Chem.Mol | ScaffoldExtractionResult],
    cache_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
    dtype: np.dtype = DEFAULT_NUMPY_DTYPE,
    logger: logging.Logger | None = None,
) -> PairwiseDistanceMatrixResult:
    """Load a cached scaffold distance matrix or compute and cache it."""

    if cache_path is not None and Path(cache_path).exists():
        matrix, metadata = load_distance_cache(cache_path, metadata_path, dtype=dtype)
        return PairwiseDistanceMatrixResult(
            matrix=matrix,
            metric_name="scaffold",
            failures=_failures_from_metadata(metadata),
            metadata=metadata or {},
        )
    result = compute_pairwise_scaffold_distance_matrix(
        scaffolds=scaffolds,
        max_rounds=max_rounds,
        dtype=dtype,
        logger=logger,
    )
    if cache_path is not None:
        metadata = _result_to_metadata(result)
        save_distance_cache(result.matrix, cache_path, metadata=metadata, metadata_path=metadata_path)
        result = PairwiseDistanceMatrixResult(
            matrix=result.matrix,
            metric_name=result.metric_name,
            failures=result.failures,
            metadata=metadata,
        )
    return result


def _coerce_scaffold_mol(item: Chem.Mol | ScaffoldExtractionResult) -> Chem.Mol:
    if isinstance(item, ScaffoldExtractionResult):
        return item.scaffold_mol
    return item


def _result_to_metadata(result: PairwiseDistanceMatrixResult) -> dict[str, object]:
    metadata = dict(result.metadata)
    metadata["metric_name"] = result.metric_name
    metadata["failures"] = [
        {
            "i": failure.i,
            "j": failure.j,
            "status": failure.status,
            "message": failure.message,
        }
        for failure in result.failures
    ]
    return metadata


def _failures_from_metadata(
    metadata: dict[str, object] | None,
) -> tuple[PairwiseComputationFailure, ...]:
    if not metadata:
        return ()
    failures = metadata.get("failures", [])
    return tuple(
        PairwiseComputationFailure(
            i=int(item["i"]),
            j=int(item["j"]),
            status=str(item["status"]),
            message=str(item["message"]),
        )
        for item in failures
    )
