"""On-demand scaffold distance backends for training-time mining."""

from __future__ import annotations

import logging
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from hcmp.data.scaffold_distance.config import (
    DEFAULT_MAX_IN_MEMORY_DISTANCES,
    DEFAULT_MAX_IN_MEMORY_SCAFFOLDS,
    DEFAULT_MAX_MCS_ROUNDS,
    DEFAULT_SCAFFOLD_FAILURE_DISTANCE,
)
from hcmp.data.scaffold_distance.data_types import MoleculeTable, ScaffoldExtractionResult
from hcmp.data.scaffold_distance.scaffold import compute_scaffold_similarity, extract_expanded_scaffold


LOGGER = logging.getLogger(__name__)


class ScaffoldDistanceBackend:
    """Minimal protocol-like base class for scaffold distance lookup."""

    cache_hits: int
    cache_misses: int
    scaffold_distance_failures: int

    def get_distance(self, global_i: int, global_j: int) -> float:
        raise NotImplementedError

    def begin_batch(self) -> None:
        """Optional hook called before one training loss call."""

    def end_batch(self) -> None:
        """Optional hook called after one training loss call."""

    def stats(self) -> dict[str, int]:
        stats = {"cache_hits": self.cache_hits, "cache_misses": self.cache_misses}
        if hasattr(self, "num_multifragment_scaffold_inputs"):
            stats["num_multifragment_scaffold_inputs"] = int(
                self.num_multifragment_scaffold_inputs
            )
        if hasattr(self, "scaffold_distance_failures"):
            stats["scaffold_distance_failures"] = int(self.scaffold_distance_failures)
        return stats

    def _log_distance_failure(
        self,
        logger: logging.Logger,
        i: int,
        j: int,
        exc: Exception,
    ) -> None:
        self.scaffold_distance_failures = int(getattr(self, "scaffold_distance_failures", 0)) + 1
        count = int(self.scaffold_distance_failures)
        if count <= 20 or count % 1000 == 0:
            logger.warning(
                "Scaffold distance failed for pair (%d, %d): %s "
                "(failure_count=%d; logging first 20 and every 1000 thereafter)",
                i,
                j,
                exc,
                count,
            )


class FullMatrixScaffoldDistanceBackend(ScaffoldDistanceBackend):
    """Read scaffold distances from a precomputed full matrix."""

    def __init__(
        self,
        matrix: np.ndarray,
        molecule_table: MoleculeTable | None = None,
        metadata: dict[str, Any] | None = None,
        require_metadata: bool = False,
    ) -> None:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Scaffold distance matrix must be square.")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Scaffold distance matrix must be symmetric.")
        if not np.allclose(np.diag(matrix), 0.0):
            raise ValueError("Scaffold distance matrix diagonal must be zero.")
        self.matrix = matrix
        self.cache_hits = 0
        self.cache_misses = 0
        if molecule_table is not None:
            self._validate_identity(molecule_table, metadata, require_metadata=require_metadata)

    def get_distance(self, global_i: int, global_j: int) -> float:
        i = int(global_i)
        j = int(global_j)
        if i < 0 or j < 0 or i >= self.matrix.shape[0] or j >= self.matrix.shape[1]:
            raise IndexError(f"Scaffold matrix indices out of range: ({i}, {j}).")
        self.cache_hits += 1
        return float(self.matrix[i, j])

    def _validate_identity(
        self,
        molecule_table: MoleculeTable,
        metadata: dict[str, Any] | None,
        require_metadata: bool,
    ) -> None:
        if self.matrix.shape[0] < molecule_table.size:
            raise ValueError(
                "Scaffold distance matrix has fewer rows than the sorted molecule table."
            )
        if not metadata:
            if require_metadata:
                raise ValueError(
                    "Full scaffold distance matrix backend requires metadata with "
                    "canonical_smiles and source_row_index for identity validation."
                )
            return
        identities = _metadata_identities(metadata)
        if identities is None:
            if require_metadata:
                raise ValueError(
                    "Scaffold distance metadata is missing canonical_smiles/source_row_index rows."
                )
            return
        if len(identities) < molecule_table.size:
            raise ValueError(
                "Scaffold distance metadata has fewer identity rows than the molecule table."
            )
        for idx in range(molecule_table.size):
            expected_smiles = str(molecule_table.canonical_smiles[idx])
            expected_source = int(molecule_table.dataframe.iloc[idx]["source_row_index"])
            actual = identities[idx]
            actual_smiles = str(actual["canonical_smiles"])
            actual_source = int(actual["source_row_index"])
            if actual_smiles != expected_smiles or actual_source != expected_source:
                raise ValueError(
                    "Scaffold distance matrix metadata does not match dataset order at "
                    f"row {idx}: expected ({expected_smiles!r}, {expected_source}), "
                    f"found ({actual_smiles!r}, {actual_source})."
                )


class InMemoryScaffoldDistanceCache(ScaffoldDistanceBackend):
    """In-memory on-demand scaffold pair distance cache."""

    def __init__(
        self,
        molecule_table: MoleculeTable,
        max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
        max_in_memory_distances: int | None = DEFAULT_MAX_IN_MEMORY_DISTANCES,
        max_in_memory_scaffolds: int | None = DEFAULT_MAX_IN_MEMORY_SCAFFOLDS,
        logger: logging.Logger | None = None,
    ) -> None:
        self.molecule_table = molecule_table
        self.max_rounds = max_rounds
        self.logger = logger or LOGGER
        self.max_in_memory_distances = _normalize_cache_cap(max_in_memory_distances)
        self.max_in_memory_scaffolds = _normalize_cache_cap(max_in_memory_scaffolds)
        self._distances: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._scaffolds: OrderedDict[int, ScaffoldExtractionResult] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.scaffold_distance_failures = 0
        self.num_multifragment_scaffold_inputs = 0

    def get_distance(self, global_i: int, global_j: int) -> float:
        i, j = _canonical_pair(global_i, global_j)
        self._validate_index(i)
        self._validate_index(j)
        if i == j:
            return 0.0
        key = (i, j)
        if key in self._distances:
            self.cache_hits += 1
            return self._get_cached_distance(key)
        self.cache_misses += 1
        distance = self._compute_distance(i, j)
        self._remember_distance(key, distance)
        return distance

    def _validate_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.molecule_table.size:
            raise IndexError(f"Molecule global index {idx} is out of range.")

    def _compute_distance(self, i: int, j: int) -> float:
        try:
            match_result = compute_scaffold_similarity(
                self._get_scaffold(i).scaffold_mol,
                self._get_scaffold(j).scaffold_mol,
                max_rounds=self.max_rounds,
            )
            if match_result.status.startswith("failed"):
                raise RuntimeError(
                    f"Scaffold similarity returned failure status: {match_result.status}"
                )
            return float(match_result.distance)
        except Exception as exc:
            self._log_distance_failure(self.logger, i, j, exc)
            return float(DEFAULT_SCAFFOLD_FAILURE_DISTANCE)

    def _get_scaffold(self, idx: int) -> ScaffoldExtractionResult:
        if idx in self._scaffolds:
            self._scaffolds.move_to_end(idx)
            return self._scaffolds[idx]
        mol = Chem.Mol(self.molecule_table.mols[idx])
        if len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)) > 1:
            self.num_multifragment_scaffold_inputs += 1
        scaffold = extract_expanded_scaffold(mol)
        self._remember_scaffold(idx, scaffold)
        return scaffold

    def _get_cached_distance(self, key: tuple[int, int]) -> float:
        self._distances.move_to_end(key)
        return self._distances[key]

    def _remember_distance(self, key: tuple[int, int], distance: float) -> None:
        if self.max_in_memory_distances == 0:
            return
        self._distances[key] = float(distance)
        self._distances.move_to_end(key)
        _evict_lru(self._distances, self.max_in_memory_distances)

    def _remember_scaffold(self, idx: int, scaffold: ScaffoldExtractionResult) -> None:
        if self.max_in_memory_scaffolds == 0:
            return
        self._scaffolds[int(idx)] = scaffold
        self._scaffolds.move_to_end(int(idx))
        _evict_lru(self._scaffolds, self.max_in_memory_scaffolds)

    def stats(self) -> dict[str, int]:
        stats = super().stats()
        stats["in_memory_distances"] = len(self._distances)
        stats["in_memory_scaffolds"] = len(self._scaffolds)
        stats["max_in_memory_distances"] = int(self.max_in_memory_distances)
        stats["max_in_memory_scaffolds"] = int(self.max_in_memory_scaffolds)
        return stats


class SQLiteScaffoldDistanceCache(InMemoryScaffoldDistanceCache):
    """Persistent SQLite scaffold pair distance cache."""

    def __init__(
        self,
        molecule_table: MoleculeTable,
        cache_path: str | Path,
        max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
        sqlite_wal: bool = True,
        commit_every_misses: int = 1000,
        max_in_memory_distances: int | None = DEFAULT_MAX_IN_MEMORY_DISTANCES,
        max_in_memory_scaffolds: int | None = DEFAULT_MAX_IN_MEMORY_SCAFFOLDS,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            molecule_table=molecule_table,
            max_rounds=max_rounds,
            max_in_memory_distances=max_in_memory_distances,
            max_in_memory_scaffolds=max_in_memory_scaffolds,
            logger=logger,
        )
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.cache_path)
        if sqlite_wal:
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS scaffold_distances ("
            "i INTEGER NOT NULL, "
            "j INTEGER NOT NULL, "
            "distance REAL NOT NULL, "
            "PRIMARY KEY (i, j)"
            ")"
        )
        self.connection.commit()
        self.commit_every_misses = max(1, int(commit_every_misses))
        self._pending_inserts = 0

    def get_distance(self, global_i: int, global_j: int) -> float:
        i, j = _canonical_pair(global_i, global_j)
        self._validate_index(i)
        self._validate_index(j)
        if i == j:
            return 0.0
        key = (i, j)
        if key in self._distances:
            self.cache_hits += 1
            return self._get_cached_distance(key)
        row = self.connection.execute(
            "SELECT distance FROM scaffold_distances WHERE i = ? AND j = ?",
            key,
        ).fetchone()
        if row is not None:
            self.cache_hits += 1
            distance = float(row[0])
            self._remember_distance(key, distance)
            return distance
        self.cache_misses += 1
        distance = self._compute_distance(i, j)
        self.connection.execute(
            "INSERT OR REPLACE INTO scaffold_distances (i, j, distance) VALUES (?, ?, ?)",
            (i, j, distance),
        )
        self._pending_inserts += 1
        if self._pending_inserts >= self.commit_every_misses:
            self.flush()
        self._remember_distance(key, distance)
        return distance

    def flush(self) -> None:
        self.connection.commit()
        self._pending_inserts = 0

    def close(self) -> None:
        self.flush()
        self.connection.close()


class SQLiteGraphCacheScaffoldDistanceCache(ScaffoldDistanceBackend):
    """Persistent scaffold cache backed by canonical SMILES from graph-cache shards."""

    def __init__(
        self,
        graph_cache_dir: str | Path,
        cache_path: str | Path,
        max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
        sqlite_wal: bool = True,
        commit_every_misses: int = 1000,
        max_in_memory_distances: int | None = DEFAULT_MAX_IN_MEMORY_DISTANCES,
        max_in_memory_scaffolds: int | None = DEFAULT_MAX_IN_MEMORY_SCAFFOLDS,
        logger: logging.Logger | None = None,
    ) -> None:
        from hcmp.data.graph_cache import ShardedGraphCache

        self.graph_cache = ShardedGraphCache(graph_cache_dir)
        self.size = len(self.graph_cache)
        self.max_rounds = max_rounds
        self.logger = logger or LOGGER
        self.max_in_memory_distances = _normalize_cache_cap(max_in_memory_distances)
        self.max_in_memory_scaffolds = _normalize_cache_cap(max_in_memory_scaffolds)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.cache_path)
        if sqlite_wal:
            self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS scaffold_distances ("
            "i INTEGER NOT NULL, "
            "j INTEGER NOT NULL, "
            "distance REAL NOT NULL, "
            "PRIMARY KEY (i, j)"
            ")"
        )
        self.connection.commit()
        self._distances: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._scaffolds: OrderedDict[int, ScaffoldExtractionResult] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.scaffold_distance_failures = 0
        self.num_multifragment_scaffold_inputs = 0
        self.commit_every_misses = max(1, int(commit_every_misses))
        self._pending_inserts = 0

    def get_distance(self, global_i: int, global_j: int) -> float:
        i, j = _canonical_pair(global_i, global_j)
        self._validate_index(i)
        self._validate_index(j)
        if i == j:
            return 0.0
        key = (i, j)
        if key in self._distances:
            self.cache_hits += 1
            return self._get_cached_distance(key)
        row = self.connection.execute(
            "SELECT distance FROM scaffold_distances WHERE i = ? AND j = ?",
            key,
        ).fetchone()
        if row is not None:
            self.cache_hits += 1
            distance = float(row[0])
            self._remember_distance(key, distance)
            return distance
        self.cache_misses += 1
        distance = self._compute_distance(i, j)
        self.connection.execute(
            "INSERT OR REPLACE INTO scaffold_distances (i, j, distance) VALUES (?, ?, ?)",
            (i, j, distance),
        )
        self._pending_inserts += 1
        if self._pending_inserts >= self.commit_every_misses:
            self.flush()
        self._remember_distance(key, distance)
        return distance

    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "scaffold_distance_failures": self.scaffold_distance_failures,
            "num_multifragment_scaffold_inputs": self.num_multifragment_scaffold_inputs,
            "in_memory_distances": len(self._distances),
            "in_memory_scaffolds": len(self._scaffolds),
            "max_in_memory_distances": int(self.max_in_memory_distances),
            "max_in_memory_scaffolds": int(self.max_in_memory_scaffolds),
        }

    def flush(self) -> None:
        self.connection.commit()
        self._pending_inserts = 0

    def close(self) -> None:
        self.flush()
        self.connection.close()

    def _validate_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Molecule global index {idx} is out of range.")

    def _compute_distance(self, i: int, j: int) -> float:
        try:
            match_result = compute_scaffold_similarity(
                self._get_scaffold(i).scaffold_mol,
                self._get_scaffold(j).scaffold_mol,
                max_rounds=self.max_rounds,
            )
            if match_result.status.startswith("failed"):
                raise RuntimeError(
                    f"Scaffold similarity returned failure status: {match_result.status}"
                )
            return float(match_result.distance)
        except Exception as exc:
            self._log_distance_failure(self.logger, i, j, exc)
            return float(DEFAULT_SCAFFOLD_FAILURE_DISTANCE)

    def _get_scaffold(self, idx: int) -> ScaffoldExtractionResult:
        if idx in self._scaffolds:
            self._scaffolds.move_to_end(idx)
            return self._scaffolds[idx]
        graph = self.graph_cache[idx]
        mol = Chem.MolFromSmiles(graph.canonical_smiles)
        if mol is None:
            raise ValueError(f"Invalid cached canonical SMILES at index {idx}.")
        if len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)) > 1:
            self.num_multifragment_scaffold_inputs += 1
        scaffold = extract_expanded_scaffold(mol)
        self._remember_scaffold(idx, scaffold)
        return scaffold

    def _get_cached_distance(self, key: tuple[int, int]) -> float:
        self._distances.move_to_end(key)
        return self._distances[key]

    def _remember_distance(self, key: tuple[int, int], distance: float) -> None:
        if self.max_in_memory_distances == 0:
            return
        self._distances[key] = float(distance)
        self._distances.move_to_end(key)
        _evict_lru(self._distances, self.max_in_memory_distances)

    def _remember_scaffold(self, idx: int, scaffold: ScaffoldExtractionResult) -> None:
        if self.max_in_memory_scaffolds == 0:
            return
        self._scaffolds[int(idx)] = scaffold
        self._scaffolds.move_to_end(int(idx))
        _evict_lru(self._scaffolds, self.max_in_memory_scaffolds)


class OnTheFlyGraphCacheScaffoldDistanceBackend(ScaffoldDistanceBackend):
    """Compute scaffold distances from graph-cache SMILES with only batch-local caches."""

    def __init__(
        self,
        graph_cache_dir: str | Path,
        max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
        max_batch_distances: int | None = None,
        max_batch_scaffolds: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        from hcmp.data.graph_cache import ShardedGraphCache

        self.graph_cache = ShardedGraphCache(graph_cache_dir)
        self.size = len(self.graph_cache)
        self.max_rounds = max_rounds
        self.logger = logger or LOGGER
        self.max_batch_distances = (
            0 if max_batch_distances is None else max(0, int(max_batch_distances))
        )
        self.max_batch_scaffolds = (
            0 if max_batch_scaffolds is None else max(0, int(max_batch_scaffolds))
        )
        self._distances: OrderedDict[tuple[int, int], float] = OrderedDict()
        self._scaffolds: OrderedDict[int, ScaffoldExtractionResult] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.scaffold_distance_failures = 0
        self.num_multifragment_scaffold_inputs = 0

    def begin_batch(self) -> None:
        self._distances.clear()
        self._scaffolds.clear()

    def end_batch(self) -> None:
        self._distances.clear()
        self._scaffolds.clear()

    def get_distance(self, global_i: int, global_j: int) -> float:
        i, j = _canonical_pair(global_i, global_j)
        self._validate_index(i)
        self._validate_index(j)
        if i == j:
            return 0.0
        key = (i, j)
        if key in self._distances:
            self.cache_hits += 1
            self._distances.move_to_end(key)
            return self._distances[key]
        self.cache_misses += 1
        distance = self._compute_distance(i, j)
        self._remember_distance(key, distance)
        return distance

    def stats(self) -> dict[str, int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "scaffold_distance_failures": self.scaffold_distance_failures,
            "num_multifragment_scaffold_inputs": self.num_multifragment_scaffold_inputs,
            "batch_distances": len(self._distances),
            "batch_scaffolds": len(self._scaffolds),
            "max_batch_distances": int(self.max_batch_distances),
            "max_batch_scaffolds": int(self.max_batch_scaffolds),
        }

    def _validate_index(self, idx: int) -> None:
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Molecule global index {idx} is out of range.")

    def _compute_distance(self, i: int, j: int) -> float:
        try:
            match_result = compute_scaffold_similarity(
                self._get_scaffold(i).scaffold_mol,
                self._get_scaffold(j).scaffold_mol,
                max_rounds=self.max_rounds,
            )
            if match_result.status.startswith("failed"):
                raise RuntimeError(
                    f"Scaffold similarity returned failure status: {match_result.status}"
                )
            return float(match_result.distance)
        except Exception as exc:
            self._log_distance_failure(self.logger, i, j, exc)
            return float(DEFAULT_SCAFFOLD_FAILURE_DISTANCE)

    def _get_scaffold(self, idx: int) -> ScaffoldExtractionResult:
        if idx in self._scaffolds:
            self._scaffolds.move_to_end(idx)
            return self._scaffolds[idx]
        graph = self.graph_cache[idx]
        mol = Chem.MolFromSmiles(graph.canonical_smiles)
        if mol is None:
            raise ValueError(f"Invalid cached canonical SMILES at index {idx}.")
        if len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)) > 1:
            self.num_multifragment_scaffold_inputs += 1
        scaffold = extract_expanded_scaffold(mol)
        self._remember_scaffold(idx, scaffold)
        return scaffold

    def _remember_distance(self, key: tuple[int, int], distance: float) -> None:
        if self.max_batch_distances == 0:
            return
        self._distances[key] = float(distance)
        self._distances.move_to_end(key)
        _evict_lru(self._distances, self.max_batch_distances)

    def _remember_scaffold(self, idx: int, scaffold: ScaffoldExtractionResult) -> None:
        if self.max_batch_scaffolds == 0:
            return
        self._scaffolds[int(idx)] = scaffold
        self._scaffolds.move_to_end(int(idx))
        _evict_lru(self._scaffolds, self.max_batch_scaffolds)


class OnTheFlyMoleculeTableScaffoldDistanceBackend(InMemoryScaffoldDistanceCache):
    """Molecule-table on-the-fly backend with no cross-batch distance persistence."""

    def __init__(
        self,
        molecule_table: MoleculeTable,
        max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
        max_batch_distances: int | None = None,
        max_batch_scaffolds: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            molecule_table=molecule_table,
            max_rounds=max_rounds,
            max_in_memory_distances=0 if max_batch_distances is None else max_batch_distances,
            max_in_memory_scaffolds=0 if max_batch_scaffolds is None else max_batch_scaffolds,
            logger=logger,
        )
        self.max_batch_distances = self.max_in_memory_distances
        self.max_batch_scaffolds = self.max_in_memory_scaffolds

    def begin_batch(self) -> None:
        self._distances.clear()
        self._scaffolds.clear()

    def end_batch(self) -> None:
        self._distances.clear()
        self._scaffolds.clear()

    def stats(self) -> dict[str, int]:
        stats = super().stats()
        stats["batch_distances"] = len(self._distances)
        stats["batch_scaffolds"] = len(self._scaffolds)
        stats["max_batch_distances"] = int(self.max_batch_distances)
        stats["max_batch_scaffolds"] = int(self.max_batch_scaffolds)
        return stats


def build_scaffold_distance_backend(
    backend: str,
    molecule_table: MoleculeTable | None,
    matrix: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    cache_path: str | Path | None = None,
    max_rounds: int = DEFAULT_MAX_MCS_ROUNDS,
    graph_cache_dir: str | Path | None = None,
    sqlite_wal: bool = True,
    commit_every_misses: int = 1000,
    max_in_memory_distances: int | None = DEFAULT_MAX_IN_MEMORY_DISTANCES,
    max_in_memory_scaffolds: int | None = DEFAULT_MAX_IN_MEMORY_SCAFFOLDS,
) -> ScaffoldDistanceBackend:
    """Construct a scaffold distance backend from config-like values."""

    if backend == "full_matrix":
        if matrix is None:
            raise ValueError("distance_backend='full_matrix' requires a scaffold distance matrix.")
        if molecule_table is None:
            raise ValueError("distance_backend='full_matrix' requires molecule_table.")
        return FullMatrixScaffoldDistanceBackend(
            matrix,
            molecule_table=molecule_table,
            metadata=metadata,
            require_metadata=True,
        )
    if backend == "on_the_fly":
        if graph_cache_dir is not None:
            return OnTheFlyGraphCacheScaffoldDistanceBackend(
                graph_cache_dir=graph_cache_dir,
                max_rounds=max_rounds,
                max_batch_distances=max_in_memory_distances,
                max_batch_scaffolds=max_in_memory_scaffolds,
            )
        if molecule_table is None:
            raise ValueError("distance_backend='on_the_fly' requires molecule_table or graph_cache_dir.")
        return OnTheFlyMoleculeTableScaffoldDistanceBackend(
            molecule_table,
            max_rounds=max_rounds,
            max_batch_distances=max_in_memory_distances,
            max_batch_scaffolds=max_in_memory_scaffolds,
        )
    if backend in {"cache", "sqlite_cache"}:
        if graph_cache_dir is not None and cache_path is not None:
            return SQLiteGraphCacheScaffoldDistanceCache(
                graph_cache_dir=graph_cache_dir,
                cache_path=cache_path,
                max_rounds=max_rounds,
                sqlite_wal=sqlite_wal,
                commit_every_misses=commit_every_misses,
                max_in_memory_distances=max_in_memory_distances,
                max_in_memory_scaffolds=max_in_memory_scaffolds,
            )
        if cache_path is None:
            if molecule_table is None:
                raise ValueError("In-memory scaffold cache requires molecule_table.")
            return InMemoryScaffoldDistanceCache(
                molecule_table,
                max_rounds=max_rounds,
                max_in_memory_distances=max_in_memory_distances,
                max_in_memory_scaffolds=max_in_memory_scaffolds,
            )
        if molecule_table is None:
            raise ValueError("SQLite scaffold cache requires molecule_table unless graph_cache_dir is set.")
        return SQLiteScaffoldDistanceCache(
            molecule_table,
            cache_path=cache_path,
            max_rounds=max_rounds,
            sqlite_wal=sqlite_wal,
            commit_every_misses=commit_every_misses,
            max_in_memory_distances=max_in_memory_distances,
            max_in_memory_scaffolds=max_in_memory_scaffolds,
        )
    raise ValueError(
        "Unsupported scaffold distance backend "
        f"{backend!r}; expected 'on_the_fly', 'cache', 'sqlite_cache', or 'full_matrix'."
    )


def molecule_identity_metadata(molecule_table: MoleculeTable) -> list[dict[str, Any]]:
    """Return strict row identity metadata for a sorted molecule table."""

    return [
        {
            "canonical_smiles": str(molecule_table.canonical_smiles[idx]),
            "source_row_index": int(molecule_table.dataframe.iloc[idx]["source_row_index"]),
        }
        for idx in range(molecule_table.size)
    ]


def _metadata_identities(metadata: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "molecules" in metadata:
        return list(metadata["molecules"])
    if "canonical_smiles" in metadata and "source_row_indices" in metadata:
        smiles = list(metadata["canonical_smiles"])
        source_indices = list(metadata["source_row_indices"])
        return [
            {"canonical_smiles": smiles[idx], "source_row_index": source_indices[idx]}
            for idx in range(min(len(smiles), len(source_indices)))
        ]
    return None


def _canonical_pair(global_i: int, global_j: int) -> tuple[int, int]:
    i = int(global_i)
    j = int(global_j)
    return (i, j) if i <= j else (j, i)


def _normalize_cache_cap(value: int | None) -> int:
    if value is None:
        return 0
    return max(0, int(value))


def _evict_lru(cache: OrderedDict, max_size: int) -> None:
    while len(cache) > max_size:
        cache.popitem(last=False)
