"""Expanded-scaffold MCS distance API."""

from hcmp.data.scaffold_distance.cache import (
    FullMatrixScaffoldDistanceBackend,
    InMemoryScaffoldDistanceCache,
    OnTheFlyGraphCacheScaffoldDistanceBackend,
    OnTheFlyMoleculeTableScaffoldDistanceBackend,
    ScaffoldDistanceBackend,
    SQLiteGraphCacheScaffoldDistanceCache,
    SQLiteScaffoldDistanceCache,
    build_scaffold_distance_backend,
    molecule_identity_metadata,
)
from hcmp.data.scaffold_distance.distance import (
    compute_pairwise_scaffold_distance_matrix,
    load_or_compute_scaffold_distance_matrix,
)
from hcmp.data.scaffold_distance.io_utils import load_molecule_table
from hcmp.data.scaffold_distance.scaffold import (
    extract_expanded_scaffold,
    select_main_organic_fragment_for_scaffold,
)

__all__ = [
    "FullMatrixScaffoldDistanceBackend",
    "InMemoryScaffoldDistanceCache",
    "OnTheFlyGraphCacheScaffoldDistanceBackend",
    "OnTheFlyMoleculeTableScaffoldDistanceBackend",
    "ScaffoldDistanceBackend",
    "SQLiteGraphCacheScaffoldDistanceCache",
    "SQLiteScaffoldDistanceCache",
    "build_scaffold_distance_backend",
    "compute_pairwise_scaffold_distance_matrix",
    "extract_expanded_scaffold",
    "load_molecule_table",
    "load_or_compute_scaffold_distance_matrix",
    "molecule_identity_metadata",
    "select_main_organic_fragment_for_scaffold",
]
