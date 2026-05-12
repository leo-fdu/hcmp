"""Structured data types for scaffold distance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


@dataclass(frozen=True)
class ScaffoldExtractionResult:
    scaffold_mol: Chem.Mol
    scaffold_smiles: str
    scaffold_atom_indices: tuple[int, ...]
    scaffold_bond_indices: tuple[int, ...]
    num_atoms: int
    num_bonds: int


@dataclass(frozen=True)
class ScaffoldMatchRound:
    round_index: int
    bond_count: int
    atom_indices_a: tuple[int, ...]
    bond_indices_a: tuple[int, ...]
    atom_indices_b: tuple[int, ...]
    bond_indices_b: tuple[int, ...]


@dataclass(frozen=True)
class ScaffoldMatchResult:
    round_bond_counts: tuple[int, ...]
    matched_bond_total: int
    similarity: float
    distance: float
    status: str
    rounds: tuple[ScaffoldMatchRound, ...] = ()


@dataclass(frozen=True)
class PairwiseComputationFailure:
    i: int
    j: int
    status: str
    message: str


@dataclass(frozen=True)
class PairwiseDistanceMatrixResult:
    matrix: np.ndarray
    metric_name: str
    failures: tuple[PairwiseComputationFailure, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MoleculeTable:
    dataframe: pd.DataFrame
    smiles_column: str
    canonical_smiles: tuple[str, ...]
    mols: tuple[Chem.Mol, ...]
    source: str | None = None
    dropped_invalid_indices: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        return len(self.mols)
