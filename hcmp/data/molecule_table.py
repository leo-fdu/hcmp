"""Shared HCMP molecule-table loading and slicing helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hcmp.data.scaffold_distance.data_types import MoleculeTable
from hcmp.data.scaffold_distance.io_utils import load_molecule_table


def load_hcmp_molecule_table(
    source: str | Path | pd.DataFrame,
    smiles_column: str = "smiles",
    max_molecules: int | None = None,
    sort_by_canonical_smiles: bool = True,
    strict: bool = False,
) -> MoleculeTable:
    """Load the canonical HCMP molecule table, then apply optional debug slicing."""

    table = load_molecule_table(
        source,
        smiles_column=smiles_column,
        invalid_smiles="raise" if strict else "drop",
        sort_by_canonical_smiles=sort_by_canonical_smiles,
    )
    full_count = table.size
    if max_molecules is not None:
        print(f"max_molecules={int(max_molecules)} applied after canonicalization/sorting")
        table = slice_molecule_table(table, int(max_molecules))
        _log_molecule_selection(full_count, table)
    return table


def slice_molecule_table(molecule_table: MoleculeTable, max_molecules: int) -> MoleculeTable:
    """Return the leading rows from an already canonicalized/sorted molecule table."""

    frame = molecule_table.dataframe.head(max_molecules).reset_index(drop=True)
    return MoleculeTable(
        dataframe=frame,
        smiles_column=molecule_table.smiles_column,
        canonical_smiles=tuple(frame["canonical_smiles"].tolist()),
        mols=tuple(molecule_table.mols[:max_molecules]),
        source=molecule_table.source,
        dropped_invalid_indices=molecule_table.dropped_invalid_indices,
    )


def _log_molecule_selection(full_count: int, table: MoleculeTable) -> None:
    print(f"full valid molecule count before slicing={full_count}")
    print(f"selected molecule count after slicing={table.size}")
    if table.size:
        print(f"first selected canonical_smiles={table.canonical_smiles[0]}")
        print(f"last selected canonical_smiles={table.canonical_smiles[-1]}")
