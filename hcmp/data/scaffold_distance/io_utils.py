"""I/O helpers for scaffold distance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from hcmp.data.scaffold_distance.data_types import MoleculeTable


def smiles_to_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    return mol


def canonicalize_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(smiles_to_mol(smiles), canonical=True)


def load_molecule_table(
    source: str | Path | pd.DataFrame,
    smiles_column: str = "smiles",
    invalid_smiles: str = "raise",
    sort_by_canonical_smiles: bool = True,
) -> MoleculeTable:
    """Load and canonicalize a molecule table."""

    if invalid_smiles not in {"raise", "drop"}:
        raise ValueError("invalid_smiles must be either 'raise' or 'drop'.")
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        source_label = "<dataframe>"
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {source_path}")
        df = pd.read_csv(source_path)
        source_label = str(source_path)
    if smiles_column not in df.columns:
        raise ValueError(f"Input data is missing required column {smiles_column!r}.")

    records: list[dict[str, Any]] = []
    mols: list[Chem.Mol] = []
    dropped: list[int] = []
    for row_position, raw_value in enumerate(df[smiles_column].tolist()):
        smiles = str(raw_value).strip()
        if not smiles or smiles.lower() in {"nan", "none", "null"}:
            if invalid_smiles == "raise":
                raise ValueError(f"Row {row_position} has an empty or missing SMILES value.")
            dropped.append(row_position)
            continue
        try:
            mol = smiles_to_mol(smiles)
        except ValueError as exc:
            if invalid_smiles == "raise":
                raise ValueError(f"Row {row_position} has invalid SMILES: {smiles!r}") from exc
            dropped.append(row_position)
            continue
        record = df.iloc[row_position].to_dict()
        record["source_row_index"] = row_position
        record["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
        records.append(record)
        mols.append(mol)
    if not records:
        raise ValueError("No valid molecules were retained after SMILES parsing.")

    cleaned = pd.DataFrame(records)
    if sort_by_canonical_smiles:
        cleaned = cleaned.sort_values(
            by=["canonical_smiles", "source_row_index"],
            kind="mergesort",
        ).reset_index(drop=True)
        source_order = cleaned["source_row_index"].tolist()
        row_to_position = {
            int(record["source_row_index"]): position
            for position, record in enumerate(records)
        }
        mols = [mols[row_to_position[int(idx)]] for idx in source_order]
    return MoleculeTable(
        dataframe=cleaned,
        smiles_column=smiles_column,
        canonical_smiles=tuple(cleaned["canonical_smiles"].tolist()),
        mols=tuple(mols),
        source=source_label,
        dropped_invalid_indices=tuple(sorted(dropped)),
    )


def save_distance_matrix(matrix: np.ndarray, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(matrix))
    return output_path


def load_distance_matrix(path: str | Path, dtype: np.dtype | None = None) -> np.ndarray:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Distance matrix cache does not exist: {input_path}")
    matrix = np.load(input_path)
    return matrix.astype(dtype, copy=False) if dtype is not None else matrix


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return output_path


def load_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_distance_cache(
    matrix: np.ndarray,
    matrix_path: str | Path,
    metadata: dict[str, Any] | None = None,
    metadata_path: str | Path | None = None,
) -> None:
    save_distance_matrix(matrix, matrix_path)
    if metadata is not None and metadata_path is not None:
        save_json(metadata, metadata_path)


def load_distance_cache(
    matrix_path: str | Path,
    metadata_path: str | Path | None = None,
    dtype: np.dtype | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    matrix = load_distance_matrix(matrix_path, dtype=dtype)
    metadata = load_json(metadata_path) if metadata_path is not None and Path(metadata_path).exists() else None
    return matrix, metadata
