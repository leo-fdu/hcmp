#!/usr/bin/env python
"""Compute HCMP descriptor values from a SMILES CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.data.descriptors import DESCRIPTOR_NAMES, compute_descriptor_table
from hcmp.data.molecule_table import load_hcmp_molecule_table
from hcmp.utils.io import ensure_dir


SMILES_COLUMNS = ["smiles", "SMILES", "Smile", "canonical_smiles", "mol"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="data/bbbp_smiles.csv")
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--output-csv", default="data/processed/descriptor_values.csv")
    parser.add_argument("--epsilon", type=float, default=1.0e-4)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=256)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    smiles_column = args.smiles_column or _detect_column(frame, SMILES_COLUMNS, "SMILES")
    molecule_table = load_hcmp_molecule_table(
        args.input_csv,
        smiles_column=smiles_column,
        sort_by_canonical_smiles=True,
        max_molecules=args.max_molecules,
        strict=False,
    )
    rows = [
        (
            idx,
            str(molecule_table.dataframe.iloc[idx][smiles_column]),
            int(molecule_table.dataframe.iloc[idx]["source_row_index"]),
        )
        for idx in range(molecule_table.size)
    ]
    output = Path(args.output_csv)
    ensure_dir(output.parent)
    table = compute_descriptor_table(
        list(tqdm(rows, desc="Preparing descriptor inputs")),
        epsilon=args.epsilon,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )
    table.to_csv(output, index=False)
    print(f"Wrote descriptor values to {output}")
    print(f"Descriptor columns: {', '.join(DESCRIPTOR_NAMES)}")
    print(
        "Descriptor rows are sorted by canonical_smiles/source_row_index to match "
        "GraphDataset training order. TPSA uses includeSandP=True; regenerate "
        "descriptor thresholds together with this file."
    )


def _detect_column(frame: pd.DataFrame, candidates: list[str], description: str) -> str:
    column = _detect_optional_column(frame, candidates)
    if column is None:
        raise ValueError(
            f"Could not detect {description} column. "
            f"Tried {candidates}; available columns: {list(frame.columns)}"
        )
    return column


def _detect_optional_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


if __name__ == "__main__":
    main()
