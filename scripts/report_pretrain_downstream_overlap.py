#!/usr/bin/env python
"""Report canonical-SMILES overlap between pretraining and downstream sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-csv", required=True)
    parser.add_argument("--downstream-root", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    pretrain = pd.read_csv(args.pretrain_csv)
    pretrain_smiles = _canonical_set(pretrain, _detect_smiles(pretrain))
    rows = []
    root = Path(args.downstream_root)
    for path in sorted(root.glob("*.csv")) + sorted(root.glob("*/*.csv")):
        dataset = path.parent.name if path.parent != root else path.stem
        frame = pd.read_csv(path)
        smiles_column = _detect_smiles(frame)
        downstream_smiles = _canonical_set(frame, smiles_column)
        overlap = len(downstream_smiles & pretrain_smiles)
        rows.append(
            {
                "dataset": dataset,
                "n_downstream_molecules": len(downstream_smiles),
                "n_overlap_with_pretrain": overlap,
                "overlap_ratio": overlap / len(downstream_smiles) if downstream_smiles else 0.0,
            }
        )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("dataset").to_csv(output, index=False)
    print(f"Wrote overlap report to {output}")


def _canonical_set(frame: pd.DataFrame, smiles_column: str) -> set[str]:
    values = set()
    for smiles in frame[smiles_column].dropna().astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            values.add(Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True))
    return values


def _detect_smiles(frame: pd.DataFrame) -> str:
    for column in ["smiles", "SMILES", "canonical_smiles", "mol"]:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not detect SMILES column; columns={list(frame.columns)}")


if __name__ == "__main__":
    main()
