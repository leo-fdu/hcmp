#!/usr/bin/env python
"""Run HCMP v1 segmentation over the BBBP SMILES dataset."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.chemistry.segmentation import segment_mol
from hcmp.utils.io import ensure_dir


DEFAULT_INPUT_CSV = "data/bbbp_smiles.csv"
DEFAULT_OUTPUT_DIR = "data/segmentation/bbbp"
SMILES_COLUMN_CANDIDATES = ["smiles", "SMILES", "Smile", "canonical_smiles", "mol"]
ID_COLUMN_CANDIDATES = ["mol_id", "id", "ID", "name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-molecules", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = ensure_dir(args.output_dir)

    frame = pd.read_csv(input_path)
    smiles_column = args.smiles_column or _detect_column(
        frame,
        SMILES_COLUMN_CANDIDATES,
        "SMILES",
    )
    if args.smiles_column is not None and args.smiles_column not in frame.columns:
        raise ValueError(
            f"SMILES column {args.smiles_column!r} not found. "
            f"Available columns: {list(frame.columns)}"
        )
    id_column = _detect_optional_column(frame, ID_COLUMN_CANDIDATES)

    if args.max_molecules is not None:
        frame = frame.head(args.max_molecules)

    molecule_rows: list[dict[str, Any]] = []
    bond_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for row_index, row in tqdm(
        frame.iterrows(),
        total=len(frame),
        desc="Segmenting BBBP molecules",
    ):
        mol_id = _get_mol_id(row, row_index, id_column)
        input_smiles = row[smiles_column]
        if pd.isna(input_smiles) or not str(input_smiles).strip():
            _record_failure(
                molecule_rows,
                failed_rows,
                mol_id,
                "",
                "invalid_smiles",
                "SMILES is empty or missing.",
            )
            continue

        input_smiles = str(input_smiles).strip()
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            _record_failure(
                molecule_rows,
                failed_rows,
                mol_id,
                input_smiles,
                "invalid_smiles",
                "RDKit could not parse SMILES.",
            )
            continue

        try:
            result = segment_mol(mol, smiles=input_smiles)
        except Exception as exc:
            _record_failure(
                molecule_rows,
                failed_rows,
                mol_id,
                input_smiles,
                "segmentation_error",
                str(exc),
            )
            continue

        num_bonds = result.mol.GetNumBonds()
        num_cut_bonds = sum(label.cut_label for label in result.bond_labels)
        cut_bond_ratio = num_cut_bonds / num_bonds if num_bonds else 0.0

        molecule_rows.append(
            {
                "mol_id": mol_id,
                "input_smiles": input_smiles,
                "canonical_smiles": result.canonical_smiles,
                "num_atoms": result.mol.GetNumAtoms(),
                "num_bonds": num_bonds,
                "num_segments": len(result.segments),
                "num_cut_bonds": num_cut_bonds,
                "cut_bond_ratio": cut_bond_ratio,
                "is_unmarked": num_cut_bonds == 0,
                "status": "success",
                "error": "",
            }
        )

        bond_rows.extend(_build_bond_rows(mol_id, result))
        segment_rows.extend(_build_segment_rows(mol_id, result))

    summary_rows = [
        _build_summary_row(
            dataset="BBBP",
            input_path=str(input_path),
            total_input_molecules=len(frame),
            molecule_rows=molecule_rows,
        )
    ]

    _write_outputs(
        output_dir=output_dir,
        molecule_rows=molecule_rows,
        bond_rows=bond_rows,
        segment_rows=segment_rows,
        summary_rows=summary_rows,
        failed_rows=failed_rows,
    )

    summary = summary_rows[0]
    print("BBBP segmentation complete.")
    print(
        "Successful molecules: "
        f"{summary['successful_molecules']} / {summary['total_input_molecules']}"
    )
    print(f"Total bonds: {summary['total_bonds']}")
    print(f"Total cut bonds: {summary['total_cut_bonds']}")
    print(
        "Segmentation boundary bond ratio: "
        f"{summary['segmentation_boundary_bond_ratio']:.4f}"
    )
    print(f"Unmarked molecules: {summary['unmarked_molecules']}")
    print(f"Unmarked molecule ratio: {summary['unmarked_molecule_ratio']:.4f}")


def _detect_column(frame: pd.DataFrame, candidates: list[str], description: str) -> str:
    column = _detect_optional_column(frame, candidates)
    if column is None:
        raise ValueError(
            f"Could not auto-detect {description} column. "
            f"Tried {candidates}. Available columns: {list(frame.columns)}"
        )
    return column


def _detect_optional_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _get_mol_id(row: pd.Series, row_index: int, id_column: str | None) -> Any:
    if id_column is None:
        return row_index
    value = row[id_column]
    if pd.isna(value):
        return row_index
    return value


def _record_failure(
    molecule_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    mol_id: Any,
    input_smiles: str,
    status: str,
    error: str,
) -> None:
    molecule_rows.append(
        {
            "mol_id": mol_id,
            "input_smiles": input_smiles,
            "canonical_smiles": "",
            "num_atoms": None,
            "num_bonds": None,
            "num_segments": None,
            "num_cut_bonds": None,
            "cut_bond_ratio": None,
            "is_unmarked": None,
            "status": status,
            "error": error,
        }
    )
    failed_rows.append(
        {
            "mol_id": mol_id,
            "input_smiles": input_smiles,
            "status": status,
            "error": error,
        }
    )


def _build_bond_rows(mol_id: Any, result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in result.bond_labels:
        bond = result.mol.GetBondWithIdx(label.bond_idx)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        rows.append(
            {
                "mol_id": mol_id,
                "canonical_smiles": result.canonical_smiles,
                "bond_idx": label.bond_idx,
                "begin_atom_idx": label.begin_atom_idx,
                "end_atom_idx": label.end_atom_idx,
                "begin_atomic_num": begin_atom.GetAtomicNum(),
                "end_atomic_num": end_atom.GetAtomicNum(),
                "bond_type": str(bond.GetBondType()),
                "is_aromatic": bond.GetIsAromatic(),
                "is_conjugated": bond.GetIsConjugated(),
                "is_in_ring": bond.IsInRing(),
                "cut_label": label.cut_label,
                "begin_segment_id": label.begin_segment_id,
                "end_segment_id": label.end_segment_id,
                "begin_segment_type": label.begin_segment_type,
                "end_segment_type": label.end_segment_type,
                "reason": label.reason,
            }
        )
    return rows


def _build_segment_rows(mol_id: Any, result: Any) -> list[dict[str, Any]]:
    return [
        {
            "mol_id": mol_id,
            "segment_id": segment.segment_id,
            "segment_type": segment.segment_type,
            "priority": segment.priority,
            "atom_indices": " ".join(map(str, sorted(segment.atom_indices))),
            "bond_indices": " ".join(map(str, sorted(segment.bond_indices))),
            "reason": segment.reason,
        }
        for segment in result.segments
    ]


def _build_summary_row(
    dataset: str,
    input_path: str,
    total_input_molecules: int,
    molecule_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_rows = [row for row in molecule_rows if row["status"] == "success"]
    invalid_smiles = sum(row["status"] == "invalid_smiles" for row in molecule_rows)
    segmentation_errors = sum(
        row["status"] == "segmentation_error" for row in molecule_rows
    )
    total_bonds = sum(int(row["num_bonds"]) for row in successful_rows)
    total_cut_bonds = sum(int(row["num_cut_bonds"]) for row in successful_rows)
    unmarked_molecules = sum(bool(row["is_unmarked"]) for row in successful_rows)
    cut_counts = [int(row["num_cut_bonds"]) for row in successful_rows]
    cut_ratios = [float(row["cut_bond_ratio"]) for row in successful_rows]
    segment_counts = [int(row["num_segments"]) for row in successful_rows]
    successful_count = len(successful_rows)

    return {
        "dataset": dataset,
        "input_path": input_path,
        "total_input_molecules": total_input_molecules,
        "successful_molecules": successful_count,
        "invalid_smiles": invalid_smiles,
        "segmentation_errors": segmentation_errors,
        "total_bonds": total_bonds,
        "total_cut_bonds": total_cut_bonds,
        "segmentation_boundary_bond_ratio": (
            total_cut_bonds / total_bonds if total_bonds else 0.0
        ),
        "unmarked_molecules": unmarked_molecules,
        "unmarked_molecule_ratio": (
            unmarked_molecules / successful_count if successful_count else 0.0
        ),
        "mean_cut_bonds_per_molecule": _mean(cut_counts),
        "median_cut_bonds_per_molecule": _median(cut_counts),
        "mean_cut_bond_ratio_per_molecule": _mean(cut_ratios),
        "median_cut_bond_ratio_per_molecule": _median(cut_ratios),
        "mean_segments_per_molecule": _mean(segment_counts),
        "median_segments_per_molecule": _median(segment_counts),
    }


def _mean(values: list[int] | list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _median(values: list[int] | list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _write_outputs(
    output_dir: Path,
    molecule_rows: list[dict[str, Any]],
    bond_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(molecule_rows, columns=_molecule_columns()).to_csv(
        output_dir / "molecules.csv",
        index=False,
    )
    pd.DataFrame(bond_rows, columns=_bond_columns()).to_csv(
        output_dir / "bond_labels.csv",
        index=False,
    )
    pd.DataFrame(segment_rows, columns=_segment_columns()).to_csv(
        output_dir / "segments.csv",
        index=False,
    )
    pd.DataFrame(summary_rows, columns=_summary_columns()).to_csv(
        output_dir / "summary.csv",
        index=False,
    )
    pd.DataFrame(failed_rows, columns=_failed_columns()).to_csv(
        output_dir / "failed_smiles.csv",
        index=False,
    )


def _molecule_columns() -> list[str]:
    return [
        "mol_id",
        "input_smiles",
        "canonical_smiles",
        "num_atoms",
        "num_bonds",
        "num_segments",
        "num_cut_bonds",
        "cut_bond_ratio",
        "is_unmarked",
        "status",
        "error",
    ]


def _bond_columns() -> list[str]:
    return [
        "mol_id",
        "canonical_smiles",
        "bond_idx",
        "begin_atom_idx",
        "end_atom_idx",
        "begin_atomic_num",
        "end_atomic_num",
        "bond_type",
        "is_aromatic",
        "is_conjugated",
        "is_in_ring",
        "cut_label",
        "begin_segment_id",
        "end_segment_id",
        "begin_segment_type",
        "end_segment_type",
        "reason",
    ]


def _segment_columns() -> list[str]:
    return [
        "mol_id",
        "segment_id",
        "segment_type",
        "priority",
        "atom_indices",
        "bond_indices",
        "reason",
    ]


def _summary_columns() -> list[str]:
    return [
        "dataset",
        "input_path",
        "total_input_molecules",
        "successful_molecules",
        "invalid_smiles",
        "segmentation_errors",
        "total_bonds",
        "total_cut_bonds",
        "segmentation_boundary_bond_ratio",
        "unmarked_molecules",
        "unmarked_molecule_ratio",
        "mean_cut_bonds_per_molecule",
        "median_cut_bonds_per_molecule",
        "mean_cut_bond_ratio_per_molecule",
        "median_cut_bond_ratio_per_molecule",
        "mean_segments_per_molecule",
        "median_segments_per_molecule",
    ]


def _failed_columns() -> list[str]:
    return ["mol_id", "input_smiles", "status", "error"]


if __name__ == "__main__":
    main()
