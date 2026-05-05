#!/usr/bin/env python
"""Run HCMP v1 segmentation on sanity checks or a SMILES CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.chemistry.segmentation import segment_molecule
from hcmp.utils.io import ensure_dir, load_yaml, slugify
from hcmp.visualization.visualize_segmentation import (
    draw_segmentation_result,
    save_bond_label_table,
    save_segment_table,
)


SANITY_CHECKS = {
    "ethylene": "C=C",
    "butadiene": "C=CC=C",
    "allene": "C=C=C",
    "propyne": "CC#C",
    "acetone": "CC(=O)C",
    "acetamide": "CC(=O)N",
    "methyl_acetate": "CC(=O)OC",
    "enone": "CC=CC(=O)C",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "aniline": "Nc1ccccc1",
    "phenol": "Oc1ccccc1",
    "styrene": "C=Cc1ccccc1",
    "stilbene": "c1ccccc1C=Cc2ccccc2",
    "biphenyl": "c1ccccc1-c2ccccc2",
    "pyridine": "c1ccncc1",
    "thiophene": "c1ccsc1",
    "indole": "c1ccc2[nH]ccc2c1",
    "decalin": "C1CCC2CCCCC2C1",
    "chlorobenzene": "Clc1ccccc1",
    "ethanol": "CCO",
    "diethyl_ether": "CCOCC",
    "triethylamine": "CCN(CC)CC",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--input-csv", default=None, help="Optional CSV containing SMILES.")
    parser.add_argument("--smiles-column", default=None, help="Override config SMILES column.")
    parser.add_argument("--output-dir", default=None, help="Override config output directory.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = ensure_dir(args.output_dir or config.get("output_dir", "results/segmentation_visualization"))
    image_format = str(config.get("image_format", "png")).lower().lstrip(".")
    if image_format not in {"png", "svg"}:
        raise ValueError("image_format must be 'png' or 'svg'.")

    smiles_column = args.smiles_column or config.get("smiles_column", "smiles")
    max_molecules = config.get("max_molecules")
    draw_atom_indices = bool(config.get("draw_atom_indices", True))
    molecules = _load_molecules(args.input_csv, smiles_column)
    if max_molecules is not None:
        molecules = molecules[: int(max_molecules)]

    summary_rows = []
    for idx, (name, smiles) in enumerate(tqdm(molecules, desc="Segmenting molecules")):
        stem = f"{idx:04d}_{slugify(name)}"
        image_path = output_dir / f"{stem}.{image_format}"
        bond_csv = output_dir / f"{stem}_bond_labels.csv"
        segment_csv = output_dir / f"{stem}_segments.csv"

        try:
            result = segment_molecule(smiles)
            draw_segmentation_result(
                result,
                str(image_path),
                draw_atom_indices=draw_atom_indices,
            )
            save_bond_label_table(result, str(bond_csv))
            save_segment_table(result, str(segment_csv))

            segment_counts = {
                segment_type: sum(1 for segment in result.segments if segment.segment_type == segment_type)
                for segment_type in (
                    "ring_system",
                    "unsaturated_conjugated",
                    "heteroatom_cluster",
                    "terminal_heteroatom",
                )
            }
            summary_rows.append(
                {
                    "name": name,
                    "smiles": smiles,
                    "canonical_smiles": result.canonical_smiles,
                    "num_segments": len(result.segments),
                    "num_bonds": len(result.bond_labels),
                    "num_cut_bonds": sum(label.cut_label for label in result.bond_labels),
                    "image_path": str(image_path),
                    "bond_label_csv": str(bond_csv),
                    "segment_csv": str(segment_csv),
                    "error": None,
                    **segment_counts,
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "name": name,
                    "smiles": smiles,
                    "canonical_smiles": None,
                    "num_segments": None,
                    "num_bonds": None,
                    "num_cut_bonds": None,
                    "image_path": None,
                    "bond_label_csv": None,
                    "segment_csv": None,
                    "error": str(exc),
                }
            )

    summary_path = output_dir / "segmentation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote segmentation outputs to {output_dir}")
    print(f"Wrote summary CSV to {summary_path}")


def _load_molecules(input_csv: str | None, smiles_column: str) -> list[tuple[str, str]]:
    if input_csv is None:
        return list(SANITY_CHECKS.items())

    frame = pd.read_csv(input_csv)
    if smiles_column not in frame.columns:
        raise ValueError(f"SMILES column {smiles_column!r} not found in {input_csv}.")

    name_column = "name" if "name" in frame.columns else None
    molecules: list[tuple[str, str]] = []
    for idx, row in frame.iterrows():
        smiles = row[smiles_column]
        if pd.isna(smiles):
            continue
        name = str(row[name_column]) if name_column else f"molecule_{idx}"
        molecules.append((name, str(smiles)))
    return molecules


if __name__ == "__main__":
    main()
