"""Visualization and table exports for HCMP v1 segmentation results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from hcmp.chemistry.segmentation import BondLabel, SegmentationResult


CUT_BOND_COLOR = (0.90, 0.10, 0.12)


def get_cut_bond_indices(mol: Chem.Mol, cut_labels: list[BondLabel]) -> list[int]:
    """Return deterministic RDKit bond indices whose final cut label is 1."""

    valid_bond_indices = {bond.GetIdx() for bond in mol.GetBonds()}
    cut_bond_indices = [
        label.bond_idx
        for label in cut_labels
        if label.cut_label == 1
    ]
    invalid_bond_indices = sorted(set(cut_bond_indices) - valid_bond_indices)
    if invalid_bond_indices:
        raise ValueError(
            "Cut bond labels reference bond indices not present in drawing molecule: "
            f"{invalid_bond_indices}"
        )
    return sorted(cut_bond_indices)


def draw_segmentation_result(
    result: SegmentationResult,
    output_path: str,
    image_size: tuple[int, int] = (900, 500),
    draw_atom_indices: bool = True,
) -> None:
    """Draw a segmentation result as PNG or SVG with only cut bonds highlighted."""

    mol = Chem.Mol(result.mol)
    AllChem.Compute2DCoords(mol)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".svg":
        drawer = Draw.MolDraw2DSVG(*image_size)
    elif suffix == ".png":
        drawer = Draw.MolDraw2DCairo(*image_size)
    else:
        raise ValueError("output_path must end with .png or .svg")

    options = drawer.drawOptions()
    options.addAtomIndices = draw_atom_indices
    options.highlightBondWidthMultiplier = 5
    options.padding = 0.08
    if hasattr(options, "fillHighlights"):
        options.fillHighlights = False

    cut_bond_indices = get_cut_bond_indices(mol, result.bond_labels)
    bond_colors = {bond_idx: CUT_BOND_COLOR for bond_idx in cut_bond_indices}

    drawer.DrawMolecule(
        mol,
        highlightAtoms=[],
        highlightBonds=cut_bond_indices,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    drawing = drawer.GetDrawingText()
    if suffix == ".svg":
        output.write_text(drawing, encoding="utf-8")
    else:
        output.write_bytes(drawing)


def save_bond_label_table(result: SegmentationResult, output_csv: str) -> None:
    """Save per-bond cut labels and segment boundary reasons as CSV."""

    columns = [
        "smiles",
        "canonical_smiles",
        "bond_idx",
        "begin_atom_idx",
        "end_atom_idx",
        "cut_label",
        "reason",
        "begin_segment_id",
        "end_segment_id",
        "begin_segment_type",
        "end_segment_type",
    ]
    rows = [
        {
            "smiles": result.smiles,
            "canonical_smiles": result.canonical_smiles,
            "bond_idx": label.bond_idx,
            "begin_atom_idx": label.begin_atom_idx,
            "end_atom_idx": label.end_atom_idx,
            "cut_label": label.cut_label,
            "reason": label.reason,
            "begin_segment_id": label.begin_segment_id,
            "end_segment_id": label.end_segment_id,
            "begin_segment_type": label.begin_segment_type,
            "end_segment_type": label.end_segment_type,
        }
        for label in result.bond_labels
    ]
    _write_csv(rows, output_csv, columns)


def save_segment_table(result: SegmentationResult, output_csv: str) -> None:
    """Save segment assignments as CSV."""

    columns = [
        "smiles",
        "canonical_smiles",
        "segment_id",
        "segment_type",
        "priority",
        "atom_indices",
        "bond_indices",
        "num_atoms",
        "num_bonds",
        "reason",
    ]
    rows = [
        {
            "smiles": result.smiles,
            "canonical_smiles": result.canonical_smiles,
            "segment_id": segment.segment_id,
            "segment_type": segment.segment_type,
            "priority": segment.priority,
            "atom_indices": " ".join(map(str, sorted(segment.atom_indices))),
            "bond_indices": " ".join(map(str, sorted(segment.bond_indices))),
            "num_atoms": len(segment.atom_indices),
            "num_bonds": len(segment.bond_indices),
            "reason": segment.reason,
        }
        for segment in result.segments
    ]
    _write_csv(rows, output_csv, columns)


def _write_csv(rows: list[dict[str, object]], output_csv: str, columns: list[str]) -> None:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
