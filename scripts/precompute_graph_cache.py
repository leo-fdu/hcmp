#!/usr/bin/env python
"""Precompute sharded lightweight HCMP graph cache."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.chemistry.segmentation import segment_mol
from hcmp.data.descriptors import DESCRIPTOR_NAMES
from hcmp.data.graph_builder import default_feature_spec, mol_to_graph
from hcmp.data.graph_cache import FEATURE_VERSION, graph_to_record
from hcmp.data.molecule_table import load_hcmp_molecule_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--descriptor-values", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument(
        "--allowed-elements",
        nargs="*",
        default=["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"],
    )
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--feature-mode", choices=["hcmp", "graph_bert"], default="hcmp")
    parser.add_argument("--allow-unverified-descriptor-alignment", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv, nrows=1)
    smiles_column = args.smiles_column or _detect_smiles_column(frame)
    molecule_table = load_hcmp_molecule_table(
        args.input_csv,
        smiles_column=smiles_column,
        sort_by_canonical_smiles=True,
        max_molecules=args.max_molecules,
        strict=False,
    )
    descriptor_frame = pd.read_csv(args.descriptor_values)
    _validate_descriptor_alignment(
        molecule_table,
        descriptor_frame,
        allow_unverified=bool(args.allow_unverified_descriptor_alignment),
    )
    descriptor_frame = descriptor_frame.head(molecule_table.size).reset_index(drop=True)
    missing = [name for name in DESCRIPTOR_NAMES if name not in descriptor_frame.columns]
    if missing:
        raise ValueError(f"descriptor-values is missing columns: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_atomic_numbers = {
        Chem.GetPeriodicTable().GetAtomicNumber(symbol) for symbol in args.allowed_elements
    }
    feature_spec = default_feature_spec({"feature_mode": args.feature_mode})
    shard_size = int(args.shard_size)
    records = []
    shard_index = 0
    written = 0

    for idx in tqdm(range(molecule_table.size), desc="Caching graphs"):
        mol = molecule_table.mols[idx]
        if not _allowed_mol(mol, allowed_atomic_numbers):
            continue
        canonical_smiles = molecule_table.canonical_smiles[idx]
        source_row_index = int(molecule_table.dataframe.iloc[idx]["source_row_index"])
        result = segment_mol(mol, smiles=canonical_smiles)
        descriptor_values = [
            float(descriptor_frame.iloc[idx][descriptor_name])
            for descriptor_name in DESCRIPTOR_NAMES
        ]
        graph = mol_to_graph(
            result.mol,
            mol_id=written,
            input_smiles=canonical_smiles,
            feature_spec=feature_spec,
            cut_labels=[label.cut_label for label in result.bond_labels],
            descriptor_values=descriptor_values,
            global_idx=written,
            source_row_index=source_row_index,
        )
        records.append(graph_to_record(graph))
        written += 1
        if len(records) >= shard_size:
            _write_shard(output_dir, shard_index, records)
            shard_index += 1
            records = []
    if records:
        _write_shard(output_dir, shard_index, records)
        shard_index += 1

    manifest = {
        "input_csv": str(args.input_csv),
        "descriptor_values_path": str(args.descriptor_values),
        "n_molecules": int(written),
        "n_shards": int(shard_index),
        "shard_size": int(shard_size),
        "feature_version": FEATURE_VERSION,
        "feature_mode": args.feature_mode,
        "node_feature_dim": feature_spec.node_feature_dim,
        "edge_feature_dim": feature_spec.edge_feature_dim,
        "atom_target_fields": feature_spec.atom_target_fields,
        "bond_target_fields": feature_spec.bond_target_fields,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "allowed_elements": list(args.allowed_elements),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "feature_spec": {
            "atomic_numbers": feature_spec.atomic_numbers,
            "formal_charge_clip": feature_spec.formal_charge_clip,
            "rich_molecular_features": feature_spec.rich_molecular_features,
            "feature_mode": feature_spec.feature_mode,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote graph cache manifest to {output_dir / 'manifest.json'}")
    print(f"cached_molecules={written} n_shards={shard_index}")


def _write_shard(output_dir: Path, shard_index: int, records: list[dict]) -> None:
    import torch

    path = output_dir / f"shard_{shard_index:05d}.pt"
    torch.save({"records": records}, path)


def _allowed_mol(mol: Chem.Mol, allowed_atomic_numbers: set[int]) -> bool:
    return all(atom.GetAtomicNum() in allowed_atomic_numbers for atom in mol.GetAtoms())


def _detect_smiles_column(frame: pd.DataFrame) -> str:
    for column in ["smiles", "SMILES", "canonical_smiles", "mol"]:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not detect SMILES column from columns={list(frame.columns)}")


def _validate_descriptor_alignment(
    molecule_table,
    descriptor_frame: pd.DataFrame,
    allow_unverified: bool = False,
) -> None:
    if len(descriptor_frame) < molecule_table.size:
        raise ValueError("descriptor-values has fewer rows than the molecule table.")
    verifiable_columns = [column for column in ["canonical_smiles", "source_row_index", "source_chembl_id", "mol_id", "molecule_id"] if column in descriptor_frame.columns]
    if "canonical_smiles" not in descriptor_frame.columns:
        if allow_unverified:
            print(
                "Warning: descriptor alignment cannot be verified because canonical_smiles "
                "is missing; proceeding because --allow-unverified-descriptor-alignment was set."
            )
            return
        raise ValueError(
            "Descriptor alignment is ambiguous: descriptor-values must contain "
            "canonical_smiles, or pass --allow-unverified-descriptor-alignment explicitly."
        )
    for idx in range(molecule_table.size):
        molecule_smiles = str(molecule_table.canonical_smiles[idx])
        descriptor_smiles = str(descriptor_frame.iloc[idx]["canonical_smiles"])
        if molecule_smiles != descriptor_smiles:
            raise ValueError(
                "Descriptor values are not aligned with molecule rows: "
                f"row={idx}, molecule_canonical_smiles={molecule_smiles!r}, "
                f"descriptor_canonical_smiles={descriptor_smiles!r}."
            )
        if "source_row_index" in descriptor_frame.columns:
            molecule_source = int(molecule_table.dataframe.iloc[idx]["source_row_index"])
            descriptor_source = descriptor_frame.iloc[idx]["source_row_index"]
            if not pd.isna(descriptor_source) and molecule_source != int(descriptor_source):
                raise ValueError(
                    "Descriptor source_row_index mismatch: "
                    f"row={idx}, molecule_source_row_index={molecule_source}, "
                    f"descriptor_source_row_index={descriptor_source}."
                )
        for id_column in ["source_chembl_id", "mol_id", "molecule_id"]:
            if id_column in descriptor_frame.columns and id_column in molecule_table.dataframe.columns:
                molecule_id = str(molecule_table.dataframe.iloc[idx][id_column])
                descriptor_id = str(descriptor_frame.iloc[idx][id_column])
                if molecule_id != descriptor_id:
                    raise ValueError(
                        f"Descriptor {id_column} mismatch at row={idx}: "
                        f"molecule={molecule_id!r}, descriptor={descriptor_id!r}."
                    )
    print(
        "Verified descriptor alignment using columns: "
        + ", ".join(verifiable_columns)
    )


if __name__ == "__main__":
    main()
