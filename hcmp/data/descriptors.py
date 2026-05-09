"""HCMP descriptor computation for threshold pairwise ranking."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.EState.EState import EStateIndices
from tqdm import tqdm


DESCRIPTOR_NAMES = [
    "MolLogP",
    "HydrophobicContributionFraction",
    "MeanCrippenMR",
    "TPSA",
    "LabuteASA",
    "MeanAbsGasteigerCharge",
    "GasteigerChargeVariance",
    "NegativeChargeFraction",
    "PositiveChargeFraction",
    "MeanAbsEState",
    "EStateVariance",
    "NegativeEStateFraction",
    "PositiveEStateFraction",
    "RotatableBondFraction",
]


def compute_descriptor_values(
    mol: Chem.Mol,
    epsilon: float = 1.0e-4,
) -> dict[str, float]:
    """Compute the 14 HCMP descriptor values for one molecule."""

    work_mol = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    heavy_count = max(work_mol.GetNumHeavyAtoms(), 1)
    crippen_contribs = Crippen._GetAtomContribs(work_mol)
    logp_contribs = np.array([item[0] for item in crippen_contribs], dtype=float)
    mr_contribs = np.array([item[1] for item in crippen_contribs], dtype=float)
    charges = np.array(_gasteiger_charges(work_mol), dtype=float)
    estate = np.array(EStateIndices(work_mol), dtype=float)

    return {
        "MolLogP": float(Crippen.MolLogP(work_mol)),
        "HydrophobicContributionFraction": _fraction(logp_contribs > epsilon, heavy_count),
        "MeanCrippenMR": float(mr_contribs.mean()) if mr_contribs.size else 0.0,
        "TPSA": float(rdMolDescriptors.CalcTPSA(work_mol, includeSandP=True)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(work_mol)),
        "MeanAbsGasteigerCharge": float(np.abs(charges).mean()) if charges.size else 0.0,
        "GasteigerChargeVariance": float(charges.var()) if charges.size else 0.0,
        "NegativeChargeFraction": _fraction(charges < -epsilon, heavy_count),
        "PositiveChargeFraction": _fraction(charges > epsilon, heavy_count),
        "MeanAbsEState": float(np.abs(estate).mean()) if estate.size else 0.0,
        "EStateVariance": float(estate.var()) if estate.size else 0.0,
        "NegativeEStateFraction": _fraction(estate < -epsilon, heavy_count),
        "PositiveEStateFraction": _fraction(estate > epsilon, heavy_count),
        "RotatableBondFraction": (
            float(Descriptors.NumRotatableBonds(work_mol)) / float(heavy_count)
        ),
    }


def compute_descriptor_table(
    smiles_rows: list[tuple[Any, str]],
    epsilon: float = 1.0e-4,
    num_workers: int = 1,
    chunksize: int = 256,
) -> pd.DataFrame:
    """Compute descriptor rows with stable status/error fields."""

    include_source_row_index = any(len(row) >= 3 for row in smiles_rows)
    progress_kwargs = {
        "total": len(smiles_rows),
        "desc": "Computing descriptors",
    }
    if num_workers <= 1:
        rows = [
            compute_descriptor_row(item, epsilon)
            for item in tqdm(smiles_rows, **progress_kwargs)
        ]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            rows = list(
                tqdm(
                    executor.map(
                        compute_descriptor_row,
                        smiles_rows,
                        repeat(epsilon),
                        chunksize=chunksize,
                    ),
                    **progress_kwargs,
                )
            )
    columns = ["mol_id"]
    if include_source_row_index:
        columns.append("source_row_index")
    columns.extend(["canonical_smiles", *DESCRIPTOR_NAMES, "status", "error"])
    return pd.DataFrame(rows, columns=columns)


def compute_descriptor_row(smiles_row: tuple[Any, ...], epsilon: float) -> dict[str, Any]:
    """Compute one descriptor output row.

    This function is intentionally top-level so ProcessPoolExecutor can pickle it.
    """

    mol_id, smiles = smiles_row[0], smiles_row[1]
    source_row_index = smiles_row[2] if len(smiles_row) >= 3 else mol_id
    row: dict[str, Any] = {
        "mol_id": mol_id,
        "source_row_index": source_row_index,
        "canonical_smiles": "",
        **{name: np.nan for name in DESCRIPTOR_NAMES},
        "status": "success",
        "error": "",
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            row["status"] = "invalid_smiles"
            row["error"] = "RDKit could not parse SMILES."
            return row
        work_mol = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
        row["canonical_smiles"] = Chem.MolToSmiles(work_mol, canonical=True)
        row.update(compute_descriptor_values(work_mol, epsilon=epsilon))
    except Exception as exc:
        row["status"] = "descriptor_error"
        row["error"] = str(exc)
    return row


def _gasteiger_charges(mol: Chem.Mol) -> list[float]:
    work_mol = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(work_mol)
    values: list[float] = []
    for atom in work_mol.GetAtoms():
        raw = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0"
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            value = 0.0
        values.append(value)
    return values


def _fraction(mask: np.ndarray, denominator: int) -> float:
    return float(mask.sum()) / float(max(denominator, 1))
