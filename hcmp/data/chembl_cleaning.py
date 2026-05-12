"""ChEMBL molecule cleaning utilities for HCMP pretraining."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, rdBase

try:
    from hcmp.data.graph_builder import (
        BOND_STEREO_VALUES,
        BOND_TYPES,
        CHIRALITY_VALUES,
        DEFAULT_ATOMIC_NUMBERS,
        FeatureSpec,
        default_feature_spec,
    )
except ModuleNotFoundError:  # pragma: no cover - lets non-torch cleaning tests import.
    FeatureSpec = Any
    DEFAULT_ATOMIC_NUMBERS = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    CHIRALITY_VALUES = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    BOND_TYPES = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    BOND_STEREO_VALUES = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]

    def default_feature_spec(config: dict[str, Any] | None = None):
        config = config or {}
        return type(
            "FeatureSpecFallback",
            (),
            {
                "atomic_numbers": list(config.get("atomic_numbers", DEFAULT_ATOMIC_NUMBERS)),
                "formal_charge_clip": int(config.get("formal_charge_clip", 3)),
            },
        )()


DEFAULT_ALLOWED_ELEMENTS = [
    Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
    for atomic_num in DEFAULT_ATOMIC_NUMBERS
    if atomic_num > 1
]
CLEANED_COLUMNS = [
    "canonical_smiles",
    "source_smiles",
    "source_chembl_id",
    "source_row_index",
    "heavy_atom_count",
    "total_atom_count",
    "total_formal_charge",
    "num_fragments_original",
    "selected_fragment_smiles",
    "was_multifragment",
]


@dataclass
class CleaningConfig:
    allowed_elements: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ELEMENTS))
    min_heavy_atoms: int = 3
    max_heavy_atoms: int = 100
    deduplicate: bool = False
    feature_spec: FeatureSpec = field(default_factory=default_feature_spec)


@dataclass
class CleaningCounters:
    n_input_rows: int = 0
    n_missing_smiles: int = 0
    n_rdkit_parse_failed: int = 0
    n_sanitize_failed: int = 0
    n_multifragment_input: int = 0
    n_after_fragment_selection: int = 0
    n_failed_after_fragment_selection: int = 0
    n_unsupported_elements: int = 0
    n_size_filter_failed: int = 0
    n_featurizer_compat_failed: int = 0
    n_duplicates_removed: int = 0
    n_cleaned_final: int = 0
    formal_charge_histogram: Counter = field(default_factory=Counter)
    heavy_atom_counts: list[int] = field(default_factory=list)
    fragment_count_histogram: Counter = field(default_factory=Counter)
    element_count_histogram: Counter = field(default_factory=Counter)
    failure_reason_histogram: Counter = field(default_factory=Counter)


def clean_chembl_rows(
    rows: Iterable[dict[str, Any]],
    config: CleaningConfig,
) -> tuple[list[dict[str, Any]], CleaningCounters]:
    """Clean ChEMBL-like molecule rows into HCMP-compatible SMILES rows."""

    counters = CleaningCounters()
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        result = clean_one_molecule_row(row, config, counters)
        if result is not None:
            cleaned.append(result)

    cleaned.sort(key=lambda item: (str(item["canonical_smiles"]), int(item["source_row_index"])))
    if config.deduplicate:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in cleaned:
            smiles = str(row["canonical_smiles"])
            if smiles in seen:
                counters.n_duplicates_removed += 1
                continue
            seen.add(smiles)
            unique.append(row)
        cleaned = unique
    counters.n_cleaned_final = len(cleaned)
    _refresh_final_distribution_counters(cleaned, counters)
    return cleaned, counters


def clean_one_molecule_row(
    row: dict[str, Any],
    config: CleaningConfig,
    counters: CleaningCounters | None = None,
) -> dict[str, Any] | None:
    """Clean one row and return the output schema row, or ``None`` when dropped."""

    counters = counters or CleaningCounters()
    counters.n_input_rows += 1
    source_smiles = "" if row.get("source_smiles") is None else str(row.get("source_smiles", "")).strip()
    if not source_smiles or source_smiles.lower() == "nan":
        counters.n_missing_smiles += 1
        counters.failure_reason_histogram["missing_smiles"] += 1
        return None

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(source_smiles, sanitize=False)
    if mol is None:
        counters.n_rdkit_parse_failed += 1
        counters.failure_reason_histogram["rdkit_parse_failed"] += 1
        return None
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(mol)
    except Exception:
        counters.n_sanitize_failed += 1
        counters.failure_reason_histogram["sanitize_failed"] += 1
        return None

    fragment_sets = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    num_fragments = len(fragment_sets)
    counters.fragment_count_histogram[str(num_fragments)] += 1
    if num_fragments > 1:
        counters.n_multifragment_input += 1
    try:
        selected = select_main_organic_fragment(
            mol,
            allowed_elements=set(config.allowed_elements),
        )
        counters.n_after_fragment_selection += 1
    except Exception:
        counters.n_failed_after_fragment_selection += 1
        counters.failure_reason_histogram["fragment_selection_failed"] += 1
        return None

    selected_fragment_smiles = Chem.MolToSmiles(selected, canonical=True)
    reparsed = Chem.MolFromSmiles(selected_fragment_smiles, sanitize=True)
    if reparsed is None:
        counters.n_failed_after_fragment_selection += 1
        counters.failure_reason_histogram["canonical_reparse_failed"] += 1
        return None
    canonical_smiles = Chem.MolToSmiles(reparsed, canonical=True)

    unsupported = unsupported_element_symbols(reparsed, set(config.allowed_elements))
    if unsupported:
        counters.n_unsupported_elements += 1
        counters.failure_reason_histogram["unsupported_elements"] += 1
        return None

    heavy_atom_count = heavy_atom_count_for_mol(reparsed)
    total_atom_count = reparsed.GetNumAtoms()
    if heavy_atom_count < config.min_heavy_atoms or heavy_atom_count > config.max_heavy_atoms:
        counters.n_size_filter_failed += 1
        counters.failure_reason_histogram["size_filter_failed"] += 1
        return None

    compatibility = check_hcmp_featurizer_compatibility(reparsed, config.feature_spec)
    if not compatibility.ok:
        counters.n_featurizer_compat_failed += 1
        counters.failure_reason_histogram[f"featurizer_{compatibility.reason}"] += 1
        return None

    total_charge = total_formal_charge(reparsed)
    counters.formal_charge_histogram[str(total_charge)] += 1
    counters.heavy_atom_counts.append(heavy_atom_count)
    for atom in reparsed.GetAtoms():
        if atom.GetAtomicNum() > 1:
            counters.element_count_histogram[atom.GetSymbol()] += 1

    return {
        "canonical_smiles": canonical_smiles,
        "source_smiles": source_smiles,
        "source_chembl_id": "" if row.get("source_chembl_id") is None else str(row.get("source_chembl_id", "")),
        "source_row_index": int(row.get("source_row_index", counters.n_input_rows - 1)),
        "heavy_atom_count": int(heavy_atom_count),
        "total_atom_count": int(total_atom_count),
        "total_formal_charge": int(total_charge),
        "num_fragments_original": int(num_fragments),
        "selected_fragment_smiles": selected_fragment_smiles,
        "was_multifragment": bool(num_fragments > 1),
    }


def select_main_organic_fragment(
    mol: Chem.Mol,
    allowed_elements: set[str] | None = None,
) -> Chem.Mol:
    """Select one deterministic main organic fragment without mutating ``mol``."""

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Expected a non-empty RDKit molecule.")
    allowed_elements = allowed_elements or set(DEFAULT_ALLOWED_ELEMENTS)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        raise ValueError("Could not extract molecule fragments.")
    if len(fragments) == 1:
        return Chem.Mol(fragments[0])

    def sort_key(fragment: Chem.Mol) -> tuple[int, int, int, int, str]:
        has_carbon = any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
        uses_allowed_elements = not unsupported_element_symbols(fragment, allowed_elements)
        heavy_count = heavy_atom_count_for_mol(fragment)
        total_count = fragment.GetNumAtoms()
        smiles = Chem.MolToSmiles(fragment, canonical=True)
        return (
            0 if has_carbon else 1,
            0 if uses_allowed_elements else 1,
            -heavy_count,
            -total_count,
            smiles,
        )

    selected = min(fragments, key=sort_key)
    if selected.GetNumAtoms() == 0:
        raise ValueError("Selected fragment is empty.")
    return Chem.Mol(selected)


@dataclass(frozen=True)
class FeaturizerCompatibility:
    ok: bool
    reason: str = "ok"


def check_hcmp_featurizer_compatibility(
    mol: Chem.Mol,
    feature_spec: FeatureSpec | None = None,
) -> FeaturizerCompatibility:
    """Check whether a molecule is representable by the current HCMP featurizer."""

    feature_spec = feature_spec or default_feature_spec()
    supported_atomic_numbers = set(int(value) for value in feature_spec.atomic_numbers)
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 1:
            continue
        if atomic_num not in supported_atomic_numbers:
            return FeaturizerCompatibility(False, "unsupported_atom_type")
        if atom.GetChiralTag() not in CHIRALITY_VALUES:
            return FeaturizerCompatibility(False, "unsupported_chirality")
    for bond in mol.GetBonds():
        if bond.GetBondType() not in BOND_TYPES:
            return FeaturizerCompatibility(False, "unsupported_bond_type")
        if bond.GetStereo() not in BOND_STEREO_VALUES:
            return FeaturizerCompatibility(False, "unsupported_bond_stereo")
    try:
        from hcmp.data.graph_builder import mol_to_graph

        mol_to_graph(mol, feature_spec=feature_spec)
    except ModuleNotFoundError:
        pass
    except Exception:
        return FeaturizerCompatibility(False, "graph_construction_failed")
    return FeaturizerCompatibility(True)


def unsupported_element_symbols(mol: Chem.Mol, allowed_elements: set[str]) -> set[str]:
    return {
        atom.GetSymbol()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetSymbol() not in allowed_elements
    }


def heavy_atom_count_for_mol(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def total_formal_charge(mol: Chem.Mol) -> int:
    return sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())


def cleaning_report_dict(
    counters: CleaningCounters,
    config: CleaningConfig,
) -> dict[str, Any]:
    return {
        "n_input_rows": counters.n_input_rows,
        "n_missing_smiles": counters.n_missing_smiles,
        "n_rdkit_parse_failed": counters.n_rdkit_parse_failed,
        "n_sanitize_failed": counters.n_sanitize_failed,
        "n_multifragment_input": counters.n_multifragment_input,
        "n_after_fragment_selection": counters.n_after_fragment_selection,
        "n_failed_after_fragment_selection": counters.n_failed_after_fragment_selection,
        "n_unsupported_elements": counters.n_unsupported_elements,
        "n_size_filter_failed": counters.n_size_filter_failed,
        "n_featurizer_compat_failed": counters.n_featurizer_compat_failed,
        "n_duplicates_removed": counters.n_duplicates_removed,
        "n_cleaned_final": counters.n_cleaned_final,
        "allowed_elements": list(config.allowed_elements),
        "min_heavy_atoms": config.min_heavy_atoms,
        "max_heavy_atoms": config.max_heavy_atoms,
        "deduplicate_enabled": config.deduplicate,
        "formal_charge_histogram": dict(sorted(counters.formal_charge_histogram.items())),
        "heavy_atom_count_summary": summarize_numeric(counters.heavy_atom_counts),
        "fragment_count_histogram": dict(sorted(counters.fragment_count_histogram.items())),
        "element_count_histogram": dict(sorted(counters.element_count_histogram.items())),
        "failure_reason_histogram": dict(sorted(counters.failure_reason_histogram.items())),
    }


def summarize_numeric(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def _refresh_final_distribution_counters(
    cleaned: list[dict[str, Any]],
    counters: CleaningCounters,
) -> None:
    counters.formal_charge_histogram = Counter()
    counters.heavy_atom_counts = []
    counters.element_count_histogram = Counter()
    for row in cleaned:
        counters.formal_charge_histogram[str(int(row["total_formal_charge"]))] += 1
        counters.heavy_atom_counts.append(int(row["heavy_atom_count"]))
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]), sanitize=True)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                counters.element_count_histogram[atom.GetSymbol()] += 1
