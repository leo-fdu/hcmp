#!/usr/bin/env python
"""Prepare cleaned ChEMBL SMILES corpora for HCMP pretraining."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.data.chembl_cleaning import (
    CLEANED_COLUMNS,
    CleaningConfig,
    cleaning_report_dict,
    clean_chembl_rows,
)
from hcmp.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv", default=None)
    input_group.add_argument("--input-sqlite", default=None)
    parser.add_argument("--smiles-column", default="canonical_smiles")
    parser.add_argument("--id-column", default="molecule_chembl_id")
    parser.add_argument("--output-dir", default="data/chembl")
    parser.add_argument(
        "--allowed-elements",
        nargs="+",
        default=None,
        help="Allowed element symbols before featurizer compatibility checks.",
    )
    parser.add_argument("--min-heavy-atoms", type=int, default=3)
    parser.add_argument("--max-heavy-atoms", type=int, default=100)
    parser.add_argument("--subset-sizes", nargs="*", type=int, default=[100000, 500000, 1000000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--csv-chunksize", type=int, default=100000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cleaned_dir = ensure_dir(output_dir / "cleaned")
    reports_dir = ensure_dir(output_dir / "reports")
    full_cleaned_path = cleaned_dir / "chembl_clean_full.csv"

    config = CleaningConfig(
        allowed_elements=_allowed_elements(args.allowed_elements),
        min_heavy_atoms=int(args.min_heavy_atoms),
        max_heavy_atoms=int(args.max_heavy_atoms),
        deduplicate=bool(args.deduplicate),
    )
    input_path = Path(args.input_csv or args.input_sqlite)
    if args.input_csv:
        rows = _iter_csv_rows(
            Path(args.input_csv),
            smiles_column=args.smiles_column,
            id_column=args.id_column,
            chunksize=args.csv_chunksize,
        )
    else:
        rows = _iter_sqlite_rows(Path(args.input_sqlite))

    cleaned_rows, counters = clean_chembl_rows(tqdm(rows, desc="Cleaning ChEMBL rows"), config)
    cleaned_frame = pd.DataFrame(cleaned_rows, columns=CLEANED_COLUMNS)
    cleaned_frame.to_csv(full_cleaned_path, index=False)

    report = cleaning_report_dict(counters, config)
    report["input_file"] = str(input_path)
    report["full_cleaned_file"] = str(full_cleaned_path)
    report["cleaning_config"] = _cleaning_config_dict(config)
    report_path = reports_dir / "chembl_cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary_csv(report, reports_dir / "chembl_cleaning_summary.csv")

    subset_manifest = _write_subsets(
        cleaned_frame,
        cleaned_dir=cleaned_dir,
        reports_dir=reports_dir,
        subset_sizes=[int(value) for value in args.subset_sizes],
        seed=int(args.seed),
        input_file=input_path,
        full_cleaned_file=full_cleaned_path,
        config=config,
    )
    print(f"Wrote full cleaned ChEMBL corpus to {full_cleaned_path} ({len(cleaned_frame)} rows)")
    print(f"Wrote cleaning report to {report_path}")
    print(f"Wrote subset manifest to {subset_manifest}")


def _allowed_elements(values: list[str] | None) -> list[str]:
    if values is not None:
        return list(dict.fromkeys(str(value) for value in values))
    from hcmp.data.chembl_cleaning import DEFAULT_ALLOWED_ELEMENTS

    return list(DEFAULT_ALLOWED_ELEMENTS)


def _iter_csv_rows(
    path: Path,
    smiles_column: str,
    id_column: str,
    chunksize: int,
) -> Iterable[dict[str, Any]]:
    row_offset = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if smiles_column not in chunk.columns:
            raise ValueError(
                f"SMILES column {smiles_column!r} not found in {path}; "
                f"available columns={list(chunk.columns)}"
            )
        has_id = id_column in chunk.columns
        smiles_values = chunk[smiles_column].tolist()
        id_values = chunk[id_column].tolist() if has_id else [""] * len(chunk)
        for local_idx, (smiles, chembl_id) in enumerate(zip(smiles_values, id_values, strict=False)):
            yield {
                "source_smiles": smiles,
                "source_chembl_id": chembl_id,
                "source_row_index": row_offset + local_idx,
            }
        row_offset += len(chunk)


def _iter_sqlite_rows(path: Path) -> Iterable[dict[str, Any]]:
    connection = sqlite3.connect(path)
    try:
        query = _chembl_sql_query(connection)
        for source_row_index, chembl_id, smiles in connection.execute(query):
            yield {
                "source_smiles": smiles,
                "source_chembl_id": "" if chembl_id is None else str(chembl_id),
                "source_row_index": int(source_row_index),
            }
    finally:
        connection.close()


def _chembl_sql_query(connection: sqlite3.Connection) -> str:
    tables = _sqlite_tables(connection)
    if "compound_structures" in tables:
        structure_columns = _sqlite_columns(connection, "compound_structures")
        if "canonical_smiles" not in structure_columns:
            raise _sqlite_schema_error(connection, "compound_structures is missing canonical_smiles")
        id_expr = "NULL"
        join_expr = ""
        order_expr = "cs.molregno" if "molregno" in structure_columns else "cs.rowid"
        if "molregno" in structure_columns and "molecule_dictionary" in tables:
            molecule_columns = _sqlite_columns(connection, "molecule_dictionary")
            chembl_column = _first_existing(
                molecule_columns,
                ["chembl_id", "molecule_chembl_id", "mol_chembl_id"],
            )
            if chembl_column and "molregno" in molecule_columns:
                id_expr = f"md.{chembl_column}"
                join_expr = "LEFT JOIN molecule_dictionary md ON md.molregno = cs.molregno"
        return (
            "SELECT ROW_NUMBER() OVER (ORDER BY {order_expr}) - 1 AS source_row_index, "
            "{id_expr} AS source_chembl_id, cs.canonical_smiles AS source_smiles "
            "FROM compound_structures cs {join_expr} "
            "WHERE cs.canonical_smiles IS NOT NULL "
            "ORDER BY {order_expr}"
        ).format(order_expr=order_expr, id_expr=id_expr, join_expr=join_expr)

    candidates: list[tuple[str, str, str | None]] = []
    for table_name in tables:
        columns = _sqlite_columns(connection, table_name)
        smiles_column = _first_existing(columns, ["canonical_smiles", "smiles", "standard_smiles"])
        if smiles_column is None:
            continue
        id_column = _first_existing(columns, ["molecule_chembl_id", "chembl_id", "mol_chembl_id"])
        candidates.append((table_name, smiles_column, id_column))
    if candidates:
        table_name, smiles_column, id_column = sorted(candidates)[0]
        id_expr = id_column if id_column is not None else "NULL"
        return (
            f"SELECT ROW_NUMBER() OVER (ORDER BY rowid) - 1 AS source_row_index, "
            f"{id_expr} AS source_chembl_id, {smiles_column} AS source_smiles "
            f"FROM {table_name} WHERE {smiles_column} IS NOT NULL ORDER BY rowid"
        )
    raise _sqlite_schema_error(connection, "could not find a table with a SMILES column")


def _sqlite_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _sqlite_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_name})")]


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _sqlite_schema_error(connection: sqlite3.Connection, reason: str) -> ValueError:
    table_columns = {table: _sqlite_columns(connection, table) for table in _sqlite_tables(connection)}
    return ValueError(
        "Could not infer ChEMBL SQLite molecule structure query: "
        f"{reason}. Available tables/columns={table_columns}"
    )


def _write_subsets(
    cleaned_frame: pd.DataFrame,
    cleaned_dir: Path,
    reports_dir: Path,
    subset_sizes: list[int],
    seed: int,
    input_file: Path,
    full_cleaned_file: Path,
    config: CleaningConfig,
) -> Path:
    rng = np.random.default_rng(seed)
    actual_subset_sizes: dict[str, int] = {}
    written_files: dict[str, str] = {}
    num_rows = len(cleaned_frame)
    for size in subset_sizes:
        key = str(size)
        if num_rows < size:
            print(f"Warning: requested subset size {size} exceeds cleaned corpus size {num_rows}; skipping.")
            actual_subset_sizes[key] = 0
            continue
        indices = np.sort(rng.choice(num_rows, size=size, replace=False))
        subset = cleaned_frame.iloc[indices].sort_values(
            ["canonical_smiles", "source_row_index"],
            kind="mergesort",
        )
        subset_path = cleaned_dir / f"chembl_clean_{_format_subset_size(size)}.csv"
        subset.to_csv(subset_path, index=False)
        actual_subset_sizes[key] = int(len(subset))
        written_files[key] = str(subset_path)

    manifest = {
        "seed": seed,
        "requested_subset_sizes": subset_sizes,
        "actual_subset_sizes": actual_subset_sizes,
        "subset_files": written_files,
        "input_file": str(input_file),
        "full_cleaned_file": str(full_cleaned_file),
        "deduplicate_enabled": bool(config.deduplicate),
        "cleaning_config": _cleaning_config_dict(config),
    }
    manifest_path = reports_dir / "chembl_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _format_subset_size(size: int) -> str:
    if size >= 1_000_000 and size % 1_000_000 == 0:
        return f"{size // 1_000_000}m"
    if size >= 1_000 and size % 1_000 == 0:
        return f"{size // 1_000}k"
    return str(size)


def _cleaning_config_dict(config: CleaningConfig) -> dict[str, Any]:
    return {
        "allowed_elements": list(config.allowed_elements),
        "min_heavy_atoms": int(config.min_heavy_atoms),
        "max_heavy_atoms": int(config.max_heavy_atoms),
        "deduplicate": bool(config.deduplicate),
        "feature_spec_atomic_numbers": list(config.feature_spec.atomic_numbers),
        "feature_spec_formal_charge_clip": int(config.feature_spec.formal_charge_clip),
    }


def _write_summary_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for key, value in report.items():
        if isinstance(value, (dict, list)):
            continue
        rows.append({"metric": key, "value": value})
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
