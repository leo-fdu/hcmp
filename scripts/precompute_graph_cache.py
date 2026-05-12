#!/usr/bin/env python
"""Precompute sharded lightweight HCMP graph cache."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.chemistry.segmentation import segment_mol
from hcmp.data.descriptors import DESCRIPTOR_NAMES


@dataclass
class CacheBuildState:
    cached_molecules: int = 0
    processed_rows: int = 0
    skipped_molecules: int = 0
    failed_molecules: int = 0
    chunks_processed: int = 0
    shard_index: int = 0
    shard_files: list[str] = field(default_factory=list)
    shard_counts: list[int] = field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--descriptor-values", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--chunk-size", type=int, default=100000)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_manifests(output_dir)
    allowed_atomic_numbers = {
        Chem.GetPeriodicTable().GetAtomicNumber(symbol) for symbol in args.allowed_elements
    }
    from hcmp.data.graph_builder import default_feature_spec

    feature_spec = default_feature_spec({"feature_mode": args.feature_mode})
    shard_size = _positive_int(args.shard_size, "--shard-size")
    chunk_size = _positive_int(args.chunk_size, "--chunk-size")
    state = CacheBuildState()

    try:
        _precompute_chunked(
            args=args,
            smiles_column=smiles_column,
            output_dir=output_dir,
            allowed_atomic_numbers=allowed_atomic_numbers,
            feature_spec=feature_spec,
            shard_size=shard_size,
            chunk_size=chunk_size,
            state=state,
        )
    except Exception as exc:
        _write_incomplete_manifest(output_dir, args, state, str(exc))
        print(
            "Graph cache precompute failed: "
            f"chunks_processed={state.chunks_processed} "
            f"processed_rows={state.processed_rows} "
            f"cached_molecules={state.cached_molecules} error={exc}"
        )
        raise

    manifest = _build_manifest(
        args=args,
        feature_spec=feature_spec,
        state=state,
        shard_size=shard_size,
        chunk_size=chunk_size,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote graph cache manifest to {output_dir / 'manifest.json'}")
    print(
        f"cached_molecules={state.cached_molecules} "
        f"skipped_molecules={state.skipped_molecules} "
        f"failed_molecules={state.failed_molecules} "
        f"n_shards={state.shard_index}"
    )


def _precompute_chunked(
    *,
    args: argparse.Namespace,
    smiles_column: str,
    output_dir: Path,
    allowed_atomic_numbers: set[int],
    feature_spec,
    shard_size: int,
    chunk_size: int,
    state: CacheBuildState,
) -> None:
    from hcmp.data.graph_builder import mol_to_graph
    from hcmp.data.graph_cache import graph_to_record

    clean_reader = pd.read_csv(args.input_csv, chunksize=chunk_size)
    descriptor_reader = pd.read_csv(args.descriptor_values, chunksize=chunk_size)
    max_molecules = int(args.max_molecules) if args.max_molecules is not None else None

    for chunk_index, clean_chunk in enumerate(clean_reader):
        if max_molecules is not None:
            remaining = max_molecules - state.processed_rows
            if remaining <= 0:
                break
            clean_chunk = clean_chunk.head(remaining)
        if len(clean_chunk) == 0:
            break

        row_start = state.processed_rows
        row_end = row_start + len(clean_chunk) - 1
        print(f"[chunk {chunk_index}] rows {row_start}-{row_end} start")
        try:
            descriptor_chunk = next(descriptor_reader)
        except StopIteration as exc:
            raise ValueError(
                f"[chunk {chunk_index}] descriptor-values ended before input rows "
                f"{row_start}-{row_end}."
            ) from exc
        descriptor_chunk = descriptor_chunk.head(len(clean_chunk))
        if len(descriptor_chunk) < len(clean_chunk):
            raise ValueError(
                f"[chunk {chunk_index}] descriptor-values has fewer rows than input chunk: "
                f"clean_rows={len(clean_chunk)} descriptor_rows={len(descriptor_chunk)}."
            )
        print(f"[chunk {chunk_index}] read clean/descriptor done")

        clean_chunk = clean_chunk.reset_index(drop=True)
        descriptor_chunk = descriptor_chunk.reset_index(drop=True)
        _validate_descriptor_chunk_columns(descriptor_chunk)

        records: list[dict[str, Any]] = []
        chunk_cached_start = state.cached_molecules
        chunk_skipped_start = state.skipped_molecules
        chunk_failed_start = state.failed_molecules
        alignment_verified_rows = 0

        iterator = range(len(clean_chunk))
        for local_idx in tqdm(iterator, desc=f"[chunk {chunk_index}] caching graphs"):
            global_row_index = row_start + local_idx
            clean_row = clean_chunk.iloc[local_idx]
            descriptor_row = descriptor_chunk.iloc[local_idx]
            try:
                mol, canonical_smiles = _mol_from_row(
                    clean_row,
                    smiles_column,
                    global_row_index,
                )
                _validate_descriptor_row_alignment(
                    clean_row=clean_row,
                    descriptor_row=descriptor_row,
                    canonical_smiles=canonical_smiles,
                    global_row_index=global_row_index,
                    chunk_index=chunk_index,
                    local_idx=local_idx,
                    allow_unverified=bool(args.allow_unverified_descriptor_alignment),
                )
                alignment_verified_rows += 1
                if _descriptor_status_is_bad(descriptor_row):
                    state.skipped_molecules += 1
                    continue
                if not _allowed_mol(mol, allowed_atomic_numbers):
                    state.skipped_molecules += 1
                    continue
                result = segment_mol(mol, smiles=canonical_smiles)
                descriptor_values = [
                    float(descriptor_row[descriptor_name])
                    for descriptor_name in DESCRIPTOR_NAMES
                ]
                descriptor_mol_id = _optional_int(
                    descriptor_row.get("mol_id"),
                    default=global_row_index,
                )
                source_row_index = _source_row_identity(
                    clean_row,
                    descriptor_row,
                    global_row_index,
                )
                graph = mol_to_graph(
                    result.mol,
                    mol_id=descriptor_mol_id,
                    input_smiles=canonical_smiles,
                    feature_spec=feature_spec,
                    cut_labels=[label.cut_label for label in result.bond_labels],
                    descriptor_values=descriptor_values,
                    global_idx=state.cached_molecules,
                    source_row_index=source_row_index,
                )
                records.append(graph_to_record(graph))
                state.cached_molecules += 1
            except DescriptorAlignmentError:
                raise
            except Exception as exc:
                state.failed_molecules += 1
                print(
                    f"[chunk {chunk_index}] graph_error row={global_row_index} "
                    f"canonical_smiles={_safe_smiles(clean_row, smiles_column)!r} "
                    f"error={exc}"
                )
                continue
            if len(records) >= shard_size:
                _flush_shard(output_dir, records, state)

        if alignment_verified_rows:
            print(f"[chunk {chunk_index}] alignment verified rows={alignment_verified_rows}")
        else:
            print(f"[chunk {chunk_index}] no valid molecules available for alignment")
        if records:
            _flush_shard(output_dir, records, state)
        state.processed_rows += len(clean_chunk)
        state.chunks_processed += 1
        print(
            f"[chunk {chunk_index}] cached={state.cached_molecules - chunk_cached_start} "
            f"skipped={state.skipped_molecules - chunk_skipped_start} "
            f"failed={state.failed_molecules - chunk_failed_start}"
        )
        print(f"[chunk {chunk_index}] done")
        del clean_chunk, descriptor_chunk, records
        gc.collect()


def _flush_shard(output_dir: Path, records: list[dict[str, Any]], state: CacheBuildState) -> None:
    shard_name = _write_shard(output_dir, state.shard_index, records)
    state.shard_files.append(shard_name)
    state.shard_counts.append(len(records))
    print(
        f"wrote shard {shard_name} records={len(records)} "
        f"cached_molecules={state.cached_molecules}"
    )
    state.shard_index += 1
    records.clear()


def _write_shard(output_dir: Path, shard_index: int, records: list[dict]) -> str:
    import torch

    shard_name = f"shard_{shard_index:05d}.pt"
    path = output_dir / shard_name
    torch.save({"records": records}, path)
    return shard_name


def _allowed_mol(mol: Chem.Mol, allowed_atomic_numbers: set[int]) -> bool:
    return all(atom.GetAtomicNum() in allowed_atomic_numbers for atom in mol.GetAtoms())


def _detect_smiles_column(frame: pd.DataFrame) -> str:
    for column in ["smiles", "SMILES", "canonical_smiles", "mol"]:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not detect SMILES column from columns={list(frame.columns)}")


def _validate_descriptor_chunk_columns(descriptor_chunk: pd.DataFrame) -> None:
    missing = [name for name in DESCRIPTOR_NAMES if name not in descriptor_chunk.columns]
    if missing:
        raise ValueError(f"descriptor-values is missing columns: {missing}")


def _mol_from_row(
    clean_row: pd.Series,
    smiles_column: str,
    global_row_index: int,
) -> tuple[Chem.Mol, str]:
    raw_smiles = str(clean_row[smiles_column]).strip()
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        raise ValueError(f"Row {global_row_index} has invalid SMILES: {raw_smiles!r}")
    return mol, Chem.MolToSmiles(mol, canonical=True)


def _validate_descriptor_row_alignment(
    *,
    clean_row: pd.Series,
    descriptor_row: pd.Series,
    canonical_smiles: str,
    global_row_index: int,
    chunk_index: int,
    local_idx: int,
    allow_unverified: bool,
) -> None:
    if "canonical_smiles" not in descriptor_row.index:
        if allow_unverified:
            return
        raise DescriptorAlignmentError(
            _format_alignment_error(
                chunk_index=chunk_index,
                global_row_index=global_row_index,
                local_idx=local_idx,
                field="canonical_smiles",
                clean_row=clean_row,
                descriptor_row=descriptor_row,
                clean_smiles=canonical_smiles,
                descriptor_smiles=None,
                detail=(
                    "descriptor-values must contain canonical_smiles, or pass "
                    "--allow-unverified-descriptor-alignment explicitly"
                ),
            )
        )
    descriptor_smiles = str(descriptor_row["canonical_smiles"])
    if canonical_smiles != descriptor_smiles:
        _raise_alignment_error(
            chunk_index=chunk_index,
            global_row_index=global_row_index,
            local_idx=local_idx,
            field="canonical_smiles",
            clean_row=clean_row,
            descriptor_row=descriptor_row,
            clean_smiles=canonical_smiles,
            descriptor_smiles=descriptor_smiles,
        )
    clean_source = _nullable_int(clean_row.get("source_row_index"))
    descriptor_source = _nullable_int(descriptor_row.get("source_row_index"))
    if clean_source is not None and descriptor_source is not None:
        if clean_source != descriptor_source:
            _raise_alignment_error(
                chunk_index=chunk_index,
                global_row_index=global_row_index,
                local_idx=local_idx,
                field="source_row_index",
                clean_row=clean_row,
                descriptor_row=descriptor_row,
                clean_smiles=canonical_smiles,
                descriptor_smiles=descriptor_smiles,
            )
    elif clean_source is None and descriptor_source is not None:
        if descriptor_source != int(global_row_index):
            _raise_alignment_error(
                chunk_index=chunk_index,
                global_row_index=global_row_index,
                local_idx=local_idx,
                field="source_row_index",
                clean_row=clean_row,
                descriptor_row=descriptor_row,
                clean_smiles=canonical_smiles,
                descriptor_smiles=descriptor_smiles,
            )
    for id_column in ["mol_id", "molecule_id", "source_chembl_id"]:
        if id_column in descriptor_row.index and id_column in clean_row.index:
            clean_id = str(clean_row[id_column])
            descriptor_id = str(descriptor_row[id_column])
            if clean_id != descriptor_id:
                _raise_alignment_error(
                    chunk_index=chunk_index,
                    global_row_index=global_row_index,
                    local_idx=local_idx,
                    field=id_column,
                    clean_row=clean_row,
                    descriptor_row=descriptor_row,
                    clean_smiles=canonical_smiles,
                    descriptor_smiles=descriptor_smiles,
                    detail=f"clean_{id_column}={clean_id!r}; descriptor_{id_column}={descriptor_id!r}",
                )


class DescriptorAlignmentError(ValueError):
    """Descriptor CSV rows do not match the clean molecule CSV rows."""


def _raise_alignment_error(
    *,
    chunk_index: int,
    global_row_index: int,
    local_idx: int,
    field: str,
    clean_row: pd.Series,
    descriptor_row: pd.Series,
    clean_smiles: str | None,
    descriptor_smiles: str | None,
    detail: str | None = None,
) -> None:
    raise DescriptorAlignmentError(
        _format_alignment_error(
            chunk_index=chunk_index,
            global_row_index=global_row_index,
            local_idx=local_idx,
            field=field,
            clean_row=clean_row,
            descriptor_row=descriptor_row,
            clean_smiles=clean_smiles,
            descriptor_smiles=descriptor_smiles,
            detail=detail,
        )
    )


def _format_alignment_error(
    *,
    chunk_index: int,
    global_row_index: int,
    local_idx: int,
    field: str,
    clean_row: pd.Series,
    descriptor_row: pd.Series,
    clean_smiles: str | None,
    descriptor_smiles: str | None,
    detail: str | None = None,
) -> str:
    clean_source = _nullable_int(clean_row.get("source_row_index"))
    descriptor_source = _nullable_int(descriptor_row.get("source_row_index"))
    message = (
        f"[chunk {chunk_index}] descriptor alignment mismatch: "
        f"row_offset={global_row_index} local_row={local_idx} field={field}; "
        f"clean_source_row_index={clean_source}; "
        f"descriptor_source_row_index={descriptor_source}; "
        f"clean_smiles={clean_smiles!r}; "
        f"descriptor_smiles={descriptor_smiles!r}"
    )
    if detail:
        message += f"; {detail}"
    return message


def _descriptor_status_is_bad(descriptor_row: pd.Series) -> bool:
    return (
        "status" in descriptor_row.index
        and not pd.isna(descriptor_row["status"])
        and str(descriptor_row["status"]) != "success"
    )


def _source_row_identity(
    clean_row: pd.Series,
    descriptor_row: pd.Series,
    global_row_index: int,
) -> int:
    if "source_row_index" in descriptor_row.index and not pd.isna(descriptor_row["source_row_index"]):
        return int(descriptor_row["source_row_index"])
    if "source_row_index" in clean_row.index and not pd.isna(clean_row["source_row_index"]):
        return int(clean_row["source_row_index"])
    return int(global_row_index)


def _nullable_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _safe_smiles(clean_row: pd.Series, smiles_column: str) -> str:
    if smiles_column not in clean_row.index or pd.isna(clean_row[smiles_column]):
        return ""
    return str(clean_row[smiles_column])


def _optional_int(value: Any, default: int) -> int:
    if value is None or pd.isna(value):
        return int(default)
    return int(value)


def _positive_int(value: int, name: str) -> int:
    int_value = int(value)
    if int_value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int_value


def _build_manifest(
    *,
    args: argparse.Namespace,
    feature_spec,
    state: CacheBuildState,
    shard_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    from hcmp.data.graph_cache import FEATURE_VERSION

    return {
        "input_csv": str(args.input_csv),
        "descriptor_values_path": str(args.descriptor_values),
        "n_molecules": int(state.cached_molecules),
        "cached_molecules": int(state.cached_molecules),
        "processed_input_rows": int(state.processed_rows),
        "skipped_molecules": int(state.skipped_molecules),
        "failed_molecules": int(state.failed_molecules),
        "n_shards": int(state.shard_index),
        "shard_size": int(shard_size),
        "chunk_size": int(chunk_size),
        "chunks_processed": int(state.chunks_processed),
        "shard_files": list(state.shard_files),
        "shard_counts": list(state.shard_counts),
        "feature_version": FEATURE_VERSION,
        "feature_mode": args.feature_mode,
        "node_feature_dim": feature_spec.node_feature_dim,
        "edge_feature_dim": feature_spec.edge_feature_dim,
        "atom_target_fields": feature_spec.atom_target_fields,
        "bond_target_fields": feature_spec.bond_target_fields,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "allowed_elements": list(args.allowed_elements),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "completed": True,
        "feature_spec": {
            "atomic_numbers": feature_spec.atomic_numbers,
            "formal_charge_clip": feature_spec.formal_charge_clip,
            "rich_molecular_features": feature_spec.rich_molecular_features,
            "feature_mode": feature_spec.feature_mode,
        },
    }


def _write_incomplete_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    state: CacheBuildState,
    error: str,
) -> None:
    incomplete = {
        "completed": False,
        "input_csv": str(args.input_csv),
        "descriptor_values_path": str(args.descriptor_values),
        "cached_molecules": int(state.cached_molecules),
        "processed_input_rows": int(state.processed_rows),
        "skipped_molecules": int(state.skipped_molecules),
        "failed_molecules": int(state.failed_molecules),
        "chunks_processed": int(state.chunks_processed),
        "n_shards": int(state.shard_index),
        "shard_files": list(state.shard_files),
        "shard_counts": list(state.shard_counts),
        "error": error,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "manifest.incomplete.json"
    path.write_text(json.dumps(incomplete, indent=2), encoding="utf-8")
    print(f"Wrote incomplete graph cache manifest to {path}")


def _remove_stale_manifests(output_dir: Path) -> None:
    for name in ["manifest.json", "manifest.incomplete.json"]:
        path = output_dir / name
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
