"""Sharded lightweight graph cache helpers."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("hcmp.data.graph_cache requires torch.") from exc

from hcmp.data.graph_builder import GraphData


FEATURE_VERSION = "hcmp_graph_cache_v2"


def graph_to_record(graph: GraphData) -> dict[str, Any]:
    """Convert a GraphData object into a torch-saveable lightweight record."""

    return {
        "mol_id": graph.mol_id,
        "input_smiles": graph.input_smiles,
        "canonical_smiles": graph.canonical_smiles,
        "node_features": graph.node_features.cpu(),
        "edge_features": graph.edge_features.cpu(),
        "edge_index": graph.edge_index.cpu(),
        "atom_type_targets": graph.atom_type_targets.cpu(),
        "bond_type_targets": graph.bond_type_targets.cpu(),
        "graph_bert_atom_targets": (
            {key: value.cpu() for key, value in graph.graph_bert_atom_targets.items()}
            if graph.graph_bert_atom_targets is not None
            else None
        ),
        "graph_bert_bond_targets": (
            {key: value.cpu() for key, value in graph.graph_bert_bond_targets.items()}
            if graph.graph_bert_bond_targets is not None
            else None
        ),
        "cut_labels": graph.cut_labels.cpu() if graph.cut_labels is not None else None,
        "descriptor_values": (
            graph.descriptor_values.cpu() if graph.descriptor_values is not None else None
        ),
        "global_idx": graph.global_idx,
        "source_row_index": graph.source_row_index,
    }


def record_to_graph(record: dict[str, Any]) -> GraphData:
    """Rebuild GraphData from a cached record."""

    return GraphData(
        mol_id=record.get("mol_id"),
        input_smiles=str(record.get("input_smiles", record.get("canonical_smiles", ""))),
        canonical_smiles=str(record["canonical_smiles"]),
        node_features=record["node_features"].float(),
        edge_features=record["edge_features"].float(),
        edge_index=record["edge_index"].long(),
        atom_type_targets=record["atom_type_targets"].long(),
        bond_type_targets=record["bond_type_targets"].long(),
        graph_bert_atom_targets=(
            {key: value.long() for key, value in record["graph_bert_atom_targets"].items()}
            if record.get("graph_bert_atom_targets") is not None
            else None
        ),
        graph_bert_bond_targets=(
            {key: value.long() for key, value in record["graph_bert_bond_targets"].items()}
            if record.get("graph_bert_bond_targets") is not None
            else None
        ),
        cut_labels=(
            record["cut_labels"].float() if record.get("cut_labels") is not None else None
        ),
        descriptor_values=(
            record["descriptor_values"].float()
            if record.get("descriptor_values") is not None
            else None
        ),
        global_idx=record.get("global_idx"),
        source_row_index=record.get("source_row_index"),
    )


class ShardedGraphCache:
    """Small LRU reader for graph-cache shards."""

    def __init__(self, cache_dir: str | Path, max_loaded_shards: int = 4) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Graph cache manifest not found: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.shard_size = int(self.manifest["shard_size"])
        self.n_molecules = int(
            self.manifest.get("n_molecules", self.manifest.get("cached_molecules", 0))
        )
        self.shard_files = list(self.manifest.get("shard_files", []))
        self.shard_counts = [int(count) for count in self.manifest.get("shard_counts", [])]
        if self.shard_counts:
            if sum(self.shard_counts) != self.n_molecules:
                raise ValueError(
                    "Graph cache manifest shard_counts do not sum to n_molecules: "
                    f"sum={sum(self.shard_counts)} n_molecules={self.n_molecules}."
                )
            if self.shard_files and len(self.shard_files) != len(self.shard_counts):
                raise ValueError(
                    "Graph cache manifest shard_files and shard_counts length mismatch: "
                    f"shard_files={len(self.shard_files)} shard_counts={len(self.shard_counts)}."
                )
            self._shard_offsets = _cumulative_offsets(self.shard_counts)
        else:
            self._shard_offsets = []
        self.max_loaded_shards = max(1, int(max_loaded_shards))
        self._loaded: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self.load_counts: dict[int, int] = {}

    def __len__(self) -> int:
        return self.n_molecules

    def __getitem__(self, idx: int) -> GraphData:
        index = int(idx)
        if index < 0 or index >= self.n_molecules:
            raise IndexError(index)
        shard_index, offset = self._locate(index)
        records = self._load_shard(shard_index)
        return record_to_graph(records[offset])

    def shard_bounds(self, shard_index: int) -> tuple[int, int]:
        if self.shard_counts:
            start = self._shard_offsets[int(shard_index)]
            return start, start + self.shard_counts[int(shard_index)]
        start = int(shard_index) * self.shard_size
        return start, min(start + self.shard_size, self.n_molecules)

    def _locate(self, index: int) -> tuple[int, int]:
        if not self.shard_counts:
            return index // self.shard_size, index % self.shard_size
        shard_index = bisect_right(self._shard_offsets, index) - 1
        if shard_index < 0 or shard_index >= len(self.shard_counts):
            raise IndexError(index)
        return shard_index, index - self._shard_offsets[shard_index]

    def _load_shard(self, shard_index: int) -> list[dict[str, Any]]:
        if shard_index in self._loaded:
            self._loaded.move_to_end(shard_index)
            return self._loaded[shard_index]
        path = self.cache_dir / (
            self.shard_files[shard_index]
            if self.shard_files
            else f"shard_{shard_index:05d}.pt"
        )
        self.load_counts[shard_index] = self.load_counts.get(shard_index, 0) + 1
        if self.load_counts[shard_index] == 1 or self.load_counts[shard_index] % 10 == 0:
            print(
                "graph_cache_load "
                f"cache_dir={self.cache_dir} shard={shard_index} "
                f"load_count={self.load_counts[shard_index]}"
            )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
        self._loaded[shard_index] = records
        while len(self._loaded) > self.max_loaded_shards:
            self._loaded.popitem(last=False)
        return records


def _cumulative_offsets(counts: list[int]) -> list[int]:
    offsets: list[int] = []
    total = 0
    for count in counts:
        offsets.append(total)
        total += int(count)
    return offsets
