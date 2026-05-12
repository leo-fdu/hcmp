"""Dataset and dataloader interfaces for HCMP training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
import random

import pandas as pd
from rdkit import Chem

from hcmp.chemistry.segmentation import segment_mol
from hcmp.data.descriptors import DESCRIPTOR_NAMES, compute_descriptor_values
from hcmp.data.scaffold_distance.data_types import MoleculeTable


@dataclass
class ProcessedDataPaths:
    molecules_csv: str = "data/processed/molecules.csv"
    graphs_pt: str = "data/processed/graphs.pt"
    bond_labels_csv: str = "data/processed/bond_labels.csv"
    descriptor_values_csv: str = "data/processed/descriptor_values.csv"
    descriptor_thresholds_csv: str = "data/processed/descriptor_thresholds.csv"
    descriptor_thresholds_json: str = "data/processed/descriptor_thresholds.json"
    prop_rank_pairs_parquet: str = "data/processed/prop_rank_pairs.parquet"
    scaffold_triplets_csv: str = "data/processed/scaffold_triplets.csv"


def load_successful_molecules(path: str) -> pd.DataFrame:
    """Load successful molecule rows from the stable processed schema."""

    frame = pd.read_csv(path)
    return frame[frame["status"] == "success"].reset_index(drop=True)


class GraphDataset:
    """Prebuilt graph dataset backed by a sorted MoleculeTable."""

    def __init__(
        self,
        molecule_table: MoleculeTable,
        feature_spec: Any,
        descriptor_values: str | Path | pd.DataFrame | None = None,
        descriptor_names: list[str] | None = None,
        recompute_descriptors: bool = False,
    ) -> None:
        from hcmp.data.graph_builder import mol_to_graph

        self.molecule_table = molecule_table
        self.feature_spec = feature_spec
        self.graphs = []
        self.source_row_to_index: dict[int, int] = {}
        self.descriptor_names = list(descriptor_names or DESCRIPTOR_NAMES)
        descriptor_frame = (
            _load_descriptor_values_for_molecule_table(
                molecule_table,
                descriptor_values,
                self.descriptor_names,
            )
            if descriptor_values is not None
            else None
        )
        for idx, mol in enumerate(molecule_table.mols):
            canonical_smiles = molecule_table.canonical_smiles[idx]
            source_row_index = int(molecule_table.dataframe.iloc[idx]["source_row_index"])
            self.source_row_to_index[source_row_index] = idx
            result = segment_mol(mol, smiles=canonical_smiles)
            graph_descriptor_values = None
            if descriptor_frame is not None:
                graph_descriptor_values = [
                    float(descriptor_frame.iloc[idx][descriptor_name])
                    for descriptor_name in self.descriptor_names
                ]
            elif recompute_descriptors:
                computed = compute_descriptor_values(result.mol)
                graph_descriptor_values = [
                    computed[descriptor_name] for descriptor_name in self.descriptor_names
                ]
            self.graphs.append(
                mol_to_graph(
                    result.mol,
                    mol_id=idx,
                    input_smiles=canonical_smiles,
                    feature_spec=feature_spec,
                    cut_labels=[label.cut_label for label in result.bond_labels],
                    descriptor_values=graph_descriptor_values,
                    global_idx=idx,
                    source_row_index=source_row_index,
                )
            )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[int(idx)]


class LazyGraphDataset:
    """Construct graph tensors in __getitem__ for smoke tests and debugging."""

    def __init__(
        self,
        molecule_table: MoleculeTable,
        feature_spec: Any,
        descriptor_values: str | Path | pd.DataFrame | None = None,
        descriptor_names: list[str] | None = None,
        recompute_descriptors: bool = False,
    ) -> None:
        self.molecule_table = molecule_table
        self.feature_spec = feature_spec
        self.descriptor_names = list(descriptor_names or DESCRIPTOR_NAMES)
        self.recompute_descriptors = bool(recompute_descriptors)
        self.descriptor_frame = (
            _load_descriptor_values_for_molecule_table(
                molecule_table,
                descriptor_values,
                self.descriptor_names,
            )
            if descriptor_values is not None
            else None
        )

    def __len__(self) -> int:
        return self.molecule_table.size

    def __getitem__(self, idx: int):
        from hcmp.data.graph_builder import mol_to_graph

        index = int(idx)
        mol = self.molecule_table.mols[index]
        canonical_smiles = self.molecule_table.canonical_smiles[index]
        source_row_index = int(self.molecule_table.dataframe.iloc[index]["source_row_index"])
        result = segment_mol(mol, smiles=canonical_smiles)
        graph_descriptor_values = None
        if self.descriptor_frame is not None:
            graph_descriptor_values = [
                float(self.descriptor_frame.iloc[index][descriptor_name])
                for descriptor_name in self.descriptor_names
            ]
        elif self.recompute_descriptors:
            computed = compute_descriptor_values(result.mol)
            graph_descriptor_values = [
                computed[descriptor_name] for descriptor_name in self.descriptor_names
            ]
        return mol_to_graph(
            result.mol,
            mol_id=index,
            input_smiles=canonical_smiles,
            feature_spec=self.feature_spec,
            cut_labels=[label.cut_label for label in result.bond_labels],
            descriptor_values=graph_descriptor_values,
            global_idx=index,
            source_row_index=source_row_index,
        )


class CachedGraphDataset:
    """Dataset backed by a sharded precomputed graph cache."""

    def __init__(self, cache_dir: str | Path, max_loaded_shards: int = 4) -> None:
        from hcmp.data.graph_cache import ShardedGraphCache

        self.cache = ShardedGraphCache(cache_dir, max_loaded_shards=max_loaded_shards)
        self.manifest = self.cache.manifest
        self.shard_size = int(self.manifest["shard_size"])
        self.n_shards = int(self.manifest["n_shards"])

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int):
        return self.cache[int(idx)]

    def shard_bounds(self, shard_idx: int) -> tuple[int, int]:
        return self.cache.shard_bounds(int(shard_idx))


class ShardAwareBatchSampler:
    """Yield cache-mode batches by shuffling shards, then rows within each shard."""

    def __init__(
        self,
        dataset: CachedGraphDataset,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        shard_order = list(range(self.dataset.n_shards))
        rng.shuffle(shard_order)
        print(
            "shard_aware_sampler_epoch "
            f"epoch={self.epoch} n_shards={len(shard_order)} batch_size={self.batch_size}"
        )
        for shard_idx in shard_order:
            start, end = self.dataset.shard_bounds(shard_idx)
            indices = list(range(start, end))
            rng.shuffle(indices)
            print(
                "shard_aware_sampler_shard "
                f"epoch={self.epoch} shard={shard_idx} n_indices={len(indices)}"
            )
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class DescriptorPairDataset:
    """Descriptor pair dataset returning graph_i, graph_j, descriptor_id, label."""

    def __init__(
        self,
        pairs: str | Path | pd.DataFrame,
        graph_dataset: GraphDataset,
    ) -> None:
        self.pairs = _read_table(pairs).reset_index(drop=True)
        self.graph_dataset = graph_dataset
        self.mol_id_to_index = {
            graph.mol_id: idx for idx, graph in enumerate(graph_dataset.graphs)
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[int(idx)]
        mol_i = row["mol_i"] if "mol_i" in row else row["i"]
        mol_j = row["mol_j"] if "mol_j" in row else row["j"]
        descriptor_id = _descriptor_id(row)
        return (
            self.graph_dataset[self._resolve_graph_index(mol_i)],
            self.graph_dataset[self._resolve_graph_index(mol_j)],
            descriptor_id,
            float(row["label"]),
        )

    def _resolve_graph_index(self, value: Any) -> int:
        int_value = int(value)
        if int_value in self.graph_dataset.source_row_to_index:
            return self.graph_dataset.source_row_to_index[int_value]
        if value in self.mol_id_to_index:
            return self.mol_id_to_index[value]
        if int_value in self.mol_id_to_index:
            return self.mol_id_to_index[int_value]
        if 0 <= int_value < len(self.graph_dataset):
            return int_value
        raise KeyError(f"Could not resolve molecule id/index {value!r}.")


class ScaffoldTripletDataset:
    """Scaffold triplet dataset returning anchor, positive, and negative graphs."""

    def __init__(
        self,
        triplets: str | Path | pd.DataFrame,
        graph_dataset: GraphDataset,
        min_distance_gap: float = 0.15,
    ) -> None:
        self.triplets = _read_table(triplets).reset_index(drop=True)
        self.graph_dataset = graph_dataset
        self.min_distance_gap = min_distance_gap
        self._validate_triplets()

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        row = self.triplets.iloc[int(idx)]
        return (
            self.graph_dataset[int(row["anchor_idx"])],
            self.graph_dataset[int(row["positive_idx"])],
            self.graph_dataset[int(row["negative_idx"])],
        )

    def _validate_triplets(self) -> None:
        required = {"anchor_idx", "positive_idx", "negative_idx", "d_ap", "d_an"}
        missing = required - set(self.triplets.columns)
        if missing:
            raise ValueError(f"Triplet table is missing columns: {sorted(missing)}")
        bad_gap = self.triplets[self.triplets["d_ap"] + self.min_distance_gap >= self.triplets["d_an"]]
        if len(bad_gap) > 0:
            raise ValueError("Triplet table contains rows that violate the scaffold min_gap rule.")
        max_index = len(self.graph_dataset) - 1
        for column in ["anchor_idx", "positive_idx", "negative_idx"]:
            if ((self.triplets[column] < 0) | (self.triplets[column] > max_index)).any():
                raise ValueError(f"Triplet column {column!r} contains out-of-range graph indices.")


def graph_collate(batch: list[Any]):
    from hcmp.data.graph_builder import collate_graphs

    return collate_graphs(batch)


def pair_collate(batch: list[tuple[Any, Any, int, float]]):
    import torch

    graphs_i, graphs_j, descriptor_ids, labels = zip(*batch, strict=False)
    return (
        graph_collate(list(graphs_i)),
        graph_collate(list(graphs_j)),
        torch.tensor(descriptor_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def triplet_collate(batch: list[tuple[Any, Any, Any]]) -> dict[str, Any]:
    anchors, positives, negatives = zip(*batch, strict=False)
    return {
        "anchor": graph_collate(list(anchors)),
        "positive": graph_collate(list(positives)),
        "negative": graph_collate(list(negatives)),
    }


class MultiTaskDataLoader:
    """Simple wrapper around the main graph-batch loader."""

    def __init__(self, graph_loader: Iterable) -> None:
        self.graph_loader = graph_loader

    def iterate_graph(self):
        return iter(self.graph_loader)


def _read_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            fallback = path.with_suffix(path.suffix + ".csv")
            if fallback.exists():
                return pd.read_csv(fallback)
            raise
    return pd.read_csv(path)


def _load_descriptor_values_for_molecule_table(
    molecule_table: MoleculeTable,
    descriptor_values: str | Path | pd.DataFrame,
    descriptor_names: list[str],
) -> pd.DataFrame:
    frame = _read_table(descriptor_values).reset_index(drop=True)
    descriptor_columns = [column for column in frame.columns if column in DESCRIPTOR_NAMES]
    if descriptor_columns != descriptor_names:
        raise ValueError(
            "Descriptor value columns do not match the expected descriptor order. "
            f"expected={descriptor_names}, found={descriptor_columns}"
        )
    if len(frame) < molecule_table.size:
        raise ValueError(
            "Descriptor values row count does not match the molecule table: "
            f"descriptor_rows={len(frame)}, molecule_rows={molecule_table.size}."
        )
    if len(frame) > molecule_table.size:
        print(
            "Descriptor values table has extra rows; using the leading rows that match "
            f"the current molecule table size ({molecule_table.size})."
        )
        frame = frame.head(molecule_table.size).reset_index(drop=True)
    if "canonical_smiles" not in frame.columns:
        raise ValueError("Descriptor values table must contain canonical_smiles for identity validation.")

    for idx in range(molecule_table.size):
        molecule_smiles = str(molecule_table.canonical_smiles[idx])
        descriptor_smiles = str(frame.iloc[idx]["canonical_smiles"])
        molecule_source = (
            int(molecule_table.dataframe.iloc[idx]["source_row_index"])
            if "source_row_index" in molecule_table.dataframe.columns
            else None
        )
        descriptor_source = (
            int(frame.iloc[idx]["source_row_index"])
            if "source_row_index" in frame.columns and not pd.isna(frame.iloc[idx]["source_row_index"])
            else None
        )
        if molecule_smiles != descriptor_smiles or (
            molecule_source is not None
            and descriptor_source is not None
            and molecule_source != descriptor_source
        ):
            raise ValueError(
                "Descriptor values table does not match molecule table order. "
                f"first_mismatch_row={idx}; "
                f"molecule_canonical_smiles={molecule_smiles!r}; "
                f"descriptor_canonical_smiles={descriptor_smiles!r}; "
                f"molecule_source_row_index={molecule_source}; "
                f"descriptor_source_row_index={descriptor_source}."
            )
    if "status" in frame.columns:
        bad_status = frame[frame["status"] != "success"]
        if len(bad_status) > 0:
            first = int(bad_status.index[0])
            raise ValueError(
                "Descriptor values table contains non-success rows for training. "
                f"first_bad_row={first}; status={bad_status.iloc[0]['status']!r}."
            )
    return frame


def _descriptor_id(row: pd.Series) -> int:
    if "descriptor_id" in row:
        return int(row["descriptor_id"])
    if "descriptor_k" in row:
        value = row["descriptor_k"]
        return int(value) if not isinstance(value, str) else DESCRIPTOR_NAMES.index(value)
    if "descriptor_name" in row:
        return DESCRIPTOR_NAMES.index(str(row["descriptor_name"]))
    raise KeyError("Pair row must contain descriptor_id, descriptor_k, or descriptor_name.")
