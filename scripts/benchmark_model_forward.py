#!/usr/bin/env python
"""Lightweight HCMP model forward benchmark."""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hcmp_debug.yaml")
    parser.add_argument("--max-molecules", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention-impl", choices=["loop", "vectorized", "both"], default="both")
    parser.add_argument("--pooling-impl", choices=["loop", "vectorized", "both"], default="both")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("Torch is required for benchmarking.") from exc

    from hcmp.training.trainer import move_graph_batch_to_device

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    config = deepcopy(load_config(args.config))
    batch, feature_spec, descriptor_names = _build_batch(config, args.max_molecules)
    batch = move_graph_batch_to_device(batch, device)

    for attention_impl, pooling_impl in _impl_pairs(args.attention_impl, args.pooling_impl):
        model = _build_model(config, feature_spec, descriptor_names, attention_impl, pooling_impl).to(device)
        model.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(min(5, args.num_steps)):
                _forward(model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(args.num_steps):
                _forward(model, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

        avg_seconds = elapsed / max(1, args.num_steps)
        memory_mb = ""
        if device.type == "cuda":
            memory_mb = f"{torch.cuda.max_memory_allocated(device) / (1024 * 1024):.3f}"
        print(
            "benchmark "
            f"attention_impl={attention_impl} "
            f"pooling_impl={pooling_impl} "
            f"device={device} "
            f"batch_size={batch.node_batch.max().item() + 1 if batch.node_batch.numel() else 0} "
            f"num_molecules={len(batch.mol_ids)} "
            f"avg_seconds_per_step={avg_seconds:.6g} "
            f"steps_per_second={1.0 / avg_seconds if avg_seconds > 0 else 0.0:.6g} "
            f"max_cuda_memory_allocated_mb={memory_mb}"
        )


def _build_batch(config: dict, max_molecules: int):
    from torch.utils.data import DataLoader

    from hcmp.data.datasets import GraphDataset, graph_collate
    from hcmp.data.descriptors import DESCRIPTOR_NAMES
    from hcmp.data.graph_builder import default_feature_spec
    from hcmp.data.molecule_table import load_hcmp_molecule_table

    data_config = config["data"]
    table = load_hcmp_molecule_table(
        data_config["input_csv"],
        smiles_column=data_config.get("smiles_column", "smiles"),
        sort_by_canonical_smiles=True,
        max_molecules=max_molecules,
        strict=False,
    )
    threshold_names = None
    threshold_path = data_config.get("descriptor_thresholds")
    if threshold_path is not None and Path(threshold_path).exists():
        threshold_names = list(pd.read_csv(threshold_path)["descriptor_name"])
    descriptor_names = threshold_names or DESCRIPTOR_NAMES
    feature_spec = default_feature_spec(config.get("features", {}))
    dataset = GraphDataset(
        table,
        feature_spec,
        descriptor_values=data_config.get("descriptor_values_path"),
        descriptor_names=descriptor_names,
        recompute_descriptors=bool(data_config.get("recompute_descriptors_in_dataset", False)),
    )
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=graph_collate)
    return next(iter(loader)), feature_spec, descriptor_names


def _build_model(
    config: dict,
    feature_spec,
    descriptor_names: list[str],
    attention_impl: str,
    pooling_impl: str,
):
    from hcmp.models.hcmp_model import HCMPModel

    encoder_config = config["model"]["encoder"]
    return HCMPModel(
        feature_spec=feature_spec,
        hidden_dim=int(encoder_config["hidden_dim"]),
        num_layers=int(encoder_config["num_layers"]),
        encoder_type=str(encoder_config.get("encoder_type", encoder_config.get("type", "graph_transformer"))),
        num_heads=int(encoder_config.get("num_heads", 4)),
        dropout=float(encoder_config.get("dropout", 0.1)),
        attention_impl=attention_impl,
        pooling_impl=pooling_impl,
        enabled_heads={
            "bert": bool(config["bert"]["enabled"]),
            "cut_seg": bool(config["cut_seg"]["enabled"]),
            "prop_rank": bool(config["prop_rank"]["enabled"]),
            "scaf_triplet": bool(config["scaf_triplet"]["enabled"]),
        },
    )


def _impl_pairs(attention_impl: str, pooling_impl: str) -> list[tuple[str, str]]:
    if attention_impl == "both" and pooling_impl == "both":
        return [("loop", "loop"), ("vectorized", "vectorized")]
    attentions = ["loop", "vectorized"] if attention_impl == "both" else [attention_impl]
    poolings = ["loop", "vectorized"] if pooling_impl == "both" else [pooling_impl]
    return [(attention, pooling) for attention in attentions for pooling in poolings]


def _forward(model, batch):
    return model(
        batch.node_features,
        batch.edge_index,
        batch.edge_features,
        node_batch=batch.node_batch,
        edge_batch=batch.edge_batch,
    )


if __name__ == "__main__":
    main()
