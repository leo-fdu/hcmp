#!/usr/bin/env python
"""Check loop and vectorized HCMP forward paths for numerical equivalence."""

from __future__ import annotations

import argparse
import sys
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("Torch is required for the equivalence check.") from exc

    from hcmp.training.trainer import move_graph_batch_to_device

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    base_config = deepcopy(load_config(args.config))
    base_config.setdefault("model", {}).setdefault("encoder", {})["dropout"] = 0.0
    base_config["training"]["max_molecules"] = args.max_molecules
    batch, feature_spec, descriptor_names = _build_batch(base_config, args.max_molecules)
    batch = move_graph_batch_to_device(batch, device)

    torch.manual_seed(1234)
    loop_model = _build_model(base_config, feature_spec, descriptor_names, "loop", "loop").to(device)
    vectorized_model = _build_model(
        base_config,
        feature_spec,
        descriptor_names,
        "vectorized",
        "vectorized",
    ).to(device)
    vectorized_model.load_state_dict(loop_model.state_dict())
    loop_model.eval()
    vectorized_model.eval()

    with torch.no_grad():
        loop_output = loop_model(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            node_batch=batch.node_batch,
            edge_batch=batch.edge_batch,
        )
        vectorized_output = vectorized_model(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            node_batch=batch.node_batch,
            edge_batch=batch.edge_batch,
        )

    max_diffs = {
        name: _max_abs_diff(getattr(loop_output, name), getattr(vectorized_output, name))
        for name in [
            "node_embeddings",
            "edge_embeddings",
            "graph_embedding",
            "atom_logits",
            "bond_logits",
            "cut_logits",
            "descriptor_scores",
            "scaffold_projection",
        ]
    }
    for name, diff in max_diffs.items():
        print(f"{name}: max_abs_diff={diff:.10g}")
    worst = max(max_diffs.values()) if max_diffs else 0.0
    if worst >= args.tolerance:
        raise SystemExit(
            f"Vectorized equivalence check failed: worst max_abs_diff={worst:.10g}, "
            f"tolerance={args.tolerance:.10g}"
        )
    print(f"Vectorized equivalence check passed: worst max_abs_diff={worst:.10g}")


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
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=graph_collate,
    )
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
        dropout=float(encoder_config.get("dropout", 0.0)),
        attention_impl=attention_impl,
        pooling_impl=pooling_impl,
        enabled_heads={
            "bert": bool(config["bert"]["enabled"]),
            "cut_seg": bool(config["cut_seg"]["enabled"]),
            "prop_rank": bool(config["prop_rank"]["enabled"]),
            "scaf_triplet": bool(config["scaf_triplet"]["enabled"]),
        },
    )


def _max_abs_diff(left, right) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return float("inf")
    return float((left - right).abs().max().detach().cpu()) if left.numel() else 0.0


if __name__ == "__main__":
    main()
