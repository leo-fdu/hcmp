#!/usr/bin/env python
"""Generate formal pretraining experiment matrix configs."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.utils.config import load_config


MATRIX = {
    "hcmp_bert": {"bert": True, "cut_seg": False, "prop_rank": False, "scaf_triplet": False},
    "hcmp_bert_cut": {"bert": True, "cut_seg": True, "prop_rank": False, "scaf_triplet": False},
    "hcmp_bert_prop": {"bert": True, "cut_seg": False, "prop_rank": True, "scaf_triplet": False},
    "hcmp_bert_scaf": {"bert": True, "cut_seg": False, "prop_rank": False, "scaf_triplet": True},
    "hcmp_bert_cut_prop": {"bert": True, "cut_seg": True, "prop_rank": True, "scaf_triplet": False},
    "hcmp_bert_cut_scaf": {"bert": True, "cut_seg": True, "prop_rank": False, "scaf_triplet": True},
    "hcmp_bert_prop_scaf": {"bert": True, "cut_seg": False, "prop_rank": True, "scaf_triplet": True},
    "hcmp_full": {"bert": True, "cut_seg": True, "prop_rank": True, "scaf_triplet": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus-name", required=True)
    parser.add_argument("--graph-cache-dir", default=None, help="Deprecated alias for --hcmp-graph-cache-dir.")
    parser.add_argument("--hcmp-graph-cache-dir", default=None)
    parser.add_argument("--graph-bert-cache-dir", default=None)
    parser.add_argument("--descriptor-thresholds", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.hcmp_graph_cache_dir is None:
        args.hcmp_graph_cache_dir = args.graph_cache_dir
    if args.hcmp_graph_cache_dir is None:
        raise ValueError("--hcmp-graph-cache-dir is required.")
    if args.graph_bert_cache_dir is None:
        args.graph_bert_cache_dir = args.hcmp_graph_cache_dir

    base = load_config(args.base_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_scratch_config = output_dir / "scratch.yaml"
    if stale_scratch_config.exists():
        stale_scratch_config.unlink()

    _write_config(output_dir / "graph_bert.yaml", _graph_bert_config(base, args))
    for model_id, flags in MATRIX.items():
        config = copy.deepcopy(base)
        _stamp_common(config, args, model_id)
        _set_feature_mode(config, "hcmp", args.hcmp_graph_cache_dir)
        _set_on_the_fly_scaffold_backend(config)
        config["model_family"] = "hcmp"
        for key, enabled in flags.items():
            config[key]["enabled"] = bool(enabled)
        config["pretrain_objectives"] = dict(flags)
        config["output"]["run_dir"] = f"runs/pretrain/{args.corpus_name}/{model_id}/seed{args.seed}"
        _write_config(output_dir / f"{model_id}.yaml", config)
    print(f"Wrote pretraining matrix configs to {output_dir}")


def _stamp_common(config: dict, args, model_id: str) -> None:
    config["model_id"] = model_id
    config["run_name"] = model_id
    config["corpus_name"] = args.corpus_name
    config["seed"] = int(args.seed)
    config["pretrain_seed"] = int(args.seed)
    config.setdefault("training", {})["use_amp"] = False
    config.setdefault("dataset", {})["graph_mode"] = "cache"
    config["dataset"]["shard_aware_sampling"] = True
    config.setdefault("data", {})["descriptor_thresholds"] = args.descriptor_thresholds


def _set_feature_mode(config: dict, feature_mode: str, graph_cache_dir: str) -> None:
    config.setdefault("dataset", {})["feature_mode"] = feature_mode
    config["dataset"]["graph_cache_dir"] = graph_cache_dir
    config.setdefault("features", {})["feature_mode"] = feature_mode
    config["features"]["rich_molecular_features"] = feature_mode == "graph_bert"


def _graph_bert_config(base: dict, args) -> dict:
    config = copy.deepcopy(base)
    _stamp_common(config, args, "graph_bert")
    _set_feature_mode(config, "graph_bert", args.graph_bert_cache_dir)
    config["model_family"] = "graph_bert"
    config["baseline"] = "traditional_graph_bert"
    config["graph_bert"] = _graph_bert_section()
    _set_on_the_fly_scaffold_backend(config)
    config["cut_seg"]["enabled"] = False
    config["prop_rank"]["enabled"] = False
    config["scaf_triplet"]["enabled"] = False
    config["pretrain_objectives"] = {
        "bert": True,
        "cut_seg": False,
        "prop_rank": False,
        "scaf_triplet": False,
    }
    config["output"]["run_dir"] = f"runs/pretrain/{args.corpus_name}/graph_bert/seed{args.seed}"
    return config


def _set_on_the_fly_scaffold_backend(config: dict) -> None:
    scaf_config = config.setdefault("scaf_triplet", {})
    scaf_config["distance_backend"] = "on_the_fly"
    scaf_config["cache_path"] = None
    scaf_config["max_in_memory_distances"] = 0
    scaf_config["max_in_memory_scaffolds"] = 0


def _graph_bert_section() -> dict:
    return {
        "atom_target_fields": [
            "atomic_number",
            "formal_charge",
            "degree",
            "hybridization",
            "aromaticity",
            "num_hydrogens",
            "ring_membership",
            "chirality",
        ],
        "bond_target_fields": [
            "bond_type",
            "conjugation",
            "aromaticity",
            "ring_membership",
            "stereo",
        ],
    }


def _write_config(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(config, handle, sort_keys=False)
        else:
            handle.write(_dump_simple_yaml(config))


def _dump_simple_yaml(value, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_dump_simple_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{pad}{key}: {_format_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{pad}-")
                lines.append(_dump_simple_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{pad}- {_format_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{pad}{_format_scalar(value)}\n"


def _format_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    return str(value)


if __name__ == "__main__":
    main()
