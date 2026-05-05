#!/usr/bin/env python
"""Print or run downstream finetuning matrix commands."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--downstream-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Deprecated alias for --downstream-seeds.")
    parser.add_argument("--smiles-column", default=None, help="Global SMILES column override.")
    parser.add_argument("--label-column", default=None, help="Global label column override.")
    parser.add_argument(
        "--column-map",
        default=None,
        help="JSON/YAML mapping of dataset to smiles_column/label_column overrides.",
    )
    parser.add_argument("--run", action="store_true", help="Run commands instead of printing them.")
    args = parser.parse_args()
    downstream_seeds = args.downstream_seeds if args.downstream_seeds is not None else args.seeds
    if downstream_seeds is None:
        raise ValueError("--downstream-seeds is required.")
    column_map = _load_column_map(args.column_map)

    commands = []
    for dataset in args.datasets:
        smiles_column, label_column = _column_overrides_for_dataset(
            dataset,
            column_map,
            global_smiles_column=args.smiles_column,
            global_label_column=args.label_column,
        )
        for split in args.splits:
            for model in args.models:
                for seed in downstream_seeds:
                    checkpoint = "none" if model == "scratch" else str(Path(args.pretrain_root) / model / f"seed{args.pretrain_seed}" / "checkpoints" / "final.pt")
                    output_dir = Path(args.output_root) / dataset / split / model / f"seed{seed}"
                    command = [
                        "python",
                        "scripts/finetune_downstream.py",
                        "--dataset",
                        dataset,
                        "--split",
                        split,
                        "--checkpoint",
                        checkpoint,
                        "--model-id",
                        model,
                        "--seed",
                        str(seed),
                        "--pretrain-seed",
                        str(args.pretrain_seed),
                        "--data-root",
                        args.data_root,
                        "--output-dir",
                        str(output_dir),
                    ]
                    if smiles_column is not None:
                        command.extend(["--smiles-column", smiles_column])
                    if label_column is not None:
                        command.extend(["--label-column", label_column])
                    commands.append(command)
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command))
        if args.run:
            subprocess.run(command, check=True)


def _load_column_map(path_like: str | None) -> dict:
    if path_like is None:
        return {}
    path = Path(path_like)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        if yaml is None:
            from hcmp.utils.io import load_yaml

            data = load_yaml(path)
        else:
            data = yaml.safe_load(text)
    return data or {}


def _column_overrides_for_dataset(
    dataset: str,
    column_map: dict,
    global_smiles_column: str | None = None,
    global_label_column: str | None = None,
) -> tuple[str | None, str | None]:
    entry = column_map.get(dataset, column_map.get(dataset.lower(), {}))
    if entry is None:
        entry = {}
    if not isinstance(entry, dict):
        raise ValueError(f"Column map entry for {dataset!r} must be a mapping.")
    smiles_column = entry.get("smiles_column", entry.get("smiles")) or global_smiles_column
    label_column = entry.get("label_column", entry.get("label")) or global_label_column
    return smiles_column, label_column


if __name__ == "__main__":
    main()
