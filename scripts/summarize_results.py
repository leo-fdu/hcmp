#!/usr/bin/env python
"""Aggregate downstream final_metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = []
    for path in Path(args.downstream_root).glob("*/*/*/seed*/final_metrics.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("test_metrics_at_best_val", {})
        for metric, value in metrics.items():
            rows.append(
                {
                    "model_id": data["model_id"],
                    "dataset": data["dataset"],
                    "split": data["split"],
                    "seed": data["seed"],
                    "metric": metric,
                    "value": value,
                }
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "downstream_results.csv", index=False)
    if results.empty:
        pd.DataFrame().to_csv(output_dir / "summary_results.csv", index=False)
        return
    summary = (
        results.groupby(["model_id", "dataset", "split", "metric"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n_seeds"})
    )
    for baseline in ["scratch", "graph_bert", "hcmp_bert"]:
        baseline_values = summary[summary["model_id"] == baseline][
            ["dataset", "split", "metric", "mean"]
        ].rename(columns={"mean": f"{baseline}_mean"})
        summary = summary.merge(baseline_values, on=["dataset", "split", "metric"], how="left")
        summary[f"delta_vs_{baseline}"] = summary["mean"] - summary[f"{baseline}_mean"]
        summary = summary.drop(columns=[f"{baseline}_mean"])
    summary.to_csv(output_dir / "summary_results.csv", index=False)
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
