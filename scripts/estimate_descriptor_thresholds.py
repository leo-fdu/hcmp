#!/usr/bin/env python
"""Estimate descriptor ranking thresholds with shuffled adjacent pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.data.descriptors import DESCRIPTOR_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-values", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--quantile", type=float, default=0.70)
    parser.add_argument("--num-shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    values = pd.read_csv(args.descriptor_values)
    missing = [name for name in DESCRIPTOR_NAMES if name not in values.columns]
    if missing:
        raise ValueError(f"Descriptor values are missing columns: {missing}")
    if "status" in values.columns:
        values = values[values["status"] == "success"].reset_index(drop=True)

    matrix = values[DESCRIPTOR_NAMES].to_numpy(dtype=np.float32, copy=True)
    rng = np.random.default_rng(int(args.seed))
    diffs = []
    for _ in range(int(args.num_shuffles)):
        perm = rng.permutation(matrix.shape[0])
        adjacent = np.abs(matrix[perm[1:]] - matrix[perm[:-1]])
        diffs.append(adjacent)
    sampled = np.concatenate(diffs, axis=0) if diffs else np.empty((0, len(DESCRIPTOR_NAMES)))
    thresholds = np.nanquantile(sampled, float(args.quantile), axis=0)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "descriptor": DESCRIPTOR_NAMES,
            "threshold": thresholds,
            "quantile": float(args.quantile),
            "num_pairs_used": int(sampled.shape[0]),
            "num_shuffles": int(args.num_shuffles),
            "seed": int(args.seed),
        }
    )
    frame.to_csv(output, index=False)
    print(f"Wrote descriptor thresholds to {output}")
    print(f"num_pairs_used={sampled.shape[0]}")


if __name__ == "__main__":
    main()
