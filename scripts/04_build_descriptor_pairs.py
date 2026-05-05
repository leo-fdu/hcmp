#!/usr/bin/env python
"""Build descriptor thresholds and balanced descriptor ranking pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.data.descriptor_pairs import (
    build_balanced_descriptor_pairs,
    estimate_descriptor_thresholds,
)
from hcmp.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-values", default="data/processed/descriptor_values.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--threshold-quantile", type=float, default=0.70)
    parser.add_argument("--max-pairs-per-descriptor", type=int, default=None)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    values = pd.read_csv(args.descriptor_values)
    if args.max_molecules is not None:
        values = _slice_descriptor_values(values, int(args.max_molecules))
    thresholds = estimate_descriptor_thresholds(
        values,
        quantile=args.threshold_quantile,
        max_pairs_per_descriptor=args.max_pairs_per_descriptor,
        random_seed=args.seed,
    )
    pairs = build_balanced_descriptor_pairs(
        values,
        thresholds,
        max_pairs_per_descriptor=args.max_pairs_per_descriptor,
        random_seed=args.seed,
    )
    thresholds_csv = output_dir / "descriptor_thresholds.csv"
    thresholds_json = output_dir / "descriptor_thresholds.json"
    pairs_path = output_dir / "prop_rank_pairs.parquet"
    thresholds.to_csv(thresholds_csv, index=False)
    thresholds_json.write_text(
        json.dumps(thresholds.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    written_pairs_path = _write_parquet_or_csv(pairs, pairs_path)
    print(f"Wrote descriptor thresholds to {thresholds_csv} and {thresholds_json}")
    print(f"Wrote {len(pairs)} descriptor ranking pairs to {written_pairs_path}")
    print(
        "Training uses descriptor_values.csv as the descriptor source of truth; "
        "regenerate descriptor values and thresholds together, especially after "
        "the TPSA includeSandP=True update."
    )


def _write_parquet_or_csv(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception as exc:
        fallback = path.with_suffix(path.suffix + ".csv")
        frame.to_csv(fallback, index=False)
        print(f"Could not write parquet ({exc}); wrote CSV fallback to {fallback}")
        return fallback


def _slice_descriptor_values(values: pd.DataFrame, max_molecules: int) -> pd.DataFrame:
    full_count = len(values)
    sliced = values.head(max_molecules).reset_index(drop=True)
    print(f"max_molecules={max_molecules} applied after canonicalization/sorting")
    print(f"full valid molecule count before slicing={full_count}")
    print(f"selected molecule count after slicing={len(sliced)}")
    if len(sliced) > 0 and "canonical_smiles" in sliced.columns:
        print(f"first selected canonical_smiles={sliced.iloc[0]['canonical_smiles']}")
        print(f"last selected canonical_smiles={sliced.iloc[-1]['canonical_smiles']}")
    return sliced


if __name__ == "__main__":
    main()
