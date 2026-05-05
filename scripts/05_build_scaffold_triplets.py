#!/usr/bin/env python
"""Build scaffold triplets using expanded-scaffold MCS distance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.data.scaffold_triplets import build_scaffold_triplets_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default="data/bbbp_smiles.csv")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output-path", default="data/processed/scaffold_triplets.csv")
    parser.add_argument("--distance-cache-path", default="data/processed/scaffold_distance.npy")
    parser.add_argument("--metadata-path", default="data/processed/scaffold_triplets_metadata.json")
    parser.add_argument("--min-distance-gap", type=float, default=0.15)
    parser.add_argument("--max-mcs-rounds", type=int, default=3)
    parser.add_argument("--max-triplets", type=int, default=10000)
    args = parser.parse_args()

    triplets, metadata = build_scaffold_triplets_from_csv(
        input_csv=args.input_csv,
        smiles_column=args.smiles_column,
        output_path=args.output_path,
        distance_cache_path=args.distance_cache_path,
        metadata_path=args.metadata_path,
        min_distance_gap=args.min_distance_gap,
        max_mcs_rounds=args.max_mcs_rounds,
        max_triplets=args.max_triplets,
    )
    requested_output = Path(args.output_path)
    actual_output = requested_output if requested_output.exists() else requested_output.with_suffix(requested_output.suffix + ".csv")
    print(f"Wrote {len(triplets)} scaffold triplets to {actual_output}")
    print(f"Wrote scaffold triplet metadata to {args.metadata_path}")
    print(
        "Scaffold triplet metadata: "
        f"num_molecules={metadata['num_molecules']}, "
        f"num_valid_triplets={metadata['num_valid_triplets']}, "
        f"max_mcs_rounds={metadata['max_mcs_rounds']}, "
        f"failures={metadata['failures']}"
    )


if __name__ == "__main__":
    main()
