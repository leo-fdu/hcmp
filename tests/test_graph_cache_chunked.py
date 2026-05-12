from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from hcmp.data.descriptors import compute_descriptor_table
from hcmp.data.datasets import CachedGraphDataset


def _load_precompute_module():
    spec = importlib.util.spec_from_file_location(
        "hcmp_precompute_graph_cache_script",
        Path(__file__).resolve().parents[1] / "scripts" / "precompute_graph_cache.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_descriptor_alignment_requires_matching_clean_source_when_present():
    module = _load_precompute_module()
    clean_row = pd.Series({"canonical_smiles": "CCO", "source_row_index": 100})
    descriptor_row = pd.Series({"canonical_smiles": "CCO", "source_row_index": 0})

    with pytest.raises(module.DescriptorAlignmentError) as exc_info:
        module._validate_descriptor_row_alignment(
            clean_row=clean_row,
            descriptor_row=descriptor_row,
            canonical_smiles="CCO",
            global_row_index=0,
            chunk_index=2,
            local_idx=3,
            allow_unverified=False,
        )

    message = str(exc_info.value)
    assert "[chunk 2]" in message
    assert "row_offset=0" in message
    assert "local_row=3" in message
    assert "clean_source_row_index=100" in message
    assert "descriptor_source_row_index=0" in message
    assert "clean_smiles='CCO'" in message
    assert "descriptor_smiles='CCO'" in message


def test_precompute_graph_cache_chunked_writes_readable_variable_shards(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is required for graph cache precompute")

    clean_path = tmp_path / "clean.csv"
    descriptor_path = tmp_path / "descriptors.csv"
    output_dir = tmp_path / "graph_cache"
    smiles = ["CC", "CCC", "CCCl", "CCO"]
    source_row_indices = [100, 101, 102, 103]
    pd.DataFrame(
        {
            "canonical_smiles": smiles,
            "source_row_index": source_row_indices,
        }
    ).to_csv(clean_path, index=False)
    compute_descriptor_table(
        [
            (idx, value, source_row_index)
            for idx, (value, source_row_index) in enumerate(
                zip(smiles, source_row_indices, strict=True)
            )
        ]
    ).to_csv(descriptor_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/precompute_graph_cache.py",
            "--input-csv",
            str(clean_path),
            "--smiles-column",
            "canonical_smiles",
            "--descriptor-values",
            str(descriptor_path),
            "--output-dir",
            str(output_dir),
            "--feature-mode",
            "hcmp",
            "--chunk-size",
            "2",
            "--shard-size",
            "3",
        ],
        check=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is True
    assert manifest["cached_molecules"] == 4
    assert manifest["shard_counts"] == [2, 2]

    dataset = CachedGraphDataset(output_dir)
    assert len(dataset) == 4
    assert [dataset[idx].canonical_smiles for idx in range(4)] == smiles
    assert [dataset[idx].source_row_index for idx in range(4)] == source_row_indices
