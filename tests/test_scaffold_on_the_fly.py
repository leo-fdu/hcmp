from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from hcmp.data.scaffold_distance.cache import build_scaffold_distance_backend
from hcmp.data.scaffold_distance.io_utils import load_molecule_table
from hcmp.utils.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_on_the_fly_scaffold_backend_clears_batch_local_caches():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is required for scaffold triplet loss")

    import torch

    from hcmp.training.losses import sampled_in_batch_scaffold_triplet_loss

    table = load_molecule_table(
        pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]}),
        smiles_column="smiles",
        invalid_smiles="raise",
        sort_by_canonical_smiles=True,
    )
    backend = build_scaffold_distance_backend(
        "on_the_fly",
        molecule_table=table,
        max_in_memory_distances=32,
        max_in_memory_scaffolds=16,
    )
    projection = torch.randn((table.size, 8), dtype=torch.float32)
    global_indices = torch.arange(table.size, dtype=torch.long)

    _loss, metrics = sampled_in_batch_scaffold_triplet_loss(
        projection,
        global_indices,
        backend,
        candidate_pairs_per_anchor=2,
    )

    assert metrics["num_sampled_candidate_pairs"] > 0
    assert backend.stats()["batch_distances"] == 0
    assert backend.stats()["batch_scaffolds"] == 0


def test_tiny_on_the_fly_training_smoke_and_resume(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is required for training smoke test")
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML is required to write the smoke-test config")

    import torch
    import yaml

    input_csv = tmp_path / "molecules.csv"
    input_csv.write_text("smiles\nCCO\nc1ccccc1\nCC(=O)O\nCCN\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    config = load_config("configs/hcmp_pretrain.yaml")
    assert config["output"]["save_every_steps"] == 1000
    config["data"]["input_csv"] = str(input_csv)
    config["data"]["smiles_column"] = "smiles"
    config["dataset"]["graph_mode"] = "eager"
    config["dataset"]["graph_cache_dir"] = None
    config["training"]["num_epochs"] = 1
    config["training"]["graph_batch_size"] = 4
    config["training"]["device"] = "cpu"
    config["training"]["use_amp"] = False
    config["training"]["max_total_steps"] = 1
    config["output"]["run_dir"] = str(run_dir)
    config["output"]["save_every_steps"] = 1
    config["output"]["save_last"] = False
    config["output"]["save_best"] = False
    config["output"]["save_final"] = False
    config["model"]["encoder"]["hidden_dim"] = 16
    config["model"]["encoder"]["num_layers"] = 1
    config["model"]["encoder"]["num_heads"] = 2
    config["model"]["heads"]["mlp_hidden_dim"] = 16
    config["cut_seg"]["enabled"] = False
    config["prop_rank"]["enabled"] = False
    config["scaf_triplet"]["enabled"] = True
    config["scaf_triplet"]["distance_backend"] = "on_the_fly"
    config["scaf_triplet"]["cache_path"] = None
    config["scaf_triplet"]["max_in_memory_distances"] = 0
    config["scaf_triplet"]["max_in_memory_scaffolds"] = 0
    config["curriculum"]["enabled"] = False
    config["loss_balancing"]["weights"] = {
        "bert": 1.0,
        "cut_seg": 0.0,
        "prop_rank": 0.0,
        "scaf_triplet": 0.1,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/train_hcmp.py", "--config", str(config_path)],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "step_checkpoint_saved global_step=1 step_in_epoch=1" in result.stdout
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_kind"] == "step"
    assert checkpoint["global_step"] == 1
    assert checkpoint["step_in_epoch"] == 1
    assert checkpoint["config"]["output"]["save_every_steps"] == 1
    resume_result = subprocess.run(
        [
            sys.executable,
            "scripts/train_hcmp.py",
            "--config",
            str(config_path),
            "--resume-from",
            str(checkpoint_path),
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "epoch=1 starting" not in resume_result.stdout
    assert "step_checkpoint_saved" not in resume_result.stdout
    log_frame = pd.read_csv(run_dir / "train_log.csv")
    assert list(log_frame["global_step"]) == [1]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 1
    assert checkpoint["step_in_epoch"] == 1
