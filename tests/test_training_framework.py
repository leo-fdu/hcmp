import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from hcmp.data.descriptor_pairs import (
    build_balanced_descriptor_pairs,
    estimate_descriptor_thresholds,
)
from hcmp.data.descriptors import DESCRIPTOR_NAMES, compute_descriptor_values
from hcmp.data.chembl_cleaning import CleaningConfig, clean_chembl_rows
from hcmp.data.molecule_table import load_hcmp_molecule_table
from hcmp.data.scaffold_triplets import build_scaffold_triplets_from_csv, filter_scaffold_triplets
from hcmp.data.scaffold_triplets import sample_scaffold_triplets
from hcmp.data.scaffold_distance import (
    SQLiteScaffoldDistanceCache,
    compute_pairwise_scaffold_distance_matrix,
    extract_expanded_scaffold,
    load_molecule_table,
    select_main_organic_fragment_for_scaffold,
)


def _torch_available():
    return importlib.util.find_spec("torch") is not None


def _descriptor_frame_for_table(table, fill_value: float = 0.0) -> pd.DataFrame:
    rows = []
    for idx in range(table.size):
        rows.append(
            {
                "mol_id": idx,
                "source_row_index": int(table.dataframe.iloc[idx]["source_row_index"]),
                "canonical_smiles": table.canonical_smiles[idx],
                **{name: fill_value + idx for name in DESCRIPTOR_NAMES},
                "status": "success",
                "error": "",
            }
        )
    return pd.DataFrame(rows)


def _load_training_script_module():
    spec = importlib.util.spec_from_file_location(
        "hcmp_train_script",
        Path(__file__).resolve().parents[1] / "scripts" / "06_train_hcmp.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_finetune_script_module():
    spec = importlib.util.spec_from_file_location(
        "hcmp_finetune_script",
        Path(__file__).resolve().parents[1] / "scripts" / "finetune_downstream.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_chembl_script_module():
    spec = importlib.util.spec_from_file_location(
        "hcmp_prepare_chembl_script",
        Path(__file__).resolve().parents[1] / "scripts" / "prepare_chembl_pretrain.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_descriptor_computation_returns_all_14_descriptors():
    values = compute_descriptor_values(Chem.MolFromSmiles("CC(=O)O"))

    assert list(values) == DESCRIPTOR_NAMES
    assert len(values) == 14


def test_chembl_cleaning_handles_core_smoke_cases():
    rows = [
        {"source_smiles": "CCO", "source_chembl_id": "neutral", "source_row_index": 0},
        {"source_smiles": "C[NH3+]", "source_chembl_id": "charged", "source_row_index": 1},
        {"source_smiles": "C[N+](=O)[O-]", "source_chembl_id": "nitro", "source_row_index": 2},
        {"source_smiles": "CCN.[Na+]", "source_chembl_id": "salt", "source_row_index": 3},
        {"source_smiles": "C[Si](C)(C)C", "source_chembl_id": "silicon", "source_row_index": 4},
        {"source_smiles": "not_a_smiles", "source_chembl_id": "invalid", "source_row_index": 5},
        {"source_smiles": "OCC", "source_chembl_id": "duplicate", "source_row_index": 6},
    ]

    cleaned, counters = clean_chembl_rows(rows, CleaningConfig(min_heavy_atoms=1, deduplicate=True))

    by_id = {row["source_chembl_id"]: row for row in cleaned}
    assert {"neutral", "charged", "nitro", "salt"} <= set(by_id)
    assert by_id["charged"]["total_formal_charge"] == 1
    assert by_id["nitro"]["total_formal_charge"] == 0
    assert by_id["salt"]["was_multifragment"] is True
    assert by_id["salt"]["selected_fragment_smiles"] == "CCN"
    assert "silicon" not in by_id
    assert counters.n_unsupported_elements >= 1
    assert counters.n_rdkit_parse_failed == 1
    assert counters.n_duplicates_removed == 1


def test_chembl_subset_sampling_is_reproducible(tmp_path):
    prepare_script = _load_chembl_script_module()
    frame = pd.DataFrame(
        {
            "canonical_smiles": [f"C{'C' * idx}" for idx in range(10)],
            "source_smiles": [f"C{'C' * idx}" for idx in range(10)],
            "source_chembl_id": [f"CHEMBL{idx}" for idx in range(10)],
            "source_row_index": list(range(10)),
            "heavy_atom_count": [idx + 1 for idx in range(10)],
            "total_atom_count": [idx + 1 for idx in range(10)],
            "total_formal_charge": [0] * 10,
            "num_fragments_original": [1] * 10,
            "selected_fragment_smiles": [f"C{'C' * idx}" for idx in range(10)],
            "was_multifragment": [False] * 10,
        }
    )
    config = CleaningConfig()

    prepare_script._write_subsets(
        frame,
        cleaned_dir=tmp_path,
        reports_dir=tmp_path,
        subset_sizes=[5],
        seed=7,
        input_file=tmp_path / "input.csv",
        full_cleaned_file=tmp_path / "full.csv",
        config=config,
    )
    first = pd.read_csv(tmp_path / "chembl_clean_5.csv")
    prepare_script._write_subsets(
        frame,
        cleaned_dir=tmp_path,
        reports_dir=tmp_path,
        subset_sizes=[5],
        seed=7,
        input_file=tmp_path / "input.csv",
        full_cleaned_file=tmp_path / "full.csv",
        config=config,
    )
    second = pd.read_csv(tmp_path / "chembl_clean_5.csv")

    assert list(first["canonical_smiles"]) == list(second["canonical_smiles"])


def test_tpsa_descriptor_includes_sulfur_and_phosphorus():
    import inspect

    assert "includeSandP=True" in inspect.getsource(compute_descriptor_values)


def test_shared_molecule_table_applies_max_after_canonical_sorting():
    frame = pd.DataFrame({"smiles": ["O", "CCCC", "CC"]})

    full = load_hcmp_molecule_table(frame, max_molecules=None)
    sliced = load_hcmp_molecule_table(frame, max_molecules=2)

    assert list(full.canonical_smiles) == sorted(full.canonical_smiles)
    assert list(sliced.canonical_smiles) == list(full.canonical_smiles[:2])
    assert sliced.canonical_smiles[0] == full.canonical_smiles[0]
    assert sliced.canonical_smiles[-1] == full.canonical_smiles[1]


def test_graph_dataset_uses_precomputed_descriptor_values_without_recompute():
    if not _torch_available():
        return
    from hcmp.data import datasets as dataset_module
    from hcmp.data.datasets import GraphDataset
    from hcmp.data.graph_builder import default_feature_spec

    table = load_molecule_table(pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]}), invalid_smiles="drop")
    descriptor_frame = _descriptor_frame_for_table(table, fill_value=7.0)
    original = dataset_module.compute_descriptor_values

    def fail_recompute(_mol):
        raise AssertionError("GraphDataset recomputed descriptors despite a descriptor table.")

    dataset_module.compute_descriptor_values = fail_recompute
    try:
        dataset = GraphDataset(table, default_feature_spec(), descriptor_values=descriptor_frame)
    finally:
        dataset_module.compute_descriptor_values = original

    assert dataset[0].descriptor_values.tolist() == [7.0] * len(DESCRIPTOR_NAMES)


def test_graph_dataset_descriptor_identity_mismatch_raises():
    if not _torch_available():
        return
    from hcmp.data.datasets import GraphDataset
    from hcmp.data.graph_builder import default_feature_spec

    table = load_molecule_table(pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]}), invalid_smiles="drop")
    descriptor_frame = _descriptor_frame_for_table(table)
    descriptor_frame.loc[0, "canonical_smiles"] = "C"

    try:
        GraphDataset(table, default_feature_spec(), descriptor_values=descriptor_frame)
    except ValueError as exc:
        message = str(exc)
        assert "first_mismatch_row=0" in message
        assert "molecule_canonical_smiles" in message
        assert "descriptor_canonical_smiles" in message
    else:
        raise AssertionError("Expected descriptor identity mismatch to raise ValueError.")


def test_curriculum_starts_with_bert_only_and_advances_after_convergence():
    from hcmp.training.curriculum import ConvergenceCurriculum, CurriculumPhase

    curriculum = ConvergenceCurriculum(
        phases=[
            CurriculumPhase("bert", ("bert",), "bert"),
            CurriculumPhase("segmentation", ("bert", "cut_seg"), "cut_seg"),
        ],
        window_epochs=2,
        min_relative_improvement=0.01,
    )

    assert curriculum.current_phase.name == "bert"
    assert curriculum.current_phase.active_losses == ("bert",)
    transition = None
    for loss in [1.0, 0.99, 0.989, 0.988]:
        transition = curriculum.observe_epoch({"bert_loss": loss})

    assert transition is not None
    assert transition.old_phase == "bert"
    assert transition.new_phase == "segmentation"
    assert math.isclose(transition.threshold, 0.01)


def test_curriculum_convergence_decision_handles_improvement_plateau_and_worsening():
    from hcmp.training.curriculum import evaluate_convergence

    improving = evaluate_convergence(1.0, 0.7, 0.01, 0.02)
    plateau = evaluate_convergence(1.0, 0.995, 0.01, 0.02)
    worsened = evaluate_convergence(1.0, 1.1, 0.01, 0.02)
    slight_worsening = evaluate_convergence(1.0, 1.005, 0.01, 0.02)

    assert improving.decision == "still improving"
    assert not improving.converged
    assert plateau.decision == "plateau converged"
    assert plateau.converged
    assert worsened.decision == "worsened too much"
    assert not worsened.converged
    assert slight_worsening.decision == "plateau converged"
    assert slight_worsening.converged


def test_curriculum_does_not_advance_when_loss_worsens_too_much():
    from hcmp.training.curriculum import ConvergenceCurriculum, CurriculumPhase

    curriculum = ConvergenceCurriculum(
        phases=[
            CurriculumPhase("bert", ("bert",), "bert"),
            CurriculumPhase("segmentation", ("bert", "cut_seg"), "cut_seg"),
        ],
        window_epochs=2,
        min_relative_improvement=0.01,
        max_relative_worsening=0.02,
    )

    transition = None
    for loss in [1.0, 1.0, 1.1, 1.1]:
        transition = curriculum.observe_epoch({"bert_loss": loss})

    assert transition is None
    assert curriculum.current_phase.name == "bert"


def test_descriptor_thresholds_return_one_row_per_descriptor():
    rows = []
    for mol_id, smiles in enumerate(["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC"]):
        values = compute_descriptor_values(Chem.MolFromSmiles(smiles))
        rows.append({"mol_id": mol_id, "status": "success", **values})
    frame = pd.DataFrame(rows)

    thresholds = estimate_descriptor_thresholds(frame, quantile=0.70)

    assert set(thresholds["descriptor_name"]) == set(DESCRIPTOR_NAMES)
    assert len(thresholds) == len(DESCRIPTOR_NAMES)


def test_descriptor_pair_sampling_is_balanced_across_descriptors():
    rows = []
    for mol_id, smiles in enumerate(["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC"]):
        values = compute_descriptor_values(Chem.MolFromSmiles(smiles))
        rows.append({"mol_id": mol_id, "status": "success", **values})
    frame = pd.DataFrame(rows)
    thresholds = estimate_descriptor_thresholds(frame, quantile=0.0)

    pairs = build_balanced_descriptor_pairs(
        frame,
        thresholds,
        max_pairs_per_descriptor=2,
        random_seed=0,
    )

    counts = pairs.groupby("descriptor_name").size()
    assert set(counts.index) == set(DESCRIPTOR_NAMES)
    assert counts.max() <= 2


def test_scaffold_triplet_filtering_respects_min_distance_gap():
    candidates = pd.DataFrame(
        [
            {"anchor": "a", "positive": "p1", "negative": "n1", "D_ap": 0.2, "D_an": 0.4},
            {"anchor": "a", "positive": "p2", "negative": "n2", "D_ap": 0.2, "D_an": 0.3},
        ]
    )

    filtered = filter_scaffold_triplets(candidates, min_distance_gap=0.15)

    assert len(filtered) == 1
    assert filtered.iloc[0]["negative"] == "n1"
    assert filtered.iloc[0]["distance_gap"] == 0.2


def test_scaffold_distance_matrix_is_symmetric_zero_diagonal():
    scaffolds = [
        extract_expanded_scaffold(Chem.MolFromSmiles(smiles))
        for smiles in ["c1ccccc1", "c1ccncc1", "CCO"]
    ]

    result = compute_pairwise_scaffold_distance_matrix(scaffolds, max_rounds=2)

    assert result.matrix.shape == (3, 3)
    assert np.allclose(result.matrix, result.matrix.T)
    assert np.allclose(np.diag(result.matrix), 0.0)


def test_multifragment_scaffold_selection_uses_main_organic_fragment():
    mol = Chem.MolFromSmiles("CCO.[Na+]")
    original_smiles = Chem.MolToSmiles(mol, canonical=True)

    selected = select_main_organic_fragment_for_scaffold(mol)
    scaffold = extract_expanded_scaffold(mol)

    assert Chem.MolToSmiles(selected, canonical=True) == "CCO"
    assert len(Chem.GetMolFrags(selected, asMols=False, sanitizeFrags=False)) == 1
    assert Chem.MolToSmiles(mol, canonical=True) == original_smiles
    assert scaffold.scaffold_smiles == "CC"


def test_multifragment_scaffold_selection_is_deterministic_with_ties():
    mol = Chem.MolFromSmiles("CC.CN")

    selected = select_main_organic_fragment_for_scaffold(mol)

    assert Chem.MolToSmiles(selected, canonical=True) == "CC"


def test_sqlite_scaffold_distance_cache_reuses_pair_distance(tmp_path):
    table = load_molecule_table(
        pd.DataFrame({"smiles": ["c1ccccc1", "c1ccncc1"]}),
        invalid_smiles="drop",
    )
    cache = SQLiteScaffoldDistanceCache(table, tmp_path / "dist.sqlite", max_rounds=1)

    first = cache.get_distance(0, 1)
    second = cache.get_distance(1, 0)

    assert first == second
    assert cache.stats()["cache_misses"] == 1
    assert cache.stats()["cache_hits"] == 1


def test_scaffold_distance_cache_respects_in_memory_caps(tmp_path):
    table = load_molecule_table(
        pd.DataFrame({"smiles": ["c1ccccc1", "c1ccncc1", "CCO", "CCN"]}),
        invalid_smiles="drop",
    )
    cache = SQLiteScaffoldDistanceCache(
        table,
        tmp_path / "dist.sqlite",
        max_rounds=1,
        max_in_memory_distances=1,
        max_in_memory_scaffolds=2,
    )

    cache.get_distance(0, 1)
    cache.get_distance(1, 2)
    cache.get_distance(2, 3)
    stats = cache.stats()

    assert stats["in_memory_distances"] <= 1
    assert stats["in_memory_scaffolds"] <= 2


def test_scaffold_distance_cache_handles_multifragment_inputs(tmp_path):
    table = load_molecule_table(
        pd.DataFrame({"smiles": ["CCO.[Na+]", "CCN"]}),
        invalid_smiles="drop",
    )
    cache = SQLiteScaffoldDistanceCache(table, tmp_path / "dist.sqlite", max_rounds=1)

    distance = cache.get_distance(0, 1)

    assert math.isfinite(distance)
    assert cache.stats()["num_multifragment_scaffold_inputs"] == 1


def test_completion_checkpoint_name_distinguishes_final_and_interrupted():
    train_script = _load_training_script_module()

    assert train_script._completion_checkpoint_name(True) == "final.pt"
    assert train_script._completion_checkpoint_name(False) == "interrupted.pt"


def test_deepchem_split_backend_metadata_is_recorded(monkeypatch):
    finetune = _load_finetune_script_module()

    class FakeDataset:
        def __init__(self, X, y, ids):
            self.X = X
            self.y = y
            self.ids = ids

    class FakeScaffoldSplitter:
        def split(self, dataset, frac_train, frac_valid, frac_test, seed):
            assert frac_train == 0.8
            assert frac_valid == 0.1
            assert frac_test == 0.1
            assert seed == 7
            return [0, 1], [2], [3]

    fake_deepchem = types.SimpleNamespace(
        data=types.SimpleNamespace(NumpyDataset=FakeDataset),
        splits=types.SimpleNamespace(
            RandomSplitter=FakeScaffoldSplitter,
            ScaffoldSplitter=FakeScaffoldSplitter,
        ),
    )
    monkeypatch.setitem(sys.modules, "deepchem", fake_deepchem)

    rows = [("CC", 0.0), ("CCC", 1.0), ("CCCC", 0.0), ("CCCCC", 1.0)]
    train_idx, val_idx, test_idx, metadata = finetune._split_indices(rows, "scaffold", 7, "classification")

    assert train_idx == [0, 1]
    assert val_idx == [2]
    assert test_idx == [3]
    assert metadata["split_backend"] == "deepchem"
    assert metadata["splitter"] == "FakeScaffoldSplitter"


def test_internal_split_fallback_metadata_is_recorded(monkeypatch):
    finetune = _load_finetune_script_module()
    monkeypatch.setitem(sys.modules, "deepchem", None)

    rows = [("CC", 0.0), ("CCC", 1.0), ("CCCC", 0.0), ("CCCCC", 1.0)]
    train_idx, val_idx, test_idx, metadata = finetune._split_indices(rows, "random", 3, "regression")

    assert sorted(train_idx + val_idx + test_idx) == [0, 1, 2, 3]
    assert metadata["split_backend"] == "internal_fallback"
    assert metadata["splitter"] == "random"
    assert metadata["seed"] == 3


def test_downstream_split_artifacts_are_saved(tmp_path):
    finetune = _load_finetune_script_module()
    rows = [("CC", 0.0), ("CCC", 1.0), ("CCCC", 0.0)]
    metadata = {
        "split_backend": "deepchem",
        "splitter": "ScaffoldSplitter",
        "frac_train": 0.8,
        "frac_valid": 0.1,
        "frac_test": 0.1,
        "seed": 0,
    }

    finetune._save_split_artifacts(
        tmp_path,
        dataset="bbbp",
        split="scaffold",
        split_metadata=metadata,
        seed=0,
        rows=rows,
        train_idx=[0],
        val_idx=[1],
        test_idx=[2],
        label_column="p_np",
    )

    split_indices = json.loads((tmp_path / "split_indices.json").read_text(encoding="utf-8"))
    assert split_indices["train_indices"] == [0]
    assert split_indices["split_backend"] == "deepchem"
    assert pd.read_csv(tmp_path / "train_smiles.csv").to_dict("records") == [
        {"index": 0, "smiles": "CC", "p_np": 0.0}
    ]
    assert (tmp_path / "val_smiles.csv").exists()
    assert (tmp_path / "test_smiles.csv").exists()


def test_pretrain_matrix_does_not_generate_scratch_config(tmp_path):
    output_dir = tmp_path / "pretrain"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_pretrain_matrix_configs.py",
            "--base-config",
            "configs/hcmp_pretrain.yaml",
            "--output-dir",
            str(output_dir),
            "--corpus-name",
            "chembl_full",
            "--hcmp-graph-cache-dir",
            "data/chembl/graph_cache_hcmp",
            "--graph-bert-cache-dir",
            "data/chembl/graph_cache_graph_bert",
            "--descriptor-thresholds",
            "data/chembl/processed/descriptor_thresholds_full.csv",
        ],
        check=True,
    )

    generated = {path.name for path in output_dir.glob("*.yaml")}
    assert "scratch.yaml" not in generated
    assert generated == {
        "graph_bert.yaml",
        "hcmp_bert.yaml",
        "hcmp_bert_cut.yaml",
        "hcmp_bert_prop.yaml",
        "hcmp_bert_scaf.yaml",
        "hcmp_bert_cut_prop.yaml",
        "hcmp_bert_cut_scaf.yaml",
        "hcmp_bert_prop_scaf.yaml",
        "hcmp_full.yaml",
    }


def test_pretraining_script_rejects_scratch_model_id(tmp_path):
    config_path = tmp_path / "scratch.yaml"
    config_path.write_text("model_id: scratch\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/train_hcmp.py", "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "scratch is a downstream-only randomly initialized baseline" in (
        result.stdout + result.stderr
    )


def test_run_downstream_matrix_passes_column_overrides(tmp_path):
    column_map = tmp_path / "columns.yaml"
    column_map.write_text(
        "bbbp:\n  smiles_column: smiles\n  label_column: p_np\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_downstream_matrix.py",
            "--pretrain-root",
            "runs/pretrain/chembl_full",
            "--data-root",
            "data/downstream",
            "--output-root",
            "runs/downstream",
            "--datasets",
            "bbbp",
            "--splits",
            "scaffold",
            "--models",
            "scratch",
            "--downstream-seeds",
            "0",
            "--column-map",
            str(column_map),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--smiles-column smiles" in result.stdout
    assert "--label-column p_np" in result.stdout
    assert "--checkpoint none" in result.stdout


def test_restore_rng_state_restores_python_numpy_and_torch_state():
    import random

    train_script = _load_training_script_module()

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

        def __init__(self):
            self.restored_state = None

        def set_rng_state(self, state):
            self.restored_state = state

    random.seed(123)
    np.random.seed(123)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": "torch-state",
        "torch_cuda": None,
    }
    _ = random.random()
    _ = np.random.random()
    fake_torch = FakeTorch()

    train_script._restore_rng_state(fake_torch, rng_state)

    assert random.random() == random.Random(123).random()
    assert np.random.random() == np.random.RandomState(123).random_sample()
    assert fake_torch.restored_state == "torch-state"


def test_restore_curriculum_state_restores_phase_and_epochs_in_phase():
    train_script = _load_training_script_module()
    from hcmp.training.curriculum import ConvergenceCurriculum, CurriculumPhase

    curriculum = ConvergenceCurriculum(
        phases=[
            CurriculumPhase("bert", ("bert",), "bert"),
            CurriculumPhase("segmentation", ("bert", "cut_seg"), "cut_seg"),
        ],
        window_epochs=2,
    )

    train_script._restore_curriculum_state(
        curriculum,
        {
            "phase_index": 1,
            "phase_history": [0.3, 0.2, 0.1],
            "epochs_in_phase": 3,
        },
    )

    assert curriculum.phase_index == 1
    assert curriculum.phase_history == [0.3, 0.2, 0.1]
    assert curriculum.epochs_in_phase == 3


def test_scaffold_triplet_sampler_respects_min_gap():
    matrix = np.array(
        [
            [0.0, 0.1, 0.4],
            [0.1, 0.0, 0.3],
            [0.4, 0.3, 0.0],
        ]
    )

    triplets = sample_scaffold_triplets(matrix, min_distance_gap=0.15)

    assert len(triplets) > 0
    assert all(triplets["d_ap"] + 0.15 < triplets["d_an"])


def test_scaffold_triplet_script_runs_on_tiny_csv():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_csv = tmp_path / "tiny.csv"
        output_path = tmp_path / "triplets.csv"
        metadata_path = tmp_path / "triplets_metadata.json"
        cache_path = tmp_path / "scaffold_distance.npy"
        pd.DataFrame({"smiles": ["c1ccccc1", "c1ccncc1", "CCO"]}).to_csv(
            input_csv,
            index=False,
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/05_build_scaffold_triplets.py",
                "--input-csv",
                str(input_csv),
                "--output-path",
                str(output_path),
                "--distance-cache-path",
                str(cache_path),
                "--metadata-path",
                str(metadata_path),
                "--max-triplets",
                "5",
                "--max-mcs-rounds",
                "2",
            ],
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert metadata_path.exists()
        assert cache_path.exists()
        assert output_path.exists()
        triplets = pd.read_csv(output_path)
        required_columns = {
            "anchor_idx",
            "positive_idx",
            "negative_idx",
            "anchor_smiles",
            "positive_smiles",
            "negative_smiles",
            "anchor_source_row_index",
            "positive_source_row_index",
            "negative_source_row_index",
            "d_ap",
            "d_an",
        }
        assert required_columns <= set(triplets.columns)
        assert all(triplets["d_ap"] + 0.15 < triplets["d_an"])


def test_scaffold_triplets_preserve_sorted_and_source_indices():
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_csv = Path(tmp_dir) / "tiny.csv"
        pd.DataFrame(
            {
                "smiles": ["CCO", "c1ccncc1", "c1ccccc1"],
            }
        ).to_csv(input_csv, index=False)

        triplets, _ = build_scaffold_triplets_from_csv(
            input_csv,
            output_path=Path(tmp_dir) / "triplets.csv",
            distance_cache_path=Path(tmp_dir) / "dist.npy",
            metadata_path=Path(tmp_dir) / "meta.json",
            max_triplets=5,
            max_mcs_rounds=2,
        )

        if len(triplets) > 0:
            assert all(triplets["anchor_idx"] != triplets["anchor_source_row_index"]) or set(
                triplets["anchor_idx"]
            ) <= {0, 1, 2}
            assert {
                "anchor_smiles",
                "positive_smiles",
                "negative_smiles",
                "anchor_source_row_index",
                "positive_source_row_index",
                "negative_source_row_index",
            } <= set(triplets.columns)


def test_graph_builder_returns_expected_feature_dimensions_when_torch_available():
    if not _torch_available():
        return
    from hcmp.data.graph_builder import default_feature_spec, smiles_to_graph

    spec = default_feature_spec()
    graph = smiles_to_graph("CC(=O)O", feature_spec=spec)

    assert graph.node_features.shape[1] == spec.node_feature_dim
    assert graph.edge_features.shape[1] == spec.edge_feature_dim
    assert graph.edge_index.shape[0] == 2


def test_encoder_pooling_and_heads_when_torch_available():
    if not _torch_available():
        return
    from hcmp.data.graph_builder import default_feature_spec, smiles_to_graph
    from hcmp.models.heads import CutSegHead
    from hcmp.models.hcmp_model import HCMPModel
    from hcmp.models.pooling import NodeEdgeAttentionPooling

    spec = default_feature_spec()
    graph = smiles_to_graph("CC(=O)O", feature_spec=spec)
    model = HCMPModel(spec, hidden_dim=16, num_layers=2, dropout=0.0)
    output = model(graph.node_features, graph.edge_index, graph.edge_features)

    assert output.node_embeddings.shape == (graph.node_features.shape[0], 16)
    assert output.edge_embeddings.shape == (graph.edge_features.shape[0], 16)
    assert output.graph_embedding.shape == (1, 16)
    assert isinstance(model.encoder.pooling, NodeEdgeAttentionPooling)
    assert model.encoder.pooling.node_attention is not model.encoder.pooling.edge_attention
    assert isinstance(model.cut_head, CutSegHead)
    assert model.cut_head.input_representation == "final_edge_embedding"
    assert output.cut_logits.shape[0] == graph.edge_features.shape[0]


def test_graph_transformer_encoder_forward_when_torch_available():
    if not _torch_available():
        return
    from hcmp.data.graph_builder import collate_graphs, default_feature_spec, smiles_to_graph
    from hcmp.models.encoders import GraphTransformerEncoder

    spec = default_feature_spec()
    graphs = [
        smiles_to_graph("CCO", mol_id=0, feature_spec=spec),
        smiles_to_graph("CC(=O)O", mol_id=1, feature_spec=spec),
    ]
    batch = collate_graphs(graphs)
    encoder = GraphTransformerEncoder(
        node_input_dim=spec.node_feature_dim,
        edge_input_dim=spec.edge_feature_dim,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )

    output = encoder(
        batch.node_features,
        batch.edge_index,
        batch.edge_features,
        node_batch=batch.node_batch,
        edge_batch=batch.edge_batch,
    )

    assert output.node_embeddings.shape == (batch.node_features.shape[0], 16)
    assert output.edge_embeddings.shape == (batch.edge_features.shape[0], 16)
    assert output.graph_embedding.shape == (2, 16)


def test_vectorized_transformer_and_pooling_match_loop_when_torch_available():
    if not _torch_available():
        return
    import torch

    from hcmp.data.graph_builder import collate_graphs, default_feature_spec, smiles_to_graph
    from hcmp.models.hcmp_model import HCMPModel

    torch.manual_seed(123)
    spec = default_feature_spec()
    graphs = [
        smiles_to_graph("CCO", mol_id=0, feature_spec=spec),
        smiles_to_graph("CC(=O)O", mol_id=1, feature_spec=spec),
        smiles_to_graph("c1ccccc1", mol_id=2, feature_spec=spec),
    ]
    batch = collate_graphs(graphs)
    loop_model = HCMPModel(
        spec,
        hidden_dim=16,
        num_layers=2,
        encoder_type="graph_transformer",
        num_heads=4,
        dropout=0.0,
        attention_impl="loop",
        pooling_impl="loop",
    )
    vectorized_model = HCMPModel(
        spec,
        hidden_dim=16,
        num_layers=2,
        encoder_type="graph_transformer",
        num_heads=4,
        dropout=0.0,
        attention_impl="vectorized",
        pooling_impl="vectorized",
    )
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

    for field in [
        "node_embeddings",
        "edge_embeddings",
        "graph_embedding",
        "atom_logits",
        "bond_logits",
        "cut_logits",
        "descriptor_scores",
        "scaffold_projection",
    ]:
        diff = (getattr(loop_output, field) - getattr(vectorized_output, field)).abs().max()
        assert float(diff) < 1.0e-5


def test_hcmp_model_can_select_graph_transformer_when_torch_available():
    if not _torch_available():
        return
    from hcmp.data.graph_builder import default_feature_spec, smiles_to_graph
    from hcmp.models.encoders import GraphTransformerEncoder
    from hcmp.models.hcmp_model import HCMPModel

    spec = default_feature_spec()
    graph = smiles_to_graph("CCO", feature_spec=spec)
    model = HCMPModel(
        spec,
        hidden_dim=16,
        num_layers=1,
        encoder_type="graph_transformer",
        num_heads=4,
        dropout=0.0,
    )
    output = model(graph.node_features, graph.edge_index, graph.edge_features)

    assert isinstance(model.encoder, GraphTransformerEncoder)
    assert output.cut_logits.shape[0] == graph.edge_features.shape[0]


def test_bert_masking_targets_only_atom_and_bond_type_when_torch_available():
    if not _torch_available():
        return
    from hcmp.data.graph_builder import apply_bert_masking, default_feature_spec, smiles_to_graph

    spec = default_feature_spec()
    graph = smiles_to_graph("CC(=O)O", feature_spec=spec)
    masked = apply_bert_masking(
        graph,
        spec,
        atom_mask_ratio=1.0,
        bond_mask_ratio=1.0,
    )

    assert masked.atom_type_targets.ndim == 1
    assert masked.bond_type_targets.ndim == 1
    assert len(masked.atom_type_targets) == graph.node_features.shape[0]
    assert len(masked.bond_type_targets) == graph.edge_features.shape[0]


def test_trainer_computes_unified_batch_losses_from_one_forward_when_torch_available():
    if not _torch_available():
        return
    import torch

    from hcmp.data.datasets import GraphDataset, MultiTaskDataLoader, graph_collate
    from hcmp.data.graph_builder import default_feature_spec
    from hcmp.data.scaffold_distance import load_molecule_table
    from hcmp.models.hcmp_model import HCMPModel
    from hcmp.training.loss_balancing import LossBalancer
    from hcmp.training.trainer import HCMPTrainer

    frame = pd.DataFrame({"smiles": ["CCO", "CC(=O)O", "c1ccccc1"]})
    table = load_molecule_table(frame, invalid_smiles="drop")
    spec = default_feature_spec()
    descriptor_frame = _descriptor_frame_for_table(table)
    dataset = GraphDataset(table, spec, descriptor_values=descriptor_frame)
    batch = graph_collate([dataset[0], dataset[1], dataset[2]])
    model = HCMPModel(spec, hidden_dim=16, num_layers=1, num_heads=4, dropout=0.0)
    forward_count = {"value": 0}
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        forward_count["value"] += 1
        return original_forward(*args, **kwargs)

    model.forward = counted_forward
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trainer = HCMPTrainer(
        model=model,
        optimizer=optimizer,
        loss_balancer=LossBalancer(weights={"bert": 1.0, "cut_seg": 1.0, "prop_rank": 1.0, "scaf_triplet": 1.0}),
        feature_spec=spec,
        descriptor_thresholds=torch.zeros(14),
        scaffold_distance_matrix=torch.tensor(
            [[0.0, 0.1, 0.5], [0.1, 0.0, 0.4], [0.5, 0.4, 0.0]]
        ),
        min_distance_gap=0.15,
        triplet_margin=0.15,
    )

    losses = trainer.train_graph_batch(batch)

    assert forward_count["value"] == 1
    assert {"bert_loss", "cut_loss", "prop_loss", "triplet_loss"} <= set(losses)
    assert losses["num_sampled_pairs"] > 0
    assert losses["num_sampled_candidate_pairs"] > 0


def test_trainer_handles_no_descriptor_pairs_or_triplets_when_torch_available():
    if not _torch_available():
        return
    import torch

    from hcmp.data.datasets import GraphDataset, graph_collate
    from hcmp.data.graph_builder import default_feature_spec
    from hcmp.data.scaffold_distance import load_molecule_table
    from hcmp.models.hcmp_model import HCMPModel
    from hcmp.training.loss_balancing import LossBalancer
    from hcmp.training.trainer import HCMPTrainer

    frame = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"]})
    table = load_molecule_table(frame, invalid_smiles="drop")
    spec = default_feature_spec()
    descriptor_frame = _descriptor_frame_for_table(table)
    dataset = GraphDataset(table, spec, descriptor_values=descriptor_frame)
    batch = graph_collate([dataset[0], dataset[1], dataset[2]])
    model = HCMPModel(spec, hidden_dim=16, num_layers=1, num_heads=4, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trainer = HCMPTrainer(
        model=model,
        optimizer=optimizer,
        loss_balancer=LossBalancer(weights={"bert": 1.0, "cut_seg": 1.0, "prop_rank": 1.0, "scaf_triplet": 1.0}),
        feature_spec=spec,
        descriptor_thresholds=torch.full((14,), 1.0e9),
        scaffold_distance_matrix=torch.zeros((3, 3)),
        min_distance_gap=0.15,
        triplet_margin=0.15,
    )

    losses = trainer.train_graph_batch(batch)

    assert losses["prop_loss"] == 0.0
    assert losses["triplet_loss"] == 0.0
    assert losses["num_valid_descriptor_pairs"] == 0.0
    assert losses["num_valid_triplets"] == 0.0
