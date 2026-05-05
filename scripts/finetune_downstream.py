#!/usr/bin/env python
"""Finetune HCMP/Graph-BERT checkpoints on MoleculeNet-style CSV datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CLASSIFICATION = {"bbbp", "hiv"}
REGRESSION = {"esol", "freesolv", "lipophilicity"}
graph_collate = None
default_feature_spec = None
smiles_to_graph = None
HCMPModel = None
move_graph_batch_to_device = None
_DownstreamModel = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["random", "scaffold"], default="random")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain-seed", type=int, default=0)
    parser.add_argument("--data-root", default="data/downstream")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=None, help="Alias for --max-epochs.")
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset
    except ModuleNotFoundError as exc:
        raise SystemExit("Torch is required for downstream finetuning. Install torch and retry.") from exc

    _ensure_training_imports()
    _seed_everything(args.seed, torch)
    if args.epochs is not None:
        args.max_epochs = args.epochs
    if args.checkpoint is None:
        if args.model_id == "scratch":
            args.checkpoint = "none"
        else:
            raise ValueError("--checkpoint is required unless --model-id scratch.")
    task_type = "classification" if args.dataset.lower() in CLASSIFICATION else "regression"
    frame = _load_downstream_frame(args.data_root, args.dataset, data_path=args.data_path)
    smiles_column = args.smiles_column or _detect_column(frame, _dataset_smiles_candidates(args.dataset))
    label_column = args.label_column or _detect_label_column(frame, smiles_column, args.dataset)
    rows = _build_rows(frame, smiles_column, label_column, task_type)
    feature_spec, model_config, state_dict, checkpoint_config, checkpoint_metadata = _checkpoint_context(args.checkpoint, torch)
    dataset = _GraphLabelDataset(rows, feature_spec)
    train_idx, val_idx, test_idx, split_metadata = _split_indices(rows, args.split, args.seed, task_type)
    loaders = {
        "train": DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=_collate),
        "val": DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=_collate),
        "test": DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=_collate),
    }

    device = torch.device(args.device)
    encoder = HCMPModel(feature_spec=feature_spec, enabled_heads={"bert": False, "cut_seg": False, "prop_rank": False, "scaf_triplet": False}, **model_config)
    if state_dict is not None:
        encoder.load_state_dict(state_dict, strict=False)
    model = _DownstreamModel(encoder, int(model_config["hidden_dim"]), task_type).to(device)
    downstream_config = (checkpoint_config or {}).get("downstream", {})
    encoder_lr = float(downstream_config.get("encoder_lr", 2.0e-4))
    head_lr = float(downstream_config.get("head_lr", 2.0e-4))
    weight_decay = float(downstream_config.get("weight_decay", 1.0e-2))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_split_artifacts(
        output_dir,
        dataset=args.dataset,
        split=args.split,
        split_metadata=split_metadata,
        seed=args.seed,
        rows=rows,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        label_column=label_column,
    )
    metrics_path = output_dir / "metrics.csv"
    best = None
    best_epoch = 0
    bad_epochs = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "split", "loss", *(_metric_names(task_type))])
        writer.writeheader()
        for epoch in range(1, args.max_epochs + 1):
            _train_epoch(model, loaders["train"], criterion, optimizer, device)
            epoch_metrics = {}
            for split_name in ["train", "val", "test"]:
                loss, metrics = _evaluate(model, loaders[split_name], criterion, device, task_type)
                row = {"epoch": epoch, "split": split_name, "loss": loss, **metrics}
                writer.writerow(row)
                epoch_metrics[split_name] = row
            handle.flush()
            val_primary = _primary_metric(epoch_metrics["val"], task_type)
            improved = _improved(val_primary, best, task_type)
            if improved:
                best = val_primary
                best_epoch = epoch
                bad_epochs = 0
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, output_dir / "best.pt")
            else:
                bad_epochs += 1
            if epoch >= args.min_epochs and bad_epochs >= args.patience:
                break
    if not (output_dir / "best.pt").exists():
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, output_dir / "best.pt")
        best_epoch = epoch

    final_loss, final_test = _evaluate(model, loaders["test"], criterion, device, task_type)
    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    _, best_test = _evaluate(model, loaders["test"], criterion, device, task_type)
    final = {
        "dataset": args.dataset,
        "split": args.split,
        "model_id": args.model_id,
        "model_family": checkpoint_metadata.get("model_family", "graph_bert" if args.model_id in {"scratch", "graph_bert"} else "hcmp"),
        "feature_mode": checkpoint_metadata.get("feature_mode", feature_spec.feature_mode),
        "seed": args.seed,
        "downstream_seed": args.seed,
        "pretrain_seed": args.pretrain_seed,
        "checkpoint": args.checkpoint,
        "checkpoint_path": args.checkpoint,
        "smiles_column": smiles_column,
        "label_column": label_column,
        "split_backend": split_metadata["split_backend"],
        "splitter": split_metadata["splitter"],
        "frac_train": split_metadata["frac_train"],
        "frac_valid": split_metadata["frac_valid"],
        "frac_test": split_metadata["frac_test"],
        "metric_primary": "ROC-AUC" if task_type == "classification" else "MAE",
        "best_epoch": best_epoch,
        "best_val_metric": best,
        "test_metrics_at_best_val": best_test,
        "final_test_metrics": {"loss": final_loss, **final_test},
    }
    (output_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"Wrote downstream metrics to {output_dir}")


class _GraphLabelDataset:
    def __init__(self, rows, feature_spec) -> None:
        self.rows = rows
        self.feature_spec = feature_spec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        smiles, label = _row_smiles_label(self.rows[int(idx)])
        return smiles_to_graph(smiles, mol_id=idx, feature_spec=self.feature_spec), float(label)


def _build_downstream_model_class():
    import torch
    from torch import nn

    class DownstreamModel(nn.Module):
        def __init__(self, encoder, hidden_dim: int, task_type: str) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(hidden_dim, 1)
            self.task_type = task_type

        def forward(self, batch):
            output = self.encoder(
                batch.node_features,
                batch.edge_index,
                batch.edge_features,
                node_batch=batch.node_batch,
                edge_batch=batch.edge_batch,
            )
            return self.head(output.graph_embedding).squeeze(-1)

    return DownstreamModel


def _collate(batch):
    import torch

    graphs, labels = zip(*batch, strict=False)
    return graph_collate(list(graphs)), torch.tensor(labels, dtype=torch.float32)


def _train_epoch(model, loader, criterion, optimizer, device) -> None:
    model.train()
    for batch, labels in loader:
        batch = move_graph_batch_to_device(batch, device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()


def _evaluate(model, loader, criterion, device, task_type: str):
    import torch

    model.eval()
    preds = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch, y in loader:
            batch = move_graph_batch_to_device(batch, device)
            y = y.to(device)
            logits = model(batch)
            losses.append(float(criterion(logits, y).detach().cpu()))
            preds.extend(logits.detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())
    metrics = _classification_metrics(labels, preds) if task_type == "classification" else _regression_metrics(labels, preds)
    return (sum(losses) / len(losses) if losses else 0.0), metrics


def _classification_metrics(labels, logits) -> dict[str, float]:
    y = np.asarray(labels, dtype=float)
    scores = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
    pred = (scores >= 0.5).astype(float)
    try:
        from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

        return {
            "ROC-AUC": float(roc_auc_score(y, scores)),
            "PR-AUC": float(average_precision_score(y, scores)),
            "accuracy": float(accuracy_score(y, pred)),
        }
    except Exception as exc:
        if not getattr(_classification_metrics, "_warned", False):
            print(f"Warning: sklearn classification metrics unavailable or invalid ({exc}); using internal fallback.")
            _classification_metrics._warned = True
    return {"ROC-AUC": _roc_auc(y, scores), "PR-AUC": _pr_auc(y, scores), "accuracy": float((pred == y).mean())}


def _regression_metrics(labels, preds) -> dict[str, float]:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(preds, dtype=float)
    err = p - y
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(mean_squared_error(y, p, squared=False)),
            "R2": float(r2_score(y, p)),
        }
    except Exception as exc:
        if not getattr(_regression_metrics, "_warned", False):
            print(f"Warning: sklearn regression metrics unavailable ({exc}); using internal fallback.")
            _regression_metrics._warned = True
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err * err)) / denom if denom > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _roc_auc(y, scores) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def _pr_auc(y, scores) -> float:
    order = np.argsort(-scores)
    y_sorted = y[order]
    positives = y.sum()
    if positives <= 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    recall = tp / positives
    return float(np.trapz(precision, recall))


def _split_indices(rows, split: str, seed: int, task_type: str = "regression"):
    frac_train = 0.8
    frac_valid = 0.1
    frac_test = 0.1
    deepchem_result = _deepchem_split_indices(
        rows,
        split=split,
        seed=seed,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
    )
    if deepchem_result is not None:
        train_idx, val_idx, test_idx, metadata = deepchem_result
        return train_idx, val_idx, test_idx, metadata

    print(
        f"Warning: using internal fallback downstream {split} splitter; "
        "DeepChem is unavailable or splitter invocation failed.",
        file=sys.stderr,
    )
    if split == "random":
        if task_type == "classification":
            train_idx, val_idx, test_idx = _stratified_random_indices(rows, seed)
        else:
            indices = list(range(len(rows)))
            random.Random(seed).shuffle(indices)
            n = len(indices)
            train_idx = indices[: int(frac_train * n)]
            val_idx = indices[int(frac_train * n): int((frac_train + frac_valid) * n)]
            test_idx = indices[int((frac_train + frac_valid) * n):]
        splitter_name = "stratified_random" if task_type == "classification" else "random"
    else:
        groups = {}
        for idx, row in enumerate(rows):
            smiles, _label = _row_smiles_label(row)
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else smiles
            groups.setdefault(scaffold, []).append(idx)
        grouped = sorted(groups.values(), key=lambda item: (-len(item), item[0]))
        indices = [idx for group in grouped for idx in group]
        n = len(indices)
        train_idx = indices[: int(frac_train * n)]
        val_idx = indices[int(frac_train * n): int((frac_train + frac_valid) * n)]
        test_idx = indices[int((frac_train + frac_valid) * n):]
        splitter_name = "bemis_murcko_greedy"
    metadata = {
        "split_backend": "internal_fallback",
        "splitter": splitter_name,
        "frac_train": frac_train,
        "frac_valid": frac_valid,
        "frac_test": frac_test,
        "seed": int(seed),
    }
    return train_idx, val_idx, test_idx, metadata


def _deepchem_split_indices(
    rows,
    split: str,
    seed: int,
    frac_train: float,
    frac_valid: float,
    frac_test: float,
):
    try:
        import deepchem as dc

        splitter = dc.splits.RandomSplitter() if split == "random" else dc.splits.ScaffoldSplitter()
        smiles = [_row_smiles_label(row)[0] for row in rows]
        labels = np.asarray([_row_smiles_label(row)[1] for row in rows], dtype=float)
        dataset = dc.data.NumpyDataset(
            X=np.zeros((len(rows), 1), dtype=np.float32),
            y=labels,
            ids=np.asarray(smiles, dtype=object),
        )
        train_idx, val_idx, test_idx = splitter.split(
            dataset,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=seed,
        )
    except Exception as exc:
        if not isinstance(exc, ModuleNotFoundError):
            print(f"Warning: DeepChem {split} splitter failed ({exc}); falling back internally.", file=sys.stderr)
        return None
    metadata = {
        "split_backend": "deepchem",
        "splitter": type(splitter).__name__,
        "frac_train": frac_train,
        "frac_valid": frac_valid,
        "frac_test": frac_test,
        "seed": int(seed),
    }
    return list(map(int, train_idx)), list(map(int, val_idx)), list(map(int, test_idx)), metadata


def _stratified_random_indices(rows, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[float, list[int]] = {}
    for idx, row in enumerate(rows):
        _smiles, label = _row_smiles_label(row)
        by_label.setdefault(float(label), []).append(idx)
    split_parts = {"train": [], "val": [], "test": []}
    for indices in by_label.values():
        rng.shuffle(indices)
        n = len(indices)
        split_parts["train"].extend(indices[: int(0.8 * n)])
        split_parts["val"].extend(indices[int(0.8 * n): int(0.9 * n)])
        split_parts["test"].extend(indices[int(0.9 * n):])
    for values in split_parts.values():
        rng.shuffle(values)
    return split_parts["train"], split_parts["val"], split_parts["test"]


def _checkpoint_context(checkpoint_path: str, torch):
    if checkpoint_path == "none":
        feature_spec = default_feature_spec({"feature_mode": "graph_bert"})
        metadata = {"model_family": "graph_bert", "feature_mode": "graph_bert"}
        return feature_spec, _default_model_config(), None, {"downstream": {}}, metadata
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    feature_spec = default_feature_spec(checkpoint.get("feature_spec", config.get("features", {})))
    model_config = _model_config_from_checkpoint(config)
    metadata = {
        "model_family": checkpoint.get("model_family", config.get("model_family", "hcmp")),
        "feature_mode": checkpoint.get("feature_mode", feature_spec.feature_mode),
    }
    return feature_spec, model_config, checkpoint.get("model_state_dict"), config, metadata


def _model_config_from_checkpoint(config: dict) -> dict:
    encoder = config.get("model", {}).get("encoder", {})
    pooling = config.get("model", {}).get("pooling", {})
    return {
        "hidden_dim": int(encoder.get("hidden_dim", 256)),
        "num_layers": int(encoder.get("num_layers", 6)),
        "encoder_type": str(encoder.get("encoder_type", "graph_transformer")),
        "num_heads": int(encoder.get("num_heads", 8)),
        "dropout": float(encoder.get("dropout", 0.1)),
        "attention_impl": str(encoder.get("attention_impl", "vectorized")),
        "pooling_impl": str(pooling.get("pooling_impl", "vectorized")),
    }


def _default_model_config() -> dict:
    return {"hidden_dim": 256, "num_layers": 6, "encoder_type": "graph_transformer", "num_heads": 8, "dropout": 0.1, "attention_impl": "vectorized", "pooling_impl": "vectorized"}


def _load_downstream_frame(root: str, dataset: str, data_path: str | None = None) -> pd.DataFrame:
    if data_path is not None:
        return pd.read_csv(data_path)
    aliases = {"esol": ["esol", "solv"], "freesolv": ["freesolv"], "lipophilicity": ["lipophilicity"], "bbbp": ["bbbp"], "hiv": ["hiv"]}
    names = aliases.get(dataset.lower(), [dataset])
    candidates = []
    for name in names:
        candidates.extend([Path(root) / name / f"{name}.csv", Path(root) / name / "data.csv", Path(root) / f"{name}.csv"])
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find downstream CSV for {dataset}: {candidates}")


def _build_rows(frame: pd.DataFrame, smiles_column: str, label_column: str, task_type: str):
    rows = []
    for _, row in frame[[smiles_column, label_column]].dropna().iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_column]))
        if mol is not None:
            label = _normalize_classification_label(row[label_column]) if task_type == "classification" else float(row[label_column])
            rows.append((Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True), label))
    return rows


def _row_smiles_label(row) -> tuple[str, float]:
    if isinstance(row, dict):
        return str(row["smiles"]), float(row["label"])
    return str(row[0]), float(row[1])


def _save_split_artifacts(
    output_dir: Path,
    dataset: str,
    split: str,
    split_metadata: dict,
    seed: int,
    rows,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    label_column: str,
) -> None:
    payload = {
        "dataset": dataset,
        "split": split,
        **split_metadata,
        "seed": int(seed),
        "train_indices": [int(idx) for idx in train_idx],
        "val_indices": [int(idx) for idx in val_idx],
        "test_indices": [int(idx) for idx in test_idx],
    }
    (output_dir / "split_indices.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_split_smiles(output_dir / "train_smiles.csv", rows, train_idx, label_column)
    _write_split_smiles(output_dir / "val_smiles.csv", rows, val_idx, label_column)
    _write_split_smiles(output_dir / "test_smiles.csv", rows, test_idx, label_column)


def _write_split_smiles(path: Path, rows, indices: list[int], label_column: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "smiles", label_column])
        writer.writeheader()
        for idx in indices:
            smiles, label = _row_smiles_label(rows[int(idx)])
            writer.writerow({"index": int(idx), "smiles": smiles, label_column: label})


def _detect_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not detect column from {candidates}; columns={list(frame.columns)}")


def _detect_label_column(frame, smiles_column: str, dataset: str) -> str:
    for column in _dataset_label_candidates(dataset):
        if column in frame.columns and column != smiles_column:
            return column
    excluded = {smiles_column, "mol_id", "name", "split", "scaffold"}
    numeric = [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]
    if not numeric:
        raise ValueError("Could not detect numeric label column.")
    return numeric[-1]


def _dataset_smiles_candidates(dataset: str) -> list[str]:
    return ["smiles", "SMILES", "mol", "canonical_smiles"]


def _dataset_label_candidates(dataset: str) -> list[str]:
    return {
        "bbbp": ["p_np", "label", "y"],
        "hiv": ["HIV_active", "label", "y"],
        "esol": ["measured log solubility in mols per litre", "y", "label"],
        "freesolv": ["expt", "y", "label"],
        "lipophilicity": ["exp", "y", "label"],
    }.get(dataset.lower(), ["y", "label"])


def _normalize_classification_label(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    numeric = float(value)
    if numeric in {0.0, 1.0}:
        return numeric
    if numeric == -1.0:
        return 0.0
    raise ValueError(f"Unexpected classification label {value!r}; expected 0/1, -1/1, or boolean.")


def _metric_names(task_type: str) -> list[str]:
    return ["ROC-AUC", "PR-AUC", "accuracy"] if task_type == "classification" else ["MAE", "RMSE", "R2"]


def _primary_metric(row: dict, task_type: str) -> float:
    return float(row["ROC-AUC"] if task_type == "classification" else row["MAE"])


def _improved(value: float, best: float | None, task_type: str) -> bool:
    if math.isnan(float(value)):
        return False
    if best is None:
        return True
    if math.isnan(float(best)):
        return True
    return value > best if task_type == "classification" else value < best


def _seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _ensure_training_imports() -> None:
    global graph_collate
    global default_feature_spec
    global smiles_to_graph
    global HCMPModel
    global move_graph_batch_to_device
    global _DownstreamModel
    if _DownstreamModel is not None:
        return
    from hcmp.data.datasets import graph_collate as imported_graph_collate
    from hcmp.data.graph_builder import default_feature_spec as imported_default_feature_spec
    from hcmp.data.graph_builder import smiles_to_graph as imported_smiles_to_graph
    from hcmp.models.hcmp_model import HCMPModel as ImportedHCMPModel
    from hcmp.training.trainer import move_graph_batch_to_device as imported_move_graph_batch_to_device

    graph_collate = imported_graph_collate
    default_feature_spec = imported_default_feature_spec
    smiles_to_graph = imported_smiles_to_graph
    HCMPModel = ImportedHCMPModel
    move_graph_batch_to_device = imported_move_graph_batch_to_device
    _DownstreamModel = _build_downstream_model_class()


if __name__ == "__main__":
    main()
