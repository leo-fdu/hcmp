#!/usr/bin/env python
"""Run the HCMP multi-task training loop."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hcmp.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hcmp_pretrain.yaml")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--descriptor-thresholds", default=None)
    parser.add_argument("--scaffold-distance", default=None)
    parser.add_argument("--scaffold-distance-cache", default=None)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every-epochs", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--attention-impl", choices=["loop", "vectorized"], default=None)
    parser.add_argument("--pooling-impl", choices=["loop", "vectorized"], default=None)
    parser.add_argument(
        "--allow-cpu-full-run",
        action="store_true",
        help="Explicitly allow CPU training when the dataset looks like a full-corpus run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if str(config.get("model_id", config.get("run_name", ""))).lower() == "scratch":
        raise SystemExit(
            "scratch is a downstream-only randomly initialized baseline and should not be pretrained."
        )

    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise SystemExit("Torch is required to run training. Install torch and retry.") from exc

    from hcmp.data.datasets import CachedGraphDataset, GraphDataset, LazyGraphDataset, MultiTaskDataLoader, ShardAwareBatchSampler, graph_collate
    from hcmp.data.descriptors import DESCRIPTOR_NAMES
    from hcmp.data.graph_builder import default_feature_spec
    from hcmp.data.molecule_table import load_hcmp_molecule_table
    from hcmp.data.scaffold_distance import build_scaffold_distance_backend
    from hcmp.models.hcmp_model import HCMPModel
    from hcmp.training.curriculum import build_curriculum, curriculum_to_metadata
    from hcmp.training.loss_balancing import build_loss_balancer
    from hcmp.training.trainer import HCMPTrainer

    config = deepcopy(config)
    data_config = config["data"]
    train_config = config.setdefault("training", {})
    output_config = config.setdefault("output", {})
    model_config = config.setdefault("model", {})
    encoder_config = model_config.setdefault("encoder", {})
    pooling_config = model_config.setdefault("pooling", {})
    input_csv = args.input_csv or data_config["input_csv"]
    smiles_column = args.smiles_column or data_config.get("smiles_column", "smiles")
    max_molecules = args.max_molecules if args.max_molecules is not None else train_config.get("max_molecules")
    num_epochs = args.num_epochs if args.num_epochs is not None else int(train_config.get("num_epochs", 1))
    device_name = args.device or train_config.get("device", "cpu")
    if args.attention_impl is not None:
        encoder_config["attention_impl"] = args.attention_impl
    if args.pooling_impl is not None:
        pooling_config["pooling_impl"] = args.pooling_impl
    if args.save_every_epochs is not None:
        output_config["save_every_epochs"] = args.save_every_epochs
    if args.save_every_steps is not None:
        output_config["save_every_steps"] = args.save_every_steps
    if args.output_dir is not None:
        output_config["run_dir"] = args.output_dir
    train_config["num_epochs"] = int(num_epochs)
    train_config["max_molecules"] = max_molecules
    train_config["device"] = str(device_name)

    device = _resolve_device(torch, device_name)
    print(f"Using device: {device}")

    objective_flags = config.get("pretrain_objectives", {})
    if objective_flags:
        for key, section in [
            ("bert", "bert"),
            ("cut_seg", "cut_seg"),
            ("prop_rank", "prop_rank"),
            ("scaf_triplet", "scaf_triplet"),
        ]:
            if key in objective_flags:
                config[section]["enabled"] = bool(objective_flags[key])
    prop_rank_enabled = bool(config["prop_rank"]["enabled"])
    descriptor_values_path = data_config.get("descriptor_values_path")
    recompute_descriptors = bool(data_config.get("recompute_descriptors_in_dataset", False))
    if prop_rank_enabled and descriptor_values_path is None and not recompute_descriptors:
        raise ValueError(
            "prop_rank is enabled, but data.descriptor_values_path is missing. "
            "Run scripts/03_compute_descriptors.py and set data.descriptor_values_path, "
            "or set data.recompute_descriptors_in_dataset=true for an explicit debug fallback."
        )
    if prop_rank_enabled and descriptor_values_path is not None and not Path(descriptor_values_path).exists():
        raise ValueError(f"Descriptor values file does not exist: {descriptor_values_path}")

    descriptor_thresholds = None
    threshold_descriptor_names = None
    if prop_rank_enabled:
        threshold_path = args.descriptor_thresholds or data_config.get("descriptor_thresholds")
        descriptor_thresholds, threshold_descriptor_names = _load_descriptor_thresholds(
            threshold_path,
            torch,
            DESCRIPTOR_NAMES,
            required=True,
        )
    feature_spec = default_feature_spec(config.get("features", {}))
    dataset_config = config.setdefault("dataset", {})
    graph_mode = str(dataset_config.get("graph_mode", data_config.get("graph_mode", "cache")))
    graph_cache_dir = dataset_config.get("graph_cache_dir", data_config.get("graph_cache_dir"))
    feature_mode = str(dataset_config.get("feature_mode", config.get("features", {}).get("feature_mode", feature_spec.feature_mode)))
    config.setdefault("features", {})["feature_mode"] = feature_mode
    dataset_config["feature_mode"] = feature_mode
    feature_spec = default_feature_spec(config.get("features", {}))
    model_family = str(config.get("model_family", "graph_bert" if config.get("baseline") == "traditional_graph_bert" else "hcmp"))
    config["model_family"] = model_family
    if model_family == "graph_bert" and feature_mode != "graph_bert":
        raise ValueError("model_family='graph_bert' requires feature_mode='graph_bert'.")
    molecule_table = None
    if graph_mode == "cache":
        if graph_cache_dir is None:
            raise ValueError("dataset.graph_mode='cache' requires dataset.graph_cache_dir.")
        graph_dataset = CachedGraphDataset(
            graph_cache_dir,
            max_loaded_shards=int(dataset_config.get("max_loaded_shards", 4)),
        )
        cached_feature_spec = graph_dataset.manifest.get("feature_spec")
        manifest_feature_mode = graph_dataset.manifest.get("feature_mode", cached_feature_spec.get("feature_mode") if isinstance(cached_feature_spec, dict) else None)
        if manifest_feature_mode != feature_mode:
            raise ValueError(
                "Graph cache feature_mode mismatch: "
                f"requested feature_mode={feature_mode!r} from cache {graph_cache_dir}, "
                f"but manifest contains feature_mode={manifest_feature_mode!r}."
            )
        if cached_feature_spec is not None:
            manifest_spec = default_feature_spec(cached_feature_spec)
            if manifest_spec.node_feature_dim != feature_spec.node_feature_dim or manifest_spec.edge_feature_dim != feature_spec.edge_feature_dim:
                raise ValueError(
                    "Graph cache feature dimensions do not match config.features. "
                    "Regenerate the cache with matching feature options."
                )
    elif graph_mode == "lazy":
        molecule_table = load_hcmp_molecule_table(
            input_csv,
            smiles_column=smiles_column,
            sort_by_canonical_smiles=True,
            max_molecules=int(max_molecules) if max_molecules is not None else None,
            strict=False,
        )
        graph_dataset = LazyGraphDataset(
            molecule_table,
            feature_spec,
            descriptor_values=descriptor_values_path if prop_rank_enabled and descriptor_values_path else None,
            descriptor_names=threshold_descriptor_names or DESCRIPTOR_NAMES,
            recompute_descriptors=recompute_descriptors,
        )
    elif graph_mode == "eager":
        molecule_table = load_hcmp_molecule_table(
            input_csv,
            smiles_column=smiles_column,
            sort_by_canonical_smiles=True,
            max_molecules=int(max_molecules) if max_molecules is not None else None,
            strict=False,
        )
        graph_dataset = GraphDataset(
            molecule_table,
            feature_spec,
            descriptor_values=descriptor_values_path if prop_rank_enabled and descriptor_values_path else None,
            descriptor_names=threshold_descriptor_names or DESCRIPTOR_NAMES,
            recompute_descriptors=recompute_descriptors,
        )
    else:
        raise ValueError("dataset.graph_mode must be one of: cache, lazy, eager.")
    batch_size = int(train_config.get("graph_batch_size", 16))
    batch_sampler = None
    if graph_mode == "cache" and bool(dataset_config.get("shard_aware_sampling", True)):
        batch_sampler = ShardAwareBatchSampler(
            graph_dataset,
            batch_size=batch_size,
            seed=int(config.get("seed", train_config.get("seed", 0))),
            drop_last=bool(train_config.get("drop_last", False)),
        )
        graph_loader = DataLoader(graph_dataset, batch_sampler=batch_sampler, collate_fn=graph_collate)
    else:
        graph_loader = DataLoader(
            graph_dataset,
            batch_size=batch_size,
            shuffle=bool(train_config.get("shuffle", True)),
            drop_last=bool(train_config.get("drop_last", False)),
            collate_fn=graph_collate,
        )

    multitask_loader = MultiTaskDataLoader(graph_loader=graph_loader)
    scaf_config = config["scaf_triplet"]
    scaffold_backend_name = str(scaf_config.get("distance_backend", "full_matrix"))
    scaffold_distance_matrix = None
    scaffold_distance_metadata = None
    scaffold_cache_path = args.scaffold_distance_cache or scaf_config.get("cache_path")
    if scaffold_backend_name == "on_the_fly":
        scaffold_cache_path = None
    if scaffold_backend_name == "full_matrix":
        scaffold_path = args.scaffold_distance or data_config.get("scaffold_distance")
        scaffold_distance_matrix, scaffold_distance_metadata = _load_scaffold_distance(scaffold_path)
    scaffold_distance_backend = None
    if bool(scaf_config.get("enabled", True)):
        scaffold_distance_default = 0 if scaffold_backend_name == "on_the_fly" else 1_000_000
        scaffold_object_default = 0 if scaffold_backend_name == "on_the_fly" else 200_000
        scaffold_distance_backend = build_scaffold_distance_backend(
            scaffold_backend_name,
            molecule_table=molecule_table,
            matrix=scaffold_distance_matrix,
            metadata=scaffold_distance_metadata,
            cache_path=scaffold_cache_path,
            max_rounds=int(scaf_config.get("max_mcs_rounds", 3)),
            graph_cache_dir=graph_cache_dir if graph_mode == "cache" else None,
            sqlite_wal=bool(scaf_config.get("sqlite_wal", True)),
            commit_every_misses=int(scaf_config.get("commit_every_misses", 1000)),
            max_in_memory_distances=_int_config_default(
                scaf_config.get("max_in_memory_distances"),
                scaffold_distance_default,
            ),
            max_in_memory_scaffolds=_int_config_default(
                scaf_config.get("max_in_memory_scaffolds"),
                scaffold_object_default,
            ),
        )
    model = HCMPModel(
        feature_spec=feature_spec,
        hidden_dim=int(encoder_config["hidden_dim"]),
        num_layers=int(encoder_config["num_layers"]),
        encoder_type=str(
            encoder_config.get("encoder_type", encoder_config.get("type", "graph_transformer"))
        ),
        num_heads=int(encoder_config.get("num_heads", 4)),
        dropout=float(encoder_config["dropout"]),
        attention_impl=str(encoder_config.get("attention_impl", "loop")),
        pooling_impl=str(pooling_config.get("pooling_impl", "loop")),
        enabled_heads={
            "bert": bool(config["bert"]["enabled"]),
            "cut_seg": bool(config["cut_seg"]["enabled"]),
            "prop_rank": prop_rank_enabled,
            "scaf_triplet": bool(config["scaf_triplet"]["enabled"]),
        },
        model_family=model_family,
    )
    optimizer_config = config.setdefault("optimizer", {})
    optimizer_name = str(optimizer_config.get("name", train_config.get("optimizer", "AdamW")))
    if optimizer_name.lower() != "adamw":
        raise ValueError("Pretraining now expects optimizer.name='AdamW'.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config.get("lr", train_config.get("learning_rate", 1.0e-4))),
        weight_decay=float(optimizer_config.get("weight_decay", 1.0e-2)),
        betas=tuple(float(value) for value in optimizer_config.get("betas", [0.9, 0.999])),
        eps=float(optimizer_config.get("eps", 1.0e-8)),
    )
    max_total_steps = train_config.get("max_total_steps", config.get("curriculum", {}).get("max_total_steps"))
    max_total_steps = int(max_total_steps) if max_total_steps is not None else None
    scheduler = _build_scheduler(torch, optimizer, config.get("scheduler", {}), max_total_steps)
    trainer = HCMPTrainer(
        model=model,
        optimizer=optimizer,
        loss_balancer=build_loss_balancer(config["loss_balancing"]),
        feature_spec=feature_spec,
        cut_pos_weight=config["cut_seg"]["pos_weight"],
        atom_mask_ratio=float(config["bert"]["atom_mask_ratio"]),
        bond_mask_ratio=float(config["bert"]["bond_mask_ratio"]),
        descriptor_thresholds=descriptor_thresholds,
        scaffold_distance_backend=scaffold_distance_backend,
        partners_per_anchor=int(config["prop_rank"].get("partners_per_anchor", 10)),
        min_distance_gap=float(scaf_config.get("min_distance_gap", 0.15)),
        triplet_margin=float(scaf_config.get("margin", 0.15)),
        candidate_pairs_per_anchor=int(scaf_config.get("candidate_pairs_per_anchor", 10)),
        device=device,
        scheduler=scheduler,
        use_amp=bool(train_config.get("use_amp", True)),
        gradient_clip_norm=train_config.get("gradient_clip_norm", None),
    )

    enabled_losses = {
        "bert": bool(config["bert"]["enabled"]),
        "cut_seg": bool(config["cut_seg"]["enabled"]),
        "prop_rank": prop_rank_enabled,
        "scaf_triplet": bool(scaf_config.get("enabled", True)),
    }
    curriculum = build_curriculum(config, enabled_losses)
    config["n_molecules"] = len(graph_dataset)
    _guard_cpu_full_run(
        device=device,
        n_molecules=len(graph_dataset),
        config=config,
        max_molecules=max_molecules,
        allow_cpu_full_run=args.allow_cpu_full_run,
    )
    run_dir = _prepare_run_dir(output_config)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _save_config_yaml(config, run_dir / "config.yaml")
    _save_metadata(
        run_dir / "metadata.json",
        device=str(device),
        model=model,
        descriptor_names=threshold_descriptor_names or DESCRIPTOR_NAMES,
        feature_spec=feature_spec,
        run_dir=run_dir,
        config=config,
        curriculum=curriculum,
        graph_cache_manifest=(
            str(Path(graph_cache_dir) / "manifest.json") if graph_cache_dir else None
        ),
        n_molecules=len(graph_dataset),
    )
    log_handle = (run_dir / "train_log.csv").open("a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_handle, fieldnames=_TRAIN_LOG_COLUMNS)
    if log_handle.tell() == 0:
        log_writer.writeheader()
        log_handle.flush()

    start_epoch = 0
    best_metric = None
    global_step = 0
    resume_step_in_epoch = 0
    resume_skip_steps = 0
    resumed_checkpoint_epoch = 0
    resumed_at_max_total_steps = False
    if args.resume_from is not None:
        checkpoint = _load_checkpoint(torch, args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("grad_scaler_state_dict") is not None:
            trainer.scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        _restore_curriculum_state(curriculum, checkpoint.get("curriculum_state"))
        _restore_loss_balancer_state(trainer.loss_balancer, checkpoint.get("loss_balancer_state"))
        checkpoint_rng_state = checkpoint.get("rng_state")
        if checkpoint_rng_state is None and (
            checkpoint.get("python_random_state") is not None or checkpoint.get("numpy_rng_state") is not None
        ):
            checkpoint_rng_state = {
                "python": checkpoint.get("python_random_state"),
                "numpy": checkpoint.get("numpy_rng_state"),
                "torch": checkpoint.get("torch_rng_state"),
                "torch_cuda": checkpoint.get("torch_cuda_rng_state"),
            }
        _restore_rng_state(torch, checkpoint_rng_state)
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        resumed_checkpoint_epoch = checkpoint_epoch
        global_step = int(checkpoint.get("global_step", checkpoint.get("final_global_step", 0)))
        resume_step_in_epoch = int(checkpoint.get("step_in_epoch", 0) or 0)
        if (
            checkpoint.get("training_status") == "interrupted"
            or checkpoint.get("checkpoint_kind") == "step"
        ) and resume_step_in_epoch > 0:
            start_epoch = max(0, checkpoint_epoch - 1)
            resume_skip_steps = resume_step_in_epoch
        else:
            start_epoch = checkpoint_epoch
        best_metric = checkpoint.get("best_metric")
        resumed_at_max_total_steps = max_total_steps is not None and global_step >= max_total_steps
        print(
            f"Resumed from {args.resume_from} at epoch={start_epoch} "
            f"global_step={global_step} step_in_epoch={resume_step_in_epoch}"
        )

    print(
        "HCMP training setup: "
        f"graphs={len(graph_dataset)}, "
        f"descriptor_thresholds={'yes' if descriptor_thresholds is not None else 'no'}, "
        f"scaffold_backend={scaffold_backend_name}, "
        f"run_dir={run_dir}"
    )
    training_completed = False
    last_completed_epoch = resumed_checkpoint_epoch if resumed_at_max_total_steps else start_epoch
    final_status = "failed"
    stopping_reason = None
    interrupted_exception = None
    curriculum_history: list[dict[str, object]] = []
    unit = str(config.get("curriculum", {}).get("unit", "epoch"))
    eval_every_steps = int(config.get("curriculum", {}).get("eval_every_steps", 1000))
    max_phase_steps = config.get("curriculum", {}).get("max_phase_steps")
    max_phase_steps = int(max_phase_steps) if max_phase_steps is not None else None
    save_every_steps = int(output_config.get("save_every_steps", 0) or 0)
    stop_after_final_phase_plateau = bool(config.get("curriculum", {}).get("stop_after_final_phase_plateau", True))
    current_step_in_epoch = resume_step_in_epoch
    if resumed_at_max_total_steps:
        stopping_reason = "max_total_steps"
        final_status = "max_total_steps"
        training_completed = True
    try:
        epoch_range = range(start_epoch, start_epoch) if resumed_at_max_total_steps else range(start_epoch, num_epochs)
        for epoch in epoch_range:
            epoch_number = epoch + 1
            last_completed_epoch = epoch_number
            if batch_sampler is not None:
                batch_sampler.set_epoch(epoch_number)
            print(
                f"epoch={epoch_number} starting global_step={global_step} "
                f"phase={curriculum.current_phase.name}"
            )
            interval_totals: dict[str, float] = {}
            interval_count = 0
            for step_in_epoch, batch in enumerate(multitask_loader.iterate_graph(), start=1):
                if resume_skip_steps > 0 and step_in_epoch <= resume_skip_steps:
                    continue
                if resume_skip_steps > 0:
                    resume_skip_steps = 0
                current_step_in_epoch = step_in_epoch
                phase = curriculum.current_phase
                started = time.perf_counter()
                losses = trainer.train_graph_batch(batch, active_losses=phase.active_losses)
                seconds_per_step = time.perf_counter() - started
                global_step += 1
                interval_count += 1
                for key, value in losses.items():
                    interval_totals[key] = interval_totals.get(key, 0.0) + float(value)
                monitor_metric = _monitor_metric(losses, phase.monitor)
                _write_train_log_row(
                    log_writer,
                    log_handle,
                    global_step=global_step,
                    epoch=epoch_number,
                    step_in_epoch=step_in_epoch,
                    phase=phase,
                    losses=losses,
                    monitor=monitor_metric,
                    epochs_in_phase=curriculum.epochs_in_phase,
                    learning_rate=_learning_rate(optimizer),
                    seconds_per_step=seconds_per_step,
                    device=str(device),
                )
                should_eval = (
                    (unit == "step" and global_step % eval_every_steps == 0)
                    or unit == "epoch" and step_in_epoch == len(graph_loader)
                )
                phase_steps = global_step - int(getattr(curriculum, "phase_start_step", 0))
                curriculum_event_happened = False
                if should_eval and interval_count:
                    mean_losses = {
                        key: value / interval_count for key, value in interval_totals.items()
                    }
                    was_final_phase = bool(getattr(curriculum, "is_final_phase", True))
                    transition = curriculum.observe_point(mean_losses, global_step=global_step)
                    if transition is not None:
                        curriculum_event_happened = True
                        if was_final_phase and stop_after_final_phase_plateau:
                            stopping_reason = "convergence_final_phase"
                            final_status = "completed"
                            curriculum_history.append(
                                {
                                    "global_step": global_step,
                                    "epoch": epoch_number,
                                    "phase": transition.old_phase,
                                    "reason": stopping_reason,
                                    "relative_improvement": transition.relative_improvement,
                                }
                            )
                            print(
                                "training_stop "
                                f"phase={transition.old_phase} global_step={global_step} "
                                f"reason={stopping_reason}"
                            )
                        else:
                            curriculum_history.append(
                                {
                                    "global_step": global_step,
                                    "epoch": epoch_number,
                                    "old_phase": transition.old_phase,
                                    "new_phase": transition.new_phase,
                                    "reason": "convergence",
                                    "relative_improvement": transition.relative_improvement,
                                }
                            )
                            print(
                                "curriculum_transition "
                                f"old_phase={transition.old_phase} new_phase={transition.new_phase} "
                                f"global_step={global_step} reason=convergence"
                            )
                    interval_totals = {}
                    interval_count = 0
                if not curriculum_event_happened and max_phase_steps is not None and phase_steps >= max_phase_steps:
                    old_phase = curriculum.current_phase.name
                    if getattr(curriculum, "is_final_phase", True):
                        stopping_reason = "max_phase_steps_final_phase"
                        final_status = "completed"
                        curriculum_event_happened = True
                        curriculum_history.append(
                            {
                                "global_step": global_step,
                                "epoch": epoch_number,
                                "phase": old_phase,
                                "reason": stopping_reason,
                            }
                        )
                        print(
                            "training_stop "
                            f"phase={old_phase} global_step={global_step} reason={stopping_reason}"
                        )
                    else:
                        curriculum.phase_index += 1
                        curriculum.phase_history = []
                        curriculum.phase_start_step = global_step
                        curriculum_event_happened = True
                        curriculum_history.append(
                            {
                                "global_step": global_step,
                                "epoch": epoch_number,
                                "old_phase": old_phase,
                                "new_phase": curriculum.current_phase.name,
                                "reason": "max_phase_steps",
                            }
                        )
                        print(
                            "curriculum_transition "
                            f"old_phase={old_phase} new_phase={curriculum.current_phase.name} "
                            f"global_step={global_step} reason=max_phase_steps"
                        )
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    _save_checkpoint(
                        checkpoints_dir / "last.pt",
                        torch=torch,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch_number,
                        config=config,
                        curriculum=curriculum,
                        loss_balancer=trainer.loss_balancer,
                        best_metric=best_metric,
                        descriptor_names=threshold_descriptor_names or DESCRIPTOR_NAMES,
                        feature_spec=feature_spec,
                        global_step=global_step,
                        step_in_epoch=current_step_in_epoch,
                        scheduler=scheduler,
                        grad_scaler=trainer.scaler,
                        stopping_reason=stopping_reason,
                        checkpoint_kind="step",
                    )
                    print(
                        "step_checkpoint_saved "
                        f"global_step={global_step} step_in_epoch={current_step_in_epoch}",
                        flush=True,
                    )
                if max_total_steps is not None and global_step >= max_total_steps:
                    stopping_reason = "max_total_steps"
                    final_status = "max_total_steps"
                    break
                if stopping_reason in {"convergence_final_phase", "max_phase_steps_final_phase"}:
                    break
            last_completed_epoch = epoch_number
            if _as_bool(output_config.get("save_last", True)):
                _save_checkpoint(
                    checkpoints_dir / "last.pt",
                    torch=torch,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch_number,
                    config=config,
                    curriculum=curriculum,
                    loss_balancer=trainer.loss_balancer,
                    best_metric=best_metric,
                    descriptor_names=threshold_descriptor_names or DESCRIPTOR_NAMES,
                    feature_spec=feature_spec,
                    global_step=global_step,
                    step_in_epoch=current_step_in_epoch,
                    scheduler=scheduler,
                    grad_scaler=trainer.scaler,
                    stopping_reason=stopping_reason,
                    checkpoint_kind="epoch",
                )
            save_every = int(output_config.get("save_every_epochs", 0) or 0)
            if save_every > 0 and epoch_number % save_every == 0:
                _save_checkpoint(
                    checkpoints_dir / f"hcmp_epoch_{epoch_number:04d}.pt",
                    torch=torch,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch_number,
                    config=config,
                    curriculum=curriculum,
                    loss_balancer=trainer.loss_balancer,
                    best_metric=best_metric,
                    descriptor_names=threshold_descriptor_names or DESCRIPTOR_NAMES,
                    feature_spec=feature_spec,
                    global_step=global_step,
                    step_in_epoch=current_step_in_epoch,
                    scheduler=scheduler,
                    grad_scaler=trainer.scaler,
                    stopping_reason=stopping_reason,
                    checkpoint_kind="epoch",
                )
            if stopping_reason in {"max_total_steps", "convergence_final_phase", "max_phase_steps_final_phase"}:
                break
        training_completed = True
        if final_status == "failed":
            final_status = "completed"
            stopping_reason = "completed"
    except KeyboardInterrupt as exc:
        interrupted_exception = exc
        final_status = "interrupted"
        stopping_reason = "manual_interrupt"
        print("Training interrupted; saving interrupted checkpoint.")
    except BaseException as exc:
        interrupted_exception = exc
        final_status = "failed"
        raise
    finally:
        if scaffold_distance_backend is not None and hasattr(scaffold_distance_backend, "flush"):
            scaffold_distance_backend.flush()
        if scaffold_distance_backend is not None and hasattr(scaffold_distance_backend, "close"):
            scaffold_distance_backend.close()
        (run_dir / "curriculum_history.json").write_text(
            json.dumps(curriculum_history, indent=2),
            encoding="utf-8",
        )
        _update_run_metadata_status(
            run_dir / "run_metadata.json",
            stopping_reason,
            final_status,
            global_step,
            last_completed_epoch,
            getattr(curriculum.current_phase, "name", None),
        )
        _update_run_metadata_status(
            run_dir / "metadata.json",
            stopping_reason,
            final_status,
            global_step,
            last_completed_epoch,
            getattr(curriculum.current_phase, "name", None),
        )
        try:
            if training_completed:
                if _as_bool(output_config.get("save_final", True)):
                    _save_checkpoint(
                        checkpoints_dir / "final.pt",
                        torch=torch,
                        model=model,
                        optimizer=optimizer,
                        epoch=last_completed_epoch,
                        config=config,
                        curriculum=curriculum,
                        loss_balancer=trainer.loss_balancer,
                        best_metric=best_metric,
                        descriptor_names=threshold_descriptor_names or DESCRIPTOR_NAMES,
                        feature_spec=feature_spec,
                        training_status=final_status,
                        global_step=global_step,
                        step_in_epoch=current_step_in_epoch,
                        scheduler=scheduler,
                        grad_scaler=trainer.scaler,
                        stopping_reason=stopping_reason,
                        checkpoint_kind="final",
                    )
            else:
                _save_checkpoint(
                    checkpoints_dir / "interrupted.pt",
                    torch=torch,
                    model=model,
                    optimizer=optimizer,
                    epoch=last_completed_epoch,
                    config=config,
                    curriculum=curriculum,
                    loss_balancer=trainer.loss_balancer,
                    best_metric=best_metric,
                    descriptor_names=threshold_descriptor_names or DESCRIPTOR_NAMES,
                    feature_spec=feature_spec,
                    training_status=final_status,
                    exception=interrupted_exception,
                    global_step=global_step,
                    step_in_epoch=current_step_in_epoch,
                    scheduler=scheduler,
                    grad_scaler=trainer.scaler,
                    stopping_reason=stopping_reason,
                    checkpoint_kind="interrupted",
                )
        except Exception as save_exc:
            if training_completed:
                raise
            print(f"Warning: failed to save interrupted checkpoint: {save_exc}")
        finally:
            log_handle.close()


def _load_descriptor_thresholds(path_like, torch, descriptor_names, required: bool = False):
    if path_like is None:
        if required:
            raise ValueError("prop_rank is enabled, but data.descriptor_thresholds is missing.")
        return None, None
    path = Path(path_like)
    if not path.exists():
        if required:
            raise ValueError(f"Descriptor threshold file does not exist: {path}")
        return None, None
    frame = pd.read_csv(path)
    descriptor_column = "descriptor_name" if "descriptor_name" in frame.columns else "descriptor"
    threshold_column = "threshold_value" if "threshold_value" in frame.columns else "threshold"
    threshold_names = list(frame[descriptor_column])
    if threshold_names != list(descriptor_names):
        raise ValueError(
            "Descriptor threshold rows do not match the expected descriptor order. "
            f"expected={list(descriptor_names)}, found={threshold_names}"
        )
    threshold_by_name = dict(zip(frame[descriptor_column], frame[threshold_column], strict=False))
    missing = [name for name in descriptor_names if name not in threshold_by_name]
    if missing:
        raise ValueError(f"Descriptor threshold file is missing descriptors: {missing}")
    values = [float(threshold_by_name[name]) for name in descriptor_names]
    return torch.tensor(values, dtype=torch.float32), threshold_names


def _load_scaffold_distance(path_like):
    if path_like is None:
        return None, None
    path = Path(path_like)
    if not path.exists():
        return None, None
    matrix = np.load(path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Scaffold distance matrix must be square.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Scaffold distance matrix must be symmetric.")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("Scaffold distance matrix diagonal must be zero.")
    metadata_path = Path(str(path) + ".metadata.json")
    metadata = None
    if metadata_path.exists():
        from hcmp.data.scaffold_distance.io_utils import load_json

        metadata = load_json(metadata_path)
    return matrix.astype(np.float32, copy=False), metadata


_TRAIN_LOG_COLUMNS = [
    "global_step",
    "epoch",
    "step_in_epoch",
    "phase",
    "active_losses",
    "monitor",
    "epochs_in_phase",
    "bert_loss",
    "cut_seg_loss",
    "prop_rank_loss",
    "scaf_triplet_loss",
    "weighted_total_loss",
    "num_valid_descriptor_pairs",
    "num_valid_triplets",
    "scaffold_cache_hits",
    "scaffold_cache_misses",
    "scaffold_distance_failures",
    "lr",
    "seconds_per_step",
    "gpu_memory_allocated_mb",
    "device",
]

_LOSS_TO_REPORTED_KEY = {
    "bert": "bert_loss",
    "cut_seg": "cut_loss",
    "prop_rank": "prop_rank_loss",
    "scaf_triplet": "scaf_triplet_loss",
}


def _resolve_device(torch, device_name: str):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _guard_cpu_full_run(
    device,
    n_molecules: int,
    config: dict,
    max_molecules,
    allow_cpu_full_run: bool,
) -> None:
    if getattr(device, "type", str(device).split(":", 1)[0]) != "cpu":
        return
    threshold = int(config.get("training", {}).get("cpu_full_run_molecule_threshold", 100000))
    corpus_name = str(config.get("corpus_name", "")).lower()
    input_csv = str(config.get("data", {}).get("input_csv", "")).lower()
    looks_full = (
        int(n_molecules) >= threshold
        or ("chembl_full" in corpus_name and max_molecules is None)
        or ("chembl_clean_full" in input_csv and max_molecules is None)
    )
    if not looks_full:
        return
    message = (
        "Refusing CPU training for a dataset that looks like a full ChEMBL-scale run "
        f"(n_molecules={n_molecules}, threshold={threshold}). Use --device cuda for cloud "
        "training, or pass --allow-cpu-full-run if this CPU run is intentional."
    )
    if allow_cpu_full_run:
        print(f"Warning: {message}")
        return
    raise RuntimeError(message)


def _prepare_run_dir(output_config: dict) -> Path:
    run_dir = output_config.get("run_dir")
    if run_dir is None:
        root_dir = Path(output_config.get("root_dir", "runs/pretrain"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = root_dir / timestamp
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def _save_config_yaml(config: dict, path: Path) -> None:
    try:
        import yaml
    except ModuleNotFoundError:
        path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _save_metadata(
    path: Path,
    device: str,
    model,
    descriptor_names: list[str],
    feature_spec,
    run_dir: Path,
    config: dict,
    curriculum,
    graph_cache_manifest: str | None,
    n_molecules: int,
) -> None:
    from hcmp.training.curriculum import curriculum_to_metadata

    metadata = {
        "project": "HCMP v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "descriptor_names": list(descriptor_names),
        "feature_spec": asdict(feature_spec),
        "run_dir": str(run_dir),
        "model_id": config.get("model_id", config.get("run_name")),
        "run_name": config.get("run_name", config.get("model_id")),
        "objective_flags": _objective_flags(config),
        "corpus_name": config.get("corpus_name"),
        "n_molecules": int(n_molecules),
        "graph_cache_manifest": graph_cache_manifest,
        "descriptor_thresholds": config.get("data", {}).get("descriptor_thresholds"),
        "scaffold_cache": config.get("scaf_triplet", {}).get("cache_path"),
        "seed": config.get("seed", config.get("training", {}).get("seed", 0)),
        "pretrain_seed": config.get("pretrain_seed", config.get("seed", config.get("training", {}).get("seed", 0))),
        "model_family": config.get("model_family", "graph_bert" if config.get("baseline") == "traditional_graph_bert" else "hcmp"),
        "feature_mode": config.get("features", {}).get("feature_mode", config.get("dataset", {}).get("feature_mode")),
        "graph_cache_dir": config.get("dataset", {}).get("graph_cache_dir"),
        "scheduler_config": config.get("scheduler", {}),
        "use_amp": config.get("training", {}).get("use_amp", False),
        "gradient_clip_norm": config.get("training", {}).get("gradient_clip_norm"),
        "final_stopping_enabled": bool(config.get("curriculum", {}).get("stop_after_final_phase_plateau", True)),
        "final_stopping_reason": config.get("final_stopping_reason"),
        "training_status": "running",
        "final_global_step": 0,
        "final_epoch": 0,
        "final_phase": getattr(curriculum.current_phase, "name", None),
        "model_config": config.get("model", {}),
        "optimizer_config": config.get("optimizer", {}),
        "curriculum_config": config.get("curriculum", {}),
        "generated_curriculum": curriculum_to_metadata(curriculum),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_train_log_row(
    writer: csv.DictWriter,
    handle,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    phase,
    losses: dict[str, float],
    monitor: float,
    epochs_in_phase: int,
    learning_rate: float,
    seconds_per_step: float,
    device: str,
) -> None:
    row = {
        "global_step": int(global_step),
        "epoch": epoch,
        "step_in_epoch": int(step_in_epoch),
        "phase": phase.name,
        "active_losses": "|".join(phase.active_losses),
        "monitor": phase.monitor,
        "epochs_in_phase": epochs_in_phase,
        "bert_loss": _format_float(losses.get("bert_loss", 0.0)),
        "cut_seg_loss": _format_float(losses.get("cut_seg_loss", losses.get("cut_loss", 0.0))),
        "prop_rank_loss": _format_float(losses.get("prop_rank_loss", 0.0)),
        "scaf_triplet_loss": _format_float(losses.get("scaf_triplet_loss", 0.0)),
        "weighted_total_loss": _format_float(
            losses.get("weighted_total_loss", losses.get("total_loss", 0.0))
        ),
        "num_valid_descriptor_pairs": _format_float(
            losses.get("num_valid_descriptor_pairs", 0.0)
        ),
        "num_valid_triplets": _format_float(losses.get("num_valid_triplets", 0.0)),
        "scaffold_cache_hits": _format_float(losses.get("cache_hits", 0.0)),
        "scaffold_cache_misses": _format_float(losses.get("cache_misses", 0.0)),
        "scaffold_distance_failures": _format_float(
            losses.get("scaffold_distance_failures", 0.0)
        ),
        "lr": _format_float(learning_rate),
        "seconds_per_step": _format_float(seconds_per_step),
        "gpu_memory_allocated_mb": _format_float(_gpu_memory_allocated_mb(device)),
        "device": device,
    }
    writer.writerow(row)
    handle.flush()


def _save_checkpoint(
    path: Path,
    torch,
    model,
    optimizer,
    epoch: int,
    config: dict,
    curriculum,
    loss_balancer,
    best_metric: float | None,
    descriptor_names: list[str],
    feature_spec,
    training_status: str = "running",
    exception: BaseException | None = None,
    global_step: int = 0,
    step_in_epoch: int = 0,
    scheduler=None,
    grad_scaler=None,
    stopping_reason: str | None = None,
    checkpoint_kind: str | None = None,
) -> None:
    from hcmp.training.curriculum import curriculum_to_metadata

    rng_state = _rng_state(torch)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "step_in_epoch": int(step_in_epoch),
        "config": config,
        "curriculum_state": _curriculum_state(curriculum),
        "loss_balancer_state": {
            "method": getattr(loss_balancer, "method", None),
            "weights": getattr(loss_balancer, "weights", None),
        },
        "best_metric": best_metric,
        "descriptor_names": list(descriptor_names),
        "feature_spec": asdict(feature_spec),
        "rng_state": rng_state,
        "python_random_state": rng_state.get("python_random_state"),
        "numpy_rng_state": rng_state.get("numpy_rng_state"),
        "torch_rng_state": rng_state.get("torch"),
        "torch_cuda_rng_state": rng_state.get("torch_cuda"),
        "training_status": training_status,
        "checkpoint_kind": checkpoint_kind,
        "final_stopping_reason": stopping_reason,
        "final_global_step": int(global_step),
        "final_epoch": int(epoch),
        "final_phase": getattr(curriculum.current_phase, "name", None),
        "model_id": config.get("model_id", config.get("run_name")),
        "run_name": config.get("run_name", config.get("model_id")),
        "model_family": config.get("model_family", "graph_bert" if config.get("baseline") == "traditional_graph_bert" else "hcmp"),
        "feature_mode": config.get("features", {}).get("feature_mode", config.get("dataset", {}).get("feature_mode")),
        "backbone": config.get("model", {}).get("encoder", {}).get("encoder_type", "graph_transformer"),
        "atom_target_fields": getattr(feature_spec, "atom_target_fields", []),
        "bond_target_fields": getattr(feature_spec, "bond_target_fields", []),
        "objective_flags": _objective_flags(config),
        "corpus_name": config.get("corpus_name"),
        "n_molecules": config.get("n_molecules"),
        "graph_cache_manifest": (
            str(Path(config.get("dataset", {}).get("graph_cache_dir", "")) / "manifest.json")
            if config.get("dataset", {}).get("graph_cache_dir")
            else None
        ),
        "descriptor_thresholds": config.get("data", {}).get("descriptor_thresholds"),
        "scaffold_cache": config.get("scaf_triplet", {}).get("cache_path"),
        "seed": config.get("seed", config.get("training", {}).get("seed", 0)),
        "pretrain_seed": config.get("pretrain_seed", config.get("seed", config.get("training", {}).get("seed", 0))),
        "model_config": config.get("model", {}),
        "optimizer_config": config.get("optimizer", {}),
        "scheduler_config": config.get("scheduler", {}),
        "curriculum_config": config.get("curriculum", {}),
        "generated_curriculum": curriculum_to_metadata(curriculum),
    }
    if exception is not None:
        checkpoint["exception_type"] = type(exception).__name__
        checkpoint["exception_message"] = str(exception)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def _load_checkpoint(torch, path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _completion_checkpoint_name(training_completed: bool) -> str:
    return "final.pt" if training_completed else "interrupted.pt"


def _rng_state(torch) -> dict:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        "python": python_state,
        "numpy": numpy_state,
        "python_random_state": python_state,
        "numpy_rng_state": numpy_state,
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(torch, rng_state: dict | None) -> None:
    if not rng_state:
        print("Warning: checkpoint has no rng_state; resume will continue with current RNG state.")
        return
    try:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        else:
            print("Warning: checkpoint rng_state is missing python state.")
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        else:
            print("Warning: checkpoint rng_state is missing numpy state.")
        if "torch" in rng_state:
            torch_state = rng_state["torch"]
            if hasattr(torch_state, "detach"):
                torch_state = torch_state.detach().cpu()
            torch.set_rng_state(torch_state)
        else:
            print("Warning: checkpoint rng_state is missing torch state.")
        torch_cuda_state = rng_state.get("torch_cuda")
        if torch_cuda_state is not None and torch.cuda.is_available():
            normalized_cuda_state = [
                item.detach().cpu() if hasattr(item, "detach") else item
                for item in torch_cuda_state
            ]
            torch.cuda.set_rng_state_all(normalized_cuda_state)
    except Exception as exc:
        print(f"Warning: failed to restore rng_state from checkpoint: {exc}")


def _curriculum_state(curriculum) -> dict:
    state = {
        "class": type(curriculum).__name__,
        "phase_index": getattr(curriculum, "phase_index", None),
        "phase_history": list(getattr(curriculum, "phase_history", [])),
        "phase_start_step": getattr(curriculum, "phase_start_step", None),
        "epochs_in_phase": getattr(curriculum, "epochs_in_phase", None),
    }
    for name in ["monitor_history", "best_values", "window_buffers"]:
        if hasattr(curriculum, name):
            state[name] = getattr(curriculum, name)
    return state


def _restore_curriculum_state(curriculum, state: dict | None) -> None:
    if not state:
        print("Warning: checkpoint has no curriculum_state; using freshly built curriculum.")
        return
    restored_any = False
    if hasattr(curriculum, "phase_index") and state.get("phase_index") is not None:
        curriculum.phase_index = int(state["phase_index"])
        restored_any = True
    if hasattr(curriculum, "phase_history") and state.get("phase_history") is not None:
        curriculum.phase_history = [float(value) for value in state["phase_history"]]
        restored_any = True
    if hasattr(curriculum, "phase_start_step") and state.get("phase_start_step") is not None:
        curriculum.phase_start_step = int(state["phase_start_step"])
        restored_any = True
    elif hasattr(curriculum, "phase_history") and state.get("epochs_in_phase") is not None:
        print(
            "Warning: checkpoint curriculum_state has epochs_in_phase but no phase_history; "
            "using existing phase_history values."
        )
    if hasattr(curriculum, "epochs_in_phase") and state.get("epochs_in_phase") is not None:
        try:
            setattr(curriculum, "epochs_in_phase", int(state["epochs_in_phase"]))
            restored_any = True
        except (AttributeError, TypeError):
            if (
                hasattr(curriculum, "phase_history")
                and len(getattr(curriculum, "phase_history")) != int(state["epochs_in_phase"])
            ):
                print(
                    "Warning: checkpoint epochs_in_phase could not be set directly and "
                    "does not match restored phase_history length."
                )
    for name in ["monitor_history", "best_values", "window_buffers"]:
        if hasattr(curriculum, name):
            if name in state:
                setattr(curriculum, name, state[name])
                restored_any = True
            else:
                print(f"Warning: checkpoint curriculum_state is missing {name}.")
    if not restored_any:
        print("Warning: checkpoint curriculum_state did not match this curriculum object.")


def _restore_loss_balancer_state(loss_balancer, state: dict | None) -> None:
    if not state:
        return
    if "method" in state and state["method"] is not None:
        loss_balancer.method = state["method"]
    if "weights" in state and state["weights"] is not None:
        loss_balancer.weights = dict(state["weights"])


def _move_optimizer_state_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _monitor_metric(losses: dict[str, float], monitor_name: str) -> float:
    key = _LOSS_TO_REPORTED_KEY.get(str(monitor_name), None)
    if key is not None and key in losses:
        return float(losses[key])
    return float(losses.get("total_loss", 0.0))


def _is_improved(metric: float, best_metric: float | None) -> bool:
    if best_metric is None:
        return True
    return float(metric) < float(best_metric)


def _learning_rate(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _build_scheduler(torch, optimizer, scheduler_config: dict, max_total_steps) -> object | None:
    if not scheduler_config or str(scheduler_config.get("name", "none")).lower() in {"none", ""}:
        return None
    name = str(scheduler_config.get("name")).lower()
    if name != "cosine_with_warmup":
        raise ValueError("Only scheduler.name='cosine_with_warmup' is supported.")
    total_steps = int(max_total_steps or scheduler_config.get("total_steps", 100000))
    warmup_steps = int(round(total_steps * float(scheduler_config.get("warmup_ratio", 0.05))))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _gpu_memory_allocated_mb(device: str) -> float | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        return float(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return None


def _update_run_metadata_status(
    path: Path,
    stopping_reason: str | None,
    training_status: str,
    global_step: int,
    epoch: int,
    final_phase: str | None,
) -> None:
    if not path.exists():
        return
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    metadata.update(
        {
            "final_stopping_reason": stopping_reason,
            "training_status": training_status,
            "final_global_step": int(global_step),
            "final_epoch": int(epoch),
            "final_phase": final_phase,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _objective_flags(config: dict) -> dict[str, bool]:
    return {
        "bert": bool(config.get("bert", {}).get("enabled", False)),
        "cut_seg": bool(config.get("cut_seg", {}).get("enabled", False)),
        "prop_rank": bool(config.get("prop_rank", {}).get("enabled", False)),
        "scaf_triplet": bool(config.get("scaf_triplet", {}).get("enabled", False)),
    }


def _format_float(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


def _int_config_default(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


if __name__ == "__main__":
    main()
