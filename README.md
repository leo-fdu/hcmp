# HCMP v1

HCMP v1 is a research codebase for **Hierarchical Conservative Molecular
Pretraining** on molecular graphs. The current codebase includes segmentation
labels, descriptor utilities, expanded-scaffold distance utilities, graph-cache
precomputation, multi-objective pretraining, and downstream full finetuning.

This is engineering code for staged experiments. Validate each scale before
spending cloud compute: small smoke test -> 10k graph-cache test -> 100k cloud
dry run -> full ChEMBL pretraining matrix.

## Setup

Create an environment with Python 3.10 or newer, then install the package:

```bash
pip install -e .
```

If RDKit is difficult to resolve from PyPI, install it with conda/mamba first:

```bash
mamba install -c conda-forge rdkit
pip install -e .
```

DeepChem is optional. When available, downstream random/scaffold splits use
DeepChem splitters; otherwise the finetuning script logs a warning and uses the
internal fallback splitter.

## Pretraining Workflow

Prepare descriptor thresholds and graph caches for both feature modes:

```bash
python scripts/estimate_descriptor_thresholds.py ...
python scripts/precompute_graph_cache.py ... --feature-mode hcmp
python scripts/precompute_graph_cache.py ... --feature-mode graph_bert
```

Generate the formal pretraining config matrix:

```bash
python scripts/make_pretrain_matrix_configs.py \
  --base-config configs/hcmp_pretrain.yaml \
  --output-dir configs/experiments/pretrain \
  --corpus-name chembl_full \
  --hcmp-graph-cache-dir data/chembl/graph_cache_hcmp \
  --graph-bert-cache-dir data/chembl/graph_cache_graph_bert \
  --descriptor-thresholds data/chembl/processed/descriptor_thresholds_full.csv
```

Run pretraining on CUDA for cloud-scale work:

```bash
python scripts/train_hcmp.py \
  --config configs/experiments/pretrain/hcmp_full.yaml \
  --device cuda
```

The pretraining matrix contains `graph_bert`, the HCMP ablations, and
`hcmp_full`. `scratch` is not a pretraining model; it is only a downstream
randomly initialized baseline.

Checkpoint names are:

```text
checkpoints/last.pt
checkpoints/final.pt
checkpoints/interrupted.pt
```

If `training.device: cuda` is requested and CUDA is unavailable, training fails
early. CPU smoke tests are allowed, but CPU runs that look like full ChEMBL-scale
training require `--allow-cpu-full-run`.

## Downstream Workflow

Run the downstream matrix:

```bash
python scripts/run_downstream_matrix.py ...
```

Summarize results:

```bash
python scripts/summarize_results.py ...
```

`scripts/finetune_downstream.py` performs full finetuning. It accepts explicit
column overrides:

```bash
python scripts/finetune_downstream.py \
  --dataset bbbp \
  --data-path data/downstream/bbbp.csv \
  --split scaffold \
  --model-id scratch \
  --output-dir runs/smoke_downstream/bbbp_scaffold_scratch \
  --epochs 2 \
  --smiles-column smiles \
  --label-column p_np
```

`scripts/run_downstream_matrix.py` can pass global `--smiles-column` and
`--label-column` overrides, or a JSON/YAML dataset column map.

Every downstream run saves the exact split:

```text
split_indices.json
train_smiles.csv
val_smiles.csv
test_smiles.csv
```

The split metadata records whether DeepChem or the internal fallback splitter
was used.

## Segmentation Debugging

The HCMP segmenter is a deterministic priority-based bond segmentation rule:

1. Ring-system segments
2. Non-ring multiple-bond-seeded, RDKit-conjugation-expanded motifs
3. Connected heteroatom clusters
4. Terminal heteroatom segments

Run the visualization workflow:

```bash
python scripts/01_debug_segmentation.py --config configs/segmentation.yaml
```

Outputs are written under `results/segmentation_visualization/`.

## Tests

Run:

```bash
python -m pytest
```
