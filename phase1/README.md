# RDO Pipeline — Running Instructions

## Overview

Four notebooks implement the full Refusal Direction Optimization (RDO) pipeline, from raw datasets to a trained refusal direction `r` ready for Phase 1 geometric auditing.

```
00_dim_notebook.ipynb     ← Your existing notebook (prerequisite)
01_dataset_curation.ipynb ← Load SALADBENCH + ALPACA, save splits
02_target_generation.ipynb← Bootstrap targets using DIM direction
03_rdo_training.ipynb     ← Run Algorithm 1 (RDO optimization loop)
04_evaluation.ipynb       ← Compare RDO vs DIM: ASR, geometry, examples
```

---

## Prerequisites

### 1. Environment

```bash
pip install nnsight datasets transformers torch tqdm matplotlib numpy
```

### 2. Hugging Face access

You need access to `meta-llama/Llama-3.1-8B-Instruct`. Log in:
```bash
huggingface-cli login
```

### 3. DIM direction artifacts (from your existing notebook)

Before running Notebook 02, save these two files from your existing DIM notebook:

```python
import torch, json, os
os.makedirs('dim_outputs', exist_ok=True)

# refusal_dir: the unit-normed DIM direction vector at layer 14
torch.save(refusal_dir, 'dim_outputs/direction.pt')

# Save metadata (layer index and token position used)
json.dump({"layer": 14, "pos": -1}, open('dim_outputs/direction_metadata.json', 'w'))

# Optionally: save per-layer mean-diff vectors for the layer-norm visualization
torch.save(mean_diffs, 'dim_outputs/mean_diffs.pt')
```

---

## Execution Order

### Notebook 01 — Dataset Curation

**Run first. No GPU required.**

- Loads SALADBENCH (harmful) and ALPACA (harmless) from HuggingFace
- Splits into train/val (80/20), balances to equal lengths
- Saves JSON splits to `data/saladbench_splits/`
- Produces category distribution and length distribution visualizations

**Output directory:** `data/saladbench_splits/`

---

### Notebook 02 — Target Generation

**Run second. Requires GPU. This is the most time-consuming step.**

- Loads model via `nnsight.LanguageModel`
- Generates three target types for every training prompt using DIM direction:
  - `t_answer`: 30 tokens under DIM ablation (all layers) on harmful prompts
  - `t_refusal`: 30 tokens under DIM addition (at `best_layer`) on harmless prompts, truncated at first `.`
  - `t_retain`: 29 tokens clean generation on harmless prompts
- Caches all outputs to disk — re-running is instant if cache exists
- Includes sanity-check cell (3 examples) to verify intervention quality before full generation

**Output directory:** `data/saladbench_splits/targets/`

**Estimated time:** 2–8 hours depending on GPU and dataset size. Run with `batch_size=4` for 24GB VRAM; reduce to `batch_size=1` for 16GB.

---

### Notebook 03 — RDO Training

**Run third. Requires GPU.**

- Loads all data from Notebooks 01 and 02
- Optionally filters training data by bypass score (removes already-bypassed harmful and already-refused harmless examples)
- Builds `RDODataset` and `DataLoader` with label masking
- Runs Algorithm 1 optimization loop:
  - Gradient accumulation over 16 micro-steps
  - AdamW with amsgrad (lr=1e-2, betas=(0.9, 0.98))
  - Riemannian gradient descent: tangent projection before step, re-normalization after
  - Early stopping with LR reduction ×2 (patience=5)
- Produces training diagnostic plots (loss curves, cosine to DIM, norm stability)
- Saves best vector (lowest training loss) as `rdo_outputs/rdo_direction.pt`

**Output directory:** `rdo_outputs/`

**Key outputs:**
- `rdo_direction.pt` — the trained refusal direction `r` (float32, unit norm, shape `(d_model,)`)
- `rdo_metadata.json` — hyperparameters and training summary

**Estimated time:** 30 min – 3 hours depending on dataset size and early stopping.

---

### Notebook 04 — Evaluation

**Run last. Requires GPU.**

- Compares DIM and RDO directions on the held-out validation set
- Computes Attack Success Rate (ASR) for both directions
- Computes induced refusal rate (addition) for both directions
- Computes geometric relationship: cosine similarity and principal angle between DIM and RDO
- Produces side-by-side generation examples for qualitative inspection
- Creates publication-ready comparison figure

**Output directory:** `eval_outputs/`

**Key outputs:**
- `asr_bypass_distributions.png` — bypass score histograms
- `asr_summary.png` — ASR and induced refusal bar charts
- `dim_vs_rdo_full_comparison.png` — combined geometric + performance figure
- `qualitative_examples.json` — 5 generation triplets (baseline / DIM / RDO)
- `full_results.json` — all numeric results in one file

---

## Directory Structure After All Notebooks

```
project_root/
├── dim_outputs/               ← Your existing DIM artifacts
│   ├── direction.pt
│   ├── direction_metadata.json
│   └── mean_diffs.pt
├── data/
│   └── saladbench_splits/
│       ├── harmful_train.json
│       ├── harmful_val.json
│       ├── harmless_train.json
│       ├── harmless_val.json
│       ├── category_distribution.png
│       ├── length_distribution.png
│       └── targets/
│           ├── harmful_targets.json   ← t_answer
│           ├── harmless_targets.json  ← t_refusal + t_retain
│           └── dim_norms_by_layer.png
├── rdo_outputs/
│   ├── rdo_direction.pt        ← PRIMARY OUTPUT: the trained r
│   ├── all_rdo_vectors.pt      ← r at every optimizer step
│   ├── rdo_metadata.json
│   ├── bypass_score_distribution.png
│   └── training_diagnostics.png
└── eval_outputs/
    ├── asr_bypass_distributions.png
    ├── asr_summary.png
    ├── dim_vs_rdo_full_comparison.png
    ├── qualitative_examples.json
    └── full_results.json
```

---

## Hyperparameter Reference

| Parameter | Value | Location | Description |
|---|---|---|---|
| `lr` | `1e-2` | NB03 | Initial learning rate (reduces ×10 per patience trigger) |
| `batch_size` | `1` | NB03 | Micro-batch size per nnsight trace call |
| `effective_batch` | `16` | NB03 | Effective batch via gradient accumulation |
| `patience` | `5` | NB03 | Optimizer steps without improvement before LR reduce |
| `n_lr_reduce` | `2` | NB03 | Max LR reductions before stopping |
| `ablation_lambda` | `1.0` | NB03 | Weight for L_ablation (CE on harmful) |
| `addition_lambda` | `0.2` | NB03 | Weight for L_addition (CE on harmless+refusal) |
| `retain_lambda` | `1.0` | NB03 | Weight for L_retain (KL on harmless) |
| `num_target_tokens` | `30` | NB02/NB03 | Token length for t_answer and t_refusal |
| `retain_tokens` | `29` | NB02 | Token length for t_retain |
| `filter_data` | `True` | NB03 | Whether to filter by bypass score |

---

## Connecting to Your Proposal (Phase 1)

The output of this pipeline — `rdo_outputs/rdo_direction.pt` — is the seed for Phase 1 of your geometric auditing framework. The next step is to:

1. Run RDO **per harm category** (modify NB01 to filter SALADBENCH by category, re-run NB02 and NB03)
2. Stack the resulting per-category directions into matrices `B_t ∈ R^{d×k}`
3. Run SVD on each `B_t` to estimate effective dimensionality
4. Compute cross-category Gram matrices `G_ij = Q_i^T Q_j` and take their SVD to get principal angles
