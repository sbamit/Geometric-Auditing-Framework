# Geometric Auditing Framework — CLAUDE.md

## Project Overview

This project implements a **geometric auditing framework** for studying how large language models
represent and process harmful content. The central hypothesis is that a model's refusal behaviour
is governed by a low-dimensional geometric subspace of its residual stream — a "refusal cone" —
and that different harm categories occupy geometrically distinct directions within that cone.

The pipeline produces per-category **Difference-in-Means (DIM) refusal directions**, validates
them empirically via directional ablation, and will ultimately train optimised **RDO directions**
that are better separable across categories.

**Primary model:** Llama-3.1-8B-Instruct (4096-dim hidden states, 32 decoder layers)

---

## Environment

```bash
conda activate geometric_audit   # Python 3.8
```

Key packages: `transformers==4.46.3`, `torch`, `einops`, `datasets`, `tqdm`, `matplotlib`, `numpy`

**Model path** must be set manually in each notebook's Cell 2 config block:
```python
MODEL_PATH = '/path/to/your/Llama-3.1-8B-Instruct'  # update this
```
The model is loaded with `device_map='auto'` and `torch_dtype=torch.float16`.

---

## Phase 1 Pipeline

### Notebook Execution Order

```
01_dataset_curation.ipynb       ← Load SALADBENCH + ALPACA, save global splits
01b_category_splits.ipynb       ← Split SALADBENCH into per-category subsets
01c_dim_per_category_notebook.ipynb  ← Compute per-category DIM refusal directions
02_target_generation.ipynb      ← Generate t_answer / t_refusal / t_retain targets
03_rdo_training.ipynb           ← Train RDO direction per category (Algorithm 1)
04_evaluation.ipynb             ← Compare DIM vs RDO: ASR, geometry, examples
```

**No GPU needed** for Notebooks 01 and 01b. GPU required for all others.

---

## Dataset (Notebook 01b)

**Source:** SALADBENCH (21,318 harmful prompts) + ALPACA (17,054 harmless prompts)

**Instruction column:** `question` (not `augmented_question` — many SALADBENCH versions
leave the augmented column empty)
**Category column:** `3-category` (66 fine-grained labels)

### Category Sizes (post-filtering)

| Category   | Raw matches | Train | Val |
|------------|-------------|-------|-----|
| `violence`   | 2,566       | 301   | 76  |
| `self_harm`  | 377         | 301   | 76  |
| `cybercrime` | 766         | 301   | 76  |
| unmatched    | 17,609      | —     | —   |

Train/val split is 80/20, capped at the smallest category (self_harm: 377 total → 301/76).
The harmless pool (ALPACA, 17,054 prompts) is shared across all categories as a common baseline.

### Category Mapping Strategy

Two-tier matching per record:
1. Substring match on the `3-category` field (e.g. `'O56: Violent Crimes'`, `'O62: Self-Harm'`, `'O38: Cyber Attack'`)
2. Keyword fallback on the instruction text if field match fails

`violence` covers hate speech, violent crimes, terrorism, weapons, bullying.
`self_harm` covers self-harm, suicide, eating disorders.
`cybercrime` covers cyber attacks, malware, hacking, surveillance.

### Output Files

```
phase1/data/saladbench_splits/
├── harmful_train.json / harmful_val.json     ← global splits (not per-category)
├── harmless_train.json / harmless_val.json   ← ALPACA splits
├── categories/
│   ├── violence/   harmful_train.json, harmful_val.json, harmless_train.json, harmless_val.json
│   ├── self_harm/  ...
│   └── cybercrime/ ...
```

---

## DIM Refusal Directions (Notebook 01c)

### Method

For each category, compute:

```
v_DIM^(c) = (μ_harmful^(c) − μ_harmless) / ‖μ_harmful^(c) − μ_harmless‖
```

where μ is the mean residual-stream activation at the last token position (`pos=-1`) at a
chosen layer, over 128 prompts from the category's `harmful_train.json` and the shared ALPACA
harmless pool.

### Computed Results

| Category   | Layer | Peak Layer | Direction Norm | N_harmful | N_harmless |
|------------|-------|------------|----------------|-----------|------------|
| violence   | 14    | **32**     | 1.000008       | 128       | 128        |
| self_harm  | 14    | **32**     | 0.999996       | 128       | 128        |
| cybercrime | 14    | **32**     | 0.999995       | 128       | 128        |

**Note:** The peak mean-diff norm is at layer 32 (final layer) for all categories, not layer 14.
Re-running with `LAYER=32` in Cell 2 of 01c may strengthen the directions. Layer 14 is used
as a conservative choice per the original RDO paper recommendation.

### Pairwise Cosine Similarities

|           | violence | self_harm | cybercrime |
|-----------|----------|-----------|------------|
| violence  | 1.000    | **0.960** | 0.871      |
| self_harm | 0.960    | 1.000     | 0.860      |
| cybercrime| 0.871    | 0.860     | 1.000      |

**Warning:** `violence` and `self_harm` directions are nearly collinear (cosine = 0.96).
These two categories are not well-separated in the model's representation space at layer 14.
`cybercrime` is the most geometrically distinct (~0.87 from both others).

### Output Files

```
phase1/dim_outputs/
├── direction.pt / direction_metadata.json / mean_diffs.pt  ← global DIM (from 00_dim_notebook)
├── direction_cosine_similarity.png
├── violence/   direction.pt, direction_metadata.json, mean_diffs.pt, dim_norm_by_layer.png
├── self_harm/  ...
└── cybercrime/ ...
```

---

## Ablation Experiment Results (Notebook 01c, Cell 10)

### Setup

- **Val prompts:** 16 per category from `harmful_val.json` (held out from DIM training)
- **Baseline:** `model.generate()` with no hooks — model generates normally (should refuse)
- **Intervention:** DIM direction ablated from all 32 decoder layers simultaneously via
  forward hooks: `x ← x − (x · v̂)v̂`
- **Max new tokens:** 128, greedy decoding (`do_sample=False`)

### Attack Success Rate

| Category   | Bypasses | Refusals Held | N/A (baseline not refusing) | **Bypass Rate** |
|------------|----------|---------------|------------------------------|-----------------|
| violence   | 6        | 8             | 2                            | 43% of applicable |
| self_harm  | 4        | 10            | 2                            | 29% of applicable |
| cybercrime | 7        | 5             | 4                            | 58% of applicable |
| **Total**  | **17**   | **23**        | **8**                        | **~42% overall** |

### Qualitative Observations

- **Cybercrime direction is the strongest**: ablation produces working Python keylogger code,
  malware creation guides, surveillance hacking steps, and website hijacking scripts.
- **Self-harm direction is the weakest**: consistent with the high violence/self_harm cosine
  similarity (0.96) — the direction carries less category-specific signal.
- **Hard floors remain**: nerve agent synthesis, direct suicide methods, and short explicit
  commands ("install malware on server") resist ablation across all categories. These may be
  encoded by additional circuits beyond the refusal direction.
- **Soft bypasses are common**: the model often prefaces harmful content with
  "Warning: for educational purposes only" before complying. Safety fine-tuning partially
  reasserts itself even without the geometric refusal signal.
- **A prior global DIM experiment on Llama-2-7B-Chat** achieved 56% full bypass + 6% partial
  bypass (62% lenient ASR) on 32 AdvBench prompts. Results stored in
  `phase1/dim_outputs/dim_results.json`.

---

## Known Technical Issues and Fixes

### 1. Stale Forward Hooks After Failed Runs

**Symptom:** `model.generate()` triggers an old hook even when `fwd_hooks=[]` is passed.
Old hooks are left dangling on decoder layers when a generation call crashes before the
`finally` block can run.

**Fix:** Call `_clear_decoder_hooks(model, NUM_LAYERS)` at the start of `get_generations_01c`.
This sweeps `model.layers.{0..31}._forward_hooks` and clears any leftover handles.
Decoder-layer hooks are safe to clear; accelerate's device-dispatch hooks live on `model.model`,
not on individual decoder layers.

### 2. Device Mismatch: direction on CPU, activation on CUDA

**Symptom:** `RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu`
inside `direction_ablation_hook` → `einops.einsum`.

**Root cause:** DIM directions (`all_directions[cat]`) are computed and stored on CPU in Cell 6.
During generation, `activation` arrives on `cuda:0` via the forward hook.

**Fix:** Add `.to(activation.device)` inside `direction_ablation_hook`:
```python
direction = direction.float().to(activation.device)
```

### 3. transformers 4.46 + device_map='auto' Incompatibility with Custom Generation Loop

**Symptom:** `RuntimeError` when calling `model(input_ids=..., use_cache=False)` in a manual
token-by-token generation loop with `device_map='auto'`.

**Fix:** Replace the custom loop with `model.generate(use_cache=False)`. With `use_cache=False`,
`model.generate()` performs a full forward pass at every generation step, so forward hooks still
fire at every layer for every token — identical ablation behaviour, but accelerate-compatible.

### 4. MODEL_PATH Placeholder

**Symptom:** `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`

**Fix:** Update `MODEL_PATH` in Cell 2 of each notebook to your local checkpoint path before
running anything else.

---

## Generation Utility Functions (defined in 01c Cell 9)

```python
direction_ablation_hook(activation, hook, direction)
    # Orthogonal projection: x ← x − (x·v̂)v̂
    # Casts to float32; moves direction to activation.device before einsum

_clear_decoder_hooks(model, num_layers)
    # Removes stale _forward_hooks from model.layers.0 .. model.layers.{num_layers-1}

get_generations_01c(model, instructions, max_tokens_generated, batch_size, fwd_hooks)
    # Batched generation via model.generate(use_cache=False)
    # Registers hooks per batch; always removes them in try/finally
    # Calls _clear_decoder_hooks at entry to neutralise stale state
```

Hook format:
```python
hook_fn   = functools.partial(direction_ablation_hook, direction=cat_dir)
fwd_hooks = [(f'model.layers.{l}', hook_fn) for l in range(NUM_LAYERS)]
```

---

## Next Steps

1. **Re-run 01c with `LAYER=32`** — peak mean-diff norm is at the final layer for all three
   categories; using layer 32 may produce stronger, more separable DIM directions.

2. **Expand the val set** — currently only 16 prompts per category. Increase `N_INST_TEST`
   to 32–64 for more reliable ASR estimates.

3. **Compute category-specific RDO directions** — run Notebooks 02 and 03 per category using:
   ```python
   DIM_DIR    = "dim_outputs/{cat}"
   SPLITS_DIR = "data/saladbench_splits/categories/{cat}"
   OUTPUT_DIR = "rdo_outputs/{cat}"
   ```

4. **Stack and decompose the direction matrix** — once all three RDO directions exist, stack
   them into `B ∈ ℝ^{4096×3}`, run SVD, and estimate the effective dimensionality of the
   refusal subspace.

5. **Compute cross-category principal angles** — form Gram matrices `G_ij = Q_i^T Q_j`
   between the per-category subspaces and take their SVD to test the geometric separability
   hypothesis.

6. **Address violence/self_harm collinearity** — cosine similarity of 0.96 suggests these
   categories share nearly identical refusal geometry at layer 14. Options: use a later layer,
   use per-category RDO (which optimises for separability), or expand self_harm training data
   beyond 377 prompts.
