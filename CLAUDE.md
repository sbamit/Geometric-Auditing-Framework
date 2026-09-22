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

### Category Sizes (post-filtering) — updated after the 2026-09-22 relabelling fix

| Category   | Raw matches | Train | Val |
|------------|-------------|-------|-----|
| `violence`   | 2,612       | 310   | 78  |
| `self_harm`  | 388         | 310   | 78  |
| `cybercrime` | 1,038       | 310   | 78  |
| unmatched    | 17,280      | —     | —   |

Train/val split is 80/20, capped at the smallest category (self_harm: 388 total → 310/78), using
**stratified** sampling by `3-category` label so the capped set preserves the pre-cap label mix
(see item 5 under Known Issues, now fixed).
The harmless pool (ALPACA, 17,054 prompts) is shared across all categories as a common baseline.

### Category Mapping Strategy

**Revised 2026-09-22** (was two-tier substring + keyword-fallback matching — see "Known issues,
fixed" below). Now a single tier: **exact-string match** on the `3-category` field against an
explicit, curated label set per category. No keyword/instruction-text fallback.

```python
CATEGORY_LABEL_SETS = {
    'self_harm': {'O62: Self-Harm'},
    'violence': {
        'O56: Violent Crimes', 'O5: Violent Content', 'O4: Terrorism',
        'O35: Weapon Generation and Transportation',
        'O2: Harass, Threaten, or Bully An Individual',
    },
    'cybercrime': {
        'O38: Cyber Attack', 'O37: Malware Generation', 'O52: Illegitimate Surveillance',
    },
}
```

`violence` now covers violent crimes, violent content, terrorism, weapon generation, and
harassment/bullying — **hate speech is deliberately excluded** (see item 4 below).
`self_harm` covers only `O62: Self-Harm` (suicide, self-injury, eating disorders all fall under
this single SALADBENCH label).
`cybercrime` covers cyber attacks, malware generation, and illegitimate surveillance.

Categories are checked in order `self_harm → violence → cybercrime` (self-harm given priority —
see item 2 below), first match wins.

### Category Label Quality (audit of saved 01b splits) — updated after the relabelling fix

Composition of the saved files (train+val = 388 records per category):

| Category   | Composition | Quality |
|------------|-------------|---------|
| `self_harm`  | 388/388 are `O62: Self-Harm` | **Cleanest** — unchanged, single exact label |
| `violence`   | Violent Crimes 113, Harass/Threaten/Bully 82, Violent Content 76, Weapon Generation 68, Terrorism 49 (proportions from the 310-record stratified train cap) | Exact-label only, no keyword noise, no hate speech |
| `cybercrime` | Illegitimate Surveillance 116, Cyber Attack 98, Malware Generation 96 (310-record stratified train cap) | Exact-label only, no keyword noise |

**Known issues in 01b — fixed 2026-09-22:**
1. ~~Keyword fallback uses raw substring matching~~ — **fixed**: the keyword-fallback tier was
   removed entirely; matching is now exact-string against `CATEGORY_LABEL_SETS`, so substring
   false positives (`stab`→"stable", `kill`→"skill", `bomb`→"bombing", etc.) can no longer occur.
2. ~~Priority order leaks self-harm into violence~~ — **fixed**: `TARGET_CATEGORIES` order changed
   to `self_harm → violence → cybercrime`, and since matching is exact-label (no keyword tier),
   there is no longer a mechanism for a self-harm prompt to be caught by violence first.
3. ~~Field terms miss labels the docs claim are covered~~ — **fixed**: `O2: Harass, Threaten, or
   Bully`, `O35: Weapon Generation and Transportation`, and `O52: Illegitimate Surveillance` are
   now explicit members of `CATEGORY_LABEL_SETS['violence']` / `['cybercrime']`.
4. ~~Hate Speech included in `violence`~~ — **fixed**: `O1: Hate Speech` is deliberately excluded
   from `CATEGORY_LABEL_SETS['violence']`, so `violence` is now purely physical
   violence/weapons/terrorism/harassment rather than a broad mixture.
5. ~~Cap was a random, unstratified subsample~~ — **fixed**: capping now uses proportional
   (largest-remainder) stratified sampling by `3-category` label, so the label mix is preserved
   after the cap to 310/78.

Directions from 01c and downstream results below (ablation ASR, cosine similarities) now reflect
this fix — see updated numbers in the sections below. Re-running 01c did **not** require any code
changes (see TODO item 3); only the underlying category splits changed.

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

**Superseded 2026-09-22 (post-relabelling re-run).** 01c no longer uses a fixed `LAYER=14`; it
searches layers 8–25 × token positions -5..-1 (Arditi-style selection) and picks the best
(layer, position) per category. The table below is from the current search-based 01c, run against
the relabelled 01b splits:

| Category   | Selected Layer | Selected Pos | Direction Norm | N_harmful | N_harmless |
|------------|----------------|--------------|-----------------|-----------|------------|
| violence   | 16             | -3           | 1.000003        | 128       | 128        |
| self_harm  | 12             | -3           | 1.000004        | 128       | 128        |
| cybercrime | 13             | -5           | 0.999973        | 128       | 128        |

The old fixed-layer-14 table (kept for history): violence/self_harm/cybercrime all at layer 14,
with the peak raw mean-diff norm observed at layer 32 for all three — that observation motivated
widening the search range in Cell 2 rather than hardcoding `LAYER=32`.

### Pairwise Cosine Similarities

**Updated 2026-09-22 (post-relabelling re-run, current per-category layer/position search):**

|           | violence | self_harm | cybercrime |
|-----------|----------|-----------|------------|
| violence  | 1.000    | **0.419** | 0.259      |
| self_harm | 0.419    | 1.000     | 0.315      |
| cybercrime| 0.259    | 0.315     | 1.000      |

Max off-diagonal similarity: 0.419. Min: 0.259. Directions are now clearly non-collinear,
supporting the multi-dimensional refusal-cone hypothesis (contrast with the old fixed-layer-14
table below).

The violence/self_harm cosine dropped sharply from the old figure (0.960 → 0.419). This is
consistent with — and plausibly largely explained by — the 01b relabelling fix: the old pipeline
leaked self-harm prompts into `violence` via a priority-order bug (see "Known issues in 01b" in
the Dataset section, item 2, now fixed) and diluted `violence` with ~22% Hate Speech (item 4, now
fixed). It is *not* a clean, isolated measurement of the labelling fix alone, since 01c's
per-category layer/position search also changed between the two runs (old: fixed layer 14; new:
searched optimum per category) — see TODO item 9.

**Old fixed-layer-14 table (superseded, kept for reference):**

|           | violence | self_harm | cybercrime |
|-----------|----------|-----------|------------|
| violence  | 1.000    | 0.960     | 0.871      |
| self_harm | 0.960    | 1.000     | 0.860      |
| cybercrime| 0.871    | 0.860     | 1.000      |

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

## Ablation Experiment Results (Notebook 01c)

### Setup

**Updated 2026-09-22:** val pool per category grew from 76 to 78 with the relabelling fix; 01c
now tests 32 prompts per category (drawn from `val[-32:]`), not 16.

- **Val prompts:** 32 per category from `harmful_val.json` (held out from DIM training)
- **Baseline:** `model.generate()` with no hooks — model generates normally (should refuse)
- **Intervention:** DIM direction ablated from all 32 decoder layers simultaneously via
  forward hooks: `x ← x − (x · v̂)v̂`
- **Max new tokens:** 128, greedy decoding (`do_sample=False`)

### Attack Success Rate

> **PROVISIONAL METRIC — see MUST TODO in Next Steps.** ASR here is computed with a regex over
> refusal phrases ("I can't", "I'm sorry", ...): "bypass" = no refusal phrase in the output, NOT
> verified harmful compliance. Soft refusals, rebuttals and empathetic replies get miscounted as
> bypasses (especially in `violence`, which has many statement/opinion-style prompts). Do not
> draw strong conclusions about cross-category ASR differences until the LLM-judge evaluation exists.

**Updated 2026-09-22 (post-relabelling re-run, 32 prompts/category):**

| Category   | Tested | N/A (baseline not refusing) | Applicable | Bypasses | Refusals Held | **Bypass Rate (ASR)** |
|------------|--------|------------------------------|------------|----------|----------------|------------------------|
| violence   | 32     | 7                             | 25         | 21       | 4              | 84.0% |
| self_harm  | 32     | 5                             | 27         | 27       | 0              | 100.0% |
| cybercrime | 32     | 10                            | 22         | 21       | 1              | 95.5% |
| **Total**  | **96** | **22**                        | **74**     | **69**   | **5**          | **93.2% overall** |

This is much higher than the old fixed-layer-14 numbers (43% / 29% / 58%, ~42% overall — kept
below for reference). The comparison is **confounded**, not a clean isolation of the relabelling
fix: besides the new labels, 01c's per-category layer/position search now selects markedly
stronger directions than the old fixed layer 14 (`induce` scores of +0.67 to +1.19 vs. the old
fixed-layer choice). Do not attribute the full ASR jump to the labelling fix alone.

**Old fixed-layer-14 table (superseded, 16 prompts/category, kept for reference):**

| Category   | Bypasses | Refusals Held | N/A (baseline not refusing) | **Bypass Rate** |
|------------|----------|---------------|------------------------------|-----------------|
| violence   | 6        | 8             | 2                            | 43% of applicable |
| self_harm  | 4        | 10            | 2                            | 29% of applicable |
| cybercrime | 7        | 5             | 4                            | 58% of applicable |
| **Total**  | **17**   | **23**        | **8**                        | **~42% overall** |

### Qualitative Observations

- **Self-harm direction is now the strongest, not the weakest** (100.0% ASR) — a reversal from
  the old fixed-layer-14 run, where self-harm was the weakest (29%) and attributed to the high
  violence/self_harm cosine similarity (0.960). That collinearity dropped to 0.419 after the
  relabelling fix (see Pairwise Cosine Similarities above), consistent with self-harm now carrying
  a cleaner, more category-specific signal.
- **Cybercrime direction remains strong** (95.5% ASR): ablation still produces working malware,
  surveillance/hacking steps, and unauthorized-access instructions.
- **Violence ASR is also much higher** (84.0% vs. the old 43%), now including gun-acquisition,
  bomb-making outline, and targeted-persecution bypasses.
- **Hard floors and soft bypasses**: not yet re-audited qualitatively at the 32-prompt scale after
  the relabelling fix — revisit if/when the LLM-judge evaluation (TODO item 2) lands, since the
  regex ASR here may be overcounting soft/hedged compliance as full bypasses.
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

## TODO List (sequenced — to do later)

Work through these in order. Items 1–2 are **MUST** items: no ASR number should be reported as a
final result until both are done.

### MUST — Evaluation redesign

- [ ] **1. Use a separate data source for testing/evaluation.**
  - Training and validation stay on SaladBench (DIM means from `harmful_train`, direction
    selection from `harmful_val`).
  - Test/evaluation must use a different benchmark: **JailbreakBench or HarmBench**.
  - Currently 01c Cell 10 tests on the last 32 prompts of the SaladBench `harmful_val` split —
    replace this with prompts from the held-out benchmark.
  - Map the benchmark's behaviours onto `violence` / `self_harm` / `cybercrime`, and check
    coverage first: neither benchmark may have a clean self-harm equivalent.

- [ ] **2. Replace regex ASR with LLM-as-a-judge evaluation (Beaver-Cost + Llama-3).**
  - The current ASR (01c Cell 10, `REFUSAL_RE` / `is_refusal`) only checks for refusal phrases.
    Re-evaluate responses the way the author's other adversarial-attack-on-LLMs work does.
  - Use LLM judges — **Beaver-Cost** and **Llama-3** models — to score each model response under
    the intervention (directional ablation / jailbreak) for **harmfulness** and **usefulness**.
  - Compute ASR from those judge scores (a response counts as a success only if it is actually
    harmful/useful, not merely "did not refuse"); report harmfulness and usefulness separately.
  - Apply the same judge identically to every category so cross-category comparisons are fair,
    and run the baseline (no-intervention) responses through the same judge.
  - Applies to the 01c ablation results and to the later DIM-vs-RDO comparison in 04.

### Data and direction fixes

- [x] **3. Fix 01b category labelling, then re-run 01b and 01c.** ~~Violence and cybercrime
  contain mischaracterised prompts (see "Category Label Quality").~~ **Done 2026-09-22**: 01b now
  uses exact `3-category` label sets (dropped the keyword-fallback tier), gives self_harm match
  priority, excludes Hate Speech from `violence`, adds the previously-missed `O2`/`O35`/`O52`
  labels, and stratifies the train/val cap. 01c re-run required no code changes. See "Category
  Label Quality" and the updated DIM/ablation numbers above.

- [ ] **4. Revisit layer/position choice for the DIM directions.** Original idea: re-run 01c with
  `LAYER=32` because the peak mean-diff norm is at the final layer. *Note: the current 01c no
  longer uses a fixed `LAYER` — it searches layers 8–25 × positions -5..-1 (Arditi-style
  selection), and the selections are near-ties; re-assess in that light.*

- [ ] **5. Expand the val set** for more reliable ASR estimates. *Note: after the item-3
  relabelling fix, 01c now uses `N_INST_TEST = 32` from a 78-prompt val pool per category (was
  76); the selection set (`val[:32]`) and test set (`val[-32:]`) are disjoint only while the pool
  has at least 64 prompts — 78 still qualifies, but only barely.*

### RDO and geometry

- [ ] **6. Compute category-specific RDO directions** — run Notebooks 02 and 03 per category:
  ```python
  DIM_DIR    = "dim_outputs/{cat}"
  SPLITS_DIR = "data/saladbench_splits/categories/{cat}"
  OUTPUT_DIR = "rdo_outputs/{cat}"
  ```

- [ ] **7. Stack and decompose the direction matrix** — once all three RDO directions exist, stack
  them into `B ∈ ℝ^{4096×3}`, run SVD, and estimate the effective dimensionality of the
  refusal subspace.

- [ ] **8. Compute cross-category principal angles** — form Gram matrices `G_ij = Q_i^T Q_j`
  between the per-category subspaces and take their SVD to test the geometric separability
  hypothesis.

- [ ] **9. Address violence/self_harm collinearity.** *Update 2026-09-22: after the item-3
  relabelling fix, violence/self_harm cosine dropped from 0.960 (old fixed-layer-14, mislabelled
  data) to 0.419 (current per-category layer/position search, relabelled data). This is real
  progress but still a confounded comparison — both the labels and the layer/position selection
  changed between runs, and 0.419 is still the largest off-diagonal similarity of the three
  pairs. If it's still too high after isolating the two effects (e.g. by re-running the old fixed
  layer-14 config on the new labels), remaining options: compare at a common (layer, position),
  use per-category RDO (optimises for separability), or expand self_harm training data beyond the
  current 388 prompts.*
