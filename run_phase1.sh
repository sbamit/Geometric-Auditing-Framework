#!/usr/bin/env bash
# =============================================================================
# run_phase1.sh — Phase 1: Concept-Specific Refusal Cone Extraction
# =============================================================================
# Usage:
#   bash run_phase1.sh               # full run with default arguments
#   bash run_phase1.sh --seed_only   # only compute seed directions (fast test)
#   bash run_phase1.sh --cone_dim 1  # override any argument
#
# All extra arguments are forwarded directly to extract_activations.py.
# =============================================================================

set -euo pipefail   # exit on error, undefined var, pipe failure

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE1_DIR="$REPO_ROOT/phase1"
EXTERNAL_DIR="$REPO_ROOT/external/geometry-of-refusal-copy"
ENV_FILE="$EXTERNAL_DIR/.env"
ENV_EXAMPLE="$EXTERNAL_DIR/.env_example"
OUTPUT_DIR="$REPO_ROOT/data/activations"

CONDA_ENV="geometric_audit"
PYTHON="$(conda run -n $CONDA_ENV which python)"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# =============================================================================
# STEP 1 — Verify conda environment
# =============================================================================
info "Conda environment : $CONDA_ENV"
info "Python            : $PYTHON"
info "Repository root   : $REPO_ROOT"

conda info --envs | grep -q "$CONDA_ENV" \
    || error "Conda environment '$CONDA_ENV' not found. Create it first."

# =============================================================================
# STEP 2 — Install missing Python dependencies
# =============================================================================
info "Checking Python dependencies..."

# nnsight requires PyTorch >= 2.0.  The environment currently has 1.13,
# so we upgrade torch first (CUDA 11.7 wheel) then install the rest.
TORCH_VER=$(conda run -n $CONDA_ENV python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "0")
TORCH_MAJOR=$(echo "$TORCH_VER" | cut -d. -f1)

if [ "$TORCH_MAJOR" -lt 2 ]; then
    warn "PyTorch $TORCH_VER detected — nnsight requires >= 2.0."
    info "Upgrading PyTorch (torch only — torchvision excluded to avoid pin conflicts)..."
    # Install torch without pinning torchvision: pip will resolve the right torch
    # version for nnsight's requirements. We do not use torchvision in this project.
    conda run -n $CONDA_ENV pip install --quiet \
        "torch" \
        --index-url https://download.pytorch.org/whl/cu117
    info "PyTorch upgraded."
fi

# Install remaining missing packages
MISSING_PKGS=()
for pkg in nnsight bitsandbytes datasets python-dotenv; do
    conda run -n $CONDA_ENV python -c "import ${pkg//-/_}" &>/dev/null \
        || MISSING_PKGS+=("$pkg")
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    info "Installing: ${MISSING_PKGS[*]}"
    # nnsight 0.5+ requires transformers>=4.48 which dropped Python 3.8 — pin to 0.4.11
    conda run -n $CONDA_ENV pip install --quiet \
        "${MISSING_PKGS[@]/nnsight/nnsight==0.4.11}"
    info "Dependencies installed."
else
    info "All dependencies present."
fi

# Downgrade nnsight if a broken 0.5.x version slipped in
NNSIGHT_VER=$(conda run -n $CONDA_ENV python -c "import nnsight; print(nnsight.__version__)" 2>/dev/null || echo "0")
NNSIGHT_MAJOR=$(echo "$NNSIGHT_VER" | cut -d. -f1)
NNSIGHT_MINOR=$(echo "$NNSIGHT_VER" | cut -d. -f2)
if [ "$NNSIGHT_MAJOR" -gt 0 ] || [ "$NNSIGHT_MINOR" -ge 5 ]; then
    warn "nnsight $NNSIGHT_VER detected — 0.5+ requires transformers>=4.48 (incompatible with Python 3.8)."
    info "Downgrading nnsight to 0.4.11..."
    conda run -n $CONDA_ENV pip install --quiet "nnsight==0.4.11"
    info "nnsight pinned to 0.4.11."
fi

# =============================================================================
# STEP 3 — Set up .env file
# =============================================================================
if [ ! -f "$ENV_FILE" ]; then
    warn ".env not found — creating from template at $ENV_FILE"
    cp "$ENV_EXAMPLE" "$ENV_FILE"

    # Fill in sensible defaults
    HF_CACHE="${HOME}/.cache/huggingface"
    SAVE_DIR="$REPO_ROOT/data/results"
    mkdir -p "$SAVE_DIR"

    sed -i "s|path-to-huggingface-cache|${HF_CACHE}|g" "$ENV_FILE"
    sed -i "s|./results|${SAVE_DIR}|g"                  "$ENV_FILE"

    warn "Edit $ENV_FILE to add your WANDB_ENTITY / WANDB_PROJECT if needed."
else
    info ".env found at $ENV_FILE"
fi

# Export env vars into current shell so the script can read them
set -a
source "$ENV_FILE"
set +a

# =============================================================================
# STEP 4 — Verify local model path
# =============================================================================
MODEL_PATH="/home/samuel/research/llmattacks/llm-attacks/DIR/Llama-3.1-8B-Instruct"
info "Checking local model path..."
if [ -f "$MODEL_PATH/config.json" ]; then
    info "Model found at $MODEL_PATH"
else
    error "Model not found at $MODEL_PATH — check the path is mounted and accessible."
fi

# =============================================================================
# STEP 5 — Create output directory
# =============================================================================
mkdir -p "$OUTPUT_DIR"
info "Output directory  : $OUTPUT_DIR"

# =============================================================================
# STEP 6 — Run extract_activations.py
# =============================================================================
# Default arguments — all can be overridden by passing flags to this script.
#
#   --model        HuggingFace model id (gated; needs HF login)
#   --layer        Residual-stream layer to hook.  16 is mid-network for a
#                  32-layer model and is where refusal geometry is strongest
#                  according to the DIM paper.
#   --cone_dim     k = 3 basis vectors per cone (B_t ∈ R^{3 × d})
#   --n_harmful    32 harmful prompts per category (matches local CSV size)
#   --n_harmless   32 harmless prompts per category
#   --epochs       1 full pass over the dataset (use 2–3 for stronger cones)
#   --spec_lambda  0.5 concept-specificity penalty weight
#   --out          where B_t tensors and targets are saved

info "Launching extract_activations.py..."
echo ""

conda run -n $CONDA_ENV \
    python "$PHASE1_DIR/extract_activations.py" \
        --model       "$MODEL_PATH" \
        --layer       16 \
        --cone_dim    3 \
        --n_harmful   32 \
        --n_harmless  32 \
        --epochs      5 \
        --spec_lambda 0.5 \
        --out         "$OUTPUT_DIR" \
        "$@"    # forward any extra CLI args (e.g. --seed_only, --cone_dim 1)

# =============================================================================
# STEP 7 — Summary
# =============================================================================
echo ""
info "Phase 1 complete.  Output files:"
echo ""
for concept in violence cybercrime selfharm; do
    for ext in seed_dir.pt targets.json cone.pt; do
        f="$OUTPUT_DIR/${concept}_${ext}"
        if [ -f "$f" ]; then
            SIZE=$(du -sh "$f" | cut -f1)
            echo "  [$SIZE]  $f"
        fi
    done
done
echo ""
echo "  $OUTPUT_DIR/metadata.json"
echo "  $OUTPUT_DIR/prompts.json"
echo ""
info "Next step: python phase1/rank_analysis.py --activations_dir $OUTPUT_DIR"
