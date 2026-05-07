"""
Phase 1 - Day 2: Build B_t Matrices + Rank Analysis
=====================================================
Goal: Construct concept-specific generator matrices B_t via bootstrapped
      mean-difference probing, compute SVD, and plot singular value spectra
      and effective rank per concept.

Inputs:  activations/ directory from Day 1
Outputs: bt_matrices/  — B_t tensors for each concept
         figures/fig1_singular_value_spectra.png
         figures/fig1b_effective_rank_table.png

Usage:
    python phase1_day2_rank_analysis.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.linalg import svd

# ── Config ────────────────────────────────────────────────────────────────────
ACT_DIR    = "activations"
BT_DIR     = "bt_matrices"
FIG_DIR    = "figures"
os.makedirs(BT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CONCEPTS        = ["violence", "cybercrime", "selfharm"]
K_BOOTSTRAP     = 30          # number of bootstrap columns in B_t
N_SAMPLE        = 20          # prompts sampled per bootstrap iteration (harmful + benign each)
EPSILON_VALUES  = [0.05, 0.10, 0.20]   # thresholds for effective rank estimation
SEED            = 42

rng = np.random.default_rng(SEED)

COLORS = {
    "violence":   "#e05c5c",
    "cybercrime": "#5c8de0",
    "selfharm":   "#5cb85c",
}

LABELS = {
    "violence":   "Violence",
    "cybercrime": "Cybercrime",
    "selfharm":   "Self-harm",
}

# ── Build B_t via bootstrapped mean-difference probing ───────────────────────
def build_Bt(harmful_acts: np.ndarray,
             benign_acts:  np.ndarray,
             K: int = K_BOOTSTRAP,
             n: int = N_SAMPLE) -> np.ndarray:
    """
    Returns B_t of shape (d, K).
    Each column = normalized(mean_harmful_sample - mean_benign_sample).
    """
    d = harmful_acts.shape[1]
    Bt = np.zeros((d, K), dtype=np.float32)

    for k in range(K):
        # Sample with replacement
        hi = rng.choice(len(harmful_acts), size=min(n, len(harmful_acts)), replace=True)
        bi = rng.choice(len(benign_acts),  size=min(n, len(benign_acts)),  replace=True)
        diff = harmful_acts[hi].mean(axis=0) - benign_acts[bi].mean(axis=0)
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            diff /= norm
        Bt[:, k] = diff

    return Bt

# ── Effective rank via singular value threshold ───────────────────────────────
def effective_rank(singular_values: np.ndarray, epsilon: float) -> int:
    """Number of singular values above epsilon * sigma_1."""
    threshold = epsilon * singular_values[0]
    return int(np.sum(singular_values > threshold))

# ── Bootstrapped rank distribution ───────────────────────────────────────────
def bootstrapped_rank_distribution(harmful_acts, benign_acts,
                                   n_boot=50, K=K_BOOTSTRAP,
                                   epsilon=0.10) -> np.ndarray:
    """
    Repeatedly build B_t and compute effective rank.
    Returns array of shape (n_boot,).
    """
    ranks = []
    for _ in range(n_boot):
        Bt = build_Bt(harmful_acts, benign_acts, K=K)
        sv = svd(Bt, compute_uv=False)
        ranks.append(effective_rank(sv, epsilon))
    return np.array(ranks)

# ── Figure 1: Singular value spectra ─────────────────────────────────────────
def plot_singular_value_spectra(sv_dict: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: raw singular values
    ax = axes[0]
    for concept, sv in sv_dict.items():
        ax.plot(range(1, len(sv) + 1), sv,
                color=COLORS[concept], label=LABELS[concept],
                linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Component index", fontsize=12)
    ax.set_ylabel("Singular value", fontsize=12)
    ax.set_title("Singular value spectra of $B_t$", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Right: normalized (σ_k / σ_1) with threshold lines
    ax = axes[1]
    for concept, sv in sv_dict.items():
        sv_norm = sv / sv[0]
        ax.plot(range(1, len(sv) + 1), sv_norm,
                color=COLORS[concept], label=LABELS[concept],
                linewidth=2, marker='o', markersize=4)

    for eps, ls in zip(EPSILON_VALUES, ['--', '-.', ':']):
        ax.axhline(eps, linestyle=ls, color='gray', linewidth=1.2,
                   label=f"ε = {eps}")

    ax.set_xlabel("Component index", fontsize=12)
    ax.set_ylabel(r"Normalized singular value $\sigma_k / \sigma_1$", fontsize=12)
    ax.set_title(r"Normalized spectra with thresholds $\tau = \varepsilon \cdot \sigma_1$",
                 fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

# ── Figure 1b: Effective rank table + boxplot ─────────────────────────────────
def plot_rank_summary(rank_dists: dict, rank_table: dict, save_path: str):
    """
    rank_dists: {concept: np.array of bootstrapped ranks}
    rank_table: {concept: {epsilon: rank}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: boxplot of bootstrapped ranks
    ax = axes[0]
    data   = [rank_dists[c] for c in CONCEPTS]
    labels = [LABELS[c] for c in CONCEPTS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, concept in zip(bp['boxes'], CONCEPTS):
        patch.set_facecolor(COLORS[concept])
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Effective rank  (ε = 0.10)", fontsize=12)
    ax.set_title("Bootstrapped rank distribution per concept", fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)

    # Right: table
    ax = axes[1]
    ax.axis('off')
    col_labels = ["Concept"] + [f"ε = {e}" for e in EPSILON_VALUES]
    table_data = []
    for concept in CONCEPTS:
        row = [LABELS[concept]] + [str(rank_table[concept][e]) for e in EPSILON_VALUES]
        table_data.append(row)

    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.4, 2.0)
    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#404040')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i, concept in enumerate(CONCEPTS):
        table[i + 1, 0].set_facecolor(COLORS[concept])
        table[i + 1, 0].set_text_props(color='white', fontweight='bold')
    ax.set_title("Effective rank by threshold τ = ε · σ₁", fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 1 - Day 2: B_t Construction + Rank Analysis")
    print("=" * 60)

    benign_acts = torch.load(os.path.join(ACT_DIR, "benign.pt")).numpy()
    print(f"Benign activations: {benign_acts.shape}")

    sv_dict        = {}
    rank_table     = {c: {} for c in CONCEPTS}
    rank_dists     = {}
    Bt_dict        = {}

    for concept in CONCEPTS:
        print(f"\n── {LABELS[concept]} ──")
        harmful_acts = torch.load(
            os.path.join(ACT_DIR, f"{concept}_harmful.pt")
        ).numpy()
        print(f"  Harmful activations: {harmful_acts.shape}")

        # Build B_t
        print(f"  Building B_t with K={K_BOOTSTRAP} bootstrap columns...")
        Bt = build_Bt(harmful_acts, benign_acts)
        Bt_dict[concept] = Bt
        torch.save(torch.tensor(Bt), os.path.join(BT_DIR, f"Bt_{concept}.pt"))
        print(f"  B_t shape: {Bt.shape}  → saved to bt_matrices/Bt_{concept}.pt")

        # SVD
        sv = svd(Bt, compute_uv=False)
        sv_dict[concept] = sv
        print(f"  Top-5 singular values: {sv[:5].round(4)}")

        # Effective rank per epsilon
        for eps in EPSILON_VALUES:
            r = effective_rank(sv, eps)
            rank_table[concept][eps] = r
            print(f"  Effective rank (ε={eps}): {r}")

        # Bootstrapped rank distribution
        print(f"  Computing bootstrapped rank distribution (50 iters)...")
        rank_dists[concept] = bootstrapped_rank_distribution(harmful_acts, benign_acts)
        print(f"  Rank dist  mean={rank_dists[concept].mean():.1f}  "
              f"std={rank_dists[concept].std():.1f}  "
              f"range=[{rank_dists[concept].min()}, {rank_dists[concept].max()}]")

    # ── Save B_t dict for Day 3 ──
    np.save(os.path.join(BT_DIR, "sv_dict.npy"), sv_dict)

    # ── Figures ──
    print("\n── Generating figures ──")
    plot_singular_value_spectra(
        sv_dict,
        os.path.join(FIG_DIR, "fig1_singular_value_spectra.png")
    )
    plot_rank_summary(
        rank_dists,
        rank_table,
        os.path.join(FIG_DIR, "fig1b_effective_rank_table.png")
    )

    print("\n✓ Day 2 complete.")
    print("  Next: run phase1_day3_principal_angles.py")

if __name__ == "__main__":
    main()
