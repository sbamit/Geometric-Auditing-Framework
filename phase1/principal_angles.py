"""
Phase 1 - Day 3: Principal Angle Computation + Entanglement Visualization
==========================================================================
Goal: Compute principal angles between all pairs of refusal subspaces,
      visualize the full entanglement spectrum, and produce a summary heatmap.

Inputs:  bt_matrices/ from Day 2
Outputs: figures/fig2_principal_angle_spectra.png
         figures/fig2b_entanglement_heatmap.png
         figures/fig2c_pairwise_comparison.png

Usage:
    python phase1_day3_principal_angles.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import svd, qr
from itertools import combinations

# ── Config ────────────────────────────────────────────────────────────────────
BT_DIR  = "bt_matrices"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

CONCEPTS = ["violence", "cybercrime", "selfharm"]
LABELS   = {
    "violence":   "Violence",
    "cybercrime": "Cybercrime",
    "selfharm":   "Self-harm",
}
PAIR_COLORS = {
    ("violence",   "selfharm"):   "#9c59d1",
    ("violence",   "cybercrime"): "#e08c3a",
    ("cybercrime", "selfharm"):   "#3aade0",
}

# ── Step 1: Load B_t matrices ─────────────────────────────────────────────────
def load_Bt(concept: str) -> np.ndarray:
    path = os.path.join(BT_DIR, f"Bt_{concept}.pt")
    return torch.load(path).numpy()

# ── Step 2: QR decomposition → orthonormal basis Q_t ─────────────────────────
def orthonormal_basis(Bt: np.ndarray) -> np.ndarray:
    """
    QR decomposition of B_t. Returns Q of shape (d, k) with orthonormal columns.
    Only keeps columns whose diagonal R entry exceeds a numerical threshold,
    which discards numerically dependent generators.
    """
    Q, R = qr(Bt, mode='economic')
    # Keep columns where |R_kk| > threshold (numerical rank filtering)
    diag_R  = np.abs(np.diag(R))
    keep    = diag_R > 1e-6 * diag_R.max()
    return Q[:, keep]

# ── Step 3: Gram matrix G_ij = Q_i^T Q_j and principal angle cosines ─────────
def principal_angle_cosines(Qi: np.ndarray, Qj: np.ndarray) -> np.ndarray:
    """
    Computes cosines of principal angles between column spaces of Bi and Bj
    via SVD of the cross-subspace Gram matrix G_ij = Q_i^T Q_j.

    Returns sigma: array of singular values in descending order.
    These equal cos(θ_ℓ) for the principal angles θ_ℓ ∈ [0, π/2].
    """
    G = Qi.T @ Qj                      # (k_i, k_j)
    sigma = svd(G, compute_uv=False)   # singular values of G_ij
    return np.clip(sigma, 0.0, 1.0)    # clip numerical noise outside [0,1]

# ── Figure 2: Full principal angle spectra for all pairs ─────────────────────
def plot_principal_angle_spectra(spectra: dict, save_path: str):
    """
    spectra: { (concept_i, concept_j): np.array of cosines }
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for pair, sigma in spectra.items():
        label = f"{LABELS[pair[0]]} ↔ {LABELS[pair[1]]}"
        color = PAIR_COLORS[pair]
        ax.plot(range(1, len(sigma) + 1), sigma,
                color=color, label=label,
                linewidth=2.2, marker='o', markersize=5)

    ax.set_xlabel("Principal angle index  ℓ", fontsize=12)
    ax.set_ylabel(r"$\sigma_\ell = \cos(\theta_\ell)$", fontsize=12)
    ax.set_title("Principal angle spectra between refusal subspaces\n"
                 r"($\sigma_\ell \approx 1$: strongly entangled, "
                 r"$\sigma_\ell \approx 0$: near-orthogonal)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

# ── Figure 2b: Entanglement heatmap (summary statistic per pair) ──────────────
def plot_entanglement_heatmap(spectra: dict, save_path: str):
    """
    Heatmap showing max principal angle cosine for each concept pair.
    """
    n = len(CONCEPTS)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1.0)   # self-overlap is always 1

    idx = {c: i for i, c in enumerate(CONCEPTS)}
    for pair, sigma in spectra.items():
        i, j = idx[pair[0]], idx[pair[1]]
        val = float(sigma.max()) if len(sigma) > 0 else 0.0
        matrix[i, j] = val
        matrix[j, i] = val

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: max cosine heatmap
    ax = axes[0]
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=r"Max $\sigma_\ell$ (max principal angle cosine)")
    labels_list = [LABELS[c] for c in CONCEPTS]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_list, fontsize=11)
    ax.set_yticklabels(labels_list, fontsize=11)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}",
                    ha='center', va='center', fontsize=12,
                    color='white' if matrix[i,j] > 0.6 else 'black',
                    fontweight='bold')
    ax.set_title(r"Max principal angle cosine $\max_\ell \sigma_\ell$", fontsize=13)

    # Right: mean cosine heatmap
    ax = axes[1]
    matrix_mean = np.zeros((n, n))
    np.fill_diagonal(matrix_mean, 1.0)
    for pair, sigma in spectra.items():
        i, j = idx[pair[0]], idx[pair[1]]
        val = float(sigma.mean()) if len(sigma) > 0 else 0.0
        matrix_mean[i, j] = val
        matrix_mean[j, i] = val

    im2 = ax.imshow(matrix_mean, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax, label=r"Mean $\sigma_\ell$ (mean principal angle cosine)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_list, fontsize=11)
    ax.set_yticklabels(labels_list, fontsize=11)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix_mean[i,j]:.2f}",
                    ha='center', va='center', fontsize=12,
                    color='white' if matrix_mean[i,j] > 0.6 else 'black',
                    fontweight='bold')
    ax.set_title(r"Mean principal angle cosine $\langle \sigma_\ell \rangle$", fontsize=13)

    plt.suptitle("Inter-concept refusal subspace entanglement", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

# ── Figure 2c: High vs low entanglement pair side-by-side ─────────────────────
def plot_high_vs_low_entanglement(spectra: dict, save_path: str):
    """
    Side-by-side comparison of the most and least entangled concept pair.
    """
    max_sigma = {pair: s.max() for pair, s in spectra.items()}
    most_entangled  = max(max_sigma, key=max_sigma.get)
    least_entangled = min(max_sigma, key=max_sigma.get)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, pair, title_tag in [
        (axes[0], most_entangled,  "Most entangled"),
        (axes[1], least_entangled, "Least entangled"),
    ]:
        sigma = spectra[pair]
        color = PAIR_COLORS[pair]
        label = f"{LABELS[pair[0]]} ↔ {LABELS[pair[1]]}"
        ax.bar(range(1, len(sigma) + 1), sigma,
               color=color, alpha=0.75, width=0.6)
        ax.plot(range(1, len(sigma) + 1), sigma,
                color=color, linewidth=2, marker='o', markersize=5)
        ax.set_xlabel("Principal angle index  ℓ", fontsize=12)
        ax.set_ylabel(r"$\sigma_\ell = \cos(\theta_\ell)$", fontsize=12)
        ax.set_title(f"{title_tag}\n{label}\n"
                     r"max $\sigma_\ell$ = " + f"{sigma.max():.3f}", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle("Principal angle spectrum: high vs low entanglement", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")

# ── Summary statistics printout ───────────────────────────────────────────────
def print_summary(spectra: dict, Qs: dict):
    print("\n── Principal Angle Summary ──")
    for pair, sigma in spectra.items():
        print(f"\n  {LABELS[pair[0]]} ↔ {LABELS[pair[1]]}")
        print(f"    Q_i rank: {Qs[pair[0]].shape[1]}   Q_j rank: {Qs[pair[1]].shape[1]}")
        print(f"    G_ij shape: ({Qs[pair[0]].shape[1]}, {Qs[pair[1]].shape[1]})")
        print(f"    Number of principal angles: {len(sigma)}")
        print(f"    Cosines: {sigma.round(4)}")
        print(f"    Max cosine (most entangled direction): {sigma.max():.4f}")
        print(f"    Mean cosine:                          {sigma.mean():.4f}")
        near_ortho = int(np.sum(sigma < 0.1))
        strong     = int(np.sum(sigma > 0.7))
        print(f"    Near-orthogonal directions (σ<0.1): {near_ortho}/{len(sigma)}")
        print(f"    Strongly entangled  (σ>0.7):        {strong}/{len(sigma)}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 1 - Day 3: Principal Angle Computation")
    print("=" * 60)

    # Load B_t matrices and compute orthonormal bases Q_t
    Bts = {c: load_Bt(c) for c in CONCEPTS}
    Qs  = {}
    for concept, Bt in Bts.items():
        Q = orthonormal_basis(Bt)
        Qs[concept] = Q
        print(f"[{LABELS[concept]}]  B_t shape: {Bt.shape}  →  Q shape: {Q.shape}")

    # Compute principal angles for all pairs
    spectra = {}
    print("\n── Computing cross-subspace Gram matrices ──")
    for ci, cj in combinations(CONCEPTS, 2):
        pair = (ci, cj)
        Qi, Qj = Qs[ci], Qs[cj]
        sigma = principal_angle_cosines(Qi, Qj)
        spectra[pair] = sigma
        print(f"  G_{ci[:3]},{cj[:3]}  shape: ({Qi.shape[1]}, {Qj.shape[1]})  "
              f"→  σ: {sigma.round(3)}")

    # Summary
    print_summary(spectra, Qs)

    # Figures
    print("\n── Generating figures ──")
    plot_principal_angle_spectra(
        spectra,
        os.path.join(FIG_DIR, "fig2_principal_angle_spectra.png")
    )
    plot_entanglement_heatmap(
        spectra,
        os.path.join(FIG_DIR, "fig2b_entanglement_heatmap.png")
    )
    plot_high_vs_low_entanglement(
        spectra,
        os.path.join(FIG_DIR, "fig2c_pairwise_comparison.png")
    )

    # Save spectra for potential Day 4 use (Phase 3 preview)
    np.save(os.path.join(BT_DIR, "spectra.npy"), spectra)
    for (ci, cj), sigma in spectra.items():
        np.save(os.path.join(BT_DIR, f"sigma_{ci}_{cj}.npy"), sigma)
    print("\n  Spectra saved to bt_matrices/")

    print("\n✓ Day 3 complete. Phase 1 preliminary results generated.")
    print("  Figures in figures/:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"    {f}")

if __name__ == "__main__":
    main()
