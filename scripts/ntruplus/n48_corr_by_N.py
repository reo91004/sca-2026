#!/usr/bin/env python3
"""F9: measured corr vs N curve, demonstrating σ_α saturation.

Uses n64_diagnose data (K=4 N=64 lane=0, 16 cases × 6 N-levels)
plus a new K=8 lane=0 N=32 evaluation for f=16 (TOP-1 case).

The formula `corr_max = b·σ_HW / sqrt((b·σ_HW)² + σ_α² + σ_w²/N)` predicts
corr → b·σ_HW / sqrt((b·σ_HW)² + σ_α²) as N → ∞ (saturates).

Plot shows:
  - 16 cases of K=4 N=64 with their N-curve (measured corr per N=2/4/8/16/32/64)
  - high-SNR case (f=16) reaches TOP-1 already at N=2
  - mid-SNR cases plateau before reaching null max ≈ 0.5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/ntruplus/phase4"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    rows = np.load(RESULTS / "n64_diagnose.npz", allow_pickle=True)["rows"]
    print(f"K=4 N=64 cases: {len(rows)}")

    # 6 N levels: assume N=2,4,8,16,32,64 (corr_n list of 6)
    N_levels = np.array([2, 4, 8, 16, 32, 64])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: corr vs N, all 16 cases
    ax = axes[0]
    abs_fc = np.array([r["abs_f_c"] for r in rows])
    snr_top = np.array([max(r["snr_n"]) for r in rows])
    cmap = plt.cm.viridis
    sort_idx = np.argsort(snr_top)[::-1]
    for ri in sort_idx[:8]:
        r = rows[ri]
        corrs = np.array(r["corr_n"], dtype=float)
        # color by max SNR
        c = cmap(min(snr_top[ri] / 0.3, 1.0))
        label = f"k={r['key']} s={r['slot']} f={r['true_f']} (|fc|={r['abs_f_c']})"
        ax.plot(N_levels, corrs, marker="o", color=c, alpha=0.7,
                 linewidth=1.5, label=label)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5,
                label="recovery threshold ≈0.5")
    ax.axhline(0.3, color="orange", linestyle=":", alpha=0.5,
                label="top-100 threshold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_levels)
    ax.set_xticklabels([str(n) for n in N_levels])
    ax.set_xlabel("N (chosen-CT traces averaged)")
    ax.set_ylabel("|corr| (CPA, V2 fixed PoI)")
    ax.set_title("F9a: corr vs N — top 8 cases by SNR\n"
                  "(K=4 N=64 lane=0, 16 cases)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    # Right: rank vs N
    ax = axes[1]
    for ri in sort_idx[:8]:
        r = rows[ri]
        ranks = np.array(r["rank_n"], dtype=float) + 1   # 1-based
        c = cmap(min(snr_top[ri] / 0.3, 1.0))
        label = f"k={r['key']} s={r['slot']} f={r['true_f']}"
        ax.plot(N_levels, ranks, marker="o", color=c, alpha=0.7,
                 linewidth=1.5, label=label)
    ax.axhline(1, color="green", linestyle="--", alpha=0.4, label="TOP-1")
    ax.axhline(10, color="orange", linestyle=":", alpha=0.4, label="top-10")
    ax.axhline(100, color="red", linestyle=":", alpha=0.4, label="top-100")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(N_levels)
    ax.set_xticklabels([str(n) for n in N_levels])
    ax.set_xlabel("N")
    ax.set_ylabel("rank (1-based)")
    ax.set_title("F9b: rank vs N — same cases\n"
                  "Most cases plateau (no recovery); some reach top-100")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("F9: corr saturation with N (σ_α-bounded)\n"
                  "N>16 yields no improvement; high-SNR cases recover at N=2",
                  y=1.02)
    fig.tight_layout()
    out = FIGS / "F9_corr_by_N.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F9 → {out}")


if __name__ == "__main__":
    main()
