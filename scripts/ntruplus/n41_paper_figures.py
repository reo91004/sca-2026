#!/usr/bin/env python3
"""Paper figures: 5 visualizations from existing npz results.

F1: b vs |f_c| scatter (n33) — recovery threshold dependence
F2: per-(lane, slot) drift heatmap (n34)
F3: M1 vs M5 corr scatter for top cases (n37)
F4: Recovery rate by pipeline (n39)
F5: Cross-lane SNR @ peak vs predict_poi (n25)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/ntruplus/phase4"
FIGS = ROOT / "results/ntruplus/phase4/figures"
FIGS.mkdir(parents=True, exist_ok=True)


def load(name):
    return np.load(RESULTS / name, allow_pickle=True)["rows"]


def fig1_b_vs_fc():
    """b coefficient vs |f_centered|, with recovery threshold."""
    rows = load("b_distribution.npz")
    abs_fc = np.array([r["abs_f_c"] for r in rows])
    b = np.array([r["b_oracle"] for r in rows])
    corr = np.array([r["corr_oracle"] for r in rows])
    sigma_a = np.array([r["sigma_a_oracle"] for r in rows])
    sigma_HW = np.array([r["sigma_HW"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(abs_fc, b * 1000, c=corr, s=18, cmap="viridis",
                     alpha=0.7, edgecolor="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("corr_oracle (b·σ_HW / √(...+σ_α²))")
    ax.axhline(1.8, color="red", linestyle="--", alpha=0.5,
                label="recovery threshold b≈0.0018")
    ax.set_xlabel("|f_centered|")
    ax.set_ylabel("b coefficient × 1000")
    ax.set_title(
        "b distribution vs |f_centered| (208 cases, all batches)\n"
        "small-|f_c| → high b (recovery zone)")
    ax.set_xscale("symlog", linthresh=10)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "F1_b_vs_fc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] F1 → {out}")


def fig2_per_slot_drift():
    """per-(lane, slot) median drift heatmap."""
    rows = load("per_slot_drift.npz")
    lanes = sorted(set(r["lane"] for r in rows))
    drift_grid = np.full((len(lanes), 4), np.nan)
    n_grid = np.zeros((len(lanes), 4), dtype=int)
    for li, lane in enumerate(lanes):
        for slot in range(4):
            sub = [r for r in rows if r["lane"] == lane and r["slot"] == slot]
            if sub:
                drift_grid[li, slot] = np.median([r["drift"] for r in sub])
                n_grid[li, slot] = len(sub)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(drift_grid, cmap="RdBu_r", vmin=-25, vmax=25,
                    aspect="auto")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("median drift (samples)")
    for li in range(len(lanes)):
        for slot in range(4):
            v = drift_grid[li, slot]
            if not np.isnan(v):
                ax.text(slot, li, f"{int(v):+d}\nn={n_grid[li, slot]}",
                        ha="center", va="center",
                        color="white" if abs(v) > 12 else "black",
                        fontsize=10, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"slot {i}" for i in range(4)])
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([f"lane {l}" for l in lanes])
    ax.set_title("Per-(lane, slot) oracle PoI drift\n"
                  "lane=80 의 24-sample slot spread (paper finding)")
    fig.tight_layout()
    out = FIGS / "F2_per_slot_drift.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] F2 → {out}")


def fig3_m1_vs_m5():
    """M1 vs M5 corr scatter, highlighting top cases."""
    rows = load("montgomery_all.npz")
    M1 = np.array([r["M1_corr"] for r in rows])
    M5 = np.array([r["M5_corr"] for r in rows])
    rk_M1 = np.array([r["rk_M1"] for r in rows])
    rk_M5 = np.array([r["rk_M5"] for r in rows])
    abs_fc = np.array([r["abs_f_c"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    # all cases gray
    ax.scatter(M1, M5, s=10, c="lightgray", alpha=0.5, edgecolor="none",
                label="all 208 cases")
    # top-100 in M1 only (excl M5 top-100)
    m1_only = (rk_M1 < 100) & (rk_M5 >= 100)
    ax.scatter(M1[m1_only], M5[m1_only], s=80, c="tab:blue",
                marker="o", edgecolor="black", linewidth=0.8,
                label=f"M1 top-100 only ({m1_only.sum()})")
    m5_only = (rk_M5 < 100) & (rk_M1 >= 100)
    ax.scatter(M1[m5_only], M5[m5_only], s=80, c="tab:orange",
                marker="^", edgecolor="black", linewidth=0.8,
                label=f"M5 top-100 only ({m5_only.sum()})")
    both = (rk_M1 < 100) & (rk_M5 < 100)
    ax.scatter(M1[both], M5[both], s=120, c="tab:red",
                marker="*", edgecolor="black", linewidth=0.8,
                label=f"both top-100 ({both.sum()})")
    ax.plot([0, 0.7], [0, 0.7], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("M1 |corr| = |corr(trace, HW(γ·f mod q))|")
    ax.set_ylabel("M5 |corr| = |corr(trace, HW(montgomery_reduce(γ·f)))|")
    ax.set_title(
        "M1 vs M5 channel separation (208 cases)\n"
        "Different cells leak via different channels")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "F3_m1_vs_m5.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] F3 → {out}")


def fig4_recovery_by_pipeline():
    """Recovery rate at top-1, 10, 100, 500 by pipeline."""
    rows = load("full_stack.npz")
    n = len(rows)
    pipelines = [("M1 baseline", "rk_M1_pred"),
                 ("M1 (slot-PoI)", "rk_M1_slot"),
                 ("M5 (slot-PoI)", "rk_M5_slot"),
                 ("Full stack\n(slot+Zsum)", "rk_full")]
    thresholds = [1, 10, 100, 500]
    rates = np.zeros((len(pipelines), len(thresholds)))
    for i, (_, key) in enumerate(pipelines):
        ranks = np.array([r[key] for r in rows])
        for j, t in enumerate(thresholds):
            rates[i, j] = (ranks < t).sum() / n * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pipelines))
    bw = 0.2
    colors = ["#2E86C1", "#28B463", "#F39C12", "#E74C3C"]
    for j, t in enumerate(thresholds):
        ax.bar(x + (j - 1.5) * bw, rates[:, j], bw,
                label=f"top-{t}", color=colors[j], alpha=0.85)
        for i in range(len(pipelines)):
            cnt = int(rates[i, j] * n / 100)
            ax.text(x[i] + (j - 1.5) * bw, rates[i, j] + 0.3,
                     f"{cnt}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pipelines])
    ax.set_ylabel("Recovery rate (%)")
    ax.set_title(f"Recovery rate by pipeline ({n} cases)\n"
                  "Full stack: 1 TOP-1, 2 top-10, 9 top-100, 45 top-500")
    ax.legend(title="threshold", loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGS / "F4_recovery_by_pipeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] F4 → {out}")


def fig5_lane_snr():
    """Cross-lane SNR @ predict_poi vs @ peak."""
    rows = load("lane_snr_compare.npz")
    lanes = sorted(set(r["lane"] for r in rows))
    snr_pred = {l: [] for l in lanes}
    snr_peak = {l: [] for l in lanes}
    for r in rows:
        snr_pred[r["lane"]].append(r["snr_pred"])
        snr_peak[r["lane"]].append(r["snr_peak"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    positions = np.arange(len(lanes))
    for ax, data, title in [
        (axes[0], snr_pred, "SNR @ predict_poi (baseline)"),
        (axes[1], snr_peak, "SNR @ true peak in ±100"),
    ]:
        bp = ax.boxplot([data[l] for l in lanes], positions=positions,
                          widths=0.6, patch_artist=True, showmeans=True,
                          meanprops=dict(marker="D", markerfacecolor="red",
                                          markeredgecolor="red", markersize=6))
        colors = ["#3498DB", "#16A085", "#E67E22", "#8E44AD"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"lane={l}\nn={len(data[l])}" for l in lanes])
        ax.set_title(title)
        ax.set_ylabel("SNR (b·σ_HW / σ_noise)")
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(0.28, color="red", linestyle="--", alpha=0.4,
                    label="recovery threshold ≈0.28")
        ax.legend(loc="upper right")
    fig.suptitle("Cross-lane SNR distribution\n"
                  "lane=64/80 worst @ predict_poi, comparable @ peak"
                  " (drift fixable)", y=1.02)
    fig.tight_layout()
    out = FIGS / "F5_lane_snr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F5 → {out}")


def main() -> int:
    fig1_b_vs_fc()
    fig2_per_slot_drift()
    fig3_m1_vs_m5()
    fig4_recovery_by_pipeline()
    fig5_lane_snr()
    print(f"\n[OK] all 5 figures saved to {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
