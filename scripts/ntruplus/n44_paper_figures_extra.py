#!/usr/bin/env python3
"""Additional paper figures F6-F8 (capture-free, from existing npz).

F6: N-curve for top-recovered cases (recompute from traces).
F7: corr saturation curve — measured vs formula (from b_distribution + sigma_a).
F8: dual channel separation by |f_centered| (M1, M5 corr distribution stratified).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

RESULTS = ROOT / "results/ntruplus768/phase4"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

from ntruplus.params import D, POLYBYTES, Q, QINV  # noqa: E402
from ntruplus.codec import center, from_bytes  # noqa: E402

MASK16 = (1 << 16) - 1
SIGN16 = 1 << 15

LANE_SLOT_OFFSET = {
    (0, 0): 24, (0, 1): 24, (0, 2): 24, (0, 3): 24,
    (64, 0): 12, (64, 1): 12, (64, 2): 11, (64, 3): -2,
    (80, 0): 8, (80, 1): 7, (80, 2): -3, (80, 3): -3,
    (128, 0): 0, (128, 1): -2, (128, 2): 0, (128, 3): -1,
}


def montgomery_reduce(a: int) -> int:
    a = int(a)
    t = (a * QINV) & MASK16
    if t & SIGN16:
        t -= 1 << 16
    t = (a - t * Q) >> 16
    return t & MASK16


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def fig6_ncurve():
    """N-curve for top-recovered cases."""
    cases = [
        (ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz",
         5, 0, 1, 16, "f=16 (small |f_c|, M1 channel)"),
        (ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz",
         7, 0, 3, 882, "f=882 (large |f_c|, M5+full channel)"),
        (ROOT / "traces/ntruplus768/phase3/wideg_lane64_K8N32.npz",
         1, 64, 0, 3455, "|f_c|=2 (small, full)"),
        (ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32_b.npz",
         5, 128, 2, 872, "f=872 (large, M1)"),
    ]
    Ns = [2, 4, 8, 16, 32]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2E86C1", "#E74C3C", "#28B463", "#F39C12"]

    for ax, mode, title in [
        (axes[0], "M1", "M1 baseline (predict_poi, profile-free)"),
        (axes[1], "full", "Full stack (slot+Zsum, calibrated)"),
    ]:
        for ci, (path, key, lane, slot, true_f, label) in enumerate(cases):
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=True)
            traces = z["traces"].astype(np.float32)
            sk_blobs = z["sk_blobs"]
            gammas = z["gammas"].astype(int)
            K, L, G, N_max, T = traces.shape
            cands = np.arange(1, Q, dtype=np.int64)
            prod = (gammas[None, :].astype(np.int64) * cands[:, None])
            mat_mod = prod % Q
            ch_M1 = np.zeros_like(mat_mod, dtype=np.float32)
            ch_M5 = np.zeros_like(mat_mod, dtype=np.float32)
            for gi in range(G):
                for ci_ in range(Q - 1):
                    ch_M1[ci_, gi] = bin(int(mat_mod[ci_, gi])).count("1")
                    ch_M5[ci_, gi] = bin(montgomery_reduce(int(prod[ci_, gi]))).count("1")
            ch_M1z = (ch_M1 - ch_M1.mean(axis=1, keepdims=True)) / \
                     (ch_M1.std(axis=1, keepdims=True) + 1e-9)
            ch_M5z = (ch_M5 - ch_M5.mean(axis=1, keepdims=True)) / \
                     (ch_M5.std(axis=1, keepdims=True) + 1e-9)
            slot_off = LANE_SLOT_OFFSET[(lane, slot)]
            poi_pred = predict_poi(lane)
            poi_full = poi_pred + slot_off
            ti = (true_f % Q) - 1
            ranks = []
            for n_use in Ns:
                if n_use > N_max:
                    ranks.append(np.nan)
                    continue
                if mode == "M1":
                    x = traces[key, 0, :, :n_use, poi_pred].mean(axis=1)
                    xz = (x - x.mean()) / (x.std() + 1e-9)
                    s1 = np.abs(ch_M1z @ xz) / G
                    rk = int((s1 > s1[ti]).sum())
                else:
                    x = traces[key, 0, :, :n_use, poi_full].mean(axis=1)
                    xz = (x - x.mean()) / (x.std() + 1e-9)
                    s1 = np.abs(ch_M1z @ xz) / G
                    s5 = np.abs(ch_M5z @ xz) / G
                    z1 = (s1 - s1.mean()) / (s1.std() + 1e-9)
                    z5 = (s5 - s5.mean()) / (s5.std() + 1e-9)
                    score = z1 + z5
                    rk = int((score > score[ti]).sum())
                ranks.append(rk + 1)  # 1-based for log
            ax.plot(Ns, ranks, marker="o", linewidth=2,
                    color=colors[ci], label=label)
        ax.axhline(1, color="green", linestyle="--", alpha=0.5,
                    label="TOP-1")
        ax.axhline(10, color="orange", linestyle=":", alpha=0.5,
                    label="top-10")
        ax.axhline(100, color="red", linestyle=":", alpha=0.5,
                    label="top-100")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N (chosen-CT traces per γ)")
        ax.set_ylabel("rank (1 = perfect, 1-based)")
        ax.set_xticks(Ns)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("F6: Minimum-traces N-curve for top-recovered cases\n"
                  "f=16 reaches TOP-1 at N=2 (M1); f=882 at N=16 (full stack)",
                  y=1.02)
    fig.tight_layout()
    out = FIGS / "F6_ncurve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F6 → {out}")


def fig7_corr_saturation():
    """corr_oracle vs measured corr — σ_α saturation."""
    rows = np.load(RESULTS / "b_distribution.npz", allow_pickle=True)["rows"]
    abs_fc = np.array([r["abs_f_c"] for r in rows])
    b_oracle = np.array([r["b_oracle"] for r in rows])
    sigma_HW = np.array([r["sigma_HW"] for r in rows])
    sigma_a = np.array([r["sigma_a_oracle"] for r in rows])
    corr_oracle = np.array([r["corr_oracle"] for r in rows])

    # Saturation formula: corr_max = b·σ_HW / sqrt((b·σ_HW)² + σ_α²)
    bsH = b_oracle * sigma_HW
    corr_formula = bsH / np.sqrt(bsH**2 + sigma_a**2 + 1e-18)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter b·σ_HW vs σ_α
    ax = axes[0]
    sc = ax.scatter(bsH, sigma_a, c=corr_oracle, cmap="viridis", s=18,
                     alpha=0.7, edgecolor="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("corr_oracle")
    # diagonal: bsH = sigma_a (corr = 1/√2 ≈ 0.707)
    xline = np.linspace(0, 0.005, 100)
    ax.plot(xline, xline, "k--", alpha=0.4,
             label="b·σ_HW = σ_α (corr=0.707)")
    ax.axvline(0.0027, color="red", linestyle=":", alpha=0.5,
                label="b·σ_HW threshold @ σ_α=0.0027")
    ax.set_xlabel("b · σ_HW (signal)")
    ax.set_ylabel("σ_α (saturating noise)")
    ax.set_title("Signal vs saturating noise (208 cases)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: histogram of corr_oracle, threshold lines
    ax = axes[1]
    ax.hist(corr_oracle, bins=40, color="#3498DB", alpha=0.7,
             edgecolor="black")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7,
                label=f"recovery threshold ≈0.5 ({(corr_oracle >= 0.5).sum()} cases)")
    ax.axvline(0.3, color="orange", linestyle=":", alpha=0.7,
                label=f"top-100 threshold ≈0.3 ({(corr_oracle >= 0.3).sum()} cases)")
    ax.axvline(np.median(corr_oracle), color="green", linestyle="-",
                alpha=0.7, label=f"median {np.median(corr_oracle):.2f}")
    ax.set_xlabel("corr_oracle = b·σ_HW / √((b·σ_HW)² + σ_α²)")
    ax.set_ylabel("Count")
    ax.set_title("corr_oracle distribution (saturation ceiling)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("F7: σ_α saturation mechanism (paper-grade insight)\n"
                  "Recovery requires b·σ_HW ≥ σ_α (4/208 cases at corr ≥ 0.5)",
                  y=1.02)
    fig.tight_layout()
    out = FIGS / "F7_saturation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F7 → {out}")


def fig8_dual_channel_by_fc():
    """M1 vs M5 channel split by |f_centered|."""
    rows = np.load(RESULTS / "montgomery_all.npz", allow_pickle=True)["rows"]
    M1 = np.array([r["M1_corr"] for r in rows])
    M5 = np.array([r["M5_corr"] for r in rows])
    abs_fc = np.array([r["abs_f_c"] for r in rows])
    rk_M1 = np.array([r["rk_M1"] for r in rows])
    rk_M5 = np.array([r["rk_M5"] for r in rows])

    # bins of |f_c|
    bins = [(0, 16), (16, 50), (50, 200), (200, 500), (500, 1000), (1000, 2000)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bin_centers = []
    M1_med = []; M5_med = []
    M1_p90 = []; M5_p90 = []
    counts = []
    for lo, hi in bins:
        mask = (abs_fc >= lo) & (abs_fc < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        M1_med.append(np.median(M1[mask]))
        M5_med.append(np.median(M5[mask]))
        M1_p90.append(np.percentile(M1[mask], 90))
        M5_p90.append(np.percentile(M5[mask], 90))
        counts.append(n)

    # Left: median + p90 corr by |f_c| bin
    ax = axes[0]
    x = np.arange(len(bin_centers))
    bw = 0.35
    ax.bar(x - bw/2, M1_med, bw, color="#2E86C1", alpha=0.85,
            label="M1 median")
    ax.bar(x + bw/2, M5_med, bw, color="#E74C3C", alpha=0.85,
            label="M5 median")
    ax.scatter(x - bw/2, M1_p90, marker="^", color="#1B4F72",
                edgecolor="black", linewidth=0.6, s=80, zorder=10,
                label="M1 p90")
    ax.scatter(x + bw/2, M5_p90, marker="^", color="#922B21",
                edgecolor="black", linewidth=0.6, s=80, zorder=10,
                label="M5 p90")
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{b[0]}, {b[1]})\nn={c}" for b, c in zip(bins, counts)],
                       fontsize=9)
    ax.set_ylabel("|corr|")
    ax.set_xlabel("|f_centered| bin")
    ax.set_title("Channel corr distribution by |f_c|")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4,
                label="recovery threshold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # Right: top-100 hit by |f_c| bin
    ax = axes[1]
    M1_t100 = []; M5_t100 = []; either_t100 = []
    for lo, hi in bins:
        mask = (abs_fc >= lo) & (abs_fc < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        M1_t100.append((rk_M1[mask] < 100).sum())
        M5_t100.append((rk_M5[mask] < 100).sum())
        either_t100.append(((rk_M1[mask] < 100) | (rk_M5[mask] < 100)).sum())
    ax.bar(x - bw/2, M1_t100, bw, color="#2E86C1", alpha=0.85,
            label="M1 top-100")
    ax.bar(x + bw/2, M5_t100, bw, color="#E74C3C", alpha=0.85,
            label="M5 top-100")
    # mark "either" with text
    for i, (m1, m5, both) in enumerate(zip(M1_t100, M5_t100, either_t100)):
        total = max(m1, m5)
        ax.text(i, total + 0.1, f"either={both}", ha="center", fontsize=8,
                 color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{b[0]}, {b[1]})\nn={c}" for b, c in zip(bins, counts)],
                       fontsize=9)
    ax.set_ylabel("# top-100")
    ax.set_xlabel("|f_centered| bin")
    ax.set_title("Top-100 recovery by channel and |f_c|")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("F8: Dual channel separation — M1 small-|f_c|, M5 large-|f_c|\n"
                  "Cells leak via different channels (paper finding)", y=1.02)
    fig.tight_layout()
    out = FIGS / "F8_dual_channel_fc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F8 → {out}")


def main() -> int:
    fig8_dual_channel_by_fc()  # fastest
    fig7_corr_saturation()
    fig6_ncurve()  # slow (load 4 trace files)
    print(f"\n[OK] F6-F8 figures saved to {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
