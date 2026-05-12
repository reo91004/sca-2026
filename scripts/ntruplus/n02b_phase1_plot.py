#!/usr/bin/env python3
"""Plot Phase 1 SNR + max|t| over time + region histogram."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    res = np.load(ROOT / "results/ntruplus768/phase1/d_map.npz")
    snr = res["snr"]
    t   = res["max_t"]
    null = res["null_max_t"]
    T = len(snr)
    p99 = np.percentile(null, 99.9)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(snr, lw=0.4)
    axes[0].set_ylabel("cross-key SNR")
    axes[0].set_title(f"NTRU+768 Phase 1 — natural D map (K=8, N=20, T={T})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, lw=0.4, color="tab:orange")
    axes[1].axhline(p99, color="grey", lw=0.6, ls="--",
                    label=f"null p99.9 = {p99:.2f}")
    axes[1].set_ylabel("pairwise max |t|")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    # density of leaky samples (max|t| > 10) — 100 bins across the trace
    bins = 100
    edges = np.linspace(0, T, bins + 1).astype(int)
    counts = np.zeros(bins)
    leaky = np.where(t > 10)[0]
    for x in leaky:
        b = int(np.searchsorted(edges, x, side="right") - 1)
        if 0 <= b < bins:
            counts[b] += 1
    axes[2].bar(edges[:-1], counts, width=T / bins, color="tab:red", alpha=0.7)
    axes[2].set_ylabel("# samples max|t|>10\n(per 1% bin)")
    axes[2].set_xlabel("sample index")
    axes[2].grid(alpha=0.3)

    out = ROOT / "results/ntruplus768/phase1/d_map_overview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"[OK] saved {out}")

    # detect hot regions
    print("\n[REGIONS] top 10 sample windows by SNR (window=200):")
    win = 200
    smooth = np.convolve(snr, np.ones(win) / win, mode="same")
    idx = np.argsort(smooth)[-30:][::-1]
    seen = set()
    out_regions = []
    for i in idx:
        b = int(i) // win
        if b in seen:
            continue
        seen.add(b)
        out_regions.append((int(i), float(smooth[i])))
        if len(out_regions) >= 10:
            break
    for i, v in out_regions:
        print(f"  sample {i:5d}  smoothed SNR = {v:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
