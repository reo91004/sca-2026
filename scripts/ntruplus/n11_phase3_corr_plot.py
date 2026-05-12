#!/usr/bin/env python3
"""Plot per-lane |corr| profile around predicted basemul PoI for v3 data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402


def main() -> int:
    z = np.load(ROOT / "traces/ntruplus768/phase3/scout_hw1.npz",
                allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk = bytes(z["sk_blob"])
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    L, G, N, T = traces.shape
    f_ntt = center(from_bytes(sk[:POLYBYTES]))

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=False)
    for li, ax in zip(range(L), axes.flat):
        flat = traces[li].reshape(G * N, T)
        flv = int(f_ntt[D * int(lanes[li]) + 0])
        lab = np.array([bin((int(g) * flv) % Q).count("1") for g in gammas],
                        np.float32)
        flat_lab = np.repeat(lab, N).astype(np.float32)
        if flat_lab.std() < 1e-6:
            continue
        tz = (flat - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-12)
        lz = (flat_lab - flat_lab.mean()) / (flat_lab.std() + 1e-12)
        corr_t = np.abs((tz * lz[:, None]).mean(axis=0))

        pred = int(3014 + 33 * (int(lanes[li]) >> 1) + 16 * (int(lanes[li]) & 1))
        # Plot ±100 around predicted PoI
        lo = max(pred - 100, 0); hi = min(pred + 101, T)
        ax.plot(np.arange(lo, hi), corr_t[lo:hi], lw=0.6)
        ax.axvline(pred, color="red", lw=0.6, alpha=0.7,
                   label=f"pred {pred}")
        argmax = int(corr_t.argmax())
        ax.axvline(argmax, color="green", lw=0.6, alpha=0.7,
                   label=f"actual {argmax}, |c|={corr_t[argmax]:.3f}")
        ax.set_title(f"lane {int(lanes[li])} (iter {int(lanes[li])>>1}, "
                     f"sub {int(lanes[li])&1}), label var "
                     f"{float(lab.var()):.2f}", fontsize=9)
        ax.set_xlabel("sample"); ax.set_ylabel("|corr|")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Phase 3 v3 — |corr(trace, HW(γ·f mod q))| around predicted basemul PoI",
                 fontsize=11)
    fig.tight_layout()
    out = ROOT / "results/ntruplus768/phase3/scout_hw1_corr_profile.png"
    fig.savefig(out, dpi=130)
    print(f"[OK] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
