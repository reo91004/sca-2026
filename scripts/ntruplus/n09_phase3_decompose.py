#!/usr/bin/env python3
"""Per-sample decomposition: at each t, measure HW(γ)-corr vs HW(γ·f)-corr.

If γ-byte leakage and γ·f leakage live in different time regions, we can
identify "clean f" samples — those where HW(γ·f mod q) corr is high while
HW(γ) corr is low.

For each lane separately:
    A_t = corr(trace[t], HW(γ·f mod q))      (f-dependent, our target)
    B_t = corr(trace[t], HW(γ) of unsigned 12-bit)  (γ-byte leakage)
    Δ_t = |A_t| − |B_t|

A sample with positive Δ and high |A| is a clean f-leak point. Conversely,
high |A| with even higher |B| is just γ-byte echo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402


def hw_unsigned(v: int) -> int:
    return bin(int(v) & 0xFFFF).count("1")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--slot", type=int, default=0)
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)        # (L, G, N, T)
    sk = bytes(z["sk_blob"])
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    L, G, N, T = traces.shape

    f_ntt = center(from_bytes(sk[:POLYBYTES]))

    print(f"[INFO] traces shape={traces.shape}  L={L}  G={G}")

    # Pre-compute label vectors: for each (l, g), label_A = HW(γ·f mod q),
    # label_B = HW(γ unsigned 12-bit). Both are per-design constants
    # (replicated N times when flat).
    labA = np.zeros((L, G), dtype=np.float32)
    labB = np.zeros((L, G), dtype=np.float32)
    for li, lane in enumerate(lanes):
        flv = int(f_ntt[D * lane + args.slot])
        for gi, g in enumerate(gammas):
            labA[li, gi] = hw_unsigned((int(g) * flv) % Q)
            labB[li, gi] = hw_unsigned(int(g) % Q)

    # per-cell corr at each sample for each lane (separately for A and B)
    corr_A = np.zeros((L, T), dtype=np.float32)
    corr_B = np.zeros((L, T), dtype=np.float32)
    for li in range(L):
        flat_tr = traces[li].reshape(G * N, T)
        labA_flat = np.repeat(labA[li], N).astype(np.float32)
        labB_flat = np.repeat(labB[li], N).astype(np.float32)
        if labA_flat.std() < 1e-6 or labB_flat.std() < 1e-6:
            continue
        tz = (flat_tr - flat_tr.mean(axis=0)) / (flat_tr.std(axis=0) + 1e-12)
        corr_A[li] = (((labA_flat - labA_flat.mean()) / labA_flat.std())[:, None]
                      * tz).mean(axis=0)
        corr_B[li] = (((labB_flat - labB_flat.mean()) / labB_flat.std())[:, None]
                      * tz).mean(axis=0)

    aA = np.abs(corr_A); aB = np.abs(corr_B)
    delta = aA - aB

    # Per-lane cleanest sample by Δ
    print(f"\n{'lane':>5} {'best_Δ':>8} {'@t':>5} {'|A|':>6} {'|B|':>6} "
          f"{'argmax|A|':>11} {'argmax|B|':>10}")
    for li in range(L):
        if aA[li].max() < 1e-6:
            continue
        ti = int(delta[li].argmax())
        print(f"{lanes[li]:>5} {delta[li, ti]:>8.4f} {ti:>5d} "
              f"{aA[li, ti]:>6.3f} {aB[li, ti]:>6.3f} "
              f"{int(aA[li].argmax()):>11d} {int(aB[li].argmax()):>10d}")

    # global: pool across lanes — count samples where Δ > 0.1 AND |A| > 0.4
    clean_mask = (delta > 0.10) & (aA > 0.40)
    counts_per_lane = clean_mask.sum(axis=1)
    print(f"\n[STAT] clean f-leak samples (Δ>0.10, |A|>0.40) per lane: "
          f"{counts_per_lane.tolist()}")
    print(f"[STAT] total across all lanes: {int(clean_mask.sum())}  "
          f"(of {L * T} cells)")

    np.savez_compressed(f"{args.out_prefix}.npz",
                         corr_A=corr_A, corr_B=corr_B,
                         delta=delta, clean_mask=clean_mask,
                         labA=labA, labB=labB,
                         lanes=lanes, gammas=gammas)

    md = []
    md.append(f"# Phase 3 sample-decomposition")
    md.append("")
    md.append(f"- traces: L={L}, G={G}, N={N}, T={T}")
    md.append(f"- HW(γ·f mod q) (`A`) vs HW(γ unsigned 12-bit) (`B`).")
    md.append(f"- Δ = |A| − |B|. Clean f-leak: Δ > 0.10 AND |A| > 0.40.")
    md.append("")
    md.append("## per-lane best Δ sample")
    md.append("| lane | bestΔ | @sample | \\|A\\| | \\|B\\| | argmax\\|A\\| | argmax\\|B\\| |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for li in range(L):
        if aA[li].max() < 1e-6:
            continue
        ti = int(delta[li].argmax())
        md.append(f"| {lanes[li]} | {delta[li, ti]:.4f} | {ti} | "
                  f"{aA[li, ti]:.3f} | {aB[li, ti]:.3f} | "
                  f"{int(aA[li].argmax())} | {int(aB[li].argmax())} |")
    md.append("")
    md.append(f"## clean-leak summary")
    md.append(f"- clean f-leak samples per lane: {counts_per_lane.tolist()}")
    md.append(f"- total clean cells: {int(clean_mask.sum())} / {L*T}")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {args.out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
