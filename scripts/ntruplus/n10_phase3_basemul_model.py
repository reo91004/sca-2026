#!/usr/bin/env python3
"""Linear basemul-iter timing model + per-lane signal at predicted PoI.

From v3 (HW=1 γ) we observed three PASS lanes at samples that match
basemul iteration timing: lane index k → iter (k>>1) at sample
S + 33.2 · iter. Use this to *predict* every lane's PoI window, then
measure |corr| there with a small ±W window-mean.

If lanes that previously FAILED show signal at the predicted window, the
"failure" was noise-peak masking — not absence of leakage. That distinction
matters for held-out attack design.
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
    p.add_argument("--input", type=Path,
                   default=ROOT / "traces/ntruplus768/phase3/scout_hw1.npz")
    p.add_argument("--out-prefix", type=Path,
                   default=ROOT / "results/ntruplus768/phase3/basemul_model")
    p.add_argument("--slot", type=int, default=0)
    p.add_argument("--base", type=int, default=3014)
    p.add_argument("--per-iter", type=float, default=33.2)
    p.add_argument("--per-subiter", type=int, default=16,
                   help="extra offset for the second basemul of a pair")
    p.add_argument("--window", type=int, default=10,
                   help="±half-window of samples to average around predicted PoI")
    p.add_argument("--n-shuffles", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026_05_08)
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)        # (L, G, N, T)
    sk = bytes(z["sk_blob"])
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    L, G, N, T = traces.shape
    f_ntt = center(from_bytes(sk[:POLYBYTES]))

    rng = np.random.default_rng(args.seed)

    print(f"[INFO] traces={traces.shape}  using S={args.base}, "
          f"per_iter={args.per_iter}, per_subiter={args.per_subiter}, ±W={args.window}")
    print(f"\n{'lane':>5} {'iter':>5} {'sub':>4} {'pred_PoI':>9} {'real':>6} "
          f"{'null μ':>7} {'null σ':>7} {'z':>6} verdict")

    rows = []
    for li, lane in enumerate(lanes):
        iter_idx = lane >> 1
        sub = lane & 1
        pred_poi = int(args.base + args.per_iter * iter_idx + args.per_subiter * sub)
        t_lo = max(pred_poi - args.window, 0)
        t_hi = min(pred_poi + args.window + 1, T)

        # max |corr| across the predicted window — preserves narrow signal
        flv = int(f_ntt[D * lane + args.slot])
        labA = np.array([hw_unsigned((int(g) * flv) % Q) for g in gammas],
                        dtype=np.float32)
        flat_lab = np.repeat(labA, N).astype(np.float32)
        if flat_lab.std() < 1e-6:
            continue
        # per-sample corr in window
        win_traces = traces[li, ..., t_lo:t_hi].reshape(G * N, t_hi - t_lo)
        tz = (win_traces - win_traces.mean(axis=0)) / (win_traces.std(axis=0) + 1e-12)
        lab_z = (flat_lab - flat_lab.mean()) / (flat_lab.std() + 1e-12)
        per_sample = (tz * lab_z[:, None]).mean(axis=0)
        argmax_local = int(np.abs(per_sample).argmax())
        real = float(per_sample[argmax_local])
        actual_t = t_lo + argmax_local

        # null: shuffle labels for the same in-window samples (same max-over-window)
        nulls = np.zeros(args.n_shuffles, dtype=np.float32)
        for s in range(args.n_shuffles):
            sh = flat_lab.copy(); rng.shuffle(sh)
            sh_z = (sh - sh.mean()) / (sh.std() + 1e-12)
            nulls[s] = float(np.abs((tz * sh_z[:, None]).mean(axis=0)).max())
        nm, ns = float(nulls.mean()), float(nulls.std())
        zsc = (abs(real) - nm) / max(ns, 1e-9)
        verdict = "PASS" if zsc > 3 else ("WEAK" if zsc > 1.5 else "FAIL")
        print(f"{lane:>5} {iter_idx:>5} {sub:>4} {pred_poi:>9} actual={actual_t:>5} "
              f"{real:>+6.3f} {nm:>7.4f} {ns:>7.4f} {zsc:>6.2f}  {verdict}")
        rows.append(dict(lane=int(lane), iter=int(iter_idx), sub=int(sub),
                          pred_poi=pred_poi, real=real, null_mean=nm,
                          null_std=ns, z=zsc, verdict=verdict))

    # If most lanes show consistent positive z (above 1.5) at the predicted
    # PoI, basemul is the universal leak source.
    zs = np.array([r["z"] for r in rows])
    pos = (zs > 1.5).sum()
    print(f"\n[STAT] {pos}/{len(rows)} lanes show z > 1.5 at predicted PoI window")

    np.savez_compressed(f"{args.out_prefix}.npz", rows=np.array(rows, dtype=object))
    md = []
    md.append("# Phase 3 — basemul-iter timing model evaluation")
    md.append("")
    md.append(f"- traces: {traces.shape}, slot={args.slot}, ±W={args.window}")
    md.append(f"- predicted PoI = {args.base} + {args.per_iter}·(lane>>1) "
              f"+ {args.per_subiter}·(lane & 1)")
    md.append("")
    md.append("| lane | iter | sub | pred PoI | real | null μ | null σ | z | verdict |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        md.append(f"| {r['lane']} | {r['iter']} | {r['sub']} | {r['pred_poi']} | "
                  f"{r['real']:+.3f} | {r['null_mean']:.4f} | "
                  f"{r['null_std']:.4f} | {r['z']:.2f} | {r['verdict']} |")
    md.append("")
    md.append(f"## summary")
    md.append(f"- {pos}/{len(rows)} lanes show z > 1.5 at predicted PoI window.")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {args.out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
