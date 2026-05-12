#!/usr/bin/env python3
"""Phase 4 — held-out attack: profile on K-1 keys, recover f_ntt of K-th.

Per-lane attack — and per-slot recovery of all 4 NTT coefficients in the
lane. A single slot=0 chosen-CT triggers the basemul computation
    r[i] = γ · f_ntt[4·lane + i]  mod q   for i ∈ {0,1,2,3}
so the trace at the basemul PoI carries information about ALL FOUR f
coefficients at once. We profile a separate ridge model per slot.

Per-lane workflow (held-out key h):
    1. Predict basemul PoI from sk-independent timing model
       PoI(k) = base + α·(k>>1) + β·(k & 1).
    2. For each slot s ∈ {0,1,2,3}:
         a. Profile a ridge: trace_window → HW(γ · f[4·lane + s] mod q).
            Train data = K-1 keys, all G·N traces.
         b. On held-out, predict per-γ HW.
         c. Score every candidate f̂ ∈ {1, …, q-1} by Σ(pred_HW −
            HW_candidate)². Report rank percentile.

Threat model: PoI is timing-only (sk-independent). sk_blobs[h] is used
only as ground truth to compute rank metrics, not as predictor input.
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


def predict_poi(lane: int, base: float, per_iter: float, per_subiter: float) -> int:
    return int(base + per_iter * (lane >> 1) + per_subiter * (lane & 1))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=ROOT / "traces/ntruplus768/phase3/multikey_hw1.npz")
    p.add_argument("--out-prefix", type=Path,
                   default=ROOT / "results/ntruplus768/phase4/recover_hw1")
    p.add_argument("--slots", type=str, default="0,1,2,3",
                   help="comma-separated slots within lane to recover. "
                        "All four slots leak at the same PoI from a single "
                        "slot=0 chosen-CT design.")
    p.add_argument("--base", type=float, default=3014)
    p.add_argument("--per-iter", type=float, default=33.0)
    p.add_argument("--per-subiter", type=float, default=16)
    p.add_argument("--window", type=int, default=10,
                   help="±half-window of samples around predicted PoI for features")
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)        # (K, L, G, N, T)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")
    print(f"[INFO] timing model: PoI = {args.base} + {args.per_iter}·iter "
          f"+ {args.per_subiter}·sub")

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    slots_to_recover = [int(s) for s in args.slots.split(",")]
    print(f"[INFO] recovering slots {slots_to_recover} per lane")

    # Pre-compute candidate HW table: (Q-1, G) where row r holds HWs of
    # (r+1)·γ_g mod q for all γ. Reused for every slot/lane/key.
    cands = np.arange(1, Q, dtype=np.int32)
    mat = (cands[:, None] * gammas[None, :].astype(np.int64)) % Q
    cand_hw = np.zeros_like(mat, dtype=np.float32)
    for gi in range(G):
        cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]

    rows = []
    for held in range(K):
        train_keys = [k for k in range(K) if k != held]
        for li, lane in enumerate(lanes):
            poi = predict_poi(int(lane), args.base, args.per_iter, args.per_subiter)
            t_lo = max(poi - args.window, 0)
            t_hi = min(poi + args.window + 1, T)
            W = t_hi - t_lo

            train_feats = traces[train_keys, li, ..., t_lo:t_hi].reshape(-1, W)
            mu = train_feats.mean(axis=0); sd = train_feats.std(axis=0) + 1e-9
            X = (train_feats - mu) / sd
            test_traces = traces[held, li, ..., t_lo:t_hi].reshape(G * N, W)
            Xte = (test_traces - mu) / sd

            for slot in slots_to_recover:
                # train labels for this (lane, slot)
                train_labs = np.zeros((len(train_keys), G), dtype=np.float32)
                for ti, ki in enumerate(train_keys):
                    flv = int(f_arr[ki, D * lane + slot])
                    for gi, g in enumerate(gammas):
                        train_labs[ti, gi] = hw_unsigned((int(g) * flv) % Q)
                ytr = np.repeat(train_labs.reshape(-1), N).astype(np.float32)
                # ridge fit
                beta = ridge_fit(X, ytr - ytr.mean(), lam=args.ridge_lambda)
                intercept = ytr.mean()
                # predict held-out
                pred_per_trace = Xte @ beta + intercept
                pred_per_g = pred_per_trace.reshape(G, N).mean(axis=1)
                # score candidates
                true_f = int(f_arr[held, D * lane + slot]) % Q
                if true_f == 0:
                    continue              # f=0 is degenerate; skip
                score = -((cand_hw - pred_per_g[None, :]) ** 2).sum(axis=1)
                true_idx = true_f - 1
                rank = int((score > score[true_idx]).sum())
                top1 = (rank == 0)
                top5 = (rank < 5)
                top10 = (rank < 10)
                top100 = (rank < 100)
                percentile = 100.0 * (1.0 - rank / (Q - 1))
                rows.append(dict(
                    held=held, lane=int(lane), slot=slot, poi=poi,
                    true_f=true_f, rank=rank, percentile=percentile,
                    top1=int(top1), top5=int(top5), top10=int(top10),
                    top100=int(top100),
                ))

    print(f"\n{'held':>5} {'lane':>5} {'slot':>4} {'poi':>5} {'true_f':>7} "
          f"{'rank':>6} {'pctile':>7} {'top1':>5} {'top5':>5} {'top10':>5} {'top100':>6}")
    for r in rows:
        print(f"{r['held']:>5} {r['lane']:>5} {r['slot']:>4} {r['poi']:>5} "
              f"{r['true_f']:>7} {r['rank']:>6} {r['percentile']:>7.2f} "
              f"{r['top1']:>5} {r['top5']:>5} {r['top10']:>5} {r['top100']:>6}")

    pcts = np.array([r["percentile"] for r in rows])
    top1s = np.array([r["top1"] for r in rows])
    top5s = np.array([r["top5"] for r in rows])
    top10s = np.array([r["top10"] for r in rows])
    top100s = np.array([r["top100"] for r in rows])
    print(f"\n[STAT] over {len(rows)} (held, lane, slot) tests:")
    print(f"[STAT] mean percentile = {pcts.mean():.2f} (random=50.00)")
    print(f"[STAT] top1 hit rate = {top1s.mean():.3f} (random≈{1/(Q-1):.5f})")
    print(f"[STAT] top5 hit rate = {top5s.mean():.3f}")
    print(f"[STAT] top10 hit rate = {top10s.mean():.3f}")
    print(f"[STAT] top100 hit rate = {top100s.mean():.3f}")
    # per-lane breakdown — which lanes' slots actually recover?
    by_lane = {}
    for r in rows:
        by_lane.setdefault(r["lane"], []).append(r["percentile"])
    print(f"\n[STAT] mean percentile per lane:")
    for lane, ps in sorted(by_lane.items()):
        print(f"  lane {lane:>4}: mean {np.mean(ps):>6.2f} (n={len(ps)})")

    np.savez_compressed(f"{args.out_prefix}.npz",
                         rows=np.array(rows, dtype=object),
                         pcts=pcts, top1s=top1s, top5s=top5s, top10s=top10s)

    md = []
    md.append(f"# Phase 4 — held-out attack (K={K}, multikey HW=1 γ)")
    md.append("")
    md.append(f"- traces: K={K}, L={L}, G={G}, N={N}, T={T}")
    md.append(f"- PoI model: {args.base} + {args.per_iter}·iter + {args.per_subiter}·sub, ±W={args.window}")
    md.append(f"- ridge λ = {args.ridge_lambda}")
    md.append(f"- slots recovered per lane: {slots_to_recover}")
    md.append("")
    md.append("## summary")
    md.append(f"- {len(rows)} (held key, lane, slot) tests")
    md.append(f"- mean percentile: {pcts.mean():.2f}  (random = 50.00)")
    md.append(f"- top1: {top1s.mean():.3f}, top5: {top5s.mean():.3f}, "
              f"top10: {top10s.mean():.3f}, top100: {top100s.mean():.3f}")
    md.append("")
    md.append("## per-lane mean percentile (across held keys × slots)")
    md.append("| lane | n | mean percentile |")
    md.append("|---:|---:|---:|")
    for lane, ps in sorted(by_lane.items()):
        md.append(f"| {lane} | {len(ps)} | {np.mean(ps):.2f} |")
    md.append("")
    md.append("## detail rows")
    md.append("| held | lane | slot | PoI | true f | rank | pctile | top1 | top5 | top10 | top100 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['held']} | {r['lane']} | {r['slot']} | {r['poi']} | "
                  f"{r['true_f']} | {r['rank']} | {r['percentile']:.2f} | "
                  f"{r['top1']} | {r['top5']} | {r['top10']} | {r['top100']} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {args.out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
