#!/usr/bin/env python3
"""Phase 4 — profile-free CPA per victim session (single-key candidate
enumeration).

Standard chosen-CT CPA: inside one victim session (one key) we have
N traces per (lane, slot, γ). For every candidate f̂ ∈ {1, …, q-1}, we
predict label vector HW((γ·f̂) mod q) and compute |corr| with the trace
sample at the predicted basemul PoI. The candidate with maximum |corr|
is the recovered f̂.

Threat model:
  - Trace inputs only (no µ′, no sk for prediction).
  - PoI is sk-independent timing model: PoI = 3014 + 33·iter + 16·sub.
  - Per-candidate scoring uses |corr| within ±W of predicted PoI: each
    candidate picks its own best PoI sample. This absorbs small per-key
    PoI drift while staying sk-indep (no sk-driven sample selection).
  - sk_blobs is consulted only at scoring time to compute rank.

Per (key, lane, slot) we report rank / percentile / top-k.
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


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=ROOT / "traces/ntruplus768/phase3/multikey_hw1.npz")
    p.add_argument("--out-prefix", type=Path,
                   default=ROOT / "results/ntruplus/phase4/cpa_per_key")
    p.add_argument("--window", type=int, default=10,
                   help="±half-window around predicted PoI for per-candidate "
                        "best-sample search")
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)        # (K, L, G, N, T)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")
    print(f"[INFO] PoI search ±W = {args.window} samples around prediction")

    # ground truth f
    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    # Pre-compute candidate HW table:
    #   cand_hw[c, g] = HW( (gammas[g] * (c+1)) mod q )   for c=0..Q-2
    cands = np.arange(1, Q, dtype=np.int64)
    mat = (gammas[None, :].astype(np.int64) * cands[:, None]) % Q  # (Q-1, G)
    cand_hw = np.zeros_like(mat, dtype=np.float32)
    for gi in range(G):
        cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]
    # per-trace expand: (Q-1, G·N) — same hw for each of the N replicates
    cand_hw_expanded = np.repeat(cand_hw, N, axis=1).astype(np.float32)
    # standardize per candidate (axis=1 across G·N)
    cand_mean = cand_hw_expanded.mean(axis=1, keepdims=True)
    cand_std = cand_hw_expanded.std(axis=1, keepdims=True) + 1e-9
    cand_z = (cand_hw_expanded - cand_mean) / cand_std

    rows = []
    for ki in range(K):
        for li, lane in enumerate(lanes):
            poi = predict_poi(int(lane))
            t_lo = max(poi - args.window, 0)
            t_hi = min(poi + args.window + 1, T)
            W = t_hi - t_lo

            # sample-level traces in window: (G·N, W)
            X = traces[ki, li, ..., t_lo:t_hi].reshape(G * N, W)
            # standardize each PoI-sample column
            X_mean = X.mean(axis=0, keepdims=True)
            X_std = X.std(axis=0, keepdims=True) + 1e-9
            X_z = (X - X_mean) / X_std       # (G·N, W)

            # Pearson corr at each PoI sample for every candidate:
            #   ρ[c, w] = sum_n (cand_z[c, n] · X_z[n, w]) / (G·N)
            #   |ρ| 의 sample-축 max → per-candidate best-PoI score.
            # cand_z (Q-1, G·N) @ X_z (G·N, W) = (Q-1, W)
            corr = (cand_z @ X_z) / (G * N)
            score = np.abs(corr).max(axis=1)        # (Q-1,)
            best_poi_per_cand = corr.argmax(axis=1) # only for diagnostics

            for slot in range(4):
                # f-value at this lane/slot
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                # CPA score for THIS slot uses true_g labels =
                # HW(γ·f mod q) for slot. But slots share traces — only
                # the candidate enumeration (cand_hw) and rank are
                # computed against the true f for THIS slot. trace and
                # cand_hw are identical across slots; what changes is
                # which cand index is the "true" one.
                # Rebuild slot-specific candidate score by re-using the
                # same `score` array — score is invariant of slot.
                # Just look up the rank of cand_idx = flv-1.
                true_idx = flv - 1
                rank = int((score > score[true_idx]).sum())
                pct = 100.0 * (1.0 - rank / (Q - 1))
                top1 = int(rank == 0); top5 = int(rank < 5)
                top10 = int(rank < 10); top100 = int(rank < 100)
                argmax_cand = int(np.argmax(score))
                rows.append(dict(
                    key=ki, lane=int(lane), slot=slot, poi_pred=poi,
                    poi_actual=t_lo + int(best_poi_per_cand[true_idx]),
                    true_f=flv, argmax_f=argmax_cand + 1,
                    score_true=float(score[true_idx]),
                    score_argmax=float(score[argmax_cand]),
                    rank=rank, percentile=pct,
                    top1=top1, top5=top5, top10=top10, top100=top100,
                ))

    # ------------------------------------------------------------------
    # Per-slot caveat: slot 0..3 ALL share the same trace samples, but
    # the basemul writes 4 different output coefficients sequentially.
    # The CPA above uses the entire window — which is sample-global, so
    # the |corr| profile inside the window will peak at slot-specific
    # samples (slot 0 ≈ poi, slot 1 ≈ poi+δ1, etc.) for different f.
    # Since per-cand argmax over the window is used as score, all slots
    # in the same lane will have the same `score` vector — but their
    # `true_idx` differs, so each slot gets its own rank.
    # ------------------------------------------------------------------

    # Print summary
    print(f"\n{'k':>2} {'lane':>5} {'slot':>4} {'pPoI':>5} {'aPoI':>5} "
          f"{'true_f':>7} {'arg_f':>6} {'sT':>5} {'sA':>5} {'rank':>5} "
          f"{'pct':>6} {'t1':>3} {'t5':>3} {'t10':>4} {'t100':>4}")
    for r in rows:
        print(f"{r['key']:>2} {r['lane']:>5} {r['slot']:>4} {r['poi_pred']:>5} "
              f"{r['poi_actual']:>5} {r['true_f']:>7} {r['argmax_f']:>6} "
              f"{r['score_true']:.3f} {r['score_argmax']:.3f} {r['rank']:>5} "
              f"{r['percentile']:>6.2f} {r['top1']:>3} {r['top5']:>3} "
              f"{r['top10']:>4} {r['top100']:>4}")

    pcts = np.array([r["percentile"] for r in rows])
    top1s = np.array([r["top1"] for r in rows])
    top5s = np.array([r["top5"] for r in rows])
    top10s = np.array([r["top10"] for r in rows])
    top100s = np.array([r["top100"] for r in rows])
    print(f"\n[STAT] over {len(rows)} (key, lane, slot) tests:")
    print(f"[STAT] mean percentile = {pcts.mean():.2f} (random=50.00)")
    print(f"[STAT] top1 = {top1s.mean():.3f} (random≈{1/(Q-1):.5f})")
    print(f"[STAT] top5 = {top5s.mean():.3f}, top10 = {top10s.mean():.3f}, "
          f"top100 = {top100s.mean():.3f}")

    # per-lane breakdown
    by_lane = {}
    for r in rows:
        by_lane.setdefault(r["lane"], []).append(r["percentile"])
    print(f"\n[STAT] mean percentile per lane:")
    for lane, ps in sorted(by_lane.items()):
        print(f"  lane {lane:>4}: mean {np.mean(ps):>6.2f}  (n={len(ps)})")

    # per-slot breakdown
    by_slot = {}
    for r in rows:
        by_slot.setdefault(r["slot"], []).append(r["percentile"])
    print(f"\n[STAT] mean percentile per slot:")
    for slot, ps in sorted(by_slot.items()):
        print(f"  slot {slot}: mean {np.mean(ps):>6.2f}  (n={len(ps)})")

    np.savez_compressed(f"{args.out_prefix}.npz",
                         rows=np.array(rows, dtype=object),
                         pcts=pcts, top1s=top1s, top5s=top5s,
                         top10s=top10s, top100s=top100s)

    md = []
    md.append(f"# Phase 4 — profile-free CPA per victim session (K={K})")
    md.append("")
    md.append(f"- traces: K={K}, L={L}, G={G}, N={N}, T={T}")
    md.append(f"- PoI prediction: 3014 + 33·iter + 16·sub, ±W={args.window}")
    md.append(f"- score(f̂) = max_{{poi ∈ ±W}} |corr(trace[poi], HW(γ·f̂ mod q))|")
    md.append("")
    md.append("## summary")
    md.append(f"- {len(rows)} (key, lane, slot) tests")
    md.append(f"- mean percentile: {pcts.mean():.2f} (random=50.00)")
    md.append(f"- top1: {top1s.mean():.3f}, top5: {top5s.mean():.3f}, "
              f"top10: {top10s.mean():.3f}, top100: {top100s.mean():.3f}")
    md.append("")
    md.append("## per-lane mean percentile")
    md.append("| lane | n | mean pctile |")
    md.append("|---:|---:|---:|")
    for lane, ps in sorted(by_lane.items()):
        md.append(f"| {lane} | {len(ps)} | {np.mean(ps):.2f} |")
    md.append("")
    md.append("## per-slot mean percentile")
    md.append("| slot | n | mean pctile |")
    md.append("|---:|---:|---:|")
    for slot, ps in sorted(by_slot.items()):
        md.append(f"| {slot} | {len(ps)} | {np.mean(ps):.2f} |")
    md.append("")
    md.append("## detail rows")
    md.append("| k | lane | slot | pPoI | aPoI | true f | arg f | "
              "score_true | score_argmax | rank | pctile | t1 | t5 | t10 | t100 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['key']} | {r['lane']} | {r['slot']} | "
                  f"{r['poi_pred']} | {r['poi_actual']} | {r['true_f']} | "
                  f"{r['argmax_f']} | {r['score_true']:.3f} | "
                  f"{r['score_argmax']:.3f} | {r['rank']} | "
                  f"{r['percentile']:.2f} | {r['top1']} | {r['top5']} | "
                  f"{r['top10']} | {r['top100']} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {args.out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
