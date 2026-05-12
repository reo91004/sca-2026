#!/usr/bin/env python3
"""Phase 4 — attack-valid CPA on wide-γ capture (HW=1 + HW=2, G=78).

For each (key, lane, slot ∈ {0,1,2,3}) within a single victim session,
compute V2 (mean-over-N) CPA score per candidate f̂ ∈ [1, q-1] using
predicted PoI = 3014 + 33·iter + 16·sub. Report rank, percentile,
top-k.

Also runs the same diagnostic SNR estimate (n16-style) on this dataset
to show the attack-valid-vs-structural-ceiling story is consistent.
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
                   default=ROOT / "traces/ntruplus768/phase3/wideg_lane0_n32.npz")
    p.add_argument("--out-prefix", type=Path,
                   default=ROOT / "results/ntruplus/phase4/wideg_lane0_attack")
    p.add_argument("--window", type=int, default=10)
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)        # (K, L, G, N, T)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")
    print(f"[INFO] G={G} (HW=1 + HW=2)  N={N}")
    null_max_pred = float(np.sqrt(2 * np.log(Q - 1) / G))
    print(f"[INFO] structural null-max ceiling: ≈ {null_max_pred:.3f}")

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    cands = np.arange(1, Q, dtype=np.int64)
    mat = (gammas[None, :].astype(np.int64) * cands[:, None]) % Q  # (Q-1, G)
    cand_hw = np.zeros_like(mat, dtype=np.float32)
    for gi in range(G):
        cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]
    cand_hw_z = (cand_hw - cand_hw.mean(axis=1, keepdims=True)) / \
                (cand_hw.std(axis=1, keepdims=True) + 1e-9)

    rows = []
    for ki in range(K):
        for li, lane in enumerate(lanes):
            poi_pred = predict_poi(int(lane))

            # ---- attack-valid V2: mean over N at fixed PoI ----
            x_mean_fixed = traces[ki, li, ..., poi_pred].mean(axis=1)   # (G,)
            xm_z = (x_mean_fixed - x_mean_fixed.mean()) / (x_mean_fixed.std() + 1e-9)
            score_v2 = np.abs(cand_hw_z @ xm_z) / G                     # (Q-1,)
            null99_v2 = float(np.percentile(score_v2, 99))
            null_max_v2 = float(score_v2.max())

            # ---- attack-valid V2w: mean over N + per-cand best in ±W ----
            t_lo = max(poi_pred - args.window, 0)
            t_hi = min(poi_pred + args.window + 1, T)
            ww = t_hi - t_lo
            x_mean_win = traces[ki, li, ..., t_lo:t_hi].mean(axis=1)    # (G, W)
            xmw_mean = x_mean_win.mean(axis=0, keepdims=True)
            xmw_std = x_mean_win.std(axis=0, keepdims=True) + 1e-9
            xmw_z = (x_mean_win - xmw_mean) / xmw_std                    # (G, W)
            corr_w = (cand_hw_z @ xmw_z) / G                             # (Q-1, W)
            score_v2w = np.abs(corr_w).max(axis=1)
            null99_v2w = float(np.percentile(score_v2w, 99))

            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                ti = flv - 1
                # V2 (fixed PoI)
                v2_true = float(score_v2[ti])
                v2_rank = int((score_v2 > v2_true).sum())
                v2_pct = 100.0 * (1.0 - v2_rank / (Q - 1))
                # V2w (window)
                v2w_true = float(score_v2w[ti])
                v2w_rank = int((score_v2w > v2w_true).sum())
                v2w_pct = 100.0 * (1.0 - v2w_rank / (Q - 1))
                rows.append(dict(
                    key=ki, lane=int(lane), slot=slot, poi=poi_pred,
                    true_f=flv,
                    v2_true=v2_true, v2_null99=null99_v2,
                    v2_max=null_max_v2,
                    v2_rank=v2_rank, v2_pct=v2_pct,
                    v2w_true=v2w_true, v2w_null99=null99_v2w,
                    v2w_rank=v2w_rank, v2w_pct=v2w_pct,
                ))

    print(f"\n{'k':>2} {'lane':>5} {'slot':>4} {'poi':>5} {'true_f':>6} "
          f"{'V2_T':>5} {'V2_99':>5} {'V2_M':>5} {'V2_pct':>6} "
          f"{'V2w_T':>5} {'V2w_99':>6} {'V2w_pct':>7}")
    for r in rows:
        print(f"{r['key']:>2} {r['lane']:>5} {r['slot']:>4} {r['poi']:>5} "
              f"{r['true_f']:>6} {r['v2_true']:.3f} {r['v2_null99']:.3f} "
              f"{r['v2_max']:.3f} {r['v2_pct']:>6.2f} "
              f"{r['v2w_true']:.3f} {r['v2w_null99']:.3f} {r['v2w_pct']:>7.2f}")

    v2_pcts = np.array([r["v2_pct"] for r in rows])
    v2w_pcts = np.array([r["v2w_pct"] for r in rows])
    v2_top1 = (np.array([r["v2_rank"] for r in rows]) == 0).mean()
    v2_top10 = (np.array([r["v2_rank"] for r in rows]) < 10).mean()
    v2_top100 = (np.array([r["v2_rank"] for r in rows]) < 100).mean()
    v2w_top1 = (np.array([r["v2w_rank"] for r in rows]) == 0).mean()
    v2w_top10 = (np.array([r["v2w_rank"] for r in rows]) < 10).mean()
    v2w_top100 = (np.array([r["v2w_rank"] for r in rows]) < 100).mean()
    v2_above99 = np.mean([r["v2_true"] >= r["v2_null99"] for r in rows])
    v2w_above99 = np.mean([r["v2w_true"] >= r["v2w_null99"] for r in rows])

    print(f"\n[STAT] over {len(rows)} (key, lane, slot) tests:")
    print(f"  V2  (fixed PoI):  pctile mean = {v2_pcts.mean():.2f}, "
          f"top1 = {v2_top1:.3f}, top10 = {v2_top10:.3f}, "
          f"top100 = {v2_top100:.3f}")
    print(f"  V2w (±W={args.window}):    pctile mean = {v2w_pcts.mean():.2f}, "
          f"top1 = {v2w_top1:.3f}, top10 = {v2w_top10:.3f}, "
          f"top100 = {v2w_top100:.3f}")
    print(f"  P(V2 true |corr| ≥ null99): {v2_above99:.3f}")
    print(f"  P(V2w true |corr| ≥ null99): {v2w_above99:.3f}")

    np.savez_compressed(f"{args.out_prefix}.npz",
                         rows=np.array(rows, dtype=object),
                         v2_pcts=v2_pcts, v2w_pcts=v2w_pcts)

    md = []
    md.append(f"# Phase 4 — wide-γ CPA attack (G={G}, K={K}, lane={lanes[0]})")
    md.append("")
    md.append(f"- traces: K={K}, L={L}, G={G} (HW=1+HW=2), N={N}, T={T}")
    md.append(f"- structural null-max at G={G}: ≈ {null_max_pred:.3f}")
    md.append(f"- V2: mean-over-N at fixed predicted PoI, |corr|")
    md.append(f"- V2w: mean-over-N at fixed PoI ±W={args.window}, max|corr|")
    md.append("")
    md.append("## summary (attack-valid)")
    md.append(f"- {len(rows)} (key, lane, slot) tests")
    md.append(f"- V2: mean pctile = {v2_pcts.mean():.2f}, "
              f"top1 = {v2_top1:.3f}, top10 = {v2_top10:.3f}, "
              f"top100 = {v2_top100:.3f}")
    md.append(f"- V2w: mean pctile = {v2w_pcts.mean():.2f}, "
              f"top1 = {v2w_top1:.3f}, top10 = {v2w_top10:.3f}, "
              f"top100 = {v2w_top100:.3f}")
    md.append(f"- P(V2 true |corr| ≥ null 99-th): {v2_above99:.3f}")
    md.append(f"- P(V2w true |corr| ≥ null 99-th): {v2w_above99:.3f}")
    md.append("")
    md.append("## detail rows")
    md.append("| k | lane | slot | PoI | true f | V2 |corr| | V2 99% | V2 max | "
              "V2 pctile | V2w |corr| | V2w 99% | V2w pctile |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['key']} | {r['lane']} | {r['slot']} | {r['poi']} | "
                  f"{r['true_f']} | {r['v2_true']:.3f} | {r['v2_null99']:.3f} | "
                  f"{r['v2_max']:.3f} | {r['v2_pct']:.2f} | "
                  f"{r['v2w_true']:.3f} | {r['v2w_null99']:.3f} | "
                  f"{r['v2w_pct']:.2f} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {args.out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
