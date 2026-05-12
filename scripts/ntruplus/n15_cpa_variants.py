#!/usr/bin/env python3
"""Phase 4 — CPA variant ablations + SNR diagnostic.

Three pieces in one run:

  V1 attack-valid, fixed-PoI CPA — score(f̂) = |corr(trace[poi_pred], HW(γ·f̂))|
       Removes the per-candidate window-argmax search of n14
       (which inflates spurious peaks via multiple comparison).

  V2 attack-valid, mean-trace CPA — average trace[poi_pred] across N
       per γ, giving a length-G sample. Then |corr| against HW(γ·f̂)
       over G=12 points. Lower noise per point, fewer points.

  D1 diagnostic SNR — for each (key, lane, slot) report:
       true_f's |corr| at fixed PoI, the rank of true_f, and the 99-th
       percentile of |corr| across all 3456 candidates (null max
       proxy). Tells us whether true_f signal is above the spurious
       ceiling.

D1 uses true_f for label / rank and is therefore NOT attack-valid;
treat its output as data-side information, not as a recovery result.
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
                   default=ROOT / "results/ntruplus768/phase4/cpa_variants")
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    cands = np.arange(1, Q, dtype=np.int64)
    mat = (gammas[None, :].astype(np.int64) * cands[:, None]) % Q  # (Q-1, G)
    cand_hw = np.zeros_like(mat, dtype=np.float32)
    for gi in range(G):
        cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]

    # for V1 (per-trace) we expand to (Q-1, G·N) since N traces share γ
    cand_hw_perTrace = np.repeat(cand_hw, N, axis=1).astype(np.float32)
    cand_hw_pT_z = (cand_hw_perTrace - cand_hw_perTrace.mean(axis=1, keepdims=True)) / \
                    (cand_hw_perTrace.std(axis=1, keepdims=True) + 1e-9)
    cand_hw_g = cand_hw                                  # (Q-1, G)
    cand_hw_g_z = (cand_hw_g - cand_hw_g.mean(axis=1, keepdims=True)) / \
                  (cand_hw_g.std(axis=1, keepdims=True) + 1e-9)

    rows = []
    for ki in range(K):
        for li, lane in enumerate(lanes):
            poi = predict_poi(int(lane))

            # V1 prep: trace[poi] flattened over (G, N)
            x_trace = traces[ki, li, ..., poi].reshape(-1)   # (G·N,)
            x_z = (x_trace - x_trace.mean()) / (x_trace.std() + 1e-9)
            # all candidates corr (Q-1,) at fixed PoI:
            v1_score = np.abs(cand_hw_pT_z @ x_z) / (G * N)

            # V2 prep: mean over N
            x_mean = traces[ki, li, ..., poi].mean(axis=1)   # (G,)
            xm_z = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
            v2_score = np.abs(cand_hw_g_z @ xm_z) / G

            # D1: null distribution
            v1_99 = float(np.percentile(v1_score, 99))
            v2_99 = float(np.percentile(v2_score, 99))
            v1_max = float(v1_score.max())
            v2_max = float(v2_score.max())

            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                ti = flv - 1
                v1_true = float(v1_score[ti])
                v2_true = float(v2_score[ti])
                v1_rank = int((v1_score > v1_score[ti]).sum())
                v2_rank = int((v2_score > v2_score[ti]).sum())
                v1_pct = 100.0 * (1.0 - v1_rank / (Q - 1))
                v2_pct = 100.0 * (1.0 - v2_rank / (Q - 1))
                rows.append(dict(
                    key=ki, lane=int(lane), slot=slot, poi=poi, true_f=flv,
                    v1_true=v1_true, v1_99=v1_99, v1_max=v1_max,
                    v1_rank=v1_rank, v1_pct=v1_pct,
                    v2_true=v2_true, v2_99=v2_99, v2_max=v2_max,
                    v2_rank=v2_rank, v2_pct=v2_pct,
                ))

    print(f"\n{'k':>2} {'lane':>5} {'slot':>4} {'poi':>5} {'true_f':>7} "
          f"{'V1_T':>5} {'V1_99':>5} {'V1_M':>5} {'V1_pct':>6} "
          f"{'V2_T':>5} {'V2_99':>5} {'V2_M':>5} {'V2_pct':>6}")
    for r in rows:
        print(f"{r['key']:>2} {r['lane']:>5} {r['slot']:>4} {r['poi']:>5} "
              f"{r['true_f']:>7} {r['v1_true']:.3f} {r['v1_99']:.3f} "
              f"{r['v1_max']:.3f} {r['v1_pct']:>6.2f} "
              f"{r['v2_true']:.3f} {r['v2_99']:.3f} {r['v2_max']:.3f} "
              f"{r['v2_pct']:>6.2f}")

    v1_pcts = np.array([r["v1_pct"] for r in rows])
    v2_pcts = np.array([r["v2_pct"] for r in rows])
    v1_top1 = (np.array([r["v1_rank"] for r in rows]) == 0).mean()
    v2_top1 = (np.array([r["v2_rank"] for r in rows]) == 0).mean()
    v1_top10 = (np.array([r["v1_rank"] for r in rows]) < 10).mean()
    v2_top10 = (np.array([r["v2_rank"] for r in rows]) < 10).mean()

    # SNR ratio: how often does true |corr| beat null 99-th?
    v1_above99 = np.mean([r["v1_true"] >= r["v1_99"] for r in rows])
    v2_above99 = np.mean([r["v2_true"] >= r["v2_99"] for r in rows])

    print("\n[STAT] V1 (fixed-PoI per-trace CPA):")
    print(f"  mean pctile = {v1_pcts.mean():.2f}  top1 = {v1_top1:.3f}  "
          f"top10 = {v1_top10:.3f}")
    print(f"  D1: P(true |corr| ≥ 99-th pctile null) = {v1_above99:.3f}")
    print("[STAT] V2 (fixed-PoI mean-over-N CPA):")
    print(f"  mean pctile = {v2_pcts.mean():.2f}  top1 = {v2_top1:.3f}  "
          f"top10 = {v2_top10:.3f}")
    print(f"  D1: P(true |corr| ≥ 99-th pctile null) = {v2_above99:.3f}")

    # per-lane mean for both
    by_lane_v1 = {}
    by_lane_v2 = {}
    for r in rows:
        by_lane_v1.setdefault(r["lane"], []).append(r["v1_pct"])
        by_lane_v2.setdefault(r["lane"], []).append(r["v2_pct"])
    print("\n[STAT] per-lane mean pctile:")
    print(f"  {'lane':>5}  {'V1':>6}  {'V2':>6}  n")
    for lane in sorted(by_lane_v1.keys()):
        print(f"  {lane:>5}  {np.mean(by_lane_v1[lane]):>6.2f}  "
              f"{np.mean(by_lane_v2[lane]):>6.2f}  {len(by_lane_v1[lane])}")

    np.savez_compressed(f"{args.out_prefix}.npz",
                         rows=np.array(rows, dtype=object),
                         v1_pcts=v1_pcts, v2_pcts=v2_pcts)

    md = []
    md.append(f"# Phase 4 — CPA variants + SNR diagnostic (K={K})")
    md.append("")
    md.append(f"- traces: K={K}, L={L}, G={G}, N={N}, T={T}")
    md.append(f"- PoI fixed at predicted = 3014 + 33·iter + 16·sub")
    md.append(f"- V1: per-trace CPA at fixed PoI (no per-cand window argmax)")
    md.append(f"- V2: per-γ mean-trace CPA at fixed PoI")
    md.append(f"- D1: not attack-valid — uses true_f to compute corr")
    md.append("")
    md.append("## Attack-valid summary")
    md.append(f"- V1 mean pctile: {v1_pcts.mean():.2f}  top1: {v1_top1:.3f}  top10: {v1_top10:.3f}")
    md.append(f"- V2 mean pctile: {v2_pcts.mean():.2f}  top1: {v2_top1:.3f}  top10: {v2_top10:.3f}")
    md.append("")
    md.append("## D1 (diagnostic, NOT a recovery result)")
    md.append(f"- V1: P(true |corr| ≥ 99-th pctile null) = {v1_above99:.3f}")
    md.append(f"- V2: P(true |corr| ≥ 99-th pctile null) = {v2_above99:.3f}")
    md.append("")
    md.append("## per-lane mean pctile (V1 / V2)")
    md.append("| lane | n | V1 | V2 |")
    md.append("|---:|---:|---:|---:|")
    for lane in sorted(by_lane_v1.keys()):
        md.append(f"| {lane} | {len(by_lane_v1[lane])} | "
                  f"{np.mean(by_lane_v1[lane]):.2f} | "
                  f"{np.mean(by_lane_v2[lane]):.2f} |")
    md.append("")
    md.append("## detail rows")
    md.append("| k | lane | slot | PoI | true f | V1 |corr| | V1 99% | V1 max | "
              "V1 pctile | V2 |corr| | V2 99% | V2 max | V2 pctile |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['key']} | {r['lane']} | {r['slot']} | {r['poi']} | "
                  f"{r['true_f']} | {r['v1_true']:.3f} | {r['v1_99']:.3f} | "
                  f"{r['v1_max']:.3f} | {r['v1_pct']:.2f} | "
                  f"{r['v2_true']:.3f} | {r['v2_99']:.3f} | "
                  f"{r['v2_max']:.3f} | {r['v2_pct']:.2f} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {args.out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
