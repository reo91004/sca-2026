#!/usr/bin/env python3
"""Profiled PoI attack: apply lane-specific median drift offset.

The lane→time mapping is sk-independent (same firmware, same hardware).
Per n25 diagnostic, the median oracle drift per lane is:
  lane=0   +24
  lane=64  +10
  lane=80  -12
  lane=128 -2

This is "calibration once, apply forever" — a standard SCA practice.
Use lane-specific offset on predict_poi → V2 CPA.

Compare:
  V2_pred — predict_poi(lane) (current baseline)
  V2_prof — predict_poi(lane) + lane_offset (profiled)

Reports: top-1, top-10, top-100 across all batches per lane.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402


# Profiled offsets from n25 oracle median drift
LANE_OFFSET = {0: 24, 64: 10, 80: -12, 128: -2}


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def profiled_poi(lane: int) -> int:
    return predict_poi(lane) + LANE_OFFSET.get(lane, 0)


def main() -> int:
    batches = [
        ("K=4 r1 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N32.npz"),
        ("K=4 N=64 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N64.npz"),
        ("K=8 r1 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz"),
        ("K=8 r2 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32_b.npz"),
        ("K=8 ln=64", ROOT / "traces/ntruplus768/phase3/wideg_lane64_K8N32.npz"),
        ("K=4 ln=80", ROOT / "traces/ntruplus768/phase3/wideg_lane80_K4N32.npz"),
        ("K=8 r1 ln=128", ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32.npz"),
        ("K=8 r2 ln=128", ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32_b.npz"),
    ]

    all_rows = []
    for batch_name, path in batches:
        if not path.exists():
            print(f"[SKIP] {path}")
            continue
        z = np.load(path, allow_pickle=True)
        traces = z["traces"].astype(np.float32)
        sk_blobs = z["sk_blobs"]
        gammas = z["gammas"].astype(int)
        ln = z["lanes"].astype(int)
        K, L, G, N, T = traces.shape
        lane = int(ln[0])

        f_arr = np.zeros((K, 768), dtype=np.int16)
        for ki in range(K):
            f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

        cands = np.arange(1, Q, dtype=np.int64)
        mat = (gammas[None, :].astype(np.int64) * cands[:, None]) % Q
        cand_hw = np.zeros_like(mat, dtype=np.float32)
        for gi in range(G):
            cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]
        cand_hw_z = (cand_hw - cand_hw.mean(axis=1, keepdims=True)) / \
                    (cand_hw.std(axis=1, keepdims=True) + 1e-9)

        poi_pred = predict_poi(lane)
        poi_prof = profiled_poi(lane)

        for ki in range(K):
            x_pred = traces[ki, 0, ..., poi_pred].mean(axis=1)
            x_prof = traces[ki, 0, ..., poi_prof].mean(axis=1)
            xp_z = (x_pred - x_pred.mean()) / (x_pred.std() + 1e-9)
            xq_z = (x_prof - x_prof.mean()) / (x_prof.std() + 1e-9)
            score_pred = np.abs(cand_hw_z @ xp_z) / G
            score_prof = np.abs(cand_hw_z @ xq_z) / G
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                v2p_T = float(score_pred[ti])
                v2q_T = float(score_prof[ti])
                rk_p = int((score_pred > v2p_T).sum())
                rk_q = int((score_prof > v2q_T).sum())
                all_rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, abs_f_c=abs(fc),
                    poi_pred=poi_pred, poi_prof=poi_prof,
                    v2_pred_T=v2p_T, v2_pred_rank=rk_p,
                    v2_prof_T=v2q_T, v2_prof_rank=rk_q,
                ))

    n = len(all_rows)
    print(f"\n=== Profiled PoI vs predict_poi ({n} cases) ===\n")

    # overall
    rks_p = np.array([r["v2_pred_rank"] for r in all_rows])
    rks_q = np.array([r["v2_prof_rank"] for r in all_rows])
    print(f"Overall recovery:")
    print(f"  V2 predict_poi : top-1 {(rks_p==0).sum()}/{n} ({(rks_p==0).mean()*100:.2f}%), "
          f"top-10 {(rks_p<10).sum()}/{n}, top-100 {(rks_p<100).sum()}/{n}, "
          f"top-500 {(rks_p<500).sum()}/{n}")
    print(f"  V2 profiled    : top-1 {(rks_q==0).sum()}/{n} ({(rks_q==0).mean()*100:.2f}%), "
          f"top-10 {(rks_q<10).sum()}/{n}, top-100 {(rks_q<100).sum()}/{n}, "
          f"top-500 {(rks_q<500).sum()}/{n}")

    # per-lane breakdown
    print(f"\nPer-lane recovery (top-100):")
    print(f"{'lane':>5} {'n':>5} {'predict':>9} {'profiled':>9} {'gain':>5}")
    for ln in sorted(set(r["lane"] for r in all_rows)):
        rows = [r for r in all_rows if r["lane"] == ln]
        rks_p_l = np.array([r["v2_pred_rank"] for r in rows])
        rks_q_l = np.array([r["v2_prof_rank"] for r in rows])
        n_l = len(rows)
        t100_p = (rks_p_l < 100).sum()
        t100_q = (rks_q_l < 100).sum()
        gain = t100_q - t100_p
        print(f"{ln:>5} {n_l:>5} {t100_p:>3}/{n_l:>3}    "
              f"{t100_q:>3}/{n_l:>3}   {gain:>+3d}")

    # detail: cases that flipped (gained or lost rank)
    print(f"\nCases gained (rank_prof < rank_pred and now top-500):")
    for r in sorted(all_rows, key=lambda r: r["v2_prof_rank"]):
        if r["v2_prof_rank"] < r["v2_pred_rank"] and r["v2_prof_rank"] < 500:
            print(f"  {r['batch']:>15} lane={r['lane']} k={r['key']} sl={r['slot']} "
                  f"f={r['true_f']} |f_c|={r['abs_f_c']}: rk "
                  f"{r['v2_pred_rank']} → {r['v2_prof_rank']}")

    out = ROOT / "results/ntruplus768/phase4/profiled_poi.npz"
    np.savez_compressed(out, rows=np.array(all_rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
