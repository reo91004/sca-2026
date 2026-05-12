#!/usr/bin/env python3
"""V2w window-size sweep on lane=64 K=8 capture.

Hypothesis: lane=64 oracle PoI is +10 samples drift from predict_poi
(median, std=15, |max|=48). V2w window=10 misses it. Wider window catches
the peak but null max also grows. Find the best W.
"""

from __future__ import annotations

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
    z = np.load(ROOT / "traces/ntruplus768/phase3/wideg_lane64_K8N32.npz",
                 allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    lane = int(lanes[0])
    poi = predict_poi(lane)
    print(f"[INFO] lane={lane}, predict_poi={poi}")

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

    windows = [10, 15, 20, 25, 30, 40, 50, 75, 100]
    print(f"\n{'W':>4} | "
          + " | ".join(f"{lbl:>9}" for lbl in [
              "top-1", "top-10", "top-50", "top-100",
              "≥null99", "median rank", "null99 mean", "null max"
          ]))
    print("-" * 100)
    rows = []
    for W in windows:
        t_lo = max(poi - W, 0); t_hi = min(poi + W + 1, T)
        ranks = []
        nulls99 = []
        nullmax = []
        above99_cnt = 0
        for ki in range(K):
            x_mean_win = traces[ki, 0, ..., t_lo:t_hi].mean(axis=1)  # (G, W)
            xmw_z = (x_mean_win - x_mean_win.mean(axis=0, keepdims=True)) / \
                    (x_mean_win.std(axis=0, keepdims=True) + 1e-9)
            corr_w = (cand_hw_z @ xmw_z) / G  # (Q-1, W)
            score_w = np.abs(corr_w).max(axis=1)  # (Q-1,)
            null99 = float(np.percentile(score_w, 99))
            nm = float(score_w.max())
            nulls99.append(null99)
            nullmax.append(nm)
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                ti = flv - 1
                rank = int((score_w > score_w[ti]).sum())
                ranks.append(rank)
                if score_w[ti] >= null99:
                    above99_cnt += 1
        ranks = np.array(ranks)
        n = len(ranks)
        top1 = (ranks == 0).sum()
        top10 = (ranks < 10).sum()
        top50 = (ranks < 50).sum()
        top100 = (ranks < 100).sum()
        med_rank = int(np.median(ranks))
        nulls99 = np.array(nulls99)
        nullmax = np.array(nullmax)
        rows.append(dict(W=W, top1=top1, top10=top10, top50=top50,
                          top100=top100, above99=above99_cnt,
                          median_rank=med_rank,
                          null99_mean=float(nulls99.mean()),
                          null_max=float(nullmax.max())))
        print(f"{W:>4} | "
              f"{top1:>4}/{n} | {top10:>4}/{n} | {top50:>4}/{n} | "
              f"{top100:>4}/{n} | {above99_cnt:>4}/{n} | {med_rank:>11} | "
              f"{nulls99.mean():>11.3f} | {nullmax.max():>8.3f}")

    out = ROOT / "results/ntruplus768/phase4/v2w_window_sweep_lane64.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
