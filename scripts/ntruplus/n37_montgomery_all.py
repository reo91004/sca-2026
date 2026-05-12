#!/usr/bin/env python3
"""Apply M5 (Montgomery-reduced HW) to all batches + M1+M5 ensemble.

n36 found M5 = HW(montgomery_reduce(γ·f)) — the EXACT operand model
of basemul `r[0] = montgomery_reduce(c[0]*f[0] - ...)` — recovers a NEW
top-100 case in K=4 N=64 that M1 (HW(γ·f mod q)) missed. Apply M5 to all
batches + combine with M1 (max ensemble).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q, QINV  # noqa: E402

MASK16 = (1 << 16) - 1
SIGN16 = 1 << 15


def montgomery_reduce(a):
    a = int(a)
    t = (a * QINV) & MASK16
    if t & SIGN16:
        t -= 1 << 16
    t = (a - t * Q) >> 16
    return t & MASK16


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


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

    rows = []
    for batch_name, path in batches:
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        traces = z["traces"].astype(np.float32)
        sk_blobs = z["sk_blobs"]
        gammas = z["gammas"].astype(int)
        ln = z["lanes"].astype(int)
        K, L, G, N, T = traces.shape
        lane = int(ln[0])
        poi = predict_poi(lane)
        f_arr = np.zeros((K, 768), dtype=np.int16)
        for ki in range(K):
            f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

        cands = np.arange(1, Q, dtype=np.int64)
        # M1 table
        prod = (gammas[None, :].astype(np.int64) * cands[:, None])  # (Q-1, G)
        mat_mod = prod % Q
        ch_M1 = np.zeros_like(mat_mod, dtype=np.float32)
        ch_M5 = np.zeros_like(mat_mod, dtype=np.float32)
        for gi in range(G):
            for ci in range(Q - 1):
                ch_M1[ci, gi] = bin(int(mat_mod[ci, gi])).count("1")
                ch_M5[ci, gi] = bin(montgomery_reduce(int(prod[ci, gi]))).count("1")
        ch_M1z = (ch_M1 - ch_M1.mean(axis=1, keepdims=True)) / \
                 (ch_M1.std(axis=1, keepdims=True) + 1e-9)
        ch_M5z = (ch_M5 - ch_M5.mean(axis=1, keepdims=True)) / \
                 (ch_M5.std(axis=1, keepdims=True) + 1e-9)

        for ki in range(K):
            x_mean = traces[ki, 0, ..., poi].mean(axis=1)
            xz = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
            score_M1 = np.abs(ch_M1z @ xz) / G
            score_M5 = np.abs(ch_M5z @ xz) / G
            score_E = np.maximum(score_M1, score_M5)
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                rk_M1 = int((score_M1 > score_M1[ti]).sum())
                rk_M5 = int((score_M5 > score_M5[ti]).sum())
                rk_E = int((score_E > score_E[ti]).sum())
                rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, abs_f_c=abs(fc),
                    M1_corr=float(score_M1[ti]),
                    M5_corr=float(score_M5[ti]),
                    rk_M1=rk_M1, rk_M5=rk_M5, rk_E=rk_E,
                ))
        print(f"  done {batch_name}")

    n = len(rows)
    print(f"\n=== Montgomery HW model on all batches ({n} cases) ===\n")

    rks_M1 = np.array([r["rk_M1"] for r in rows])
    rks_M5 = np.array([r["rk_M5"] for r in rows])
    rks_E = np.array([r["rk_E"] for r in rows])
    cs_M1 = np.array([r["M1_corr"] for r in rows])
    cs_M5 = np.array([r["M5_corr"] for r in rows])

    print(f"  V2 M1   : top-1 {(rks_M1==0).sum()}/{n}, "
          f"top-10 {(rks_M1<10).sum()}/{n}, top-100 {(rks_M1<100).sum()}/{n}, "
          f"top-500 {(rks_M1<500).sum()}/{n}")
    print(f"  V2 M5   : top-1 {(rks_M5==0).sum()}/{n}, "
          f"top-10 {(rks_M5<10).sum()}/{n}, top-100 {(rks_M5<100).sum()}/{n}, "
          f"top-500 {(rks_M5<500).sum()}/{n}")
    print(f"  V2 ens  : top-1 {(rks_E==0).sum()}/{n}, "
          f"top-10 {(rks_E<10).sum()}/{n}, top-100 {(rks_E<100).sum()}/{n}, "
          f"top-500 {(rks_E<500).sum()}/{n}")
    print(f"  M5 max corr: {cs_M5.max():.3f}, M1 max: {cs_M1.max():.3f}")

    # per-lane
    print(f"\nPer-lane top-100:")
    print(f"{'lane':>5} {'n':>4} {'M1':>5} {'M5':>5} {'ens':>5}")
    for lane in sorted(set(r["lane"] for r in rows)):
        sub = [r for r in rows if r["lane"] == lane]
        rps = np.array([r["rk_M1"] for r in sub])
        rqs = np.array([r["rk_M5"] for r in sub])
        res = np.array([r["rk_E"] for r in sub])
        n_l = len(sub)
        print(f"{lane:>5} {n_l:>4} {(rps<100).sum():>2}/{n_l:>2} "
              f"{(rqs<100).sum():>3}/{n_l:>2} "
              f"{(res<100).sum():>3}/{n_l:>2}")

    # cases where M5 finds a new top-100 (not in M1)
    print(f"\nM5-only top-100 (not in M1 top-100):")
    for r in rows:
        if r["rk_M5"] < 100 and r["rk_M1"] >= 100:
            print(f"  {r['batch']:>15} k={r['key']} sl={r['slot']} "
                  f"f={r['true_f']} |f_c|={r['abs_f_c']}: "
                  f"M1 rk {r['rk_M1']} → M5 rk {r['rk_M5']} "
                  f"(M1 corr {r['M1_corr']:.3f}, M5 corr {r['M5_corr']:.3f})")

    # cases where M5 BEATS M1 by significant margin in top-500
    print(f"\nM5 large gain (M5 rk - M1 rk ≤ -500, both ≤ 500):")
    for r in rows:
        if r["rk_M5"] < 500 and r["rk_M1"] - r["rk_M5"] >= 500:
            print(f"  {r['batch']:>15} k={r['key']} sl={r['slot']} "
                  f"f={r['true_f']} |f_c|={r['abs_f_c']}: "
                  f"M1 rk {r['rk_M1']} → M5 rk {r['rk_M5']}")

    out = ROOT / "results/ntruplus/phase4/montgomery_all.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
