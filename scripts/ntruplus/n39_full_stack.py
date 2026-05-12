#!/usr/bin/env python3
"""Full attack stack: per-(lane, slot) PoI + M1+M5 Zsum ensemble.

Combines:
  - per-(lane, slot) median drift offset (n34 oracle calibration)
  - M1 (HW(γ·f mod q)) + M5 (HW(montgomery_reduce(γ·f))) Z-sum (n38)

For each (key, slot), use slot-specific PoI = predict_poi + offset(lane, slot).
At that PoI, compute M1 & M5 V2 corr scores. Z-normalize per model, sum.
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

LANE_SLOT_OFFSET = {
    (0, 0): 24, (0, 1): 24, (0, 2): 24, (0, 3): 24,
    (64, 0): 12, (64, 1): 12, (64, 2): 11, (64, 3): -2,
    (80, 0): 8, (80, 1): 7, (80, 2): -3, (80, 3): -3,
    (128, 0): 0, (128, 1): -2, (128, 2): 0, (128, 3): -1,
}


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
        ("K=8 ln=80", ROOT / "traces/ntruplus768/phase3/wideg_lane80_K8N32.npz"),
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
        poi_pred = predict_poi(lane)
        f_arr = np.zeros((K, 768), dtype=np.int16)
        for ki in range(K):
            f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

        cands = np.arange(1, Q, dtype=np.int64)
        prod = (gammas[None, :].astype(np.int64) * cands[:, None])
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

        # Pre-compute scores at unique slot offsets (limited set per lane)
        offsets = sorted(set(LANE_SLOT_OFFSET[(lane, s)] for s in range(4)))
        score_cache = {}  # (key_idx, offset) -> (score_M1, score_M5)
        for off in offsets:
            poi_use = poi_pred + off
            for ki in range(K):
                x = traces[ki, 0, ..., poi_use].mean(axis=1)
                xz = (x - x.mean()) / (x.std() + 1e-9)
                s1 = np.abs(ch_M1z @ xz) / G
                s5 = np.abs(ch_M5z @ xz) / G
                score_cache[(ki, off)] = (s1, s5)
        # Also pre-compute at predict_poi for baseline
        baseline_cache = {}
        for ki in range(K):
            x = traces[ki, 0, ..., poi_pred].mean(axis=1)
            xz = (x - x.mean()) / (x.std() + 1e-9)
            s1 = np.abs(ch_M1z @ xz) / G
            s5 = np.abs(ch_M5z @ xz) / G
            baseline_cache[ki] = (s1, s5)

        for ki in range(K):
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                # baseline: M1 at predict_poi
                s1_b, s5_b = baseline_cache[ki]
                rk_M1_pred = int((s1_b > s1_b[ti]).sum())
                # full stack: M1+M5 Zsum at slot-specific PoI
                slot_off = LANE_SLOT_OFFSET[(lane, slot)]
                s1, s5 = score_cache[(ki, slot_off)]
                z1 = (s1 - s1.mean()) / (s1.std() + 1e-9)
                z5 = (s5 - s5.mean()) / (s5.std() + 1e-9)
                score_full = z1 + z5
                score_M1_only = s1
                score_M5_only = s5
                rk_M1 = int((s1 > s1[ti]).sum())
                rk_M5 = int((s5 > s5[ti]).sum())
                rk_full = int((score_full > score_full[ti]).sum())
                rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, abs_f_c=abs(fc),
                    rk_M1_pred=rk_M1_pred,   # original baseline
                    rk_M1_slot=rk_M1,         # slot-PoI M1
                    rk_M5_slot=rk_M5,         # slot-PoI M5
                    rk_full=rk_full,           # slot-PoI Zsum
                ))

    n = len(rows)
    print(f"\n=== Full stack: per-slot PoI + Zsum ({n} cases) ===\n")

    for label, key in [("M1 baseline (predict_poi)", "rk_M1_pred"),
                       ("M1 (slot-PoI)            ", "rk_M1_slot"),
                       ("M5 (slot-PoI)            ", "rk_M5_slot"),
                       ("Full stack (slot+Zsum)   ", "rk_full")]:
        rks = np.array([r[key] for r in rows])
        print(f"  {label}: top-1 {(rks==0).sum()}/{n}, "
              f"top-10 {(rks<10).sum()}/{n}, "
              f"top-100 {(rks<100).sum()}/{n}, "
              f"top-500 {(rks<500).sum()}/{n}")

    # Per-lane
    print(f"\nPer-lane top-100:")
    print(f"{'lane':>5} {'n':>4} {'M1 pred':>8} {'full':>5}")
    for lane in sorted(set(r["lane"] for r in rows)):
        sub = [r for r in rows if r["lane"] == lane]
        rps = np.array([r["rk_M1_pred"] for r in sub])
        rfs = np.array([r["rk_full"] for r in sub])
        n_l = len(sub)
        print(f"{lane:>5} {n_l:>4} {(rps<100).sum():>3}/{n_l:>2}      "
              f"{(rfs<100).sum():>3}/{n_l:>2}")

    # Full-stack top-100 cases
    print(f"\nFull-stack top-100 cases:")
    print(f"{'batch':>15} {'k':>3} {'sl':>3} {'f':>5} {'|f_c|':>5} "
          f"{'M1 pred':>8} {'M1 sl':>6} {'M5 sl':>6} {'full':>5}")
    for r in sorted(rows, key=lambda r: r["rk_full"]):
        if r["rk_full"] < 100:
            print(f"{r['batch']:>15} {r['key']:>3} {r['slot']:>3} "
                  f"{r['true_f']:>5} {r['abs_f_c']:>5} "
                  f"{r['rk_M1_pred']:>8} {r['rk_M1_slot']:>6} "
                  f"{r['rk_M5_slot']:>6} {r['rk_full']:>5}")

    out = ROOT / "results/ntruplus768/phase4/full_stack.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
