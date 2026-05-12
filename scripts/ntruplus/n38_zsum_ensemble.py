#!/usr/bin/env python3
"""Z-score sum ensemble of M1 + M5 HW models.

Take Z(corr_M1) + Z(corr_M5) for each candidate, where Z subtracts the
across-candidate mean and divides by std. If M1 and M5 noise are
independent, this is the optimal linear combination (energy detector).
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

        for ki in range(K):
            x_mean = traces[ki, 0, ..., poi].mean(axis=1)
            xz = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
            score_M1 = np.abs(ch_M1z @ xz) / G
            score_M5 = np.abs(ch_M5z @ xz) / G
            # Z-score normalization across candidates
            zM1 = (score_M1 - score_M1.mean()) / (score_M1.std() + 1e-9)
            zM5 = (score_M5 - score_M5.mean()) / (score_M5.std() + 1e-9)
            score_ZS = zM1 + zM5
            score_ZE = np.sqrt(zM1 ** 2 + zM5 ** 2)  # energy
            score_MAX = np.maximum(score_M1, score_M5)
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                rk_M1 = int((score_M1 > score_M1[ti]).sum())
                rk_M5 = int((score_M5 > score_M5[ti]).sum())
                rk_MAX = int((score_MAX > score_MAX[ti]).sum())
                rk_ZS = int((score_ZS > score_ZS[ti]).sum())
                rk_ZE = int((score_ZE > score_ZE[ti]).sum())
                rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, abs_f_c=abs(fc),
                    rk_M1=rk_M1, rk_M5=rk_M5,
                    rk_MAX=rk_MAX, rk_ZS=rk_ZS, rk_ZE=rk_ZE,
                ))

    n = len(rows)
    print(f"\n=== M1+M5 ensemble strategies ({n} cases) ===\n")

    for label, key in [("M1     ", "rk_M1"), ("M5     ", "rk_M5"),
                       ("MAX    ", "rk_MAX"),
                       ("Zsum   ", "rk_ZS"),
                       ("Zenergy", "rk_ZE")]:
        rks = np.array([r[key] for r in rows])
        print(f"  V2 {label}: top-1 {(rks==0).sum()}/{n}, "
              f"top-10 {(rks<10).sum()}/{n}, "
              f"top-100 {(rks<100).sum()}/{n}, "
              f"top-500 {(rks<500).sum()}/{n}")

    # per-lane Zsum
    print(f"\nPer-lane top-100:")
    print(f"{'lane':>5} {'n':>4} {'M1':>5} {'M5':>5} {'MAX':>5} "
          f"{'Zsum':>5} {'Zen':>5}")
    for lane in sorted(set(r["lane"] for r in rows)):
        sub = [r for r in rows if r["lane"] == lane]
        n_l = len(sub)
        cells = []
        for key in ["rk_M1", "rk_M5", "rk_MAX", "rk_ZS", "rk_ZE"]:
            rks = np.array([r[key] for r in sub])
            cells.append(f"{(rks<100).sum():>2}/{n_l:>2}")
        print(f"{lane:>5} {n_l:>4} " + " ".join(cells))

    # Zsum unique top-100 (not in M1)
    print(f"\nZsum top-100 cases (showing M1 vs Zsum rank):")
    for r in sorted(rows, key=lambda r: r["rk_ZS"]):
        if r["rk_ZS"] < 100:
            print(f"  {r['batch']:>15} k={r['key']} sl={r['slot']} "
                  f"f={r['true_f']} |f_c|={r['abs_f_c']}: "
                  f"M1 rk {r['rk_M1']:>4}, M5 rk {r['rk_M5']:>4}, "
                  f"Zsum rk {r['rk_ZS']:>4}")

    out = ROOT / "results/ntruplus768/phase4/zsum_ensemble.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
