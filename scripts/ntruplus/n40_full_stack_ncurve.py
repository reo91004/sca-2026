#!/usr/bin/env python3
"""N-curve for full-stack pipeline on the new TOP-1 case (f=882) and others.

For each top-stack-recovered (key, lane, slot), subsample N ∈ {2,4,8,16,32}
and compute full-stack rank to characterize minimum N.
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
    (80, 0): 9, (80, 1): -4, (80, 2): -13, (80, 3): -15,
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
    # Cases of interest from n39 full-stack top-100
    cases = [
        # (batch_path, key, lane, slot, true_f, label)
        (ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz",
         7, 0, 3, 882, "K=8 r1 ln=0 k=7 s=3 f=882"),
        (ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz",
         5, 0, 1, 16, "K=8 r1 ln=0 k=5 s=1 f=16"),
        (ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32_b.npz",
         5, 128, 2, 872, "K=8 r2 ln=128 k=5 s=2 f=872"),
        (ROOT / "traces/ntruplus768/phase3/wideg_lane64_K8N32.npz",
         1, 64, 0, 3455, "K=8 ln=64 k=1 s=0 f=3455 (|fc|=2)"),
        (ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N64.npz",
         3, 0, 0, 1617, "K=4 N=64 k=3 s=0 f=1617"),
        (ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32_b.npz",
         7, 0, 2, 517, "K=8 r2 k=7 s=2 f=517"),
    ]

    Ns = [2, 4, 8, 16, 32]
    print(f"\n=== Full-stack N-curve for top-recovered cases ===\n")
    print(f"{'case':>40} | "
          + " | ".join(f"N={n:>2}" for n in Ns))

    for path, key, lane, slot, true_f, label in cases:
        if not path.exists():
            print(f"  [SKIP] {path}")
            continue
        z = np.load(path, allow_pickle=True)
        traces = z["traces"].astype(np.float32)
        sk_blobs = z["sk_blobs"]
        gammas = z["gammas"].astype(int)
        K, L, G, N_max, T = traces.shape
        # build label tables
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
        slot_off = LANE_SLOT_OFFSET[(lane, slot)]
        poi_use = predict_poi(lane) + slot_off
        ti = (true_f % Q) - 1
        ranks_full = []
        ranks_M1 = []
        for n_use in Ns:
            if n_use > N_max:
                ranks_full.append("-")
                ranks_M1.append("-")
                continue
            x = traces[key, 0, :, :n_use, poi_use].mean(axis=1)
            xz = (x - x.mean()) / (x.std() + 1e-9)
            s1 = np.abs(ch_M1z @ xz) / G
            s5 = np.abs(ch_M5z @ xz) / G
            z1 = (s1 - s1.mean()) / (s1.std() + 1e-9)
            z5 = (s5 - s5.mean()) / (s5.std() + 1e-9)
            score_full = z1 + z5
            rk_full = int((score_full > score_full[ti]).sum())
            rk_M1 = int((s1 > s1[ti]).sum())
            ranks_full.append(rk_full)
            ranks_M1.append(rk_M1)
        rks_str = " | ".join(f"{r:>4}" for r in ranks_full)
        print(f"{label:>40} | {rks_str}  (full-stack)")
        rks_M1_str = " | ".join(f"{r:>4}" for r in ranks_M1)
        print(f"{'  (M1 only):':>40} | {rks_M1_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
