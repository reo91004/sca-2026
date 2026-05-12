#!/usr/bin/env python3
"""HW model variants on K=4 N=64 wide-γ.

Test if different HW labels give different b·σ_HW or σ_a behavior.
Variants:
  M1 HW(γ·f mod q)              — baseline (12-bit)
  M2 HW(γ·f mod q) − HW(γ)      — γ-baseline removed (HD approx)
  M3 HW(γ·f mod q) low byte     — 8-bit operand
  M4 HW(γ·f mod q) high 4-bit   — 4-bit MSB
  M5 HW(montgomery(γ·f))        — montgomery-reduced 16-bit
  M6 HW((γ·f mod q) ⊕ γ)        — XOR pattern
  M7 HW(low_byte(γ·f) ⊕ low_byte(γ))  — byte-level XOR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q, QINV, R  # noqa: E402

POI0 = 3014


def hw(x):
    return bin(int(x)).count("1")


def montgomery_reduce(a):
    a = int(a)
    t = (a * QINV) & ((1 << 16) - 1)
    if t & (1 << 15):
        t -= 1 << 16
    t = (a - t * Q) >> 16
    return t & ((1 << 16) - 1)


def make_labels(gammas, flv, model):
    G = len(gammas)
    out = np.zeros(G, dtype=np.float32)
    for gi, g in enumerate(gammas):
        v = (int(g) * flv) % Q
        if model == "M1":
            out[gi] = hw(v)
        elif model == "M2":
            out[gi] = hw(v) - hw(int(g))
        elif model == "M3":
            out[gi] = hw(v & 0xFF)
        elif model == "M4":
            out[gi] = hw((v >> 8) & 0xF)
        elif model == "M5":
            out[gi] = hw(montgomery_reduce(int(g) * flv))
        elif model == "M6":
            out[gi] = hw(v ^ int(g))
        elif model == "M7":
            out[gi] = hw((v & 0xFF) ^ (int(g) & 0xFF))
    return out


def main() -> int:
    z = np.load(ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N64.npz",
                 allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] K=4 N=64 lane=0 traces shape: {traces.shape}")

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    models = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    cands = np.arange(1, Q, dtype=np.int64)

    # precompute all candidate label tables for each model
    cand_hw_z_by_model = {}
    for m in models:
        print(f"  computing {m} candidate table...")
        ch = np.zeros((Q - 1, G), dtype=np.float32)
        for ci, c in enumerate(cands):
            ch[ci] = make_labels(gammas, int(c), m)
        ch_z = (ch - ch.mean(axis=1, keepdims=True)) / \
               (ch.std(axis=1, keepdims=True) + 1e-9)
        cand_hw_z_by_model[m] = ch_z

    # for each (key, slot), compute V2 corr at predict_poi for each model
    rows = []
    print(f"\n=== Per-model V2 corr at predict_poi=3014 (16 cases) ===")
    print(f"{'k':>2} {'sl':>3} {'true_f':>7} {'|f_c|':>5} | "
          + " | ".join(f"{m:>7}" for m in models))
    for ki in range(K):
        for slot in range(4):
            flv = int(f_arr[ki, slot]) % Q
            if flv == 0:
                continue
            fc = flv if flv <= Q // 2 else flv - Q
            ti = flv - 1
            x_mean = traces[ki, 0, ..., POI0].mean(axis=1)
            x_z = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
            scores = {}
            ranks = {}
            for m in models:
                score = np.abs(cand_hw_z_by_model[m] @ x_z) / G
                scores[m] = float(score[ti])
                ranks[m] = int((score > score[ti]).sum())
            rows.append(dict(key=ki, slot=slot, true_f=flv, abs_f_c=abs(fc),
                              scores=scores, ranks=ranks))
            corrs_str = " | ".join(f"{scores[m]:>7.3f}" for m in models)
            print(f"{ki:>2} {slot:>3} {flv:>7} {abs(fc):>5} | {corrs_str}")

    # summary by model
    print(f"\n=== Per-model summary ===")
    print(f"{'model':>5} | "
          + " | ".join(f"{x:>9}" for x in
                        ["med corr", "max corr",
                         "top-1", "top-10", "top-100", "top-500"]))
    for m in models:
        cs = np.array([r["scores"][m] for r in rows])
        rs = np.array([r["ranks"][m] for r in rows])
        n = len(rs)
        print(f"{m:>5} | {np.median(cs):>9.3f} | {cs.max():>9.3f} | "
              f"{(rs==0).sum():>3}/{n} | {(rs<10).sum():>3}/{n} | "
              f"{(rs<100).sum():>3}/{n} | {(rs<500).sum():>3}/{n}")

    out = ROOT / "results/ntruplus768/phase4/hw_models_n64.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
