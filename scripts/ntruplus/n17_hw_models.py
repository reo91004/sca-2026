#!/usr/bin/env python3
"""HW model ablation — same multikey data, 4 candidate label functions.

For each (key, lane, slot, predicted PoI) compute SNR (signal vs noise
of OLS fit) under different label functions:

  M1 unsigned    HW(γ·f mod q)               in [0, q-1]
  M2 centered    HW(centered(γ·f mod q))     16-bit two's complement,
                                              v in [-q/2, q/2]
  M3 montgomery  HW(montgomery_reduce(γ·f))  16-bit two's complement,
                                              v in (-q, q)
  M4 nibble HW   sum of 4-bit nibble HWs of unsigned γ·f mod q
  M5 byte HW     sum of byte HWs (low-byte vs high-byte separately
                                  reported as M5a, M5b)

Diagnostic only — uses true f to compute label/SNR; not an attack-valid
recovery. Goal: which HW model is the best leak predictor for this
target firmware?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q, QINV  # noqa: E402


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def hw_u(v: int) -> int:
    return bin(int(v) & 0xFFFF).count("1")


def hw_centered(v: int) -> int:
    """v is in [0, q-1]; return HW of its centered 16-bit two's compl rep."""
    if v > Q // 2:
        v -= Q
    return bin(int(v) & 0xFFFF).count("1")


def montgomery_reduce(a: int) -> int:
    """Standard 16-bit Montgomery reduce: out in (-q, q)."""
    u = (a * QINV) & 0xFFFF
    if u > 0x7FFF:
        u -= 0x10000
    t = a - u * Q
    out = t >> 16
    return out


def hw_montgomery(gf_unsigned: int, gamma: int) -> int:
    """gf_unsigned = γ·f mod q. We want HW(montgomery_reduce(γ·f))
    where γ·f is the 32-bit product (not yet reduced)."""
    # We need original γ·f integer, not mod q.
    raise NotImplementedError("compute via montgomery_reduce(gamma * f)")


def hw_nibble(v: int) -> int:
    """Sum of 4-bit nibble HWs (= HW unsigned, structurally same as M1
    but flag as separate model in case nibble-level ALU lookup leaks)."""
    return hw_u(v)  # numerically identical to popcount, kept for clarity


def hw_low_byte(v: int) -> int:
    return bin(int(v) & 0xFF).count("1")


def hw_high_byte(v: int) -> int:
    return bin((int(v) >> 8) & 0xFF).count("1")


def main() -> int:
    z = np.load(ROOT / "traces/ntruplus768/phase3/multikey_hw1.npz",
                allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    label_funcs = {
        "M1_unsigned": lambda gf, gamma, fv: hw_u(gf),
        "M2_centered": lambda gf, gamma, fv: hw_centered(gf),
        "M3_montgomery": lambda gf, gamma, fv: bin(montgomery_reduce(int(gamma) * int(fv)) & 0xFFFF).count("1"),
        "M4_low_byte": lambda gf, gamma, fv: hw_low_byte(gf),
        "M5_high_byte": lambda gf, gamma, fv: hw_high_byte(gf),
    }

    print(f"\n{'model':>14}  {'mean SNR':>9}  {'max SNR':>8}  "
          f"{'P(|c|>null99)':>13}")

    summary = {}
    for name, lf in label_funcs.items():
        snrs = []
        true_above_null99 = []
        for ki in range(K):
            for li, lane in enumerate(lanes):
                poi = predict_poi(int(lane))
                x_all = traces[ki, li, ..., poi].reshape(-1)   # (G·N,)

                # null reference: per-candidate corr distribution at this poi
                cands = np.arange(1, Q, dtype=np.int64)
                # build per-candidate label using same lf — for unsigned-style
                # M1/M4/M5, label depends only on (γ·c mod q). For M3 (montgomery)
                # label depends on actual γ·c. M2 like M1.
                cand_lab = np.zeros((Q - 1, G), dtype=np.float32)
                for gi, g in enumerate(gammas):
                    for ci, c in enumerate(cands):
                        gf = (int(g) * int(c)) % Q
                        cand_lab[ci, gi] = lf(gf, int(g), int(c))
                cand_lab_pT = np.repeat(cand_lab, N, axis=1)
                # standardize each candidate row
                cand_z = (cand_lab_pT - cand_lab_pT.mean(axis=1, keepdims=True)) / \
                         (cand_lab_pT.std(axis=1, keepdims=True) + 1e-9)
                x_z = (x_all - x_all.mean()) / (x_all.std() + 1e-9)
                all_corr = np.abs(cand_z @ x_z) / (G * N)
                null99 = float(np.percentile(all_corr, 99))

                for slot in range(4):
                    flv = int(f_arr[ki, D * lane + slot]) % Q
                    if flv == 0:
                        continue
                    # label per (γ, n)
                    lab_g = np.array([lf((int(g) * flv) % Q, int(g), flv)
                                       for g in gammas], dtype=np.float32)
                    lab_pT = np.repeat(lab_g, N).astype(np.float32)
                    Hm = lab_pT - lab_pT.mean()
                    if Hm.var() < 1e-9:
                        continue
                    Xm = x_all - x_all.mean()
                    b = float((Hm * Xm).mean() / Hm.var())
                    sigma_sig = abs(b) * float(lab_g.std())
                    resid = x_all - (lab_pT.mean() + b * lab_pT)
                    sigma_noi = float(resid.std())
                    snr = sigma_sig / max(sigma_noi, 1e-12)
                    snrs.append(snr)
                    true_idx = flv - 1
                    true_corr = float(all_corr[true_idx])
                    true_above_null99.append(true_corr >= null99)

        snr_arr = np.array(snrs)
        above = np.array(true_above_null99)
        summary[name] = (snr_arr, above)
        print(f"{name:>14}  {snr_arr.mean():>9.4f}  {snr_arr.max():>8.4f}  "
              f"{above.mean():>13.3f}")

    # Save summary
    out = ROOT / "results/ntruplus768/phase4/hw_models.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: v[0] for k, v in summary.items()})
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
