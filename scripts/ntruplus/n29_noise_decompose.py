#!/usr/bin/env python3
"""Noise decomposition: within-g (averages with N) vs across-g residual.

Hypothesis: V2 corr saturates at high N because the per-g residual contains
a γ-dependent baseline α[g] that doesn't average out. The SNR formula
corr ≈ SNR/√(SNR²+1/N) assumes white noise → overpredicts.

Test: compute σ_within = std(X[g,n] - mean_n(X[g,:])) — averages out 1/√N.
       σ_across = std(mean_n(X[g,:]) - α₀ - b·H_g) — saturates.
       True saturation: corr_∞ = (b · std(H_g)) / σ_across.
       Compare to measured V2 corr at N=64.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402

POI0 = 3014


def find_peak(traces_block, gammas, flv, poi_pred, N, window=50):
    G, _, T = traces_block.shape
    lo = max(poi_pred - window, 0); hi = min(poi_pred + window + 1, T)
    lab = np.array([bin((int(g) * flv) % Q).count("1") for g in gammas],
                    dtype=np.float32)
    lab_pT = np.repeat(lab, N).astype(np.float32)
    if lab_pT.std() < 1e-9:
        return poi_pred
    x_block = traces_block[..., lo:hi].reshape(G * N, hi - lo)
    x_z = (x_block - x_block.mean(axis=0)) / (x_block.std(axis=0) + 1e-9)
    lab_z = (lab_pT - lab_pT.mean()) / (lab_pT.std() + 1e-9)
    per_sample = (x_z * lab_z[:, None]).mean(axis=0)
    return lo + int(np.abs(per_sample).argmax())


def decompose(traces_block, gammas, flv, sample, N_use):
    """Return σ_within, σ_across, b, σ_HW for diagnostic."""
    G = traces_block.shape[0]
    sub = traces_block[:, :N_use, sample]   # (G, N_use)

    # within-g noise: sub[g, n] - mean_n(sub[g, :])
    g_mean = sub.mean(axis=1, keepdims=True)
    within = sub - g_mean
    sigma_within = float(within.std())

    # across-g baseline + signal
    sub_mean = sub.mean(axis=1)   # (G,)
    lab_g = np.array([bin((int(g) * flv) % Q).count("1") for g in gammas],
                      dtype=np.float32)
    sigma_HW = float(lab_g.std())
    Hm = lab_g - lab_g.mean()
    Xm = sub_mean - sub_mean.mean()
    if Hm.var() < 1e-9:
        return sigma_within, 0.0, 0.0, sigma_HW
    b = float((Hm * Xm).mean() / Hm.var())
    resid_across = Xm - b * Hm   # (G,)
    sigma_across = float(resid_across.std())
    return sigma_within, sigma_across, abs(b), sigma_HW


def main() -> int:
    z = np.load(ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N64.npz",
                 allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    Ns = [2, 4, 8, 16, 32, 64]

    # focus on highest-SNR case: k=2 s=0 f=2016 (SNR=0.237 from prior diag)
    print("=== Noise decomposition (top SNR case k=2 s=0 f=2016) ===")
    for k_pick, s_pick in [(2, 0), (1, 0), (0, 1), (2, 3)]:
        flv = int(f_arr[k_pick, s_pick]) % Q
        poi = find_peak(traces[k_pick, 0], gammas, flv, POI0, N, window=50)
        print(f"\nk={k_pick} slot={s_pick} f={flv} (peak PoI={poi}, "
              f"drift {poi - POI0:+d}):")
        print(f"{'N':>4} {'sig_w':>8} {'sig_a':>8} {'b':>10} {'sig_HW':>7} "
              f"{'SNR_w':>7} {'SNR_a':>7} {'corr_pred_w':>12} "
              f"{'corr_pred_a':>12}")
        for N_use in Ns:
            sw, sa, b, sH = decompose(traces[k_pick, 0], gammas, flv,
                                        poi, N_use)
            snr_w = (b * sH) / max(sw, 1e-9)
            snr_a = (b * sH) / max(sa, 1e-9)
            corr_w = snr_w / np.sqrt(snr_w**2 + 1.0/N_use)  # naive (white noise)
            corr_a = b * sH / np.sqrt((b * sH)**2 + sa**2)   # ideal (saturating)
            print(f"{N_use:>4} {sw:>8.3f} {sa:>8.4f} {b:>+10.5f} {sH:>7.4f} "
                  f"{snr_w:>7.4f} {snr_a:>7.4f} {corr_w:>12.4f} "
                  f"{corr_a:>12.4f}")

    # Aggregate analysis: for all 16 cases, compute saturation prediction
    # vs measured corr at N=64
    print("\n=== Aggregate saturation prediction vs measured V2 corr (N=64) ===")
    print(f"{'k':>2} {'sl':>3} {'true_f':>7} {'b':>10} {'sig_w':>8} "
          f"{'sig_a':>8} {'pred_64':>8} {'measured':>9} {'ratio':>6}")
    rows = []

    cands = np.arange(1, Q, dtype=np.int64)
    mat = (gammas[None, :].astype(np.int64) * cands[:, None]) % Q
    cand_hw = np.zeros_like(mat, dtype=np.float32)
    for gi in range(G):
        cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]
    cand_hw_z = (cand_hw - cand_hw.mean(axis=1, keepdims=True)) / \
                (cand_hw.std(axis=1, keepdims=True) + 1e-9)

    for ki in range(K):
        for slot in range(4):
            flv = int(f_arr[ki, D * 0 + slot]) % Q
            if flv == 0:
                continue
            sw, sa, b, sH = decompose(traces[ki, 0], gammas, flv, POI0, N)
            # ideal-noise saturation (using sigma_a as effective noise)
            corr_pred = (b * sH) / np.sqrt((b * sH)**2 + sa**2)
            # measured V2 corr at predicted PoI (formula matches n23)
            x_mean = traces[ki, 0, ..., POI0].mean(axis=1)
            xm_z = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
            score = np.abs(cand_hw_z @ xm_z) / G
            measured = float(score[flv - 1])
            ratio = measured / max(corr_pred, 1e-9)
            rows.append(dict(k=ki, slot=slot, true_f=flv,
                              b=b, sig_w=sw, sig_a=sa,
                              pred=corr_pred, measured=measured, ratio=ratio))
            print(f"{ki:>2} {slot:>3} {flv:>7} {b:>+10.5f} {sw:>8.3f} "
                  f"{sa:>8.4f} {corr_pred:>8.4f} {measured:>9.4f} "
                  f"{ratio:>6.3f}")

    ratios = np.array([r["ratio"] for r in rows])
    preds = np.array([r["pred"] for r in rows])
    print(f"\n  ratio (measured/predicted_a) over 16 cases: "
          f"median={np.median(ratios):.3f}, mean={ratios.mean():.3f}")
    print(f"  predicted_a (sat ceiling using sig_a) range: "
          f"{preds.min():.3f}–{preds.max():.3f}, median={np.median(preds):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
