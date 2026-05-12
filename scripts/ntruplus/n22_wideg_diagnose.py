#!/usr/bin/env python3
"""Diagnostic on wide-γ data — SNR + empirical null distribution analysis.

Three pieces:

  D1 SNR per slot at lane 0 (uses true f → diagnostic only).
  D2 empirical null max distribution: among 3456 candidates, what's the
     mean / 99% / max |corr|? Compare to Gaussian-iid prediction
     √(2·log(Q-1)/G) ≈ 0.456.
  D3 candidate-label correlations: pairwise corr among candidate label
     vectors (sample-and-test, ~100 random pairs). If labels are highly
     correlated, effective G < 78 and null max should be larger.
  D4 per-sample SNR profile around predicted PoI (where is the leak
     actually peaking?).
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
    z = np.load(ROOT / "traces/ntruplus768/phase3/wideg_lane0_n32.npz",
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

    # =================================================================
    # D1 — SNR estimate per slot
    # =================================================================
    print("\n=== D1: per-slot SNR at predicted PoI ===")
    print(f"{'slot':>4} {'true_f':>7} {'lab_var':>8} {'b':>10} "
          f"{'sig_sig':>8} {'sig_noi':>8} {'SNR':>6}")
    snrs = []
    poi = predict_poi(int(lanes[0]))
    x_all = traces[0, 0, ..., poi].reshape(-1)   # (G·N,)
    for slot in range(4):
        flv = int(f_arr[0, D * lanes[0] + slot]) % Q
        lab_g = np.array([bin((int(g) * flv) % Q).count("1")
                           for g in gammas], dtype=np.float32)
        lab_var = float(lab_g.var())
        lab_pT = np.repeat(lab_g, N).astype(np.float32)
        Hm = lab_pT - lab_pT.mean()
        Xm = x_all - x_all.mean()
        b = float((Hm * Xm).mean() / Hm.var()) if Hm.var() > 1e-9 else 0.0
        sigma_sig = abs(b) * float(lab_g.std())
        resid = x_all - (lab_pT.mean() + b * lab_pT)
        sigma_noi = float(resid.std())
        snr = sigma_sig / max(sigma_noi, 1e-12)
        snrs.append(snr)
        print(f"{slot:>4} {flv:>7} {lab_var:>8.3f} {b:>+10.5f} "
              f"{sigma_sig:>8.5f} {sigma_noi:>8.5f} {snr:>6.3f}")

    # =================================================================
    # D2 — empirical null max distribution at predicted PoI
    # =================================================================
    print("\n=== D2: empirical null distribution (V2 mean-trace) ===")
    cands = np.arange(1, Q, dtype=np.int64)
    mat = (gammas[None, :].astype(np.int64) * cands[:, None]) % Q  # (Q-1, G)
    cand_hw = np.zeros_like(mat, dtype=np.float32)
    for gi in range(G):
        cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]
    cand_z = (cand_hw - cand_hw.mean(axis=1, keepdims=True)) / \
             (cand_hw.std(axis=1, keepdims=True) + 1e-9)

    # mean trace over N at predicted PoI: (G,)
    x_mean = traces[0, 0, ..., poi].mean(axis=1)
    xm_z = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
    score = np.abs(cand_z @ xm_z) / G

    null_max_predicted = float(np.sqrt(2 * np.log(Q - 1) / G))
    print(f"  empirical mean    = {score.mean():.3f}")
    print(f"  empirical median  = {np.median(score):.3f}")
    print(f"  empirical 95%     = {np.percentile(score, 95):.3f}")
    print(f"  empirical 99%     = {np.percentile(score, 99):.3f}")
    print(f"  empirical 99.9%   = {np.percentile(score, 99.9):.3f}")
    print(f"  empirical max     = {score.max():.3f}")
    print(f"  predicted null max (G={G}, iid): {null_max_predicted:.3f}")
    # rank of true f
    for slot in range(4):
        flv = int(f_arr[0, D * lanes[0] + slot]) % Q
        ti = flv - 1
        rank = int((score > score[ti]).sum())
        pct = 100.0 * (1.0 - rank / (Q - 1))
        print(f"  slot {slot}: true_f={flv}  |corr|={score[ti]:.3f}  "
              f"rank={rank}  pctile={pct:.1f}")

    # =================================================================
    # D3 — candidate-label cross-correlations (effective G)
    # =================================================================
    print("\n=== D3: candidate-label pairwise correlations ===")
    rng = np.random.default_rng(2026_05_08)
    n_pairs = 1000
    pair_corrs = np.zeros(n_pairs, dtype=np.float32)
    for i in range(n_pairs):
        a, b_ = rng.integers(1, Q, size=2)
        if a == b_:
            continue
        v_a = cand_hw[a - 1]; v_b = cand_hw[b_ - 1]
        if v_a.std() < 1e-9 or v_b.std() < 1e-9:
            continue
        pair_corrs[i] = np.corrcoef(v_a, v_b)[0, 1]
    print(f"  pairwise |corr| (n=1000 candidate pairs):")
    print(f"    mean   = {np.abs(pair_corrs).mean():.3f}")
    print(f"    median = {np.median(np.abs(pair_corrs)):.3f}")
    print(f"    99%    = {np.percentile(np.abs(pair_corrs), 99):.3f}")
    print(f"    max    = {np.abs(pair_corrs).max():.3f}")
    # rough effective G estimator: G_eff ≈ G · (1 - ρ²)
    rho2 = float((pair_corrs ** 2).mean())
    g_eff = G * (1 - rho2)
    print(f"  E[ρ²] across pairs = {rho2:.3f}")
    print(f"  rough effective G = G·(1−E[ρ²]) ≈ {g_eff:.1f}")
    null_max_eff = float(np.sqrt(2 * np.log(Q - 1) / g_eff))
    print(f"  predicted null max with G_eff={g_eff:.1f}: {null_max_eff:.3f}")

    # =================================================================
    # D4 — per-sample SNR profile around predicted PoI (true-f label)
    # =================================================================
    print("\n=== D4: per-sample |corr|(trace[t], HW(γ·f_true mod q)) profile ===")
    print(f"  predicted PoI = {poi}, scanning ±100")
    lo = max(poi - 100, 0); hi = min(poi + 101, T)
    for slot in range(4):
        flv = int(f_arr[0, D * lanes[0] + slot]) % Q
        lab = np.array([bin((int(g) * flv) % Q).count("1")
                         for g in gammas], dtype=np.float32)
        lab_pT = np.repeat(lab, N).astype(np.float32)
        if lab_pT.std() < 1e-9:
            continue
        x_block = traces[0, 0, ..., lo:hi].reshape(G * N, hi - lo)
        x_z = (x_block - x_block.mean(axis=0)) / (x_block.std(axis=0) + 1e-9)
        lab_z = (lab_pT - lab_pT.mean()) / (lab_pT.std() + 1e-9)
        per_sample = (x_z * lab_z[:, None]).mean(axis=0)
        argmax = int(np.abs(per_sample).argmax())
        actual_t = lo + argmax
        peak = float(per_sample[argmax])
        print(f"  slot {slot}: true_f={flv}  argmax_t={actual_t} "
              f"(drift={actual_t - poi:+d})  |c|peak={abs(peak):.3f}  "
              f"|c|@PoI={abs(per_sample[poi - lo]):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
