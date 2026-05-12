#!/usr/bin/env python3
"""K=4 N=64 wide-γ deep diagnostic.

Three pieces:
  D1 N-curve: subsample N ∈ {2, 4, 8, 16, 32, 64} for each (key, slot).
     measure SNR_N and corr_N, compare to formula corr = SNR / √(SNR² + 1/N).
  D2 PoI drift comparison vs predict_poi(0)=3014.
  D3 cross-batch SNR distribution (this batch vs K=4 N=32 vs K=8 N=32).
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


def compute_snr_corr(traces_block, gammas, flv, sample, N_use, cand_hw_z):
    """Compute SNR and V2 |corr| for first N_use traces at given sample."""
    G = traces_block.shape[0]
    sub = traces_block[:, :N_use, :]   # (G, N_use, T)
    # SNR
    x = sub[..., sample].reshape(-1)
    lab_g = np.array([bin((int(g) * flv) % Q).count("1") for g in gammas],
                      dtype=np.float32)
    lab_pT = np.repeat(lab_g, N_use).astype(np.float32)
    Hm = lab_pT - lab_pT.mean()
    Xm = x - x.mean()
    if Hm.var() < 1e-9:
        return 0.0, 0.0, 0
    b = float((Hm * Xm).mean() / Hm.var())
    sigma_sig = abs(b) * float(lab_g.std())
    resid = x - (lab_pT.mean() + b * lab_pT)
    sigma_noi = float(resid.std())
    snr = sigma_sig / max(sigma_noi, 1e-12)
    # V2 corr at given sample
    x_mean = sub[..., sample].mean(axis=1)
    xm_z = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
    score = np.abs(cand_hw_z @ xm_z) / G
    ti = flv - 1
    v2_t = float(score[ti])
    rank = int((score > score[ti]).sum())
    return snr, v2_t, rank


def find_peak_drift(traces_block, gammas, flv, poi_pred, N, window=50):
    """Find peak |corr| sample with true f label."""
    G, _, T = traces_block.shape
    lo = max(poi_pred - window, 0); hi = min(poi_pred + window + 1, T)
    lab = np.array([bin((int(g) * flv) % Q).count("1") for g in gammas],
                    dtype=np.float32)
    lab_pT = np.repeat(lab, N).astype(np.float32)
    if lab_pT.std() < 1e-9:
        return 0
    x_block = traces_block[..., lo:hi].reshape(G * N, hi - lo)
    x_z = (x_block - x_block.mean(axis=0)) / (x_block.std(axis=0) + 1e-9)
    lab_z = (lab_pT - lab_pT.mean()) / (lab_pT.std() + 1e-9)
    per_sample = (x_z * lab_z[:, None]).mean(axis=0)
    return lo + int(np.abs(per_sample).argmax())


def main() -> int:
    z = np.load(ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N64.npz",
                 allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces {traces.shape}")

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

    Ns = [2, 4, 8, 16, 32, 64]

    # D1 + D2 — N-curve + drift
    print("\n=== D1+D2: N-curve and PoI drift per (key, slot) ===")
    print(f"{'k':>2} {'sl':>3} {'true_f':>7} {'|f_c|':>5} {'drift':>6} | "
          + " | ".join(f"{f'corr_N={n}':>9}" for n in Ns) + " | SNR(N=64)")
    rows = []
    for ki in range(K):
        for slot in range(4):
            flv = int(f_arr[ki, D * 0 + slot]) % Q
            if flv == 0:
                continue
            fc = flv if flv <= Q // 2 else flv - Q
            drift = find_peak_drift(traces[ki, 0], gammas, flv, POI0, N) - POI0
            row = dict(key=ki, slot=slot, true_f=flv, abs_f_c=abs(fc),
                       drift=drift, snr_n=[], corr_n=[], rank_n=[])
            corrs_str = []
            for n_use in Ns:
                snr, corr, rank = compute_snr_corr(traces[ki, 0], gammas,
                                                    flv, POI0, n_use, cand_hw_z)
                row["snr_n"].append(snr)
                row["corr_n"].append(corr)
                row["rank_n"].append(rank)
                corrs_str.append(f"{corr:.3f}")
            rows.append(row)
            snr64 = row["snr_n"][-1]
            print(f"{ki:>2} {slot:>3} {flv:>7} {abs(fc):>5} {drift:>+6d} | "
                  + " | ".join(f"{c:>9}" for c in corrs_str)
                  + f" | {snr64:.3f}")

    # D2 summary
    drifts = np.array([r["drift"] for r in rows])
    print(f"\nDrift summary: median={int(np.median(drifts)):+d} "
          f"std={int(drifts.std())} |max|={int(np.abs(drifts).max())}")

    # D1 summary: corr/predicted ratio at each N
    print(f"\nN-curve formula check: corr / (SNR/√(SNR² + 1/N))")
    print(f"{'N':>3} | "
          + " | ".join(f"{x:>9}" for x in
                        ["mean ratio", "median", "max snr_n", "max corr_n"]))
    for ni, n_use in enumerate(Ns):
        snrs = np.array([r["snr_n"][ni] for r in rows])
        corrs = np.array([r["corr_n"][ni] for r in rows])
        # predicted corr from SNR
        pred = snrs / np.sqrt(snrs ** 2 + 1.0 / n_use)
        ratio = corrs / np.maximum(pred, 1e-9)
        # exclude near-zero SNR cases
        mask = snrs > 0.05
        if mask.sum() > 0:
            print(f"{n_use:>3} | {ratio[mask].mean():>9.3f} | "
                  f"{np.median(ratio[mask]):>9.3f} | "
                  f"{snrs.max():>9.3f} | {corrs.max():>9.3f}")
        else:
            print(f"{n_use:>3} | (insufficient SNR>0.05 cases)")

    # D3 — small-|f_c| cases specifically (most informative)
    print(f"\n=== D3: small |f_c| (<150) cases ===")
    small = sorted([r for r in rows if r["abs_f_c"] < 150],
                   key=lambda r: r["abs_f_c"])
    for r in small:
        print(f"  k={r['key']} slot={r['slot']} f={r['true_f']} "
              f"|f_c|={r['abs_f_c']} drift={r['drift']:+d}")
        print(f"     SNR_N: " + " ".join(f"{s:.3f}" for s in r["snr_n"]))
        print(f"     corr_N: " + " ".join(f"{c:.3f}" for c in r["corr_n"]))
        print(f"     rank_N: " + " ".join(f"{rk}" for rk in r["rank_n"]))

    out = ROOT / "results/ntruplus/phase4/n64_diagnose.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
