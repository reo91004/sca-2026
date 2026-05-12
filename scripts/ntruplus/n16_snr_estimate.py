#!/usr/bin/env python3
"""Quantitative SNR estimate at each (key, lane, slot, predicted PoI).

Per (key, lane) we have N=8 traces × G=12 γ. At the predicted PoI sample,
each trace value y is modeled as
    y = a + b · HW(γ·f_true mod q) + ε
where ε is zero-mean noise. We fit (a, b) by least squares per
(key, lane, slot) — this DOES use sk and is therefore a diagnostic,
not an attack-valid recovery.

Reported per (key, lane, slot):
  - σ_signal = |b| · std(HW(γ·f mod q) across γ)        — leak amplitude
  - σ_noise  = residual std around (a + b·HW(...))       — noise std
  - SNR = σ_signal / σ_noise
  - implied minimum N (per-γ replicates) for ~99% attack-valid
    candidate distinguishability under Gaussian noise model

Implied N: under V2 (mean-over-N) the corr SNR scales as
    corr_SNR(N) ≈ b · √N · √(G−1) · σ_HW / σ_noise (asymptotic)
For 3456 candidates, max-of-3456-null |corr| ≈ √(2·log(3456) / G)
under standardization. We invert to required N.
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

    print(f"\n{'k':>2} {'lane':>5} {'slot':>4} {'true_f':>7} "
          f"{'lab_var':>8} {'b':>8} {'sig_sig':>8} {'sig_noi':>8} "
          f"{'SNR':>6} {'N_req99':>9}")
    rows = []
    for ki in range(K):
        for li, lane in enumerate(lanes):
            poi = predict_poi(int(lane))
            x_all = traces[ki, li, ..., poi].reshape(-1)   # (G·N,)
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                # label per (γ, n) — same HW for all n with same γ
                lab_g = np.array([bin((int(g) * flv) % Q).count("1")
                                   for g in gammas], dtype=np.float32)
                lab_var = float(lab_g.var())
                lab_pT = np.repeat(lab_g, N)               # (G·N,)
                # OLS fit y = a + b · HW
                Hm = lab_pT - lab_pT.mean()
                Xm = x_all - x_all.mean()
                if Hm.var() < 1e-9:
                    continue
                b = float((Hm * Xm).mean() / (Hm.var()))
                a = float(x_all.mean() - b * lab_pT.mean())
                resid = x_all - (a + b * lab_pT)
                sigma_sig = abs(b) * float(lab_g.std())   # signal across γ
                sigma_noi = float(resid.std())
                snr = sigma_sig / max(sigma_noi, 1e-12)

                # N required so that "true |corr|" beats max-of-Q-1 null
                # using mean-over-N variant (V2):
                #   corr_true ≈ snr · √N / √(snr²·N + 1)
                # null max ≈ √(2·log(Q-1) / G)
                # solve corr_true ≥ null_max
                null_max = float(np.sqrt(2 * np.log(Q - 1) / G))
                if snr < 1e-6:
                    n_req = float("inf")
                else:
                    # corr = snr·√N / √(snr²·N + 1) ≥ null_max
                    # → snr²·N ≥ null_max²·(snr²·N + 1)
                    # → N · snr²(1 - null_max²) ≥ null_max²
                    # → N ≥ null_max² / (snr²·(1 - null_max²))
                    if null_max >= 1:
                        n_req = float("inf")
                    else:
                        n_req = (null_max ** 2) / (snr ** 2 * (1 - null_max ** 2))
                rows.append(dict(
                    key=ki, lane=int(lane), slot=slot, true_f=flv,
                    lab_var=lab_var, b=b, sigma_sig=sigma_sig,
                    sigma_noi=sigma_noi, snr=snr,
                    n_required=n_req,
                ))
                print(f"{ki:>2} {lane:>5} {slot:>4} {flv:>7} {lab_var:>8.3f} "
                      f"{b:>+8.4f} {sigma_sig:>7.4f} {sigma_noi:>7.4f} "
                      f"{snr:>6.3f} {n_req:>11.1f}")

    snrs = np.array([r["snr"] for r in rows])
    n_reqs = np.array([r["n_required"] for r in rows])
    finite = np.isfinite(n_reqs) & (n_reqs > 0)
    null_max = float(np.sqrt(2 * np.log(Q - 1) / G))
    print(f"\n[STAT] over {len(rows)} (key, lane, slot):")
    print(f"  SNR median = {np.median(snrs):.3f}, max = {snrs.max():.3f}")
    print(f"  null-max ceiling at G={G}: {null_max:.3f}")
    print(f"  G threshold for finite N: G > {2 * np.log(Q - 1):.1f}")
    if finite.any():
        print(f"  N required (median, finite) = {np.median(n_reqs[finite]):.1f}")
        print(f"  N required (90-th pctile, finite) = "
              f"{np.percentile(n_reqs[finite], 90):.1f}")
    else:
        print(f"  no finite N — null-max ≥ 1 (G={G} is below structural "
              f"threshold {int(np.ceil(2 * np.log(Q - 1)))})")

    # per-lane summary
    by_lane = {}
    for r in rows:
        by_lane.setdefault(r["lane"], []).append(r)
    print(f"\n[STAT] per lane: median SNR / median N→99%")
    for lane, rs in sorted(by_lane.items()):
        snr_l = np.array([r["snr"] for r in rs])
        nreq = np.array([r["n_required"] for r in rs])
        nf = np.isfinite(nreq) & (nreq > 0)
        if nf.any():
            print(f"  lane {lane:>4}: SNR_med={np.median(snr_l):.3f}  "
                  f"N_med={np.median(nreq[nf]):.1f}  "
                  f"(n={len(rs)}, finite={nf.sum()})")

    out = ROOT / "results/ntruplus/phase4/snr_estimate.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
