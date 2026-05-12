#!/usr/bin/env python3
"""Cross-lane SNR comparison (diagnostic, uses true f).

For each lane batch, compute per-(key, slot) diagnostic SNR at predicted PoI
and at the true peak in ±100 window. Output:

  - SNR distribution per lane (matched and unmatched by |f_centered|)
  - PoI drift (predicted vs true peak) per lane
  - Whether predict_poi(lane) actually hits the leak peak

This separates two failure modes for a low-SNR lane:
  (i) intrinsic: leak power at that lane is genuinely smaller, OR
  (ii) PoI-prediction error: the leak is real but predict_poi misses it.

Diagnostic only — uses true f. Not attack-valid.
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


def compute_snr(traces_block, gammas, flv, sample, N):
    G = traces_block.shape[0]
    x = traces_block[..., sample].reshape(-1)
    lab_g = np.array([bin((int(g) * flv) % Q).count("1") for g in gammas],
                      dtype=np.float32)
    lab_pT = np.repeat(lab_g, N).astype(np.float32)
    Hm = lab_pT - lab_pT.mean()
    if Hm.var() < 1e-9:
        return 0.0
    Xm = x - x.mean()
    b = float((Hm * Xm).mean() / Hm.var())
    sigma_sig = abs(b) * float(lab_g.std())
    resid = x - (lab_pT.mean() + b * lab_pT)
    sigma_noi = float(resid.std())
    return sigma_sig / max(sigma_noi, 1e-12)


def find_peak_sample(traces_block, gammas, flv, poi_pred, N, window=100):
    """Search for the |corr| peak in ±window around poi_pred (true f)."""
    G = traces_block.shape[0]
    T = traces_block.shape[-1]
    lo = max(poi_pred - window, 0); hi = min(poi_pred + window + 1, T)
    lab = np.array([bin((int(g) * flv) % Q).count("1") for g in gammas],
                    dtype=np.float32)
    lab_pT = np.repeat(lab, N).astype(np.float32)
    if lab_pT.std() < 1e-9:
        return poi_pred, 0.0
    x_block = traces_block[..., lo:hi].reshape(G * N, hi - lo)
    x_z = (x_block - x_block.mean(axis=0)) / (x_block.std(axis=0) + 1e-9)
    lab_z = (lab_pT - lab_pT.mean()) / (lab_pT.std() + 1e-9)
    per_sample = (x_z * lab_z[:, None]).mean(axis=0)
    argmax = int(np.abs(per_sample).argmax())
    return lo + argmax, float(per_sample[argmax])


def main() -> int:
    batches = [
        ("lane=0 K=4", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N32.npz"),
        ("lane=0 K=8a", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz"),
        ("lane=0 K=8b", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32_b.npz"),
        ("lane=64 K=8", ROOT / "traces/ntruplus768/phase3/wideg_lane64_K8N32.npz"),
        ("lane=80 K=4", ROOT / "traces/ntruplus768/phase3/wideg_lane80_K4N32.npz"),
        ("lane=80 K=8", ROOT / "traces/ntruplus768/phase3/wideg_lane80_K8N32.npz"),
        ("lane=128 K=8a", ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32.npz"),
        ("lane=128 K=8b", ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32_b.npz"),
    ]

    all_rows = []
    for batch_name, path in batches:
        if not path.exists():
            print(f"[SKIP] {path}")
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

        for ki in range(K):
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                snr_pred = compute_snr(traces[ki, 0], gammas, flv, poi_pred, N)
                peak_t, peak_c = find_peak_sample(traces[ki, 0], gammas, flv,
                                                    poi_pred, N, window=100)
                snr_peak = compute_snr(traces[ki, 0], gammas, flv, peak_t, N)
                all_rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, f_centered=fc, abs_f_c=abs(fc),
                    snr_pred=snr_pred, snr_peak=snr_peak,
                    peak_t=peak_t, drift=peak_t - poi_pred,
                    peak_corr=peak_c,
                ))

    if not all_rows:
        print("[ERR] no batches loaded")
        return 1

    print(f"\n=== Cross-lane SNR comparison ({len(all_rows)} cases) ===\n")

    # group by lane
    lanes = sorted(set(r["lane"] for r in all_rows))
    for ln in lanes:
        rs = [r for r in all_rows if r["lane"] == ln]
        snrs_pred = np.array([r["snr_pred"] for r in rs])
        snrs_peak = np.array([r["snr_peak"] for r in rs])
        drifts = np.array([r["drift"] for r in rs])
        abs_fc = np.array([r["abs_f_c"] for r in rs])
        print(f"--- lane = {ln} ({len(rs)} cases) ---")
        print(f"  SNR @ predicted PoI: median={np.median(snrs_pred):.4f}, "
              f"90%={np.percentile(snrs_pred, 90):.4f}, max={snrs_pred.max():.4f}")
        print(f"  SNR @ peak in ±100:  median={np.median(snrs_peak):.4f}, "
              f"90%={np.percentile(snrs_peak, 90):.4f}, max={snrs_peak.max():.4f}")
        print(f"  drift (peak - pred): median={int(np.median(drifts)):+d}, "
              f"std={int(drifts.std())}, |max|={int(np.abs(drifts).max())}")
        print(f"  |f_c|: median={int(np.median(abs_fc))}, "
              f"min={int(abs_fc.min())}, max={int(abs_fc.max())}")
        # SNR vs |f_c| binned
        bins = [(0, 32), (32, 100), (100, 500), (500, 2000)]
        for lo, hi in bins:
            mask = (abs_fc >= lo) & (abs_fc < hi)
            if mask.sum() == 0:
                continue
            snr_bin = snrs_peak[mask]
            print(f"    |f_c|∈[{lo},{hi}): n={mask.sum()}, "
                  f"SNR median={np.median(snr_bin):.4f}, max={snr_bin.max():.4f}")
        print()

    # SNR vs |f_c| matched comparison: pick bins, compare lanes
    print("=== SNR (peak) by |f_c| bin × lane ===")
    print(f"{'|f_c| bin':>15} | " + " | ".join(f"{f'lane={ln}':>12}" for ln in lanes))
    for lo, hi in [(0, 32), (32, 100), (100, 300), (300, 800), (800, 2000)]:
        cells = []
        for ln in lanes:
            rs = [r for r in all_rows if r["lane"] == ln
                  and lo <= r["abs_f_c"] < hi]
            if not rs:
                cells.append("           -")
                continue
            snrs = np.array([r["snr_peak"] for r in rs])
            cells.append(f"  n={len(rs)} m={np.median(snrs):.3f}")
        print(f"{f'[{lo},{hi})':>15} | " + " | ".join(cells))

    # detect intrinsic lane SNR difference: median peak SNR per lane
    # (averaged across all f_c — apples-to-apples is not perfect but tells us
    # if a lane is dramatically lower across the board)
    print("\n=== Lane-level peak SNR summary ===")
    for ln in lanes:
        rs = [r for r in all_rows if r["lane"] == ln]
        snrs = np.array([r["snr_peak"] for r in rs])
        # exclude near-zero |f_c| outliers (which dominate)
        mid = [r["snr_peak"] for r in rs if 100 <= r["abs_f_c"] < 1000]
        if mid:
            mid = np.array(mid)
            print(f"  lane={ln:>3}: n={len(rs)}, "
                  f"all peak SNR median={np.median(snrs):.4f}, "
                  f"|f_c|∈[100,1000) median={np.median(mid):.4f}")

    out = ROOT / "results/ntruplus/phase4/lane_snr_compare.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=np.array(all_rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
