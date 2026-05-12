#!/usr/bin/env python3
"""b coefficient distribution across all batches.

Per n29 finding, V2 corr saturates at b·σ_HW / √((b·σ_HW)² + σ_a²).
Recovery requires b ≥ ~0.0018 (so b·σ_HW ≈ σ_α ≈ 0.0027).

This script computes b per (key, lane, slot) at:
  - predict_poi
  - oracle peak (true f)
and reports distribution + threshold-crossing rate per lane.
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


def fit_b(traces_block, gammas, flv, sample, N):
    """Compute b, σ_HW, σ_w, σ_a at given sample for a (key, slot)."""
    G = traces_block.shape[0]
    sub = traces_block[..., sample]   # (G, N)
    g_mean = sub.mean(axis=1)          # (G,)
    within = sub - g_mean[:, None]
    sigma_w = float(within.std())
    lab_g = np.array([bin((int(g) * flv) % Q).count("1") for g in gammas],
                      dtype=np.float32)
    sigma_HW = float(lab_g.std())
    Hm = lab_g - lab_g.mean()
    Xm = g_mean - g_mean.mean()
    if Hm.var() < 1e-9:
        return 0.0, sigma_HW, sigma_w, 0.0
    b = float((Hm * Xm).mean() / Hm.var())
    resid_a = Xm - b * Hm
    sigma_a = float(resid_a.std())
    return abs(b), sigma_HW, sigma_w, sigma_a


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
                b_p, sH, sw_p, sa_p = fit_b(traces[ki, 0], gammas, flv,
                                              poi_pred, N)
                poi_or = find_peak(traces[ki, 0], gammas, flv, poi_pred, N,
                                     window=50)
                b_o, _, sw_o, sa_o = fit_b(traces[ki, 0], gammas, flv,
                                             poi_or, N)
                # corr at oracle PoI
                bsH_o = b_o * sH
                corr_o = bsH_o / np.sqrt(bsH_o ** 2 + sa_o ** 2)
                rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, abs_f_c=abs(fc),
                    b_pred=b_p, b_oracle=b_o,
                    sigma_HW=sH, sigma_a_pred=sa_p, sigma_a_oracle=sa_o,
                    bsH_oracle=bsH_o, corr_oracle=corr_o,
                ))

    n = len(rows)
    print(f"\n=== b coefficient distribution ({n} cases) ===\n")

    bs_p = np.array([r["b_pred"] for r in rows])
    bs_o = np.array([r["b_oracle"] for r in rows])
    bsHs_o = np.array([r["bsH_oracle"] for r in rows])
    sigma_a_o = np.array([r["sigma_a_oracle"] for r in rows])
    cs_o = np.array([r["corr_oracle"] for r in rows])
    sH_arr = np.array([r["sigma_HW"] for r in rows])

    print(f"b @ predict_poi: min={bs_p.min():.5f}, "
          f"median={np.median(bs_p):.5f}, max={bs_p.max():.5f}")
    print(f"b @ oracle peak: min={bs_o.min():.5f}, "
          f"median={np.median(bs_o):.5f}, max={bs_o.max():.5f}")
    print(f"σ_HW (depends on f): min={sH_arr.min():.3f}, "
          f"median={np.median(sH_arr):.3f}, max={sH_arr.max():.3f}")
    print(f"σ_a @ oracle: min={sigma_a_o.min():.4f}, "
          f"median={np.median(sigma_a_o):.4f}, max={sigma_a_o.max():.4f}")
    print(f"corr_oracle (b·σ_HW / √(...+σ_a²)):")
    print(f"  min={cs_o.min():.3f}, median={np.median(cs_o):.3f}, "
          f"max={cs_o.max():.3f}")
    print(f"  ≥ 0.5: {(cs_o >= 0.5).sum()}/{n} "
          f"({(cs_o >= 0.5).mean()*100:.1f}%)")
    print(f"  ≥ 0.4: {(cs_o >= 0.4).sum()}/{n}")
    print(f"  ≥ 0.3: {(cs_o >= 0.3).sum()}/{n}")
    print(f"  ≥ 0.25: {(cs_o >= 0.25).sum()}/{n}")

    # per-lane b distribution
    print(f"\nPer-lane b @ oracle peak:")
    print(f"{'lane':>5} {'n':>4} {'b_med':>8} {'b_max':>8} {'b≥0.0018':>10} "
          f"{'corr≥0.5':>10}")
    for lane in sorted(set(r["lane"] for r in rows)):
        sub = [r for r in rows if r["lane"] == lane]
        bs = np.array([r["b_oracle"] for r in sub])
        cs = np.array([r["corr_oracle"] for r in sub])
        print(f"{lane:>5} {len(sub):>4} {np.median(bs):>8.5f} "
              f"{bs.max():>8.5f} {(bs >= 0.0018).sum():>5}/{len(sub):>2}    "
              f"{(cs >= 0.5).sum():>5}/{len(sub):>2}")

    # b distribution by |f_c| bin
    print(f"\nb @ oracle by |f_c| bin:")
    print(f"{'|f_c|':>15} {'n':>4} {'b_med':>8} {'σ_HW_med':>9} "
          f"{'corr_med':>9}")
    for lo, hi in [(0, 16), (16, 50), (50, 200), (200, 500),
                    (500, 1000), (1000, 1800)]:
        sub = [r for r in rows if lo <= r["abs_f_c"] < hi]
        if not sub:
            continue
        bs = np.array([r["b_oracle"] for r in sub])
        sH = np.array([r["sigma_HW"] for r in sub])
        cs = np.array([r["corr_oracle"] for r in sub])
        print(f"   [{lo:>4},{hi:>4}) {len(sub):>4} {np.median(bs):>8.5f} "
              f"{np.median(sH):>9.3f} {np.median(cs):>9.3f}")

    out = ROOT / "results/ntruplus768/phase4/b_distribution.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
