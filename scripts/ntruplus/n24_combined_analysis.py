#!/usr/bin/env python3
"""Combined analysis across all wide-γ K=4/8 lane=0 N=32 batches.

Aggregates attack-valid CPA results from:
  - wideg_lane0_K4N32.npz (K=4 round 1)
  - wideg_lane0_K8N32.npz (K=8 round 1, contains TOP-1 case)
  - wideg_lane0_K8N32_b.npz (K=8 round 2)

Produces:
  - Total attack-valid recovery rate
  - Per-rank distribution (top-1, top-10, top-100, top-500)
  - SNR distribution
  - Recovery vs |f_centered| scatter
  - Final summary suitable for paper.
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


def compute_snr(traces_block, gammas, flv, poi, N):
    G = traces_block.shape[0]
    x = traces_block[..., poi].reshape(-1)
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


def main() -> int:
    batches = [
        ("K=4 r1 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N32.npz"),
        ("K=8 r1 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz"),
        ("K=8 r2 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32_b.npz"),
        ("K=8 ln=64", ROOT / "traces/ntruplus768/phase3/wideg_lane64_K8N32.npz"),
        ("K=8 ln=80", ROOT / "traces/ntruplus768/phase3/wideg_lane80_K8N32.npz"),
        ("K=8 r1 ln=128", ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32.npz"),
        ("K=8 r2 ln=128", ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32_b.npz"),
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
        K, L, G, N, T = traces.shape
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

        ln_z = z["lanes"].astype(int)
        lane = int(ln_z[0])
        poi = predict_poi(lane)
        for ki in range(K):
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                # f_centered for analysis
                f_centered = flv if flv <= Q // 2 else flv - Q

                # V2 fixed PoI
                x_mean = traces[ki, 0, ..., poi].mean(axis=1)
                xm_z = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
                score = np.abs(cand_hw_z @ xm_z) / G
                ti = flv - 1
                v2_T = float(score[ti])
                v2_99 = float(np.percentile(score, 99))
                v2_max = float(score.max())
                rank = int((score > score[ti]).sum())
                pct = 100.0 * (1.0 - rank / (Q - 1))

                # V2w window
                t_lo = max(poi - 10, 0); t_hi = min(poi + 11, T)
                x_mean_win = traces[ki, 0, ..., t_lo:t_hi].mean(axis=1)
                xmw_z = (x_mean_win - x_mean_win.mean(axis=0, keepdims=True)) / \
                        (x_mean_win.std(axis=0, keepdims=True) + 1e-9)
                corr_w = (cand_hw_z @ xmw_z) / G
                score_w = np.abs(corr_w).max(axis=1)
                v2w_T = float(score_w[ti])
                v2w_99 = float(np.percentile(score_w, 99))
                rank_w = int((score_w > score_w[ti]).sum())
                pct_w = 100.0 * (1.0 - rank_w / (Q - 1))

                snr = compute_snr(traces[ki, 0], gammas, flv, poi, N)

                all_rows.append(dict(
                    batch=batch_name, key=ki, slot=slot, true_f=flv,
                    f_centered=f_centered, abs_f_c=abs(f_centered),
                    snr=snr,
                    v2_T=v2_T, v2_99=v2_99, v2_max=v2_max,
                    v2_rank=rank, v2_pct=pct,
                    v2w_T=v2w_T, v2w_99=v2w_99,
                    v2w_rank=rank_w, v2w_pct=pct_w,
                    above_99=v2_T >= v2_99,
                    above_99_w=v2w_T >= v2w_99,
                ))

    if not all_rows:
        print("[ERR] no batches loaded")
        return 1

    n = len(all_rows)
    print(f"\n=== Cumulative attack-valid analysis ({n} cases) ===\n")
    snrs = np.array([r["snr"] for r in all_rows])
    abs_fc = np.array([r["abs_f_c"] for r in all_rows])
    v2_ranks = np.array([r["v2_rank"] for r in all_rows])
    v2w_ranks = np.array([r["v2w_rank"] for r in all_rows])
    above99 = np.array([r["above_99"] for r in all_rows])
    above99w = np.array([r["above_99_w"] for r in all_rows])

    print(f"SNR distribution:")
    print(f"  median = {np.median(snrs):.3f}")
    print(f"  90th pctile = {np.percentile(snrs, 90):.3f}")
    print(f"  max = {snrs.max():.3f}")
    print(f"  cases with SNR >= 0.5: {(snrs >= 0.5).sum()}")
    print(f"  cases with SNR >= 0.3: {(snrs >= 0.3).sum()}")
    print(f"  cases with SNR >= 0.2: {(snrs >= 0.2).sum()}")
    print(f"  cases with SNR >= 0.1: {(snrs >= 0.1).sum()}")
    print()
    print(f"|f_centered| distribution:")
    print(f"  median = {np.median(abs_fc):.0f}")
    print(f"  cases with |f_c| < 16: {(abs_fc < 16).sum()}")
    print(f"  cases with |f_c| < 32: {(abs_fc < 32).sum()}")
    print(f"  cases with |f_c| < 100: {(abs_fc < 100).sum()}")
    print()
    print(f"V2 (fixed PoI) recovery:")
    print(f"  top-1:    {(v2_ranks == 0).sum()}/{n}  ({(v2_ranks == 0).mean()*100:.2f}%)")
    print(f"  top-10:   {(v2_ranks < 10).sum()}/{n}  ({(v2_ranks < 10).mean()*100:.2f}%)")
    print(f"  top-100:  {(v2_ranks < 100).sum()}/{n}  ({(v2_ranks < 100).mean()*100:.2f}%)")
    print(f"  top-500:  {(v2_ranks < 500).sum()}/{n}  ({(v2_ranks < 500).mean()*100:.2f}%)")
    print(f"  >= null 99-th: {above99.sum()}/{n}  ({above99.mean()*100:.2f}%)")
    print()
    print(f"V2w (per-cand best PoI in ±10) recovery:")
    print(f"  top-1:    {(v2w_ranks == 0).sum()}/{n}  ({(v2w_ranks == 0).mean()*100:.2f}%)")
    print(f"  top-10:   {(v2w_ranks < 10).sum()}/{n}  ({(v2w_ranks < 10).mean()*100:.2f}%)")
    print(f"  top-100:  {(v2w_ranks < 100).sum()}/{n}  ({(v2w_ranks < 100).mean()*100:.2f}%)")
    print(f"  >= null 99-th: {above99w.sum()}/{n}  ({above99w.mean()*100:.2f}%)")
    print()
    print(f"Recovered cases (V2 top-100 OR V2w top-100):")
    print(f"{'batch':>8} {'key':>4} {'slot':>4} {'true_f':>7} {'f_c':>5} {'|f_c|':>5} "
          f"{'SNR':>6} {'V2_pct':>7} {'V2w_pct':>8}")
    for r in all_rows:
        if r["v2_rank"] < 100 or r["v2w_rank"] < 100:
            print(f"{r['batch']:>8} {r['key']:>4} {r['slot']:>4} {r['true_f']:>7} "
                  f"{r['f_centered']:>+5d} {r['abs_f_c']:>5} {r['snr']:>6.3f} "
                  f"{r['v2_pct']:>7.2f} {r['v2w_pct']:>8.2f}")

    # save
    out = ROOT / "results/ntruplus/phase4/combined.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=np.array(all_rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
