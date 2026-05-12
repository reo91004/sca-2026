#!/usr/bin/env python3
"""Attack-valid CPA + per-key SNR on K=4 wide-γ capture.

For each (key, slot) at lane 0, run V2 (mean-over-N CPA at fixed
predicted PoI) + V2w (window ±W). Report rank, percentile, top-k.
Also report per-(key, slot) SNR (diagnostic, uses true f).

The 16 attack cases (4 keys × 4 slots) provide enough statistical
power to detect whether at least one key/slot has SNR sufficient to
break the empirical null max ceiling (~0.66 at G=78).
"""

from __future__ import annotations

import argparse
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
    """Diagnostic SNR at a single (key, lane, slot, poi) sample."""
    x = traces_block[..., poi].reshape(-1)   # (G·N,)
    G = traces_block.shape[0]
    lab_g = np.array([bin((int(g) * flv) % Q).count("1")
                       for g in gammas], dtype=np.float32)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N32.npz")
    p.add_argument("--out-prefix", type=Path,
                   default=ROOT / "results/ntruplus/phase4/wideg_K4_attack")
    p.add_argument("--window", type=int, default=10)
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)        # (K, L, G, N, T)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")
    print(f"[INFO] G={G} N={N} (HW=1+HW=2)")

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

    rows = []
    for ki in range(K):
        for li, lane in enumerate(lanes):
            poi_pred = predict_poi(int(lane))
            t_lo = max(poi_pred - args.window, 0)
            t_hi = min(poi_pred + args.window + 1, T)
            ww = t_hi - t_lo

            # V2 fixed PoI: mean over N at predicted PoI
            x_mean_fixed = traces[ki, li, ..., poi_pred].mean(axis=1)
            xm_z = (x_mean_fixed - x_mean_fixed.mean()) / (x_mean_fixed.std() + 1e-9)
            score_v2 = np.abs(cand_hw_z @ xm_z) / G
            null99_v2 = float(np.percentile(score_v2, 99))
            null_max_v2 = float(score_v2.max())

            # V2w window: mean over N, then per-cand best in window
            x_mean_win = traces[ki, li, ..., t_lo:t_hi].mean(axis=1)   # (G, W)
            xmw_z = (x_mean_win - x_mean_win.mean(axis=0, keepdims=True)) / \
                    (x_mean_win.std(axis=0, keepdims=True) + 1e-9)
            corr_w = (cand_hw_z @ xmw_z) / G
            score_v2w = np.abs(corr_w).max(axis=1)
            null99_v2w = float(np.percentile(score_v2w, 99))
            null_max_v2w = float(score_v2w.max())

            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                ti = flv - 1
                v2_true = float(score_v2[ti])
                v2w_true = float(score_v2w[ti])
                v2_rank = int((score_v2 > v2_true).sum())
                v2w_rank = int((score_v2w > v2w_true).sum())
                v2_pct = 100.0 * (1.0 - v2_rank / (Q - 1))
                v2w_pct = 100.0 * (1.0 - v2w_rank / (Q - 1))
                snr = compute_snr(traces[ki, li], gammas, flv, poi_pred, N)
                rows.append(dict(
                    key=ki, lane=int(lane), slot=slot, poi=poi_pred,
                    true_f=flv, snr=snr,
                    v2_true=v2_true, v2_99=null99_v2, v2_max=null_max_v2,
                    v2_rank=v2_rank, v2_pct=v2_pct,
                    v2w_true=v2w_true, v2w_99=null99_v2w, v2w_max=null_max_v2w,
                    v2w_rank=v2w_rank, v2w_pct=v2w_pct,
                ))

    print(f"\n{'k':>2} {'slot':>4} {'true_f':>7} {'SNR':>6} "
          f"{'V2_T':>5} {'V2_99':>5} {'V2_M':>5} {'V2_pct':>7} "
          f"{'V2w_T':>5} {'V2w_99':>5} {'V2w_M':>5} {'V2w_pct':>7}")
    for r in rows:
        print(f"{r['key']:>2} {r['slot']:>4} {r['true_f']:>7} {r['snr']:>6.3f} "
              f"{r['v2_true']:.3f} {r['v2_99']:.3f} {r['v2_max']:.3f} "
              f"{r['v2_pct']:>7.2f} "
              f"{r['v2w_true']:.3f} {r['v2w_99']:.3f} {r['v2w_max']:.3f} "
              f"{r['v2w_pct']:>7.2f}")

    v2_pcts = np.array([r["v2_pct"] for r in rows])
    v2w_pcts = np.array([r["v2w_pct"] for r in rows])
    snrs = np.array([r["snr"] for r in rows])
    top1_v2 = (np.array([r["v2_rank"] for r in rows]) == 0).mean()
    top10_v2 = (np.array([r["v2_rank"] for r in rows]) < 10).mean()
    top100_v2 = (np.array([r["v2_rank"] for r in rows]) < 100).mean()
    top1_v2w = (np.array([r["v2w_rank"] for r in rows]) == 0).mean()
    top10_v2w = (np.array([r["v2w_rank"] for r in rows]) < 10).mean()
    top100_v2w = (np.array([r["v2w_rank"] for r in rows]) < 100).mean()
    above99_v2 = np.mean([r["v2_true"] >= r["v2_99"] for r in rows])
    above99_v2w = np.mean([r["v2w_true"] >= r["v2w_99"] for r in rows])

    print(f"\n[STAT] over {len(rows)} (key, slot) cases:")
    print(f"  SNR median = {np.median(snrs):.3f}, max = {snrs.max():.3f}")
    print(f"  V2 mean pctile  = {v2_pcts.mean():.2f}, "
          f"top1 = {top1_v2:.3f}, top10 = {top10_v2:.3f}, "
          f"top100 = {top100_v2:.3f}")
    print(f"  V2w mean pctile = {v2w_pcts.mean():.2f}, "
          f"top1 = {top1_v2w:.3f}, top10 = {top10_v2w:.3f}, "
          f"top100 = {top100_v2w:.3f}")
    print(f"  P(V2 true ≥ null99): {above99_v2:.3f}")
    print(f"  P(V2w true ≥ null99): {above99_v2w:.3f}")

    # per-key breakdown
    print(f"\n[STAT] per-key:")
    for ki in range(K):
        ks = [r for r in rows if r["key"] == ki]
        if not ks:
            continue
        kv2 = np.array([r["v2_pct"] for r in ks])
        ksnr = np.array([r["snr"] for r in ks])
        print(f"  key {ki}: SNR_med={np.median(ksnr):.3f}  "
              f"V2_pct_mean={kv2.mean():.2f}  "
              f"top10={(np.array([r['v2_rank'] for r in ks]) < 10).mean():.3f}")

    np.savez_compressed(f"{args.out_prefix}.npz",
                         rows=np.array(rows, dtype=object))

    md = []
    md.append(f"# Phase 4 — wide-γ K=4 lane=0 attack (G={G}, N={N})")
    md.append("")
    md.append(f"- traces: K={K}, L={L}, G={G}, N={N}, T={T}")
    md.append("")
    md.append("## summary (attack-valid)")
    md.append(f"- {len(rows)} (key, slot) cases")
    md.append(f"- SNR median {np.median(snrs):.3f}, max {snrs.max():.3f}")
    md.append(f"- V2 mean pctile {v2_pcts.mean():.2f}, top1 {top1_v2:.3f}, "
              f"top10 {top10_v2:.3f}, top100 {top100_v2:.3f}")
    md.append(f"- V2w mean pctile {v2w_pcts.mean():.2f}, top1 {top1_v2w:.3f}, "
              f"top10 {top10_v2w:.3f}, top100 {top100_v2w:.3f}")
    md.append(f"- P(V2 true ≥ null99) {above99_v2:.3f}")
    md.append(f"- P(V2w true ≥ null99) {above99_v2w:.3f}")
    md.append("")
    md.append("## detail rows")
    md.append("| k | slot | true f | SNR | V2 |corr| | V2 99% | V2 max | V2 pctile | "
              "V2w |corr| | V2w 99% | V2w max | V2w pctile |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['key']} | {r['slot']} | {r['true_f']} | {r['snr']:.3f} | "
                  f"{r['v2_true']:.3f} | {r['v2_99']:.3f} | {r['v2_max']:.3f} | "
                  f"{r['v2_pct']:.2f} | {r['v2w_true']:.3f} | {r['v2w_99']:.3f} | "
                  f"{r['v2w_max']:.3f} | {r['v2w_pct']:.2f} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {args.out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
