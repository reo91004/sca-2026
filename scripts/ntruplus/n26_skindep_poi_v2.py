#!/usr/bin/env python3
"""sk-independent PoI calibration + V2 CPA with empirical PoI.

Hypothesis: predict_poi(lane) has a lane-dependent offset (lane=0 +24,
lane=64 +10, lane=80 −12, lane=128 −2; from n25). Using sk-independent
PoI calibration via per-sample γ-variance (no f needed), we can refine
the PoI per (key, lane) and then run V2 CPA at that empirical PoI.

The γ-variance metric: for each sample t in ±W of predict_poi,
compute Var_g(mean_over_N(traces[ki, li, g, :, t])) across G γ values.
Peak variance → most γ-sensitive sample → likely basemul region.

This is attack-valid: only uses public γ values (chosen by attacker), no f.

Compares 3 PoI strategies:
  - V2 fixed: predict_poi(lane) (current baseline)
  - V2 emp:   sk-indep empirical PoI (this script)
  - V2 oracle: true peak (diagnostic upper bound; uses f)
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


def empirical_poi(traces_block, poi_pred, window=50):
    """sk-indep PoI: per-sample γ-variance peak in ±window."""
    G, N, T = traces_block.shape
    lo = max(poi_pred - window, 0); hi = min(poi_pred + window + 1, T)
    mean_per_g = traces_block[..., lo:hi].mean(axis=1)  # (G, W)
    var_per_t = mean_per_g.var(axis=0)                    # (W,)
    argmax = int(var_per_t.argmax())
    return lo + argmax, float(var_per_t[argmax])


def true_peak_poi(traces_block, gammas, flv, poi_pred, N, window=50):
    """Diagnostic: peak of |corr| using true f label."""
    G, _, T = traces_block.shape
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


def cpa_at(traces_block, gammas, sample, cand_hw_z):
    """V2 CPA at given sample: |corr(mean_trace_at[g, sample], HW(γ·cand))|."""
    G, _, _ = traces_block.shape
    x_mean = traces_block[..., sample].mean(axis=1)  # (G,)
    xm_z = (x_mean - x_mean.mean()) / (x_mean.std() + 1e-9)
    score = np.abs(cand_hw_z @ xm_z) / G
    return score


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-prefix", type=Path, required=True)
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces {traces.shape}, lanes={lanes}, window=±{args.window}")

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
            poi_emp, var_emp = empirical_poi(traces[ki, li], poi_pred,
                                               window=args.window)
            score_pred = cpa_at(traces[ki, li], gammas, poi_pred, cand_hw_z)
            score_emp = cpa_at(traces[ki, li], gammas, poi_emp, cand_hw_z)
            null99_pred = float(np.percentile(score_pred, 99))
            null99_emp = float(np.percentile(score_emp, 99))
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                v2_pred_T = float(score_pred[ti])
                v2_emp_T = float(score_emp[ti])
                rank_pred = int((score_pred > score_pred[ti]).sum())
                rank_emp = int((score_emp > score_emp[ti]).sum())
                pct_pred = 100.0 * (1.0 - rank_pred / (Q - 1))
                pct_emp = 100.0 * (1.0 - rank_emp / (Q - 1))

                # diagnostic oracle
                poi_or, peak_corr = true_peak_poi(traces[ki, li], gammas, flv,
                                                   poi_pred, N, window=args.window)
                score_or = cpa_at(traces[ki, li], gammas, poi_or, cand_hw_z)
                rank_or = int((score_or > score_or[ti]).sum())
                pct_or = 100.0 * (1.0 - rank_or / (Q - 1))
                rows.append(dict(
                    key=ki, lane=int(lane), slot=slot, true_f=flv,
                    f_centered=fc, abs_f_c=abs(fc),
                    poi_pred=poi_pred, poi_emp=poi_emp, poi_or=poi_or,
                    drift_emp=poi_emp - poi_pred,
                    drift_or=poi_or - poi_pred,
                    v2_pred_T=v2_pred_T, v2_pred_99=null99_pred,
                    v2_pred_rank=rank_pred, v2_pred_pct=pct_pred,
                    v2_emp_T=v2_emp_T, v2_emp_99=null99_emp,
                    v2_emp_rank=rank_emp, v2_emp_pct=pct_emp,
                    v2_or_rank=rank_or, v2_or_pct=pct_or,
                    var_emp=var_emp,
                ))

    n = len(rows)
    print(f"\n=== sk-indep PoI vs predict_poi vs oracle ({n} cases) ===\n")

    drifts_emp = np.array([r["drift_emp"] for r in rows])
    drifts_or = np.array([r["drift_or"] for r in rows])
    print(f"PoI drift summary:")
    print(f"  empirical (γ-var):   median={int(np.median(drifts_emp)):+d} "
          f"std={int(drifts_emp.std())} |max|={int(np.abs(drifts_emp).max())}")
    print(f"  oracle (true peak):  median={int(np.median(drifts_or)):+d} "
          f"std={int(drifts_or.std())} |max|={int(np.abs(drifts_or).max())}")
    # how often does empirical agree with oracle?
    diff_eo = drifts_emp - drifts_or
    print(f"  |emp - oracle|: median={int(np.median(np.abs(diff_eo)))} "
          f"≤5: {(np.abs(diff_eo) <= 5).mean()*100:.1f}%, "
          f"≤10: {(np.abs(diff_eo) <= 10).mean()*100:.1f}%")
    print()

    print(f"Recovery rate comparison:")
    rks_pred = np.array([r["v2_pred_rank"] for r in rows])
    rks_emp = np.array([r["v2_emp_rank"] for r in rows])
    rks_or = np.array([r["v2_or_rank"] for r in rows])
    for label, rks in [("V2 fixed (predict)", rks_pred),
                        ("V2 emp (γ-var)   ", rks_emp),
                        ("V2 oracle        ", rks_or)]:
        print(f"  {label}: top-1 {(rks==0).sum()}/{n}, "
              f"top-10 {(rks<10).sum()}/{n}, "
              f"top-100 {(rks<100).sum()}/{n}")

    # detail rows: where empirical PoI helps
    print(f"\nCases where V2 emp helps (rank emp < rank pred):")
    print(f"{'k':>3} {'sl':>3} {'true_f':>7} {'|f_c|':>5} {'drft_e':>6} {'drft_o':>6} "
          f"{'rk_pr':>6} {'rk_em':>6} {'rk_or':>6}")
    helps = sorted([r for r in rows if r["v2_emp_rank"] < r["v2_pred_rank"]],
                   key=lambda r: r["v2_emp_rank"])[:25]
    for r in helps:
        print(f"{r['key']:>3} {r['slot']:>3} {r['true_f']:>7} {r['abs_f_c']:>5} "
              f"{r['drift_emp']:>+6d} {r['drift_or']:>+6d} "
              f"{r['v2_pred_rank']:>6} {r['v2_emp_rank']:>6} {r['v2_or_rank']:>6}")

    np.savez_compressed(f"{args.out_prefix}.npz",
                         rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {args.out_prefix}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
