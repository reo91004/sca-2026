#!/usr/bin/env python3
"""Phase 4.6-G200 PoI/window diagnostic.

The compact G=200 scout did not improve fixed-PoI recovery.  This script checks
whether the signal is merely shifted away from the timing model, or whether the
trace is intrinsically weak for this victim.

It compares:
  - pred:   timing-only predict_poi(lane)
  - slot:   calibrated per-(lane, slot) offset from Phase 4
  - emp:    sk-independent gamma-variance PoI per lane
  - winmax: per-candidate max over a local window (diagnostic, larger null)
  - oracle: true-candidate best sample in the window (diagnostic upper bound)

Only pred/slot/emp are attack-compatible.  winmax/oracle use a broader or
secret-referenced selection and are diagnostics, not paper claims.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q, QINV  # noqa: E402

MASK16 = (1 << 16) - 1
SIGN16 = 1 << 15

LANE_SLOT_OFFSET = {
    (0, 0): 24, (0, 1): 24, (0, 2): 24, (0, 3): 24,
    (64, 0): 12, (64, 1): 12, (64, 2): 11, (64, 3): -2,
    (80, 0): 8, (80, 1): 7, (80, 2): -3, (80, 3): -3,
    (128, 0): 0, (128, 1): -2, (128, 2): 0, (128, 3): -1,
}


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def montgomery_reduce(a: int) -> int:
    a = int(a)
    t = (a * QINV) & MASK16
    if t & SIGN16:
        t -= 1 << 16
    t = (a - t * Q) >> 16
    return t & MASK16


def rank_of(score: np.ndarray, true_idx: int) -> int:
    return int((score > score[true_idx]).sum())


def summarize(rows: list[dict], key: str) -> dict:
    rks = np.array([r[key] for r in rows if r[key] >= 0], dtype=np.int64)
    if len(rks) == 0:
        return dict(n=0, top1=0, top10=0, top100=0, top500=0,
                    median=-1, best=-1)
    return dict(
        n=int(len(rks)),
        top1=int((rks == 0).sum()),
        top10=int((rks < 10).sum()),
        top100=int((rks < 100).sum()),
        top500=int((rks < 500).sum()),
        median=int(np.median(rks)),
        best=int(rks.min()),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--window", type=int, default=96)
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)  # (K=1, L, G, N, T)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    assert K == 1, "n56 expects K=1 single-victim trace"

    print(f"[INFO] traces={traces.shape} lanes={list(lanes)} window=+/-{args.window}")

    f_arr = center(from_bytes(bytes(sk_blobs[0])[:POLYBYTES]))

    cands = np.arange(1, Q, dtype=np.int64)
    prod = gammas[None, :].astype(np.int64) * cands[:, None]
    mat_mod = prod % Q
    ch_m1 = np.zeros_like(mat_mod, dtype=np.float32)
    ch_m5 = np.zeros_like(mat_mod, dtype=np.float32)
    for gi in range(G):
        for ci in range(Q - 1):
            ch_m1[ci, gi] = bin(int(mat_mod[ci, gi])).count("1")
            ch_m5[ci, gi] = bin(montgomery_reduce(int(prod[ci, gi]))).count("1")
    ch_m1z = (ch_m1 - ch_m1.mean(axis=1, keepdims=True)) / (
        ch_m1.std(axis=1, keepdims=True) + 1e-9
    )
    ch_m5z = (ch_m5 - ch_m5.mean(axis=1, keepdims=True)) / (
        ch_m5.std(axis=1, keepdims=True) + 1e-9
    )

    rows: list[dict] = []
    for li, lane_np in enumerate(lanes):
        lane = int(lane_np)
        poi_pred = predict_poi(lane)
        lo = max(0, poi_pred - args.window)
        hi = min(T, poi_pred + args.window + 1)
        rel_pred = poi_pred - lo
        if not (0 <= rel_pred < hi - lo):
            print(f"[WARN] lane={lane}: predicted PoI out of range")
            continue

        x_win = traces[0, li, :, :, lo:hi].mean(axis=1)  # (G, W)
        xz_win = (x_win - x_win.mean(axis=0, keepdims=True)) / (
            x_win.std(axis=0, keepdims=True) + 1e-9
        )
        score_m1_win = np.abs(ch_m1z @ xz_win) / G  # (Q-1, W)
        score_m5_win = np.abs(ch_m5z @ xz_win) / G

        gamma_var = x_win.var(axis=0)
        rel_emp = int(gamma_var.argmax())
        poi_emp = lo + rel_emp

        score_m1_pred = score_m1_win[:, rel_pred]
        score_m5_pred = score_m5_win[:, rel_pred]
        score_m1_emp = score_m1_win[:, rel_emp]
        score_m5_emp = score_m5_win[:, rel_emp]
        score_m1_winmax = score_m1_win.max(axis=1)
        score_m5_winmax = score_m5_win.max(axis=1)

        for slot in range(4):
            flv = int(f_arr[D * lane + slot]) % Q
            if flv == 0:
                continue
            fc = flv if flv <= Q // 2 else flv - Q
            ti = flv - 1

            off = LANE_SLOT_OFFSET.get((lane, slot), 0)
            poi_slot = poi_pred + off
            rel_slot = poi_slot - lo
            if 0 <= rel_slot < hi - lo:
                score_m1_slot = score_m1_win[:, rel_slot]
                score_m5_slot = score_m5_win[:, rel_slot]
                rk_m1_slot = rank_of(score_m1_slot, ti)
                rk_m5_slot = rank_of(score_m5_slot, ti)
            else:
                rk_m1_slot = -1
                rk_m5_slot = -1

            rel_m1_or = int(score_m1_win[ti].argmax())
            rel_m5_or = int(score_m5_win[ti].argmax())
            poi_m1_or = lo + rel_m1_or
            poi_m5_or = lo + rel_m5_or

            rows.append(dict(
                lane=lane,
                slot=slot,
                true_f=flv,
                abs_f_c=abs(fc),
                poi_pred=poi_pred,
                poi_slot=poi_slot,
                poi_emp=poi_emp,
                poi_m1_or=poi_m1_or,
                poi_m5_or=poi_m5_or,
                drift_slot=poi_slot - poi_pred,
                drift_emp=poi_emp - poi_pred,
                drift_m1_or=poi_m1_or - poi_pred,
                drift_m5_or=poi_m5_or - poi_pred,
                rk_m1_pred=rank_of(score_m1_pred, ti),
                rk_m5_pred=rank_of(score_m5_pred, ti),
                rk_m1_slot=rk_m1_slot,
                rk_m5_slot=rk_m5_slot,
                rk_m1_emp=rank_of(score_m1_emp, ti),
                rk_m5_emp=rank_of(score_m5_emp, ti),
                rk_m1_winmax=rank_of(score_m1_winmax, ti),
                rk_m5_winmax=rank_of(score_m5_winmax, ti),
                rk_m1_oracle=rank_of(score_m1_win[:, rel_m1_or], ti),
                rk_m5_oracle=rank_of(score_m5_win[:, rel_m5_or], ti),
                s_m1_true_pred=float(score_m1_pred[ti]),
                s_m5_true_pred=float(score_m5_pred[ti]),
                s_m1_true_win=float(score_m1_winmax[ti]),
                s_m5_true_win=float(score_m5_winmax[ti]),
            ))

    methods = [
        ("M1 pred", "rk_m1_pred"),
        ("M5 pred", "rk_m5_pred"),
        ("M1 slot", "rk_m1_slot"),
        ("M5 slot", "rk_m5_slot"),
        ("M1 emp", "rk_m1_emp"),
        ("M5 emp", "rk_m5_emp"),
        ("M1 winmax", "rk_m1_winmax"),
        ("M5 winmax", "rk_m5_winmax"),
        ("M1 oracle", "rk_m1_oracle"),
        ("M5 oracle", "rk_m5_oracle"),
    ]

    print(f"\n=== Phase 4.6-G200 PoI diagnostic ({len(rows)} cases) ===\n")
    print(f"{'method':<12} {'top1':>7} {'top10':>7} {'top100':>8} "
          f"{'top500':>8} {'best':>6} {'median':>7}")
    summaries = []
    for label, key in methods:
        s = summarize(rows, key)
        summaries.append(dict(label=label, key=key, **s))
        print(f"{label:<12} {s['top1']:>3}/{s['n']:<3} {s['top10']:>3}/{s['n']:<3} "
              f"{s['top100']:>4}/{s['n']:<3} {s['top500']:>4}/{s['n']:<3} "
              f"{s['best']:>6} {s['median']:>7}")

    print("\nBest rows by diagnostic oracle:")
    print(f"{'kind':<10} {'lane':>5} {'slot':>4} {'true_f':>7} {'|f_c|':>6} "
          f"{'rank':>6} {'drift':>6}")
    for label, key, drift_key in [
        ("M1 oracle", "rk_m1_oracle", "drift_m1_or"),
        ("M5 oracle", "rk_m5_oracle", "drift_m5_or"),
        ("M1 winmax", "rk_m1_winmax", "drift_m1_or"),
        ("M5 winmax", "rk_m5_winmax", "drift_m5_or"),
    ]:
        best = sorted(rows, key=lambda r: r[key])[:4]
        for r in best:
            print(f"{label:<10} {r['lane']:>5} {r['slot']:>4} {r['true_f']:>7} "
                  f"{r['abs_f_c']:>6} {r[key]:>6} {r[drift_key]:>+6d}")

    np.savez_compressed(
        f"{args.out_prefix}.npz",
        rows=np.array(rows, dtype=object),
        summaries=np.array(summaries, dtype=object),
    )

    md = [
        "# Phase 4.6-G200 PoI/window diagnostic",
        "",
        f"- input: `{args.input.name}`",
        f"- traces shape: K={K} L={L} G={G} N={N} T={T}",
        f"- window: +/-{args.window} around timing-only predict_poi(lane)",
        "- `pred`, `slot`, `emp` are attack-compatible PoI choices.",
        "- `winmax` and `oracle` are diagnostics, not paper claims.",
        "",
        "## Aggregate",
        "| method | top-1 | top-10 | top-100 | top-500 | best rank | median rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['label']} | {s['top1']}/{s['n']} | {s['top10']}/{s['n']} "
            f"| {s['top100']}/{s['n']} | {s['top500']}/{s['n']} "
            f"| {s['best']} | {s['median']} |"
        )
    md.extend([
        "",
        "## Best Diagnostic Rows",
        "| kind | lane | slot | true f | abs f_c | rank | drift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, key, drift_key in [
        ("M1 oracle", "rk_m1_oracle", "drift_m1_or"),
        ("M5 oracle", "rk_m5_oracle", "drift_m5_or"),
        ("M1 winmax", "rk_m1_winmax", "drift_m1_or"),
        ("M5 winmax", "rk_m5_winmax", "drift_m5_or"),
    ]:
        best = sorted(rows, key=lambda r: r[key])[:4]
        for r in best:
            md.append(
                f"| {label} | {r['lane']} | {r['slot']} | {r['true_f']} "
                f"| {r['abs_f_c']} | {r[key]} | {r[drift_key]:+d} |"
            )
    md.append("")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"\n[OK] saved -> {args.out_prefix}.{{npz,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
