#!/usr/bin/env python3
"""Phase 4.6-G200 oracle-selection null.

n56 showed that a true-candidate oracle PoI can make every case top-100.  That
selection is secret-referenced and can be badly biased: any candidate may look
good at the sample where its own score peaks.

This script applies the same self-oracle selection to every candidate and
compares the true candidate against that null distribution.  If the true
candidate's self-oracle rank is typical under all candidates, the n56 oracle
upper bound should be treated as selection bias rather than leakage evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q, QINV  # noqa: E402

MASK16 = (1 << 16) - 1
SIGN16 = 1 << 15


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def montgomery_reduce(a: int) -> int:
    a = int(a)
    t = (a * QINV) & MASK16
    if t & SIGN16:
        t -= 1 << 16
    t = (a - t * Q) >> 16
    return t & MASK16


def self_oracle_ranks(score_win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank every candidate at the sample where that candidate peaks."""
    rel_best = score_win.argmax(axis=1)
    best = score_win[np.arange(score_win.shape[0]), rel_best]
    ranks = np.empty(score_win.shape[0], dtype=np.int32)
    for ci, (rel, val) in enumerate(zip(rel_best, best)):
        ranks[ci] = int((score_win[:, int(rel)] > float(val)).sum())
    return ranks, rel_best.astype(np.int32)


def rank_counts(ranks: np.ndarray) -> dict:
    return dict(
        n=int(len(ranks)),
        top1=int((ranks == 0).sum()),
        top10=int((ranks < 10).sum()),
        top100=int((ranks < 100).sum()),
        top500=int((ranks < 500).sum()),
        median=int(np.median(ranks)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--window", type=int, default=96)
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    assert K == 1, "n57 expects K=1 single-victim trace"
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

    true_rows: list[dict] = []
    null_rows: list[dict] = []
    for li, lane_np in enumerate(lanes):
        lane = int(lane_np)
        poi = predict_poi(lane)
        lo = max(0, poi - args.window)
        hi = min(T, poi + args.window + 1)
        x_win = traces[0, li, :, :, lo:hi].mean(axis=1)
        xz_win = (x_win - x_win.mean(axis=0, keepdims=True)) / (
            x_win.std(axis=0, keepdims=True) + 1e-9
        )
        for label, ch_z in [("M1", ch_m1z), ("M5", ch_m5z)]:
            score_win = np.abs(ch_z @ xz_win) / G
            ranks_all, rel_best = self_oracle_ranks(score_win)
            null_rows.append(dict(
                model=label,
                lane=lane,
                ranks_all=ranks_all,
                rel_best=rel_best,
                counts=rank_counts(ranks_all),
            ))
            for slot in range(4):
                flv = int(f_arr[D * lane + slot]) % Q
                if flv == 0:
                    continue
                ti = flv - 1
                fc = flv if flv <= Q // 2 else flv - Q
                # Lower rank is better; p_self is how much of the self-oracle
                # null is at least as good as the true candidate.
                p_self = float((ranks_all <= ranks_all[ti]).mean())
                true_rows.append(dict(
                    model=label,
                    lane=lane,
                    slot=slot,
                    true_f=flv,
                    abs_f_c=abs(fc),
                    self_rank=int(ranks_all[ti]),
                    self_drift=int(rel_best[ti]) + lo - poi,
                    self_p=p_self,
                ))

    print("\n=== Self-oracle null by model/lane ===\n")
    print(f"{'model':>5} {'lane':>5} {'top1':>8} {'top10':>8} {'top100':>9} "
          f"{'top500':>9} {'median':>7}")
    for r in null_rows:
        c = r["counts"]
        print(f"{r['model']:>5} {r['lane']:>5} {c['top1']:>4}/{c['n']:<4} "
              f"{c['top10']:>4}/{c['n']:<4} {c['top100']:>5}/{c['n']:<4} "
              f"{c['top500']:>5}/{c['n']:<4} {c['median']:>7}")

    print("\n=== True candidates under self-oracle null ===\n")
    print(f"{'model':>5} {'top1':>7} {'top10':>7} {'top100':>8} "
          f"{'top500':>8} {'median':>7} {'median p':>9}")
    summaries = []
    for model in ["M1", "M5"]:
        ranks = np.array([r["self_rank"] for r in true_rows if r["model"] == model])
        ps = np.array([r["self_p"] for r in true_rows if r["model"] == model])
        c = rank_counts(ranks)
        summaries.append(dict(model=model, **c, median_p=float(np.median(ps))))
        print(f"{model:>5} {c['top1']:>3}/{c['n']:<3} {c['top10']:>3}/{c['n']:<3} "
              f"{c['top100']:>4}/{c['n']:<3} {c['top500']:>4}/{c['n']:<3} "
              f"{c['median']:>7} {np.median(ps):>9.3f}")

    print("\nBest true rows:")
    print(f"{'model':>5} {'lane':>5} {'slot':>4} {'true_f':>7} {'|f_c|':>6} "
          f"{'rank':>6} {'p_self':>8} {'drift':>6}")
    for r in sorted(true_rows, key=lambda x: x["self_rank"])[:12]:
        print(f"{r['model']:>5} {r['lane']:>5} {r['slot']:>4} {r['true_f']:>7} "
              f"{r['abs_f_c']:>6} {r['self_rank']:>6} {r['self_p']:>8.3f} "
              f"{r['self_drift']:>+6d}")

    np.savez_compressed(
        f"{args.out_prefix}.npz",
        true_rows=np.array(true_rows, dtype=object),
        null_rows=np.array(null_rows, dtype=object),
        summaries=np.array(summaries, dtype=object),
    )

    md = [
        "# Phase 4.6-G200 oracle-selection null",
        "",
        f"- input: `{args.input.name}`",
        f"- traces shape: K={K} L={L} G={G} N={N} T={T}",
        f"- window: +/-{args.window}",
        "- Every candidate is scored at its own best sample in the window.",
        "",
        "## True Candidates",
        "| model | top-1 | top-10 | top-100 | top-500 | median rank | median self-null p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['model']} | {s['top1']}/{s['n']} | {s['top10']}/{s['n']} "
            f"| {s['top100']}/{s['n']} | {s['top500']}/{s['n']} "
            f"| {s['median']} | {s['median_p']:.3f} |"
        )
    md.extend([
        "",
        "## Null By Lane",
        "| model | lane | top-1 | top-10 | top-100 | top-500 | median rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in null_rows:
        c = r["counts"]
        md.append(
            f"| {r['model']} | {r['lane']} | {c['top1']}/{c['n']} "
            f"| {c['top10']}/{c['n']} | {c['top100']}/{c['n']} "
            f"| {c['top500']}/{c['n']} | {c['median']} |"
        )
    md.append("")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"\n[OK] saved -> {args.out_prefix}.{{npz,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
