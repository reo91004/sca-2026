#!/usr/bin/env python3
"""Phase 4.6-G200 cross-validated window CPA.

Previous G200 diagnostics showed that unconstrained oracle/window selection is
dominated by sample-selection bias.  This script tests whether a candidate's
best local PoI is stable across independent gamma subsets:

  1. Split gamma designs into even/odd folds.
  2. On fold A, choose each candidate's best sample in a local window.
  3. Score that same candidate on fold B at the chosen sample.
  4. Repeat B->A and average the held-out scores.

This is attack-compatible: no true secret is used for sample selection.
If the leakage is real and locally stable, the true candidate should keep a
good rank on held-out gamma values.  If not, the prior oracle gains were mostly
overfit to sample/candidate choices.
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


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-9)


def corr_window(ch_z: np.ndarray, x_win: np.ndarray) -> np.ndarray:
    """Return |corr| over candidates x samples for one gamma fold."""
    xz = (x_win - x_win.mean(axis=0, keepdims=True)) / (
        x_win.std(axis=0, keepdims=True) + 1e-9
    )
    return np.abs(ch_z @ xz) / x_win.shape[0]


def cv_score(score_sel: np.ndarray, score_eval: np.ndarray) -> np.ndarray:
    """Per-candidate best sample on selection fold, evaluated on held-out fold."""
    rel = score_sel.argmax(axis=1)
    return score_eval[np.arange(score_eval.shape[0]), rel]


def rank_of(score: np.ndarray, true_idx: int) -> int:
    return int((score > score[true_idx]).sum())


def counts(ranks: np.ndarray) -> dict:
    return dict(
        n=int(len(ranks)),
        top1=int((ranks == 0).sum()),
        top10=int((ranks < 10).sum()),
        top100=int((ranks < 100).sum()),
        top500=int((ranks < 500).sum()),
        best=int(ranks.min()) if len(ranks) else -1,
        median=int(np.median(ranks)) if len(ranks) else -1,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--windows", default="16,32,64,96")
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    windows = [int(x) for x in args.windows.split(",") if x]

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    assert K == 1, "n59 expects K=1 single-victim trace"
    print(f"[INFO] traces={traces.shape} lanes={list(lanes)} windows={windows}")

    even = np.arange(G) % 2 == 0
    odd = ~even
    folds = [(even, odd), (odd, even)]
    print(f"[INFO] gamma folds: even={even.sum()} odd={odd.sum()}")

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

    rows: list[dict] = []
    for W in windows:
        lane_scores: dict[tuple[int, str], np.ndarray] = {}
        for li, lane_np in enumerate(lanes):
            lane = int(lane_np)
            poi = predict_poi(lane)
            lo = max(0, poi - W)
            hi = min(T, poi + W + 1)
            x_all = traces[0, li, :, :, lo:hi].mean(axis=1)  # (G, W)

            model_scores = {}
            for model, ch in [("M1", ch_m1), ("M5", ch_m5)]:
                fold_scores = []
                for sel_mask, eval_mask in folds:
                    ch_sel = ch[:, sel_mask]
                    ch_eval = ch[:, eval_mask]
                    ch_sel_z = (ch_sel - ch_sel.mean(axis=1, keepdims=True)) / (
                        ch_sel.std(axis=1, keepdims=True) + 1e-9
                    )
                    ch_eval_z = (ch_eval - ch_eval.mean(axis=1, keepdims=True)) / (
                        ch_eval.std(axis=1, keepdims=True) + 1e-9
                    )
                    score_sel = corr_window(ch_sel_z, x_all[sel_mask])
                    score_eval = corr_window(ch_eval_z, x_all[eval_mask])
                    fold_scores.append(cv_score(score_sel, score_eval))
                model_scores[model] = np.mean(fold_scores, axis=0)

            lane_scores[(lane, "M1")] = model_scores["M1"]
            lane_scores[(lane, "M5")] = model_scores["M5"]
            lane_scores[(lane, "Full")] = zscore(model_scores["M1"]) + zscore(model_scores["M5"])

        for lane_np in lanes:
            lane = int(lane_np)
            for slot in range(4):
                flv = int(f_arr[D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                for model in ["M1", "M5", "Full"]:
                    score = lane_scores[(lane, model)]
                    rows.append(dict(
                        window=W,
                        model=model,
                        lane=lane,
                        slot=slot,
                        true_f=flv,
                        abs_f_c=abs(fc),
                        rank=rank_of(score, ti),
                    ))

    summaries = []
    print("\n=== G200 cross-validated window CPA ===\n")
    print(f"{'W':>4} {'model':>5} {'top1':>7} {'top10':>7} {'top100':>8} "
          f"{'top500':>8} {'best':>6} {'median':>7}")
    for W in windows:
        for model in ["M1", "M5", "Full"]:
            ranks = np.array([
                r["rank"] for r in rows
                if r["window"] == W and r["model"] == model
            ], dtype=np.int64)
            c = counts(ranks)
            summaries.append(dict(window=W, model=model, **c))
            print(f"{W:>4} {model:>5} {c['top1']:>3}/{c['n']:<3} "
                  f"{c['top10']:>3}/{c['n']:<3} {c['top100']:>4}/{c['n']:<3} "
                  f"{c['top500']:>4}/{c['n']:<3} {c['best']:>6} {c['median']:>7}")

    print("\nBest rows:")
    print(f"{'W':>4} {'model':>5} {'lane':>5} {'slot':>4} {'true_f':>7} "
          f"{'|f_c|':>6} {'rank':>6}")
    for r in sorted(rows, key=lambda x: x["rank"])[:16]:
        print(f"{r['window']:>4} {r['model']:>5} {r['lane']:>5} {r['slot']:>4} "
              f"{r['true_f']:>7} {r['abs_f_c']:>6} {r['rank']:>6}")

    np.savez_compressed(
        f"{args.out_prefix}.npz",
        rows=np.array(rows, dtype=object),
        summaries=np.array(summaries, dtype=object),
    )

    md = [
        "# Phase 4.6-G200 cross-validated window CPA",
        "",
        f"- input: `{args.input.name}`",
        f"- traces shape: K={K} L={L} G={G} N={N} T={T}",
        f"- windows: {windows}",
        "- gamma split: even/odd; select candidate sample on one fold, score on the other.",
        "",
        "## Aggregate",
        "| W | model | top-1 | top-10 | top-100 | top-500 | best rank | median rank |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['window']} | {s['model']} | {s['top1']}/{s['n']} "
            f"| {s['top10']}/{s['n']} | {s['top100']}/{s['n']} "
            f"| {s['top500']}/{s['n']} | {s['best']} | {s['median']} |"
        )
    md.extend([
        "",
        "## Best Rows",
        "| W | model | lane | slot | true f | abs f_c | rank |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for r in sorted(rows, key=lambda x: x["rank"])[:16]:
        md.append(
            f"| {r['window']} | {r['model']} | {r['lane']} | {r['slot']} "
            f"| {r['true_f']} | {r['abs_f_c']} | {r['rank']} |"
        )
    md.append("")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"\n[OK] saved -> {args.out_prefix}.{{npz,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
