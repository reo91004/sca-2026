#!/usr/bin/env python3
"""Phase 4.6-G200 attack-compatible window reducer sweep.

n56/n57 closed the secret-referenced oracle path as selection bias.  This
script tries candidate-independent reducers over the local PoI window:
mean, top-k mean, softmax, and max.  These reducers do not use the true secret
to pick a sample, so they are useful as negative/positive engineering gates.
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


def rank_of(score: np.ndarray, true_idx: int) -> int:
    return int((score > score[true_idx]).sum())


def topk_mean(score_win: np.ndarray, k: int) -> np.ndarray:
    k = min(k, score_win.shape[1])
    part = np.partition(score_win, score_win.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def softmax_score(score_win: np.ndarray, tau: float) -> np.ndarray:
    x = tau * score_win
    m = x.max(axis=1, keepdims=True)
    return (np.log(np.exp(x - m).mean(axis=1)) + m[:, 0]) / tau


def zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-9)


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
    assert K == 1, "n58 expects K=1 single-victim trace"
    print(f"[INFO] traces={traces.shape} lanes={list(lanes)} windows={windows}")

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

    reducer_defs = [
        ("mean", lambda s: s.mean(axis=1)),
        ("top3", lambda s: topk_mean(s, 3)),
        ("top5", lambda s: topk_mean(s, 5)),
        ("top9", lambda s: topk_mean(s, 9)),
        ("soft5", lambda s: softmax_score(s, 5.0)),
        ("soft10", lambda s: softmax_score(s, 10.0)),
        ("max", lambda s: s.max(axis=1)),
    ]

    rows: list[dict] = []
    for W in windows:
        lane_scores: dict[tuple[int, str, str], np.ndarray] = {}
        for li, lane_np in enumerate(lanes):
            lane = int(lane_np)
            poi = predict_poi(lane)
            lo = max(0, poi - W)
            hi = min(T, poi + W + 1)
            x_win = traces[0, li, :, :, lo:hi].mean(axis=1)
            xz_win = (x_win - x_win.mean(axis=0, keepdims=True)) / (
                x_win.std(axis=0, keepdims=True) + 1e-9
            )
            score_m1_win = np.abs(ch_m1z @ xz_win) / G
            score_m5_win = np.abs(ch_m5z @ xz_win) / G
            for rname, reducer in reducer_defs:
                s1 = reducer(score_m1_win)
                s5 = reducer(score_m5_win)
                lane_scores[(lane, rname, "M1")] = s1
                lane_scores[(lane, rname, "M5")] = s5
                lane_scores[(lane, rname, "Full")] = zscore(s1) + zscore(s5)

        for lane_np in lanes:
            lane = int(lane_np)
            for slot in range(4):
                flv = int(f_arr[D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                for rname, _ in reducer_defs:
                    for model in ["M1", "M5", "Full"]:
                        score = lane_scores[(lane, rname, model)]
                        rows.append(dict(
                            window=W,
                            reducer=rname,
                            model=model,
                            lane=lane,
                            slot=slot,
                            true_f=flv,
                            abs_f_c=abs(fc),
                            rank=rank_of(score, ti),
                        ))

    summaries = []
    print("\n=== G200 attack-compatible window reducer sweep ===\n")
    print(f"{'W':>4} {'reducer':>7} {'model':>5} {'top1':>7} {'top10':>7} "
          f"{'top100':>8} {'top500':>8} {'best':>6} {'median':>7}")
    for W in windows:
        for rname, _ in reducer_defs:
            for model in ["M1", "M5", "Full"]:
                ranks = np.array([
                    r["rank"] for r in rows
                    if r["window"] == W and r["reducer"] == rname and r["model"] == model
                ], dtype=np.int64)
                c = counts(ranks)
                summaries.append(dict(window=W, reducer=rname, model=model, **c))
                print(f"{W:>4} {rname:>7} {model:>5} {c['top1']:>3}/{c['n']:<3} "
                      f"{c['top10']:>3}/{c['n']:<3} {c['top100']:>4}/{c['n']:<3} "
                      f"{c['top500']:>4}/{c['n']:<3} {c['best']:>6} {c['median']:>7}")

    np.savez_compressed(
        f"{args.out_prefix}.npz",
        rows=np.array(rows, dtype=object),
        summaries=np.array(summaries, dtype=object),
    )

    md = [
        "# Phase 4.6-G200 window reducer sweep",
        "",
        f"- input: `{args.input.name}`",
        f"- traces shape: K={K} L={L} G={G} N={N} T={T}",
        f"- windows: {windows}",
        "- All reducers are candidate-independent and attack-compatible.",
        "",
        "## Aggregate",
        "| W | reducer | model | top-1 | top-10 | top-100 | top-500 | best rank | median rank |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['window']} | {s['reducer']} | {s['model']} "
            f"| {s['top1']}/{s['n']} | {s['top10']}/{s['n']} "
            f"| {s['top100']}/{s['n']} | {s['top500']}/{s['n']} "
            f"| {s['best']} | {s['median']} |"
        )
    md.append("")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"\n[OK] saved -> {args.out_prefix}.{{npz,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
