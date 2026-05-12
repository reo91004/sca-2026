#!/usr/bin/env python3
"""Export top candidate coefficient lists for Phase 4.5/4.6.

This reruns the same CPA scoring used by n43/n45, but keeps the top candidate
values for each (victim, lane, slot, channel).  The output is intentionally
compact and is meant as input for later candidate-set / sparse-recovery work.
No new captures are taken.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q, QINV  # noqa: E402

DEFAULT_OUT = ROOT / "results/ntruplus768/phase45/candidate_export"
MASK16 = (1 << 16) - 1
SIGN16 = 1 << 15

LANE_SLOT_OFFSET = {
    (0, 0): 24, (0, 1): 24, (0, 2): 24, (0, 3): 24,
    (64, 0): 12, (64, 1): 12, (64, 2): 11, (64, 3): -2,
    (80, 0): 8, (80, 1): 7, (80, 2): -3, (80, 3): -3,
    (128, 0): 0, (128, 1): -2, (128, 2): 0, (128, 3): -1,
}
CALIBRATED = {0, 64, 80, 128}
CHANNELS = ["M1", "M5", "M1_slot", "M5_slot", "Full"]


def slot_offset_interp(lane: int, slot: int) -> int:
    if (lane, slot) in LANE_SLOT_OFFSET:
        return LANE_SLOT_OFFSET[(lane, slot)]
    cal = sorted(CALIBRATED)
    if lane <= cal[0]:
        return LANE_SLOT_OFFSET[(cal[0], slot)]
    if lane >= cal[-1]:
        return LANE_SLOT_OFFSET[(cal[-1], slot)]
    for i in range(len(cal) - 1):
        l1, l2 = cal[i], cal[i + 1]
        if l1 <= lane <= l2:
            o1 = LANE_SLOT_OFFSET[(l1, slot)]
            o2 = LANE_SLOT_OFFSET[(l2, slot)]
            t = (lane - l1) / (l2 - l1)
            return int(round(o1 + t * (o2 - o1)))
    return 0


def montgomery_reduce(a: int) -> int:
    a = int(a)
    t = (a * QINV) & MASK16
    if t & SIGN16:
        t -= 1 << 16
    t = (a - t * Q) >> 16
    return t & MASK16


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def top_candidates(score: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, score.size)
    idx = np.argpartition(score, -k)[-k:]
    idx = idx[np.argsort(score[idx])[::-1]]
    return (idx + 1).astype(np.int16), score[idx].astype(np.float32)


def rank_of(score: np.ndarray, true_f: int) -> int:
    return int((score > score[true_f - 1]).sum())


def hypothesis_mats(gammas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cands = np.arange(1, Q, dtype=np.int64)
    prod = gammas[None, :].astype(np.int64) * cands[:, None]
    mat_mod = prod % Q
    ch_m1 = np.zeros_like(mat_mod, dtype=np.float32)
    ch_m5 = np.zeros_like(mat_mod, dtype=np.float32)
    for gi in range(gammas.size):
        for ci in range(Q - 1):
            ch_m1[ci, gi] = bin(int(mat_mod[ci, gi])).count("1")
            ch_m5[ci, gi] = bin(montgomery_reduce(int(prod[ci, gi]))).count("1")
    ch_m1 = (ch_m1 - ch_m1.mean(axis=1, keepdims=True)) / \
        (ch_m1.std(axis=1, keepdims=True) + 1e-9)
    ch_m5 = (ch_m5 - ch_m5.mean(axis=1, keepdims=True)) / \
        (ch_m5.std(axis=1, keepdims=True) + 1e-9)
    return ch_m1, ch_m5


def score_at(traces_li: np.ndarray, poi: int, ch_m1: np.ndarray,
             ch_m5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # traces_li shape: (G, N, T)
    x = traces_li[..., poi].mean(axis=1)
    xz = (x - x.mean()) / (x.std() + 1e-9)
    s1 = np.abs(ch_m1 @ xz) / xz.size
    s5 = np.abs(ch_m5 @ xz) / xz.size
    return s1, s5


def evaluate_file(path: Path, top_k: int) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, _L, _G, _N, T = traces.shape
    if K != 1:
        raise ValueError(f"{path.name}: expected K=1, got K={K}")

    victim = hashlib.sha256(bytes(sk_blobs[0])).hexdigest()[:16]
    f_arr = center(from_bytes(bytes(sk_blobs[0])[:POLYBYTES]))
    ch_m1, ch_m5 = hypothesis_mats(gammas)

    out: list[dict] = []
    for li, lane_raw in enumerate(lanes):
        lane = int(lane_raw)
        poi_pred = predict_poi(lane)
        if not (0 <= poi_pred < T):
            continue
        s1_pred, s5_pred = score_at(traces[0, li], poi_pred, ch_m1, ch_m5)
        slot_scores: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for off in {slot_offset_interp(lane, s) for s in range(4)}:
            poi = poi_pred + off
            if 0 <= poi < T:
                slot_scores[off] = score_at(traces[0, li], poi, ch_m1, ch_m5)

        for slot in range(4):
            true_f = int(f_arr[D * lane + slot]) % Q
            if true_f == 0:
                continue
            fc = true_f if true_f <= Q // 2 else true_f - Q
            off = slot_offset_interp(lane, slot)
            if off in slot_scores:
                s1_slot, s5_slot = slot_scores[off]
                z1 = (s1_slot - s1_slot.mean()) / (s1_slot.std() + 1e-9)
                z5 = (s5_slot - s5_slot.mean()) / (s5_slot.std() + 1e-9)
                s_full = z1 + z5
            else:
                s1_slot, s5_slot = s1_pred, s5_pred
                s_full = np.full_like(s1_pred, np.nan, dtype=np.float32)

            scores = {
                "M1": s1_pred,
                "M5": s5_pred,
                "M1_slot": s1_slot,
                "M5_slot": s5_slot,
                "Full": s_full,
            }
            cand = {}
            cand_scores = {}
            ranks = {}
            for ch, score in scores.items():
                if np.isnan(score).all():
                    cand[ch] = np.array([], dtype=np.int16)
                    cand_scores[ch] = np.array([], dtype=np.float32)
                    ranks[ch] = -1
                    continue
                c, sc = top_candidates(score, top_k)
                cand[ch] = c
                cand_scores[ch] = sc
                ranks[ch] = rank_of(score, true_f)

            out.append(dict(
                victim=victim, file=path.name, lane=lane, slot=slot,
                true_f=true_f, abs_f_c=abs(fc), ranks=ranks,
                candidates=cand, candidate_scores=cand_scores,
            ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in args.inputs:
        if not path.exists():
            print(f"[SKIP] {path}: not found")
            continue
        part = evaluate_file(path, args.top_k)
        rows.extend(part)
        print(f"[INFO] {path.name}: exported {len(part)} rows")

    np.savez_compressed(
        f"{args.out_prefix}.npz",
        rows=np.array(rows, dtype=object),
        channels=np.array(CHANNELS, dtype=object),
        top_k=np.array(args.top_k, dtype=np.int32),
    )

    md = ["# Phase 4.5/4.6 candidate export", "",
          f"- rows: {len(rows)}",
          f"- top_k: {args.top_k}", "",
          "## True Top-100 Rows", "",
          "| victim | lane | slot | true f | |f_c| | channel | rank | top5 candidates |",
          "|---|---:|---:|---:|---:|---|---:|---|"]
    for r in rows:
        for ch in CHANNELS:
            rk = int(r["ranks"][ch])
            if 0 <= rk < 100:
                top5 = ",".join(str(int(x)) for x in r["candidates"][ch][:5])
                md.append(f"| {r['victim'][:8]} | {r['lane']} | {r['slot']} "
                          f"| {r['true_f']} | {r['abs_f_c']} | {ch} | {rk} | {top5} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md) + "\n")
    print(f"[OK] wrote {args.out_prefix}.{{npz,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
