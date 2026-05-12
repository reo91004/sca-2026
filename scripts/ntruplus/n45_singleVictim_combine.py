#!/usr/bin/env python3
"""Combine multiple Phase 4.5 single-victim captures into one cumulative table.

Each capture has K=1 (one victim sk). For paper-grade single-victim claim,
the same sk would be ideal — but firmware 'k' regenerates fresh sk per
session. So we report per-victim statistics (each session's lanes), and
treat the cumulative as a multi-victim demonstration of the
single-victim multi-lane recovery method.

Inputs: list of trace files (each is a different victim).
Output: combined per-(victim, lane, slot) recovery table + stats.
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

LANE_SLOT_OFFSET = {
    (0, 0): 24, (0, 1): 24, (0, 2): 24, (0, 3): 24,
    (64, 0): 12, (64, 1): 12, (64, 2): 11, (64, 3): -2,
    (80, 0): 8, (80, 1): 7, (80, 2): -3, (80, 3): -3,
    (128, 0): 0, (128, 1): -2, (128, 2): 0, (128, 3): -1,
}
CALIBRATED = {0, 64, 80, 128}


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


def evaluate_one(input_path: Path):
    """Return list of per-(lane, slot) row dicts and victim sk hash."""
    z = np.load(input_path, allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    assert K == 1, f"{input_path.name} expects K=1"
    import hashlib
    sk_hash = hashlib.sha256(bytes(sk_blobs[0])).hexdigest()[:16]

    f_arr = center(from_bytes(bytes(sk_blobs[0])[:POLYBYTES]))
    cands = np.arange(1, Q, dtype=np.int64)
    prod = (gammas[None, :].astype(np.int64) * cands[:, None])
    mat_mod = prod % Q
    ch_M1 = np.zeros_like(mat_mod, dtype=np.float32)
    ch_M5 = np.zeros_like(mat_mod, dtype=np.float32)
    for gi in range(G):
        for ci in range(Q - 1):
            ch_M1[ci, gi] = bin(int(mat_mod[ci, gi])).count("1")
            ch_M5[ci, gi] = bin(montgomery_reduce(int(prod[ci, gi]))).count("1")
    ch_M1z = (ch_M1 - ch_M1.mean(axis=1, keepdims=True)) / \
             (ch_M1.std(axis=1, keepdims=True) + 1e-9)
    ch_M5z = (ch_M5 - ch_M5.mean(axis=1, keepdims=True)) / \
             (ch_M5.std(axis=1, keepdims=True) + 1e-9)

    rows = []
    for li, lane in enumerate(lanes):
        lane = int(lane)
        poi_pred = predict_poi(lane)
        if not (0 <= poi_pred < T):
            continue
        x_pred = traces[0, li, ..., poi_pred].mean(axis=1)
        xz_pred = (x_pred - x_pred.mean()) / (x_pred.std() + 1e-9)
        s1_b = np.abs(ch_M1z @ xz_pred) / G
        s5_b = np.abs(ch_M5z @ xz_pred) / G

        slot_offsets = {s: slot_offset_interp(lane, s) for s in range(4)}
        s_cache = {}
        for off in set(slot_offsets.values()):
            poi_use = poi_pred + off
            if not (0 <= poi_use < T):
                continue
            x_off = traces[0, li, ..., poi_use].mean(axis=1)
            xz_off = (x_off - x_off.mean()) / (x_off.std() + 1e-9)
            s1 = np.abs(ch_M1z @ xz_off) / G
            s5 = np.abs(ch_M5z @ xz_off) / G
            s_cache[off] = (s1, s5)

        for slot in range(4):
            flv = int(f_arr[D * lane + slot]) % Q
            if flv == 0:
                continue
            fc = flv if flv <= Q // 2 else flv - Q
            ti = flv - 1
            rk_M1_pred = int((s1_b > s1_b[ti]).sum())
            rk_M5_pred = int((s5_b > s5_b[ti]).sum())
            slot_off = slot_offsets[slot]
            if slot_off in s_cache:
                s1, s5 = s_cache[slot_off]
                rk_M1_slot = int((s1 > s1[ti]).sum())
                rk_M5_slot = int((s5 > s5[ti]).sum())
                z1 = (s1 - s1.mean()) / (s1.std() + 1e-9)
                z5 = (s5 - s5.mean()) / (s5.std() + 1e-9)
                score_full = z1 + z5
                rk_full = int((score_full > score_full[ti]).sum())
            else:
                rk_M1_slot = rk_M1_pred
                rk_M5_slot = rk_M5_pred
                rk_full = -1
            rows.append(dict(
                victim=sk_hash, lane=lane, slot=slot,
                true_f=flv, abs_f_c=abs(fc),
                calibrated=(lane in CALIBRATED),
                rk_M1_pred=rk_M1_pred, rk_M5_pred=rk_M5_pred,
                rk_M1_slot=rk_M1_slot, rk_M5_slot=rk_M5_slot,
                rk_full=rk_full,
                file=input_path.name, N=N, G=G,
            ))
    return rows, sk_hash, N, G, L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="Phase 4.5 trace files (one or more victims)")
    ap.add_argument("--out-prefix", type=Path,
                    default=ROOT / "results/ntruplus/phase45/combined")
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    victims = {}
    for inp in args.inputs:
        if not inp.exists():
            print(f"[SKIP] {inp}: not found")
            continue
        rows, sk_hash, N, G, L = evaluate_one(inp)
        victims[sk_hash] = dict(file=inp.name, N=N, G=G, L=L,
                                  cases=len(rows))
        all_rows.extend(rows)
        print(f"[INFO] {inp.name}: victim={sk_hash}  L={L}  N={N}  cases={len(rows)}")

    print(f"\n=== {len(all_rows)} cases over {len(victims)} victims ===\n")
    if not all_rows:
        return 0
    print(f"{'pipeline':>30} {'top-1':>6} {'top-10':>7} {'top-100':>8} {'top-500':>8}")
    print("-" * 72)
    for label, key in [("M1 baseline (predict_poi)    ", "rk_M1_pred"),
                       ("M5 baseline (predict_poi)    ", "rk_M5_pred"),
                       ("M1 slot-PoI (calibrated only)", "rk_M1_slot"),
                       ("M5 slot-PoI (calibrated only)", "rk_M5_slot"),
                       ("Full stack (Zsum, calib only)", "rk_full")]:
        rks = np.array([r[key] for r in all_rows if r[key] >= 0])
        if len(rks) == 0:
            continue
        print(f"{label:>30} {(rks==0).sum():>3}/{len(rks):<3} "
              f"{(rks<10).sum():>3}/{len(rks):<3} "
              f"{(rks<100).sum():>3}/{len(rks):<3} "
              f"{(rks<500).sum():>3}/{len(rks):<3}")

    # Per-victim breakdown
    print(f"\nPer-victim summary:")
    print(f"{'victim':>17} {'L':>4} {'N':>3} {'cases':>6} "
          f"{'M1 t100':>8} {'M1 t500':>8} {'full t100':>10}")
    for v in victims:
        sub = [r for r in all_rows if r["victim"] == v]
        m1_t100 = sum(1 for r in sub if r["rk_M1_pred"] < 100)
        m1_t500 = sum(1 for r in sub if r["rk_M1_pred"] < 500)
        full_t100 = sum(1 for r in sub
                        if r["rk_full"] >= 0 and r["rk_full"] < 100)
        info = victims[v]
        print(f"{v:>17} {info['L']:>4} {info['N']:>3} {info['cases']:>6} "
              f"{m1_t100:>8} {m1_t500:>8} {full_t100:>10}")

    # Per-(victim, lane) heatmap data
    print(f"\nPer-(victim, lane) M1 baseline top-100 hits:")
    for v in victims:
        sub = [r for r in all_rows if r["victim"] == v]
        lanes = sorted(set(r["lane"] for r in sub))
        for lane in lanes:
            subl = [r for r in sub if r["lane"] == lane]
            t100 = sum(1 for r in subl if r["rk_M1_pred"] < 100)
            if t100 > 0:
                cases = ", ".join(f"sl={r['slot']}/f={r['true_f']}/rk={r['rk_M1_pred']}"
                                   for r in subl if r["rk_M1_pred"] < 100)
                print(f"  victim={v[:8]} lane={lane:>3}: {t100} hit ({cases})")

    np.savez_compressed(f"{args.out_prefix}.npz",
                        rows=np.array(all_rows, dtype=object),
                        victims=np.array(victims, dtype=object))

    md = [f"# Phase 4.5 — Single-victim multi-lane (combined)", "",
          f"- Files combined: {len(args.inputs)}",
          f"- Total cases: {len(all_rows)}",
          f"- Unique victims: {len(victims)}", ""]
    md.append("## Aggregate")
    md.append("| pipeline | top-1 | top-10 | top-100 | top-500 |")
    md.append("|---|---:|---:|---:|---:|")
    for label, key in [("M1 baseline (predict_poi)", "rk_M1_pred"),
                       ("M5 baseline (predict_poi)", "rk_M5_pred"),
                       ("M1 slot-PoI (calibrated)", "rk_M1_slot"),
                       ("M5 slot-PoI (calibrated)", "rk_M5_slot"),
                       ("Full stack (Zsum, calib)", "rk_full")]:
        rks = np.array([r[key] for r in all_rows if r[key] >= 0])
        if len(rks) == 0:
            md.append(f"| {label} | — | — | — | — |")
            continue
        md.append(f"| {label} | {(rks==0).sum()}/{len(rks)} "
                  f"| {(rks<10).sum()}/{len(rks)} "
                  f"| {(rks<100).sum()}/{len(rks)} "
                  f"| {(rks<500).sum()}/{len(rks)} |")
    md.append("")
    md.append("## Per-victim")
    md.append("| victim | L | N | cases | M1 t100 | full t100 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for v in victims:
        sub = [r for r in all_rows if r["victim"] == v]
        m1_t100 = sum(1 for r in sub if r["rk_M1_pred"] < 100)
        full_t100 = sum(1 for r in sub
                         if r["rk_full"] >= 0 and r["rk_full"] < 100)
        info = victims[v]
        md.append(f"| {v} | {info['L']} | {info['N']} | {info['cases']} "
                  f"| {m1_t100} | {full_t100} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md) + "\n")
    print(f"\n[OK] saved → {args.out_prefix}.{{npz,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
