#!/usr/bin/env python3
"""Phase 4.5 — single-victim multi-lane attack-valid evaluator.

Adapted from n39 (full stack) for multi-lane traces with K=1 fixed sk.
For each (lane, slot):
  - rk_M1_pred: M1 (HW(γ·f mod q)) at predict_poi(lane) — profile-free.
  - rk_full:    M1+M5 Zsum at slot-PoI = predict + LANE_SLOT_OFFSET[(lane,slot)]
                (fallback to 0 for non-calibrated lanes — those evaluate equal
                 to M1+M5 Zsum at predict_poi).

Profile-free baseline: rk_M1_pred (no oracle).
Profiled (calibrated lanes only): rk_full uses LANE_SLOT_OFFSET fitted from
  Phase 4 multi-key data (n34 oracle median across 28 victims; sk-independent
  firmware constant approximation).
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
    """sk-independent linear interpolation between calibrated lanes.

    For lanes outside {0, 64, 80, 128} (the calibrated set from Phase 4
    multi-key data), use linear interpolation to estimate slot drift.
    Constant extrapolation outside the [0, 128] range.
    """
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)        # (K=1, L, G, N, T)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")
    print(f"[INFO] lanes ({L}): {list(lanes)}")
    print(f"[INFO] G={G} (HW=1+HW=2)  N={N}\n")
    assert K == 1, "n43 expects K=1 single-victim"

    # Single victim sk
    f_arr = center(from_bytes(bytes(sk_blobs[0])[:POLYBYTES]))  # (768,)

    # Pre-compute candidate hypothesis matrices (M1, M5)
    cands = np.arange(1, Q, dtype=np.int64)
    prod = (gammas[None, :].astype(np.int64) * cands[:, None])     # (Q-1, G)
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
            print(f"[WARN] lane={lane}: predict_poi {poi_pred} out of range [0, {T})")
            continue

        # M1 baseline at predict_poi
        x_pred = traces[0, li, ..., poi_pred].mean(axis=1)        # (G,)
        xz_pred = (x_pred - x_pred.mean()) / (x_pred.std() + 1e-9)
        s1_b = np.abs(ch_M1z @ xz_pred) / G
        s5_b = np.abs(ch_M5z @ xz_pred) / G

        # full stack uses (lane, slot) offset (calibrated or linear interp)
        slot_offsets = {}
        for slot in range(4):
            slot_offsets[slot] = slot_offset_interp(lane, slot)
        unique_offs = sorted(set(slot_offsets.values()))
        s_cache = {}
        for off in unique_offs:
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
                rk_full = -1   # cannot compute (PoI out of range)

            rows.append(dict(
                lane=lane, slot=slot,
                true_f=flv, abs_f_c=abs(fc),
                calibrated=(lane in CALIBRATED),
                rk_M1_pred=rk_M1_pred,
                rk_M5_pred=rk_M5_pred,
                rk_M1_slot=rk_M1_slot,
                rk_M5_slot=rk_M5_slot,
                rk_full=rk_full,
            ))

    # Print summary
    n = len(rows)
    print(f"=== Single-victim attack ({n} cases, {L} lanes × ≤4 slots) ===\n")
    for label, key in [("M1 baseline (predict_poi)  ", "rk_M1_pred"),
                       ("M5 (predict_poi)            ", "rk_M5_pred"),
                       ("M1 (slot-PoI, calibrated)   ", "rk_M1_slot"),
                       ("M5 (slot-PoI, calibrated)   ", "rk_M5_slot"),
                       ("Full stack (slot+Zsum, calib)", "rk_full")]:
        rks = np.array([r[key] for r in rows if r[key] >= 0])
        if len(rks) == 0:
            continue
        print(f"  {label}: top-1 {(rks==0).sum()}/{len(rks)}, "
              f"top-10 {(rks<10).sum()}/{len(rks)}, "
              f"top-100 {(rks<100).sum()}/{len(rks)}, "
              f"top-500 {(rks<500).sum()}/{len(rks)}")

    # Per-lane top-100 (M1 baseline — main claim)
    print(f"\nPer-lane M1 baseline top-100 (profile-free):")
    print(f"{'lane':>5} {'cases':>6} {'top1':>5} {'top10':>5} {'top100':>6} {'top500':>6}")
    for lane in sorted(set(r["lane"] for r in rows)):
        sub = [r for r in rows if r["lane"] == lane]
        rs = np.array([r["rk_M1_pred"] for r in sub])
        print(f"{lane:>5} {len(sub):>6} {(rs==0).sum():>5} "
              f"{(rs<10).sum():>5} {(rs<100).sum():>6} {(rs<500).sum():>6}")

    # Top-100 cases listing
    print(f"\nM1 baseline top-100 cases:")
    print(f"{'lane':>5} {'slot':>4} {'true_f':>7} {'|f_c|':>6} {'rk_M1_pred':>11}")
    for r in sorted(rows, key=lambda r: r["rk_M1_pred"]):
        if r["rk_M1_pred"] < 100:
            print(f"{r['lane']:>5} {r['slot']:>4} {r['true_f']:>7} "
                  f"{r['abs_f_c']:>6} {r['rk_M1_pred']:>11}")

    print(f"\nFull stack top-100 cases (calibrated lanes only):")
    print(f"{'lane':>5} {'slot':>4} {'true_f':>7} {'|f_c|':>6} {'rk_full':>8}")
    for r in sorted(rows, key=lambda r: r["rk_full"] if r["rk_full"] >= 0 else 9999):
        if 0 <= r["rk_full"] < 100:
            print(f"{r['lane']:>5} {r['slot']:>4} {r['true_f']:>7} "
                  f"{r['abs_f_c']:>6} {r['rk_full']:>8}")

    # Save
    np.savez_compressed(f"{args.out_prefix}.npz",
                        rows=np.array(rows, dtype=object))

    # Markdown
    md = [f"# Phase 4.5 — single-victim multi-lane attack",
          f"", f"- input: `{args.input.name}`",
          f"- traces shape: K={K} L={L} G={G} N={N} T={T}",
          f"- victim sk_blob hash: see meta", f""]
    md.append("## Aggregate")
    md.append(f"| pipeline | top-1 | top-10 | top-100 | top-500 |")
    md.append(f"|---|---:|---:|---:|---:|")
    for label, key in [("M1 baseline (predict_poi)", "rk_M1_pred"),
                       ("M5 baseline (predict_poi)", "rk_M5_pred"),
                       ("M1 slot-PoI (calibrated)", "rk_M1_slot"),
                       ("M5 slot-PoI (calibrated)", "rk_M5_slot"),
                       ("Full stack Zsum (calibrated)", "rk_full")]:
        rks = np.array([r[key] for r in rows if r[key] >= 0])
        if len(rks) == 0:
            md.append(f"| {label} | — | — | — | — |")
            continue
        md.append(f"| {label} | {(rks==0).sum()}/{len(rks)} "
                  f"| {(rks<10).sum()}/{len(rks)} "
                  f"| {(rks<100).sum()}/{len(rks)} "
                  f"| {(rks<500).sum()}/{len(rks)} |")
    md.append("")
    md.append("## Per-lane M1 baseline (profile-free)")
    md.append(f"| lane | cases | top-1 | top-10 | top-100 | top-500 |")
    md.append(f"|---:|---:|---:|---:|---:|---:|")
    for lane in sorted(set(r["lane"] for r in rows)):
        sub = [r for r in rows if r["lane"] == lane]
        rs = np.array([r["rk_M1_pred"] for r in sub])
        md.append(f"| {lane} | {len(sub)} | {(rs==0).sum()} "
                  f"| {(rs<10).sum()} | {(rs<100).sum()} | {(rs<500).sum()} |")
    md.append("")
    md.append("## Top-100 (M1 baseline)")
    md.append(f"| lane | slot | true f | \\|f_c\\| | rk_M1_pred |")
    md.append(f"|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: r["rk_M1_pred"]):
        if r["rk_M1_pred"] < 100:
            md.append(f"| {r['lane']} | {r['slot']} | {r['true_f']} "
                      f"| {r['abs_f_c']} | {r['rk_M1_pred']} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md) + "\n")
    print(f"\n[OK] saved → {args.out_prefix}.{{npz,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
