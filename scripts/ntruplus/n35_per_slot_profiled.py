#!/usr/bin/env python3
"""Per-(lane, slot) profiled PoI attack.

n34 found lane=80 has SLOT-DEPENDENT drifts (+9, -4, -13, -15 — spread
24 samples) while lane=0 is uniform (+24 all slots). Apply per-(lane, slot)
median drift offset to predict_poi.

Calibration is sk-independent (firmware/hardware constant), so this is
attack-valid: profile once, attack many keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402

# Per-(lane, slot) offsets from n34 weighted-by-peak_corr median
# These are firmware constants from oracle calibration.
LANE_SLOT_OFFSET = {
    (0, 0): 24, (0, 1): 24, (0, 2): 24, (0, 3): 24,
    (64, 0): 12, (64, 1): 12, (64, 2): 11, (64, 3): -2,
    (80, 0): 9, (80, 1): -4, (80, 2): -13, (80, 3): -15,
    (128, 0): 0, (128, 1): -2, (128, 2): 0, (128, 3): -1,
}


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def main() -> int:
    batches = [
        ("K=4 r1 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N32.npz"),
        ("K=4 N=64 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N64.npz"),
        ("K=8 r1 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz"),
        ("K=8 r2 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32_b.npz"),
        ("K=8 ln=64", ROOT / "traces/ntruplus768/phase3/wideg_lane64_K8N32.npz"),
        ("K=4 ln=80", ROOT / "traces/ntruplus768/phase3/wideg_lane80_K4N32.npz"),
        ("K=8 r1 ln=128", ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32.npz"),
        ("K=8 r2 ln=128", ROOT / "traces/ntruplus768/phase3/wideg_lane128_K8N32_b.npz"),
    ]

    rows = []
    for batch_name, path in batches:
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        traces = z["traces"].astype(np.float32)
        sk_blobs = z["sk_blobs"]
        gammas = z["gammas"].astype(int)
        ln = z["lanes"].astype(int)
        K, L, G, N, T = traces.shape
        lane = int(ln[0])
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

        poi_pred = predict_poi(lane)
        # Pre-compute scores at each unique (lane, slot) PoI
        slot_offsets = sorted(set(LANE_SLOT_OFFSET[(lane, s)] for s in range(4)))
        score_by_offset = {}
        for off in slot_offsets:
            poi_use = poi_pred + off
            for ki in range(K):
                key_id = (ki, off)
                x = traces[ki, 0, ..., poi_use].mean(axis=1)
                xz = (x - x.mean()) / (x.std() + 1e-9)
                score_by_offset[key_id] = np.abs(cand_hw_z @ xz) / G

        for ki in range(K):
            x_p = traces[ki, 0, ..., poi_pred].mean(axis=1)
            xpz = (x_p - x_p.mean()) / (x_p.std() + 1e-9)
            score_pred = np.abs(cand_hw_z @ xpz) / G
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                slot_off = LANE_SLOT_OFFSET[(lane, slot)]
                score_slot = score_by_offset[(ki, slot_off)]
                rk_pred = int((score_pred > score_pred[ti]).sum())
                rk_slot = int((score_slot > score_slot[ti]).sum())
                rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, abs_f_c=abs(fc), slot_offset=slot_off,
                    score_pred=float(score_pred[ti]),
                    score_slot=float(score_slot[ti]),
                    rk_pred=rk_pred, rk_slot=rk_slot,
                ))

    n = len(rows)
    print(f"\n=== Per-(lane, slot) profiled PoI ({n} cases) ===\n")

    rks_p = np.array([r["rk_pred"] for r in rows])
    rks_s = np.array([r["rk_slot"] for r in rows])
    print(f"  V2 predict     : top-1 {(rks_p==0).sum()}/{n}, "
          f"top-10 {(rks_p<10).sum()}/{n}, top-100 {(rks_p<100).sum()}/{n}, "
          f"top-500 {(rks_p<500).sum()}/{n}")
    print(f"  V2 (lane,slot) : top-1 {(rks_s==0).sum()}/{n}, "
          f"top-10 {(rks_s<10).sum()}/{n}, top-100 {(rks_s<100).sum()}/{n}, "
          f"top-500 {(rks_s<500).sum()}/{n}")

    # per-lane
    print(f"\nPer-lane top-100:")
    print(f"{'lane':>5} {'n':>4} {'pred':>6} {'(l,s)':>7}")
    for lane in sorted(set(r["lane"] for r in rows)):
        sub = [r for r in rows if r["lane"] == lane]
        rps = np.array([r["rk_pred"] for r in sub])
        rss = np.array([r["rk_slot"] for r in sub])
        n_l = len(sub)
        print(f"{lane:>5} {n_l:>4} {(rps<100).sum():>3}/{n_l:>2} "
              f"{(rss<100).sum():>4}/{n_l:>2}")

    # per-slot for lane=80 (most affected)
    print(f"\nlane=80 per-slot detail:")
    for slot in range(4):
        sub = [r for r in rows if r["lane"] == 80 and r["slot"] == slot]
        if not sub:
            continue
        rps = np.array([r["rk_pred"] for r in sub])
        rss = np.array([r["rk_slot"] for r in sub])
        print(f"  slot {slot} (offset {LANE_SLOT_OFFSET[(80, slot)]:+d}): "
              f"n={len(sub)} pred top-100 {(rps<100).sum()}, "
              f"slot top-100 {(rss<100).sum()}, "
              f"med rank pred {int(np.median(rps))} slot {int(np.median(rss))}")

    # gain cases
    print(f"\nCases gained (rank_slot < rank_pred) into top-200:")
    for r in sorted(rows, key=lambda r: r["rk_slot"]):
        if r["rk_slot"] < r["rk_pred"] and r["rk_slot"] < 200:
            print(f"  {r['batch']:>15} lane={r['lane']} k={r['key']} sl={r['slot']} "
                  f"f={r['true_f']} |f_c|={r['abs_f_c']} (off={r['slot_offset']:+d}): "
                  f"rk {r['rk_pred']} → {r['rk_slot']}")

    out = ROOT / "results/ntruplus/phase4/per_slot_profiled.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
