#!/usr/bin/env python3
"""Per-slot oracle drift pattern across all batches.

Each lane processes 4 slots in basemul. Slots may have intra-iter offsets
(since they're separate variable-uses within the same basemul). If so,
per-slot offsets could refine PoI further than the lane-only median.

Output: drift table by (lane, slot) over all batches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def find_peak(traces_block, gammas, flv, poi_pred, N, window=80):
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
    return lo + argmax, float(np.abs(per_sample[argmax]))


def main() -> int:
    batches = [
        ("K=4 r1 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N32.npz"),
        ("K=4 N=64 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N64.npz"),
        ("K=8 r1 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz"),
        ("K=8 r2 ln=0", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32_b.npz"),
        ("K=8 ln=64", ROOT / "traces/ntruplus768/phase3/wideg_lane64_K8N32.npz"),
        ("K=4 ln=80", ROOT / "traces/ntruplus768/phase3/wideg_lane80_K4N32.npz"),
        ("K=8 ln=80", ROOT / "traces/ntruplus768/phase3/wideg_lane80_K8N32.npz"),
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
        poi_pred = predict_poi(lane)
        f_arr = np.zeros((K, 768), dtype=np.int16)
        for ki in range(K):
            f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))
        for ki in range(K):
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                peak_t, peak_c = find_peak(traces[ki, 0], gammas, flv,
                                             poi_pred, N, window=80)
                drift = peak_t - poi_pred
                rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, abs_f_c=abs(fc),
                    drift=drift, peak_corr=peak_c,
                ))

    print(f"\n=== Per-slot drift pattern ({len(rows)} cases) ===\n")

    # per (lane, slot)
    print(f"{'lane':>5} {'slot':>5} {'n':>3} {'med':>5} {'mean':>6} "
          f"{'std':>5} {'q25':>5} {'q75':>5}")
    for lane in sorted(set(r["lane"] for r in rows)):
        for slot in range(4):
            sub = [r for r in rows if r["lane"] == lane and r["slot"] == slot]
            if not sub:
                continue
            drifts = np.array([r["drift"] for r in sub])
            print(f"{lane:>5} {slot:>5} {len(sub):>3} "
                  f"{int(np.median(drifts)):>+5d} "
                  f"{drifts.mean():>+6.1f} {int(drifts.std()):>5d} "
                  f"{int(np.percentile(drifts, 25)):>+5d} "
                  f"{int(np.percentile(drifts, 75)):>+5d}")

    # weighted by peak_corr (higher peak = more reliable drift estimate)
    print(f"\n=== Per-slot drift, weighted by peak_corr (top half only) ===")
    print(f"{'lane':>5} {'slot':>5} {'n':>3} {'wmed':>5} {'high-corr cases':>20}")
    for lane in sorted(set(r["lane"] for r in rows)):
        for slot in range(4):
            sub = [r for r in rows if r["lane"] == lane and r["slot"] == slot]
            if len(sub) < 2:
                continue
            sub_sorted = sorted(sub, key=lambda r: r["peak_corr"],
                                 reverse=True)[:max(len(sub) // 2, 2)]
            drifts = np.array([r["drift"] for r in sub_sorted])
            cases = ", ".join(
                f"k{r['key']}({r['drift']:+d}, c={r['peak_corr']:.2f})"
                for r in sub_sorted[:3])
            print(f"{lane:>5} {slot:>5} {len(sub_sorted):>3} "
                  f"{int(np.median(drifts)):>+5d}  {cases}")

    out = ROOT / "results/ntruplus768/phase4/per_slot_drift.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
