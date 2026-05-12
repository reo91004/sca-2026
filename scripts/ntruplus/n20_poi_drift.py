#!/usr/bin/env python3
"""Diagnostic — per-key/per-lane sample-level corr profile around predicted PoI.

For each (key, lane, slot) compute |corr(trace[t], HW(γ·f mod q))| for
every sample t in [poi_pred - 50, poi_pred + 50]. Locate argmax t_actual
and report:
  - per-key drift |t_actual - poi_pred|
  - max |corr| at any sample in window (true-label analysis — diagnostic
    only, uses true f)
  - corr at predicted PoI

Goal: are we losing signal because PoI prediction is off by a few
samples? If t_actual drifts >5 cycles per key, fixed-PoI CPA leaves
information on the table.
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


def main() -> int:
    z = np.load(ROOT / "traces/ntruplus768/phase3/multikey_hw1.npz",
                allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    print(f"\n{'k':>2} {'lane':>5} {'slot':>4} {'true_f':>7} {'PoI':>5} "
          f"{'t_act':>6} {'drift':>6} {'|c|@PoI':>8} {'|c|max':>7}")
    rows = []
    for ki in range(K):
        for li, lane in enumerate(lanes):
            poi_pred = predict_poi(int(lane))
            lo = max(poi_pred - 50, 0); hi = min(poi_pred + 51, T)
            x_block = traces[ki, li, ..., lo:hi].reshape(G * N, hi - lo)
            x_z = (x_block - x_block.mean(axis=0)) / (x_block.std(axis=0) + 1e-9)
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                lab = np.array([bin((int(g) * flv) % Q).count("1")
                                 for g in gammas], dtype=np.float32)
                lab_pT = np.repeat(lab, N).astype(np.float32)
                if lab_pT.std() < 1e-9:
                    continue
                lab_z = (lab_pT - lab_pT.mean()) / (lab_pT.std() + 1e-9)
                per_sample = (x_z * lab_z[:, None]).mean(axis=0)
                idx_max = int(np.abs(per_sample).argmax())
                t_actual = lo + idx_max
                drift = t_actual - poi_pred
                c_at_poi = float(per_sample[poi_pred - lo]) if 0 <= poi_pred - lo < (hi - lo) else 0.0
                c_max = float(per_sample[idx_max])
                rows.append(dict(
                    key=ki, lane=int(lane), slot=slot, true_f=flv,
                    poi_pred=poi_pred, t_actual=t_actual, drift=drift,
                    c_at_poi=c_at_poi, c_max=c_max,
                ))
                print(f"{ki:>2} {lane:>5} {slot:>4} {flv:>7} {poi_pred:>5} "
                      f"{t_actual:>6} {drift:>+6} {c_at_poi:>+8.3f} {c_max:>+7.3f}")

    drifts = np.array([r["drift"] for r in rows])
    c_pois = np.array([abs(r["c_at_poi"]) for r in rows])
    c_maxs = np.array([abs(r["c_max"]) for r in rows])
    print(f"\n[STAT] over {len(rows)} (key, lane, slot):")
    print(f"  drift mean = {drifts.mean():+.2f}  std = {drifts.std():.2f}  "
          f"|drift| max = {np.abs(drifts).max()}")
    print(f"  |corr@PoI| median = {np.median(c_pois):.3f}, "
          f"|corr|max median = {np.median(c_maxs):.3f}")
    print(f"  drift histogram:")
    bins = [-50, -20, -10, -5, 0, 5, 10, 20, 50]
    hist, edges = np.histogram(drifts, bins=bins)
    for i, h in enumerate(hist):
        print(f"    [{edges[i]:>+4}, {edges[i+1]:>+4}): {h}")

    out = ROOT / "results/ntruplus768/phase4/poi_drift.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
