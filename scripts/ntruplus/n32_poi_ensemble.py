#!/usr/bin/env python3
"""PoI ensemble: max(score@predict, score@profiled).

For each candidate, take max of |corr| at two specific PoIs:
  - predict_poi(lane)
  - predict_poi(lane) + lane_offset (profiled)

Null grows by factor ~√(2) (two PoIs) but signal at the better PoI is preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402

LANE_OFFSET = {0: 24, 64: 10, 80: -12, 128: -2}


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

        poi_p = predict_poi(lane)
        poi_q = poi_p + LANE_OFFSET.get(lane, 0)

        for ki in range(K):
            x_p = traces[ki, 0, ..., poi_p].mean(axis=1)
            x_q = traces[ki, 0, ..., poi_q].mean(axis=1)
            xp_z = (x_p - x_p.mean()) / (x_p.std() + 1e-9)
            xq_z = (x_q - x_q.mean()) / (x_q.std() + 1e-9)
            score_p = np.abs(cand_hw_z @ xp_z) / G
            score_q = np.abs(cand_hw_z @ xq_z) / G
            score_e = np.maximum(score_p, score_q)
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                fc = flv if flv <= Q // 2 else flv - Q
                ti = flv - 1
                rk_p = int((score_p > score_p[ti]).sum())
                rk_q = int((score_q > score_q[ti]).sum())
                rk_e = int((score_e > score_e[ti]).sum())
                rows.append(dict(
                    batch=batch_name, lane=lane, key=ki, slot=slot,
                    true_f=flv, abs_f_c=abs(fc),
                    score_p=float(score_p[ti]), score_q=float(score_q[ti]),
                    score_e=float(score_e[ti]),
                    null99_p=float(np.percentile(score_p, 99)),
                    null99_q=float(np.percentile(score_q, 99)),
                    null99_e=float(np.percentile(score_e, 99)),
                    rk_p=rk_p, rk_q=rk_q, rk_e=rk_e,
                ))

    n = len(rows)
    print(f"\n=== PoI ensemble: predict ∨ profiled ({n} cases) ===\n")

    rks_p = np.array([r["rk_p"] for r in rows])
    rks_q = np.array([r["rk_q"] for r in rows])
    rks_e = np.array([r["rk_e"] for r in rows])
    n99_p = np.array([r["null99_p"] for r in rows])
    n99_q = np.array([r["null99_q"] for r in rows])
    n99_e = np.array([r["null99_e"] for r in rows])

    for label, rks in [("predict_poi  ", rks_p),
                       ("profiled     ", rks_q),
                       ("ensemble (max)", rks_e)]:
        print(f"  V2 {label}: top-1 {(rks==0).sum()}/{n} ({(rks==0).mean()*100:.2f}%), "
              f"top-10 {(rks<10).sum()}/{n}, "
              f"top-100 {(rks<100).sum()}/{n}, "
              f"top-500 {(rks<500).sum()}/{n}")

    print(f"\nNull99 mean (across cases):")
    print(f"  predict_poi   : {n99_p.mean():.3f}")
    print(f"  profiled      : {n99_q.mean():.3f}")
    print(f"  ensemble (max): {n99_e.mean():.3f}")

    # Per-lane breakdown
    print(f"\nPer-lane top-100 recovery:")
    print(f"{'lane':>5} {'n':>4} {'pred':>6} {'prof':>6} {'ens':>6}")
    for lane in sorted(set(r["lane"] for r in rows)):
        sub = [r for r in rows if r["lane"] == lane]
        rps = np.array([r["rk_p"] for r in sub])
        rqs = np.array([r["rk_q"] for r in sub])
        res = np.array([r["rk_e"] for r in sub])
        n_l = len(sub)
        print(f"{lane:>5} {n_l:>4} {(rps<100).sum():>2}/{n_l:>2} "
              f"{(rqs<100).sum():>3}/{n_l:>2} {(res<100).sum():>3}/{n_l:>2}")

    out = ROOT / "results/ntruplus768/phase4/poi_ensemble.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
