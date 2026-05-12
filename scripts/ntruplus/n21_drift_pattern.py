#!/usr/bin/env python3
"""Diagnostic — is per-key best-PoI drift sk-dependent or noise-driven?

If drift is sk-dependent (e.g., key-specific cache/pipeline state) →
fixing the timing model won't help; we need per-key alignment.
If drift is noise-driven (random), then the "best PoI" location at
N=8 is dominated by noise spikes — capturing more N would let the
true peak emerge.

Test: for each (key, lane, slot), compute |corr| at every sample in
±50 with two different N partitions:
  N4_a = first 4 traces, N4_b = last 4 traces
If best-PoI shifts a lot between halves → noise-driven (each half
picks a different noise spike).
If best-PoI consistent → sk-dependent (real signal at consistent
location, just noisy).
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
    if N < 8:
        raise RuntimeError("need N≥8 to split into halves")

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")
    print(f"\n{'k':>2} {'lane':>5} {'slot':>4} {'true_f':>7} "
          f"{'PoI':>5} {'tA':>5} {'tB':>5} {'|tA-tB|':>8} "
          f"{'|c|A':>6} {'|c|B':>6}")
    rows = []
    for ki in range(K):
        for li, lane in enumerate(lanes):
            poi = predict_poi(int(lane))
            lo = max(poi - 50, 0); hi = min(poi + 51, T)
            for slot in range(4):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                lab = np.array([bin((int(g) * flv) % Q).count("1")
                                 for g in gammas], dtype=np.float32)
                lab_pT_h = np.repeat(lab, N // 2).astype(np.float32)
                lab_z = (lab_pT_h - lab_pT_h.mean()) / (lab_pT_h.std() + 1e-9)

                # half A: first 4 traces; half B: last 4 traces
                halfA = traces[ki, li, :, :N//2, lo:hi].reshape(-1, hi - lo)
                halfB = traces[ki, li, :, N//2:, lo:hi].reshape(-1, hi - lo)
                # standardize each sample column (per-half)
                aA = (halfA - halfA.mean(axis=0)) / (halfA.std(axis=0) + 1e-9)
                aB = (halfB - halfB.mean(axis=0)) / (halfB.std(axis=0) + 1e-9)
                cA = (aA * lab_z[:, None]).mean(axis=0)
                cB = (aB * lab_z[:, None]).mean(axis=0)
                iA = int(np.abs(cA).argmax()); iB = int(np.abs(cB).argmax())
                tA = lo + iA; tB = lo + iB
                diff = abs(tA - tB)
                rows.append(dict(
                    key=ki, lane=int(lane), slot=slot, true_f=flv,
                    poi=poi, tA=tA, tB=tB, diff=diff,
                    cA_max=float(cA[iA]), cB_max=float(cB[iB]),
                ))
                print(f"{ki:>2} {lane:>5} {slot:>4} {flv:>7} "
                      f"{poi:>5} {tA:>5} {tB:>5} {diff:>8} "
                      f"{abs(cA[iA]):>+6.3f} {abs(cB[iB]):>+6.3f}")

    diffs = np.array([r["diff"] for r in rows])
    print(f"\n[STAT] over {len(rows)} cases:")
    print(f"  |tA - tB| (sample diff between two halves of N=8):")
    print(f"    median = {np.median(diffs):.1f}")
    print(f"    mean   = {diffs.mean():.1f}")
    print(f"    fraction with |diff| ≤ 5:  {(diffs <= 5).mean():.3f}")
    print(f"    fraction with |diff| ≤ 10: {(diffs <= 10).mean():.3f}")
    print(f"    fraction with |diff| ≤ 20: {(diffs <= 20).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
