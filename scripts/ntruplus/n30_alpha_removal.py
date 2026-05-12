#!/usr/bin/env python3
"""α-removal experiment: subtract cross-key γ-baseline before V2 CPA.

Hypothesis: At fixed PoI, Y[k, g] = α[g] + b_k·H_{f_k}[g] + noise,
where α[g] is sk-independent γ-baseline. Mean_k(Y[k, g]) ≈ α[g] when
mean_k(b_k·H_{f_k}[g]) ≈ 0 (random f's).

α-removed Y'[k, g] = Y[k, g] - mean_(k'≠k)(Y[k', g])

Then V2 CPA on Y' should have lower σ_a → higher corr → better recovery.

Test variants:
  V2_raw    — original V2 (baseline)
  V2_alpha1 — subtract leave-one-out cross-key mean
  V2_alpha2 — subtract cross-batch mean (load multiple npz)
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
    # Load multiple lane=0 batches for richer α estimate.
    batches = [
        ("K=4 N=32", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N32.npz"),
        ("K=4 N=64", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K4N64.npz"),
        ("K=8 N=32a", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32.npz"),
        ("K=8 N=32b", ROOT / "traces/ntruplus768/phase3/wideg_lane0_K8N32_b.npz"),
    ]

    poi = predict_poi(0)
    all_y = []   # list of (key_id_global, batch_id, Y[g])
    all_meta = []   # (batch_name, k_local, gammas)
    gammas_ref = None
    for bi, (name, path) in enumerate(batches):
        if not path.exists():
            print(f"[SKIP] {path}")
            continue
        z = np.load(path, allow_pickle=True)
        traces = z["traces"].astype(np.float32)
        sk_blobs = z["sk_blobs"]
        gammas = z["gammas"].astype(int)
        K, L, G, N, T = traces.shape
        if gammas_ref is None:
            gammas_ref = gammas
        else:
            if not np.array_equal(gammas, gammas_ref):
                print(f"[WARN] {name}: gammas differ from first batch")
        f_arr = np.zeros((K, 768), dtype=np.int16)
        for ki in range(K):
            f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))
        for ki in range(K):
            Y_g = traces[ki, 0, ..., poi].mean(axis=1)   # (G,)
            all_y.append(Y_g)
            all_meta.append(dict(batch=name, key=ki, f=f_arr[ki]))

    Y_mat = np.stack(all_y, axis=0)   # (Ktotal, G)
    print(f"[INFO] Y matrix shape: {Y_mat.shape}, gammas={len(gammas_ref)}")

    # candidate label table
    cands = np.arange(1, Q, dtype=np.int64)
    G = len(gammas_ref)
    mat = (gammas_ref[None, :].astype(np.int64) * cands[:, None]) % Q
    cand_hw = np.zeros_like(mat, dtype=np.float32)
    for gi in range(G):
        cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]
    cand_hw_z = (cand_hw - cand_hw.mean(axis=1, keepdims=True)) / \
                (cand_hw.std(axis=1, keepdims=True) + 1e-9)

    Ktotal = len(all_meta)
    rows = []
    print(f"\n=== α-removal experiment: {Ktotal} keys (lane=0) at PoI={poi} ===")
    print(f"{'idx':>3} {'batch':>10} {'k':>2} {'sl':>3} {'true_f':>7} "
          f"{'|f_c|':>5} {'corr_raw':>9} {'corr_α1':>9} "
          f"{'rk_raw':>7} {'rk_α1':>7}")
    for i, m in enumerate(all_meta):
        # leave-one-out alpha estimate
        idx_other = [j for j in range(Ktotal) if j != i]
        alpha1 = Y_mat[idx_other].mean(axis=0)
        Y_raw = Y_mat[i]
        Y_alpha1 = Y_raw - alpha1

        # z-normalize
        z_raw = (Y_raw - Y_raw.mean()) / (Y_raw.std() + 1e-9)
        z_a1 = (Y_alpha1 - Y_alpha1.mean()) / (Y_alpha1.std() + 1e-9)
        score_raw = np.abs(cand_hw_z @ z_raw) / G
        score_a1 = np.abs(cand_hw_z @ z_a1) / G

        for slot in range(4):
            flv = int(m["f"][slot]) % Q
            if flv == 0:
                continue
            fc = flv if flv <= Q // 2 else flv - Q
            ti = flv - 1
            corr_raw = float(score_raw[ti])
            corr_a1 = float(score_a1[ti])
            rank_raw = int((score_raw > corr_raw).sum())
            rank_a1 = int((score_a1 > corr_a1).sum())
            rows.append(dict(idx=i, batch=m["batch"], key=m["key"], slot=slot,
                              true_f=flv, abs_f_c=abs(fc),
                              corr_raw=corr_raw, corr_a1=corr_a1,
                              rank_raw=rank_raw, rank_a1=rank_a1))
            print(f"{i:>3} {m['batch']:>10} {m['key']:>2} {slot:>3} {flv:>7} "
                  f"{abs(fc):>5} {corr_raw:>9.3f} {corr_a1:>9.3f} "
                  f"{rank_raw:>7} {rank_a1:>7}")

    # summary
    n = len(rows)
    rks_raw = np.array([r["rank_raw"] for r in rows])
    rks_a1 = np.array([r["rank_a1"] for r in rows])
    cs_raw = np.array([r["corr_raw"] for r in rows])
    cs_a1 = np.array([r["corr_a1"] for r in rows])
    print(f"\n=== summary ({n} cases) ===")
    print(f"  Recovery (top-1):   raw {(rks_raw==0).sum()}/{n}  "
          f"α1 {(rks_a1==0).sum()}/{n}")
    print(f"  Recovery (top-10):  raw {(rks_raw<10).sum()}/{n}  "
          f"α1 {(rks_a1<10).sum()}/{n}")
    print(f"  Recovery (top-100): raw {(rks_raw<100).sum()}/{n}  "
          f"α1 {(rks_a1<100).sum()}/{n}")
    print(f"  Recovery (top-500): raw {(rks_raw<500).sum()}/{n}  "
          f"α1 {(rks_a1<500).sum()}/{n}")
    print(f"  Median corr: raw={np.median(cs_raw):.3f} α1={np.median(cs_a1):.3f}")
    print(f"  Max corr:    raw={cs_raw.max():.3f} α1={cs_a1.max():.3f}")

    # check α-removed cases that flipped
    print(f"\n  Cases where α1 helped (rank_α1 < rank_raw):")
    for r in sorted(rows, key=lambda r: r["rank_a1"]):
        if r["rank_a1"] < r["rank_raw"] and r["rank_a1"] < 100:
            print(f"   {r['batch']:>10} k={r['key']} sl={r['slot']} f={r['true_f']} "
                  f"|f_c|={r['abs_f_c']}: rk {r['rank_raw']} → {r['rank_a1']}")

    out = ROOT / "results/ntruplus/phase4/alpha_removal.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=object))
    print(f"\n[OK] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
