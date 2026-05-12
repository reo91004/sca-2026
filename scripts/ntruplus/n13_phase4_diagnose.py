#!/usr/bin/env python3
"""Phase 4 diagnostic — locate why held-out (cross-key) recovery fails.

Phase 4 (n08) showed mean percentile 50.79 ≈ random. Possible causes:

  (A) Multikey capture SNR too low (N=8 vs scout N=24) — even within-key
      recovery would fail.
  (B) Cross-key transfer fails (per-key DC offset / amplitude / clock
      drift) — within-key works but ridge model doesn't generalize.
  (C) Per-trace label entropy issue — trained ridge collapses despite
      apparent corr at PoI.

We run three tests:

  T1 within-key recovery: per key, split N=8 into 4 train / 4 test
      (same key, same f, same γ), profile ridge, evaluate rank.

  T2 per-key z-norm cross-key recovery: per-key normalize traces (subtract
      per-key mean trace, divide per-key std), then run same held-out
      attack as n08.

  T3 mean-prediction cross-key recovery: instead of ridge on raw window,
      use the simpler estimator pred_HW(γ) = a + b · trace[poi]
      (single-sample regression). If single-PoI corr is real, this should
      work. If it doesn't, ridge isn't the culprit.

Output: per-test mean percentile, per-lane summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402


def hw_unsigned(v: int) -> int:
    return bin(int(v) & 0xFFFF).count("1")


def predict_poi(lane: int) -> int:
    return int(3014 + 33.0 * (lane >> 1) + 16 * (lane & 1))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def score_rank(pred_per_g: np.ndarray, gammas: np.ndarray, true_f: int) -> tuple[int, float]:
    cands = np.arange(1, Q, dtype=np.int32)
    mat = (cands[:, None] * gammas[None, :].astype(np.int64)) % Q
    G = len(gammas)
    cand_hw = np.zeros_like(mat, dtype=np.float32)
    for gi in range(G):
        cand_hw[:, gi] = [bin(int(v)).count("1") for v in mat[:, gi]]
    score = -((cand_hw - pred_per_g[None, :]) ** 2).sum(axis=1)
    true_idx = (true_f % Q) - 1
    rank = int((score > score[true_idx]).sum())
    pct = 100.0 * (1.0 - rank / (Q - 1))
    return rank, pct


def main() -> int:
    z = np.load(ROOT / "traces/ntruplus768/phase3/multikey_hw1.npz",
                allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    sk_blobs = z["sk_blobs"]
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    K, L, G, N, T = traces.shape
    W = 10

    f_arr = np.zeros((K, 768), dtype=np.int16)
    for ki in range(K):
        f_arr[ki] = center(from_bytes(bytes(sk_blobs[ki])[:POLYBYTES]))

    print(f"[INFO] traces (K, L, G, N, T) = {traces.shape}")

    # ------------------------------------------------------------------
    # T1: within-key, split N=8 → 4 train / 4 test
    # ------------------------------------------------------------------
    print("\n[T1] within-key recovery (N=4 train / N=4 test, same key)")
    t1_pcts = []
    for held_key in range(K):
        for li, lane in enumerate(lanes):
            poi = predict_poi(int(lane))
            t_lo = max(poi - W, 0); t_hi = min(poi + W + 1, T)
            ww = t_hi - t_lo
            for slot in range(4):
                flv = int(f_arr[held_key, D * lane + slot]) % Q
                if flv == 0:
                    continue
                # build (G, N) trace block for this key/lane
                blk = traces[held_key, li, ..., t_lo:t_hi]    # (G, N, W)
                labs = np.array([hw_unsigned((int(g) * flv) % Q) for g in gammas],
                                dtype=np.float32)
                # train: traces[:, :4], test: traces[:, 4:]
                Xtr = blk[:, :4].reshape(-1, ww)
                ytr = np.repeat(labs, 4)
                Xte = blk[:, 4:].reshape(-1, ww)
                mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0) + 1e-9
                Xtr_z = (Xtr - mu) / sd; Xte_z = (Xte - mu) / sd
                beta = ridge_fit(Xtr_z, ytr - ytr.mean(), lam=1.0)
                intercept = ytr.mean()
                pred_per_g = (Xte_z @ beta + intercept).reshape(G, 4).mean(axis=1)
                _, pct = score_rank(pred_per_g, gammas, flv)
                t1_pcts.append((int(lane), slot, pct))
    t1_arr = np.array([x[2] for x in t1_pcts])
    print(f"[T1] n={len(t1_arr)}  mean pctile = {t1_arr.mean():.2f}  "
          f"(random=50.00, max=100.00)")
    by_lane = {}
    for ln, sl, pc in t1_pcts:
        by_lane.setdefault(ln, []).append(pc)
    for ln, ps in sorted(by_lane.items()):
        print(f"  T1 lane {ln:>4}: mean {np.mean(ps):>6.2f}  (n={len(ps)})")

    # ------------------------------------------------------------------
    # T2: per-key z-normed cross-key recovery
    # ------------------------------------------------------------------
    print("\n[T2] cross-key with per-key trace normalization")
    # per-key normalize: each key's traces[ki] → subtract per-key mean,
    # divide by per-key std (shared across L, G, N; per-sample within key)
    traces_n = traces.copy()
    for ki in range(K):
        mu_k = traces_n[ki].mean(axis=(0, 1, 2), keepdims=True)
        sd_k = traces_n[ki].std(axis=(0, 1, 2), keepdims=True) + 1e-9
        traces_n[ki] = (traces_n[ki] - mu_k) / sd_k

    t2_pcts = []
    for held in range(K):
        train_keys = [k for k in range(K) if k != held]
        for li, lane in enumerate(lanes):
            poi = predict_poi(int(lane))
            t_lo = max(poi - W, 0); t_hi = min(poi + W + 1, T)
            ww = t_hi - t_lo
            train_feats = traces_n[train_keys, li, ..., t_lo:t_hi].reshape(-1, ww)
            mu = train_feats.mean(axis=0); sd = train_feats.std(axis=0) + 1e-9
            X = (train_feats - mu) / sd
            test_traces = traces_n[held, li, ..., t_lo:t_hi].reshape(G * N, ww)
            Xte = (test_traces - mu) / sd
            for slot in range(4):
                train_labs = np.zeros((len(train_keys), G), dtype=np.float32)
                for ti, ki in enumerate(train_keys):
                    flv = int(f_arr[ki, D * lane + slot]) % Q
                    for gi, g in enumerate(gammas):
                        train_labs[ti, gi] = hw_unsigned((int(g) * flv) % Q)
                ytr = np.repeat(train_labs.reshape(-1), N).astype(np.float32)
                beta = ridge_fit(X, ytr - ytr.mean(), lam=1.0)
                intercept = ytr.mean()
                pred_per_trace = Xte @ beta + intercept
                pred_per_g = pred_per_trace.reshape(G, N).mean(axis=1)
                true_f = int(f_arr[held, D * lane + slot]) % Q
                if true_f == 0:
                    continue
                _, pct = score_rank(pred_per_g, gammas, true_f)
                t2_pcts.append((int(lane), slot, pct))
    t2_arr = np.array([x[2] for x in t2_pcts])
    print(f"[T2] n={len(t2_arr)}  mean pctile = {t2_arr.mean():.2f}")
    by_lane2 = {}
    for ln, sl, pc in t2_pcts:
        by_lane2.setdefault(ln, []).append(pc)
    for ln, ps in sorted(by_lane2.items()):
        print(f"  T2 lane {ln:>4}: mean {np.mean(ps):>6.2f}  (n={len(ps)})")

    # ------------------------------------------------------------------
    # T3: single-sample (PoI only) cross-key
    # ------------------------------------------------------------------
    print("\n[T3] cross-key, single-sample PoI regression (no window)")
    t3_pcts = []
    for held in range(K):
        train_keys = [k for k in range(K) if k != held]
        for li, lane in enumerate(lanes):
            poi = predict_poi(int(lane))
            train_x = traces[train_keys, li, ..., poi].reshape(-1)  # (3*G*N,)
            test_x = traces[held, li, ..., poi].reshape(-1)         # (G*N,)
            for slot in range(4):
                train_labs = np.zeros((len(train_keys), G), dtype=np.float32)
                for ti, ki in enumerate(train_keys):
                    flv = int(f_arr[ki, D * lane + slot]) % Q
                    for gi, g in enumerate(gammas):
                        train_labs[ti, gi] = hw_unsigned((int(g) * flv) % Q)
                ytr = np.repeat(train_labs.reshape(-1), N).astype(np.float32)
                # simple linear fit: y = a + b * x
                xm = train_x.mean(); ym = ytr.mean()
                cov = ((train_x - xm) * (ytr - ym)).mean()
                var = ((train_x - xm) ** 2).mean() + 1e-12
                b = cov / var
                a = ym - b * xm
                pred_per_g = (a + b * test_x).reshape(G, N).mean(axis=1)
                true_f = int(f_arr[held, D * lane + slot]) % Q
                if true_f == 0:
                    continue
                _, pct = score_rank(pred_per_g, gammas, true_f)
                t3_pcts.append((int(lane), slot, pct))
    t3_arr = np.array([x[2] for x in t3_pcts])
    print(f"[T3] n={len(t3_arr)}  mean pctile = {t3_arr.mean():.2f}")
    by_lane3 = {}
    for ln, sl, pc in t3_pcts:
        by_lane3.setdefault(ln, []).append(pc)
    for ln, ps in sorted(by_lane3.items()):
        print(f"  T3 lane {ln:>4}: mean {np.mean(ps):>6.2f}  (n={len(ps)})")

    # ------------------------------------------------------------------
    # T4: cross-key correlation between per-key trace and per-key labels
    #     at the predicted PoI sample. If signal is sk-universal, the
    #     pooled cross-key (trace[poi], HW) corr should be > random.
    # ------------------------------------------------------------------
    print("\n[T4] pooled cross-key |corr(trace[poi], HW(γ·f mod q))|")
    for li, lane in enumerate(lanes):
        poi = predict_poi(int(lane))
        for slot in range(4):
            xs = []; ls = []
            for ki in range(K):
                flv = int(f_arr[ki, D * lane + slot]) % Q
                if flv == 0:
                    continue
                xs.append(traces[ki, li, ..., poi].reshape(-1))
                lab = np.array([hw_unsigned((int(g) * flv) % Q) for g in gammas],
                               dtype=np.float32)
                ls.append(np.repeat(lab, N))
            x = np.concatenate(xs); y = np.concatenate(ls)
            corr = float(np.corrcoef(x, y)[0, 1])
            print(f"  T4 lane {lane:>4} slot {slot}: pooled corr = {corr:+.3f}  "
                  f"(n={len(x)})")
        if li >= 2:
            break  # first 3 lanes only — diagnostic, not exhaustive

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
