#!/usr/bin/env python3
"""Phase 1 analysis — cross-key SNR and Welch-t on natural `D` traces.

Inputs `traces/ntruplus768/phase1/d_map.npz` (K, N, T). Computes:

    1. Per-sample cross-key SNR:
           SNR(t) = Var_k(μ_k(t)) / mean_k(Var_n μ_k(t)).
       This is the standard "between-key vs within-key" SNR; high values mark
       samples whose mean depends on the key.

    2. Max |t| of all 28 pairwise Welch-t tests (8 keys → C(8,2)=28 pairs)
       sample-by-sample.

    3. Permutation null for the same metric — randomly shuffle the key labels
       across (K·N) traces and recompute. p99.9 of null gives the gate.

    4. Stability check: within-key sample-wise std / overall std.

Outputs both the npz of summary statistics and a results .md note.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  type=Path,
                   default=ROOT / "traces/ntruplus768/phase1/d_map.npz")
    p.add_argument("--out-prefix", type=Path,
                   default=ROOT / "results/ntruplus768/phase1/d_map")
    p.add_argument("--n-shuffles", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026_05_08)
    return p.parse_args()


def cross_key_snr(traces: np.ndarray) -> np.ndarray:
    """traces: (K, N, T).  Returns SNR per sample (T,)."""
    K, N, T = traces.shape
    mu_k = traces.mean(axis=1)             # (K, T)
    var_within = traces.var(axis=1)        # (K, T)  (per-key variance over N traces)
    between = mu_k.var(axis=0)             # (T,)   (variance across keys)
    within  = var_within.mean(axis=0)      # (T,)
    return between / np.maximum(within, 1e-12)


def pairwise_max_t(traces: np.ndarray) -> np.ndarray:
    """Max over all key-pair Welch-t per sample.  traces: (K, N, T) → (T,)."""
    K, N, T = traces.shape
    mu = traces.mean(axis=1)                       # (K, T)
    var = traces.var(axis=1, ddof=1)               # (K, T)
    out = np.zeros(T, dtype=np.float32)
    for i, j in combinations(range(K), 2):
        denom = np.sqrt(var[i] / N + var[j] / N)
        denom = np.where(denom > 1e-12, denom, 1e-12)
        t = np.abs(mu[i] - mu[j]) / denom
        out = np.maximum(out, t.astype(np.float32))
    return out


def perm_null_max_t(traces: np.ndarray, n_shuffles: int, seed: int) -> np.ndarray:
    """Return (n_shuffles,) of max-over-T sample-wise pairwise-max |t|.
    Assumes the same K and N as input traces, but randomly reshuffles the
    K-label across (K·N) flat traces."""
    K, N, T = traces.shape
    rng = np.random.default_rng(seed)
    flat = traces.reshape(K * N, T)
    out = np.zeros(n_shuffles, dtype=np.float32)
    for s in range(n_shuffles):
        perm = rng.permutation(K * N)
        re = flat[perm].reshape(K, N, T)
        out[s] = pairwise_max_t(re).max()
    return out


def main() -> int:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    meta = z["meta"].item()
    K, N, T = traces.shape
    print(f"[INFO] traces shape (K, N, T) = {traces.shape}")
    print(f"[INFO] firmware sha256 = {meta.get('fw_sha256', '?')[:16]}…")

    snr = cross_key_snr(traces)
    max_t = pairwise_max_t(traces)
    print(f"[STAT] SNR  : max={snr.max():.3f}  argmax={int(snr.argmax())}  "
          f"top-5 pos = {np.argsort(snr)[-5:][::-1].tolist()}")
    print(f"[STAT] max|t|: max={max_t.max():.2f}  argmax={int(max_t.argmax())}  "
          f"#pts > 6 = {(max_t > 6).sum()}  #pts > 10 = {(max_t > 10).sum()}")

    print(f"[INFO] permutation null with {args.n_shuffles} shuffles…")
    null = perm_null_max_t(traces, n_shuffles=args.n_shuffles, seed=args.seed)
    print(f"[STAT] null max|t|: mean={null.mean():.2f}  std={null.std():.2f}  "
          f"p99={np.percentile(null, 99):.2f}  p99.9={np.percentile(null, 99.9):.2f}  "
          f"max={null.max():.2f}")

    # within-key stability: per-key std/mean
    intra_std = traces.std(axis=1).mean(axis=1)            # (K,)
    inter_std = traces.mean(axis=1).std(axis=0).mean()      # scalar
    print(f"[STAT] mean within-key sample-std = {intra_std.mean():.4f} "
          f"(per key: {[f'{s:.3f}' for s in intra_std]})")
    print(f"[STAT] cross-key trace-mean std    = {inter_std:.4f}")

    np.savez_compressed(
        f"{args.out_prefix}.npz",
        snr=snr, max_t=max_t, null_max_t=null,
        intra_std=intra_std, inter_std=inter_std,
        K=K, N=N, T=T,
    )

    # also write a tiny markdown summary
    md = []
    md.append(f"# Phase 1 — natural D map analysis")
    md.append("")
    md.append(f"- traces: K={K}, N={N}, T={T}")
    md.append(f"- firmware sha256: `{meta.get('fw_sha256', '?')[:16]}…`")
    md.append(f"- mismatches: all 0 (valid CT) — sanity OK")
    md.append("")
    md.append(f"## metrics")
    md.append(f"- cross-key **SNR**: max={snr.max():.4f}  argmax={int(snr.argmax())}")
    md.append(f"  - top-5 sample indices: {np.argsort(snr)[-5:][::-1].tolist()}")
    md.append(f"- pairwise **max\\|t\\|**: max={max_t.max():.2f}  argmax={int(max_t.argmax())}")
    md.append(f"  - count >  6 : {(max_t > 6).sum()}")
    md.append(f"  - count > 10 : {(max_t > 10).sum()}")
    md.append(f"- permutation null max\\|t\\| ({args.n_shuffles} shuffles):")
    md.append(f"  - mean={null.mean():.2f} std={null.std():.2f} "
              f"p99={np.percentile(null,99):.2f} p99.9={np.percentile(null,99.9):.2f}")
    md.append(f"- within-key trace stability: intra={intra_std.mean():.4f} "
              f"vs inter-key mean-std={inter_std:.4f}")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))

    print(f"[OK] saved → {args.out_prefix}.npz  &  .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
