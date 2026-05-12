#!/usr/bin/env python3
"""Phase 2 — sk/ct-byte HW correlation analysis on Phase 1 traces.

Threat-model role: positive control. We use sk dump as a *label oracle*
to decide whether byte-level Hamming-weight templates can predict the
secret given trace-only features. A pass tells us simple HW templates work
on this firmware; a fail does not exclude richer Phase 3 attacks.

Why this script supersedes a naive "max |corr| over (bytes × samples)"
analysis: with only K=8 distinct keys, the per-cell |corr| of two length-8
vectors saturates near 1 by chance. Taking max over a 64×5000 grid then
sees ≈p99.9 of the null at 0.998 — purely a multiple-testing artifact,
not signal. The right comparison is per-cell: does the *distribution of
cell-wise z-scores* under the real labels deviate from the null
distribution under shuffled labels?

Outputs per-cell |corr|, and reports:
    • count of cells whose real |corr| exceeds the per-cell null p99.9
      (i.e. survives a per-cell threshold),
    • the same count under the null itself (reference rate),
    • top-10 cells with positions, values, and per-cell z.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))


def hw(byte_array: np.ndarray) -> np.ndarray:
    arr = byte_array.astype(np.uint8)
    out = np.zeros_like(arr, dtype=np.int8)
    for s in range(8):
        out += ((arr >> s) & 1).astype(np.int8)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  type=Path,
                   default=ROOT / "traces/ntruplus768/phase1/d_map.npz")
    p.add_argument("--out-prefix", type=Path,
                   default=ROOT / "results/ntruplus768/phase2/skunpack")
    p.add_argument("--label-source", choices=("sk", "ct"), default="sk",
                   help="byte-HW label source: sk_blobs (sk-leak hypothesis) "
                        "or ct (ct-leak hypothesis using fingerprint as proxy).")
    p.add_argument("--byte-range", type=str, default="0:64")
    p.add_argument("--sample-range", type=str, default="0:5000")
    p.add_argument("--n-shuffles", type=int, default=500,
                   help="Permutation shuffles for per-cell null estimation.")
    p.add_argument("--seed", type=int, default=2026_05_08)
    return p.parse_args()


def parse_range(s: str) -> tuple[int, int]:
    a, b = s.split(":")
    return int(a), int(b)


def per_key_mean(traces: np.ndarray) -> np.ndarray:
    """traces (K, N, T) → (K, T)."""
    return traces.mean(axis=1)


def corr_per_cell(means: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """means (K, nt), labels (K, nb) → |corr| of shape (nb, nt).

    Vectorized: standardize means and labels, then dot.
    """
    K, nt = means.shape
    _, nb = labels.shape
    mz = means - means.mean(axis=0, keepdims=True)
    ms = mz / np.sqrt((mz ** 2).sum(axis=0, keepdims=True) + 1e-30)  # (K, nt)
    lz = labels - labels.mean(axis=0, keepdims=True)
    ls = lz / np.sqrt((lz ** 2).sum(axis=0, keepdims=True) + 1e-30)   # (K, nb)
    # corr[i, t] = sum_k ls[k, i] * ms[k, t]
    return np.abs(ls.T @ ms).astype(np.float32)                       # (nb, nt)


def get_ct_byte_labels(z: np.lib.npyio.NpzFile,
                       b_lo: int, b_hi: int) -> tuple[np.ndarray, str]:
    """Return per-key proxy CT bytes for the requested byte range.

    Phase 1 doesn't preserve the full ciphertext, only ct_fp (16-byte
    sha3-256 prefix) — but ct_fp depends entirely on the (random) ct, so
    HW(ct_fp[i]) is a valid CT-dependent label per key. We test cells
    [0, min(16, b_hi-b_lo)) of the fingerprint.
    """
    fp = z["ct_fps"]                       # (K, 16)
    K = fp.shape[0]
    nb = b_hi - b_lo
    if nb > 16:
        nb = 16
    out = fp[:, :nb].astype(np.uint8)
    note = (f"CT bytes proxied via sha3_256(ct)[:{nb}] fingerprint "
            f"(full ct not preserved in Phase 1 dump).")
    return out, note


def main() -> int:
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix = Path(f"{args.out_prefix}_{args.label_source}")

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)
    K, N, T = traces.shape
    print(f"[INFO] traces (K, N, T) = {traces.shape}")

    b_lo, b_hi = parse_range(args.byte_range)
    t_lo, t_hi = parse_range(args.sample_range)
    print(f"[INFO] byte-range [{b_lo}, {b_hi})  sample-range [{t_lo}, {t_hi})")

    means = per_key_mean(traces)[:, t_lo:t_hi]                    # (K, nt)

    if args.label_source == "sk":
        sk_blobs = z["sk_blobs"]
        bytes_arr = sk_blobs[:, b_lo:b_hi]
        label_note = f"sk_blobs[:, {b_lo}:{b_hi}] (NTT-domain f_ntt 12-bit packed)"
    else:
        bytes_arr, label_note = get_ct_byte_labels(z, b_lo, b_hi)
        b_lo, b_hi = 0, bytes_arr.shape[1]
    nb = bytes_arr.shape[1]
    nt = means.shape[1]
    print(f"[INFO] labels: {label_note}  →  shape (K, nb)={bytes_arr.shape}")

    hw_labels = hw(bytes_arr).astype(np.float32)                  # (K, nb)
    real = corr_per_cell(means, hw_labels)                        # (nb, nt)

    # Per-cell null distribution via key-permutation
    rng = np.random.default_rng(args.seed)
    print(f"[INFO] estimating per-cell null with {args.n_shuffles} shuffles…")
    n_cells = nb * nt
    null_acc = np.zeros((args.n_shuffles, nb, nt), dtype=np.float32)
    for s in range(args.n_shuffles):
        perm = rng.permutation(K)
        null_acc[s] = corr_per_cell(means, hw_labels[perm, :])
    null_mean = null_acc.mean(axis=0)
    null_std  = null_acc.std(axis=0)
    null_p999 = np.percentile(null_acc, 99.9, axis=0)              # per-cell p99.9
    z_per_cell = (real - null_mean) / np.maximum(null_std, 1e-6)

    # Multiple-testing-aware report
    survive = (real > null_p999)             # boolean (nb, nt)
    expected_under_null = 0.001 * n_cells    # by definition, ~0.1% of cells
    n_survive = int(survive.sum())

    # top-10 cells by z (ranked by raw z, not by |corr|)
    flat_idx = np.argsort(z_per_cell.ravel())[-10:][::-1]
    print(f"\n[STAT] real max |corr| = {real.max():.4f}")
    print(f"[STAT] cells exceeding per-cell null p99.9: {n_survive} "
          f"(expected ~{expected_under_null:.0f} under null = 0.1% of {n_cells})")
    print(f"[STAT] z_per_cell  max={z_per_cell.max():.2f}  "
          f"mean={z_per_cell.mean():.2f}  std={z_per_cell.std():.2f}")
    print(f"\n[STAT] top-10 cells by z:")
    print(f"  {'rank':>4} {'byte':>6} {'sample':>7} {'|corr|':>8} {'z':>7}")
    for rank, fi in enumerate(flat_idx, 1):
        bi, ti = np.unravel_index(int(fi), z_per_cell.shape)
        print(f"  {rank:>4d} {b_lo + int(bi):>6d} {t_lo + int(ti):>7d}  "
              f"{real[bi, ti]:>8.4f} {z_per_cell[bi, ti]:>7.2f}")

    # how many cells in the empirical null itself exceeded null p99.9?
    survive_null_per_shuffle = (null_acc > null_p999[None, :, :]).sum(axis=(1, 2))
    null_survive_mean = float(survive_null_per_shuffle.mean())
    print(f"\n[STAT] reference: in {args.n_shuffles} shuffles each had on "
          f"average {null_survive_mean:.1f} cells > per-cell p99.9 (sanity).")

    # Pass / fail gate
    if n_survive > 5 * expected_under_null and z_per_cell.max() > 4:
        verdict = "PASS"
    elif n_survive > 2 * expected_under_null:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    print(f"[STAT] gate verdict: {verdict}")

    np.savez_compressed(
        f"{out_prefix}.npz",
        real=real, null_mean=null_mean, null_std=null_std, null_p999=null_p999,
        z_per_cell=z_per_cell, survive=survive,
        b_lo=b_lo, b_hi=b_hi, t_lo=t_lo, t_hi=t_hi,
        label_source=args.label_source,
        n_survive=n_survive, expected=expected_under_null, verdict=verdict,
    )

    md = []
    md.append(f"# Phase 2 — byte-HW triage  (label = {args.label_source})")
    md.append("")
    md.append(f"- traces: K={K}, N={N}, T={T}")
    md.append(f"- byte-range [{b_lo}, {b_hi}), sample-range [{t_lo}, {t_hi})")
    md.append(f"- labels: {label_note}")
    md.append("")
    md.append(f"## per-cell null analysis ({args.n_shuffles} shuffles)")
    md.append(f"- real max\\|corr\\| = {real.max():.4f}")
    md.append(f"- cells > per-cell null p99.9: **{n_survive}** "
              f"(expected ≈ {expected_under_null:.0f}; "
              f"ratio = {n_survive / max(expected_under_null,1):.2f}×)")
    md.append(f"- z_per_cell: max = {z_per_cell.max():.2f}, "
              f"mean = {z_per_cell.mean():.2f}, std = {z_per_cell.std():.2f}")
    md.append(f"- gate verdict: **{verdict}**")
    md.append("")
    md.append("## top-10 cells by z")
    md.append("| rank | byte | sample | \\|corr\\| | z |")
    md.append("|---:|---:|---:|---:|---:|")
    for rank, fi in enumerate(flat_idx, 1):
        bi, ti = np.unravel_index(int(fi), z_per_cell.shape)
        md.append(f"| {rank} | {b_lo + int(bi)} | {t_lo + int(ti)} | "
                  f"{real[bi, ti]:.4f} | {z_per_cell[bi, ti]:.2f} |")
    Path(f"{out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
