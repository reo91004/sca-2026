#!/usr/bin/env python3
"""Branch R weak-recovery pressure from predicted latent `mu' histograms.

This script reuses the held-out-key ridge evaluator from
``s2_z_lowdim_analyze.py``. It does not read target responses. It asks a more
attack-facing question than the low-dimensional z-scores:

    if we predict `mu_byte_hw` for detector-grid chosen ciphertexts, how much
    sparse support candidate entropy would the predicted byte histogram remove?

The entropy number is intentionally conservative. It uses only the
``alpha=128, shift=0`` support detector and treats the predicted byte counts as
byte-local support constraints. It does not yet combine overlapping shifted
histograms, signs, public-key validation, or candidate enumeration.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.smaug.params import SMAUG1  # noqa: E402
from scripts.smaug.s2_z_lowdim_analyze import (  # noqa: E402
    Config,
    cached_labels,
    evaluate,
    load_dataset,
)


@dataclass(frozen=True)
class DesignIndex:
    alpha64_j0: int | None
    alpha128_j0: int | None
    alpha192_j0: int | None
    alpha128_shifts: tuple[tuple[int, int], ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--sample-range", default="1024:5000")
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--n-features", type=int, default=32)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument("--hs", type=int, default=SMAUG1.hs)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "s4_mu_recovery_pressure.txt",
    )
    return p.parse_args()


def log2_comb(n: int, k: int) -> float:
    if not (0 <= k <= n):
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def project_counts_to_sum(values: np.ndarray, total: int, cap: int = 8) -> np.ndarray:
    """Round byte counts to integers in [0, cap] with a fixed global sum."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, float(cap))
    out = np.floor(v).astype(np.int64)
    out = np.clip(out, 0, cap)
    need = int(total - out.sum())
    frac = v - np.floor(v)
    if need > 0:
        order = np.argsort(-frac, kind="stable")
        for idx in order:
            if need <= 0:
                break
            add = min(cap - int(out[idx]), need)
            out[idx] += add
            need -= add
    elif need < 0:
        order = np.argsort(frac, kind="stable")
        for idx in order:
            if need >= 0:
                break
            sub = min(int(out[idx]), -need)
            out[idx] -= sub
            need += sub
    if need != 0:
        raise ValueError(f"cannot project counts to total={total}; residual {need}")
    return out


def design_index(designs: list[list[tuple[int, int]]]) -> DesignIndex:
    found = {64: None, 128: None, 192: None}
    support_shifts = []
    for i, terms in enumerate(designs):
        if len(terms) != 1:
            continue
        coef, alpha = terms[0]
        if coef == 0 and alpha in found and found[alpha] is None:
            found[alpha] = i
        if alpha == 128 and coef % 8 == 0:
            support_shifts.append((i, coef // 8))
    return DesignIndex(found[64], found[128], found[192], tuple(support_shifts))


def support_entropy_from_byte_counts(counts: np.ndarray) -> float:
    counts_i = np.asarray(counts, dtype=np.int64)
    return float(sum(log2_comb(8, int(k)) for k in counts_i))


def log2_add(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    hi = max(a, b)
    lo = min(a, b)
    return hi + math.log2(1.0 + 2.0 ** (lo - hi))


def support_entropy_from_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    total: int,
    cap: int = 8,
) -> float:
    """log2 #support patterns with byte counts in [lower, upper] and sum=total."""
    lo = np.clip(np.asarray(lower, dtype=np.int64), 0, cap)
    hi = np.clip(np.asarray(upper, dtype=np.int64), 0, cap)
    if lo.shape != hi.shape:
        raise ValueError(f"interval shape mismatch: {lo.shape} vs {hi.shape}")
    dp = np.full(total + 1, float("-inf"), dtype=np.float64)
    dp[0] = 0.0
    for a, b in zip(lo.tolist(), hi.tolist()):
        nxt = np.full(total + 1, float("-inf"), dtype=np.float64)
        for used in range(total + 1):
            if dp[used] == float("-inf"):
                continue
            for k in range(int(a), int(b) + 1):
                if used + k <= total:
                    val = float(dp[used]) + log2_comb(cap, k)
                    nxt[used + k] = log2_add(float(nxt[used + k]), val)
        dp = nxt
    return float(dp[total])


def main() -> int:
    args = parse_args()
    ds = load_dataset(args.inputs, args.component)
    if args.sample_range:
        lo_s, hi_s = args.sample_range.split(":")
        lo = int(lo_s) if lo_s else 0
        hi = int(hi_s) if hi_s else ds.traces.shape[-1]
        ds = type(ds)(
            traces=ds.traces[..., lo:hi],
            sks=ds.sks,
            design_terms=ds.design_terms,
            design_c2=ds.design_c2,
            pkfps=ds.pkfps,
        )
    cfg = Config(args.block, args.n_features, args.ridge, args.feature_mode)
    real = evaluate(ds, "mu_byte_hw", cfg)
    pred = np.asarray(real["pred"], dtype=np.float64)
    true = np.asarray(real["true"], dtype=np.float64)
    s, d = ds.traces.shape[:2]
    pred = pred.reshape(s, d, -1)
    true = true.reshape(s, d, -1)

    idx = design_index(ds.design_terms)
    if idx.alpha128_j0 is None:
        raise SystemExit("need alpha=128, coef=0 design for support entropy")

    prior_support_bits = log2_comb(SMAUG1.n, args.hs)
    prior_ternary_bits = prior_support_bits + args.hs

    reductions = []
    oracle_reductions = []
    tolerant_reductions = []
    avg_reductions = []
    avg_tolerant_reductions = []
    support_l1 = []
    support_mae = []
    support_maxerr = []
    avg_support_l1 = []
    avg_support_mae = []
    pred_sums = []
    true_sums = []
    for key_i in range(s):
        pred_counts = pred[key_i, idx.alpha128_j0]
        true_counts = true[key_i, idx.alpha128_j0].astype(np.int64)
        projected = project_counts_to_sum(pred_counts, args.hs)
        pred_entropy = support_entropy_from_byte_counts(projected)
        true_entropy = support_entropy_from_byte_counts(true_counts)
        err = np.abs(projected - true_counts)
        tolerant_entropy = support_entropy_from_intervals(
            projected - err,
            projected + err,
            args.hs,
        )
        reductions.append(prior_support_bits - pred_entropy)
        oracle_reductions.append(prior_support_bits - true_entropy)
        tolerant_reductions.append(prior_support_bits - tolerant_entropy)
        support_l1.append(float(np.abs(projected - true_counts).sum()))
        support_mae.append(float(np.abs(projected - true_counts).mean()))
        support_maxerr.append(float(np.max(err)))
        pred_sums.append(int(projected.sum()))
        true_sums.append(int(true_counts.sum()))

        if idx.alpha128_shifts:
            pred_views = []
            true_views = []
            for design_i, shift_bytes in idx.alpha128_shifts:
                # Output byte b corresponds to secret byte (b - shift) mod 32.
                # Roll left by shift to recover canonical secret-byte order.
                pred_views.append(np.roll(pred[key_i, design_i], -shift_bytes))
                true_views.append(np.roll(true[key_i, design_i], -shift_bytes))
            pred_avg = np.mean(np.stack(pred_views), axis=0)
            true_avg = np.mean(np.stack(true_views), axis=0).round().astype(np.int64)
            projected_avg = project_counts_to_sum(pred_avg, args.hs)
            avg_entropy = support_entropy_from_byte_counts(projected_avg)
            avg_err = np.abs(projected_avg - true_avg)
            avg_tolerant_entropy = support_entropy_from_intervals(
                projected_avg - avg_err,
                projected_avg + avg_err,
                args.hs,
            )
            avg_reductions.append(prior_support_bits - avg_entropy)
            avg_tolerant_reductions.append(prior_support_bits - avg_tolerant_entropy)
            avg_support_l1.append(float(avg_err.sum()))
            avg_support_mae.append(float(avg_err.mean()))

    lines = [
        "Branch R mu-byte histogram recovery pressure",
        f"S={s}, D={d}, sample_range={args.sample_range}, cfg={cfg.name}",
        f"inputs={[p.name for p in args.inputs]}",
        "",
        "Low-dimensional prediction:",
        f"  exact={real['exact']:.4f}",
        f"  rounded_MAE={real['rounded_mae']:.4f}",
        f"  corr={real['corr']:.4f}",
        "",
        "Conservative support entropy from alpha=128, shift=0 byte counts:",
        f"  prior_support_bits=log2(C(256,{args.hs}))={prior_support_bits:.2f}",
        f"  prior_ternary_bits=prior_support_bits+{args.hs} signs={prior_ternary_bits:.2f}",
        f"  predicted_support_entropy_reduction_bits mean={np.mean(reductions):.2f} "
        f"std={np.std(reductions, ddof=1):.2f}",
        f"  oracle_support_entropy_reduction_bits mean={np.mean(oracle_reductions):.2f} "
        f"std={np.std(oracle_reductions, ddof=1):.2f}",
        f"  true_containing_tolerant_reduction_bits mean={np.mean(tolerant_reductions):.2f} "
        f"std={np.std(tolerant_reductions, ddof=1):.2f}",
        f"  projected_support_byte_count_L1 mean={np.mean(support_l1):.2f} "
        f"std={np.std(support_l1, ddof=1):.2f}",
        f"  projected_support_byte_count_MAE mean={np.mean(support_mae):.3f}",
        f"  projected_support_byte_count_max_error mean={np.mean(support_maxerr):.2f}",
        f"  projected_sum_range={min(pred_sums)}..{max(pred_sums)} "
        f"true_sum_range={min(true_sums)}..{max(true_sums)}",
        "  note=raw predicted reduction may exclude the true support; the tolerant",
        "       reduction inflates each byte interval just enough to include it.",
        "",
    ]

    if avg_reductions:
        lines.extend([
            "Shift-averaged support pressure (alpha=128, all byte-aligned shifts):",
            f"  n_shift_views={len(idx.alpha128_shifts)}",
            f"  raw_avg_support_reduction_bits mean={np.mean(avg_reductions):.2f} "
            f"std={np.std(avg_reductions, ddof=1):.2f}",
            f"  true_containing_avg_tolerant_reduction_bits mean={np.mean(avg_tolerant_reductions):.2f} "
            f"std={np.std(avg_tolerant_reductions, ddof=1):.2f}",
            f"  shift_avg_support_byte_count_L1 mean={np.mean(avg_support_l1):.2f} "
            f"std={np.std(avg_support_l1, ddof=1):.2f}",
            f"  shift_avg_support_byte_count_MAE mean={np.mean(avg_support_mae):.3f}",
            "",
        ])

    if idx.alpha64_j0 is not None and idx.alpha192_j0 is not None:
        plus_l1 = np.abs(np.rint(pred[:, idx.alpha64_j0]) - true[:, idx.alpha64_j0]).sum(axis=1)
        minus_l1 = np.abs(np.rint(pred[:, idx.alpha192_j0]) - true[:, idx.alpha192_j0]).sum(axis=1)
        lines.extend([
            "Detector histogram sanity (alpha=64/192, shift=0):",
            f"  alpha64 byte-count L1 mean={np.mean(plus_l1):.2f} std={np.std(plus_l1, ddof=1):.2f}",
            f"  alpha192 byte-count L1 mean={np.mean(minus_l1):.2f} std={np.std(minus_l1, ddof=1):.2f}",
            "",
        ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
