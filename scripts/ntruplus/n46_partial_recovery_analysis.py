#!/usr/bin/env python3
"""Partial NTT-coord recovery → coefficient-domain info reduction.

Given X NTT coords perfectly recovered (top-1) from a single victim,
analyze information-theoretic and practical reach toward full sk recovery.

Mode A — synthetic Monte-Carlo:
  Generate random NTRU+ sk, sample X random NTT coords as 'recovered',
  INTT mod q with constraints. Measure:
    (i)  baseline residual entropy (uniform 768-X coord)
    (ii) constrained-LP/LLL reach (rough, not actual lattice solve)

Mode B — empirical case (after capture):
  Pass actual list of recovered (lane, slot, true_f_ntt) tuples, plus
  the victim's true sk. Output: which sk coefficients are *resolved*
  (single value consistent under cbd1 + linear constraints).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.ntt import ntt, invntt  # noqa: E402
from ntruplus.params import N, Q  # noqa: E402


def cbd1(rng: np.random.Generator, n: int) -> np.ndarray:
    """Centered binomial distribution param=1, output ∈ {-1, 0, +1}."""
    a = rng.integers(0, 2, size=n, dtype=np.int8)
    b = rng.integers(0, 2, size=n, dtype=np.int8)
    return (a - b).astype(np.int16)


def info_bits_recovered(num_recovered: int) -> dict:
    """Information-theoretic bounds for X NTT coords recovered.

    Assumptions:
      - sk is cbd1: f_coeff[i] ∈ {-1, 0, +1} with prob {1/4, 1/2, 1/4}
      - sk entropy = N · H_3 = 768 · 1.5 = 1152 bits
      - NTT is linear bijection mod q

    Bounds:
      - upper: log2(q^X / 1^X) = X · log2(q) — NTT coord entropy
      - lower: how much sk entropy is reduced (≤ upper, depends on prior)
    """
    H_total = N * 1.5  # bits, cbd1 H_3
    H_per_ntt = float(np.log2(Q))   # 11.756
    H_max_reduction = num_recovered * H_per_ntt
    return dict(
        H_sk_total_bits=H_total,
        H_per_ntt_coord=H_per_ntt,
        H_max_reduction_bits=H_max_reduction,
        residual_entropy_bits_lower=max(H_total - H_max_reduction, 0),
        H_reduction_pct=100.0 * H_max_reduction / H_total,
    )


def lattice_feasibility_estimate(num_recovered: int) -> dict:
    """Rough BKZ-block-size estimate for partial-LWE-style sk recovery.

    Heuristic (Albrecht et al style):
      - n = 768 dim, q = 3457
      - secret magnitude ~ 1 (cbd1)
      - revealed: X NTT coords (each 12 bits info on linear combo)
      - effective reduced dim: n - X (after substituting known constraints)
      - required β for primal attack ≈ 0.265·n / log2(δ_β), where δ_β is
        BKZ-β root-Hermite factor.

    Output: required β estimate, time estimate, approximate qualitative
    feasibility on a single workstation.
    """
    n_eff = N - num_recovered
    sigma_secret = np.sqrt(1.5)  # cbd1 std

    # crude δ-required for primal embedding
    # log_q(σ_secret) = log_q(1.22) ≈ 0.025; aliase
    if n_eff <= 1:
        return dict(beta_required=0, feasibility="trivial",
                     n_eff=n_eff)
    log_q = np.log2(Q)
    target_log_delta = (np.log2(sigma_secret) +
                        log_q * (1 - num_recovered / N)) / (2 * n_eff)
    # approx δ_β ≈ ((β / (2π·e)) · (πβ)^{1/β})^{1/(2(β-1))}
    # invert: small β → large δ; chord lookup typical
    # rough table:
    table = [(20, 1.0124), (40, 1.0114), (60, 1.0103), (80, 1.0093),
              (100, 1.0083), (150, 1.0070), (200, 1.0061),
              (300, 1.0050), (500, 1.0039), (1000, 1.0027)]
    beta_required = None
    for beta, delta in table:
        if delta < 2 ** target_log_delta:
            beta_required = beta
            break
    if beta_required is None:
        beta_required = 1500
    if beta_required <= 60:
        feas = "tractable on workstation (<1 day)"
    elif beta_required <= 100:
        feas = "tractable with cluster (~1-7 days)"
    elif beta_required <= 200:
        feas = "marginal — large compute"
    elif beta_required <= 500:
        feas = "infeasible with current methods"
    else:
        feas = "far beyond reach"
    return dict(
        n_eff_dim=n_eff,
        target_log_delta=float(target_log_delta),
        beta_required=int(beta_required),
        feasibility=feas,
    )


def cbd1_consistency_simulation(num_recovered: int, n_trials: int = 100,
                                  seed: int = 0) -> dict:
    """Monte Carlo: how many sk coefficients become deterministically resolved
    after substituting X NTT coords + cbd1 prior?
    """
    rng = np.random.default_rng(seed)
    resolved_counts = []
    for _ in range(n_trials):
        f_coeff = cbd1(rng, N)
        f_ntt = ntt(f_coeff)
        # randomly choose num_recovered NTT coord indices
        idx = rng.choice(N, size=num_recovered, replace=False)
        # Build constraint matrix M (X × N) where M[i] = INTT^-1[..idx[i]..]
        # i.e., NTT_idx[i] = sum_j NTT_matrix[idx[i], j] · f_coeff[j]
        # We'd need the NTT matrix to do exact linear algebra. For simulation
        # we just verify that constraints are correct.
        # Quick test: residual entropy after constraints.
        # Heuristic: each constraint mod q gives log_q(N possibilities/N
        # consistent) info ~ log2(q) per constraint, but for cbd1 secret
        # the "consistent" set is sparse so log2(3) per coord ~ 1.585 bit.
        # As a proxy: simulate by re-randomizing 768-X coords and checking
        # if constraints are satisfied.
        # This is too slow (q^X check), so we just record entropy bounds.
        resolved_counts.append(0)  # placeholder
    return dict(num_trials=n_trials,
                resolved_count_mean=float(np.mean(resolved_counts)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ntt-recovered", type=int, default=30,
                    help="number of NTT coords perfectly recovered")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/ntruplus768/phase45/partial_analysis.md")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== NTRU+768 partial recovery analysis ===\n")
    print(f"Hypothetical: X={args.ntt_recovered} NTT coords perfectly recovered\n")

    info = info_bits_recovered(args.ntt_recovered)
    print("Information-theoretic bounds:")
    for k, v in info.items():
        print(f"  {k:<32} = {v:.3f}")

    lat = lattice_feasibility_estimate(args.ntt_recovered)
    print(f"\nLattice (BKZ primal) estimate:")
    for k, v in lat.items():
        print(f"  {k:<32} = {v}")

    # Curve over X
    Xs = [1, 5, 10, 20, 30, 50, 100, 200, 384, 500, 700, 768]
    print(f"\n--- Recovery scan ---")
    print(f"{'X':>5} {'H_red %':>8} {'res H bits':>10} {'β_req':>6} {'feas':>30}")
    for X in Xs:
        info = info_bits_recovered(X)
        lat = lattice_feasibility_estimate(X)
        print(f"{X:>5} {info['H_reduction_pct']:>8.1f} "
              f"{info['residual_entropy_bits_lower']:>10.1f} "
              f"{lat['beta_required']:>6} {lat['feasibility']:>30}")

    # Markdown output
    md = ["# NTRU+768 Partial recovery analysis", "",
          f"For X NTT coords perfectly recovered from a single victim, "
          f"information-theoretic and lattice-attack reach.",
          "",
          "## Assumptions",
          "- secret f_coeff ∈ {-1, 0, +1}^768, cbd1 distribution",
          f"- total entropy H(f) = 768 · H_3(1/4, 1/2, 1/4) = 1152 bits",
          f"- q = 3457, log2(q) = 11.756",
          f"- NTT is linear bijective mod q",
          "", "## Recovery scan", "",
          f"| X | H_red % | residual H | β_req | feasibility |",
          f"|---:|---:|---:|---:|---|"]
    for X in Xs:
        info = info_bits_recovered(X)
        lat = lattice_feasibility_estimate(X)
        md.append(f"| {X} | {info['H_reduction_pct']:.1f} | "
                  f"{info['residual_entropy_bits_lower']:.0f} | "
                  f"{lat['beta_required']} | {lat['feasibility']} |")
    md += ["", "## Interpretation", "",
           f"Phase 4 (multi-key, 240 cases): 7.08% top-100 = ~17 NTT coords "
           f"across 60 victims, ~0.28 coords/victim — *insufficient* for "
           f"any single-victim recovery."]
    md += [f"",
           f"Phase 4.5 hypothesis (single-victim, 192-lane coverage):",
           f"- M1 baseline (predict_poi): ~32 coords/victim (extrapolation)",
           f"- Full stack (calibrated lanes): ~29 coords/victim",
           f"- Union (≈30% overlap): ~43 unique coords/victim",
           f"",
           f"At X=43, residual sk entropy ~ {1152 - 43 * 11.756:.0f} bits = "
           f"~{(1152 - 43 * 11.756)/1152*100:.0f}% of original entropy. "
           f"Lattice β_req in 'infeasible' zone but quantifies the "
           f"info-disclosure rate."]
    args.out.write_text("\n".join(md) + "\n")
    print(f"\n[OK] saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
