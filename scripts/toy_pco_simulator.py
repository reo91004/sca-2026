#!/usr/bin/env python3
"""Toy PCO simulator — 99% noisy 1-bit oracle 가정 하 trace cost 분석.

목적
----
Pilot 4 가 입증한 1-bit FO accept/reject oracle (99% cross-sk accuracy) 으로
표준 Ravi 2020 PCO 진행 시 sparse MLWR sk 복구에 필요한 trace count 추정.

가정
----
- target sk: ternary {-1, 0, +1}, sparse HW=70/256 per polynomial, k=2 polynomials (smaug1)
- oracle: per-query 1-bit answer with error rate ε = 0.01 (Pilot 4 보수적)
- query schedule (단순화): 각 sk coefficient s_i 의 sign 을 binary test
    Q1. "s_i ∈ {-1, 0}?"  (≥ 0 의 부정)
    Q2. "s_i ∈ { 0, +1}?" (≤ 0 의 부정)
  두 답 (a, b) 으로 sk[i] 결정:
    (T, T) → s_i = 0
    (F, T) → s_i = +1
    (T, F) → s_i = -1
    (F, F) → 모순 (오라클 노이즈)

Per-question majority vote over N samples → P(correct) = 1 - F_binom(N/2; N, 1-ε).

분석:
  per-coefficient correct probability vs N (queries per question)
  per-polynomial expected correct count vs N
  total queries to recover sk with > 99% confidence
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.special import betainc

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def majority_vote_correct_prob(p_single: float, n: int) -> float:
    """N noisy queries, majority vote correct probability.

    P(majority correct) = P(X >= ceil(N/2)) where X ~ Binom(N, p_single).
    For odd N, exact regularized incomplete beta. We use the regularized
    incomplete beta closed form: P(Binom(N, p) >= k) = I_p(k, N-k+1).
    """
    k = (n // 2) + 1  # majority threshold
    return float(betainc(k, n - k + 1, p_single))


def main() -> int:
    out_dir = _REPO / "results"
    out_dir.mkdir(exist_ok=True)

    eps_oracle = 0.01  # 1% error from Pilot 4 (99% accuracy)
    p_single = 1 - eps_oracle
    n_coefs_per_poly = 256
    k_poly = 2  # smaug1 k=2 polynomials
    hs = 70    # sparse HW per polynomial
    n_questions_per_coef = 2

    # === 1. Majority vote analysis: P(correct per question) vs N ===
    print(f"[1] Per-question accuracy (oracle ε={eps_oracle:.3f})")
    print(f"  N=1:  {majority_vote_correct_prob(p_single, 1)*100:.4f}%")
    print(f"  N=3:  {majority_vote_correct_prob(p_single, 3)*100:.6f}%")
    print(f"  N=5:  {majority_vote_correct_prob(p_single, 5)*100:.7f}%")
    print(f"  N=7:  {majority_vote_correct_prob(p_single, 7)*100:.8f}%")
    print(f"  N=11: {majority_vote_correct_prob(p_single, 11)*100:.9f}%")
    print(f"  N=21: {majority_vote_correct_prob(p_single, 21)*100:.11f}%")

    # === 2. Per-coefficient correct probability ===
    # Each coef requires 2 questions, each with N traces. Per coef = 2N traces.
    # P(coef correct) = P(both questions correct) = mvc(N)^2.
    # P(full poly correct) = P(coef correct)^256.
    # P(full sk correct, k=2 polys) = P(poly correct)^2.

    print(f"\n[2] Recovery probability vs N per question")
    print(f"  N | per-coef | per-poly (256 coefs) | full-sk (2 polys) | total trace count")
    Ns = [1, 3, 5, 7, 11, 15, 21, 31, 51, 101]
    rows = []
    for N in Ns:
        p_q = majority_vote_correct_prob(p_single, N)
        p_coef = p_q ** n_questions_per_coef
        p_poly = p_coef ** n_coefs_per_poly
        p_sk = p_poly ** k_poly
        total = N * n_questions_per_coef * n_coefs_per_poly * k_poly
        rows.append((N, p_q, p_coef, p_poly, p_sk, total))
        print(f"  {N:>2d} | {p_q*100:7.4f}% | {p_poly*100:14.6f}%       | {p_sk*100:14.6f}%   | {total:>5d}")

    # find smallest N s.t. P(full sk) > 0.99
    N_thresh = next((r[0] for r in rows if r[4] > 0.99), None)
    if N_thresh is not None:
        rec_total = next(r[5] for r in rows if r[0] == N_thresh)
        print(f"\n  → minimum N for P(full sk) > 99% : N={N_thresh}, total queries={rec_total}")
    else:
        print(f"\n  → P(full sk) > 99% not reached in tested N")

    # === 3. Time cost analysis ===
    print(f"\n[3] Time cost (capture + serial overhead)")
    capture_time_s = 3.3e-3   # samples 24400 / 7.4 MHz
    serial_overhead_s = 0.2   # per-trace 'I'+'L'+'D' + ack rounds, conservative
    per_trace_s = capture_time_s + serial_overhead_s
    print(f"  per-trace = capture {capture_time_s*1000:.1f}ms + serial {serial_overhead_s*1000:.0f}ms = {per_trace_s*1000:.0f}ms")

    if N_thresh is not None:
        elapsed_s = rec_total * per_trace_s
        print(f"  full sk recovery (Pilot 4 99% oracle): {rec_total:>5d} traces × {per_trace_s*1000:.0f}ms ≈ {elapsed_s:.1f}s = {elapsed_s/60:.1f} min")

    # === 4. Sensitivity to oracle accuracy ===
    print(f"\n[4] Sensitivity — required N to reach P(full sk) > 99% as oracle ε varies")
    print(f"  {'ε':>5s} | {'p_single':>9s} | {'N_min':>5s} | {'total queries':>12s} | {'time (min)':>9s}")
    for eps in [0.005, 0.01, 0.02, 0.05, 0.10]:
        p_s = 1 - eps
        for N in range(1, 200, 2):  # try odd N
            p_q = majority_vote_correct_prob(p_s, N)
            p_sk = (p_q ** n_questions_per_coef) ** (n_coefs_per_poly * k_poly)
            if p_sk > 0.99:
                total = N * n_questions_per_coef * n_coefs_per_poly * k_poly
                t_min = total * per_trace_s / 60
                print(f"  {eps*100:4.1f}% | {p_s*100:7.3f}% | {N:>5d} | {total:>12d} | {t_min:>9.1f}")
                break
        else:
            print(f"  {eps*100:4.1f}% | {p_s*100:7.3f}% | (N>200) | — | —")

    # === 5. Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    Ns_dense = np.arange(1, 102, 2)
    for eps in [0.005, 0.01, 0.02, 0.05]:
        p_s = 1 - eps
        p_qs = [majority_vote_correct_prob(p_s, N) for N in Ns_dense]
        p_sks = [(pq ** n_questions_per_coef) ** (n_coefs_per_poly * k_poly) for pq in p_qs]
        axes[0].plot(Ns_dense, [1 - p for p in p_sks], label=f"ε={eps*100:.1f}%")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("N traces per question")
    axes[0].set_ylabel("P(any sk error)")
    axes[0].axhline(0.01, color="r", ls="--", label="P(error)=1%")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Pilot 4 oracle ε=1% → N≈{N_thresh} per question")

    Ns_grid = [1, 3, 5, 7, 11, 15, 21, 31, 51, 101]
    rec_totals = [N * n_questions_per_coef * n_coefs_per_poly * k_poly for N in Ns_grid]
    times_s = [t * per_trace_s for t in rec_totals]
    axes[1].plot(Ns_grid, times_s, "o-")
    axes[1].set_xlabel("N traces per question")
    axes[1].set_ylabel("total recovery time (s)")
    axes[1].set_title(f"sk recovery time vs N per question (per-trace {per_trace_s*1000:.0f}ms)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "toy_pco_simulator.png", dpi=120)
    plt.close(fig)
    print(f"\n  → {out_dir / 'toy_pco_simulator.png'}")

    # === 6. Empirical Monte Carlo simulation (sanity vs analytical) ===
    print(f"\n[5] Empirical Monte Carlo simulation (10000 trials, oracle ε=1%, N=11 per question)")
    rng = np.random.default_rng(42)
    n_trials = 10000
    N = 11
    correct_count = 0
    for _ in range(n_trials):
        # ground truth sparse ternary sk
        sk = np.zeros(n_coefs_per_poly * k_poly, dtype=np.int8)
        for kp in range(k_poly):
            offsets = rng.choice(n_coefs_per_poly, size=hs, replace=False)
            signs = rng.choice([-1, +1], size=hs)
            sk[kp * n_coefs_per_poly + offsets] = signs

        # for each coef, simulate 2 questions × N noisy queries
        all_correct = True
        for i in range(len(sk)):
            true = sk[i]
            # Q1: "s_i in {-1, 0}" (i.e., s_i <= 0)
            q1_truth = int(true <= 0)
            q2_truth = int(true >= 0)
            answers_q1 = rng.binomial(1, q1_truth * (1 - eps_oracle) + (1 - q1_truth) * eps_oracle, size=N)
            answers_q2 = rng.binomial(1, q2_truth * (1 - eps_oracle) + (1 - q2_truth) * eps_oracle, size=N)
            decided_q1 = int(answers_q1.sum() > N / 2)
            decided_q2 = int(answers_q2.sum() > N / 2)
            # decode
            if decided_q1 == 1 and decided_q2 == 1:
                pred = 0
            elif decided_q1 == 0 and decided_q2 == 1:
                pred = 1
            elif decided_q1 == 1 and decided_q2 == 0:
                pred = -1
            else:
                pred = 99  # contradiction
            if pred != true:
                all_correct = False
                break
        if all_correct:
            correct_count += 1
    p_emp = correct_count / n_trials
    p_ana = (majority_vote_correct_prob(p_single, N) ** 2) ** (n_coefs_per_poly * k_poly)
    print(f"  empirical full-sk recovery success: {p_emp*100:.2f}%  ({correct_count}/{n_trials})")
    print(f"  analytical:                        {p_ana*100:.2f}%")
    print(f"  agreement: {abs(p_emp - p_ana):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
