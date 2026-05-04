#!/usr/bin/env python3
"""PCO pilot 분석 — 'D' trace 의 어느 단계 (early=indcpa_dec, late=FO/hash) 에서
µ' 가 leak 되는지, 1-bit classifier 가 cross-design 작동하는지 평가.

질문 1: max|t| 위치 — early (≤ 50%) 또는 late (≥ 50%)?
  - early: indcpa_dec 의 µ' bit leak (이미 알던 것)
  - late: indcpa_enc/SHAKE 의 operand-power leak (PCO 의 핵심)

질문 2: 1-bit classifier accuracy
  - same-target session 안에서 train/test split
  - "이 trace 가 design A 인가 B 인가?" 분류기 — 100% 정확하면 µ' 정보 충분히 leak

질문 3: HW(µ') leak 의 양적 측정
  - design HW 가 0, 39, 31 → 3 designs 의 *pairwise* TVLA 비교
  - HW 차이 큰 pair 의 |t| 가 더 강하면 HW model 적합
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.zeros(a.shape[1], dtype=np.float64)
    ma = a.mean(axis=0); mb = b.mean(axis=0)
    va = a.var(axis=0, ddof=1).clip(min=1e-12)
    vb = b.var(axis=0, ddof=1).clip(min=1e-12)
    return (ma - mb) / np.sqrt(va / a.shape[0] + vb / b.shape[0])


def main() -> int:
    d = np.load("traces/pco_pilot.npz", allow_pickle=True)
    T = d["traces"]
    flags = d["mismatch_flags"]
    meta = d["meta"].item()
    labels = np.asarray(meta["label_design"])
    n_total, n_samples = T.shape
    designs = list(meta["design_names"])
    mu_hw = meta["mu_hw_per_design"]
    print(f"[ANALYSIS] {n_total} traces, {n_samples} samples")
    print(f"  designs: {designs}, HW: {mu_hw}")
    print(f"  mismatch flags (1 = reject): {int(flags.sum())}/{n_total} (all reject expected)\n")

    # === 1. pairwise TVLA + max|t| location ===
    print("[1] Pairwise design TVLA (where in trace is the leak?)")
    pairs = [(designs[0], designs[1]), (designs[0], designs[2]), (designs[1], designs[2])]
    Path("results").mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 9), sharex=True)
    for i, (a, b) in enumerate(pairs):
        Ta = T[labels == a]; Tb = T[labels == b]
        t = welch_t(Ta, Tb)
        idx = int(np.argmax(np.abs(t)))
        max_t = float(np.abs(t).max())
        n_above = int((np.abs(t) > 4.5).sum())
        early_max = float(np.abs(t[:n_samples // 2]).max())
        late_max = float(np.abs(t[n_samples // 2:]).max())
        loc = "EARLY" if idx < n_samples // 2 else "LATE"
        print(f"  {a:10s} vs {b:10s}: max|t|={max_t:5.2f} @ sample {idx:>5d} ({loc}), "
              f"early_max={early_max:.2f}, late_max={late_max:.2f}, |t|>4.5: {n_above}")
        axes[i].plot(np.abs(t), lw=0.4)
        axes[i].axhline(4.5, color="r", ls="--", lw=0.5)
        axes[i].axvline(n_samples // 2, color="k", ls=":", lw=0.5,
                        label=f"early/late boundary @ {n_samples//2}")
        axes[i].set_title(f"{a} vs {b} (HW {mu_hw[a]} vs {mu_hw[b]}, max|t|={max_t:.2f} @ {idx})")
        axes[i].set_ylabel("|t|")
        axes[i].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig("results/pco_pilot_tvla.png", dpi=120)
    plt.close(fig)
    print("  → results/pco_pilot_tvla.png")

    # === 2. 1-bit classifier (single-trace) accuracy ===
    print("\n[2] Single-trace 1-bit classifier (same target, target-internal)")
    # For each pair: train threshold on first half, test on second half.
    rng = np.random.default_rng(42)
    for a, b in pairs:
        Ta = T[labels == a]; Tb = T[labels == b]
        # train/test split (50/50)
        na, nb = len(Ta), len(Tb)
        na_tr = na // 2; nb_tr = nb // 2
        ord_a = rng.permutation(na); ord_b = rng.permutation(nb)
        Ta_tr, Ta_te = Ta[ord_a[:na_tr]], Ta[ord_a[na_tr:]]
        Tb_tr, Tb_te = Tb[ord_b[:nb_tr]], Tb[ord_b[nb_tr:]]
        # train: Welch-t on training set, pick PoI
        t_tr = welch_t(Ta_tr, Tb_tr)
        poi = int(np.argmax(np.abs(t_tr)))
        # threshold: midpoint of mean(Ta_tr[:, poi]) and mean(Tb_tr[:, poi])
        mu_a = float(Ta_tr[:, poi].mean())
        mu_b = float(Tb_tr[:, poi].mean())
        thr = (mu_a + mu_b) / 2.0
        sign_pred_a = +1 if mu_a > mu_b else -1
        # test: predict label per trace
        def predict(Tx):
            return np.where(sign_pred_a * (Tx[:, poi] - thr) > 0,
                            a, b)
        pred_a = predict(Ta_te)
        pred_b = predict(Tb_te)
        acc_a = float((pred_a == a).mean())
        acc_b = float((pred_b == b).mean())
        acc = (acc_a + acc_b) / 2.0
        print(f"  {a:10s} vs {b:10s}: PoI sample {poi:>5d}, threshold accuracy = {acc*100:.1f}% "
              f"(class {a}: {acc_a*100:.1f}%, class {b}: {acc_b*100:.1f}%)")

    # === 3. HW model — does |t| scale with HW difference? ===
    print("\n[3] HW(µ') model check")
    print(f"  HW table: {mu_hw}")
    pairs_with_hw = [(a, b, abs(mu_hw[a] - mu_hw[b])) for a, b in pairs]
    print("  HW diff | max|t|")
    for a, b, hw_diff in pairs_with_hw:
        Ta = T[labels == a]; Tb = T[labels == b]
        t = welch_t(Ta, Tb)
        print(f"  {hw_diff:>3d}     | {float(np.abs(t).max()):5.2f}  ({a} vs {b})")

    # === 4. Verdict ===
    print("\n[4] Verdict — late-trace leak indicator")
    # 우리 'D' command 가 indcpa_dec + indcpa_enc + hash 모두 포함.
    # 'D' command 의 indcpa_dec phase = trace 의 ~첫 1/3 정도 (clkgen_x1 에서)
    # Late trace (≥ 1/2 위치) 에서 max|t| > 5 면 → FO/SHAKE 가 µ' 를 leak.
    z_a64 = welch_t(T[labels == designs[0]], T[labels == designs[1]])
    z_a192 = welch_t(T[labels == designs[0]], T[labels == designs[2]])
    a64_a192 = welch_t(T[labels == designs[1]], T[labels == designs[2]])
    late_zone = slice(n_samples // 2, n_samples)
    late_max = max(np.abs(z_a64[late_zone]).max(),
                   np.abs(z_a192[late_zone]).max(),
                   np.abs(a64_a192[late_zone]).max())
    early_max = max(np.abs(z_a64[:n_samples//2]).max(),
                    np.abs(z_a192[:n_samples//2]).max(),
                    np.abs(a64_a192[:n_samples//2]).max())
    print(f"  early (sample 0—{n_samples//2}) max|t| = {float(early_max):.2f}")
    print(f"  late  (sample {n_samples//2}—{n_samples}) max|t| = {float(late_max):.2f}")
    if late_max > 5.0:
        print("  ✓ Late-trace leak observable — FO/SHAKE phase carries µ'-dependent signal")
        print("    → MV-PC oracle viable. Next: full PCO pipeline.")
    elif late_max > 3.5:
        print("  △ Marginal late-trace signal — borderline. Need higher N or better PoI.")
    else:
        print("  ✗ No late-trace leak. FO/SHAKE phase appears constant-time-clean.")
        print("    → Pure MV-PC via SHAKE operand power not viable on this setup.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
