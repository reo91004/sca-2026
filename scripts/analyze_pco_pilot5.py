#!/usr/bin/env python3
"""PCO pilot 5 분석 — same-design cross-sk distinguishability.

핵심 검증:
  Q1. zero design (모든 sk 공유 µ'=0) 의 trace 가 sk 마다 cluster 되는가?
      → 이건 *z-leak* (rejection K = SHAKE(z, ct)) 검증. z 는 uniform 32B, sk 의
         sparse ternary 부분이 아니므로 sk recovery 에 직접 안 도움.
  Q2. a192 design (µ' = sk support indicator) 의 trace 가 sk 마다 cluster 되는가?
      → 이건 *µ'-leak* 검증. µ' bit = (sk[i] != 0). 직접 sparse sk support 추출 가능.
  Q3. zero 의 cluster 강도 << a192 의 cluster 강도 → µ' 가 dominant leak (good for SCA)
       zero ≈ a192 → z 가 dominant leak (only K_rej-related)
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


def welch_t(a, b):
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.zeros(a.shape[1], dtype=np.float64)
    ma = a.mean(0); mb = b.mean(0)
    va = a.var(0, ddof=1).clip(min=1e-12); vb = b.var(0, ddof=1).clip(min=1e-12)
    return (ma - mb) / np.sqrt(va / a.shape[0] + vb / b.shape[0])


def main() -> int:
    out_dir = _REPO / "results"
    out_dir.mkdir(exist_ok=True)
    d = np.load(_REPO / "traces/pco_pilot5.npz", allow_pickle=True)
    T = d["traces"]
    flags = d["mismatch_flags"]
    meta = d["meta"].item()
    n_total, n_samples = T.shape
    n_sessions = meta["n_sessions"]
    n_per_design = meta["n_per_design"]
    label_design = np.asarray(meta["label_design"])
    label_session = np.asarray(meta["label_session"])

    sk_hexes = meta["sk_pke_hex_per_session"]
    print(f"[ANALYSIS] {n_total} traces, {n_samples} samples, {n_sessions} sessions")
    for s in range(n_sessions):
        print(f"  session {s}: sk={sk_hexes[s][:16]}, "
              f"HW(µ'_zero)={meta['mu_zero_hw_per_session'][s]}, "
              f"HW(µ'_a192)={meta['mu_a192_hw_per_session'][s]}")

    # split
    groups = {}
    for design in ("zero", "a192"):
        for s in range(n_sessions):
            mask = (label_design == design) & (label_session == s)
            groups[(design, s)] = T[mask]
            print(f"  {design} sk{s}: {int(mask.sum())} traces")

    # === Q1+Q2: pairwise inter-sk same-design Welch-t ===
    print("\n[1] Inter-sk SAME-design Welch-t (cluster strength)")
    print("    same design (same µ' under that ct, but DIFFERENT sk → different z)")
    LATE_START = 20000
    POI = 23284  # pilot 3 정한 LATE PoI

    for design in ("zero", "a192"):
        print(f"\n  Design '{design}':")
        max_ts = []
        for i in range(n_sessions):
            for j in range(i + 1, n_sessions):
                Ti = groups[(design, i)]
                Tj = groups[(design, j)]
                t = welch_t(Ti, Tj)
                max_t_late = float(np.abs(t[LATE_START:]).max())
                idx_late = int(LATE_START + np.argmax(np.abs(t[LATE_START:])))
                t_at_poi = float(np.abs(t[POI]))
                max_ts.append(max_t_late)
                print(f"    sk{i} vs sk{j}: late max|t|={max_t_late:5.2f} @ {idx_late:>5d}, "
                      f"|t|@POI={t_at_poi:5.2f}")
        avg_late = float(np.mean(max_ts))
        print(f"    → avg late max|t| = {avg_late:.2f}")

    # === Q3: cluster strength comparison ===
    print("\n[2] Comparison — does µ' (a192) leak more than z (zero)?")
    avg_zero, avg_a192 = [], []
    for i in range(n_sessions):
        for j in range(i + 1, n_sessions):
            avg_zero.append(float(np.abs(welch_t(groups[("zero", i)], groups[("zero", j)]))[LATE_START:].max()))
            avg_a192.append(float(np.abs(welch_t(groups[("a192", i)], groups[("a192", j)]))[LATE_START:].max()))
    print(f"  zero (z-only leak): avg={np.mean(avg_zero):5.2f}, std={np.std(avg_zero):5.2f}")
    print(f"  a192 (z + µ' leak):  avg={np.mean(avg_a192):5.2f}, std={np.std(avg_a192):5.2f}")
    ratio = np.mean(avg_a192) / max(np.mean(avg_zero), 1e-6)
    print(f"  ratio (a192/zero) = {ratio:.2f}")
    if ratio > 1.5:
        print(f"  ✓ µ' (sk-support) leak dominates → sparse sk SUPPORT recoverable")
    elif ratio > 0.8:
        print(f"  △ µ' and z leak similar — only z-recoverable directly")
    else:
        print(f"  ✗ z leak dominates — sk SUPPORT not directly recoverable")

    # === multi-class ANOVA-style: 4 sk classifier within each design ===
    print("\n[3] 4-way sk classifier (which sk produced this trace?)")
    rng = np.random.default_rng(42)
    for design in ("zero", "a192"):
        # 50/50 train/test split per sk
        Xs_tr, ys_tr, Xs_te, ys_te = [], [], [], []
        for s in range(n_sessions):
            G = groups[(design, s)]
            n = len(G)
            order = rng.permutation(n)
            Xs_tr.append(G[order[:n//2]]); ys_tr.append(np.full(n//2, s))
            Xs_te.append(G[order[n//2:]]); ys_te.append(np.full(n - n//2, s))
        Xtr = np.concatenate(Xs_tr); ytr = np.concatenate(ys_tr)
        Xte = np.concatenate(Xs_te); yte = np.concatenate(ys_te)

        # PoI = sample with max F-statistic style (variance ratio across classes)
        # simple: argmax of sum of pairwise |t| at sample
        late_t_sum = np.zeros(n_samples)
        for i in range(n_sessions):
            for j in range(i + 1, n_sessions):
                t = welch_t(Xtr[ytr == i], Xtr[ytr == j])
                late_t_sum += np.abs(t)
        # restrict to LATE
        late_t_sum[:LATE_START] = 0
        poi = int(np.argmax(late_t_sum))

        # 1-NN classifier on (Xtr[:, poi], ytr) → predict ytest
        # use template: per-class mean at poi
        means = np.array([Xtr[ytr == s, poi].mean() for s in range(n_sessions)])
        # predict: closest mean
        pred = np.array([np.argmin(np.abs(Xte[k, poi] - means)) for k in range(len(Xte))])
        acc = float((pred == yte).mean())
        random_acc = 1.0 / n_sessions
        print(f"  design {design}: PoI={poi}, 4-way acc = {acc*100:5.1f}% (random = {random_acc*100:.1f}%)")

    # === plot: traces overlaid by sk ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, design in zip(axes, ("zero", "a192")):
        for s in range(n_sessions):
            G = groups[(design, s)]
            ax.plot(G.mean(0), lw=0.5, alpha=0.8, label=f"sk{s}")
        ax.set_xlim(LATE_START, n_samples)
        ax.set_ylabel(f"{design} mean trace")
        ax.legend()
        ax.axvline(POI, color="r", ls="--", lw=0.5, label="POI 23284")
    axes[1].set_xlabel("sample")
    fig.suptitle("Mean trace per sk, late zone (overlay)")
    fig.tight_layout()
    fig.savefig(out_dir / "pco_pilot5_means.png", dpi=120)
    plt.close(fig)
    print(f"\n  → {out_dir / 'pco_pilot5_means.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
