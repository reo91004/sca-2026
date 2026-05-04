#!/usr/bin/env python3
"""PCO pilot 6 분석 — bit-level cross-sk µ' leak detection.

데이터: 4 sk × 4 designs × 32 traces = 512 traces. 각 trace 의 µ' bit prediction (256 bits) 보유.

전략:
  Step 1. 각 (sample s, bit i) 에 대해 trace[s] 와 µ'[i] 의 Pearson correlation 계산.
          이걸 sk-by-sk 로 분리 → cross-sk correlation matrix.
  Step 2. (s, i) 쌍 중 모든 sk 에서 correlation 의 sign 이 일치 + |corr| 가 큰 것 선별.
          이게 µ'[i] 의 cross-sk leak PoI.
  Step 3. 만약 그런 (s, i) 쌍이 있으면 → cross-sk µ' bit oracle 가능 → sparse sk 복구.
          없으면 → 'D' trace 의 µ'-bit leak 도 cross-key fail.
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


def main() -> int:
    out_dir = _REPO / "results"
    out_dir.mkdir(exist_ok=True)
    d = np.load(_REPO / "traces/pco_pilot6.npz", allow_pickle=True)
    T = d["traces"].astype(np.float64)
    flags = d["mismatch_flags"]
    mu = d["mu_per_trace"].astype(np.float64)  # (N, 256)
    meta = d["meta"].item()
    label_session = np.asarray(meta["label_session"])
    label_design = np.asarray(meta["label_design"])
    n_sessions = meta["n_sessions"]

    n_total, n_samples = T.shape
    n_bits = mu.shape[1]
    print(f"[ANALYSIS] {n_total} traces, {n_samples} samples, {n_bits} bits")
    print(f"  sessions: {n_sessions}, designs: {meta['design_names']}")

    LATE_START = 20000
    n_late = n_samples - LATE_START
    T_late = T[:, LATE_START:]

    # === Step 1 — per-session correlation between trace[s] and µ'[i] ===
    print("\n[1] Per-session Pearson |corr| max per bit (LATE zone only)")
    # corr_sk[k, s, i] = corr_within_session_k(trace[s], µ'[i])
    # naive: per-session, per-bit, per-sample correlation
    sess_traces = []
    sess_mu = []
    for k in range(n_sessions):
        m = label_session == k
        sess_traces.append(T_late[m])  # (n_per_sk, n_late)
        sess_mu.append(mu[m])           # (n_per_sk, 256)

    # Vectorized Pearson per (s, i): corr = (Σ(t-mt)(b-mb)) / (sqrt(Σ(t-mt)^2) * sqrt(Σ(b-mb)^2))
    def per_session_corr(Tk, Mk):
        # Tk: (N, n_late), Mk: (N, 256)
        Tc = Tk - Tk.mean(0, keepdims=True)
        Mc = Mk - Mk.mean(0, keepdims=True)
        nT = np.sqrt((Tc**2).sum(0)).clip(min=1e-12)  # (n_late,)
        nM = np.sqrt((Mc**2).sum(0)).clip(min=1e-12)  # (256,)
        # numerator: Tc.T @ Mc → (n_late, 256)
        num = Tc.T @ Mc
        return num / (nT[:, None] * nM[None, :])

    corr_per_sk = np.array([per_session_corr(t, m) for t, m in zip(sess_traces, sess_mu)])
    # shape: (n_sessions, n_late, 256)
    print(f"  corr_per_sk shape: {corr_per_sk.shape}")
    abs_max_per_sk = np.abs(corr_per_sk).max(axis=(1, 2))
    print(f"  |corr| max per session: {[f'{x:.3f}' for x in abs_max_per_sk]}")

    # === Step 2 — cross-sk consistency ===
    # For each (s, i), check if all 4 sk have same-sign correlation AND mean |corr| is high.
    # mean of corr (with sign) across sk:
    mean_signed_corr = corr_per_sk.mean(axis=0)  # (n_late, 256)
    # consistency = (count of same-sign sk) / n_sessions
    sign_corr = np.sign(corr_per_sk)
    # all 4 same sign:
    consistent_pos = (sign_corr > 0).all(axis=0)
    consistent_neg = (sign_corr < 0).all(axis=0)
    consistent = consistent_pos | consistent_neg  # (n_late, 256)
    print(f"  consistent (s,i) pairs: {int(consistent.sum())} / {n_late * n_bits} "
          f"= {consistent.sum()/(n_late*n_bits)*100:.2f}%")

    # cross-sk score: |mean_signed_corr| * consistent
    score = np.abs(mean_signed_corr) * consistent.astype(float)
    flat_top = np.argsort(score, axis=None)[::-1][:20]
    print(f"\n[2] Top 20 cross-sk consistent (sample, bit) pairs:")
    print(f"  {'sample':>7s}  {'bit':>4s}  {'mean_corr':>10s}  {'corr_per_sk':>30s}")
    for idx in flat_top:
        s, i = np.unravel_index(idx, score.shape)
        s_full = LATE_START + s
        per_sk = [f"{corr_per_sk[k, s, i]:+.3f}" for k in range(n_sessions)]
        print(f"  {s_full:>7d}  {i:>4d}  {mean_signed_corr[s, i]:>+9.4f}  {','.join(per_sk):>30s}")

    # === Step 3 — verdict ===
    # 비교: random baseline 의 cross-sk consistent rate.
    # 4-coin flip 에서 모두 same direction = 2/16 = 12.5% (under H0 of independence)
    # uniform sample-bit pairs 의 consistent rate 가 통계적 baseline
    # 1) consistent_rate by chance ~ 12.5% (4 sessions same sign)
    # 2) max score under H0 = ?
    # Heuristic: if max score > 0.5, that's strong leak (corr ≈ 0.5 cross-sk).
    print(f"\n[3] Verdict")
    max_score = float(score.max())
    s_top, i_top = np.unravel_index(int(np.argmax(score)), score.shape)
    print(f"  max cross-sk score = {max_score:.4f} at sample {LATE_START + s_top}, bit {i_top}")

    # also, average per-sk |corr| at top point
    top_corrs = corr_per_sk[:, s_top, i_top]
    print(f"  per-sk corr at this point: {top_corrs}")
    if max_score > 0.5:
        print(f"  ✓ STRONG cross-sk µ' bit leak detected — sparse sk recovery viable")
    elif max_score > 0.25:
        print(f"  △ Moderate signal — borderline, needs higher N")
    else:
        print(f"  ✗ No cross-sk µ' bit leak — 'D' trace also random cross-key for sparse sk recovery")

    # === Plot ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].imshow(np.abs(mean_signed_corr).T, aspect="auto",
                   extent=(LATE_START, n_samples, 0, n_bits),
                   origin="lower", cmap="viridis", vmax=0.3)
    axes[0].set_title("Cross-sk |mean_signed_corr| heatmap (LATE zone)")
    axes[0].set_xlabel("sample"); axes[0].set_ylabel("bit i")

    # per-session top-bit at top-sample
    axes[1].plot(np.abs(mean_signed_corr).max(axis=1), label="cross-sk |mean_corr| max over bits")
    axes[1].plot(score.max(axis=1), label="cross-sk consistency-weighted score")
    xs = np.arange(n_late) + LATE_START
    axes[1].set_xticks(np.linspace(0, n_late, 6))
    axes[1].set_xticklabels([str(int(x)) for x in np.linspace(LATE_START, n_samples, 6)])
    axes[1].set_xlabel("sample"); axes[1].set_ylabel("|corr|")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pco_pilot6_cpa.png", dpi=120)
    plt.close(fig)
    print(f"  → {out_dir / 'pco_pilot6_cpa.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
