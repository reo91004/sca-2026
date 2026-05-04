#!/usr/bin/env python3
"""Cross-seed alignment diagnosis.

Step 1 leakage_phase_scan 에서 9 seeds 의 max|t| 위치가 capture 마다 다름을 확인.
이게 단순 shift 인지 (→ cross-correlation 으로 보정 가능) 아니면 완전 random layout 인지
판정.

방법:
1. 각 npz 의 oracle-pair Welch-t trace t_i(sample) 산출.
2. seed 0 의 |t_0| 를 reference 로 잡고, 다른 seed 들의 |t_k| 와 cross-correlation
   → optimal shift Δ_k.
3. Δ_k 분포가 작고 (예: ±100 sample) shift-corrected 후 hot region 들이 같은 위치에
   합쳐지면 alignment-after schedule 가능. shift 가 ±수천이거나 corrected 후도
   peak 가 흩어지면 layout 자체가 random.

산출:
    results/alignment_diag.npz / alignment_diag.png / 콘솔 출력.
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


def welch_t_abs(T: np.ndarray, la: np.ndarray, ap: int, an: int) -> np.ndarray:
    a = T[la == ap]; b = T[la == an]
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.zeros(T.shape[1], dtype=np.float64)
    ma = a.mean(axis=0); mb = b.mean(axis=0)
    va = a.var(axis=0, ddof=1).clip(min=1e-12)
    vb = b.var(axis=0, ddof=1).clip(min=1e-12)
    return np.abs((ma - mb) / np.sqrt(va / a.shape[0] + vb / b.shape[0]))


def best_shift(ref: np.ndarray, sig: np.ndarray, max_shift: int = 4000) -> tuple[int, float]:
    """argmax_d corr(ref, sig shifted by d). |d| ≤ max_shift."""
    n = ref.size
    rc = ref - ref.mean(); rc /= max(rc.std(), 1e-12)
    sc = sig - sig.mean(); sc /= max(sc.std(), 1e-12)
    # full cross-correlation, but limit to ±max_shift.
    full = np.correlate(rc, sc, mode="full")
    center = n - 1
    lo = max(0, center - max_shift); hi = min(full.size, center + max_shift + 1)
    sub = full[lo:hi]
    d = int(np.argmax(sub) + lo - center)  # positive d = sig is shifted left
    score = float(sub.max() / n)
    return d, score


def main() -> int:
    paths = sorted(Path("traces").glob("attack_seed*.npz"))
    paths = [p for p in paths if "smaug3" not in p.name and "smaug5" not in p.name]
    paths += [Path("traces/attack_const_c1_n256.npz")]
    paths = [p for p in paths if p.exists()]
    print(f"[INFO] {len(paths)} captures")

    sigs = []
    names = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        T = d["traces"]; meta = d["meta"].item()
        ap = int(meta["alpha_pos"]); an = int(meta["alpha_neg"])
        la = np.asarray(meta["label_alpha"])
        t = welch_t_abs(T, la, ap, an)
        sigs.append(t)
        names.append(p.name)
        print(f"  {p.name:45s} max|t|={float(t.max()):6.2f} @{int(np.argmax(t)):>5d}")

    # reference: seed 0
    ref = sigs[0]
    n_samp = ref.size

    print(f"\n[shift] reference = {names[0]} (length={n_samp})")
    shifts = []
    scores = []
    aligned = [ref.copy()]
    for i in range(1, len(sigs)):
        d, sc = best_shift(ref, sigs[i], max_shift=4000)
        shifts.append(d); scores.append(sc)
        # apply shift
        if d >= 0:
            a = np.zeros_like(sigs[i])
            a[d:] = sigs[i][:n_samp - d]
        else:
            a = np.zeros_like(sigs[i])
            a[:n_samp + d] = sigs[i][-d:]
        aligned.append(a)
        print(f"  {names[i]:45s} shift={d:+6d}  norm-corr={sc:.3f}")

    aligned = np.stack(aligned, axis=0)
    cum_max = np.max(aligned, axis=0)
    cum_mean = np.mean(aligned, axis=0)
    print(f"\n  pre-align  cumulative max|t|: {float(np.max(np.stack(sigs))):.2f}")
    print(f"  post-align cumulative max|t|: {float(cum_max.max()):.2f}")
    print(f"  post-align cumulative mean|t|: {float(cum_mean.max()):.2f}")
    if cum_mean.max() > 4.0:
        # consensus peak emerged
        peak = int(np.argmax(cum_mean))
        print(f"  ★ consensus peak at sample {peak} (mean|t|={float(cum_mean[peak]):.2f})")
    else:
        print("  [note] no strong consensus peak even after shift — likely layout-level mismatch")

    # save
    out = Path("results/alignment_diag.npz")
    np.savez(out,
             pre_align=np.stack(sigs),
             post_align=aligned,
             shifts=np.array([0] + shifts),
             scores=np.array([1.0] + scores),
             names=np.array(names))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for s in sigs:
        axes[0].plot(s, lw=0.3, alpha=0.5)
    axes[0].set_title("pre-alignment |t| (9 seeds overlay)")
    axes[0].axhline(4.5, color="r", ls="--", lw=0.5)
    axes[0].set_ylabel("|t|")
    for s in aligned:
        axes[1].plot(s, lw=0.3, alpha=0.5)
    axes[1].plot(cum_mean, color="k", lw=1.0, label="mean")
    axes[1].set_title("post-alignment |t| (after best shift, ±4000)")
    axes[1].axhline(4.5, color="r", ls="--", lw=0.5)
    axes[1].set_ylabel("|t|"); axes[1].set_xlabel("sample")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("results/alignment_diag.png", dpi=120)
    plt.close(fig)
    print("\n[OK] results/alignment_diag.{npz,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
