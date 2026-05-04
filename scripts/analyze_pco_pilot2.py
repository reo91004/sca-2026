#!/usr/bin/env python3
"""PCO pilot 2 분석 — 1-bit µ' detectability + HW model fit + cross-session stability.

Pilot 2 의 6 designs (zero, a4, a64, a128, a192, a252) × Session A + (zero, a192) × Session B.

평가:
  Q1. 1-bit µ' 차이 detectable? — zero vs a4, zero vs a252 의 max|t|
  Q2. HW(µ') vs max|t| monotone? — 모든 pair scatter
  Q3. Cross-session leak position stable? — Session A vs B 의 sample 22924 leak 비교
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


def load_session(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = d["meta"].item()
    return {
        "T": d["traces"],
        "flags": d["mismatch_flags"],
        "labels": np.asarray(meta["label_design"]),
        "designs": list(meta["design_names"]),
        "mu_hw": meta["mu_hw_per_design"],
        "ct_fp": meta["ct_fp_per_design"],
        "session": meta.get("session", "?"),
        "sk_hex": meta.get("sk_pke_bytes_hex", "")[:16],
    }


def split_classifier_acc(Ta: np.ndarray, Tb: np.ndarray, name_a: str, name_b: str,
                         seed: int = 42) -> tuple[float, int, float, float]:
    rng = np.random.default_rng(seed)
    na, nb = len(Ta), len(Tb)
    na_tr = na // 2; nb_tr = nb // 2
    ord_a = rng.permutation(na); ord_b = rng.permutation(nb)
    Ta_tr, Ta_te = Ta[ord_a[:na_tr]], Ta[ord_a[na_tr:]]
    Tb_tr, Tb_te = Tb[ord_b[:nb_tr]], Tb[ord_b[nb_tr:]]
    t_tr = welch_t(Ta_tr, Tb_tr)
    poi = int(np.argmax(np.abs(t_tr)))
    mu_a = float(Ta_tr[:, poi].mean())
    mu_b = float(Tb_tr[:, poi].mean())
    thr = (mu_a + mu_b) / 2.0
    sign_a = +1 if mu_a > mu_b else -1
    pred_a = np.where(sign_a * (Ta_te[:, poi] - thr) > 0, name_a, name_b)
    pred_b = np.where(sign_a * (Tb_te[:, poi] - thr) > 0, name_a, name_b)
    acc_a = float((pred_a == name_a).mean())
    acc_b = float((pred_b == name_b).mean())
    return (acc_a + acc_b) / 2.0, poi, acc_a, acc_b


def main() -> int:
    out_dir = _REPO / "results"
    out_dir.mkdir(exist_ok=True)

    sess_a = load_session(_REPO / "traces/pco_pilot2.npz")
    sess_b = load_session(_REPO / "traces/pco_pilot2_session_b.npz")

    n_samples = sess_a["T"].shape[1]
    designs_a = sess_a["designs"]
    mu_hw = sess_a["mu_hw"]
    print(f"[Session A] {sess_a['T'].shape[0]} traces, {n_samples} samples, sk={sess_a['sk_hex']}")
    print(f"  designs: {designs_a}")
    print(f"  HW: {mu_hw}")
    print(f"  flags (1=reject): {int(sess_a['flags'].sum())}/{len(sess_a['flags'])}\n")
    print(f"[Session B] {sess_b['T'].shape[0]} traces, sk={sess_b['sk_hex']}")
    print(f"  designs: {sess_b['designs']}, HW: {sess_b['mu_hw']}\n")
    if sess_a["sk_hex"] != sess_b["sk_hex"]:
        print(f"  [WARN] sk differs between sessions!")

    # === Q1. 1-bit µ' detectability ===
    print("[Q1] 1-bit µ' difference detectability")
    print("  (zero vs a4, zero vs a252 are *small* HW perturbations)")
    pairs_q1 = [("zero", "a4"), ("zero", "a252"), ("zero", "a192"),
                ("zero", "a64"), ("zero", "a128")]
    T = sess_a["T"]; labels = sess_a["labels"]
    for a, b in pairs_q1:
        if a not in designs_a or b not in designs_a:
            continue
        Ta = T[labels == a]; Tb = T[labels == b]
        t = welch_t(Ta, Tb)
        idx = int(np.argmax(np.abs(t)))
        max_t = float(np.abs(t).max())
        loc = "EARLY" if idx < n_samples // 2 else "LATE"
        hw_diff = abs(mu_hw[a] - mu_hw[b])
        acc, poi, acc_a, acc_b = split_classifier_acc(Ta, Tb, a, b)
        print(f"  {a:6s} vs {b:6s}: HW_diff={hw_diff:>3d}, max|t|={max_t:5.2f} @ {idx:>5d} ({loc}), "
              f"1-bit acc={acc*100:5.1f}% (a={acc_a*100:.1f}, b={acc_b*100:.1f})")

    # === Q2. HW(µ') vs max|t| scatter ===
    print("\n[Q2] HW model fit — all pairs (always vs zero)")
    print("  HW_diff | max|t| (LATE) | sample idx | 1-bit acc")
    hw_diffs = []
    max_ts = []
    accs = []
    for d in designs_a:
        if d == "zero": continue
        Ta = T[labels == "zero"]; Tb = T[labels == d]
        t = welch_t(Ta, Tb)
        late_zone = slice(n_samples // 2, n_samples)
        idx_late = int(n_samples // 2 + np.argmax(np.abs(t[late_zone])))
        max_t_late = float(np.abs(t[late_zone]).max())
        hw_diff = abs(mu_hw["zero"] - mu_hw[d])
        acc, _, _, _ = split_classifier_acc(Ta, Tb, "zero", d)
        print(f"  {d:6s}: {hw_diff:>3d}     | {max_t_late:5.2f}        | {idx_late:>5d}     | {acc*100:5.1f}%")
        hw_diffs.append(hw_diff)
        max_ts.append(max_t_late)
        accs.append(acc)

    # scatter HW vs max|t|
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(hw_diffs, max_ts, s=80)
    for d, x, y in zip([d for d in designs_a if d != "zero"], hw_diffs, max_ts):
        axes[0].annotate(d, (x, y), fontsize=9, xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel("HW(µ') diff vs zero")
    axes[0].set_ylabel("max|t| (late half)")
    axes[0].set_title("HW vs leak strength — vs zero design")
    axes[0].axhline(4.5, color="r", ls="--", lw=0.5, label="|t|=4.5 threshold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(hw_diffs, [a * 100 for a in accs], s=80)
    for d, x, y in zip([d for d in designs_a if d != "zero"], hw_diffs, accs):
        axes[1].annotate(d, (x, y * 100), fontsize=9, xytext=(5, 5), textcoords="offset points")
    axes[1].set_xlabel("HW(µ') diff vs zero")
    axes[1].set_ylabel("1-bit classifier accuracy (%)")
    axes[1].set_title("HW vs single-trace classifier")
    axes[1].axhline(50, color="k", ls=":", lw=0.5, label="random 50%")
    axes[1].axhline(95, color="g", ls="--", lw=0.5, label="95% (oracle threshold)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pco_pilot2_hw_fit.png", dpi=120)
    plt.close(fig)
    print(f"  → {out_dir / 'pco_pilot2_hw_fit.png'}")

    # === Q3. Cross-session leak position stability ===
    print("\n[Q3] Cross-session leak position stability (zero vs a192)")
    Ta_a = sess_a["T"][sess_a["labels"] == "zero"]
    Tb_a = sess_a["T"][sess_a["labels"] == "a192"]
    t_a = welch_t(Ta_a, Tb_a)
    idx_a = int(np.argmax(np.abs(t_a)))
    max_t_a = float(np.abs(t_a).max())
    print(f"  Session A: max|t|={max_t_a:5.2f} @ sample {idx_a:>5d}")

    Ta_b = sess_b["T"][sess_b["labels"] == "zero"]
    Tb_b = sess_b["T"][sess_b["labels"] == "a192"]
    t_b = welch_t(Ta_b, Tb_b)
    idx_b = int(np.argmax(np.abs(t_b)))
    max_t_b = float(np.abs(t_b).max())
    print(f"  Session B: max|t|={max_t_b:5.2f} @ sample {idx_b:>5d}")
    print(f"  Δsample = {abs(idx_a - idx_b):>5d}")

    # cross-session classifier transfer:
    #   train PoI on Session A, apply to Session B traces
    rng = np.random.default_rng(42)
    poi_a_train = int(np.argmax(np.abs(t_a)))
    mu_a_zero = float(Ta_a[:, poi_a_train].mean())
    mu_a_a192 = float(Tb_a[:, poi_a_train].mean())
    thr_a = (mu_a_zero + mu_a_a192) / 2.0
    sign_a = +1 if mu_a_zero > mu_a_a192 else -1
    pred_b_zero = np.where(sign_a * (Ta_b[:, poi_a_train] - thr_a) > 0, "zero", "a192")
    pred_b_a192 = np.where(sign_a * (Tb_b[:, poi_a_train] - thr_a) > 0, "zero", "a192")
    acc_b_z = float((pred_b_zero == "zero").mean())
    acc_b_a = float((pred_b_a192 == "a192").mean())
    cross_acc = (acc_b_z + acc_b_a) / 2.0
    print(f"  Cross-session classifier (PoI={poi_a_train} from A, applied to B):")
    print(f"    accuracy = {cross_acc*100:5.1f}% (zero={acc_b_z*100:.1f}%, a192={acc_b_a*100:.1f}%)")

    # plot t-traces overlay
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(np.abs(t_a), lw=0.4, color="C0", label=f"Session A (max|t|={max_t_a:.2f} @ {idx_a})")
    ax.plot(np.abs(t_b), lw=0.4, color="C1", alpha=0.7, label=f"Session B (max|t|={max_t_b:.2f} @ {idx_b})")
    ax.axhline(4.5, color="r", ls="--", lw=0.5)
    ax.axvline(idx_a, color="C0", ls=":", lw=0.5)
    ax.axvline(idx_b, color="C1", ls=":", lw=0.5)
    ax.set_xlabel("sample"); ax.set_ylabel("|t|")
    ax.set_title("zero vs a192 — Welch-t cross-session overlay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pco_pilot2_cross_session.png", dpi=120)
    plt.close(fig)
    print(f"  → {out_dir / 'pco_pilot2_cross_session.png'}")

    # === Verdict ===
    print("\n[Verdict] PCO oracle viability")
    print(f"  Q1 1-bit detectability: see zero vs a4 / a252 above")
    a4_acc = None
    a252_acc = None
    for d in designs_a:
        if d == "a4":
            Ta = T[labels == "zero"]; Tb = T[labels == "a4"]
            a4_acc = split_classifier_acc(Ta, Tb, "zero", "a4")[0]
        if d == "a252":
            Ta = T[labels == "zero"]; Tb = T[labels == "a252"]
            a252_acc = split_classifier_acc(Ta, Tb, "zero", "a252")[0]
    if a4_acc is not None:
        verdict_q1 = "PASS" if a4_acc > 0.85 else ("MARGINAL" if a4_acc > 0.6 else "FAIL")
        print(f"  Q1: zero vs a4 1-bit acc = {a4_acc*100:.1f}% → {verdict_q1}")
    print(f"  Q2: HW model fit — see scatter plot")
    print(f"  Q3: cross-session A→B classifier = {cross_acc*100:.1f}% "
          f"({'PASS' if cross_acc > 0.85 else 'MARGINAL' if cross_acc > 0.6 else 'FAIL'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
