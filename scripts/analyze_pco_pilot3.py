#!/usr/bin/env python3
"""PCO pilot 3 분석 — accept vs reject distinguisher in late trace.

목적
----
late-trace leak 의 *근본 source* 를 결정:
  (a) accept/reject 가 distinguishable → FO check 의 1-bit oracle (PCO oracle).
  (b) reject 들끼리만 ct-byte 로 distinguishable → ct-byte SHAKE 누설만 (sk-uninformative).

평가
----
- pair 1: accept vs reject_zero       — 핵심 PCO test
- pair 2: accept vs reject_a192       — 다른 reject ct 로 robustness
- pair 3: reject_zero vs reject_a192  — pilot 1/2 의 ct-byte leak 재확인 (control)

late-trace max|t| 비교:
  pair1, pair2 ≫ pair3 이면 → accept/reject 가 강한 signal (PCO 가능)
  pair1, pair2 ≈ pair3 이면 → ct-byte leak only (PCO 불가)
  pair1 ≈ pair2 ≪ pair3 이면 → ct-byte leak 만 (accept 와 reject 둘다 같은 dist)

또한 *accept 내부 분산* 측정: 'e' 가 매 회 다른 valid ct 생성 → accept traces 가
ct 마다 다르면 high variance, σ_late_acc 큼. reject 들은 fixed ct → σ_late_rej 작음.
그래서 reject 끼리 비교가 "control" 그룹.
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


def split_classifier(Ta: np.ndarray, Tb: np.ndarray, name_a: str, name_b: str,
                     seed: int = 42) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    na, nb = len(Ta), len(Tb)
    na_tr, nb_tr = na // 2, nb // 2
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
    return float(((pred_a == name_a).mean() + (pred_b == name_b).mean()) / 2.0), poi


def main() -> int:
    out_dir = _REPO / "results"
    out_dir.mkdir(exist_ok=True)
    d = np.load(_REPO / "traces/pco_pilot3.npz", allow_pickle=True)
    T = d["traces"]
    flags = d["mismatch_flags"]
    meta = d["meta"].item()
    labels = np.asarray(meta["label_design"])
    n_total, n_samples = T.shape
    designs = list(meta["design_names"])

    n_acc = int(np.sum(labels == "accept"))
    n_rz = int(np.sum(labels == "reject_zero"))
    n_ra = int(np.sum(labels == "reject_a192"))
    print(f"[ANALYSIS] {n_total} traces, {n_samples} samples")
    print(f"  groups: accept={n_acc}, reject_zero={n_rz}, reject_a192={n_ra}")
    # validate flags
    flags_acc = flags[labels == "accept"]
    flags_rz = flags[labels == "reject_zero"]
    flags_ra = flags[labels == "reject_a192"]
    print(f"  flags(0=acc): accept{int(flags_acc.sum())}/{n_acc}, "
          f"reject_zero{int(flags_rz.sum())}/{n_rz}, reject_a192{int(flags_ra.sum())}/{n_ra}")
    n_acc_real = int((flags_acc == 0).sum())
    print(f"  → real accepts: {n_acc_real}/{n_acc}, "
          f"rejects(rz): {int((flags_rz == 1).sum())}/{n_rz}, "
          f"rejects(ra): {int((flags_ra == 1).sum())}/{n_ra}")
    if n_acc_real < n_acc:
        print(f"  [WARN] {n_acc - n_acc_real} 'accept' traces actually rejected — likely 'e/d' state mismatch")

    # Filter to clean groups
    T_acc = T[(labels == "accept") & (flags == 0)]
    T_rz = T[(labels == "reject_zero") & (flags == 1)]
    T_ra = T[(labels == "reject_a192") & (flags == 1)]
    print(f"  clean: accept={len(T_acc)}, reject_zero={len(T_rz)}, reject_a192={len(T_ra)}\n")

    # === pairwise TVLA + classifier ===
    print("[1] Pairwise TVLA + classifier")
    pairs = [
        ("accept",      "reject_zero",  T_acc, T_rz),
        ("accept",      "reject_a192",  T_acc, T_ra),
        ("reject_zero", "reject_a192",  T_rz,  T_ra),
    ]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 9), sharex=True)
    results = []
    for i, (na, nb, Ta, Tb) in enumerate(pairs):
        t = welch_t(Ta, Tb)
        idx = int(np.argmax(np.abs(t)))
        max_t = float(np.abs(t).max())
        late = slice(n_samples // 2, n_samples)
        max_t_late = float(np.abs(t[late]).max())
        idx_late = int(n_samples // 2 + np.argmax(np.abs(t[late])))
        loc = "EARLY" if idx < n_samples // 2 else "LATE"
        acc, poi = split_classifier(Ta, Tb, na, nb)
        print(f"  {na:12s} vs {nb:12s}: max|t|={max_t:5.2f} @ {idx:>5d} ({loc}), "
              f"late_max={max_t_late:5.2f} @ {idx_late}, classifier={acc*100:5.1f}%")
        results.append((na, nb, max_t, idx, max_t_late, idx_late, acc, poi))
        axes[i].plot(np.abs(t), lw=0.4)
        axes[i].axhline(4.5, color="r", ls="--", lw=0.5)
        axes[i].axvline(n_samples // 2, color="k", ls=":", lw=0.5)
        axes[i].set_title(f"{na} vs {nb} (max|t|={max_t:.2f} @ {idx}, classifier={acc*100:.1f}%)")
        axes[i].set_ylabel("|t|")
    axes[-1].set_xlabel("sample")
    fig.tight_layout()
    fig.savefig(out_dir / "pco_pilot3_tvla.png", dpi=120)
    plt.close(fig)
    print(f"  → {out_dir / 'pco_pilot3_tvla.png'}")

    # === Verdict ===
    print("\n[2] Verdict — is accept vs reject the dominant signal?")
    pair_acc_rz = next(r for r in results if r[0] == "accept" and r[1] == "reject_zero")
    pair_acc_ra = next(r for r in results if r[0] == "accept" and r[1] == "reject_a192")
    pair_rz_ra = next(r for r in results if r[0] == "reject_zero" and r[1] == "reject_a192")

    t_ar = pair_acc_rz[4]  # late_max
    t_ar2 = pair_acc_ra[4]
    t_rr = pair_rz_ra[4]

    print(f"  accept vs reject_zero  late max|t| = {t_ar:5.2f}, classifier = {pair_acc_rz[6]*100:.1f}%")
    print(f"  accept vs reject_a192  late max|t| = {t_ar2:5.2f}, classifier = {pair_acc_ra[6]*100:.1f}%")
    print(f"  reject_zero vs reject_a192 late max|t| = {t_rr:5.2f}, classifier = {pair_rz_ra[6]*100:.1f}%")

    if min(t_ar, t_ar2) > t_rr * 1.5:
        print(f"\n  ✓ STRONG accept-vs-reject signal — exceeds ct-byte ctrl by ≥1.5×")
        print(f"    → PCO oracle viable. Trace late-window can detect FO accept/reject.")
    elif min(t_ar, t_ar2) > t_rr * 0.7:
        print(f"\n  △ accept-vs-reject signal *similar* to ct-byte leak.")
        print(f"    May or may not give a *new* oracle — accept's high variance (random ct)")
        print(f"    can mask difference. Need same-ct accept (deterministic encap).")
    else:
        print(f"\n  ✗ accept-vs-reject signal *weaker* than ct-byte ctrl.")
        print(f"    Late-trace leak is mostly ct-byte SHAKE absorb (public).")
        print(f"    PCO accept/reject not viable from this PoI.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
