#!/usr/bin/env python3
"""PCO pilot 4 분석 — cross-session/cross-sk classifier transfer.

Train PoI on session A (pilot3, sk_A) → apply to session B (pilot4, sk_B). And reverse.

만약 전이 100% → oracle 이 sk-universal. attacker 가 own calibration → target session.
만약 전이 < session-internal acc → per-session calibration 필요. 이래도 attack-valid.
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
    ma = a.mean(0); mb = b.mean(0)
    va = a.var(0, ddof=1).clip(min=1e-12); vb = b.var(0, ddof=1).clip(min=1e-12)
    return (ma - mb) / np.sqrt(va / a.shape[0] + vb / b.shape[0])


def load_pilot(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = d["meta"].item()
    labels = np.asarray(meta["label_design"])
    flags = d["mismatch_flags"]
    T = d["traces"]
    return {
        "T_acc": T[(labels == "accept") & (flags == 0)],
        "T_rz":  T[(labels == "reject_zero") & (flags == 1)],
        "T_ra":  T[(labels == "reject_a192") & (flags == 1)],
        "sk16":  meta.get("sk_pke_bytes_hex", "")[:16],
        "n_samples": T.shape[1],
    }


def train_classifier(Ta_tr, Tb_tr, na, nb, late_only=True, late_start=20000):
    """Returns (poi, threshold, sign_a, mu_a, mu_b)."""
    t = welch_t(Ta_tr, Tb_tr)
    if late_only:
        ts = np.abs(t).copy(); ts[:late_start] = 0
        poi = int(np.argmax(ts))
    else:
        poi = int(np.argmax(np.abs(t)))
    mu_a = float(Ta_tr[:, poi].mean())
    mu_b = float(Tb_tr[:, poi].mean())
    thr = (mu_a + mu_b) / 2
    sign_a = +1 if mu_a > mu_b else -1
    return poi, thr, sign_a, mu_a, mu_b


def apply_classifier(Tx, poi, thr, sign_a, name_a, name_b):
    return np.where(sign_a * (Tx[:, poi] - thr) > 0, name_a, name_b)


def acc_pair(Ta, Tb, poi, thr, sign_a, na, nb):
    pa = apply_classifier(Ta, poi, thr, sign_a, na, nb)
    pb = apply_classifier(Tb, poi, thr, sign_a, na, nb)
    return float((pa == na).mean()), float((pb == nb).mean())


def main() -> int:
    pa = load_pilot(_REPO / "traces/pco_pilot3.npz")
    pb = load_pilot(_REPO / "traces/pco_pilot4_session_b.npz")
    print(f"[A] pilot 3: sk={pa['sk16']}, accept={len(pa['T_acc'])}, "
          f"rz={len(pa['T_rz'])}, ra={len(pa['T_ra'])}")
    print(f"[B] pilot 4: sk={pb['sk16']}, accept={len(pb['T_acc'])}, "
          f"rz={len(pb['T_rz'])}, ra={len(pb['T_ra'])}")
    if pa['sk16'] == pb['sk16']:
        print("  [WARN] same sk — cross-session test invalidated")
    else:
        print("  ✓ different sk between sessions")

    print("\n[1] In-session training (LATE-only PoI)")
    pairs = [
        ("accept", "reject_zero", "T_acc", "T_rz"),
        ("accept", "reject_a192", "T_acc", "T_ra"),
        ("reject_zero", "reject_a192", "T_rz", "T_ra"),
    ]
    for name, sess in [("A (pilot3)", pa), ("B (pilot4)", pb)]:
        print(f"  Session {name}:")
        for na, nb, ka, kb in pairs:
            Ta, Tb = sess[ka], sess[kb]
            # 50/50 split
            rng = np.random.default_rng(42)
            n_a, n_b = len(Ta) // 2, len(Tb) // 2
            oa = rng.permutation(len(Ta)); ob = rng.permutation(len(Tb))
            Ta_tr, Ta_te = Ta[oa[:n_a]], Ta[oa[n_a:]]
            Tb_tr, Tb_te = Tb[ob[:n_b]], Tb[ob[n_b:]]
            poi, thr, sign, *_ = train_classifier(Ta_tr, Tb_tr, na, nb, late_only=True)
            acc_a, acc_b = acc_pair(Ta_te, Tb_te, poi, thr, sign, na, nb)
            print(f"    {na:12s} vs {nb:12s}: poi={poi:>5d}, acc=({acc_a*100:5.1f}, {acc_b*100:5.1f})")

    print("\n[2] CROSS-session transfer (LATE-only)")
    for src_name, src, dst_name, dst in [
        ("A→B", pa, "B", pb),
        ("B→A", pb, "A", pa),
    ]:
        print(f"  {src_name} (train on session {src_name[0]}, apply to session {dst_name})")
        for na, nb, ka, kb in pairs:
            poi, thr, sign, *_ = train_classifier(src[ka], src[kb], na, nb, late_only=True)
            acc_a, acc_b = acc_pair(dst[ka], dst[kb], poi, thr, sign, na, nb)
            avg = (acc_a + acc_b) / 2
            verdict = "✓" if avg > 0.95 else ("△" if avg > 0.70 else "✗")
            print(f"    {na:12s} vs {nb:12s}: poi={poi:>5d}, "
                  f"acc=({acc_a*100:5.1f}, {acc_b*100:5.1f}), avg={avg*100:5.1f}% {verdict}")

    print("\n[3] Late-trace |t| overlay (accept vs reject_zero)")
    out_dir = _REPO / "results"
    out_dir.mkdir(exist_ok=True)
    t_a = welch_t(pa["T_acc"], pa["T_rz"])
    t_b = welch_t(pb["T_acc"], pb["T_rz"])
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(np.abs(t_a), lw=0.4, color="C0", label=f"Session A sk={pa['sk16']}")
    ax.plot(np.abs(t_b), lw=0.4, color="C1", alpha=0.7, label=f"Session B sk={pb['sk16']}")
    ax.axhline(4.5, color="r", ls="--", lw=0.5)
    ax.set_xlim(20000, pa["n_samples"])
    ax.set_xlabel("sample"); ax.set_ylabel("|t|")
    ax.set_title("accept vs reject_zero — late-trace |t| cross-session")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pco_pilot4_cross_session.png", dpi=120)
    plt.close(fig)
    print(f"  → {out_dir / 'pco_pilot4_cross_session.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
