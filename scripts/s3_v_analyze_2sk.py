#!/usr/bin/env python3
"""S3 — V capture 두 sk 비교 (Welch-t cross-sk).

S1 ('T') 와 S2 ('Z') 의 cross-sk Welch-t 와 직접 비교 :
  - 'T' max|t|=1294 (clean isolated poly_mul_acc, but not attack-valid)
  - 'Z' max|t|=771  (full indcpa_dec, includes setup loops)
  - 'V' max|t|=?    (sub-triggered vec_vec_mult_add, near attack-valid trace
                       but firmware-instrumented)

V 가 |t|>=771 (S2 보다 같거나 강) 이면 sub-trigger 가 누설 영역 cover 되었다는
증거. V 가 |t| 훨씬 강 (예 1000+) 이면 setup loop 가 실제로 noise 였음을 입증.
V 가 약하면 sub-trigger 위치 잘못이거나 vec_vec_mult_add 영역 밖에 leak 이
있다는 신호.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sk-a", type=Path, default=_REPO / "traces" / "s3_v_skA_a4_n200.npz")
    p.add_argument("--sk-b", type=Path, default=_REPO / "traces" / "s3_v_skB_a4_n200.npz")
    p.add_argument("--out-prefix", type=Path, default=_REPO / "results" / "s3_v_2sk")
    return p.parse_args()


def _load(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    return data["traces"], data["meta"].item()


def main() -> int:
    args = parse_args()
    A, ma = _load(args.sk_a)
    B, mb = _load(args.sk_b)
    if ma["sk_pke_hex"] == mb["sk_pke_hex"]:
        print("[FAIL] same sk")
        return 1
    n_a, T = A.shape
    n_b = B.shape[0]
    print(f"[INFO] N_A={n_a} N_B={n_b} T={T}")

    mean_a = A.mean(axis=0)
    mean_b = B.mean(axis=0)
    var_a = A.var(axis=0, ddof=1)
    var_b = B.var(axis=0, ddof=1)
    se = np.sqrt(var_a / n_a + var_b / n_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mean_a - mean_b) / se
    t = np.where(np.isfinite(t), t, 0.0)

    abs_t = np.abs(t)
    n_leak = int(np.sum(abs_t > 4.5))
    n_strong = int(np.sum(abs_t > 8.0))
    max_t = float(abs_t.max())
    argmax_t = int(np.argmax(abs_t))

    print(f"[RESULT] V cross-sk Welch-t : max|t|={max_t:.2f} @ sample {argmax_t}")
    print(f"         |t|>4.5: {n_leak} samples")
    print(f"         |t|>8.0: {n_strong} samples")
    print(f"         baseline : S1='T' 1294, S2='Z' 771")

    out_pre = args.out_prefix
    out_pre.parent.mkdir(parents=True, exist_ok=True)
    txt = (
        f"sk_a={args.sk_a.name}\nsk_b={args.sk_b.name}\n"
        f"N_A={n_a}\nN_B={n_b}\nT={T}\n"
        f"max_abs_t={max_t:.4f}\nargmax={argmax_t}\n"
        f"n_t_gt_4_5={n_leak}\nn_t_gt_8={n_strong}\n"
        f"vs_S1_T_1294={'pass' if max_t > 800 else 'weak'}\n"
        f"vs_S2_Z_771={'pass' if max_t > 600 else 'weak'}\n"
    )
    out_pre.with_suffix(".txt").write_text(txt)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(mean_a, label="sk_A", color="C0", lw=0.5)
    ax[0].plot(mean_b, label="sk_B", color="C1", lw=0.5)
    ax[0].legend(); ax[0].set_ylabel("mean trace")
    ax[1].plot(t, color="k", lw=0.5)
    ax[1].axhline(4.5, ls="--", color="r", lw=0.5)
    ax[1].axhline(-4.5, ls="--", color="r", lw=0.5)
    ax[1].set_ylabel("Welch-t")
    ax[1].set_xlabel("sample")
    ax[1].set_title(f"max|t|={max_t:.2f} @ {argmax_t}, leaky pts {n_leak}")
    fig.tight_layout()
    fig.savefig(out_pre.with_suffix(".png"), dpi=120)
    print(f"[OK] wrote {out_pre.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
