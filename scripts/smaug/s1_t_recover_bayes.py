#!/usr/bin/env python3
"""S1 — 3-class Bayes recovery (LOO).

각 좌표 k 의 best PoI t_k 에서, training 의 (M[t_k], sk[k]) 분포를 보고:
  μ_c = E[M[t_k] | sk[k]=c]  for c ∈ {-1, 0, +1}
  σ   = pooled within-class std
  prior π_c = empirical (sparse: π_0 ≈ 0.73, π_±1 ≈ 0.135)

추론: argmax_c π_c · N(M_test[t_k]; μ_c, σ) (= 3-class Gaussian Bayes).

추가 옵션:
  --conservative : posterior margin 이 작으면 0 으로 강제 (default off).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.smaug import codec as _codec  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s1_main_*c0_j0_a1*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument(
        "--conservative",
        action="store_true",
        help="margin 작으면 0 으로 강제 (FP 줄임)",
    )
    p.add_argument("--margin-thresh", type=float, default=0.5,
                   help="conservative 모드에서 posterior diff threshold")
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s1_recover_bayes",
    )
    return p.parse_args()


def _load(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["traces"], data["meta"].item()


def main() -> int:
    args = parse_args()
    means: list[np.ndarray] = []
    sks: list[np.ndarray] = []
    pkfps: list[str] = []
    for p in args.inputs:
        try:
            traces, meta = _load(p)
        except Exception:
            continue
        if "sk_pke_hex" not in meta:
            continue
        if meta["pk_fp16"] in pkfps:
            continue
        means.append(traces.mean(axis=0).astype(np.float64))
        full = _codec.unpack_sx(bytes.fromhex(meta["sk_pke_hex"])).astype(np.int64)
        sks.append(full[args.component * 256 : (args.component + 1) * 256])
        pkfps.append(meta["pk_fp16"])
    M = np.stack(means, axis=0)
    sk_t = np.stack(sks, axis=0)
    S, T = M.shape
    K = sk_t.shape[1]
    print(f"[INFO] S={S}, T={T}, K={K}")

    # 3-class label per (s, k): -1, 0, +1
    label_to_idx = {-1: 0, 0: 1, 1: 2}
    idx_to_ternary = {0: -1, 1: 0, 2: 1}
    L = np.zeros_like(sk_t, dtype=np.int64)
    for s in range(S):
        for k in range(K):
            L[s, k] = label_to_idx[int(sk_t[s, k])]

    accs = []
    supps = []
    signs = []
    margins_overall = []
    for hold in range(S):
        train_idx = [s for s in range(S) if s != hold]
        Mt = M[train_idx]
        Lt = L[train_idx]
        # PoI selection per coord — variance 기반 valid coords
        poi_t = -np.ones(K, dtype=np.int64)
        mu_per_class = np.zeros((K, 3))
        sigma_per_coord = np.zeros(K)
        prior = np.zeros((K, 3))
        Mc = Mt - Mt.mean(axis=0, keepdims=True)
        Mn = np.linalg.norm(Mc, axis=0, keepdims=True)

        for k in range(K):
            lk = Lt[:, k]
            uniq = np.unique(lk)
            if len(uniq) < 2:
                continue
            # PoI: maximize |Pearson(M[:,t], hw_int16(label))|.
            hw = np.array([
                bin(int(np.int16({0:-1,1:0,2:+1}[c])) & 0xFFFF).count("1")
                for c in lk
            ], dtype=np.float64)
            if hw.var(ddof=1) <= 0:
                continue
            Hc = hw - hw.mean()
            denom = (np.linalg.norm(Hc) * Mn[0])
            denom = np.where(denom == 0, 1.0, denom)
            r = (Hc @ Mc) / denom
            tb = int(np.argmax(np.abs(r)))
            poi_t[k] = tb
            # 클래스별 평균 + pooled within-class variance
            mu_c = []
            var_c = []
            n_c = []
            for cls in range(3):
                mask = lk == cls
                if mask.sum() == 0:
                    mu_c.append(np.nan)
                    var_c.append(np.nan)
                    n_c.append(0)
                else:
                    vals = Mt[mask, tb]
                    mu_c.append(float(vals.mean()))
                    if mask.sum() > 1:
                        var_c.append(float(vals.var(ddof=1)))
                    else:
                        var_c.append(0.0)
                    n_c.append(int(mask.sum()))
            mu_per_class[k] = mu_c
            # pooled std (within-class)
            total_var = 0.0
            total_dof = 0
            for v, n in zip(var_c, n_c):
                if not np.isnan(v) and n > 1:
                    total_var += v * (n - 1)
                    total_dof += n - 1
            sigma_per_coord[k] = (
                np.sqrt(total_var / total_dof) if total_dof > 0 else 1e-3
            )
            # prior (training)
            for cls in range(3):
                prior[k, cls] = (lk == cls).sum() / lk.size

        # Test
        Mte = M[hold]
        sk_te = sk_t[hold]
        ternary_pred = np.zeros(K, dtype=np.int64)
        margins = np.zeros(K)
        for k in range(K):
            if poi_t[k] < 0 or sigma_per_coord[k] <= 0:
                ternary_pred[k] = 0
                continue
            x = Mte[poi_t[k]]
            mu = mu_per_class[k]
            sig = sigma_per_coord[k]
            # log posterior (omit constant)
            lp = np.zeros(3)
            for cls in range(3):
                if np.isnan(mu[cls]) or prior[k, cls] == 0:
                    lp[cls] = -np.inf
                else:
                    lp[cls] = (
                        -0.5 * ((x - mu[cls]) / sig) ** 2
                        + np.log(prior[k, cls])
                    )
            best_cls = int(np.argmax(lp))
            sorted_lp = np.sort(lp)
            margin = float(sorted_lp[-1] - sorted_lp[-2])
            margins[k] = margin
            if args.conservative and margin < args.margin_thresh:
                ternary_pred[k] = 0
            else:
                ternary_pred[k] = idx_to_ternary[best_cls]

        coord_acc = float((ternary_pred == sk_te).mean())
        supp_mask = sk_te != 0
        supp_acc = float(((ternary_pred != 0) == supp_mask).mean()) if supp_mask.any() else 1.0
        sign_mask = supp_mask & (ternary_pred != 0)
        sign_acc = (
            float(
                (np.sign(ternary_pred[sign_mask]) == np.sign(sk_te[sign_mask])).mean()
            )
            if sign_mask.any()
            else 0.0
        )
        accs.append(coord_acc)
        supps.append(supp_acc)
        signs.append(sign_acc)
        margins_overall.append(float(margins.mean()))

    base_zero = float((sk_t == 0).mean())
    print(
        f"[LOOCV {S}-fold]\n"
        f"  mean coord_acc = {np.mean(accs):.4f} (predict-zero baseline = {base_zero:.4f})\n"
        f"  mean supp_acc  = {np.mean(supps):.4f}\n"
        f"  mean sign_acc  = {np.mean(signs):.4f}\n"
        f"  mean margin    = {np.mean(margins_overall):.4f}"
    )

    # 통계 표
    summary_lines = [
        f"S1 3-class Bayes LOO recovery (conservative={args.conservative})",
        f"S={S} sk, K=256, T=24400",
        f"baseline predict-zero = {base_zero:.4f}",
        f"mean coord_acc = {np.mean(accs):.4f}",
        f"mean supp_acc  = {np.mean(supps):.4f}",
        f"mean sign_acc  = {np.mean(signs):.4f}",
        "",
        "per-sk:",
    ]
    for h, ca, sa, sg in zip(range(S), accs, supps, signs):
        summary_lines.append(f"  hold={h:2d} coord={ca:.4f} supp={sa:.4f} sign={sg:.4f}")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".txt").write_text("\n".join(summary_lines) + "\n")
    print(f"[OK] {args.out_prefix.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
