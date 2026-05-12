#!/usr/bin/env python3
"""S2 — 'Z' chosen-CT trace 에 3-class Bayes LOO recovery.

S1 ('T') 와 같은 held-out-key evaluation skeleton for the `Z` trace. The HW
models here are candidates, not implementation claims. In particular, `shift`
is retained as a legacy hypothesis unless re-confirmed by disassembly.

option :
  --hw-model {raw|shift|both}
       raw   : HW(sk_int16)            ('T' 식)
       shift : HW(sk_int16 << 11)      (legacy shifted-secret hypothesis)
       both  : 두 model 의 PoI 중 더 좋은 것 사용 (per-coord)
  --conservative : margin < threshold 면 0 으로 강제 (default off)
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


def _load(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["traces"], data["meta"].item()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s2_z_sk*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument(
        "--hw-model", choices=("raw", "shift", "both"), default="shift",
    )
    p.add_argument("--conservative", action="store_true")
    p.add_argument("--margin-thresh", type=float, default=1.0)
    p.add_argument(
        "--sample-range",
        type=str,
        default=None,
        help="제한 샘플 범위 (예: '5000:22000'). Default = 모든 샘플.",
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s2_z_recover",
    )
    return p.parse_args()


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
        if meta.get("cmd") != "Z":
            continue
        if meta.get("pk_fp16") in pkfps:
            continue
        means.append(traces.mean(axis=0).astype(np.float64))
        full = _codec.unpack_sx(bytes.fromhex(meta["sk_pke_hex"])).astype(np.int64)
        sks.append(full[args.component * 256 : (args.component + 1) * 256])
        pkfps.append(meta["pk_fp16"])

    M = np.stack(means, axis=0)
    sk_t = np.stack(sks, axis=0)
    S, T = M.shape
    K = sk_t.shape[1]
    sample_lo = 0
    sample_hi = T
    if args.sample_range:
        lo, hi = args.sample_range.split(":")
        sample_lo = int(lo) if lo else 0
        sample_hi = int(hi) if hi else T
        M = M[:, sample_lo:sample_hi]
    print(
        f"[INFO] S={S} unique 'Z' captures, T={T}, K={K}, "
        f"hw_model={args.hw_model}, sample_range=[{sample_lo}:{sample_hi}]"
    )

    label_to_idx = {-1: 0, 0: 1, 1: 2}
    idx_to_ternary = {0: -1, 1: 0, 2: 1}
    L = np.zeros_like(sk_t)
    for s in range(S):
        for k in range(K):
            L[s, k] = label_to_idx[int(sk_t[s, k])]

    # HW maps
    def hw_for(label_value: int) -> int:
        sign_val = idx_to_ternary[label_value]
        if args.hw_model == "raw":
            return bin(int(sign_val) & 0xFFFF).count("1")
        elif args.hw_model == "shift":
            return bin((int(sign_val) << 11) & 0xFFFF).count("1")
        else:
            # 'both' 처리는 LOOCV 안에서 (per-coord)
            return bin(int(sign_val) & 0xFFFF).count("1")

    accs = []
    supps = []
    signs = []
    for hold in range(S):
        train_idx = [s for s in range(S) if s != hold]
        Mt = M[train_idx]
        Lt = L[train_idx]
        Mc = Mt - Mt.mean(axis=0, keepdims=True)
        Mn = np.linalg.norm(Mc, axis=0, keepdims=True)
        poi_t = -np.ones(K, dtype=np.int64)
        mu_per_class = np.full((K, 3), np.nan)
        sigma_per_coord = np.zeros(K)
        prior = np.zeros((K, 3))

        for k in range(K):
            lk = Lt[:, k]
            uniq = np.unique(lk)
            if len(uniq) < 2:
                continue
            # PoI: variant — 'both' 면 두 모델 다 시도해서 best |r| 선택
            best_t = -1
            best_absr = -1.0
            for model_choice in (
                ("raw", "shift") if args.hw_model == "both" else (args.hw_model,)
            ):
                hw = np.array([
                    bin((int(idx_to_ternary[c]) << (11 if model_choice == "shift" else 0))
                        & 0xFFFF).count("1")
                    for c in lk
                ], dtype=np.float64)
                if hw.var(ddof=1) <= 0:
                    continue
                Hc = hw - hw.mean()
                denom = (np.linalg.norm(Hc) * Mn[0])
                denom = np.where(denom == 0, 1.0, denom)
                r = (Hc @ Mc) / denom
                tb = int(np.argmax(np.abs(r)))
                ab = float(abs(r[tb]))
                if ab > best_absr:
                    best_absr = ab
                    best_t = tb
            if best_t < 0:
                continue
            poi_t[k] = best_t
            # 클래스별 평균 + pooled within-class variance at this PoI
            for cls in range(3):
                mask = lk == cls
                if mask.sum() == 0:
                    continue
                mu_per_class[k, cls] = float(Mt[mask, best_t].mean())
            total_var = 0.0
            total_dof = 0
            for cls in range(3):
                mask = lk == cls
                if mask.sum() > 1:
                    v = Mt[mask, best_t].var(ddof=1)
                    total_var += v * (mask.sum() - 1)
                    total_dof += mask.sum() - 1
            sigma_per_coord[k] = (
                np.sqrt(total_var / total_dof) if total_dof > 0 else 1e-3
            )
            for cls in range(3):
                prior[k, cls] = (lk == cls).sum() / lk.size

        Mte = M[hold]
        sk_te = sk_t[hold]
        ternary_pred = np.zeros(K, dtype=np.int64)
        for k in range(K):
            if poi_t[k] < 0 or sigma_per_coord[k] <= 0:
                ternary_pred[k] = 0
                continue
            x = Mte[poi_t[k]]
            mu = mu_per_class[k]
            sig = sigma_per_coord[k]
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
            if sign_mask.any() else 0.0
        )
        accs.append(coord_acc)
        supps.append(supp_acc)
        signs.append(sign_acc)

    base_zero = float((sk_t == 0).mean())
    print(
        f"[LOO {S}-fold, hw_model={args.hw_model}, conservative={args.conservative}]\n"
        f"  mean coord_acc = {np.mean(accs):.4f} (predict-zero baseline = {base_zero:.4f})\n"
        f"  mean supp_acc  = {np.mean(supps):.4f}\n"
        f"  mean sign_acc  = {np.mean(signs):.4f}"
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".txt").write_text(
        f"S2 'Z' LOO 3-class Bayes\n"
        f"S={S} sk, T={T}, K={K}\n"
        f"hw_model={args.hw_model}, conservative={args.conservative}\n"
        f"baseline pred-zero = {base_zero:.4f}\n"
        f"mean coord_acc = {np.mean(accs):.4f}\n"
        f"mean supp_acc  = {np.mean(supps):.4f}\n"
        f"mean sign_acc  = {np.mean(signs):.4f}\n"
    )
    print(f"[OK] {args.out_prefix.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
