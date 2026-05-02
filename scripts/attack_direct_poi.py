#!/usr/bin/env python3
"""Phase H/F — Direct attack PoI learning (cross-domain transfer 우회).

기존 analyze_attack.py 가 random-µ profile 의 PoI 를 chosen-CT attack 에 transfer
하다 sign 반전 + signal weak 문제가 있었다.

이 스크립트는 *chosen-CT attack data 자체* 위에서 PoI 학습:
  - 여러 design 의 µ′ 가 deterministic per-design but design 간 변화 → 각 비트 i 의
    µ′_i 가 design 별 0 또는 1 분포.
  - 그 cross-design split 위에서 trace sample t × bit_i 의 Welch t-test → PoI.

이게 paper 의 핵심 contribution: 'cross-domain transfer 문제 우회 + direct attack
PoI 학습 → sk recovery 정확도 향상'.

용법:
    scripts/attack_direct_poi.py traces/H_attack_n512.npz \\
        --design-pos H_const_a64 --design-neg H_const_a192 \\
        --component 0 --out-prefix results/direct_poi_n512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.analysis import sparse_recover  # noqa: E402
from host.smaug.codec import unpack_sx  # noqa: E402
from host.smaug import params as _params  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", type=Path, help="run_h_attack.py 의 multi-design .npz")
    p.add_argument("--design-pos", type=str, default="H_const_a64",
                   help="oracle pair 의 +det 디자인 이름")
    p.add_argument("--design-neg", type=str, default="H_const_a192",
                   help="oracle pair 의 -det 디자인 이름")
    p.add_argument("--component", type=int, default=0, help="sk 의 component (0 또는 1)")
    p.add_argument("--out-prefix", type=str, default="results/direct_poi")
    p.add_argument("--min-class-size", type=int, default=100,
                   help="bit 별 두 클래스 최소 개수 (작으면 unreliable PoI)")
    return p.parse_args()


def learn_direct_attack_poi(traces: np.ndarray, mus: np.ndarray,
                            min_class: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """direct attack data 위 per-bit PoI/sign 학습.

    traces: (N, T)
    mus: (N, 32) — each trace's µ′ (32 bytes = 256 bits, LSB-first per byte)
    min_class: bit 의 두 클래스 (0/1) 최소 trace 수. 미달시 -1 (invalid).

    Returns: (poi, t_score, sign), 각 (256,).
    """
    N, _ = traces.shape
    bit_mat = np.zeros((N, 256), dtype=np.int8)
    for i in range(N):
        for k in range(256):
            bit_mat[i, k] = (int(mus[i, k // 8]) >> (k % 8)) & 1

    poi = np.zeros(256, dtype=np.int64)
    t_score = np.zeros(256, dtype=np.float64)
    sign = np.zeros(256, dtype=np.int8)
    for bi in range(256):
        bits = bit_mat[:, bi]
        n0 = int((bits == 0).sum()); n1 = int((bits == 1).sum())
        if n0 < min_class or n1 < min_class:
            poi[bi] = -1
            continue
        m0 = traces[bits == 0].mean(axis=0)
        m1 = traces[bits == 1].mean(axis=0)
        v0 = traces[bits == 0].var(axis=0, ddof=1).clip(min=1e-12)
        v1 = traces[bits == 1].var(axis=0, ddof=1).clip(min=1e-12)
        se = np.sqrt(v0 / n0 + v1 / n1)
        t = (m1 - m0) / se
        poi[bi] = int(np.argmax(np.abs(t)))
        t_score[bi] = float(t[poi[bi]])
        sign[bi] = +1 if t_score[bi] > 0 else -1
    return poi, t_score, sign


def attack_with_oracle_pair(traces: np.ndarray, design_idx: np.ndarray,
                            d_pos: int, d_neg: int,
                            poi: np.ndarray, sign: np.ndarray,
                            ) -> np.ndarray:
    """oracle pair 분류 → per-i signed score d.

    d[i] = sign[i] * (mean_pos[poi[i]] - mean_neg[poi[i]]).
    """
    mean_p = traces[design_idx == d_pos].mean(axis=0)
    mean_n = traces[design_idx == d_neg].mean(axis=0)
    diff = mean_p - mean_n
    d = np.zeros(256)
    for i in range(256):
        if poi[i] >= 0:
            d[i] = float(sign[i]) * float(diff[int(poi[i])])
    return d


def signed_score_to_posterior(d: np.ndarray) -> np.ndarray:
    sigma = max(float(d.std()), 1e-9)
    pos = np.abs(d[d > 0]); neg = np.abs(d[d < 0])
    mu = ((pos.mean() if pos.size else 1.0) + (neg.mean() if neg.size else 1.0)) / 2
    log_neg = -0.5 * ((d + mu) / sigma) ** 2
    log_zer = -0.5 * (d / sigma) ** 2
    log_pos = -0.5 * ((d - mu) / sigma) ** 2
    L = np.stack([log_neg, log_zer, log_pos], axis=1)
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L); P /= P.sum(axis=1, keepdims=True)
    return P


def main() -> int:
    args = parse_args()
    p = _params.SMAUG1
    hs = p.hs

    print(f"[INFO] {args.npz}")
    d = np.load(args.npz, allow_pickle=True)
    traces = d["traces"]; mus = d["mu_prime"]; meta = d["meta"].item()
    names = list(meta["design_names"])
    idx = np.array(meta["design_index_per_trace"])
    sk_bytes = bytes.fromhex(meta["sk_pke_bytes_hex"])
    sk = unpack_sx(sk_bytes).reshape(p.module_rank, p.n).astype(np.int64)
    s_gt = sk[args.component].astype(np.int8)
    print(f"  shape: {traces.shape}, designs: {names}, sk[{args.component}] HW={int(np.count_nonzero(s_gt))}")

    poi, t_score, sign = learn_direct_attack_poi(traces, mus,
                                                 min_class=args.min_class_size)
    valid = poi >= 0
    n_valid = int(valid.sum())
    n_signal = int((np.abs(t_score[valid]) > 4.5).sum())
    print(f"\n[Direct attack PoI]")
    print(f"  valid bits: {n_valid}/256")
    print(f"  max|t|={float(np.abs(t_score[valid]).max()) if n_valid else 0:.2f}")
    print(f"  mean|t|={float(np.abs(t_score[valid]).mean()) if n_valid else 0:.2f}")
    print(f"  bits |t|>4.5: {n_signal}/{n_valid}")

    if args.design_pos not in names or args.design_neg not in names:
        print(f"[ERROR] design_pos/neg not in {names}")
        return 1
    d_pos_i = names.index(args.design_pos)
    d_neg_i = names.index(args.design_neg)

    d_score = attack_with_oracle_pair(traces, idx, d_pos_i, d_neg_i, poi, sign)
    corr = float(np.corrcoef(d_score, s_gt.astype(float))[0, 1])
    print(f"\n[Oracle pair attack: {args.design_pos} vs {args.design_neg}, comp={args.component}]")
    print(f"  d vs sk corr: r={corr:.4f}")

    sigma = max(d_score.std(), 1e-9)
    print("\n  Raw threshold:")
    for k in [0.5, 1.0, 1.5, 2.0]:
        th = k * sigma
        pred = np.where(d_score > th, +1, np.where(d_score < -th, -1, 0)).astype(np.int8)
        m = sparse_recover.accuracy(pred, s_gt)
        print(f"    th={k:.1f}σ: bit_acc={m['bit_accuracy']:.3f} "
              f"sup={m['support_accuracy']:.3f} sign={m['sign_accuracy']:.3f} "
              f"HW={int(np.count_nonzero(pred))}")

    P = signed_score_to_posterior(d_score)
    print("\n  sparse_recover (HW=70 MAP):")
    s_pred = sparse_recover.greedy_sparse_recover(P, hs)
    m_sp = sparse_recover.accuracy(s_pred, s_gt)
    print(f"    bit_acc={m_sp['bit_accuracy']:.3f} "
          f"sup={m_sp['support_accuracy']:.3f} sign={m_sp['sign_accuracy']:.3f}")

    # save
    out = Path(args.out_prefix + "_summary.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write(f"# Direct attack PoI — {args.npz.name}\n")
        fh.write(f"oracle pair: {args.design_pos} vs {args.design_neg}, component={args.component}\n")
        fh.write(f"valid bits: {n_valid}/256, signal |t|>4.5: {n_signal}\n")
        fh.write(f"max|t|={float(np.abs(t_score[valid]).max()) if n_valid else 0:.2f}\n")
        fh.write(f"d vs sk corr: r={corr:.4f}\n")
        fh.write(f"sparse_recover hs=70: bit_acc={m_sp['bit_accuracy']:.3f}, "
                 f"sign_acc={m_sp['sign_accuracy']:.3f}\n")
    print(f"\n[OK] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
