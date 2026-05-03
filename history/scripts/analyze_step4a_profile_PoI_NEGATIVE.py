#!/usr/bin/env python3
"""profile + attack 트레이스로 full sk 복구 + ground truth 정확도 계산.

흐름:
  1) Profile (random µ × N) → per-bit Welch t-test → per-i PoI map.
  2) Attack (4 chosen-CT, c1=α 상수, α=64/192 × component=0/1) → per-CT mean trace.
  3) 각 i ∈ [0, n) 에 대해:
       d_i = (mean_α=64 - mean_α=192) at PoI_i × sign_per_i
       d_i > +threshold → s_i = +1
       d_i < -threshold → s_i = -1
       |d_i| < threshold → s_i = 0
  4) X dump 의 ground truth 와 비교.

용법:
    scripts/analyze_attack.py \\
        --profile traces/profile_random_mu_n1000.npz \\
        --attack traces/attack_const_c1.npz \\
        [--png results/attack_full_sk.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.smaug.codec import unpack_sx  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--attack", type=Path, required=True)
    p.add_argument("--png", type=Path, default=None)
    p.add_argument("--threshold-sigma", type=float, default=1.0,
                   help="0 결정 임계 (σ 단위, attack diff std 기준)")
    return p.parse_args()


def learn_per_i_poi(prof_traces: np.ndarray, prof_mu: np.ndarray,
                    n_bits: int = 256
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """profile 데이터로 per-i PoI 학습.

    반환: (poi_per_i, t_per_i, sign_per_i).
    sign_per_i = +1 if mu1_template > mu0_template at PoI (i.e., bit=1 → 더 큼).
    """
    n_traces, n_samples = prof_traces.shape
    bit_mat = np.zeros((n_traces, n_bits), dtype=np.int8)
    for bi in range(n_bits):
        bit_mat[:, bi] = (prof_mu[:, bi // 8] >> (bi % 8)) & 1

    poi = np.zeros(n_bits, dtype=np.int64)
    t_max = np.zeros(n_bits, dtype=np.float64)
    sign = np.zeros(n_bits, dtype=np.int8)

    for bi in range(n_bits):
        bits = bit_mat[:, bi]
        n0 = int((bits == 0).sum()); n1 = int((bits == 1).sum())
        if n0 < 5 or n1 < 5:
            poi[bi] = -1; t_max[bi] = 0; sign[bi] = 0; continue
        m0 = prof_traces[bits == 0].mean(axis=0)
        m1 = prof_traces[bits == 1].mean(axis=0)
        v0 = prof_traces[bits == 0].var(axis=0, ddof=1)
        v1 = prof_traces[bits == 1].var(axis=0, ddof=1)
        se = np.maximum(np.sqrt(v0/n0 + v1/n1), 1e-12)
        t = (m1 - m0) / se
        poi[bi] = int(np.argmax(np.abs(t)))
        t_max[bi] = float(t[poi[bi]])
        sign[bi] = +1 if t_max[bi] > 0 else -1
    return poi, t_max, sign


def classify_attack(attack_traces: np.ndarray, attack_labels: np.ndarray,
                    poi: np.ndarray, sign: np.ndarray,
                    alpha_pos: int, alpha_neg: int,
                    component: int, threshold_sigma: float = 1.0
                    ) -> np.ndarray:
    """단일 component 의 ternary 비밀 벡터 (256 계수) 추정.

    label_component[i] == component 인 trace 만 추출 → α=pos/neg 분리 → 평균 →
    diff = mean_pos - mean_neg → PoI_i 에서 부호로 ternary 결정.
    """
    label_comp = attack_labels[:, 0]
    label_alpha = attack_labels[:, 1]
    mask_p = (label_comp == component) & (label_alpha == alpha_pos)
    mask_n = (label_comp == component) & (label_alpha == alpha_neg)
    if not mask_p.any() or not mask_n.any():
        raise ValueError(f"missing chosen-CT for component={component}, "
                         f"α=({alpha_pos}, {alpha_neg})")
    mean_p = attack_traces[mask_p].mean(axis=0)
    mean_n = attack_traces[mask_n].mean(axis=0)
    diff = mean_p - mean_n
    sigma = float(np.std(diff))
    th = threshold_sigma * sigma

    n_bits = poi.size
    pred = np.zeros(n_bits, dtype=np.int8)
    for i in range(n_bits):
        if poi[i] < 0: pred[i] = 0; continue
        d = int(sign[i]) * float(diff[int(poi[i])])
        if d > th: pred[i] = +1
        elif d < -th: pred[i] = -1
        else: pred[i] = 0
    return pred


def main() -> int:
    args = parse_args()

    prof = np.load(args.profile, allow_pickle=True)
    prof_traces = prof['traces']; prof_mu = prof['mu_prime']
    print(f"[profile] {prof_traces.shape}, mu shape {prof_mu.shape}")

    poi, t_max, sign = learn_per_i_poi(prof_traces, prof_mu, n_bits=256)
    valid = poi >= 0
    print(f"  PoI learned for {int(valid.sum())}/256 bits")
    print(f"  Welch |t|: max={float(np.abs(t_max[valid]).max()):.2f}, "
          f"mean={float(np.abs(t_max[valid]).mean()):.2f}, "
          f"min={float(np.abs(t_max[valid]).min()):.2f}")
    noise_floor = float(np.sqrt(2 * np.log(prof_traces.shape[1])))
    print(f"  noise floor √(2·ln·n_samples) ≈ {noise_floor:.2f}")
    print(f"  bits with |t| > noise_floor: "
          f"{int((np.abs(t_max[valid]) > noise_floor).sum())}/{int(valid.sum())}")

    att = np.load(args.attack, allow_pickle=True)
    att_traces = att['traces']
    ameta = att['meta'].item()
    label_comp = np.asarray(ameta['label_component'])
    label_alpha = np.asarray(ameta['label_alpha'])
    labels = np.column_stack([label_comp, label_alpha])
    sk_hex = ameta['sk_pke_bytes_hex']
    sk_bytes = bytes.fromhex(sk_hex)
    s0_gt = unpack_sx(sk_bytes[:64])
    s1_gt = unpack_sx(sk_bytes[64:128])
    print(f"\n[attack] {att_traces.shape}, components={sorted(set(label_comp.tolist()))}, "
          f"α={sorted(set(label_alpha.tolist()))}")
    print(f"  GT s[0] HW={int((s0_gt!=0).sum())}, s[1] HW={int((s1_gt!=0).sum())}")

    ap = int(ameta['alpha_pos']); an = int(ameta['alpha_neg'])
    print(f"\n=== Classification ===")
    s0_pred = classify_attack(att_traces, labels, poi, sign, ap, an,
                              component=0, threshold_sigma=args.threshold_sigma)
    s1_pred = classify_attack(att_traces, labels, poi, sign, ap, an,
                              component=1, threshold_sigma=args.threshold_sigma)

    def report(pred, gt, name):
        correct = int((pred == gt).sum())
        n = len(gt)
        # confusion: per-class precision/recall
        c_match = {}
        for cls in (-1, 0, +1):
            tp = int(((pred == cls) & (gt == cls)).sum())
            fp = int(((pred == cls) & (gt != cls)).sum())
            fn = int(((pred != cls) & (gt == cls)).sum())
            c_match[cls] = (tp, fp, fn)
        print(f"\n  {name}: {correct}/{n} = {100*correct/n:.1f}% accuracy")
        print(f"    GT dist: -1:{int((gt==-1).sum())} 0:{int((gt==0).sum())} +1:{int((gt==+1).sum())}")
        print(f"    pred:    -1:{int((pred==-1).sum())} 0:{int((pred==0).sum())} +1:{int((pred==+1).sum())}")
        for cls in (-1, 0, +1):
            tp, fp, fn = c_match[cls]
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            print(f"    s={cls:+d}: TP={tp}, FP={fp}, FN={fn}, "
                  f"prec={prec:.2f}, recall={rec:.2f}")

    report(s0_pred, s0_gt, "s[0]")
    report(s1_pred, s1_gt, "s[1]")

    # full sk accuracy
    full_correct = int((s0_pred == s0_gt).sum() + (s1_pred == s1_gt).sum())
    full_n = 512
    print(f"\n  full sk: {full_correct}/{full_n} = {100*full_correct/full_n:.1f}% "
          f"({full_n - full_correct} 비트 오류)")

    if args.png:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(14, 7))
            for ax, gt, pred, name in zip(axes, [s0_gt, s1_gt],
                                          [s0_pred, s1_pred],
                                          ['s[0]', 's[1]']):
                ax.scatter(np.arange(256), gt, label='ground truth', s=12,
                           color='black', marker='_')
                ax.scatter(np.arange(256), pred, label='predicted',
                           s=10, alpha=0.6, color='red', marker='+')
                err = pred != gt
                ax.scatter(np.arange(256)[err], pred[err], label='errors',
                           s=30, color='orange', marker='x')
                ax.set_ylabel(f'{name}')
                ax.set_yticks([-1, 0, 1])
                ax.set_xlim(-2, 258); ax.grid(alpha=0.3)
                acc = float((pred == gt).mean())
                ax.set_title(f'{name}: pred vs GT, acc={100*acc:.1f}%')
                ax.legend(loc='upper right', fontsize=8)
            axes[-1].set_xlabel('coefficient index i')
            fig.tight_layout()
            args.png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.png, dpi=110)
            print(f"[OK] {args.png}")
        except Exception as e:
            print(f"[WARN] PNG fail: {e}")

    return 0 if (s0_pred == s0_gt).all() and (s1_pred == s1_gt).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
