#!/usr/bin/env python3
"""Phase H/F 통합 — analyze_attack 결과 위 sparse_recover (HW=HS MAP) 후처리.

기존 analyze_attack.py 가 ternary 분류 결과 (s_pred ∈ {-1, 0, +1}) 만 제공.
이 스크립트는 *signed score d_i* (raw) 를 추출 → posterior π_i ∈ Δ^3 →
sparse_recover.greedy 로 HW=HS constraint MAP 복구.

용법:
    scripts/analyze_attack_with_sparse.py \\
        --profile traces/profile_random_mu_n1000.npz \\
        --attack traces/attack_const_c1.npz \\
        [--out-prefix results/sparse_v1]
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

# 재사용
sys.path.insert(0, str(_REPO / "scripts"))
from analyze_attack import learn_per_i_poi  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--attack", type=Path, required=True)
    p.add_argument("--out-prefix", type=str, default="results/sparse_v1")
    return p.parse_args()


def extract_signed_score(attack_traces: np.ndarray,
                         attack_labels: np.ndarray,
                         poi: np.ndarray, sign: np.ndarray,
                         alpha_pos: int, alpha_neg: int,
                         component: int) -> np.ndarray:
    """component 의 256 비트에 대한 signed score d_i.

    d_i = sign[i] * (mean_p[poi[i]] - mean_n[poi[i]])
    """
    label_comp = attack_labels[:, 0]
    label_alpha = attack_labels[:, 1]
    mask_p = (label_comp == component) & (label_alpha == alpha_pos)
    mask_n = (label_comp == component) & (label_alpha == alpha_neg)
    if not mask_p.any() or not mask_n.any():
        raise ValueError(f"missing for component={component}")
    mean_p = attack_traces[mask_p].mean(axis=0)
    mean_n = attack_traces[mask_n].mean(axis=0)
    diff = mean_p - mean_n

    n_bits = poi.size
    d = np.zeros(n_bits, dtype=np.float64)
    for i in range(n_bits):
        if poi[i] < 0:
            d[i] = 0.0
            continue
        d[i] = float(sign[i]) * float(diff[int(poi[i])])
    return d


def signed_score_to_posterior(d: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """signed score d ∈ R → ternary posterior (n, 3).

    Model: 정답 s ∈ {-1, 0, +1} 라면 d 의 likelihood:
        s = +1 → d ~ N(+μ, σ²)   (signal 양수)
        s = -1 → d ~ N(-μ, σ²)   (signal 음수)
        s =  0 → d ~ N(0,    σ²) (signal 없음)
    여기서 μ = expected magnitude (= |d| 의 어느 정도). 단순화: μ = d 의 표본 std.

    posterior ∝ N(d - μ_class, σ²) for each class.
    """
    n = d.size
    sigma = sigma if sigma is not None else float(np.std(d))
    if sigma <= 0:
        # uniform fallback
        return np.full((n, 3), 1.0 / 3.0)
    # μ class 위치: 정답 magnitude 추정. d 의 |d| 분포의 mean 또는 quantile.
    mu_pos = float(np.mean(np.abs(d[d > 0]))) if (d > 0).any() else 1.0
    mu_neg = float(np.mean(np.abs(d[d < 0]))) if (d < 0).any() else 1.0
    mu_mag = (mu_pos + mu_neg) / 2.0
    # likelihoods (Gaussian, unnormalized — exponent 만)
    log_neg = -0.5 * ((d - (-mu_mag)) / sigma) ** 2
    log_zer = -0.5 * (d / sigma) ** 2
    log_pos = -0.5 * ((d - mu_mag) / sigma) ** 2
    # posterior = softmax over rows
    log_stack = np.stack([log_neg, log_zer, log_pos], axis=1)  # (n, 3)
    log_stack -= log_stack.max(axis=1, keepdims=True)  # 안정화
    p = np.exp(log_stack)
    p /= p.sum(axis=1, keepdims=True)
    return p


def main() -> int:
    args = parse_args()
    p_smaug = _params.SMAUG1
    hs = p_smaug.hs

    print(f"[INFO] profile = {args.profile}, attack = {args.attack}, hs={hs}")

    prof = np.load(args.profile, allow_pickle=True)
    prof_traces = prof['traces']
    prof_mu = prof['mu_prime']
    print(f"  profile {prof_traces.shape}")

    poi, t_max, sign = learn_per_i_poi(prof_traces, prof_mu, n_bits=256)
    valid = poi >= 0
    print(f"  PoI learned for {int(valid.sum())}/256 bits, max|t|={float(np.abs(t_max[valid]).max()):.2f}")

    att = np.load(args.attack, allow_pickle=True)
    att_traces = att['traces']
    ameta = att['meta'].item()
    label_comp = np.asarray(ameta['label_component'])
    label_alpha = np.asarray(ameta['label_alpha'])
    labels = np.column_stack([label_comp, label_alpha])
    sk_hex = ameta['sk_pke_bytes_hex']
    sk_bytes = bytes.fromhex(sk_hex)
    s0_gt = unpack_sx(sk_bytes[:64]).astype(np.int8)
    s1_gt = unpack_sx(sk_bytes[64:128]).astype(np.int8)
    print(f"  attack {att_traces.shape}, GT s[0] HW={int((s0_gt!=0).sum())}, "
          f"s[1] HW={int((s1_gt!=0).sum())}")

    ap = int(ameta['alpha_pos']); an = int(ameta['alpha_neg'])

    print(f"\n[Signed score extraction]")
    d0 = extract_signed_score(att_traces, labels, poi, sign, ap, an, component=0)
    d1 = extract_signed_score(att_traces, labels, poi, sign, ap, an, component=1)
    print(f"  s[0] d range: [{d0.min():.4f}, {d0.max():.4f}], std={d0.std():.4f}")
    print(f"  s[1] d range: [{d1.min():.4f}, {d1.max():.4f}], std={d1.std():.4f}")

    print(f"\n[Posterior + sparse_recover]")
    post0 = signed_score_to_posterior(d0)
    post1 = signed_score_to_posterior(d1)

    # Baseline raw (threshold-based) for comparison
    sigma0 = float(np.std(d0))
    sigma1 = float(np.std(d1))
    th0 = sigma0 * 1.0
    th1 = sigma1 * 1.0
    raw0 = np.where(d0 > th0, +1, np.where(d0 < -th0, -1, 0)).astype(np.int8)
    raw1 = np.where(d1 > th1, +1, np.where(d1 < -th1, -1, 0)).astype(np.int8)

    # Sparse recovery (HW constraint)
    s0_sparse = sparse_recover.greedy_sparse_recover(post0, hs)
    s1_sparse = sparse_recover.greedy_sparse_recover(post1, hs)

    def report(pred, gt, name):
        m = sparse_recover.accuracy(pred, gt)
        print(f"  {name}: bit_acc={m['bit_accuracy']:.3f} "
              f"support_acc={m['support_accuracy']:.3f} "
              f"sign_acc={m['sign_accuracy']:.3f} "
              f"HW(pred)={int(np.count_nonzero(pred))}")

    print("\n  Raw threshold (threshold_sigma=1.0):")
    report(raw0, s0_gt, "  s[0] raw")
    report(raw1, s1_gt, "  s[1] raw")

    print("\n  After sparse_recover (HW=70 MAP):")
    report(s0_sparse, s0_gt, "  s[0] sparse")
    report(s1_sparse, s1_gt, "  s[1] sparse")

    # Combined full sk
    raw_full = int((raw0 == s0_gt).sum() + (raw1 == s1_gt).sum())
    sparse_full = int((s0_sparse == s0_gt).sum() + (s1_sparse == s1_gt).sum())
    print(f"\n  Full sk: raw={raw_full}/512 ({100*raw_full/512:.1f}%), "
          f"sparse={sparse_full}/512 ({100*sparse_full/512:.1f}%)")

    # Save
    out_dir = (_REPO / args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = _REPO / (args.out_prefix + "_summary.txt")
    with txt_path.open("w") as fh:
        fh.write(f"# sparse_recover post-processing — profile={args.profile.name}, "
                 f"attack={args.attack.name}\n\n")
        fh.write(f"profile: {prof_traces.shape}, max|t|={float(np.abs(t_max[valid]).max()):.2f}\n")
        fh.write(f"attack:  {att_traces.shape}\n")
        fh.write(f"GT:  s[0] HW={int((s0_gt!=0).sum())}, s[1] HW={int((s1_gt!=0).sum())}\n\n")

        fh.write("Raw threshold:\n")
        m0 = sparse_recover.accuracy(raw0, s0_gt)
        m1 = sparse_recover.accuracy(raw1, s1_gt)
        fh.write(f"  s[0] bit={m0['bit_accuracy']:.3f} sup={m0['support_accuracy']:.3f} "
                 f"sign={m0['sign_accuracy']:.3f} HW={int(np.count_nonzero(raw0))}\n")
        fh.write(f"  s[1] bit={m1['bit_accuracy']:.3f} sup={m1['support_accuracy']:.3f} "
                 f"sign={m1['sign_accuracy']:.3f} HW={int(np.count_nonzero(raw1))}\n")
        fh.write(f"  full sk: {raw_full}/512 ({100*raw_full/512:.1f}%)\n\n")

        fh.write("Sparse_recover (HW=70 MAP):\n")
        m0s = sparse_recover.accuracy(s0_sparse, s0_gt)
        m1s = sparse_recover.accuracy(s1_sparse, s1_gt)
        fh.write(f"  s[0] bit={m0s['bit_accuracy']:.3f} sup={m0s['support_accuracy']:.3f} "
                 f"sign={m0s['sign_accuracy']:.3f} HW={int(np.count_nonzero(s0_sparse))}\n")
        fh.write(f"  s[1] bit={m1s['bit_accuracy']:.3f} sup={m1s['support_accuracy']:.3f} "
                 f"sign={m1s['sign_accuracy']:.3f} HW={int(np.count_nonzero(s1_sparse))}\n")
        fh.write(f"  full sk: {sparse_full}/512 ({100*sparse_full/512:.1f}%)\n")
    print(f"\n[OK] saved {txt_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 7))
        for ax, gt, raw, sp, name in zip(axes, [s0_gt, s1_gt],
                                         [raw0, raw1], [s0_sparse, s1_sparse],
                                         ['s[0]', 's[1]']):
            ax.scatter(np.arange(256), gt, label='GT', s=14, color='black', marker='_')
            err_raw = raw != gt
            ax.scatter(np.arange(256)[err_raw], raw[err_raw], label='raw err',
                       s=18, alpha=0.5, color='gray', marker='x')
            err_sp = sp != gt
            ax.scatter(np.arange(256)[err_sp], sp[err_sp], label='sparse err',
                       s=22, alpha=0.7, color='red', marker='+')
            raw_acc = float((raw == gt).mean())
            sp_acc = float((sp == gt).mean())
            ax.set_title(f"{name}: raw={100*raw_acc:.1f}% → sparse_recover={100*sp_acc:.1f}%")
            ax.set_yticks([-1, 0, 1])
            ax.set_xlim(-2, 258); ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        axes[-1].set_xlabel('coefficient index i')
        fig.tight_layout()
        png_path = _REPO / (args.out_prefix + ".png")
        fig.savefig(png_path, dpi=120)
        print(f"[OK] saved {png_path}")
    except Exception as e:
        print(f"[WARN] plot fail: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
