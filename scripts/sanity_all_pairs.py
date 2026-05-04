#!/usr/bin/env python3
"""Sanity check — *모든* (calib, target) pair 의 cross-key schedule transfer.

지금까지 실험들이 calib=seed1 한 가지로만 cross-key 평가했다. 사용자 질문:
"정말 계속 랜덤으로 수렴해? 비밀키 여러개 측정해도?"

이 스크립트는 9 distinct smaug1 keys 의 9×8=72 ordered pair 를 모두 sweep:
  (a) µ′-profile schedule (mu-prime PoI from calib applied to target)
  (b) FFT design-window schedule (calib FFT-PoI applied to target FFT)

random baseline 의 95% CI 안에 모든 pair 가 들어가는지 검증.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from scripts.analyze_multi_seed import analyze_one, learn_schedule_from_npz  # noqa: E402
from scripts.fft_design_window import cross_key_eval as fft_cross_key  # noqa: E402


def random_baseline(n: int, hs: int) -> tuple[float, float]:
    """sparse_recover 의 random expectation + std (binomial approx).

    bit_acc per coef = P(both zero) + P(both nonzero)/2.
    full_sk_acc = mean of bit_acc across n coefs.
    Var ≈ p(1-p)/n with p = mean.
    """
    pz = (n - hs) / float(n); pn = hs / float(n)
    p = pz * pz + pn * pn * 0.5
    std = (p * (1 - p) / n) ** 0.5
    return p, std


def main() -> int:
    paths = sorted(Path("traces").glob("attack_seed*.npz"))
    paths = [p for p in paths if "smaug3" not in p.name and "smaug5" not in p.name]
    paths += [Path("traces/attack_const_c1_n256.npz")]
    paths = [p for p in paths if p.exists()]
    K = len(paths)
    n_total = 512  # 2 components × 256
    hs_total = 140  # 2 × 70
    p_rand, p_std = random_baseline(n_total, hs_total)
    ci_lo = p_rand - 1.96 * p_std
    ci_hi = p_rand + 1.96 * p_std
    print(f"[INFO] {K} distinct keys")
    print(f"[INFO] random baseline (HW={hs_total}/{n_total} cstr): "
          f"{p_rand:.4f} ± {p_std:.4f} (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])\n")

    # === (a) mu-prime schedule transfer (full cross-pair matrix) ===
    print(f"=== (a) µ′-profile schedule transfer ({K}×{K-1}={K*(K-1)} pairs) ===")
    M_mu = np.full((K, K), np.nan)
    for i, calib in enumerate(paths):
        sched = learn_schedule_from_npz(calib)
        sk_calib = sched["_meta"]["sk_hash_short"]
        for j, target in enumerate(paths):
            if i == j:
                continue
            d = np.load(target, allow_pickle=True)
            sk_tgt = d["meta"].item()["sk_pke_bytes_hex"][:16]
            if sk_tgt == sk_calib:
                continue
            r = analyze_one(target, poi_source="schedule", schedule=sched)
            M_mu[i, j] = r["full_sk_acc"]
        print(f"  calib={paths[i].name:35s} → mean over targets: "
              f"{np.nanmean(M_mu[i]):.4f} (max {np.nanmax(M_mu[i]):.4f})")

    valid = M_mu[~np.isnan(M_mu)]
    print(f"\n  µ′-schedule across {len(valid)} pairs:")
    print(f"    mean ± std: {valid.mean():.4f} ± {valid.std():.4f}")
    print(f"    min/median/max: {valid.min():.4f} / {np.median(valid):.4f} / {valid.max():.4f}")
    n_above = int((valid > ci_hi).sum())
    n_below = int((valid < ci_lo).sum())
    print(f"    pairs above random 95% CI: {n_above}/{len(valid)}")
    print(f"    pairs below random 95% CI: {n_below}/{len(valid)}")

    # === (b) FFT-magnitude schedule transfer ===
    print(f"\n=== (b) FFT-magnitude design-window cross-key ===")
    M_fft = np.full((K, K), np.nan)
    for i, calib in enumerate(paths):
        for j, target in enumerate(paths):
            if i == j: continue
            try:
                r = fft_cross_key(calib, target)
                M_fft[i, j] = r["full_sk_acc"]
            except Exception as e:
                print(f"    [skip] {paths[i].name}→{paths[j].name}: {e}")
        print(f"  calib={paths[i].name:35s} → mean: {np.nanmean(M_fft[i]):.4f}, "
              f"max: {np.nanmax(M_fft[i]):.4f}")

    valid_f = M_fft[~np.isnan(M_fft)]
    print(f"\n  FFT-schedule across {len(valid_f)} pairs:")
    print(f"    mean ± std: {valid_f.mean():.4f} ± {valid_f.std():.4f}")
    print(f"    min/median/max: {valid_f.min():.4f} / {np.median(valid_f):.4f} / {valid_f.max():.4f}")
    n_above_f = int((valid_f > ci_hi).sum())
    print(f"    pairs above random 95% CI: {n_above_f}/{len(valid_f)}")

    # save
    Path("results").mkdir(exist_ok=True)
    np.savez("results/sanity_all_pairs.npz",
             paths=np.array([p.name for p in paths]),
             M_mu_schedule=M_mu, M_fft_schedule=M_fft,
             ci_lo=ci_lo, ci_hi=ci_hi, p_rand=p_rand)
    print(f"\n[OK] results/sanity_all_pairs.npz")

    print(f"\n=== verdict ===")
    if n_above == 0 and n_above_f == 0:
        print("  ✓ All-pairs sweep: NO pair (out of {}+{}) exceeds random 95% CI.".format(
            len(valid), len(valid_f)))
        print("  → Random level claim is robust across all calib/target combinations.")
    else:
        print(f"  △ Some pairs above CI: µ′ {n_above}, FFT {n_above_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
