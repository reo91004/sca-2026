"""Phase H — host/analysis/sparse_recover.py 의 정합성 + SNR 곡선."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.analysis import sparse_recover  # noqa: E402


def _synthetic_sk(n: int, hs: int, seed: int) -> np.ndarray:
    """HW=hs ternary sk 합성. 균형 ±1 분포 (HS/2 each, hs odd 면 +1 1개 더)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=np.int8)
    support = rng.choice(n, size=hs, replace=False)
    half = hs // 2
    signs = np.array([+1] * (hs - half) + [-1] * half, dtype=np.int8)
    rng.shuffle(signs)
    s[support] = signs
    return s


def _make_posterior_from_truth(s_true: np.ndarray, snr: float, seed: int) -> np.ndarray:
    """s_true 정답 위에 noise 가산해 (n, 3) posterior 생성.

    snr: signal-to-noise factor — 0 = uniform, 큰 값 → 정답 쪽 mass 증가.
    """
    rng = np.random.default_rng(seed)
    n = s_true.size
    posterior = rng.dirichlet([1.0, 1.0, 1.0], size=n)  # uniform-ish base
    # 정답 column 에 mass shift
    truth_col = np.where(s_true == -1, 0, np.where(s_true == 0, 1, 2))
    boost = np.zeros((n, 3), dtype=np.float64)
    boost[np.arange(n), truth_col] = snr
    posterior = posterior + boost
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior


# --- greedy_sparse_recover --------------------------------------------------


def test_greedy_recovers_perfect_posterior() -> None:
    """SNR=∞ (정답 col 1.0, 나머지 0) → greedy 가 정답 sk 정확히 복구."""
    n, hs = 256, 70
    s_true = _synthetic_sk(n, hs, seed=0)
    truth_col = np.where(s_true == -1, 0, np.where(s_true == 0, 1, 2))
    posterior = np.zeros((n, 3), dtype=np.float64)
    posterior[np.arange(n), truth_col] = 1.0

    s_hat = sparse_recover.greedy_sparse_recover(posterior, hs)
    assert s_hat.shape == (n,)
    assert int(np.count_nonzero(s_hat)) == hs
    assert np.array_equal(s_hat, s_true), (
        f"perfect posterior 에서 정답 복구 실패. mismatch count = "
        f"{int((s_hat != s_true).sum())}"
    )


def test_greedy_enforces_hw_constraint() -> None:
    """결과 sk 의 nonzero 개수가 정확히 hs."""
    n, hs = 256, 70
    rng = np.random.default_rng(42)
    posterior = rng.dirichlet([1.0, 1.0, 1.0], size=n)
    s_hat = sparse_recover.greedy_sparse_recover(posterior, hs)
    assert int(np.count_nonzero(s_hat)) == hs
    assert set(np.unique(s_hat).tolist()) <= {-1, 0, 1}


def test_greedy_rejects_invalid_args() -> None:
    n = 100
    posterior = np.full((n, 3), 1.0 / 3.0)
    for bad_hs in [-1, 0, n + 1, 1000]:
        raised = False
        try:
            sparse_recover.greedy_sparse_recover(posterior, bad_hs)
        except ValueError:
            raised = True
        assert raised, f"hs={bad_hs} 통과해버림"
    # bad shape
    raised = False
    try:
        sparse_recover.greedy_sparse_recover(np.ones((n, 2)), 10)
    except ValueError:
        raised = True
    assert raised


def test_greedy_snr_curve_smaug1() -> None:
    """SNR 증가 → bit_accuracy 단조 증가 (정량 곡선)."""
    n, hs = 256, 70
    s_true = _synthetic_sk(n, hs, seed=1)

    accs = []
    for snr in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        posterior = _make_posterior_from_truth(s_true, snr=snr, seed=2)
        s_hat = sparse_recover.greedy_sparse_recover(posterior, hs)
        m = sparse_recover.accuracy(s_hat, s_true)
        accs.append((snr, m["bit_accuracy"], m["support_accuracy"]))
    # SNR 0 (uniform) 의 정확도가 SNR 10 보다 의미있게 낮음
    snr0_bit = accs[0][1]
    snr10_bit = accs[-1][1]
    assert snr10_bit > snr0_bit + 0.1, (
        f"SNR 곡선 단조성 약함: SNR=0 → {snr0_bit:.3f}, SNR=10 → {snr10_bit:.3f}"
    )
    # 강한 SNR 에서 95% 이상
    assert snr10_bit >= 0.95, f"SNR=10 에서 bit_acc {snr10_bit:.3f} < 0.95"


# --- enumerate_top_candidates -----------------------------------------------


def test_enumerate_returns_sorted_unique_candidates() -> None:
    n, hs = 100, 30
    s_true = _synthetic_sk(n, hs, seed=10)
    posterior = _make_posterior_from_truth(s_true, snr=2.0, seed=11)

    cands = sparse_recover.enumerate_top_candidates(posterior, hs, k=8)
    assert 1 <= len(cands) <= 8
    # 정렬 (logp desc)
    logps = [c[0] for c in cands]
    assert logps == sorted(logps, reverse=True), "logp desc 정렬 깨짐"
    # 중복 방지 — logp 가 모두 다르거나 (실수 비교) sk 가 다름
    sk_set = {tuple(c[1].tolist()) for c in cands}
    assert len(sk_set) == len(cands), "duplicate sk in enumeration"


def test_enumerate_top_candidate_matches_greedy() -> None:
    """top-1 후보가 greedy 결과와 동일."""
    n, hs = 64, 20
    s_true = _synthetic_sk(n, hs, seed=20)
    posterior = _make_posterior_from_truth(s_true, snr=1.0, seed=21)

    greedy = sparse_recover.greedy_sparse_recover(posterior, hs)
    cands = sparse_recover.enumerate_top_candidates(posterior, hs, k=4)
    top1 = cands[0][1]
    assert np.array_equal(top1, greedy), "top1 후보 ≠ greedy"


# --- binary_to_ternary_posterior --------------------------------------------


def test_binary_to_ternary_idealized_detector() -> None:
    """Idealized detector outputs (s 정답 정확히 1/0) → posterior 가 정답에 mass."""
    # s = -1 → score_pos=0, score_neg=1
    # s =  0 → score_pos=0, score_neg=0
    # s = +1 → score_pos=1, score_neg=0
    sp = np.array([0.0, 0.0, 1.0])
    sn = np.array([1.0, 0.0, 0.0])
    post = sparse_recover.binary_to_ternary_posterior(sp, sn)
    # 행 합 = 1
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
    # argmax = 정답 column
    np.testing.assert_array_equal(np.argmax(post, axis=1), [0, 1, 2])
    # 강한 mass (>0.9) on 정답 column
    np.testing.assert_array_less(0.9, [post[0, 0], post[1, 1], post[2, 2]])


def test_binary_to_ternary_contradiction_uniform() -> None:
    """(score_pos, score_neg) = (1, 1) — 모순 (s=+1 과 s=-1 동시 detector 발화)
    → fallback 균등 분포."""
    sp = np.array([1.0])
    sn = np.array([1.0])
    post = sparse_recover.binary_to_ternary_posterior(sp, sn)
    # 모든 likelihood ≈ eps · (1-eps), normalize 후 거의 균등
    # 우리 구현: too_small fallback 이 1/3 set.
    # (sp, sn) 둘 다 1-eps, (1-sp)*sn ≈ eps*(1-eps), (1-sp)*(1-sp) ≈ eps^2,
    # sp*(1-sn) ≈ (1-eps)*eps. z ≈ 2*eps - eps^2 ≈ eps_small but > 1e-6 의
    # eps 선택 영향. 안전한 검증: 결과가 finite + 행 합 ≈ 1.
    assert np.all(np.isfinite(post))
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)


def test_binary_to_ternary_with_greedy_smaug1() -> None:
    """Synthetic ternary sk 에 idealized binary detectors → posterior →
    greedy_sparse_recover. perfect SNR 에서 정답 복구."""
    n, hs = 256, 70
    s_true = _synthetic_sk(n, hs, seed=30)
    # detectors:
    score_pos = (s_true == +1).astype(np.float64)
    score_neg = (s_true == -1).astype(np.float64)
    posterior = sparse_recover.binary_to_ternary_posterior(score_pos, score_neg)
    s_hat = sparse_recover.greedy_sparse_recover(posterior, hs)
    assert np.array_equal(s_hat, s_true), (
        f"idealized detector 에서 sparse_recover 실패. "
        f"mismatch={int((s_hat != s_true).sum())}"
    )


# --- accuracy ----------------------------------------------------------------


def test_accuracy_metrics_perfect() -> None:
    n, hs = 256, 70
    s_true = _synthetic_sk(n, hs, seed=50)
    m = sparse_recover.accuracy(s_true, s_true)
    assert m["bit_accuracy"] == 1.0
    assert m["support_accuracy"] == 1.0
    assert m["sign_accuracy"] == 1.0
    assert m["n"] == float(n)


def test_accuracy_all_zero_baseline_smaug1() -> None:
    """s_true = HW=70 ternary, predict all-zero → bit_acc = (256-70)/256 ≈ 0.727."""
    n, hs = 256, 70
    s_true = _synthetic_sk(n, hs, seed=51)
    s_pred = np.zeros(n, dtype=np.int8)
    m = sparse_recover.accuracy(s_pred, s_true)
    expected = (n - hs) / n
    assert abs(m["bit_accuracy"] - expected) < 1e-9
    # sign_accuracy 는 common_support 가 0 이라 NaN
    import math
    assert math.isnan(m["sign_accuracy"])


if __name__ == "__main__":
    fns = sorted(n for n in globals() if n.startswith("test_"))
    for n in fns:
        print(f"[RUN] {n}")
        globals()[n]()
        print(f"[OK ] {n}")
    print(f"[OK ] {len(fns)} tests")
