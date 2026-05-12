"""Phase H — sparse ternary recovery (HW=HS constraint MAP).

SMAUG-T sk 의 prior: HW(s) = HS = 70 per poly (smaug1). chosen-CT SCA 의 noisy
classifier 출력 위에서 이 *전역 sparse 제약* 을 강제해 recovery 정확도를 끌어
올린다 — 단일 비트 분류 SNR 한계 (예: 56%) 를 후처리로 우회.

입력 형태:
    posterior: shape (n, 3), 각 행이 (P[s=-1], P[s=0], P[s=+1]).
    hs: 전역 HW(s) 제약 값 (smaug1: 70).

API:
    greedy_sparse_recover(posterior, hs) → ŝ ∈ {-1, 0, +1}^n
        O(n log n). Top-HS support score 위치를 nonzero 로, 각각 sign argmax.
        Baseline.

    enumerate_top_candidates(posterior, hs, k) → [(log_prob, ŝ), …] (k 개)
        Top-K support pattern (1-swap neighborhood) 후보 enumeration. 각
        후보의 log-prob 정렬. public key 검증 등 후속 사용.

    binary_to_ternary_posterior(score_pos, score_neg) → posterior (n, 3)
        Oracle pair (α_pos, α_neg) 의 binary detector 출력을 Bayes 로 ternary
        posterior 로 변환. (1, 1) 응답 = 모순 → P(s=*) = 0.

수학:

    MAP problem:
        ŝ = argmax_s  Σ_i log P(s_i | data)
        s.t.  s_i ∈ {-1, 0, +1},  |{i : s_i ≠ 0}| = hs

    Greedy sub-optimal but fast: support score = log P(s≠0|data) − log P(s=0|data)
    내림차순 위치 선택, 각 위치 sign argmax.

    Optimal: integer programming. n=256, hs=70 라 C(256, 70) ≈ 2^140 — 직접
    enumeration 불가능. 1-swap neighborhood 가 실용적.
"""

from __future__ import annotations

import numpy as np


def greedy_sparse_recover(
    posterior: np.ndarray,
    hs: int,
) -> np.ndarray:
    """Greedy MAP under HW=hs constraint. O(n log n).

    Top-hs support-score 위치 선정 → 각 위치에서 sign argmax.

    Parameters
    ----------
    posterior : (n, 3) — column 0=P(-1), 1=P(0), 2=P(+1). Row sum 임의 (정규화 X).
    hs : 강제 HW (= number of nonzero positions).

    Returns
    -------
    ŝ : (n,) int8, 값 ∈ {-1, 0, +1}, np.count_nonzero(ŝ) == hs.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    if posterior.ndim != 2 or posterior.shape[1] != 3:
        raise ValueError(f"posterior shape {posterior.shape} != (n, 3)")
    n = posterior.shape[0]
    if not (0 < hs <= n):
        raise ValueError(f"hs={hs} ∉ (0, {n}]")

    # support score = P(nonzero) (정확히 P(-1) + P(+1)).
    # ties: argpartition 가 임의 결정 — 결과 결정성 위해 stable sort.
    support_score = posterior[:, 0] + posterior[:, 2]
    # 큰 순 정렬, top hs 위치 선정.
    order = np.argsort(-support_score, kind="stable")
    selected = order[:hs]

    out = np.zeros(n, dtype=np.int8)
    for i in selected:
        if posterior[i, 0] >= posterior[i, 2]:
            out[i] = -1
        else:
            out[i] = +1
    # invariant
    assert int(np.count_nonzero(out)) == hs
    return out


def _logp(posterior: np.ndarray, s: np.ndarray, eps: float = 1e-12) -> float:
    """Σ_i log P(s_i | data). s ∈ {-1, 0, +1}^n, posterior (n, 3)."""
    cols = np.where(s == -1, 0, np.where(s == 0, 1, 2))
    p = posterior[np.arange(s.size), cols]
    return float(np.sum(np.log(np.maximum(p, eps))))


def enumerate_top_candidates(
    posterior: np.ndarray,
    hs: int,
    k: int = 32,
) -> list[tuple[float, np.ndarray]]:
    """Top-k 후보 sk under HW=hs (1-swap neighborhood of greedy MAP).

    Greedy MAP 한 후, support 의 한 위치를 *bumped-out* 한 자리와 swap 한
    1-neighborhood 후보들 logp 계산 후 정렬해 상위 k 개 반환.

    Parameters
    ----------
    posterior : (n, 3)
    hs : sparse constraint
    k : 반환할 후보 수 (기본 32)

    Returns
    -------
    [(log_prob, ŝ), …] sorted by log_prob desc, 길이 ≤ k.

    Notes
    -----
    1-swap 만으로는 조합 탐색이 좁음. 더 큰 k 가 필요하면 multi-swap 으로
    확장 — 단, 시간 복잡도 O(k · n) → O(k · n^2). 본 구현은 baseline.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    n = posterior.shape[0]
    if k < 1:
        raise ValueError(f"k={k} < 1")

    # 1) Greedy MAP baseline.
    s_base = greedy_sparse_recover(posterior, hs)
    logp_base = _logp(posterior, s_base)
    candidates: list[tuple[float, np.ndarray]] = [(logp_base, s_base.copy())]

    # 2) 1-swap neighborhood: support 한 자리 i_in (현재 nonzero) 를 빼고,
    #    out-of-support 자리 j_out (현재 zero) 를 넣은 후보.
    in_support = np.where(s_base != 0)[0]
    out_support = np.where(s_base == 0)[0]

    # 효율: support score 정렬해 좋은 swap 후보 우선 (greedy 와 동일 metric).
    support_score = posterior[:, 0] + posterior[:, 2]
    out_sorted = out_support[np.argsort(-support_score[out_support], kind="stable")]
    in_sorted = in_support[np.argsort(support_score[in_support], kind="stable")]

    # 모든 1-swap 후보 logp 계산. (n_in × n_out 가능, 너무 많으면 잘라서.)
    seen_logps = {logp_base}
    for j_out in out_sorted[: max(k * 2, 32)]:
        for i_in in in_sorted[: max(k * 2, 32)]:
            s_new = s_base.copy()
            s_new[i_in] = 0
            # j_out 위치 sign 결정
            s_new[j_out] = -1 if posterior[j_out, 0] >= posterior[j_out, 2] else +1
            assert int(np.count_nonzero(s_new)) == hs
            lp = _logp(posterior, s_new)
            # 중복 회피 (greedy 가 정확히 swap_score 0 인 자리 만들 수 있음)
            key = round(lp, 8)
            if key in seen_logps:
                continue
            seen_logps.add(key)
            candidates.append((lp, s_new))
            if len(candidates) >= k * 4:  # 일정량 모은 후 정렬
                break
        if len(candidates) >= k * 4:
            break

    candidates.sort(key=lambda x: -x[0])
    return candidates[:k]


def binary_to_ternary_posterior(
    score_pos: np.ndarray,
    score_neg: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Oracle pair (α_pos, α_neg) 의 binary detector 출력 → ternary posterior.

    Detector ground truth (E3a partition table 기반):
        s = -1: α_pos → µ′ = 0, α_neg → µ′ = 1
        s =  0: α_pos → µ′ = 0, α_neg → µ′ = 0
        s = +1: α_pos → µ′ = 1, α_neg → µ′ = 0

    score_pos ∈ [0, 1] = P(µ′ = 1 | α_pos detector)
    score_neg ∈ [0, 1] = P(µ′ = 1 | α_neg detector)

    Likelihood (independent under noise):
        P(scores | s = -1) ∝ (1 - score_pos) · score_neg
        P(scores | s =  0) ∝ (1 - score_pos) · (1 - score_neg)
        P(scores | s = +1) ∝ score_pos · (1 - score_neg)

    Posterior = likelihood · uniform_prior, normalized per position.

    Parameters
    ----------
    score_pos, score_neg : (n,) float, 값 ∈ [0, 1].

    Returns
    -------
    posterior : (n, 3) — column 0=P(-1), 1=P(0), 2=P(+1).
    """
    sp = np.clip(np.asarray(score_pos, dtype=np.float64), eps, 1.0 - eps)
    sn = np.clip(np.asarray(score_neg, dtype=np.float64), eps, 1.0 - eps)
    if sp.shape != sn.shape:
        raise ValueError(f"score shape mismatch: {sp.shape} vs {sn.shape}")
    n = sp.size

    posterior = np.zeros((n, 3), dtype=np.float64)
    posterior[:, 0] = (1.0 - sp) * sn          # s = -1
    posterior[:, 1] = (1.0 - sp) * (1.0 - sn)  # s =  0
    posterior[:, 2] = sp * (1.0 - sn)          # s = +1
    # (sp, sn) 둘 다 1 에 가까우면 모든 likelihood 거의 0 — 모순 응답.
    # uniform fallback 으로 P=1/3 each.
    z = posterior.sum(axis=1, keepdims=True)
    too_small = z < 1e-6
    posterior = np.where(too_small, 1.0 / 3.0, posterior / np.where(z > 0, z, 1.0))
    return posterior


def accuracy(s_pred: np.ndarray, s_true: np.ndarray) -> dict[str, float]:
    """ŝ vs s_true 의 정확도 metrics.

    Returns dict:
        bit_accuracy : 256-coef 정확 비율
        support_accuracy : (s ≠ 0) 매칭 비율 (HW=HS 위치 검출)
        sign_accuracy : support 안에서 sign 정확도
    """
    s_pred = np.asarray(s_pred, dtype=np.int8)
    s_true = np.asarray(s_true, dtype=np.int8)
    if s_pred.shape != s_true.shape:
        raise ValueError(f"shape mismatch: {s_pred.shape} vs {s_true.shape}")

    n = s_pred.size
    bit_acc = float(np.mean(s_pred == s_true))

    support_pred = s_pred != 0
    support_true = s_true != 0
    support_acc = float(np.mean(support_pred == support_true))

    common_support = support_pred & support_true
    if common_support.any():
        sign_acc = float(np.mean(s_pred[common_support] == s_true[common_support]))
    else:
        sign_acc = float("nan")

    return {
        "bit_accuracy": bit_acc,
        "support_accuracy": support_acc,
        "sign_accuracy": sign_acc,
        "n": float(n),
    }
