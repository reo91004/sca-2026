"""Welch t-test 기반 TVLA (Test Vector Leakage Assessment).

표준 절차 (Goodwill et al. 2011, ISO/IEC 17825 부록):
  - 두 클래스 A, B 의 트레이스 행렬을 받아 시간점별 t-statistic 을 계산.
  - 임계값 ±4.5 (양측 검정 p < 1e-5, df > 1000 한계) 를 초과하는 시간점이
    하나라도 있으면 *first-order leakage 존재* 로 판정.

본 모듈은 numpy 벡터화로 (N_a, T) + (N_b, T) → (T,) t 를 한 번에 계산.
샘플 수가 한쪽이라도 < 2 이면 ValueError. 분산이 0 인 시간점은 t=0 으로
강제 (분모 보호). 평균/표준편차/PoI 까지 한 객체에 묶어 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TvlaResult:
    """Welch t-test 결과 (시간점별)."""

    t: np.ndarray  # (T,) float64 — 시간점별 t-statistic
    mu_a: np.ndarray
    mu_b: np.ndarray
    var_a: np.ndarray
    var_b: np.ndarray
    n_a: int
    n_b: int
    threshold: float = 4.5

    @property
    def max_abs_t(self) -> float:
        return float(np.max(np.abs(self.t)))

    @property
    def leaky_idx(self) -> np.ndarray:
        """|t| > threshold 인 시간점 인덱스 (PoI 후보)."""
        return np.flatnonzero(np.abs(self.t) > self.threshold)

    def summary(self) -> str:
        leaky = self.leaky_idx
        return (
            f"Welch-t: N_A={self.n_a} N_B={self.n_b} "
            f"max|t|={self.max_abs_t:.2f} "
            f"leaky points (|t|>{self.threshold})={len(leaky)}"
        )


def welch_t(
    a: np.ndarray,
    b: np.ndarray,
    *,
    threshold: float = 4.5,
) -> TvlaResult:
    """t_i = (μ_A,i − μ_B,i) / sqrt(σ²_A,i / N_A + σ²_B,i / N_B).

    a, b : (N, T) 형 ndarray — 같은 샘플 길이 T 필요.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a, b must be 2D (N, T)")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"sample-length mismatch: a.T={a.shape[1]} != b.T={b.shape[1]}"
        )
    n_a, t_len = a.shape
    n_b = b.shape[0]
    if n_a < 2 or n_b < 2:
        raise ValueError(f"need N >= 2 in both classes (got {n_a}, {n_b})")

    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    mu_a = a64.mean(axis=0)
    mu_b = b64.mean(axis=0)
    # ddof=1: 표본분산 (Welch 전제는 unbiased estimator)
    var_a = a64.var(axis=0, ddof=1)
    var_b = b64.var(axis=0, ddof=1)

    denom = np.sqrt(var_a / n_a + var_b / n_b)
    # 분모가 0 인 시간점 (양쪽 클래스 모두 상수) → t=0 으로 정의.
    safe = denom > 0
    t = np.zeros(t_len, dtype=np.float64)
    t[safe] = (mu_a[safe] - mu_b[safe]) / denom[safe]

    return TvlaResult(
        t=t,
        mu_a=mu_a,
        mu_b=mu_b,
        var_a=var_a,
        var_b=var_b,
        n_a=int(n_a),
        n_b=int(n_b),
        threshold=float(threshold),
    )


def difference_of_means(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """단순 평균 차분 — 누출 위치 시각화용 (TVLA 보조)."""
    if a.shape[1] != b.shape[1]:
        raise ValueError("sample-length mismatch")
    return a.astype(np.float64).mean(axis=0) - b.astype(np.float64).mean(axis=0)
