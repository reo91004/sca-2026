"""E3c (oracle pair) 트레이스 → ternary 비밀 계수 추정.

run_e3c.py 가 저장한 .npz 의 형식:
    traces[i]                — (samples,) float32
    responses[i, 0]          — 'D' mismatch flag
    meta.label_alpha[i]      — chosen-CT 의 α (= alpha_pos 또는 alpha_neg)
    meta.label_k[i]          — c1 component (0 또는 1)
    meta.label_j[i]          — 단항 X^j 의 j

분석 절차:
  1) trace 를 (alpha, k, j) 로 그룹핑.
  2) 같은 (k, j) 위에서 alpha_pos / alpha_neg 두 그룹의 trace 평균.
  3) 각 그룹에서 PoI (외부 입력 또는 휴리스틱) 의 평균 amplitude →
     이진 결정 (µ′ = 0 or 1).
  4) (µ_pos, µ_neg) join 으로 ternary {-1, 0, +1} 결정.

PoI 가 외부에서 안 주어진 경우 (이 모듈의 default) 휴리스틱:
    각 (k, j) 의 alpha_pos vs alpha_neg 트레이스의 sample-wise 차분
    →  std 가 가장 큰 sample 을 PoI 로. 다 위치 공통 PoI 가 있으면
    그게 message-bit 처리부.

본 모듈의 핵심 함수:
    classify_e3c(npz_path, *, poi=None) → dict[(k, j)] = predicted_s
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import load_capture


@dataclass
class E3cResult:
    """oracle pair 분석 결과."""

    predictions: dict[tuple[int, int], int]  # (k, j) → s ∈ {-1, 0, +1}
    inconsistent: list[tuple[int, int]]      # (1, 1) 응답이 나온 위치 (모순)
    poi_idx: int                             # 분석에 쓴 PoI 시간점 (heuristic 또는 입력)
    alpha_pos: int
    alpha_neg: int

    @property
    def n_positions(self) -> int:
        return len(self.predictions)

    def histogram(self) -> dict[int, int]:
        h = {-1: 0, 0: 0, 1: 0}
        for v in self.predictions.values():
            h[v] += 1
        return h


def _group_by_label(meta: dict, traces: np.ndarray) -> dict[tuple[int, int, int], np.ndarray]:
    """(alpha, k, j) → trace 인덱스 배열."""
    alpha = np.asarray(meta["label_alpha"], dtype=np.int64)
    k = np.asarray(meta["label_k"], dtype=np.int64)
    j = np.asarray(meta["label_j"], dtype=np.int64)
    if not (alpha.shape[0] == k.shape[0] == j.shape[0] == traces.shape[0]):
        raise ValueError("label arrays length mismatch with traces")
    out: dict[tuple[int, int, int], list[int]] = {}
    for i in range(traces.shape[0]):
        key = (int(alpha[i]), int(k[i]), int(j[i]))
        out.setdefault(key, []).append(i)
    return {key: np.asarray(idx, dtype=np.int64) for key, idx in out.items()}


def _heuristic_poi(traces: np.ndarray, alpha_pos: int, alpha_neg: int,
                   meta: dict) -> int:
    """PoI 입력이 없으면: 두 alpha 그룹 전체 평균의 sample-wise 차분 |Δμ|
    가 최대인 시간점."""
    alpha = np.asarray(meta["label_alpha"], dtype=np.int64)
    pos_mean = traces[alpha == alpha_pos].mean(axis=0)
    neg_mean = traces[alpha == alpha_neg].mean(axis=0)
    delta = np.abs(pos_mean - neg_mean)
    return int(np.argmax(delta))


def classify_e3c(
    npz_path,
    *,
    poi: int | None = None,
    threshold: float | None = None,
) -> E3cResult:
    cap = load_capture(npz_path)
    meta = cap.meta
    if "label_alpha" not in meta:
        raise ValueError(
            f"{cap.path}: meta 에 label_alpha 없음 — run_e3c.py 산출물 아님"
        )
    alpha_pos = int(meta["alpha_pos"])
    alpha_neg = int(meta["alpha_neg"])

    if poi is None:
        poi = _heuristic_poi(cap.traces, alpha_pos, alpha_neg, meta)

    if threshold is None:
        # 모든 trace 의 PoI sample 의 median 을 임계값으로
        threshold = float(np.median(cap.traces[:, poi]))

    groups = _group_by_label(meta, cap.traces)
    # (k, j) → (alpha_pos 평균 PoI, alpha_neg 평균 PoI)
    by_kj: dict[tuple[int, int], dict[int, float]] = {}
    for (alpha, k, j), idx in groups.items():
        kj = (k, j)
        by_kj.setdefault(kj, {})[alpha] = float(cap.traces[idx, poi].mean())

    predictions: dict[tuple[int, int], int] = {}
    inconsistent: list[tuple[int, int]] = []
    for kj, vals in by_kj.items():
        if alpha_pos not in vals or alpha_neg not in vals:
            continue
        bit_pos = int(vals[alpha_pos] > threshold)
        bit_neg = int(vals[alpha_neg] > threshold)
        if (bit_pos, bit_neg) == (0, 0):
            predictions[kj] = 0
        elif (bit_pos, bit_neg) == (1, 0):
            predictions[kj] = 1
        elif (bit_pos, bit_neg) == (0, 1):
            predictions[kj] = -1
        else:
            # (1, 1) 모순 — sign-separating partition 의 join 에서 발생 안 해야
            inconsistent.append(kj)
            predictions[kj] = 0  # placeholder, 실패 표시

    return E3cResult(
        predictions=predictions,
        inconsistent=inconsistent,
        poi_idx=int(poi),
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
    )
