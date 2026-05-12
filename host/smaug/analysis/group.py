"""트레이스 클래스 분할 헬퍼.

TVLA 또는 chosen-CT classifier 검증을 위해, (N, T) 트레이스 배열을
응답값 또는 외부 라벨 기준으로 두 그룹 (또는 그 이상) 으로 가른다.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def split_by_response_byte(
    traces: np.ndarray,
    responses: np.ndarray,
    *,
    byte_index: int = 0,
    value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """response[:, byte_index] == value 인지에 따라 트레이스를 둘로 가른다.

    SMAUG 의 dec mismatch flag (resp[:,0] == 0 vs != 0) 같은 단순 분기에
    바로 쓴다. 길이 0 그룹이 나오면 그대로 빈 배열로 반환.
    """
    if responses.ndim != 2:
        raise ValueError("responses must be (N, R) — use io.load_capture")
    if not (0 <= byte_index < responses.shape[1]):
        raise IndexError(
            f"byte_index={byte_index} out of range R={responses.shape[1]}"
        )
    mask = responses[:, byte_index] == value
    return traces[mask], traces[~mask]


def split_by_labels(
    traces: np.ndarray,
    labels: np.ndarray,
    *,
    classes: Iterable[int] | None = None,
) -> dict[int, np.ndarray]:
    """외부 라벨 (chosen-CT 인덱스 등) 으로 클래스별 dict 반환."""
    if labels.shape[0] != traces.shape[0]:
        raise ValueError(
            f"labels N={labels.shape[0]} != traces N={traces.shape[0]}"
        )
    if classes is None:
        classes = sorted(int(c) for c in np.unique(labels))
    return {int(c): traces[labels == c] for c in classes}


def fixed_vs_random_split(
    traces: np.ndarray,
    is_fixed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """TVLA 표준 분할: is_fixed (bool, N) 에 따라 (fixed, random) 반환."""
    if is_fixed.dtype != bool:
        is_fixed = is_fixed.astype(bool)
    if is_fixed.shape[0] != traces.shape[0]:
        raise ValueError("is_fixed length mismatch")
    return traces[is_fixed], traces[~is_fixed]
