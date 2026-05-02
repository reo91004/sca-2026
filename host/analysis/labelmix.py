"""여러 .npz 를 외부 라벨로 한 컨테이너에 합치고 그룹별로 split.

E3b 는 두 클래스 (μ=zero, μ=one) 를 각자 .npz 로 저장 — 단순 비교는
이미 plot_tvla.py 가 받는다. labelmix 는 *세 개 이상의 sweep* 또는
*외부 라벨 (예측 µ′ 계급, oracle 응답 등)* 로 클래스를 합칠 때 쓴다.

핵심 함수:
    LabelMix.from_npz_files(items)   여러 .npz + 라벨 → 한 객체
    .traces(label)                   특정 라벨의 트레이스 행렬
    .group_pair(label_a, label_b)    plot_tvla 가 받을 (a, b) 튜플
    .summary()                       라벨별 N, samples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .io import Capture, load_capture


@dataclass
class LabelMix:
    """여러 캡처를 외부 라벨로 묶은 컨테이너.

    내부 자료구조:
        _by_label[label] = (concatenated_traces, concatenated_responses, [meta...])
    """

    _by_label: dict[str, tuple[np.ndarray, np.ndarray, list[dict]]] = field(
        default_factory=dict
    )
    samples: int | None = None  # 첫 추가 시 결정, 이후 검증

    @classmethod
    def from_npz_files(
        cls,
        items: Iterable[tuple[str | Path, str]],
    ) -> "LabelMix":
        """items = [(npz_path, label), ...]. 같은 라벨이 여러 번 나오면 concat."""
        mix = cls()
        for path, label in items:
            cap = load_capture(path)
            mix.add_capture(cap, label)
        return mix

    def add_capture(self, cap: Capture, label: str) -> None:
        if self.samples is None:
            self.samples = cap.samples
        elif cap.samples != self.samples:
            raise ValueError(
                f"sample-length 불일치: {self.samples} (기존) vs {cap.samples} ({cap.path})"
            )
        if label in self._by_label:
            t_old, r_old, m_old = self._by_label[label]
            t_new = np.concatenate([t_old, cap.traces], axis=0)
            r_new = np.concatenate([r_old, cap.responses], axis=0)
            m_new = m_old + [cap.meta]
        else:
            t_new, r_new, m_new = cap.traces, cap.responses, [cap.meta]
        self._by_label[label] = (t_new, r_new, m_new)

    def labels(self) -> list[str]:
        return sorted(self._by_label.keys())

    def n(self, label: str) -> int:
        return int(self._by_label[label][0].shape[0])

    def traces(self, label: str) -> np.ndarray:
        return self._by_label[label][0]

    def responses(self, label: str) -> np.ndarray:
        return self._by_label[label][1]

    def metas(self, label: str) -> list[dict]:
        return list(self._by_label[label][2])

    def group_pair(self, label_a: str, label_b: str) -> tuple[np.ndarray, np.ndarray]:
        if label_a not in self._by_label:
            raise KeyError(f"unknown label {label_a!r} (have {self.labels()})")
        if label_b not in self._by_label:
            raise KeyError(f"unknown label {label_b!r} (have {self.labels()})")
        return self.traces(label_a), self.traces(label_b)

    def summary(self) -> str:
        lines = [
            f"LabelMix samples={self.samples}, labels={len(self._by_label)}:"
        ]
        for label in self.labels():
            t, r, _ = self._by_label[label]
            lines.append(f"  {label!r}: N={t.shape[0]} resp_shape={r.shape}")
        return "\n".join(lines)


def relabel_by_response_byte(
    cap: Capture,
    *,
    byte_index: int = 0,
) -> dict[int, np.ndarray]:
    """응답의 byte_index 값별로 트레이스 인덱스를 쪼개 dict 반환.

    예: oracle pair 실험에서 'D' 응답 (mismatch flag) 가 0/1 두 가지 →
    두 클래스. 또는 host 가 외부에서 응답 byte 를 oracle 결과로 집어넣은 경우.
    """
    if cap.responses.ndim != 2:
        raise ValueError("Capture.responses must be 2D")
    col = cap.responses[:, byte_index]
    out: dict[int, np.ndarray] = {}
    for v in np.unique(col):
        out[int(v)] = np.flatnonzero(col == v)
    return out
