"""캡처 트레이스 .npz 표준 로더.

`host/capture.py` 가 저장하는 .npz 는 다음 키를 갖는다:

    traces      (N, T) float32   ADC 샘플
    responses   (N, R) uint8     SimpleSerial ack 페이로드 (R=resp_len)
    meta        ()    object     dict — captured_at, scope_sn, samples,
                                   gain_db, target, cmd, send_len, resp_len,
                                   ss_ver, baud, n_attempted, n_timeouts

이 모듈의 `load_capture` 는 위 표준을 강제하고, 누락/형식 위반을
즉시 에러로 알린다 — 분석 스크립트가 .npz 한두 줄만 보고도 그대로
믿을 수 있게 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Capture:
    """단일 캡처 .npz 의 정상화된 view."""

    traces: np.ndarray  # (N, T) float32
    responses: np.ndarray  # (N, R) uint8
    meta: dict[str, Any]
    path: Path

    @property
    def n(self) -> int:
        return int(self.traces.shape[0])

    @property
    def samples(self) -> int:
        return int(self.traces.shape[1])

    def summary(self) -> str:
        return (
            f"{self.path.name}: N={self.n} S={self.samples} "
            f"target={self.meta.get('target','?')} cmd={self.meta.get('cmd','?')} "
            f"gain={self.meta.get('gain_db','?')}dB "
            f"timeouts={self.meta.get('n_timeouts',0)}"
        )


def load_capture(path: str | Path) -> Capture:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"capture not found: {p}")
    npz = np.load(p, allow_pickle=True)

    missing = [k for k in ("traces", "responses", "meta") if k not in npz.files]
    if missing:
        raise ValueError(f"{p}: 필수 키 누락 {missing} (capture.py 산출물 아님)")

    traces = npz["traces"]
    responses = npz["responses"]
    meta_obj = npz["meta"].item() if npz["meta"].dtype == object else npz["meta"]
    if not isinstance(meta_obj, dict):
        raise ValueError(f"{p}: meta 가 dict 아님 ({type(meta_obj).__name__})")

    if traces.ndim != 2:
        raise ValueError(f"{p}: traces.ndim={traces.ndim} (expected 2)")
    if traces.dtype != np.float32:
        raise ValueError(f"{p}: traces.dtype={traces.dtype} (expected float32)")
    if responses.ndim not in (1, 2):
        raise ValueError(f"{p}: responses.ndim={responses.ndim} (expected 1 or 2)")
    if responses.dtype != np.uint8:
        raise ValueError(f"{p}: responses.dtype={responses.dtype} (expected uint8)")
    if responses.shape[0] != traces.shape[0]:
        raise ValueError(
            f"{p}: responses.shape[0]={responses.shape[0]} != "
            f"traces.shape[0]={traces.shape[0]}"
        )

    # responses 는 (N,) 또는 (N, R) — 항상 2D 로 정규화 (R=1 도 명시).
    if responses.ndim == 1:
        responses = responses.reshape(-1, 1)

    return Capture(traces=traces, responses=responses, meta=dict(meta_obj), path=p)


def save_capture(
    path: str | Path,
    traces: np.ndarray,
    responses: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    """capture.py 와 호환되는 형식으로 .npz 저장.

    분석 단계에서 합성/필터링한 트레이스를 다시 저장할 때 쓴다.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if traces.dtype != np.float32:
        traces = traces.astype(np.float32, copy=False)
    if responses.dtype != np.uint8:
        responses = responses.astype(np.uint8, copy=False)
    np.savez_compressed(
        p,
        traces=traces,
        responses=responses,
        meta=np.array(meta, dtype=object),
    )
    return p
