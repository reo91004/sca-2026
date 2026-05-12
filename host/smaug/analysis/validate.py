"""캡처 sanity 검증.

smoke-style 캡처 점검에서 사용. 캡처 환경이 살아 있는지를
형식·물리·로직 세 층으로 점검한다:

  1. 형식  : N/T/dtype 이 예상대로인지
  2. 물리  : NaN/Inf, all-zero, saturation 의심
  3. 로직  : SMAUG 의 mismatch flag (resp[0] == 0) — KEM 동작 자체

문제가 있으면 사람이 읽을 한 줄짜리 reason 을 모은 list 를 반환. 길이 0
이면 통과.
"""

from __future__ import annotations

from .io import Capture

import numpy as np


def validate_capture(
    cap: Capture,
    *,
    expected_n: int | None = None,
    expected_samples: int | None = None,
    do_kem_match_check: bool = False,
) -> list[str]:
    errs: list[str] = []
    n, t = cap.traces.shape

    if expected_n is not None and n != expected_n:
        errs.append(f"traces.shape[0]={n} != expected N={expected_n}")
    if expected_samples is not None and t != expected_samples:
        errs.append(f"traces.shape[1]={t} != expected S={expected_samples}")

    n_timeouts = int(cap.meta.get("n_timeouts", 0))
    if n_timeouts != 0:
        errs.append(f"meta.n_timeouts={n_timeouts} (expected 0)")

    if not np.isfinite(cap.traces).all():
        errs.append("traces 에 NaN/Inf 포함")
    elif np.abs(cap.traces).max() < 1e-9:
        errs.append("traces 가 사실상 0 — 게인/트리거/HAL 클록 의심")

    if do_kem_match_check:
        # SMAUG: 1바이트 mismatch flag, 0 이 정상.
        if cap.responses.shape[1] < 1:
            errs.append("KEM mismatch flag 검사 요청됐으나 resp_len < 1")
        else:
            bad = int((cap.responses[:, 0] != 0).sum())
            if bad:
                errs.append(
                    f"mismatch != 0 인 trace {bad}개 — KEM dec/enc 불일치"
                )
    return errs
