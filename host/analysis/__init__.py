"""sca-2026 트레이스 분석 패키지.

현재 사용처: **SMAUG-T 트랙** (`scripts/smaug/s*` 및 `tests/smaug/test_*`).
NTRU+ 트랙은 자체 분석 코드를 `scripts/ntruplus/n*` 안에 내장하고 있다.
향후 NTRU+ 가 공유 분석 도구를 쓰게 되면 트랙별 분리 검토.

표준 진입점:
    from host.analysis import io, viz, tvla, group, validate

각 모듈은 numpy/matplotlib 만 의존하며, ChipWhisperer 가 설치되지 않은
오프라인 호스트에서도 그대로 동작한다 (캡처는 host.capture, 분석은 여기).
"""

from . import group, io, sparse_recover, tvla, validate, viz  # noqa: F401

__all__ = ["group", "io", "sparse_recover", "tvla", "validate", "viz"]
