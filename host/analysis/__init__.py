"""sca-2026 트레이스 분석 패키지.

표준 진입점:
    from host.analysis import io, viz, tvla, group, validate

각 모듈은 numpy/matplotlib 만 의존하며, ChipWhisperer 가 설치되지 않은
오프라인 호스트에서도 그대로 동작한다 (캡처는 host.capture, 분석은 여기).
"""

from . import e3c, group, io, labelmix, sparse_recover, tvla, validate, viz  # noqa: F401

__all__ = [
    "e3c", "group", "io", "labelmix", "sparse_recover",
    "tvla", "validate", "viz",
]
