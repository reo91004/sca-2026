"""SMAUG-T 프로토콜/스펙 시뮬레이터 (host 측, numpy 만 의존).

이 패키지는 보드와 통신하지 않는다 — chosen-CT 빌더가 바이트 단위로
정확히 무엇을 보드에 보낼지를 *오프라인* 으로 만들고 round-trip 검증한다.

서브모듈:
    params        — smaug{1,3,5} 파라미터 dataclass
    codec         — Compress/Decompress, polynomial pack/unpack
    ciphertext    — c1/c2 ↔ 672 B 직렬화
    chosen        — chosen-CT 빌더 (μ′=0/1, c1=αX^j 형태)
"""

from . import chosen, ciphertext, codec, params, sk_partition  # noqa: F401

__all__ = ["chosen", "ciphertext", "codec", "params", "sk_partition"]
