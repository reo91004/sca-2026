"""SMAUG-T ciphertext 직렬화: (c1, c2) ↔ CIPHERTEXT_BYTES.

c1 ∈ R_p^k         (k = MODULE_RANK)   → 256 byte × k
c2 ∈ R_p′                              → 160 byte
이어 붙여 CIPHERTEXT_BYTES = ctpolyvec_bytes + ctpoly2_bytes 가 된다.

SMAUG-T pack.c::Rp_vec_to_bytes / Rp2_to_bytes 와 같은 layout 을 따르도록
구현 — 각 polynomial 을 그 자체로 packed 한 뒤 단순 concat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import pack_rp, pack_rp2, unpack_rp, unpack_rp2
from .params import SmaugParams


@dataclass(frozen=True)
class Ciphertext:
    """SMAUG-T ciphertext = (c1 polyvec, c2 poly).

    c1.shape = (module_rank, n)  ints in [0, p)
    c2.shape = (n,)              ints in [0, p′)
    """

    c1: np.ndarray
    c2: np.ndarray
    params: SmaugParams

    def to_bytes(self) -> bytes:
        return ct_to_bytes(self.params, self.c1, self.c2)

    @classmethod
    def from_bytes(cls, buf: bytes, params: SmaugParams) -> "Ciphertext":
        c1, c2 = ct_from_bytes(params, buf)
        return cls(c1=c1, c2=c2, params=params)


def ct_to_bytes(p: SmaugParams, c1: np.ndarray, c2: np.ndarray) -> bytes:
    """(c1, c2) → CIPHERTEXT_BYTES.

    c1.shape == (module_rank, n) — 행 단위로 pack_rp 적용 후 concat.
    c2.shape == (n,)            — pack_rp2.
    """
    if c1.shape != (p.module_rank, p.n):
        raise ValueError(
            f"c1.shape={c1.shape} != ({p.module_rank}, {p.n})"
        )
    if c2.shape != (p.n,):
        raise ValueError(f"c2.shape={c2.shape} != ({p.n},)")
    if p.log_p != 8:
        raise NotImplementedError(
            f"ct_to_bytes 현재 LOG_P=8 (smaug1) 만 지원 — log_p={p.log_p}"
        )
    if p.log_p2 != 5:
        raise NotImplementedError(
            f"ct_to_bytes 현재 LOG_P2=5 (smaug1) 만 지원 — log_p2={p.log_p2}"
        )

    parts: list[bytes] = []
    for k in range(p.module_rank):
        parts.append(pack_rp(c1[k]))
    parts.append(pack_rp2(c2))
    out = b"".join(parts)
    if len(out) != p.ciphertext_bytes:
        raise AssertionError(
            f"packed length {len(out)} != {p.ciphertext_bytes} — 파라미터 불일치"
        )
    return out


def ct_from_bytes(p: SmaugParams, buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(buf) != p.ciphertext_bytes:
        raise ValueError(f"len={len(buf)} != {p.ciphertext_bytes}")
    if p.log_p != 8 or p.log_p2 != 5:
        raise NotImplementedError(
            "ct_from_bytes 현재 smaug1 (LOG_P=8, LOG_P2=5) 만 지원"
        )

    off = 0
    c1 = np.empty((p.module_rank, p.n), dtype=np.int64)
    for k in range(p.module_rank):
        c1[k] = unpack_rp(buf[off:off + p.ctpoly1_bytes])
        off += p.ctpoly1_bytes
    c2 = unpack_rp2(buf[off:off + p.ctpoly2_bytes])
    off += p.ctpoly2_bytes
    assert off == p.ciphertext_bytes
    return c1, c2
