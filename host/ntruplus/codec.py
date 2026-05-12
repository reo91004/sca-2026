"""poly_tobytes / poly_frombytes — 12-bit packed coefficient encoding.

Mirrors upstream `poly.c` `poly_tobytes` / `poly_frombytes`. Two coefficients
share three bytes:

    r[3i+0] = (t0      ) & 0xff
    r[3i+1] = (t0 >> 8 ) | (t1 << 4)
    r[3i+2] = (t1 >> 4 ) & 0xff

with t = (a + (a >> 15) & q) — i.e. negative coefficients are mapped to
[0, q) before packing.

frombytes reads twelve unsigned bits; centred values are in [0, 0xfff], not
restricted to [0, q). A valid encoder always emits values in [0, q) so the
12-bit output equals the centred residue mod q.
"""

from __future__ import annotations

import numpy as np

from .params import N, POLYBYTES, Q


def to_bytes(a: np.ndarray) -> bytes:
    """Pack a length-N int16 polynomial into POLYBYTES bytes (12-bit per coef)."""
    if a.shape != (N,):
        raise ValueError(f"expected shape ({N},), got {a.shape}")
    a = a.astype(np.int32)             # avoid int16 wrap on +Q
    t = a + ((a >> 15) & Q)            # mask negatives → [0, q)
    if (t < 0).any() or (t >= 1 << 12).any():
        raise ValueError("coefficient out of 12-bit range after centering")

    out = np.zeros(POLYBYTES, dtype=np.uint8)
    pairs = N // 2
    t0 = t[0::2]
    t1 = t[1::2]
    out[0::3] = (t0 & 0xff).astype(np.uint8)
    out[1::3] = ((t0 >> 8) | ((t1 << 4) & 0xff)).astype(np.uint8)
    out[2::3] = ((t1 >> 4) & 0xff).astype(np.uint8)
    return bytes(out)


def from_bytes(buf: bytes | bytearray | np.ndarray) -> np.ndarray:
    """Unpack POLYBYTES bytes into N coefficients in [0, 0xfff]."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.shape != (POLYBYTES,):
        raise ValueError(f"expected {POLYBYTES} bytes, got {arr.shape}")
    out = np.empty(N, dtype=np.int16)
    b0 = arr[0::3].astype(np.int32)
    b1 = arr[1::3].astype(np.int32)
    b2 = arr[2::3].astype(np.int32)
    out[0::2] = ((b0 >> 0) | (b1 << 8)) & 0xfff
    out[1::2] = ((b1 >> 4) | (b2 << 4)) & 0xfff
    return out


def center(x: np.ndarray) -> np.ndarray:
    """Map [0, q) → centred residues in (-q/2, q/2]. Operates element-wise."""
    y = (x.astype(np.int32)) % Q
    y = np.where(y > Q // 2, y - Q, y)
    return y.astype(np.int16)
