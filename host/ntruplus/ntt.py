"""Reference NTRU+768 NTT / inverse-NTT / basemul, ported from upstream `ntt.c`.

Goal here is *bit-exact* parity with the firmware's int16 arithmetic so we can
predict labels (HW(γ·f̂_k mod q), reduction-bit, etc.) for SCA work — not raw
performance. Everything is plain Python/numpy, scalar where the C is scalar.

Conventions:
    poly is a numpy int16 array of shape (N,), values in centred residue
    representation (-q/2, q/2]. After `ntt()` it stores Montgomery-form NTT
    coefficients identical to what the board's `ntt()` produces.
"""

from __future__ import annotations

import numpy as np

from .params import (
    N,
    NINV,
    OMEGA,
    Q,
    QINV,
    R,
    RINV,
    RSQ,
    TWO_NINV,
    ZETAS,
    ZMINUSZ5INV,
)


def _i16(x: int) -> int:
    """Python int → int16 with two's complement wrap."""
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def montgomery_reduce(a: int) -> int:
    """`a * R^{-1} mod q`, output centred in {-q+1, …, q-1}.

    Upstream:
        t = (int16_t)a * QINV;
        t = (a - (int32_t)t * Q) >> 16;
    """
    a = _i32(int(a))
    t = _i16(_i16(a) * QINV)
    t = _i32(a - t * Q)
    return t >> 16


def barrett_reduce(a: int) -> int:
    """Centred residue mod q in {-(q+1)/2, …, (q+1)/2}."""
    a = _i16(int(a))
    v = ((1 << 26) + Q // 2) // Q
    t = _i32((v * a + (1 << 25)) >> 26)
    t *= Q
    return _i16(a - t)


def fqmul(a: int, b: int) -> int:
    return montgomery_reduce(int(a) * int(b))


# ---- forward NTT (in-place on a copy) ------------------------------------

def ntt(r_in: np.ndarray) -> np.ndarray:
    r = np.array(r_in, dtype=np.int16, copy=True)
    k = 1

    zeta1 = int(ZETAS[k]); k += 1
    half = N // 2
    for i in range(half):
        t1 = fqmul(zeta1, int(r[i + half]))
        r[i + half] = _i16(int(r[i]) + int(r[i + half]) - t1)
        r[i]        = _i16(int(r[i]) + t1)

    for start in range(0, N, 384):
        zeta1 = int(ZETAS[k]); k += 1
        zeta2 = int(ZETAS[k]); k += 1
        for i in range(start, start + 128):
            t1 = fqmul(zeta1, int(r[i + 128]))
            t2 = fqmul(zeta2, int(r[i + 256]))
            t3 = fqmul(OMEGA, t1 - t2)
            r[i + 256] = _i16(int(r[i]) - t1 - t3)
            r[i + 128] = _i16(int(r[i]) - t2 + t3)
            r[i      ] = _i16(int(r[i]) + t1 + t2)

    step = 64
    while step >= 4:
        for start in range(0, N, step << 1):
            zeta1 = int(ZETAS[k]); k += 1
            for i in range(start, start + step):
                t1 = fqmul(zeta1, int(r[i + step]))
                r[i + step] = _i16(int(r[i]) - t1)
                r[i]        = _i16(int(r[i]) + t1)
        step >>= 1

    for i in range(N):
        r[i] = barrett_reduce(int(r[i]))
    return r


# ---- inverse NTT ---------------------------------------------------------

def invntt(r_in: np.ndarray) -> np.ndarray:
    r = np.array(r_in, dtype=np.int16, copy=True)
    k = 191

    step = 4
    while step <= 64:
        for start in range(0, N, step << 1):
            zeta1 = int(ZETAS[k]); k -= 1
            for i in range(start, start + step):
                t1 = int(r[i + step])
                r[i + step] = fqmul(zeta1, t1 - int(r[i]))
                r[i]        = barrett_reduce(int(r[i]) + t1)
        step <<= 1

    for start in range(0, N, 384):
        zeta2 = int(ZETAS[k]); k -= 1
        zeta1 = int(ZETAS[k]); k -= 1
        for i in range(start, start + 128):
            t1 = fqmul(OMEGA, int(r[i + 128]) - int(r[i]))
            t2 = fqmul(zeta1, int(r[i + 256]) - int(r[i]) + t1)
            t3 = fqmul(zeta2, int(r[i + 256]) - int(r[i + 128]) - t1)
            r[i      ] = _i16(int(r[i]) + int(r[i + 128]) + int(r[i + 256]))
            r[i + 128] = t2
            r[i + 256] = t3

    half = N // 2
    for i in range(half):
        t1 = _i16(int(r[i]) + int(r[i + half]))
        t2 = fqmul(ZMINUSZ5INV, int(r[i]) - int(r[i + half]))
        r[i       ] = fqmul(NINV,     t1 - t2)
        r[i + half] = fqmul(TWO_NINV, t2)
    return r


# ---- basemul (per 4-coefficient lane) ------------------------------------

def basemul_lane(a: np.ndarray, b: np.ndarray, zeta: int) -> np.ndarray:
    """Multiply two 4-coef NTT lanes inside Zq[X] / (X^4 - zeta).

    Direct port of upstream `basemul()`. Both inputs and output are int16
    centered residues. zeta is the lane-specific twiddle.
    """
    a = a.astype(np.int32)
    b = b.astype(np.int32)
    r = np.zeros(4, dtype=np.int32)

    r[0] = montgomery_reduce(a[1] * b[3] + a[2] * b[2] + a[3] * b[1])
    r[1] = montgomery_reduce(a[2] * b[3] + a[3] * b[2])
    r[2] = montgomery_reduce(a[3] * b[3])

    r[0] = montgomery_reduce(r[0] * zeta + a[0] * b[0])
    r[1] = montgomery_reduce(r[1] * zeta + a[0] * b[1] + a[1] * b[0])
    r[2] = montgomery_reduce(r[2] * zeta + a[0] * b[2] + a[1] * b[1] + a[2] * b[0])
    r[3] = montgomery_reduce(a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0])

    r[0] = montgomery_reduce(r[0] * RSQ)
    r[1] = montgomery_reduce(r[1] * RSQ)
    r[2] = montgomery_reduce(r[2] * RSQ)
    r[3] = montgomery_reduce(r[3] * RSQ)
    return r.astype(np.int16)


def basemul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full polynomial basemul in NTT domain (192 4-coef lanes, 96 zeta pairs)."""
    out = np.zeros(N, dtype=np.int16)
    for i in range(N // 8):
        zeta = int(ZETAS[96 + i])
        out[8 * i:8 * i + 4]     = basemul_lane(a[8 * i:8 * i + 4],
                                                 b[8 * i:8 * i + 4], +zeta)
        out[8 * i + 4:8 * i + 8] = basemul_lane(a[8 * i + 4:8 * i + 8],
                                                 b[8 * i + 4:8 * i + 8], -zeta)
    return out


def naive_poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply in R_q = Z_q[X] / (X^N - X^(N/2) + 1).  Reference for tests.

    Loops only over non-zero coefficients of a and b; the dense path is too
    slow for N=768 in pure Python and not needed for our sparse tests.

    Reduction: X^N = X^(N/2) - 1, applied recursively for k ≥ N. The closed
    form for k ∈ [N, 2N) is:
        k ∈ [N, 3N/2):    +X^(k - N/2)  −X^(k - N)
        k ∈ [3N/2, 2N):   −X^(k - 3N/2)
    """
    half = N // 2
    out = np.zeros(N, dtype=np.int64)
    nz_a = np.flatnonzero(a)
    nz_b = np.flatnonzero(b)
    for i in nz_a:
        ai = int(a[i])
        for j in nz_b:
            v = ai * int(b[j])
            k = int(i) + int(j)
            if k < N:
                out[k] += v
            elif k < N + half:
                out[k - half] += v
                out[k - N]    -= v
            else:                           # k ∈ [3N/2, 2N)
                out[k - 3 * half] -= v
    out %= Q
    out = np.where(out > Q // 2, out - Q, out)
    return out.astype(np.int16)
