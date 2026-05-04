"""SMAUG-T Compress/Decompress 와 polynomial pack/unpack.

수학적 정의 (Kyber-style, SMAUG-T spec 과 동치):

    Compress(x, d, q)   = ⌊ 2^d · x / q ⌉  mod 2^d
    Decompress(x, d, q) = ⌊ q · x / 2^d ⌉

여기서 ⌊·⌉ 은 round-to-nearest (ties: away from zero — Python 의 int(x + 0.5) 와
일치하지 않으므로 이 모듈은 numpy 정수산술로 명시 구현). 본 모듈은:

  - 위 정의를 곧이곧대로 옮긴 reference 함수 (`compress_ref`, `decompress_ref`),
  - SMAUG-T C 코드의 비트 트릭을 옮긴 fast 함수 (`compress_bit`, `decompress_bit`)
  - 두 함수가 [0, q) 전 도메인에서 일치한다는 사실을 tests/ 에서 검증.

Polynomial pack/unpack 은 R_p (8 bit/coef → 1 byte/coef) 와 R_p′ (5 bit/coef
→ 5 byte / 8 coefs) 두 가지를 다룬다. 이게 SMAUG-T smaug1 의 pack.c 가
하는 일과 정확히 같다 (objdump 로 검증).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Compress / Decompress
# ---------------------------------------------------------------------------

def compress_ref(x: np.ndarray, d: int, log_q: int) -> np.ndarray:
    """Reference Compress: ⌊2^d · x / q⌉ mod 2^d (integer-only).

    x ∈ [0, q) 가정 — 음수/오버플로 입력은 호출 측이 mod q 로 보낸다.
    round-to-nearest, ties-to-positive-infinity (Kyber/SMAUG-T 코드와 일치).
    """
    if d <= 0 or d >= log_q:
        raise ValueError(f"need 0 < d < log_q (got d={d}, log_q={log_q})")
    q = 1 << log_q
    p = 1 << d
    x64 = np.asarray(x, dtype=np.int64) % q
    # ⌊(2*p*x + q) / (2*q)⌉  ↔  Kyber 정의 (q 더해서 round)
    return ((2 * p * x64 + q) // (2 * q)) % p


def compress_bit(x: np.ndarray, d: int, log_q: int) -> np.ndarray:
    """SMAUG-T parameters.h 의 RD_ADD/RD_AND 비트 트릭 등가물.

    수식: ((x << (16 - log_q)) + 2^(15 - d)) >> (16 - d)  &  (2^d − 1)
    """
    if d <= 0 or d >= log_q:
        raise ValueError(f"need 0 < d < log_q (got d={d}, log_q={log_q})")
    if not (0 < (16 - log_q) and (16 - d) <= 16):
        raise ValueError("requires 0 < 16-log_q and 16-d <= 16")
    q = 1 << log_q
    rd_add = 1 << (15 - d)
    x_u16 = (np.asarray(x, dtype=np.int64) % q).astype(np.uint32)
    shifted = (x_u16 << (16 - log_q)) & 0xFFFF  # 16-bit wrap
    rounded = (shifted + rd_add) & 0xFFFF
    out = (rounded >> (16 - d)) & ((1 << d) - 1)
    return out.astype(np.int64)


def decompress(x: np.ndarray, d: int, log_q: int) -> np.ndarray:
    """⌊q · x / 2^d⌉ — Kyber 정의대로. 입력은 [0, 2^d).

    log_q ≥ d 이므로 q/2^d 는 정수 (여기 SMAUG-T 모든 레벨 해당). 따라서
    round-to-nearest 가 정수 곱으로 단순해진다.
    """
    if d <= 0 or d > log_q:
        raise ValueError(f"need 0 < d <= log_q (got d={d}, log_q={log_q})")
    p = 1 << d
    q = 1 << log_q
    x64 = np.asarray(x, dtype=np.int64) % p
    return (q * x64) // p  # (q/p)·x; q%p == 0 이라 정확히 나눠 떨어짐


def fixed_point_set(d: int, log_q: int) -> np.ndarray:
    """U_d := { x ∈ Z_q : Decompress(Compress(x, d), d) == x }.

    SMAUG-T smaug1 의 c1 (d=8, log_q=10) → U = {0, 4, 8, ..., q-4}.
                 c2 (d=5, log_q=10) → U = {0, 32, 64, ..., q-32}.
    """
    q = 1 << log_q
    cands = np.arange(q, dtype=np.int64)
    c = compress_ref(cands, d, log_q)
    return cands[decompress(c, d, log_q) == cands]


# ---------------------------------------------------------------------------
# Polynomial pack / unpack
# ---------------------------------------------------------------------------

def pack_bitstream(coeffs: np.ndarray, bits_per_coef: int) -> bytes:
    """Generic little-endian bitstream packer — SMAUG-T 의 모든 pack_R2_d 와 등가.

    upstream src/packring.c 의 LOG_P ∈ {8, 9} 와 LOG_P_PRIME ∈ {3, 4, 5, 7}
    모두 *연속된 little-endian bit stream* 으로 환원된다 (직접 검증 완료):

        bit position bp = i * bits_per_coef + b
        out[bp >> 3] bit (bp & 7) = (coeffs[i] >> b) & 1

    예: bits_per_coef=8 → trivially 1 byte per coef (pack_R2_8).
        bits_per_coef=9 → 8 coef = 9 byte (pack_R2_9, 64+8 비트 split 도
                          연속 bitstream 과 일치, src 라인-by-라인 검증).
        bits_per_coef=5 → 8 coef = 5 byte (pack_R2_5).

    n_bytes = ceil(n * bits_per_coef / 8). 이게 ctpoly{1,2}_bytes 와 정확히
    일치 — SmaugParams 의 derived property 와 cross-check.
    """
    if bits_per_coef <= 0 or bits_per_coef > 16:
        raise ValueError(f"bits_per_coef={bits_per_coef} out of (0, 16]")
    a = np.asarray(coeffs, dtype=np.int64)
    n = a.size
    mask = (1 << bits_per_coef) - 1
    a_masked = (a & mask).astype(np.uint64)

    total_bits = n * bits_per_coef
    n_bytes = (total_bits + 7) // 8
    out = bytearray(n_bytes)
    for i, c in enumerate(a_masked.tolist()):
        # c 는 uint, bit 0..bits_per_coef-1 가 의미. 시작 bit 위치 bp_start = i*bits_per_coef.
        bp = i * bits_per_coef
        # byte-align 한 chunk 별로 OR — Python int 의 임의 비트 시프트 활용.
        byte_idx = bp >> 3
        bit_off = bp & 7
        # c 를 bit_off 만큼 left shift 하면 byte stream 의 byte_idx 부터 OR 가능.
        shifted = c << bit_off
        # 최대 (bit_off + bits_per_coef) 비트, 즉 ⌈/8⌉ byte.
        nb = (bit_off + bits_per_coef + 7) // 8
        for k in range(nb):
            out[byte_idx + k] |= (shifted >> (8 * k)) & 0xFF
    return bytes(out)


def unpack_bitstream(buf: bytes, bits_per_coef: int, n_coefs: int) -> np.ndarray:
    """pack_bitstream 의 역."""
    if bits_per_coef <= 0 or bits_per_coef > 16:
        raise ValueError(f"bits_per_coef={bits_per_coef} out of (0, 16]")
    expected = (n_coefs * bits_per_coef + 7) // 8
    if len(buf) != expected:
        raise ValueError(
            f"unpack_bitstream: len={len(buf)} != expected {expected} "
            f"(n={n_coefs}, bits={bits_per_coef})"
        )
    mask = (1 << bits_per_coef) - 1
    out = np.zeros(n_coefs, dtype=np.int64)
    for i in range(n_coefs):
        bp = i * bits_per_coef
        byte_idx = bp >> 3
        bit_off = bp & 7
        nb = (bit_off + bits_per_coef + 7) // 8
        # 필요한 byte chunk 를 Python int 로 합쳐 mask.
        word = 0
        for k in range(nb):
            word |= int(buf[byte_idx + k]) << (8 * k)
        out[i] = (word >> bit_off) & mask
    return out


# ---------------------------------------------------------------------------
# Level-aware wrappers — params 로부터 bits/coef 와 length 자동 결정.
# 기존 signature 와의 호환을 위해 smaug1 default 로 동작.
# ---------------------------------------------------------------------------

def pack_rp(coeffs: np.ndarray, bits_per_coef: int = 8, n_coefs: int = 256) -> bytes:
    """R_p polynomial → byte stream. smaug1 default (LOG_P=8, n=256).

    smaug3/5 호출자는 bits_per_coef=p.log_p 로 전달.
    """
    a = np.asarray(coeffs, dtype=np.int64)
    if a.shape != (n_coefs,):
        raise ValueError(f"pack_rp expects ({n_coefs},), got {a.shape}")
    return pack_bitstream(a, bits_per_coef)


def unpack_rp(buf: bytes, bits_per_coef: int = 8, n_coefs: int = 256) -> np.ndarray:
    return unpack_bitstream(buf, bits_per_coef, n_coefs)


def pack_rp2(coeffs: np.ndarray, bits_per_coef: int = 5, n_coefs: int = 256) -> bytes:
    """R_p′ polynomial → byte stream. smaug1 default (LOG_P2=5, n=256)."""
    a = np.asarray(coeffs, dtype=np.int64)
    if a.shape != (n_coefs,):
        raise ValueError(f"pack_rp2 expects ({n_coefs},), got {a.shape}")
    return pack_bitstream(a, bits_per_coef)


def unpack_rp2(buf: bytes, bits_per_coef: int = 5, n_coefs: int = 256) -> np.ndarray:
    return unpack_bitstream(buf, bits_per_coef, n_coefs)


# ---------------------------------------------------------------------------
# Sx (sk polynomial) packing — 4 ternary coeff per byte (2 bit each)
# ---------------------------------------------------------------------------

def unpack_sx(packed: bytes) -> np.ndarray:
    """SMAUG-T 의 Sx_to_bytes 의 역 — 2-bit ternary unpack.

    인코딩 (smaug archive 의 nm 으로 export 된 Sx_to_bytes 기준, 'X' 명령
    dump 의 hist {-1: ~30, 0: ~186, +1: ~40} 로 검증, total HS = 70/poly):
        00 → 0,  01 → +1,  11 → -1,  10 → unused/error

    LWE_N=256 coeffs/poly → SKPOLY_BYTES = 64. LSB pair (bit 0-1) 가 coef 인덱스
    4i, 그 다음 (2-3) 가 4i+1, ...

    입력: bytes (길이 임의, 보통 64 또는 128).
    출력: int8 numpy array, shape = (len(packed) * 4,), 값 ∈ {-1, 0, +1}.
    """
    arr = np.frombuffer(packed, dtype=np.uint8).astype(np.int64)
    out = np.zeros(arr.size * 4, dtype=np.int8)
    for k in range(4):
        two_bit = (arr >> (k * 2)) & 0x3
        # vectorize: 0→0, 1→+1, 3→-1, 2→0 (defensive default)
        slot = np.zeros_like(two_bit)
        slot[two_bit == 1] = 1
        slot[two_bit == 3] = -1
        out[k::4] = slot.astype(np.int8)
    return out


def pack_sx(coeffs: np.ndarray) -> bytes:
    """unpack_sx 의 역: ternary {-1,0,+1} → 2-bit packed bytes (4 coeff/byte).

    인코딩 -1 → 11, 0 → 00, +1 → 01. 입력 길이는 4 의 배수여야 함.
    """
    a = np.asarray(coeffs, dtype=np.int64)
    if a.size % 4 != 0:
        raise ValueError(f"pack_sx needs len multiple of 4, got {a.size}")
    if not np.all((a == -1) | (a == 0) | (a == 1)):
        raise ValueError("pack_sx coeffs must be in {-1, 0, +1}")
    # map ternary → 2-bit code: 0→0, +1→1, -1→3
    code = np.zeros(a.size, dtype=np.uint8)
    code[a == 1] = 1
    code[a == -1] = 3
    code4 = code.reshape(-1, 4)
    out = np.zeros(code4.shape[0], dtype=np.uint8)
    for k in range(4):
        out |= (code4[:, k] << (k * 2))
    return out.tobytes()
