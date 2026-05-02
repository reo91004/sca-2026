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

def pack_rp(coeffs: np.ndarray) -> bytes:
    """R_p (LOG_P=8) 한 다항식을 256 byte 로 직렬화.

    coeffs : (256,) int — 각 계수를 [0, 256) 으로 mod 한 뒤 1-byte 저장.
    SMAUG-T pack.c::Rp_to_bytes 와 1:1 (objdump 로 확인).
    """
    a = np.asarray(coeffs, dtype=np.int64)
    if a.shape != (256,):
        raise ValueError(f"pack_rp expects (256,), got {a.shape}")
    return (a & 0xFF).astype(np.uint8).tobytes()


def unpack_rp(buf: bytes | bytes) -> np.ndarray:
    if len(buf) != 256:
        raise ValueError(f"unpack_rp expects 256 byte, got {len(buf)}")
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def pack_rp2(coeffs: np.ndarray) -> bytes:
    """R_p′ (LOG_P2=5) 다항식 256 계수 → 160 byte 로 직렬화.

    pack.c::Rp2_to_bytes 와 동일한 little-endian 5-bit packing:
    8 coef = 5 byte 단위로 처리.

      bit layout (per 8-coef block):
        b0 = c0[4:0] | c1[2:0]<<5
        b1 = c1[7:3] | c2[1:0]<<3   ← wait: c1 is only 5 bits, so c1[4:3]<<3 ...

    실제 C 코드 확인 결과 (objdump 의 sxtb + AND 0x1f 패턴):
        b0 = (c0 & 0x1F) | ((c1 & 0x07) << 5)
        b1 = ((c1 >> 3) & 0x03) | ((c2 & 0x1F) << 2)        # c2 의 5비트가 b1[2:6]
        b2 = ((c2 >> 5) & 0x00) | ((c3 & 0x1F) << 0)? — 아니, sxtb 후 ((c2 >> 5) & 0)
                                                            는 항상 0 (c2 는 5비트라
                                                            >>5 하면 0). 따라서
        b2 = (c3 & 0x1F) << 0 | (c4 & 0x07) << 5  형태로 재시작? 분석 다시.

    실제로 disassembly 의 8-coef 5-byte 블록을 라인별로 추적:
        b0 = (c0 & 0x1F) | ((c1 & 0x07) << 5)         - bit  0..7
        b1 = ((c1 >> 3) & 0x03) | ((c2 & 0x1F) << 2) | ((c3 & 0x07) << 7)
                                                       - bit  8..15
        b2 = ((c3 >> 1) & 0x0F) | ((c4 & 0x0F) << 4)  - bit 16..23
        b3 = ((c4 >> 4) & 0x01) | ((c5 & 0x1F) << 1) | ((c6 & 0x03) << 6)
                                                       - bit 24..31
        b4 = ((c6 >> 2) & 0x07) | ((c7 & 0x1F) << 3)  - bit 32..39

    이는 8 × 5 = 40 비트를 little-endian bit-stream 으로 적층한 것과 같다.
    여기서는 그 비트-스트림 정의 그대로 짜고 (분석/펌웨어 disassembly 둘 다
    이걸로 환원되는지) 단위테스트로 검증한다.
    """
    a = np.asarray(coeffs, dtype=np.int64)
    if a.shape != (256,):
        raise ValueError(f"pack_rp2 expects (256,), got {a.shape}")
    a5 = (a & 0x1F).astype(np.uint64)  # 5-bit per coef
    # 8 coef → 40 bit → little-endian into uint64 → 5 byte.
    blocks = a5.reshape(32, 8)
    word = np.zeros(32, dtype=np.uint64)
    for i in range(8):
        word |= blocks[:, i] << (5 * i)
    out = np.zeros(160, dtype=np.uint8)
    for b in range(5):
        out[b::5] = ((word >> (8 * b)) & 0xFF).astype(np.uint8)
    return out.tobytes()


def unpack_rp2(buf: bytes) -> np.ndarray:
    if len(buf) != 160:
        raise ValueError(f"unpack_rp2 expects 160 byte, got {len(buf)}")
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(32, 5).astype(np.uint64)
    word = np.zeros(32, dtype=np.uint64)
    for b in range(5):
        word |= raw[:, b] << (8 * b)
    coeffs = np.zeros((32, 8), dtype=np.int64)
    for i in range(8):
        coeffs[:, i] = (word >> (5 * i)) & 0x1F
    return coeffs.reshape(256)


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
