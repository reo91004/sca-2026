"""Compress/Decompress 정합성과 polynomial pack/unpack round-trip.

검증 대상:
  - compress_ref vs compress_bit 가 [0, q) 전 도메인에서 일치 (codec 두 구현)
  - fixed_point_set 이 SMAUG-T smaug1 의 c1, c2 에 대해 사이즈 256/32 가 맞음
  - pack_rp / unpack_rp round-trip (256 byte ↔ 256 coef)
  - pack_rp2 / unpack_rp2 round-trip (160 byte ↔ 256 × 5-bit)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 sys.path 에 추가 (raw 실행 모드 대응)
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.smaug import codec  # noqa: E402
from host.smaug import params  # noqa: E402


def test_smaug1_params_consistent() -> None:
    p = params.SMAUG1
    p.assert_consistent()
    assert p.q == 1024
    assert p.p == 256
    assert p.p2 == 32
    assert p.t == 2
    assert p.ctpoly1_bytes == 256
    assert p.ctpolyvec_bytes == 512
    assert p.ctpoly2_bytes == 160
    assert p.ciphertext_bytes == 672


def test_compress_ref_bit_agree_full_domain() -> None:
    """두 Compress 구현이 [0, q) 전 도메인에서 동일."""
    log_q = 10
    q = 1 << log_q
    x = np.arange(q, dtype=np.int64)
    for d in (8, 5):  # smaug1 의 c1, c2 modulus
        ref = codec.compress_ref(x, d, log_q)
        bit = codec.compress_bit(x, d, log_q)
        assert np.array_equal(ref, bit), (
            f"compress_ref 와 compress_bit 가 d={d} 에서 불일치: "
            f"first diff at x={int(np.argmax(ref != bit))}"
        )


def test_fixed_point_set_smaug1() -> None:
    """U_p (d=8) = 256 개, U_p′ (d=5) = 32 개, 모두 등간격."""
    log_q = 10
    Up = codec.fixed_point_set(8, log_q)
    Up2 = codec.fixed_point_set(5, log_q)
    assert Up.size == 256
    assert Up2.size == 32
    assert int(Up[0]) == 0
    assert int(Up[1] - Up[0]) == 4   # q/p = 1024/256 = 4
    assert int(Up2[1] - Up2[0]) == 32  # q/p′ = 32
    # 마지막 원소: q - step
    assert int(Up[-1]) == 1024 - 4
    assert int(Up2[-1]) == 1024 - 32


def test_decompress_inverts_compress_on_fixed_point_set() -> None:
    log_q = 10
    for d in (8, 5):
        U = codec.fixed_point_set(d, log_q)
        c = codec.compress_ref(U, d, log_q)
        back = codec.decompress(c, d, log_q)
        assert np.array_equal(back, U), f"d={d} fixed-point 깨짐"


def test_pack_rp_round_trip_random() -> None:
    rng = np.random.default_rng(42)
    coeffs = rng.integers(0, 256, size=256, dtype=np.int64)
    buf = codec.pack_rp(coeffs)
    assert len(buf) == 256
    back = codec.unpack_rp(buf)
    assert np.array_equal(back, coeffs)


def test_pack_rp_round_trip_full_domain() -> None:
    """0..255 전 도메인을 한 다항식에 다 채워도 round-trip."""
    coeffs = np.arange(256, dtype=np.int64)
    back = codec.unpack_rp(codec.pack_rp(coeffs))
    assert np.array_equal(back, coeffs)


def test_pack_rp2_round_trip_random() -> None:
    rng = np.random.default_rng(7)
    coeffs = rng.integers(0, 32, size=256, dtype=np.int64)
    buf = codec.pack_rp2(coeffs)
    assert len(buf) == 160
    back = codec.unpack_rp2(buf)
    assert np.array_equal(back, coeffs)


def test_pack_rp2_round_trip_edge_values() -> None:
    """경계값들: all-zero, all-31, alternating, 단항."""
    for arr in (
        np.zeros(256, dtype=np.int64),
        np.full(256, 31, dtype=np.int64),
        np.tile(np.array([0, 31], dtype=np.int64), 128),
        np.eye(256, dtype=np.int64)[:, 0] * 17,  # 한 위치만 17, 나머지 0
    ):
        back = codec.unpack_rp2(codec.pack_rp2(arr))
        assert np.array_equal(back, arr & 0x1F)


def test_pack_sx_unpack_sx_round_trip_random() -> None:
    rng = np.random.default_rng(11)
    n_coef = 256  # smaug1 LWE_N
    coeffs = rng.choice([-1, 0, 1], size=n_coef, p=[0.137, 0.726, 0.137]).astype(np.int64)
    packed = codec.pack_sx(coeffs)
    assert len(packed) == n_coef // 4   # SKPOLY_BYTES = LWE_N / 4
    back = codec.unpack_sx(packed)
    assert np.array_equal(back, coeffs)


def test_pack_sx_unpack_sx_edge_values() -> None:
    cases = [
        np.zeros(256, dtype=np.int64),
        np.full(256, +1, dtype=np.int64),
        np.full(256, -1, dtype=np.int64),
        np.tile([+1, -1, 0, 0], 64).astype(np.int64),
        np.tile([-1, +1, +1, -1], 64).astype(np.int64),
    ]
    for arr in cases:
        back = codec.unpack_sx(codec.pack_sx(arr))
        assert np.array_equal(back, arr), \
            f"round-trip failed for arr starts {arr[:8].tolist()}"


def test_unpack_sx_smaug1_hamming_property() -> None:
    """smaug1 sk: HS=70 per poly, balanced (sum 의 평균 ≈ 0)."""
    rng = np.random.default_rng(13)
    HS = 70; n_coef = 256
    nonzero_idx = rng.choice(n_coef, size=HS, replace=False)
    signs = rng.choice([-1, +1], size=HS)
    s = np.zeros(n_coef, dtype=np.int64); s[nonzero_idx] = signs
    packed = codec.pack_sx(s)
    back = codec.unpack_sx(packed)
    assert np.array_equal(back, s)
    assert int((back != 0).sum()) == HS


def test_pack_sx_rejects_invalid_values() -> None:
    bad = np.array([2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    try:
        codec.pack_sx(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("pack_sx should reject non-ternary values")


if __name__ == "__main__":
    # raw 실행 시: 이 모듈의 모든 test_* 함수를 차례로 호출.
    fns = sorted(n for n in globals() if n.startswith("test_"))
    for n in fns:
        print(f"[RUN] {n}")
        globals()[n]()
        print(f"[OK ] {n}")
    print(f"[OK ] {len(fns)} tests")
