"""Host predictor for SMAUG-T poly_mul_acc — closed form vs convolution 일치 검증.

검증 항목:
  T1. negacyclic_mul 가 X^256 + 1 quotient ring 에서 self-product 와 일치
  T2. 단항 b = α·X^j 일 때 negacyclic_mul == monomial_predict (closed form)
  T3. ternary sk × 단항 b: int16 wraparound 후 firmware-식 byte 표현 32B 가 결정성

이 검증을 통과하면 host predictor 가 archive 동작 (disasm 로 reverse-engineer)
과 수학적으로 동치 — round-trip 실험에서 mismatch 가 발견되면 firmware 가
아니라 capture pipeline 또는 보드 sk-dump 단계 문제로 좁혀진다.
"""

from __future__ import annotations

import numpy as np

from host.smaug.poly_mul import (
    LWE_N,
    board_response_mod_p,
    monomial_b,
    monomial_predict,
    negacyclic_mul,
    negacyclic_mul_int16,
    negacyclic_mul_mod_p,
)


def _rand_ternary(n: int, hs: int, rng: np.random.Generator) -> np.ndarray:
    """랜덤 sparse ternary 다항식: |support|=hs, 각 좌표 ±1 random."""
    assert 0 < hs <= n
    out = np.zeros(n, dtype=np.int64)
    sup = rng.choice(n, size=hs, replace=False)
    signs = rng.choice([-1, 1], size=hs)
    out[sup] = signs
    return out


def _ref_negacyclic_naive(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """이중 루프 정의식: res[i] = sum_{p+q ≡ i (mod n)} sgn(p+q < n) · a[p]·b[q].

    speed 는 무시하고 정의에 가장 충실한 reference. negacyclic_mul (numpy.convolve
    기반) 의 결과와 모든 좌표가 같으면 numpy 경로의 정확성이 단정된다.
    """
    a64 = np.asarray(a, dtype=np.int64)
    b64 = np.asarray(b, dtype=np.int64)
    res = np.zeros(n, dtype=np.int64)
    for p in range(n):
        ap = int(a64[p])
        if ap == 0:
            continue
        for q in range(n):
            bq = int(b64[q])
            if bq == 0:
                continue
            s = p + q
            if s < n:
                res[s] += ap * bq
            else:
                res[s - n] -= ap * bq
    return res


def test_negacyclic_mul_matches_naive_reference():
    """T1: numpy.convolve fold 가 이중 루프 정의식과 일치."""
    rng = np.random.default_rng(0xC0FFEE)
    for trial in range(5):
        a = _rand_ternary(LWE_N, hs=70, rng=rng)
        # b 는 임의 dense int16 — 보드 'T' 의 host_b 는 host 통제니까 dense 가능
        b = rng.integers(-128, 128, size=LWE_N, dtype=np.int64)
        got = negacyclic_mul(a, b)
        want = _ref_negacyclic_naive(a, b, LWE_N)
        assert got.shape == want.shape == (LWE_N,)
        if not np.array_equal(got, want):
            diff = np.where(got != want)[0]
            raise AssertionError(
                f"trial {trial}: negacyclic_mul mismatch at {diff[:5].tolist()} "
                f"got[{diff[0]}]={got[diff[0]]} want={want[diff[0]]}"
            )


def test_monomial_closed_form_matches_convolution():
    """T2: b = α·X^j 일 때 closed form == numpy convolve."""
    rng = np.random.default_rng(0xDEADBEEF)
    for trial in range(20):
        sk = _rand_ternary(LWE_N, hs=70, rng=rng)
        idx = int(rng.integers(0, LWE_N))
        # alpha 는 보드 firmware 가 int16 으로 받음. host predictor 도 int16 cast.
        alpha = int(rng.integers(-10000, 10001))
        b = monomial_b(idx, alpha)
        full = negacyclic_mul(sk, b)
        closed = monomial_predict(sk, idx, alpha)
        if not np.array_equal(full, closed):
            diff = np.where(full != closed)[0]
            raise AssertionError(
                f"trial {trial} idx={idx} alpha={alpha}: "
                f"closed form != convolve at {diff[:5].tolist()}"
            )


def test_int16_wrap_byte_layout():
    """T3: int16 wrap 후 little-endian 32B 표현이 결정성.

    firmware 의 `simpleserial_put('r', 32, (uint8_t *)t_out)` 가 보내는
    바이트 패턴 = `t_out[0..15].tobytes()` (little-endian int16). 본 테스트는
    *mathematical reference* (Toom-Cook 잔여 무시) 의 결정성만 확인.
    """
    rng = np.random.default_rng(0xFEEDFACE)
    sk = _rand_ternary(LWE_N, hs=70, rng=rng)
    b = monomial_b(idx=200, alpha=12345)
    out = negacyclic_mul_int16(sk, b)
    assert out.dtype == np.int16
    assert out.shape == (LWE_N,)
    head32 = out[:16].tobytes()
    assert len(head32) == 32, f"expected 32 bytes, got {len(head32)}"
    out2 = negacyclic_mul_int16(sk, b)
    assert out2[:16].tobytes() == head32, "int16 wrap not deterministic"


def test_mod_p_centered_signed():
    """T4: ``negacyclic_mul_mod_p(signed=True)`` 가 [-p/2, p/2) 안에 들어간다.

    smaug1 LOG_P=8 → p=256. 결과 ∈ [-128, 128). monomial 케이스로 sk·alpha
    값과 직접 비교.
    """
    rng = np.random.default_rng(0xCAFEBABE)
    sk = _rand_ternary(LWE_N, hs=70, rng=rng)
    log_p = 8
    p = 1 << log_p

    for trial in range(10):
        idx = int(rng.integers(0, LWE_N))
        alpha = int(rng.integers(-200, 200))  # 범위 작게 → mod p 후 signed 안에 들어감
        b = monomial_b(idx, alpha)
        mod_p_signed = negacyclic_mul_mod_p(sk, b, log_p=log_p, signed=True)
        assert mod_p_signed.shape == (LWE_N,)
        assert np.all(mod_p_signed >= -p // 2)
        assert np.all(mod_p_signed < p // 2)

        # closed-form 과 일치 (mod p, signed) 비교
        cf = monomial_predict(sk, idx, alpha)
        cf_modp_signed = cf % p
        cf_modp_signed = np.where(
            cf_modp_signed >= p // 2, cf_modp_signed - p, cf_modp_signed
        )
        if not np.array_equal(mod_p_signed, cf_modp_signed):
            diff = np.where(mod_p_signed != cf_modp_signed)[0][:5]
            raise AssertionError(
                f"trial {trial} idx={idx} alpha={alpha}: "
                f"mod_p mismatch closed_form vs convolve at {diff.tolist()}"
            )


def test_board_response_mod_p_unsigned_round_trip():
    """T5: ``board_response_mod_p`` 가 raw int16 → mod p 변환을 정확히 한다.

    임의 int16 버퍼를 만들고 mod p 결과를 numpy 직접 계산 + 본 함수 결과와
    비교.
    """
    rng = np.random.default_rng(0xBADF00D)
    raw = rng.integers(-32768, 32768, size=16, dtype=np.int16)
    log_p = 8
    p = 1 << log_p

    raw_bytes = raw.tobytes()
    assert len(raw_bytes) == 32

    # signed=True 비교
    mod_signed = (raw.astype(np.int64) % p)
    mod_signed = np.where(mod_signed >= p // 2, mod_signed - p, mod_signed)
    got_signed = board_response_mod_p(raw_bytes, log_p=log_p, signed=True)
    assert np.array_equal(got_signed, mod_signed)

    # signed=False 비교
    mod_unsigned = (raw.astype(np.int64) % p)
    got_unsigned = board_response_mod_p(raw_bytes, log_p=log_p, signed=False)
    assert np.array_equal(got_unsigned, mod_unsigned)


def test_mod_p_explains_observed_high_bit_residue():
    """T6: 관찰된 firmware bias (24576 = 96·p, etc.) 가 mod-p 모델로 설명됨.

    실측 데이터 (s1_t_diag.py 가 보드에서 받은 값) 에서:
        host_predict = 1, brd_int16 = 24577 = 24576 + 1 = 96·256 + 1
        (24577 mod 256) = 1 = host_predict ✓

    본 테스트는 그 *논리* 가 일관됨을 확인한다 (보드 무관, 순수 산술).
    """
    log_p = 8
    p = 1 << log_p
    # examples from real board capture
    cases = [
        (1, 24577),     # host=1, brd=24577 (96·p + 1)
        (0, 24576),     # host=0, brd=24576 (96·p)
        (-1, 32767),    # host=-1, brd=32767 (= 128·p - 1, signed-mod = -1)
        (2, 24578),
        (1024, 25600),  # host=1024 ≡ 0 (mod p), brd=25600 ≡ 0 (mod p)
        (-2048, -2048), # host=-2048 ≡ 0 (mod p), brd=-2048 ≡ 0
    ]
    for host_val, brd_val in cases:
        host_modp = host_val % p
        brd_modp = brd_val % p
        if host_modp != brd_modp:
            raise AssertionError(
                f"host {host_val} mod p={host_modp}, brd {brd_val} mod p={brd_modp}"
            )
