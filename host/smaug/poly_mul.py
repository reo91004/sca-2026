"""SMAUG-T `poly_mul_acc` host-side predictor.

archive disassembly (smaug1.a, ``cryptolab_smaug1_poly_mul_acc``) 로 검증한 사실:

  * 시그니처 : `void poly_mul_acc(const int16_t a[256], const int16_t b[256],
                                  int16_t res[256])`.
  * 내부에서 ``toom_cook_4way`` 가 길이-512 int16 임시 버퍼에 polynomial
    convolution `tmp[m] = sum_{p+q=m} a[p]*b[q]` 을 *모듈러 산술* 로 계산.
    Toom-Cook 4-way 의 보간 단계가 `1/3 = 0xAAAB`, `1/15 = 0xEEEF` 등 mod-2^16
    역원을 사용 — 이론적으로 mod 2^16 정확하지만, 실제 implementation 은
    *high-bit 잔여* 가 남는다 (eval points {0,±1,±2,...} 의 division 단계가
    낮은 bit 만 cancel). 결과 :
        firmware 응답 int16  ≡  (a * b)[i]   (mod p)
                              ≠  raw int16
    *원천 disasm 검증 + alpha={1..2048} sweep + idx sweep 으로 입증.*
  * SMAUG-T 가 이 함수를 부르는 indcpa_dec 는 결과를 어차피 ``round_t``
    (= compress to log_p bits → log_t bits) 로 잘라쓰기 때문에 mod-p 만
    맞으면 algorithmic 으로 충분. firmware 가 high-bit precision 을 유지하지
    않는 이유.
  * X^256 + 1 negacyclic 환원 : ``res[i] += tmp[i] - tmp[i + 256]`` 자체는
    mod 2^16 add/sub 라 *형태* 는 옳음 (단지 tmp[] 안에 high-bit 잔여가 있음).

따라서 host 측 ground truth (round-trip 검증용) :
    poly_mul_acc(a, b)[i]  ≡  ( (a * b) mod (X^256 + 1) )[i]   (mod p)

`a, b ∈ Z` 정수 다항식이라고 가정하고 numpy int64 로 convolution 한 뒤,
*반드시 mod p 한 결과* 를 보드 응답의 `mod p` 와 비교한다 (정확한 int16 비교는
불가능 — high-bit 잔여 때문).

SCA 관점:
    - 보드 power trace 는 *실제 int16 산술* 누설 → Hamming weight model 은
      예측된 mod-p 값으로는 불완전. 하지만 low log_p bits = 정확히 예측 가능,
      power trace 의 dominant signal 도 보통 low byte 에 강해서 HW(value mod p)
      는 여전히 강력한 predictor.

Sanity 단항 케이스 (`a = sk`, `b = α·X^j`) :

    res[i]  ≡  α · sk[(i - j) mod 256] · ( +1  if i ≥ j else -1 )   (mod p)

이건 b 가 단항일 때의 closed form 이며, ``negacyclic_mul`` 와 ``monomial_predict``
두 경로가 같은 *raw int* 값을 내는지 단위테스트로 cross-check 한다 (raw int 비교는
우리 host predictor 의 두 경로 일관성 — 보드와의 비교는 별도로 mod p 적용).
"""

from __future__ import annotations

import numpy as np

LWE_N = 256


def negacyclic_mul(a: np.ndarray, b: np.ndarray, *, n: int = LWE_N) -> np.ndarray:
    """`(a · b) mod (X^n + 1)` 를 정수 산술로 계산.

    a, b : shape (n,), 정수 dtype (int64 로 cast 후 계산).
    반환 : shape (n,) int64. *int16 wraparound 적용 전* 값. 호출자가 필요
    하면 ``np.int16`` 캐스트로 firmware 응답 과 동일한 16-bit 표현을 얻는다.
    """
    if a.shape != (n,) or b.shape != (n,):
        raise ValueError(f"need shape ({n},), got a={a.shape} b={b.shape}")
    a64 = np.asarray(a, dtype=np.int64)
    b64 = np.asarray(b, dtype=np.int64)
    full = np.convolve(a64, b64)  # 길이 2n-1
    # full[m] = sum_{p+q=m} a[p]*b[q], m ∈ [0, 2n-2].  X^n = -1 폴드:
    # res[i] = full[i] - full[i+n], i ∈ [0, n-1].  i+n > 2n-2 인 i 는 0 으로.
    res = np.zeros(n, dtype=np.int64)
    res[:] = full[:n]
    # full 의 길이는 2n-1 = 511. full[n .. 2n-2] 는 511-n = 255 개.
    # i.e. i+n in [n, 2n-1) 일 때만 fold; i = 0..n-2.
    res[: n - 1] -= full[n : 2 * n - 1]
    # i = n-1 의 경우 i+n = 2n-1 >= 2n-1 (out of full) → 추가 항 0.
    return res


def negacyclic_mul_int16(a: np.ndarray, b: np.ndarray, *, n: int = LWE_N) -> np.ndarray:
    """`negacyclic_mul` + int16 wraparound (mathematical reference).

    *주의* : 이 값은 firmware 'T' 응답의 raw int16 과 일치하지 *않는다*.
    Toom-Cook 4-way 가 high-bit 잔여를 남기기 때문 (모듈 docstring 참고). raw
    int16 비교를 원하면 firmware 의 정확한 산술 경로 (1/3·1/15 modular 보간)
    까지 emulate 해야 함 — 본 프로젝트에서는 SCA 목적상 mod-p 비교로 충분.
    """
    res64 = negacyclic_mul(a, b, n=n)
    return res64.astype(np.int16)


def negacyclic_mul_mod_p(
    a: np.ndarray,
    b: np.ndarray,
    *,
    log_p: int,
    n: int = LWE_N,
    signed: bool = True,
) -> np.ndarray:
    """`negacyclic_mul(a, b) mod 2^log_p` — firmware 'T' 응답과 mod p 일치.

    log_p : SMAUG-T 의 LOG_P (smaug1=8, smaug3=9, smaug5=9). c1 가 R_p 안에서
            동작하는 modulus 와 동일.
    signed: True 면 [-(p/2), +p/2) signed centered, False 면 [0, p) unsigned.
            firmware 응답의 raw int16 도 mod p 로 wrap 하면 두 표현 모두 가능
            (양쪽 다 검증).

    검증 :
        host_mod  = negacyclic_mul_mod_p(sk, b, log_p=8, signed=True)
        board_mod = (np.frombuffer(resp_32B, dtype=int16).astype(int64)) & 0xFF
                    이후 signed = brd > 127 인 byte 를 -256 → signed 변환
        host_mod == board_mod  좌표별 일치
    """
    if log_p <= 0 or log_p > 16:
        raise ValueError(f"log_p {log_p} out of (0, 16]")
    p = 1 << log_p
    res64 = negacyclic_mul(a, b, n=n)
    res_modp = res64 % p  # python mod 는 항상 [0, p)
    if signed:
        # [-(p/2), p/2) signed centered : 값 ≥ p/2 면 −= p
        res_modp = np.where(res_modp >= p // 2, res_modp - p, res_modp)
    return res_modp.astype(np.int64)


def board_response_mod_p(resp_32B: bytes, *, log_p: int, signed: bool = True) -> np.ndarray:
    """firmware 'T' 응답 32B 를 16 개 int16 로 파싱 후 mod p 표현.

    log_p == 8 인 smaug1 의 경우 32B 응답에서 짝수 위치 (low byte) 만 의미
    있는 비트. 이 함수는 그것을 명시적으로 추출해 host predictor 와 비교
    가능한 형태로 반환.
    """
    if len(resp_32B) != 32:
        raise ValueError(f"resp len {len(resp_32B)} != 32")
    raw = np.frombuffer(resp_32B, dtype=np.int16).astype(np.int64)
    p = 1 << log_p
    mod = raw % p
    if signed:
        mod = np.where(mod >= p // 2, mod - p, mod)
    return mod


def monomial_b(idx: int, alpha: int, *, n: int = LWE_N) -> np.ndarray:
    """Sparse host_b 다항식 = `α · X^idx`.

    'T' 명령 입력 (component, idx, alpha) 중 idx 와 alpha 로 만든 b. firmware 가
    `t_host_b[idx] = (int16_t)alpha` 로 한 좌표만 채운다.
    """
    if not (0 <= idx < n):
        raise ValueError(f"idx={idx} not in [0, {n})")
    b = np.zeros(n, dtype=np.int64)
    b[idx] = int(np.int16(alpha))  # int16 cast: alpha 가 int16 representable 보장
    return b


def monomial_predict(
    sk_poly: np.ndarray,
    idx: int,
    alpha: int,
    *,
    n: int = LWE_N,
) -> np.ndarray:
    """Closed-form: b = α·X^idx 일 때 `poly_mul_acc(sk_poly, b)` 의 정수값.

    `res[i] = α · sk[(i - idx) mod n] · sgn_wrap(i, idx)`
    where `sgn_wrap = +1 if i ≥ idx else -1`.

    이 식은 ``negacyclic_mul(sk_poly, monomial_b(idx, alpha))`` 와 동일해야 하며
    (단위테스트 ``tests/test_poly_mul.py`` 가 검증).
    """
    if sk_poly.shape != (n,):
        raise ValueError(f"sk_poly shape must be ({n},), got {sk_poly.shape}")
    if not (0 <= idx < n):
        raise ValueError(f"idx={idx} not in [0, {n})")
    sk = np.asarray(sk_poly, dtype=np.int64)
    a = int(np.int16(alpha))
    out = np.empty(n, dtype=np.int64)
    # i in [idx, n-1] : +sk[i - idx]
    out[idx:] = a * sk[: n - idx]
    # i in [0, idx-1] : -sk[i - idx + n]
    if idx > 0:
        out[:idx] = -a * sk[n - idx :]
    return out
