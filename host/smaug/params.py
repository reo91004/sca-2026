"""SMAUG-T 보안 레벨별 파라미터.

include/crypto_kem/smaug{1,3,5}/parameters.h 의 매크로를 그대로 옮긴다.
런타임에 헤더를 다시 파싱하지 않고 정적 dict 로 굳혀, host 측 시뮬레이터
가 빌드 산출물 없이도 동작하게 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmaugParams:
    """단일 보안 레벨의 SMAUG-T 파라미터 묶음."""

    name: str
    n: int                # LWE_N (poly degree)
    module_rank: int      # MODULE_RANK
    log_q: int            # LOG_Q     (public modulus)
    log_p: int            # LOG_P     (c1 modulus)
    log_p2: int           # LOG_P2    (c2 modulus)
    log_t: int            # LOG_T     (message modulus, 1 = binary)
    hs: int               # Hamming weight of secret coefficient vector
    public_key_bytes: int
    secret_key_bytes: int
    ciphertext_bytes: int
    delta_bytes: int      # message μ size
    shared_secret_bytes: int

    @property
    def q(self) -> int:
        return 1 << self.log_q

    @property
    def p(self) -> int:
        return 1 << self.log_p

    @property
    def p2(self) -> int:
        return 1 << self.log_p2

    @property
    def t(self) -> int:
        return 1 << self.log_t

    @property
    def ctpoly1_bytes(self) -> int:
        return self.log_p * self.n // 8

    @property
    def ctpoly2_bytes(self) -> int:
        return self.log_p2 * self.n // 8

    @property
    def ctpolyvec_bytes(self) -> int:
        return self.ctpoly1_bytes * self.module_rank

    @property
    def skpoly_bytes(self) -> int:
        # SKPOLY_BYTES = LWE_N / 4 (2-bit ternary packed), spec 정의.
        return self.n // 4

    @property
    def pke_secret_key_bytes(self) -> int:
        # PKE_SECRETKEY_BYTES = SKPOLYVEC_BYTES = SKPOLY_BYTES * MODULE_RANK
        # smaug1=128, smaug3=192, smaug5=256. attack 의 'X' dump 가 이 부분.
        return self.skpoly_bytes * self.module_rank

    def assert_consistent(self) -> None:
        """parameters.h 의 합산 관계가 dataclass 와 일치하는지 검증."""
        if self.ctpolyvec_bytes + self.ctpoly2_bytes != self.ciphertext_bytes:
            raise ValueError(
                f"{self.name}: ctpolyvec({self.ctpolyvec_bytes}) + "
                f"ctpoly2({self.ctpoly2_bytes}) != "
                f"CIPHERTEXT_BYTES({self.ciphertext_bytes})"
            )
        if self.delta_bytes != self.n // 8:
            raise ValueError(
                f"{self.name}: delta_bytes={self.delta_bytes} != n/8={self.n // 8}"
            )


# parameters.h 에서 그대로:
SMAUG1 = SmaugParams(
    name="smaug1",
    n=256, module_rank=2,
    log_q=10, log_p=8, log_p2=5, log_t=1,
    hs=70,
    public_key_bytes=672,
    secret_key_bytes=160 + 672,
    ciphertext_bytes=672,
    delta_bytes=32,
    shared_secret_bytes=32,
)

# spec values cross-checked against include/crypto_kem/smaug{3,5}/{parameters,api}.h.
# CRYPTO_PUBLICKEYBYTES / CRYPTO_CIPHERTEXTBYTES 는 api.h 에 hardcoded 된 값.
SMAUG3 = SmaugParams(
    name="smaug3",
    n=256, module_rank=3,
    log_q=11, log_p=9, log_p2=4, log_t=1,
    hs=88,
    public_key_bytes=1088,        # api.h: CRYPTO_PUBLICKEYBYTES = 1088
    secret_key_bytes=224 + 1088,  # api.h: CRYPTO_SECRETKEYBYTES = 224 + 1088
    ciphertext_bytes=992,         # api.h: CRYPTO_CIPHERTEXTBYTES = 992
    delta_bytes=32,
    shared_secret_bytes=32,
)

# smaug5 (KpqC SMAUG-T v4.0): MODULE_RANK = 4 (NOT 5 — common 잘못된 짐작),
# HS = 87 (not 152).
SMAUG5 = SmaugParams(
    name="smaug5",
    n=256, module_rank=4,
    log_q=11, log_p=9, log_p2=7, log_t=1,
    hs=87,
    public_key_bytes=1440,        # api.h: CRYPTO_PUBLICKEYBYTES = 1440
    secret_key_bytes=352 + 1440,  # api.h: CRYPTO_SECRETKEYBYTES = 352 + 1440
    ciphertext_bytes=1376,        # api.h: CRYPTO_CIPHERTEXTBYTES = 1376
    delta_bytes=32,
    shared_secret_bytes=32,
)


BY_NAME: dict[str, SmaugParams] = {
    "smaug1": SMAUG1,
    "smaug3": SMAUG3,
    "smaug5": SMAUG5,
}


def get(name: str) -> SmaugParams:
    if name not in BY_NAME:
        raise KeyError(f"unknown SMAUG-T level: {name} (try {list(BY_NAME)})")
    return BY_NAME[name]
