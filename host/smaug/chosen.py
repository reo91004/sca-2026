"""Chosen-ciphertext 빌더 (host 측, 보드와 통신하지 않음).

이 모듈은 docs/idea.md 의 µ′ classifier 전략을 SMAUG-T smaug1 의
Compress/Decompress fixed-point 집합 위에 직접 옮긴 코드.

수학 (smaug1, T=2):

    µ′_i = ⌊ (t / p)·⟨c1, s⟩_i + (t / p′)·c2_i ⌉ mod t
         = ⌊ (1 / 128)·⟨c1, s⟩_i + (1 / 16)·c2_i ⌉ mod 2

c1 ∈ R_p^k 의 계수는 fixed-point set U_p = {0, 4, 8, …, q − 4}, c2 ∈ R_p′
의 계수는 U_p′ = {0, 32, 64, …, q − 32}. host 가 만든 c1/c2 의 모든 계수가
이 집합 안에 들어와야 *Decompress(Compress(·))* 가 항등이 되어, 보드가
실제로 받는 다항식이 host 가 의도한 그것과 정확히 일치한다.

빌더:
    build_mu_constant(p, mu_bit)        c1=0, c2 = round-up/down threshold
    build_monomial_c1(p, k, j, alpha)   c1 = α · X^j (component k 만 단항)
"""

from __future__ import annotations

import numpy as np

from .codec import fixed_point_set
from .ciphertext import Ciphertext
from .params import SmaugParams


def build_mu_constant(
    params: SmaugParams,
    mu_bit: int,
) -> Ciphertext:
    """c1 = 0, c2 = const · 1 (모든 계수 동일).

    target µ′ bit:
        mu_bit=0 → c2 의 모든 계수를 *0* (round 결과 bit 0)
        mu_bit=1 → c2 의 모든 계수를 *p′/2 의 fixed-point* (round 결과 bit 1)

    논리:
        µ′_i = ⌊ (1/16) · c2_i ⌉ mod 2
        c2_i ∈ {0, 32, …, 992} (U_p′). c2_i = 0 → µ′_i = 0 명백.
        c2_i = 16·k 형태가 정확히 round-up boundary 인데, U_p′ 는 32 단위라
        정확한 0.5 LSB 가 없다. 가장 가까운 fixed-point: c2_i = 16
        (U_p′ 가 아님!) → 32 (U_p′ 안). c2 = 32 → µ′_i = ⌊32/16⌉ = 2 → bit 0. 망.

        실제로 t/p′ = 2/32 = 1/16. round(c/16) mod 2.
            c=0   → 0  → bit 0
            c=8   → 0.5 → 1 → bit 1   (round-half-up)
            c=16  → 1  → bit 1
            c=24  → 1.5 → 2 → bit 0
            c=32  → 2  → bit 0
        패턴: bit 1 영역은 c ∈ [8, 24) 와 그 주기 (32 단위). 따라서 c2 의
        모든 계수를 *fixed-point set 안에서* µ′=1 을 만들기 위해선 c=8 또는
        16 또는 등이 필요한데, U_p′ = {0, 32, 64, …} 에는 8 도 16 도 없다.

        해결: U_p′ 가 32 단위라는 건 *Decompress(Compress(c))* round-trip 의
        보존 영역. 하지만 c2 를 *bytes 로 보드에 보낼 때* 는 5비트 (Compress
        결과) 도메인이라 c2_compressed ∈ [0, 32). 따라서 보드 위에서
        Decompress 한 값은 32 의 배수 — fixed-point. mu_bit=1 을 만들려면
        Decompressed c2_i 가 [8, 24) 안에 있어야 하는데 32 의 배수만 가능
        하므로 **c1=0 + 어떤 c2 로도 µ′=1 만으로 채울 수는 없다**.

        ⇒ 이 함수는 mu_bit=0 만 지원. mu_bit=1 일 때는 구조상 불가능 —
           idea.md 가 가정한 "all-zero / all-one µ′" 중 *all-one 은 c1 도 동원해야*
           가능하다 (예: c1 의 한 계수를 큰 값으로 둬서 ⟨c1, s⟩ 가 더해지게).

    실험적 우회:
        - build_mu_constant(p, 0) 으로 trivial 클래스 A 만 만들고,
        - 클래스 B 는 build_monomial_c1(p, 0, 0, alpha=4·k) 같은 한 계수 nonzero
          ct 로 두면 µ′ 의 *일부 bit* 가 갈리는 ct 가 된다 — TVLA 표적이
          되는 차분이 그 자리에서 발생.

    그래서 이 함수는 mu_bit=0 한 가지만 만들고, mu_bit=1 요청은 명시적
    NotImplementedError 로 띄운다. 이게 현재 idea.md 의 한계.
    """
    if mu_bit == 0:
        c1 = np.zeros((params.module_rank, params.n), dtype=np.int64)
        c2 = np.zeros(params.n, dtype=np.int64)
        return Ciphertext(c1=c1, c2=c2, params=params)
    raise NotImplementedError(
        "mu_bit=1 모든-1 µ′ 는 c2 만으로 구성 불가 (U_p′={0,32,…} 의 round-trip "
        "결과로는 µ′ bit=1 을 만들 수 없음). 대안은 docs/idea.md §'두 번째 공격 "
        "지점' 또는 build_monomial_c1 + sk-dependent 차이를 이용. "
        "추후 chosen-CT 분류기 설계를 다시 한다."
    )


def build_monomial_c1(
    params: SmaugParams,
    component: int,
    coef_idx: int,
    alpha: int,
) -> Ciphertext:
    """c1 = α · X^j (component k 의 단항), c2 = 0.

    α 는 fixed-point set U_p (= {0, 4, 8, …, q − 4} for smaug1) 안에서 골라야
    하고, 입력값이 그 집합에 없으면 ValueError. coef_idx ∈ [0, n).

    이 ct 는 ⟨c1, s⟩_i = α · s_{(i − coef_idx) mod n} (cyclic with anticyclic
    sign at wrap) 형태가 되어, µ′_i 가 *단일 비밀 계수* 분류기로 작용한다.
    """
    if not (0 <= component < params.module_rank):
        raise IndexError(f"component {component} out of [0, {params.module_rank})")
    if not (0 <= coef_idx < params.n):
        raise IndexError(f"coef_idx {coef_idx} out of [0, {params.n})")

    fp = fixed_point_set(params.log_p, params.log_q)
    if alpha not in fp.tolist():
        raise ValueError(
            f"alpha={alpha} ∉ U_p (size {fp.size}). 시작점 후보: "
            f"{fp[:5].tolist()} … {fp[-3:].tolist()}"
        )

    c1 = np.zeros((params.module_rank, params.n), dtype=np.int64)
    c1[component, coef_idx] = alpha
    c2 = np.zeros(params.n, dtype=np.int64)
    return Ciphertext(c1=c1, c2=c2, params=params)


def chunkify(ct_bytes: bytes, chunk_size: int = 32) -> list[tuple[int, bytes]]:
    """보드 'I' 명령을 위해 ct 를 (idx, chunk) 리스트로 자른다.

    smaug1: 672 / 32 = 21 chunks. idx 는 0..20, 각 chunk 는 32 byte.
    """
    if len(ct_bytes) % chunk_size != 0:
        raise ValueError(
            f"ct length {len(ct_bytes)} not divisible by chunk={chunk_size}"
        )
    n_chunks = len(ct_bytes) // chunk_size
    return [
        (i, ct_bytes[i * chunk_size:(i + 1) * chunk_size])
        for i in range(n_chunks)
    ]
