"""Chosen-ciphertext 빌더 (host 측, 보드와 통신하지 않음).

이 모듈은 docs/idea.md 의 µ′ classifier 전략을 SMAUG-T 의 ciphertext
직렬화 도메인 위에 직접 옮긴 코드.

수학 (smaug1, T=2):

    µ′_i = ⌊ (t / p)·⟨c1, s⟩_i + (t / p′)·c2_i ⌉ mod t
         = ⌊ (1 / 128)·⟨c1, s⟩_i + (1 / 16)·c2_i ⌉ mod 2

c1 ∈ R_p^k 의 계수는 [0, p), c2 ∈ R_p′ 의 계수는 [0, p′) 이다. 이 값들이
그대로 ciphertext byte stream 에 pack 되고, 펌웨어의 indcpa_dec 도 같은 R_p /
R_p′ 도메인 값을 복호화 식에 사용한다. q-domain fixed-point 값(예: smaug1 의
320, 576, ...)을 여기 넣으면 pack 시 하위 log_p 비트만 남아 다른 ciphertext 와
alias 되므로 허용하지 않는다.

빌더:
    build_mu_constant(p, mu_bit)        c1=0, c2 = round-up/down threshold
    build_monomial_c1(p, k, j, alpha)   c1 = α · X^j (component k 만 단항)
"""

from __future__ import annotations

import numpy as np

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

    α 는 ciphertext 의 R_p 계수이므로 0 <= α < p 이어야 한다.
    coef_idx ∈ [0, n).

    이 ct 는 ⟨c1, s⟩_i = α · s_{(i − coef_idx) mod n} (cyclic with anticyclic
    sign at wrap) 형태가 되어, µ′_i 가 *단일 비밀 계수* 분류기로 작용한다.
    """
    if not (0 <= component < params.module_rank):
        raise IndexError(f"component {component} out of [0, {params.module_rank})")
    if not (0 <= coef_idx < params.n):
        raise IndexError(f"coef_idx {coef_idx} out of [0, {params.n})")

    if not (0 <= int(alpha) < params.p):
        raise ValueError(f"alpha={alpha} outside R_p=[0, {params.p})")

    c1 = np.zeros((params.module_rank, params.n), dtype=np.int64)
    c1[component, coef_idx] = int(alpha)
    c2 = np.zeros(params.n, dtype=np.int64)
    return Ciphertext(c1=c1, c2=c2, params=params)


def build_multi_term_c1(
    params: SmaugParams,
    coefs: dict[tuple[int, int], int],
) -> Ciphertext:
    """c1[m, l] = α for each (m, l) → α 매핑. 나머지 0. c2 = 0.

    coefs : {(component, coef_idx): alpha}. component ∈ [0, module_rank),
    coef_idx ∈ [0, n), alpha ∈ [0, p). 한 component 의 한 자리만 비어있어도
    build_monomial_c1 과 동치.

    범용 빌더 — Phase H 의 multi-term chosen-CT 에 사용.
    """
    c1 = np.zeros((params.module_rank, params.n), dtype=np.int64)
    for (m, l), alpha in coefs.items():
        if not (0 <= m < params.module_rank):
            raise IndexError(f"component {m} ∉ [0, {params.module_rank})")
        if not (0 <= l < params.n):
            raise IndexError(f"coef_idx {l} ∉ [0, {params.n})")
        if not (0 <= int(alpha) < params.p):
            raise ValueError(f"alpha={alpha} outside R_p=[0, {params.p}) at (m={m}, l={l})")
        c1[m, l] = int(alpha)
    c2 = np.zeros(params.n, dtype=np.int64)
    return Ciphertext(c1=c1, c2=c2, params=params)


def build_combined_c1(
    params: SmaugParams,
    coefs0: dict[int, int] | None = None,
    coefs1: dict[int, int] | None = None,
) -> Ciphertext:
    """R^2 cross-component multi-term c1 — eprint25 OT-FDA-CK 영감.

    coefs0 = {l: α_l}  →  c1[0] = Σ_l α_l · X^l
    coefs1 = {k: α_k}  →  c1[1] = Σ_k α_k · X^k

    예시:
        build_combined_c1(params, {10: 64}, {50: 64})
            → c1[0] = 64·X^10, c1[1] = 64·X^50.
            → ⟨c1, s⟩_i = 64·(s[0]_(i-10) + s[1]_(i-50))  (sign flip on wrap)

    한 chosen-CT 가 두 component 를 동시에 probe.
    """
    coefs0 = coefs0 or {}
    coefs1 = coefs1 or {}
    if params.module_rank < 2 and coefs1:
        raise ValueError(
            f"coefs1 non-empty but module_rank={params.module_rank} < 2"
        )
    coefs: dict[tuple[int, int], int] = {}
    for l, a in coefs0.items():
        coefs[(0, l)] = a
    for k, a in coefs1.items():
        coefs[(1, k)] = a
    return build_multi_term_c1(params, coefs)


def predict_mu_prime(
    params: SmaugParams,
    c1: np.ndarray,
    sk: np.ndarray,
    c2: np.ndarray | None = None,
) -> np.ndarray:
    """spec 정의로 µ′ 256 비트를 직접 계산 (보드 응답과 round-trip 검증용).

    µ′_i = ⌊ (t/p)·⟨c1, s⟩_i + (t/p′)·c2_i ⌉ mod t

    R = Z_q[X]/(X^n+1) 안 곱:
        ⟨c1, s⟩(X) = Σ_m c1[m](X) · sk[m](X)  mod (X^n + 1)

    coefficient i:
        ⟨c1, s⟩_i = Σ_(m, l) c1[m, l] · sk[m]_((i-l) mod n) · sign_l(i)
    여기서 sign_l(i) = +1 if i ≥ l else -1 (anticyclic wrap 부호).

    반환: shape (n,), dtype int64, 값 ∈ [0, t).

    사용:
        sk = unpack_sx(board_X_response)  # ternary
        c1 = build_multi_term_c1(params, {...}).c1
        mu_predicted = predict_mu_prime(params, c1, sk)
        # board: 'I' inject + 'Z' indcpa_dec → mu_board (32B → 256 bits)
        # 합격: np.array_equal(mu_predicted, mu_board)
    """
    if c1.shape != (params.module_rank, params.n):
        raise ValueError(f"c1.shape={c1.shape} != ({params.module_rank}, {params.n})")
    if sk.shape != (params.module_rank, params.n):
        raise ValueError(f"sk.shape={sk.shape} != ({params.module_rank}, {params.n})")

    n = params.n
    inner = np.zeros(n, dtype=np.int64)

    for m in range(params.module_rank):
        # 일반 다항식 곱 (Z 위) → negacyclic reduce.
        full = np.convolve(
            c1[m].astype(np.int64),
            sk[m].astype(np.int64),
        )
        # full.shape = (2n-1,). Reduce: result[i] = full[i] - full[i+n] for i<n.
        # full[i+n] 가 존재하는 i 범위: i < n-1.
        head = full[:n].copy()
        tail = np.zeros(n, dtype=np.int64)
        tail_len = full.size - n  # = n - 1
        if tail_len > 0:
            tail[:tail_len] = full[n:n + tail_len]
        inner += head - tail

    # mod q 후 signed range (-q/2, q/2] 로 변환 — round 정의가 signed 대상.
    inner_mod = inner % params.q
    half = params.q // 2
    inner_signed = ((inner_mod + half) % params.q) - half

    if c2 is None:
        c2 = np.zeros(n, dtype=np.int64)
    else:
        if c2.shape != (n,):
            raise ValueError(f"c2.shape={c2.shape} != ({n},)")
        c2 = np.asarray(c2, dtype=np.int64) % params.p2

    # µ′_i = round_half_up( (t/p)·inner_signed + (t/p′)·c2 ) mod t
    #      = floor( (numerator) / (2·p·p′) ) mod t
    # numerator = 2·t·(p′·inner + p·c2) + p·p′
    t, p, p2 = params.t, params.p, params.p2
    numerator = 2 * t * (inner_signed * p2 + c2 * p) + p * p2
    denom = 2 * p * p2
    rounded = numerator // denom
    return (rounded % t).astype(np.int64)


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
