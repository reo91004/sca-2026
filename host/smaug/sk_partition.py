"""E3a — sk-known partition table 생성기 (host-only, 보드 없음).

핵심 질문:

    chosen ciphertext c1 = α·X^j (단항, c2=0) 를 SMAUG-T smaug1 의 PKE.Dec
    에 넣으면, 출력 메시지 µ′_i 가 비밀 계수 s_{(i-j) mod n} ∈ {-1, 0, +1}
    의 *어떤* 클래스 함수가 되는가?

이걸 host 에서 *spec 정의를 그대로 numpy 로* 돌려 답한다.

수식 (smaug1):

    ⟨c1, s⟩ = α · X^j · s   (R_q 안 anticyclic 곱셈)
    µ′_i    = ⌊ (t/p) · ⟨c1, s⟩_i + (t/p′) · c2_i ⌉ mod t
            = ⌊ (1/128) · α · sgn(i-j) · s_{(i-j) mod n} ⌉ mod 2   (c2 = 0)

여기서 sgn(i-j) 는 anticyclic wrap 부호: i ≥ j → +, 아니면 −.

본 모듈의 출력은 다음 형태의 partition matrix:

    P[α, s] ∈ {0, 1} ⊂ Z_t

행 = α ∈ U_p (smaug1 fixed-point set, 256 개), 열 = s ∈ {-1, 0, +1}.
P[α, s] 가 µ′_i 의 *예측값* 이다. 이 행렬에서:
  - α 행이 (0, _, _) ↔ (s=-1) 이 다른 두 클래스와 분리되는지
  - separability index = #(분리되는 α) / 256
를 통해 message-only chosen-CT 가 *support 만* 주는지 *sign 까지* 주는지
사전에 numeric 으로 답할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import fixed_point_set
from .params import SmaugParams


@dataclass(frozen=True)
class PartitionStats:
    """E3a 의 핵심 결과 요약."""

    alphas: np.ndarray       # (M,) — sweep 에 쓴 α 값들 (정렬됨)
    classes: np.ndarray      # (3,) — [-1, 0, +1] 고정
    matrix: np.ndarray       # (M, 3) ∈ {0, 1} — P[α_idx, s_idx]
    fully_separating_idx: np.ndarray  # 세 클래스 모두 다른 µ′ 를 만드는 α 인덱스
    sign_separating_idx: np.ndarray   # +1/-1 이 갈리는 α 인덱스 (s=0 이 어디 붙든 무관)
    support_only_idx: np.ndarray      # +1/-1 같고 0 만 다른 α — support oracle

    def summary(self) -> str:
        m = self.matrix.shape[0]
        return (
            f"partition over {m} α values: "
            f"fully_separating={self.fully_separating_idx.size} "
            f"sign_separating={self.sign_separating_idx.size} "
            f"support_only={self.support_only_idx.size} "
            f"trivial={m - (self.fully_separating_idx.size + self.support_only_idx.size)}"
        )


def predict_mu_prime_bit(
    params: SmaugParams,
    alpha: int,
    secret_value: int,
    *,
    allow_negative_alpha: bool = True,
) -> int:
    """단일 (α, s) 에 대한 µ′_i ∈ {0, 1} 을 spec 정의로 직접 계산.

    µ′_i = ⌊ (t/p) · α · s ⌉ mod t,  c2 = 0 가정.
    α 는 *signed effective value* 로 받는다 (anticyclic wrap 의 -α 도 다룸).
    """
    if secret_value not in (-1, 0, +1):
        raise ValueError(f"secret_value must be -1/0/+1, got {secret_value}")
    if not allow_negative_alpha and alpha < 0:
        raise ValueError("alpha < 0 disabled")
    t = params.t
    p = params.p
    # ⌊ x ⌉ = floor(x + 0.5)  (round-half-up)
    raw = alpha * secret_value
    # spec: µ′ ∈ R_t, 즉 [0, t) 로 정규화. mod t 후 [0, t) 로.
    # round-half-up: floor((2*t*x + p) / (2*p))  to integer, then mod t.
    rounded = (2 * t * raw + p) // (2 * p) - (1 if (2 * t * raw + p) < 0 and (2 * t * raw + p) % (2 * p) != 0 else 0)
    # numpy floor 가 음수에서 -inf 방향이라 위 식이 정확. 단순화:
    # 정수 라운드: round_half_up(x) = floor(x + 0.5). 우리는 x = t·raw / p.
    # 수치 안전을 위해 분수 비교로 다시:
    num = 2 * t * raw + p   # = 2p·(t·raw/p + 0.5)
    den = 2 * p
    # floor division — Python // 는 항상 floor (음수도 −∞ 방향) 이라 OK.
    rounded = num // den
    return int(rounded % t)


def build_partition_table(
    params: SmaugParams,
    *,
    use_negative_alpha: bool = False,
) -> PartitionStats:
    """모든 α ∈ U_p 에 대해 (s=-1, 0, +1) 분류표 작성.

    use_negative_alpha=True 면 anticyclic wrap 으로 발생하는 -α 도 함께
    sweep — 실제 c1 = α·X^j 가 만든 ⟨c1,s⟩ 의 i-번째 계수가 +α·s 또는
    -α·s 가 되는 두 위치를 모두 다루기 위해.
    """
    Up = fixed_point_set(params.log_p, params.log_q)  # (P,) — smaug1: 256 개
    if use_negative_alpha:
        # ±α 는 mod q 로 표현됨. -α ≡ q-α. fixed-point set 의 q-α 는 다른
        # alpha 가 아닌 같은 set 의 wrap. 즉 효과는 sign flip 만.
        # 단순히 "sign 두 방향" 을 따로 검사하려면 effective_alpha = ±α 로
        # 두 set 합집합을 다룬다 (signed integer 평가).
        alphas = np.concatenate([Up, -Up[Up != 0]])
        alphas = np.sort(np.unique(alphas))
    else:
        alphas = Up.copy()

    matrix = np.zeros((alphas.size, 3), dtype=np.int8)
    for ai, a in enumerate(alphas.tolist()):
        for si, s in enumerate((-1, 0, 1)):
            matrix[ai, si] = predict_mu_prime_bit(params, int(a), int(s))

    # 분류 카테고리:
    full_sep = []
    sign_sep = []
    support_only = []
    for ai in range(alphas.size):
        row = tuple(int(x) for x in matrix[ai])
        # set 사이즈로 분류 클래스 수.
        unique = set(row)
        if len(unique) == 3:
            full_sep.append(ai)
        elif row[0] != row[2]:
            # +1 과 -1 이 다름 → sign 분리. (0 이 둘 중 어디든 붙든)
            sign_sep.append(ai)
        elif row[0] == row[2] and row[1] != row[0]:
            # ±1 같고 0 만 다름 → support oracle (zero/nonzero).
            support_only.append(ai)
        # else: 셋 다 같음 → trivial (정보 없음).

    return PartitionStats(
        alphas=alphas,
        classes=np.array([-1, 0, 1], dtype=np.int8),
        matrix=matrix,
        fully_separating_idx=np.array(full_sep, dtype=np.int64),
        sign_separating_idx=np.array(sign_sep, dtype=np.int64),
        support_only_idx=np.array(support_only, dtype=np.int64),
    )


def best_alphas_for_full_separation(
    stats: PartitionStats,
    n: int = 5,
) -> list[tuple[int, tuple[int, int, int]]]:
    """fully-separating α 중 처음 n 개를 (alpha, (µ_-1, µ_0, µ_+1)) 로."""
    out = []
    for ai in stats.fully_separating_idx[:n]:
        row = tuple(int(x) for x in stats.matrix[ai])
        out.append((int(stats.alphas[ai]), row))
    return out


@dataclass(frozen=True)
class OraclePair:
    """두 chosen-CT 의 결합으로 ternary 분류를 만드는 α 쌍.

    SMAUG-T smaug1 의 partition table 분석으로 발견된 결정적 결과:
      α ∈ {64, 320, 576, 832} → (µ_-1, µ_0, µ_+1) = (0, 0, 1) → +1 detector
      α ∈ {192, 448, 704, 960} → (1, 0, 0)                   → -1 detector

    두 oracle 의 join (각 secret 위치에서 두 chosen-CT 를 모두 dec) 으로:
      (0, 0) → s = 0 확정       ← 두 detector 모두 침묵
      (1, 0) → s = +1 확정      ← +1 detector 만 발화
      (0, 1) → s = -1 확정      ← -1 detector 만 발화
      (1, 1) → 불가능           ← s 가 동시에 +1 과 -1 일 수 없음
    """

    alpha_pos: int     # +1 detector α
    alpha_neg: int     # -1 detector α
    pos_row: tuple[int, int, int]
    neg_row: tuple[int, int, int]

    def predict_pair(self, s: int) -> tuple[int, int]:
        """단일 secret value s ∈ {-1, 0, +1} 에 대해 두 oracle 의 응답."""
        if s not in (-1, 0, 1):
            raise ValueError(s)
        col = {-1: 0, 0: 1, 1: 2}[s]
        return (self.pos_row[col], self.neg_row[col])


def predict_mu_prime_pair(
    params: SmaugParams,
    alpha: int,
    s_a: int,
    s_b: int,
    *,
    sign_a: int = +1,
    sign_b: int = +1,
) -> int:
    """Phase H — 2-term c1 = α·(sign_a·X^l + sign_b·X^k) 일 때 µ′_i ∈ {0, 1}.

    s_a = sk[m]_((i-l) mod n), s_b = sk[m']_((i-k) mod n) 의 effective ternary
    값. sign_a/b 는 anticyclic wrap 의 부호 (l 또는 k 가 i 보다 크면 -1).
    같은 component 두 항이든 다른 component 든 같은 식으로 동작 (host 가
    어떤 (m, m') 에서 가져왔는지에 따라 호출 측이 결정).

    µ′_i = round_half_up( (t/p) · α · (sign_a·s_a + sign_b·s_b) ) mod t
    """
    if s_a not in (-1, 0, 1) or s_b not in (-1, 0, 1):
        raise ValueError(f"s_a, s_b must be -1/0/+1, got ({s_a}, {s_b})")
    if sign_a not in (-1, +1) or sign_b not in (-1, +1):
        raise ValueError(f"sign_a, sign_b must be ±1, got ({sign_a}, {sign_b})")
    t, p = params.t, params.p
    raw = int(alpha) * (sign_a * s_a + sign_b * s_b)
    num = 2 * t * raw + p
    den = 2 * p
    return int((num // den) % t)


def build_partition_table_2term(
    params: SmaugParams,
    *,
    sign_a: int = +1,
    sign_b: int = +1,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase H — 2-term partition table.

    각 α ∈ U_p 에 대해 9 outcomes (s_a × s_b ∈ {-1, 0, +1}^2) 의 µ′ bit.

    Returns:
        alphas (M,): U_p (smaug1: 256 개)
        matrix (M, 9): row 순서 = lexicographic on (s_a, s_b):
            col 0 = (-1, -1), 1 = (-1, 0), 2 = (-1, +1),
            col 3 = ( 0, -1), 4 = ( 0, 0), 5 = ( 0, +1),
            col 6 = (+1, -1), 7 = (+1, 0), 8 = (+1, +1)
        값 ∈ {0, 1}.
    """
    Up = fixed_point_set(params.log_p, params.log_q)
    alphas = Up.copy()
    pairs = [(sa, sb) for sa in (-1, 0, 1) for sb in (-1, 0, 1)]
    matrix = np.zeros((alphas.size, len(pairs)), dtype=np.int8)
    for ai, a in enumerate(alphas.tolist()):
        for ci, (sa, sb) in enumerate(pairs):
            matrix[ai, ci] = predict_mu_prime_pair(
                params, int(a), int(sa), int(sb),
                sign_a=sign_a, sign_b=sign_b,
            )
    return alphas, matrix


def classify_2term_alphas(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """2-term matrix 의 행을 *binary outcome split* 카테고리로 분류.

    matrix shape = (M, 9), 값 ∈ {0, 1}. n_ones[ai] = matrix[ai].sum() ∈ [0, 9].

    카테고리:
      "split_k_of_9" — 9 outcomes 중 k 개가 µ′=1 (나머지 9-k 가 0).

    high-entropy split (4_of_9 또는 5_of_9) 가 정보 함량 가장 큼. 1_of_9 또는
    8_of_9 는 specific (s_a, s_b) detector — query 효율적으로 사용 가능.
    0_of_9, 9_of_9 는 trivial (모든 (s_a, s_b) 가 같은 µ′ — α 자체가 정보 없음).
    """
    if matrix.ndim != 2 or matrix.shape[1] != 9:
        raise ValueError(f"expected (M, 9) shape, got {matrix.shape}")
    n_ones = matrix.sum(axis=1)
    cats: dict[str, np.ndarray] = {}
    for k in range(10):
        cats[f"split_{k}_of_9"] = np.where(n_ones == k)[0].astype(np.int64)
    return cats


def find_oracle_pairs(stats: PartitionStats) -> list[OraclePair]:
    """ternary 분류용 (α_pos, α_neg) 쌍을 모두 찾는다.

    조건:
      pos_row = (0, 0, 1)  AND  neg_row = (1, 0, 0)
    smaug1 에서는 (4 × 4) = 16 개 쌍이 나온다.
    """
    pos_target = (0, 0, 1)
    neg_target = (1, 0, 0)
    pos_alphas: list[int] = []
    neg_alphas: list[int] = []
    for ai in stats.sign_separating_idx:
        row = tuple(int(x) for x in stats.matrix[ai])
        a = int(stats.alphas[ai])
        if row == pos_target:
            pos_alphas.append(a)
        elif row == neg_target:
            neg_alphas.append(a)

    pairs: list[OraclePair] = []
    for ap in pos_alphas:
        for an in neg_alphas:
            pairs.append(OraclePair(
                alpha_pos=ap, alpha_neg=an,
                pos_row=pos_target, neg_row=neg_target,
            ))
    return pairs
