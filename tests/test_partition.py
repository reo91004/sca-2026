"""E3a partition table 의 형식과 핵심 정합성."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.smaug import params, sk_partition  # noqa: E402


def test_predict_mu_prime_bit_basic_smaug1() -> None:
    p = params.SMAUG1
    # α=0 → 모든 s 에 대해 µ′=0
    for s in (-1, 0, 1):
        assert sk_partition.predict_mu_prime_bit(p, 0, s) == 0
    # α=128, s=+1 → ⌊128/128⌉=1 mod 2 → 1
    assert sk_partition.predict_mu_prime_bit(p, 128, 1) == 1
    # α=128, s=-1 → ⌊-128/128⌉=-1 mod 2 → 1
    assert sk_partition.predict_mu_prime_bit(p, 128, -1) == 1
    # α=128, s=0  → 0
    assert sk_partition.predict_mu_prime_bit(p, 128, 0) == 0
    # α=64, s=+1  → ⌊0.5⌉ = 1 (round-half-up) → 1 mod 2 = 1
    assert sk_partition.predict_mu_prime_bit(p, 64, 1) == 1
    # α=64, s=-1  → ⌊-0.5⌉ = 0 (round-half-up) → 0
    assert sk_partition.predict_mu_prime_bit(p, 64, -1) == 0


def test_partition_table_smaug1_shape() -> None:
    p = params.SMAUG1
    stats = sk_partition.build_partition_table(p)
    # smaug1: U_p 사이즈 = 256
    assert stats.alphas.size == 256
    assert stats.matrix.shape == (256, 3)
    assert tuple(int(x) for x in stats.classes) == (-1, 0, 1)
    # 모든 항목 ∈ {0, 1}
    assert set(stats.matrix.ravel().tolist()) <= {0, 1}


def test_partition_table_alpha0_row_is_all_zero() -> None:
    p = params.SMAUG1
    stats = sk_partition.build_partition_table(p)
    a0_idx = int(np.where(stats.alphas == 0)[0][0])
    assert (stats.matrix[a0_idx] == 0).all(), "α=0 행이 모두 0 이어야 함"


def test_partition_has_some_full_separation_or_signal() -> None:
    """smaug1 에서 적어도 어떤 α 가 sign 분리를 만든다는 사실 numeric 으로 확인.

    이게 통과하면 *message-only* chosen-CT 로 sign 회복 가능성이 0 이 아님.
    실패하면 모든 α 에서 ±1 이 같은 클래스 → message-only 로는 support 만.
    """
    p = params.SMAUG1
    stats = sk_partition.build_partition_table(p)
    sign_or_full = stats.fully_separating_idx.size + stats.sign_separating_idx.size
    assert sign_or_full > 0, (
        "어떤 α 도 +1/-1 을 분리 못 함 — message-only sign recovery 수학적으로 불가능. "
        "이 결과는 docs/idea.md §정합성 점검 노트 에 사실관계로 추가 필요."
    )


def test_partition_summary_runs() -> None:
    p = params.SMAUG1
    stats = sk_partition.build_partition_table(p)
    s = stats.summary()
    assert "fully_separating" in s
    # 디버그 시 사람이 보고 싶은 형태 확인
    print(s)


def test_oracle_pairs_discoverable_smaug1() -> None:
    """smaug1 에서 (α=64, α=192) 류의 oracle pair 가 ternary 분류를 만든다.

    2 chosen-CT join 으로 단일 secret 의 -1/0/+1 완전 분류:
       (0, 0) → s = 0
       (1, 0) → s = +1
       (0, 1) → s = -1
       (1, 1) → 불가능 (모순)
    """
    p = params.SMAUG1
    stats = sk_partition.build_partition_table(p)
    pairs = sk_partition.find_oracle_pairs(stats)
    assert len(pairs) > 0, "smaug1 에서 oracle pair 가 0 개 — partition 가설 깨짐"

    # 첫 pair 의 정확성: 모든 s ∈ {-1, 0, +1} 에 대해 join 이 unique.
    pr = pairs[0]
    pred_for_neg = pr.predict_pair(-1)
    pred_for_zero = pr.predict_pair(0)
    pred_for_pos = pr.predict_pair(1)
    assert pred_for_neg == (0, 1), pred_for_neg
    assert pred_for_zero == (0, 0), pred_for_zero
    assert pred_for_pos == (1, 0), pred_for_pos
    # (1, 1) 은 어떤 s 도 만들지 않음 — 모순 응답이라는 제약.
    seen = {pr.predict_pair(s) for s in (-1, 0, 1)}
    assert (1, 1) not in seen


def test_oracle_pair_count_smaug1() -> None:
    """R_p 직렬화 도메인에서는 대표 oracle pair (64, 192) 한 쌍."""
    p = params.SMAUG1
    stats = sk_partition.build_partition_table(p)
    pairs = sk_partition.find_oracle_pairs(stats)
    assert len(pairs) == 1, f"oracle pair count = {len(pairs)}, 기대 1"
    assert (pairs[0].alpha_pos, pairs[0].alpha_neg) == (64, 192)


def test_predict_mu_prime_pair_basic() -> None:
    """2-term predict — alpha=0 모두 0, alpha=128 / s_a + s_b = ±1 → 1."""
    p = params.SMAUG1
    # alpha=0
    for sa in (-1, 0, 1):
        for sb in (-1, 0, 1):
            assert sk_partition.predict_mu_prime_pair(p, 0, sa, sb) == 0
    # alpha=128: ⌊128·(s_a + s_b)/128⌉ mod 2.
    # s_a + s_b ∈ {-2, -1, 0, +1, +2} → bit = (s_a+s_b) mod 2 = |s_a+s_b| mod 2.
    cases = [
        (-1, -1, 0),  # -2 mod 2 = 0
        (-1, 0, 1),   # -1 mod 2 = 1
        (-1, 1, 0),   # 0
        (0, 0, 0),
        (0, 1, 1),
        (1, 1, 0),    # 2 mod 2 = 0
    ]
    for sa, sb, expected in cases:
        got = sk_partition.predict_mu_prime_pair(p, 128, sa, sb)
        assert got == expected, f"alpha=128, ({sa},{sb}): got {got}, expected {expected}"


def test_predict_mu_prime_pair_sign_flip() -> None:
    """sign_a=-1 은 s_a 에 대한 부호 반전과 동치."""
    p = params.SMAUG1
    for alpha in (64, 128, 192):
        for sa, sb in [(-1, 0), (1, -1), (0, 1)]:
            base = sk_partition.predict_mu_prime_pair(p, alpha, sa, sb,
                                                     sign_a=+1, sign_b=+1)
            flipped = sk_partition.predict_mu_prime_pair(p, alpha, -sa, sb,
                                                        sign_a=+1, sign_b=+1)
            via_sign = sk_partition.predict_mu_prime_pair(p, alpha, sa, sb,
                                                         sign_a=-1, sign_b=+1)
            assert flipped == via_sign, (
                f"alpha={alpha} (sa,sb)=({sa},{sb}): flipped={flipped} "
                f"via_sign={via_sign}"
            )
            # base 와 via_sign 은 일반적으로 다름 — 일관성 확인용
            del base


def test_partition_table_2term_shape_smaug1() -> None:
    p = params.SMAUG1
    alphas, matrix = sk_partition.build_partition_table_2term(p)
    assert alphas.size == 256, f"|U_p| smaug1 = 256, got {alphas.size}"
    assert matrix.shape == (256, 9)
    assert set(matrix.ravel().tolist()) <= {0, 1}


def test_partition_table_2term_alpha0_all_zero() -> None:
    p = params.SMAUG1
    alphas, matrix = sk_partition.build_partition_table_2term(p)
    a0 = int(np.where(alphas == 0)[0][0])
    assert (matrix[a0] == 0).all()


def test_partition_table_2term_central_zero_pair_is_zero() -> None:
    """모든 α 에서 (s_a, s_b) = (0, 0) → µ′ = 0 (raw=0)."""
    p = params.SMAUG1
    alphas, matrix = sk_partition.build_partition_table_2term(p)
    # col index = (s_a, s_b)=(0,0) → 1*3+1=4
    assert (matrix[:, 4] == 0).all()


def test_classify_2term_split_categories_partition_full() -> None:
    """모든 α 가 어떤 split_k 카테고리에 정확히 1개씩 들어간다."""
    p = params.SMAUG1
    _, matrix = sk_partition.build_partition_table_2term(p)
    cats = sk_partition.classify_2term_alphas(matrix)
    total = sum(idx.size for idx in cats.values())
    assert total == matrix.shape[0], f"sum split sizes = {total}, expected 256"
    # 한 α 가 두 카테고리에 들어가지 않음
    seen = set()
    for idx_arr in cats.values():
        for ai in idx_arr.tolist():
            assert ai not in seen, f"alpha index {ai} in multiple splits"
            seen.add(ai)


def test_2term_high_entropy_alphas_exist() -> None:
    """smaug1 에서 4_of_9 또는 5_of_9 split 의 high-entropy α 가 *최소 1개* 존재.

    즉 한 chosen-CT 의 µ′ outcome 이 9 (s_a, s_b) pair 중 4 또는 5 에서 µ′=1 로
    가는 α 가 있어 entropy 가 거의 최대 (log2 9 ≈ 3.17 의 binary 분포 ≈ 0.99).
    이게 H 의 multi-term leak amp 의 numeric 증거.
    """
    p = params.SMAUG1
    _, matrix = sk_partition.build_partition_table_2term(p)
    cats = sk_partition.classify_2term_alphas(matrix)
    high_entropy = cats["split_4_of_9"].size + cats["split_5_of_9"].size
    assert high_entropy > 0, (
        "smaug1 2-term partition 에서 high-entropy split 0 — "
        "multi-term 가 정보 추가하지 않음 (가설 깨짐)"
    )


if __name__ == "__main__":
    fns = sorted(n for n in globals() if n.startswith("test_"))
    for n in fns:
        print(f"[RUN] {n}")
        globals()[n]()
        print(f"[OK ] {n}")
    print(f"[OK ] {len(fns)} tests")
