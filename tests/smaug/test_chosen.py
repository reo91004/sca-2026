"""chosen-CT 빌더 + chunkify 의 모양/제약 검증."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.smaug import chosen, params  # noqa: E402
from scripts.smaug.s2_z_lowdim_analyze import label_from_mu_bits  # noqa: E402
from scripts.smaug.s2_z_lowdim_analyze import label_from_fo_downstream  # noqa: E402
from scripts.smaug.s4_c2_pair_analyze import PairDataset, labels_for_pair  # noqa: E402


def test_mu_zero_is_all_zero_ct() -> None:
    p = params.SMAUG1
    ct = chosen.build_mu_constant(p, mu_bit=0)
    buf = ct.to_bytes()
    assert len(buf) == p.ciphertext_bytes
    assert all(b == 0 for b in buf)


def test_mu_one_unsupported_explicit() -> None:
    """mu_bit=1 c2-only 는 fixed-point 구조상 불가능 — NotImplementedError."""
    p = params.SMAUG1
    raised = False
    try:
        chosen.build_mu_constant(p, mu_bit=1)
    except NotImplementedError:
        raised = True
    assert raised, "mu_bit=1 build 가 NotImplementedError 를 안 냄 — 수학적 한계 누락"


def test_monomial_c1_alpha_must_be_in_rp_domain() -> None:
    p = params.SMAUG1
    # alpha 는 ciphertext R_p 계수이므로 0 <= alpha < p 이면 통과.
    ct = chosen.build_monomial_c1(p, component=0, coef_idx=10, alpha=3)
    assert ct.c1.shape == (p.module_rank, p.n)
    assert ct.c2.shape == (p.n,)
    nonzero = np.argwhere(ct.c1 != 0)
    assert nonzero.shape == (1, 2), f"단항 위치 1개여야 함, got {nonzero}"
    assert tuple(nonzero[0].tolist()) == (0, 10)
    assert int(ct.c1[0, 10]) == 3
    assert (ct.c2 == 0).all()

    # q-domain fixed-point 값 320은 R_p 밖이며, pack alias를 막기 위해 거부.
    raised = False
    try:
        chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=320)
    except ValueError:
        raised = True
    assert raised, "R_p 밖 alpha 가 ValueError 안 던짐"


def test_multi_term_c1_shape_and_placement() -> None:
    """build_multi_term_c1 가 dict 의 (m, l) 자리에 정확히 alpha 를 둔다."""
    p = params.SMAUG1
    coefs = {(0, 5): 64, (0, 50): 192, (1, 200): 64}
    ct = chosen.build_multi_term_c1(p, coefs)
    assert ct.c1.shape == (p.module_rank, p.n)
    assert ct.c2.shape == (p.n,)
    assert (ct.c2 == 0).all()
    # 명시된 자리 정확
    for (m, l), a in coefs.items():
        assert int(ct.c1[m, l]) == a, f"c1[{m},{l}] = {ct.c1[m, l]} != {a}"
    # 그 외 자리 0
    expected_nonzero = set(coefs.keys())
    for m in range(p.module_rank):
        for l in range(p.n):
            if (m, l) in expected_nonzero:
                continue
            assert int(ct.c1[m, l]) == 0


def test_multi_term_c1_rejects_invalid_alpha() -> None:
    p = params.SMAUG1
    raised = False
    try:
        chosen.build_multi_term_c1(p, {(0, 0): p.p})
    except ValueError:
        raised = True
    assert raised


def test_multi_term_c1_rejects_oor_index() -> None:
    p = params.SMAUG1
    for bad in [(p.module_rank, 0), (0, p.n), (-1, 0), (0, -1)]:
        raised = False
        try:
            chosen.build_multi_term_c1(p, {bad: 4})
        except (IndexError, ValueError):
            raised = True
        assert raised, f"bad index {bad} 통과해버림"


def test_combined_c1_cross_component() -> None:
    """build_combined_c1 가 coefs0 → c1[0], coefs1 → c1[1] 로 라우팅."""
    p = params.SMAUG1
    ct = chosen.build_combined_c1(p, {10: 64}, {50: 192})
    assert int(ct.c1[0, 10]) == 64
    assert int(ct.c1[1, 50]) == 192
    # 다른 자리 모두 0
    nonzero = np.argwhere(ct.c1 != 0)
    assert nonzero.shape == (2, 2)


def test_combined_c1_equivalent_to_multi_term() -> None:
    p = params.SMAUG1
    a = chosen.build_combined_c1(p, {10: 64}, {50: 192})
    b = chosen.build_multi_term_c1(p, {(0, 10): 64, (1, 50): 192})
    assert np.array_equal(a.c1, b.c1)
    assert np.array_equal(a.c2, b.c2)


def _random_ternary_sk(p, seed: int) -> np.ndarray:
    """테스트용 ternary sk (HW 제약 없는 임의값)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=(p.module_rank, p.n)).astype(np.int64)


def test_mu_label_matches_full_predictor_for_component_design() -> None:
    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0x5204)
    terms = [(7, 64), (101, 192), (203, 128)]
    c2_alpha = 8
    ct = chosen.build_multi_term_c1(p, {(0, coef): alpha for coef, alpha in terms})
    ct.c2[:] = c2_alpha
    bits = chosen.predict_mu_prime(p, ct.c1, sk, ct.c2).astype(np.float64)

    assert np.array_equal(label_from_mu_bits(sk[0], terms, c2_alpha, "mu_bit"), bits)
    assert np.array_equal(
        label_from_mu_bits(sk[0], terms, c2_alpha, "mu_byte_hw"),
        bits.reshape(32, 8).sum(axis=1),
    )
    assert np.array_equal(
        label_from_mu_bits(sk[0], terms, c2_alpha, "mu_block16_hw"),
        bits.reshape(16, 16).sum(axis=1),
    )


def test_c2_pair_flip_labels_are_xor_of_mu_bits() -> None:
    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0xC2D1)
    terms = [(0, 128)]
    base_c2 = 0
    delta_c2 = 1
    b0 = label_from_mu_bits(sk[0], terms, base_c2, "mu_bit").astype(np.int8)
    b1 = label_from_mu_bits(sk[0], terms, delta_c2, "mu_bit").astype(np.int8)
    flip = np.bitwise_xor(b0, b1).reshape(32, 8).sum(axis=1)

    pds = PairDataset(
        xmean=np.zeros((1, 1, 2), dtype=np.float64),
        xvar=np.ones((1, 1, 2), dtype=np.float64),
        counts=np.ones((1, 1), dtype=np.float64),
        sks=sk[0][None, :],
        base_terms=[terms],
        base_c2=[base_c2],
        delta_c2=[delta_c2],
        pair_indices=[(0, 1)],
        pkfps=["test"],
    )
    assert np.array_equal(labels_for_pair(pds, "flip_byte_hw")[0, 0], flip)


def test_fo_downstream_labels_are_byte_hw_vectors() -> None:
    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0xF0)
    pk = bytes((i * 17 + 3) & 0xFF for i in range(p.public_key_bytes))
    terms = [(0, 128)]
    assert label_from_fo_downstream(sk[0], terms, 15, pk, "fo_kr0_byte_hw").shape == (32,)
    assert label_from_fo_downstream(sk[0], terms, 15, pk, "fo_kr1_byte_hw").shape == (32,)
    assert label_from_fo_downstream(sk[0], terms, 15, pk, "fo_kr64_byte_hw").shape == (64,)


def test_predict_mu_prime_monomial_matches_partition_predict() -> None:
    """단항 c1 = α·X^l 의 predict_mu_prime 가 partition predict_mu_prime_bit 와
    모든 i 에서 일치 (anticyclic wrap 부호 포함)."""
    from host.smaug import sk_partition

    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0xCAFE)
    alpha = 64
    l = 33
    ct = chosen.build_monomial_c1(p, component=0, coef_idx=l, alpha=alpha)
    mu_pred = chosen.predict_mu_prime(p, ct.c1, sk)
    assert mu_pred.shape == (p.n,)
    # i 별 비교: ⟨c1, s⟩_i = α · sign_l(i) · sk[0]_((i-l) mod n)
    for i in range(p.n):
        sign = +1 if i >= l else -1
        sval = int(sk[0, (i - l) % p.n])
        eff_alpha = alpha * sign
        bit_expected = sk_partition.predict_mu_prime_bit(p, eff_alpha, sval)
        assert int(mu_pred[i]) == bit_expected, (
            f"i={i} l={l} sign={sign} s={sval}: pred={mu_pred[i]} vs "
            f"predict_bit({eff_alpha}, {sval})={bit_expected}"
        )


def test_predict_mu_prime_constant_c1_leaks_all_secret_positions() -> None:
    """c1 = α (l=0 의 단항) → ⟨c1, s⟩_i = α · s[0]_i for all i (no wrap).

    우리 컨벤션 발견 (2026-05-02) 의 기본 케이스 — 단일 chosen-CT 가 256 비밀
    계수 모두를 µ′_0..255 로 동시에 leak.
    """
    from host.smaug import sk_partition

    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0xBEEF)
    alpha = 128
    ct = chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=alpha)
    mu_pred = chosen.predict_mu_prime(p, ct.c1, sk)
    for i in range(p.n):
        sval = int(sk[0, i])
        bit_expected = sk_partition.predict_mu_prime_bit(p, alpha, sval)
        assert int(mu_pred[i]) == bit_expected, f"i={i} s={sval}"


def test_predict_mu_prime_2term_matches_pair_no_wrap() -> None:
    """2-term c1 = α(X^l + X^k) 의 predict_mu_prime 가 predict_mu_prime_pair 와
    일치 (i ≥ max(l, k) 영역 — wrap 없음, sign_a=+1, sign_b=+1)."""
    from host.smaug import sk_partition

    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0xDEAD)
    alpha = 64
    l, k = 5, 50
    coefs = {(0, l): alpha, (0, k): alpha}
    ct = chosen.build_multi_term_c1(p, coefs)
    mu_pred = chosen.predict_mu_prime(p, ct.c1, sk)
    # i ≥ max(l, k) → 둘 다 wrap 없음 → sign_a = sign_b = +1
    for i in range(max(l, k), p.n):
        sa = int(sk[0, (i - l) % p.n])
        sb = int(sk[0, (i - k) % p.n])
        bit_expected = sk_partition.predict_mu_prime_pair(
            p, alpha, sa, sb, sign_a=+1, sign_b=+1,
        )
        assert int(mu_pred[i]) == bit_expected, (
            f"i={i} l={l} k={k} s_a={sa} s_b={sb}"
        )


def test_predict_mu_prime_2term_anticyclic_wrap_signs() -> None:
    """l > k > 0 인 2-term c1 = α(X^l + X^k) 의 wrap 영역 (i < k):
    sign_a = sign_b = -1 → µ′ = pair predict with sign_a=-1, sign_b=-1.

    부분 wrap 영역 (k ≤ i < l): sign_a=-1, sign_b=+1.
    """
    from host.smaug import sk_partition

    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0x1234)
    alpha = 128
    l, k = 100, 30
    ct = chosen.build_multi_term_c1(p, {(0, l): alpha, (0, k): alpha})
    mu_pred = chosen.predict_mu_prime(p, ct.c1, sk)
    for i in range(p.n):
        sa = int(sk[0, (i - l) % p.n])
        sb = int(sk[0, (i - k) % p.n])
        sign_a = +1 if i >= l else -1
        sign_b = +1 if i >= k else -1
        bit_expected = sk_partition.predict_mu_prime_pair(
            p, alpha, sa, sb, sign_a=sign_a, sign_b=sign_b,
        )
        assert int(mu_pred[i]) == bit_expected, (
            f"i={i} l={l} k={k} sign_a={sign_a} sign_b={sign_b} "
            f"s_a={sa} s_b={sb}: pred={mu_pred[i]} expected={bit_expected}"
        )


def test_predict_mu_prime_combined_cross_component() -> None:
    """c1[0] = α·X^l, c1[1] = α·X^k → ⟨c1, s⟩_i = α(sign_l·s[0]_(i-l) + sign_k·s[1]_(i-k))."""
    from host.smaug import sk_partition

    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0x5678)
    alpha = 64
    l, k = 5, 80
    ct = chosen.build_combined_c1(p, {l: alpha}, {k: alpha})
    mu_pred = chosen.predict_mu_prime(p, ct.c1, sk)
    for i in range(p.n):
        sa = int(sk[0, (i - l) % p.n])
        sb = int(sk[1, (i - k) % p.n])
        sign_a = +1 if i >= l else -1
        sign_b = +1 if i >= k else -1
        bit_expected = sk_partition.predict_mu_prime_pair(
            p, alpha, sa, sb, sign_a=sign_a, sign_b=sign_b,
        )
        assert int(mu_pred[i]) == bit_expected, f"i={i}"


def test_chunkify_smaug1_21_chunks() -> None:
    p = params.SMAUG1
    ct = chosen.build_mu_constant(p, mu_bit=0)
    chunks = chosen.chunkify(ct.to_bytes(), chunk_size=32)
    assert len(chunks) == 21
    indexes = [c[0] for c in chunks]
    assert indexes == list(range(21))
    for idx, data in chunks:
        assert len(data) == 32, f"chunk {idx} len={len(data)} != 32"


def test_chunkify_rejects_non_divisible() -> None:
    raised = False
    try:
        chosen.chunkify(b"x" * 100, chunk_size=32)
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = sorted(n for n in globals() if n.startswith("test_"))
    for n in fns:
        print(f"[RUN] {n}")
        globals()[n]()
        print(f"[OK ] {n}")
    print(f"[OK ] {len(fns)} tests")
