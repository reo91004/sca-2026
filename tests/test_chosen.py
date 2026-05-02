"""chosen-CT 빌더 + chunkify 의 모양/제약 검증."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.smaug import chosen, params  # noqa: E402


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


def test_monomial_c1_alpha_must_be_in_fixed_point_set() -> None:
    p = params.SMAUG1
    # alpha=4 는 U_p (= multiples of 4) 에 들어 있음 → 통과
    ct = chosen.build_monomial_c1(p, component=0, coef_idx=10, alpha=4)
    assert ct.c1.shape == (p.module_rank, p.n)
    assert ct.c2.shape == (p.n,)
    nonzero = np.argwhere(ct.c1 != 0)
    assert nonzero.shape == (1, 2), f"단항 위치 1개여야 함, got {nonzero}"
    assert tuple(nonzero[0].tolist()) == (0, 10)
    assert int(ct.c1[0, 10]) == 4
    assert (ct.c2 == 0).all()

    # alpha=3 은 U_p 밖 → ValueError
    raised = False
    try:
        chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=3)
    except ValueError:
        raised = True
    assert raised, "fixed-point set 밖 alpha 가 ValueError 안 던짐"


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
