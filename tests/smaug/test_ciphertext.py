"""ct 직렬화 round-trip + 길이/형식 검증."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.smaug import ciphertext, params  # noqa: E402


def test_ct_zero_round_trip_smaug1() -> None:
    p = params.SMAUG1
    c1 = np.zeros((p.module_rank, p.n), dtype=np.int64)
    c2 = np.zeros(p.n, dtype=np.int64)
    buf = ciphertext.ct_to_bytes(p, c1, c2)
    assert len(buf) == p.ciphertext_bytes
    assert all(b == 0 for b in buf)
    c1b, c2b = ciphertext.ct_from_bytes(p, buf)
    assert np.array_equal(c1b, c1)
    assert np.array_equal(c2b, c2)


def test_ct_random_round_trip_smaug1() -> None:
    p = params.SMAUG1
    rng = np.random.default_rng(0)
    c1 = rng.integers(0, p.p, size=(p.module_rank, p.n), dtype=np.int64)
    c2 = rng.integers(0, p.p2, size=p.n, dtype=np.int64)
    buf = ciphertext.ct_to_bytes(p, c1, c2)
    assert len(buf) == p.ciphertext_bytes
    c1b, c2b = ciphertext.ct_from_bytes(p, buf)
    assert np.array_equal(c1b, c1)
    assert np.array_equal(c2b, c2)


def test_ct_length_invariant_smaug1() -> None:
    p = params.SMAUG1
    # CTPOLYVEC + CTPOLY2 = CIPHERTEXT
    assert p.ctpolyvec_bytes + p.ctpoly2_bytes == p.ciphertext_bytes
    # smaug1 구체 수치
    assert p.ctpolyvec_bytes == 512
    assert p.ctpoly2_bytes == 160
    assert p.ciphertext_bytes == 672


if __name__ == "__main__":
    fns = sorted(n for n in globals() if n.startswith("test_"))
    for n in fns:
        print(f"[RUN] {n}")
        globals()[n]()
        print(f"[OK ] {n}")
    print(f"[OK ] {len(fns)} tests")
