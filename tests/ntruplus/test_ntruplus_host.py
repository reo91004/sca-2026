#!/usr/bin/env python3
"""Round-trip tests for the host/ntruplus package.

These run with no board — they verify that our Python port of upstream
`Reference_Implementation/NTRU+768/{poly.c, ntt.c}` is internally consistent.
Board ↔ host parity is checked separately by tests/smoke_ntruplus.py once
the board is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus import codec, ntt as nttmod  # noqa: E402
from ntruplus.params import N, Q, ZETAS  # noqa: E402
from ntruplus.chosen import selected_lane  # noqa: E402


def test_zetas_table_size() -> None:
    assert ZETAS.shape == (192,)


def test_codec_roundtrip_random() -> None:
    rng = np.random.default_rng(2026_05_08)
    for _ in range(8):
        a = rng.integers(-(Q // 2), Q // 2 + 1, size=N, dtype=np.int16)
        buf = codec.to_bytes(a)
        b = codec.from_bytes(buf)
        b_centered = codec.center(b)
        assert np.array_equal(a % Q, b % Q), "raw mod-q roundtrip failed"
        assert np.array_equal(b_centered, codec.center(a)), "centered mismatch"
    print("[OK] codec roundtrip — 8 random polys")


def test_codec_specific_vectors() -> None:
    # All-zero
    z = np.zeros(N, dtype=np.int16)
    assert codec.from_bytes(codec.to_bytes(z)).tolist() == z.tolist()
    # All q-1 (max valid)
    m = np.full(N, Q - 1, dtype=np.int16)
    assert np.array_equal(codec.from_bytes(codec.to_bytes(m)), m)
    # Negative max ((-(q-1)/2)) maps to (q-1)/2 + 1
    n_min = -((Q - 1) // 2)
    arr = np.full(N, n_min, dtype=np.int16)
    out = codec.from_bytes(codec.to_bytes(arr))
    assert np.all(out == (n_min % Q))
    print("[OK] codec edge cases")


def test_ntt_invntt_identity() -> None:
    rng = np.random.default_rng(2026_05_08)
    a = rng.integers(-(Q // 2), Q // 2 + 1, size=N, dtype=np.int16)
    a_ntt = nttmod.ntt(a)
    a_back = nttmod.invntt(a_ntt)
    # invntt produces values in centred residue (after the final fqmul);
    # mod-q equality is the right comparison.
    diff = (a_back.astype(np.int32) - a.astype(np.int32)) % Q
    assert np.all(diff == 0), f"NTT∘INTT mismatch, max diff={np.max(diff)}"
    print("[OK] ntt ∘ invntt = identity (mod q)")


def test_basemul_matches_naive() -> None:
    """basemul(NTT(a), NTT(b)) ≡ NTT(a · b in R_q).  Use a small test."""
    rng = np.random.default_rng(2026_05_08)
    # naive_poly_mul is O(N^2); skip for full N=768. Use only a sparse test:
    a = np.zeros(N, dtype=np.int16); b = np.zeros(N, dtype=np.int16)
    # a = 1 + X^5 - X^100,  b = 2 - X^7
    a[0] = 1; a[5] = 1; a[100] = -1
    b[0] = 2; b[7] = -1
    naive = nttmod.naive_poly_mul(a, b)
    a_ntt = nttmod.ntt(a)
    b_ntt = nttmod.ntt(b)
    prod_ntt = nttmod.basemul(a_ntt, b_ntt)
    prod = nttmod.invntt(prod_ntt)
    diff = (prod.astype(np.int32) - naive.astype(np.int32)) % Q
    assert np.all(diff == 0), f"basemul vs naive mismatch, max diff={np.max(diff)}"
    print("[OK] basemul ∘ NTT == NTT ∘ naive_poly_mul")


def test_selected_lane_layout() -> None:
    # lane=3, slot=0, gamma=42 → c_ntt[12]=42, others 0
    buf, c = selected_lane(lane=3, gamma=42, slot=0)
    assert c[12] == 42
    nonzero = np.flatnonzero(c)
    assert nonzero.tolist() == [12]
    # round-trip via codec
    c2 = codec.center(codec.from_bytes(buf))
    assert np.array_equal(c2, c)
    # slot=2 puts γ at 4*lane+2
    _, c3 = selected_lane(lane=10, gamma=-100, slot=2)
    assert c3[42] == -100
    nonzero = np.flatnonzero(c3)
    assert nonzero.tolist() == [42]
    print("[OK] selected_lane layout + codec roundtrip")


def test_basemul_lane_isolation() -> None:
    """For a chosen-CT with single NTT-domain coefficient,
    basemul(c, f) leaves the active lane carrying γ·f and zeros elsewhere."""
    rng = np.random.default_rng(2026_05_08)
    f = rng.integers(-(Q // 2), Q // 2 + 1, size=N, dtype=np.int16)
    f_ntt = nttmod.ntt(f)

    lane = 7
    slot = 0
    gamma = 100
    _, c = selected_lane(lane=lane, gamma=gamma, slot=slot)

    m = nttmod.basemul(c, f_ntt)
    # m[4*lane + i] should equal (γ * f_ntt[4*lane + i]) mod q (for slot=0)
    for i in range(4):
        expected = (gamma * int(f_ntt[4 * lane + i])) % Q
        got = int(m[4 * lane + i]) % Q
        assert expected == got, f"lane[{i}] expected {expected}, got {got}"

    # All other lanes must be zero (mod q)
    nonzero = []
    for k in range(N):
        if 4 * lane <= k < 4 * lane + 4:
            continue
        if int(m[k]) % Q != 0:
            nonzero.append(k)
    # NOTE: basemul groups two lanes per zeta; the *paired* lane (8*pair + 1
    # vs 8*pair) is independent (uses -zeta), so isolation holds. Verify.
    assert not nonzero, f"unexpected nonzero outside active lane: {nonzero[:8]}"
    print(f"[OK] selected_lane isolation: lane={lane} slot={slot} γ={gamma}")


def main() -> int:
    test_zetas_table_size()
    test_codec_roundtrip_random()
    test_codec_specific_vectors()
    test_ntt_invntt_identity()
    test_basemul_matches_naive()
    test_selected_lane_layout()
    test_basemul_lane_isolation()
    print("\n[ALL PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
