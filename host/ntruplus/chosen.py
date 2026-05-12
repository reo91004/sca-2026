"""Chosen-ciphertext generators for NTRU+768 SCA.

Two families used in the attack plan:

  * `selected_lane(lane, gamma, slot=0)` — main path. NTT-domain ciphertext
    with a single lane active. After board's `poly_basemul(&m1, &c, &f)`,
    only `m1[4*lane + slot]` carries the secret leak `γ * f_ntt[4*lane+slot]`.
    All other m1 lanes are deterministic functions of γ alone.

  * `monomial_pair(j, alpha, i, beta)` — fallback path. NTT-domain c with two
    monomials (α at index j, β at index i). Used by the coefficient-domain
    threshold sweep in Phase 5 — note the indices are still NTT-domain because
    NTRU+ ciphertexts are stored in NTT.

Public-design parity rule: every CT this module emits is fully determined by
its public parameters; secret-shuffle nulls in Phase 3 must work even when
the same byte sequence is sent.
"""

from __future__ import annotations

import numpy as np

from .codec import to_bytes
from .params import D, N, N_LANES, Q


def _validate_gamma(gamma: int) -> int:
    """Map any integer γ to centred residue (-q/2, q/2]. Both unsigned [0, q)
    and signed centred forms are accepted; γ ≡ q is rejected since it equals 0
    and that's a degenerate chosen-CT we never want to send by accident."""
    g = int(gamma) % Q
    if g == 0:
        raise ValueError(f"gamma must be ≢ 0 (mod q={Q}); got {gamma}")
    if g > Q // 2:
        g -= Q
    return g


def selected_lane(lane: int, gamma: int, slot: int = 0) -> tuple[bytes, np.ndarray]:
    """Return (ct_bytes, c_ntt) where c_ntt has exactly one active coefficient.

    Args:
        lane:  index in [0, N_LANES)  — chooses the 4-coef block.
        gamma: chosen NTT-domain coefficient value, centred residue.
        slot:  position within the lane (0..D-1). Default 0.

    The active coefficient is at NTT index `4*lane + slot`. Board basemul
    will leave m1[lane] = (γ at slot, …) shifted by f_ntt[lane] in the
    polynomial ring Zq[X]/(X^4 - ζ_lane).
    """
    if not 0 <= lane < N_LANES:
        raise ValueError(f"lane {lane} out of range [0, {N_LANES})")
    if not 0 <= slot < D:
        raise ValueError(f"slot {slot} out of range [0, {D})")
    g = _validate_gamma(gamma)

    c = np.zeros(N, dtype=np.int16)
    c[D * lane + slot] = g
    return to_bytes(c), c


def monomial_pair(j: int, alpha: int, i: int, beta: int) -> tuple[bytes, np.ndarray]:
    """NTT-domain ciphertext with two non-zero coefficients.

    Used by Phase 5 fallback as a smaller-design analogue of SMAUG-T's
    coefficient-domain `c = α X^j + β X^i`. Here j, i are indices in the
    NTT-domain layout; their behaviour through `basemul` is *not* a simple
    monomial in coefficient domain, but the same chosen-bit calibration
    methodology (signed local Δ) applies.
    """
    if not 0 <= j < N or not 0 <= i < N:
        raise ValueError("indices out of range")
    if i == j:
        raise ValueError("i == j; use selected_lane()")
    a = _validate_gamma(alpha)
    b = _validate_gamma(beta)

    c = np.zeros(N, dtype=np.int16)
    c[j] = a
    c[i] = b
    return to_bytes(c), c
