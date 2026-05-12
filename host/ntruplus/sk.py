"""Secret-key parsing helpers.

NTRU+ sk layout (upstream `crypto_kem_keypair_derand`):

    [0           .. POLYBYTES )         f_ntt        (12-bit packed)
    [POLYBYTES   .. 2*POLYBYTES)        hinv_ntt     (12-bit packed)
    [2*POLYBYTES .. +SYMBYTES )         hash_f(pk) = H(pk) seed for implicit reject

The secret polynomial f stored on disk is **already in NTT domain** — it is
written by the keypair routine after `poly_ntt(&f)`, so the bytes there encode
the NTT-domain coefficients (12-bit each, in [0, q)).

For our chosen-CT attack the only secret we want is `f_ntt`, since
`poly_basemul(&m1, &c, &f)` consumes it directly during decap.
"""

from __future__ import annotations

import numpy as np

from .codec import center, from_bytes
from .params import POLYBYTES, SECRETKEYBYTES, SYMBYTES


def parse_sk(sk_bytes: bytes) -> dict:
    """Split an sk blob into f_ntt (centred), hinv_ntt (centred), pk_seed (32B)."""
    if len(sk_bytes) != SECRETKEYBYTES:
        raise ValueError(f"sk size {len(sk_bytes)} != {SECRETKEYBYTES}")
    f_raw    = from_bytes(sk_bytes[:POLYBYTES])
    hinv_raw = from_bytes(sk_bytes[POLYBYTES:2 * POLYBYTES])
    pk_seed  = bytes(sk_bytes[2 * POLYBYTES:])
    assert len(pk_seed) == SYMBYTES
    return {
        "f_ntt":    center(f_raw),
        "hinv_ntt": center(hinv_raw),
        "pk_seed":  pk_seed,
    }


def f_ntt_unsigned(sk_bytes: bytes) -> np.ndarray:
    """Return f_ntt directly as the [0, q) byte values (no centring)."""
    return from_bytes(sk_bytes[:POLYBYTES])
