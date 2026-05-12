"""Host-side NTRU+ helpers for chosen-CT SCA on STM32F4 firmware.

Mirrors upstream `Reference_Implementation/NTRU+768/` (KpqC final, commit
`621c667`). The local archive `lib/crypto_kem/ntruplus768.a` has zetas[192]
binary-identical to upstream; this package is the host-side counterpart.

Modules:
    params  — N, q, R, RINV, RSQ, QINV, OMEGA, ... and zetas[192]
    codec   — poly_tobytes / poly_frombytes (12-bit packing)
    ntt     — barrett_reduce, montgomery_reduce, ntt, invntt, basemul
    chosen  — selected-lane CT generator (NTT-domain, then poly_tobytes)
    sk      — sk parsing helpers (f, hinv, h_pk_seed)

Conventions:
    All polynomials carried as numpy int16 arrays of shape (N,). NTT-domain
    polynomials use the 4-radix layout described by upstream `ntt()` —
    coefficients [4k .. 4k+3] are an element of Zq[X]/(X^4 - zeta_i) where
    zeta_i = zetas[96+(k>>1)] · (-1)^(k&1).
"""

from . import params  # noqa: F401
from . import codec   # noqa: F401
from . import ntt     # noqa: F401
from . import chosen  # noqa: F401
from . import sk      # noqa: F401
