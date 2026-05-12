"""NTRU+768 parameters and the 192-entry Montgomery-form zeta table.

Source of truth: upstream `Reference_Implementation/NTRU+768/{params.h, ntt.c}`,
commit 621c667. Verified by extracting the .rodata section of
`lib/crypto_kem/ntruplus768.a:ntt.c.o` — all 192 entries match.
"""

from __future__ import annotations

import numpy as np


# ---- scheme parameters ---------------------------------------------------

ALGNAME = "ntruplus768"

N = 768
Q = 3457
D = 4               # NTT base ring degree (X^4 - zeta_i)

SYMBYTES  = 32
SSBYTES   = 32
POLYBYTES = 1152    # = 3 * N / 2  (12 bits per coefficient, packed pairs)

PUBLICKEYBYTES  = POLYBYTES                     # h_ntt only
SECRETKEYBYTES  = (POLYBYTES << 1) + SYMBYTES   # f_ntt | hinv_ntt | H(pk)
CIPHERTEXTBYTES = POLYBYTES


# ---- Montgomery / NTT constants ------------------------------------------

# All in centered representation, R = 2^16, q = 3457.
R            = -147   # 2^16 mod q   (centered)
RINV         = -682   # R^{-1} mod q
RSQ          = 867    # R^2 mod q
QINV         = 12929  # q^{-1} mod 2^16  (unsigned 16-bit; matches reduce.h)

OMEGA        = -886   # omega * R mod q                  (omega: 6-th root)
ZMINUSZ5INV  = -1665  # (z - z^5)^{-1} * R mod q,  z = zeta^((n/d)/6)
NINV         = -811   # (n/d)^{-1} * R mod q
TWO_NINV     = -1622  # 2 * (n/d)^{-1} * R mod q


# ---- twiddle table -------------------------------------------------------

# Montgomery form: zeta_i_real = zetas[i] * R^{-1} mod q.
# Layout used by ntt(): k=1 root for first split, k=2,3 for next two,
# k=4..63 for the 32-step radix-2 layers, k=64..95 unused by ntt() (used
# only by basemul indices 96+i for i = 0..N/8-1 = 0..95).
ZETAS: np.ndarray = np.array([
    -147, -1033, -682, -248, -708, 682, 1, -722,
    -723, -257, -1124, -867, -256, 1484, 1262, -1590,
    1611, 222, 1164, -1346, 1716, -1521, -357, 395,
    -455, 639, 502, 655, -699, 541, 95, -1577,
    -1241, 550, -44, 39, -820, -216, -121, -757,
    -348, 937, 893, 387, -603, 1713, -1105, 1058,
    1449, 837, 901, 1637, -569, -1617, -1530, 1199,
    50, -830, -625, 4, 176, -156, 1257, -1507,
    -380, -606, 1293, 661, 1428, -1580, -565, -992,
    548, -800, 64, -371, 961, 641, 87, 630,
    675, -834, 205, 54, -1081, 1351, 1413, -1331,
    -1673, -1267, -1558, 281, -1464, -588, 1015, 436,
    223, 1138, -1059, -397, -183, 1655, 559, -1674,
    277, 933, 1723, 437, -1514, 242, 1640, 432,
    -1583, 696, 774, 1671, 927, 514, 512, 489,
    297, 601, 1473, 1130, 1322, 871, 760, 1212,
    -312, -352, 443, 943, 8, 1250, -100, 1660,
    -31, 1206, -1341, -1247, 444, 235, 1364, -1209,
    361, 230, 673, 582, 1409, 1501, 1401, 251,
    1022, -1063, 1053, 1188, 417, -1391, -27, -1626,
    1685, -315, 1408, -1248, 400, 274, -1543, 32,
    -1550, 1531, -1367, -124, 1458, 1379, -940, -1681,
    22, 1709, -275, 1108, 354, -1728, -968, 858,
    1221, -218, 294, -732, -1095, 892, 1588, -779,
], dtype=np.int16)
assert ZETAS.shape == (192,), ZETAS.shape


# ---- NTT layout helpers --------------------------------------------------

# Each lane is a 4-coefficient block. There are N/4 = 192 lanes, but
# basemul groups two lanes per zeta:
#   lane (8*i) uses     zetas[96 + i]
#   lane (8*i + 1) uses -zetas[96 + i]
# so a "design pair" indexed by i ∈ [0, N/8) covers two lanes (8 coeffs).
N_LANES      = N // D            # 192
N_LANE_PAIRS = N // (2 * D)      # 96


def lane_zeta(lane_idx: int) -> int:
    """Return the NTT-domain modulus parameter ζ for a given 4-coef lane.

    lane_idx ∈ [0, N_LANES). The lane occupies coefficients
    [4*lane_idx, 4*lane_idx + 4) and represents Zq[X]/(X^4 - ζ).
    """
    if not 0 <= lane_idx < N_LANES:
        raise ValueError(f"lane_idx out of range: {lane_idx}")
    pair = lane_idx >> 1                # which entry in zetas[96 ..]
    sign = 1 if (lane_idx & 1) == 0 else -1
    return int(sign * int(ZETAS[96 + pair]))
