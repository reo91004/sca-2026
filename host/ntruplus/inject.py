"""ChipWhisperer SimpleSerial wrappers for the NTRU+ firmware.

Encapsulates the F → B* → I* → L → D pipeline so callers don't have to
manage chunk indices, fingerprints, or response framing. Designed for
use by per-design capture loops in scripts/ntruplus/.

Firmware command set (see firmware/simpleserial-ntruplus/simpleserial-ntruplus.c):

    F  : new keypair, persisted on board.        ack = sha3_256(pk)[0:16]
    B  : pk chunk dump (idx).                    ack = 32 B chunk
    I  : ct_inj chunk inject (idx + 32 B).       ack = 1 B status (0=OK)
    L  : declare ct_inj loaded.                  ack = sha3_256(ct_inj)[0:16]
    D  : crypto_kem_dec(ss_dec, ct_inj, sk).     ack = 1 B mismatch flag
    X  : sk chunk dump (idx).                    ack = 32 B chunk

For ntruplus768 the chunk size is 32 B (POLYBYTES = 1152 = 36 × 32). pk is
also 1152 B. sk is 2336 B = 73 × 32. ct is 1152 B = 36 × 32.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .params import (
    CIPHERTEXTBYTES,
    POLYBYTES,
    PUBLICKEYBYTES,
    SECRETKEYBYTES,
)


CT_CHUNK = 32
PK_CHUNK = 32
SK_CHUNK = 32

assert POLYBYTES        % CT_CHUNK == 0
assert PUBLICKEYBYTES   % PK_CHUNK == 0
assert SECRETKEYBYTES   % SK_CHUNK == 0
N_CT_CHUNKS = CIPHERTEXTBYTES // CT_CHUNK   # 36
N_PK_CHUNKS = PUBLICKEYBYTES  // PK_CHUNK   # 36
N_SK_CHUNKS = SECRETKEYBYTES  // SK_CHUNK   # 73


def _sha3_256_16(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()[:16]


@dataclass
class FirmwareError(RuntimeError):
    cmd: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.cmd}] {self.detail}"


def keygen_persistent(target, *, timeout_ms: int = 5000) -> bytes:
    """Send 'F'. Return sha3_256(pk)[:16] fingerprint reported by the board."""
    target.simpleserial_write("F", b"")
    fp = target.simpleserial_read("r", 16, timeout=timeout_ms)
    if fp is None or len(fp) != 16:
        raise FirmwareError("F", f"bad ack len: {0 if fp is None else len(fp)}")
    return bytes(fp)


def dump_pk(target, *, timeout_ms: int = 5000) -> bytes:
    """Read all PK chunks via 'B' and concatenate into a PUBLICKEYBYTES blob."""
    out = bytearray()
    for idx in range(N_PK_CHUNKS):
        target.simpleserial_write("B", bytes([idx]))
        chunk = target.simpleserial_read("r", PK_CHUNK, timeout=timeout_ms)
        if chunk is None or len(chunk) != PK_CHUNK:
            raise FirmwareError("B", f"chunk {idx}: bad len")
        out.extend(chunk)
    if len(out) != PUBLICKEYBYTES:
        raise FirmwareError("B", f"total {len(out)} != {PUBLICKEYBYTES}")
    return bytes(out)


def dump_sk(target, *, timeout_ms: int = 5000) -> bytes:
    """Read all SK chunks via 'X'. WARNING: calibration only — do NOT use as
    inference input under threat model."""
    out = bytearray()
    for idx in range(N_SK_CHUNKS):
        target.simpleserial_write("X", bytes([idx]))
        chunk = target.simpleserial_read("r", SK_CHUNK, timeout=timeout_ms)
        if chunk is None or len(chunk) != SK_CHUNK:
            raise FirmwareError("X", f"chunk {idx}: bad len")
        out.extend(chunk)
    if len(out) != SECRETKEYBYTES:
        raise FirmwareError("X", f"total {len(out)} != {SECRETKEYBYTES}")
    return bytes(out)


def inject_ct(target, ct_bytes: bytes, *, timeout_ms: int = 5000,
              verify_fingerprint: bool = True) -> bytes:
    """Push a chosen ciphertext to the board, then 'L' for fingerprint check.

    Returns the 16-byte fingerprint reported by the board (sha3_256(ct_inj)).
    If `verify_fingerprint` is True, raises if it disagrees with the host
    expectation — that's the codec parity gate.
    """
    if len(ct_bytes) != CIPHERTEXTBYTES:
        raise ValueError(f"ct len {len(ct_bytes)} != {CIPHERTEXTBYTES}")
    for idx in range(N_CT_CHUNKS):
        chunk = ct_bytes[idx * CT_CHUNK:(idx + 1) * CT_CHUNK]
        target.simpleserial_write("I", bytes([idx]) + chunk)
        ack = target.simpleserial_read("r", 1, timeout=timeout_ms)
        if ack is None or len(ack) != 1 or ack[0] != 0:
            got = "None" if ack is None else f"{ack[0]} (len={len(ack)})"
            raise FirmwareError("I", f"chunk {idx}: status={got}")
    target.simpleserial_write("L", b"")
    fp_board = target.simpleserial_read("r", 16, timeout=timeout_ms)
    if fp_board is None or len(fp_board) != 16:
        raise FirmwareError("L", f"bad ack len: {0 if fp_board is None else len(fp_board)}")
    if verify_fingerprint:
        fp_host = _sha3_256_16(ct_bytes)
        if bytes(fp_board) != fp_host:
            raise FirmwareError(
                "L",
                f"fingerprint mismatch: board={bytes(fp_board).hex()} host={fp_host.hex()}",
            )
    return bytes(fp_board)


def decap(target, *, timeout_ms: int = 10000) -> int:
    """Trigger 'D'. Returns the 1-byte mismatch flag.

    Caller must arm the scope BEFORE writing 'D'.
    """
    target.simpleserial_write("D", b"")
    ack = target.simpleserial_read("r", 1, timeout=timeout_ms)
    if ack is None or len(ack) != 1:
        raise FirmwareError("D", f"bad ack len: {0 if ack is None else len(ack)}")
    return int(ack[0])
