#!/usr/bin/env python3
"""Minimal inject + L round-trip — ct fingerprint mismatch 진단.

E1 capture 에서 board fp != host expected. 단순 inject 후 fp 비교.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import chipwhisperer as cw  # noqa: E402

from host.chosen_ct import CHUNK_BYTES, reset_target  # noqa: E402
from host.cw_serial import pick_serial  # noqa: E402
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import params as _params  # noqa: E402


def _ack(target, n: int, timeout_ms: int = 10000) -> bytes:
    a = target.simpleserial_read("r", n, timeout=timeout_ms)
    if a is None or len(a) != n:
        raise RuntimeError(f"ack {-1 if a is None else len(a)} != {n}")
    return bytes(a)


def main() -> int:
    p = _params.get("smaug1")
    sn = pick_serial(None)
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = 24400
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = 25.0
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = 38400
    target.flush()
    time.sleep(0.3)

    try:
        target.simpleserial_write("F", b"")
        pk_fp = _ack(target, 16)
        print(f"pk_fp16 = {pk_fp.hex()}")

        # Build ct: c1 = α=64 at component=0, coef_idx=0
        ct = _chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=64)
        ct_bytes = ct.to_bytes()
        host_fp = hashlib.sha3_256(ct_bytes).hexdigest()[:32]
        print(f"host expected ct fp:  {host_fp}")
        print(f"ct_bytes len = {len(ct_bytes)}, first 16: {ct_bytes[:16].hex()}")
        print(f"ct_bytes[200:216]:  {ct_bytes[200:216].hex()}")
        print(f"ct_bytes[600:616]:  {ct_bytes[600:616].hex()}")

        chunks = _chosen.chunkify(ct_bytes, chunk_size=CHUNK_BYTES)
        print(f"chunks: {len(chunks)} × {CHUNK_BYTES} bytes")

        # Inject and verify per-chunk
        for idx, data in chunks:
            assert len(data) == CHUNK_BYTES
            target.simpleserial_write("I", bytes([idx]) + data)
            st = _ack(target, 1, timeout_ms=2000)
            if st[0] != 0:
                raise RuntimeError(f"'I' chunk {idx} status={st[0]}")
            if idx in (0, 10, 20):
                print(f"  chunk {idx}: ack ok, sent {data.hex()[:32]}...")

        target.simpleserial_write("L", b"")
        board_fp = _ack(target, 16).hex()
        print(f"board returned ct fp: {board_fp}")
        print(f"match: {board_fp == host_fp}")

        # If mismatch, do another inject right after with same bytes
        print("\nRetry — inject same ct again:")
        for idx, data in chunks:
            target.simpleserial_write("I", bytes([idx]) + data)
            st = _ack(target, 1, timeout_ms=2000)
            if st[0] != 0:
                raise RuntimeError(f"'I' chunk {idx} status={st[0]}")
        target.simpleserial_write("L", b"")
        board_fp2 = _ack(target, 16).hex()
        print(f"board returned ct fp (2nd): {board_fp2}")
        print(f"match host: {board_fp2 == host_fp}, match prev: {board_fp2 == board_fp}")

    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
