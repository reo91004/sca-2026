#!/usr/bin/env python3
"""[DEBUG] 'T' isolated poly_mul_acc 명령 round-trip 검증 (0/6 fail, NEGATIVE).

Paper Section 8.2 ("What didn't work"). Phase F (Toom-Cook SASCA) 시작 시
firmware 에 'T' 명령 추가해 isolated `poly_mul_acc` 의 sub-trigger 캡처
시도. **host predict_mul 과 보드 응답이 일치 안 해 0/6 round-trip fail** —
SMAUG-T archive 가 internal storage form (q-modular unsigned 또는 NTT-friendly
변형) 사용 추정.

목적:
    'T' payload = (component, idx, alpha) → firmware: host_b 을 sparse
    initialize (host_b[idx]=alpha) + trigger_high → poly_mul_acc(sk, host_b,
    out) → trigger_low → out 의 첫 32B 응답.
    host: numpy 로 sk · host_b mod (X^256+1) 의 첫 16 int16 계산해 보드 응답
    과 비교.

핵심 발견 (negative):
    - 보드 응답 byte 가 host 가정 (single int16 plain) 과 일치 안 함.
    - 패턴: board 32704 (0x7FC0) vs expected -64 (0xFFC0) — 한 비트 차이.
      또 board -32768 (INT16_MIN) vs expected 0.
    - 결론: archive 의 internal layout 분석 안 한 채 round-trip 어려움.
      → Phase F pivot — 'Z' (전체 indcpa_dec) trace windowing 으로 대체.

Note (history/):
    'T' 명령 자체는 firmware 에 keep (paper Section 8.2 mention). 이 debug
    script 는 negative finding 의 *원천 evidence*.

용법 (legacy):
    history/scripts/debug_T_command_NEGATIVE.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402


def _ack(target, n: int, timeout_ms: int = 10000) -> bytes:
    a = target.simpleserial_read("r", n, timeout=timeout_ms)
    if a is None or len(a) != n:
        raise RuntimeError(f"ack {-1 if a is None else len(a)} != {n}")
    return bytes(a)


def predict_mul_acc(a_int16: np.ndarray, b_int16: np.ndarray, n: int = 256) -> np.ndarray:
    """SMAUG-T poly_mul_acc 의 numpy 시뮬레이션: out = a · b mod (X^n + 1).

    a, b : (n,) int16. out : (n,) int16 (overflow 자연 truncate).
    """
    a64 = a_int16.astype(np.int64)
    b64 = b_int16.astype(np.int64)
    full = np.convolve(a64, b64)  # 2n-1
    out = np.zeros(n, dtype=np.int64)
    out[:n] = full[:n]
    if full.size > n:
        out[:full.size - n] -= full[n:]
    return out.astype(np.int16)


def main() -> int:
    p = _params.SMAUG1
    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(None)
    print(f"[INFO] sn={sn}")

    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = 24400
    scope.gain.db = 25.0
    scope.clock.adc_src = "clkgen_x4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = 38400
    target.flush()
    time.sleep(0.3)

    try:
        target.simpleserial_write("F", b"")
        pk_fp16 = _ack(target, 16)
        print(f"  pk_fp16 = {pk_fp16.hex()}")

        sk_pke = bytearray()
        for idx in range(4):
            target.simpleserial_write("X", bytes([idx]))
            sk_pke.extend(_ack(target, 32, timeout_ms=2000))
        sk_unpacked_flat = _codec.unpack_sx(bytes(sk_pke))  # (512,) ternary
        sk = sk_unpacked_flat.reshape(p.module_rank, p.n).astype(np.int16)
        print(f"  sk HW per poly: {[int(np.count_nonzero(sk[m])) for m in range(p.module_rank)]}")

        # 다양한 test cases
        test_cases = [
            (0,  10, 64),
            (0,  50, 192),
            (0, 200, 64),
            (1,   5, 128),
            (1, 100, 64),
            (1, 255, 4),
        ]
        passes = 0
        for comp, idx, alpha in test_cases:
            payload = bytes([comp]) + idx.to_bytes(2, 'big') + alpha.to_bytes(2, 'big')
            target.simpleserial_write("T", payload)
            scope.arm()  # 'T' has trigger
            target.simpleserial_write("T", payload)  # actually arm before write
            # Re-do: arm scope first, then write T
            break  # restructure below
        # Restructure — separate arm + write
        passes = 0
        fails = 0
        for comp, idx, alpha in test_cases:
            payload = bytes([comp]) + idx.to_bytes(2, 'big') + alpha.to_bytes(2, 'big')
            scope.arm()
            target.simpleserial_write("T", payload)
            cap_status = scope.capture()
            ack = target.simpleserial_read("r", 32, timeout=10000)
            if ack is None or len(ack) != 32:
                print(f"  FAIL (ack) comp={comp} idx={idx} alpha={alpha}")
                fails += 1
                continue
            board_out = np.frombuffer(bytes(ack), dtype=np.int16)
            # host 시뮬레이션
            host_b = np.zeros(p.n, dtype=np.int16)
            host_b[idx] = alpha
            expected = predict_mul_acc(sk[comp], host_b)[:16]
            match = int(np.sum(board_out == expected))
            ok = match == 16
            print(f"  [{'OK ' if ok else 'FAIL'}] comp={comp} idx={idx:3d} alpha={alpha:4d}  "
                  f"match={match}/16  board[0:4]={board_out[:4].tolist()} "
                  f"exp[0:4]={expected[:4].tolist()}  cap_status={cap_status}")
            if ok:
                passes += 1
            else:
                fails += 1
        print(f"\n[SUMMARY] T command: {passes}/{len(test_cases)} pass, {fails} fail")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
