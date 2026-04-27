#!/usr/bin/env python3
"""
SMAUG-T 부채널 트레이스 캡처 (ChipWhisperer-Lite + CW308T-STM32F4 / STM32F415).

사용법:
    python3 host/capture.py -n 1000 -s 24400 -o traces/smaug1_dec.npz

흐름:
    1. CW1173 (F415가 매달린 보드, sn=TARGET_SN) 만 사용. 금지 시리얼은 거부.
    2. scope.default_setup() 후 ADC 샘플 수 / 게인 / 트리거 설정.
    3. SimpleSerial v2.1 타겟으로 sca-2026/firmware/simpleserial-smaug 펌웨어와 통신.
    4. 캡처 루프:
           scope.arm() -> target.simpleserial_write('p', b'')
                       -> scope.capture() (펌웨어가 trigger_high()/_low() 사이의 dec 실행)
                       -> target.simpleserial_read('r', 1) (1바이트 mismatch flag)
                       -> scope.get_last_trace()
    5. 트레이스를 numpy .npz로 저장: traces (N,T), responses (N,), timestamp 등.

전제:
    - host/upload.py 로 simpleserial-smaug-CW308_STM32F4.{hex,bin}을 이미 플래시했을 것.
    - SimpleSerial 명령은 'p' (full pipeline) 로 시작. 'd' (dec only) 등은 향후 chosen-CT
      공격 시 추가.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chipwhisperer as cw
import numpy as np

from cw_serial import TARGET_SN, pick_serial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SMAUG-T crypto_kem_dec 부채널 트레이스 캡처",
    )
    p.add_argument(
        "-n", "--num-traces",
        type=int, default=100,
        help="수집할 트레이스 개수 (기본 100)",
    )
    p.add_argument(
        "-s", "--samples",
        type=int, default=24400,
        help="트레이스당 ADC 샘플 수 (CW-Lite 최대 24400)",
    )
    p.add_argument(
        "-g", "--gain-db",
        type=float, default=25.0,
        help="LNA 게인 dB (기본 25.0)",
    )
    p.add_argument(
        "-c", "--cmd",
        choices=("p", "d"),
        default="p",
        help="펌웨어에 보낼 명령: p=full pipeline / d=decaps만 (기본 p). "
             "'d'를 사용하려면 별도 keypair/encaps 사전준비가 필요.",
    )
    p.add_argument(
        "-o", "--output",
        type=Path, required=True,
        help="저장할 .npz 경로",
    )
    p.add_argument(
        "--serial", default=None,
        help=(
            f"CW1173 시리얼 (기본 자동 선택: {TARGET_SN} 가 있으면 그것, "
            "없고 forbidden 제외 단일 보드면 그것)"
        ),
    )
    p.add_argument(
        "--baud", type=int, default=230400,
        help="SimpleSerial UART 보레이트 (기본 230400; CW SS2 표준)",
    )
    p.add_argument(
        "--timeout-ms", type=int, default=5000,
        help="명령당 응답 타임아웃 (ms)",
    )
    return p.parse_args()


def setup_scope(sn: str, samples: int, gain_db: float):
    scope = cw.scope(sn=sn)
    scope.default_setup()                # CW-Lite + STM32F4 표준 셋업

    scope.adc.samples = samples
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = gain_db
    scope.clock.adc_src = "clkgen_x4"    # 4x 샘플링
    scope.trigger.triggers = "tio4"      # CW308T-STM32F4 trigger pad
    return scope


def setup_target(scope, baud: int):
    target = cw.target(scope, cw.targets.SimpleSerial2)
    target.baud = baud
    target.flush()
    return target


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sn = pick_serial(args.serial)
    print(f"[INFO] CW1173 sn={sn}")

    scope = setup_scope(sn, args.samples, args.gain_db)
    target = setup_target(scope, args.baud)
    print(f"[INFO] scope.adc.samples={scope.adc.samples} gain={scope.gain.db} dB")
    print(f"[INFO] target SS_VER_2_1 baud={target.baud}")

    cmd = args.cmd.encode("ascii")
    n = args.num_traces

    traces = np.empty((n, args.samples), dtype=np.float32)
    responses = np.empty((n,), dtype=np.uint8)
    timeouts = 0
    started = time.time()

    try:
        for i in range(n):
            scope.arm()
            target.simpleserial_write(cmd, bytearray())

            # capture() -> True if the trigger never fired in time
            if scope.capture():
                timeouts += 1
                print(f"[WARN] trace {i}: scope.capture() timeout")
                continue

            ack = target.simpleserial_read("r", 1, timeout=args.timeout_ms)
            if ack is None or len(ack) != 1:
                timeouts += 1
                print(f"[WARN] trace {i}: SimpleSerial response missing/short")
                continue

            traces[i] = scope.get_last_trace()
            responses[i] = ack[0]

            if (i + 1) % 50 == 0 or i == n - 1:
                elapsed = time.time() - started
                rate = (i + 1) / max(elapsed, 1e-9)
                print(f"[PROG] {i + 1}/{n}  ({rate:.1f} tr/s)  timeouts={timeouts}")

    finally:
        try:
            target.dis()
        except Exception:
            pass
        try:
            scope.dis()
        except Exception:
            pass

    captured = n - timeouts
    np.savez_compressed(
        args.output,
        traces=traces[:n],
        responses=responses[:n],
        meta=np.array(
            {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "scope_sn": sn,
                "samples": args.samples,
                "gain_db": args.gain_db,
                "cmd": args.cmd,
                "baud": args.baud,
                "n_attempted": n,
                "n_timeouts": timeouts,
            },
            dtype=object,
        ),
    )
    print(
        f"[OK] saved {captured}/{n} traces to {args.output} "
        f"(shape={traces.shape}, dtype={traces.dtype})"
    )
    return 0 if timeouts == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
