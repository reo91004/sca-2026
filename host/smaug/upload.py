#!/usr/bin/env python3
"""SMAUG-T 펌웨어 업로더 (CW1173 ChipWhisperer-Lite + CW308T-STM32F4 / STM32F415).

CW 보드 선택은 ``cw_serial.pick_serial`` 정책을 따른다:
    - TARGET_SN 보드가 있으면 그것을 사용
    - 없고 forbidden 제외 단일 보드면 그것을 자동 선택
    - 명시적인 ``--serial`` 은 위 정책을 무시 (단, FORBIDDEN_SN 은 항상 거부)

흐름은 정상 동작하는 레퍼런스 프로젝트와 같은 ChipWhisperer 기본 패턴을 쓴다:
    scope = cw.scope(sn=...)
    target = cw.target(scope, cw.targets.SimpleSerial)
    scope.default_setup()
    cw.program_target(scope, cw.programmers.STM32FProgrammer, fw_path)

사용 예:
    python3 host/smaug/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chipwhisperer as cw

from host.cw_serial import TARGET_SN, pick_serial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ChipWhisperer-Lite 경유 STM32F415 SMAUG-T 펌웨어 업로더",
    )
    p.add_argument(
        "firmware",
        type=Path,
        help="플래시할 .hex 펌웨어 (Intel HEX, 0x08000000 베이스)",
    )
    p.add_argument(
        "-s",
        "--serial",
        default=None,
        help=(
            f"CW1173 시리얼 명시 (미지정시 자동 선택; 표준 F415 sn={TARGET_SN})"
        ),
    )
    p.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="STM32 부트로더 UART 보레이트 (기본: 115200)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.firmware.is_file():
        sys.exit(f"[ABORT] 파일이 없음: {args.firmware}")
    if args.firmware.suffix.lower() != ".hex":
        sys.exit(
            f"[ABORT] .hex 만 지원 (받은 확장자: {args.firmware.suffix}). "
            "make 가 동시에 만드는 simpleserial-smaug-*.hex 를 쓰라."
        )

    sn = pick_serial(args.serial)
    fw_path = args.firmware.resolve()

    print(f"[INFO] CW1173 sn={sn} 에 연결")
    scope = cw.scope(sn=sn)
    target = cw.target(scope, cw.targets.SimpleSerial)
    prog = cw.programmers.STM32FProgrammer
    time.sleep(0.05)
    scope.default_setup()
    try:
        print(f"[INFO] 플래시 중: {fw_path}")
        cw.program_target(
            scope,
            prog,
            str(fw_path),
            baud=args.baud,
        )
        print("[OK] 플래시 완료")
    finally:
        try:
            target.dis()
        except Exception:
            pass
        try:
            scope.dis()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
