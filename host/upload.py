#!/usr/bin/env python3
"""SMAUG-T 펌웨어 업로더 (CW1173 ChipWhisperer-Lite + CW308T-STM32F4 / STM32F415).

CW 보드 선택은 ``cw_serial.pick_serial`` 정책을 따른다:
    - TARGET_SN 보드가 있으면 그것을 사용
    - 없고 forbidden 제외 단일 보드면 그것을 자동 선택
    - 명시적인 ``--serial`` 은 위 정책을 무시 (단, FORBIDDEN_SN 은 항상 거부)

사용 예:
    python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.bin
    python3 host/upload.py path/to/firmware.hex --no-verify
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import chipwhisperer as cw
from chipwhisperer.capture.utils.IntelHex import IntelHex

from cw_serial import TARGET_SN, pick_serial


# STM32 내장 부트로더가 기대하는 플래시 시작 주소
STM32_FLASH_BASE = 0x08000000


def to_hex_path(fw_path: Path) -> tuple[Path, Path | None]:
    """
    .hex이면 그대로 사용, .bin이면 STM32_FLASH_BASE 오프셋으로 임시 .hex 변환.
    반환: (programmer에 넘길 경로, 정리해야 할 임시 파일 경로 또는 None)
    """
    suffix = fw_path.suffix.lower()
    if suffix == ".hex":
        return fw_path, None
    if suffix == ".bin":
        ih = IntelHex()
        ih.loadbin(str(fw_path), offset=STM32_FLASH_BASE)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".hex", delete=False, prefix="smaug_fw_"
        )
        tmp.close()
        ih.write_hex_file(tmp.name)
        return Path(tmp.name), Path(tmp.name)
    sys.exit(f"[ABORT] 지원하지 않는 펌웨어 확장자: {suffix} ({fw_path})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ChipWhisperer-Lite 경유 STM32F415 SMAUG-T 펌웨어 업로더",
    )
    p.add_argument(
        "firmware",
        type=Path,
        help="플래시할 펌웨어 (.bin 또는 .hex)",
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
        "--no-verify",
        action="store_true",
        help="플래시 후 베리파이 생략",
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

    sn = pick_serial(args.serial)
    fw_path, tmp_to_clean = to_hex_path(args.firmware.resolve())

    print(f"[INFO] CW1173 sn={sn} 에 연결")
    scope = cw.scope(sn=sn)
    try:
        # CW308T-STM32F4 표준 셋업: HS2=CLKGEN, tio1/2 = serial, 7.37MHz 등.
        scope.default_setup()

        prog = cw.programmers.STM32FProgrammer(baud=args.baud)
        prog.scope = scope

        print(f"[INFO] STM32 부트로더 진입 및 칩 식별")
        prog.open()
        prog.find()
        prog.erase()

        print(f"[INFO] 플래시 중: {fw_path} (verify={not args.no_verify})")
        prog.program(str(fw_path), memtype="flash", verify=not args.no_verify)
        prog.close()
        print("[OK] 플래시 완료")
    finally:
        try:
            scope.dis()
        except Exception:
            pass
        if tmp_to_clean is not None:
            try:
                os.unlink(tmp_to_clean)
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
