#!/usr/bin/env python3
"""
SMAUG-T SCA 트레이스 캡처 (ChipWhisperer-Lite + CW308T-STM32F4 / STM32F415).

타겟: SMAUG-T KEM (firmware/simpleserial-smaug)

사용법:
    # SMAUG-T full pipeline (기본)
    python3 host/smaug/capture.py -n 1000 -s 24400 -o traces/smaug1_dec.npz

    # HQC custom RM encode_single ('e' 명령, 16바이트 응답)
    python3 host/smaug/capture.py --target hqc -c e --send-len 1 --resp-len 16 \\
        -n 1000 -s 24400 -o traces/hqc_custom_e.npz

흐름:
    1. CW1173 (F415가 매달린 보드, sn=TARGET_SN) 만 사용. 금지 시리얼은 거부.
    2. scope.default_setup() 후 ADC 샘플 수 / 게인 / 트리거 설정.
    3. SimpleSerial v1.1 타겟으로 펌웨어와 통신 (양 firmware 모두 v1_1로 통일).
    4. 캡처 루프:
           scope.arm() -> target.simpleserial_write(cmd, payload)
                       -> scope.capture()
                       -> target.simpleserial_read('r', resp_len)
                       -> scope.get_last_trace()
    5. 트레이스를 numpy .npz로 저장: traces (N,T), responses (N,), timestamp 등.

전제:
    - host/smaug/upload.py 로 해당 firmware .hex 를 이미 플래시했을 것.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chipwhisperer as cw
import numpy as np

from host.cw_serial import TARGET_SN, pick_serial


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ChipWhisperer SCA 트레이스 캡처 (SMAUG-T / HQC)",
    )
    p.add_argument(
        "--target", choices=("smaug", "hqc"), default="smaug",
        help="대상 KEM (기본 smaug). 메타에만 기록되며, 실제 동작은 cmd/길이로 결정.",
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
        default="p",
        help=(
            "펌웨어에 보낼 SimpleSerial 명령 1글자 (기본 'p'). "
            "SMAUG: 'p' full pipeline / 'd' decaps만 / 'k' keypair / 'e' encaps. "
            "HQC: 'e' encode_single (16B) / 'p' encode_full (16B) / 'c' code_encode (pqclean)."
        ),
    )
    p.add_argument(
        "--send-len", type=int, default=0,
        help="명령 페이로드 바이트 수 (zero-fill, 기본 0)",
    )
    p.add_argument(
        "--resp-len", type=int, default=1,
        help="응답 'r' 바이트 수 (기본 1; HQC encode 계열은 16)",
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
        "--baud", type=int, default=38400,
        help="SimpleSerial UART 보레이트 (기본 38400; SS_VER_1_1 표준)",
    )
    p.add_argument(
        "--timeout-ms", type=int, default=5000,
        help="명령당 응답 타임아웃 (ms)",
    )
    p.add_argument(
        "--firmware-hex", type=Path, default=None,
        help=".npz meta 에 firmware_hex_sha256 을 기록할 .hex 경로 (옵션)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="meta 에 기록할 host-side seed (chosen-CT 재현용; 기본=0)",
    )
    p.add_argument(
        "--label", default=None,
        help="meta 에 기록할 사람 읽기용 라벨 (기본: 'p'/'D' 등 cmd 그대로)",
    )
    return p.parse_args()


def _git_rev_short() -> str:
    """현재 HEAD 의 짧은 sha. 실패 시 'unknown' (no-throw)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_scope(sn: str, samples: int, gain_db: float):
    import time
    scope = cw.scope(sn=sn)
    scope.default_setup()                # CW-Lite + STM32F4 표준 셋업

    scope.adc.samples = samples
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = gain_db
    scope.clock.adc_src = "clkgen_x4"    # 4x 샘플링
    scope.trigger.triggers = "tio4"      # CW308T-STM32F4 trigger pad

    # nRST 토글 + 부팅 대기 — default_setup() 직후 통신이 안정되도록.
    # (실측: 이 reset 없이는 첫 simpleserial_read 가 즉시 timeout.)
    scope.io.nrst = "low"
    time.sleep(0.05)
    scope.io.nrst = "high_z"
    time.sleep(0.5)
    return scope


def setup_target(scope, baud: int):
    target = cw.target(scope, cw.targets.SimpleSerial)
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
    print(f"[INFO] target=SS_VER_1_1 baud={target.baud} "
          f"target_kind={args.target} cmd='{args.cmd}' "
          f"send_len={args.send_len} resp_len={args.resp_len}")

    cmd = args.cmd  # SimpleSerial v1 takes a str, not bytes
    n = args.num_traces

    traces = np.empty((n, args.samples), dtype=np.float32)
    # responses: (N, resp_len) uint8 — full ack payload per trace
    responses = np.zeros((n, args.resp_len), dtype=np.uint8)
    timeouts = 0
    started = time.time()
    payload = bytearray(args.send_len)

    try:
        for i in range(n):
            scope.arm()
            target.simpleserial_write(cmd, payload)

            # capture() -> True if the trigger never fired in time
            if scope.capture():
                timeouts += 1
                print(f"[WARN] trace {i}: scope.capture() timeout")
                continue

            ack = target.simpleserial_read("r", args.resp_len, timeout=args.timeout_ms)
            if ack is None or len(ack) != args.resp_len:
                timeouts += 1
                got = 0 if ack is None else len(ack)
                print(f"[WARN] trace {i}: SimpleSerial response len={got} "
                      f"(expected {args.resp_len})")
                continue

            traces[i] = scope.get_last_trace()
            responses[i, :] = ack

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

    fw_sha = None
    if args.firmware_hex is not None:
        if args.firmware_hex.exists():
            fw_sha = _file_sha256(args.firmware_hex)
        else:
            print(f"[WARN] --firmware-hex 경로 없음: {args.firmware_hex}")

    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope_sn": sn,
        "samples": args.samples,
        "gain_db": args.gain_db,
        "target": args.target,
        "cmd": args.cmd,
        "send_len": args.send_len,
        "resp_len": args.resp_len,
        "ss_ver": "SS_VER_1_1",
        "baud": args.baud,
        "n_attempted": n,
        "n_timeouts": timeouts,
        # 재현성 보강 (모든 .npz 가 어떤 펌웨어/git 상태에서 나왔는지 추적):
        "git_rev": _git_rev_short(),
        "firmware_hex": str(args.firmware_hex) if args.firmware_hex else None,
        "firmware_hex_sha256": fw_sha,
        "seed": 0 if args.seed is None else int(args.seed),
        "label": args.label or args.cmd,
    }
    np.savez_compressed(
        args.output,
        traces=traces[:n],
        responses=responses[:n],
        meta=np.array(meta, dtype=object),
    )
    print(
        f"[OK] saved {captured}/{n} traces to {args.output} "
        f"(shape={traces.shape}, dtype={traces.dtype})"
    )
    return 0 if timeouts == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
