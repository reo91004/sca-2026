#!/usr/bin/env python3
"""ADC 윈도우 크기별 throughput 비교.

같은 보드 상태 (이전 'F'/'M' 결과 영속) 위에서 'D' n 회를 두 윈도우 설정으로
캡처. tr/s 와 추정 full-sweep 시간 출력.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target  # noqa: E402


def _capture(scope, target, n: int, samples: int) -> tuple[float, int]:
    timeouts = 0
    started = time.time()
    for i in range(n):
        scope.arm()
        target.simpleserial_write("D", b"")
        if scope.capture():
            timeouts += 1
            continue
        ack = target.simpleserial_read("r", 1, timeout=10000)
        if ack is None or len(ack) != 1:
            timeouts += 1
            continue
        _ = scope.get_last_trace()
    return time.time() - started, timeouts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", type=int, default=100)
    p.add_argument("--gain-db", type=float, default=25.0)
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--serial", default=None)
    args = p.parse_args()

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(args.serial)
    print(f"[INFO] sn={sn}")
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = args.gain_db
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()
    time.sleep(0.3)

    # 보드는 이미 'F'/'M' 상태가 영속되어 있다고 가정 — 이전 E3c 캡처 직후라
    # ct_inj 가 채워져 있고 sk 도 있다. 'D' 만 보내면 됨.
    cases = [
        ("full window", 0, 24400),
        ("narrow µ' window", 5500, 1500),
        ("very narrow µ' window", 5750, 200),
    ]
    results = []
    for label, off, samp in cases:
        scope.adc.offset = off
        scope.adc.samples = samp
        # warm-up 5 trace
        _capture(scope, target, 5, samp)
        elapsed, t_out = _capture(scope, target, args.n, samp)
        rate = args.n / elapsed if elapsed > 0 else 0
        results.append((label, off, samp, elapsed, rate, t_out))
        print(f"  [{label}] offset={off:5d} samples={samp:5d}: "
              f"{elapsed:.2f}s for {args.n} → {rate:.2f} tr/s "
              f"(timeouts={t_out})")

    print()
    print("=== full-sweep estimate (256 positions × 2 α × N traces) ===")
    for label, off, samp, _, rate, _ in results:
        for N in (16, 64, 256, 500, 1000):
            total = 256 * 2 * N
            sec = total / rate
            print(f"  [{label:25s}] N={N:4d}: {total:6d} trace ≈ "
                  f"{sec/3600:.2f} h ({sec/60:.1f} min)")

    try: target.dis()
    except Exception: pass
    try: scope.dis()
    except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
