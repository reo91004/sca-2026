#!/usr/bin/env python3
"""Phase 4.6 — compact G=200 capture scout.

The Phase 4.5/4.6 candidate-set null check showed that unioning broad top-100
sets does not beat the set-size null.  The next useful experiment is therefore
score sharpness: increase G while keeping the trace window compact.

Defaults:
  K=1, lanes=0,64,80,128, G=200, N=8, samples=6000

This captures only the early decap window containing the calibrated basemul
PoIs for these lanes, reducing artifact size compared to full 24400-sample
captures.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chipwhisperer as cw
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

from cw_serial import pick_serial  # noqa: E402
from ntruplus import inject as nti  # noqa: E402
from ntruplus.chosen import selected_lane  # noqa: E402


def gammas_hw123(max_g: int = 200) -> list[int]:
    hw1 = [1 << i for i in range(12)]
    hw2 = [(1 << i) + (1 << j) for i in range(12) for j in range(i + 1, 12)]
    hw3 = [
        (1 << i) + (1 << j) + (1 << k)
        for i in range(12)
        for j in range(i + 1, 12)
        for k in range(j + 1, 12)
    ]
    base = sorted(set(hw1 + hw2))
    if max_g <= len(base):
        return base[:max_g]
    need = max_g - len(base)
    # Evenly cover the HW=3 list instead of taking only the smallest values.
    idx = np.linspace(0, len(hw3) - 1, num=need, dtype=int)
    extra = [hw3[int(i)] for i in idx]
    return sorted(set(base + extra))[:max_g]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-K", "--keys", type=int, default=1)
    ap.add_argument("-L", "--lanes", type=str, default="0,64,80,128")
    ap.add_argument("-N", "--traces", type=int, default=8)
    ap.add_argument("-G", "--gammas", type=int, default=200)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("-s", "--samples", type=int, default=6000)
    ap.add_argument("--decimate", type=int, default=4)
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "traces/ntruplus768/phase46/g200_calib_K1L4N8_s6000.npz")
    ap.add_argument("--baud", type=int, default=38400)
    ap.add_argument("--timeout-ms", type=int, default=10000)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lanes = [int(x) for x in args.lanes.split(",") if x]
    gammas = gammas_hw123(args.gammas)
    K, L, G, N = args.keys, len(lanes), len(gammas), args.traces
    total = K * L * G * N
    print(f"[INFO] capture matrix: K={K} L={L} G={G} N={N} total={total}")
    print(f"[INFO] lanes={lanes}")
    print(f"[INFO] samples={args.samples} decimate={args.decimate}")

    sn = pick_serial(None)
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = args.samples
    scope.adc.offset = 0
    scope.adc.decimate = args.decimate
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = 25.0
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    scope.io.nrst = "low"
    time.sleep(0.05)
    scope.io.nrst = "high_z"
    time.sleep(0.5)

    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()

    traces = np.empty((K, L, G, N, args.samples), dtype=np.float32)
    sk_blobs = np.zeros((K, 2336), dtype=np.uint8)
    pk_blobs = np.zeros((K, 1152), dtype=np.uint8)

    timeouts = 0
    t0 = time.time()
    try:
        for ki in range(K):
            pk_fp = nti.keygen_persistent(target, timeout_ms=args.timeout_ms)
            sk_blobs[ki] = list(nti.dump_sk(target, timeout_ms=args.timeout_ms))
            pk_blobs[ki] = list(nti.dump_pk(target, timeout_ms=args.timeout_ms))
            print(f"[INFO] key {ki + 1}/{K} pk_fp={pk_fp.hex()[:16]}")
            for li, lane in enumerate(lanes):
                for gi, gamma in enumerate(gammas):
                    ct, _ = selected_lane(lane=lane, gamma=gamma, slot=args.slot)
                    nti.inject_ct(target, ct, timeout_ms=args.timeout_ms,
                                  verify_fingerprint=False)
                    for ni in range(N):
                        scope.arm()
                        target.simpleserial_write("D", b"")
                        if scope.capture():
                            timeouts += 1
                            continue
                        ack = target.simpleserial_read("r", 1, timeout=args.timeout_ms)
                        if ack is None or len(ack) != 1:
                            timeouts += 1
                            continue
                        traces[ki, li, gi, ni] = scope.get_last_trace()
                elapsed = time.time() - t0
                done = ((ki * L) + li + 1) * G * N
                rate = done / max(elapsed, 1e-9)
                print(f"[PROG] key {ki + 1}/{K} lane {li + 1}/{L} "
                      f"done={done}/{total} rate={rate:.2f} tr/s timeouts={timeouts}")
    finally:
        try:
            target.dis()
        except Exception:
            pass
        try:
            scope.dis()
        except Exception:
            pass

    fw_path = ROOT / ("firmware/simpleserial-ntruplus/"
                      "simpleserial-ntruplus-CW308_STM32F4-ntruplus768.hex")
    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope_sn": sn,
        "samples": args.samples,
        "decimate": args.decimate,
        "target": "ntruplus768",
        "phase": "4.6-g200-compact",
        "K": K,
        "lanes": lanes,
        "gammas": gammas,
        "N": N,
        "G_count": G,
        "n_timeouts": timeouts,
        "fw_hex": str(fw_path),
        "fw_sha256": hashlib.sha256(fw_path.read_bytes()).hexdigest() if fw_path.exists() else None,
        "slot_used": args.slot,
        "note": "compact early-window G=200 HW<=3 subset scout",
    }
    np.savez_compressed(
        args.output,
        traces=traces,
        sk_blobs=sk_blobs,
        pk_blobs=pk_blobs,
        lanes=np.array(lanes, dtype=np.int32),
        gammas=np.array(gammas, dtype=np.int32),
        meta=np.array(meta, dtype=object),
    )
    print(f"[OK] saved -> {args.output} shape={traces.shape} timeouts={timeouts} "
          f"elapsed={time.time() - t0:.1f}s")
    return 0 if timeouts == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
