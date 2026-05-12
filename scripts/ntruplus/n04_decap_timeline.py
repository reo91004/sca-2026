#!/usr/bin/env python3
"""Map the full crypto_kem_dec timeline by varying scope.adc.decimate.

Goal — determine *where* in the trace `poly_basemul(&m1, &c, &f)` sits, so
Phase 3 can target it with the correct ADC window.

Decap runs continuously while the trigger is high (cmd_decap_inject), so a
single long capture catches the full operation if the sample rate is
slowed enough. With decimate=4 we get 1 sample / core cycle on the
ChipWhisperer-Lite, i.e. 24400 samples ≈ 3.3 ms of CPU time. NTRU+768
decap on Cortex-M4 is empirically near 100 k cycles, so a single
decimate=4 trace catches roughly 24 % of the workload — but it's enough to
calibrate where we are.

We additionally capture *two* per-key valid traces to estimate noise vs
location signal under decimate=4.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chipwhisperer as cw
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from cw_serial import pick_serial  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("-K", type=int, default=4, help="distinct keys")
    p.add_argument("-N", type=int, default=4, help="traces per key")
    p.add_argument("-o", "--output", type=Path,
                   default=ROOT / "traces/ntruplus768/phase1/timeline_dec4.npz")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sn = pick_serial(None)
    print(f"[INFO] CW1173 sn={sn}")

    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = 24400
    scope.adc.offset = 0
    scope.adc.decimate = args.decimate
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = 25.0
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    scope.io.nrst = "low"; time.sleep(0.05)
    scope.io.nrst = "high_z"; time.sleep(0.5)

    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = 38400
    target.flush()

    K = args.K
    N = args.N
    T = scope.adc.samples
    print(f"[INFO] decimate={args.decimate}  effective samples-per-cycle="
          f"{4 / args.decimate}  total samples={T}")

    traces = np.empty((K, N, T), dtype=np.float32)
    try:
        for k in range(K):
            target.simpleserial_write("k", b"")
            ack = target.simpleserial_read("r", 16, timeout=5000)
            assert ack is not None and len(ack) == 16
            target.simpleserial_write("e", b"")
            ack = target.simpleserial_read("r", 16, timeout=5000)
            assert ack is not None and len(ack) == 16
            for n in range(N):
                scope.arm()
                target.simpleserial_write("d", b"")
                if scope.capture():
                    print(f"[WARN] key {k} trace {n}: timeout")
                    continue
                a = target.simpleserial_read("r", 1, timeout=10000)
                if a is None or len(a) != 1:
                    print(f"[WARN] key {k} trace {n}: no ack")
                    continue
                traces[k, n] = scope.get_last_trace()
            print(f"[PROG] key {k+1}/{K}")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass

    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "decimate": args.decimate,
        "samples": T,
        "samples_per_cycle": 4 / args.decimate,
        "target": "ntruplus768",
        "phase": "1-timeline",
        "K": K, "N": N,
    }
    np.savez_compressed(args.output, traces=traces,
                         meta=np.array(meta, dtype=object))
    print(f"[OK] saved → {args.output}  shape={traces.shape}")

    # quick analysis: cross-key SNR profile to find hot regions on full timeline
    K, N, T = traces.shape
    mu_k = traces.mean(axis=1)
    var_within = traces.var(axis=1)
    snr = mu_k.var(axis=0) / np.maximum(var_within.mean(axis=0), 1e-12)
    cycles_per_sample = args.decimate / 4
    print(f"\n[ANALYSIS] full decap snapshot, ~{T * cycles_per_sample:.0f} core cycles total")
    # find regions of contiguous high SNR
    thr = np.percentile(snr, 99)
    above = snr > thr
    boundaries = np.where(np.diff(above.astype(int)) != 0)[0]
    if above[0]:
        boundaries = np.r_[0, boundaries]
    if above[-1]:
        boundaries = np.r_[boundaries, len(snr) - 1]
    regs = []
    for i in range(0, len(boundaries) - 1, 2):
        a = int(boundaries[i]); b = int(boundaries[i + 1])
        if b - a < 20:
            continue
        regs.append((a, b, snr[a:b + 1].mean()))
    regs.sort(key=lambda r: -r[2])
    print(f"[ANALYSIS] top hot regions (samples → core-cycles, mean SNR):")
    for a, b, m in regs[:8]:
        print(f"  samples [{a:5d}, {b:5d}]  cycles [{int(a*cycles_per_sample):5d}, "
              f"{int(b*cycles_per_sample):5d}]  mean SNR = {m:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
