#!/usr/bin/env python3
"""Phase 4 — wide-γ capture for structural multi-comparison ceiling break.

Combines HW=1 (12) and HW=2 (66) γ sets for G=78. With Q=3457:
   null-max(|corr|) ≈ √(2·log(3456)/78) ≈ 0.456
This is below the |corr|=1 saturation, so attack-valid CPA can in
principle distinguish true f as N grows. n16 SNR diagnostic showed
SNR median 0.112; for V2 (mean-over-N) corr to clear 0.456 ceiling we
need N > null_max² / (snr²·(1−null_max²)) ≈ 0.208 / (0.013·0.792) ≈ 20.

Default K=1, L=1 (lane 0 only — strongest leak), N=32 → 2496 traces.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from cw_serial import pick_serial  # noqa: E402
from ntruplus import inject as nti  # noqa: E402
from ntruplus.chosen import selected_lane  # noqa: E402


def gammas_hw1_hw2() -> list[int]:
    g1 = [1 << i for i in range(12)]                          # 12
    g2 = [(1 << i) + (1 << j) for i in range(12) for j in range(i + 1, 12)]  # 66
    return sorted(set(g1 + g2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-K", "--keys", type=int, default=1)
    p.add_argument("-L", "--lanes", type=str, default="0")
    p.add_argument("-N", "--traces", type=int, default=32)
    p.add_argument("--slot", type=int, default=0)
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("-o", "--output", type=Path,
                   default=ROOT / "traces/ntruplus768/phase3/wideg_lane0.npz")
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--timeout-ms", type=int, default=10000)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lanes = [int(x) for x in args.lanes.split(",")]
    gammas = gammas_hw1_hw2()
    K, L, G, N = args.keys, len(lanes), len(gammas), args.traces
    print(f"[INFO] capture matrix: K={K} × L={L} × G={G} (HW1+HW2) × N={N} = "
          f"{K*L*G*N} traces total")
    print(f"[INFO] γ list: {gammas[:5]} ... {gammas[-5:]} (G={G})")

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
    scope.io.nrst = "low"; time.sleep(0.05)
    scope.io.nrst = "high_z"; time.sleep(0.5)

    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()

    T = args.samples
    traces = np.empty((K, L, G, N, T), dtype=np.float32)
    sk_blobs = np.zeros((K, 2336), dtype=np.uint8)
    pk_blobs = np.zeros((K, 1152), dtype=np.uint8)

    timeouts = 0
    t0 = time.time()
    try:
        for ki in range(K):
            pk_fp = nti.keygen_persistent(target, timeout_ms=args.timeout_ms)
            sk_blobs[ki] = list(nti.dump_sk(target, timeout_ms=args.timeout_ms))
            pk_blobs[ki] = list(nti.dump_pk(target, timeout_ms=args.timeout_ms))
            print(f"[INFO] key {ki+1}/{K}  pk_fp={pk_fp.hex()[:16]}…")
            for li, lane in enumerate(lanes):
                for gi, gamma in enumerate(gammas):
                    ct, _ = selected_lane(lane=lane, gamma=gamma, slot=args.slot)
                    nti.inject_ct(target, ct,
                                   timeout_ms=args.timeout_ms,
                                   verify_fingerprint=False)
                    for n in range(N):
                        scope.arm()
                        target.simpleserial_write("D", b"")
                        if scope.capture():
                            timeouts += 1; continue
                        ack = target.simpleserial_read("r", 1,
                                                        timeout=args.timeout_ms)
                        if ack is None or len(ack) != 1:
                            timeouts += 1; continue
                        traces[ki, li, gi, n] = scope.get_last_trace()
                print(f"[PROG] key {ki+1}/{K} lane {li+1}/{L}  "
                      f"elapsed={time.time()-t0:.1f}s")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass

    fw_path = ROOT / ("firmware/simpleserial-ntruplus/"
                      "simpleserial-ntruplus-CW308_STM32F4-ntruplus768.hex")
    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope_sn": sn,
        "samples": T, "decimate": args.decimate,
        "samples_per_cycle": 4 / args.decimate,
        "target": "ntruplus768", "phase": "4-wide-gamma",
        "K": K, "lanes": lanes, "gammas": gammas, "N": N,
        "L_count": L, "G_count": G,
        "n_timeouts": timeouts,
        "fw_hex": str(fw_path),
        "fw_sha256": hashlib.sha256(fw_path.read_bytes()).hexdigest(),
        "git_rev": __import__("subprocess").check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]
        ).decode().strip(),
        "calibration_only_sk": True,
        "slot_used": args.slot,
    }
    np.savez_compressed(
        args.output,
        traces=traces, sk_blobs=sk_blobs, pk_blobs=pk_blobs,
        lanes=np.array(lanes, dtype=np.int32),
        gammas=np.array(gammas, dtype=np.int32),
        meta=np.array(meta, dtype=object),
    )
    print(f"[OK] saved → {args.output}  shape={traces.shape}  "
          f"timeouts={timeouts}  total={time.time()-t0:.1f}s")
    return 0 if timeouts == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
