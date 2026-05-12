#!/usr/bin/env python3
"""Phase 3 — multi-slot capture (slot ∈ {0, 1, 2, 3}) for higher label entropy.

For each lane, sweep all 4 NTT-domain slots within the lane. Each slot
gives a label of HW(γ·f_ntt[4·lane + slot] mod q), so a single lane
yields 4× more (slot, γ) labels than a slot=0-only design.

Combined with G=12 HW=1 γ values, each lane gets 48 distinct labels —
enough to overcome the lane-specific label-variance bottleneck observed
in v3 (lanes 1, 2, 3, 8, 32 had label var < 2.0 with slot=0 only).

Used after multikey held-out shows partial recovery; this script
broadens label coverage for the next capture round.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-K", "--keys", type=int, default=4)
    p.add_argument("-L", "--lanes", type=str, default="0,2,8,32,80,128")
    p.add_argument("-G", "--gammas", type=str,
                   default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    p.add_argument("-S", "--slots", type=str, default="0,1,2,3")
    p.add_argument("-N", "--traces", type=int, default=6)
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("-o", "--output", type=Path,
                   default=ROOT / "traces/ntruplus768/phase3/multislot.npz")
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--timeout-ms", type=int, default=10000)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lanes = [int(x) for x in args.lanes.split(",")]
    gammas = [int(x) for x in args.gammas.split(",")]
    slots = [int(x) for x in args.slots.split(",")]
    K, L, G, S, N = args.keys, len(lanes), len(gammas), len(slots), args.traces
    print(f"[INFO] capture matrix: K={K} × L={L} × G={G} × S={S} × N={N} = "
          f"{K*L*G*S*N} traces total")

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
    traces = np.empty((K, L, S, G, N, T), dtype=np.float32)
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
                for si, slot in enumerate(slots):
                    for gi, gamma in enumerate(gammas):
                        ct, _ = selected_lane(lane=lane, gamma=gamma, slot=slot)
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
                            traces[ki, li, si, gi, n] = scope.get_last_trace()
                print(f"[PROG] key {ki+1} lane {li+1}/{L}  "
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
        "target": "ntruplus768", "phase": "3-multislot",
        "K": K, "lanes": lanes, "gammas": gammas, "slots": slots, "N": N,
        "L_count": L, "G_count": G, "S_count": S,
        "n_timeouts": timeouts,
        "fw_hex": str(fw_path),
        "fw_sha256": hashlib.sha256(fw_path.read_bytes()).hexdigest(),
        "git_rev": __import__("subprocess").check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]
        ).decode().strip(),
        "calibration_only_sk": True,
    }
    np.savez_compressed(
        args.output,
        traces=traces, sk_blobs=sk_blobs, pk_blobs=pk_blobs,
        lanes=np.array(lanes, dtype=np.int32),
        gammas=np.array(gammas, dtype=np.int32),
        slots=np.array(slots, dtype=np.int32),
        meta=np.array(meta, dtype=object),
    )
    print(f"[OK] saved → {args.output}  shape={traces.shape}  to={timeouts}")
    return 0 if timeouts == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
