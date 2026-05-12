#!/usr/bin/env python3
"""Phase 3 scout — selected-lane chosen-CT capture on a single (known) key.

For a fixed sk on the board, send chosen ciphertexts whose NTT-domain
representation has exactly one non-zero coefficient at position 4*lane+slot
with value γ. The decap's `poly_basemul(&m1, &c, &f)` then computes
m1[4*lane+i] = γ · f_ntt[4*lane+i] mod q for i = 0..3, with all other
coefficients of m1 being zero.

Capture matrix (default):
    L lanes × G γ-values × N traces = 8 × 6 × 10 = 480 traces.

After capture we save (a) raw traces, (b) the host-known sk so labels can
be computed for the scout analysis, (c) per-design metadata. Phase 3
inference under the threat model uses sk only at the *profiling* stage,
not for held-out attack.

Note on timing: decimate=4 (1 sample/core-cycle). 24400 samples covers
~24400 cycles, which from the Phase 1 timeline scan covers the early
poly_frombytes calls plus the bulk of poly_basemul and beyond.
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
from ntruplus.params import N_LANES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-L", "--lanes", type=str, default="0,1,2,3,8,32,80,128",
                   help="comma-separated NTT-domain lane indices to test")
    p.add_argument("-G", "--gammas", type=str, default="1,2,4,16,64,256",
                   help="comma-separated γ values (centered residue mod 3457)")
    p.add_argument("-N", "--traces", type=int, default=10,
                   help="traces per (lane, γ) design")
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--slot", type=int, default=0,
                   help="lane slot (0..3) for the active coefficient")
    p.add_argument("-o", "--output", type=Path,
                   default=ROOT / "traces/ntruplus768/phase3/scout.npz")
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--timeout-ms", type=int, default=10000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lanes = [int(x) for x in args.lanes.split(",")]
    gammas = [int(x) for x in args.gammas.split(",")]
    for L in lanes:
        if not (0 <= L < N_LANES):
            raise ValueError(f"lane {L} out of [0, {N_LANES})")
    L_count, G_count, N = len(lanes), len(gammas), args.traces
    print(f"[INFO] design = {L_count} lanes × {G_count} γ × {N} traces "
          f"= {L_count * G_count * N} captures")

    sn = pick_serial(None)
    print(f"[INFO] CW1173 sn={sn}")
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = args.samples
    scope.adc.offset = 0
    scope.adc.decimate = args.decimate
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = args.gain_db
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    scope.io.nrst = "low"; time.sleep(0.05)
    scope.io.nrst = "high_z"; time.sleep(0.5)

    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()

    T = args.samples
    traces = np.empty((L_count, G_count, N, T), dtype=np.float32)
    mismatches = np.zeros((L_count, G_count, N), dtype=np.uint8)

    timeouts = 0
    t0 = time.time()
    try:
        # 1 keypair, persisted, sk dumped for label computation (calibration).
        pk_fp = nti.keygen_persistent(target, timeout_ms=args.timeout_ms)
        print(f"[INFO] sha3(pk)[:16] = {pk_fp.hex()}")
        sk_blob = nti.dump_sk(target, timeout_ms=args.timeout_ms)
        pk_blob = nti.dump_pk(target, timeout_ms=args.timeout_ms)

        for li, lane in enumerate(lanes):
            for gi, gamma in enumerate(gammas):
                ct_bytes, _ = selected_lane(lane=lane, gamma=gamma, slot=args.slot)
                # inject once per (lane, γ); only the trace varies
                fp_board = nti.inject_ct(target, ct_bytes,
                                          timeout_ms=args.timeout_ms,
                                          verify_fingerprint=True)
                for n in range(N):
                    scope.arm()
                    target.simpleserial_write("D", b"")
                    if scope.capture():
                        timeouts += 1
                        print(f"[WARN] lane={lane} γ={gamma} n={n}: scope timeout")
                        continue
                    ack = target.simpleserial_read("r", 1, timeout=args.timeout_ms)
                    if ack is None or len(ack) != 1:
                        timeouts += 1
                        print(f"[WARN] lane={lane} γ={gamma} n={n}: no ack")
                        continue
                    traces[li, gi, n] = scope.get_last_trace()
                    mismatches[li, gi, n] = ack[0]
            print(f"[PROG] lane {li+1}/{L_count} (idx={lane}) "
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
        "samples": T,
        "decimate": args.decimate,
        "samples_per_cycle": 4 / args.decimate,
        "gain_db": args.gain_db,
        "target": "ntruplus768",
        "phase": "3-scout",
        "lanes": lanes, "gammas": gammas, "slot": args.slot,
        "L_count": L_count, "G_count": G_count, "N": N,
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
        traces=traces, mismatches=mismatches,
        sk_blob=np.frombuffer(sk_blob, dtype=np.uint8),
        pk_blob=np.frombuffer(pk_blob, dtype=np.uint8),
        lanes=np.array(lanes, dtype=np.int32),
        gammas=np.array(gammas, dtype=np.int32),
        meta=np.array(meta, dtype=object),
    )
    print(f"[OK] saved → {args.output}  shape={traces.shape}  "
          f"mismatches={int(mismatches.sum())}  timeouts={timeouts}")
    return 0 if timeouts == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
