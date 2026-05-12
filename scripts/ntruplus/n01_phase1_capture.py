#!/usr/bin/env python3
"""Phase 1 — natural `D` capture map for NTRU+768.

For each of K keys, take one valid encapsulation (`e`) and N decap traces
(`d`) on the resulting (ct, sk). The decap path on the board exercises:

    poly_frombytes(c)
    poly_frombytes(f)
    poly_frombytes(hinv)
    poly_basemul(m1, c, f)              ← attack hook B
    poly_invntt(m1)
    poly_crepmod3(m1, m1)
    poly_ntt(m2 = m1)
    poly_sub; poly_basemul(r2, c, hinv)  ← FO recompute step 1
    poly_tobytes(buf1, r2); hash_g
    poly_sotp_decode                     ← attack hook D
    hash_h; poly_cbd1; poly_ntt; poly_tobytes; verify

The trigger window covers the entire decap. CW-Lite's 24400-sample budget at
ADC=4·clkgen will only see a fraction of the workload; this script also
optionally walks `--offset` to map the full timeline.

Outputs an npz with shape (K, N, T) traces, plus per-key sk + pk dumps for
later calibration use (clearly tagged as calibration-only — do not use as
oracle labels under threat model).

Usage:
    python3 scripts/n01_phase1_capture.py -K 8 -N 20 -s 24400 \
        -o traces/ntruplus768/phase1/d_map.npz
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-K", "--keys", type=int, default=8,
                   help="number of fresh keypairs (default 8)")
    p.add_argument("-N", "--traces", type=int, default=20,
                   help="traces per key (fixed valid CT) (default 20)")
    p.add_argument("-s", "--samples", type=int, default=24400,
                   help="ADC samples (default 24400 == CW-Lite max)")
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--offset", type=int, default=0,
                   help="ADC trigger offset in samples (default 0)")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0,
                   help="metadata seed (firmware RNG is hardware-seeded; this "
                        "is for replay tracking only).")
    return p.parse_args()


def setup_scope(sn: str, samples: int, gain_db: float, offset: int):
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = samples
    scope.adc.offset = offset
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = gain_db
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    scope.io.nrst = "low"; time.sleep(0.05)
    scope.io.nrst = "high_z"; time.sleep(0.5)
    return scope


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sn = pick_serial(None)
    print(f"[INFO] CW1173 sn={sn}")

    scope = setup_scope(sn, args.samples, args.gain_db, args.offset)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()

    K = args.keys
    N = args.traces
    T = args.samples

    traces = np.empty((K, N, T), dtype=np.float32)
    mismatches = np.zeros((K, N), dtype=np.uint8)
    pk_fps = np.zeros((K, 16), dtype=np.uint8)
    ct_fps = np.zeros((K, 16), dtype=np.uint8)
    sk_blobs = np.zeros((K, 2336), dtype=np.uint8)   # ntruplus768 SECRETKEYBYTES
    pk_blobs = np.zeros((K, 1152), dtype=np.uint8)

    t0 = time.time()
    timeouts = 0

    try:
        for k in range(K):
            # Fresh keypair (uses board RNG; pk fingerprint we read here is
            # purely a sanity check that responses came back before the
            # capture loop starts).
            target.simpleserial_write("k", b"")
            pk_fp = target.simpleserial_read("r", 16, timeout=args.timeout_ms)
            if pk_fp is None or len(pk_fp) != 16:
                raise RuntimeError(f"key {k}: bad k-ack")
            pk_fps[k] = list(pk_fp)

            # Bring the actual pk + sk to host (calibration only).
            pk_blobs[k] = list(nti.dump_pk(target, timeout_ms=args.timeout_ms))
            sk_blobs[k] = list(nti.dump_sk(target, timeout_ms=args.timeout_ms))

            target.simpleserial_write("e", b"")
            ct_fp = target.simpleserial_read("r", 16, timeout=args.timeout_ms)
            if ct_fp is None or len(ct_fp) != 16:
                raise RuntimeError(f"key {k}: bad e-ack")
            ct_fps[k] = list(ct_fp)

            for n in range(N):
                scope.arm()
                target.simpleserial_write("d", b"")
                if scope.capture():
                    timeouts += 1
                    print(f"[WARN] key {k} trace {n}: scope timeout")
                    continue
                ack = target.simpleserial_read("r", 1, timeout=args.timeout_ms)
                if ack is None or len(ack) != 1:
                    timeouts += 1
                    print(f"[WARN] key {k} trace {n}: missing ack")
                    continue
                traces[k, n] = scope.get_last_trace()
                mismatches[k, n] = ack[0]
            elapsed = time.time() - t0
            print(f"[PROG] key {k+1}/{K}  pk_fp={bytes(pk_fps[k][:8]).hex()} "
                  f"ct_fp={bytes(ct_fps[k][:8]).hex()} "
                  f"mism={int(mismatches[k].sum())} "
                  f"({elapsed:.1f}s)")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass

    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope_sn": sn,
        "samples": T,
        "offset": args.offset,
        "gain_db": args.gain_db,
        "target": "ntruplus768",
        "phase": "1",
        "label": "natural_D_map",
        "K": K, "N": N,
        "n_timeouts": timeouts,
        "fw_hex": "firmware/simpleserial-ntruplus/simpleserial-ntruplus-CW308_STM32F4-ntruplus768.hex",
        "fw_sha256": hashlib.sha256(
            (ROOT / "firmware/simpleserial-ntruplus/simpleserial-ntruplus-CW308_STM32F4-ntruplus768.hex").read_bytes()
        ).hexdigest(),
        "git_rev": __import__("subprocess").check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]
        ).decode().strip(),
        "seed": args.seed,
        "calibration_only_sk": True,    # sk_blobs are calibration; do NOT use
                                        # for inference under threat model.
    }
    np.savez_compressed(
        args.output,
        traces=traces, mismatches=mismatches,
        pk_fps=pk_fps, ct_fps=ct_fps,
        sk_blobs=sk_blobs, pk_blobs=pk_blobs,
        meta=np.array(meta, dtype=object),
    )
    print(f"[OK] saved → {args.output}  shape={traces.shape}  to={timeouts}")
    return 0 if timeouts == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
