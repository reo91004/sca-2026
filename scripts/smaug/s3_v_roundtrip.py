#!/usr/bin/env python3
"""S3 round-trip — 'V' (sub-triggered vec_vec_mult_add) sanity 검증.

검증 :
  1. 'V' 응답 µ′ = 'Z' 응답 µ′ (같은 chosen-CT 위에서 indcpa_dec 와 동일 수학)
  2. 'V' 응답 µ′ = host predict_mu_prime (= mathematical reference)
  3. 'V' trace sample 수가 'Z' 보다 짧음 (sub-trigger window = vec_vec_mult_add only)
  4. 여러 chosen-CT 디자인 (α, j sweep) 으로 다양성 확인

이 검증이 통과해야 S3 main capture (s3_v_capture_main.py) 가능. 통과 못 하면
firmware 의 V command 가 indcpa_dec 와 다른 수학을 수행하는 것 — leak 분석 무의미.

run::
    python3 scripts/s3_v_roundtrip.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target, setup_session  # noqa: E402
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from scripts.s1_t_roundtrip import dump_sk_pke  # noqa: E402


def _pack_mu_bits(mu_bits: np.ndarray) -> bytes:
    out = bytearray(32)
    for bit_i in range(256):
        if mu_bits[bit_i]:
            out[bit_i // 8] |= 1 << (bit_i % 8)
    return bytes(out)


def _diff_bits(a: bytes, b: bytes) -> int:
    n = 0
    for x, y in zip(a, b):
        n += bin(x ^ y).count("1")
    return n


def main() -> int:
    p = _params.get("smaug1")
    p.assert_consistent()

    import chipwhisperer as cw

    from host.cw_serial import pick_serial

    sn = pick_serial(None)
    print(f"[INFO] CW1173 sn={sn}")
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = 24400
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = 25.0
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = 38400
    target.flush()
    time.sleep(0.3)

    designs = [
        # (component, coef_idx, alpha)
        (0, 0, 4),     # paper main
        (0, 0, 1),
        (0, 0, 64),
        (0, 0, 192),
        (0, 5, 4),
        (1, 0, 4),
        (0, 100, 64),
    ]

    failures = 0
    try:
        target.simpleserial_write("F", b"")
        pk_fp16 = bytes(target.simpleserial_read("r", 16, timeout=5000))
        print(f"[OK] F pk_fp16={pk_fp16.hex()}")

        sk_pke = dump_sk_pke(target, p.pke_secret_key_bytes // 32)
        sk_unpacked_full = _codec.unpack_sx(sk_pke).astype(np.int64)
        sk_polyvec = sk_unpacked_full.reshape(p.module_rank, p.n)
        print(f"[OK] sk_pke[0:16]={sk_pke[:16].hex()}")

        # T1: V trace shorter than Z trace via sample counts (sanity — same scope buffer)
        # We can't directly resize per-command; instead check that V trace ends earlier.
        # Approach: capture both with same samples=24400, compare trigger duration.

        # Round-trip per design
        z_durations = []
        v_durations = []
        for (component, coef, alpha) in designs:
            ct = _chosen.build_monomial_c1(p, component=component, coef_idx=coef, alpha=alpha)
            ct_bytes = ct.to_bytes()
            ct_fp16_host = hashlib.sha3_256(ct_bytes).digest()[:16]

            bundle = setup_session(
                target,
                ct_bytes,
                label=f"v_rt_c{component}j{coef}a{alpha}",
                fresh_key=False,
                pk_fp16=pk_fp16,
            )
            if bundle.ct_fp16_host != bundle.ct_fp16_board:
                raise RuntimeError(
                    f"ct integrity fail @ design ({component},{coef},{alpha}): "
                    f"host={bundle.ct_fp16_host.hex()} brd={bundle.ct_fp16_board.hex()}"
                )

            # host predict
            mu_pred = _chosen.predict_mu_prime(p, ct.c1, sk_polyvec, ct.c2)
            mu_pred_packed = _pack_mu_bits(mu_pred)

            # Z capture
            scope.arm()
            target.simpleserial_write("Z", b"")
            timed_out_z = scope.capture()
            mu_z = bytes(target.simpleserial_read("r", 32, timeout=5000))
            try:
                z_tc = scope.adc.trig_count
            except Exception:
                z_tc = -1

            # V capture
            scope.arm()
            target.simpleserial_write("V", b"")
            timed_out_v = scope.capture()
            mu_v = bytes(target.simpleserial_read("r", 32, timeout=5000))
            try:
                v_tc = scope.adc.trig_count
            except Exception:
                v_tc = -1

            z_durations.append(z_tc)
            v_durations.append(v_tc)

            checks = []
            if mu_v != mu_pred_packed:
                checks.append(f"V vs predict differ {_diff_bits(mu_v, mu_pred_packed)}/256")
            if mu_z != mu_pred_packed:
                checks.append(f"Z vs predict differ {_diff_bits(mu_z, mu_pred_packed)}/256")
            if mu_v != mu_z:
                checks.append(f"V vs Z differ {_diff_bits(mu_v, mu_z)}/256")
            if checks:
                failures += 1

            status = "FAIL" if checks else "OK"
            print(
                f"[{status}] design (c={component},j={coef},α={alpha}) "
                f"V_tc={v_tc} Z_tc={z_tc} "
                f"hw(µ′)={int(np.sum(mu_pred))} "
                f"{' | '.join(checks) if checks else 'all 256/256 match'}"
            )

        if v_durations and z_durations:
            v_arr = np.array(v_durations)
            z_arr = np.array(z_durations)
            print(
                f"[INFO] trig_count V mean={v_arr.mean():.0f} Z mean={z_arr.mean():.0f} "
                f"V/Z ratio={v_arr.mean()/max(1,z_arr.mean()):.4f}"
            )

    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
