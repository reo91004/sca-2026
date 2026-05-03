#!/usr/bin/env python3
"""'Z' (indcpa_dec only) 와 'X' (sk dump) 명령 검증 + throughput.

흐름:
  1) F (keygen) → pk_fp16 출력
  2) X 4번 호출 → sk PKE 영역 (128B) 받아 ternary 디코드 (예상: HS=70, balanced)
  3) M (μ=zero) → ct_inj 채움
  4) D 50 회 → throughput (full crypto_kem_dec)
  5) Z 50 회 → throughput (indcpa_dec only) + µ' 비트 검증
  6) D vs Z throughput ratio 출력
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

from host.chosen_ct import _MU_PATTERNS, reset_target  # noqa: E402


def _ss_w(target, cmd, payload=b""):
    target.simpleserial_write(cmd, bytes(payload))


def _ss_ack(target, n, timeout_ms=10000):
    ack = target.simpleserial_read("r", n, timeout=timeout_ms)
    if ack is None or len(ack) != n:
        raise RuntimeError(f"ack len {-1 if ack is None else len(ack)} != {n}")
    return bytes(ack)


def unpack_sx(packed: bytes) -> np.ndarray:
    """SMAUG-T Sx_to_bytes 의 역 — 4 ternary coeff per byte, 2-bit each.
    인코딩 가설 (대부분의 SMAUG 류 구현):
      0  → 00, +1 → 01, -1 → 11   (또는 11 → -1, 10 unused)
    256-coeff poly = 64B. 4-coeff per byte, LSB pair = coeff index 4i.
    """
    out = np.zeros(len(packed) * 4, dtype=np.int8)
    for bi, b in enumerate(packed):
        for k in range(4):
            two = (b >> (k * 2)) & 0x3
            if two == 0: out[bi * 4 + k] = 0
            elif two == 1: out[bi * 4 + k] = +1
            elif two == 3: out[bi * 4 + k] = -1
            else: out[bi * 4 + k] = 0xFF  # unexpected (10) — 인코딩 가설 깨짐
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", type=int, default=50)
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--serial", default=None)
    args = p.parse_args()

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(args.serial)
    print(f"[INFO] sn={sn}")
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = 24400
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = 25.0
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()
    time.sleep(0.3)

    try:
        # 1) F
        print("\n[1] F (keygen)")
        _ss_w(target, "F"); pk_fp = _ss_ack(target, 16)
        print(f"  pk_fp16 = {pk_fp.hex()}")

        # 2) X × 4
        print("\n[2] X × 4 (sk PKE dump)")
        sk_raw = b""
        for idx in range(4):
            _ss_w(target, "X", bytes([idx]))
            chunk = _ss_ack(target, 32)
            sk_raw += chunk
            print(f"  X[{idx}] = {chunk.hex()}")
        print(f"  sk_raw 128B = {sk_raw[:32].hex()}…")
        # 디코드 — PKE sk = 두 polyvec, 각 64B = 256 coeffs
        s0 = unpack_sx(sk_raw[:64])
        s1 = unpack_sx(sk_raw[64:128])
        print(f"  s[0] hist: -1={int((s0==-1).sum())}, 0={int((s0==0).sum())}, "
              f"+1={int((s0==+1).sum())}, ?={int((s0==0xFF).sum())}")
        print(f"  s[1] hist: -1={int((s1==-1).sum())}, 0={int((s1==0).sum())}, "
              f"+1={int((s1==+1).sum())}, ?={int((s1==0xFF).sum())}")
        nz = int((s0 != 0).sum() + (s1 != 0).sum())
        print(f"  total nonzero = {nz}  (expected HS=70 for smaug1)")
        print(f"  s[0][0..15] = {s0[:16].tolist()}")
        print(f"  s[0][50] = {int(s0[50])}, s[0][100] = {int(s0[100])}, "
              f"s[0][200] = {int(s0[200])}")

        # 3) M (μ=zero)
        print("\n[3] M (μ=zero)")
        _ss_w(target, "M", _MU_PATTERNS["zero"])
        ct_fp = _ss_ack(target, 16)
        print(f"  ct_fp16 = {ct_fp.hex()}")

        # 4) D throughput
        print(f"\n[4] D × {args.n} (full crypto_kem_dec)")
        # warm-up
        for _ in range(3):
            scope.arm(); _ss_w(target, "D")
            scope.capture(); _ = _ss_ack(target, 1)
        t0 = time.time()
        for _ in range(args.n):
            scope.arm(); _ss_w(target, "D")
            if scope.capture(): print(' [scope timeout]'); continue
            _ = _ss_ack(target, 1)
            _ = scope.get_last_trace()
        d_elapsed = time.time() - t0
        d_rate = args.n / d_elapsed
        print(f"  {d_elapsed:.2f}s, {d_rate:.3f} tr/s")

        # 5) Z throughput + µ' 비트 확인
        print(f"\n[5] Z × {args.n} (indcpa_dec only)")
        for _ in range(3):
            scope.arm(); _ss_w(target, "Z")
            scope.capture(); _ = _ss_ack(target, 16)
        t0 = time.time()
        mu_first = None
        for i in range(args.n):
            scope.arm(); _ss_w(target, "Z")
            if scope.capture(): print(' [scope timeout]'); continue
            mp = _ss_ack(target, 16)
            _ = scope.get_last_trace()
            if i == 0: mu_first = mp
        z_elapsed = time.time() - t0
        z_rate = args.n / z_elapsed
        print(f"  {z_elapsed:.2f}s, {z_rate:.3f} tr/s")
        if mu_first is not None:
            bits = []
            for byte in mu_first:
                for b in range(8):
                    bits.append((byte >> b) & 1)
            print(f"  µ'[0..15] = {mu_first.hex()}")
            print(f"  µ'_i bits[0..15] = {bits[:16]}")
            print(f"  µ' Hamming weight (out of 128) = {sum(bits)}")
            # 정상 ct (M-injected μ=zero, deterministic seed=0) 의 PKE.dec 결과:
            # 1-δ 확률로 µ' = μ = zero. 즉 µ' 모두 0 이어야 함.

        # 6) 비교
        print("\n=== 결과 ===")
        print(f"  D 'full dec'  : {d_rate:.3f} tr/s")
        print(f"  Z 'indcpa_dec': {z_rate:.3f} tr/s")
        print(f"  속도비 Z/D    : {z_rate/d_rate:.2f}x")
        print(f"  추정 — 256위치×2α×N trace full sweep:")
        for N in (16, 64, 128, 256, 500):
            d_h = (256*2*N) / d_rate / 3600
            z_h = (256*2*N) / z_rate / 3600
            print(f"    N={N:4d}: D={d_h:6.2f}h, Z={z_h:6.2f}h")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
