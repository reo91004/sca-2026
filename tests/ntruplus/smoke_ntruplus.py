#!/usr/bin/env python3
"""ntruplus768 펌웨어 baseline + chosen-CT plumbing smoke.

A 그룹 (k/e/d/p) — 정상 ct 파이프라인 mismatch=0 인지.
B 그룹 (F/B/I/L/D) — pk dump 후 host 가 만든 selected-lane chosen-CT 를
   inject 하고, 'L' 의 sha3_256(ct_inj)[:16] 가 host 가 재계산한
   sha3_256(host_ct)[:16] 와 일치하는지 검증 (codec parity gate).

이 smoke 는 ntruplus768 hex 가 이미 플래시된 상태에서 동작한다.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

import chipwhisperer as cw  # noqa: E402

from cw_serial import pick_serial  # noqa: E402
from ntruplus import inject as nti  # noqa: E402
from ntruplus.chosen import selected_lane  # noqa: E402
from ntruplus.params import CIPHERTEXTBYTES  # noqa: E402


def hexdump16(b: bytes | None) -> str:
    if b is None:
        return "<None>"
    return " ".join(f"{x:02x}" for x in b[:16])


def expect(label: str, ok: bool, *, fatal: bool = False) -> None:
    tag = "OK " if ok else "FAIL"
    print(f"  [{tag}] {label}")
    if fatal and not ok:
        sys.exit(1)


def main() -> int:
    sn = pick_serial(None)
    print(f"[INFO] CW1173 sn={sn}")
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.basic_mode = "rising_edge"
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"

    # nRST toggle + boot wait — capture.py 와 동일. 이 reset 없이는
    # 첫 simpleserial_read 가 즉시 timeout 한다.
    scope.io.nrst = "low"
    time.sleep(0.05)
    scope.io.nrst = "high_z"
    time.sleep(0.5)

    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = 38400
    target.flush()

    try:
        # --- A 그룹 ---
        print("[A] baseline smoke")

        target.simpleserial_write("k", b"")
        pk_fp = target.simpleserial_read("r", 16, timeout=5000)
        expect(f"k → pk[0:16] = {hexdump16(pk_fp)}", pk_fp is not None and len(pk_fp) == 16, fatal=True)

        target.simpleserial_write("e", b"")
        ct_fp = target.simpleserial_read("r", 16, timeout=5000)
        expect(f"e → ct[0:16] = {hexdump16(ct_fp)}", ct_fp is not None and len(ct_fp) == 16, fatal=True)

        target.simpleserial_write("d", b"")
        d_resp = target.simpleserial_read("r", 1, timeout=5000)
        expect(f"d → mismatch = {d_resp[0] if d_resp else 'None'} (0=OK)",
               d_resp is not None and d_resp[0] == 0, fatal=True)

        target.simpleserial_write("p", b"")
        p_resp = target.simpleserial_read("r", 1, timeout=10000)
        expect(f"p → mismatch = {p_resp[0] if p_resp else 'None'} (0=OK)",
               p_resp is not None and p_resp[0] == 0, fatal=True)

        # --- B 그룹: F → B → I → L → D ---
        # F : 새 keypair 영속화. fingerprint 응답.
        # B : pk chunk dump (chunks 27 = 864 / 32) — 첫 두 chunk 읽고 비교
        # I : ct_inj 27 chunks 모두 0 으로 채움. (실제 정합 ct 는 host-side
        #     reference 코드 없이는 못 만든다. 여기선 보드 응답이 형식 OK,
        #     이후 'D' 가 mismatch=1 로 응답하는지만 확인 — 0×N ct 는 거의
        #     확실히 invalid → ss_dec ≠ ss_enc.)
        # L : sha3_256(ct_inj)[0:16]
        # D : decap (ct_inj=zeros), trigger ON. mismatch flag 1B
        print("[B] chosen-CT plumbing smoke")

        target.simpleserial_write("F", b"")
        f_resp = target.simpleserial_read("r", 16, timeout=5000)
        expect(f"F → sha3_256(pk)[0:16] = {hexdump16(f_resp)}",
               f_resp is not None and len(f_resp) == 16, fatal=True)

        # pk chunk dump (idx=0)
        target.simpleserial_write("B", bytes([0]))
        b0 = target.simpleserial_read("r", 32, timeout=5000)
        expect(f"B[0] → 32B chunk = {hexdump16(b0)}",
               b0 is not None and len(b0) == 32, fatal=True)

        # CT inject : host 가 만든 selected-lane chosen-CT.
        #   c_ntt[4*lane] = gamma, 그 외 0  → board basemul 시 lane 만 살림.
        # Phase 0b 의 핵심: host 의 12-bit pack 결과가 board 의 poly_frombytes
        # 후 다시 sha3_256 하면 동일한 [:16] 지문을 만들어야 한다 (codec
        # parity gate).
        host_ct, _ = selected_lane(lane=3, gamma=42, slot=0)
        N_CT_CHUNKS = nti.N_CT_CHUNKS  # 36 for ntruplus768
        for idx in range(N_CT_CHUNKS):
            chunk = host_ct[idx * 32:(idx + 1) * 32]
            target.simpleserial_write("I", bytes([idx]) + chunk)
            r = target.simpleserial_read("r", 1, timeout=5000)
            if r is None or r[0] != 0:
                expect(f"I[{idx}] → status = {r[0] if r else 'None'}", False, fatal=True)
        expect(f"I × {N_CT_CHUNKS} chunks (selected-lane) all status=0", True)

        target.simpleserial_write("L", b"")
        l_resp = target.simpleserial_read("r", 16, timeout=5000)
        expect(f"L → board sha3_256(ct_inj)[:16] = {hexdump16(l_resp)}",
               l_resp is not None and len(l_resp) == 16, fatal=True)

        host_fp = hashlib.sha3_256(host_ct).digest()[:16]
        expect(f"L parity: host sha3_256(host_ct)[:16] = {hexdump16(host_fp)}",
               bytes(l_resp) == host_fp, fatal=True)

        target.simpleserial_write("D", b"")
        d_inj = target.simpleserial_read("r", 1, timeout=10000)
        # ss_enc was 0-init by 'F'. chosen-CT decapsulation produces some
        # ss_dec that very likely differs → mismatch=1. 형식만 확인.
        expect(f"D (selected-lane ct) → mismatch flag = {d_inj[0] if d_inj else 'None'} (형식만)",
               d_inj is not None and len(d_inj) == 1, fatal=True)

        # X : sk chunk dump (73 chunks for ntruplus768 = 2336 / 32). 첫 chunk 만.
        target.simpleserial_write("X", bytes([0]))
        x0 = target.simpleserial_read("r", 32, timeout=5000)
        expect(f"X[0] → 32B sk chunk = {hexdump16(x0)}",
               x0 is not None and len(x0) == 32, fatal=True)

        print("\n[OK] all smoke checks passed")
        return 0

    finally:
        try:
            target.dis()
        except Exception:
            pass
        try:
            scope.dis()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
