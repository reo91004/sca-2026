#!/usr/bin/env python3
"""S1 main capture — fixed (sk, component, idx, alpha) 로 N traces 모은다.

목적
====
poly_mul_acc 의 *처음 ~6100 cycles* 에서 sk content 가 전력 흔적에 보이는지
확인. Toom-Cook 4-way 의 evaluation phase (chunk 분할 + 첫 t_k 평가) 가
captured 영역 안에 들어가야 sk 좌표별 leakage 가 가능.

전략
====
* SAME sk : 'F' 1회 → sk 영속, 'X' 4 회로 sk_pke dump (ground truth).
* SAME (component, idx, alpha) : N traces 의 의미적 input 동일.
* N=500 로 mean trace 의 SNR √500 ≈ 22× 향상 → 단일 trace |t|≈4 noise floor 가
  90+ 정도까지 나오리라 기대.

산출물
======
``traces/s1_main_<seed_tag>.npz``
    - traces (N, 24400) float32
    - ack (N, 32) uint8 — 'T' 응답 (mod p 검증용)
    - sk_pke (128,) uint8 — 'X' 0..3 dump
    - meta : pk_fp16, sk_pke_hex, captured_at, gain, samples, ...

이후 분석은 ``scripts/s1_t_analyze.py`` 로.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from host.smaug.poly_mul import board_response_mod_p, negacyclic_mul_mod_p  # noqa: E402
from scripts.s1_t_roundtrip import dump_sk_pke, issue_t  # noqa: E402


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "-n", "--num-traces", type=int, default=500,
        help="capture 할 trace 수 (기본 500)",
    )
    p.add_argument(
        "-s", "--samples", type=int, default=24400,
        help="trace 당 ADC sample 수 (CW-Lite 최대 24400)",
    )
    p.add_argument(
        "-g", "--gain-db", type=float, default=25.0,
        help="LNA gain dB",
    )
    p.add_argument(
        "--component", type=int, default=0, choices=(0, 1),
        help="'T' component (0 또는 1, smaug1 module_rank=2)",
    )
    p.add_argument(
        "--idx", type=int, default=0,
        help="'T' idx (0..255)",
    )
    p.add_argument(
        "--alpha", type=int, default=1,
        help="'T' alpha (signed int16)",
    )
    p.add_argument(
        "--firmware-hex",
        type=Path,
        default=_REPO
        / "firmware"
        / "simpleserial-smaug"
        / "simpleserial-smaug-CW308_STM32F4.hex",
    )
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--serial", default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="저장 .npz 경로 (default: traces/s1_main_<auto>.npz)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    p = _params.get("smaug1")
    if args.idx < 0 or args.idx >= p.n:
        raise SystemExit(f"--idx {args.idx} out of [0, {p.n})")

    out_path = args.out
    if out_path is None:
        tag = f"c{args.component}_j{args.idx}_a{args.alpha}_n{args.num_traces}"
        out_path = _REPO / "traces" / f"s1_main_{tag}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fw_sha = _file_sha256(args.firmware_hex) if args.firmware_hex.exists() else None
    print(f"[INFO] firmware_hex={args.firmware_hex.name}")
    print(f"[INFO] firmware_sha256={fw_sha}")

    import chipwhisperer as cw  # noqa: E402
    from host.cw_serial import pick_serial  # noqa: E402

    sn = pick_serial(args.serial)
    print(f"[INFO] sn={sn}")
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = args.samples
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = args.gain_db
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"

    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()
    time.sleep(0.3)

    started = time.time()
    try:
        target.simpleserial_write("F", b"")
        pk_fp16_b = target.simpleserial_read("r", 16, timeout=5000)
        if pk_fp16_b is None:
            raise RuntimeError("F ack 없음")
        pk_fp16 = bytes(pk_fp16_b)
        print(f"[OK] F pk_fp16={pk_fp16.hex()}")
        sk_pke = dump_sk_pke(target, p.pke_secret_key_bytes // 32)
        print(f"[OK] X sk_pke[{len(sk_pke)}B]={sk_pke[:8].hex()}…")
        sk_unpacked = _codec.unpack_sx(sk_pke).astype(np.int64)
        sk_polys = [
            sk_unpacked[i * p.n : (i + 1) * p.n] for i in range(p.module_rank)
        ]

        # 사전 sanity : 'T' 1 회 호출 + mod-p 비교
        from host.smaug.poly_mul import monomial_b
        b_pred = monomial_b(args.idx, args.alpha, n=p.n)
        host_modp = negacyclic_mul_mod_p(
            sk_polys[args.component], b_pred, log_p=p.log_p, signed=True
        )[:16]

        resp_first = issue_t(target, args.component, args.idx, args.alpha)
        board_modp = board_response_mod_p(resp_first, log_p=p.log_p, signed=True)
        if not np.array_equal(host_modp, board_modp):
            raise RuntimeError(
                "[FAIL] capture 전 host predict ≠ board 'T' (mod p) — "
                f"host={host_modp.tolist()} brd={board_modp.tolist()}"
            )
        print(f"[OK] sanity 'T' 응답 mod p == host predict (16 좌표)")

        # 본 capture
        traces = np.empty((args.num_traces, args.samples), dtype=np.float32)
        acks = np.zeros((args.num_traces, 32), dtype=np.uint8)
        timeouts = 0
        n_ok = 0
        # 'T' payload : 5B (component, idx_BE, alpha_BE)
        comp_byte = args.component & 0xFF
        idx_BE = bytes([(args.idx >> 8) & 0xFF, args.idx & 0xFF])
        a16 = int(np.int16(args.alpha)) & 0xFFFF
        alpha_BE = bytes([(a16 >> 8) & 0xFF, a16 & 0xFF])
        payload = bytes([comp_byte]) + idx_BE + alpha_BE
        if len(payload) != 5:
            raise AssertionError("payload len != 5")

        for i in range(args.num_traces):
            scope.arm()
            target.simpleserial_write("T", payload)
            if scope.capture():
                timeouts += 1
                print(f"[WARN] scope timeout at trace {i}")
                continue
            ack = target.simpleserial_read("r", 32, timeout=5000)
            if ack is None or len(ack) != 32:
                timeouts += 1
                print(f"[WARN] ack short at trace {i}")
                continue
            traces[i] = scope.get_last_trace().astype(np.float32)
            acks[i] = bytearray(ack)
            n_ok += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - started
                print(
                    f"[PROG] {i+1}/{args.num_traces} "
                    f"({elapsed:.1f}s, {(i+1)/elapsed:.2f} tr/s)"
                )

        # capture 후 'T' 결과 한 번 더 확인 — sk 변경 없음 검증
        resp_last = issue_t(target, args.component, args.idx, args.alpha)
        board_modp_last = board_response_mod_p(resp_last, log_p=p.log_p, signed=True)
        if not np.array_equal(host_modp, board_modp_last):
            raise RuntimeError(
                "[FAIL] capture 후 'T' 응답이 host 와 다름 — sk 손상 의심"
            )
        print(f"[OK] capture 후 sanity 'T' 응답 일관성 유지")

        # 모든 trace 의 ack 가 동일한지 — board 결과가 결정적인지 확인
        unique_acks = np.unique(acks[:n_ok], axis=0)
        if unique_acks.shape[0] != 1:
            raise RuntimeError(
                f"[FAIL] {n_ok} traces 안에서 ack 가 {unique_acks.shape[0]} 종 — "
                "보드 sk 또는 t_out 잡음 의심"
            )
        print(f"[OK] ack determinism: 모든 {n_ok} traces ack 일치")

        meta = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "scope_sn": sn,
            "samples": args.samples,
            "gain_db": args.gain_db,
            "target": "smaug1",
            "cmd": "T",
            "component": args.component,
            "idx": args.idx,
            "alpha": int(args.alpha),
            "ss_ver": "SS_VER_1_1",
            "baud": args.baud,
            "n_attempted": args.num_traces,
            "n_ok": n_ok,
            "n_timeouts": timeouts,
            "firmware_hex": str(args.firmware_hex),
            "firmware_hex_sha256": fw_sha,
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_hex": sk_pke.hex(),
            "level": "smaug1",
            "log_p": p.log_p,
            "module_rank": p.module_rank,
            "lwe_n": p.n,
            "label": (
                f"s1_main_c{args.component}_j{args.idx}_a{args.alpha}_n{args.num_traces}"
            ),
        }
        np.savez_compressed(
            out_path,
            traces=traces[:n_ok],
            acks=acks[:n_ok],
            meta=np.array(meta, dtype=object),
        )
        elapsed = time.time() - started
        print(
            f"[OK] saved {n_ok}/{args.num_traces} traces -> {out_path}  "
            f"({elapsed:.1f}s, shape=({n_ok},{args.samples}))"
        )
    finally:
        try:
            target.dis()
        except Exception:
            pass
        try:
            scope.dis()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
