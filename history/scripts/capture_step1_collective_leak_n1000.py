#!/usr/bin/env python3
"""[STEP 1] Collective µ flip leakage 첫 측정 (TVLA, N=1000).

Paper Section 8.1 (SNR phase transition) 에서 인용된 첫 evidence — "SMAUG-T
trace 에 SCA-readable 신호 가 존재" 라는 사실의 시드 측정.

목적:
    같은 sk 위 *알려진 µ pattern* (μ=all-zero vs μ=all-one) 의 PKE-labeled
    ciphertext 를 'M' 명령으로 만들고, 'D' (full crypto_kem_dec) 로 N=1000
    trace 두 클래스 캡처. 클래스간 mean diff TVLA → max|t|=8.99, leaky 221
    points. 같은 sk 라 차분이 *순수 message-leakage* (+ noise) 만 남음.

핵심 발견:
    - SMAUG-T smaug1 trace 에 µ flip 의 collective leakage 가 분명히 존재
    - 다만 *single-bit µ′_i flip* 분리는 noise floor 미달 (E3p 단계 확인)
      → paper 의 *direct attack PoI* 가 이 한계 우회

Note (history/):
    paper main attack (8 traces, 100% recovery) 은 이 데이터 안 씀. 결과 만
    paper 에 인용 (max|t|=8.99 수치). 발견 과정 evidence 로 보존.

용법 (legacy):
    history/scripts/capture_step1_collective_leak_n1000.py -n 1000

산출물 (legacy):
    traces/E3b_zero.npz, traces/E3b_one.npz
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import (  # noqa: E402
    MU_BYTES, _MU_PATTERNS, reset_target,
)


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--num-traces", type=int, default=1000)
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--out-zero", type=Path,
                   default=_REPO / "traces" / "E3b_zero.npz")
    p.add_argument("--out-one", type=Path,
                   default=_REPO / "traces" / "E3b_one.npz")
    p.add_argument("--firmware-hex", type=Path,
                   default=_REPO / "firmware" / "simpleserial-smaug"
                   / "simpleserial-smaug-CW308_STM32F4.hex")
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--serial", default=None)
    return p.parse_args()


def _setup_scope_and_target(sn: str, samples: int, gain_db: float, baud: int):
    import chipwhisperer as cw
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = samples
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = gain_db
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = baud
    target.flush()
    time.sleep(0.3)
    return scope, target


def _ss_write(target, cmd: str, payload: bytes = b"") -> None:
    target.simpleserial_write(cmd, bytes(payload))


def _ss_ack(target, n: int, timeout_ms: int = 10000) -> bytes:
    ack = target.simpleserial_read("r", n, timeout=timeout_ms)
    if ack is None or len(ack) != n:
        raise RuntimeError(f"ack len {-1 if ack is None else len(ack)} != {n}")
    return bytes(ack)


def capture_one_class(
    scope, target, n_traces: int, samples: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """현재 보드 상태 (sk + ct_inj 영속) 위에서 'D' n 회 → trace + ack."""
    traces = np.empty((n_traces, samples), dtype=np.float32)
    responses = np.zeros((n_traces, 1), dtype=np.uint8)
    timeouts = 0
    started = time.time()
    for i in range(n_traces):
        scope.arm()
        target.simpleserial_write("D", b"")
        if scope.capture():
            timeouts += 1
            print(f"[WARN] trace {i}: scope timeout")
            continue
        ack = target.simpleserial_read("r", 1, timeout=10000)
        if ack is None or len(ack) != 1:
            timeouts += 1
            continue
        traces[i] = scope.get_last_trace()
        responses[i, 0] = int(ack[0])
        if (i + 1) % 100 == 0 or i == n_traces - 1:
            elapsed = time.time() - started
            rate = (i + 1) / max(elapsed, 1e-9)
            print(f"  [PROG] {i+1}/{n_traces} ({rate:.2f} tr/s)  timeouts={timeouts}")
    return traces, responses, timeouts


def main() -> int:
    args = parse_args()
    args.out_zero.parent.mkdir(parents=True, exist_ok=True)
    args.out_one.parent.mkdir(parents=True, exist_ok=True)

    from host.cw_serial import pick_serial
    sn = pick_serial(args.serial)
    print(f"[INFO] CW1173 sn={sn}")

    fw_sha = _file_sha256(args.firmware_hex) if args.firmware_hex.exists() else None
    git = _git_rev()

    scope, target = _setup_scope_and_target(sn, args.samples, args.gain_db, args.baud)
    try:
        # 1) 단일 keypair
        print("=== F (keygen) ===")
        _ss_write(target, "F")
        pk_fp16 = _ss_ack(target, 16)
        print(f"  pk_fp16 = {pk_fp16.hex()}")

        # 2) μ = zero, ct 채우기
        mu_zero = _MU_PATTERNS["zero"]
        print("=== M (μ=zero) ===")
        _ss_write(target, "M", mu_zero)
        ct_zero_fp = _ss_ack(target, 16)
        print(f"  ct_fp16(μ=0)  = {ct_zero_fp.hex()}")

        # 3) zero 캡처
        print(f"=== capture {args.num_traces} traces (μ=zero, 'D') ===")
        tz, rz, toz = capture_one_class(scope, target, args.num_traces, args.samples)

        # 4) μ = one, ct 다시 채우기 (sk 그대로)
        mu_one = _MU_PATTERNS["one"]
        print("=== M (μ=one) ===")
        _ss_write(target, "M", mu_one)
        ct_one_fp = _ss_ack(target, 16)
        print(f"  ct_fp16(μ=1)  = {ct_one_fp.hex()}")
        if ct_zero_fp == ct_one_fp:
            raise RuntimeError("ct_inj 가 μ 변경에도 같음 — indcpa_enc 작동 의심")

        # 5) one 캡처
        print(f"=== capture {args.num_traces} traces (μ=one, 'D') ===")
        to, ro, tooo = capture_one_class(scope, target, args.num_traces, args.samples)

        # 6) 두 .npz 저장
        common_meta = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "scope_sn": sn,
            "samples": args.samples,
            "gain_db": args.gain_db,
            "target": "smaug",
            "cmd": "D",
            "send_len": 0,
            "resp_len": 1,
            "ss_ver": "SS_VER_1_1",
            "baud": args.baud,
            "git_rev": git,
            "firmware_hex": str(args.firmware_hex),
            "firmware_hex_sha256": fw_sha,
            "seed": 0,
            "pk_fp16": pk_fp16.hex(),
            "experiment": "E3b_pke_labeled",
        }
        for path, traces, resp, timeouts, mu_label, ct_fp in (
            (args.out_zero, tz, rz, toz, "zero", ct_zero_fp),
            (args.out_one,  to, ro, tooo, "one",  ct_one_fp),
        ):
            meta = dict(common_meta)
            meta.update({
                "n_attempted": args.num_traces,
                "n_timeouts": timeouts,
                "label": f"E3b_{mu_label}",
                "mu_pattern": mu_label,
                "ct_fp16_board": ct_fp.hex(),
            })
            np.savez_compressed(
                path, traces=traces, responses=resp,
                meta=np.array(meta, dtype=object),
            )
            print(f"[OK] saved {path} (timeouts={timeouts})")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass

    # 자동 viz — 캡처 직후 즉시 PNG 생성. 별도 명령으로 다시 돌릴 필요 없게.
    try:
        from host.analysis import io as a_io
        from host.analysis import tvla as a_tvla
        from host.analysis import viz as a_viz
        cap_a = a_io.load_capture(args.out_zero)
        cap_b = a_io.load_capture(args.out_one)
        result = a_tvla.welch_t(cap_a.traces, cap_b.traces)
        print(f"[TVLA] {result.summary()}")
        png = _REPO / "results" / "E3b_tvla_zero_vs_one.png"
        a_viz.plot_tvla(result, png,
                        title="E3b: μ=zero vs μ=one (PKE-labeled, fixed sk)")
        # overview 두 클래스 평균/σ 도 별도 PNG
        overview_zero = _REPO / "results" / "E3b_overview_zero.png"
        overview_one  = _REPO / "results" / "E3b_overview_one.png"
        a_viz.plot_overview(cap_a, overview_zero)
        a_viz.plot_overview(cap_b, overview_one)
        print(f"[VIZ] {png}")
        print(f"[VIZ] {overview_zero}")
        print(f"[VIZ] {overview_one}")
    except Exception as e:
        print(f"[WARN] auto-viz 실패 (캡처 자체는 OK): {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
