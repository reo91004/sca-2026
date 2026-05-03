#!/usr/bin/env python3
"""[STEP 3] random-µ profile 캡처 — cross-domain transfer 시도 (NEGATIVE).

Paper Section 5.1 (profile-PoI cross-domain transfer fail). 표준 PQC SCA
pipeline 의 *first attempt* — random µ × N=200/1000 trace 로 profile-PoI 학습
→ chosen-CT attack 에 transfer. **결국 sign 반전 + 56% 정확도** (zero-baseline
73% 미달) 로 fail. paper 의 *direct attack PoI* (Section 5) method 가 이걸
우회 (chosen-CT data 자체에서 PoI 학습).

목적:
    같은 sk 위 random 32-byte µ × N → M (indcpa_enc, seed=zero fixed) → Z
    (indcpa_dec). Z 응답이 µ′ = µ (1-δ correctness) 라 각 trace 의 256 µ′_i
    ground truth 즉시 확보 → 각 i 별 Welch t PoI 학습.

핵심 발견 (negative):
    - per-bit |t|: N=200 → 1000 (5×) 했는데 4.97 → 5.19 (변화 적음)
    - √N 스케일링 안 따라감 → noise peak 만 봄 (signal 미달)
    - profile 학습 PoI 가 chosen-CT attack 에 sign 반전 (corr -0.086)
    - Direct attack PoI (paper Section 5) 가 이걸 회피 (corr +0.99 ~ +1.00)

Note (history/):
    paper main result 는 random-µ profile 안 씀. 이 캡처는 *negative finding
    evidence*. 결과만 paper Section 5.1 에 인용.

용법 (legacy):
    history/scripts/capture_step3_random_mu_profile.py -n 1000

산출물 (legacy):
    traces/profile_random_mu_n1000.npz (30 MB, 1000 trace)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target  # noqa: E402


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO, stderr=subprocess.DEVNULL)
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
    p.add_argument("-n", "--num-traces", type=int, default=200)
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--seed", type=int, default=0xCAFE,
                   help="random µ 생성 시드 (재현성). 0 이면 OS entropy.")
    p.add_argument("--out", type=Path,
                   default=_REPO / "traces" / "profile_random_mu.npz")
    p.add_argument("--firmware-hex", type=Path,
                   default=_REPO / "firmware" / "simpleserial-smaug"
                   / "simpleserial-smaug-CW308_STM32F4.hex")
    p.add_argument("--serial", default=None)
    return p.parse_args()


def _ack(target, n: int, timeout_ms: int = 10000) -> bytes:
    a = target.simpleserial_read("r", n, timeout=timeout_ms)
    if a is None or len(a) != n:
        raise RuntimeError(f"ack {-1 if a is None else len(a)} != {n}")
    return bytes(a)


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed) if args.seed else None
    rand_bytes = (lambda n: rng.bytes(n)) if rng else (lambda n: secrets.token_bytes(n))

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(args.serial)
    print(f"[INFO] sn={sn}, n={args.num_traces}, samples={args.samples}")

    fw_sha = _file_sha256(args.firmware_hex) if args.firmware_hex.exists() else None
    git = _git_rev()

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

    try:
        target.simpleserial_write("F", b"")
        pk_fp16 = _ack(target, 16)
        print(f"  pk_fp16 = {pk_fp16.hex()}")

        # sk dump for ground truth
        sk_pke = bytearray()
        for idx in range(4):
            target.simpleserial_write("X", bytes([idx]))
            sk_pke.extend(_ack(target, 32, timeout_ms=2000))
        print(f"  sk_pke (X×4) = {bytes(sk_pke[:32]).hex()}…")

        traces = np.empty((args.num_traces, args.samples), dtype=np.float32)
        mu_used = np.zeros((args.num_traces, 32), dtype=np.uint8)
        mu_prime = np.zeros((args.num_traces, 32), dtype=np.uint8)
        timeouts = 0

        # warm-up
        for _ in range(3):
            mu = rand_bytes(32)
            target.simpleserial_write("M", mu); _ack(target, 16)
            scope.arm(); target.simpleserial_write("Z", b"")
            scope.capture(); _ack(target, 32)

        started = time.time()
        for i in range(args.num_traces):
            mu = rand_bytes(32)
            target.simpleserial_write("M", mu)
            ct_fp = _ack(target, 16, timeout_ms=2000)  # M ack
            scope.arm()
            target.simpleserial_write("Z", b"")
            if scope.capture():
                timeouts += 1
                print(f"[WARN] {i}: scope timeout"); continue
            mp = _ack(target, 32, timeout_ms=10000)
            traces[i] = scope.get_last_trace()
            mu_used[i] = np.frombuffer(mu, dtype=np.uint8)
            mu_prime[i] = np.frombuffer(mp, dtype=np.uint8)
            if (i + 1) % 50 == 0 or i == args.num_traces - 1:
                e = time.time() - started; r = (i + 1) / e
                print(f"  [PROG] {i+1}/{args.num_traces} ({r:.2f} tr/s) timeouts={timeouts}")

        # 1-δ correctness 검증 — µ' == µ 인 비율
        match = int(np.all(mu_used == mu_prime, axis=1).sum())
        print(f"\n[INFO] µ' == µ exact-match: {match}/{args.num_traces} "
              f"({100*match/args.num_traces:.1f}%)")
        per_bit_match = int((mu_used == mu_prime).sum())
        per_bit_total = mu_used.size
        print(f"  per-byte match: {per_bit_match}/{per_bit_total} "
              f"({100*per_bit_match/per_bit_total:.2f}%)")

        meta = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "scope_sn": sn,
            "samples": args.samples,
            "gain_db": args.gain_db,
            "target": "smaug",
            "cmd": "Z",
            "send_len": 0,
            "resp_len": 32,
            "ss_ver": "SS_VER_1_1",
            "baud": args.baud,
            "n_attempted": args.num_traces,
            "n_timeouts": timeouts,
            "git_rev": git,
            "firmware_hex": str(args.firmware_hex),
            "firmware_hex_sha256": fw_sha,
            "seed": args.seed,
            "label": "profile_random_mu",
            "experiment": "profile_random_mu",
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_bytes_hex": bytes(sk_pke).hex(),
        }
        np.savez_compressed(
            args.out, traces=traces,
            mu_used=mu_used, mu_prime=mu_prime,
            meta=np.array(meta, dtype=object),
        )
        print(f"[OK] saved {args.out} traces.shape={traces.shape}")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
