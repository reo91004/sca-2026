#!/usr/bin/env python3
"""[MAIN] Multi-term + R^2 cross-component chosen-CT 캡처 (paper Section 3.4).

Paper Section 3.4 (Multi-term + R^2 cross-component). multi-term chosen-CT
의 보드 round-trip 검증 (predict_mu_prime ≡ Z 응답) + HW gain 측정 (paper
Section 8.1 SNR phase transition 의 *원천 데이터*).

디자인 카탈로그 (`DESIGNS` dict):
    H_const_a64           c1[0] = 64                    # 단항 baseline
    H_const_a192          c1[0] = 192                   # 단항 baseline
    H_c1_const_a64        c1[1] = 64                    # s[1] oracle pair (+det)
    H_c1_const_a192       c1[1] = 192                   # s[1] oracle pair (-det)
    H_2term_l5_k50_a64    c1[0] = 64·(X^5 + X^50)       # 2-term
    H_2term_l10_k200_a64  c1[0] = 64·(X^10 + X^200)     # 2-term diff
    H_3term_l_5_50_200    c1[0] = 64·(X^5 + X^50 + X^200) # 3-term
    H_combined_l10_k50    c1[0]=64·X^10, c1[1]=64·X^50  # R^2 cross
    H_combined_l0_k0      c1[0]=64,       c1[1]=64       # R^2 const

각 design × N trace (Z 명령). 응답 32B µ′ + sk PKE dump → host 측 round-trip
검증.

용법:
    scripts/run_h_attack.py -n 64
    scripts/run_h_attack.py -n 512 --designs H_const_a64,H_const_a192 \\
        --out traces/H_attack_n512.npz

산출물:
    traces/H_attack.npz (default), 또는 --out 지정.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import CHUNK_BYTES, reset_target  # noqa: E402
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import params as _params  # noqa: E402


# Design 카탈로그 — coefs dict 형태로 build_multi_term_c1 입력.
# alpha 는 모두 U_p 안 (smaug1: multiples of 4).
DESIGNS: dict[str, dict[tuple[int, int], int]] = {
    "H_const_a64":          {(0, 0): 64},
    "H_const_a192":         {(0, 0): 192},
    "H_c1_const_a64":       {(1, 0): 64},   # component 1 oracle pair (s[1] 분리용)
    "H_c1_const_a192":      {(1, 0): 192},
    "H_2term_l5_k50_a64":   {(0, 5): 64, (0, 50): 64},
    "H_2term_l10_k200_a64": {(0, 10): 64, (0, 200): 64},
    "H_3term_l_5_50_200":   {(0, 5): 64, (0, 50): 64, (0, 200): 64},
    "H_combined_l10_k50":   {(0, 10): 64, (1, 50): 64},
    "H_combined_l0_k0":     {(0, 0): 64, (1, 0): 64},
}


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
    p.add_argument("-n", "--num-per-design", type=int, default=64,
                   help="design 당 trace 수")
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--designs", type=str, default=None,
                   help="comma-separated 디자인 이름. 미지정시 전부.")
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--out", type=Path,
                   default=_REPO / "traces" / "H_attack.npz")
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


def inject_ct(target, ct_bytes: bytes) -> bytes:
    chunks = _chosen.chunkify(ct_bytes, chunk_size=CHUNK_BYTES)
    for idx, data in chunks:
        target.simpleserial_write("I", bytes([idx]) + data)
        st = _ack(target, 1, timeout_ms=2000)
        if st[0] != 0:
            raise RuntimeError(f"'I' chunk {idx} status={st[0]}")
    target.simpleserial_write("L", b"")
    return _ack(target, 16)


def capture_n(scope, target, n: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    traces = np.empty((n, samples), dtype=np.float32)
    mus = np.zeros((n, 32), dtype=np.uint8)
    for i in range(n):
        scope.arm()
        target.simpleserial_write("Z", b"")
        if scope.capture():
            raise RuntimeError(f"scope timeout at trace {i}")
        a = target.simpleserial_read("r", 32, timeout=10000)
        if a is None or len(a) != 32:
            raise RuntimeError(f"Z ack at trace {i}")
        traces[i] = scope.get_last_trace()
        mus[i] = bytearray(a)
    return traces, mus


def _design_to_json(coefs: dict[tuple[int, int], int]) -> str:
    """tuple keys → JSON-serializable list of [m, l, alpha]."""
    return json.dumps([[m, l, a] for (m, l), a in coefs.items()])


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    p = _params.get("smaug1")

    if args.designs is None:
        design_names = list(DESIGNS.keys())
    else:
        design_names = [d.strip() for d in args.designs.split(",") if d.strip()]
        for d in design_names:
            if d not in DESIGNS:
                raise SystemExit(
                    f"unknown design '{d}'. choose from {list(DESIGNS.keys())}"
                )

    # simulator 검증 — 보드 캡처 전 수학 정합성 확인.
    print(f"[INFO] {len(design_names)} 디자인 × N={args.num_per_design} trace each")
    for name in design_names:
        coefs = DESIGNS[name]
        # build_multi_term_c1 가 ValueError 없이 통과하면 스펙 정합.
        ct = _chosen.build_multi_term_c1(p, coefs)
        nonzero = int(np.count_nonzero(ct.c1))
        print(f"  - {name}: nonzero c1 entries = {nonzero}, coefs = {coefs}")

    import chipwhisperer as cw  # noqa: E402
    from host.cw_serial import pick_serial  # noqa: E402
    sn = pick_serial(args.serial)
    print(f"[INFO] sn={sn}")

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

        sk_pke = bytearray()
        for idx in range(4):
            target.simpleserial_write("X", bytes([idx]))
            sk_pke.extend(_ack(target, 32, timeout_ms=2000))
        print(f"  sk_pke[0:8] = {bytes(sk_pke[:8]).hex()}…")

        all_traces: list[np.ndarray] = []
        all_mus: list[np.ndarray] = []
        design_label: list[int] = []  # design_idx per trace
        ct_fps: list[str] = []
        design_specs: list[str] = []
        started = time.time()
        for d_idx, name in enumerate(design_names, 1):
            coefs = DESIGNS[name]
            ct = _chosen.build_multi_term_c1(p, coefs)
            fp = inject_ct(target, ct.to_bytes())
            ct_fps.append(fp.hex())
            design_specs.append(_design_to_json(coefs))
            t, m = capture_n(scope, target, args.num_per_design, args.samples)
            all_traces.append(t)
            all_mus.append(m)
            design_label.extend([d_idx - 1] * args.num_per_design)
            elapsed = time.time() - started
            n_so_far = d_idx * args.num_per_design
            print(f"  [{d_idx}/{len(design_names)}] {name} "
                  f"({elapsed:.1f}s, {n_so_far/elapsed:.2f} tr/s)")

        traces = np.concatenate(all_traces, axis=0)
        mus = np.concatenate(all_mus, axis=0)
        labels = np.asarray(design_label, dtype=np.int64)

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
            "n_attempted": traces.shape[0],
            "git_rev": git,
            "firmware_hex": str(args.firmware_hex),
            "firmware_hex_sha256": fw_sha,
            "label": "H_attack_multi_term",
            "experiment": "phase_h_multi_term",
            "design_names": design_names,
            "design_specs_json": design_specs,
            "n_per_design": args.num_per_design,
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_bytes_hex": bytes(sk_pke).hex(),
            "ct_fp16_per_design": ct_fps,
            "design_index_per_trace": labels.tolist(),
        }
        np.savez_compressed(
            args.out, traces=traces, mu_prime=mus,
            meta=np.array(meta, dtype=object),
        )
        print(f"[OK] saved {args.out} traces.shape={traces.shape}")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass

    # 후속 분석은 별도 스크립트 (analyze_h_attack.py 또는 인라인 jupyter).
    # 핵심 검증: 호스트 predict_mu_prime(c1, unpack_sx(sk_pke)) == 보드 µ′
    print("[NEXT] analyze with: python3 -c \"from host.smaug.chosen import predict_mu_prime; ...\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
