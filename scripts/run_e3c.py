#!/usr/bin/env python3
"""E3c 캡처: oracle pair (α_pos, α_neg) 단항 chosen-CT 로 secret coefficient
ternary 분류 (보드 측, 같은 sk).

핵심:
  α=64+256k → +1 detector  (µ′_i = 1 iff s = +1)
  α=192+256k → -1 detector  (µ′_i = 1 iff s = -1)

각 (k, j) 위치에서 두 chosen-CT 를 dec → 두 응답을 join 해 ternary.

사용:
  scripts/run_e3c.py --positions 0,1,5,10,50 --component 0 \\
      --alpha-pos 64 --alpha-neg 192 -n 4

저장 형식:
  traces/E3c_<αp>_<αn>_<comp>.npz — 모든 위치 트레이스 concat
  responses 의 첫 byte = 'D' 의 mismatch (참고용)
  meta.position_labels = (alpha, k, j) tuple 리스트, trace[i] 의 라벨
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
    CHUNK_BYTES, INJECT_PAYLOAD_LEN, reset_target,
)
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import params as _params  # noqa: E402


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO,
            stderr=subprocess.DEVNULL,
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
    p.add_argument("--component", type=int, default=0, help="c1 component k")
    p.add_argument("--positions", type=str, required=True,
                   help="콤마로 구분된 j 인덱스 리스트 (0..255)")
    p.add_argument("--alpha-pos", type=int, default=64, help="+1 detector α")
    p.add_argument("--alpha-neg", type=int, default=192, help="-1 detector α")
    p.add_argument("-n", "--num-per-position", type=int, default=4)
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--serial", default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="기본 traces/E3c_<αp>_<αn>_k<comp>.npz")
    p.add_argument("--cmd", choices=["D", "Z"], default="Z",
                   help="trace 캡처 명령 — D=full crypto_kem_dec, "
                        "Z=indcpa_dec only (3.4x 빠름, re-enc/cmov 누설 제거).")
    p.add_argument("--firmware-hex", type=Path,
                   default=_REPO / "firmware" / "simpleserial-smaug"
                   / "simpleserial-smaug-CW308_STM32F4.hex")
    return p.parse_args()


def _setup(sn: str, samples: int, gain_db: float, baud: int):
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


def _ack(target, n: int, timeout_ms: int = 10000) -> bytes:
    a = target.simpleserial_read("r", n, timeout=timeout_ms)
    if a is None or len(a) != n:
        raise RuntimeError(f"ack {-1 if a is None else len(a)} != {n}")
    return bytes(a)


def inject_ct(target, ct_bytes: bytes) -> bytes:
    """'I' × N 으로 ct_inj 채우고 'L' 로 무결성 지문 받기."""
    chunks = _chosen.chunkify(ct_bytes, chunk_size=CHUNK_BYTES)
    for idx, data in chunks:
        payload = bytes([idx]) + data
        target.simpleserial_write("I", payload)
        st = _ack(target, 1, timeout_ms=2000)
        if st[0] != 0:
            raise RuntimeError(f"'I' chunk {idx} status={st[0]}")
    target.simpleserial_write("L", b"")
    return _ack(target, 16)


def capture_n(scope, target, n: int, samples: int, cmd: str = "Z"
              ) -> tuple[np.ndarray, np.ndarray]:
    """cmd='D' (full crypto_kem_dec, 1B mismatch) 또는 'Z' (indcpa_dec, 16B µ').
    Z 는 µ' 비트가 그대로 응답에 들어와 ground truth 검증 + 분류기 비교에 유용."""
    ack_len = 16 if cmd == "Z" else 1
    traces = np.empty((n, samples), dtype=np.float32)
    resp = np.zeros((n, ack_len), dtype=np.uint8)
    for i in range(n):
        scope.arm()
        target.simpleserial_write(cmd, b"")
        if scope.capture():
            raise RuntimeError(f"scope timeout at trace {i}")
        a = target.simpleserial_read("r", ack_len, timeout=10000)
        if a is None or len(a) != ack_len:
            raise RuntimeError(f"{cmd} ack at trace {i}")
        traces[i] = scope.get_last_trace()
        resp[i, :] = bytearray(a)
    return traces, resp


def dump_sk(target, total_bytes: int = 128, chunk_bytes: int = 32) -> bytes:
    """'X' × ⌈total/chunk⌉ 로 sk PKE 영역 dump. smaug1: 128B."""
    out = bytearray()
    n_chunks = total_bytes // chunk_bytes
    for idx in range(n_chunks):
        target.simpleserial_write("X", bytes([idx]))
        chunk = target.simpleserial_read("r", chunk_bytes, timeout=2000)
        if chunk is None or len(chunk) != chunk_bytes:
            raise RuntimeError(
                f"X[{idx}] ack len {-1 if chunk is None else len(chunk)}")
        out.extend(chunk)
    return bytes(out)


def main() -> int:
    args = parse_args()
    js = sorted({int(x) for x in args.positions.split(",")})
    if not all(0 <= j < 256 for j in js):
        raise SystemExit("[FAIL] positions out of [0, 256)")
    if args.alpha_pos % 4 != 0 or args.alpha_neg % 4 != 0:
        raise SystemExit("[FAIL] α 값이 fixed-point set 밖 (multiples of 4 필요)")

    p = _params.get("smaug1")
    if args.component not in (0, 1):
        raise SystemExit("[FAIL] smaug1 의 component 는 0 또는 1")

    out = args.out or _REPO / "traces" / (
        f"E3c_ap{args.alpha_pos}_an{args.alpha_neg}_k{args.component}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    from host.cw_serial import pick_serial
    sn = pick_serial(args.serial)
    print(f"[INFO] CW1173 sn={sn}, positions={js}, "
          f"α_pos={args.alpha_pos}, α_neg={args.alpha_neg}, "
          f"comp={args.component}, n/pos={args.num_per_position}")

    fw_sha = _file_sha256(args.firmware_hex) if args.firmware_hex.exists() else None
    git = _git_rev()

    scope, target = _setup(sn, args.samples, args.gain_db, args.baud)
    try:
        # 단일 keypair (모든 위치/α 공유)
        target.simpleserial_write("F", b"")
        pk_fp16 = _ack(target, 16)
        print(f"  pk_fp16 = {pk_fp16.hex()}")

        # sk PKE 영역 dump — 분류기 ground truth (필수). smaug1: 128B.
        sk_pke_bytes = dump_sk(target, total_bytes=128, chunk_bytes=32)
        print(f"  sk_pke (X×4) = {sk_pke_bytes[:32].hex()}…")

        # (k, j, α) 라벨 트레이스를 한 컨테이너에 모음
        label_alpha: list[int] = []
        label_k:     list[int] = []
        label_j:     list[int] = []
        all_traces: list[np.ndarray] = []
        all_resp:   list[np.ndarray] = []
        ct_fps: list[str] = []

        sweep = []
        for j in js:
            sweep.append((args.alpha_pos, args.component, j))
            sweep.append((args.alpha_neg, args.component, j))

        started = time.time()
        for n_done, (alpha, k, j) in enumerate(sweep, 1):
            ct = _chosen.build_monomial_c1(p, component=k, coef_idx=j, alpha=alpha)
            fp = inject_ct(target, ct.to_bytes())
            ct_fps.append(fp.hex())
            t, r = capture_n(scope, target, args.num_per_position, args.samples,
                             cmd=args.cmd)
            all_traces.append(t)
            all_resp.append(r)
            label_alpha.extend([alpha] * args.num_per_position)
            label_k.extend([k] * args.num_per_position)
            label_j.extend([j] * args.num_per_position)
            elapsed = time.time() - started
            print(f"  [{n_done}/{len(sweep)}] α={alpha:>4d} k={k} j={j:>3d} "
                  f"({elapsed:.1f}s)")

        traces = np.concatenate(all_traces, axis=0)
        responses = np.concatenate(all_resp, axis=0)
        meta = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "scope_sn": sn,
            "samples": args.samples,
            "gain_db": args.gain_db,
            "target": "smaug",
            "cmd": args.cmd,
            "send_len": 0,
            "resp_len": 16 if args.cmd == "Z" else 1,
            "ss_ver": "SS_VER_1_1",
            "baud": args.baud,
            "n_attempted": traces.shape[0],
            "n_timeouts": 0,
            "git_rev": git,
            "firmware_hex": str(args.firmware_hex),
            "firmware_hex_sha256": fw_sha,
            "seed": 0,
            "label": "E3c_oracle_pair",
            "experiment": "E3c_oracle_pair",
            "alpha_pos": args.alpha_pos,
            "alpha_neg": args.alpha_neg,
            "component": args.component,
            "positions": js,
            "n_per_position": args.num_per_position,
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_bytes_hex": sk_pke_bytes.hex(),  # ground truth (PKE sk 128B)
            "ct_fp16_per_sweep": ct_fps,
            # trace[i] 의 (alpha, k, j) 라벨 — int list (np 직렬화 안전)
            "label_alpha": label_alpha,
            "label_k":     label_k,
            "label_j":     label_j,
        }
        np.savez_compressed(
            out, traces=traces, responses=responses,
            meta=np.array(meta, dtype=object),
        )
        print(f"[OK] saved {out} shape={traces.shape}")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
