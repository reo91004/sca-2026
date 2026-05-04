#!/usr/bin/env python3
"""[MAIN] Chosen-CT attack 캡처 — paper main result reproducer.

Paper Section 4 (Method overview), Section 7 (N curve). 4 chosen-CT × N trace
캡처 — paper main attack 의 *데이터 캡처* 부분.

Chosen-CT 디자인 (Paper Section 3 컨벤션 정정 결과):
  c1 = α (j=0 의 상수 다항식) → ⟨c1, s⟩_i = α·s_i 가 모든 i 에 성립
  → µ′_i = round_t(α·s_i) 가 256 비밀 계수를 한 번에 leak.

4 chosen-CT (oracle pair × 2 component):
  s[0]: c1[0]=α=64 (+det), c1[0]=α=192 (-det)
  s[1]: c1[1]=α=64,        c1[1]=α=192

호출 흐름 (per call):
  F (영속 keygen) → X×4 (sk PKE ground truth dump, attack 자체엔 불필요지만
  검증용) → 4 chosen-CT × N trace (각각 'I'×21 inject + 'L' fingerprint +
  'Z'×N indcpa_dec capture).

용법 (paper main):
    scripts/run_attack.py -n 2     # 8 traces, 3.2 sec — paper main result
    scripts/run_attack.py -n 128   # multi-seed evaluation 용 (5 seeds × N=128)

산출물:
    traces/attack_const_c1.npz (default, 또는 --out 으로 지정)

다음 단계: scripts/attack_direct_poi.py (single-seed) 또는
          scripts/analyze_multi_seed.py (multi-seed paper main analyzer).
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

from host.chosen_ct import CHUNK_BYTES, reset_target  # noqa: E402
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import params as _params  # noqa: E402


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
    p.add_argument("--level", choices=("smaug1", "smaug3", "smaug5"),
                   default="smaug1",
                   help="SMAUG-T 보안 레벨 (펌웨어가 SMAUG_LEVEL=N 로 빌드돼 있어야 함)")
    p.add_argument("-n", "--num-per-ct", type=int, default=64,
                   help="chosen-CT 당 attack trace 수 (averaging 효과)")
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--alpha-pos", type=int, default=None,
                   help="oracle pair α_pos. None 이면 level 별 기본값 (smaug1=64, smaug3/5=128).")
    p.add_argument("--alpha-neg", type=int, default=None,
                   help="oracle pair α_neg. None 이면 level 별 기본값 (smaug1=192, smaug3/5=384).")
    p.add_argument("--alphas", type=str, default=None,
                   help="추가 α list (comma-separated). 예: --alphas 132 (support α). "
                        "지정 시 (alpha_pos, alpha_neg, *extras) sweep — designs/comp = 2 + len(extras).")
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--out", type=Path, default=None,
                   help="출력 .npz. None 이면 traces/attack_const_c1_<level>.npz.")
    p.add_argument("--firmware-hex", type=Path, default=None,
                   help="플래시할 .hex. None 이면 simpleserial-smaug-CW308_STM32F4-<level>.hex.")
    p.add_argument("--serial", default=None)
    return p.parse_args()


def default_oracle_pair(p) -> tuple[int, int]:
    """level 별 oracle pair (α_pos, α_neg) — sk_partition.find_oracle_pairs 의 첫 항목."""
    if p.log_p == 8:   # smaug1
        return (64, 192)
    if p.log_p == 9:   # smaug3, smaug5
        return (128, 384)
    raise ValueError(f"unknown log_p={p.log_p}, can't pick default oracle pair")


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


def main() -> int:
    args = parse_args()

    p = _params.get(args.level)

    if args.out is None:
        args.out = _REPO / "traces" / f"attack_const_c1_{args.level}.npz"
    if args.firmware_hex is None:
        args.firmware_hex = (_REPO / "firmware" / "simpleserial-smaug" /
                             f"simpleserial-smaug-CW308_STM32F4-{args.level}.hex")
    if args.alpha_pos is None or args.alpha_neg is None:
        a_pos, a_neg = default_oracle_pair(p)
        if args.alpha_pos is None:
            args.alpha_pos = a_pos
        if args.alpha_neg is None:
            args.alpha_neg = a_neg

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(args.serial)
    print(f"[INFO] level={args.level} (k={p.module_rank}, hs={p.hs}, log_p={p.log_p}), "
          f"sn={sn}, n/CT={args.num_per_ct}, α_pos={args.alpha_pos}, α_neg={args.alpha_neg}")
    print(f"[INFO] {p.module_rank} components × 2 alphas = {2 * p.module_rank} chosen-CT designs")

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

        # PKE sk dump (32-byte chunks). chunk count = pke_secret_key_bytes / 32.
        # smaug1: 4, smaug3: 6, smaug5: 8.
        n_sk_chunks = p.pke_secret_key_bytes // 32
        sk_pke = bytearray()
        for idx in range(n_sk_chunks):
            target.simpleserial_write("X", bytes([idx]))
            sk_pke.extend(_ack(target, 32, timeout_ms=2000))
        assert len(sk_pke) == p.pke_secret_key_bytes

        # chosen-CT designs: each component × α list.
        # 기본 (oracle pair only): designs/comp = 2.
        # --alphas 로 추가: designs/comp = 2 + len(extras), e.g. (128, 384, 132) for smaug3.
        extra_alphas = []
        if args.alphas:
            extra_alphas = [int(x) for x in args.alphas.split(",") if x.strip()]
        alpha_list = (args.alpha_pos, args.alpha_neg, *extra_alphas)
        sweep = []
        for component in range(p.module_rank):
            for alpha in alpha_list:
                sweep.append((component, alpha))

        all_traces: list[np.ndarray] = []
        all_mus: list[np.ndarray] = []
        all_labels = []  # per-trace (component, alpha)
        ct_fps = []
        started = time.time()
        for n_done, (comp, alpha) in enumerate(sweep, 1):
            ct = _chosen.build_monomial_c1(p, component=comp, coef_idx=0,
                                           alpha=alpha)
            fp = inject_ct(target, ct.to_bytes())
            ct_fps.append(fp.hex())
            t, m = capture_n(scope, target, args.num_per_ct, args.samples)
            all_traces.append(t)
            all_mus.append(m)
            all_labels.extend([(comp, alpha)] * args.num_per_ct)
            elapsed = time.time() - started
            print(f"  [{n_done}/{len(sweep)}] component={comp} α={alpha:3d} "
                  f"({elapsed:.1f}s, {(n_done * args.num_per_ct)/elapsed:.2f} tr/s)")

        traces = np.concatenate(all_traces, axis=0)
        mus = np.concatenate(all_mus, axis=0)
        labels = np.asarray(all_labels, dtype=np.int64)  # (N_total, 2)

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
            "label": "attack_const_c1",
            "experiment": "attack_const_c1",
            "level": args.level,
            "module_rank": p.module_rank,
            "hs": p.hs,
            "log_p": p.log_p,
            "alpha_pos": args.alpha_pos,
            "alpha_neg": args.alpha_neg,
            "components": list(range(p.module_rank)),
            "n_per_ct": args.num_per_ct,
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_bytes_hex": bytes(sk_pke).hex(),
            "ct_fp16_per_sweep": ct_fps,
            "label_component": labels[:, 0].tolist(),
            "label_alpha":     labels[:, 1].tolist(),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
