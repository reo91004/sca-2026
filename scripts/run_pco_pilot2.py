#!/usr/bin/env python3
"""PCO pilot 2 — per-coefficient sparse chosen-CT + HW model fit + cross-session check.

Pilot 1 결과: 'D' command late trace (sample 22924) 에 max|t|=31.93,
zero vs c0_a192 single-trace 100%. 단 두 ct 의 µ' Hamming distance 가 31 bits.

Pilot 2 핵심 질문:
  Q1. *1-bit* µ' 차이도 late-trace 에서 detectable 한가? (PCO 는 1-bit oracle 필요)
  Q2. HW(µ') vs max|t| 가 monotone 인가? (HW model fit)
  Q3. Late-trace leak 의 sample 위치가 capture session 마다 stable 한가?

설계:
  - 같은 target sk (board RAM 'F' 후 reuse, no reset)
  - 6 designs:
      D0: zero       (c1=0, c2=0)              → µ' HW = 0
      D1: c1[0,0]=4  (smallest perturbation)   → µ' HW ≈ 0-1 (대부분 0)
      D2: c1[0,0]=64 (oracle pair +)            → µ' HW ≈ 35-40
      D3: c1[0,0]=128 (intermediate)            → µ' HW ≈ 35
      D4: c1[0,0]=192 (oracle pair -)            → µ' HW ≈ 30-35
      D5: c1[0,0]=252 (smallest perturbation, near-zero)  → µ' HW ≈ 0-1
  - N=64 'D' trace per design
  - + 두 번째 capture session (same sk, no reset, no F): D0 + D2 만 → 위치 stability 확인

산출:
  traces/pco_pilot2.npz (전체)
  traces/pco_pilot2_session_b.npz (cross-session subset)
"""

from __future__ import annotations

import hashlib
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
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from host.smaug.ciphertext import Ciphertext  # noqa: E402


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


def capture_n_decap(scope, target, n: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    traces = np.empty((n, samples), dtype=np.float32)
    flags = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        scope.arm()
        target.simpleserial_write("D", b"")
        if scope.capture():
            raise RuntimeError(f"scope timeout at trace {i}")
        a = target.simpleserial_read("r", 1, timeout=10000)
        if a is None or len(a) != 1:
            raise RuntimeError(f"D ack at trace {i}")
        traces[i] = scope.get_last_trace()
        flags[i] = a[0]
    return traces, flags


def run_session(scope, target, p, sk, designs_def, n_per_ct, samples) -> dict:
    all_traces = []; all_flags = []; all_labels = []
    mu_per: dict[str, np.ndarray] = {}
    fp_per: dict[str, str] = {}

    for name, c1, c2 in designs_def:
        ct = Ciphertext(c1=c1, c2=c2, params=p)
        ct_bytes = ct.to_bytes()
        host_fp = hashlib.sha3_256(ct_bytes).hexdigest()[:32]
        fp = inject_ct(target, ct_bytes)
        if fp.hex() != host_fp:
            raise RuntimeError(f"ct fp mismatch host={host_fp} board={fp.hex()}")
        mu_pred = _chosen.predict_mu_prime(p, c1, sk, c2=c2)
        mu_per[name] = mu_pred
        fp_per[name] = host_fp
        print(f"    design {name:10s} HW(µ')={int(mu_pred.sum()):>3d}  ct_fp={host_fp[:16]}...")
        tr, fl = capture_n_decap(scope, target, n_per_ct, samples)
        all_traces.append(tr); all_flags.append(fl)
        all_labels.extend([name] * n_per_ct)
    return {
        "traces": np.concatenate(all_traces, axis=0),
        "flags": np.concatenate(all_flags, axis=0),
        "labels": np.array(all_labels),
        "mu_pred": mu_per,
        "ct_fp": fp_per,
    }


def main() -> int:
    level = "smaug1"
    n_per_ct = 64
    samples = 24400
    p = _params.get(level)

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(None)
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = samples
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = 25.0
    scope.clock.adc_src = "clkgen_x1"   # 7.4 MHz to fit full crypto_kem_dec
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = 38400
    target.flush()
    time.sleep(0.3)

    try:
        # keygen + sk dump
        target.simpleserial_write("F", b"")
        pk_fp16 = _ack(target, 16)
        n_sk_chunks = p.pke_secret_key_bytes // 32
        sk_pke = bytearray()
        for idx in range(n_sk_chunks):
            target.simpleserial_write("X", bytes([idx]))
            sk_pke.extend(_ack(target, 32, timeout_ms=2000))
        sk = _codec.unpack_sx(bytes(sk_pke)).reshape(p.module_rank, p.n).astype(np.int64)
        sk_hex = bytes(sk_pke).hex()
        print(f"  sk={sk_hex[:16]}, pk_fp={pk_fp16.hex()}")

        # Build 6 designs
        zero_c1 = np.zeros((p.module_rank, p.n), dtype=np.int64)
        zero_c2 = np.zeros(p.n, dtype=np.int64)
        designs_def = [("zero", zero_c1, zero_c2)]
        for alpha in (4, 64, 128, 192, 252):
            c = _chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=alpha)
            designs_def.append((f"a{alpha}", c.c1, c.c2))

        # === Session A — main pilot ===
        print(f"\n[Session A] 6 designs × N={n_per_ct} = {6 * n_per_ct} traces")
        sess_a = run_session(scope, target, p, sk, designs_def, n_per_ct, samples)
        meta_a = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "scope_sn": sn, "samples": samples, "gain_db": 25.0,
            "clock_adc_src": "clkgen_x1",
            "cmd": "D", "level": level,
            "module_rank": p.module_rank, "hs": p.hs, "log_p": p.log_p,
            "n_per_design": n_per_ct,
            "design_names": [d[0] for d in designs_def],
            "ct_fp_per_design": sess_a["ct_fp"],
            "mu_hw_per_design": {n: int(sess_a["mu_pred"][n].sum()) for n in sess_a["mu_pred"]},
            "label_design": sess_a["labels"].tolist(),
            "mismatch_flag_per_trace": sess_a["flags"].tolist(),
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_bytes_hex": sk_hex,
            "session": "A",
        }
        out_a = _REPO / "traces/pco_pilot2.npz"
        np.savez_compressed(out_a, traces=sess_a["traces"],
                            mismatch_flags=sess_a["flags"],
                            meta=np.array(meta_a, dtype=object))
        print(f"  [OK] saved {out_a}, traces.shape={sess_a['traces'].shape}")

        # === Session B — same sk re-capture (no reset/F) for cross-session check ===
        # Subset: just zero + a192 (the 100% classifier pair)
        print(f"\n[Session B] same sk re-capture, zero + a192 only, N={n_per_ct} each")
        time.sleep(2.0)
        target.flush()
        designs_def_b = [("zero", zero_c1, zero_c2)]
        c192 = _chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=192)
        designs_def_b.append(("a192", c192.c1, c192.c2))
        sess_b = run_session(scope, target, p, sk, designs_def_b, n_per_ct, samples)

        # Verify same sk via 'X'
        sk_b = bytearray()
        for idx in range(n_sk_chunks):
            target.simpleserial_write("X", bytes([idx]))
            sk_b.extend(_ack(target, 32, timeout_ms=2000))
        if bytes(sk_b).hex() != sk_hex:
            raise RuntimeError(f"sk drift between sessions: A={sk_hex[:16]} B={sk_b.hex()[:16]}")
        print(f"  sk verified across sessions: {sk_hex[:16]}")

        meta_b = dict(meta_a)
        meta_b["session"] = "B"
        meta_b["design_names"] = [d[0] for d in designs_def_b]
        meta_b["ct_fp_per_design"] = sess_b["ct_fp"]
        meta_b["mu_hw_per_design"] = {n: int(sess_b["mu_pred"][n].sum()) for n in sess_b["mu_pred"]}
        meta_b["label_design"] = sess_b["labels"].tolist()
        meta_b["mismatch_flag_per_trace"] = sess_b["flags"].tolist()
        meta_b["captured_at"] = datetime.now(timezone.utc).isoformat()
        out_b = _REPO / "traces/pco_pilot2_session_b.npz"
        np.savez_compressed(out_b, traces=sess_b["traces"],
                            mismatch_flags=sess_b["flags"],
                            meta=np.array(meta_b, dtype=object))
        print(f"  [OK] saved {out_b}, traces.shape={sess_b['traces'].shape}")

    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
