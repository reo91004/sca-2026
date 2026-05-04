#!/usr/bin/env python3
"""PCO pilot 4 — cross-session/cross-sk accept-vs-reject classifier transfer.

Pilot 3 result: same-sk session 안 100% accept/reject classifier @ sample 23284.
Pilot 4 핵심 질문: 다른 sk (board 'F' 로 new keygen) 에서도 같은 PoI @ 23284 가
1-bit oracle 으로 작동하는가?

만약 YES → oracle 이 universal. attacker 가 자기 own calibration trace 만으로
                            target session 안 sk 복구 가능.
만약 NO  → per-session calibration 필수 (하지만 그래도 attack-valid — 'e'+'d' 로
                            calibration set 만들고 그걸로 chosen-CT trace 분류).

설계
----
- Session B: reset_target → new 'F' (different sk)
- Same protocol: accept (e+d, N=64), reject_zero (I+L+D, N=64), reject_a192 (N=64)
- Save to traces/pco_pilot4_session_b.npz
- Cross-session evaluation:
  * train PoI on Pilot 3 session A → apply to session B
  * train PoI on session B itself → compare
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


def capture_accept(scope, target, n, samples):
    traces = np.empty((n, samples), dtype=np.float32)
    flags = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        target.simpleserial_write("e", b"")
        _ = _ack(target, 16, timeout_ms=10000)
        scope.arm()
        target.simpleserial_write("d", b"")
        if scope.capture():
            raise RuntimeError(f"scope timeout accept {i}")
        a = target.simpleserial_read("r", 1, timeout=10000)
        if a is None or len(a) != 1:
            raise RuntimeError(f"d ack {i}")
        traces[i] = scope.get_last_trace()
        flags[i] = a[0]
    return traces, flags


def capture_reject(scope, target, n, samples):
    traces = np.empty((n, samples), dtype=np.float32)
    flags = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        scope.arm()
        target.simpleserial_write("D", b"")
        if scope.capture():
            raise RuntimeError(f"scope timeout reject {i}")
        a = target.simpleserial_read("r", 1, timeout=10000)
        if a is None or len(a) != 1:
            raise RuntimeError(f"D ack {i}")
        traces[i] = scope.get_last_trace()
        flags[i] = a[0]
    return traces, flags


def main() -> int:
    level = "smaug1"
    n_per_design = 64
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
    scope.clock.adc_src = "clkgen_x1"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = 38400
    target.flush()
    time.sleep(0.3)

    try:
        target.simpleserial_write("F", b"")
        pk_fp16 = _ack(target, 16)
        n_sk_chunks = p.pke_secret_key_bytes // 32
        sk_pke = bytearray()
        for idx in range(n_sk_chunks):
            target.simpleserial_write("X", bytes([idx]))
            sk_pke.extend(_ack(target, 32, timeout_ms=2000))
        sk_hex = bytes(sk_pke).hex()
        print(f"  Session B sk={sk_hex[:16]}, pk_fp={pk_fp16.hex()}")

        # Group A: accept
        print(f"\n[A] accept (e+d), N={n_per_design}")
        tr_a, fl_a = capture_accept(scope, target, n_per_design, samples)
        print(f"  flags: 0={int((fl_a==0).sum())}/{n_per_design}")

        # Group B: reject zero
        print(f"\n[B] reject_zero, N={n_per_design}")
        zero_c1 = np.zeros((p.module_rank, p.n), dtype=np.int64)
        zero_c2 = np.zeros(p.n, dtype=np.int64)
        ct_zero = Ciphertext(c1=zero_c1, c2=zero_c2, params=p).to_bytes()
        host_fp_z = hashlib.sha3_256(ct_zero).hexdigest()[:32]
        fp_z = inject_ct(target, ct_zero)
        if fp_z.hex() != host_fp_z:
            raise RuntimeError("zero ct fp mismatch")
        tr_b, fl_b = capture_reject(scope, target, n_per_design, samples)
        print(f"  flags: 1={int((fl_b==1).sum())}/{n_per_design}")

        # Group C: reject a192
        print(f"\n[C] reject_a192, N={n_per_design}")
        c192 = _chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=192)
        ct_192 = Ciphertext(c1=c192.c1, c2=c192.c2, params=p).to_bytes()
        host_fp_c = hashlib.sha3_256(ct_192).hexdigest()[:32]
        fp_c = inject_ct(target, ct_192)
        if fp_c.hex() != host_fp_c:
            raise RuntimeError("a192 ct fp mismatch")
        tr_c, fl_c = capture_reject(scope, target, n_per_design, samples)
        print(f"  flags: 1={int((fl_c==1).sum())}/{n_per_design}")

        traces = np.concatenate([tr_a, tr_b, tr_c], axis=0)
        flags = np.concatenate([fl_a, fl_b, fl_c], axis=0)
        labels = (["accept"] * n_per_design + ["reject_zero"] * n_per_design
                  + ["reject_a192"] * n_per_design)

        meta = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "scope_sn": sn, "samples": samples, "gain_db": 25.0,
            "clock_adc_src": "clkgen_x1",
            "level": level,
            "module_rank": p.module_rank, "hs": p.hs, "log_p": p.log_p,
            "n_per_design": n_per_design,
            "design_names": ["accept", "reject_zero", "reject_a192"],
            "label_design": labels,
            "mismatch_flag_per_trace": flags.tolist(),
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_bytes_hex": sk_hex,
            "session": "B",
            "note": "Pilot 4 — different sk than pilot 3. Cross-session classifier transfer test.",
        }
        out = _REPO / "traces/pco_pilot4_session_b.npz"
        np.savez_compressed(out, traces=traces,
                            mismatch_flags=flags,
                            meta=np.array(meta, dtype=object))
        print(f"\n  [OK] saved {out}, traces.shape={traces.shape}")

    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
