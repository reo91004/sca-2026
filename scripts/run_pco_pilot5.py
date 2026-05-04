#!/usr/bin/env python3
"""PCO pilot 5 — same chosen-CT, multiple sk → does the LATE PoI carry sk info?

Pilot 4 showed: cross-sk classifier transfer 99%. But that just confirms the
oracle (accept vs reject) is sk-universal — it doesn't show that our chosen-CT
traces *themselves* carry sk-bit info.

Pilot 5 핵심 질문:
  zero/a192 같은 chosen-CT 는 *모든 sk 에서 reject*. 같은 reject-class 안에서
  sk 마다 trace 가 distinguishable 한가? 즉:
    sample 23284 의 power 가 sk 의 함수인가? 아니면 단순 FO branch indicator 인가?

설계
----
  4 sk × 2 designs (zero, a192) × N=32 = 256 traces
  - 매 sk: reset_target → 'F' 새 keygen → 'X' dump sk → 'I'+'L'+'D' 각 design 32회
  - 모두 reject (mismatch=1) — 같은 class 에 속함
  - PoI 23284 의 trace 값이 sk 마다 cluster 가 다르면 → sk-bit leak 존재
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


def session_for_sk(scope, target, p, n_per_design, samples):
    """Single sk session: 'F' keygen → dump sk → 2 designs of N traces each."""
    target.simpleserial_write("F", b"")
    pk_fp16 = _ack(target, 16)
    n_sk_chunks = p.pke_secret_key_bytes // 32
    sk_pke = bytearray()
    for idx in range(n_sk_chunks):
        target.simpleserial_write("X", bytes([idx]))
        sk_pke.extend(_ack(target, 32, timeout_ms=2000))
    sk = _codec.unpack_sx(bytes(sk_pke)).reshape(p.module_rank, p.n).astype(np.int64)
    sk_hex = bytes(sk_pke).hex()

    # zero
    zero_c1 = np.zeros((p.module_rank, p.n), dtype=np.int64)
    zero_c2 = np.zeros(p.n, dtype=np.int64)
    ct_zero = Ciphertext(c1=zero_c1, c2=zero_c2, params=p).to_bytes()
    fp_z_host = hashlib.sha3_256(ct_zero).hexdigest()[:32]
    fp_z = inject_ct(target, ct_zero)
    if fp_z.hex() != fp_z_host:
        raise RuntimeError("zero ct fp mismatch")
    tr_z, fl_z = capture_reject(scope, target, n_per_design, samples)

    # a192
    c192 = _chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=192)
    ct_192 = Ciphertext(c1=c192.c1, c2=c192.c2, params=p).to_bytes()
    fp_a_host = hashlib.sha3_256(ct_192).hexdigest()[:32]
    fp_a = inject_ct(target, ct_192)
    if fp_a.hex() != fp_a_host:
        raise RuntimeError("a192 ct fp mismatch")
    tr_a, fl_a = capture_reject(scope, target, n_per_design, samples)

    mu_zero = _chosen.predict_mu_prime(p, zero_c1, sk, c2=zero_c2)
    mu_a192 = _chosen.predict_mu_prime(p, c192.c1, sk, c2=c192.c2)
    return {
        "sk_hex": sk_hex,
        "pk_fp16": pk_fp16.hex(),
        "tr_zero": tr_z, "fl_zero": fl_z,
        "tr_a192": tr_a, "fl_a192": fl_a,
        "mu_zero_hw": int(mu_zero.sum()),
        "mu_a192_hw": int(mu_a192.sum()),
        "mu_zero": mu_zero, "mu_a192": mu_a192,
    }


def main() -> int:
    level = "smaug1"
    n_per_design = 32
    samples = 24400
    p = _params.get(level)

    n_sessions = 4

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(None)

    sessions = []
    for s in range(n_sessions):
        print(f"\n=== Session {s+1}/{n_sessions} ===")
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
            sess = session_for_sk(scope, target, p, n_per_design, samples)
            sessions.append(sess)
            print(f"  sk={sess['sk_hex'][:16]}, pk_fp={sess['pk_fp16']}")
            print(f"  HW(µ'_zero)={sess['mu_zero_hw']}, HW(µ'_a192)={sess['mu_a192_hw']}")
            print(f"  zero rejects: {int((sess['fl_zero']==1).sum())}/{n_per_design}")
            print(f"  a192 rejects: {int((sess['fl_a192']==1).sum())}/{n_per_design}")
        finally:
            try: target.dis()
            except Exception: pass
            try: scope.dis()
            except Exception: pass

    # save
    traces = np.concatenate([s["tr_zero"] for s in sessions]
                          + [s["tr_a192"] for s in sessions], axis=0)
    flags = np.concatenate([s["fl_zero"] for s in sessions]
                         + [s["fl_a192"] for s in sessions])
    labels_design = (["zero"] * (n_sessions * n_per_design)
                   + ["a192"] * (n_sessions * n_per_design))
    labels_session = []
    for label_grp in ("zero", "a192"):
        for s_idx in range(n_sessions):
            labels_session.extend([s_idx] * n_per_design)

    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope_sn": sn, "samples": samples, "gain_db": 25.0,
        "clock_adc_src": "clkgen_x1",
        "level": level,
        "n_sessions": n_sessions,
        "n_per_design": n_per_design,
        "label_design": labels_design,
        "label_session": labels_session,
        "mismatch_flag_per_trace": flags.tolist(),
        "sk_pke_hex_per_session": [s["sk_hex"] for s in sessions],
        "pk_fp16_per_session": [s["pk_fp16"] for s in sessions],
        "mu_zero_hw_per_session": [s["mu_zero_hw"] for s in sessions],
        "mu_a192_hw_per_session": [s["mu_a192_hw"] for s in sessions],
        "mu_zero_per_session_hex": [bytes(np.packbits(s["mu_zero"], bitorder="little")).hex() for s in sessions],
        "mu_a192_per_session_hex": [bytes(np.packbits(s["mu_a192"], bitorder="little")).hex() for s in sessions],
        "note": "Pilot 5 — multi-sk same-design reject traces. Test if LATE PoI carries sk info.",
    }
    out = _REPO / "traces/pco_pilot5.npz"
    np.savez_compressed(out, traces=traces,
                        mismatch_flags=flags,
                        meta=np.array(meta, dtype=object))
    print(f"\n[OK] saved {out}, traces.shape={traces.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
