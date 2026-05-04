#!/usr/bin/env python3
"""PCO pilot 6 — diverse µ' patterns × multiple sk for bit-level cross-sk CPA.

Pilot 5 finding: reject trace 의 cluster signal 은 z (random rejection seed) 가
dominant. sparse MLWR sx 직접 복구 안 됨.

Pilot 6 핵심 질문:
  Q. 'D' trace 의 LATE zone (indcpa_enc 재암호화 단계 포함) 에서 µ'[bit_i] 가
     *cross-sk* leak 되는가?
     - YES → sparse sk 복구 path. indcpa_dec 단계 (E1-E5 random) 와 다른 phase.
     - NO  → 'D' trace 도 cross-key µ'-bit leak 없음 → SMAUG-T trace SCA 한계 확인.

설계
----
4 sk × 4 designs (cyclic shifts) × N=32 = 512 traces

designs (모두 c2=0):
  j0 : c1[0, 0]=192   → µ'[i] = (sk[0][i]   != 0)
  j64: c1[0, 64]=192  → µ'[i] = (sk[0][i-64]!= 0)
  j128: c1[0,128]=192 → µ'[i] = (sk[0][i-128] != 0)
  j192: c1[0,192]=192 → µ'[i] = (sk[0][i-192] != 0)

각 (design, sk) 별 µ' 가 sk[0] 의 cyclic shift. 16 unique patterns.

분석 (analyze_pco_pilot6.py):
  for sample s in LATE:
    for bit_i in [0, 256):
      compute Welch-t on traces partitioned by µ'_per_(design,sk)[bit_i] ∈ {0, 1}
  → 강한 peaks at (s, i) pairs cross-sk consistent → µ'-bit leak

산출:
  traces/pco_pilot6.npz
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


def session_for_sk(scope, target, p, design_specs, n_per_design, samples):
    """Session for one sk: keygen + N traces per design."""
    target.simpleserial_write("F", b"")
    pk_fp16 = _ack(target, 16)
    n_sk_chunks = p.pke_secret_key_bytes // 32
    sk_pke = bytearray()
    for idx in range(n_sk_chunks):
        target.simpleserial_write("X", bytes([idx]))
        sk_pke.extend(_ack(target, 32, timeout_ms=2000))
    sk = _codec.unpack_sx(bytes(sk_pke)).reshape(p.module_rank, p.n).astype(np.int64)
    sk_hex = bytes(sk_pke).hex()

    out_designs = []
    for name, j_idx, alpha in design_specs:
        c = _chosen.build_monomial_c1(p, component=0, coef_idx=j_idx, alpha=alpha)
        ct_bytes = Ciphertext(c1=c.c1, c2=c.c2, params=p).to_bytes()
        host_fp = hashlib.sha3_256(ct_bytes).hexdigest()[:32]
        fp = inject_ct(target, ct_bytes)
        if fp.hex() != host_fp:
            raise RuntimeError(f"ct fp mismatch for {name}")
        mu_pred = _chosen.predict_mu_prime(p, c.c1, sk, c2=c.c2)
        traces, flags = capture_reject(scope, target, n_per_design, samples)
        out_designs.append({
            "name": name, "traces": traces, "flags": flags,
            "mu_pred_bits": mu_pred.astype(np.uint8),
            "mu_hw": int(mu_pred.sum()),
            "ct_fp": host_fp,
        })
    return {
        "sk_hex": sk_hex, "pk_fp16": pk_fp16.hex(),
        "designs": out_designs,
    }


def main() -> int:
    level = "smaug1"
    n_per_design = 32
    samples = 24400
    p = _params.get(level)
    n_sessions = 4

    # 4 designs: c1 = 192 · X^j for j ∈ {0, 64, 128, 192}
    # all have HW(µ') = HW(sk[0]) = HS = 70 but different bit patterns (cyclic shifts).
    design_specs = [
        ("j0",   0,   192),
        ("j64",  64,  192),
        ("j128", 128, 192),
        ("j192", 192, 192),
    ]

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(None)

    sessions = []
    for s_idx in range(n_sessions):
        print(f"\n=== Session {s_idx+1}/{n_sessions} ===")
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
            sess = session_for_sk(scope, target, p, design_specs, n_per_design, samples)
            sessions.append(sess)
            print(f"  sk={sess['sk_hex'][:16]}, pk_fp={sess['pk_fp16']}")
            for d in sess["designs"]:
                rejs = int((d["flags"] == 1).sum())
                print(f"    {d['name']:6s}: HW(µ')={d['mu_hw']}, rej={rejs}/{n_per_design}")
        finally:
            try: target.dis()
            except Exception: pass
            try: scope.dis()
            except Exception: pass

    # save
    all_traces = []
    all_flags = []
    label_design = []
    label_session = []
    mu_per_trace = []  # (n_total, 256) uint8
    for s_idx, sess in enumerate(sessions):
        for d in sess["designs"]:
            all_traces.append(d["traces"])
            all_flags.append(d["flags"])
            label_design.extend([d["name"]] * n_per_design)
            label_session.extend([s_idx] * n_per_design)
            mu_per_trace.extend([d["mu_pred_bits"]] * n_per_design)
    traces = np.concatenate(all_traces, axis=0)
    flags = np.concatenate(all_flags)
    mu_per_trace = np.array(mu_per_trace, dtype=np.uint8)

    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope_sn": sn, "samples": samples, "gain_db": 25.0,
        "clock_adc_src": "clkgen_x1",
        "level": level,
        "n_sessions": n_sessions,
        "n_per_design": n_per_design,
        "design_specs": [list(s) for s in design_specs],
        "design_names": [s[0] for s in design_specs],
        "label_design": label_design,
        "label_session": label_session,
        "mismatch_flag_per_trace": flags.tolist(),
        "sk_pke_hex_per_session": [s["sk_hex"] for s in sessions],
        "pk_fp16_per_session": [s["pk_fp16"] for s in sessions],
        "mu_hw_per_design_per_session": [
            {d["name"]: d["mu_hw"] for d in s["designs"]} for s in sessions
        ],
        "note": "Pilot 6 — diverse µ' patterns (4 cyclic shifts × 4 sk). Bit-level cross-sk CPA.",
    }
    out = _REPO / "traces/pco_pilot6.npz"
    np.savez_compressed(
        out, traces=traces,
        mismatch_flags=flags,
        mu_per_trace=mu_per_trace,  # (n_total, 256) bit prediction per trace
        meta=np.array(meta, dtype=object),
    )
    print(f"\n[OK] saved {out}, traces.shape={traces.shape}, mu_per_trace.shape={mu_per_trace.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
