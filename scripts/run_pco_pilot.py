#!/usr/bin/env python3
"""PCO pilot — 'D' (full crypto_kem_dec) trace capture for FO/SHAKE leakage assessment.

목적
----
사용자 분석에 따라 cross-key trace SCA 가 random 인 SMAUG-T smaug1 에서
threat model 을 *target-key-internal PCO oracle* 로 전환 가능한지 측정.

PCO oracle 의 핵심 = trace 에서 *single bit* "accept/reject" 또는 µ' 의 어떤
feature (예: HW) 를 추출. 이게 cross-key 와 무관 (target session 안 self-comparison).

본 pilot 의 *제한적* 평가:
- 정확한 accept ct (= valid Encrypt(pk, µ, G(µ,...))) 는 host-side full PKE.Enc
  구현 필요 → 시간 비쌈. 따라서 본 pilot 은 *서로 다른 µ' 를 만드는* chosen-CT
  여러 개 사이의 'D' trace TVLA 만 평가.
- 만약 *late trace* (indcpa_enc + FO + final hash 단계) 에서 max|t| > 5 면 →
  µ' 가 SHAKE operand 로 들어가 HW model leak. MV-PC oracle 가능성.
- max|t| 가 *early trace* (indcpa_dec poly mul 단계) 에만 있으면 → 우리가 이미
  알던 indcpa_dec 의 µ' bit leak 만 보임. PCO 추가 단서 없음.

캡처 설계
---------
- target sk fixed (board RAM 에 'F' 로 keygen)
- 3 designs (예상되는 µ' HW 가 충분히 다른 ct):
  - design Z: c1=0, c2=0 (zero ct) → µ' = 0...0 (HW=0)
  - design 64: c1[0,0]=64 → µ'_i = round_t(64·sk_i / 256) 일부 1
  - design 192: c1[0,0]=192 → µ'_i = round_t(192·sk_i / 256) 일부 1 (보완적)
- 각 design × N=64 'D' trace
- 'D' trigger 가 full crypto_kem_dec 를 감싸므로 sample rate 낮춤 (clkgen_x2 → x1)
  로 24400 sample 안에 full dec 들어가게.

산출
----
- traces/pco_pilot.npz: traces (N×3, samples), label_alpha, mu_pred_per_design,
  mismatch_flags
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
    """N times 'D' (full crypto_kem_dec). returns (N, samples) trace + (N,) mismatch flag."""
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


def main() -> int:
    level = "smaug1"
    n_per_ct = 64
    samples = 24400
    gain_db = 25.0
    baud = 38400
    p = _params.get(level)

    fw_path = (_REPO / "firmware" / "simpleserial-smaug" /
               f"simpleserial-smaug-CW308_STM32F4-{level}.hex")

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(None)
    print(f"[PCO pilot] level={level}, n/CT={n_per_ct}, samples={samples}")

    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = samples
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = gain_db
    # 'D' 는 full crypto_kem_dec → indcpa_dec 보다 길다.
    # x4 (29 MHz) 에서 24400 sample = 840 µs. crypto_kem_dec 는 ~1500-2500 µs 추정.
    # x1 (7.4 MHz) 로 낮추면 24400 sample = 3.3 ms — full dec capture 가능.
    scope.clock.adc_src = "clkgen_x1"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = baud
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
        sk_short = bytes(sk_pke).hex()[:16]
        print(f"  sk={sk_short}, pk_fp={pk_fp16.hex()}")

        # 3 designs, computing predicted µ' per design
        designs: list[tuple[str, np.ndarray]] = []
        # Design A: zero ct → µ' = 0
        cA_c1 = np.zeros((p.module_rank, p.n), dtype=np.int64)
        cA_c2 = np.zeros(p.n, dtype=np.int64)
        designs.append(("zero", cA_c1, cA_c2))
        # Design B: c1[0,0] = 64 → µ' = leaked s[0]
        cB = _chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=64)
        designs.append(("c0_a64", cB.c1, cB.c2))
        # Design C: c1[0,0] = 192
        cC = _chosen.build_monomial_c1(p, component=0, coef_idx=0, alpha=192)
        designs.append(("c0_a192", cC.c1, cC.c2))

        all_traces: list[np.ndarray] = []
        all_flags: list[np.ndarray] = []
        all_labels: list[str] = []
        mu_per_design: dict[str, np.ndarray] = {}
        ct_fp_per_design: dict[str, str] = {}
        started = time.time()

        for name, c1, c2 in designs:
            from host.smaug.ciphertext import Ciphertext
            ct = Ciphertext(c1=c1, c2=c2, params=p)
            ct_bytes = ct.to_bytes()
            host_fp = hashlib.sha3_256(ct_bytes).hexdigest()[:32]
            fp = inject_ct(target, ct_bytes)
            board_fp = fp.hex()
            if board_fp != host_fp:
                raise RuntimeError(
                    f"ct fp mismatch host={host_fp} board={board_fp} — wrong firmware?")
            mu_pred = _chosen.predict_mu_prime(p, c1, sk, c2=c2)
            mu_hw = int(mu_pred.sum())
            ct_fp_per_design[name] = host_fp
            mu_per_design[name] = mu_pred
            print(f"  design {name:10s} ct_fp={host_fp[:16]}... mu_pred_HW={mu_hw}/256")
            tr, fl = capture_n_decap(scope, target, n_per_ct, samples)
            all_traces.append(tr)
            all_flags.append(fl)
            all_labels.extend([name] * n_per_ct)
            print(f"    captured {n_per_ct} traces, mismatch_flag mean={fl.mean():.2f}, "
                  f"sum_zero={int((fl == 0).sum())}/{n_per_ct} (= accept count)")

        traces = np.concatenate(all_traces, axis=0)
        flags = np.concatenate(all_flags, axis=0)
        labels = np.array(all_labels)
        elapsed = time.time() - started
        print(f"\n  total {traces.shape[0]} traces, {elapsed:.1f}s")

        meta = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "scope_sn": sn,
            "samples": samples,
            "gain_db": gain_db,
            "clock_adc_src": "clkgen_x1",
            "target": "smaug",
            "cmd": "D",
            "ss_ver": "SS_VER_1_1",
            "baud": baud,
            "level": level,
            "module_rank": p.module_rank,
            "hs": p.hs,
            "log_p": p.log_p,
            "n_per_design": n_per_ct,
            "design_names": [d[0] for d in designs],
            "ct_fp_per_design": ct_fp_per_design,
            "mu_pred_per_design_hex":
                {n: bytes(np.packbits(mu_per_design[n], bitorder="little")).hex()
                 for n in mu_per_design},
            "mu_hw_per_design": {n: int(mu_per_design[n].sum()) for n in mu_per_design},
            "label_design": labels.tolist(),
            "mismatch_flag_per_trace": flags.tolist(),
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_bytes_hex": bytes(sk_pke).hex(),
        }
        out = _REPO / "traces/pco_pilot.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, traces=traces, mismatch_flags=flags,
                            meta=np.array(meta, dtype=object))
        print(f"\n[OK] saved {out}, traces.shape={traces.shape}")

    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
