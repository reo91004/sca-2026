#!/usr/bin/env python3
"""S2 main capture — chosen-CT 'Z' (indcpa_dec only) trace × N for fixed sk.

흐름 :
  1. 'F' fresh keypair, sk persistent.
  2. 'X' x 4 → sk_pke ground truth.
  3. host 가 chosen-CT (c1 = α·X^j component k, c2=0) 빌드.
  4. 'I' x 21 chunks + 'L' 무결성 검증 (host vs board sha3_256).
  5. 첫 'Z' 호출: µ′ 응답 = host predict_mu_prime 와 byte-for-byte 일치 검증.
     이게 통과해야 capture 시작 (아니면 보드와 host 가 다른 ct 인 것).
  6. 'Z' × N traces capture. µ′ 응답 무시 (이번 capture 는 attack-valid trace
     dump 이지 instrumented oracle 사용 X).
  7. 저장 — traces, acks, sk_pke_hex, ct_fp16, meta.

S1 ('T') 와 의도적으로 동일한 capture 파라미터:
  - clkgen_x4, samples=24400, gain=25.0
  - 같은 보드, 같은 firmware (smaug1)
  - chosen-CT 의 c1 = α·X^0 component 0 → vec_vec_mult_add(sk, [c1=4·X^0, c1[1]=0])
    결과 = sk_polyvec[0] * 4 mod (X^256+1) — 수학적으로 'T' (c=0, j=0, α=1) ×4
    동치.

run::
    python3 scripts/s2_z_capture_main.py -n 200 --alpha 4 \
        --out traces/s2_z_skX_a4_n200.npz
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target, setup_session  # noqa: E402
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from scripts.smaug.s1_t_roundtrip import dump_sk_pke  # noqa: E402


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--num-traces", type=int, default=200)
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument(
        "--adc-offset", type=int, default=0,
        help="trigger 후 skip 할 ADC sample 수 (vec_vec_mult_add 영역 정렬용)",
    )
    p.add_argument(
        "--component", type=int, default=0, choices=(0, 1),
        help="chosen c1 의 어떤 component (default 0)",
    )
    p.add_argument(
        "--coef", type=int, default=0,
        help="chosen c1 의 단항 X^j 의 j (default 0)",
    )
    p.add_argument(
        "--alpha", type=int, default=4,
        help="c1[component, coef] 의 R_p 계수. smaug1 multiples of 4 → fixed-point.",
    )
    p.add_argument(
        "--firmware-hex",
        type=Path,
        default=_REPO
        / "firmware"
        / "simpleserial-smaug"
        / "simpleserial-smaug-CW308_STM32F4.hex",
    )
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--serial", default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    p = _params.get("smaug1")
    p.assert_consistent()
    if not (0 <= args.alpha < p.p):
        raise SystemExit(f"--alpha {args.alpha} ∉ [0, {p.p})")
    out_path = args.out
    if out_path is None:
        tag = f"c{args.component}_j{args.coef}_a{args.alpha}_n{args.num_traces}"
        out_path = _REPO / "traces" / f"s2_z_{tag}_TEMP.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fw_sha = _file_sha256(args.firmware_hex) if args.firmware_hex.exists() else None
    print(f"[INFO] firmware_hex={args.firmware_hex.name} sha256={fw_sha}")

    # chosen-CT 빌드 + bytes
    ct = _chosen.build_monomial_c1(
        p, component=args.component, coef_idx=args.coef, alpha=args.alpha
    )
    ct_bytes = ct.to_bytes()
    if len(ct_bytes) != p.ciphertext_bytes:
        raise SystemExit(
            f"ct len {len(ct_bytes)} != CIPHERTEXT_BYTES {p.ciphertext_bytes}"
        )
    ct_fp16_host = hashlib.sha3_256(ct_bytes).digest()[:16]
    print(
        f"[INFO] chosen-CT c{args.component}·X^{args.coef}·{args.alpha}, "
        f"len={len(ct_bytes)}B, ct_fp16_host={ct_fp16_host.hex()}"
    )

    import chipwhisperer as cw  # noqa: E402
    from host.cw_serial import pick_serial  # noqa: E402

    sn = pick_serial(args.serial)
    print(f"[INFO] CW1173 sn={sn}")
    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = args.samples
    scope.adc.offset = args.adc_offset
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = args.gain_db
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()
    time.sleep(0.3)

    started = time.time()
    try:
        # 1) F
        target.simpleserial_write("F", b"")
        pk_fp16 = bytes(target.simpleserial_read("r", 16, timeout=5000))
        print(f"[OK] F pk_fp16={pk_fp16.hex()}")

        # 2) X (sk_pke dump)
        sk_pke = dump_sk_pke(target, p.pke_secret_key_bytes // 32)
        print(f"[OK] X sk_pke[0:8]={sk_pke[:8].hex()}…")
        sk_unpacked_full = _codec.unpack_sx(sk_pke).astype(np.int64)
        sk_polyvec = sk_unpacked_full.reshape(p.module_rank, p.n)

        # 3) inject ct into the resident key session. Do not call 'F' again:
        # sk_pke above must label the exact key used by this 'Z' capture.
        bundle = setup_session(
            target,
            ct_bytes,
            label="s2_z_chosen",
            fresh_key=False,
            pk_fp16=pk_fp16,
        )
        if bundle.ct_fp16_host != bundle.ct_fp16_board:
            raise RuntimeError(
                f"ct integrity fail: host={bundle.ct_fp16_host.hex()} "
                f"board={bundle.ct_fp16_board.hex()}"
            )
        print(f"[OK] inject + 'L' verify ct_fp16={bundle.ct_fp16_board.hex()}")

        # 4) 첫 'Z' — µ′ 응답 vs predict_mu_prime
        target.simpleserial_write("Z", b"")
        first_mu_resp = target.simpleserial_read("r", 32, timeout=5000)
        if first_mu_resp is None or len(first_mu_resp) != 32:
            raise RuntimeError("Z first ack fail")
        first_mu_resp = bytes(first_mu_resp)
        # predict
        mu_pred = _chosen.predict_mu_prime(p, ct.c1, sk_polyvec, ct.c2)
        # mu_pred shape (256,), values ∈ {0, 1}. Pack to bytes (LSB first within byte).
        mu_pred_packed = bytearray(32)
        for bit_i in range(256):
            if mu_pred[bit_i]:
                mu_pred_packed[bit_i // 8] |= 1 << (bit_i % 8)
        if bytes(mu_pred_packed) != first_mu_resp:
            # detail diagnose
            n_diff_bits = 0
            for byte_i in range(32):
                xor = mu_pred_packed[byte_i] ^ first_mu_resp[byte_i]
                n_diff_bits += bin(xor).count("1")
            raise RuntimeError(
                f"µ′ round-trip fail: {n_diff_bits}/256 bits differ. "
                f"host={bytes(mu_pred_packed).hex()} brd={first_mu_resp.hex()}"
            )
        print(f"[OK] µ′ round-trip 256/256 bits 일치")

        # 5) 'Z' x N capture
        traces = np.empty((args.num_traces, args.samples), dtype=np.float32)
        acks = np.zeros((args.num_traces, 32), dtype=np.uint8)
        timeouts = 0
        n_ok = 0
        for i in range(args.num_traces):
            scope.arm()
            target.simpleserial_write("Z", b"")
            if scope.capture():
                timeouts += 1
                print(f"[WARN] scope timeout @ {i}")
                continue
            ack = target.simpleserial_read("r", 32, timeout=5000)
            if ack is None or len(ack) != 32:
                timeouts += 1
                print(f"[WARN] short ack @ {i}")
                continue
            traces[i] = scope.get_last_trace().astype(np.float32)
            acks[i] = bytearray(ack)
            n_ok += 1
            if (i + 1) % 100 == 0:
                el = time.time() - started
                print(f"[PROG] {i+1}/{args.num_traces}  ({el:.1f}s, {(i+1)/el:.2f} tr/s)")

        # ack determinism
        unique_acks = np.unique(acks[:n_ok], axis=0)
        if unique_acks.shape[0] != 1:
            raise RuntimeError(
                f"ack 다양성 {unique_acks.shape[0]} — 보드 sk 변경 또는 µ′ 비결정성 의심"
            )
        # 첫 µ′ 와 동일한지 마지막 검증
        if bytes(acks[0]) != first_mu_resp:
            raise RuntimeError(
                "first capture ack != initial round-trip ack — race condition?"
            )
        print(f"[OK] {n_ok} traces ack 일관성")

        # adc trig_count
        try:
            tc = scope.adc.trig_count
            print(f"[INFO] scope.adc.trig_count = {tc}")
        except Exception:
            tc = -1

        # rename out path with pk_fp16 short (uniqueness across sks)
        if "TEMP" in str(out_path):
            tag2 = pk_fp16.hex()[:8]
            out_path = out_path.with_name(out_path.name.replace("TEMP", tag2))

        meta = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "scope_sn": sn,
            "samples": args.samples,
            "gain_db": args.gain_db,
            "adc_offset": args.adc_offset,
            "target": "smaug1",
            "cmd": "Z",
            "level": "smaug1",
            "log_p": p.log_p,
            "module_rank": p.module_rank,
            "lwe_n": p.n,
            "ss_ver": "SS_VER_1_1",
            "baud": args.baud,
            "n_attempted": args.num_traces,
            "n_ok": n_ok,
            "n_timeouts": timeouts,
            "firmware_hex": str(args.firmware_hex),
            "firmware_hex_sha256": fw_sha,
            "pk_fp16": pk_fp16.hex(),
            "sk_pke_hex": sk_pke.hex(),
            "ct_bytes_sha256": hashlib.sha256(ct_bytes).hexdigest(),
            "ct_fp16_board": bundle.ct_fp16_board.hex(),
            "ct_fp16_host": bundle.ct_fp16_host.hex(),
            "ct_label": f"c{args.component}_X{args.coef}_a{args.alpha}",
            "alpha": args.alpha,
            "component": args.component,
            "coef_idx": args.coef,
            "first_mu_resp_hex": first_mu_resp.hex(),
            "trig_count": int(tc),
            "label": f"s2_z_{out_path.stem}",
        }
        np.savez_compressed(
            out_path,
            traces=traces[:n_ok],
            acks=acks[:n_ok],
            meta=np.array(meta, dtype=object),
        )
        el = time.time() - started
        print(f"[OK] saved {n_ok}/{args.num_traces} -> {out_path}  ({el:.1f}s)")

    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
