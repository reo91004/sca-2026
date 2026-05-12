#!/usr/bin/env python3
"""S1U matrix capture — isolated multi-term `poly_mul_acc` diagnostic.

Firmware command `U` is the multi-term sibling of diagnostic `T`. It triggers
only around:

    poly_mul_acc(sk[component], public_b, out)

where `public_b` is built from up to eight public `(coef, alpha)` terms. By
default this script shifts each ciphertext-domain alpha left by 8 before sending
it, matching the `c1 << 8` convention used by `vec_vec_mult_add` inside `Z`.

This is diagnostic-only. Its purpose is to check whether the Toom/Karatsuba
labels that weakly appear in S2 are visible in an isolated multiplication window
before we spend more capture time on natural `Z`/`D` windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from host.smaug.poly_mul import board_response_mod_p, negacyclic_mul_mod_p  # noqa: E402
from scripts.s1_t_roundtrip import dump_sk_pke, int16_wrap, issue_u  # noqa: E402


def _file_sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_int_list(text: str) -> list[int]:
    out = [int(part.strip(), 0) for part in text.split(",") if part.strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def _make_designs(args: argparse.Namespace, p) -> list[list[tuple[int, int]]]:
    if args.design_file is not None:
        data = json.loads(args.design_file.read_text())
        raw = data["selected"] if "selected" in data else data
        designs = []
        for item in raw:
            terms = item["terms"] if isinstance(item, dict) else item
            designs.append([(int(coef), int(alpha)) for coef, alpha in terms])
        return designs
    rng = np.random.default_rng(args.design_seed)
    alpha_choices = _parse_int_list(args.alpha_choices)
    designs: list[list[tuple[int, int]]] = []
    for _ in range(args.num_designs):
        coefs = rng.choice(p.n, size=args.terms, replace=False)
        alphas = rng.choice(np.asarray(alpha_choices, dtype=np.int64), size=args.terms)
        designs.append([(int(c), int(a)) for c, a in zip(coefs, alphas)])
    return designs


def _payload_terms(terms: list[tuple[int, int]], shift: int) -> list[tuple[int, int]]:
    return [(coef, int16_wrap(int(alpha) << shift)) for coef, alpha in terms]


def _b_from_terms(terms: list[tuple[int, int]], n: int) -> np.ndarray:
    b = np.zeros(n, dtype=np.int64)
    for coef, alpha in terms:
        b[int(coef)] += int16_wrap(alpha)
    return b


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-keys", type=int, default=4)
    p.add_argument("-n", "--num-traces", type=int, default=100)
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--component", type=int, default=0, choices=(0, 1))
    p.add_argument("--num-designs", type=int, default=8)
    p.add_argument("--terms", type=int, default=4)
    p.add_argument("--design-seed", type=int, default=20260505)
    p.add_argument("--design-file", type=Path, default=None)
    p.add_argument("--alpha-choices", default="32,64,96,128,160,192,224")
    p.add_argument(
        "--alpha-shift",
        type=int,
        default=8,
        help="Shift public R_p alpha before sending to U. Use 8 to mimic Z c1<<8.",
    )
    p.add_argument(
        "--firmware-hex",
        type=Path,
        default=_REPO / "firmware" / "simpleserial-smaug" / "simpleserial-smaug-CW308_STM32F4.hex",
    )
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--serial", default=None)
    p.add_argument("--out-dir", type=Path, default=_REPO / "traces")
    p.add_argument("--tag", default="s1_u_matrix")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    p = _params.get("smaug1")
    p.assert_consistent()
    designs = _make_designs(args, p)
    if any(len(terms) > 8 for terms in designs):
        raise SystemExit("U supports at most 8 terms per design")
    for terms in designs:
        for coef, alpha in terms:
            if not (0 <= coef < p.n):
                raise SystemExit(f"coef {coef} outside [0, {p.n})")
            if not (0 <= alpha < p.p):
                raise SystemExit(f"alpha {alpha} outside R_p=[0, {p.p})")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fw_sha = _file_sha256(args.firmware_hex)

    import chipwhisperer as cw  # noqa: E402
    from host.cw_serial import pick_serial  # noqa: E402

    sn = pick_serial(args.serial)
    print(f"[INFO] CW1173 sn={sn}")
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

    started = time.time()
    try:
        for key_i in range(args.num_keys):
            target.simpleserial_write("F", b"")
            pk_fp16_b = target.simpleserial_read("r", 16, timeout=5000)
            if pk_fp16_b is None or len(pk_fp16_b) != 16:
                raise RuntimeError("F ack failed")
            pk_fp16 = bytes(pk_fp16_b)
            sk_pke = dump_sk_pke(target, p.pke_secret_key_bytes // 32)
            sk_full = _codec.unpack_sx(sk_pke).astype(np.int64)
            sk = sk_full[args.component * p.n : (args.component + 1) * p.n]
            print(f"[KEY {key_i:02d}] pk={pk_fp16.hex()[:8]} sk[0:4]={sk_pke[:4].hex()}")

            all_traces = []
            all_acks = []
            design_meta = []
            min_ok = args.num_traces
            for design_i, terms in enumerate(designs):
                payload_terms = _payload_terms(terms, args.alpha_shift)
                b = _b_from_terms(payload_terms, p.n)
                host_modp = negacyclic_mul_mod_p(sk, b, log_p=p.log_p, signed=True)[:16]
                first = issue_u(target, args.component, payload_terms)
                board_modp = board_response_mod_p(first, log_p=p.log_p, signed=True)
                if not np.array_equal(host_modp, board_modp):
                    raise RuntimeError(
                        f"U round-trip fail key={key_i} design={design_i}: "
                        f"host={host_modp.tolist()} board={board_modp.tolist()}"
                    )

                traces = np.empty((args.num_traces, args.samples), dtype=np.float32)
                acks = np.zeros((args.num_traces, 32), dtype=np.uint8)
                n_ok = 0
                for trace_i in range(args.num_traces):
                    scope.arm()
                    target.simpleserial_write(
                        "U",
                        bytes([args.component & 0xFF, len(payload_terms) & 0xFF])
                        + b"".join(
                            bytes([
                                (coef >> 8) & 0xFF,
                                coef & 0xFF,
                                ((int16_wrap(alpha) & 0xFFFF) >> 8) & 0xFF,
                                (int16_wrap(alpha) & 0xFFFF) & 0xFF,
                            ])
                            for coef, alpha in payload_terms
                        )
                        + bytes(4 * (8 - len(payload_terms))),
                    )
                    if scope.capture():
                        print(f"[WARN] timeout key={key_i} design={design_i} trace={trace_i}")
                        continue
                    ack = target.simpleserial_read("r", 32, timeout=5000)
                    if ack is None or len(ack) != 32:
                        print(f"[WARN] ack fail key={key_i} design={design_i} trace={trace_i}")
                        continue
                    traces[n_ok] = scope.get_last_trace().astype(np.float32)
                    acks[n_ok] = bytearray(ack)
                    n_ok += 1
                if n_ok == 0:
                    raise RuntimeError(f"no traces for key={key_i} design={design_i}")
                if not np.all(acks[:n_ok] == np.frombuffer(first, dtype=np.uint8)):
                    raise RuntimeError(f"ack nondeterminism key={key_i} design={design_i}")
                min_ok = min(min_ok, n_ok)
                all_traces.append(traces[:n_ok])
                all_acks.append(acks[:n_ok])
                design_meta.append(
                    {
                        "component": args.component,
                        "terms": terms,
                        "payload_terms": payload_terms,
                        "alpha_shift": args.alpha_shift,
                        "first_resp_hex": first.hex(),
                    }
                )
                print(
                    f"[KEY {key_i:02d}] design {design_i+1}/{len(designs)} "
                    f"terms={terms} payload={payload_terms} ok={n_ok}/{args.num_traces}"
                )

            traces_out = np.stack([x[:min_ok] for x in all_traces], axis=0)
            acks_out = np.stack([x[:min_ok] for x in all_acks], axis=0)
            meta = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "scope_sn": sn,
                "samples": args.samples,
                "gain_db": args.gain_db,
                "target": "smaug1",
                "cmd": "U",
                "capture_kind": "u_matrix",
                "level": "smaug1",
                "module_rank": p.module_rank,
                "lwe_n": p.n,
                "hs": p.hs,
                "baud": args.baud,
                "n_attempted_per_design": args.num_traces,
                "n_ok_per_design": int(min_ok),
                "firmware_hex": str(args.firmware_hex),
                "firmware_hex_sha256": fw_sha,
                "pk_fp16": pk_fp16.hex(),
                "sk_pke_hex": sk_pke.hex(),
                "component": args.component,
                "design_seed": args.design_seed,
                "design_terms": designs,
                "alpha_shift": args.alpha_shift,
                "label": f"{args.tag}_k{key_i:02d}_{pk_fp16.hex()[:8]}",
            }
            out_path = args.out_dir / f"{args.tag}_k{key_i:02d}_d{len(designs)}_n{min_ok}_{pk_fp16.hex()[:8]}.npz"
            np.savez_compressed(
                out_path,
                traces=traces_out,
                acks=acks_out,
                designs=np.array(design_meta, dtype=object),
                meta=np.array(meta, dtype=object),
            )
            elapsed = time.time() - started
            print(f"[OK] saved {out_path} shape={traces_out.shape} elapsed={elapsed:.1f}s")
    finally:
        try:
            target.dis()
        except Exception:
            pass
        try:
            scope.dis()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
