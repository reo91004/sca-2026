#!/usr/bin/env python3
"""S2/S3 matrix capture — multiple public chosen-CT designs per resident key.

This is the next dataset after the negative single-design S2.5 profiler:
for each fresh key, keep the key resident, dump sk for profiling labels, then
inject several public monomial or multi-term c1 designs and capture Z, V, W, or R
traces for each design.

The saved NPZ has:
  traces  : (num_designs, n_ok_min, samples) float32
  acks    : (num_designs, n_ok_min, 32) uint8
  designs : object array of {component, terms, ct fingerprints}
  meta    : object dict with resident pk/sk and capture settings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target, setup_session  # noqa: E402
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug.ciphertext import Ciphertext  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from scripts.s1_t_roundtrip import dump_sk_pke  # noqa: E402


def _file_sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _pack_mu_bits(bits: np.ndarray) -> bytes:
    out = bytearray(32)
    for i, bit in enumerate(bits):
        if int(bit):
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _parse_coef_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part, 0))
    if not out:
        raise ValueError("empty coef list")
    return out


def _parse_int_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part, 0))
    if not out:
        raise ValueError("empty int list")
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
    if args.design_mode == "monomial":
        return [[(coef, args.alpha)] for coef in _parse_coef_list(args.coefs)]
    if args.design_mode == "detector-grid":
        alphas = _parse_int_list(args.detector_alphas)
        return [
            [(int(coef), int(alpha))]
            for alpha in alphas
            for coef in _parse_coef_list(args.coefs)
        ]
    if args.design_mode == "random-monomial":
        coefs = rng.choice(p.n, size=args.num_designs, replace=False)
        return [[(int(coef), args.alpha)] for coef in coefs]
    if args.design_mode == "random-multiterm":
        alpha_choices = _parse_int_list(args.alpha_choices)
        designs: list[list[tuple[int, int]]] = []
        for _ in range(args.num_designs):
            coefs = rng.choice(p.n, size=args.terms, replace=False)
            alphas = rng.choice(
                np.asarray(alpha_choices, dtype=np.int64),
                size=args.terms,
                replace=True,
            )
            designs.append([(int(c), int(a)) for c, a in zip(coefs, alphas)])
        return designs
    raise ValueError(f"unknown design mode {args.design_mode}")


def _cmd_payload(args: argparse.Namespace) -> bytes:
    if args.cmd == "W":
        return bytes([args.component])
    return b""


def _apply_c2(args: argparse.Namespace, p, ct: Ciphertext) -> Ciphertext:
    if args.c2_mode == "zero":
        return ct
    if args.c2_mode == "constant":
        c2 = np.full(p.n, int(args.c2_alpha), dtype=np.int64)
        return Ciphertext(c1=ct.c1, c2=c2, params=p)
    raise ValueError(f"unknown c2 mode {args.c2_mode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-keys", type=int, default=4)
    p.add_argument("-n", "--num-traces", type=int, default=50)
    p.add_argument("-s", "--samples", type=int, default=24400)
    p.add_argument("-g", "--gain-db", type=float, default=25.0)
    p.add_argument("--adc-offset", type=int, default=0)
    p.add_argument("--cmd", choices=("Z", "V", "W", "R"), default="Z")
    p.add_argument("--component", type=int, default=0, choices=(0, 1))
    p.add_argument("--alpha", type=int, default=4)
    p.add_argument("--c2-mode", choices=("zero", "constant"), default="zero")
    p.add_argument("--c2-alpha", type=int, default=0)
    p.add_argument(
        "--design-mode",
        choices=("monomial", "detector-grid", "random-monomial", "random-multiterm"),
        default="monomial",
    )
    p.add_argument("--num-designs", type=int, default=8)
    p.add_argument("--terms", type=int, default=4)
    p.add_argument("--design-seed", type=int, default=0xD3516E)
    p.add_argument(
        "--design-file",
        type=Path,
        default=None,
        help="JSON design list or scorer output containing selected terms.",
    )
    p.add_argument(
        "--alpha-choices",
        default="32,64,96,128,160,192,224",
        help="Comma-separated alpha choices for random-multiterm mode.",
    )
    p.add_argument(
        "--detector-alphas",
        default="64,128,192",
        help="Comma-separated alpha values for detector-grid mode.",
    )
    p.add_argument(
        "--coefs",
        default="0,32,64,96,128,160,192,224",
        help="Comma-separated X^j coefficients to probe.",
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
        "--out-dir",
        type=Path,
        default=_REPO / "traces",
    )
    p.add_argument("--tag", default="s2_z_matrix")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    p = _params.get("smaug1")
    p.assert_consistent()
    if not (0 <= args.alpha < p.p):
        raise SystemExit(f"alpha {args.alpha} outside [0, {p.p})")
    if not (0 <= args.c2_alpha < p.p2):
        raise SystemExit(f"c2-alpha {args.c2_alpha} outside [0, {p.p2})")
    designs = _make_designs(args, p)
    for terms in designs:
        if not terms:
            raise SystemExit("empty design")
        for coef, alpha in terms:
            if not (0 <= coef < p.n):
                raise SystemExit(f"coef {coef} outside [0, {p.n})")
            if not (0 <= alpha < p.p):
                raise SystemExit(f"alpha {alpha} outside [0, {p.p})")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fw_sha = _file_sha256(args.firmware_hex)

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
        for key_i in range(args.num_keys):
            target.simpleserial_write("F", b"")
            pk_fp16 = bytes(target.simpleserial_read("r", 16, timeout=5000))
            if len(pk_fp16) != 16:
                raise RuntimeError("F ack failed")
            sk_pke = dump_sk_pke(target, p.pke_secret_key_bytes // 32)
            sk_unpacked = _codec.unpack_sx(sk_pke).astype(np.int64).reshape(p.module_rank, p.n)
            print(f"[KEY {key_i:02d}] pk={pk_fp16.hex()[:8]} sk[0:4]={sk_pke[:4].hex()}")

            all_traces = []
            all_acks = []
            design_meta = []
            min_ok = args.num_traces

            for design_i, terms in enumerate(designs):
                ct = _chosen.build_multi_term_c1(
                    p,
                    {(args.component, coef): alpha for coef, alpha in terms},
                )
                ct = _apply_c2(args, p, ct)
                ct_bytes = ct.to_bytes()
                bundle = setup_session(
                    target,
                    ct_bytes,
                    label=f"matrix_{args.design_mode}_d{design_i}",
                    fresh_key=False,
                    pk_fp16=pk_fp16,
                )
                if bundle.ct_fp16_host != bundle.ct_fp16_board:
                    raise RuntimeError(
                        f"ct integrity fail design={design_i}: "
                        f"host={bundle.ct_fp16_host.hex()} board={bundle.ct_fp16_board.hex()}"
                    )

                payload = _cmd_payload(args)
                target.simpleserial_write(args.cmd, payload)
                first_resp = bytes(target.simpleserial_read("r", 32, timeout=5000))
                if len(first_resp) != 32:
                    raise RuntimeError(
                        f"{args.cmd} first response failed key={key_i} "
                        f"design={design_i}: len={len(first_resp)}"
                    )
                if args.cmd in ("Z", "V", "R"):
                    pred_mu = _pack_mu_bits(
                        _chosen.predict_mu_prime(p, ct.c1, sk_unpacked, ct.c2)
                    )
                    diff = sum(bin(a ^ b).count("1") for a, b in zip(pred_mu, first_resp))
                    if diff != 0:
                        raise RuntimeError(
                            f"mu' round-trip fail key={key_i} design={design_i}: "
                            f"{diff} bits"
                        )

                traces = np.empty((args.num_traces, args.samples), dtype=np.float32)
                acks = np.zeros((args.num_traces, 32), dtype=np.uint8)
                n_ok = 0
                for trace_i in range(args.num_traces):
                    scope.arm()
                    target.simpleserial_write(args.cmd, payload)
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
                    raise RuntimeError(f"no traces captured for key={key_i} design={design_i}")
                if not np.all(acks[:n_ok] == np.frombuffer(first_resp, dtype=np.uint8)):
                    raise RuntimeError(f"ack nondeterminism key={key_i} design={design_i}")

                min_ok = min(min_ok, n_ok)
                all_traces.append(traces[:n_ok])
                all_acks.append(acks[:n_ok])
                design_meta.append({
                    "component": args.component,
                    "coef_idx": terms[0][0] if len(terms) == 1 else None,
                    "alpha": terms[0][1] if len(terms) == 1 else None,
                    "terms": terms,
                    "c2_mode": args.c2_mode,
                    "c2_alpha": int(args.c2_alpha) if args.c2_mode == "constant" else 0,
                    "ct_fp16_host": bundle.ct_fp16_host.hex(),
                    "ct_fp16_board": bundle.ct_fp16_board.hex(),
                    "ct_sha256": hashlib.sha256(ct_bytes).hexdigest(),
                    "first_resp_hex": first_resp.hex(),
                    "first_mu_resp_hex": first_resp.hex() if args.cmd in ("Z", "V", "R") else None,
                })
                print(
                    f"[KEY {key_i:02d}] design {design_i+1}/{len(designs)} "
                    f"terms={terms} ok={n_ok}/{args.num_traces}"
                )

            traces_out = np.stack([x[:min_ok] for x in all_traces], axis=0)
            acks_out = np.stack([x[:min_ok] for x in all_acks], axis=0)
            meta = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "scope_sn": sn,
                "samples": args.samples,
                "gain_db": args.gain_db,
                "adc_offset": args.adc_offset,
                "target": "smaug1",
                "cmd": args.cmd,
                "capture_kind": "matrix",
                "level": "smaug1",
                "module_rank": p.module_rank,
                "lwe_n": p.n,
                "hs": p.hs,
                "ss_ver": "SS_VER_1_1",
                "baud": args.baud,
                "n_attempted_per_design": args.num_traces,
                "n_ok_per_design": int(min_ok),
                "firmware_hex": str(args.firmware_hex),
                "firmware_hex_sha256": fw_sha,
                "pk_fp16": pk_fp16.hex(),
                "sk_pke_hex": sk_pke.hex(),
                "component": args.component,
                "alpha": args.alpha,
                "c2_mode": args.c2_mode,
                "c2_alpha": int(args.c2_alpha) if args.c2_mode == "constant" else 0,
                "design_mode": args.design_mode,
                "design_seed": args.design_seed,
                "terms_per_design": args.terms if args.design_mode == "random-multiterm" else 1,
                "coefs": [terms[0][0] if len(terms) == 1 else None for terms in designs],
                "design_terms": designs,
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
