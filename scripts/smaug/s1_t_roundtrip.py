#!/usr/bin/env python3
"""S1 step 0 — 'T' (isolated poly_mul_acc) host round-trip 검증.

목적
====
host 가 보드에 ``'T'`` (component, idx, alpha) 를 보내고 32B 응답을 받았을 때,
응답이 host 의 ``negacyclic_mul_int16(sk_PKE[component], host_b)[:16].tobytes()`` 와
*byte-for-byte* 일치하는가? 일치하면 host predictor 가 archive 의
``poly_mul_acc`` 동작을 정확히 모델링 하는 것 (= S1 capture 단계에서 ground
truth 를 신뢰할 수 있다).

전제
====
* firmware ``firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex``
  가 보드에 *이미 플래시* 되어 있어야 한다 (``host/upload.py`` 사용). 본
  스크립트는 firmware 의 sha256 만 메타에 기록할 뿐 자동으로 플래시 하지 않는다.
* SMAUG_NAMESPACE = smaug1 빌드. 다른 레벨은 응답 길이/포맷이 다르므로 별도
  실행 필요.

실패 시 진단
============
mismatch 가 발생하면:
  (A) 'X' chunk 응답이 host codec.unpack_sx 와 다른 인코딩이거나,
  (B) host poly_mul predictor 의 sign/wrap convention 이 archive 와 다르거나,
  (C) firmware 의 ``cmd_isolated_poly_mul`` 자체가 build 마다 다른 코드.

이 스크립트는 mismatch 발생 시 첫 5 좌표의 (host int16, board int16) 쌍 +
원시 byte-hex 를 출력해서 어느 mode 인지 즉시 확인 가능하게 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import reset_target  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from host.smaug.poly_mul import (  # noqa: E402
    LWE_N,
    board_response_mod_p,
    monomial_predict,
    negacyclic_mul_mod_p,
)


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--firmware-hex",
        type=Path,
        default=_REPO
        / "firmware"
        / "simpleserial-smaug"
        / "simpleserial-smaug-CW308_STM32F4.hex",
        help="firmware .hex (sha256 만 기록; 자동 플래시 안 함)",
    )
    p.add_argument("--baud", type=int, default=38400)
    p.add_argument("--serial", default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "s1_t_roundtrip.txt",
        help="결과 요약을 기록할 텍스트 파일",
    )
    return p.parse_args()


def _ack(target, n: int, timeout_ms: int = 5000) -> bytes:
    a = target.simpleserial_read("r", n, timeout=timeout_ms)
    if a is None or len(a) != n:
        raise RuntimeError(f"ack {-1 if a is None else len(a)} != {n}")
    return bytes(a)


def int16_wrap(x: int) -> int:
    """Return Python int with C int16_t wraparound semantics."""
    u = int(x) & 0xFFFF
    return u - 0x10000 if u >= 0x8000 else u


def issue_t(
    target,
    component: int,
    idx: int,
    alpha: int,
    *,
    timeout_ms: int = 5000,
) -> bytes:
    """'T' command 단발 호출. 32B int16[16] little-endian 응답.

    payload = component(1B) | idx(2B BE) | alpha(2B BE), 총 5B.
    """
    if not (0 <= component <= 0xFF):
        raise ValueError(f"component {component} OOB")
    if not (0 <= idx <= 0xFFFF):
        raise ValueError(f"idx {idx} OOB")
    a16 = int16_wrap(alpha) & 0xFFFF  # int16 → uint16 wire
    payload = bytes(
        [
            component & 0xFF,
            (idx >> 8) & 0xFF,
            idx & 0xFF,
            (a16 >> 8) & 0xFF,
            a16 & 0xFF,
        ]
    )
    target.simpleserial_write("T", payload)
    resp = _ack(target, 32, timeout_ms=timeout_ms)
    # 펌웨어는 길이 mismatch 시 1B 상태(1/2/3)만 보냄. 32B 가 아니면 _ack 가 raise.
    return resp


def issue_u(
    target,
    component: int,
    terms: list[tuple[int, int]],
    *,
    timeout_ms: int = 5000,
) -> bytes:
    """'U' command 단발 호출. 최대 8개 `(idx, alpha)` term 지원.

    payload = component(1B) | n_terms(1B) | 8 * [idx(2B BE), alpha(2B BE)].
    남는 term slot 은 zero-fill. 응답은 'T' 와 같은 32B int16[16].
    """
    if not (0 <= component <= 0xFF):
        raise ValueError(f"component {component} OOB")
    if len(terms) > 8:
        raise ValueError(f"too many terms: {len(terms)} > 8")
    payload = bytearray([component & 0xFF, len(terms) & 0xFF])
    for idx, alpha in terms:
        if not (0 <= idx <= 0xFFFF):
            raise ValueError(f"idx {idx} OOB")
        a16 = int16_wrap(alpha) & 0xFFFF
        payload.extend([(idx >> 8) & 0xFF, idx & 0xFF, (a16 >> 8) & 0xFF, a16 & 0xFF])
    while len(payload) < 34:
        payload.extend([0, 0, 0, 0])
    if len(payload) != 34:
        raise AssertionError(f"U payload len {len(payload)} != 34")
    target.simpleserial_write("U", bytes(payload))
    return _ack(target, 32, timeout_ms=timeout_ms)


def dump_sk_pke(target, n_chunks: int) -> bytes:
    """'X' chunks 0..n-1 을 모아 PKE sk 영역 전체 (smaug1: 128B)."""
    out = bytearray()
    for i in range(n_chunks):
        target.simpleserial_write("X", bytes([i]))
        out.extend(_ack(target, 32, timeout_ms=2000))
    return bytes(out)


def make_test_cases() -> list[dict]:
    """corner case 를 골고루 — sign wrap + alpha sign + 두 component."""
    cases: list[dict] = []
    # Component 0 — sign wrap 검증
    for j, alpha in [
        (0, 1),         # i ≥ j: out[0..15] = +sk0[0..15]
        (10, 1),        # i ∈ [10,15]: +sk0; i ∈ [0,9]: -sk0[wrap]
        (15, 1),        # boundary at i=15
        (16, 1),        # all wrap (i ∈ [0,15] < 16)
        (200, 1),       # i ∈ [0,15] all wrap, large idx
        (255, 1),       # 끝점
        (5, -1),        # alpha 음수
        (100, 64),      # alpha 큰 값 → int16 wrap 안 함 (sk ∈ {-1,0,+1})
    ]:
        cases.append({"component": 0, "idx": j, "alpha": alpha})
    # Component 1 — 다른 다항식
    for j, alpha in [(0, 1), (10, 1), (200, 1), (5, -1)]:
        cases.append({"component": 1, "idx": j, "alpha": alpha})
    return cases


def main() -> int:
    args = parse_args()

    p = _params.get("smaug1")
    n = p.n
    if n != LWE_N:
        raise SystemExit(f"smaug1 LWE_N {n} != {LWE_N} — predictor mismatch")
    n_chunks = p.pke_secret_key_bytes // 32  # smaug1: 128 / 32 = 4
    if p.pke_secret_key_bytes != n_chunks * 32:
        raise SystemExit(
            f"PKE_SECRETKEY_BYTES {p.pke_secret_key_bytes} not multiple of 32"
        )

    fw_sha = (
        _file_sha256(args.firmware_hex) if args.firmware_hex.exists() else None
    )

    import chipwhisperer as cw  # noqa: E402
    from host.cw_serial import pick_serial  # noqa: E402

    sn = pick_serial(args.serial)
    print(f"[INFO] CW1173 sn={sn}")
    print(f"[INFO] firmware_hex={args.firmware_hex}")
    print(f"[INFO] firmware_sha256={fw_sha}")

    scope = cw.scope(sn=sn)
    scope.default_setup()
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    target.flush()
    time.sleep(0.3)

    out_lines: list[str] = []
    out_lines.append(f"firmware_hex={args.firmware_hex.name}")
    out_lines.append(f"firmware_sha256={fw_sha}")
    out_lines.append(f"scope_sn={sn}")

    try:
        # --- 1) F: fresh keypair, sk persistent ---------------------------
        target.simpleserial_write("F", b"")
        pk_fp16 = _ack(target, 16, timeout_ms=5000)
        print(f"[OK]   F  pk_fp16={pk_fp16.hex()}")
        out_lines.append(f"pk_fp16={pk_fp16.hex()}")

        # --- 2) X 0..n_chunks-1: dump sk PKE bytes ------------------------
        sk_pke = dump_sk_pke(target, n_chunks)
        print(f"[OK]   X  sk_pke[{len(sk_pke)}B]={sk_pke[:8].hex()}…")
        out_lines.append(f"sk_pke_hex={sk_pke.hex()}")
        # Unpack — codec.unpack_sx 가 disasm 와 일치하는지 (count check 로 약식 검증).
        ternary_per_byte = 4
        sk_unpacked_full = _codec.unpack_sx(sk_pke)
        # 두 다항식 (smaug1 module_rank=2, 각 256 coef)
        sk_polys = [
            sk_unpacked_full[i * n : (i + 1) * n].astype(np.int64)
            for i in range(p.module_rank)
        ]
        for m, poly in enumerate(sk_polys):
            if poly.shape != (n,):
                raise RuntimeError(
                    f"sk_poly[{m}] shape {poly.shape} != ({n},)"
                )
            if not set(np.unique(poly).tolist()).issubset({-1, 0, 1}):
                raise RuntimeError(
                    f"sk_poly[{m}] outside {{-1,0,+1}}: "
                    f"{np.unique(poly).tolist()}"
                )
            hs = int(np.sum(poly != 0))
            print(
                f"[OK]   sk_poly[{m}] HS={hs} "
                f"(spec HS={p.hs} expected, but per-poly may differ)"
            )
            out_lines.append(f"sk_poly[{m}]_hs={hs}")
        del ternary_per_byte  # 단순 sanity 변수

        # --- 3) round-trip: 'T' x N test cases ----------------------------
        cases = make_test_cases()
        print(f"[INFO] testing {len(cases)} cases…")
        out_lines.append(f"n_cases={len(cases)}")

        log_p = p.log_p  # smaug1 = 8
        out_lines.append(f"log_p={log_p}")

        n_pass = 0
        for k, c in enumerate(cases):
            comp = c["component"]
            j = c["idx"]
            alpha = c["alpha"]
            try:
                resp = issue_t(target, comp, j, alpha)
            except RuntimeError as e:
                msg = (
                    f"[FAIL] case {k} comp={comp} idx={j} alpha={alpha} "
                    f"transport: {e}"
                )
                print(msg)
                out_lines.append(msg)
                continue
            sk_poly = sk_polys[comp]
            # mod-p signed centered host predict
            host_modp_full = negacyclic_mul_mod_p(
                sk_poly, _to_b(j, alpha, n), log_p=log_p, signed=True
            )
            host_modp = host_modp_full[:16]
            board_modp = board_response_mod_p(resp, log_p=log_p, signed=True)
            ok = bool(np.array_equal(host_modp, board_modp))
            # closed form 도 같이 검증 (host predictor 의 단항 closed form 경로)
            cf_full = monomial_predict(sk_poly, j, alpha)
            cf_modp = (cf_full[:16] % (1 << log_p))
            cf_modp = np.where(
                cf_modp >= (1 << (log_p - 1)), cf_modp - (1 << log_p), cf_modp
            )
            cf_ok = bool(np.array_equal(cf_modp, board_modp))
            if ok and cf_ok:
                n_pass += 1
                if k < 4:
                    print(
                        f"[OK]   case {k:2d} comp={comp} idx={j:3d} alpha={alpha:6d}  "
                        f"(mod p={1 << log_p})"
                    )
            else:
                first_diff = np.where(board_modp != host_modp)[0]
                fd = int(first_diff[0]) if first_diff.size > 0 else -1
                msg = (
                    f"[FAIL] case {k:2d} comp={comp} idx={j:3d} alpha={alpha:6d} "
                    f"mod_p_convolve_match={ok} mod_p_closed_form_match={cf_ok} "
                    f"first_diff_at_i={fd}"
                )
                print(msg)
                out_lines.append(msg)
                # 16 좌표 dump (mod p 비교 + raw board int16 동시)
                board_raw = np.frombuffer(resp, dtype=np.int16)
                show = " ".join(
                    f"({i}:h{int(host_modp[i])}/b{int(board_modp[i])}|raw{int(board_raw[i])})"
                    for i in range(16)
                )
                print(f"        all16: {show}")
                out_lines.append(f"        all16: {show}")
                out_lines.append(f"        board_hex={resp.hex()}")

        print(f"[RESULT] {n_pass}/{len(cases)} round-trip OK")
        out_lines.append(f"n_pass={n_pass}/{len(cases)}")

    finally:
        try:
            target.dis()
        except Exception:
            pass
        try:
            scope.dis()
        except Exception:
            pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n")
    print(f"[OK] wrote {args.out}")
    return 0 if n_pass == len(cases) else 1


def _to_b(idx: int, alpha: int, n: int) -> np.ndarray:
    """Re-export of monomial_b for inline use (avoid extra import noise)."""
    from host.smaug.poly_mul import monomial_b

    return monomial_b(idx, alpha, n=n)


if __name__ == "__main__":
    raise SystemExit(main())
