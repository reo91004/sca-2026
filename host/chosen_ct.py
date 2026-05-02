"""Chosen-ciphertext Serial 세션 (SMAUG-T 'F'/'I'/'L'/'D' 명령).

펌웨어 상주 sk + ct_inj 위에 chosen ct 를 host 가 주입하고, 무결성 지문을
host 측 시뮬레이터의 sha3_256 기대값과 비교해 *전송 sanity* 까지 보장한
뒤, 이후 capture loop 가 'D' 만 반복 호출해도 동일 ct 위에서 트레이스를
모을 수 있게 만든다.

흐름:

    SimpleSerial 연결
       └─► 'F' (keygen)         → pk_fp16
       └─► 'I' × 21 (chunk 0..20)
       └─► 'L' (digest)         → ct_fp16 (host 의 sha3_256(ct) 와 비교)
       └─► (이 시점 이후 'D' 호출은 모두 같은 sk + 같은 ct_inj)
       └─► capture.py -c D ...

CLI:
    python3 -m host.chosen_ct setup --ct mu_zero --out-fingerprint <path>
    python3 -m host.chosen_ct setup --ct monomial --component 0 --coef 0 \\
        --alpha 4 --out-fingerprint <path>

세션은 함수형으로도 사용 가능 (기존 capture.py 와 동일 scope 객체 공유):

    from host.chosen_ct import setup_session, Bundle
    bundle = setup_session(scope, target, ciphertext_bytes)
    # 이후 같은 scope/target 으로 capture loop 실행
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 이 모듈은 chipwhisperer 가 *없어도* import 만은 동작해야 한다 (단위테스트
# 환경 호환). 실제 setup_session 호출 시점에서만 cw 모듈을 가져온다.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import params as _params  # noqa: E402


CHUNK_BYTES = 32
INJECT_PAYLOAD_LEN = 1 + CHUNK_BYTES  # idx + data; firmware 와 일치
MU_BYTES = 32                          # smaug1 DELTA_BYTES; firmware 와 일치
SEED_BYTES = 32                        # 펌웨어 'M' 은 zero-fixed seed 만 사용
MENC_PAYLOAD_LEN = MU_BYTES            # 32; SS_VER_1_1 한도 64 미만 (>=64 거부)


@dataclass(frozen=True)
class Bundle:
    """세션 셋업의 모든 결과물 — 캡처 단계에 그대로 넘긴다.

    두 가지 모드를 모두 다룬다:
      (chunk-inject) host 가 ct 를 만들어 'I' 로 주입 → ct_bytes 는 host 가 안다.
      (PKE-labeled)  보드의 indcpa_enc 가 ct 를 만든다 → ct_bytes = None,
                      대신 mu_seed 와 보드측 sha3 를 보관한다.
    """

    pk_fp16: bytes
    ct_bytes: bytes | None       # PKE-labeled 모드는 None (보드만 안다)
    ct_fp16_host: bytes | None   # host 가 미리 계산한 sha3 (chunk-inject 모드만)
    ct_fp16_board: bytes         # 보드측 sha3
    label: str
    # PKE-labeled 모드 전용:
    mu: bytes | None = None      # 32 B μ
    seed: bytes | None = None    # 32 B seed

    @property
    def integrity_ok(self) -> bool:
        """chunk-inject 모드만 host vs board sha3 비교가 의미 있음.
        PKE-labeled 는 *결정성* 만 별도로 verify_pke_determinism 로 검사."""
        if self.ct_fp16_host is None:
            # PKE-labeled — 무결성 자체를 host가 미리 알 수 없으므로 항상 True 반환.
            return True
        return self.ct_fp16_host == self.ct_fp16_board


def reset_target(scope: Any, *, settle_s: float = 0.5) -> None:
    """STM32 를 nRST 로 강제 리셋하고 부팅 대기.

    `scope.default_setup()` 직후 바로 SimpleSerial write 를 시도하면
    보드가 아직 ready 상태가 아니라 응답이 오지 않는다 (실측 확인).
    nRST 라인을 low → high_z 토글 후 settle_s 만큼 기다리면 'p' / 'F'
    가 즉시 응답한다.
    """
    import time
    scope.io.nrst = "low"
    time.sleep(0.05)
    scope.io.nrst = "high_z"
    time.sleep(settle_s)


def _ss_write(target: Any, cmd: str, payload: bytes = b"") -> None:
    """SimpleSerial v1.1 명령 송신 (chipwhisperer 6.0)."""
    target.simpleserial_write(cmd, bytes(payload))


def _ss_read_ack(target: Any, n_bytes: int, timeout_ms: int = 2000) -> bytes:
    ack = target.simpleserial_read("r", n_bytes, timeout=timeout_ms)
    if ack is None or len(ack) != n_bytes:
        got = -1 if ack is None else len(ack)
        raise RuntimeError(
            f"SimpleSerial ack 길이 {got} != 기대 {n_bytes}"
        )
    return bytes(ack)


def setup_session_pke_labeled(
    target: Any,
    mu: bytes,
    seed: bytes,
    *,
    label: str = "pke_labeled",
    verify_determinism: bool = True,
) -> Bundle:
    """PKE-labeled 모드: 'F' → 'M(μ, seed)' 로 보드의 indcpa_enc 가 ct 를 만든다.

    SMAUG-T spec v4.0 Theorem 2 (PKE.PKE 1-δ correctness): 다음 'D' 의
    decryption 결과 µ′ = μ (LWR error 무시 가능). 따라서 host 는 보드가
    무엇을 decapsulate 할지 *암호학적으로* 안다. fixed-point 함정 없이
    임의 μ 패턴 (all-0/1, alternating, random-known) 을 강제할 수 있다.

    무결성 검증: verify_determinism=True 면 'M(μ, seed)' 를 두 번 호출해
    같은 sha3_256(ct_inj)[:16] 가 나오는지 확인한다 (보드의 indcpa_enc 가
    decisive — randombytes 안 부른다는 사실 검증).
    """
    if len(mu) != MU_BYTES:
        raise ValueError(f"mu len {len(mu)} != {MU_BYTES}")
    if len(seed) != SEED_BYTES:
        raise ValueError(f"seed len {len(seed)} != {SEED_BYTES}")
    # 현 펌웨어 ('M' payload = 32B μ-only) 는 seed 인자를 무시 — host
    # API 의 seed 는 *기록용* 으로만 보존되고 보드에는 zero seed 가 들어간다.
    # SS_VER_1_1 의 simpleserial_addcmd 가 len >= 64 거부 (max 63B) 이라
    # μ + seed = 64B 페이로드를 만들 수 없어서 내린 결정. 추후 별도 'N'
    # (seed inject 32B) 명령 추가하면 host seed 도 통제 가능.

    target.flush()

    _ss_write(target, "F")
    pk_fp16 = _ss_read_ack(target, 16)

    payload = bytes(mu)            # 32B μ-only
    _ss_write(target, "M", payload)
    fp_a = _ss_read_ack(target, 16)

    if verify_determinism:
        _ss_write(target, "M", payload)
        fp_b = _ss_read_ack(target, 16)
        if fp_a != fp_b:
            raise RuntimeError(
                f"PKE-labeled 결정성 깨짐: M(μ) 가 두 번 호출에 다른 ct 출력 — "
                f"fp_a={fp_a.hex()} fp_b={fp_b.hex()} "
                f"(indcpa_enc 가 zero-seed 위에서 deterministic 해야 함)"
            )

    return Bundle(
        pk_fp16=pk_fp16,
        ct_bytes=None,
        ct_fp16_host=None,
        ct_fp16_board=fp_a,
        label=label,
        mu=bytes(mu),
        seed=bytes(seed),
    )


def setup_session(
    target: Any,
    ct_bytes: bytes,
    *,
    chunk_size: int = CHUNK_BYTES,
    inject_timeout_ms: int = 2000,
    label: str = "chosen_ct",
) -> Bundle:
    """보드에 'F' → 'I' × N → 'L' 시퀀스를 보내고, 무결성 검증 + Bundle 반환.

    target  : chipwhisperer 의 SimpleSerial 객체 (이미 연결됨).
    ct_bytes: host 가 주입할 ct. len == params.ciphertext_bytes 여야 함.
    """
    if len(ct_bytes) % chunk_size != 0:
        raise ValueError(
            f"ct length {len(ct_bytes)} not divisible by chunk_size={chunk_size}"
        )

    target.flush()

    # 1) 'F' fresh keygen
    _ss_write(target, "F")
    pk_fp16 = _ss_read_ack(target, 16)

    # 2) 'I' chunk inject — 'I' 페이로드는 [idx 1B][data CHUNK B]
    chunks = _chosen.chunkify(ct_bytes, chunk_size=chunk_size)
    for idx, data in chunks:
        if len(data) != chunk_size:
            raise AssertionError(f"chunk[{idx}] len {len(data)} != {chunk_size}")
        payload = bytes([idx]) + data
        if len(payload) != INJECT_PAYLOAD_LEN:
            raise AssertionError(
                f"inject payload {len(payload)} != {INJECT_PAYLOAD_LEN}"
            )
        _ss_write(target, "I", payload)
        status = _ss_read_ack(target, 1, timeout_ms=inject_timeout_ms)
        if status[0] != 0:
            raise RuntimeError(
                f"'I' chunk[{idx}] status={status[0]} (1=idx OOR, 2=len OOR)"
            )

    # 3) 'L' 무결성 지문 비교
    _ss_write(target, "L")
    ct_fp16_board = _ss_read_ack(target, 16)
    ct_fp16_host = hashlib.sha3_256(ct_bytes).digest()[:16]

    bundle = Bundle(
        pk_fp16=pk_fp16,
        ct_bytes=ct_bytes,
        ct_fp16_host=ct_fp16_host,
        ct_fp16_board=ct_fp16_board,
        label=label,
    )
    if not bundle.integrity_ok:
        raise RuntimeError(
            f"ct 무결성 불일치 (label={label}): "
            f"host={ct_fp16_host.hex()} board={ct_fp16_board.hex()} — "
            f"chunk inject 중 손실 추정"
        )
    return bundle


# ---------------------------------------------------------------------------
# CLI: scope 한 번 잡고 setup 만 돌리는 스탠드얼론 모드.
# ---------------------------------------------------------------------------

_MU_PATTERNS = {
    "zero":  bytes(MU_BYTES),                                # 0x00 × 32
    "one":   bytes([0xFF] * MU_BYTES),                       # 0xFF × 32
    "55":    bytes([0x55] * MU_BYTES),                       # alternating bits
    "AA":    bytes([0xAA] * MU_BYTES),                       # alternating bits, 반대 phase
}


def _decode_hex(s: str, expect: int, name: str) -> bytes:
    try:
        out = bytes.fromhex(s)
    except ValueError as e:
        raise SystemExit(f"[FAIL] --{name} hex 파싱 실패: {e}")
    if len(out) != expect:
        raise SystemExit(f"[FAIL] --{name} 길이 {len(out)} != {expect}")
    return out


def _build_ct(args: argparse.Namespace) -> tuple[bytes, str]:
    p = _params.get(args.level)
    if args.ct == "mu_zero":
        ct = _chosen.build_mu_constant(p, mu_bit=0)
        return ct.to_bytes(), "mu_zero"
    if args.ct == "monomial":
        if args.component is None or args.coef is None or args.alpha is None:
            raise SystemExit(
                "[FAIL] monomial 모드는 --component, --coef, --alpha 가 필요"
            )
        ct = _chosen.build_monomial_c1(
            p,
            component=args.component,
            coef_idx=args.coef,
            alpha=args.alpha,
        )
        label = f"mono_k{args.component}_j{args.coef}_a{args.alpha}"
        return ct.to_bytes(), label
    raise SystemExit(f"[FAIL] unknown --ct={args.ct}")


def _cli_setup(args: argparse.Namespace) -> int:
    # chipwhisperer 는 여기서야 import — 단위테스트는 영향 없음.
    import chipwhisperer as cw  # noqa: F401

    from host.cw_serial import pick_serial

    sn = pick_serial(args.serial)
    scope = cw.scope(sn=sn)
    scope.default_setup()
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = args.baud
    try:
        if args.ct == "labeled":
            mu = _MU_PATTERNS.get(args.mu_pattern) if args.mu_pattern else None
            if mu is None:
                if not args.mu:
                    raise SystemExit(
                        "[FAIL] --ct labeled 는 --mu-pattern {zero,one,55,AA} "
                        "또는 --mu HEX (64자) 필요"
                    )
                mu = _decode_hex(args.mu, MU_BYTES, "mu")
            seed = (
                _decode_hex(args.seed, SEED_BYTES, "seed")
                if args.seed
                else bytes(SEED_BYTES)
            )
            label = args.label or f"labeled_{args.mu_pattern or 'custom'}"
            print(f"[INFO] CW1173 sn={sn}, label={label}, mu={mu.hex()[:16]}…, "
                  f"seed={seed.hex()[:16]}…")
            bundle = setup_session_pke_labeled(target, mu, seed, label=label)
            print(f"[OK] pk_fp16     = {bundle.pk_fp16.hex()}")
            print(f"[OK] ct_fp16_brd = {bundle.ct_fp16_board.hex()}")
            print(f"[OK] determinism verified (M called twice, same sha3)")
            if args.out_fingerprint:
                Path(args.out_fingerprint).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out_fingerprint).write_text(
                    f"label {label}\n"
                    f"mode pke_labeled\n"
                    f"pk_fp16 {bundle.pk_fp16.hex()}\n"
                    f"ct_fp16 {bundle.ct_fp16_board.hex()}\n"
                    f"mu {mu.hex()}\n"
                    f"seed {seed.hex()}\n"
                )
                print(f"[OK] fingerprint -> {args.out_fingerprint}")
        else:
            ct_bytes, label = _build_ct(args)
            print(f"[INFO] CW1173 sn={sn}, label={label}, ct_len={len(ct_bytes)}")
            bundle = setup_session(target, ct_bytes, label=label)
            print(f"[OK] pk_fp16     = {bundle.pk_fp16.hex()}")
            print(f"[OK] ct_fp16_brd = {bundle.ct_fp16_board.hex()}")
            print(f"[OK] ct_fp16_hst = {bundle.ct_fp16_host.hex()}")
            if args.out_fingerprint:
                Path(args.out_fingerprint).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out_fingerprint).write_text(
                    f"label {label}\n"
                    f"mode chunk_inject\n"
                    f"pk_fp16 {bundle.pk_fp16.hex()}\n"
                    f"ct_fp16 {bundle.ct_fp16_board.hex()}\n"
                )
                print(f"[OK] fingerprint -> {args.out_fingerprint}")
            if args.dump_ct:
                Path(args.dump_ct).parent.mkdir(parents=True, exist_ok=True)
                Path(args.dump_ct).write_bytes(ct_bytes)
                print(f"[OK] ct bytes    -> {args.dump_ct}")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="host.chosen_ct", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="action", required=True)

    s = sub.add_parser("setup", help="F/I/L 또는 F/M 시퀀스로 보드에 sk + ct_inj 영속")
    s.add_argument("--level", default="smaug1", help="smaug{1,3,5} (기본 smaug1)")
    s.add_argument(
        "--ct", choices=("mu_zero", "monomial", "labeled"), default="mu_zero",
        help=("mu_zero / monomial: chunk-inject 모드 (host 가 ct 를 직접 만들고 'I' 로 주입). "
              "labeled: PKE-labeled 모드 (보드의 indcpa_enc 가 μ/seed 로 ct 를 만듦, "
              "spec Theorem 2 로 µ′=μ 보장)"),
    )
    # chunk-inject 모드 인자
    s.add_argument("--component", type=int, default=None,
                   help="[monomial] c1 component k (0..module_rank-1)")
    s.add_argument("--coef", type=int, default=None,
                   help="[monomial] 단항 X^j 의 j (0..n-1)")
    s.add_argument("--alpha", type=int, default=None,
                   help="[monomial] α 값. fixed-point set 안이어야 함")
    # PKE-labeled 모드 인자
    s.add_argument("--mu-pattern", choices=tuple(_MU_PATTERNS.keys()), default=None,
                   help="[labeled] preset μ 패턴 (zero/one/55/AA)")
    s.add_argument("--mu", default=None,
                   help="[labeled] μ 32-byte hex (64 자). --mu-pattern 보다 우선")
    s.add_argument("--seed", default=None,
                   help="[labeled] seed 32-byte hex (기본 0x00 × 32)")
    s.add_argument("--label", default=None,
                   help="meta/fingerprint 에 적을 사람 읽기용 라벨")
    # 공통
    s.add_argument("--serial", default=None,
                   help="CW1173 시리얼 (기본 자동 선택)")
    s.add_argument("--baud", type=int, default=38400)
    s.add_argument("--out-fingerprint", type=Path, default=None,
                   help="pk/ct 지문을 텍스트로 남길 경로")
    s.add_argument("--dump-ct", type=Path, default=None,
                   help="host 가 만든 ct bytes 를 파일로 dump (chunk-inject 모드만)")
    s.set_defaults(fn=_cli_setup)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
