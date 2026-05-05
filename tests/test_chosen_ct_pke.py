"""host/chosen_ct.py PKE-labeled 모드의 mock 검증.

실제 chipwhisperer 객체 없이 *프로토콜 sequence* (F → M → 검증성 M)
가 정확한 페이로드/cmd 로 진행되는지 확인.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import hashlib

from host import chosen_ct  # noqa: E402


class _MockTarget:
    """SimpleSerial v1.1 동작을 흉내내는 단순 mock."""

    def __init__(self, *, fixed_pk_fp16: bytes, fixed_ct_fp16: bytes,
                 deterministic: bool = True) -> None:
        self._fixed_pk_fp16 = fixed_pk_fp16
        self._fixed_ct_fp16 = fixed_ct_fp16
        self._deterministic = deterministic
        self.write_log: list[tuple[str, bytes]] = []
        self._pending_ack: bytes | None = None
        self._call_no = 0

    def flush(self) -> None:
        pass

    def simpleserial_write(self, cmd: str, data: bytes) -> None:
        self.write_log.append((cmd, bytes(data)))
        if cmd == "F":
            self._pending_ack = self._fixed_pk_fp16
        elif cmd == "I":
            self._pending_ack = b"\x00"
        elif cmd == "L":
            self._pending_ack = self._fixed_ct_fp16
        elif cmd == "M":
            self._call_no += 1
            if self._deterministic:
                self._pending_ack = self._fixed_ct_fp16
            else:
                # 호출마다 다른 응답 → 결정성 검증 fail 시뮬레이션
                self._pending_ack = bytes([self._call_no & 0xFF] * 16)
        else:
            raise AssertionError(f"unexpected cmd {cmd!r}")

    def simpleserial_read(self, prefix: str, n: int, timeout: int = 0):
        assert prefix == "r"
        ack = self._pending_ack
        self._pending_ack = None
        return ack


def test_pke_labeled_normal_flow() -> None:
    pk = bytes.fromhex("aa" * 16)
    ct = bytes.fromhex("bb" * 16)
    t = _MockTarget(fixed_pk_fp16=pk, fixed_ct_fp16=ct, deterministic=True)
    mu = bytes(32)
    seed = bytes(32)
    bundle = chosen_ct.setup_session_pke_labeled(t, mu, seed, label="zero_zero")
    # 보드 응답이 그대로 반영
    assert bundle.pk_fp16 == pk
    assert bundle.ct_fp16_board == ct
    assert bundle.ct_bytes is None  # PKE-labeled 는 host 가 ct 를 모름
    assert bundle.mu == mu and bundle.seed == seed
    assert bundle.integrity_ok is True

    # 호출 시퀀스: F, M, M (결정성 verify 한 번 더)
    cmds = [c for c, _ in t.write_log]
    assert cmds == ["F", "M", "M"], f"sequence = {cmds}"
    # 현재 펌웨어 'M' payload = μ 32B only (seed 는 펌웨어 zero-fixed).
    # SS_VER_1_1 한도 < 64 라 64B 페이로드는 등록 거부됨.
    payloads = [d for c, d in t.write_log if c == "M"]
    assert all(p == mu for p in payloads)
    assert all(len(p) == 32 for p in payloads)


def test_pke_labeled_skip_determinism_check() -> None:
    pk = bytes(16); ct = bytes(16)
    t = _MockTarget(fixed_pk_fp16=pk, fixed_ct_fp16=ct, deterministic=False)
    bundle = chosen_ct.setup_session_pke_labeled(
        t, bytes(32), bytes(32), verify_determinism=False
    )
    # M 한 번만 — fp_a 그대로 반영
    cmds = [c for c, _ in t.write_log]
    assert cmds == ["F", "M"]
    assert bundle.ct_fp16_board == bytes([1] * 16)  # mock 의 첫 호출 응답


def test_pke_labeled_raises_on_nondeterminism() -> None:
    pk = bytes(16); ct = bytes(16)
    t = _MockTarget(fixed_pk_fp16=pk, fixed_ct_fp16=ct, deterministic=False)
    raised = False
    try:
        chosen_ct.setup_session_pke_labeled(
            t, bytes(32), bytes(32), verify_determinism=True
        )
    except RuntimeError as e:
        raised = "결정성" in str(e) or "determini" in str(e).lower()
    assert raised, "두 번 호출 응답이 다르면 RuntimeError 가 떠야 함"


def test_pke_labeled_rejects_bad_lengths() -> None:
    t = _MockTarget(fixed_pk_fp16=bytes(16), fixed_ct_fp16=bytes(16))
    raised = 0
    try:
        chosen_ct.setup_session_pke_labeled(t, bytes(31), bytes(32))
    except ValueError:
        raised += 1
    try:
        chosen_ct.setup_session_pke_labeled(t, bytes(32), bytes(33))
    except ValueError:
        raised += 1
    assert raised == 2


def test_chunk_inject_can_reuse_resident_key() -> None:
    pk = bytes.fromhex("aa" * 16)
    ct_bytes = bytes(range(64))
    ct_fp16 = hashlib.sha3_256(ct_bytes).digest()[:16]
    t = _MockTarget(fixed_pk_fp16=pk, fixed_ct_fp16=ct_fp16)

    bundle = chosen_ct.setup_session(
        t,
        ct_bytes,
        label="resident",
        fresh_key=False,
        pk_fp16=pk,
    )

    assert bundle.pk_fp16 == pk
    assert bundle.ct_fp16_board == ct_fp16
    assert bundle.ct_fp16_host == ct_fp16
    assert bundle.integrity_ok is True
    cmds = [c for c, _ in t.write_log]
    assert cmds == ["I", "I", "L"], f"sequence = {cmds}"
    assert t.write_log[0][1] == bytes([0]) + ct_bytes[:32]
    assert t.write_log[1][1] == bytes([1]) + ct_bytes[32:]


if __name__ == "__main__":
    fns = sorted(n for n in globals() if n.startswith("test_"))
    for n in fns:
        print(f"[RUN] {n}")
        globals()[n]()
        print(f"[OK ] {n}")
    print(f"[OK ] {len(fns)} tests")
