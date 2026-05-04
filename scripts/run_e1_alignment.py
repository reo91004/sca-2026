#!/usr/bin/env python3
"""E1 — same-sk re-capture for alignment cause isolation.

한 script run 안에서 두 capture session 을 연속 수행:
  session A: 정상 reset + F (keygen) + capture
  session B: reset/F 스킵, 같은 board RAM 의 sk 그대로 capture

두 session 의 trace 를 cross-correlation 해 alignment 가 OK 인지 random 인지 결정.

가설:
- align OK (norm-corr ≫ 0.1, shift 작음) → 같은 sk 위에서는 trace 정렬됨
   → 기존 9 dataset 의 cross-session 깨짐은 *다른 sk 때문*. 즉 sk-dependent
   timing 또는 keygen 의 RNG 영향. 이 경우 firmware 에 sub-trigger 추가하거나
   *같은 sk 로 모든 keypair-test 캡처* (key swap firmware) 가 필요.
- align RANDOM (norm-corr ≤ 0.1) → 같은 sk 위에서도 align 안 됨
   → indcpa_dec 자체에 hardware-state 의존성 (ART prefetch 등). DTW
   alignment 또는 implementation-derived schedule 만이 답.

산출: traces/E1_sessionA.npz, traces/E1_sessionB.npz, results/E1_alignment.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.chosen_ct import CHUNK_BYTES, reset_target  # noqa: E402
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from scripts.run_attack import (  # noqa: E402
    _ack,
    _file_sha256,
    _git_rev,
    capture_n,
    inject_ct,
)


def _do_session(scope, target, p, *, alpha_pos: int, alpha_neg: int,
                 n_per_ct: int, samples: int, do_keygen: bool) -> dict:
    """한 session = (optional F) + X dump + 4 design × n_per_ct capture.

    do_keygen=False 면 F 와 reset 모두 skip → board RAM sk 그대로.
    """
    if do_keygen:
        target.simpleserial_write("F", b"")
        pk_fp16 = _ack(target, 16)
    else:
        pk_fp16 = b"\x00" * 16

    n_sk_chunks = p.pke_secret_key_bytes // 32
    sk_pke = bytearray()
    for idx in range(n_sk_chunks):
        target.simpleserial_write("X", bytes([idx]))
        sk_pke.extend(_ack(target, 32, timeout_ms=2000))

    import hashlib
    sk_unpacked = _codec.unpack_sx(bytes(sk_pke)).reshape(
        p.module_rank, p.n).astype(np.int64)

    sweep = []
    for component in range(p.module_rank):
        for alpha in (alpha_pos, alpha_neg):
            sweep.append((component, alpha))

    all_traces: list[np.ndarray] = []
    all_mus: list[np.ndarray] = []
    all_labels: list = []
    ct_fps: list[str] = []
    started = time.time()
    for n_done, (comp, alpha) in enumerate(sweep, 1):
        ct = _chosen.build_monomial_c1(p, component=comp, coef_idx=0, alpha=alpha)
        ct_bytes = ct.to_bytes()
        host_fp = hashlib.sha3_256(ct_bytes).hexdigest()[:32]
        fp = inject_ct(target, ct_bytes)
        board_fp = fp.hex()
        ct_fps.append(board_fp)
        if board_fp != host_fp:
            raise RuntimeError(
                f"ct fingerprint mismatch host={host_fp} board={board_fp} — wrong firmware?"
            )
        t, m = capture_n(scope, target, n_per_ct, samples)
        # round-trip µ′ check (first trace)
        mu_pred = _chosen.predict_mu_prime(p, ct.c1, sk_unpacked)
        mu_actual = np.zeros(256, dtype=int)
        for k in range(256):
            mu_actual[k] = (int(m[0, k // 8]) >> (k % 8)) & 1
        if not (mu_pred == mu_actual).all():
            n_match = int((mu_pred == mu_actual).sum())
            raise RuntimeError(
                f"µ′ round-trip mismatch ({n_match}/256) at comp={comp} α={alpha}"
            )
        all_traces.append(t)
        all_mus.append(m)
        all_labels.extend([(comp, alpha)] * n_per_ct)
        elapsed = time.time() - started
        print(f"    [{n_done}/{len(sweep)}] comp={comp} α={alpha:3d} "
              f"({elapsed:.1f}s, {(n_done * n_per_ct)/elapsed:.2f} tr/s)")

    return {
        "traces": np.concatenate(all_traces, axis=0),
        "mu_prime": np.concatenate(all_mus, axis=0),
        "labels": np.asarray(all_labels, dtype=np.int64),
        "pk_fp16": pk_fp16.hex(),
        "sk_pke_bytes_hex": bytes(sk_pke).hex(),
        "ct_fps": ct_fps,
    }


def _save_session(out_path: Path, sess: dict, *, level: str, p, alpha_pos: int,
                  alpha_neg: int, n_per_ct: int, samples: int, gain_db: float,
                  baud: int, fw_path: Path, fw_sha: str | None,
                  git: str, scope_sn: str, label: str) -> None:
    from datetime import datetime, timezone
    meta = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope_sn": scope_sn,
        "samples": samples,
        "gain_db": gain_db,
        "target": "smaug",
        "cmd": "Z",
        "send_len": 0,
        "resp_len": 32,
        "ss_ver": "SS_VER_1_1",
        "baud": baud,
        "n_attempted": int(sess["traces"].shape[0]),
        "git_rev": git,
        "firmware_hex": str(fw_path),
        "firmware_hex_sha256": fw_sha,
        "label": label,
        "experiment": "E1_same_sk",
        "level": level,
        "module_rank": p.module_rank,
        "hs": p.hs,
        "log_p": p.log_p,
        "alpha_pos": alpha_pos,
        "alpha_neg": alpha_neg,
        "components": list(range(p.module_rank)),
        "n_per_ct": n_per_ct,
        "pk_fp16": sess["pk_fp16"],
        "sk_pke_bytes_hex": sess["sk_pke_bytes_hex"],
        "ct_fp16_per_sweep": sess["ct_fps"],
        "label_component": sess["labels"][:, 0].tolist(),
        "label_alpha": sess["labels"][:, 1].tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, traces=sess["traces"], mu_prime=sess["mu_prime"],
                        meta=np.array(meta, dtype=object))
    print(f"    [OK] saved {out_path} traces.shape={sess['traces'].shape}")


def main() -> int:
    level = "smaug1"
    n_per_ct = 128
    samples = 24400
    gain_db = 25.0
    baud = 38400
    p = _params.get(level)
    alpha_pos, alpha_neg = 64, 192
    fw_path = (_REPO / "firmware" / "simpleserial-smaug" /
               f"simpleserial-smaug-CW308_STM32F4-{level}.hex")
    fw_sha = _file_sha256(fw_path) if fw_path.exists() else None
    git = _git_rev()

    import chipwhisperer as cw
    from host.cw_serial import pick_serial
    sn = pick_serial(None)
    print(f"[E1] level={level}, sn={sn}, n/CT={n_per_ct}, α=({alpha_pos},{alpha_neg})")

    scope = cw.scope(sn=sn)
    scope.default_setup()
    scope.adc.samples = samples
    scope.adc.offset = 0
    scope.adc.basic_mode = "rising_edge"
    scope.gain.db = gain_db
    scope.clock.adc_src = "clkgen_x4"
    scope.trigger.triggers = "tio4"
    reset_target(scope)
    target = cw.target(scope, cw.targets.SimpleSerial)
    target.baud = baud
    target.flush()
    time.sleep(0.3)

    try:
        # === Session A — fresh keygen ===
        print("\n[E1/A] fresh keygen + capture")
        sess_a = _do_session(scope, target, p,
                             alpha_pos=alpha_pos, alpha_neg=alpha_neg,
                             n_per_ct=n_per_ct, samples=samples, do_keygen=True)
        sk_short = sess_a["sk_pke_bytes_hex"][:16]
        print(f"    sk hash: {sk_short}")
        _save_session(_REPO / "traces/E1_sessionA.npz", sess_a,
                      level=level, p=p, alpha_pos=alpha_pos, alpha_neg=alpha_neg,
                      n_per_ct=n_per_ct, samples=samples, gain_db=gain_db,
                      baud=baud, fw_path=fw_path, fw_sha=fw_sha,
                      git=git, scope_sn=sn, label="E1_sessionA")

        # === Session B — same sk, no reset, no F ===
        # 단 short delay 후 capture 재개. board RAM 의 sk 가 그대로 남아있어야.
        print("\n[E1/B] same sk (no reset, no F) + capture")
        time.sleep(2.0)
        target.flush()
        sess_b = _do_session(scope, target, p,
                             alpha_pos=alpha_pos, alpha_neg=alpha_neg,
                             n_per_ct=n_per_ct, samples=samples, do_keygen=False)
        sk_b_short = sess_b["sk_pke_bytes_hex"][:16]
        if sk_b_short != sk_short:
            raise RuntimeError(
                f"[ERROR] sk mismatch — A={sk_short} B={sk_b_short}. "
                f"board RAM was reset between sessions — E1 invalid.")
        print(f"    sk hash verified: {sk_b_short}")
        _save_session(_REPO / "traces/E1_sessionB.npz", sess_b,
                      level=level, p=p, alpha_pos=alpha_pos, alpha_neg=alpha_neg,
                      n_per_ct=n_per_ct, samples=samples, gain_db=gain_db,
                      baud=baud, fw_path=fw_path, fw_sha=fw_sha,
                      git=git, scope_sn=sn, label="E1_sessionB")
    finally:
        try: target.dis()
        except Exception: pass
        try: scope.dis()
        except Exception: pass

    print("\n[E1] capture done. Now run scripts/alignment_diagnose_pair.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
