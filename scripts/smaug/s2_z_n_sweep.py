#!/usr/bin/env python3
"""S2 — N 별 cross-sk Welch-t 측정 ('Z'). 분석 SNR 의존성 가시화.

s2_z_skA.npz, s2_z_skB.npz (각 N=200) 으로 N=50, 100, 200 별 |Welch-t| 측정.
N 가 늘면 |t| 도 sqrt(N) 으로 늘어남이 expectation.

또 30-sk dataset 에서 최강 cross-sk std 위치 (예: 19713) 의 단일 좌표 회귀
fit 의 R^2 를 N 별로 측정 — sample 단계에서 sk-dependent component 가 *크다*
는 것을 객관화.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402


def _load(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["traces"], data["meta"].item()


def main() -> int:
    A_path = _REPO / "traces" / "s2_z_skA.npz"
    B_path = _REPO / "traces" / "s2_z_skB.npz"
    if not A_path.exists() or not B_path.exists():
        print("skA/skB 없음 — 먼저 N=200 capture 실행")
        return 1
    A, mA = _load(A_path)
    B, mB = _load(B_path)
    print(f"[INFO] sk_A pk={mA['pk_fp16'][:16]}…, N={A.shape[0]}, T={A.shape[1]}")
    print(f"[INFO] sk_B pk={mB['pk_fp16'][:16]}…, N={B.shape[0]}, T={B.shape[1]}")
    n_max = min(A.shape[0], B.shape[0])

    print("\n[N sweep] cross-sk Welch-t (sk_A vs sk_B)")
    for N in (10, 25, 50, 100, 150, 200):
        if N > n_max:
            continue
        ma = A[:N].mean(axis=0)
        mb = B[:N].mean(axis=0)
        sa = A[:N].std(axis=0, ddof=1)
        sb = B[:N].std(axis=0, ddof=1)
        se = np.sqrt(sa**2 / N + sb**2 / N)
        t = (ma - mb) / np.where(se == 0, 1, se)
        max_t = float(np.abs(t).max())
        max_t_at = int(np.argmax(np.abs(t)))
        print(f"  N={N:4d}: max |t|={max_t:8.1f} @ sample {max_t_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
