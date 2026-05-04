"""host/analysis/ 핵심 — Welch-t 구현 정합성 (synthetic ground truth)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.analysis import group, tvla  # noqa: E402


def test_welch_t_recovers_injected_leak() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=(500, 800)).astype(np.float32)
    b = rng.normal(0, 1, size=(500, 800)).astype(np.float32)
    b[:, 200:210] += 0.5
    r = tvla.welch_t(a, b)
    peak = int(np.argmax(np.abs(r.t)))
    assert 200 <= peak < 210, f"peak {peak} not in injected window 200..209"
    assert r.max_abs_t > 4.5, f"max|t|={r.max_abs_t} < 4.5"
    assert r.leaky_idx.size > 0


def test_welch_t_zero_signal_zero_t() -> None:
    """양쪽 모두 같은 분포 → 평균적으로 t≈0, 큰 값 거의 없음."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=(500, 400)).astype(np.float32)
    b = rng.normal(0, 1, size=(500, 400)).astype(np.float32)
    r = tvla.welch_t(a, b)
    # 우연한 큰 값 1~2 개는 허용. 본격적인 PoI 가 안 나와야 함.
    assert r.leaky_idx.size <= 5, (
        f"H0 데이터에서 leaky points 가 너무 많음 ({r.leaky_idx.size})"
    )


def test_welch_t_rejects_short_arrays() -> None:
    raised = False
    try:
        tvla.welch_t(np.zeros((1, 10), dtype=np.float32),
                     np.zeros((1, 10), dtype=np.float32))
    except ValueError:
        raised = True
    assert raised


def test_split_by_response_byte_basic() -> None:
    traces = np.arange(40, dtype=np.float32).reshape(10, 4)
    resp = np.zeros((10, 1), dtype=np.uint8)
    resp[3, 0] = 1
    resp[7, 0] = 1
    a, b = group.split_by_response_byte(traces, resp, byte_index=0, value=0)
    assert a.shape[0] == 8
    assert b.shape[0] == 2
    assert (a == traces[[0, 1, 2, 4, 5, 6, 8, 9]]).all()
    assert (b == traces[[3, 7]]).all()


def test_design_window_poi_keeps_signed_design_difference() -> None:
    """µ′ label 없이 α_pos/α_neg 디자인 차분만으로 window별 PoI를 잡는다.

    sign_v 를 t-score 부호로 곱하면 모든 점수가 양수가 되어 secret sign이
    사라지므로, design-window 경로는 raw mean_pos - mean_neg 부호를 보존해야 한다.
    """
    from scripts.analyze_multi_seed import _learn_poi_from_design_windows

    rng = np.random.default_rng(123)
    n_bits = 4
    samples = 40
    pos = rng.normal(0, 0.05, size=(8, samples)).astype(np.float32)
    neg = rng.normal(0, 0.05, size=(8, samples)).astype(np.float32)
    # 각 10-sample window 안에 부호가 다른 design 차분을 주입.
    for bi, amp in enumerate([1.0, -1.0, 0.8, -0.8]):
        j = bi * 10 + 4
        pos[:, j] += amp
        neg[:, j] -= amp

    poi, t_score, sign_v = _learn_poi_from_design_windows(
        pos, neg, n_bits=n_bits, window_start=0, window_end=samples)
    assert poi.tolist() == [4, 14, 24, 34]
    assert np.all(sign_v == 1)
    assert t_score[0] > 0 and t_score[1] < 0


if __name__ == "__main__":
    fns = sorted(n for n in globals() if n.startswith("test_"))
    for n in fns:
        print(f"[RUN] {n}")
        globals()[n]()
        print(f"[OK ] {n}")
    print(f"[OK ] {len(fns)} tests")
