#!/usr/bin/env python3
"""[UTIL] Welch-t TVLA 2-class plot dev tool.

Paper Section 8.1 (SNR phase transition) 의 |t| 곡선 시각화 dev tool. 두
.npz (또는 한 .npz + 응답분기) 의 클래스간 Welch-t 결과 PNG.

Note (history/):
    분석 라이브러리는 host/analysis/tvla.py (main flow). 이 plot 은 dev
    visualization. paper figure 생성 시 별도 작성.

사용 모드 (legacy):
  (a) 두 캡처 비교: scripts/util_plot_tvla.py A.npz B.npz --png out.png
  (b) 단일 캡처 + 응답 분기:
      scripts/plot_tvla.py traces/X.npz --split-response-byte 0 \\
          --split-value 0 --png results/E2_tvla_X.png

분석 정의:
    t_i = (μ_A,i − μ_B,i) / sqrt(σ²_A,i / N_A + σ²_B,i / N_B)
    threshold = 4.5  (TVLA 표준; |t|>4.5 인 시간점이 PoI 후보)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from host.analysis import group as a_group  # noqa: E402
from host.analysis import io as a_io  # noqa: E402
from host.analysis import tvla as a_tvla  # noqa: E402
from host.analysis import viz as a_viz  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz_a", type=Path, help="클래스 A .npz")
    p.add_argument("npz_b", type=Path, nargs="?", default=None,
                   help="클래스 B .npz (생략 시 --split-response-byte 사용)")
    p.add_argument("--split-response-byte", type=int, default=None,
                   help="단일 .npz 모드: 이 응답 바이트 인덱스로 분기")
    p.add_argument("--split-value", type=int, default=0,
                   help="단일 .npz 모드: byte == value 가 클래스 A (기본 0)")
    p.add_argument("--threshold", type=float, default=4.5,
                   help="TVLA threshold (기본 4.5)")
    p.add_argument("--png", type=Path, default=None,
                   help="출력 PNG (기본: results/<a-stem>__vs__<b-stem>_tvla.png)")
    p.add_argument("--title", type=str, default=None,
                   help="figure title 직접 지정")
    return p.parse_args()


def _load_two_classes(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str, str]:
    """두 클래스 (a, b) 트레이스 + 사람 읽기용 라벨을 반환."""
    cap_a = a_io.load_capture(args.npz_a)
    if args.npz_b is not None:
        cap_b = a_io.load_capture(args.npz_b)
        if cap_a.samples != cap_b.samples:
            raise SystemExit(
                f"[FAIL] sample-length 불일치: A={cap_a.samples} B={cap_b.samples}"
            )
        return cap_a.traces, cap_b.traces, args.npz_a.stem, args.npz_b.stem

    if args.split_response_byte is None:
        raise SystemExit(
            "[FAIL] B 캡처를 안 줬으면 --split-response-byte 가 필요"
        )

    a_t, b_t = a_group.split_by_response_byte(
        cap_a.traces, cap_a.responses,
        byte_index=args.split_response_byte, value=args.split_value,
    )
    label_a = f"{args.npz_a.stem}[resp[{args.split_response_byte}]=={args.split_value}]"
    label_b = f"{args.npz_a.stem}[resp[{args.split_response_byte}]!={args.split_value}]"
    if a_t.shape[0] < 2 or b_t.shape[0] < 2:
        raise SystemExit(
            f"[FAIL] 분기 후 각 클래스 N이 부족: |A|={a_t.shape[0]}, |B|={b_t.shape[0]}"
        )
    return a_t, b_t, label_a, label_b


def main() -> int:
    args = parse_args()
    a_t, b_t, label_a, label_b = _load_two_classes(args)
    print(f"[TVLA] A={label_a} N={a_t.shape[0]}")
    print(f"[TVLA] B={label_b} N={b_t.shape[0]}")

    result = a_tvla.welch_t(a_t, b_t, threshold=args.threshold)
    print(f"[TVLA] {result.summary()}")

    leaky = result.leaky_idx
    if leaky.size:
        # 첫/마지막/최댓값 인덱스만 한 줄 요약
        peak = int(np.argmax(np.abs(result.t)))
        print(
            f"[TVLA] leaky range: first={int(leaky[0])} "
            f"last={int(leaky[-1])} peak@{peak} (t={result.t[peak]:+.2f})"
        )
    else:
        print("[TVLA] no leaky points — 트리거/게인/캡처 윈도우 점검 필요")

    if args.png is None:
        if args.npz_b is not None:
            stem = f"{args.npz_a.stem}__vs__{args.npz_b.stem}_tvla"
        else:
            stem = f"{args.npz_a.stem}_split{args.split_response_byte}_tvla"
        png = _REPO_ROOT / "results" / f"{stem}.png"
    else:
        png = args.png

    title = args.title or f"TVLA: {label_a}  vs  {label_b}"
    out = a_viz.plot_tvla(result, png, title=title)
    print(f"[VIZ] saved {out}")
    return 0 if leaky.size else 2  # 누출 없으면 비정상 종료 (사람이 알아채라)


if __name__ == "__main__":
    raise SystemExit(main())
