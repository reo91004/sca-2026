#!/usr/bin/env python3
"""캡처 .npz 한 개를 받아 overview PNG 생성 + sanity validation.

scripts/smoke.sh 의 [E/F] 단계를 모듈로 분리한 thin wrapper. 단독 실행도
가능 (재시각화, 새 캡처 산출물 검증).

사용법:
    scripts/plot_overview.py traces/smoke_smaug1.npz \\
        --png results/E1_overview_smaug1.png

    scripts/plot_overview.py traces/foo.npz --kem-match-check \\
        --expected-n 1000 --expected-samples 24400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# host 패키지를 루트에서 import 할 수 있도록 sys.path 보정.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from host.analysis import io as a_io  # noqa: E402
from host.analysis import validate as a_validate  # noqa: E402
from host.analysis import viz as a_viz  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", type=Path, help="입력 .npz (host/capture.py 산출물)")
    p.add_argument("--png", type=Path, default=None,
                   help="출력 PNG (기본: results/<npz-stem>_overview.png)")
    p.add_argument("--expected-n", type=int, default=None,
                   help="N 검증 (없으면 skip)")
    p.add_argument("--expected-samples", type=int, default=None,
                   help="T 검증 (없으면 skip)")
    p.add_argument("--kem-match-check", action="store_true",
                   help="SMAUG 1바이트 mismatch flag 가 모두 0 인지 검증")
    p.add_argument("--n-overlay", type=int, default=5,
                   help="평균 plot 위에 겹쳐 그릴 무작위 트레이스 개수 (기본 5)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cap = a_io.load_capture(args.npz)
    print(f"[VAL] {cap.summary()}")
    print(f"[VAL] meta: {cap.meta}")

    errs = a_validate.validate_capture(
        cap,
        expected_n=args.expected_n,
        expected_samples=args.expected_samples,
        do_kem_match_check=args.kem_match_check,
    )
    if errs:
        print("[VAL] 실패 원인:")
        for e in errs:
            print(f"       - {e}")
        return 1
    print("[VAL] OK — 캡처 sanity 통과")

    png = args.png or _REPO_ROOT / "results" / f"{args.npz.stem}_overview.png"
    out = a_viz.plot_overview(cap, png, n_overlay=args.n_overlay)
    print(f"[VIZ] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
