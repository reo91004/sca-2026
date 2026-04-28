"""sca-2026 공통: CW1173 ChipWhisperer-Lite 시리얼 선택 정책.

벤치 환경:
    이 사용자의 책상에는 CW1173 두 대가 동시에 꽂혀 있을 수 있다.
        - TARGET_SN  -> CW308T-STM32F4 (STM32F415)  : SCA 대상
        - FORBIDDEN_SN -> 별도 DUT             : F415용 펌웨어를 올리면 안 됨
    한 대만 꽂혀 있을 때는 자동으로 그것을 쓰되, FORBIDDEN_SN 만 꽂혀
    있다면 안전을 위해 거부한다. 자세한 규칙은 ``pick_serial`` docstring 참조.
"""

from __future__ import annotations

import sys

import chipwhisperer as cw


TARGET_SN = ""
FORBIDDEN_SN = "50203220594a48303330373133323037"


def pick_serial(
    requested: str | None,
    target_sn: str = TARGET_SN,
    forbidden_sn: str = FORBIDDEN_SN,
) -> str:
    """사용할 CW1173 시리얼을 결정한다.

    규칙:
        1. ``forbidden_sn`` 보드는 어떤 경로로도 절대 사용하지 않는다.
        2. ``requested`` 가 주어지면 그것을 사용한다 (1) 위반시 거부.
        3. 자동 선택:
            - ``target_sn`` 보드가 연결되어 있으면 그것을 쓴다.
            - 그렇지 않고 forbidden 을 제외한 보드가 정확히 1개면 그것을
              쓴다 (벤치 한 대 구성용 폴백).
            - 0개면 안내와 함께 종료.
            - 2개 이상이면 모호하므로 ``--serial`` 명시를 요구.
    """
    if requested == forbidden_sn:
        sys.exit(
            f"[ABORT] sn={forbidden_sn} 보드는 F415가 아닌 다른 DUT 용으로 "
            "이 스크립트에서 사용 금지."
        )

    devices = cw.list_devices()
    by_sn = {d.get("sn"): d.get("name") for d in devices}

    if requested is not None:
        if requested not in by_sn:
            seen = ", ".join(f"{n}({s})" for s, n in by_sn.items()) or "<없음>"
            sys.exit(
                f"[ABORT] sn={requested} 보드를 찾지 못함. "
                f"연결된 NewAE 장치: {seen}"
            )
        return requested

    candidates = [s for s in by_sn if s != forbidden_sn]

    if target_sn in candidates:
        return target_sn

    if len(candidates) == 1:
        sn = candidates[0]
        print(
            f"[INFO] 표준 F415 sn({target_sn}) 보드 없음. "
            f"단일 연결 보드 sn={sn} ({by_sn[sn]}) 자동 선택."
        )
        return sn

    if not candidates:
        if forbidden_sn in by_sn:
            sys.exit(
                f"[ABORT] 연결된 보드가 sn={forbidden_sn} (F415 아님) 뿐. "
                "F415 보드를 연결하라."
            )
        sys.exit("[ABORT] CW1173 보드가 하나도 연결되지 않음.")

    listing = ", ".join(f"{by_sn[s]}({s})" for s in candidates)
    sys.exit(
        f"[ABORT] 사용할 CW 보드가 모호함 (후보 {len(candidates)}개). "
        f"--serial 로 명시하라. 후보: {listing}"
    )
