# NTRU+768 트랙

Chosen-CT NTT-domain CPA on NTRU+768 (KpqC final, GitHub v1.0 / 2026-01-30),
ChipWhisperer-Lite + CW308T-STM32F415.

## 진입점

처음 읽는다면:

1. **[FLOW.md](FLOW.md)** — 트랙 한 눈에 (problem → experiments → results timeline). 5~10 분.
2. **[HANDOFF.md](HANDOFF.md)** — Paper-grade 핵심 결과 + 재시작 instructions. 한 세션 분량.
3. **[EXPERIMENTS.md](EXPERIMENTS.md)** — 매 phase 의 가정·과정·결과 누적 로그 (역시간순, 가장 큰 문서).

전략·gate 가 궁금하면 [PLAN.md](PLAN.md), paper outline 은 [PAPER_OUTLINE.md](PAPER_OUTLINE.md).

## 문서 역할

| 파일 | 역할 | 갱신 빈도 |
|---|---|---|
| `FLOW.md` | navigational summary (트랙 전체 흐름) | minor — phase 결정 시점에만 |
| `PLAN.md` | 전략·gate·산출물 규약 (single source of truth) | medium — 새 phase 시작/종료 |
| `EXPERIMENTS.md` | 매 phase 실험 로그 (역시간순) | high — 매 실험 후 entry 추가 |
| `HANDOFF.md` | paper-grade 결과 + 재시작 가이드 | low — phase 종료/세션 인계 시 |
| `PAPER_OUTLINE.md` | paper section / figure 안내 | low — paper 작업 시점 |

## 코드 위치

| 경로 | 설명 |
|---|---|
| `scripts/ntruplus/n01_*` … `n59_*` | 분석/캡처 스크립트 (n* prefix, phase 순 번호) |
| `scripts/ntruplus/run_main_capture.sh`, `run_phase45_chain.sh`, `phase45_status.sh` | shell driver |
| `host/ntruplus/` | host-side NTT/codec/inject 패키지 |
| `host/cw_serial.py` | ChipWhisperer serial picker (양 트랙 공유) |
| `firmware/simpleserial-ntruplus/` | STM32F415 펌웨어 |
| `tests/ntruplus/` | round-trip + smoke 테스트 |
| `traces/ntruplus768/<phase>/` | 캡처 산출물 (git-ignored) |
| `results/ntruplus/<phase>/` | 분석 산출물 (git-ignored) |
