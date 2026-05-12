# NTRU+768 트랙 — Problem · Experiments · Results Flow

이 문서는 NTRU+ chosen-CT NTT-domain CPA 라인을 "어떤 문제 → 어떤 실험 →
어떤 결과" 흐름으로 한 눈에 보여주는 navigational summary 다. 한 entry
당 1~2 줄의 결정/결과만 담고, 자세한 가정·수치·다음 단계는 source-of-truth
문서들 (`PLAN.md`, `EXPERIMENTS.md`, `HANDOFF.md`) 에 있다.

타겟: NTRU+768 (KpqC final, GitHub v1.0 / 2026-01-30) · ChipWhisperer-Lite +
CW308T-STM32F415 · pqm4-style C reference.

상태 요약 (2026-05-11): **Phase 4 完, Phase 4.5/4.6 partial-recovery 보강 完, paper-grade**.

## 트랙 전체 흐름

| Phase | 문제 / 가정 | 실험 | 핵심 결과 |
|---|---|---|---|
| 0a | host-side NTT/codec 가 보드와 정합한가? | `host/ntruplus/` 패키지 + round-trip test | NTT, codec, basemul 100% round-trip. |
| 0b | 펌웨어가 정상 동작하는가? | smoke + codec parity (`tests/ntruplus/smoke_ntruplus.py`) | hex 빌드/플래시/encap/decap PASS. |
| 1 | natural `D` (valid CT) 의 leak map 은? | K=8 sk × N=20 trace, sample-wise TVLA + bytewise HW | basemul 시간대에서 sk-dependent leak 확인, PoI 후보 식별. |
| 1-timeline | `D` 명령의 어느 함수 호출 구간이 leak 인가? | decimate=4 timeline mapping | `poly_basemul(m1, c, f)` 가 단연 우세 (attack hook B 위치). |
| 2 | byte-HW triage 가 PoI 를 좁히는가? | K=8, N=20 byte-HW CPA | per-byte z 약함 — 단일 byte HW 로 부족. Phase 3 selected-lane 으로 전환. |
| 3-scout v1/v2/v3 | NTT-domain selected-lane CPA 의 single-key signal 은? | G={6,12} 동일 γ, c=α·X^j 다중 lane 시도 | v1 PASS 가 γ-byte 오염 → v3 const-γ set 으로 정화, 4 slots/lane 식별. |
| 3-multikey | cross-key 로도 PoI 일관성이 있는가? | K=4 sk × 동일 designs | per-key CPA + Bonferroni 통계 안정. |
| 3-decompose / basemul-model | leak source 가 mod-q 인가 Montgomery 인가? | basemul-timing model + dual-channel HW (M1, M5) | M1 (mod-q) 와 M5 (montgomery_reduce(γ·f)) 둘 다 분리 가능한 channel. |
| 4-recover | multi-victim 으로 TOP-1 NTT 좌표 회수 가능한가? | 240 cases (4 lanes × 28 keys × 4 slots) | **2 perfect TOP-1** (f=16, f=882), **17 unique top-100 / 240 = 7.08%**, 4 lanes 모두 회수. |
| 4-mechanistic | 어떤 mechanistic finding 이 paper-quality 인가? | M1/M5 dual-channel, Zsum ensemble, full-stack | 5종 finding 정리 (HANDOFF.md §3.3). |
| 4.5 | single victim multi-lane sk leakage 도 가능한가? | 2 victims × 88 lanes × 4 slots = 352 cases (later 3rd victim) | M1 t100 6/352, full 7/352, lane=104 near-TOP-1. 192-lane projection 31% entropy reduction. |
| 4.6 | wide-coverage 3rd victim + G=200 compact scout 으로 회수가 강화되는가? | G=200 single-victim, 16 calib lanes | M1/full top-100 0/16, M5 1/16. **simple G 확장 = negative scout**. partial-recovery 로만 보고. |
| 4.6-G200 diag | G200 single-victim 의 oracle 분명 PoI 가 self-oracle selection bias 인가? | PoI/window diagnostic + cv window CPA | true PoI 의 oracle gain 이 candidate-null 과 일치 → bias 로 설명. real attack-compatible PoI 는 t100 0~1/16 에 머무름. |

## 현재 상태 (2026-05-11)

- Paper-grade 결과: **2 perfect TOP-1 + 17/240 top-100 multi-victim**.
- single-victim multi-lane (Phase 4.5/4.6): **partial NTT-coordinate information disclosure** 로 conservative 하게 표기. full sk recovery 는 현재 회수율로는 lattice 까지 도달 X.
- 다음 단계 (HANDOFF.md §4) : paper writing 시작 가능. extension 후보는 (a) lattice-aided recovery threshold 추정, (b) profiled / template attack, (c) Karatsuba/Toom-Cook sub-step SASCA.

## 알려진 잘못된 가정 / 정정

- **µ′-label 으로 PoI 학습은 IND-CCA 위반** — Phase 4 attack-valid 결과는 모두 attack-side 만 PoI 사용 (µ′ 미지) 으로 산출. early 분석 중 "PoI 학습이 µ′ 라벨 사용" 한 결과는 corrigendum 으로 정정.
- NTT 좌표 회수 ≠ full sk recovery. lattice 단계는 별도 future work.

## 어디서 읽어야 하는가

| 목적 | 파일 |
|---|---|
| 트랙 전략·gate (single source of truth) | `PLAN.md` (구 `NTRUplus.md`) |
| 매 phase 의 가정·과정·결과 실험 로그 (역시간순) | `EXPERIMENTS.md` (구 `IDEA.md`) |
| Paper-grade 핵심 결과 + 재시작 가이드 | `HANDOFF.md` |
| Paper section / figure outline | `PAPER_OUTLINE.md` |
| 트랙 진입점 (README) | `README.md` |
| Cross-track hub | `../README.md` |
