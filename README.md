# SCA Workbench — KpqC SCA on ChipWhisperer-Lite + STM32F415

Two independent side-channel research tracks live in this repo:

| Track | 진입점 | Status (2026-05-11) |
|---|---|---|
| **NTRU+768** chosen-CT NTT-domain CPA | [`docs/ntruplus/`](docs/ntruplus/README.md) | Paper-grade Phase 4 종료 — 2 perfect TOP-1 + 17/240 top-100 partial NTT-coordinate recovery |
| **SMAUG-T** smaug{1,3,5} chosen-CT SCA | [`docs/smaug/`](docs/smaug/README.md) | Trace-only chosen-CT cross-key fail. PCO oracle 은 sk-universal 이지만 µ′-bit 직접 leak 없음 |

각 트랙의 한 눈 timeline (problem → experiments → results) 은 트랙별
`FLOW.md` 에 있다:

- [`docs/ntruplus/README.md`](docs/ntruplus/README.md) — Phase 0 → 4.6 overview, paper-grade results, reproduction (+ `experiments.md` for phase-by-phase log)
- [`docs/smaug/FLOW.md`](docs/smaug/FLOW.md) — S1/S2/S2.x/S3/S1U/S4 + branch (C2, R, Y, D-Pair, S5) + PCO pilots 1~6

자세한 가정·수치·다음 단계는 각 트랙 `README.md` 를 따라가면 source-
of-truth 문서들 (NTRU+ HANDOFF/PLAN/EXPERIMENTS, SMAUG-T IDEA + branch md)
로 연결된다.

## Repository layout

```
sca-2026/
├── README.md                    (이 문서 — cross-track hub)
├── docs/
│   ├── README.md                (docs hub)
│   ├── ntruplus/                (NTRU+ 트랙 문서)
│   ├── smaug/                   (SMAUG-T 트랙 문서)
│   └── paper/                   (HQC reference PDFs, 공유)
├── firmware/
│   ├── simpleserial-ntruplus/   (NTRU+ STM32F415 펌웨어)
│   ├── simpleserial-smaug/      (SMAUG-T 펌웨어)
│   └── simpleserial-hqc/        (HQC reference firmware)
├── host/
│   ├── ntruplus/                (NTRU+ codec/NTT/inject 패키지)
│   ├── smaug/                   (SMAUG-T ciphertext/codec/poly_mul 패키지)
│   ├── analysis/                (TVLA + sparse-recover 분석 라이브러리)
│   ├── cw_serial.py             (양 트랙 공유 — ChipWhisperer serial picker)
│   ├── chosen_ct.py, upload.py, capture.py   (SMAUG-T 전용)
├── scripts/
│   ├── ntruplus/                (n01_*~n59_* + shell drivers)
│   └── smaug/                   (s1_*/s2_*/s3_*/s4_* + smoke.sh)
├── tests/
│   ├── ntruplus/                (round-trip + smoke)
│   ├── smaug/                   (analyzer 정합성 + oracle 테스트)
│   └── run_all.py               (양 트랙 일괄 실행)
├── results/{ntruplus,smaug}/    (분석 산출물, git-ignored 단 디렉토리만 유지)
├── traces/                      (캡처 산출물, git-ignored)
└── history/                     (top-level archive — 옛 자료)
```

## 빠른 실행

```bash
# 양 트랙 테스트 (host/codec/analysis 정합성)
python3 tests/run_all.py

# NTRU+ 캡처/분석 entry
python3 scripts/ntruplus/n01_phase1_capture.py --help
python3 scripts/ntruplus/n39_full_stack.py --help

# SMAUG-T 캡처/분석 entry
python3 scripts/smaug/s1_t_capture_main.py --help
python3 scripts/smaug/s2_z_analyze.py --help
```

캡처 보드는 자동으로 시리얼을 선택한다 (`host/cw_serial.py` 의
`pick_serial` 정책 — TARGET_SN 우선, FORBIDDEN_SN 거부).

## 펌웨어 빌드/플래시

```bash
# SMAUG-T
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4 SMAUG_LEVEL=1
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex

# NTRU+ (scripts/ntruplus/ 의 캡처 스크립트가 hex 경로 hard-code; --help 참조)
```

## 정정 (Corrigendum)

이전 commit 들에 남아 있을 수 있는 "instrumented µ′-label" 결과 (예 :
SMAUG-T `8 traces / 9 keypairs / 100%`) 는 **IND-CCA 위반** 이며 attack-valid
주장에서 제외되었다. 자세한 내용은 [`docs/smaug/FLOW.md`](docs/smaug/FLOW.md)
와 메모리 `corrigendum_2026_05_04.md` 참조.
