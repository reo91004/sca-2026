# SCA Workbench — KpqC SCA on ChipWhisperer-Lite + STM32F415

Two independent side-channel research tracks live in this repo:

| Track | Entry point | Status (2026-05-12) |
|---|---|---|
| **NTRU+768** chosen-CT NTT-domain CPA | [`docs/ntruplus/README.md`](docs/ntruplus/README.md) | Active — paper-grade Phase 4 종료: 2 perfect TOP-1 + 17/240 top-100 partial NTT-coordinate recovery |
| **SMAUG-T** smaug{1,3,5} chosen-CT SCA | [`docs/smaug/README.md`](docs/smaug/README.md) | **Archived** — trace-only chosen-CT cross-key fail. PCO oracle 은 sk-universal 이지만 µ′-bit 직접 leak 없음 |

각 트랙 문서는 한 페이지 진입점 (`README.md`) + phase-by-phase 상세 로그
(`experiments.md`) 두 파일이다. NTRU+ 트랙은 paper writing 직전, SMAUG-T
트랙은 코드/문서는 보존하되 데이터(traces/results) 는 2026-05-12 정리에서
제거된 archived 상태.

## Repository layout

```
sca-2026/
├── README.md                    (이 문서 — cross-track hub)
├── docs/
│   ├── README.md                (docs hub)
│   ├── ntruplus/                (NTRU+ 트랙 문서: README + experiments)
│   ├── smaug/                   (SMAUG-T 트랙 문서: README + experiments)
│   └── paper/                   (HQC reference PDFs, 공유)
├── firmware/
│   ├── simpleserial-ntruplus/   (NTRU+ STM32F415 펌웨어)
│   └── simpleserial-smaug/      (SMAUG-T STM32F415 펌웨어)
├── host/                        (NTRU+ 와 SMAUG-T 동일한 self-contained 구조)
│   ├── cw_serial.py             (양 트랙 공유 — ChipWhisperer serial picker)
│   ├── ntruplus/                (NTRU+ codec/NTT/inject 패키지)
│   └── smaug/                   (SMAUG-T 패키지)
│       ├── analysis/            (TVLA + sparse-recover)
│       ├── capture.py           (캡처 driver)
│       ├── chosen_ct.py         (chosen-CT serial 세션)
│       ├── upload.py            (펌웨어 업로더)
│       └── (ciphertext, codec, poly_mul, sk_partition, chosen, params)
├── scripts/
│   ├── ntruplus/                (n01_*~n59_* + shell drivers)
│   └── smaug/                   (s1_*/s2_*/s3_*/s4_* + smoke.sh)
├── tests/
│   ├── ntruplus/                (round-trip + smoke)
│   ├── smaug/                   (analyzer 정합성 + oracle 테스트)
│   └── run_all.py               (양 트랙 일괄 실행)
├── results/
│   ├── ntruplus/                (NTRU+ paper-grade 산출물; git-ignored, dir만 유지)
│   └── smaug/                   (.gitkeep 만 보존 — 트랙 archived 로 삭제)
├── traces/
│   ├── ntruplus768/             (Phase 1/3/4.5/4.6 paper evidence ~5.6 GB)
│   └── (직속 — git-ignored; SMAUG-T traces 는 archived 로 삭제)
└── upstream/                    (libopencm3 / pqm4_sdk reference; git-ignored)
```

`host/` 양 트랙 모두 self-contained package: NTRU+ 는 `host/ntruplus/`,
SMAUG-T 는 `host/smaug/` 안에 codec / chosen-CT / capture / upload / analysis
가 모두 들어 있다. 공유 모듈은 `host/cw_serial.py` 하나뿐 (시리얼 선택).

## 빠른 실행

```bash
# 양 트랙 host/codec/analysis 정합성 테스트 (89/89 OK)
python3 tests/run_all.py

# NTRU+ — paper-grade 분석 진입점
python3 scripts/ntruplus/n01_phase1_capture.py --help
python3 scripts/ntruplus/n39_full_stack.py --help
python3 scripts/ntruplus/n24_combined_analysis.py

# SMAUG-T — archived (코드는 보존됨, 데이터는 새로 캡처해야 동작)
python3 scripts/smaug/s1_t_capture_main.py --help
python3 scripts/smaug/s2_z_analyze.py --help
```

캡처 보드는 자동으로 시리얼을 선택한다 (`host/cw_serial.py` 의
`pick_serial` 정책 — TARGET_SN 우선, FORBIDDEN_SN 거부).

## 펌웨어 빌드/플래시

```bash
# NTRU+
make -C firmware/simpleserial-ntruplus PLATFORM=CW308_STM32F4 NTRUPLUS_LEVEL=768
# (NTRU+ 캡처 스크립트가 hex 경로 hard-code 후 cw.program_target 자체 호출)

# SMAUG-T
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4 SMAUG_LEVEL=1
python3 -m host.smaug.upload firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
```

## Corrigendum

이전 git history (특히 commit `db3670d` 이전) 의 SMAUG-T "8 traces / 9
keypairs / 100%" 류 claim 은 **IND-CCA 위반** (µ′ label 로 PoI 학습) 으로
정정되어 paper-grade attack-valid 결과에서 제외되었다. 자세한 내용은
[`docs/smaug/README.md`](docs/smaug/README.md) §5 와 메모리
`corrigendum_2026_05_04.md`.
