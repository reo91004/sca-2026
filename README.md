# SMAUG-T SCA Workbench

> Paper draft: `docs/latex/main.tex` · Experiment notes: `docs/experiments.md`

KpqC 2025 우승 KEM **SMAUG-T** 의 chosen-ciphertext side-channel 실험
저장소입니다. ChipWhisperer-Lite (CW1173) + Vcc-shunt + STM32F415 unmasked
reference 구현 위에서 smaug1/3/5의 chosen-CT trace, diagnostic oracle, known-key
calibration 경로를 분리해 평가합니다.

2026-05-04 정합성 점검 + score 분해 후 기준:
- `Z`의 32B µ′ 응답과 `X` sk dump는 **instrumented oracle / known-key calibration 전용**.
- 순수 trace SCA 분석은 target µ′/sk 사용 금지. design label + trace 만.
- smaug3/5의 empirical mapping은 *독립* calibration keypair에서 학습 (`--mapping calibrated`).

**Threat model 별 결과 (n=256, hs=70/88/87, sparse_recover HW=hs constraint)**

| threat model | smaug1 (k=2) | smaug3 (k=3) | smaug5 (k=4) |
|---|---:|---:|---:|
| trace SCA, design label only (attack-valid) | 0.562 ± 0.009 (≈ random 0.565) | — | — |
| trace SCA, cross-session µ′-profiled schedule | 0.564 ± 0.009 (transfer fail) | — | — |
| board response oracle, host predict (no calib) | — | 0.637 (2-α) / 0.850 (3-α) | 0.880 (3-α) |
| board response oracle, **calibrated cross-seed (3-α)** | — | **1.000** (양방향) | **1.000** (양방향) |
| µ′-leak instrumented (= same-session µ′ 사용) | 1.000 (어제 결과) | 1.000 | 1.000 |

핵심:
- smaug1 trace SCA 는 capture-session 간 alignment 가 random (norm-corr ≤ 0.1, 단순 shift 로 보정 불가) → 현재 setup 으로는 attack-valid full-sk recovery 불가.
- smaug3/5 의 `calibrated` mapping 은 *trace 와 무관하게 board µ′ 응답만* 사용. 3-α (oracle pair + support α=132) 일 때 cross-key 100%. boundary mismatch 가 *device-specific* 이고 *key-specific 이 아님*을 입증.
- 2-α 만으로는 (0,0) tuple 의 boundary ambiguity (sk=-1 의 ~73% 가 (0,0) 으로 떨어짐) 를 calibration 도 못 푼다 → 3-α mandatory.

| 평가 metric | 값 |
|---|---|
| Setup | CW-Lite + Vcc shunt + STM32F415 |
| Trace SCA path | random level (alignment fail) — 재캡처 protocol 필요 |
| Board oracle path | smaug3 3-α / smaug5 3-α calibrated cross-seed 100% |
| Keypairs evaluated | 9 smaug1, 4 smaug3 (2-α + 3-α), 2 smaug5 (3-α) |
| Method | (a) trace Welch-t per coef-window — fail; (b) calibrated tuple→class — 3-α 100% |

## 빠른 reproduce

```bash
# 1. 펌웨어 빌드 + 플래시 (한 번)
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4 SMAUG_LEVEL=1
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex

# 2. Multi-seed 캡처 (5 keypairs × N=128, ≈25분)
for s in 1 2 3 4 5; do
    python3 scripts/run_attack.py -n 128 --out traces/attack_seed${s}.npz
done

# 3. trace SCA path (attack-valid, 현재 setup 에서는 random level)
python3 scripts/analyze_multi_seed.py traces/attack_seed{1..5}.npz \
    --poi-source design-window --out-prefix results/multi_seed_design_window
# → full_sk ≈ 0.562 ≈ random baseline 0.565. alignment 진단:
python3 scripts/leakage_phase_scan.py
python3 scripts/alignment_diagnose.py

# 4. cross-session µ′-profiled schedule (negative result)
python3 scripts/analyze_multi_seed.py traces/attack_seed{2..5}.npz \
    --poi-source schedule --schedule-source traces/attack_seed1.npz \
    --out-prefix results/cross_key_schedule
# → 0.564 ≈ random. capture-session 간 alignment 가 random.

# 5. board response oracle path (smaug3/5, calibrated cross-seed)
python3 scripts/analyze_multibit.py traces/attack_smaug3_seed3_3alpha_n128.npz \
    --mapping calibrated --calibration traces/attack_smaug3_seed4_3alpha_n2.npz \
    --out-prefix results/smaug3_3a_calib
# → full_sk 1.000 (cross-key). 단 board µ′ 응답 직접 dump 가정.
```

## 디렉토리 구조

```
sca-2026/
├── README.md                 ← 이 파일
├── docs/                     ← paper draft, 살아있는 메모 (gitignored)
│   ├── paper_draft.md        ←   paper 본문 markdown (Abstract + 10 sections)
│   ├── countermeasures.md    ←   masking / shuffling 분석
│   ├── HANDOFF.md            ←   세션 인계 메모
│   ├── idea.md               ←   method 도출 history
│   ├── experiments.md        ←   실험 매트릭스 + 결과
│   └── notes/toom_cook_layout.md   ←   smaug1.a 의 함수 cycle 분석
├── firmware/simpleserial-smaug/    ← STM32F415 펌웨어
│   ├── makefile              ←   CW 빌드 시스템 (FPUUSE=1, SMAUG_LEVEL=1)
│   ├── simpleserial-smaug.c  ←   F/I/L/M/D/Z/X/T 명령
│   └── smaug_rng.c           ←   rng_get_random_blocking → CW HAL
├── lib/crypto_kem/           ← SMAUG-T 정적 아카이브 (Cortex-M4F hard-float)
│   ├── smaug{1,3,5}.a        ←   레벨별
│   └── hqc128/               ←   비교 baseline
├── include/                  ← SMAUG-T + HQC 헤더
├── host/
│   ├── capture.py            ← 트레이스 캡처 일반
│   ├── chosen_ct.py          ← chosen-CT 세션 (F/I/L wrapper)
│   ├── cw_serial.py          ← CW1173 시리얼 자동 선택
│   ├── upload.py             ← .hex 플래시
│   ├── smaug/                ← spec 시뮬레이터
│   │   ├── params.py         ←   smaug{1,3,5} 파라미터
│   │   ├── codec.py          ←   Compress/Decompress, pack/unpack
│   │   ├── ciphertext.py     ←   672B serialize
│   │   ├── chosen.py         ←   chosen-CT 빌더 + predict_mu_prime
│   │   └── sk_partition.py   ←   partition table + oracle pair (1-term + 2-term)
│   └── analysis/             ← trace 분석 라이브러리
│       ├── io.py             ←   Capture .npz 로드/저장
│       ├── group.py          ←   응답/라벨 분할
│       ├── tvla.py           ←   Welch-t TVLA
│       ├── sparse_recover.py ←   HW=HS MAP recovery (paper 핵심)
│       ├── validate.py       ←   캡처 sanity
│       └── viz.py            ←   plot helper
├── scripts/                  ← MAIN reproducer (4 files)
│   ├── smoke.sh              ←   preflight + build + flash + smoke capture
│   ├── run_attack.py         ←   MAIN capture (4 chosen-CT × N)
│   ├── run_h_attack.py       ←   Phase H multi-term capture (검증)
│   ├── attack_direct_poi.py  ←   single-seed 분석
│   └── analyze_multi_seed.py ←   ★ multi-seed (paper main result generator)
├── tests/                    ← 64 unit tests (수학/분석 정합성)
├── results/                  ← 분석 결과 (gitignored)
├── traces/                   ← 캡처 .npz (gitignored)
└── history/                  ← 발견 과정 + negative results (paper evidence trail)
    ├── README.md             ←   각 파일의 paper-claim mapping
    ├── scripts/              ←   12 files (Step 1-5 + UTIL + DEBUG)
    ├── host_analysis/        ←   2 files (e3c, labelmix dev tools)
    └── tests/                ←   1 file (test_labelmix)
```

## Paper main flow (4 scripts)

| 파일 | Paper Section | 역할 |
|---|---|---|
| `scripts/run_attack.py` | §4 Method overview | 4 chosen-CT × N capture (메인 capture) |
| `scripts/run_h_attack.py` | §3.4 Multi-term + R^2 | multi-term chosen-CT capture (검증) |
| `scripts/attack_direct_poi.py` | §5 Direct attack PoI | single-seed analysis |
| `scripts/analyze_multi_seed.py` | §6-7 Component-specific PoI + N curve | **paper main result** generator |

발견 과정 코드 (`history/`) 는 paper 의 *evidence trail* — 각 step 이 paper 의
어느 negative result / 측정값과 연결되는지 `history/README.md` 에 정리.

## Setup

### 하드웨어
- ChipWhisperer-Lite (CW1173, NewAE VID 0x2b3e PID 0xace2)
- CW308 UFO + CW308T-STM32F4 도터보드 (STM32F415RGTx)
- 20-pin 리본 + USB

### 호스트 의존성
```
arm-none-eabi-gcc 10.x+
Python 3.10+, chipwhisperer 6.0.0, numpy ≥ 1.26
GNU make
ChipWhisperer firmware tree (외부 클론, makefile 의 CW_FW_PATH)
```

### 빌드
```bash
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4 SMAUG_LEVEL=1
# 산출물: simpleserial-smaug-CW308_STM32F4.{elf,hex,bin}
# ROM ≈ 33 KB / RAM ≈ 7 KB (smaug1)
```

### 플래시
```bash
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
```

## Tests

```bash
python3 tests/run_all.py
# === 64/64 OK ===
```

64 unit tests 가 *수학/분석 정합성* 검증:
- `test_codec.py`: Compress/Decompress 비트 트릭 ↔ ref 일치
- `test_chosen.py`: chosen-CT 빌더 + `predict_mu_prime` round-trip
- `test_partition.py`: partition table (1-term + 2-term) + oracle pair 검증
- `test_sparse_recover.py`: greedy MAP + posterior + accuracy synthetic curve
- `test_chosen_ct_pke.py`: PKE-labeled chosen-CT session

## SimpleSerial 명령 셋 (firmware)

| 명령 | 페이로드 | 트리거 | 응답 | 역할 |
|---|---|---|---|---|
| `k` | 0 | OFF | 16B pk[0:16] | KEM keypair |
| `e` | 0 | OFF | 16B ct[0:16] | KEM encaps |
| `d` | 0 | ON | 1B mismatch | KEM decaps |
| `p` | 0 | dec | 1B mismatch | k → e → d 파이프라인 |
| `F` | 0 | OFF | 16B sha3(pk) | persistent keygen |
| `I` | 33B | OFF | 1B status | ct chunk inject (32B × 21) |
| `L` | 0 | OFF | 16B sha3(ct) | inject load done |
| `M` | 32B μ | OFF | 16B sha3(ct) | indcpa_enc(ct, pk, μ, seed=0×32) |
| `D` | 0 | ON | 1B mismatch | crypto_kem_dec(ss, ct_inj, sk) |
| `Z` | 0 | ON | **32B µ′** | indcpa_dec only (paper main) |
| `X` | 1B chunk_idx | OFF | 32B sk bytes | sk PKE dump (검증용 only) |
| `T` | 5B (comp+idx+α) | ON | 32B out | isolated poly_mul_acc — paper §8.2 negative |

## Troubleshooting

자세한 setup 이슈 (LD VFP mismatch, multiple PQCLEAN_randombytes, USB
ChipWhisperer 못 찾음, F415 가 아닌 보드만 꽂혀 있음, scope.io.nrst
boot timing 등) 는 `docs/HANDOFF.md` §"트러블슈팅 / 알려진 이슈" 참조.

## 참고

- SMAUG-T v4.0: KpqC Round 2 (2025), `cryptolab` 네임스페이스 PQM4 port
- CW HAL tree: `chipwhisperer/firmware/mcu/hal/chipwhisperer-fw-extra/stm32f4/`
- SimpleSerial v2.1: `chipwhisperer/firmware/mcu/simpleserial/`
