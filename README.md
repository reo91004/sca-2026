# Real-Trace Single-Stage SCA on SMAUG-T smaug1 (8 traces, 100%)

> Paper draft: `docs/paper_draft.md` · Countermeasures analysis: `docs/countermeasures.md`

KpqC 2025 우승 KEM **SMAUG-T smaug1** 의 long-term secret key 를
**4 chosen-CT × N=2 = 8 power traces (~3.2 sec capture)** 로 100% 복구하는
side-channel attack. ChipWhisperer-Lite (CW1173) + Vcc-shunt + STM32F415
unmasked reference 구현, **no offline templates**, no EM probe.

| 평가 metric | 값 |
|---|---|
| Trace count | **8** (= 4 chosen-CT × N=2) |
| Capture time | **~3.2 seconds** |
| Keypairs evaluated | **9** (random) |
| Full sk recovery | **100% ± 0%** (9/9) |
| Setup | CW-Lite + Vcc shunt + STM32F415 |
| Method | Direct attack PoI + component-specific PoI + sparse_recover |

## 빠른 reproduce

```bash
# 1. 펌웨어 빌드 + 플래시 (한 번)
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4 SMAUG_LEVEL=1
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex

# 2. Multi-seed 캡처 (5 keypairs × N=128, ≈25분)
for s in 1 2 3 4 5; do
    python3 scripts/run_attack.py -n 128 --out traces/attack_seed${s}.npz
done

# 3. 분석 — paper main result generator
python3 scripts/analyze_multi_seed.py traces/attack_seed{1..5}.npz
# → "full_sk_acc: mean=1.000, std=0.000, min=1.000, max=1.000" 출력

# 4. 또는 minimum cost (8 traces, 3.2 sec) 시연
python3 scripts/run_attack.py -n 2 --out traces/attack_quick.npz
python3 scripts/analyze_multi_seed.py traces/attack_quick.npz
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
├── tests/                    ← 64 unit tests (수학 정합성)
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
# === 63/63 OK ===
```

64 unit tests 가 *수학 정합성* 검증:
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
