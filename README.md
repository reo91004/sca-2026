# sca-2026 — SMAUG-T 부채널 분석 환경 (CW1173 + STM32F415)

ChipWhisperer-Lite (CW1173) + CW308T-STM32F4 (STM32F415RGTx) 위에서 한국 표준
PQC KEM 후보 **SMAUG-T** 의 `crypto_kem_dec` 에 대해 power-trace 기반
사이드채널 분석을 수행하기 위한 최소 빌드/캡처 환경.

* SMAUG-T 핵심 코드는 PQM4 설계의 미리 빌드된 정적 아카이브
  `lib/crypto_kem/smaug{1,3,5}.a` 로 제공되며, 그 외부 의존은
  `memcpy`, `memset`, `rng_get_random_blocking` 단 세 개뿐이다.
* `firmware/simpleserial-smaug/` 가 이 아카이브를 ChipWhisperer 의 SimpleSerial
  v2.1 펌웨어 빌드 시스템에 그대로 끼워 넣어 STM32F415용 `.hex` / `.bin` 을
  생성한다.
* `host/upload.py` 가 빌드 산출물을 CW-Lite 부트로더 경유로 플래시하고,
  `host/capture.py` 가 캡처 루프를 돌려 `.npz` 트레이스를 저장한다.

## 디렉토리

```
sca-2026/
├── docs/                              ← 설계/실험/세션 인계 문서
│   ├── idea.md                        ←   공격 가설 (chosen-CT message recovery)
│   ├── experiments.md                 ←   E0–E5 실험 매트릭스
│   ├── HANDOFF.md                     ←   세션 인계용 살아있는 메모
│   └── paper/2020-912.pdf             ←   참고 논문 (Kyber chosen-CT SCA)
├── lib/crypto_kem/                    ← SMAUG-T 정적 아카이브 (Cortex-M4F, hard-float)
│   ├── smaug1.a                       ←   레벨 1 (LWE_N=256, MODULE_RANK=2)
│   ├── smaug3.a                       ←   레벨 3
│   ├── smaug5.a                       ←   레벨 5
│   └── hqc128/                        ←   비교 baseline (HQC PQClean)
├── include/
│   ├── crypto_kem/smaug{1,3,5}/*.h    ← 레벨별 헤더 (api.h, parameters.h, …)
│   ├── crypto_kem/hqc128/*.h          ← HQC 헤더
│   └── randombytes.h                  ← kem.h 가 #include <randombytes.h> 함
├── firmware/
│   ├── simpleserial-smaug/            ← SMAUG-T 타겟 펌웨어 (메인)
│   │   ├── makefile                   ←   CW 펌웨어 빌드 시스템 (FPUUSE=1, SMAUG_LEVEL=…)
│   │   ├── simpleserial-smaug.c       ←   main + k/e/d/p 콜백, trigger_high()/_low()
│   │   └── smaug_rng.c                ←   rng_get_random_blocking → CW HAL get_rand()
│   └── simpleserial-hqc/              ← HQC baseline 펌웨어 (옵션)
├── host/                              ← PC 측 도구
│   ├── cw_serial.py                   ←   CW1173 시리얼 자동 선택 정책 (공통)
│   ├── upload.py                      ←   STM32 부트로더 경유 플래시
│   ├── capture.py                     ←   부채널 트레이스 캡처 루프
│   └── analysis/                      ←   오프라인 분석 패키지 (numpy/matplotlib)
│       ├── io.py                      ←     .npz 로드/저장 (Capture 객체)
│       ├── group.py                   ←     클래스 분할 (response/label/fixed-vs-random)
│       ├── tvla.py                    ←     Welch-t TVLA
│       ├── validate.py                ←     캡처 sanity 검증
│       └── viz.py                     ←     PNG 산출 (overview, TVLA)
├── scripts/                           ← 실행 진입점
│   ├── smoke.sh                       ←   preflight → build → flash → capture → viz
│   ├── plot_overview.py               ←   .npz → overview PNG (E0/E1)
│   └── plot_tvla.py                   ←   두 .npz → Welch-t PNG (E2/E3)
├── traces/                            ← 캡처 .npz (gitignored)
└── results/                           ← 분석 PNG 산출물 (gitignored)
```

## 준비물

### 하드웨어
* ChipWhisperer-Lite (CW1173, NewAE VID 0x2b3e PID 0xace2)
* CW308 UFO 베이스보드 + CW308T-STM32F4 도터보드 (STM32F415RGTx)
* CW1173 ↔ CW308 20-pin 리본 케이블, USB

### 소프트웨어 (호스트)
* arm-none-eabi-gcc 10.x 이상 (`apt install gcc-arm-none-eabi`)
* Python 3.10+, `chipwhisperer 6.0.0`, `numpy ≥ 1.26`
* GNU make
* ChipWhisperer 펌웨어 트리 — 외부 경로 (기본값:
  `/home/pacl/Documents/Repository/chipwhisperer/firmware/mcu`).
  다른 경로면 makefile 의 `CW_FW_PATH` 또는 환경변수로 덮어쓴다.

### 빌드시 핵심 컴파일러 옵션
| 옵션 | 값 | 이유 |
|---|---|---|
| `-mcpu=cortex-m4` | F415 코어 | CW HAL 디폴트 |
| `-mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16` | hard-float ABI | `smaug{1,3,5}.a` 가 hard-float 로 빌드되어 있음 (`FPUUSE=1` 필수) |
| `-DSTM32F415RGTx -DSTM32F415xx` | F415 | CW HAL Makefile 이 자동 부여 |
| `-DSS_VER_1_1=2 -DSS_VER=SS_VER_1_1` | SimpleSerial v1.1 | 호스트 `cw.targets.SimpleSerial` 와 짝, baud=38400 |

## 빌드

```bash
cd firmware/simpleserial-smaug

# 레벨 1 (기본)
make PLATFORM=CW308_STM32F4

# 레벨 3 / 5
make PLATFORM=CW308_STM32F4 SMAUG_LEVEL=3
make PLATFORM=CW308_STM32F4 SMAUG_LEVEL=5

# 외부 chipwhisperer 트리 경로가 다르면
make PLATFORM=CW308_STM32F4 CW_FW_PATH=/path/to/chipwhisperer/firmware/mcu
```

산출물 (예: 레벨 1):
```
simpleserial-smaug-CW308_STM32F4.elf   ← 디버그 / nm용
simpleserial-smaug-CW308_STM32F4.hex   ← 부트로더용
simpleserial-smaug-CW308_STM32F4.bin   ← 0x08000000 로드, 부트로더 / 디버거용
```

레벨 1 기준 점유: ROM 32 828 B / RAM 4 912 B (각 3 % 미만).

## 플래시

```bash
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
```

`.hex` 만 받는다 (`make` 가 동시에 만드는 산출물). flow:
`cw.scope().default_setup()` → `STM32FProgrammer.open() → find() → erase() → program()`.

> `.bin` 직접 플래시는 지원하지 않음. chipwhisperer 6.0.0 의
> `IntelHex.loadbin/write_hex_file` 가 Py3 와 호환되지 않아 `.bin → 임시 .hex`
> 변환 경로가 깨져 있다. `.hex` 경로는 native 그대로 잘 작동한다.

## 트레이스 캡처

```bash
mkdir -p traces
python3 host/capture.py -n 1000 -s 24400 -o traces/smaug1_dec.npz
```

옵션:

| 인자 | 기본 | 설명 |
|---|---|---|
| `-n / --num-traces` | 100 | 수집할 트레이스 개수 |
| `-s / --samples` | 24400 | 트레이스당 ADC 샘플 (CW-Lite 최대 24400) |
| `-g / --gain-db` | 25.0 | LNA 게인 |
| `-c / --cmd` | `p` | `p`=full pipeline / `d`=dec only (사전 keypair/encaps 필요) |
| `-o / --output` | (필수) | 저장할 `.npz` 경로 |
| `--baud` | 38400 | SimpleSerial UART 보레이트 (SS_VER_1_1 표준) |
| `--target` | smaug | SMAUG / HQC 캡처 분기 (메타에 기록) |
| `--send-len` | 0 | 명령 페이로드 바이트 수 |
| `--resp-len` | 1 | 응답 `r` 바이트 수 |

캡처 결과 `.npz` 는 다음 키를 갖는다:

```python
import numpy as np
d = np.load("traces/smaug1_dec.npz", allow_pickle=True)
d["traces"].shape       # (N, samples), float32
d["responses"].shape    # (N, R), uint8 — SimpleSerial ack 페이로드
d["meta"].item()        # dict: captured_at, scope_sn, samples, gain_db, cmd, baud, …
```

권장: 직접 `np.load` 보다 `host.analysis.io.load_capture(path)` 를 쓰면
형식 검증 + sanity 점검을 한 번에 한다 (자세한 건 §"분석 / 시각화").

## 스모크 테스트

`scripts/smoke.sh` 가 preflight → build → flash → capture → validate → viz
까지 한 번에 돈다. 환경 변수로 동작 토글:

```bash
scripts/smoke.sh                                   # SMAUG-T level 1 (기본)
LEVEL=3 scripts/smoke.sh                           # SMAUG level 3
NUM_TRACES=1000 SAMPLES=24400 scripts/smoke.sh     # 1k trace
SKIP_BUILD=1 SKIP_FLASH=1 scripts/smoke.sh         # capture + viz only
SKIP_CAPTURE=1 NPZ=traces/foo.npz scripts/smoke.sh # 기존 .npz 재시각화
TARGET=hqc-custom scripts/smoke.sh                 # HQC baseline 비교용
```

## 분석 / 시각화

오프라인 분석은 `host/analysis/` 패키지가 SSOT (단일 진실의 원천). 진입점
스크립트 두 개가 있고, `scripts/smoke.sh` 도 내부적으로 같은 모듈을 부른다.

```bash
# E0/E1: 단일 캡처 평균 ± σ + 무작위 오버레이 PNG
scripts/plot_overview.py traces/smoke_smaug1.npz \
    --png results/E1_overview_smaug1.png

# E2: 두 캡처 간 Welch-t TVLA (fixed-vs-random 또는 fixed-vs-fixed)
scripts/plot_tvla.py traces/E2_fixed_mu0.npz traces/E2_random_mu.npz \
    --png results/E2_tvla_fixed_vs_random.png

# 단일 .npz 를 응답 바이트로 분기 (SMAUG mismatch flag 등)
scripts/plot_tvla.py traces/X.npz --split-response-byte 0 --split-value 0 \
    --png results/E2_tvla_X_split.png
```

분석 모듈을 직접 import 해서 쓸 수도 있다:

```python
from host.analysis import io, tvla, viz, group

cap_a = io.load_capture("traces/E3_mu0.npz")
cap_b = io.load_capture("traces/E3_mu1.npz")
result = tvla.welch_t(cap_a.traces, cap_b.traces)
print(result.summary())                       # max|t|, leaky 시간점 수
viz.plot_tvla(result, "results/E3_tvla.png")
```

실험 매트릭스(E0–E5) 와 합격 기준은 `docs/experiments.md`. 작업 인계는
`docs/HANDOFF.md` 를 참고/갱신.

## SimpleSerial 명령 셋

현재 펌웨어는 SS v1.1 로 동작하며 두 그룹의 명령을 노출한다.

**(A) baseline / smoke** — 페이로드 0 바이트.

| 코드 | 동작 | 트리거 | 응답 (`r`) |
|---|---|---|---|
| `k` | `crypto_kem_keypair(pk, sk)` | OFF | `pk[0:16]` (지문) |
| `e` | `crypto_kem_enc(ct, ss_enc, pk)` | OFF | `ct[0:16]` |
| `d` | `crypto_kem_dec(ss_dec, ct, sk)` | **ON** (high → dec → low) | 1 B `mismatch` (0 = ss_dec == ss_enc) |
| `p` | keypair → enc → (trigger high) dec (trigger low) — 한 방 파이프라인 | dec 구간에만 ON | 1 B `mismatch` |

**(B) chosen-ciphertext** — `host/chosen_ct.py setup` 가 자동으로 묶어 호출.

| 코드 | 동작 | 페이로드 | 트리거 | 응답 (`r`) |
|---|---|---|---|---|
| `F` | fresh keypair + sk 영속, ct_inj/ss_enc 0-init | 0 B | OFF | 16 B `sha3_256(pk)[:16]` |
| `I` | ct_inj 의 [idx·32, idx·32+32) 에 32 B 주입 | 33 B = `[idx 1 B][data 32 B]` | OFF | 1 B status (0=OK / 1=idx OOR / 2=len OOR) |
| `L` | 무결성 지문 (host 가 host-side `sha3_256(ct)[:16]` 와 비교) | 0 B | OFF | 16 B `sha3_256(ct_inj)[:16]` |
| `M` | `indcpa_enc(ct_inj, pk, μ, seed=0×32)` PKE-labeled valid ct | 32 B = μ (seed 펌웨어 zero-fixed) | OFF | 16 B `sha3_256(ct_inj)[:16]` |
| `D` | `crypto_kem_dec(ss, ct_inj, sk)` 영속 sk 사용 | 0 B | **ON** (high → dec → low) | 1 B `mismatch` (vs ss_enc) |

전부 보드에 상주하는 정적 버퍼 (`pk` 672 B, `sk` 832 B, `ct` 672 B,
`ct_inj` 672 B, `ss_*` 32 B) 위에서 동작한다. SS v1.1 페이로드 한도가
< 64 B (`addcmd len >= 64 reject`) 라 ct 672 B 는 'I' 명령 21 회로 나눠
주입하고 (chunk = 32 B), 'M' 도 μ-only 32 B 로 한도 안에 들어간다.

`host/chosen_ct.py` 가 'F'/'I'/'L' 시퀀스를 한 함수 (`setup_session`) 로
묶어 무결성 검증까지 한다. `host/smaug/` 패키지가 host 측 ct 를
spec-정확한 byte 로 만들어준다 (Compress/Decompress, pack/unpack, chunkify).

> **주의 — boot timing**: `scope.default_setup()` 직후 즉시 SimpleSerial
> write 를 시도하면 보드가 ready 상태가 아니라 응답이 안 온다.
> `host/capture.py::setup_scope` 와 `host/chosen_ct.py::reset_target`
> 가 `scope.io.nrst` 를 toggle 후 0.5 s 기다리는 형태로 보호한다 — 직접
> `cw.scope()` 를 잡는다면 같은 reset 시퀀스를 재현해야 한다.

> **주의 — SS_VER_1_1 페이로드 한도**: simpleserial.c (chipwhisperer
> firmware tree) 의 `simpleserial_addcmd` 가 `if (len >= MAX_SS_LEN=64)
> reject` 라 정확히 64 B 페이로드도 거부 (실측). 따라서 v1.1 모드 최대
> payload 는 **63 byte**. SMAUG-T 의 'M' 명령이 μ(32) ‖ seed(32) = 64
> 였는데 등록 거부 → 응답 없음. 현재 'M' 은 μ 32 B 만 받고 seed 는 펌웨어
> 에서 `0×32` 로 fixed (host 가 seed 도 통제하려면 별도 'N' 명령 추가).

## CW 보드 자동 선택 정책 (`host/cw_serial.py`)

이 사용자의 벤치에는 동시에 최대 두 대의 CW1173 이 꽂혀 있을 수 있고, 그
중 한 대는 F415 가 아닌 다른 DUT 가 매달려 있다. 따라서:

* `TARGET_SN = 5020 3220 3130 3854 3030 3332 3732 3039` → F415 (사용 대상)
* `FORBIDDEN_SN = 5020 3220 594a 4830 3330 3731 3332 3037` → 다른 DUT (절대 금지)

`pick_serial(requested)` 의 결정 규칙:

1. `--serial` 가 `FORBIDDEN_SN` 이면 무조건 거부.
2. `--serial` 가 명시되면 그것을 사용 (연결되어 있다는 전제).
3. 자동 (인자 없음):
   * `TARGET_SN` 이 연결되어 있으면 그것을 사용.
   * 그렇지 않고 forbidden 을 제외한 보드가 정확히 1개면 그것을 자동 선택
     (벤치 한 대만 있는 환경의 폴백). `[INFO]` 로 알린다.
   * 0 개면 종료.
   * 2 개 이상이면 모호하므로 `--serial` 명시 요구.

이 정책은 `upload.py`, `capture.py` 모두 공유한다. `cw_serial.py` 단위 검증은
스크립트가 자동 mock 으로 10 개 시나리오를 모두 통과하는 것을 확인했다.

## 트러블슈팅 / 알려진 이슈

### `LD: ... uses VFP register arguments, ... does not`
hard-float 와 soft-float 객체를 섞은 것이다. makefile 의 `FPUUSE = 1` 이
지워지지 않았는지 확인하라. SMAUG-T 아카이브는 hard-float 으로 빌드되어 있다.

### `multiple definition of PQCLEAN_randombytes`
`smaug{1,3,5}.a` 안의 `randombytes.c.o` 가 `PQCLEAN_randombytes` 를 정의한다.
PQM4 의 `libpqm4hal.a` 도 같은 심볼을 정의하므로, **두 라이브러리를 동시에
링크하면 안 된다.** 본 환경은 PQM4 HAL 을 빼고 CW HAL 만 사용한다.

### `undefined reference to rng_get_random_blocking`
`smaug_rng.c` 가 `SRC` 에 들어 있는지 확인하라. 이 함수는 SMAUG-T 의
`randombytes.c.o` 가 호출하는 단일 외부 의존이며, 본 환경은 CW HAL 의
`get_rand()` (= `HAL_RNG_GenerateRandomNumber` 래퍼) 로 라우팅한다.

### `ChipWhisperer 보드를 못 찾음`
* `lsusb | grep 2b3e:ace2` 로 USB 인식 확인.
* 다른 프로세스 (Jupyter, 다른 캡처 스크립트) 가 점유 중인지 확인.
* `udev` 규칙이 빠져 있어 `/dev/bus/usb/...` 가 read-only 면 `newae.rules`
  설치가 필요할 수 있다 (ChipWhisperer 공식 인스톨 가이드 참조).

### F415 가 아닌 보드만 꽂혀 있음 → `[ABORT]`
의도된 동작이다. F415 펌웨어를 다른 DUT 에 올리면 페리페럴 매핑이 어긋나
브릭/오작동을 유발할 수 있어 정책적으로 막는다. 진짜로 진행하려면 보드를
바꿔 꽂거나 `host/cw_serial.py` 의 상수를 갱신하라.

## 참고

* SMAUG-T 원천: KpqC Round 2 (2025) — `cryptolab` 네임스페이스로 빌드된 PQM4
  포트.
* CW HAL 트리: `chipwhisperer/firmware/mcu/hal/chipwhisperer-fw-extra/stm32f4/`
  (외부 의존, 사용자가 별도 클론).
* SimpleSerial v2.1 사양: `chipwhisperer/firmware/mcu/simpleserial/`.
