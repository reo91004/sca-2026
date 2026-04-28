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
├── lib/crypto_kem/                  ← SMAUG-T 정적 아카이브 (Cortex-M4F, hard-float)
│   ├── smaug1.a                     ←   레벨 1 (LWE_N=256, MODULE_RANK=2)
│   ├── smaug3.a                     ←   레벨 3
│   └── smaug5.a                     ←   레벨 5
├── include/
│   ├── crypto_kem/smaug{1,3,5}/*.h  ← 레벨별 헤더 (api.h, parameters.h, …)
│   └── randombytes.h                ← kem.h 가 #include <randombytes.h> 함
├── firmware/simpleserial-smaug/
│   ├── makefile                     ← CW 펌웨어 빌드 시스템 (FPUUSE=1, SMAUG_LEVEL=…)
│   ├── simpleserial-smaug.c         ← main + k/e/d/p 콜백, trigger_high()/_low()
│   └── smaug_rng.c                  ← rng_get_random_blocking → CW HAL get_rand()
└── host/
    ├── cw_serial.py                 ← CW1173 시리얼 자동 선택 정책 (공통)
    ├── upload.py                    ← STM32 부트로더 경유 플래시
    └── capture.py                   ← 부채널 트레이스 캡처 루프
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
| `-DSS_VER_2_1=3 -DSS_VER=SS_VER_2_1` | SimpleSerial v2.1 | 호스트 `cw.targets.SimpleSerial2` 와 짝 |

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
| `--baud` | 230400 | SimpleSerial UART 보레이트 |

캡처 결과 `.npz` 는 다음 키를 갖는다:

```python
import numpy as np
d = np.load("traces/smaug1_dec.npz", allow_pickle=True)
d["traces"].shape       # (N, samples), float32
d["responses"].shape    # (N,), uint8 — dec 결과 mismatch flag
d["meta"].item()        # dict: captured_at, scope_sn, samples, gain_db, cmd, baud, …
```

## SimpleSerial 명령 셋

펌웨어는 SS v2.1 로 동작하며 4 개의 명령을 노출한다. 모두 페이로드 0 바이트.

| 코드 | 동작 | 트리거 | 응답 (`r`) |
|---|---|---|---|
| `k` | `crypto_kem_keypair(pk, sk)` | OFF | `pk[0:16]` (지문) |
| `e` | `crypto_kem_enc(ct, ss_enc, pk)` | OFF | `ct[0:16]` |
| `d` | `crypto_kem_dec(ss_dec, ct, sk)` | **ON** (high → dec → low) | 1 B `mismatch` (0 = ss_dec == ss_enc) |
| `p` | keypair → enc → (trigger high) dec (trigger low) — 한 방 파이프라인 | dec 구간에만 ON | 1 B `mismatch` |

전부 보드에 상주하는 정적 버퍼 (`pk` 672 B, `sk` 832 B, `ct` 672 B,
`ss_*` 32 B) 위에서 동작한다. SimpleSerial 의 64 B/프레임 한도 때문에
호스트가 pk/ct 를 통째로 받아오지 않는다.

> chosen-CT 공격이 필요해지면, 추가로 'i' (inject 64 B chunk of CT) 와
> 'D' (decaps with already-injected CT, fixed sk) 같은 명령을 더 붙이면 된다.
> 현재 펌웨어는 KEM 동작 자체와 dec-구간 트레이스 수집의 베이스라인을
> 검증하기 위한 최소 셋이다.

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
