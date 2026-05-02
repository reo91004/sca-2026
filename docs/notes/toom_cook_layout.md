# SMAUG-T smaug1 Toom-Cook 4-way layout (F1 분석, 2026-05-03)

`arm-none-eabi-objdump -d lib/crypto_kem/smaug1.a` 위에서 추출한 함수 layout.

## Call hierarchy (indcpa_dec → poly_mul_acc → toom_cook_4way → karatsuba_simple)

```
indcpa_dec (0x29e, 295 inst, 4096B stack)
  ├── matrix/vec helpers
  └── ... (ct decompress, sk multiplication via vec ops)

vec_vec_mult_add (0xdc, 1564B stack)        ← R^2 vector inner product
  └── poly_mul_acc (1 call per component)

matrix_vec_mult_add (0x220, 1056B stack)    ← matrix × vector
  └── poly_mul_acc (multiple calls)

poly_mul_acc (0xf42, 74 inst, 1048B stack)  ← entry to Toom-Cook
  └── toom_cook_4way (1 call)

toom_cook_4way (0x712, 651 inst, 3672B stack)
  └── karatsuba_simple × 7 (Toom-Cook-4 의 evaluation points)

karatsuba_simple (0x000, 601 inst, 9 muls, 368B stack)  ← 64×64 sub-mul
  └── memset + recursive school-book mul (9 explicit muls — 추정 base-case
      분기 포함 더 큰 work)
```

## Cycle estimate (Cortex-M4, single-cycle MUL, 1.5 cycle avg per inst)

| 함수 | inst | cycle | sample @ clkgen_x4 (29.5 MS/s) |
|---|---:|---:|---:|
| karatsuba_simple (1회) | 601 | ~1500 | ~6000 |
| toom_cook_4way (자체) | 651 | ~1000 | ~4000 |
| toom_cook_4way (전체) | 5808 | ~11500 | ~46000 |
| poly_mul_acc | 5882 | ~11600 | ~46400 |
| indcpa_dec (전체) | ~5000 | ~50000 | ~200000 |

ADC 윈도우 한도: 24400 samples (CW-Lite).

## 캡처 mode 옵션

| mode | clkgen | sample/cycle | 24400 sample 영역 (cycles) | poly_mul_acc 한 호출 잡힘? |
|---|---|---:|---:|---|
| A (현재 default) | x4 (29.5 MS/s) | 4 | 6100 cycles | ❌ (절반 정도) |
| B | x2 (14.77 MS/s) | 2 | 12200 cycles | ✓ (margin 있음) |
| C | x1 (7.38 MS/s) | 1 | 24400 cycles | ✓ (큰 margin) |

**결정**: F2 에서 mode A 우선 (기존 setup 과 호환, 4× oversampling 의 SCA 이점).
poly_mul_acc 의 *첫 절반* (≈ toom_cook 의 evaluation + 처음 3-4× karatsuba) 만 잡음.
필요시 mode B 로 확장 → 전체 한 호출 분석.

## Sub-trigger 디자인 (F2)

Approach 1 (선택 — 기존 인프라 재사용):
- 새 명령 'T' = host 가 sparse host_b (예: alpha at one position) 를 보내고
  firmware 가 trigger_high → poly_mul_acc(out, sk[0], host_b) → trigger_low.
- HW template 학습용 isolated mul (chosen-CT 와 같은 sparse input pattern).

Approach 2 (확장 — F7 단계에서):
- 'Z' 명령 그대로 사용 (전체 indcpa_dec) + 분석 측에서 trace 의 어느 영역이
  poly_mul_acc 호출인지 매핑 (이번 F1 의 cycle layout 활용).

F2 는 Approach 1 으로 시작.
