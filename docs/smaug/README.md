# SMAUG-T 트랙

Chosen-CT SCA on SMAUG-T smaug{1,3,5} (k ∈ {2,3,4}) via ChipWhisperer-Lite +
CW308T-STM32F415.

## 진입점

처음 읽는다면:

1. **[FLOW.md](FLOW.md)** — 트랙 전체 timeline (problem → experiments → results, branch 별 1~2 줄). 10~15 분.
2. **[IDEA.md](IDEA.md)** — 메인 누적 로그. S1/S2/S2.5~S2.13/S3/S1U/S4/D-Pair phase 가 한 파일.

각 branch 의 자세한 데이터는 옆 `.md` 들에 있다:

| 파일 | branch |
|---|---|
| `c2_threshold_pairs.md` | C2 — paired threshold tomography (c1, c2 pair → mu_delta) |
| `r_mu_roundpack.md` | R — µ′ round/pack leakage |
| `fo_downstream.md` | Y — FO downstream replay |
| `d_pair_oracle.md` | D-Pair — trace-only µ′-change oracle on `crypto_kem_dec` |
| `s5_trace_scaling.md` | S5 — trace-count scaling / detection-limit audit |
| `notes/toom_cook_layout.md` | Toom/Karatsuba state mapping reference |

## 문서 역할

| 파일 | 역할 | 갱신 빈도 |
|---|---|---|
| `FLOW.md` | navigational summary (branch timeline 한 표) | minor — branch 종료 시점 |
| `IDEA.md` | 메인 phase 누적 로그 (S1/S2/S2.5~S2.13/S3/S1U/S4/D-Pair) | high (active 기간), low (현재 종료) |
| branch `.md` | 각 branch 의 가설·실험·수치·결정 | branch 단위 갱신 |

## 코드 위치

| 경로 | 설명 |
|---|---|
| `scripts/smaug/s1_*` | S1 — `T` poly_mul_acc diagnostic |
| `scripts/smaug/s1_u_*` | S1U — multi-term isolated poly_mul_acc |
| `scripts/smaug/s2_z_*` | S2 — `Z` chosen-CT indcpa_dec (matrix, low-dim, multi-term, scored designs 모두 포함) |
| `scripts/smaug/s3_v_*` | S3 — `V` vec_vec_mult_add |
| `scripts/smaug/s4_*` | S4 — c2 pair, mu recovery pressure, pair distance oracle |
| `scripts/smaug/smoke.sh` | shell driver |
| `host/smaug/` | host-side ciphertext/codec/poly_mul/sk_partition 패키지 |
| `host/chosen_ct.py`, `host/capture.py`, `host/upload.py` | SMAUG-T 전용 시리얼/캡처/업로더 |
| `host/cw_serial.py` | ChipWhisperer serial picker (양 트랙 공유) |
| `host/analysis/` | TVLA / sparse-recover 분석 라이브러리 |
| `firmware/simpleserial-smaug/` | STM32F415 펌웨어 |
| `tests/smaug/` | round-trip + analyzer 정합성 테스트 |
| `traces/<...>` | 캡처 산출물 (git-ignored) |
| `results/smaug/<...>` | 분석 산출물 (git-ignored) |

## 종료 상태 (2026-05-09)

표준 chosen-CT trace-based SCA 는 **cross-key sk 복구 불가**. PCO oracle 은
sk-universal 이지만 µ′-bit 직접 leak 없음 — 1-bit PCO 채널로만 sk recovery
가능. 자세한 결과/정정은 [FLOW.md](FLOW.md) 와 메모리
`corrigendum_2026_05_04.md` 참조.
