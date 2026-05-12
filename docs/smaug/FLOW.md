# SMAUG-T 트랙 — Problem · Experiments · Results Flow

이 문서는 SMAUG-T chosen-CT SCA 라인을 "어떤 문제 → 어떤 실험 → 어떤
결과" 흐름으로 한 눈에 보여주는 navigational summary 다. 한 entry 당
1~2 줄의 결정/결과만 담고, 각 branch 의 자세한 가정·수치는 source-
of-truth 문서들 (`IDEA.md`, branch 별 `.md`) 에 있다.

타겟: SMAUG-T smaug{1,3,5} (k ∈ {2,3,4}) · ChipWhisperer-Lite +
CW308T-STM32F415 · pqm4-style C reference.

상태 요약 (2026-05-09): **chosen-CT trace-only SCA 로 cross-key sk
복구 X**. PCO oracle 은 sk-universal (cross-sk 99% transfer) 이지만
µ′-bit 직접 leak 없음 — 표준 1-bit PCO 채널로만 가능.

## 트랙 전체 흐름

| Branch / Phase | 문제 / 가정 | 실험 | 핵심 결과 |
|---|---|---|---|
| S1 — `T` poly_mul_acc | diagnostic. c=0, j=0, α=1 한 환경에서 sk-dependent leak 이 존재하나? | per-sk capture + cross-sk Welch-t + per-coord HW CPA + LOO recover | cross-sk t 값 있음. 단일 coord HW 만으로는 LOO sk 복구 X. |
| S2 — `Z` indcpa_dec | chosen-CT 가 실제 attack-valid signal 을 주는가? | per-sk capture + cross-sk Welch-t | per-bit `|t|` 가 N=1000 에서도 noise floor 4~5 에 갇힘. **CW-Lite per-bit FD 한계**. |
| S2.5 — multivariate / matrix / low-dim | 단일 byte HW 대신 multivariate 또는 low-dim label 로 SNR 개선? | matrix pilot, low-dim labels, scored designs | per-design score 개선되어도 cross-key wall 그대로. |
| S2.11 — Toom/Karatsuba label | disassembly 추적 후 sub-step 별 label 로 분리하면? | toom/karatsuba state labels | window scan 안에서 일부 score gain — 그러나 still cross-key fail. |
| S3 — `V` vec_vec_mult_add | 다른 함수에서 leak 가능? | V capture + low-dim Toom labels | 동일 wall. |
| S1U — multi-term `T` | 다중 항 chosen-CT 가 SNR 합산을 주는가? | multi-term isolated poly_mul_acc | partial signal 강화. 그러나 sk 복구 도달 X. |
| C2 — paired threshold (`c2_threshold_pairs.md`) | c1, c2 pair 로 mu_delta 의 임계값 추적 | c2 staircase + monomial pairs | flip_any α-bug 발견 (signed mu_delta 사용해야). pair 자체로는 sk leak 도달 X. |
| R — µ′ round/pack (`r_mu_roundpack.md`) | 채택된 µ′ bit 패턴이 round/pack 단계에서 leak 하는가? | chosen CT 로 latent µ′ 유도 + capture | leak 신호 검출됨 — 단 cross-key consistent 하지 못함. |
| Y — FO downstream (`fo_downstream.md`) | FO transform 의 re-encrypt 단계에 leak 가능? | downstream replay capture | partial leak — chosen-CT 결정 분리하지 못함. |
| D-Pair — trace-only µ′-change oracle (`d_pair_oracle.md`) | 동일 sk 에서 µ′ 차이 oracle 을 trace 만으로? | paired tomography on natural `D` | corr z ≤ +1.6 ceiling across early/late window, staircase, multi-term. **CW-Lite SNR wall 동일**. |
| S5 — trace scaling audit (`s5_trace_scaling.md`) | N 증가만으로 detection threshold 깰 수 있나? | N sweep + segment scan | N 5000 까지도 threshold 변화 미미. **scaling 만으로는 한계 못 깸**. |
| E1 / E3 — alignment (memory) | 정렬·DTW 가 cross-key fail 의 원인? | full pipeline TVLA after alignment | same-sk cross-session 100%, **cross-key 는 정렬 후도 random**. sk-dependent execution path 시사. firmware mismatch 가드 추가. |
| E5 — FFT / OT-PCA insight (memory) | trace-SCA 의 다른 representation 으로 cross-key 회복? | FFT magnitude TVLA + OT-PCA 검토 | FFT 도 cross-key fail. host_posterior_table 이 OT-PCA offline template 와 동치라 추가 이득 없음. |
| PCO Pilot 1-2 — late-trace leak | `D` 끝부분에서 chosen-CT 결정의 leak 가 보이나? | full crypto_kem_dec capture, accept vs reject | sample 22924 (94% 지점) max\|t\|=31.93. zero vs a192 single-trace 100%. **첫 cross-key-free signal**. 단 leak source = ct-byte operand power (µ′ 가 아님). |
| PCO Pilot 3 — accept/reject (`memory`) | trace 만으로 accept(valid) vs reject(chosen) 100% 분류? | 'd' vs 'D' classifier + ct-only control | late max\|t\|=33.8~34.3 vs ct-only ctrl 9.8 (3.5×). **표준 PCO 1-bit oracle 입증**. |
| PCO Pilot 4 — cross-sk transfer ★ | 다른 sk session 에 oracle 이 그대로 작동? | 4 sk × cyclic-shift designs | classifier cross-sk transfer 96.9~99.2%. **oracle = sk-universal**. PCO sk recovery 의 trace cost 추정 가능. |
| PCO Pilot 5 — z-leak vs µ′-leak | reject path 의 큰 \|t\| 가 µ′ 직접 leak 인가? | inter-sk reject trace | reject \|t\|=170~216 이지만 zero ≈ a192 (1.04). **z (rejection seed) 가 leak 우세**, sparse sx 직접 복구 X. |
| PCO Pilot 6 — µ′-bit cross-sk leak (`memory`) | 4 cyclic-shift designs 에서 µ′-bit cross-sk leak? | 4 sk × 4 designs × 32 = 512 traces | max consistent score 0.34 @ ct-byte (aliasing). **µ′-bit cross-key leak 없음**. trace-only sparse sx 회복 X 확정. |
| Toy PCO trace cost | Pilot 4 oracle 99% 가정 하 sk recovery 비용? | Monte Carlo 10000 trials | N=7 × 1024 queries ≈ 7168 traces ≈ 24 min total. 100% match. |

## 결정적 정정 (Corrigendum, 2026-05-04)

이전 직접-PoI 결과는 **µ′ label 로 PoI 학습 → IND-CCA 위반**. 다음 항목들이
정정되었다:

- "sk[0] 100% real-SCA" — instrumented 환경 (µ′ 누설) 가정 하 결과. attack-valid
  자체는 random level.
- "N=2 minimum cost SOTA" — 동일. 8 traces 100% 은 instrumented.
- "smaug3/5 extension 3-α 128/384/132" — 정정 cross-seed 평가에서는 3-α
  100%, 2-α 0.78 으로 calibrated cross-key 100% 가 paper main result.

상세는 메모리 `corrigendum_2026_05_04.md` 참조.

## 현재 상태 (2026-05-09)

- 표준 chosen-CT trace-based SCA: **cross-key sk recovery 불가능** (단일 환경 SNR + cross-key path divergence wall).
- PCO oracle: sk-universal classifier 입증 (Pilot 4 cross-sk 99%). 표준 1-bit
  PCO 채널로 sk recovery 가능 (toy cost ≈ 7168 traces / 24 min).
- µ′-bit 직접 cross-sk leak 없음 — sparse sx 의 trace 만으로 복구는 불가.
- threat-model: only `D` (full crypto_kem_dec) 이 attack-valid. `Z/V/R/Q/Y/W/T/U`
  는 모두 instrumentation.

## 어디서 읽어야 하는가

| 목적 | 파일 |
|---|---|
| 트랙 메인 timeline (S1/S2/S3/S4/D-Pair 등 누적) | `IDEA.md` |
| C2 paired threshold tomography 상세 | `c2_threshold_pairs.md` |
| R µ′ round/pack 분석 | `r_mu_roundpack.md` |
| Y FO downstream replay | `fo_downstream.md` |
| D-Pair trace-only µ′-change oracle | `d_pair_oracle.md` |
| S5 trace-count scaling audit | `s5_trace_scaling.md` |
| Toom/Karatsuba state mapping 자료 | `notes/toom_cook_layout.md` |
| 트랙 진입점 (README) | `README.md` |
| Cross-track hub | `../README.md` |
