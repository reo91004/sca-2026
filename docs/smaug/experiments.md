# SMAUG-T — Phase Experiment Log

[`README.md`](README.md) 의 "Flow at a glance" 가 트랙 전반 navigational summary,
이 문서가 phase / branch / pilot 별 detail (역시간순). 트랙 archived 이므로 새
entry 추가는 없으나 raw 한 가정·실험·결과 record 는 여기에 보존.

핵심 detail 은 git history 의 commit `f5a9a0b` 이전 docs (IDEA.md, branch md
들) 에 더 자세히 있다. 본 문서는 essentials 만.

---

## D-Pair Branch (2026-05-08~) — Trace-only μ′-change Oracle on natural `D`

**Motivation**. 이전 branches (R, Q, Y, C2) 가 isolated firmware 트리거에서
diagnostic µ′ leak 을 보였으나 natural decapsulation 으로 transfer 못함. 따라서
attacker 가 측정 불가능한 continuous secret label 회귀 대신, 더 단순한
binary "µ′ 가 바뀌었는가?" 가설로 전환.

**Threat-model rebase**. Only `D` (full crypto_kem_dec) 가 attack-valid. T, V,
W, R, Q, Y, Z 는 모두 instrumentation.

**Phases**:
- 0 — Tooling.
- 1.0 — D smoke (정합 baseline).
- 1.1 — D scout early window + α-bug 발견 (binary "any flip" with monomial
  c1 c2=(15,16) 가 public α 로 결정됨, secret 아님 — signed mu_delta 또는
  s-distinguishing pairs 써야).
- 1.1' — Secret-dependent label.
- 1.2 — D control (c1=0).
- 1.4 — Z ablation (NOT attack claim).
- 1.5 — Late-D scout (--adc-offset 24400).
- 2.1 — c2 staircase (3 secret-dependent pair types).
- 2.2 — Multi-term c1.

**Conclusion (D-Pair)**. corr z ≤ +1.6 ceiling across all variants (early/late
window, staircase, multi-term, signed mu_delta). same CW-Lite SNR wall as
the trace-only chosen-CT path. memory `d_pair_subgate_ceiling`.

---

## S4 (2026-05-06~) — Two-Way Residualized Low-Dim Reanalysis

S2.7~S2.13 의 low-dim labels 가 residualize 했을 때 sk evidence 가 남는지
재평가. S4.1, S4 c2 pair, S4 mu_recovery_pressure, S4 pair_distance_oracle.

**Resolved Negative Paths**:
- S1 diagnostic 만으로 sk distribution recovery 불가.
- single-coord HW model LOO 88% sign, 75% support — paper-grade 못 됨.
- S2 chosen-CT per-bit HW recovery 가 cross-key fail.

---

## S1U (2026-05-04~) — Multi-Term Isolated `poly_mul_acc`

다중 항 chosen-CT 가 SNR 합산을 주는지. S1U.1 에서 lower-level Karatsuba
labels 까지 내려가서 진단. partial signal 강화 확인했으나 sk 복구 도달 X.

---

## S3 (2026-04~) — `V` vec_vec_mult_add + S3.1~S3.4

다른 함수 (vec_vec_mult_add) 에서 leak 가능성. low-dim Toom labels (S3.1),
multi-term V matrix (S3.2), component-specific W replay (S3.3), nonzero-c2
add/HD labels (S3.4). 결과: same cross-key wall.

---

## S2 series (2026-04~2026-05-04) — `Z` chosen-CT indcpa_dec

핵심 메인 phase. 모든 multivariate / matrix / low-dim 시도가 cross-key fail.

- S2 — per-bit \|t\| noise floor 4-5 wall (N=1000). 메모리 `snr_limit_cwlite`.
- S2.5 — Multivariate Trace-Only Profiling.
- S2.6 — Multi-Design Z Matrix Pilot.
- S2.7 — Low-Dimensional Matrix Labels (per-design score 개선).
- S2.8 — Random Multi-Term Z Matrix Pilot.
- S2.9 — Z Window-Local Product Labels.
- S2.10 — Focused ADC-Offset Capture Around sample 17731-19779.
- S2.11 — Disassembly-Derived Toom/Karatsuba State Labels (window scan score
  gain — 그러나 cross-key 도달 X).
- S2.12 — Scored Public-Design Capture.
- S2.13 — Nested Leakage-Aware Design Selection.

---

## S1 (2026-04~) — `T` Diagnostic poly_mul_acc

가장 단순한 baseline. c=0, j=0, α=1 한 환경에서 cross-sk Welch-t + per-coord
HW CPA. cross-sk t 값 있음, 그러나 단일 coord HW 만으로 LOO sk 복구 X.
S2 의 motivation 이 됨.

---

## Branch C2 (2026-05-06) — Paired Threshold Tomography

c1, c2 pair 로 mu_delta 의 threshold 추적. c2 staircase + monomial pairs.

**핵심 발견**: flip_any α-bug — binary "any flip" 가 public α 로 결정되어
secret-dependent 아님. signed mu_delta 또는 s-distinguishing pairs 필요.
메모리 `flip_any_alpha_bug`.

**결과**: pair 자체로 sk 도달 X.

---

## Branch R (2026-05-06) — µ′ Round/Pack Leakage

채택된 µ′ bit 패턴이 round/pack 단계에서 leak 하는지 chosen CT 로 latent µ′
유도 + capture. leak 신호 검출 — 단 cross-key consistent 하지 못함.

---

## Branch Y (2026-05-06) — FO Downstream Replay

FO transform 의 re-encrypt 단계 leak. downstream replay capture.
partial leak — chosen-CT 결정 분리 X.

---

## Branch S5 (2026-05-06) — Trace-Count Scaling Audit

N 증가만으로 detection threshold 깰 수 있나? N sweep + segment scan. N=5000
까지도 변화 미미. 메모리 `s5_*` 자료 (현재 메모리에 없음 — 별도 archive).

---

## PCO Pilots (2026-05-04)

### Pilot 1 — late-trace leak

`D` 끝부분에서 chosen-CT 결정의 leak 가 보이나? full crypto_kem_dec capture,
accept vs reject 분류. 결과: sample 22924 (94%) max\|t\|=31.93. zero vs a192
single-trace 100%. **첫 cross-key-free signal**. 단 leak source = ct-byte
operand power (µ′ 아님). 메모리 `pco_pilot_late_trace_leak`.

### Pilot 2 — leak is ct-byte

HW(µ′)=0 인 두 design 도 max\|t\|=45.96 100% 구분. leak source 확정: µ′ 가
아니라 ct-byte operand power. cross-session PoI 0-sample stable (95.3%
transfer). 메모리 `pco_pilot2_findings`.

### Pilot 3 — accept vs reject 100% classifier

trace 만으로 'd' (valid) / 'D' (chosen) 100% classifier 가능?
late max\|t\|=33.8-34.3 vs ct-only ctrl 9.8 (3.5×). **표준 PCO 1-bit
oracle 입증**. 메모리 `pco_pilot3_accept_reject`.

### Pilot 4 — cross-sk transfer 99% ★

다른 sk session 간 classifier 전이 가능? 4 sk 두 session 분류기 96.9-99.2%.
**oracle sk-universal**. 첫 cross-key SCA signal in this project. PCO sk
recovery 가능성. 메모리 `pco_pilot4_cross_sk_transfer`.

### Pilot 5 — z-leak vs µ′-leak

reject path 의 큰 \|t\| 가 µ′ 직접 leak 인가? inter-sk reject trace
\|t\|=170-216 BUT zero ≈ a192 (1.04). **z (rejection seed) leak 우세**,
sparse sx 직접 복구 불가. PoI 23284 은 cleanly sk-independent (oracle 분류기
로 valid). 메모리 `pco_pilot5_z_vs_mu`.

### Pilot 6 — no µ′-bit cross-sk leak

4 sk × 4 cyclic-shift designs × 32 = 512 traces. cross-sk consistent score
max 0.34 @ ct-byte leak 위치 (aliasing). **µ′-bit cross-key leak 없음**.
trace-only sparse sx 복구 불가 확인. 메모리 `pco_pilot6_no_mu_bit_leak`.

### Toy PCO trace cost

Pilot 4 99% oracle 가정 하 sk recovery: N=7 × 1024 queries = 7168 traces
≈ 24 min total. Monte Carlo 10000 trials 100% match. 메모리 `toy_pco_trace_cost`.

---

## E-series (2026-05-04) — alignment + FFT findings

### E1 / E3 — alignment + DTW

정렬·DTW 가 cross-key fail 의 원인? full pipeline TVLA after alignment.
**same-sk cross-session 100%**, cross-key 는 정렬 후도 random.
**sk-dependent execution path 시사** — firmware mismatch guard 추가. 메모리
`e1_e3_alignment_findings`.

### E5 — FFT magnitude + OT-PCA insight

다른 representation 으로 cross-key 회복? FFT magnitude TVLA 도 cross-key
fail. OT-PCA 의 offline templates = combinatorial — 우리 host_posterior_table
이 이미 그 형태이므로 추가 이득 없음. 메모리 `e5_fft_negative_otpca_insight`.

---

## Earlier instrumented results (corrigendum 2026-05-04 정정 대상)

다음 results 는 **모두 IND-CCA 위반 (µ′ label 로 PoI 학습)** 으로 정정됨.
git history (commit `f5a9a0b` 이전) 의 IDEA.md / direct_attack_poi_breakthrough
메모리에 raw 한 형태로 보존되어 있으나, attack-valid 자체는 random level.

- 2026-05-03 direct attack PoI 돌파 — sk[0] 100%.
- 2026-05-03 N=2 minimum cost SOTA — 8 traces 100%.
- 2026-05-03 smaug3/5 extension — 3-α (128, 384, 132).
- 2026-05-03 N=512 phase transition — N=256→512 noise→signal 전환,
  max\|t\|=15.20, 1050+ leaky pts (SMAUG-T trace 에 SCA signal 존재 확인).
- 2026-05-03 H capture findings — round-trip 100%, HW gain 1.6-2.2×, TVLA
  N=64 noise-bound (N≥256 detect 필요).

**정정 paper main result**: calibrated cross-seed 평가에서 smaug3/5 3-α 100%,
2-α 0.78. 메모리 `corrigendum_2026_05_04`.

---

## Threat-Model Rules (snapshot, finalized 2026-05-08)

- attack-valid 명령 = `D` (full crypto_kem_dec). 다른 모든 명령 (T, V, W, R,
  Q, Y, Z, U) 은 instrumentation 이며 PoI/label calibration 에만 사용 가능.
- target-side 의 returned µ′, ss, dumped sk 를 inference 에 사용 X (IND-CCA).
- PoI 학습이 µ′ label 을 쓰면 IND-CCA 위반 — 직전 단계의 정정 사유.
- attack-time CPA labels 은 public chosen-CT γ + sk-independent timing
  model 만.
- alignment / DTW / FFT 도 cross-key transfer 가 검증 단계 통과해야 paper-grade.
