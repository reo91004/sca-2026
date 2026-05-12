# SMAUG-T Chosen-CT SCA — Track Overview (Archived)

ChipWhisperer-Lite + CW308T-STM32F415 위에서 SMAUG-T smaug{1,3,5} (k ∈ {2,3,4})
에 대해 진행한 chosen-ciphertext side-channel 공격 연구. **트랙 archived
(2026-05-09)** — 표준 chosen-CT trace-based SCA 로는 cross-key sk 복구가
infeasible 함을 확정하고 NTRU+ 트랙으로 pivot.

Last updated: 2026-05-12.

---

## 1. Result summary

1. **표준 chosen-CT trace-based SCA 는 cross-key sk recovery 불가**. SMAUG-T
   smaug1 (k=2, n=256, q=…, hs=70 sparse ternary) 기준, CW-Lite per-bit Welch-t
   가 N=1000 에서도 noise floor 4-5 에 갇힘. multivariate / low-dim /
   Toom/Karatsuba-state labels 모두 cross-key wall 그대로.
2. **PCO oracle 은 sk-universal**. 'd'(valid)/'D'(chosen) accept-vs-reject
   classifier 가 inter-sk 99% transfer (Pilot 4). 1-bit PCO 채널로 sk
   recovery 자체는 가능, toy cost ≈ 7168 traces / 24 min (N=7 × 1024 queries).
3. **µ′-bit 직접 cross-sk leak 은 없음**. Reject trace 의 큰 \|t\| (170-216) 은
   z (rejection seed) leak 우세이며 µ′-bit 와 분리되지 않음 (Pilot 5). 4 cyclic-
   shift designs × 4 sk × 32 traces 에서 µ′-bit cross-sk consistent score
   max 0.34 — ct-byte aliasing 수준 (Pilot 6).
4. **Corrigendum 2026-05-04**: 이전 "8 traces 100%" 류 결과는 µ′ label 로 PoI
   학습 = IND-CCA 위반. 정정 cross-seed calibrated 평가에서 smaug3/5 3-α 100%,
   2-α 0.78 만 paper-grade.

요약: trace-only attack 은 negative result. PCO 1-bit oracle 은 positive
result 이지만 sparse sx 직접 회수 불가 — 표준 PCO 채널로의 우회 가능성만.

## 2. Threat model

- **Target**: SMAUG-T smaug1 (k=2, hs=70), smaug3 (k=3, hs=88),
  smaug5 (k=4, hs=87 — NOT k=5). Module-LWR + Toom-Cook 4-way, **no NTT**.
  q=2^10·… (정확 값은 pqcmp.kr port lib/include 참조), n=256 fixed, secret
  ∈ {-1, 0, +1}.
- **Attacker capability**: chosen-CT 주입 (정합 protocol 'F/I/L/D' for
  smaug-injection), power trace, public parameters.
- **Forbidden** (IND-CCA): target-side 의 returned µ′, ss, dumped sk 사용해
  inference. 2026-05-04 corrigendum 의 핵심 정정 이유.
- **Attack-valid command** = `D` (full `crypto_kem_dec`). 다른 명령들
  ('T', 'V', 'R', 'Q', 'Y', 'W', 'Z', 'U') 는 instrumentation 으로
  PoI/label calibration 에만 사용 가능 (attack-time 사용 X).

## 3. Background

SMAUG-T 의 Toom-Cook 4-way + Karatsuba 결합 multiplication 은 NTRU+ 의
NTT 와 달리 sub-step 별 trace landmark 가 다르다. 초기 작업 가설:
multi-design chosen-CT 가 sparse ternary sk 의 좌표를 binary "did mu'
change" oracle 로 분리할 수 있을 것.

실측 결과 cross-key 환경에서:
- sk-dependent execution path 가 alignment 후도 random level 로 분산 (E1/E3
  alignment findings).
- FFT magnitude representation 도 cross-key fail.
- Trace-only sparse recovery 는 root SNR limit (per-bit FD infeasible) 에 부딪힘.

PCO branch 는 단일 sk 안에서는 accept/reject 100% classifier 가 가능했고,
이 classifier 가 sk-universal (다른 sk session 에 직접 transfer) 함을 확인.
그러나 µ′-bit 자체의 cross-sk leak 은 측정되지 않아 표준 PCO 1-bit
channel 로만 sk recovery 가능 — 별도 contribution 아님.

## 4. Flow at a glance

자세한 phase / branch entries 는 [`experiments.md`](experiments.md).

### Phase spine (main)

| Phase | 문제 | 결과 |
|---|---|---|
| S1 — `T` poly_mul_acc | diagnostic. sk-dependent leak 존재? | cross-sk t 있음. 단일 coord HW LOO recover X. |
| S2 — `Z` indcpa_dec | chosen-CT attack-valid signal? | per-bit \|t\| noise floor 4-5 wall (N=1000). |
| S2.5~S2.13 — multivariate / matrix / low-dim / Toom-Karatsuba | SNR 개선 가능? | per-design score 일부 개선, cross-key wall 그대로. |
| S3 — `V` vec_vec_mult_add | 다른 함수에서 leak? | same wall. |
| S1U — multi-term `T` | SNR 합산? | partial signal 강화, sk 복구 도달 X. |
| S4 — c2 pair / mu pressure / pair distance | residualized low-dim, paired oracle | wall 그대로. |
| D-Pair — trace-only µ′-change oracle on natural `D` | 동일 sk paired tomography | corr z ≤ +1.6 ceiling across early/late window, staircase, multi-term. |

### Branches (sub-investigations)

| Branch | 가설 | 결과 |
|---|---|---|
| C2 | c1, c2 pair 로 mu_delta threshold 추적 | flip_any α-bug 발견 (signed mu_delta 사용해야), pair 자체로 sk 도달 X. |
| R | µ′ round/pack leak | latent µ′ pattern 검출, cross-key consistent X. |
| Y | FO downstream replay | partial leak, chosen-CT 결정 분리 X. |
| S5 — trace scaling | N 증가만으로 threshold 깨기? | N=5000 까지도 변화 미미. |

### PCO pilots

| Pilot | 가설 | 결과 |
|---|---|---|
| 1 | `D` 끝부분 chosen-CT 결정 leak? | sample 22924 (94%) max\|t\|=31.93. zero vs a192 single-trace 100%. 첫 cross-key-free signal. |
| 2 | leak source 가 µ′ ? | HW(µ')=0 design 도 max\|t\|=45.96 100% 구분 → leak = ct-byte operand power. |
| 3 | trace 만으로 accept(valid) vs reject(chosen) 100%? | late max\|t\|=33.8-34.3, ctrl 9.8 (3.5×). 표준 PCO 1-bit oracle 입증. |
| 4 ★ | classifier cross-sk transfer? | inter-sk 96.9-99.2%. **oracle sk-universal**. PCO sk recovery 가능. |
| 5 | reject \|t\| 가 µ′ 직접 leak? | \|t\|=170-216 but zero≈a192 (1.04). z (rejection seed) leak 우세. |
| 6 | µ′-bit cross-sk leak? | 4 sk × 4 designs × 32 = 512 traces. max consistent 0.34 = ct-byte aliasing. **µ′-bit cross-key leak 없음**. |

### E-series (alignment + FFT)

| Pilot | 가설 | 결과 |
|---|---|---|
| E1 / E3 | 정렬·DTW 가 cross-key fail 원인? | same-sk cross-session 100%, cross-key 는 정렬 후도 random. sk-dependent execution path 시사. |
| E5 | FFT magnitude / OT-PCA 로 회복? | FFT 도 cross-key fail. host_posterior_table 이 OT-PCA offline template 와 동치라 추가 이득 없음. |

### Toy cost

Pilot 4 99% oracle 가정 하 sk recovery: N=7 × 1024 queries = 7168 traces ≈
24 min total. Monte Carlo 10000 trials 100% match.

## 5. Corrigendum 2026-05-04 (반드시 읽을 것)

이전 commit history 에 남아 있을 수 있는 다음 claims 는 **모두 IND-CCA 위반
- µ′ label 로 PoI 학습 (instrumented oracle)**:

- "sk[0] 100% real-SCA" → instrumented (µ′ leak 환경) 가정 하 결과,
  attack-valid 자체는 random level.
- "N=2 minimum cost SOTA — 8 traces 100%" → 동일.
- "smaug3/5 extension 3-α (128, 384, 132)" → same-key sk_gt cross-tab 기반.

정정 후 paper main result: **calibrated cross-seed 평가에서 smaug3/5 3-α
100%, 2-α 0.78**. 자세한 메모리는 `memory/corrigendum_2026_05_04.md`.

## 6. Figures

SMAUG-T 트랙의 분석 산출물은 results/smaug/ 에 보관되어 있었으나 2026-05-12
정리에서 모두 삭제됨 (트랙 종료 + 디스크 회수). 핵심 결과는 본 README +
experiments.md 의 텍스트 수치로 보존되고, 자세한 분석 figure 는 git history
의 commit 까지 (특히 commit `f5a9a0b`) 에서 raw 한 형태로 복구 가능하다.

## 7. Reproduction

### 7.1 펌웨어 빌드/플래시

```bash
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4 SMAUG_LEVEL=1
python3 -m host.smaug.upload firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
```

### 7.2 핵심 분석 명령 (코드는 보존, 데이터는 archived)

`traces/aligned/` 와 `traces/` 직속 SMAUG-T .npz 들은 2026-05-12 정리에서
삭제되었다. 새 캡처는 다음 명령으로 가능하지만 결론이 negative result 임을
주의:

```bash
# S1 — T diagnostic capture
python3 scripts/smaug/s1_t_capture_main.py -n 500 \
    --out traces/s1_main_new_c0_j0_a1_n500.npz

# S2 — Z chosen-CT capture
python3 scripts/smaug/s2_z_capture_main.py -n 200 --alpha 4 \
    --out traces/s2_z_new_a4_n200.npz

# Analysis after capture
python3 scripts/smaug/s1_t_final_analysis.py \
    --out-prefix results/smaug/recheck_s1_final --n-shuffles 50
python3 scripts/smaug/s2_z_analyze.py \
    --out-prefix results/smaug/recheck_s2_z_final --n-shuffles 50
```

### 7.3 호스트 단위 테스트

```bash
python3 tests/run_all.py
# 89/89 OK 가 정상
```

## 8. Limitations / Closing notes

- **트랙 종료 사유**: CW-Lite per-bit FD 한계 (SNR wall) + cross-key path
  divergence 가 fundamental. 더 좋은 측정 setup (EM probe, faster ADC) 없이는
  trace-only sk recovery 가 본 implementation 에서 infeasible.
- **PCO oracle**: sk-universal classifier 가 본 프로젝트의 첫 cross-key SCA
  signal 이지만 µ′-bit 직접 leak 없음 → 표준 PCO 채널 sk recovery 만 가능,
  그 자체는 별도 contribution 으로 새롭지 않음.
- **남은 코드**: host/smaug/, scripts/smaug/, firmware/simpleserial-smaug/,
  tests/smaug/, docs/smaug/ 모두 보존. 미래 SCA 비교 reference 또는 다른
  KEM 으로 같은 방법론 transfer 시 활용 가능.
- **다음 단계**: NTRU+ 트랙으로 pivot 완료 ([`../ntruplus/README.md`](../ntruplus/README.md)).
  SMAUG-T 트랙 재개는 hardware 보강 또는 SASCA 같은 fundamentally 다른
  접근이 필요.
