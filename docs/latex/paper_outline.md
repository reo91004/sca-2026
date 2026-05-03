# Paper Outline — SMAUG-T smaug1 chosen-CT SCA (draft, 2026-05-03)

## Working title
"Real-trace SCA on SMAUG-T via Multi-term Chosen-Ciphertext and Direct Attack
PoI Learning"

## Core contributions

1. **Chosen-CT 컨벤션 정정** (Section: Chosen-CT design)
   - 원래 c1=α·X^j 의 µ′_j ↔ s_j 가정이 잘못됨을 수학+실측으로 발견
   - 정확한 매핑: µ′_i = α · s_(i-j mod n) · sign_l(i) (anticyclic wrap)
   - 새 설계 c1=α 상수 → 256 비밀 동시 leak. **4 chosen-CT 로 full sk** (4400×
     효율, 32k+ → 64 trace)
   - 검증: tests/test_chosen.py + test_partition.py 39 tests, 4 j × 2 α = 8
     groups round-trip 100%

2. **Multi-term chosen-CT design + R^2 cross-component** (Section: Method)
   - c1[m] = Σ_l α[m, l] · X^l 일반형 빌더 (host/smaug/chosen.py)
   - R^2 cross-component: c1[0] = α·X^l, c1[1] = α·X^k (eprint25 OT-FDA-CK 영감)
   - 7 design 카탈로그: const, 2-term, 3-term, R^2 combined
   - HW gain 실측: 단항 36 → 2-term 68 → 3-term 80 (1.6-2.2× 증폭)
   - 모든 design board round-trip 100% (predict_mu_prime ≡ Z 응답)

3. **SNR phase transition (negative → positive)** (Section: Setup limits)
   - per-bit FD attack (E3-FD) on profile_random_mu N=200/1000 — noise-bound,
     평행성 (max|t| 4.97 → 5.19)
   - multi-term TVLA N=64 (max|t|=4.03), N=256 (3.64) — *random walk*
   - **N=512 phase transition**: max|t|=15.20 (3.8× jump), leaky 1379 pts
   - SMAUG-T trace 의 SCA-readable signal 이 N=256-512 사이 emerge

4. **Direct attack PoI (cross-domain transfer 우회)** ★ **메인 contribution**
   - profile (random-µ) PoI transfer: sign 반전, sparse_recover 후 raw 보다
     나빠짐 (58.2% → 55.3%, zero-baseline 72.7% 미달)
   - **Direct attack PoI**: chosen-CT data 자체에서 cross-design Welch-t →
     PoI[i] + sign[i]
   - 결과: oracle pair (α=64 + α=192) 의 cross-design split 이 sk ternary
     value 별 자동 분리:
     * sk_i=0  → 둘 다 µ′_i=0 (cross-design 차이 0, invalid bit, 자동 0 분류)
     * sk_i=+1 → α=64 µ′_i=1, α=192 µ′_i=0
     * sk_i=-1 → α=64 µ′_i=0, α=192 µ′_i=1
   - **valid bits = sk nonzero 위치 정확히** (HW=70). HW=70 sparse constraint
     디자인 자체에 implicit.

5. **sk recovery 정확도 (paper 의 main result)**
   | dataset | designs | N | valid bits | corr | raw | sparse |
   |---|---|---:|---:|---:|---:|---:|
   | attack_const_c1 | 4 | 16 | 122/256 | +0.99 (s[0]) | 100% | 100% |
   | H_attack_n1024  | 2 | 1024 | 70/256 | +1.00 | 100% | 100% |
   | (s[1] N=16) | 4 | 16 | 122 | +0.90 | 89.5% | 86% |
   - **sk[0] 100% real-SCA 복구**: 4 chosen-CT × N=16 = 64 trace, ~10 sec
   - 비교: E5-Crypto (Z µ′ artificial oracle) 64 trace 30 sec, 100%
   - **real-SCA path 의 첫 100% 결과** on SMAUG-T smaug1, CW-Lite + Vcc 션트
     setup, no EM probe.

6. **Sparse ternary recovery (HW=HS MAP)**
   - host/analysis/sparse_recover.py: greedy + 1-swap + binary_to_ternary +
     accuracy
   - Greedy MAP under HW=HS: support score = log P(s≠0|data) − log P(s=0|data),
     top-HS positions select, sign argmax.
   - 검증: synthetic SNR curve (정확도 vs noise), idealized binary detector
     → posterior → 정답 sk 정확 복구 (tests/test_sparse_recover.py 11 tests)
   - **sign_acc 100% within selected support** — sparse 가 선택한 nonzero
     위치는 sign 모두 정답.

7. **Negative findings (보강)**
   - Per-bit FD profile transfer fail (profile-PoI sign 반전)
   - 'T' 명령 isolated poly_mul_acc round-trip fail (archive internal storage
     form, q-modular 또는 NTT-friendly 추정)
   - Multi-term gain 이 *trace SNR detection* 으로 못 깸 (N=64/256 noise floor 미만)

## Section outline

1. Introduction
   - PQC SCA 배경 (HQC OT-PCA, Kyber NTT SASCA)
   - SMAUG-T smaug1 / Toom-Cook unique 한 점
   - Contribution 요약

2. SMAUG-T background
   - smaug1 parameter set (n=256, q=1024, p=256, p′=32, t=2, HS=70, MODULE_RANK=2)
   - Toom-Cook 4-way + Karatsuba (NTT 안 씀)
   - FO transform + indcpa_dec / re-enc / cmov 단계

3. Chosen-CT design + 컨벤션 정정
   - 원래 가정의 한계 (수학 도출)
   - 정확한 µ′_i = α · s_(i-j) · sign 공식
   - 새 4 chosen-CT 디자인 (full sk)

4. Multi-term + R^2 cross-component
   - host/smaug/chosen.py 빌더 수학적 정의
   - 시뮬레이터 + 보드 round-trip 검증

5. Setup + measurement
   - CW-Lite + Vcc 션트 + STM32F415 + 25 dB LNA
   - 'Z' 명령 (indcpa_dec only, 32B µ′ 응답)
   - 'X' 명령 (sk PKE dump, ground truth)

6. SNR phase transition (negative → positive)
   - profile FD attack negative (N=200/1000)
   - multi-term TVLA noise→signal regime (N=512)

7. Direct attack PoI ★
   - cross-design Welch-t method
   - oracle pair 의 sk ternary 자동 분리
   - 100% sk[0] real-SCA recovery

8. Sparse ternary recovery
   - HW=HS MAP algorithm
   - sparse_recover end-to-end

9. Discussion
   - 비교 (HQC OT-PCA, Kyber chosen-CT SPA, ML-KEM SASCA)
   - SMAUG-T 의 unique vulnerability/advantage
   - countermeasures 후보 (codeword masking, shuffling, etc.)

10. Conclusion + future work
    - sk[1] N 늘려 100%
    - 더 다양한 sk seed 평균
    - F5-F7 Toom-Cook BP-SASCA (R&D, 추후)

## Figures

1. F_phase_transition.png — SNR N=64/256/512 곡선
2. F_direct_poi_n1024.png — direct PoI 결과 (70 valid bits = sk nonzero)
3. (생성 예정) sparse_recover_compare.png — profile transfer vs direct PoI
4. attack_const_c1_full_sk.png (이미 있음) — Z oracle 100% recovery
5. attack_sca_classification.png (이미 있음) — profile-PoI 56% baseline
6. (생성 예정) cycle_layout.png — Toom-Cook 함수 사이클 layout

## Supplementary

- 코드: github 공개 (commit chain a45eef6 → b69b05f)
- 데이터셋: traces/*.npz 일부 공개
- 시뮬레이터: host/smaug/* (numpy spec implementation)
