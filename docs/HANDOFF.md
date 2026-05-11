# NTRU+ Phase 4/4.6 Handoff (2026-05-11)

이 문서는 NTRU+ branch Phase 4 (NTT-domain selected-lane chosen-CT CPA)
의 종료 시점 상태와 paper-grade 결과를 정리한다. 다음 세션에서 이어가려면
이 문서부터 읽고 시작.

## 1. 빠른 요약 (Executive Summary)

**Status**: Phase 4 attack-valid 분석 **완료**. Phase 4.5/4.6 single-victim
multi-lane 확장도 3 victims 까지 분석 완료. Paper-grade 핵심 결과는 Phase 4
multi-victim TOP-1/Top-100 이고, Phase 4.5/4.6 은 partial information
disclosure 를 보수적으로 뒷받침한다.
SMAUG-T 분기 cross-key fail 과 대조적으로, NTRU+ 분기는 **단일 victim
session 안에서 standard CPA + dual-channel HW model + per-slot PoI** 로
NTT 좌표 회수 입증.

**핵심 결과** (240 (key, lane, slot) cases, 4 lanes × 28 keys × 4 slots):
- **2 perfect TOP-1**: f=16 (M1 channel, N=2 traces 충분), f=882 (M5 channel
  + full-stack, N=16 traces 충분).
- **17 unique top-100 / 240 (7.08%)** — M1 baseline ∪ full-stack pipeline.
- 회수 lane: **0, 64, 80, 128** 모두 (lane=80 도 K=8 + refined per-slot 으로 회수).
- **5종 mechanistic finding** (paper-quality): N-saturation, dual leak channel,
  per-(lane, slot) PoI drift, b 분포 |f_c| 의존성, σ_α 한계.

**최신 single-victim 결과** (Phase 4.5/4.6, 3 victims, 88 lanes, 352 cases):
- victim 1: L=16 N=8, M1 top-100 3/64, full-stack 4/64.
- victim 2: L=24 N=16, M1 top-100 3/96, full-stack 3/96.
- victim 3: L=48 N=8, M1 top-100 3/192, M5 top-100 9/192, full-stack 1/192.
- aggregate: M1 baseline 9/352, M5 baseline 12/352, full-stack 8/352.
- 192-lane projection 은 보수적으로 M1 약 20 coords/victim, M5 약 26,
  full-stack 약 17. 이전 2-victim 기반 29-34 coords/victim 수치는 lucky-victim
  영향을 받을 수 있어 paper 본문에서는 보조로만 사용.
- channel-mixed union upper bound: M1 ∪ M5 ∪ Full 은 29/352 top-100,
  192-lane projection 약 63 coords/victim. 이는 정보 누출 후보 pool 로만 쓰고,
  single-pipeline 공격 성능과 분리해서 보고한다.
- all-channel hit-list 기준 top-100 union 은 37/352 이다. M1/M5/Full baseline
  top-100 overlap 이 거의 없으므로, 다음 단계는 단일 `Zsum` 점수 강화보다
  channel별 candidate evidence 를 보존한 후단 결합이 유망하다.
- `scripts/n52_phase45_candidate_export.py` 로 352 rows × 5 channels 의 top-100
  candidate residues 를 `results/ntruplus768/phase45/candidate_export.npz` 에
  저장했다. 다음 recovery-pressure 실험은 trace 재로딩 없이 이 파일에서 시작.
- `scripts/n53_phase45_candidate_pressure.py` 결과 all-channel union 은 top-10
  4/352, top-50 18/352, top-100 37/352. 평균 all-channel candidate set size 는
  top-10 에서 41.2, top-100 에서 371.2 이므로 현재는 direct recovery 보다
  candidate-set constraint 로 해석해야 한다.
- `scripts/n54_phase45_candidate_null.py` 로 candidate-set size 를 반영한
  uniform-rank null 을 계산했다. all-channel top-100 은 observed 37/352,
  null expected 37.80, z=-0.14. 즉 Phase 4.5/4.6 raw top-100 candidate-set
  counts 는 유의하지 않다. Single-victim section 은 leakage/candidate-export
  artifact 로 두고, paper-grade positive claim 은 Phase 4 의 TOP-1/top-rank
  cases 와 mechanistic leakage 에 둔다.

## 2. 관련 파일

### 2.1 Capture data (traces/ntruplus768/phase3/)

| File | K | N | Lane | Comment |
|---|---:|---:|---:|---|
| wideg_lane0_K4N32.npz | 4 | 32 | 0 | round 1 |
| wideg_lane0_K4N64.npz | 4 | 64 | 0 | N-saturation 입증용 |
| wideg_lane0_K8N32.npz | 8 | 32 | 0 | round 1, **f=16 TOP-1**, **f=882 TOP-1** |
| wideg_lane0_K8N32_b.npz | 8 | 32 | 0 | round 2 |
| wideg_lane64_K8N32.npz | 8 | 32 | 64 | f=−2 case rk 44 (full stack) |
| wideg_lane80_K4N32.npz | 4 | 32 | 80 | small batch |
| wideg_lane80_K8N32.npz | 8 | 32 | 80 | **2026-05-10 추가** — lane=80 첫 회수 |
| wideg_lane128_K8N32.npz | 8 | 32 | 128 | round 1 |
| wideg_lane128_K8N32_b.npz | 8 | 32 | 128 | round 2 |

각 파일 ~496MB, 24400 sample, G=78 (HW=1+HW=2 wide-γ set).

### 2.2 분석 스크립트 (scripts/)

| Script | 목적 |
|---|---|
| n13_phase4_diagnose.py | T1-T4 진단 (within-key ridge, cross-key z, single-sample, pooled corr) |
| n14_cpa_per_key.py | Profile-free CPA per victim (per-cand best PoI) |
| n15_cpa_variants.py | V1 fixed-PoI per-trace, V2 mean-over-N, D1 null distribution |
| n16_snr_estimate.py | SNR 측정 (b, σ_HW, σ_noise) |
| n17_hw_models.py | HW 모델 5 변형 (multikey K=4 G=12) |
| n18_phase4_g78_capture.py | **G=78 wide-γ capture script** (참조 사용) |
| n19_wideg_attack.py | wide-γ attack-valid evaluator (single key) |
| n22_wideg_diagnose.py | SNR + null distribution + drift profile |
| n23_wideg_K4_attack.py | **K=4/8 attack-valid evaluator (메인 분석)** |
| n24_combined_analysis.py | **누적 통계 (현재 240 cases, 7 batches)** |
| n25_lane_snr_compare.py | Cross-lane SNR @ predict vs peak |
| n26_skindep_poi_v2.py | sk-indep PoI calibration 시도 (γ-variance, FAIL) |
| n27_v2w_window_sweep.py | V2w window sweep (FAIL) |
| n28_n64_diagnose.py | K=4 N=64 N-curve diagnostic |
| n29_noise_decompose.py | **σ_a (across-g) saturation 메커니즘 입증** |
| n30_alpha_removal.py | α-removal cross-key mean (FAIL) |
| n31_profiled_poi.py | Lane median offset profiled PoI (neutral) |
| n32_poi_ensemble.py | Predict + profiled max ensemble (neutral) |
| n33_b_distribution.py | **b 분포 + 회수 가능성 |f_c| 의존성** |
| n34_per_slot_drift.py | **per-(lane, slot) drift 발견 (paper-grade)** |
| n35_per_slot_profiled.py | per-slot offset 적용 |
| n36_hw_models_n64.py | 7 HW model variants on K=4 N=64 (M5 발견) |
| n37_montgomery_all.py | **M5 (Montgomery HW) 모든 batch 적용** |
| n38_zsum_ensemble.py | **M1+M5 Z-score sum ensemble (paper-grade)** |
| n39_full_stack.py | **★ Full pipeline: per-slot PoI + Zsum (240 cases)** |
| n40_full_stack_ncurve.py | Full-stack N-curve (f=882 N=16 입증) |

### 2.3 결과 파일 (results/ntruplus768/phase4/)

| File | Comment |
|---|---|
| combined.npz | n24 누적 (240 cases) |
| montgomery_all.npz | n37 M5 across batches |
| zsum_ensemble.npz | n38 M1+M5 Zsum |
| **full_stack.npz** | **n39 final pipeline 결과** |
| b_distribution.npz | n33 b 분포 |
| per_slot_drift.npz | n34 per-slot drift |
| per_slot_profiled.npz | n35 per-slot PoI applied |
| lane_snr_compare.npz | n25 cross-lane SNR |
| (그 외 various) | 진단 / 중간 결과 |

### 2.4 문서

| File | Comment |
|---|---|
| `docs/IDEA.md` | **메인 실험 로그** — Phase 4 entries (a)-(q8) |
| `docs/NTRUplus.md` | 초기 plan (largely unchanged) |
| `docs/HANDOFF.md` | 이 문서 |
| `host/ntruplus/` | host-side NTRU+ 패키지 (codec, ntt, params 등) |

## 3. Paper-grade 핵심 결과

### 3.1 두 perfect TOP-1 cases ★★★

| Case | Channel | N | rk |
|---|---|---:|---:|
| K=8 r1 ln=0 k=5 sl=1 **f=16** (\|f_c\|=16) | M1 (mod-q HW) | **2** | **0** |
| K=8 r1 ln=0 k=7 sl=3 **f=882** (\|f_c\|=882) | M5 (Montgomery HW) | **16** | **0** |

f=16 은 small-|f_c| 케이스, f=882 는 large-|f_c| 케이스 — **두 channel
이 다른 sk-구조 좌표 군 회수**.

### 3.2 240 cases recovery 통계

| Pipeline | top-1 | top-10 | top-100 | top-500 |
|---|---:|---:|---:|---:|
| M1 baseline (predict_poi) | 1 | 1 | **10** (4.17%) | 35 |
| M1 (slot-PoI) | 1 | 1 | 8 | 37 |
| M5 (slot-PoI) | 0 | 0 | 6 | 38 |
| Full stack (slot + Zsum) | 1 | 2 | **9** (3.75%) | **45** (18.75%) |
| **Union M1 ∪ Full** | **2** | **3** | **17** (7.08%) | ~50 (~21%) |

### 3.3 5종 mechanistic finding (paper-quality)

1. **N-saturation** (n28, n29): corr 가 `b·σ_HW / √((b·σ_HW)² + σ_α²)`
   에서 saturates. σ_α 가 saturating noise. measured corr / formula(SNR/√...)
   ratio 가 N=64 에서 **0.27** (formula overpredicts 4×). N>16 무용.
2. **Dual leak channel** (n36, n37): basemul 의 leak 은 두 채널 — M1
   (HW(γ·f mod q)) AND **M5 (HW(montgomery_reduce(γ·f)))**. M5 가 large-|f_c|
   에서도 회수 가능. `r[0] = montgomery_reduce(c[0]*f[0] − …)` 가 정확한 firmware.
3. **Per-(lane, slot) PoI drift** (n34): firmware-precise calibration. lane=80
   특히 slot 들이 spread 큼. lane=0 (uniform +24), lane=64 (+12,+12,+11,−2),
   lane=80 (+8,+7,−3,−3), lane=128 (0,−2,0,−1). sk-independent (firmware
   constant), 1회 calibration 후 모든 attack 에 적용 가능 — standard SCA practice.
4. **b 분포 + |f_c| 의존성** (n33): 회수 가능성은 b ≥ 0.0018 에 의존. b 는
   |f_c| 와 상관 — small-|f_c| (b_med 0.00130) >> large-|f_c| (b_med 0.00045).
   회수 가능 풀은 structurally **|f_c|<50 + Montgomery 채널의 large-|f_c| 일부**.
5. **σ_α saturation 천장** (n30, n32): cross-key α-removal, ensemble 모두
   ceiling 못 깸. recovery 한계는 σ_α/(b·σ_HW) 비율 — fundamental.

### 3.4 잡힌 SMAUG-T 분기 한계와의 대조

- SMAUG-T branch: cross-key transfer fail (random level, |t| 4-5 stuck noise floor).
- NTRU+ branch: **단일 victim session 안에서 회수**. cross-key transfer 불필요.
  본 프로젝트 **첫 attack-valid 결과**.

## 4. 미완료 작업 / Future work

### 4.1 Phase 4.5 완료 (2026-05-10) ★

**목표**: 단일 victim 의 multi-lane sk leakage 직접 입증.

**결과 (Phase 4.5, 160 cases, 2 victims, 40 lanes)**:
- victim 1 (235e27a0, scout L=16 N=8): 3 M1 t100, 4 full-stack t100
- victim 2 (8c6aacfa, main L=24 N=16): 3 M1 t100, 3 full-stack t100
- 합산: M1 baseline 6 / 160 (3.75%), Full stack 7 / 160 (4.4%),
  M1 ∪ Full ~9 unique top-100
- 0 TOP-1 (N=8/16 + lucky-victim 부재로), 1 top-10 (lane=104 rk=2)
- top-recovered: lane=104 slot=0 f=2563 |f_c|=894 **M1 baseline rk=2**
  (cross-channel large-|f_c|), lane=128 slot=2 f=3448 |f_c|=9 (small)

**192-lane single-victim projection** (linear extrapolation):
- M1 baseline: ~29 NTT coords / victim
- Full stack: ~34 NTT coords / victim
- Union: ~43 unique top-100 / victim (n42 추정과 정확히 일치)
- **Superseded for final paper numbers by Phase 4.6**: 3-victim aggregate
  gives more conservative single-pipeline projections of ~17-26 coords/victim.

**Mechanistic finding (추가)**: linear interpolation of LANE_SLOT_OFFSET
between calibrated lanes (0, 64, 80, 128) — Phase 4 multi-key oracle
median 기반 sk-independent — is effective: M1 baseline 3 → slot-PoI 6
top-100 (main capture). Slot drift 가 lane 별로 monotonic 하지 않지만
linear interp 가 +3 coords/96 cases 회수.

**Paper claim**:
- §5.2 (Phase 4.5): single-victim multi-lane recovery 가 가능. 192-lane
  적용 시 ~30 NTT 좌표 회수, entropy reduction ~31% (153/1152 bits).
- 그러나 lattice full sk recovery 는 **infeasible** (β_req~300). Partial
  information disclosure 가 honest claim.
- Phase 4.6 반영 후에는 paper 본문 claim 을 ~17-26 coords/victim
  (single-pipeline), channel-mixed union upper bound ~63 coords/victim 로 분리한다.

**관련 산출물**:
- traces: `traces/ntruplus768/phase45/{scout_K1_L16_N8, main_K1_L24_N16}.npz`
- 분석: `results/ntruplus768/phase45/{scout, main, combined, SUMMARY}.{npz, md}`
- figures: `results/ntruplus768/phase45/figures/F10/F11/F12.png`

### 4.1b Phase 4.6 완료 (2026-05-10/11) ★

**목표**: 더 넓은 lane coverage 의 3rd victim 으로 Phase 4.5 projection 검증.

**결과 (192 cases, 1 victim, 48 lanes)**:
- victim 3 (8ce340e4bdfa80fc, wide L=48 N=8): M1 baseline 3/192 top-100,
  M5 baseline 9/192 top-100, full-stack 1/192 top-100.
- M1 top-100 cases:
  - lane=24 slot=0 f=1215 |f_c|=1215 rk=15
  - lane=76 slot=1 f=1169 |f_c|=1169 rk=23
  - lane=92 slot=2 f=565 |f_c|=565 rk=50

**3-victim cumulative (352 cases, 88 lanes)**:
| pipeline | top-1 | top-10 | top-100 | top-500 |
|---|---:|---:|---:|---:|
| M1 baseline | 0/352 | 1/352 | **9/352** | 45/352 |
| M5 baseline | 0/352 | 1/352 | **12/352** | 50/352 |
| M1 slot-PoI | 0/352 | 0/352 | 8/352 | 48/352 |
| M5 slot-PoI | 0/352 | 1/352 | 10/352 | 47/352 |
| Full stack | 0/352 | 1/352 | **8/352** | 46/352 |

**해석**:
- 48-lane victim 의 M1 yield 는 2-victim projection 보다 낮아, single-victim
  coordinate yield 에 victim-to-victim variance 가 큼.
- 보수적 paper claim: 192-lane single-victim coverage 에서 약 17-26 top-100
  NTT-coordinate disclosures / victim. Full key recovery 는 여전히 infeasible.
- `scripts/n50_phase45_overlap_projection.py` 로 union/overlap 을 재계산했다.
  M1 ∪ M5 ∪ Full top-100 은 29/352 이고 192-lane projection 은 약 63
  coords/victim 이지만, channel-mixed information-pool upper bound 로만 취급한다.
- `candidate_null.md` 반영 후 더 강한 정정: raw top-100 candidate-set counts 는
  set-size null 과 구분되지 않는다. Phase 4.5/4.6 은 단일 victim trace 에서
  channel-specific score/candidate artifacts 를 만들 수 있음을 보여주는
  engineering result 로 쓰고, recovery claim 으로는 격하한다.

**관련 산출물**:
- trace: `traces/ntruplus768/phase45/wide_K1_L48_N8.npz`
- analysis: `results/ntruplus768/phase45/wide_K1_L48_N8.{npz,md}`
- latest summary: `results/ntruplus768/phase45/SUMMARY.md`
- overlap/projection: `results/ntruplus768/phase45/overlap_projection.md`
- channel hit-list: `results/ntruplus768/phase45/channel_hitlist.md`
- candidate export: `results/ntruplus768/phase45/candidate_export.{md,npz}`
- candidate pressure: `results/ntruplus768/phase45/candidate_pressure.md`
- candidate null: `results/ntruplus768/phase45/candidate_null.md`

### 4.2 Pending (Phase 5/6, design 단계)

- **Phase 5**: coefficient-domain threshold fallback (별도 attack vector)
- **Phase 6**: SOTP / validity leakage (last-resort fallback)

이 두 phase 는 NTRU+ NTT-domain attack 의 보충적 대안. Phase 4.5 결과
나오면 우선순위 재검토.

### 4.3 NTT-domain attack 의 가능 확장

- **다른 lanes capture** (현재 0, 64, 80, 128 — 192 lanes 중 4 개만).
  각 lane K=8 N=32 ≈ 3h. 192 lanes 모두 → 24일 capture. 비현실적.
  → 4-lane sample 으로 통계적 일반화 충분.
- **K 더 늘림** (현재 lane=0 24 keys, 다른 lanes 8 keys). K 증가 = 회수
  가능 좌표 수 비례 증가. K=32 N=32 lane=0 → 6.25% × 32 ≈ 2 top-100.
  계산 부담 미미, capture 시간 ~12h.
- **G 더 늘림** (현재 G=78 = HW=1+HW=2). G=200 (HW=3 추가) → null max
  √(2·log(Q-1)/G) ≈ 0.29. 회수 가능 풀 확장 가능. Capture 시간 ~10h
  (lane 1개 K=4 N=32 에서).

### 4.4 Sparse recovery / lattice attack (full sk recovery)

현재 Phase 4 multi-victim 은 240 cases 중 17 unique top-100, Phase 4.5/4.6
single-victim aggregate 는 352 cases 중 M1 9 / M5 12 / full-stack 8 top-100.
192-lane single-victim projection 은 raw top-100 count 기준 약 17-26 좌표 /
victim 이지만, candidate-set-size null 에서는 유의하지 않다. 따라서 현재
single-victim aggregate 는 sparse {-1,0,+1} f_coeff recovery 에 쓰기 부족하다.
**현 setup 으로 full sk recovery 불가**. 가능 솔루션:

1. **G 확장 + 더 많은 lanes** → 100+ 좌표 per key 가능, lattice attack
   feasible 가능성 검토 필요.
2. **Profiled CPA (alignment templates)**: per-key 단계에서 profile (sk-known
   data 로 PoI/weight 학습) 후 unknown sk attack. (논문 main result 와는
   별개의 augmented setting).
3. Phase 5/6 (coefficient-domain threshold + SOTP) 와 결합.

## 5. Paper writing 준비 (recommended next session)

### 5.1 paper sections suggestion

1. **Introduction**: NTRU+ KEM, KpqC, side-channel motivations.
2. **Background**: NTT, basemul, chosen-CT, CPA, HW model.
3. **Threat model**: natural `crypto_kem_dec`, IND-CCA, attacker controls
   ciphertext bytes only.
4. **Attack methodology**:
   - Selected-lane chosen-CT: c[lane*4+slot] = γ, others 0.
   - sk-independent timing: predict_poi(lane).
   - Wide-γ G=78 (HW=1+HW=2 set).
   - Dual HW channels: M1, M5.
   - Per-slot PoI calibration.
   - Z-score sum ensemble.
5. **Results**: 240 cases, 2 TOP-1, 17 top-100 (7.08%).
6. **Mechanistic discussion**:
   - N-saturation: corr ceiling.
   - Per-slot drift: lane-dependent.
   - Channel structure: M1 small-|f_c|, M5 large-|f_c|.
7. **Limitations**: Full sk recovery 불가 (~7% partial), CW-Lite SNR 한계.
8. **Future work**: Phase 5/6, lattice attack, profiled extension.

### 5.2 Figures (생성 완료, `results/ntruplus768/phase4/figures/`)

생성된 8종 paper figures:
- **F1_b_vs_fc.png** (n41) — b vs |f_centered| scatter, recovery threshold
- **F2_per_slot_drift.png** (n41) — per-(lane, slot) drift heatmap, lane=80 spread
- **F3_m1_vs_m5.png** (n41) — M1 vs M5 corr scatter, channel separation
- **F4_recovery_by_pipeline.png** (n41) — recovery rate by pipeline
- **F5_lane_snr.png** (n41) — SNR distribution @ predict vs peak per lane
- **F6_ncurve.png** (n44) — minimum-traces N-curve for top-recovered cases
- **F7_saturation.png** (n44) — σ_α saturation mechanism (signal vs noise)
- **F8_dual_channel_fc.png** (n44) — dual channel separation by |f_c| bin

추가로 만들 만한 figures (필요 시):
- chosen-CT NTT layout diagram (개념도)
- per-victim recovery rate (Phase 4.5 결과 후)

## 6. 재시작 instructions (resume session)

세션 재시작 후 다음 단계로 진행 가능:

```bash
cd /home/pacl/Documents/Repository/sca-2026

# 가장 최신 결과 확인
python3 scripts/n24_combined_analysis.py
python3 scripts/n39_full_stack.py
ls -la docs/IDEA.md docs/HANDOFF.md
ls -la traces/ntruplus768/phase3/wideg_*.npz
ls -la results/ntruplus768/phase4/
```

가장 중요한 reference 문서:
1. `docs/IDEA.md` — Phase 4 entry (q8) 가 최종 cumulative
2. `docs/HANDOFF.md` — 이 문서 (paper outline 포함)
3. `host/ntruplus/` — NTRU+ codec/NTT 라이브러리 (수정 거의 없음)

핵심 reference scripts (pipeline 핵심):
- `scripts/n23_wideg_K4_attack.py` — 새 batch attack-valid 평가
- `scripts/n24_combined_analysis.py` — 누적 통계 update (batch 추가 시)
- `scripts/n39_full_stack.py` — full pipeline 실행 (M1+M5 Zsum + per-slot)
- `scripts/n40_full_stack_ncurve.py` — N-curve 분석

## 7. Task list status

```
#1. [completed] Create idea_ntru.md plan + sync overall strategy
#2. [completed] Phase 0a: host/ntruplus/ package + round-trip tests
#3. [completed] Phase 1: natural D capture map (valid CT, multi-key)
#4. [completed] Phase 2: sk-unpack triage (positive control)
#5. [completed] Phase 3: NTT selected-lane oracle (main attack)
#6. [completed] Phase 4: oracle-to-key recovery (NTT branch) ← 2026-05-10 종료
#7. [pending] Phase 5: coefficient-domain threshold fallback
#8. [pending] Phase 6: SOTP / validity leakage (last fallback)
#9. [completed] Phase 0b: smoke + host↔board codec parity
```

Phase 4 complete. Next: paper writing OR Phase 5/6 (별도 attack vector).

## 8. Memory references

자동 메모리 (`/home/pacl/.claude/projects/-home-pacl-Documents-Repository-sca-2026/memory/`):
- `ntruplus_top1_recovery.md` ★ — 2 TOP-1 cases (f=16, f=882)
- `ntruplus_montgomery_channel.md` ★★ — M1+M5 dual channel 발견
- `feedback_diagnostic_no_verdicts.md` — 진단용 결과 ≠ attack-valid 결과
- `threat_model_z_is_diagnostic.md` — only D (full crypto_kem_dec) attack-valid

이 메모리들은 다음 세션에서 자동 로드됨.

---
*Phase 4 종료: 2026-05-10*
*Total experiments: ~40 scripts (n01–n40)*
*Total capture: 7 batches × ~3h = ~21h capture time*
*Paper-grade findings: 5 mechanistic + 2 perfect TOP-1*
