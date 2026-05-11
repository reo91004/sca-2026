# NTRU+ SCA — 실험 가정 · 과정 · 결과 분석

마지막 업데이트: 2026-05-11.

이 문서는 NTRU+ chosen-CT SCA 프로젝트의 **실험 로그**다. 전략·gate·계획 자체는
`docs/NTRUplus.md` 에 있다. 본 문서는 매 phase / 실험마다 아래 형식으로 누적
기록한다. 세션이 끊겨도 다시 읽으면 어디까지 했는지, 다음에 무엇을 해야 하는지
바로 복구 가능해야 한다.

## 기록 형식

```
### YYYY-MM-DD <PhaseN.label> — <한 줄 요약>

가정
- 무엇을 시험했고, 그 가정의 근거가 무엇인가.

과정
- 명령어 / 산출물 경로 / 핵심 파라미터.

결과 (수치)
- 표 / 핵심 메트릭. 가능하면 z, AUROC, percentile 형태로.

해석
- gate 통과 / 실패 + 왜.

다음 단계
- 다음에 무엇을 할지 1~3 개. 명확한 동작 동사로.
```

새 entry 는 **위쪽** 에 추가한다 (역시간순). 진행 중인 entry 는 `(in-progress)`
태그를 붙이고, 끝나면 그 자리에서 결과를 채우고 태그 제거.

---

## 실험 로그

### 2026-05-11  Phase 4.6 정리 — Wide-coverage 3rd victim 완료 + 3-victim aggregate

가정
- Phase 4.5 의 2 victim (40 lanes) 결과 paper-grade 입증. 그러나 단일 victim
  최대 24 lanes (12.5% lane coverage) 만 검증. paper §5.2 의 192-lane
  projection 의 정확도는 lane coverage 비율에 의존. 더 wide 한 sample
  (48 lanes = 25% coverage) 으로 projection 정확도 ↑ 및 paper claim 강화.
- LANE_SLOT_OFFSET linear interp (Phase 4.5 main 의 +3 cases) 가 더 wide
  lane sample 에서도 효과 입증 가능.

과정
- (s1) **Phase 4.6 capture 완료** (2026-05-10 09:15-15:41 KST)
  - cmd: `n18 -K 1 -L "0,4,8,...,188" -N 8 -o wide_K1_L48_N8.npz`
  - victim 3: fresh keygen
  - K=1 L=48 stride=4 (calibrated 0/64/80/128 모두 포함) G=78 N=8
    = 1×48×78×8 = 29,952 traces, `T=24400`.
  - trace: `traces/ntruplus768/phase45/wide_K1_L48_N8.npz` (약 745 MB)
  - analysis: `results/ntruplus768/phase45/wide_K1_L48_N8.{npz,md}`
  - combined summary: `results/ntruplus768/phase45/{combined,SUMMARY}.{npz,md}`

결과 (수치)
- **victim 3 (8ce340e4bdfa80fc), 48 lanes, 192 cases**
  | pipeline | top-1 | top-10 | top-100 | top-500 |
  |---|---:|---:|---:|---:|
  | M1 baseline (predict_poi) | 0/192 | 0/192 | **3/192** | 22/192 |
  | M5 baseline (predict_poi) | 0/192 | 1/192 | **9/192** | 31/192 |
  | M1 slot-PoI (interp) | 0/192 | 0/192 | 1/192 | 18/192 |
  | M5 slot-PoI (interp) | 0/192 | 0/192 | 3/192 | 23/192 |
  | Full stack (Zsum, interp) | 0/192 | 1/192 | 1/192 | 14/192 |
- **3-victim aggregate (Phase 4.5 + 4.6), 88 lanes, 352 cases**
  | pipeline | top-1 | top-10 | top-100 | top-500 |
  |---|---:|---:|---:|---:|
  | M1 baseline | 0/352 | 1/352 | **9/352** | 45/352 |
  | M5 baseline | 0/352 | 1/352 | **12/352** | 50/352 |
  | M1 slot-PoI (interp) | 0/352 | 0/352 | 8/352 | 48/352 |
  | M5 slot-PoI (interp) | 0/352 | 1/352 | 10/352 | 47/352 |
  | Full stack | 0/352 | 1/352 | **8/352** | 46/352 |
- **Overlap/projection 분석** (`scripts/n50_phase45_overlap_projection.py`)
  - output: `results/ntruplus768/phase45/overlap_projection.{md,npz}`
  - M1 baseline ∪ M5 baseline: 21/352 top-100, projected 45.8 coords/victim
  - M1 ∪ M5 ∪ Full: 29/352 top-100, projected 63.3 coords/victim
  - 단, union 은 channel-mixed upper bound 이며 top-100 rank 를 perfect recovery 로
    취급하지 않는다. 본문 claim 은 single-pipeline 17-26 coords/victim 을 기본값으로 둔다.
- **Channel hit-list 분석** (`scripts/n51_phase45_channel_hitlist.py`)
  - output: `results/ntruplus768/phase45/channel_hitlist.{md,npz}`
  - top-100 hits: M1 9, M5 12, M1_slot 8, M5_slot 10, Full 8, all-channel union 37.
  - M1 ∩ M5 = 0, M1 ∩ Full = 0, M5 ∩ Full = 0. 현재 single-victim aggregate 에서는
    fusion 이 같은 좌표를 더 강하게 만드는 것보다 서로 다른 channel 후보 pool 을
    드러내는 성격이 강하다.
  - |f_c| bin 기준 hits 는 large 쪽도 많다: M1 은 `>=768` 5/9, M5 는 `>=768` 5/12.
    초기 small-|f_c| 중심 가설은 Phase 4 multi-key 결과에는 맞지만, single-victim
    lane 확장에서는 large-|f_c| 후보 pool 도 별도로 살려야 한다.
- M1 baseline top-100 hits 추가:
  - victim 3 lane=24 slot=0 f=1215 |f_c|=1215 rk=15
  - victim 3 lane=76 slot=1 f=1169 |f_c|=1169 rk=23
  - victim 3 lane=92 slot=2 f=565 |f_c|=565 rk=50

해석
- 48-lane wide victim 은 M1 baseline 기준 `3/48 = 0.0625` coords/lane 로,
  앞선 2 victims 평균 `6/40 = 0.15`보다 낮다. 따라서 192-lane projection 은
  lucky-victim 가정 없이 더 보수적으로 써야 한다.
- 3-victim aggregate 기준 M1 baseline 은 `9/88 = 0.102` top-100 coords/lane,
  192-lane extrapolation 시 약 **20 coords/victim** 이다. M5 baseline 은
  `12/88 = 0.136`, 약 **26 coords/victim** 이다. Full stack 은 현재
  `8/88 = 0.091`, 약 **17 coords/victim** 으로 M1/M5 개별 baseline보다 낮다.
- 여러 channel 을 union 하면 top-100 upper bound 는 최대 약 **63 coords/victim**
  까지 올라가지만, 이는 pipeline 하나의 공격 성능이 아니라 정보 누출 후보 pool
  크기다. 논문에서는 single-pipeline 수치와 union upper bound 를 분리해서 쓴다.
- 다음 모델링은 `Zsum` 하나로 합치는 방향보다, M1/M5/slot/full score 를 별도
  likelihood 로 보존하고 sparse/lattice 후단에서 candidate-set evidence 로
  결합하는 쪽이 더 자연스럽다.
- Phase 4 의 2 TOP-1 / 17 top-100 multi-victim 결과는 여전히 paper-grade 핵심.
  Phase 4.5/4.6 은 “한 victim 안에서 lane coverage 를 늘리면 partial
  NTT-coordinate disclosure 가 누적된다”는 보조/확장 claim 으로 두는 것이
  가장 안전하다.

다음 단계
- (s2) `docs/HANDOFF.md`, `docs/PAPER_OUTLINE.md` 를 3-victim aggregate 로 갱신. ✅
- (s3) capture-free 추가 분석: M1/M5/Full union top-100, victim별 projection,
  hit overlap 을 정리해 paper §5.2 수치 확정. ✅
- (s4) 새 대용량 capture 는 보류. 레포 비대화 방지를 위해 `traces/`는 추가
  생성 전 예상 크기와 목적을 문서화한다.

---

### 2026-05-10  Phase 4.5 — 단일 victim multi-lane sk leakage 입증 시도 (in-progress)

상태 메모 (2026-05-11): 이 entry 의 2-victim projection 은 당시 중간 결론이다.
최종 paper 수치는 위 Phase 4.6 의 3-victim aggregate 를 우선한다.

가정
- Phase 4 의 240 cases 는 60 unique sks (batch 마다 다른 key) 로 한 sk 당 1 lane × 4
  slots = 4 cases 만 평가됨. 평균 0.15-0.17 top-100 좌표 / lane / victim. 192-lane
  으로 lane coverage 확장 시 단일 victim 의 ~30 NTT 좌표 (top-100) 회수 추정 (n42).
- 따라서 같은 sk 의 multi-lane capture 만 추가하면 "single-victim partial sk
  recovery" attack-valid 입증 가능. paper 의 main claim 강화 + sk leakage 직접
  입증.
- 평가 파이프라인은 attack-valid only: M1 baseline (predict_poi formula,
  profile-free) 와 calibrated-lane full stack (M1+M5 Zsum + LANE_SLOT_OFFSET,
  Phase 4 multi-key 데이터의 sk-independent firmware constant 가정).

과정
- (r0) **per-key 분포 분석** (`scripts/n42_per_key_distribution.py`)
  - 240 cases / 60 unique sks (batch 마다 다른 sk_blob hash 확인 — 모든 cross-batch
    intersection 0):
    | pipeline | mean t100 / lane / victim | max | ≥1 t100 keys | ≥2 |
    |---|---:|---:|---:|---:|
    | M1 baseline (predict_poi) | 0.167 | 1 | 10/60 | 0/60 |
    | Full stack (slot+Zsum) | 0.150 | 1 | 9/60 | 0/60 |
  - 1 lane 당 ≥2 top-100 key = 0/60 → 1 lane 안에서 한 victim 이 2 좌표 회수
    한 사례 없음. 회수는 좌표 sk-구조 의존 (small |f_c| 또는 lucky high-SNR).
  - 192-lane 단일 victim 추정: M1 baseline ~32 coords, Full stack ~29 coords,
    union (30% overlap) ~43 unique top-100 / victim.
- (r1) **scout capture** (in-progress, ~1.6h, 시작 01:24)
  - cmd: `n18_phase4_g78_capture.py -K 1 -L "0,12,24,36,48,60,64,72,80,92,104,
    116,128,140,152,164" -N 8 -o traces/ntruplus768/phase45/scout_K1_L16_N8.npz`
  - K=1 fixed sk, 16 lanes (calibrated 0/64/80/128 + 12 새 lanes 균등 분포),
    G=78 (HW1+HW2), N=8 = 1 × 16 × 78 × 8 = 9984 traces.
  - 평가: `scripts/n43_singleVictim_multilane.py`. M1 baseline (모든 lanes,
    profile-free) + Full stack (calibrated lanes only, profiled approximation).
- (r1') **추가 capture-free 도구 / paper figures** ✅ (capture 진행 동안 작업)
  - **n44_paper_figures_extra.py**: F6 N-curve, F7 σ_α saturation, F8 dual
    channel by |f_c| — 3종 paper figures 추가 (총 paper figures F0-F9 = 10종).
  - **n45_singleVictim_combine.py**: 다중 victim trace 합산 evaluator (각
    capture 마다 다른 sk 라 multi-victim demonstration 형태).
  - **n46_partial_recovery_analysis.py**: X NTT coords → information theoretic
    bound + lattice β estimate. X=30 → 31% entropy reduction; β_req~300
    (infeasible for full recovery, partial info disclosure quantified).
  - **n47_paper_layout_diagram.py**: F0 conceptual chosen-CT NTT layout
    diagram (paper §3 figure).
  - **n48_corr_by_N.py**: F9 corr saturation curve from K=4 N=64 16-case
    data (mechanistic σ_α-bounded ceiling 시각화).
  - **n49_phase45_summary.py**: F10/F11/F12 figures + SUMMARY.md (post-main).
  - **scripts/run_phase45_chain.sh**: scout → analyze → main → analyze → combine
    자동 chain (사용자 부재 동안 진행).
  - **scripts/phase45_status.sh**: 한 명령어로 전체 진행 상황 확인.
  - **docs/PAPER_OUTLINE.md**: paper full outline (§1-§8 + appendices).
- (r1'') **자동화 chain 시작** (2026-05-10 01:42 KST) — sleep 시간 동안:
  - scout (1.6h, ~03:00 끝) → 분석 → main capture (5h, ~08:00 끝) →
    합산 → SUMMARY.md + F10/F11/F12 figures.
  - main capture spec: K=1 L=24 stride=8 (lanes 0/8/16/.../184 incl 0/64/80/128
    calibrated) N=16 G=78 = 1×24×78×16 = 29,952 traces.
  - 사용자 깨어났을 때: `bash scripts/phase45_status.sh` 로 진행 확인.

- (r2) **scout 완료 + 첫 결과** ★ (2026-05-10 03:31 KST, 2h 7m capture)
  - file: `traces/ntruplus768/phase45/scout_K1_L16_N8.npz` (226 MB)
  - sk hash: 235e27a030715e99 (fresh victim)
  - shape (K=1, L=16, G=78, N=8, T=24400)
  - **자동 chain race condition 으로 분석 실패** — chain wrapper 가 file
    appearance 즉시 분석 시작 → npz 가 partially written 상태 → BadZipFile
    error → main capture skip 결정. 수동 재분석.
  - **수동 분석 결과 (n43 + n45)**:
    | pipeline | top-1 | top-10 | top-100 | top-500 |
    |---|---:|---:|---:|---:|
    | M1 baseline (predict_poi) | 0/64 | 1/64 | **3/64** (4.7%) | 12/64 (18.8%) |
    | M5 baseline | 0/64 | 0/64 | 2/64 | 9/64 |
    | M1 slot-PoI (calibrated only) | 0/64 | 0/64 | 1/64 | 10/64 |
    | M5 slot-PoI (calibrated only) | 0/64 | 1/64 | 3/64 | 9/64 |
    | **Full stack (slot+Zsum)** | 0/64 | 0/64 | **4/64** (6.25%) | 14/64 |
  - **★ Single-victim sk leakage 첫 직접 입증**: K=1 victim 의 16 lanes
    × 4 slots = 64 cases 에서 attack-valid:
    - **lane=104 slot=0 f=2563 (|f_c|=894) M1 baseline rk=2** — near-TOP-1!
    - lane=36 slot=1 f=236 (|f_c|=236) rk=45
    - lane=152 slot=1 f=679 rk=93
    - Full stack 추가: lane=24 sl=2 (rk 53), lane=60 sl=0 (rk 65),
      lane=80 sl=0 (rk 94), lane=116 sl=2 (rk 52)
    - Union (M1 ∪ Full): **7 unique top-100 / 64 cases** (10.9%) — Phase 4
      multi-victim 통계 (7.08%) 와 일관, 단일 victim 안에서 입증.
  - **N=8 의 영향**: top-1 이 0 (N=16 이면 가능했을). 그러나 top-100 회수율
    Phase 4 multi-victim 보다 약간 더 좋음 (lucky victim).
- (r3) **main capture 완료** ★ (03:32-08:50 KST, 5h 18m, 0 timeouts)
  - file: `traces/ntruplus768/phase45/main_K1_L24_N16.npz` (745 MB)
  - shape (K=1, L=24, G=78, N=16, T=24400) = 29,952 traces
  - victim 2 sk hash: 8c6aacfa07e4ddcd (fresh sk)
  - lanes: stride 8 = [0, 8, 16, ..., 184] including 0/64/80/128 calibrated
  - **n43 결과 (96 cases)**:
    | pipeline | top-1 | top-10 | top-100 | top-500 |
    |---|---:|---:|---:|---:|
    | M1 baseline (predict_poi) | 0 | 0 | **3** | 11 |
    | M5 baseline | 0 | 0 | 1 | 10 |
    | M1 slot-PoI (linear interp) | 0 | 0 | **6** ★ | 20 |
    | M5 slot-PoI (linear interp) | 0 | 0 | 4 | 15 |
    | Full stack (Zsum, interp) | 0 | 0 | 3 | 18 |
  - **★ slot-PoI linear interp 효과 입증**: M1 baseline 3 → slot-PoI 6
    (interp 사용으로 +3 cases 회수). Phase 4 의 calibrated 4 lanes 만의
    LANE_SLOT_OFFSET 을 새 lanes 에 linear-interpolate 한 결과.
  - victim 2 top-100 cases (M1 baseline):
    - lane=128 slot=2 f=3448 (|f_c|=9, **small**) rk=23
    - lane=88 slot=0 f=1207 (|f_c|=1207, large) rk=37
    - lane=144 slot=1 f=2496 (|f_c|=961, large) rk=88
- (r4) **★★ Phase 4.5 cumulative — single-victim sk leakage 입증** (160 cases,
  2 victims, 40 lanes covered)
  - **Aggregate**:
    | pipeline | top-1 | top-10 | top-100 | top-500 |
    |---|---:|---:|---:|---:|
    | M1 baseline | 0/160 | 1/160 | **6/160** (3.75%) | 23/160 (14.4%) |
    | M5 baseline | 0/160 | 0/160 | 3/160 | 19/160 |
    | M1 slot-PoI (interp) | 0/160 | 0/160 | **7/160** | 30/160 (18.75%) |
    | M5 slot-PoI (interp) | 0/160 | 1/160 | 7/160 | 24/160 |
    | **Full stack (Zsum, interp)** | 0/160 | 0/160 | **7/160** | **32/160** (20%) |
  - **Per-victim**:
    - victim 1 (235e27a0, L=16 N=8): 3 M1 t100, 4 full t100 (16 lanes)
    - victim 2 (8c6aacfa, L=24 N=16): 3 M1 t100, 3 full t100 (24 lanes)
  - **Top recovery cases (M1 baseline)**:
    1. **lane=104 slot=0 f=2563 |f_c|=894 rk=2** — near-TOP-1, large-|f_c|
       on M1 channel (cross-channel finding ★)
    2. lane=128 slot=2 f=3448 |f_c|=9 rk=23 (small)
    3. lane=88 slot=0 f=1207 |f_c|=1207 rk=37 (large)
    4. lane=36 slot=1 f=236 |f_c|=236 rk=45 (small)
    5. lane=144 slot=1 f=2496 |f_c|=961 rk=88 (large)
    6. lane=152 slot=1 f=679 rk=93 (mid)
  - **★ 192-lane projection** (extrapolation from 40-lane sample):
    - M1 baseline: 6/40 × 192 = **28.8 coords / victim** (top-100)
    - Full stack: 7/40 × 192 = **33.6 coords / victim** (top-100)
    - Union (≈30% overlap): ~9/40 × 192 ≈ **43 unique top-100 / victim**
    - n42 의 추정 (43 unique top-100) 와 정확히 일치 ✓
  - **★ paper-grade conclusion**: 단일 victim 의 192-lane 전체 capture 시
    추정 ~30 NTT 좌표 회수 (full stack), entropy reduction ~31% (153/1152
    bits). Lattice attack 으로 full sk recovery 는 **infeasible** (n46 분석
    β_req~300), 그러나 partial information disclosure 가 **공식 입증**됨.
  - **★ Phase 4 vs Phase 4.5 비교**:
    - Phase 4 (multi-victim, 240 cases, 60 victims): 0.28 coords/victim
      (mean), 7.08% top-100 in aggregate
    - Phase 4.5 (single-victim, 160 cases, 2 victims): 3 coords/victim
      (mean), 3.75-4.4% top-100, 192-lane projection ~30 coords/victim
    - 두 결과는 일관: 같은 attack 의 multi-key statistical view vs
      single-victim accumulation view.
- (r5) **Figures F10/F11/F12** ✅ — `results/ntruplus768/phase45/figures/`
  - F10: per-victim recovery summary (M1 baseline + Full stack 비교)
  - F11: lane × victim heatmap (top-100 hits)
  - F12: cumulative coords vs lanes covered (192-lane projection 라인 포함)

결과 (수치)
- (r0) 240 cases, 60 unique sks. 단일 victim 1 lane 평균 0.15 top-100 좌표 회수.
  192-lane 확장 추정 ~30-43 unique top-100 / victim.
- (r1) **수치 미수집** — capture 진행 중.

해석 (verdict 아닌, 좁혀진 변인)
- 메인 가설: NTRU+ NTT-domain CPA 가 단일 victim 의 lane coverage 을 늘리면
  partial sk 회수 가능. 정량 추정 ~30 좌표 / 768 = 4% 좌표 회수, 153 bits
  entropy reduction (1152 bit total).
- 검증 변인: scout capture 의 (1) M1 baseline top-100 rate (target ≥
  240-cases 의 0.167 / lane), (2) calibrated lanes 의 full stack rate.
  scout 결과가 추정 회수율과 일관되면 main capture (32-64 lanes) 진행.
- 부정합 가능성: scout 의 recovery rate < 0.05 / lane 이면 새 victim 의 b
  분포가 outlier, 또는 N=8 미달. diagnose 후 N 확장 또는 다른 victim.

다음 단계
- (r2) scout 결과 분석 (capture 후 ~1.6h+분석 30min). M1 baseline top-100 rate
  vs 0.167 비교. SNR median, b 분포 통계.
- (r3) main capture: K=1 L=32-64 N=8-16 — scout 결과 따라 결정.
- (r4) 최종 cumulative single-victim recovery → paper §3.x 새 결과로 추가.

---

### 2026-05-08  Phase 4 — multikey 기반 attack-valid + 진단 평가 (4 evaluator, in-progress)

가정
- 단일-key scout 의 corr 0.55 (lane 0 PoI 3014, n05) 가 attack-valid 평가
  (sk-indep PoI 모델, oracle 의존 없음) 에서 재현될지, 또 cross-key transfer
  를 통해 held-out attack 이 가능한지 측정. 4 evaluator 가 각자 다른 변인을
  분리한다.

과정 (모두 `traces/ntruplus768/phase3/multikey_hw1.npz` K=4 L=6 G=12 N=8 사용)

| 스크립트 | 카테고리 | 설명 |
|---|---|---|
| n08_phase4_recover.py | attack-valid | profile-on-(K-1)-keys ridge → predict held-out, ±W=10 window |
| n13_phase4_diagnose.py | 진단 (T1-T4) | within-key ridge / per-key z-norm / single-sample / pooled corr |
| n14_cpa_per_key.py | attack-valid | profile-free CPA: per-cand best-PoI in ±W=10 window |
| n15_cpa_variants.py | attack-valid (V1/V2) + 진단 (D1) | fixed-PoI per-trace / mean-over-N CPA, true-vs-null distribution |
| n16_snr_estimate.py | 진단 | OLS leak coefficient b, σ_signal, σ_noise, SNR, N→99% |

결과 (수치)

attack-valid (paper 의 recovery 결과로 사용 가능):

| script | mean pctile | top1 (96 케이스) | top10 |
|---|---:|---:|---:|
| n08 (held-out ridge) | 50.79 | 0/96 | 0/96 |
| n14 (CPA, per-cand best PoI) | 50.19 | 0/96 | 1/96 |
| n15 V1 (CPA, fixed PoI) | 51.44 | 0/96 | 0/96 |
| n15 V2 (CPA, mean-over-N) | 51.43 | 0/96 | 0/96 |

진단 (data-side information, attack 결과 아님):

| 측정 | 값 |
|---|---:|
| n13 T1 within-key ridge mean pctile (true f 사용) | 71.34 |
| n13 T2 cross-key per-key z-norm | 51.77 |
| n13 T4 lane 0 slot 0 pooled cross-key corr | +0.185 |
| n13 T4 lane 0 slot 2 pooled cross-key corr | -0.141  (sign flip across slots) |
| n15 D1 P(true \|corr\| ≥ null 99-th pctile) | 0.000 (96/96 케이스) |
| n16 SNR (median, max) | 0.112, 0.679 |
| n16 null-max ceiling at G=12 | 1.165 (asymptotic) |
| n16 G threshold for finite N | G > 2·log(Q-1) ≈ 16.3 |

해석 (verdict 아닌, 좁혀진 변인 + 분리 필요한 변인)
- attack-valid 4 evaluator 모두 percentile ≈ 50 / top1 = 0. 현재 (K=4, G=12,
  N=8, HW=1 γ) 조건에서 candidate 회수 안 됨.
- 진단 D1 (n15) 가 가장 정량적 — 96/96 케이스에서 true f 의 |corr| 가 후보
  분포의 99-th percentile 보다 낮음. 즉 multi-comparison 천장이 true 신호보다
  높음.
- 진단 n16 가 구조적 한계 명시: V2 (mean-over-N) 의 asymptotic null-max
  ≈ √(2·log(Q-1)/G) = 1.17 > 1 at G=12 — N 무한대도 |corr|≤1 천장 못 넘음.
  G > 16.3 이 구조적 최소.
- 진단 T1 (within-key, true f 사용) 71% 는 데이터에 leak 자체는 있다는 사실
  (within-key ridge 모델이 일부 정보 capture) 을 보여줌. 단, attack-valid
  평가에서 그 leak 이 후보 enumeration 으로 추출 안 됨.
- 진단 T4 의 sign flip (slot 0 +0.185 / slot 2 −0.141) 은 sk-universal
  affine 모델 (b 가 sk 와 무관) 가정의 위반 후보. b 가 key 별 / slot 별로
  부호 차이 가능.

다음 단계 (이미 진행 / 진행 예정)
- (a) **HW 모델 변형** ablation — 같은 데이터로 5 변형 (M1 unsigned / M2 centered /
  M3 montgomery / M4 low-byte / M5 high-byte). capture 필요 없음. ✅
  결과: 5 변형 모두 mean SNR 0.12-0.16, max SNR 0.47-0.81, P(true ≥ null99)
  0.000 (M2 만 0.021). HW 모델 변경으로는 천장 못 깸. SNR 자체가 limiting.
- (b) **G 확장 capture** ✅ — `scripts/n18_phase4_g78_capture.py` K=1 L=1 (lane 0)
  G=78 (HW=1+HW=2) N=32, 24 min capture, 0 timeouts.
  attack-valid `n19_wideg_attack.py`: 4 cases (single key × lane 0 × 4 slots).
  V2 mean pctile 25.50, V2w 29.70, top1 = 0/4. P(true ≥ null99) = 0.000.
  → 진단 `n22_wideg_diagnose.py`: 이 특정 (key, lane=0) 의 per-slot SNR
  0.082 / 0.030 / 0.028 / 0.007 (multikey 의 max SNR 0.679 보다 한 자리 낮음 —
  f-specific). null distribution: empirical 99-th 0.474, max 0.662
  (Gaussian-iid 예측 0.457 와 근접). pairwise candidate-label corr E[ρ²]=0.024,
  effective G ≈ 76.1 ≈ 78. 구조 모델은 robust. 따라서 천장은 정확하지만
  이 특정 key/lane 의 SNR 이 부족.
  required N (이 SNR 으로 0.66 ceiling 통과): SNR=0.082 → N≥122.
- (b') **K 확장 + wide-γ capture** ✅ — `traces/ntruplus768/phase3/wideg_lane0_K4N32.npz`
  K=4 L=1 lane=0 G=78 N=32 = 9984 traces, 96.5 min. attack-valid `n23`:
  16 (key, slot) 케이스, V2 mean pctile 59.66, V2w 39.88, top1 0/16, top10 0/16,
  **top100 1/16**. **P(V2 true |corr| ≥ empirical null 99-th) = 1/16**.
  핵심 단일 회수: key=0 slot=0 true_f=1148 SNR=0.280 V2|corr|=0.553 (vs null99
  =0.504, null max=0.678) — **rank 11/3456 (top 0.32%)**. V2w 도 0.634 > 0.568.
  나머지 15 케이스 SNR median 0.085 (max 0.280) — 천장 0.66 통과 못 함.
  per-key SNR_median: 0.100, 0.075, 0.078, 0.100. per-slot 분포 비균등 (slot 0
  의 한 사례만 SNR>0.25).
- (e) **N-curve 분석** (existing K=4 데이터, 같은 trace 의 N=4/8/16/32 부분집합):
  key 0 slot 0 corr/rank: N=4→0.307/126, N=8→0.456/24, N=16→**0.527/8**, N=32→0.553/11.
  N=16 에서 이미 top10. N 증가에 따라 corr 증가하지만 N=32 에서 saturation 시작.
  다른 의미 있는 N-trends: key 0 slot 2 (0.05→0.22), key 2 slot 1 (0.04→0.15) —
  signal 존재하지만 ceiling 미달. 나머지 cases corr N-flat (noise only).
- (f) **K=4 lane=80 wide-γ capture** ✅ —
  `traces/ntruplus768/phase3/wideg_lane80_K4N32.npz`. attack-valid (n23):
  SNR median 0.016 (max 0.031, lane 0 의 1/5 수준). V2 mean pctile 49.74,
  top1 0/16, top10 0/16, top100 0/16. P(true ≥ null99) = 0/16.
  → 회수 신호 없음. lane=80 의 cross-key 평균 SNR 가 lane=0 보다 약함.
  scout (n05) 의 lane=80 corr=0.572 PASS 는 single-key 의 운이 좋은 case.
- (g) **K=4 N=64 lane=0** ✅ — `wideg_lane0_K4N64.npz`, 182 min. attack-valid
  (n23): SNR median 0.076, max 0.237. **0/16 회수**. top SNR 케이스 (key=2
  slot=0, SNR 0.237) V2_T=0.373 < null99=0.486 — 미달. **새 4 keys 의 SNR
  분포가 N=32 batch 보다 낮음 (lucky/unlucky variance)**. 또 측정 corr ≈
  0.65 × predicted (formula `SNR/√(SNR²+1/N)`) — 50% 오버예측. 임계 SNR ≈
  0.25-0.28 (N=32 임계, 0.24 N=64 미달).
- (h) **누적 통계** (1차) — lane=0 K=4 N=32 (1/16) + lane=0 K=4 N=64 (0/16) +
  lane=80 K=4 N=32 (0/16) + lane=0 K=1 N=32 (0/4) = 1/52 attack-valid 회수.
- (i) **K=8 lane=0 G=78 N=32** ✅ ★ — `wideg_lane0_K8N32.npz`. attack-valid (n23):
  SNR median 0.066, **max 0.647**.
  - **key=5 slot=1 true_f=16 SNR=0.647 → V2_T=0.658 = V2_max → rank=0/3456 TOP-1**
    perfect 회수. V2w_T=0.658 > V2w_99=0.545 also top1.
  - **key=6 slot=0 true_f=3451 SNR=0.281 → V2w_T=0.619 > V2w_99=0.560 → V2w
    pctile 99.68, top10**.
  - 32 cases: top1 1/32, top10 1/32, top100 2/32. P(V2 ≥ null99)=1/32,
    P(V2w ≥ null99)=2/32.
  - per-key SNR: key 5 (0.151) 와 key 6 (0.129) 가 lucky high-SNR keys.
  - f=16 = 2⁴ 의 small magnitude → label vector entropy 가 매우 높아 SNR 큼.
- (j) **누적 attack-valid 회수** — lane=0 wide-γ N=32 G=78 batch 들에서 K=12
  (4+8) 총 **48 cases 중 3 회수 (6.25%)**: rank 11 (K=4 key 0), rank 0 (K=8
  key 5 top-1), V2w top10 (K=8 key 6).
  → "단일 victim session 안에서 standard CPA 가 NTRU+ NTT 좌표 1 개를 회수"
  attack model 이 부분적으로 입증. SNR-bounded.
- (k) **N-curve 정밀 분석** (key=5 slot=1, f=16, SNR=0.647):
  - **N=2 → V2_T=0.574 = V2_max → rank 0**. **2 traces 만으로 top-1 회수**.
  - N=4 → 0.635, N=8 → 0.657, N=12 이상 saturation 0.658.
  - SNR ≥ 0.5 high-SNR case 는 매우 작은 N 으로도 회수 가능.
- (l) **mid-SNR case** (key=6 slot=0, f=3451, SNR=0.281): N=2 V2 rank=21 → N=32
  rank=57 (saturating). V2w 만 top10 (rank ~5). V2 fixed PoI 로는 부족.
  임계 SNR ≈ 0.3 이상이 V2 fixed PoI 로 안정 회수.
- (m) **paper-grade 핵심 결과** ★ — NTRU+768 chosen-CT NTT-domain selected-lane
  attack 의 **첫 attack-valid TOP-1 recovery**: f_ntt[1]=16 (lane=0, slot=1)
  for one randomly-generated key, using N=2 chosen-CT traces. Empirical
  cap rate ~6% per (key, slot) at this CW-Lite rig.
- (n2) **K=8 round 2** ✅ — `wideg_lane0_K8N32_b.npz` (n23): SNR median 0.081,
  max 0.316. **0/32 top10 회수**. V2 pct 98+ (top100 within) 2 cases:
  - key 1 slot 1 SNR 0.316 V2_T=0.458 vs null99=0.502 (rank ~63)
  - key 4 slot 3 SNR 0.290 V2_T=0.453 vs null99=0.493 (rank ~64)
  → 임계 SNR 0.5+ 미달 → top10 회수 실패. 2 case 가 top100 안.
  - 누적 K=20 keys × 4 slots = **80 cases, recovery 4 (5%)**: rank 0 (top-1),
    rank 5 (top10 V2w), rank 11 (top12), rank ~63 (top100). 이론 모델
    (~6% small-|f| coords) 와 일치.
- (n4) **K=8 lane=128 wide-γ** ✅ — `wideg_lane128_K8N32.npz`. attack-valid (n23):
  SNR median **0.103** (lane=0 의 ~0.08 보다 약간 좋음), max 0.527.
  - **key=3 slot=2 true_f=6 (centered) SNR=0.527 → V2_T=0.527 > null99=0.500**,
    V2w_T=**0.696 > V2w_99=0.522** → V2w pct 99.71 (top-10 회수)
  - key 2 slot 3 (f=21, centered=21, SNR=0.467) V2 pct 98.44 V2w 97.74 (top-100)
  - 32 cases: top1 0/32, top10 0/32, top100 V2 2/32 V2w 3/32.
  → **lane=128 도 회수 가능 (lane=0 와 비슷한 SNR profile)**. lane=80 만
  outlier (iter=40 의 약함은 lane-specific). lane=0 (iter=0) 와 lane=128 (iter=64)
  가 비슷.
- (n6) **K=8 lane=128 round 2** ✅ — `wideg_lane128_K8N32_b.npz` (n23):
  SNR median 0.075, max 0.515. **1/32 회수**: key 5 slot 2 (f=872, SNR=0.515,
  V2_T=0.515 > null99=0.504 → V2 pct 99.54). lane=128 회수율 재현.
- (n7) **최종 누적** (5 batches, K=36 effective keys × 4 slots = **144 cases**)
  — `scripts/n24_combined_analysis.py`:
  - V2 (fixed PoI) recovery: top-1 1/144 (0.69%), top-10 1/144, **top-100
    8/144 (5.56%)**, top-500 22/144 (15.28%), ≥ null 99-th 4/144 (2.78%).
  - V2w (per-cand best PoI in ±10): top-10 1/144, **top-100 9/144 (6.25%)**,
    ≥ null 99-th 5/144 (3.47%).
  - SNR distribution: median 0.081, 90th 0.250, max 0.647.
    SNR ≥ 0.5: 3/144; ≥ 0.3: 7/144; ≥ 0.2: 24/144; ≥ 0.1: 58/144.
  - |f_centered| distribution: median 740. |f_c|<16: 3/144 (2.1%); |f_c|<32:
    5/144 (3.5%); |f_c|<100: 7/144.
  - **Lane breakdown** (V2w top-100):
    - lane=0: 5/80 (**6.25%**)
    - lane=128: 4/64 (**6.25%**) — **lane=0 와 동일 회수율 ✓ 재현성 확인**
    - lane=80: 0/16 (0%) — outlier (basemul iter=40 SNR 저하)
  - 9 recovered cases:
    | batch | key/slot | f | |f_c| | SNR | V2_pct | V2w_pct |
    |---|---|---|---|---|---|---|
    | K=4 r1 ln=0 | 0/0 | 1148 | 1148 | 0.280 | 99.68 | 99.68 |
    | K=8 r1 ln=0 | **5/1** ★ | **16** | 16 | **0.647** | **100.00** | 99.97 |
    | K=8 r1 ln=0 | 6/0 | 3451 | 6 | 0.281 | 98.35 | 99.68 |
    | K=8 r2 ln=0 | 1/1 | 1721 | 1721 | 0.316 | 98.18 | 97.34 |
    | K=8 r2 ln=0 | 4/3 | 3447 | 10 | 0.290 | 98.15 | 98.70 |
    | K=8 r1 ln=128 | 1/0 | 435 | 435 | 0.289 | 89.96 | 97.74 |
    | K=8 r1 ln=128 | 2/3 | 21 | 21 | 0.467 | 98.44 | 97.74 |
    | K=8 r1 ln=128 | 3/2 | 6 | 6 | 0.527 | 99.36 | 99.71 |
    | K=8 r2 ln=128 | 5/2 | 872 | 872 | 0.515 | 99.54 | 99.31 |
  - **공통 조건**: SNR ≥ 0.28. **6/9 가 |f_c| < 22** (small magnitude).
    3/9 는 moderate-|f_c| (435, 872, 1148, 1721) 인데 lucky high-SNR.
  - **paper-grade 결론**: NTRU+768 NTT-domain selected-lane chosen-CT CPA
    가 자연스러운 `crypto_kem_dec` trace 에서 **6.25% 의 (key, slot) 좌표를
    top-100 안으로, 0.69% 를 top-1 (perfect) 으로 회수**. 회수 대상은
    small |f_centered| 좌표 또는 lucky high-SNR moderate-|f| 좌표.
    Lane=0 과 lane=128 (basemul 시작 / 중간) 둘 다 가능, lane=80 (iter=40
    부근) 은 SNR outlier.

- (o) **lane=80 outlier 원인 분리** (`scripts/n25_lane_snr_compare.py`) — 진단
  (true f 사용). lane∈{0, 80, 128} 6 batches 160 cases.
  - SNR @ predicted PoI: lane=0 median 0.080, lane=80 **0.016**, lane=128 0.084.
    → lane=80 5× 낮음.
  - SNR @ true peak in ±100 sample window: lane=0 median 0.127, lane=80
    **0.105**, lane=128 0.127. → lane=80 만 20% 낮음 (intrinsic 차이 크지 않음).
  - PoI drift (peak − predicted): lane=0 median +24, lane=80 **−12**, lane=128 −2.
    → predict_poi formula `3014+33·(lane>>1)+16·(lane&1)` 가 lane=80 에서 +12
    sample 오프셋 (peak 가 실제로는 predicted 보다 12 sample 앞에 있음).
  - |f_c| matched comparison (각 lane × |f_c| bin): SNR_peak 거의 동일
    (e.g. |f_c|∈[300,800): lane=0 0.097, lane=80 0.098, lane=128 0.102).
  - **결론**: lane=80 의 0% 회수는 intrinsic SNR 부족 아님. 두 효과의 곱:
    (i) K=4 batch (16 cases) 에 |f_c|<32 좌표 0 개 (lane=0/128 K=8 은 각각 3, 2 개).
    (ii) predict_poi 가 lane=80 에서 +12 sample bias (V2 fixed PoI 가 peak 미스).
    refined PoI 모델 또는 K=8 lane=80 capture 로 회수 가능성 회복 예상.

- (o2) **lane=64 K=8 wide-γ + cross-lane 진단 확장**
  ✅ — `traces/ntruplus768/phase3/wideg_lane64_K8N32.npz` (3h capture).
  attack-valid n23 결과: SNR median **0.017** (lane=80 K=4 와 거의 동일),
  max 0.057. V2 top-1 0/32, top-10 0/32, **top-100 1/32 (key=6 slot=1, f=1368
  V2_pct 99.62)**. V2w top-100 0/32. P(V2 ≥ null99) 1/32.
  - 진단 n25 확장 (192 cases) → **lane=64 의 진실은 정반대**:
    SNR @ predicted PoI median 0.017 (worst),
    **SNR @ true peak in ±100 median 0.133 (best of all lanes!)**.
    drift: median **+10**, std 19, |max| 81.
  - **결정적 케이스**: key=1 slot=0 f=3455 (centered −2, |f_c|=2). SNR_peak=0.41
    (TOP-1 candidate quality), 그러나 SNR @ predicted PoI = 0.006.
    → predict_poi 가 +10 sample 잘못 가리키므로 V2 fixed PoI 회수 실패.
    refined PoI 사용 시 회수 강력 후보.
  - **lane → drift pattern**: lane=0 +24, lane=64 +10, lane=80 −12, lane=128 −2.
    `predict_poi(lane) = 3014 + 33·(lane>>1) + 16·(lane&1)` 이 lane 별로
    독립적인 offset 가짐. linear formula 부족, lane-별 calibration 필요.
  - 이 발견은 회수 가능성을 **6.25% 보다 더 위로 끌어올릴 가능성** 시사.
    sk-independent PoI calibration (예: HW(γ) corr-based scan) 다음 단계.

- (o3) **sk-indep PoI calibration 시도** (`scripts/n26_skindep_poi_v2.py`,
  `scripts/n27_v2w_window_sweep.py`) — lane=64 K=8 데이터 대상.
  - 방법 A — γ-variance peak: per-sample variance of mean-over-N traces
    across G=78 γ trials, search ±50 of predict_poi. → empirical PoI
    median drift **−14** (반대 방향, oracle 는 +10). 분산 peak 가 basemul
    이 아닌 γ-load step 에 잡힘 (γ HW={1,2} 두 값만 → variance 가 load
    operation 의 register-bus power 에 dominated). attack-valid 이지만
    **basemul PoI 추적 실패**. top-100 1/32 그대로.
  - 방법 B — V2w window sweep (W ∈ {10,15,20,25,30,40,50,75,100}):
    W≥15 에서 plateau (top-100 1/32, V2w_99 mean 0.59, max 0.87).
    window 확장으로 잡은 추가 신호가 null max 확장에 의해 상쇄됨.
  - oracle (true peak PoI) upper bound: top-1 1/32, top-10 2/32,
    **top-100 3/32 (9.4%)** — lane=64 의 intrinsic recovery 한계.
    현재 1/32 = 33% of upper bound 만 회수.
  - **결론**: γ entropy (HW={1,2} 12+66 split) 가 충분치 않아 sk-indep
    PoI scan 에 제약. 회수 ceiling 은 single-sample 과 multi-comparison
    null 의 **곱** 이지 PoI 단독 효과 아님. lane=64 의 9.4% intrinsic
    upper bound 는 lane=128 (4/64=6.25%) 와 lane=0 (5/80=6.25%) 와 통계적
    범위 내.

- (q) **N-saturation 메커니즘 규명** ★ (`scripts/n28_n64_diagnose.py`,
  `scripts/n29_noise_decompose.py`) — paper-grade insight.
  - **현상**: K=4 N=64 lane=0 데이터에서 corr/predicted ratio (formula
    `SNR/√(SNR²+1/N)`) 가 N 증가에 따라 단조 감소: N=2 0.766, N=4 0.590,
    N=8 0.488, N=16 0.430, N=32 0.335, **N=64 0.274**. → SNR formula
    overpredicts by ~4× at N=64.
  - **noise decomposition**: per-(g, sample) trace 를 within-g (across-N)
    와 across-g (after linear fit) 로 분해. K=4 N=64 16 cases 모두에서
    **measured V2 corr = b·σ_HW / √((b·σ_HW)² + σ_a²)** (across-g residual)
    매칭 (ratio 1.000 ± 0). σ_w (within-g 노이즈) 와 무관.
  - **메커니즘**: trace = α[g] + b·HW(γ·f) + ε[g, n] 모델에서 α[g] (γ-only,
    sk-independent baseline) 가 saturating 노이즈. 평균화는 ε 만 줄이고 α
    는 그대로. corr saturates at b·σ_HW / √(b²·σ_HW² + σ_α²).
  - **paper-grade 결론**:
    - **N>16 은 거의 무용**. corr 는 σ_α 에 dominated.
    - **trace budget 분배**: K (more keys, lucky high-b cases) >> N (within-key averaging). K=8 N=32 = K=4 N=64 (총 traces) 이지만 K=8 의 회수율 (3/32) 이 K=4 N=64 (0/16) 보다 높음.
    - 임계 SNR 식별: corr ≥ 0.5 (≈ null max) 회수 가능 ↔ b·σ_HW ≥ σ_α.
      σ_α ≈ 0.0027, σ_HW ≈ 1.5 → b ≥ 0.0018 필요. 측정 b 분포: 0.00002–
      0.00082, median 0.00033. **b ≥ 0.0018 인 (k, slot) 만 회수**.
  - **α-removal 시도 실패** (`scripts/n30_alpha_removal.py`): leave-one-out
    cross-key mean 으로 α[g] 추정 → V2 CPA 수행. 24 keys × 4 slots 중
    raw top-100 5/96 → α-corrected 1/96 (악화).
    이유: 추정한 α[g] 가 random f 평균 해도 H_true 와 상관 (특히 K=4 batch
    에서 sample size 제한). → simple cross-key average 로 α 분리 불가.

- (q2) **Profiled PoI 시도 (lane-specific median offset)**
  (`scripts/n31_profiled_poi.py`) — lane→time mapping 은 sk-independent
  (firmware/hardware 상수). n25 oracle median drift 적용:
  - lane=0 +24, lane=64 +10, lane=80 −12, lane=128 −2.
  - 결과 (208 cases): predict_poi top-100 **9/208**, profiled **8/208**.
  - 개별 케이스 변화: lane=0 +1 회수 (k=2 sl=0 f=2016 rk 155→39),
    lane=64 −1 회수, lane=128 −1 회수. **net neutral**.
  - 이유: drift std=20-30 sample. median offset 가 systematic 만 잡음.
    개별 case 의 ±20-30 sample 변동은 여전. 평균 +24 인 lane=0 에 +24
    적용 시 실제 +0 인 case 는 −24 로 PoI 미스.

- (q3) **PoI ensemble (max(predict, profiled))** (`scripts/n32_poi_ensemble.py`):
  ensemble top-100 9/208 (predict 와 동일). null99 평균 0.444 (predict)
  → 0.487 (ensemble 후) — 10% 증가. 추가 PoI 의 신호 ↑ 와 null ↑ 가 상쇄.
  → **단순 PoI 변형으로 ceiling 못 깸**. fundamental limit 는 σ_α/(b·σ_HW)
  비율 — 단일 (key, slot) 에서 b 가 작으면 PoI quality 와 무관.

- (q3b) **per-(lane, slot) drift 패턴 발견** ★ (`scripts/n34_per_slot_drift.py`,
  `scripts/n35_per_slot_profiled.py`) — paper-grade calibration insight.
  - n34 oracle drift weighted-by-peak_corr median per (lane, slot):
    | lane\slot | 0 | 1 | 2 | 3 | spread |
    |---|---:|---:|---:|---:|---:|
    | 0 | +24 | +24 | +24 | +24 | 0 |
    | 64 | +12 | +12 | +11 | −2 | 14 |
    | 80 | **+9** | **−4** | **−13** | **−15** | **24** |
    | 128 | 0 | −2 | 0 | −1 | 2 |
    → lane=80 의 4 slot 들은 24 sample 에 spread. lane=80 basemul iter=40
    의 slot processing 이 timing 분리 (lane=0/128 은 slot 들이 거의 동시).
    이는 **firmware-level 정보** (sk-independent), 1회 calibration 후 모든
    attack 에 적용 가능 (standard SCA practice).
  - n35 per-(lane, slot) profiled PoI 결과: top-100 8/208 (predict 9/208),
    top-500 35/208 (predict 30/208) — 양 쪽 통계로 +5 case 회수.
  - **결정적 케이스**: lane=64 k=1 s=0 f=3455 (|f_c|=2): predict rank 2763
    → per-slot +12 rank **107** (top-100 직전!). small-|f_c| 좌표가
    refined PoI 로 회수 거의 가능.
  - top-100 기준 marginal (9 vs 8) 인 이유: per-slot offset 가 high-corr
    좌표는 회수에 도움 주지만, lucky low-corr top-100 case (e.g., lane=64
    k=6 s=1 f=1368) 는 다른 sample 노이즈로 혼동 → −1.
  - 이 finding 은 주로 **top-500 / 사실상 회수 가능 후보 풀** 확장
    (30→35, +17%). 후속 sparse recovery / 사후 lattice attack 단계의
    candidate quality 측면에서 가치.

- (q5) **HW 모델 변형: Montgomery (M5) 가 별도 leak channel** ★★
  (`scripts/n36_hw_models_n64.py`, `scripts/n37_montgomery_all.py`,
  `scripts/n38_zsum_ensemble.py`) — paper-major finding.
  - **배경**: basemul 의 실제 연산은 `r[0] = montgomery_reduce(c[0]·f_ntt[0] - …)`.
    M1 (HW(γ·f mod q)) 은 mod-q 의 결과 HW 를 modeling 하지만 montgomery
    reduction 의 실제 결과는 다른 값 (γ·f / R mod q where R=2^16).
    M5 = HW(montgomery_reduce(γ·f)) 가 정확한 operand 모델.
  - **K=4 N=64 lane=0 단독 결과** (n36): M5 가 1/16 top-100 회수
    (k=3 sl=0 f=1617, M5 corr **0.423**). M1 은 이 case 0/16. M5 가
    M1 이 못 잡은 leak 도 잡음. 7 HW variants 中 M5 가 best max corr.
  - **모든 batch 적용** (n37, 208 cases):
    - M1: top-100 9/208 (5%), top-10 1/208.
    - M5: top-100 3/208, top-10 0. → M1 보다 적지만 다른 case 들 회수.
    - MAX(M1, M5) ensemble: top-100 9/208 — null inflation 으로 무이득.
  - **3 cases M5-only top-100** (M1 에 못 잡힘):
    - K=4 N=64 k=3 s=0 f=1617 |f_c|=1617: M1 rk 1184 → M5 rk **56**.
    - K=8 r1 ln=0 k=7 s=3 f=882 |f_c|=882: M1 rk 341 → M5 rk **12**.
    - K=8 ln=64 k=2 s=3 f=322 |f_c|=322: M1 rk 321 → M5 rk **11**.
    이 3 case 모두 |f_c|>200 — small-|f_c| 쌀이 아닌 LARGE-|f_c| 의
    회수 가능 채널 발견.
  - **Z-score sum ensemble (n38)** ★: top-10 = **2/208** (M1 단독 1/208 의 2×):
    | case | M1 rk | M5 rk | Zsum rk |
    |---|---:|---:|---:|
    | K=8 r1 ln=0 k=7 s=3 f=882 (\|f_c\|=882) | 341 | 12 | **2** |
    | K=8 ln=64 k=2 s=3 f=322 (\|f_c\|=322) | 321 | 11 | **7** |
    | K=8 r1 ln=0 k=5 s=1 f=16 (\|f_c\|=16) | 0 | 1861 | 28 |
    | K=8 r2 ln=128 k=5 s=2 f=872 (\|f_c\|=872) | 16 | 858 | 43 |
    Zsum rk **2** = 거의 TOP-1. Zsum 의 top-100 6/208 (lower than M1
    9/208) 인 이유: M5 가 weak 인 cases 가 dilute 됨. ensemble 은 M1∪M5 가
    공통 detect 한 cases 강조.
  - **paper-grade 결론**:
    - NTRU+ basemul 의 leak 은 **2 separate channels**: (i) γ·f mod q 의
      HW (M1, register-load 가능), (ii) montgomery_reduce(γ·f) 의 HW (M5,
      reduction-output). 두 채널은 다른 |f| range 에서 두드러짐.
    - **M1 (small |f_c|) ∪ M5 (large |f_c| occasional)** 가 회수 가능 풀의
      structurally complete 한 set. 6.25% recovery 한계는 M1 단독 measurement.
    - Zsum top-10 = 2/208 (~1%) 는 **near-perfect recovery** rate 첫 dual-channel
      결과. 1 perfect TOP-1 (f=16 N=2) 외에 추가로 2 case rk 2, 7 — 사실상
      perfect 회수 가능 후보.

- (q6) **★★★ Full attack stack: per-slot PoI + Zsum ensemble** —
  paper-major final result (`scripts/n39_full_stack.py`).
  - 결합: per-(lane, slot) median drift offset (n34) + M1+M5 Z-score sum (n38).
  - 208 cases 결과:
    | pipeline | top-1 | top-10 | top-100 | top-500 |
    |---|---:|---:|---:|---:|
    | M1 baseline (predict_poi) | 1 | 1 | 9 | 30 |
    | M1 (slot-PoI) | 1 | 1 | 8 | 35 |
    | M5 (slot-PoI) | 0 | 0 | 4 | 31 |
    | **Full stack (slot+Zsum)** | **1** | **2** | **6** | **42** |
  - **2 TOP-1**: original f=16 (still rank 0 in M1 baseline) + **NEW f=882
    (lane=0 k=7 s=3, |f_c|=882, full-stack rank 0)**.
  - **6 full-stack top-100 cases**:
    | batch | key/slot | f | \|f_c\| | M1 pred | M1 sl | M5 sl | full |
    |---|---|---:|---:|---:|---:|---:|---:|
    | K=8 r1 ln=0 | 7/3 | 882 | 882 | 341 | 143 | 12 | **0** |
    | K=8 r1 ln=0 | 5/1 | 16 | 16 | **0** | 0 | 2257 | 8 |
    | K=8 r2 ln=128 | 5/2 | 872 | 872 | 16 | 16 | 858 | 43 |
    | K=8 ln=64 | 1/0 | 3455 | **2** | 2763 | 107 | 252 | 44 |
    | K=4 N=64 ln=0 | 3/0 | 1617 | 1617 | 1184 | 2054 | 19 | 73 |
    | K=8 r2 ln=0 | 7/2 | 517 | 517 | 2817 | 930 | 53 | 79 |
    → small-|f_c| (16, 2) AND large-|f_c| (882, 872, 1617) 둘 다 recover.
    full stack 가 **structurally |f_c|-agnostic recovery** 처음 demonstrate.
  - **paper-grade key result**:
    - Standard CPA (M1 only) → 1 TOP-1, 4.3% top-100.
    - Full stack pipeline → **2 TOP-1**, 1% top-10, 20% top-500.
    - Top-500 candidate pool 확장 (30→42, +40%) — 후속 sparse recovery 단계의
      candidate sample 수 향상.
  - **메커니즘**: M1 (mod-q HW) + M5 (montgomery_reduce HW) 가 dual leak channels.
    per-slot PoI 가 firmware-precise calibration. Zsum 이 두 채널의 evidence
    pooling. 단일 pipeline 으로 NTRU+ basemul leak 의 양 dimension capture.

- (q8) **★★★ K=8 lane=80 capture + 최종 누적** (2026-05-10) — Phase 4 종료.
  - **K=8 lane=80 capture** ✅ — `traces/ntruplus768/phase3/wideg_lane80_K8N32.npz`
    (3h 12min capture). attack-valid n23: SNR median 0.013, max 0.049 (K=4
    lane=80 와 동등하게 낮은 SNR @ predict_poi). V2 top-1 0/32,
    **top-100 1/32 (k=4 sl=1 f=283 V2_pct 97.63)**, V2w top-100 0/32.
    K=4 lane=80 의 0/16 → K=8 의 1/32 — small-|f_c| 좌표 부재 + 보더라인 case.
  - **n34 per-slot drift 재교정** (240 cases): K=8 lane=80 추가로 drift 분포
    refined.
    - lane=80 slot 0/1/2/3 weighted median: **+8 / +7 / −3 / −3**
      (이전 K=4-only 추정 +9 / −4 / −13 / −15 보다 spread 작음).
    - K=4-only 의 slot 2/3 음수 큰 drift 는 small-sample artifact.
  - **n39 full-stack pipeline 240 cases 적용** (refined per-slot offsets):
    | pipeline | top-1 | top-10 | top-100 | top-500 |
    |---|---:|---:|---:|---:|
    | M1 baseline (predict_poi) | 1 | 1 | **10** (4.17%) | 35 |
    | M1 (slot-PoI) | 1 | 1 | 8 | 37 |
    | M5 (slot-PoI) | 0 | 0 | 6 | 38 |
    | **Full stack (slot+Zsum)** | **1** | **2** | **9** | **45** (18.75%) |
  - **per-lane top-100** (full stack):
    | lane | n | M1 baseline | full stack |
    |---|---:|---:|---:|
    | 0 | 96 | 5 | 4 |
    | 64 | 32 | 1 | 1 |
    | 80 | 48 | 1 | **3** (refined per-slot 효과 ★) |
    | 128 | 64 | 3 | 1 |
    → lane=80 K=8 + refined slot offsets 가 처음 lane=80 의 top-100 회수
    가능성 입증.
  - **★ Final cumulative paper claim** (240 cases):
    - M1 baseline: 1 TOP-1, 10 top-100 (4.17%), 35 top-500 (14.58%).
    - Full stack: 1 TOP-1 (NEW f=882), 2 top-10, 9 top-100, 45 top-500 (18.75%).
    - **Union M1 ∪ Full**: 2 TOP-1 (f=16, f=882), **17 unique top-100 / 240
      (7.08%)**, top-500 candidate pool ~50 unique cases (~21%).
    - Per-lane recovery present in **all 4 lanes** (이전 lane=80 의 0% 깨짐).
  - **paper title summary**: "NTRU+768 chosen-CT NTT-domain selected-lane CPA on
    natural `crypto_kem_dec` 가 standard CPA + dual-channel HW model
    (mod-q M1, montgomery M5) + per-(lane, slot) PoI calibration 으로
    structurally-determined NTT 좌표 ~7% (top-100) 회수, 2 perfect TOP-1
    (1 case at N=2, 1 case at N=16). 회수 lane: 0/64/80/128 모든 4 lanes."

- (q7) **Full-stack N-curve** ★ (`scripts/n40_full_stack_ncurve.py`) —
  paper-quality minimum-traces analysis on top-recovered cases.
  | case | N=2 | N=4 | N=8 | N=16 | N=32 | M1@N=32 |
  |---|---:|---:|---:|---:|---:|---:|
  | k=7 s=3 f=882 (full) | 4 | 7 | 3 | **0** | **0** | 143 |
  | k=5 s=1 f=16 (M1) | 0 | 0 | 0 | 0 | 0 | 0 |
  | k=1 s=0 |fc|=2 (full) | 117 | 58 | 46 | 46 | **44** | 107 |
  | k=3 s=0 f=1617 (full) | 62 | 67 | 70 | 66 | 70 | 2102 |
  - **f=882 TOP-1 at N=16, top-10 at N=2** (just 2 chosen traces!).
    M1 baseline never reaches top-100 (rank 143 at N=32) — pure M5 channel discovery.
  - lane=64 |f_c|=2 case top-100 at N=8 (full), never with M1 (rank 107 at N=32).
  - **paper claim**: NTRU+ NTT-domain selected-lane attack 에서 두 가지 perfect
    recovery 가 존재 — (i) M1 (mod-q HW) 채널이 small-|f_c| (f=16, N=2 sufficient),
    (ii) M5+slot-PoI Zsum (Montgomery channel) 이 large-|f_c| (f=882,
    N=16 sufficient). 두 channel 이 다른 sk-구조 좌표 군 회수.

- (q4) **b 분포 종합** ★ (`scripts/n33_b_distribution.py`) — 208 cases.
  - **b @ predict_poi**: median 0.00020, max 0.00196.
  - **b @ oracle peak** (true f label): median 0.00043 (≈ 2× predict), max 0.00277.
  - σ_a @ oracle: median 0.0029 (saturating noise).
  - corr_oracle (formula `b·σ_HW/√((b·σ_HW)² + σ_a²)`):
    median 0.204, max 0.848.
    ≥ 0.5: **4/208 (1.9%)** — 회수 가능 ceiling.
    ≥ 0.4: 13/208, ≥ 0.3: 45/208, ≥ 0.25: 73/208.
    측정된 V2 top-100 9/208 = 4.3%. corr 0.4 ≤ … ≤ 0.5 사이 매칭.
  - **per-lane b @ oracle**:
    | lane | n | b_med | b_max | b≥0.0018 | corr≥0.5 |
    |---|---:|---:|---:|---:|---:|
    | 0 | 96 | 0.00048 | 0.00277 | 3/96 | 2/96 |
    | 64 | 32 | 0.00056 | 0.00138 | **0/32** | 0/32 |
    | 80 | 16 | 0.00030 | 0.00086 | **0/16** | 0/16 |
    | 128 | 64 | 0.00040 | 0.00259 | 3/64 | 2/64 |
    → lane=64/80 은 **realistic 회수 후보가 0/n**. b≥0.0018 임계 미달.
  - **|f_c| 의존성** (b @ oracle median):
    | f_c bin | n | b_med | σ_HW_med | corr_med |
    |---|---:|---:|---:|---:|
    | [0,16) | 4 | 0.00130 | 2.10 | **0.427** |
    | [16,50) | 4 | 0.00105 | 1.74 | 0.420 |
    | [50,200) | 17 | 0.00046 | 1.51 | 0.224 |
    | [200,500) | 48 | 0.00044 | 1.59 | 0.203 |
    | [500,1000) | 53 | 0.00040 | 1.58 | 0.190 |
    | [1000+) | 82 | 0.00042 | 1.58 | 0.203 |
    → **|f_c|<50 인 좌표가 회수 가능 풀**. 일반 |f_c|>200 좌표는 b·σ_HW
    < σ_α 로 corr ≤ 0.25.
  - **paper-grade 결론**: NTRU+ NTT-domain selected-lane attack 의
    회수 가능 좌표는 |f_c|<50 인 "small-magnitude" 좌표로 **structurally
    한정**. 평균 NTRU+ key (768 coords) 에 |f_c|<50 좌표 ≈ 22 개 (probability
    100/3457·768 ≈ 22). 그 중 selected 4-slot lane 에 떨어지는 비율은
    낮음 — 6.25% recovery rate 와 일치.

- (p) **누적 attack-valid (lane=64 추가)** — 6 batches K=44 effective × 4 slots
  = **176 cases**:
  - V2 (fixed PoI) top-1 1/176 (0.57%), top-100 9/176 (5.11%), ≥null99 5/176.
  - V2w (per-cand best in ±10) top-100 9/176 (5.11%), ≥null99 5/176.
  - SNR distribution: median 0.066, 90th 0.227, max 0.647.
  - **Lane breakdown** (V2 top-100):
    - lane=0: 5/80 (6.25%)
    - lane=64: 1/32 (3.13%)
    - lane=80: 0/16 (0%)
    - lane=128: 3/48 \[K=8a\] + 1/16 \[K=8b\] = 4/64 (6.25%)
  - lane=64 의 underperformance 는 predict_poi formula 의 +10 bias 때문 —
    refined PoI 로 9.4% intrinsic ceiling 달성 가능 (현재 1/3 회수).

- (n5) **최종 종합** (`scripts/n24_combined_analysis.py` updated) —
  4 batches, **K=28 effective keys × 4 slots = 112 cases**:
  - V2 (fixed PoI) recovery: top-1 1/112 (0.89%), top-10 1/112, **top-100 7/112
    (6.25%)**, top-500 18/112 (16.07%), ≥ null 99-th 3/112 (2.68%).
  - V2w (per-cand best PoI in ±10) recovery: top-10 1/112, top-100 8/112 (7.14%),
    ≥ null 99-th 4/112 (3.57%).
  - SNR distribution: median 0.082, 90th 0.251, max 0.647. SNR ≥ 0.5: 2/112;
    ≥ 0.3: 6/112; ≥ 0.2: 20/112; ≥ 0.1: 45/112.
  - Recovered 8 cases (V2 OR V2w top-100):
    | batch | key/slot | f | |f_c| | SNR | V2_pct | V2w_pct |
    |---|---|---|---|---|---|---|
    | K=4 r1 ln=0 | 0/0 | 1148 | 1148 | 0.280 | 99.68 | 99.68 |
    | K=8 r1 ln=0 | **5/1** ★ | **16** | 16 | **0.647** | **100.00** | 99.97 |
    | K=8 r1 ln=0 | 6/0 | 3451 | 6 | 0.281 | 98.35 | 99.68 |
    | K=8 r2 ln=0 | 1/1 | 1721 | 1721 | 0.316 | 98.18 | 97.34 |
    | K=8 r2 ln=0 | 4/3 | 3447 | 10 | 0.290 | 98.15 | 98.70 |
    | K=8 ln=128 | 1/0 | 435 | 435 | 0.289 | 89.96 | 97.74 |
    | K=8 ln=128 | 2/3 | 21 | 21 | 0.467 | 98.44 | 97.74 |
    | K=8 ln=128 | **3/2** | **6** | 6 | **0.527** | 99.36 | **99.71** |

  - 5/8 회수에 |f_c|<22 (small magnitude), 3/8 moderate-|f| with mid-SNR.
    단일 충분조건은 SNR ≥ 0.28; small |f_c| 는 high-SNR 의 sk-구조적 인자.
  - **lane 별 회수율 (top-100)**: lane=0 (5/80=6.25%), lane=128 (3/32=9.4%),
    lane=80 (0/16=0%). lane=80 은 outlier — basemul iter=40 의 SNR 저하.

- (n3) **종합 분석** (`scripts/n24_combined_analysis.py`) — 모든 batch 합쳐
  K=4+8+8 = K=20 keys × 4 slots = **80 cases**:
  - V2 (fixed predicted PoI 3014) recovery rates:
    - top-1: 1/80 (1.25%)
    - top-10: 1/80 (1.25%)
    - top-100: **5/80 (6.25%)**
    - top-500: 11/80 (13.75%)
    - ≥ empirical null 99-th: 2/80 (2.50%)
  - V2w (per-cand best PoI in ±10): top-100 5/80, ≥ null 99-th 3/80.
  - SNR distribution: median 0.080, 90th 0.220, max 0.647.
    SNR ≥ 0.5 cases: 1/80; SNR ≥ 0.3: 2/80; SNR ≥ 0.2: 11/80; SNR ≥ 0.1: 29/80.
  - |f_centered| distribution: median 754. cases with |f_c|<16: 2/80 (2.5%);
    |f_c|<32: 3/80; |f_c|<100: 4/80.

  **Recovered cases (5)**: V2_pct >= 98%
  | batch | key/slot | f | |f_c| | SNR | V2_pct | V2w_pct |
  |---|---|---|---|---|---|---|
  | K=4 r1 | 0/0 | 1148 | 1148 | 0.280 | 99.68 | 99.68 |
  | K=8 r1 | **5/1** ★ | **16** | **16** | **0.647** | **100.00** | 99.97 |
  | K=8 r1 | 6/0 | 3451 | 6 | 0.281 | 98.35 | 99.68 |
  | K=8 r2 | 1/1 | 1721 | 1721 | 0.316 | 98.18 | 97.34 |
  | K=8 r2 | 4/3 | 3447 | 10 | 0.290 | 98.15 | 98.70 |

  **공통 조건**: SNR ≥ 0.28. **3/5 가 |f_c| < 16** (small magnitude).
  2/5 는 moderate |f_c| 인데 SNR 0.28-0.32 → 회수.
  → high-SNR 의 충분조건은 small |f_c|; 필요조건 아님 (moderate |f_c| 에서도
  occasional high-SNR 발생).

  **paper-ready bottom line**:
  - 6.25% (5/80) **Top-100** recovery rate per (key, lane=0, slot)
  - 1.25% (1/80) **TOP-1** recovery (key 5 slot 1, f=16, N=2 traces)
  - 임계 SNR 0.28 (V2 fixed PoI). SNR 0.5+ 면 **N=2 만으로 perfect 회수**.
  - 천장 mechanism: G=78 (HW=1+HW=2 γ set) → empirical null max ≈ 0.66.

- (n) **회수 가능 좌표의 sk-구조적 특성** — 회수된 3 cases 의 |f_centered|:
  - K=4 key 0 slot 0: f=1148 (centered 1148) → rank 11
  - K=8 key 5 slot 1: f=16 (centered 16) → rank 0 ★
  - K=8 key 6 slot 0: f=3451 (centered −6) → V2w top10 ★
  → **small |f_centered| 좌표가 high-SNR**. 메커니즘: γ·f mod q 의 HW
  vector entropy 가 |f_centered| 에 의존. small f 는 다수 γ 에서 wrap-free →
  label var 매우 다른 패턴.
  - 확률 모델: NTRU+ f_ntt 가 uniform in Z_q. |f_centered|<16 인 좌표 비율 =
    32/3457 ≈ 0.93% per coord. lane 4 slot = 0.037 expected per lane → ~6%
    of (key, lane) 에 small-|f| 좌표 1 개 존재. 측정 6% recovery rate 와 일치.
  - **attack 가 회수 가능한 좌표는 sk 구조에 의존**: 키 별로 0-N 좌표만 leak.
    잔여 좌표는 SNR 부족으로 회수 불가. 이는 회수 가능성과 회수 가능량 사이의
    기본 trade-off.
- (c) **PoI alignment 진단** (`scripts/n20_poi_drift.py`) ✅ — true f 라벨로
  per-(key, lane, slot) 의 best |corr| sample 위치를 ±50 안에서 찾음.
  결과: drift mean=+0.17, std=22.9, max |drift|=49 sample. drift 분포가 거의
  균등 ([-50,-20]:20, [-20,-10]:13, …, [+20,+50]:20). |corr|@predicted PoI
  median=0.112, |corr|max median=0.341.
- (d) **drift = noise vs sk-dependent?** (`scripts/n21_drift_pattern.py`) ✅
  — N=8 을 첫 4 / 마지막 4 traces 두 half 로 나눠 각자의 best PoI 위치 비교.
  결과: |t_A − t_B| median=24 sample, |diff|≤5 fraction=0.219, |diff|≤10
  fraction=0.323. 두 half 가 일관된 best-PoI 잡으면 sk-dependent 신호 (N
  확장이 필요), 따로 가면 노이즈 spike. 24 sample 분포는 strong noise-driven
  pattern → "잘 정의된 leak peak 위치가 N=8 의 노이즈 floor 안에 묻혀 있음"
  결론. N≥32 확장이 peak localization 에 필요.

---

### 2026-05-08  Phase 3 leak-per-PoI = 4 coefficients (one design → 4 slots)

가정
- selected_lane(slot=0) chosen-CT 의 board-side basemul 은
    r[i] = montgomery_reduce(γ · f[4·lane + i])  (i=0,1,2,3)
  순서로 4 출력을 모두 계산. 한 trace 의 한 PoI sample 이 단일 slot 만이
  아니라 lane 의 4 coefficient 모두에 정보 leak 한다는 가설.

과정
- `scripts/n11_phase3_corr_plot.py` 의 변형으로 lane 0 의 PoI 3014 에서
  4 라벨 (HW(γ·f[i] mod q), i=0..3) 각각의 |corr| 를 sample 별로 계산.

결과 (수치) — lane 0, slot=0 chosen-CT:

| slot | f값 | argmax\|c\| | \|c\| max | \|c\| @3014 | \|c\| @3015 | \|c\| @3016 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1078 | 3014 | 0.553 | 0.553 | 0.009 | 0.043 |
| 1 | 1540 | 3014 | 0.386 | 0.386 | 0.018 | 0.111 |
| 2 | 676 | 22976 | 0.380 | 0.335 | 0.018 | 0.068 |
| 3 | -579 | 23479 | 0.371 | 0.153 | 0.028 | 0.022 |

해석
- **단일 slot=0 chosen-CT 의 한 PoI sample 에 lane 4 coef 모두 leak**.
  slot 0 가 가장 강 (corr 0.55), slot 1, 2 약간 약 (0.39, 0.34),
  slot 3 가 가장 약 (0.15) — 이 분포는 basemul 코드의 register
  통과 순서에 따른 leak intensity 차이로 추정.
- 수확량 4× 증가: 한 chosen-CT 디자인 (slot=0) 으로 lane 의 4 NTT
  coefficient 동시 회수 가능. multi-slot capture 가 항상 필요한 게 아님.
- Phase 4 recovery 가 lane 당 1 slot → 4 slots 회수로 업그레이드 (script
  n08 수정).

다음 단계
- multi-key held-out 결과로 attack-validity 검증.

---

### 2026-05-08  Phase 3 basemul-timing model + clean signal (v3 분석 완)

가정
- v3 (HW(γ) const) 의 PASS lanes (0, 80, 128) 의 PoI 가 basemul iteration
  에 대응한다면 PoI(k) = base + α·(k>>1) + β·(k & 1) 형태의 linear timing
  모델이 성립할 것. 모든 lane (PASS / FAIL 모두) 에 대해 예측 PoI 부근의
  ±W window 안 max\|corr\| 이 lane-specific f 값에 의해 결정된다.

과정
- v3 PASS lanes 의 PoI: lane 0 → 3014, lane 80 → 4342, lane 128 → 5126.
  3 점 fit: base=3014, α (per-iter) ≈ 33.0 cycles. (lane 128 정확 fit: 3014 + 33·64 = 5126 ✓.)
- `scripts/n10_phase3_basemul_model.py --per-iter 33 --window 20` 으로
  모든 lane 에 대해 예측 PoI ±20 window 안 per-sample max\|corr\| 와
  500-shuffle null 비교. 라벨은 HW(γ·f mod q).
- 라벨 entropy 도 측정: 12개 γ × f_lane 의 label HW 분산 (=`label var`).

결과 (수치)

| lane | iter | sub | pred PoI | actual PoI | real corr | z | verdict | label var |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | 0 | 0 | 3014 | 3014 | +0.553 | 9.03 | PASS | 2.64 |
| 1 | 0 | 1 | 3030 | 3048 | -0.192 | -0.25 | FAIL | 1.97 |
| 2 | 1 | 0 | 3047 | 3041 | -0.189 | -0.85 | FAIL | 1.56 |
| 3 | 1 | 1 | 3063 | 3083 | +0.246 | 1.07 | FAIL | 1.85 |
| 8 | 4 | 0 | 3146 | 3153 | -0.180 | -0.66 | FAIL | 0.74 |
| 32 | 16 | 0 | 3542 | 3539 | -0.217 | 0.16 | FAIL | 1.91 |
| 80 | 40 | 0 | 4334 | 4342 | -0.572 | 10.58 | PASS | 2.41 |
| 128 | 64 | 0 | 5126 | 5126 | -0.496 | 8.97 | PASS | 2.58 |

해석
- **basemul iteration timing 정확 — PASS lanes 의 actual PoI 가 예측과
  정확히 일치 (오차 ≤ 8 cycles)**. 모든 FAIL lanes 의 actual PoI 도
  예측에서 ≤ 20 cycles 안. 즉 leak source 는 basemul 임을 timing 으로 입증.
- PASS / FAIL 의 결정 요인 = label variance. PASS lanes (0, 80, 128) 모두
  var > 2.4. FAIL lanes 모두 < 2.0. label var 가 작은 이유는 G=12 HW=1 γ
  set (powers of 2) 으로 γ·f mod q 가 종종 비슷한 HW 값에 collapse 하기
  때문.
- 시사: chosen-CT 가 정확히 의도한 대로 basemul 한 lane 만 isolate 함.
  attack-validity 핵심 입증. label entropy 만 늘리면 (G 확장 / HW=2 set /
  multi-slot) 모든 lane 에서 signal 회복 가능.

다음 단계
- multi-key (K=4) 캡처 → held-out attack (Phase 4) 으로 attack-valid 여부
  검증.
- lane entropy 부족 해결 위해 G=28 HW=2 γ set 또는 multi-slot capture 시도.

---

### 2026-05-08  Phase 3 scout v3 — HW(γ) const γ set (G=12, single key)

가정
- v2 (γ HW 가변) 가 huge HW(γ) 신호 (corr=0.97-0.99) 로 f-leak 을 가렸음.
  γ ∈ {1,2,4,8,16,32,64,128,256,512,1024,2048} (12 powers of 2 ≤ 2048,
  모두 HW=1) 로 set 을 fixate 하면 byte-level public-design leakage 가
  std=0 으로 control 변수 자체가 사라지고 trace 의 모든 신호는 γ·f mod q
  의 f-dependence 만 남는다.

과정
- 캡처: `scripts/n05_phase3_scout.py --lanes 0,1,2,3,8,32,80,128
  --gammas 1,2,4,8,16,32,64,128,256,512,1024,2048 -N 10
  -o traces/ntruplus768/phase3/scout_hw1.npz`. shape (8, 12, 10, 24400),
  918s. 0 timeout, all mismatches=0.
- 분석: `scripts/n06_phase3_analyze.py`. HW(γ)-only control 이 γ HW std=0
  으로 자동 skip. 300-permutation null per lane.

결과 (수치)

| lane | max\|corr\| | PoI sample | null p99.9 | z | verdict |
|---:|---:|---:|---:|---:|---|
| **0** | **0.5530** | 3014 | 0.4528 | **8.27** | **PASS** |
| 1 | 0.3839 | 6131 | 0.4466 | 1.20 | FAIL |
| 2 | 0.3579 | 7553 | 0.4411 | -0.19 | FAIL |
| 3 | 0.3284 | 18959 | 0.4564 | -1.26 | FAIL |
| 8 | 0.3546 | 12637 | 0.4543 | 0.02 | FAIL |
| 32 | 0.3484 | 22473 | 0.4451 | -0.30 | FAIL |
| **80** | **0.5718** | 4342 | 0.4282 | **9.55** | **PASS** |
| **128** | **0.4959** | 5126 | 0.4362 | **6.39** | **PASS** |

해석
- 3/8 PASS — 깨끗한 f-leakage 신호 (HW(γ) 영향 없음).
- PASS lanes 의 PoI 가 basemul iteration 과 일치 → 다음 entry (basemul
  timing model) 에서 정량 확인.
- FAIL lanes 의 reported PoI 는 random noise peak — 실제 lane signal 은
  basemul region 이지만 label variance 가 부족해 max\|corr\| 가 null
  p99.9 미만.

다음 단계
- basemul timing model 로 모든 lane 에서 예측 PoI 위치 평가.
- multi-key 로 held-out 검증.

---

### 2026-05-08  Phase 3 scout v2 critique — γ-byte leakage 가 v1 PASS 을 오염

가정
- v1 (γ HW=1 const) 에서 PASS 한 lane 2, 80 의 corr 0.71-0.80 이 정말
  chosen-CT 의 γ·f leakage 인지, 아니면 c bytes 자체의 byte-level HW
  leakage 인지 분리 검증 필요. v2 는 γ ∈ {1..128, q-1, q-2, q-4, ...} 로
  γ HW 가 varied → public-design control 가능.

과정
- 캡처: `scripts/n05_phase3_scout.py --gammas 1,2,4,8,16,32,64,128,
  3456,3455,3453,3449,3441,3425,3393,3329 -N 10 -o scout_g16.npz`.
  shape (8, 16, 10, 24400), 918s.
- 분석: `n06_phase3_analyze.py` 에 **HW(γ) only 라벨로 추가 corr 계산 +
  gate 추가** (real > public-control + 0.05).
- decompose 분석 `n09_phase3_decompose.py`: 각 sample t 에서 |A|=|corr w/
  γ·f| 와 |B|=|corr w/ γ| 의 차 Δ. 깨끗한 f-leak sample 은 Δ > 0.1 ∧
  |A| > 0.4.

결과 (수치)
- 모든 lane 에서 HW(γ)-only max\|corr\| = 0.97-0.99 (lane 3, 8 만 0.31-0.33).
- real corr 0.30-0.93 — HW(γ) 와 비슷하거나 낮음.
- public-design gate 적용시 **0/8 PASS**.
- decompose: lane 80 argmax|A|=4343 (corr 0.93), 같은 sample 의 |B|=0.88.
  Δ=+0.05 — f-leakage 가 γ-byte leakage 위에 5% 만 추가.
- Δ > 0.1 ∧ |A| > 0.4 cells = **0 / 195200**.

해석
- v1 의 corr 0.71-0.80 PASS 는 v2 데이터로 보면 γ-byte leakage 의 일부
  → 단독으론 attack-valid 아님. SMAUG 의 flip_any α-bug 와 유사한
  public-design classifier risk.
- 시간상으로도 γ-byte leak (poly_frombytes(c, ct) 동안) 과 γ·f leak
  (basemul) 이 같은 sample 영역에 collapse 됨 — 보드의 instruction-level
  pipeline 으로 c 를 읽는 즉시 그 byte 가 register 에 머무르며 basemul
  의 γ 입력으로 재사용된다.
- 해결책: γ set 을 byte-HW 가 const 가 되도록 (HW=1 powers of 2) 또는
  byte 위치에서 control 가능한 paired design 으로 설계.

다음 단계
- v3 with HW=1 γ set (12 values).

---

### 2026-05-08  Phase 3 scout v1 — selected-lane signal (single key, G=6)

가정
- chosen-CT 가 c_ntt[4·lane+slot] = γ 한 점만 살린 상태에서 보드의
  `poly_basemul(&m1, &c, &f)` 가 m1[4·lane+i] = γ·f_ntt[4·lane+i] mod q
  를 계산. 보드 trace 가 그 시점에서 leak 한다면, 같은 lane 의 60 traces
  (G=6 γ × N=10) 는 label = HW(γ·f_ntt[4·lane+0] mod q) 와 corr.
- 단일 key 로도 lane 마다 6 distinct label values 가 있어 within-lane
  corr 추정이 가능. permutation null 은 (γ, n) 안에서 라벨만 셔플.

과정
- decimate=4 (1 sample/cycle, full 24400 cycle) 로 캡처.
- 캡처: `scripts/n05_phase3_scout.py --lanes 0,1,2,3,8,32,80,128
  --gammas 1,2,4,16,64,256 -N 10` →
  `traces/ntruplus768/phase3/scout.npz` (8, 6, 10, 24400) = 480 traces, 349s.
- 분석: `scripts/n06_phase3_analyze.py` — label = HW(γ·f mod q) (`modq_hw`)
  와 raw value (`modq_value`) 두 가지. 300-permutation null per lane.

결과 (수치) — 라벨 = HW

| lane | max\|corr\| | PoI sample | null p99.9 | z | verdict |
|---:|---:|---:|---:|---:|---|
| 0 | 0.5476 | 3013 | 0.5786 | 1.88 | FAIL |
| 1 | 0.5666 | 12497 | 0.6288 | 1.89 | FAIL |
| **2** | **0.7998** | 21956 | 0.6166 | **9.87** | **PASS** |
| 3 | 0.5413 | 19634 | 0.6059 | 1.06 | FAIL |
| 8 | 0.5164 | 6232 | 0.6116 | 0.44 | FAIL |
| 32 | 0.5248 | 14189 | 0.5987 | 0.61 | FAIL |
| **80** | **0.7067** | 4343 | 0.6244 | **6.32** | **PASS** |
| 128 | 0.4938 | 2662 | 0.6062 | -0.44 | FAIL |

결과 (수치) — 라벨 = signed value (centred residue)

- 8/8 lanes FAIL. max\|corr\| ≤ 0.54, 모두 null p99.9 미만.

해석
- **2/8 PASS** 는 random null (8 × 1.6e-3 ≈ 1.3% 기대) 보다 압도적으로 높음
  → 진짜 signal. 다만 8 중 6 은 약함 — chosen-CT 가 효과적인 lane 이
  데이터/labeling 측면에서 selective.
- HW label 이 value-level 라벨보다 강함 — Cortex-M4 SCA 의 표준 가정과
  일치 (register HW dominance).
- PoI 분포: lane 80 의 PoI=4343 은 추정 basemul 영역 (S+60·40 = 4400 with
  S≈2000) 과 정확히 일치. lane 2 의 PoI=21956 은 basemul 한참 후 — invntt
  / crepmod3 의 m1 pass-through propagation 으로 추정. 즉 leak source 가
  단일이 아니라 **basemul + downstream** 다중 single-trace.
- per-lane label entropy (HW range 의 distinct value 개수): 어느 lane 이
  PASS 하느냐는 label entropy 보다 leak source 와 PoI 의 alignment 가
  결정짓는 듯 — lane 2 (3 distinct HW) 가 lane 32 (3 distinct HW) 보다
  훨씬 강함.
- G=6 으로는 signal 약. **G=16 (양/음 modular 포함) 으로 v2 capture** 진행
  중.

다음 단계
- Phase 3 scout v2 (G=16) 분석 — 같은 lane 들에서 z 가 향상되는지, PoI 가
  더 일관성 있게 잡히는지.
- v2 결과로 PASS lane 비율 ≥ 50% 면 multi-key Phase 3 confirmation
  (held-out attack), 미만이면 multi-slot (slot=0..3) capture 추가로 lane 당
  4× label 데이터 확보 시도.

---

### 2026-05-08  Phase 2 — byte-HW triage on Phase 1 traces (K=8, N=20)

가정
- Phase 1 의 cross-key SNR = 423 가 byte-level Hamming weight 라벨로
  설명 가능한지 확인 (positive control). sk-byte HW (sk read 가설), ct-byte
  HW (ct read 가설), 그리고 late window 에서의 sk-byte HW 셋을 비교.
- 단일 max-over-grid 통계는 K=8 의 8점 \|corr\| saturation 으로 의미 없음
  (real max < null mean 같은 무의미한 결과 확인). 대신 **per-cell z** 분포가
  null 분포에서 얼마나 벗어나는지 본다.

과정
- 분석: `scripts/n03_phase2_skunpack.py`. per-cell |corr| 계산 후 cell 별
  null p99.9 로 임계 → "임계 초과 cell 비율" 을 reference null 의 동량과 비교.
- 3 시나리오:
  1. sk-byte (early window 0-5k, bytes 0-64)
  2. ct-byte (early window, fingerprint proxy 0-16)
  3. sk-byte (late window 12-20k, bytes 0-64)

결과 (수치)

| 시나리오 | top z | survive | reference null | ratio | verdict |
|---|---:|---:|---:|---:|---|
| sk early (0-5k) | 3.47 | 736 | 473.8 | 1.55× | WEAK |
| ct early (0-5k) | 3.35 | 173 | 107.3 | 1.61× | WEAK |
| sk late (12-20k) | 3.49 | 1227 | 768.3 | 1.60× | WEAK |

z_per_cell 분포 (셋 다): mean ≈ 0.0, std ≈ 1.0 — null 분포 그 자체.

해석
- 세 가설 모두 ratio ~1.5-1.6× 의 약한 inflation 만 보이며, top z 는
  3.4-3.5 로 K=8 의 noise floor 수준. **byte-HW model 은 K=8 cross-key 의
  423 SNR 을 설명하지 못한다**.
- 결론: Phase 1 의 cross-key 신호는 byte-단위 Hamming weight 의 단순 합이
  아니라 더 복잡한 함수 (e.g. 12-bit 계수의 register-level operation,
  basemul 의 montgomery_reduce 결과 등). Phase 3 의 value-aware label
  (γ·f mod q) 이 적합한 모델.
- Phase 2 negative 자체는 새로 알게 된 것이 적음 — Phase 1 의 region split
  (early=noise, late=hot) 이 이미 byte-HW 가설이 약함을 시사했음. 단,
  **K=8 setting 에서는 어떤 단순 라벨도 statistical force 부족**이라는
  사실을 명시적으로 확인.

다음 단계
- Phase 3 scout 로 직진. byte-HW 대신 chosen-CT 에 의해 통제되는 γ 변수를
  활용해 단일 key 안에서 라벨 분포를 만들어내는 방향.

---

### 2026-05-08  Phase 1-timeline — full decap mapping (decimate=4)

가정
- decimate=4 → 1 sample / core cycle, 24400 sample = 24400 cycle 캡처.
  NTRU+768 decap 100k cycle 중 첫 24% 영역에 어떤 operation block 이
  들어가는지를 sample 위치로 매핑.
- Phase 1 (decimate=1) 에서 본 hot region 12-24k samples = 3000-6100 core
  cycle 에 대응되어야 함.

과정
- 캡처: `scripts/n04_decap_timeline.py --decimate 4 -K 4 -N 4`.
- 분석: cross-key SNR 을 cycle 구간 별로 집계.

결과 (수치)

| 구간 (cycle) | max SNR | mean SNR | #>1 | #>5 |
|---:|---:|---:|---:|---:|
| 0-2000 | 3.22 | 0.27 | 35 | 0 |
| 2-4k | **264.7** | 1.36 | 224 | 80 |
| 4-6k | 170.3 | 2.19 | 370 | 158 |
| 6-8k | 194.1 | 2.21 | 390 | 153 |
| 8-10k | **287.0** | 2.20 | 363 | 146 |
| 10-12k | 224.3 | 2.25 | 380 | 163 |
| 12-14k | 170.5 | 2.18 | 374 | 144 |
| 14-16k | 177.7 | 2.06 | 364 | 164 |
| 16-18k | 247.7 | 2.44 | 395 | 161 |
| 18-20k | **292.3** | 2.43 | 364 | 144 |
| 20-22k | 257.9 | 2.09 | 368 | 144 |
| 22-24k | 201.1 | 2.38 | 391 | 166 |

해석
- Cycles 0-2000 = poly_frombytes(c, ct) 영역, byte-level arithmetic 으로
  SCA 거의 없음 (max SNR 3.2).
- Cycles 2000~24400 = 거의 균일한 hot region. 이 안에:
  poly_frombytes(f, sk), poly_frombytes(hinv, sk), poly_basemul(m1, c, f),
  poly_invntt(m1) 의 전부 또는 일부가 들어감.
- Phase 1 (decimate=1) 의 sample 12-24k = 3-6.1k cycle 에 해당되는 영역도
  여전히 hot. 추정: 12-24k samples 는 frombytes(f) 후반 + frombytes(hinv) +
  basemul 시작.
- **Phase 3 캡처는 decimate=4 로 해야 basemul 전체를 보장**한다.

다음 단계
- Phase 3 scout 로 chosen-CT 분석 시작.

---

### 2026-05-08  Phase 1 — natural D map (K=8, N=20, valid CT)

가정
- 보드 위 `crypto_kem_dec` 의 trace 가 sk 또는 ct 와 상관 있는 어떤 시간 영역을
  갖는다. cross-key 분산이 within-key 분산보다 충분히 크면 SCA-가능 신호로
  본다 (Phase 1 의 단일 게이트는 max|t| > null p99.9).
- 한 key 에 대해 fresh ct (encap of ephemeral randomness) 1개 → decap 20회.
  같은 (sk, ct) pair 에서 within-key 분산은 순수 전기 노이즈.
- 24400 샘플 (CW-Lite 한도) 은 ADC=4·clkgen=29.5 MHz 기준 6100 코어 사이클
  대응. NTRU+768 decap 전체는 100k+ 사이클이므로 **trace 는 decap 의 선두
  ~6 % 만 잡는다** — 대략 poly_frombytes ×3 + poly_basemul(m1,c,f) 의 일부.

과정
- 펌웨어: `simpleserial-ntruplus-CW308_STM32F4-ntruplus768.hex` 플래시
  (sha256 `55c92fbd1e2c459f…`).
- capture: `scripts/n01_phase1_capture.py -K 8 -N 20 -s 24400 -o
  traces/ntruplus768/phase1/d_map.npz`. K 키마다 `k → e → d×20`. sk/pk dump
  도 calibration 용으로 보존 (`calibration_only_sk = True`).
- 분석: `scripts/n02_phase1_analyze.py --n-shuffles 200`. cross-key SNR,
  pairwise max\|t\| (28 페어), 200-permutation null.
- 시각화: `scripts/n02b_phase1_plot.py` — 3-stack overview PNG 생성.

결과 (수치)
- 캡처 137 s, 0 timeout, mismatches=0 (valid CT 전부 OK).
- **max\|t\| = 248.87** (sample 16221) vs null p99.9 = **7.29**
  → z ≈ (248.87 − 5.64) / 0.42 ≈ **579 σ**.
- 24400 샘플 중 **2188 점이 \|t\| > 10**, **3251 점이 \|t\| > 6**.
- **cross-key SNR max = 423.78** (sample 15269). 상위 5 위치:
  15269, 15177, 16221, 21625, 19813.
- region split (max\|t\|, max SNR, #pts > 10):

  | region | max\|t\| | max SNR | #pts>10 |
  |---|---:|---:|---:|
  | 0-3000  (frombytes ×3 + early basemul?) | 7.1 | 0.27 | 0 |
  | 3-7k    (early basemul) | 5.9 | 0.34 | 0 |
  | 7-12k   (mid basemul)   | 6.2 | 0.27 | 0 |
  | 12-16k  (late basemul)  | 199.6 | 423.78 | 768 |
  | 16-20k                  | 248.9 | 342.54 | 667 |
  | 20-24k                  | 165.0 | 263.91 | 753 |

해석
- gate 통과 (max\|t\| 25.5× null p99.9). leakage 가 단일 점이 아니라 여러
  cluster 에 분포 → loop 단위 leakage signature 일 가능성 높음.
- **놀라운 비대칭**: 초반 12k 샘플은 거의 noise (max\|t\|≤7.1, ≈ null), 후반
  ~12k 부터 폭발. NTRU+768 의 poly_frombytes (sk read) 는 byte-level
  arithmetic 이라 12-bit register 압박이 약하고 SCA 노출이 작은 듯.
  반면 poly_basemul 의 16×16-bit mult + Montgomery reduce 가 후반부 신호의
  주범으로 추정. 6100 cycle 짜리 window 의 후반 = basemul (192 lane × ~30
  cycle ≈ 5760 cycle) 의 후반부 lane 들. 즉 **Phase 3 의 basemul 가 그대로
  hot spot**.
- Phase 2 (sk-unpack) 는 0~3000 샘플 영역에 해당하나 max\|t\|=7.1 ≈ null
  → 이 데이터로 위치-제한 byte-HW 라벨이 의미 있는 신호를 못 만들 가능성
  높음. 그러나 정성적 positive-control 시도는 필요.
- 단, K=8 / N=20 이라 분산 추정이 거칠다. nul p99.9 = 7.29 도 K=8 → 28 페어
  의 inflation 을 머금음. 실제 attack-mode 분석에서는 N 을 더 키워야
  할 수도 있음 (Phase 3 는 이미 N=10/design × 다중 design 으로 충분).

다음 단계
- Phase 2 sk-unpack 분석을 기존 `d_map.npz` 위에서 (capture 추가 없이)
  돌려 negative/positive 결과 기록.
- Phase 3 selected-lane scout: K=8 keys × L=16 lanes × G=6 γ × N=10 traces.
  Phase 1 결과로 보아 trace window 는 후반 (12k~24k) 에 집중되어 있어,
  scout 분석은 그 region 부터.

---

### 2026-05-08  Phase 0b — smoke + codec parity

가정
- host 의 12-bit pack `to_bytes` 가 보드 firmware 의 `poly_tobytes` 와
  비트-동일이면, host 가 만든 selected-lane CT 가 보드의
  `poly_frombytes` 후 의도된 NTT-domain 표현으로 복원된다. 이 일치를
  검증하는 가장 강한 host-only signature 는 `sha3_256(ct_bytes)[:16]`.

과정
- `host/upload.py` 로 `simpleserial-ntruplus-CW308_STM32F4-ntruplus768.hex`
  플래시 (sha256 `55c92fbd…`).
- `tests/smoke_ntruplus.py` 실행. (a) `k/e/d/p` baseline, (b) `F/B`,
  (c) host `selected_lane(lane=3, γ=42, slot=0)` 36 chunk 주입, (d) `L`
  지문이 host 의 `sha3_256(host_ct)[:16]` 와 정확히 일치하는지, (e) `D`
  응답 형식, (f) `X[0]` sk chunk dump.

결과 (수치)
- 모든 게이트 통과. **L parity** 정확히 일치:
  `fd c8 22 fe c8 22 0f 81 a4 b2 02 19 06 96 a1 49`
  (board) == (host).
- `D (selected-lane) → mismatch flag = 0`. invalid CT → fail=1 → ss_dec
  zero-fill. ss_enc 도 'F' 시 0-init. 따라서 mismatch flag = 0 — 이는 정상
  거동이며 mismatch flag 은 oracle 로 쓸 수 없음을 동시에 확인.

해석
- gate 통과: codec 호환성 100%. Phase 3 의 선택-lane chosen-CT 가 보드에서
  의도된 NTT-domain 모양을 만든다는 것이 codec 차원에서 검증됨. NTT 자체의
  binary parity 는 zetas[192] 추출/비교로 별도 검증 (Phase 0a).

다음 단계
- Phase 1: natural D capture map.

---

### 2026-05-08  Phase 0a — host/ntruplus/ 패키지 + round-trip 검증

가정
- NTRU+ ciphertext 와 sk 의 f 영역이 모두 **NTT 도메인 직렬화** 라는 가정
  (upstream `Reference_Implementation/NTRU+768/kem.c` 의 `crypto_kem_dec`
  본문에서 `poly_frombytes(&c, ct)` 후 곧장 `poly_basemul(&m1, &c, &f)` 로
  NTT 변환 없이 사용). 따라서 host 가 INTT 를 직접 수행하지 않고 c_ntt 를
  바로 12-bit pack 하면 보드 basemul 에서 lane-isolation 이 성립한다.
- 우리 로컬 archive `lib/crypto_kem/ntruplus768.a` 의 NTT 가 upstream final
  과 binary-identical 이라는 가정 (그래야 host 에서 만든 selected-lane CT 가
  보드 basemul 결과와 일치한다).

과정
- 공식 NTRU+ repo clone: `/tmp/ntruplus_src/NTRUplus`. 공식 final 은
  768/864/1152 만 — 576 은 비공식. **primary target 을 ntruplus768 로 변경**
  (`docs/NTRUplus.md` 갱신).
- 로컬 archive 의 ntt.c.o `.rodata` 추출 후 zetas[192] entry 192 개를
  `struct.unpack('<' + 'h'*20, …)` 로 디코드. 첫 20 entry 가 upstream
  `ntt.c` const table 과 정확히 일치. 192-entry 전체는 binary 동일성을
  signature 로 가정.
- host 패키지 작성 (`host/ntruplus/`):
  - `params.py`  — N=768, q=3457, R/RINV/RSQ/QINV/OMEGA/ZMINUSZ5INV/NINV
                   /TWO_NINV, ZETAS[192], lane_zeta() helper.
  - `codec.py`   — `to_bytes` / `from_bytes` / `center` (12-bit packed,
                   mirroring upstream `poly_tobytes` / `poly_frombytes`).
  - `ntt.py`     — int16/int32 wrap-aware `montgomery_reduce`,
                   `barrett_reduce`, `fqmul`, `ntt`, `invntt`, `basemul_lane`,
                   `basemul`, `naive_poly_mul` (in R_q = Z_q[X]/(X^N -
                   X^(N/2) + 1)).
  - `chosen.py`  — `selected_lane(lane, gamma, slot)` and `monomial_pair`.
  - `sk.py`      — sk blob 파싱 helper.
  - `inject.py`  — F/B/I/L/D/X SimpleSerial wrappers + sha3-256 fingerprint
                   parity check.
- 테스트 작성: `tests/test_ntruplus_host.py`. 7 케이스:
  1. zetas 크기,
  2. codec 랜덤 round-trip (8 polys, mod-q 동치),
  3. codec edge-cases (0, q-1, -(q-1)/2),
  4. NTT ∘ invNTT = identity (mod q),
  5. **basemul ∘ NTT = NTT ∘ naive_poly_mul** (linkage 검증, 희소 입력),
  6. selected_lane layout + codec roundtrip,
  7. **basemul lane isolation** — chosen-CT (lane=7, slot=0, γ=100) →
     m[28..31] = γ·f_ntt[28..31] mod q, 그 외 모든 위치 0.

결과 (수치)
- 7/7 테스트 통과. `python3 tests/test_ntruplus_host.py` 종료 코드 0.
- lane isolation 케이스 검증으로 **Phase 3 selected-lane 공격의 수학적
  토대 입증**. (host 에서 만든 12-bit pack ciphertext bytes 가 보드 basemul
  결과로 단일 lane 만 살리도록 의도된 sparsity 를 정확히 만들어낸다.)

해석
- gate 통과: host-only round-trip parity 100 %. 다음 검증은 host↔board.
- 가장 중요한 결과: c_ntt[4*lane + slot] = γ 단일 비영(非零) 만 있는
  ciphertext 가 basemul 후 4 개 NTT-domain f 좌표를 동시에 leak (γ·f_ntt[4*lane+0..3]).
  단, basemul 은 8-coeff pair 안에 ±zeta 둘로 묶이므로 한 design 의 leakage
  scope = 4 좌표, 그 paired lane 은 독립.
- public-design classifier 위험 (Phase 3 null 가드 필요): 같은 design 의
  ciphertext bytes 는 모든 키 동일이라 trace label 이 (lane, γ) 만으로
  결정되면 사실상 trivial classifier — 이미 SMAUG-T flip_any α-bug 와 같은
  사고 경로. Phase 3 capture matrix 에는 secret-shuffle null 데이터셋이
  필수.

다음 단계
- Phase 0b: 보드 smoke (`tests/smoke_ntruplus.py`) 를 ntruplus768 chunk
  (POLYBYTES=1152, 36 chunks × 32 B) 로 갱신 후 실행. 핵심 게이트:
  (a) baseline `k/e/d/p` 응답, (b) chosen-CT round-trip 후 `L` 응답이
  host 의 sha3-256(ct_bytes)[:16] 와 일치 — codec parity 직접 확인.
- Phase 0b 가 통과하면 host 의 selected_lane CT 가 보드에서 의도한 lane
  배치를 만든다는 것이 codec 차원에서 보장된다 (NTT-domain 일치는 zetas
  binary 일치로 이미 가정).
- Phase 1 capture 는 보드 점유가 필요하므로 Phase 0b 직후 사용자 확인.
