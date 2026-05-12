# NTRU+768 — Phase Experiment Log

[`README.md`](README.md) 의 "Flow at a glance" 표가 navigational summary,
이 문서가 phase 별 가정·실험·결과·해석을 누적한 detail 로그다 (역시간순,
신규 entry 는 맨 위).

Last entry: 2026-05-11 (Phase 4.6-G200 PoI/window diagnostic).

---

## 2026-05-11 Phase 4.6-G200 PoI/window diagnostic — oracle gain 은 selection bias

**가정**. G=200 compact scout 의 fixed/slot PoI 결과가 약한 이유가 단순 timing
miss 라면, oracle PoI (true candidate 의 best sample 직접 고름) 가 회수율을
올리되 self-oracle null 과 분리되어야 한다.

**실험**. `n56_phase46_g200_poi_diagnose.py` (window w ∈ {16, 32, 64, 96}) +
`n57_phase46_g200_oracle_null.py` (same-procedure null) +
`n58_phase46_g200_window_reducers.py` (mean/top-k/softmax) +
`n59_phase46_g200_cv_window.py` (held-out γ fold CPA).

**결과**. oracle PoI 의 top-10 gain 이 candidate-set null 과 통계적으로 분리
안 됨. attack-compatible PoI 들은 top-100 0/16 ~ 1/16 유지. window reducer
도 mean/top-k/softmax/max 모두 0/16 top-10. CV-window CPA 도 stable
local-PoI evidence 없음.

**해석**. G=200 single-victim 의 oracle 분명 PoI gain 은 self-oracle
selection bias. simple G 확장은 **negative scout**. Phase 4.6 single-victim
recovery 는 partial-information-disclosure 로 격하.

**다음**. paper writing 으로 전환. additional capture 는 victim variance
증가 가능성 검토.

---

## 2026-05-11 Phase 4.6-G200 compact scout — G 확장 score-sharpness gate

**가정**. Phase 4 의 G=78 wide-γ 대비 G=200 compact (좁은 ADC window, fewer
samples) 가 score sharpness 를 올려 회수율 증가를 가져올 수 있다.

**실험**. `n55_phase46_g200_capture.py -K 1 -L 0,64,80,128 -N 8 -G 200
-s 6000` (4 calib lanes, victim 3 기준). 총 6400 traces, 0 timeouts.

**결과**. M1/full-stack top-100 0/16, M5 top-100 1/16. **simple G 확장 =
negative scout**. follow-up diagnostic (위 entry) 으로 oracle gain 은 bias 임을
검증.

**해석**. G 확장만으로는 σ_α saturation 천장 못 깸. Phase 4.6 main result 는
3-victim aggregate (다음 entry) 로 종결.

---

## 2026-05-11 Phase 4.6 정리 — Wide-coverage 3rd victim + 3-victim aggregate

**가정**. Phase 4.5 (2 victims, 40 lanes) 의 projection 을 wide-coverage 3rd
victim (48 lanes) 으로 검증 시, single-victim coordinate yield 가 stable 한지
확인 가능.

**실험**. victim 3 (8ce340e4bdfa80fc, L=48 N=8). 분석: full-stack pipeline +
candidate export + overlap projection.

**결과 (192 cases)**:
- M1 baseline 3/192 top-100, M5 baseline 9/192, full-stack 1/192.
- 3-victim cumulative (352 cases, 88 lanes): M1 9/352, M5 12/352, Full 8/352,
  channel-mixed union 29/352.

**해석**. 48-lane victim 의 M1 yield 는 2-victim projection 보다 낮음 →
single-victim coord yield 에 victim-to-victim variance. 보수적 paper claim:
**~17-26 top-100 NTT coordinates / victim** (single-pipeline), channel-mixed
union upper bound ~63 coords/victim. Full sk recovery 는 여전히 infeasible.
`candidate_null` 반영: raw top-100 counts 는 set-size null 과 구분되지
않으므로 engineering result 로 격하.

**산출물**. `traces/ntruplus768/phase45/wide_K1_L48_N8.npz`,
`results/ntruplus/phase45/{wide_K1_L48_N8.npz, SUMMARY.md, overlap_projection.md,
channel_hitlist.md, candidate_{export,pressure,null}.md}`.

---

## 2026-05-10 Phase 4.5 — 단일 victim multi-lane sk leakage 입증

**가정**. Phase 4 (multi-victim 240 cases, 60 victims) 의 "단일 victim 안에서
회수" 정의를 직접 입증하기 위해, victim 1개 안에서 multi-lane sk leakage 를
보이면 entropy reduction 의 실측 데이터 확보.

**실험**. 2 victims:
- victim 1 (235e27a0..., L=16 stride 12, N=8) — scout.
- victim 2 (8c6aacfa..., L=24 stride 8, N=16) — main, more lanes/traces.

분석: `n43_singleVictim_multilane.py`, `n45_singleVictim_combine.py`,
`n46_partial_recovery_analysis.py`.

**결과 (160 cases)**:
- victim 1: M1 top-100 3/64, full 4/64.
- victim 2: M1 top-100 3/96, full 3/96.
- aggregate: M1 6/160 (3.75%), full 7/160 (4.4%), M1 ∪ Full ≈ 9 unique.
- 0 TOP-1, 1 top-10 (lane=104 slot=0 f=2563, \|f_c\|=894, **M1 baseline
  rk=2** — cross-channel large-\|f_c\|).
- 192-lane projection (linear extrap): M1 ~29, Full ~34, Union ~43
  coords/victim. **superseded** by Phase 4.6 conservative ~17-26 range.

**해석**. multi-lane sk leakage 단일 victim 안에서 partial 회수 가능 입증.
linear LANE_SLOT_OFFSET interpolation (calibrated lanes 0/64/80/128 의 median
slot drift 사이 보간) 은 effective (M1 baseline 3 → slot-PoI 6 top-100 in
main capture). Honest 결론: partial information disclosure, lattice 도달 X.

---

## 2026-05-08 Phase 4 — multikey 기반 attack-valid + 진단 평가

**가정**. Phase 3 scout 의 single-key signal 이 multi-key 에서도 stable 하면
240-case (4 lanes × 60 sk × 4 slots) 통계로 top-1/top-100 recovery rate 를
직접 계산 가능. dual-channel (M1, M5) + per-slot PoI calibration + Zsum
ensemble 도 평가 대상.

**실험**. K=8 capture (lane=0 wide-γ + 추가 lanes 64/80/128), 60 victims
누적. 분석: `n23_wideg_K4_attack.py` (attack-valid evaluator),
`n24_combined_analysis.py` (cumulative), `n39_full_stack.py` (full pipeline).

**결과 (240 cases)**:

| Pipeline | top-1 | top-10 | top-100 | top-500 |
|---|---:|---:|---:|---:|
| M1 baseline | 1 | 1 | 10 | 35 |
| M5 baseline | 0 | 0 | 6 | 38 |
| Full stack | 1 | 2 | 9 | 45 |
| Union M1 ∪ Full | 2 | 3 | **17** | ~50 |

- 2 perfect TOP-1: `f=16` (K=8 key 5 slot 1, M1 channel, N=2 sufficient),
  `f=882` (K=8 key 7 slot 3, M5 + full-stack, N=16 sufficient).
- 4 lanes 모두 회수 (0/64/80/128).

**해석**. 본 프로젝트 첫 attack-valid 결과. 5종 mechanistic finding (README §5)
도 함께 정립 — N-saturation, dual channel, per-slot drift, b·\|f_c\| 의존성,
σ_α ceiling. paper-grade.

---

## 2026-05-08 Phase 3 leak-per-PoI = 4 coefficients (one design → 4 slots)

**가정**. lane k 에 `c_ntt[4k + i] = γ · δ_{i,0}` 으로 chosen-CT 를 주입하면
`r[i] = γ · f̂[4k + i]` (i ∈ {0..3}) 가 basemul 출력. 즉 design 하나에서
**4 slots** 의 f̂ 좌표가 동시에 leak.

**실험**. lane=0, K=4, G=12, slot 별 separate PoI scan.

**결과**. design 하나당 4 slots all 회수 → cost amortization 4×. Phase 4
240-case (= 60 victims × 4 slots) 의 efficient evaluation 가능 근거.

---

## 2026-05-08 Phase 3 basemul-timing model + clean signal (v3 분석 완)

**가정**. v2 의 γ-byte 오염 발견 후 (다음 entry), const-γ set 으로 정화하면
selected-lane signal 이 single-key 에서 stable 하게 잡혀야.

**실험**. `n10_phase3_basemul_model.py` 로 basemul timing model 검증 +
`n11_phase3_corr_plot.py` 로 corr profile 시각화. K=1, G=12.

**결과**. clean single-key signal 확인. PoI 가 basemul 시간대 정확히 위치.
multikey 전이 (다음 phase) 진행 가능.

---

## 2026-05-08 Phase 3 scout v3 — HW(γ) const γ set (G=12, single key)

**가정**. v1/v2 의 γ-byte leakage 오염 제거 위해 HW(γ) 고정 const-γ set 만
사용. γ 가 모두 같은 HW 면 byte leak 무력화.

**실험**. `n05_phase3_scout.py` v3, K=1, G=12 (HW(γ)=1 또는 2 const).

**결과**. v3 에서는 selected-lane PoI 신호 깨끗 (γ-byte 오염 없음). 4 slots
identification 가능.

---

## 2026-05-08 Phase 3 scout v2 critique — γ-byte leakage 가 v1 PASS 을 오염

**가정**. v1 (다음 entry) 의 PASS signal 이 실제 NTT-coordinate leak 인지,
아니면 γ 의 byte HW 자체가 trace 와 correlate 하는 spurious 인지 분리 검증.

**실험**. γ-byte HW vs sk-related HW model 비교 (n09_phase3_decompose.py).

**결과**. γ-byte HW 가 v1 single-key signal 의 큰 fraction. **v1 PASS 은
γ-byte 오염**. v3 (const-γ) 으로 전환.

---

## 2026-05-08 Phase 3 scout v1 — selected-lane signal (single key, G=6)

**가정**. `c_ntt[4k] = γ`, 나머지 0 의 chosen-CT 를 K=1, G=6 wide-γ 로
주입하면 `f̂[lane·4]` 한 좌표의 HW(γ·f̂) signal 이 검출되어야.

**실험**. `n05_phase3_scout.py` v1, lane=0, K=1, G=6.

**결과**. PASS — single-key signal 검출. 단 v2/v3 검증 (위 두 entries) 으로
γ-byte 오염 발견. clean PASS 는 v3 에서.

---

## 2026-05-08 Phase 2 — byte-HW triage on Phase 1 traces (K=8, N=20)

**가정**. Phase 1 의 sk-dependent leak 가 byte-HW 단위 CPA 로 좁혀지는지.

**실험**. `n03_phase2_skunpack.py` per-byte HW CPA on Phase 1 traces.

**결과**. per-byte z 값이 약함. 단일 byte HW 로는 PoI 못 좁힘. → Phase 3
selected-lane CPA 로 전환.

---

## 2026-05-08 Phase 1-timeline — full decap mapping (decimate=4)

**가정**. `crypto_kem_dec` 의 어느 함수 호출 구간이 sk-dependent leak 의
source 인가? 

**실험**. `n04_decap_timeline.py` decimate=4 timeline mapping + 함수
boundary 마커.

**결과**. `poly_basemul(m1, c, f)` 시간대가 단연 leak source — attack hook B
의 정확한 위치. 후속 phase 의 PoI window 기준.

---

## 2026-05-08 Phase 1 — natural D map (K=8, N=20, valid CT)

**가정**. natural `D` (chosen-CT 없는 valid encap → decap) trace 에서도
sk-dependent leak 가 존재하는가?

**실험**. `n01_phase1_capture.py` K=8 sk × N=20 traces (valid CT only).
`n02_phase1_analyze.py` TVLA + byte-HW preview.

**결과**. basemul 시간대에 sk-dependent t-statistic 검출. PoI 후보 다수.
Phase 2 byte-HW triage 진행 근거.

**산출물**. `traces/ntruplus768/phase1/d_map_K8N20.npz`,
`results/ntruplus/phase1/d_map_overview.png`.

---

## 2026-05-08 Phase 0b — smoke + codec parity

**가정**. NTRU+768 펌웨어 빌드/플래시 후 host-side codec 과 보드 응답이
binary-identical 한가?

**실험**. `tests/ntruplus/smoke_ntruplus.py` — hex 빌드, F (keygen), E
(encap), D (decap) round-trip + host codec parity.

**결과**. encap fingerprint 100% match. decap ack 0 / mismatch 0. host
codec ↔ 보드 codec 정합 100%.

---

## 2026-05-08 Phase 0a — host/ntruplus/ 패키지 + round-trip 검증

**가정**. NTRU+ 의 host-side NTT/codec/inject 를 build → C reference 와 binary
parity 가 보장되는 Python 패키지로 만들면 chosen-CT plumbing 의 sanity 가
host 측 sha3_256 비교로 검증 가능.

**실험**. `host/ntruplus/{codec,ntt,params,sk,chosen,inject}.py` 신설 +
`tests/ntruplus/test_ntruplus_host.py` round-trip 테스트.

**결과**. NTT/codec/basemul round-trip 100% 통과. chosen-CT 빌더 도 sha3_256
parity 검증. Phase 1 이후 chosen-CT 캡처의 무결성 baseline.

**산출물**. `host/ntruplus/` 패키지 6 모듈.
