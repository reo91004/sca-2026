# NTRU+768 Chosen-CT NTT-Domain CPA — Track Overview

ChipWhisperer-Lite + CW308T-STM32F415 위에서 NTRU+768 (KpqC final, GitHub
v1.0 / 2026-01-30) 의 natural `crypto_kem_dec` trace 로부터 NTT-domain 비밀키
좌표를 회수하는 chosen-CT side-channel 공격. **Phase 4 종료, paper-grade**.

Last updated: 2026-05-12.

---

## 1. Result summary

**Phase 4 main**: 240 (key, lane, slot) cases (4 lanes × 60 victims × 4 slots).

| Pipeline | top-1 | top-10 | top-100 | top-500 |
|---|---:|---:|---:|---:|
| M1 baseline (HW(γ·f̂ mod q)) | 1 | 1 | 10 (4.17%) | 35 |
| M5 baseline (HW(montgomery_reduce(γ·f̂))) | 0 | 0 | 6 | 38 |
| Full stack (slot-PoI + Zsum) | 1 | 2 | 9 (3.75%) | 45 (18.75%) |
| **Union M1 ∪ Full** | **2** | **3** | **17 (7.08%)** | ~50 (~21%) |

- **2 perfect TOP-1** (`f=16` at N=2 traces via M1; `f=882` at N=16 via M5/full).
- **Recovered lanes**: 0, 64, 80, 128 (all four scanned NTT lanes).
- This is the project's first attack-valid result. Cross-key transfer is
  **not** required — single-victim CPA suffices.

**Phase 4.5/4.6 supporting**: 3 victims × 88 lanes × 4 slots = 352 single-victim
cases. M1 9/352, M5 12/352, Full 8/352 top-100. Conservative 192-lane projection
≈ 17-26 NTT coordinates / victim per single pipeline (channel-mixed union
upper bound ≈ 63/victim). Reported as **partial NTT-coordinate information
disclosure**, not full sk recovery (lattice β_req ≈ 300 remains infeasible at
this rate).

## 2. Threat model

- **Target**: NTRU+768, level smallest. Standard `simpleserial-ntruplus`
  firmware on CW308T-STM32F415. q = 3457, N = 768.
- **Attacker capability**: chosen-CT injection (`F/B/I/L/D` SimpleSerial),
  power trace at standard ADC clock, public parameters, public pk dump.
- **Forbidden**: target-side `mu'` / `ss` / `sk` dump used in inference
  (IND-CCA violation; see SMAUG-T track's 2026-05-04 corrigendum for the
  precedent). Diagnostic triggers are for **PoI calibration only**, never
  for attack-time labels.
- **Attack-valid trace** = `cmd_decap_inject` `'D'` trace. CPA labels use
  the public chosen γ and a sk-independent timing model only.

## 3. Background

NTRU+ uses a Z_q[X]/(X^N − X^(N/2) + 1) ring with q = 3457 (NTT-friendly).
`crypto_kem_dec` calls `poly_basemul(m1, c, f)` early — this is **attack hook
B**, where `c[i] = γ · δ_{i, 0}` injected via chosen-CT makes
`r[i] = γ · f̂[lane·4 + i]` for i ∈ {0..3}. CPA over `γ` reveals
HW(γ · f̂) — modeled by two complementary HW channels:

- **M1**: `HW(γ · f̂ mod q)` — mod-q reduction operand.
- **M5**: `HW(montgomery_reduce(γ · f̂))` — Montgomery reduction store operand.

The two channels recover different |f̂| regimes — M1 favors small |f̂|, M5
recovers large |f̂|. Dual-channel ensemble + per-(lane, slot) PoI drift
calibration are the paper-grade mechanisms.

## 4. Flow at a glance

자세한 entries 는 [`experiments.md`](experiments.md).

| Phase | 문제 / 가정 | 실험 | 핵심 결과 |
|---|---|---|---|
| 0a | host-side NTT/codec 가 보드와 정합한가? | `host/ntruplus/` 패키지 + round-trip | NTT/codec/basemul 100% round-trip |
| 0b | 펌웨어 동작 sanity? | smoke + codec parity | hex 빌드/플래시/encap/decap PASS |
| 1 | natural `D` leak map 은? | K=8 sk × N=20, TVLA + byte HW | basemul 시간대 sk-dependent leak 식별 |
| 1-timeline | leak 의 함수 호출 위치? | decimate=4 timeline | `poly_basemul(m1, c, f)` 우세 (hook B) |
| 2 | byte-HW triage 가 PoI 좁히나? | byte CPA | 단일 byte HW 부족 → selected-lane 으로 전환 |
| 3-scout v1/v2/v3 | NTT-domain selected-lane single-key signal? | G={6,12} const-γ, 다 lane | v3 const-γ set 으로 4 slots/lane 식별 |
| 3-multikey | cross-key PoI 일관성? | K=4 sk × 동일 designs | per-key CPA + Bonferroni 안정 |
| 3-decompose | leak source 가 mod-q 만 인가? | basemul-timing model + HW variants | **M5 (Montgomery) 채널 발견** |
| 4-recover | multi-victim TOP-1 회수 가능? | 240 cases × dual-channel + slot-PoI | **2 perfect TOP-1, 17/240 top-100** |
| 4.5 | single-victim multi-lane 가능? | 2 victims × 88 lanes | partial 회수, lane=104 near-TOP-1, 31% entropy 감소 |
| 4.6 | wide-coverage 3rd victim 으로 검증? | G=200 single-victim 16 calib lanes | 보수적 17-26 coords/victim, simple G 확장은 negative scout |

## 5. Paper-grade mechanistic findings

1. **N-saturation** (Phase 4): `corr = b·σ_HW / √((b·σ_HW)² + σ_α²)`,
   measured/formula ratio 0.27 at N=64 (formula overpredicts 4×). N > 16
   gives no further gain.
2. **Dual leak channel** (Phase 3-decompose): M1 (mod-q) + M5 (Montgomery
   reduction) are complementary; M5 reaches large |f̂| coordinates that M1
   misses.
3. **Per-(lane, slot) PoI drift** (Phase 4): firmware-precise calibration.
   Drift is sk-independent (firmware constant); one calibration applies to
   all victims. Sample drifts: lane=0 uniform +24, lane=80 (+8,+7,−3,−3),
   lane=128 (0,−2,0,−1).
4. **b · |f̂| dependence** (Phase 4): recovery requires b ≥ 0.0018; b is
   correlated with |f̂|, structurally favoring `|f̂| < 50` plus
   Montgomery-channel large-|f̂| outliers.
5. **σ_α saturation ceiling** (Phase 4): cross-key α-removal and ensemble
   methods can't pierce the σ_α/(b·σ_HW) ratio — fundamental.

## 6. Figures

핵심 결과 figure (paper-relevant). 모두 `results/ntruplus/<phase>/figures/` 에 보존.

| Figure | 설명 | Path |
|---|---|---|
| F0 | chosen-CT NTT layout 개념도 | `results/ntruplus/phase4/figures/F0_chosen_ct_layout.png` |
| F1 | b vs \|f_c\| | `results/ntruplus/phase4/figures/F1_b_vs_fc.png` |
| F2 | per-slot drift | `results/ntruplus/phase4/figures/F2_per_slot_drift.png` |
| F3 | M1 vs M5 channel comparison | `results/ntruplus/phase4/figures/F3_m1_vs_m5.png` |
| F4 | recovery by pipeline | `results/ntruplus/phase4/figures/F4_recovery_by_pipeline.png` |
| F5 | lane SNR | `results/ntruplus/phase4/figures/F5_lane_snr.png` |
| F6 | N-curve | `results/ntruplus/phase4/figures/F6_ncurve.png` |
| F7 | saturation model | `results/ntruplus/phase4/figures/F7_saturation.png` |
| F8 | dual-channel × \|f_c\| | `results/ntruplus/phase4/figures/F8_dual_channel_fc.png` |
| F9 | corr by N (saturation) | `results/ntruplus/phase4/figures/F9_corr_by_N.png` |
| F10 | per-victim recovery (Phase 4.5) | `results/ntruplus/phase45/figures/F10_per_victim_recovery.png` |
| F11 | lane heatmap (Phase 4.5) | `results/ntruplus/phase45/figures/F11_lane_heatmap.png` |
| F12 | cumulative lanes (Phase 4.5) | `results/ntruplus/phase45/figures/F12_cumulative_lanes.png` |

미리보기 (F4 — recovery by pipeline, 240 cases):

![F4 recovery by pipeline](../../results/ntruplus/phase4/figures/F4_recovery_by_pipeline.png)

## 7. Reproduction

### 7.1 펌웨어 빌드/플래시

```bash
# NTRU+768 펌웨어 빌드
make -C firmware/simpleserial-ntruplus PLATFORM=CW308_STM32F4

# 캡처 스크립트가 hex 경로를 hard-code 하므로 별도 upload 단계 없음
# (각 `scripts/ntruplus/n01_phase1_capture.py` 같은 캡처가 자체 `cw.program_target`)
```

### 7.2 핵심 캡처 명령

```bash
# Phase 1 — natural D map (K=8 sk × N=20)
python3 scripts/ntruplus/n01_phase1_capture.py -K 8 -N 20 -s 24400 \
    -o traces/ntruplus768/phase1/d_map_K8N20.npz

# Phase 3 — selected-lane wide-γ scout (lane=0, K=8, N=32, G=78)
python3 scripts/ntruplus/n18_phase4_g78_capture.py -K 8 -N 32 -G 78 -L 0 \
    -o traces/ntruplus768/phase3/wideg_lane0_K8N32.npz

# Phase 4.5 — single-victim multi-lane main capture (victim 2)
python3 scripts/ntruplus/n43_singleVictim_multilane.py -K 1 -L 24 -N 16 \
    -o traces/ntruplus768/phase45/main_K1_L24_N16.npz

# Phase 4.6 — G=200 compact scout
python3 scripts/ntruplus/n55_phase46_g200_capture.py -K 1 -L 0,64,80,128 \
    -N 8 -G 200 -s 6000 \
    -o traces/ntruplus768/phase46/g200_calib_K1L4N8_s6000.npz
```

### 7.3 핵심 분석

```bash
# Phase 4 attack-valid full pipeline (M1 + M5 + Zsum + per-slot)
python3 scripts/ntruplus/n39_full_stack.py
python3 scripts/ntruplus/n40_full_stack_ncurve.py

# Phase 4 cumulative statistics
python3 scripts/ntruplus/n24_combined_analysis.py

# Phase 4.5 candidate hit-list / projection
python3 scripts/ntruplus/n50_phase45_overlap_projection.py
python3 scripts/ntruplus/n52_phase45_candidate_export.py

# Paper figures
python3 scripts/ntruplus/n41_paper_figures.py
python3 scripts/ntruplus/n44_paper_figures_extra.py
```

### 7.4 호스트 단위 테스트

```bash
python3 tests/run_all.py
# 89/89 OK 가 정상
```

## 8. Limitations / Future work

- **Partial recovery only**: 240 cases 에서 17 top-100 (7.08%) 는 lattice
  full-sk attack 까지 도달하지 못한다 (β_req ≈ 300). 명시적으로 "partial
  NTT-coordinate information disclosure" 로 보고.
- **σ_α saturation 천장**: cross-key α-removal, ensemble 모두 fundamental
  ratio σ_α/(b·σ_HW) 못 깨짐. profile-free CPA 의 한계.
- **Phase 4.5/4.6 victim variance**: single-victim coordinate yield 가
  victim 별로 크게 다름 — 192-lane projection 은 conservative range 로만
  사용 (lucky-victim 효과 분리).
- **다음 후보**: (a) profiled / template attack 으로 σ_α 천장 도전, (b)
  Karatsuba/Toom-Cook sub-step SASCA, (c) lattice-aided recovery threshold
  추정 — 모두 별도 단계.
- **Diagnostic 분리**: 본 결과는 모두 natural `D` trace 만. PoI calibration
  은 sk-independent firmware-constant 로 1회. attack-time 에 sub-trigger
  로 µ' 등 leak 사용 안 함.
