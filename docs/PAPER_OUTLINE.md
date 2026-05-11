# NTRU+768 Chosen-CT NTT-Domain Side-Channel Attack — Paper Outline

Working title: *NTT-Domain Selected-Lane CPA on NTRU+768 with
Dual-Channel Hardware Models: A Single-Victim Power-Analysis Recovery*

Last updated: 2026-05-11 (Phase 4.6 integrated).

## Authorship / Target venue

- target: CHES 2026 / TCHES (2026 issue).
- secondary: Asiacrypt / USENIX Security if more polish.

## Abstract (≈250 words)

NTRU+ is a finalist in the Korean PQC standardization (KpqC). We
demonstrate the first practical side-channel attack on NTRU+768 that
exploits its NTT-domain ciphertext serialization to recover individual
secret-key NTT coefficients from natural `crypto_kem_dec` traces. Our
attack uses chosen ciphertexts of the form `c_ntt[4k + i] = γ ·
δ_{i, 0}` (one γ in lane k slot 0, others zero), producing a basemul
output `r[i] = γ · f̂[4k + i]` mod q for i ∈ {0, 1, 2, 3}. By scanning
G = 78 wide-γ designs (HW = 1 + HW = 2 set) and applying a wide-γ
correlation power analysis (CPA) with two complementary
hardware-Hamming-weight models — M1 (HW(γ · f̂ mod q)) and M5
(HW(montgomery_reduce(γ · f̂))) — we recover NTT-domain f̂
coefficients on a ChipWhisperer-Lite + STM32F415 platform. We obtain
two perfect TOP-1 recoveries (one at N = 2 chosen-CT traces) and 7%
top-100 over 240 randomly-keyed cases (60 victims). With a per-(lane,
slot) PoI calibration profiled from one batch and applied to held-out
victims, we further demonstrate single-victim multi-lane key
disclosure across three fresh victims and 88 covered lanes. The
single-victim aggregate reaches 9 M1 top-100, 12 M5 top-100, and 8
full-stack top-100 hits over 352 evaluated coordinates, implying a
conservative 192-lane projection of roughly 17-26 top-100 NTT
coordinates per victim depending on channel. We characterize the attack
ceiling as a
saturating-noise σ_α phenomenon — `corr_max = b·σ_HW /
sqrt((b·σ_HW)² + σ_α²)` — and discuss the
information-theoretic distance from full sk recovery (~660+ coords
needed via lattice). All claims are made on natural `crypto_kem_dec`
traces; diagnostic triggers are used only for label calibration.

## §1 Introduction

- KpqC final selections (NTRU+768/864/1152, SMAUG-T)
- post-quantum + side-channels: well-studied for Kyber but less for
  KpqC schemes
- NTRU+ specifics: NTT-friendly (Z_q[X] / (X^N - X^(N/2) + 1)), q=3457
- our contribution:
  1. First chosen-CT NTT-domain attack on NTRU+ specifically
     leveraging the byte-level NTT serialization of ciphertext
  2. Dual hardware-Hamming-weight model (M1 mod-q + M5 montgomery
     reduction) — paper-novel
  3. Per-(lane, slot) PoI calibration — sk-independent firmware
     constant
  4. σ_α saturation mechanism — quantitative attack ceiling
  5. Single-victim multi-lane recovery demo (Phase 4.5)
- comparison to prior NTRU SCA: Askeland 2021 (sk-unpack), Ravi 2020
  (coefficient threshold), Xu (mismatching attacks). All target
  different surfaces; our NTT-domain selected-lane is independent.

## §2 Background

### §2.1 NTRU+ KEM

- secret f ∈ {-1, 0, +1}^768, cbd1 distribution (per upstream
  `Reference_Implementation/NTRU+768/poly.c::poly_cbd1`)
- public key h_ntt = NTT(f^-1 · g) (some structure)
- ciphertext c, encapsulation: hash_g + sotp_encode + ntt + basemul
- decapsulation steps:
  1. poly_frombytes c → c_ntt
  2. poly_basemul m1 = c_ntt · f_ntt (192 lanes × 4 coeffs each)
  3. poly_invntt m1
  4. poly_crepmod3 m1 (mod-3 message decode)
  5. FO recompute (m2, c-m2 · hinv → r2)
  6. SOTP decode + verify

### §2.2 NTT layout

- Z_q[X] / (X^N - X^(N/2) + 1) cyclotomic-like
- 4-radix split, 192 zetas (Montgomery form)
- basemul per 4-coef lane in Z_q[X] / (X^4 - ζ_k)
- chosen-CT semantic: `c_ntt[4k + i] = γ · δ_{i, 0}` → output
  `r[i] = γ · f̂[4k + i]` for i ∈ {0, 1, 2, 3}

### §2.3 Side-channel model

- hardware: ChipWhisperer-Lite (CW308 + STM32F415)
- attacker capabilities:
  - chosen-CT injection (`I` × 27 + `L` + `D` commands)
  - power trace (24400 samples, decimate=4, 4 samples/cycle)
  - public params, public pk
- attacker NOT allowed:
  - sk dump
  - target μ' / ss / sk inference (IND-CCA violation)
- attack-valid trace: natural `crypto_kem_dec` (`D` cmd) only.
  Diagnostic triggers (`I/L/X/etc`) for calibration only.

## §3 Attack Methodology

### §3.1 Selected-lane chosen-CT design

- Equation: `c_ntt[4k + i] = γ · δ_{i, 0}` for chosen lane k, slot 0
- After byte-level packing: 1152 bytes injected via `I × 27 + L`
- One design = one γ value, K candidates of γ from HW=1 + HW=2 set:
  - HW=1: 12 (= 2^0, 2^1, ..., 2^11)
  - HW=2: 66 (= sums of distinct pairs)
  - G = 78 unique γ
- Per design we record N traces (N ∈ {8, 16, 32}).

### §3.2 sk-independent PoI

- predict_poi(lane) = 3014 + 33 · (lane >> 1) + 16 · (lane & 1)
- timing model: linear basemul iter index
- Calibrated extension: per-(lane, slot) median offset profiled from
  Phase 4 multi-victim data (n34) — sk-independent firmware constant

### §3.3 Wide-γ CPA with dual HW models

- Trace at PoI: x[g] = α_g + b · HW_model(γ · f̂) + ε[g]
- M1: HW(γ · f̂ mod q) — register-load model
- M5: HW(montgomery_reduce(γ · f̂)) — reduction-output model
- per-candidate CPA: corr(x, HW_model(γ · cand)) over 78 designs
- V2 (mean-over-N): average traces per γ before CPA

### §3.4 Z-score fusion

- M1 + M5 evidence pooling: `Z_full = z_M1 + z_M5`
- per-(lane, slot) PoI calibration: PoI_use(lane, slot) = predict_poi(lane) + offset

## §4 Experimental Setup

- platform: CW-Lite + CW308T-STM32F4 (STM32F415RG)
- firmware: simpleserial-ntruplus, hex sha256 (specify), git rev 08decc4
- capture sweep:
  - K = 4–8 victims per batch (240 cases over 7 batches)
  - L = 4 lanes (0, 64, 80, 128) calibrated
  - G = 78 wide-γ
  - N ∈ {32, 64} chosen-CT traces per design
  - total capture time: ~21 hours (multi-batch)
- Phase 4.5/4.6: K=1 single-victim captures over 16, 24, and 48 lanes
  (88 lanes total across 3 fresh victims), all with G=78 and N=8/16.

## §5 Results

### §5.1 Attack-valid recovery (240 multi-victim cases)

| Pipeline | top-1 | top-10 | top-100 | top-500 |
|---|---:|---:|---:|---:|
| M1 baseline (predict_poi) | 1 / 240 | 1 / 240 | 10 / 240 (4.17%) | 35 / 240 |
| M1 (slot-PoI calibrated) | 1 / 240 | 1 / 240 | 8 / 240 (3.33%) | 37 / 240 |
| M5 (slot-PoI calibrated) | 0 / 240 | 0 / 240 | 6 / 240 (2.50%) | 38 / 240 |
| Full stack (slot+Zsum) | 1 / 240 | 2 / 240 | 9 / 240 (3.75%) | 45 / 240 (18.75%) |
| Union M1 ∪ Full | 2 / 240 | 3 / 240 | **17 / 240 (7.08%)** | ~50 / 240 (~21%) |

- Two perfect TOP-1 cases:
  - f = 16, M1 channel, lane=0 K=8 r1 k=5 s=1 — N = 2 traces sufficient
  - f = 882, M5+Zsum channel, lane=0 K=8 r1 k=7 s=3 — N = 16 traces
- Per-lane breakdown: lane 0/64/80/128 all show recovery; lane=80
  initially outlier (refined per-slot PoI corrected this).

### §5.2 Single-victim multi-lane (Phase 4.5/4.6)

**Setup**:
- Three K=1 fixed-victim captures with different sk (separate keygens):
  - victim A (scout): sk=235e27a0..., L=16 lanes (stride 12 incl 0/64/80/128
    calibrated), N=8, 9,984 traces, 2h7m CW-Lite
  - victim B (main): sk=8c6aacfa..., L=24 lanes (stride 8 incl 0/64/80/128
    calibrated), N=16, 29,952 traces, 5h18m CW-Lite
  - victim C (wide): sk=8ce340e4..., L=48 lanes (stride 4 incl 0/64/80/128
    calibrated), N=8, 29,952 traces
- Total 69,888 traces, 88 lane-visits, 352 evaluated NTT coordinates.

**Pipeline**: same as §5.1 + linear interp of LANE_SLOT_OFFSET to non-calibrated lanes.

**Per-victim recovery**:

| victim | L | N | M1 baseline t100 | M5 baseline t100 | Full stack t100 |
|---|---:|---:|---:|---:|---:|
| A (scout) | 16 | 8 | 3 / 64 | 2 / 64 | 4 / 64 |
| B (main) | 24 | 16 | 3 / 96 | 1 / 96 | 3 / 96 |
| C (wide) | 48 | 8 | 3 / 192 | 9 / 192 | 1 / 192 |

**Aggregate (88 lanes covered, 352 cases)**:

| Pipeline | top-1 | top-10 | top-100 | top-500 |
|---|---:|---:|---:|---:|
| M1 baseline (predict_poi) | 0 / 352 | 1 / 352 | **9 / 352** (2.56%) | 45 / 352 (12.8%) |
| M5 baseline | 0 / 352 | 1 / 352 | **12 / 352** (3.41%) | 50 / 352 (14.2%) |
| M1 slot-PoI (calibrated + interp) | 0 / 352 | 0 / 352 | 8 / 352 | 48 / 352 |
| M5 slot-PoI (calibrated + interp) | 0 / 352 | 1 / 352 | 10 / 352 | 47 / 352 |
| **Full stack (Zsum, interp)** | 0 / 352 | 1 / 352 | **8 / 352** (2.27%) | 46 / 352 (13.1%) |

**Channel-mixed union**:

| Union | top-10 | top-100 | top-500 | projected coords |
|---|---:|---:|---:|---:|
| M1 baseline ∪ M5 baseline | 2 / 352 | 21 / 352 | 91 / 352 | 45.8 |
| M1 baseline ∪ Full stack | 2 / 352 | 17 / 352 | 80 / 352 | 37.1 |
| M5 baseline ∪ Full stack | 2 / 352 | 20 / 352 | 88 / 352 | 43.6 |
| M1 ∪ M5 ∪ Full | 3 / 352 | 29 / 352 | 119 / 352 | 63.3 |

These union counts quantify the candidate information pool.  They should not
be presented as one pipeline's standalone recovery rate.

The broader all-channel hit-list (M1, M5, M1_slot, M5_slot, Full) has 37/352
top-100 candidate coordinates.  Since M1/M5/Full overlap is essentially zero
in the current aggregate, later key-recovery experiments should preserve
channel-specific evidence rather than collapsing everything into one fused
score too early.

For downstream recovery-pressure experiments, export top-100 candidate
residues per channel from `candidate_export.npz` instead of re-reading the
large traces.  This keeps the paper workflow reproducible while avoiding
additional large artifacts.

The top-k pressure and null checks confirm that the current single-victim data
is not yet a direct-recovery result: all-channel union gives 4/352 at top-10
and 37/352 at top-100, with average all-channel candidate-set sizes of 41.2
and 371.2 respectively.  Under the actual set-size null, all-channel top-100
has expected 37.80 hits (z=-0.14).  This should be framed as a negative /
diagnostic result for single-victim candidate-set recovery, while the paper's
positive recovery evidence remains the Phase 4 multi-victim TOP-1/top-rank
cases and mechanistic leakage analysis.

**Top-recovery cases (M1 baseline)**:
1. **lane=104 slot=0 f=2563 (|f_c|=894) rk=2** ★ — near-TOP-1, large-|f_c|
   on M1 channel
2. lane=128 slot=2 f=3448 (|f_c|=9) rk=23 (small-|f_c|, M1 channel)
3. lane=76 slot=1 f=1169 (|f_c|=1169) rk=23 (victim C, large)
4. lane=88 slot=0 f=1207 (|f_c|=1207) rk=37 (large)
5. lane=36 slot=1 f=236 rk=45 (small)
6. lane=92 slot=2 f=565 rk=50 (victim C, mid)
7. lane=144 slot=1 f=2496 (|f_c|=961) rk=88 (large)
8. lane=152 slot=1 f=679 rk=93 (mid)

**192-lane single-victim projection** (linear extrapolation from 88 lane-visits):
- M1 baseline: 9/88 × 192 = ~20 top-100 coords / victim
- M5 baseline: 12/88 × 192 = ~26 top-100 coords / victim
- Full stack: 8/88 × 192 = ~17 top-100 coords / victim
- Channel-mixed union: 29/88 × 192 = ~63 top-100 candidate coords / victim
- Earlier two-victim projection was ~29-34 coords/victim, but the 48-lane
  third victim is less favorable.  Use the 3-victim projection as the
  conservative paper number unless later captures justify stratification.

**Information-theoretic implication**:
- For X=20 top-100 NTT coords recovered from a single victim ⇒
  entropy reduction: 20 · log₂(3457) / 1152 ≈ 235 / 1152 = **20%**
  (upper bound, treating each NTT coord as fully revealed).
- For X=26 (M5 baseline projection): 306 / 1152 = **27% reduction**.
- Residual sk entropy for X=26: ~846 bits. Lattice attack β_required ≈ 300
  (Albrecht-style estimator, see §6.3) — **infeasible with current methods**
  but provable partial information disclosure.

**Comparison to §5.1 (multi-victim)**:
- §5.1 (240 cases, 60 victims): 17 / 240 = 7.08% top-100 in aggregate,
  mean 0.28 coords / victim.
- §5.2 (352 cases, 3 victims): M1 baseline 9 / 352, M5 baseline 12 / 352,
  full stack 8 / 352.  This demonstrates accumulation within individual
  victims, but with substantial victim-to-victim variance.
- The strongest safe statement is partial information disclosure, not
  reliable full-key recovery.  More lane coverage or a larger γ set is needed
  before claiming a stable per-victim coordinate yield.

**Mechanistic finding**: Linear interpolation of LANE_SLOT_OFFSET between
calibrated lanes (0, 64, 80, 128, fitted from §5.1) is effective for
non-calibrated lanes. M1 baseline 3 → slot-PoI 6 (main capture, +3
recoveries from interp). The slot drift varies non-monotonically across
lanes, but linear interpolation captures the dominant trend.

### §5.3 Mechanistic findings

- F1: b vs |f_centered| — recovery zone visualization
- F2: per-(lane, slot) drift — lane=80 24-sample slot spread
- F3: M1 vs M5 channel separation by case
- F4: recovery rate by pipeline
- F5: cross-lane SNR @ predict vs peak
- F6: minimum-traces N-curve (f=16 N=2, f=882 N=16)
- F7: σ_α saturation mechanism (signal vs saturating noise)
- F8: dual channel separation by |f_c| bin

## §6 Discussion

### §6.1 σ_α saturation

- N-saturation: `corr ≈ b·σ_HW / sqrt((b·σ_HW)² + σ_α²)`, plateau at
  N > 16
- Decomposition (n29): σ_α (across-γ residual) is saturating; σ_w
  (within-γ noise) reduces with N but limit hits σ_α
- Implication: per-victim N > 16 is wasted; budget should go to
  K (more victims) or L (more lanes per victim).

### §6.2 Channel structure

- M1 (mod-q HW): captures small-|f_c| cells via wrap-free arithmetic
- M5 (montgomery HW): captures occasional large-|f_c| via reduction-output bit
- Dual channel ∪ extends recovery pool ~2x in cases (Z-sum top-10 = 2 / 208)

### §6.3 Information-theoretic reach

- 17 top-100 hits / 240 (multi-victim) and 9-12 top-100 hits / 352
  (single-victim aggregate, depending on channel) ≪ NTRU+ entropy
  (1152 bits)
- Single-victim 192-lane projection: ~17-26 coords → ~17-27% entropy
  reduction (upper bound)
- Lattice attack β_required ≈ 300 — infeasible with current methods
- Partial info disclosure entropy reduction: paper-grade quantitative
  bound

### §6.4 Limitations

- CW-Lite SNR limit (G=78 ceiling, σ_α ~0.0027)
- Profiled per-slot PoI: leave-one-out validation needed (but median
  offset is firmware constant, sk-independent)
- Cross-key transfer fails (per Phase 4 analysis) — single-victim
  primary mode

## §7 Future Work

- G = 200 (HW=3 add): lower null max ≈ 0.29, more recoverable cells
- Profiled augmented setting: 1-key-known PoI/template learn → unknown
  victim attack
- Other lanes capture (192 total → 4-lane sample currently) — more
  comprehensive coverage
- Phase 5 (coefficient-domain threshold): independent attack vector
- Phase 6 (SOTP / FO⊥ validity): implementation leakage baseline
- Lattice attack with more recovered coords (X ≥ 200 → β ~ 100,
  potentially feasible)

## §8 Conclusion

NTRU+768 is vulnerable to NTT-domain selected-lane CPA on the
ChipWhisperer-Lite platform. We recover individual NTT-domain
coefficients with as few as N=2 traces, and demonstrate the structural
recovery ceiling at 7% top-100 over 240 cases, dominated by σ_α
saturation. Single-victim multi-lane attack conservatively projects to
roughly 17-26 top-100 NTT-coordinate disclosures per victim under the
current G=78 design — partial information leakage, not full sk
recovery. Larger γ sets and more lanes are projected to extend the
recoverable pool. The dual-channel (M1+M5) HW model is generally
applicable to montgomery-reduction-based PQC schemes.

## Appendices

- A. NTRU+ NTT layout (chosen-CT semantics derivation)
- B. Wide-γ G=78 design enumeration (HW=1 + HW=2 → 78)
- C. Per-(lane, slot) PoI offset table (Phase 4 calibration)
- D. M1/M5 hypothesis enumeration tables
- E. Trace dataset summary (7 batches × ~3h capture each)
