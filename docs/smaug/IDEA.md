# SMAUG-T SCA Current State

Updated: 2026-05-05.

This note is the single source of truth for the current experiment direction.
Older claims based on target `mu'` labels, same-key lookup tables, PCO pilots,
DTW, FFT, and multi-alpha board-response oracles are retired. They remain useful
as negative evidence in git history, but they are not part of the active attack
path.

## Threat-Model Rules

- `T`, `V`, `X`, and `Z` are diagnostic instrumentation.
- A real attack claim may use traces only. It must not use target `mu'` responses
  or target secret-key dumps for inference.
- Diagnostic wrappers are allowed only to locate leakage and debug models.
- The active goal is a general trace model for SMAUG-T multiplication leakage,
  preferably transferable to `Z` or a natural `D` decapsulation window.

## Current Experimental Spine

### S1: `T` Diagnostic `poly_mul_acc`

Command: `T(component, idx, alpha)`.

Meaning: firmware calls isolated `poly_mul_acc(sk[component], alpha * X^idx)`.
This is not attack-valid, but it is the cleanest leakage-localization gadget.

Rechecked command:

```bash
python3 scripts/smaug/s1_t_final_analysis.py --out-prefix results/smaug/recheck_s1_final --n-shuffles 50
python3 scripts/smaug/s1_t_recover_bayes.py --out-prefix results/smaug/recheck_s1_recover_bayes --conservative --margin-thresh 1.0
```

Result:

- 30 unique keys, mixed N=200/500 traces.
- Pairwise cross-key Welch-t is very large, with first six checked pairs in the
  `736..853` max-|t| range.
- Coefficient-HW CPA has clear signal: real `99.9-pct |rho| = 0.73499`,
  shuffle-null `0.62098 +/- 0.00421`, z about `+27.07`.
- Bonferroni PoIs: `3664` real vs about `122` shuffle-null average.
- Conservative 3-class Bayes LOO: coordinate `0.7363`, support `0.7466`,
  sign `0.8778`.

Interpretation: SMAUG-T multiplication leaks secret-dependent information in a
statistically clear way. The signal is real, but this stage is a diagnostic
wrapper and cannot be stated as an end-to-end KEM attack.

### S2: `Z` Chosen-CT `indcpa_dec`

Command: chosen ciphertext injection followed by `Z`.

Meaning: `Z` calls `indcpa_dec` only. Its `mu'` response is used only for capture
sanity; analysis must use the trace and public chosen-CT labels only.

Rechecked command:

```bash
python3 scripts/smaug/s2_z_analyze.py --out-prefix results/smaug/recheck_s2_z_final --n-shuffles 50
```

Result:

- 36 unique keys, average N about 122 traces.
- Cross-key trace leakage exists, but tested single-coordinate HW models remain
  at shuffle-null level.
- Candidate models tested in the current script:
  - raw ternary int16 HW,
  - legacy shifted-sk HW,
  - high-byte shifted-sk HW,
  - four-term Toom/evaluation-sum HW.

Interpretation: `Z` is the closest current proxy for a natural decapsulation
phase. It confirms that multiplication-related leakage survives outside the
clean `T` gadget, but current simple per-coordinate models do not recover the
key.

### S2.5: `Z` Multivariate Trace-Only Profiling

Command:

```bash
python3 scripts/smaug/s2_z_multivariate.py --sweep --n-perm 100 --out-prefix results/smaug/s2_z_multivariate
```

Meaning: train a profiling model on known-key `Z` captures and predict a
held-out key from trace features only. For each held-out key, high-SNR trace
blocks are selected from training traces only, a multi-output ridge model
predicts 256 ternary coefficients, and the SMAUG-T sparse prior (`HS=70`) is
applied by selecting the largest absolute coefficient scores.

Result:

- 36 unique `Z` keys, same single chosen-CT design `c1 = 4 * X^0` component 0.
- Fixed smoke config `block=16, features=64, ridge=10` was random-level:
  coordinate `0.5653`, support `0.6039`, sign `0.4826`.
- Exploratory grid best was `block=64, features=128, ridge=1`:
  coordinate `0.5734`, support `0.6096`, sign `0.5335`.
- Permutation null for that best config: coordinate `0.5767 +/- 0.0057`,
  support `0.6136 +/- 0.0049`, sign `0.5328 +/- 0.0439`.
- Random sparse baseline: coordinate `0.5652`, support `0.6026`, sign `0.4987`.
- All-zero coordinate baseline remains `0.7266`.

Interpretation: this first-priority S2 profiling attempt is negative. The
single-design `Z` dataset has strong key-dependent trace variation, but a
general multivariate profiler does not map it to held-out coefficient recovery
above permutation null. Do not tune this branch further on the same dataset.
The next S2 experiment must add multiple public chosen-CT designs per key so
the model sees controlled variation in the multiplication input, not just
cross-key trace identity.

### S2.6: Multi-Design `Z` Matrix Pilot

Capture commands:

```bash
python3 scripts/smaug/s2_z_capture_matrix.py --num-keys 2 -n 5 --coefs 0,64 --tag s2_z_matrix_smoke
python3 scripts/smaug/s2_z_capture_matrix.py --num-keys 12 -n 20 --coefs 0,64,128,192 --tag s2_z_matrix_pilot_d4n20
```

Analysis commands:

```bash
python3 scripts/smaug/s2_z_matrix_analyze.py --inputs traces/s2_z_matrix_pilot_d4n20_k*.npz --block 32 --n-features 128 --ridge 10 --n-perm 100 --out-prefix results/smaug/s2_z_matrix_pilot_d4n20
python3 scripts/smaug/s2_z_matrix_analyze.py --inputs traces/s2_z_matrix_pilot_d4n20_k*.npz --block 8 --n-features 256 --ridge 1 --n-perm 100 --out-prefix results/smaug/s2_z_matrix_pilot_d4n20_best
```

Implementation fix before capture: `setup_session(..., fresh_key=False,
pk_fp16=...)` is now used by S2/S3 capture scripts so ciphertext injection does
not call `F` again after `X` dumps the resident key. Existing S2 captures were
checked offline: stored `sk_pke_hex` predicts each file's `first_mu_resp_hex`
for all 36/36 files, so the old dataset is internally consistent.

Result:

- Hardware smoke passed: 2 keys, 2 designs, 5 traces/design, all round-trips and
  ack determinism checks passed.
- Pilot captured 12 keys, 4 public monomial designs (`j=0,64,128,192`),
  20 traces/design. All 12 files saved with shape `(4, 20, 24400)`.
- Default matrix profiler (`block=32, features=128, ridge=10`):
  coordinate `0.5827`, support `0.6224`, sign `0.5270`; permutation null
  coordinate `0.6063 +/- 0.0198`, support `0.6367 +/- 0.0168`.
- Best real-only sweep candidate (`block=8, features=256, ridge=1`):
  coordinate `0.5863`, support `0.6263`, sign `0.5422`; permutation null
  coordinate `0.6086 +/- 0.0180`, support `0.6366 +/- 0.0140`.

Interpretation: multi-design capture is now protocol-correct and reproducible,
but this small pilot still does not beat permutation null. The likely issue is
sample/key count, not command semantics: S=12 is too small for high-dimensional
held-out profiling, and four monomial shifts are too sparse to identify the
multiplication leakage map. Next capture should either scale to many more keys
or switch to a lower-dimensional label target such as design-conditioned
byte/chunk/evaluation states before returning to full 256-coordinate recovery.

### S2.7: Low-Dimensional Matrix Labels

Command:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_pilot_d4n20_k*.npz --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_pilot_d4n20
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_pilot_d4n20_k*.npz --label-kinds support16 --block 32 --n-features 32 --ridge 100 --feature-mode snr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_support16_confirm
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_pilot_d4n20_k*.npz --label-kinds sum16 --block 16 --n-features 64 --ridge 100 --feature-mode snr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_sum16_confirm
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_pilot_d4n20_k*.npz --label-kinds support16 sum16 eval64_support eval64_sum --diff-base 0 --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_diff_pilot_d4n20
```

Meaning: instead of predicting all 256 ternary coefficients, each `(key,
design)` observation predicts lower-dimensional shifted-secret labels:
`support4`, `sum4`, packed `byte_hw`, `eval64_support`, `eval64_sum`,
`support16`, and `sum16`. The target key is always held out as a whole.
Feature selection is fold-local and uses either trace SNR or training-label
correlation. A design-difference mode (`X[d] - X[0]`, `Y[d] - Y[0]`) checks
whether any signal survives same-key common-mode removal.

Result:

- Most low-dimensional labels are null-level or worse.
- Initial 100-permutation sweep had two weak exact-accuracy bumps:
  `support16` exact `0.2526` vs null `0.1993 +/- 0.0192`, z `+2.77`;
  `sum16` exact `0.2057` vs null `0.1815 +/- 0.0105`, z `+2.31`.
- 1000-permutation confirmation:
  `support16` exact `0.2526` vs null `0.2010 +/- 0.0190`, z `+2.72`,
  but rounded-MAE z only `+1.09` and correlation z `-0.75`.
  `sum16` exact `0.2057` vs null `0.1813 +/- 0.0109`, z `+2.23`,
  but rounded-MAE z `-0.63` and correlation z `-2.43`.
- Design-difference check did not confirm a robust signal:
  `support16` exact z `+0.99`, rounded-MAE z `+0.74`, corr z `+0.47`;
  `sum16` and `eval64_sum` were worse than null. `eval64_support` difference is
  trivial for the current 64-spaced designs, giving exact `1.0` for both real
  and null.

Interpretation: the lower-dimensional checkpoint is not yet broken. The
`support16` exact-accuracy bump is worth remembering, but it is not a reliable
attack signal because it does not improve correlation, barely improves MAE, and
mostly disappears under design differencing. The current 12-key / 4-design
pilot is too small and too structurally regular to support a claim.

### S2.8: Random Multi-Term `Z` Matrix Pilot

Motivation: Kyber chosen-ciphertext SCA work shows that carefully selected
ciphertexts can magnify leakage, and multiplication-focused attacks often rely
on repeated secret-dependent multiplication patterns. To avoid the trivial
64-spaced monomial structure from S2.6/S2.7, the matrix capture script now
supports random public multi-term `c1` designs.

Capture commands:

```bash
python3 scripts/smaug/s2_z_capture_matrix.py --num-keys 1 -n 3 --design-mode random-multiterm --num-designs 2 --terms 3 --design-seed 20260505 --tag s2_z_matrix_randmt_smoke
python3 scripts/smaug/s2_z_capture_matrix.py --num-keys 12 -n 10 --design-mode random-multiterm --num-designs 8 --terms 4 --design-seed 20260505 --tag s2_z_matrix_randmt_d8n10
```

Analysis commands:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_d8n10_k*.npz --label-kinds prod16_sum prod16_abs prod16_hw prod64_sum prod64_hw --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_d8n10
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_d8n10_k*.npz --label-kinds prod16_abs prod16_hw prod64_hw --diff-base 0 --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_d8n10_diff
python3 scripts/smaug/s2_z_matrix_analyze.py --inputs traces/s2_z_matrix_randmt_d8n10_k*.npz --block 8 --n-features 256 --ridge 1 --n-perm 100 --out-prefix results/smaug/s2_z_matrix_randmt_d8n10
```

Result:

- Hardware smoke passed. The 12-key pilot captured 8 random 4-term public
  designs/key and 10 traces/design. All files saved with shape `(8, 10, 24400)`.
- Full sparse-key profiler still failed: coordinate `0.5781`, support `0.6100`,
  sign `0.5827`; permutation null coordinate `0.6036 +/- 0.0179`, support
  `0.6341 +/- 0.0156`.
- Product-label profiling did not produce a robust signal. Some exact metrics
  were above null, but exact rates were tiny or unsupported by MAE/correlation:
  `prod16_abs` exact `0.0026` vs null `0.0006 +/- 0.0007`, but rounded-MAE z
  only `+0.79` and corr z `-0.75`; `prod16_hw` exact z `+1.24`, rounded-MAE z
  `-2.34`, corr z `-3.09`.
- Design-difference product labels were also negative: `prod16_hw` exact z
  `+1.65`, rounded-MAE z `-1.35`, corr z `-1.85`; `prod64_hw` exact z `+0.36`,
  rounded-MAE z `-3.40`, corr z `-3.42`.

Interpretation: random multi-term chosen ciphertexts are protocol-correct and
reproducible, but the current `Z` full-window features still do not isolate
secret-dependent product-state leakage. The blocker now looks less like
ciphertext design syntax and more like window/model mismatch: public-input and
key/session variation dominate the full `Z` trace representation.

### S2.9: `Z` Window-Local Product Labels

Window scan:

```bash
python3 scripts/smaug/s2_z_window_scan.py --window 2048 --smooth 64 --top-k 8 --out-prefix results/smaug/s2_z_window_scan_w2048
```

Follow-up analyses:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_d8n10_k*.npz --label-kinds prod16_abs prod16_hw prod64_hw --sample-range 2438:4486 --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_win_2438_4486
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_d8n10_k*.npz --label-kinds prod16_abs prod16_hw prod64_hw --sample-range 10682:12730 --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_win_10682_12730
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_d8n10_k*.npz --label-kinds prod16_abs prod16_hw prod64_hw --sample-range 17731:19779 --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_win_17731_19779
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_d8n10_k*.npz --label-kinds prod64_hw --sample-range 17731:19779 --block 16 --n-features 32 --ridge 100 --feature-mode corr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_randmt_win_17731_19779_prod64hw_confirm
```

Meaning: compare diagnostic `T`, diagnostic `V`, and proxy `Z` key-dependent
SNR envelopes, then analyze only candidate `Z` windows. The analyzer now
supports `--sample-range lo:hi`.

Result:

- Top `Z` SNR window: `10682:12730`.
- Best `V`-template match in `Z`: `2438:4486`, correlation `0.4656`.
- Best `T`-template match in `Z`: `2274:4322`, correlation `0.3278`.
- The strongest weak product-label result was not in the top template match but
  in `17731:19779`. Confirmed with 1000 permutations:
  `prod64_hw`, `block=16`, `features=32`, `ridge=100`, `feature_mode=corr`
  gave exact `0.1930` vs null `0.1822 +/- 0.0043`, z `+2.54`;
  rounded-MAE `1.6457` vs null `1.6811 +/- 0.0195`, z `+1.82`;
  correlation `0.0405` vs null `0.0220 +/- 0.0208`, z `+0.89`.
- Four 512-sample subwindows of `17731:19779` were checked sequentially with
  200 permutations after parallel jobs stalled. No subwindow alone reproduced
  the 2048-sample result:
  `17731:18243` exact z `+0.44`, MAE z `+0.09`;
  `18243:18755` exact z `+1.01`, MAE z `+0.60`;
  `18755:19267` and `19267:19779` were worse than null.

Interpretation: this is the first window-local result where exact and MAE both
move in the right direction under a 1000-permutation null, but the signal is
still weak and correlation remains below a convincing threshold. The effect is
spread across a broad 2048-sample region rather than concentrated in one
512-sample chunk. This is a candidate window for the next capture, not an
attack success.

### S2.10: Focused ADC-Offset Capture Around `17731:19779`

Capture:

```bash
python3 scripts/smaug/s2_z_capture_matrix.py --num-keys 1 -n 3 --samples 2048 --adc-offset 17731 --design-mode random-multiterm --num-designs 2 --terms 4 --design-seed 20260505 --tag s2_z_matrix_randmt_win17731_smoke
python3 scripts/smaug/s2_z_capture_matrix.py --num-keys 12 -n 20 --samples 2048 --adc-offset 17731 --design-mode random-multiterm --num-designs 8 --terms 4 --design-seed 20260505 --tag s2_z_matrix_randmt_win17731_d8n20
```

Analysis:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds prod64_hw --block 16 --n-features 32 --ridge 100 --feature-mode corr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_prod64hw
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds prod64_hw prod16_hw prod16_abs --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_sweep
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds prod16_hw --block 16 --n-features 64 --ridge 100 --feature-mode snr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_prod16hw_confirm
```

Result:

- Hardware smoke passed. Focused pilot captured 12 keys, 8 random 4-term
  designs/key, 20 traces/design, all shape `(8, 20, 2048)`.
- The previous `prod64_hw` candidate did not strengthen:
  exact `0.1769` vs null `0.1721 +/- 0.0056`, z `+0.86`;
  rounded-MAE `1.7030` vs null `1.7883 +/- 0.0650`, z `+1.31`;
  corr `0.0081` vs null `0.0150 +/- 0.0199`, z `-0.35`.
- A broader sweep found `prod16_hw` as the best focused-window low-dimensional
  target. 1000-permutation confirmation:
  exact `0.1100` vs null `0.0939 +/- 0.0067`, z `+2.39`;
  rounded-MAE `3.5658` vs null `3.6789 +/- 0.0887`, z `+1.28`;
  corr `0.0212` vs null `-0.0010 +/- 0.0320`, z `+0.69`.

Interpretation: focused window capture is protocol-correct and more efficient,
but it did not turn the S2.9 weak signal into a robust attack signal. The
`prod16_hw` result is another weak low-dimensional candidate, not a recovery
path yet. More traces alone are unlikely to solve the model mismatch unless the
label/window alignment is improved.

### S2.11: Disassembly-Derived Toom/Karatsuba State Labels

Implementation update:

- `scripts/smaug/s2_z_lowdim_analyze.py` now has Toom-Cook labels derived from
  `smaug1.a` disassembly: a 256-coefficient input is split into four 64-coef
  chunks, seven wrapped int16 evaluation arrays are formed, and per-evaluation
  Karatsuba operand/convolution HW labels are tested.
- The model keeps the firmware shift convention: public `c1` coefficients are
  shifted left by 8 before `vec_vec_mult_add`, while `sk` is unshifted.
- Labels are trace-only at attack time: target-key labels are used only for
  held-out evaluation, not for training or inference.

Analysis:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds qprod16_hibyte --block 16 --n-features 64 --ridge 100 --feature-mode snr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_qprod16_hibyte_confirm
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom7_mul16_hw toom7_mul4_hw toom7_conv16_hw --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom7_sweep
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom0_mul4_hw toom1_mul4_hw toom2_mul4_hw toom3_mul4_hw toom4_mul4_hw toom5_mul4_hw toom6_mul4_hw --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom_points_mul4_sweep
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom0_conv16_hw toom1_conv16_hw toom2_conv16_hw toom3_conv16_hw toom4_conv16_hw toom5_conv16_hw toom6_conv16_hw --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom_points_conv16_sweep
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom2_conv16_hw --block 8 --n-features 128 --ridge 10 --feature-mode corr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom2_conv16_confirm
```

Result:

- q-domain product labels did not improve the S2.10 product candidate:
  `qprod16_hibyte` exact `0.1100` vs null `0.0939 +/- 0.0067`, z `+2.39`;
  rounded-MAE z `+1.28`; corr z `+0.69`.
- Whole-7-point Toom labels were not robust. `toom7_mul4_hw` had exact z
  `+2.78`, but rounded-MAE and correlation were below null.
- Per-point operand labels were weak. The best directional candidates were
  `toom4_mul4_hw` and `toom6_mul4_hw`, both with all metrics moving positive
  but only around z `+0.4..+1.7`.
- Per-point convolution labels produced the strongest S2 trace-only signal so
  far. 1000-permutation confirmation for `toom2_conv16_hw`:
  exact `0.0729` vs null `0.0649 +/- 0.0087`, z `+0.92`;
  rounded-MAE `6.5755` vs null `7.9657 +/- 0.4265`, z `+3.26`;
  corr `0.4932` vs null `0.3180 +/- 0.0631`, z `+2.78`.
- Time-localization did not find a single strong 512- or 1024-sample subwindow.
  The signal needs many weak features across the 2048-sample focused window;
  reducing the model to 32 selected features drops the confirmation to
  rounded-MAE z `+0.21` and corr z `+0.33`.
- Recovery-pressure check separates label identifiability from trace prediction
  accuracy. With trace-predicted `toom2_conv16_hw` labels, the real key beats
  random SMAUG1-sparse candidates only weakly: normalized mean true percentile
  `0.6296`, `0/12` keys above the 95th percentile among 5000 random candidates.
  With oracle exact labels, the same label family is discriminative:
  `12/12` keys above the 99th percentile and mean best-random margin `0.3656`.
- Design differencing (`diff-base 0`) does not fix trace prediction. It gives
  exact z `+2.63`, but rounded-MAE z only `+0.87` and corr z `+0.01`.

Interpretation: this is not key recovery yet, but it is the first focused `Z`
result where an implementation-state label, derived from actual Toom/Karatsuba
control flow, survives a 1000-permutation null on both magnitude error and
correlation. The label family is capable of constraining sparse secrets in
principle, but current trace predictions are too noisy to rank the true secret
reliably. The next active branch should improve trace-to-label accuracy, most
likely by increasing key/design diversity or using a more direct trigger/window
for the relevant Karatsuba state.

### S2.12: Scored Public-Design Capture

Motivation: S2.11 showed that exact `toom2_conv16_hw` labels can distinguish
sparse secrets, but trace-predicted labels are too noisy. Before scaling capture
blindly, we tested whether public designs can be selected to improve the
implementation-state label.

Implementation:

- `scripts/smaug/s2_z_lowdim_analyze.py` now supports `--max-traces`,
  `--max-designs`, and `--design-offset` for ablations.
- `scripts/smaug/s2_z_design_score.py` scores random public 4-term `c1` designs on
  synthetic SMAUG1-sparse secrets only. It maximizes `toom2_conv16_hw` label
  variation/effective rank without using target keys.
- `scripts/smaug/s2_z_capture_matrix.py` now accepts `--design-file`, using the
  scorer's JSON `selected` design set directly.
- `scripts/smaug/s2_z_label_pressure.py` compares trace-predicted or oracle labels
  against random sparse secret candidates.

Ablation on the original random-multiterm focused dataset:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom2_conv16_hw --max-traces 5 --block 8 --n-features 128 --ridge 10 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom2_conv16_n5_confirm
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom2_conv16_hw --max-traces 10 --block 8 --n-features 128 --ridge 10 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom2_conv16_n10_confirm
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom2_conv16_hw --max-designs 4 --block 8 --n-features 128 --ridge 10 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom2_conv16_d4_confirm
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom2_conv16_hw --design-offset 4 --max-designs 4 --block 8 --n-features 128 --ridge 10 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom2_conv16_d4b_confirm
```

Result:

- `N=5` is negative: exact z `+0.00`, rounded-MAE z `-2.12`, corr z `-1.27`.
- `N=10` is weak: exact z `+0.39`, rounded-MAE z `+0.10`, corr z `+0.66`.
- The first four original designs are better than the full eight:
  exact z `+2.22`, rounded-MAE z `+1.31`, corr z `+1.12`.
- The last four original designs are not useful:
  exact z `-2.44`, rounded-MAE z `+0.30`, corr z `+0.64`.

Interpretation: trace averaging matters, and public design choice matters. More
designs are not automatically better; some designs add label variation that is
not visible in the measured power trace.

Scored-design generation and capture:

```bash
python3 scripts/smaug/s2_z_design_score.py --num-secrets 512 --num-candidates 512 --top-k 12 --select-k 8 --out-json results/smaug/s2_z_scored_designs_toom2_512x512.json --out-txt results/smaug/s2_z_design_score_toom2_512x512.txt
python3 scripts/smaug/s2_z_capture_matrix.py --num-keys 1 -n 3 --samples 2048 --adc-offset 17731 --design-file results/smaug/s2_z_scored_designs_toom2_512x512.json --tag s2_z_matrix_scored_toom2_smoke
python3 scripts/smaug/s2_z_capture_matrix.py --num-keys 12 -n 20 --samples 2048 --adc-offset 17731 --design-file results/smaug/s2_z_scored_designs_toom2_512x512.json --tag s2_z_matrix_scored_toom2_d8n20
```

Capture result:

- Smoke passed: 1 key, 8 scored designs, 3 traces/design, all round-trips and
  ack determinism checks passed.
- Main capture passed: 12 keys, 8 scored designs, 20 traces/design. Every design
  for every key captured `20/20`; all files saved with shape `(8, 20, 2048)`.

Analysis:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_scored_toom2_d8n20_k*.npz --label-kinds toom2_conv16_hw --block 8 --n-features 128 --ridge 10 --feature-mode corr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_scored_toom2_d8n20_toom2_conv16_confirm
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_scored_toom2_d8n20_k*.npz --label-kinds toom0_conv16_hw toom1_conv16_hw toom2_conv16_hw toom3_conv16_hw toom4_conv16_hw toom5_conv16_hw toom6_conv16_hw toom0_mul4_hw toom1_mul4_hw toom2_mul4_hw toom3_mul4_hw toom4_mul4_hw toom5_mul4_hw toom6_mul4_hw --sweep --n-perm 100 --out-prefix results/smaug/s2_z_lowdim_scored_toom2_d8n20_toom_points_sweep
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_scored_toom2_d8n20_k*.npz --label-kinds toom6_conv16_hw --block 8 --n-features 32 --ridge 1 --feature-mode corr --n-perm 1000 --out-prefix results/smaug/s2_z_lowdim_scored_toom2_d8n20_toom6_conv16_confirm
python3 scripts/smaug/s2_z_label_pressure.py --inputs traces/s2_z_matrix_scored_toom2_d8n20_k*.npz --label-kind toom6_conv16_hw --block 8 --n-features 32 --ridge 1 --feature-mode corr --n-candidates 5000 --score-mode z --prediction-mode trace --out results/smaug/s2_z_label_pressure_scored_toom6_trace_z_n5000.txt
```

Result:

- Scored `toom2_conv16_hw` is negative:
  exact `0.0495` vs null `0.0569 +/- 0.0083`, z `-0.89`;
  rounded-MAE z `-0.66`; corr z `-1.74`.
- Per-point sweep found only a weak alternate `toom6_conv16_hw` exact bump.
  1000-permutation confirmation:
  exact `0.1797` vs null `0.1297 +/- 0.0170`, z `+2.94`;
  rounded-MAE z `+0.77`; corr z `+0.87`.
- `toom6_conv16_hw` does not create recovery pressure:
  normalized mean true percentile `0.5330`; only `1/12` keys above the 95th
  percentile among 5000 random sparse candidates.

Interpretation: this scored-design branch is negative for attack progress.
Optimizing public designs for synthetic label variation/effective rank did not
optimize measured leakage. It likely increased hard-to-leak high-level label
variation rather than the specific low-level transitions visible in `Z`. Do not
scale this scored design set. The next design step should be leakage-aware:
select designs using a small profiling capture and validation split, or move
to a diagnostic subtrigger that isolates the relevant Karatsuba call before
returning to natural `Z`.

### S2.13: Nested Leakage-Aware Design Selection

Motivation: S2.12 showed that synthetic public-label variation is not a good
proxy for measured leakage. The next trace-only check is whether designs can be
selected by actual trace-to-label transfer on profiling keys only. The target
key must stay held out from both design selection and model fitting.

Implementation:

- `scripts/smaug/s2_z_leakage_select.py` performs nested selection. For each outer
  held-out key, all candidate design subsets of size `K` are scored by
  leave-one-key validation on the remaining profiling keys. The selected subset
  is then used to train on all profiling keys and evaluate the held-out key.
- The permutation null permutes labels before both nested design selection and
  fitting, so selection overfit is included in the null distribution.

Analysis:

```bash
python3 scripts/smaug/s2_z_leakage_select.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kind toom2_conv16_hw --select-k 4 --n-perm 50 --out results/smaug/s2_z_leakage_select_randmt_toom2_d4_p50.txt
python3 scripts/smaug/s2_z_leakage_select.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kind toom2_conv16_hw --select-k 4 --selection-metric rounded_mae --n-perm 30 --out results/smaug/s2_z_leakage_select_randmt_toom2_d4_rmae_p30.txt
python3 scripts/smaug/s2_z_leakage_select.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kind toom2_conv16_hw --select-k 3 --selection-metric corr --n-perm 30 --out results/smaug/s2_z_leakage_select_randmt_toom2_d3_corr_p30.txt
python3 scripts/smaug/s2_z_leakage_select.py --inputs traces/s2_z_matrix_scored_toom2_d8n20_k*.npz --label-kind toom2_conv16_hw --select-k 4 --selection-metric corr --n-perm 30 --out results/smaug/s2_z_leakage_select_scored_toom2_d4_corr_p30.txt
python3 scripts/smaug/s2_z_leakage_select.py --inputs traces/s2_z_matrix_scored_toom2_d8n20_k*.npz --label-kind toom6_conv16_hw --block 8 --n-features 32 --ridge 1 --feature-mode corr --select-k 4 --selection-metric exact --n-perm 30 --out results/smaug/s2_z_leakage_select_scored_toom6_d4_exact_p30.txt
```

Result:

- On the original random focused dataset, `toom2_conv16_hw` with `select_k=4`
  improved absolute metrics over the full 8-design run:
  exact `0.0807`, rounded-MAE `5.5599`, corr `0.6324`.
  However the nested permutation null also rose:
  exact `0.0675 +/- 0.0105`, rounded-MAE `5.8263 +/- 0.4407`,
  corr `0.6013 +/- 0.0645`. The resulting z-scores were only
  exact `+1.27`, rounded-MAE `+0.60`, corr `+0.48`.
- Selected subsets were stable but not equal to the earlier first-four
  ablation: designs `d0` and `d2` were selected for all 12 held-out folds,
  `d7` for 11/12, and `d1`/`d6` split most of the fourth slot.
- Changing the selection objective did not help. `select_k=4` with
  rounded-MAE selection gave z-scores around `+0.4`, and `select_k=3` with
  corr selection made both real and null correlations very high
  (`0.7064` real vs `0.7164 +/- 0.0476` null).
- On the scored-design dataset, nested selection stayed negative for
  `toom2_conv16_hw`: exact z `-1.55`, rounded-MAE z `-1.40`, corr z `-1.05`.
- The weak scored-dataset `toom6_conv16_hw` exact bump did not become useful
  under nested selection: exact z `+0.80`, rounded-MAE z `+0.45`,
  corr z `-0.39`.

Interpretation: leakage-aware selection on the current 12-key / 8-design
captures improves real metrics in absolute terms, but the improvement is mostly
explained by selection overfit under a properly nested permutation null. This
does not support scaling the same design-selection loop. The next coherent move
is not more subset search on these captures; it is a better leakage-localization
instrument, most likely a diagnostic trigger around the relevant Toom/Karatsuba
state, followed by a return to natural `Z` once the state/window alignment is
known.

### S3: `V` Diagnostic `vec_vec_mult_add`

Command: chosen ciphertext injection followed by `V`.

Meaning: firmware reconstructs the `indcpa_dec` multiplication inputs and
triggers only around `vec_vec_mult_add`. This is diagnostic instrumentation, not
attack-valid.

Rechecked command:

```bash
python3 scripts/smaug/s3_v_analyze_2sk.py --sk-a traces/s3_v_skA_a4_n200.npz --sk-b traces/s3_v_skB_a4_n200.npz --out-prefix results/smaug/recheck_s3_v_2sk
```

Result:

- `max|t| = 36.43` at sample `23129`.
- `849` samples above `|t| > 4.5`, `251` above `|t| > 8.0`.
- This is far weaker than S1 and S2.

Interpretation: current `V` data does not explain or recover the S2 leakage. Do
not claim `V` solves the attack-valid problem. Use it only as a sanity check
unless a future capture clearly changes this result.

### S3.1: `V` Low-Dimensional Toom Label Check

Motivation: S2.11/S2.13 left two possibilities: either the Toom/Karatsuba label
family is poorly aligned to the measured state, or the natural/full `Z` window
is too mixed. Existing `V` traces isolate `vec_vec_mult_add`, so they are a
small diagnostic check before adding more firmware instrumentation.

Implementation:

- `scripts/smaug/s3_v_lowdim_analyze.py` wraps existing single-design `V` captures as
  a one-design matrix dataset and reuses the same low-dimensional evaluator and
  permutation null as S2.
- Current data has only 5 keys and one public monomial design
  (`c1 = 4 * X^0`), so this is a localization scout, not an attack result.

Analysis:

```bash
python3 scripts/smaug/s3_v_lowdim_analyze.py --inputs traces/s3_v*_a4_n200.npz --label-kinds toom0_conv16_hw toom1_conv16_hw toom2_conv16_hw toom3_conv16_hw toom4_conv16_hw toom5_conv16_hw toom6_conv16_hw --sweep --n-perm 50 --out-prefix results/smaug/s3_v_lowdim_toom_conv_sweep_p50
python3 scripts/smaug/s3_v_lowdim_analyze.py --inputs traces/s3_v*_a4_n200.npz --label-kinds toom2_conv16_hw --block 8 --n-features 128 --ridge 10 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s3_v_lowdim_toom2_fixed_p500
```

Result:

- The real-only sweep produced some high exact z-scores, but rounded-MAE and
  correlation were mostly at or below permutation null. This is consistent with
  overfitting on a 5-key, one-design diagnostic dataset.
- The fixed S2 configuration for `toom2_conv16_hw` was negative:
  exact `0.5000` vs null `0.5288 +/- 0.0271`, rounded-MAE `4.1000` vs null
  `3.1938 +/- 0.4781`, corr `0.9026` vs null `0.9274 +/- 0.0207`.

Interpretation: current `V` traces do not give a cleaner `toom2_conv16_hw`
mapping than `Z`. The next localization step should not be more `V` analysis on
the same monomial data. Instead, isolate the same multi-term public inputs used
by S2 inside `poly_mul_acc` itself.

### S1U: Multi-Term Isolated `poly_mul_acc`

Implementation:

- Firmware command `U` was added as a multi-term sibling of `T`.
- Payload is fixed 34 bytes: component, term count, then up to eight
  `(coef, alpha)` slots. The firmware builds a public `host_b` polynomial,
  triggers only around `poly_mul_acc(sk[component], host_b, out)`, and returns
  the first 32 output bytes for sanity.
- `scripts/smaug/s1_u_capture_matrix.py` captures multiple public designs per key.
  By default it sends `alpha << 8`, matching the `c1 << 8` convention inside
  `Z`/`vec_vec_mult_add`.
- `scripts/smaug/s2_z_lowdim_analyze.py` can now also load `cmd=U`,
  `capture_kind=u_matrix` files, so the same Toom labels and permutation null
  can be applied to isolated multi-term multiplication traces.

First hardware smoke to run after flashing the rebuilt firmware:

```bash
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4 SMAUG_LEVEL=1
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
python3 scripts/smaug/s1_u_capture_matrix.py --num-keys 1 -n 5 --num-designs 2 --terms 4 --tag s1_u_matrix_smoke
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s1_u_matrix_smoke_k*.npz --label-kinds toom2_conv16_hw --block 8 --n-features 64 --ridge 10 --feature-mode corr --n-perm 20 --out-prefix results/smaug/s1_u_lowdim_smoke_toom2
```

Interpretation target:

- If `U` shows a robust Toom label while `V`/`Z` do not, the blocker is window
  placement/replay/natural-trace mixing.
- If `U` is also null, the current Toom label family is probably not the
  dominant low-level leakage model for measured power, even when isolated.

Hardware result:

```bash
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
python3 scripts/smaug/s1_u_capture_matrix.py --num-keys 1 -n 5 --num-designs 2 --terms 4 --tag s1_u_matrix_smoke
python3 scripts/smaug/s1_u_capture_matrix.py --num-keys 6 -n 20 --num-designs 8 --terms 4 --design-seed 20260505 --tag s1_u_matrix_randmt_d8n20_scout
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s1_u_matrix_randmt_d8n20_scout_k*.npz --label-kinds toom2_conv16_hw --block 8 --n-features 128 --ridge 10 --feature-mode corr --n-perm 200 --out-prefix results/smaug/s1_u_lowdim_randmt_d8n20_scout_toom2_fixed_p200
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s1_u_matrix_randmt_d8n20_scout_k*.npz --label-kinds toom0_conv16_hw toom1_conv16_hw toom2_conv16_hw toom3_conv16_hw toom4_conv16_hw toom5_conv16_hw toom6_conv16_hw --sweep --n-perm 50 --out-prefix results/smaug/s1_u_lowdim_randmt_d8n20_scout_toom_conv_sweep_p50
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s1_u_matrix_randmt_d8n20_scout_k*.npz --label-kinds toom3_conv16_hw --block 8 --n-features 128 --ridge 1 --feature-mode snr --n-perm 500 --out-prefix results/smaug/s1_u_lowdim_randmt_d8n20_scout_toom3_confirm_p500
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s1_u_matrix_randmt_d8n20_scout_k*.npz --label-kinds toom3_conv16_hw --sample-range 11000:15000 --block 8 --n-features 128 --ridge 1 --feature-mode snr --n-perm 500 --out-prefix results/smaug/s1_u_lowdim_randmt_d8n20_scout_toom3_win11000_15000_p500
```

- Firmware upload passed with the new `U` command.
- `U` smoke passed: 1 key, 2 random 4-term designs, 5 traces/design, all
  round-trip and ack determinism checks passed.
- Scout capture passed: 6 keys, 8 random 4-term public designs, 20
  traces/design. Every design for every key captured `20/20`, all files saved
  with shape `(8, 20, 24400)`.
- Fixed S2 `toom2_conv16_hw` config on isolated `U` traces was only weakly
  positive:
  exact `0.0755` vs null `0.0640 +/- 0.0138`, z `+0.83`;
  rounded-MAE `5.3542` vs null `5.9092 +/- 0.4002`, z `+1.39`;
  corr `0.6474` vs null `0.5809 +/- 0.0626`, z `+1.06`.
- Per-point convolution sweep found the best scout candidate at
  `toom3_conv16_hw`, not `toom2_conv16_hw`. 500-permutation confirmation:
  exact `0.0964` vs null `0.0620 +/- 0.0104`, z `+3.30`;
  rounded-MAE `9.2474` vs null `9.7269 +/- 0.4094`, z `+1.17`;
  corr `0.2706` vs null `0.2449 +/- 0.0290`, z `+0.89`.
- Localizing `toom3_conv16_hw` to `11000:15000` did not preserve the effect:
  exact z `-0.47`, rounded-MAE z `-0.92`, corr z `+0.41`.

Interpretation: this does not support the statement "Karatsuba does not leak."
It supports a narrower statement: the current high-level Toom convolution label
family is a weak and unstable predictor even in isolated multi-term
`poly_mul_acc`. There is likely multiplication leakage, as S1 monomial `T`
already showed, but the leakage visible on this setup is not cleanly explained
by the current `toom2_conv16_hw` recovery label. The next coherent model step is
to move below whole-convolution grouped labels and test lower-level Karatsuba
operation labels or simpler operand/evaluation HW windows before returning to
natural `Z`.

### S1U.1: Lower-Level Karatsuba Labels

Motivation: whole 64-coefficient Toom convolution labels are too coarse. The
next model step splits each 64-coefficient Toom evaluation into the top-level
Karatsuba operands:

- `kara0`: low 32 coefficients,
- `kara1`: `(low + high)` cross operand,
- `kara2`: high 32 coefficients.

Implementation:

- `scripts/smaug/s2_z_lowdim_analyze.py` now supports lower-level labels:
  `toom{i}_kara{0,1,2}_conv8_hw`, `toom{i}_kara{0,1,2}_mul4_hw`,
  `toom{i}_kara{0,1,2}_op8_hw`, and `toom{i}_sec8_hw`.
- These labels are still diagnostic and high-level relative to actual C
  instruction transitions, but they are closer to `karatsuba_simple` than
  `toom{i}_conv16_hw`.

Analysis on the `U` scout:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s1_u_matrix_randmt_d8n20_scout_k*.npz --label-kinds toom3_sec8_hw toom3_kara0_conv8_hw toom3_kara1_conv8_hw toom3_kara2_conv8_hw toom3_kara0_mul4_hw toom3_kara1_mul4_hw toom3_kara2_mul4_hw toom3_kara0_op8_hw toom3_kara1_op8_hw toom3_kara2_op8_hw --sweep --n-perm 100 --out-prefix results/smaug/s1_u_lowdim_randmt_d8n20_scout_toom3_kara_low_sweep_p100
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s1_u_matrix_randmt_d8n20_scout_k*.npz --label-kinds toom3_kara2_conv8_hw --block 16 --n-features 64 --ridge 1 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s1_u_lowdim_randmt_d8n20_scout_toom3_kara2_conv8_confirm_p500
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom3_kara2_conv8_hw --block 16 --n-features 64 --ridge 1 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s2_z_lowdim_randmt_win17731_d8n20_toom3_kara2_conv8_confirm_p500
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_d8n10_k*.npz --label-kinds toom3_kara2_conv8_hw --block 16 --n-features 64 --ridge 1 --feature-mode corr --n-perm 300 --out-prefix results/smaug/s2_z_lowdim_randmt_d8n10_toom3_kara2_conv8_confirm_p300
```

Result:

- On isolated `U`, the best lower-level candidate was
  `toom3_kara2_conv8_hw`. 500-permutation confirmation:
  exact `0.1615` vs null `0.1187 +/- 0.0174`, z `+2.46`;
  rounded-MAE `3.9479` vs null `4.5996 +/- 0.3102`, z `+2.10`;
  corr `0.2801` vs null `0.1985 +/- 0.0540`, z `+1.51`.
- The same label did not transfer to focused `Z`:
  exact z `-2.12`, rounded-MAE z `-1.50`, corr z `-1.76`.
- It also did not transfer to full-window random-multiterm `Z`:
  exact z `-1.64`, rounded-MAE z `+0.02`, corr z `-0.66`.

Interpretation: this is the strongest evidence so far that the multiplication
core does leak a Karatsuba-related low-level state under isolated triggering,
but that state is not visible in the current `Z` windows. The blocker is now
less "does Karatsuba leak?" and more "where does this Karatsuba subproduct
state land inside natural decapsulation, and can it be isolated without using a
diagnostic wrapper?" The next useful capture is therefore a `Z`/`D` window scan
guided by the `U` timing/features for `toom3_kara2_conv8_hw`, not more public
design scoring.

### S3.2: Multi-Term `V` Matrix

Motivation: earlier `V` data used only a monomial public input, while the
isolated `U` signal appears on random multi-term public designs. Before blaming
natural `Z` timing entirely, we need to ask whether the same multi-term design
set leaks in the `V` replay of `vec_vec_mult_add`.

Implementation:

- `scripts/smaug/s2_z_capture_matrix.py` now supports `--cmd V`, reusing the same
  resident-key, design-matrix, round-trip, and ack-determinism checks as `Z`.
- `scripts/smaug/s2_z_lowdim_analyze.py` can load matrix captures with `cmd=V`.

Capture and analysis:

```bash
python3 scripts/smaug/s2_z_capture_matrix.py --cmd V --num-keys 1 -n 3 --design-mode random-multiterm --num-designs 2 --terms 4 --design-seed 20260505 --tag s3_v_matrix_randmt_smoke
python3 scripts/smaug/s2_z_capture_matrix.py --cmd V --num-keys 6 -n 20 --design-mode random-multiterm --num-designs 8 --terms 4 --design-seed 20260505 --tag s3_v_matrix_randmt_d8n20_scout
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_v_matrix_randmt_d8n20_scout_k*.npz --label-kinds toom3_kara2_conv8_hw --block 16 --n-features 64 --ridge 1 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s3_v_lowdim_matrix_randmt_d8n20_scout_toom3_kara2_conv8_confirm_p500
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_v_matrix_randmt_d8n20_scout_k*.npz --label-kinds toom3_sec8_hw toom3_kara0_conv8_hw toom3_kara1_conv8_hw toom3_kara2_conv8_hw toom3_kara0_mul4_hw toom3_kara1_mul4_hw toom3_kara2_mul4_hw toom3_kara0_op8_hw toom3_kara1_op8_hw toom3_kara2_op8_hw --sweep --n-perm 100 --out-prefix results/smaug/s3_v_lowdim_matrix_randmt_d8n20_scout_toom3_kara_low_sweep_p100
```

Result:

- Smoke passed.
- Scout capture passed: 6 keys, 8 random 4-term designs, 20 traces/design,
  all `20/20`, shape `(8, 20, 24400)`.
- The `U`-derived label did not survive in `V`:
  `toom3_kara2_conv8_hw` fixed confirmation gave exact `0.1615` vs null
  `0.1512 +/- 0.0185`, z `+0.55`; rounded-MAE z `+0.11`; corr z `-0.01`.
- The broader low-level Toom/Karatsuba sweep was also negative. Several labels
  had high absolute correlation, but permutation null correlations were equally
  high or higher; most rounded-MAE and correlation z-scores were negative.

Interpretation: the isolated `U` Karatsuba subproduct signal does not survive
the current `V` replay either. This narrows the blocker: it is not merely the
full `Z` window or FO/rounding code. The visible `U` leakage is likely tied to
the isolated `poly_mul_acc(sk_component, host_b)` call shape, while
`vec_vec_mult_add` replay changes enough context, accumulation, or component
mixing to hide this label. The next useful diagnostic is a component-specific
`V`-like replay that calls exactly the `poly_mul_acc` path used by
`vec_vec_mult_add` for one component at a time, rather than summing the whole
polyvec multiplication.

### S3.3: Component-Specific `W` Replay

Motivation: S3.2 showed that the `U` signal does not transfer to the full
`vec_vec_mult_add` replay. The next split is whether the signal is lost because
`V` sums both polyvec components, or because the actual `vec_vec_mult_add`
`poly_mul_acc` call shape differs from the isolated `U` diagnostic.

Important implementation finding:

- Disassembly of `cryptolab_smaug1_vec_vec_mult_add` shows that the function
  first computes `a >> mod`, calls `poly_mul_acc(a_unshifted, b, tmp)`, then
  shifts the product back left before adding into the output. Therefore the
  actual `V` multiplication state is not `poly_mul_acc(sk, c1 << 8)`.
- `scripts/smaug/s2_z_lowdim_analyze.py` keeps the old `toom*` labels for the `U`
  shifted-public hypothesis and adds `vtoom*` labels for the `V`/`W` internal
  hypothesis: unshifted public operand first, secret operand second.
- `firmware/simpleserial-smaug/simpleserial-smaug.c` adds command `W`:
  payload `component`, replay `load_from_string_sk` + `load_from_string`, then
  trigger only around `poly_mul_acc(c1_component_unshifted, sk_component, out)`.
- `scripts/smaug/s2_z_capture_matrix.py` now supports `--cmd W`. Since `W` returns a
  raw product sanity response rather than `mu'`, capture checks only response
  length and ack determinism, not round-trip `mu'`.

Capture and analysis:

```bash
python3 scripts/smaug/s2_z_capture_matrix.py --cmd W --num-keys 1 -n 3 --design-mode random-multiterm --num-designs 2 --terms 4 --design-seed 20260505 --tag s3_w_component_randmt_smoke
python3 scripts/smaug/s2_z_capture_matrix.py --cmd W --num-keys 6 -n 20 --design-mode random-multiterm --num-designs 8 --terms 4 --design-seed 20260505 --tag s3_w_component_randmt_d8n20_scout
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_w_component_randmt_d8n20_scout_k*.npz --label-kinds toom3_kara2_conv8_hw --block 16 --n-features 64 --ridge 1 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s3_w_lowdim_component_randmt_d8n20_scout_toom3_kara2_conv8_confirm_p500
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_w_component_randmt_d8n20_scout_k*.npz --label-kinds vtoom3_kara2_conv8_hw --block 16 --n-features 64 --ridge 1 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s3_w_lowdim_component_randmt_d8n20_scout_vtoom3_kara2_conv8_confirm_p500
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_w_component_randmt_d8n20_scout_k*.npz --label-kinds vtoom3_sec8_hw vtoom3_kara0_conv8_hw vtoom3_kara1_conv8_hw vtoom3_kara2_conv8_hw vtoom3_kara0_mul4_hw vtoom3_kara1_mul4_hw vtoom3_kara2_mul4_hw vtoom3_kara0_op8_hw vtoom3_kara1_op8_hw vtoom3_kara2_op8_hw --sweep --n-perm 100 --out-prefix results/smaug/s3_w_lowdim_component_randmt_d8n20_scout_vtoom3_kara_low_sweep_p100
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_w_component_randmt_d8n20_scout_k*.npz --label-kinds prod16_sum prod16_abs prod16_hw prod64_sum prod64_hw qprod16_hw qprod16_hibyte qprod64_hw qprod64_hibyte --sweep --n-perm 100 --out-prefix results/smaug/s3_w_lowdim_component_randmt_d8n20_scout_prod_sweep_p100
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_v_matrix_randmt_d8n20_scout_k*.npz --label-kinds vtoom3_kara0_conv8_hw vtoom3_kara1_conv8_hw vtoom3_kara2_conv8_hw vtoom3_kara0_mul4_hw vtoom3_kara1_mul4_hw vtoom3_kara2_mul4_hw --sweep --n-perm 100 --out-prefix results/smaug/s3_v_lowdim_matrix_randmt_d8n20_scout_vtoom3_kara_sweep_p100
```

Result:

- `W` smoke passed.
- `W` scout passed: 6 keys, 8 random 4-term designs, 20 traces/design, all
  `20/20`, shape `(8, 20, 24400)`.
- The old `U`-derived shifted-public label stayed null on `W`:
  `toom3_kara2_conv8_hw` exact z `-0.35`, rounded-MAE z `+0.57`,
  corr z `+0.63`.
- The direct `V`/`W` internal label also stayed null:
  `vtoom3_kara2_conv8_hw` exact z `+0.85`, rounded-MAE z `-0.97`,
  corr z `-0.59`.
- Broader `vtoom3` Karatsuba sweep was negative. The best-looking exact bumps
  (`vtoom3_kara1_conv8_hw` exact z `+1.80`,
  `vtoom3_kara2_op8_hw` exact z `+1.77`) had weak or negative rounded-MAE and
  correlation z-scores.
- Product/final-state labels on `W` were also negative. Several exact z-scores
  rose only because exact match is sparse; rounded-MAE and correlation were
  consistently worse than null.
- Re-analyzing the existing `V` matrix with `vtoom` labels did not rescue S3.2.
  The best candidate, `vtoom3_kara2_mul4_hw`, reached exact z `+1.69`, but
  rounded-MAE z was only `+0.57` and corr z `+0.09`.

Interpretation: this is a useful negative result. The `U` signal is real, but
it is tied to a diagnostic call shape that does not match the multiplication
state used by `vec_vec_mult_add`. The natural `V`/`Z` path multiplies unshifted
public `c1` by the secret, then shifts the product afterward; the strong
shifted-public `U` leakage is therefore not a faithful proxy for the attack
window. Next work should stop scaling `U`-derived Karatsuba labels and instead
either model the shift/add loops inside `vec_vec_mult_add` directly or return
to natural `Z`/`D` window discovery with labels grounded in the actual
disassembled `vec_vec_mult_add` sequence.

### S3.4: Nonzero-`c2` Add/HD Labels

Motivation: S3.3 suggested that the Karatsuba core is not the visible
attack-relevant state in `V`. The next hypothesis was that the shift-back and
`poly_add` loops inside `vec_vec_mult_add` leak a state closer to the final
accumulation. Existing random-multiterm captures had `c2=0`, so add and HD
labels collapsed to the same state as `tmp << 8`. To make the addition state
nontrivial, we captured `V` with constant nonzero `c2`.

Implementation:

- `scripts/smaug/s2_z_lowdim_analyze.py` now supports `vtmp*`, `vshift*`, `vadd*`,
  and `vdelta*` labels:
  - `vtmp*`: approximate `poly_mul_acc(c1_unshifted, sk)` result,
  - `vshift*`: `tmp << LOG_P`,
  - `vadd*`: `(c2 << 11) + (tmp << LOG_P)`,
  - `vdelta*`: Hamming distance between `c2 << 11` and the post-add output.
- `scripts/smaug/s2_z_capture_matrix.py` now supports `--c2-mode constant
  --c2-alpha A`; old captures default to `c2=0` in the analyzer.

Analysis on existing `c2=0` `V`/`W` captures:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_v_matrix_randmt_d8n20_scout_k*.npz --label-kinds vtmp16_hw vtmp64_hw vtmp16_lohw vtmp16_hihw vshift16_hw vshift64_hw vshift16_hihw vadd16_hw vadd64_hw vdelta16_hw vdelta64_hw --sweep --n-perm 100 --out-prefix results/smaug/s3_v_lowdim_matrix_randmt_d8n20_scout_vecadd_sweep_p100
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_w_component_randmt_d8n20_scout_k*.npz --label-kinds vtmp16_hw vtmp64_hw vtmp16_lohw vtmp16_hihw vshift16_hw vshift64_hw vshift16_hihw vadd16_hw vadd64_hw vdelta16_hw vdelta64_hw --sweep --n-perm 100 --out-prefix results/smaug/s3_w_lowdim_component_randmt_d8n20_scout_vecadd_sweep_p100
```

Result on existing captures: negative. Several labels produced exact z-scores
above +2, but rounded-MAE and correlation were consistently worse than the
permutation null. This is expected to some extent because `c2=0` makes `vadd`
and `vdelta` equivalent to shifted product labels.

Nonzero-`c2` capture and analysis:

```bash
python3 scripts/smaug/s2_z_capture_matrix.py --cmd V --c2-mode constant --c2-alpha 8 --num-keys 1 -n 3 --design-mode random-multiterm --num-designs 2 --terms 4 --design-seed 20260505 --tag s3_v_matrix_randmt_c2a8_smoke
python3 scripts/smaug/s2_z_capture_matrix.py --cmd V --c2-mode constant --c2-alpha 8 --num-keys 6 -n 20 --design-mode random-multiterm --num-designs 8 --terms 4 --design-seed 20260505 --tag s3_v_matrix_randmt_c2a8_d8n20_scout
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_v_matrix_randmt_c2a8_d8n20_scout_k*.npz --label-kinds vtmp16_lohw vtmp16_hw vshift16_hw vshift64_hw vadd16_hw vadd64_hw vdelta16_hw vdelta64_hw --sweep --n-perm 100 --out-prefix results/smaug/s3_v_lowdim_matrix_randmt_c2a8_d8n20_scout_vecadd_sweep_p100
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_v_matrix_randmt_c2a8_d8n20_scout_k*.npz --label-kinds vdelta64_hw --block 16 --n-features 128 --ridge 10 --feature-mode corr --n-perm 500 --out-prefix results/smaug/s3_v_lowdim_matrix_randmt_c2a8_d8n20_scout_vdelta64_confirm_p500
```

Result:

- Smoke passed.
- Scout capture passed: 6 keys, 8 designs, 20 traces/design, all `20/20`,
  shape `(8, 20, 24400)`.
- Sweep remained negative on rounded-MAE/correlation. The most notable result
  was `vdelta64_hw`, which had exact z `+3.73` at p100.
- p500 confirmation for `vdelta64_hw`:
  exact `0.0833` vs null `0.0379 +/- 0.0135`, z `+3.36`;
  rounded-MAE `8.5781` vs null `8.5386 +/- 0.2950`, z `-0.13`;
  corr `0.2121` vs null `0.2493 +/- 0.0675`, z `-0.55`.

Interpretation: nonzero `c2` makes the add/HD hypothesis testable, but it still
does not produce useful trace-to-label prediction. The exact bump is not enough:
the model does not improve label magnitude or correlation over null. This
retires the current coarse `vec_vec_mult_add` shift/add label family. The next
step should either move to a more local instruction/window diagnostic for the
repeated shift/add loops or stop adding diagnostic wrappers and return to
natural `Z` window discovery with a more agnostic leakage objective.

### S4.1: Two-Way Residualized Low-Dimensional Reanalysis

Motivation: previous design differencing used a single baseline design. A more
attack-valid cleanup is to remove both key common-mode and public-design effect:
for each leave-one-key-out fold, subtract the per-key design mean and the
train-key design mean, then add the train global mean. The target key's
per-key mean uses only its chosen-ciphertext traces, while public-design means
are computed from profiling keys only.

Implementation:

- `scripts/smaug/s2_z_lowdim_analyze.py` now supports
  `--residualize two-way`.
- The residualization is fold-local. Feature selection, ridge fitting, and the
  permutation null all run after this transform.
- Dataset slicing with `--max-designs`, `--max-traces`, and `--sample-range`
  now preserves `design_c2` metadata.

Commands:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_v_matrix_randmt_c2a8_d8n20_scout_k*.npz --label-kinds vdelta64_hw vshift64_hw --block 16 --n-features 128 --ridge 10 --feature-mode corr --residualize two-way --n-perm 300 --out-prefix results/smaug/s4_resid_v_c2a8_d8n20_vdelta_vshift_p300
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s2_z_matrix_randmt_win17731_d8n20_k*.npz --label-kinds toom2_conv16_hw toom3_kara2_conv8_hw --block 8 --n-features 128 --ridge 10 --feature-mode corr --residualize two-way --n-perm 300 --out-prefix results/smaug/s4_resid_z_win17731_d8n20_toom_candidates_p300
python3 scripts/smaug/s2_z_lowdim_analyze.py --inputs traces/s3_w_component_randmt_d8n20_scout_k*.npz --label-kinds vtoom3_kara1_conv8_hw vtoom3_kara2_conv8_hw vtoom3_kara2_op8_hw --block 16 --n-features 64 --ridge 1 --feature-mode corr --residualize two-way --n-perm 300 --out-prefix results/smaug/s4_resid_w_component_d8n20_vtoom3_candidates_p300
```

Results:

- `V+c2=8`, `vdelta64_hw`: exact z `+0.00`, rounded-MAE z `+0.02`,
  corr z `+0.89`.
- `V+c2=8`, `vshift64_hw`: exact z `-0.68`, rounded-MAE z `+0.62`,
  corr z `-0.03`.
- natural focused `Z`, `toom2_conv16_hw`: exact z `+0.71`,
  rounded-MAE z `+1.46`, corr z `+1.47`.
- natural focused `Z`, `toom3_kara2_conv8_hw`: exact z `-1.45`,
  rounded-MAE z `-1.69`, corr z `-1.93`.
- component-local `W`, `vtoom3_kara2_conv8_hw`: exact z `+3.98`,
  rounded-MAE z `+0.56`, corr z `+1.37`.
- component-local `W`, `vtoom3_kara2_op8_hw`: residual labels collapse to the
  null baseline, so the apparent perfect exact score is not informative.

Interpretation: two-way residualization does not rescue the current label
families. The best natural `Z` result improves slightly but remains well below
the rounded-MAE/correlation gate (`z > +3`). `W` again shows an exact-only bump
without magnitude/correlation support. This strengthens the transfer-failure
story: the blocker is not just key identity or public-design common mode.

## Resolved Negative Paths

The following directions are retired from active work:

- direct `mu'`-label PoI and the old 8-trace claim,
- same-key or target-key lookup using `Z` responses,
- PCO and MV-PC pilots based on FO accept/reject or SHAKE leakage,
- DTW/FFT/alignment attempts on the old chosen-CT data,
- smaug3/smaug5 calibrated board-response oracle claims.

Reason: they either relied on diagnostic responses, classified public data or
command choices, or failed to transfer under trace-only target-key evaluation.

## Next Priority

1. Do not scale `17731:19779` blindly. It is a useful focused-capture window,
   but current product labels remain weak.
2. Do not scale the `U`-derived shifted-public Karatsuba labels. S3.3 shows
   they do not match the actual `vec_vec_mult_add` multiplication state.
3. The next useful model step is the disassembled `vec_vec_mult_add` sequence
   itself: `a >> mod`, component `poly_mul_acc`, product `<< mod`, and
   `poly_add`. The shift/add loops may leak more attack-relevant state than the
   Karatsuba core under unshifted public inputs.
4. S3.4 retired the current coarse shift/add/HD labels. A follow-up in this
   family must be more local than whole-vector grouped HW, e.g. instruction
   window diagnostics around a single repeated shift/add loop.
5. S4.1 shows two-way residualization does not rescue the current labels. Do
   not keep tuning residualized ridge unless a more local label/window is found.
6. Only after one of these actual-`V`/natural-`Z` labels beats the permutation
   null on rounded-MAE and correlation, attempt a larger natural `D` trace
   window.

Do not add new attack branches until one of these checkpoints produces a
trace-only improvement over the current S2 null-level coordinate recovery.

## Follow-Up Branches

- Branch R (`mu'` round/pack leakage) is documented in
  `docs/r_mu_roundpack.md`. It is now the primary positive path: diagnostic `R`
  round/pack traces predict latent `mu'` byte/block Hamming weight with
  held-out keys and permutation nulls, while natural `Z` transfer remains an
  open/negative checkpoint.
- Branch C2 (`c2` paired threshold tomography) is documented in
  `docs/c2_threshold_pairs.md`. It tests same-`c1`, different-`c2` pairs to
  cancel multiplication common mode. The naive `0 -> 1` pair is label-degenerate;
  the corrected `15 -> 16` pair creates secret-dependent `mu'` flips, but
  natural `Z` remains null-level even after paired window scanning. Diagnostic
  `R` shows a weak signed `mu_delta_byte_hw` result, and diagnostic `Q` shows
  only a sub-gate localized hint. The branch currently strengthens the
  diagnostic-to-natural transfer-failure story rather than producing an
  attack-valid oracle.
- Branch Y (FO downstream replay) is documented in
  `docs/fo_downstream.md`. It moves the trigger after `indcpa_dec` and tests
  whether `mu' -> G(mu', H(pk)) -> seed' -> re-encryption -> verify/cmov`
  amplifies latent `mu'` into a stronger trace label. A 6-key scout completed
  cleanly with `24 designs x 10 traces`. Full-window analysis found the best
  label at `fo_kr1_byte_hw`, but only exact z `+1.52`, rounded-MAE z `+1.52`,
  corr z `+0.55`. A localized window scan also stayed below gate, with the best
  confirmed corr z around `+1.21`. This branch is diagnostic only and currently
  does not produce an attack-valid oracle.
- S5 trace-count scaling is documented in `docs/s5_trace_scaling.md`. It asks
  whether the failed transfer branches are merely trace-limited. The diagnostic
  `R` positive control passes even at `N=2` (`mu_byte_hw`, rMAE/corr z
  `+5.61/+4.29`), proving the audit can see real leakage. Natural `Z` remains
  sub-gate through `N=10`, `Q` bridge remains negative through `N=20`, and `Y`
  FO downstream shows a non-monotonic bump that does not persist. The only
  coherent scaling hint is diagnostic `C2-Q` `mu_delta_byte_hw` at `N=10`
  (rMAE/corr z `+2.30/+2.21`), but the same pair in natural `Z` is null-level.
  Current evidence therefore favors label/window/model mismatch over a simple
  lack of trace averaging.
- S5.6 followed up the diagnostic `C2-Q` hint with an independent `N=20`
  recapture (`6 keys x 24 designs x 20 traces`, all `20/20`). The old best
  window `9216:13192` did not reproduce (`N=20` rMAE/corr z `+0.94/+0.81`).
  A fresh window scan found `7680:11656` as the best new diagnostic window; a
  p1000 confirmation gave rMAE/corr z `+2.62/+1.76`. This is the strongest
  remaining localization hint, but it is still below the `+3/+3` gate and has
  no natural `Z` transfer, so it is not an attack-valid oracle.

## Branch D-Pair: Trace-only Paired Threshold Tomography on `crypto_kem_dec`

Date: 2026-05-08. Documented in `docs/d_pair_oracle.md`.

### Threat-Model Rebase

The paired threshold tomography idea was originally tested on `Z`
(`indcpa_dec` only). Re-reading the threat model, `Z` is also diagnostic
instrumentation — it omits the FO transform that makes the natural KEM
interface attack-valid. Only `D` (`crypto_kem_dec`, full FO) is
attack-valid for an external chosen-CT attacker. All Branch D-Pair attack
claims must come from `D`; `Z`/`R`/`Q`/`Y` results stay diagnostic.

### Phase 0 — Tooling

- `scripts/smaug/s2_z_capture_matrix.py` accepts `--cmd D`. The `D` response is
  the 1-byte `mismatch` flag; only ack length and determinism are
  validated, mu' round-trip is skipped.
- `scripts/smaug/s4_pair_distance_oracle.py` is a new analyzer that builds same-`c1`
  paired-`c2` features, evaluates held-out AUROC on `flip_any`, and returns
  ridge-regression rMAE/corr on `flip_byte_hw` as a secondary metric.
- `tests/smaug/test_pair_oracle.py` covers AUROC ties/edge cases, the
  XOR-of-`mu_bits` round-trip for `flip_any` and `flip_byte_hw`, and synthetic
  fixtures.
- `python3 tests/run_all.py` passes 82/82 (9 new oracle tests).

### Phase 1.0 — D Smoke

1 key × 2 designs × 3 traces, `--samples 24400`:

```text
shape (4, 3, 24400), elapsed 27.7s, all round-trips OK.
```

The CW-Lite buffer caps at ~24400 samples. `--samples 32000` returned
partial data with timeouts. Therefore one D capture covers ~0.83 ms,
predominantly the `indcpa_dec` portion of `crypto_kem_dec`.

### Phase 1.1 — D Scout (early window) and the α-bug

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd D --num-keys 6 -n 10 --samples 24400 \
  --design-mode detector-grid --coefs 0,8,16,24 --detector-alphas 64,128,192 \
  --c2-mode grid --c2-grid 15,16 \
  --tag s4_d_pair_b15_16_d24n10
```

6 keys × 12 c1 designs × 2 c2 × 10 traces, all `(24, 10, 24400)`, elapsed
2247s ≈ 37.5 min.

Naive pair-distance oracle, sweep best `b64_f128_r1_corr`, full window:

```text
AUROC_pooled = 0.6719 (null 0.6063+/-0.0441, z = +1.49)
AUROC_meankey= 0.6771 (null 0.5944+/-0.0460, z = +1.80)
class balance: pos_frac = 0.667 (pos=48 neg=24)
```

A direct check of the `flip_any` label across all six keys revealed an
**experiment design flaw**: the binary `flip_any` vector was *identical*
across keys for these chosen ciphertexts. For a monomial `c1 = α·X^j` with
`c2 = (15, 16)`, the bit-flip map at coordinate `i` only depends on
`s_{(i-j) mod n}` and on the (α, c2) threshold position:

```text
α =  64, c2=(15,16): flip iff s in {-1, +1}
α = 128, c2=(15,16): flip never (universal 0)
α = 192, c2=(15,16): flip iff s in {-1, +1}
```

Because SMAUG1 has HW=70 nonzero coords out of 256, "any flip" over the
full 256-coord vector is effectively constant: 1 for α=64/192, 0 for
α=128. **`flip_any` is determined by the public α**, not the secret.
Therefore the AUROC ≈ 0.67 result was the model classifying public α via
trace, not the secret. **Not an attack-valid oracle.**

### Phase 1.1' — Secret-dependent label

`flip_byte_hw[b]` (byte-wise XOR-HW of μ'(c2=a) vs μ'(c2=b)) is genuinely
secret-dependent for α=64,192: it equals the count of nonzero coordinates
within each 8-coord byte window. The α=128 designs have constant
`flip_byte_hw = 0` and only dilute the regression target.

D scout regression on `flip_byte_hw`, sweep best `b64_f128_r1_corr`,
full window:

```text
exact z = +1.22, rMAE z = +1.17, corr z = +1.56  (sub-gate)
```

D scout regression on `mu_delta_byte_hw` (signed byte-wise
HW(μ'(b)) − HW(μ'(a))), sweep best:

```text
exact z = +0.32, rMAE z = +0.21, corr z = +0.85  (null)
```

Window scan with the proper config across 0..24400 samples: no subwindow
reproduces the full-window result; signal is diffused (likely ridge
overfit on noise correlations rather than localized leakage).

### Phase 1.2 — D Control (c1=0)

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd D --num-keys 6 -n 10 --samples 24400 \
  --design-mode detector-grid --coefs 0,8,16,24 --detector-alphas 0 \
  --c2-mode grid --c2-grid 15,16 \
  --tag s4_d_pair_b15_16_c1zero_d8n10
```

c1=0 control elapsed ~13 min. With α=0, μ' depends only on c2 and gives
mu'_i=1 for both c2=15 and c2=16, so `flip_any` is always 0 (degenerate);
AUROC undefined.

Univariate paired TVLA between c2=15 and c2=16 traces:

```text
control (c1=0): max|t|=3.70 at sample 5411, frac>4.5=0%
scout   (c1≠0): max|t|=3.63 at sample 10209, frac>4.5=0%
```

**The scout's c2-pair trace difference is at the same noise level as
control's**. The first 24400 samples of D do not contain secret-dependent
paired-c2 leakage; the regression z=+1.56 from Phase 1.1' is a ridge-overfit
artifact, not a localized SCA leakage point.

### Phase 1.4 — Z Ablation (NOT attack claim)

Existing `s4_c2_z_pair_b15_16_d24n10_k*.npz` re-run through the new
analyzer. Z is diagnostic, included only for code-path validation and
relative comparison.

Sweep best `b32_f64_r100_corr`, full window:

```text
AUROC_pooled = 0.7413 (null 0.6523, z = +2.15)  -- alpha-classifier (see flaw)
flip_byte_hw: corr z = +0.99  (sub-gate, secret-honest metric)
```

Z's "AUROC z=+2.15" is also alpha-classifier inflation. The
secret-honest regression metric `corr z = +0.99` is sub-gate. Z is
slightly weaker than D on the same secret-dep label (`+1.56` vs `+0.99`),
but both are sub-gate.

### Phase 1.5 — Late-D Scout (`--adc-offset 24400`)

To capture the FO-downstream phase of `crypto_kem_dec` (which the first
24400 samples miss), the same chosen-CT matrix was re-captured with
`--adc-offset 24400`. Designs restricted to α∈{64,192} (drop the useless
α=128). Pilot (4 keys × 5 traces × 16 entries) showed corr z=+2.15 on
window 4096:8072 of late-D mu_delta_byte_hw — promising hint.

Full late-D scout (6 keys × 10 traces × 16 entries, elapsed 1502s ≈ 25
min):

```text
sweep best (full window):
  flip_byte_hw       b32_f64_r100_corr: exact z=+1.70, corr z=-0.29
  mu_delta_byte_hw   b64_f32_r100_snr:  exact z=+1.69, corr z=+0.60

window scan (best config), best window:
  4096:8072   corr z = -1.27   (pilot hint did NOT reproduce)
  12288:16264 corr z = +0.84
```

The pilot's corr z=+2.15 was a small-sample fluke. With proper power, late-D
also gives sub-gate signal. **FO downstream amplification is too weak under
this SCA setup to provide an attack-valid c2-paired oracle.**

### Phase 2.1 — c2 Staircase (3 secret-dependent pair types)

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd D --num-keys 6 -n 10 --samples 24400 \
  --design-mode detector-grid --coefs 0,8,16,24 --detector-alphas 64 \
  --c2-mode grid --c2-grid 7,8,15,16,23,24 \
  --tag s4_d_stair_a64_d24n10
```

Six keys × 4 c1 × 6 c2 × 10 traces, elapsed 2248s ≈ 37 min. Three
secret-dependent c2-pairs per c1 (delta=1):

```text
α=64 c2=(7,8):  flip iff s=0  (s=0 detector)
α=64 c2=(15,16):flip iff s≠0  (s≠0 detector, inverse)
α=64 c2=(23,24):flip iff s=0  (s=0 detector, alternate position)
```

Pooled `flip_byte_hw` sweep (12 pairs/key total):

```text
exact z = +0.44, rMAE z = -0.21, corr z = +0.42  (null)
```

Per-pair-type sweep (8 pairs/key per type):

```text
(7, 8)   mu_delta_byte_hw  best b32_f32_r10_snr:    exact z = -1.18,  corr z = -0.84
(15,16)  mu_delta_byte_hw  best b64_f64_r1_snr:     exact z = +0.08,  corr z = -0.74
(23,24)  mu_delta_byte_hw  best b32_f128_r10_snr:   exact z = +0.89,  corr z = -0.28
combined (7,8)+(23,24)     best b32_f256_r1_snr:    exact z = +0.73,  corr z = -0.52
```

All null. Different threshold positions don't help.

Univariate paired TVLA per c2-pair (240 traces × 240 traces):

```text
(7,8):   max|t| = 4.13 at sample 4123,  frac>4.5 = 0%
(15,16): max|t| = 4.61 at sample 4686,  frac>4.5 = 0%
(23,24): max|t| = 3.76 at sample 1671,  frac>4.5 = 0%
```

Univariate paired signal is essentially noise across all three pair types
on the natural D trace.

### Phase 2.2 — Multi-term c1

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd D --num-keys 6 -n 10 --samples 24400 \
  --design-mode random-multiterm --num-designs 8 --terms 4 \
  --design-seed 20260508 --c2-mode grid --c2-grid 15,16 \
  --tag s4_d_multiterm_d8n10
```

Random 4-term c1 with random alphas ∈ {32,64,96,128,160,192,224}. Each
inner_i is a sum of up to 4 secret terms scaled by α. Six keys × 8 designs
× 2 c2 × 10 traces, elapsed 1503s ≈ 25 min.

Sweep on full window:

```text
flip_byte_hw      best b8_f256_r100_snr:   exact z = -0.17, rMAE z = -0.79, corr z = -2.43
mu_delta_byte_hw  best b16_f64_r100_snr:   exact z = +1.28, rMAE z = +0.11, corr z = -2.56
```

Both labels give **negative correlation z** — the held-out predictions
correlate inverse to truth. This is overfit on noise, not signal. Multi-term
random alphas do not rescue the natural-D oracle.

### D-Pair Branch Conclusion

For SMAUG-T v4.0 on STM32F415 + CW-Lite, paired threshold tomography on
the natural `crypto_kem_dec` interface does not produce an attack-valid
oracle at the strict gate (`rMAE z > +3 AND corr z > +3`). Across the
matrix of attempted variations:

```text
attempt                                     secret-dep regr corr z
Z early-window monomial c1, c2=(15,16)      +0.99   (diagnostic, not attack)
D early-window monomial c1, c2=(15,16)      +1.56   (attack-valid)
D late-window  monomial c1, c2=(15,16)      +0.60   (full scout) / -1.27 (best window)
D staircase (7,8) + (15,16) + (23,24)        null per-pair-type, max +0.89 exact
D multi-term  random 4-term, c2=(15,16)     -2.43..-2.56 (inverted overfit)
```

The univariate paired TVLA confirms the limit: max|t| ≈ 3.7..4.6 across
240..720 traces per c2 side, regardless of c1 design or window phase.
This is consistent with the existing `snr_limit_cwlite.md` finding that
per-bit |t| stalls at 4–5 even at N=1000.

The two methodological wins from this branch are:

1. **Threat model clarification**: only `D` is attack-valid. Z/V/R/Q/Y/W/T/U
   are diagnostic instrumentation. Earlier C2 branch results on Z must be
   restated as diagnostic localization, not attack claims.
2. **`flip_any` α-bug**: any "binary did mu' flip" oracle is contaminated
   by public-α classification when c1 is monomial(α, j) with c2 chosen so
   that the threshold position is independent of s. Future paired-c2 work
   must either use signed `mu_delta_byte_hw` regression or pick c2 pairs
   that genuinely separate s-classes per coordinate.

The reusable artifact from this branch is the `D` capture path
(`--cmd D` in the matrix capture, the new `s4_pair_distance_oracle.py`
analyzer, and `tests/smaug/test_pair_oracle.py` with 9 oracle-specific tests).

### Final Pivot Recommendation

The remaining options against SMAUG-T on this CW-Lite rig are:

- **(a) Bigger SCA setup**: CW-Pro or higher-bandwidth oscilloscope, better
  EM probe. Justified if we keep SMAUG-T as the target. Hardware budget
  question.
- **(b) Different attack vector on SMAUG-T**: template attack with explicit
  per-coefficient secret model (per-pole leakage at known time samples), or
  multivariate FFT alignment with shifted templates, or single-trace EM at
  specific instruction window. All have been weakly explored in earlier
  branches with negative or sub-gate results, suggesting we are at the
  CW-Lite SNR ceiling rather than at a model-mismatch ceiling.
- **(c) NTRU+ pivot** (explicitly deferred per session scope, not executed
  in this session). NTRU+ has more classical SCA precedent and a
  structurally different attack surface (NTT pointwise multiplication,
  center-lift, decode threshold, FO re-encryption). The existing capture
  path (`D` in the matrix script, paired-c2 + multi-term + late-window
  options) can be re-targeted at NTRU+'s `crypto_kem_dec` with new firmware
  but identical analysis tooling.

Recommendation: NTRU+ pivot is the highest-EV next step. If pivot is
deferred, the SMAUG-T result should be written up as a *controlled
negative finding*: "v4.0 reference implementation on CW-Lite resists
paired-c2 threshold tomography on the natural KEM interface; pre- and
post-FO trace windows, monomial and multi-term chosen-CT designs, and
secret-dependent c2 pair staircases all return sub-gate results under
held-out-key permutation-null evaluation."
