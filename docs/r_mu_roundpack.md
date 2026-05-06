# Branch R: `mu'` Round/Pack Leakage

Goal: test whether chosen ciphertexts can induce a latent `mu'` bit pattern
whose round/pack materialization leaks trace-only weak secret information.

This branch is separate from the Toom/Karatsuba recovery path. The old path
models mixed multiplication states:

```text
secret -> Toom/Karatsuba / vec_vec_mult_add state -> trace
```

Branch R instead makes the chosen ciphertext itself a detector:

```text
secret + chosen CT -> latent mu' detector bits -> round_t/pack trace
```

The response bytes from diagnostic commands are sanity-only. Attack-facing
claims must use traces, profiling-key labels, held-out keys, and permutation
nulls.

## R1: Diagnostic Round/Pack Trigger

Firmware command:

```text
R: resident sk + injected ct
   unpack sk/ct
   reproduce V-like vec_vec_mult_add with trigger OFF
   trigger_high
   round_t loop + pack 256 bits into 32B
   trigger_low
   return 32B mu' sanity response
```

The response is checked against host `predict_mu_prime`, then ignored for SCA
analysis.

Initial chosen-CT detector grid:

```text
component = 0
c2 = 0
alpha in {64, 128, 192}
shift j in {0, 8}
```

Labels:

```text
mu_total_hw
mu_block32_hw
mu_block16_hw
mu_byte_hw
mu_bit
```

Success gate:

```text
held-out key
permutation null
target response unused
rounded-MAE z > +3
correlation z > +3
```

`exact` alone is not enough.

## Implementation

- `firmware/simpleserial-smaug/simpleserial-smaug.c`
  - added `R` command for round_t/pack-only trigger.
- `scripts/s2_z_capture_matrix.py`
  - supports `--cmd R`.
  - supports `--design-mode detector-grid`.
- `scripts/s2_z_lowdim_analyze.py`
  - loads `cmd=R` matrix captures.
  - supports `mu_total_hw`, `mu_block32_hw`, `mu_block16_hw`,
    `mu_byte_hw`, and `mu_bit` labels.
- `tests/test_chosen.py`
  - checks the new component-local `mu_*` label path against full
    `chosen.predict_mu_prime`.

Validation:

```bash
python3 -m compileall host scripts tests
python3 tests/run_all.py
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
```

Result: host tests pass (`71/71 OK`), firmware builds, flash verifies.

## R1 Smoke

Command:

```bash
python3 scripts/s2_z_capture_matrix.py --cmd R --num-keys 1 -n 3 --samples 5000 --design-mode detector-grid --coefs 0,8 --detector-alphas 64,128,192 --tag s4_r_mu_roundpack_smoke
```

Result:

- 1 key, 6 designs, 3 traces/design.
- All designs captured `3/3`.
- `R` response matches host-predicted `mu'`.
- Saved:
  `traces/s4_r_mu_roundpack_smoke_k00_d6_n3_2b009bd4.npz`.

## R1 Scout

Capture:

```bash
python3 scripts/s2_z_capture_matrix.py --cmd R --num-keys 6 -n 10 --samples 5000 --design-mode detector-grid --coefs 0,8 --detector-alphas 64,128,192 --tag s4_r_mu_roundpack_d6n10_scout
```

Result:

- 6 keys, 6 designs, 10 traces/design.
- All keys/designs captured `10/10`.
- All first responses round-trip against host `predict_mu_prime`.
- Saved:
  `traces/s4_r_mu_roundpack_d6n10_scout_k*.npz`.

Analysis:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d6n10_scout_k*.npz --label-kinds mu_total_hw mu_block32_hw mu_block16_hw mu_byte_hw --sweep --n-perm 100 --out-prefix results/s4_r_lowdim_roundpack_d6n10_mu_sweep_p100
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d6n10_scout_k*.npz --label-kinds mu_byte_hw --block 8 --n-features 32 --ridge 1 --feature-mode corr --n-perm 500 --out-prefix results/s4_r_lowdim_roundpack_d6n10_mu_byte_confirm_p500
```

Results:

- `mu_total_hw`: exact z `+3.04`, rounded-MAE z `+2.02`,
  corr z `-0.65`.
- `mu_block32_hw`: exact z `+3.06`, rounded-MAE z `+1.32`,
  corr z `+0.42`.
- `mu_block16_hw`: exact z `+3.21`, rounded-MAE z `+0.13`,
  corr z `-0.98`.
- `mu_byte_hw` p100: exact z `+8.85`, rounded-MAE z `+4.49`,
  corr z `+1.72`.
- `mu_byte_hw` p500 confirm:
  exact `0.4158` vs null `0.2594 +/- 0.0184`, z `+8.50`;
  rounded-MAE `1.0191` vs null `1.2785 +/- 0.0587`, z `+4.42`;
  corr `0.2455` vs null `0.1557 +/- 0.0510`, z `+1.76`.

Interpretation:

- Branch R is alive. This is the first matrix experiment where a trace-only
  label beats permutation null on rounded-MAE by more than `+3`.
- The positive signal is strongest at byte-level `mu'` Hamming weight.
- The strict gate is not fully met because correlation is still below `+3`.
  The next experiment should improve localization or label granularity before
  claiming robust recovery pressure.
- This is a different result from exact-only Toom/Karatsuba bumps: the
  rounded-MAE improvement is confirmed at p500.

## Next R Experiments

1. Increase detector diversity modestly:

```text
alpha in {64, 128, 192}
shift j in {0, 8, 16, 24}
N = 20 traces/design
S = 8 or 12 keys
```

2. Scan narrower sample ranges around the repeated selected blocks:

```text
600:616
1208:1232
1816:1840
2424:2448
3032:3056
3496:3504
```

3. Add detector-specific labels:

```text
alpha=64  -> plus byte/block counts
alpha=128 -> support byte/block counts
alpha=192 -> minus byte/block counts
```

4. If byte-HW remains positive with correlation improvement, quantify weak
   recovery:

```text
byte histogram error
support/sign count error
sparse candidate entropy reduction
true-key candidate percentile
```

## R2: Window and Detector-Diversity Follow-Up

Motivation: R1 showed confirmed rounded-MAE improvement for `mu_byte_hw`, but
correlation was below the strict gate. R2 tested whether the signal is localized
to a small subwindow and whether modestly increasing detector diversity improves
held-out prediction.

### R2.1 Narrow Window Scan on R1 Data

Candidate subwindows were the repeated selected blocks from R1:

```text
600:616
1208:1232
1816:1840
2424:2448
3032:3056
3496:3504
```

Command pattern:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d6n10_scout_k*.npz --label-kinds mu_byte_hw --sample-range START:END --block 8 --n-features 32 --ridge 1 --feature-mode corr --n-perm 300 --out-prefix results/s4_r_lowdim_roundpack_d6n10_mu_byte_winSTART_END_p300
```

Result: all narrow windows were negative or weaker than the full window. This
means the `mu_byte_hw` signal is not carried by one tiny 8-24 sample island; the
model needs several repeated round/pack positions together.

Broader segment scan:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d6n10_scout_k*.npz --label-kinds mu_byte_hw --sample-range 1024:5000 --block 8 --n-features 32 --ridge 1 --feature-mode corr --n-perm 500 --out-prefix results/s4_r_lowdim_roundpack_d6n10_mu_byte_seg1024_5000_confirm_p500
```

Best R1 broad window:

- `1024:5000`, `mu_byte_hw`, p500:
  exact `0.3976` vs null `0.2575 +/- 0.0178`, z `+7.88`;
  rounded-MAE `1.0252` vs null `1.2853 +/- 0.0581`, z `+4.48`;
  corr `0.2573` vs null `0.1503 +/- 0.0512`, z `+2.09`.

Interpretation: removing the earliest 1024 samples improves correlation, but
R1 still misses the strict correlation gate.

Alpha-specific subsets on R1 data:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d6n10_scout_k*.npz --label-kinds mu_byte_hw --design-offset 0 --max-designs 2 --sample-range 1024:5000 --block 8 --n-features 32 --ridge 1 --feature-mode corr --n-perm 300 --out-prefix results/s4_r_lowdim_roundpack_d6n10_mu_byte_alpha64_seg1024_5000_p300
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d6n10_scout_k*.npz --label-kinds mu_byte_hw --design-offset 2 --max-designs 2 --sample-range 1024:5000 --block 8 --n-features 32 --ridge 1 --feature-mode corr --n-perm 300 --out-prefix results/s4_r_lowdim_roundpack_d6n10_mu_byte_alpha128_seg1024_5000_p300
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d6n10_scout_k*.npz --label-kinds mu_byte_hw --design-offset 4 --max-designs 2 --sample-range 1024:5000 --block 8 --n-features 32 --ridge 1 --feature-mode corr --n-perm 300 --out-prefix results/s4_r_lowdim_roundpack_d6n10_mu_byte_alpha192_seg1024_5000_p300
```

Result: alpha-specific subsets were weak. The positive R1 signal comes from the
combined detector family rather than from one isolated alpha with only two
shifts.

### R2.2 Detector-Diversity Capture

Capture:

```bash
python3 scripts/s2_z_capture_matrix.py --cmd R --num-keys 8 -n 10 --samples 5000 --design-mode detector-grid --coefs 0,8,16,24 --detector-alphas 64,128,192 --tag s4_r_mu_roundpack_d12n10_diversity
```

Result:

- 8 keys, 12 designs, 10 traces/design.
- Designs are `alpha in {64,128,192}` crossed with shifts `{0,8,16,24}`.
- All keys/designs captured `10/10`.
- All first responses round-trip against host `predict_mu_prime`.
- Saved:
  `traces/s4_r_mu_roundpack_d12n10_diversity_k*.npz`.

Analysis:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d12n10_diversity_k*.npz --label-kinds mu_total_hw mu_block32_hw mu_block16_hw mu_byte_hw --sample-range 1024:5000 --sweep --n-perm 100 --out-prefix results/s4_r_lowdim_roundpack_d12n10_diversity_mu_seg1024_5000_sweep_p100
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_r_mu_roundpack_d12n10_diversity_k*.npz --label-kinds mu_block16_hw mu_byte_hw --sample-range 1024:5000 --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 500 --out-prefix results/s4_r_lowdim_roundpack_d12n10_diversity_mu_byte_block16_confirm_p500
```

p100 sweep:

- `mu_block32_hw`: exact z `+5.99`, rounded-MAE z `+4.07`,
  corr z `+2.37`.
- `mu_block16_hw`: exact z `+9.76`, rounded-MAE z `+6.89`,
  corr z `+5.32`.
- `mu_byte_hw`: exact z `+8.21`, rounded-MAE z `+5.15`,
  corr z `+3.77`.

p500 confirmation:

- `mu_block16_hw`:
  exact `0.3483` vs null `0.2323 +/- 0.0127`, z `+9.16`;
  rounded-MAE `1.1400` vs null `1.3738 +/- 0.0390`, z `+6.00`;
  corr `0.5284` vs null `0.3832 +/- 0.0316`, z `+4.59`.
- `mu_byte_hw`:
  exact `0.4525` vs null `0.3144 +/- 0.0127`, z `+10.84`;
  rounded-MAE `0.7708` vs null `0.9585 +/- 0.0275`, z `+6.82`;
  corr `0.4279` vs null `0.2495 +/- 0.0362`, z `+4.93`.

Interpretation:

- R2 passes the strict gate. Both `mu_byte_hw` and `mu_block16_hw` beat the
  permutation null on exact, rounded-MAE, and correlation.
- This is now the primary positive branch: trace-only prediction of latent
  `mu'` byte/block Hamming weight from a round/pack diagnostic window.
- The next step is not more ridge tuning. The next step is to quantify weak
  recovery pressure from predicted byte/block histograms, then test transfer
  from diagnostic `R` to natural `Z` or `D` windows.

Next:

1. Convert `mu_byte_hw`/`mu_block16_hw` predictions into support/sign count
   error on detector groups.
2. Estimate entropy reduction for sparse ternary candidates under byte/block
   histogram constraints.
3. Capture or window-scan natural `Z`/`D` traces for the same detector-grid CTs
   and test whether the `1024:5000` diagnostic pattern transfers.

## R3: Conservative Recovery Pressure

Goal: translate the positive `mu_byte_hw` predictor into a weak-recovery metric
without claiming full key recovery.

Implementation:

- `scripts/s4_mu_recovery_pressure.py`
  - reuses the held-out-key ridge evaluator from `s2_z_lowdim_analyze.py`.
  - uses no target response bytes.
  - uses `alpha=128, shift=0` as a conservative support-byte detector.
  - projects predicted byte counts to the fixed `HS=70` support sum.
  - reports raw predicted entropy, oracle entropy, and a true-containing
    tolerant entropy. The tolerant entropy expands each byte interval just
    enough to include the real support count, preventing overconfident claims
    when predicted constraints exclude the target.

Command:

```bash
python3 scripts/s4_mu_recovery_pressure.py --inputs traces/s4_r_mu_roundpack_d12n10_diversity_k*.npz --sample-range 1024:5000 --block 8 --n-features 32 --ridge 10 --feature-mode corr --out results/s4_r_mu_recovery_pressure_d12n10_seg1024_5000.txt
```

Result:

- `mu_byte_hw` low-dimensional prediction:
  exact `0.4525`, rounded-MAE `0.7708`, corr `0.4279`.
- Prior support entropy:
  `log2(C(256,70)) = 212.50` bits.
- Prior ternary entropy with unconstrained signs:
  `282.50` bits.
- Raw predicted support-byte constraints:
  mean reduction `107.32 +/- 24.64` bits.
- Oracle support-byte constraints:
  mean reduction `70.06 +/- 3.87` bits.
- True-containing tolerant constraints:
  mean reduction `36.08 +/- 5.32` bits.
- Projected support-byte count error:
  L1 `50.25 +/- 11.49`, MAE `1.570` per byte.
- Detector histogram sanity:
  `alpha=64` byte-count L1 `20.88 +/- 2.75`;
  `alpha=192` byte-count L1 `23.50 +/- 4.17`.
- Shift-averaged support pressure using all byte-aligned `alpha=128` shifts:
  raw reduction `99.32 +/- 21.33` bits;
  true-containing tolerant reduction `37.13 +/- 8.23` bits;
  byte-count L1 `43.75 +/- 7.21`, MAE `1.367` per byte.

Interpretation:

- The raw predicted constraints are overconfident. They can reduce the space
  more than oracle constraints because they may exclude the true support.
- A safer claim is the true-containing tolerant number: with enough tolerance
  to keep the target, the predicted support-byte histogram still removes about
  `36-37` bits of support entropy on average.
- This is a meaningful weak-recovery pressure result, not full key recovery.
  The next recovery step should combine overlapping shifted histograms and
  detector signs with calibrated uncertainty instead of hard byte constraints.

## R4: Natural `Z` Transfer Check

Goal: test whether the positive diagnostic `R` round/pack result transfers to a
natural full `indcpa_dec` (`Z`) trace for the same detector-grid CTs.

Capture:

```bash
python3 scripts/s2_z_capture_matrix.py --cmd Z --num-keys 8 -n 10 --samples 24400 --design-mode detector-grid --coefs 0,8,16,24 --detector-alphas 64,128,192 --tag s4_z_mu_detector_d12n10_transfer
```

Result:

- 8 keys, 12 designs, 10 traces/design.
- All keys/designs captured `10/10`.
- All first responses round-trip against host `predict_mu_prime`.
- Saved:
  `traces/s4_z_mu_detector_d12n10_transfer_k*.npz`.

Full-window analysis:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_z_mu_detector_d12n10_transfer_k*.npz --label-kinds mu_block16_hw mu_byte_hw --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 300 --out-prefix results/s4_z_lowdim_mu_detector_d12n10_full_mu_byte_block16_p300
```

Results:

- `mu_block16_hw`: exact z `+0.87`, rounded-MAE z `+0.39`,
  corr z `+0.16`.
- `mu_byte_hw`: exact z `+0.02`, rounded-MAE z `+0.64`,
  corr z `-1.99`.

Segment scan:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_z_mu_detector_d12n10_transfer_k*.npz --label-kinds mu_byte_hw --sample-range 0:4096 --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 200 --out-prefix results/s4_z_lowdim_mu_detector_d12n10_mu_byte_seg0_4096_p200
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_z_mu_detector_d12n10_transfer_k*.npz --label-kinds mu_byte_hw --sample-range 4096:8192 --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 200 --out-prefix results/s4_z_lowdim_mu_detector_d12n10_mu_byte_seg4096_8192_p200
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_z_mu_detector_d12n10_transfer_k*.npz --label-kinds mu_byte_hw --sample-range 8192:12288 --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 200 --out-prefix results/s4_z_lowdim_mu_detector_d12n10_mu_byte_seg8192_12288_p200
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_z_mu_detector_d12n10_transfer_k*.npz --label-kinds mu_byte_hw --sample-range 12288:16384 --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 200 --out-prefix results/s4_z_lowdim_mu_detector_d12n10_mu_byte_seg12288_16384_p200
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_z_mu_detector_d12n10_transfer_k*.npz --label-kinds mu_byte_hw --sample-range 16384:20480 --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 200 --out-prefix results/s4_z_lowdim_mu_detector_d12n10_mu_byte_seg16384_20480_p200
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_z_mu_detector_d12n10_transfer_k*.npz --label-kinds mu_byte_hw --sample-range 20480:24400 --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 200 --out-prefix results/s4_z_lowdim_mu_detector_d12n10_mu_byte_seg20480_24400_p200
```

Best segment result was still weak:

- `20480:24400`, `mu_byte_hw`:
  exact z `+1.32`, rounded-MAE z `+1.30`, corr z `-0.04`.

Sliding-window localization:

```bash
python3 scripts/s2_z_label_window_scan.py --inputs traces/s4_z_mu_detector_d12n10_transfer_k*.npz --label-kind mu_byte_hw --block 8 --n-features 32 --ridge 10 --feature-mode corr --window 3976 --stride 512 --top-k 6 --n-perm 200 --out results/s4_z_label_window_scan_mu_byte_w3976_s512_p200.txt
```

Best confirmed window:

- `14848:18824`, `mu_byte_hw`:
  exact z `+1.30`, rounded-MAE z `+1.08`, corr z `+1.41`.

Other top real-only windows were also below threshold:

```text
15360:19336  corr z +0.92
15872:19848  corr z +1.02
14336:18312  corr z +0.79
11776:15752  corr z +0.29
17408:21384  corr z +0.60
```

Interpretation:

- Diagnostic `R` gives a strong positive result, but natural `Z` transfer is
  not established by this capture.
- The likely reason is window mixing: `Z` contains unpacking, shifts,
  multiplication, round/pack, and other loop structure in one long trigger.
  The round/pack leakage that is clean in `R` is diluted or mislocalized in
  full `Z`.
- Current primary claim should therefore remain diagnostic trace-only
  round/pack leakage plus weak entropy pressure, with natural `Z` transfer
  listed as an open/negative checkpoint.
- The sliding-window scan makes the negative transfer result stronger: the
  failure is not explained by a coarse 4K segment boundary alone.

## R5: `Q` Bridge Check

Goal: separate two possible explanations for the failed natural `Z` transfer.

- If `R` leakage survives in a trigger that includes `vec_vec_mult_add`
  immediately before `round_t/pack`, then the main `Z` problem is likely
  full-decapsulation window mixing or coarse alignment.
- If `R` leakage does not survive this bridge, then the strong diagnostic
  result is fragile to multiplication-window dilution even before the rest of
  `indcpa_dec` is included.

Firmware command:

- `Q` replays the same resident-sk + chosen-CT path as `V/R`.
- It triggers around:
  `vec_vec_mult_add_namespaced(...)` followed immediately by the
  `round_t` loop and 32-byte pack.
- The response is the 32-byte `mu'` sanity value, used only for round-trip
  validation during capture and not as an analysis oracle.

Capture:

```bash
python3 scripts/s2_z_capture_matrix.py --cmd Q --num-keys 8 -n 10 --samples 24400 --design-mode detector-grid --coefs 0,8,16,24 --detector-alphas 64,128,192 --tag s4_q_mu_bridge_d12n10
```

Result:

- 8 keys, 12 designs, 10 traces/design.
- All keys/designs captured `10/10`.
- All first responses round-trip against host `predict_mu_prime`.
- Saved:
  `traces/s4_q_mu_bridge_d12n10_k*.npz`.

Full-window analysis:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_q_mu_bridge_d12n10_k*.npz --label-kinds mu_block16_hw mu_byte_hw --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 300 --out-prefix results/s4_q_lowdim_mu_bridge_d12n10_full_mu_byte_block16_p300
```

Results:

- `mu_block16_hw`: exact z `-0.55`, rounded-MAE z `-2.92`,
  corr z `-2.95`.
- `mu_byte_hw`: exact z `-1.37`, rounded-MAE z `-1.34`,
  corr z `-2.28`.

Sliding-window localization:

```bash
python3 scripts/s2_z_label_window_scan.py --inputs traces/s4_q_mu_bridge_d12n10_k*.npz --label-kind mu_byte_hw --block 8 --n-features 32 --ridge 10 --feature-mode corr --window 3976 --stride 512 --top-k 6 --n-perm 200 --out results/s4_q_label_window_scan_mu_byte_w3976_s512_p200.txt
python3 scripts/s2_z_label_window_scan.py --inputs traces/s4_q_mu_bridge_d12n10_k*.npz --label-kind mu_block16_hw --block 8 --n-features 32 --ridge 10 --feature-mode corr --window 3976 --stride 512 --top-k 6 --n-perm 200 --out results/s4_q_label_window_scan_mu_block16_w3976_s512_p200.txt
```

Best `mu_byte_hw` confirmed window:

- `5632:9608`: exact z `+1.85`, rounded-MAE z `+1.98`,
  corr z `+1.76`.

Best `mu_block16_hw` confirmed window:

- `7168:11144`: exact z `+2.75`, rounded-MAE z `+2.77`,
  corr z `+2.69`.

Stricter p500 confirmation for the best `mu_block16_hw` window:

```bash
python3 scripts/s2_z_lowdim_analyze.py --inputs traces/s4_q_mu_bridge_d12n10_k*.npz --label-kinds mu_block16_hw --sample-range 7168:11144 --block 8 --n-features 32 --ridge 10 --feature-mode corr --n-perm 500 --out-prefix results/s4_q_lowdim_mu_block16_w7168_11144_p500
```

Result:

- `mu_block16_hw`, `7168:11144`:
  exact z `+2.76`, rounded-MAE z `+2.54`, corr z `+2.40`.

Interpretation:

- `Q` does not pass the pre-defined transfer gate
  (`rounded-MAE z > +3` and `corr z > +3`).
- There is a weak localized hint around `7168:11144`, but it does not survive
  stricter confirmation as an attack-quality signal.
- Therefore the strong `R` diagnostic leakage is fragile once the preceding
  multiplication is included in the same window. The natural `Z` failure is
  not only a coarse full-window problem; the `vec_vec_mult_add` region itself
  appears to dilute or dominate the `mu'` round/pack leakage.
- The most conservative current claim is:
  isolated `round_t/pack` leaks latent `mu'` byte/block-HW strongly, but that
  leakage has not transferred to a realistic trigger containing multiplication
  or full decapsulation.
