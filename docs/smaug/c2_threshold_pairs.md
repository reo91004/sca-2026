# Branch C2: Paired Threshold Tomography

Date: 2026-05-06

## Motivation

The multiplication-centered branch is now a validated negative path for
trace-only recovery on the current implementation. Isolated multiplication
diagnostics leak, but the leakage does not transfer into natural `Z`/`V`/`W`
recovery pressure after held-out-key and permutation-null validation.

Branch R changed the target from multiplication intermediates to latent
`mu' = Dec(sk, ct)` materialization. It passed in isolated `round_t`/pack
diagnostics and produced weak support-entropy reduction, but natural `Z` and
the diagnostic bridge `Q` did not pass transfer gates.

This branch tests the most direct remaining hypothesis:

```text
s -> <c1, s> -> thresholded mu' -> pack/downstream trace
```

while subtracting the multiplication common mode with paired ciphertexts.

## Threat Model

The attacker may submit chosen ciphertexts and record traces. The attacker may
know all public ciphertext fields and may use profiling keys for supervised
training. The target key's `mu'` response, decapsulation response, or any
secret-dependent response byte is not used for inference.

Commands such as `R` or `Q` remain diagnostic only. A final attack claim must
come from natural `Z` or a full decapsulation command, not from an artificial
trigger alone.

## Core Design

For a fixed public `c1`, capture two ciphertexts:

```text
ct0 = (c1, c2 = beta)
ct1 = (c1, c2 = beta + delta)
```

Then analyze:

```text
DeltaT = T(ct1) - T(ct0)
```

The multiplication `c1 * s` should be nearly common mode. The remaining signal
should be enriched for threshold crossing, `mu'` materialization, packing, and
downstream dataflow.

Primary labels:

```text
flip_byte_hw     = HW_byte(mu'(beta + delta) xor mu'(beta))
flip_block16_hw  = HW_16(mu'(beta + delta) xor mu'(beta))
mu_delta_byte_hw = HW_byte(mu'(beta + delta)) - HW_byte(mu'(beta))
```

The `flip_*` labels are preferred because they directly measure threshold
crossings and are nonnegative.

## Required Controls

- `c1 = 0`, `c2` sweep control. If this is strong, the result is public `c2`
  or public packing leakage, not secret-dependent threshold leakage.
- Same `c1`, same `beta` null pair when possible. This checks whether pair
  differencing creates artifacts.
- Held-out key split. All designs from the target key stay out of training.
- Permutation null after pair construction.
- Fold-local feature selection only.
- Exact accuracy alone is not a success. Rounded-MAE and correlation must also
  move in the correct direction.

## Implementation Notes

Added `--c2-mode grid` to:

```text
scripts/smaug/s2_z_capture_matrix.py
```

This crosses every public `c1` design with a list of constant `c2` values and
stores per-design `c2_alpha` metadata.

Added paired analysis:

```text
scripts/smaug/s4_c2_pair_analyze.py
```

It finds same-`c1` pairs whose `c2` values differ by `--c2-delta`, builds
mean-trace differences, computes `flip_*` labels from profiling secrets, and
evaluates ridge predictors with held-out-key permutation null.

## Scout Matrix

First natural `Z` scout:

```bash
python3 scripts/smaug/s2_z_capture_matrix.py \
  --cmd Z \
  --num-keys 6 \
  -n 10 \
  --samples 24400 \
  --design-mode detector-grid \
  --coefs 0,8,16,24 \
  --detector-alphas 64,128,192 \
  --c2-mode grid \
  --c2-grid 0,1 \
  --tag s4_c2_z_pair_d24n10
```

This creates 12 public `c1` designs times 2 `c2` values, or 12 paired
observations per key for `delta=1`.

Primary analysis:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s4_c2_z_pair_d24n10_k*.npz \
  --label-kinds flip_byte_hw flip_block16_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --n-perm 300 \
  --out-prefix results/smaug/s4_c2_z_pair_d24n10_flip_p300
```

Control:

```bash
python3 scripts/smaug/s2_z_capture_matrix.py \
  --cmd Z \
  --num-keys 6 \
  -n 10 \
  --samples 24400 \
  --design-mode detector-grid \
  --coefs 0,8,16,24 \
  --detector-alphas 0 \
  --c2-mode grid \
  --c2-grid 0,1 \
  --tag s4_c2_z_public_c1zero_d8n10
```

Expected control behavior: the secret-dependent `flip_*` labels should have
zero variance or stay at null. If trace features classify public `c2` strongly
but target labels are public-only, it cannot be counted as secret leakage.

## Success Gate

Minimum gate for a live branch:

```text
rounded-MAE z > +3
corr z > +3
held-out key
target response unused
control does not explain signal
```

If the natural `Z` scout passes, the next step is to expand `beta`/`delta` and
compute posterior entropy reduction under a sparse ternary prior. If it fails,
run only one diagnostic comparison with `Q` or `R` to distinguish "no threshold
signal" from "natural transfer still suppresses it"; do not scale natural
captures blindly.

## Initial Result Log

### C2.0 implementation validation

Validation:

```bash
python3 -m compileall host scripts tests
python3 tests/run_all.py
```

Result:

```text
compileall OK
72/72 tests OK
```

The added unit test verifies that `flip_byte_hw` equals the byte grouping of
`mu'(beta + delta) xor mu'(beta)`.

### C2.1 natural Z, naive `c2 = 0 -> 1`

Capture:

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd Z \
  --num-keys 6 \
  -n 10 \
  --samples 24400 \
  --design-mode detector-grid \
  --coefs 0,8,16,24 \
  --detector-alphas 64,128,192 \
  --c2-mode grid \
  --c2-grid 0,1 \
  --tag s4_c2_z_pair_d24n10
```

All six planned keys completed with shape:

```text
(24 designs, 10 traces/design, 24400 samples)
```

Analysis:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s4_c2_z_pair_d24n10_k*.npz \
  --label-kinds flip_byte_hw flip_block16_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --n-perm 300 \
  --out-prefix results/smaug/s4_c2_z_pair_d24n10_flip_p300
```

Result:

```text
flip_byte_hw:
  exact 1.0000 vs null 1.0000+/-0.0000, z +0.00
  rMAE  0.0000 vs null 0.0000+/-0.0000, z +0.00
  corr  0.0000 vs null 0.0000+/-0.0000, z +0.00

flip_block16_hw:
  exact 1.0000 vs null 1.0000+/-0.0000, z +0.00
  rMAE  0.0000 vs null 0.0000+/-0.0000, z +0.00
  corr  0.0000 vs null 0.0000+/-0.0000, z +0.00
```

Interpretation: this is not a positive result. It is a degenerate label result:
`c2 = 0 -> 1` does not cross the `mu'` rounding threshold for these detector
designs.

Offline profiling-label grid over captured keys showed:

```text
delta=1:
  beta 7  -> 8   produces many flips but c1=0 public control flips all bits.
  beta 15 -> 16  produces secret-dependent flips while c1=0 control has no flip.
  beta 31 -> 0   is the wraparound analogue of 15 -> 16.
```

Therefore the next valid pair is `15 -> 16`, not `0 -> 1` or `7 -> 8`.

### C2.2 natural Z, threshold pair `c2 = 15 -> 16`

Capture:

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd Z \
  --num-keys 6 \
  -n 10 \
  --samples 24400 \
  --design-mode detector-grid \
  --coefs 0,8,16,24 \
  --detector-alphas 64,128,192 \
  --c2-mode grid \
  --c2-grid 15,16 \
  --tag s4_c2_z_pair_b15_16_d24n10
```

All six keys completed with shape:

```text
(24 designs, 10 traces/design, 24400 samples)
```

Primary analysis:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s4_c2_z_pair_b15_16_d24n10_k*.npz \
  --label-kinds flip_byte_hw flip_block16_hw mu_delta_byte_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --n-perm 300 \
  --out-prefix results/smaug/s4_c2_z_pair_b15_16_d24n10_flip_p300
```

Result:

```text
flip_byte_hw:
  exact 0.2049 vs null 0.2116+/-0.0139, z -0.48
  rMAE  1.4366 vs null 1.3937+/-0.0436, z -0.99
  corr  0.0166 vs null 0.0225+/-0.0476, z -0.12

flip_block16_hw:
  exact 0.1276 vs null 0.1214+/-0.0131, z +0.48
  rMAE  2.3481 vs null 2.4061+/-0.0831, z +0.70
  corr  0.0727 vs null 0.0514+/-0.0575, z +0.37

mu_delta_byte_hw:
  exact 0.3333 vs null 0.3173+/-0.0111, z +1.44
  rMAE  1.0629 vs null 1.0811+/-0.0226, z +0.81
  corr -0.0080 vs null -0.0016+/-0.0299, z -0.21
```

Config sweep:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s4_c2_z_pair_b15_16_d24n10_k*.npz \
  --label-kinds flip_byte_hw flip_block16_hw \
  --c2-delta 1 \
  --sweep \
  --n-perm 100 \
  --out-prefix results/smaug/s4_c2_z_pair_b15_16_d24n10_flip_sweep_p100
```

Best real-confirmed summaries:

```text
flip_byte_hw, b32_f64_r1_corr:
  exact z +0.82, rMAE z +0.72, corr z +0.57

flip_block16_hw, b8_f64_r1_snr:
  exact z -0.21, rMAE z -1.64, corr z -1.45
```

Interpretation: paired threshold differencing does not rescue natural `Z`.
Even after choosing a non-degenerate threshold pair, exact/rMAE/corr remain at
permutation-null level.

### C2.3 diagnostic R calibration for `c2 = 15 -> 16`

Purpose: distinguish "the `15 -> 16` label is bad" from "natural `Z` transfer
still suppresses the materialization leakage."

Capture:

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd R \
  --num-keys 6 \
  -n 10 \
  --samples 5000 \
  --design-mode detector-grid \
  --coefs 0,8,16,24 \
  --detector-alphas 64,128,192 \
  --c2-mode grid \
  --c2-grid 15,16 \
  --tag s4_c2_r_pair_b15_16_d24n10
```

All six keys completed with shape:

```text
(24 designs, 10 traces/design, 5000 samples)
```

Analysis:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s4_c2_r_pair_b15_16_d24n10_k*.npz \
  --label-kinds flip_byte_hw flip_block16_hw mu_delta_byte_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --n-perm 300 \
  --out-prefix results/smaug/s4_c2_r_pair_b15_16_d24n10_flip_p300
```

Result:

```text
flip_byte_hw:
  exact 0.3498 vs null 0.3721+/-0.0135, z -1.65
  rMAE  0.9692 vs null 0.9043+/-0.0228, z -2.84
  corr  0.4648 vs null 0.5296+/-0.0211, z -3.06

flip_block16_hw:
  exact 0.2595 vs null 0.2627+/-0.0159, z -0.20
  rMAE  1.3837 vs null 1.3495+/-0.0378, z -0.90
  corr  0.6683 vs null 0.6855+/-0.0162, z -1.06

mu_delta_byte_hw:
  exact 0.5360 vs null 0.4132+/-0.0247, z +4.97
  rMAE  0.7708 vs null 1.0490+/-0.0790, z +3.52
  corr  0.3867 vs null 0.0592+/-0.1190, z +2.75
```

Interpretation:

- Isolated `R` still contains `mu'` materialization information, but the
  nonnegative XOR flip labels are not the right target under this low-dimensional
  model. They are key/design structured enough that the permutation null is
  strong.
- Signed `mu_delta_byte_hw` is the best C2 diagnostic label. It passes exact and
  rounded-MAE gates, but correlation is slightly below the strict `+3` gate.
- Natural `Z` remains negative for the same `15 -> 16` pair, so the main C2
  hypothesis does not currently provide attack-valid recovery pressure.

### C2.4 paired window scan

Purpose: check whether C2.2 failed because the `mu'` pair signal was diluted in
the full natural `Z` window. This is still exploratory localization, not a final
attack claim.

Added:

```text
scripts/smaug/s4_c2_pair_window_scan.py
```

Natural `Z`, signed pair label:

```bash
python3 scripts/smaug/s4_c2_pair_window_scan.py \
  --inputs traces/s4_c2_z_pair_b15_16_d24n10_k*.npz \
  --label-kind mu_delta_byte_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --window 3976 \
  --stride 512 \
  --top-k 6 \
  --n-perm 200 \
  --out results/smaug/s4_c2_z_pair_b15_16_window_scan_mu_delta_byte_w3976_s512_p200.txt
```

Best confirmed windows:

```text
20424:24400:
  exact z +1.00, rMAE z +1.28, corr z +1.43

19968:23944:
  exact z +1.85, rMAE z +1.28, corr z +1.25

17920:21896:
  exact z +1.58, rMAE z +0.89, corr z +1.50
```

Natural `Z`, absolute trace difference:

```bash
python3 scripts/smaug/s4_c2_pair_window_scan.py \
  --inputs traces/s4_c2_z_pair_b15_16_d24n10_k*.npz \
  --label-kind mu_delta_byte_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --window 3976 \
  --stride 512 \
  --top-k 6 \
  --n-perm 200 \
  --absolute-diff \
  --out results/smaug/s4_c2_z_pair_b15_16_window_scan_mu_delta_byte_absdiff_w3976_s512_p200.txt
```

Best confirmed window:

```text
15360:19336:
  exact z +0.16, rMAE z -0.12, corr z +0.73
```

Diagnostic `R` calibration window scan:

```bash
python3 scripts/smaug/s4_c2_pair_window_scan.py \
  --inputs traces/s4_c2_r_pair_b15_16_d24n10_k*.npz \
  --label-kind mu_delta_byte_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --window 1024 \
  --stride 256 \
  --top-k 6 \
  --n-perm 200 \
  --out results/smaug/s4_c2_r_pair_b15_16_window_scan_mu_delta_byte_w1024_s256_p200.txt
```

Best confirmed windows:

```text
1536:2560:
  exact z +2.86, rMAE z +2.61, corr z +2.28

2048:3072:
  exact z +2.97, rMAE z +2.73, corr z +2.36

1792:2816:
  exact z +3.42, rMAE z +2.24, corr z +1.95
```

Interpretation:

- The paired window scanner can detect diagnostic `R` signal in the expected
  direction, but even the diagnostic result remains below the strict
  rounded-MAE/correlation `+3/+3` gate.
- Natural `Z` has only late-window hints around z `+1` to `+1.5`; this is not
  attack-quality.
- Absolute differencing does not help, so the natural failure is not just a
  sign convention issue in `T(ct1)-T(ct0)`.

### C2.5 diagnostic Q bridge for `c2 = 15 -> 16`

Purpose: check whether paired differencing survives when the trigger includes
the multiplication plus round/pack bridge. `Q` is diagnostic only; it is not a
natural oracle or final attack claim.

Capture:

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd Q \
  --num-keys 6 \
  -n 10 \
  --samples 24400 \
  --design-mode detector-grid \
  --coefs 0,8,16,24 \
  --detector-alphas 64,128,192 \
  --c2-mode grid \
  --c2-grid 15,16 \
  --tag s4_c2_q_pair_b15_16_d24n10
```

All six keys completed with shape:

```text
(24 designs, 10 traces/design, 24400 samples)
```

Full-window analysis:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s4_c2_q_pair_b15_16_d24n10_k*.npz \
  --label-kinds mu_delta_byte_hw flip_byte_hw flip_block16_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --n-perm 300 \
  --out-prefix results/smaug/s4_c2_q_pair_b15_16_d24n10_full_p300
```

Result:

```text
mu_delta_byte_hw:
  exact 0.3203 vs null 0.3199+/-0.0106, z +0.03
  rMAE  1.0681 vs null 1.0574+/-0.0219, z -0.49
  corr  0.0066 vs null 0.0055+/-0.0293, z +0.04

flip_byte_hw:
  exact z -1.23, rMAE z -1.16, corr z -0.98

flip_block16_hw:
  exact z -0.72, rMAE z +0.62, corr z +0.23
```

Window scan:

```bash
python3 scripts/smaug/s4_c2_pair_window_scan.py \
  --inputs traces/s4_c2_q_pair_b15_16_d24n10_k*.npz \
  --label-kind mu_delta_byte_hw \
  --c2-delta 1 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --window 3976 \
  --stride 512 \
  --top-k 6 \
  --n-perm 200 \
  --out results/smaug/s4_c2_q_pair_b15_16_window_scan_mu_delta_byte_w3976_s512_p200.txt
```

Best confirmed windows:

```text
9216:13192:
  exact z +0.63, rMAE z +2.22, corr z +2.29

7680:11656:
  exact z +0.55, rMAE z +0.85, corr z +2.08

6656:10632:
  exact z +0.78, rMAE z +1.44, corr z +2.14
```

p500 confirmation for the best window:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s4_c2_q_pair_b15_16_d24n10_k*.npz \
  --label-kinds mu_delta_byte_hw \
  --c2-delta 1 \
  --sample-range 9216:13192 \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --n-perm 500 \
  --out-prefix results/smaug/s4_c2_q_pair_b15_16_mu_delta_byte_w9216_13192_p500
```

Result:

```text
mu_delta_byte_hw, 9216:13192:
  exact 0.3286 vs null 0.3211+/-0.0095, z +0.79
  rMAE  1.0065 vs null 1.0524+/-0.0200, z +2.30
  corr  0.0815 vs null 0.0089+/-0.0331, z +2.19
```

Interpretation:

- `Q` full-window is completely null.
- `Q` has a weak localized hint in the middle of the trace, but p500
  confirmation stays below the strict rounded-MAE/correlation gate.
- The progression is now:

```text
R isolated round/pack      : weak signed mu_delta signal
Q mult + round/pack bridge : weak localized hint, gate-miss
Z natural indcpa_dec       : null-level after window scan
```

This means paired threshold tomography does not currently create an
attack-valid oracle. It does, however, localize where the diagnostic signal is
lost: adding the preceding multiplication and natural surrounding code rapidly
dilutes the round/pack delta signal.

## Current Conclusion

Branch C2 was the most coherent next attempt after Branch R, because it directly
targets the `mu'` threshold and tries to cancel multiplication common mode.
The experiment found a real design pitfall (`0 -> 1` is degenerate), a better
threshold pair (`15 -> 16`), and a weak diagnostic signed-delta signal in `R`.
The `Q` bridge shows only a sub-gate localized hint, and natural `Z` does not
show predictive `mu'`-derived pair leakage under held-out-key permutation-null
validation, even after window scanning.

This strengthens the broader project claim:

```text
SMAUG-T v4.0 diagnostic mu' materialization leakage exists,
but trace-only natural chosen-CT recovery pressure is still blocked by transfer
and window/model mismatch.
```

Do not scale `Z` C2 pairs blindly from this point. The only justified follow-up
would be a new natural command/window that isolates the actual `mu'` pack or
FO downstream region without using response bytes as an oracle.
