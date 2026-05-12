# S5: Trace-Count Scaling and Detection-Limit Audit

Date: 2026-05-06

## Question

After Branch R, C2, and Y, the key uncertainty is no longer "which label is
mathematically plausible?" but:

```text
Is the transfer failure just a trace-count problem,
or is it label/window/model mismatch?
```

S5 therefore reuses existing captures and varies the number of traces averaged
per `(key, design)` observation. If a branch contains weak but real leakage,
rounded-MAE and correlation z-scores should generally improve as `N` increases.
If z-scores stay null-level or move inconsistently, scaling that branch is not
justified.

Success gate remains:

```text
rounded-MAE z > +3
corr z > +3
held-out key
permutation null
target response unused
```

Exact-only improvements are not counted.

## S5.1 Positive Control: Diagnostic R Round/Pack

Dataset:

```text
traces/s4_r_mu_roundpack_d12n10_diversity_k*.npz
```

Label/window:

```text
mu_byte_hw, sample_range=1024:5000
```

Commands:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py \
  --inputs traces/s4_r_mu_roundpack_d12n10_diversity_k*.npz \
  --label-kinds mu_byte_hw \
  --sample-range 1024:5000 \
  --max-traces N \
  --block 8 --n-features 32 --ridge 10 --feature-mode corr \
  --n-perm 200 \
  --out-prefix results/smaug/s5_scaling_r_mu_byte_seg1024_5000_nN_p200
```

Results:

```text
N=2   exact z +8.39,  rMAE z +5.61,  corr z +4.29
N=5   exact z +8.83,  rMAE z +5.16,  corr z +3.77
N=10  exact z +10.19, rMAE z +6.09,  corr z +4.49
```

Interpretation: the audit detects real leakage when it exists. `R` is a valid
positive control.

## S5.2 Natural Z Transfer

Dataset:

```text
traces/s4_z_mu_detector_d12n10_transfer_k*.npz
```

Label/window:

```text
mu_byte_hw, sample_range=14848:18824
```

This was the best real-only window from the previous `Z` scan.

Results:

```text
N=2   exact z -0.01, rMAE z -0.02, corr z -0.74
N=5   exact z +0.34, rMAE z +0.70, corr z -0.18
N=10  exact z +1.32, rMAE z +1.15, corr z +1.63
```

Interpretation: there is a small `N=10` hint, but it remains far below gate and
does not show a strong scaling trend from `N=2` to `N=10`.

## S5.3 Q Bridge

Dataset:

```text
traces/s4_q_mu_bridge_d12n20_k*.npz
```

Label/window:

```text
mu_block16_hw, sample_range=7168:11144
```

Results:

```text
N=2   exact z -1.25, rMAE z -1.23, corr z -2.03
N=5   exact z +0.29, rMAE z -0.39, corr z -1.33
N=10  exact z -0.85, rMAE z -1.09, corr z -1.99
N=20  exact z -0.08, rMAE z -1.08, corr z -1.39
```

Interpretation: increasing from `N=2` to `N=20` does not reveal the bridge
label. This is strong evidence against a simple trace-count explanation for
the `Q` failure.

## S5.4 Y FO Downstream

Dataset:

```text
traces/s4_y_fo_downstream_d24n10_k*.npz
```

Label/window:

```text
fo_kr1_byte_hw, sample_range=9216:13192
```

Results:

```text
N=2   exact z +0.87, rMAE z -0.65, corr z -1.72
N=5   exact z +1.77, rMAE z +2.14, corr z +1.13
N=10  exact z +0.68, rMAE z +0.85, corr z +1.09
```

Interpretation: `N=5` gives a localized bump, but it does not persist at
`N=10`. This is not a reliable trace-count scaling trend.

## S5.5 C2 Paired Threshold Tomography

### Diagnostic Q Pair

Dataset:

```text
traces/s4_c2_q_pair_b15_16_d24n10_k*.npz
```

Label/window:

```text
mu_delta_byte_hw, sample_range=9216:13192, c2_delta=1
```

Results:

```text
N=2   exact z -0.51, rMAE z -0.68, corr z -0.69
N=5   exact z +0.63, rMAE z +0.28, corr z -0.10
N=10  exact z +0.76, rMAE z +2.30, corr z +2.21
```

Interpretation: this is the only branch with a real scaling hint. It still
misses the strict gate, but unlike `Y` the `N=10` bump is coherent across rMAE
and correlation. However, `Q` is diagnostic/bridge-only.

### Natural Z Pair

Dataset:

```text
traces/s4_c2_z_pair_b15_16_d24n10_k*.npz
```

Same label/window:

```text
mu_delta_byte_hw, sample_range=9216:13192, c2_delta=1
```

Results:

```text
N=2   exact z -0.76, rMAE z -1.50, corr z -0.54
N=5   exact z -1.63, rMAE z -1.27, corr z -0.50
N=10  exact z +0.37, rMAE z +0.03, corr z -0.52
```

Interpretation: the diagnostic `Q` hint does not transfer to natural `Z`.
Therefore this is not currently an attack-valid scale-up candidate.

## Conclusion

S5 separates "trace-count problem" from "mismatch problem":

```text
R diagnostic     : strong positive control, passes even at N=2
Z transfer       : weak sub-gate hint at N=10 only
Q bridge         : no improvement through N=20
Y FO downstream  : non-monotonic bump, not stable
C2-Q diagnostic  : coherent sub-gate hint at N=10
C2-Z natural     : null-level through N=10
```

The main conclusion is that current natural/bridge failures are not explained
by simply having too few traces. If the label/window/model were correct, the
positive control shows that this audit would see it.

Do not scale natural `Z`, `Q`, or `Y` blindly. The only possible focused
follow-up is a narrow diagnostic `C2-Q` recapture at higher `N` to test whether
the `+2.3/+2.2` hint crosses gate. That would still be a localization result,
not an attack-valid oracle, unless a corresponding natural `Z` or full
decapsulation transfer is later demonstrated.

## S5.6 Focused C2-Q Recapture at N=20

Because diagnostic `C2-Q` was the only S5 branch with a coherent sub-gate
scaling hint, it was recaptured independently at `N=20`.

Capture:

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd Q \
  --num-keys 6 \
  -n 20 \
  --samples 24400 \
  --design-mode detector-grid \
  --coefs 0,8,16,24 \
  --detector-alphas 64,128,192 \
  --c2-mode grid \
  --c2-grid 15,16 \
  --tag s5_c2_q_pair_b15_16_d24n20
```

Result:

```text
6 keys completed
each key: shape=(24 designs, 20 traces/design, 24400 samples)
all designs: 20/20 captures
```

First confirmation on the previous best window:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s5_c2_q_pair_b15_16_d24n20_k*.npz \
  --label-kinds mu_delta_byte_hw \
  --sample-range 9216:13192 \
  --max-traces 20 \
  --c2-delta 1 \
  --block 8 --n-features 32 --ridge 10 --feature-mode corr \
  --n-perm 500 \
  --out-prefix results/smaug/s5_c2q_d24n20_mu_delta_byte_w9216_13192_p500
```

Result:

```text
N=20, 9216:13192: exact z +0.20, rMAE z +0.94, corr z +0.81
```

The previous `N=10` bump did not reproduce at the same window. A `N=10` subset
of the new capture also failed to reproduce the earlier rounded-MAE signal:

```text
N=10, 9216:13192: exact z -0.18, rMAE z +0.14, corr z +1.99
```

Window scan on the new `N=20` capture:

```bash
python3 scripts/smaug/s4_c2_pair_window_scan.py \
  --inputs traces/s5_c2_q_pair_b15_16_d24n20_k*.npz \
  --label-kind mu_delta_byte_hw \
  --c2-delta 1 \
  --block 8 --n-features 32 --ridge 10 --feature-mode corr \
  --window 3976 --stride 512 --top-k 8 \
  --n-perm 300 \
  --out results/smaug/s5_c2q_d24n20_window_scan_mu_delta_byte_w3976_s512_p300.txt
```

Best real-only window:

```text
7680:11656: exact z +1.25, rMAE z +2.92, corr z +2.07
```

Higher-permutation confirmation:

```bash
python3 scripts/smaug/s4_c2_pair_analyze.py \
  --inputs traces/s5_c2_q_pair_b15_16_d24n20_k*.npz \
  --label-kinds mu_delta_byte_hw \
  --sample-range 7680:11656 \
  --max-traces 20 \
  --c2-delta 1 \
  --block 8 --n-features 32 --ridge 10 --feature-mode corr \
  --n-perm 1000 \
  --out-prefix results/smaug/s5_c2q_d24n20_mu_delta_byte_w7680_11656_p1000
```

Result:

```text
N=20, 7680:11656: exact z +1.22, rMAE z +2.62, corr z +1.76
```

Updated interpretation: diagnostic `C2-Q` does contain the strongest remaining
hint in the project, but the hint remains below the strict gate and is not
stable at the previous window. Since the corresponding natural `C2-Z` pair was
null-level, this should be treated as a diagnostic localization result, not an
attack-valid oracle or a scale-up justification for natural recovery.
