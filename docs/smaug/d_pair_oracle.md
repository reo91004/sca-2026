# Branch D-Pair: Trace-only μ'-change Oracle on `crypto_kem_dec`

Date: 2026-05-08

## Motivation

Earlier branches (R, Q, Y, C2) found strong diagnostic μ' leakage under
isolated firmware triggers but no transfer to natural decapsulation under a
regression label model. The user-driven retake of the threat model:

- `T`, `V`, `W`, `R`, `Q`, `Y`, `Z` are diagnostic instrumentation. Only `D`
  (full `crypto_kem_dec`) is attack-valid for an external chosen-CT attacker.
- The attacker should not regress on a continuous secret label they cannot
  measure; instead test the simpler binary hypothesis: did μ' change at all?

## Threat Model

The attacker submits a chosen ciphertext, captures one trace of
`crypto_kem_dec`, and may use *profiling* keys to learn a binary classifier.
Target keys are evaluated held-out; their decapsulation response bytes
(`ss_dec`, `mismatch`) are never used for inference. Diagnostic commands `Z`,
`Q`, `R`, `Y` may appear in *ablation* tables but must not be cited as the
result of an attack.

## Core Design

For a fixed public `c1` and adjacent `c2` values:

```text
ct_a = (c1, c2 = a)
ct_b = (c1, c2 = a + delta)
T_a, T_b : per-trace D captures
diff     : per-pair mean trace difference
```

Per (key, pair) labels (computed from profiling secrets only):

```text
flip_any        = int( μ'(sk, c1, a) != μ'(sk, c1, a+delta) )    # binary
flip_byte_hw[b] = HW( (μ'(sk,c1,a) XOR μ'(sk,c1,a+delta))[8b:8b+8] )  # 32 bytes
mu_delta_byte_hw[b] = HW(μ'(sk,c1,a+delta)[8b:8b+8]) - HW(μ'(sk,c1,a)[8b:8b+8])
```

Held-out key folds train ridge on `(diff -> label)`. Held-out AUROC is the
binary-task metric; rMAE/corr against `flip_byte_hw` is the regression aux.
A permutation null over key labels gives z-scores.

## Implementation

- `scripts/smaug/s2_z_capture_matrix.py` accepts `--cmd D`. The first response is the
  1-byte `mismatch` flag; only ack length and determinism are validated.
- `scripts/smaug/s4_pair_distance_oracle.py` builds same-`c1` `c2` pairs from the
  matrix capture, computes per-pair feature differences, evaluates held-out
  AUROC and ridge regression byte-HW metrics, runs a permutation null over key
  labels, and reports per-key AUROC.
- `tests/smaug/test_pair_oracle.py` covers AUROC ties/edge cases, the
  XOR-of-`mu_bits` round-trip for `flip_any` and `flip_byte_hw`, and synthetic
  perfect/random fixtures for the held-out evaluator.

Validation:

```bash
python3 -m compileall host scripts tests
python3 tests/run_all.py
```

Result: 82/82 tests OK (9 new oracle tests).

## Gate

A live oracle requires:

```text
held-out AUROC_pooled >= 0.85   OR   z(AUROC_pooled) > +3
held-out AUROC_meankey >= 0.85  OR   z(AUROC_meankey) > +3
control(c1=0) AUROC ~ 0.5  (within ±2 std of permutation null)
```

The control is required to rule out public-`c2` packing leakage.

## D Smoke

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd D --num-keys 1 -n 3 --samples 24400 \
  --design-mode detector-grid --coefs 0,8 --detector-alphas 128 \
  --c2-mode grid --c2-grid 15,16 \
  --tag s4_d_pair_b15_16_smoke
```

Result:

- 4 design entries (2 c1 × 2 c2), 3 traces each, all 3/3 ok.
- shape `(4, 3, 24400)`, ack length 1B (D mismatch flag), elapsed 27.7s.
- 24400 is the CW-Lite buffer cap; 32000 returned partial data with timeouts.
  Therefore the D capture covers only the first ~0.83 ms of `crypto_kem_dec`,
  predominantly the `indcpa_dec` activity. Late-window D (FO downstream) needs
  a separate `--adc-offset` capture.

## D Scout (full crypto_kem_dec, 6 keys × 12 c1 × 2 c2 × 10 traces)

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd D --num-keys 6 -n 10 --samples 24400 \
  --design-mode detector-grid --coefs 0,8,16,24 --detector-alphas 64,128,192 \
  --c2-mode grid --c2-grid 15,16 \
  --tag s4_d_pair_b15_16_d24n10
```

All 6 keys captured `(24, 10, 24400)`. Total elapsed 2247s ≈ 37.5 min.

### Initial AUROC result and a critical bug

Pair-distance oracle, default config `b8_f32_r10_corr`, full window:

```text
AUROC_pooled = 0.4696 (null 0.5391+/-0.0609, z = -1.14)  -- null
```

Sweep best config `b64_f128_r1_corr`, full window:

```text
AUROC_pooled = 0.6719 (null 0.6063+/-0.0441, z = +1.49)  -- positive but sub-gate
AUROC_meankey= 0.6771 (null 0.5944+/-0.0460, z = +1.80)  -- positive but sub-gate
rMAE z = +1.14   (regression aux)
corr z = +1.54   (regression aux)
class balance: pos_frac = 0.667 (pos=48 neg=24)
per-key AUROC: 0.812, 0.781, 0.469, 0.531, 0.812, 0.656
```

A direct check of the `flip_any` label across all six keys revealed an
**experiment design flaw**:

```text
flip_any vector across 6 keys  =  identical  [1,1,1,1, 0,0,0,0, 1,1,1,1]
```

For SMAUG1 with `c2=(15,16)` (delta=1) and a monomial `c1 = α·X^j`, the bit
flip behavior at coordinate `i` only depends on `s_{(i-j) mod n}` and the
α/c2 threshold position:

```text
α = 64 + c2=(15,16):  flip iff s in {-1, +1}      (nonzero detector)
α = 128 + c2=(15,16): flip never (universal)      (constant 0)
α = 192 + c2=(15,16): flip iff s in {-1, +1}      (nonzero detector)
```

Because SMAUG1 has HW=70 nonzero coordinates out of 256, *somewhere* in the
256-coord vector there is always at least one nonzero coordinate, so the
"any flip" reduction over 256 coords is effectively constant: 1 for α=64/192,
0 for α=128. **`flip_any` is determined by the public α**, not the secret.

Therefore the `AUROC_pooled = 0.6719` result is the model classifying
*public α* via the trace, not the secret. It is **not an attack-valid
oracle**.

### Secret-dependent labels

Per-byte `flip_byte_hw[b]` is the byte-wise XOR-HW of μ'(c2=a) and μ'(c2=b).
For α=64 c2=(15,16) it equals the count of nonzero coordinates within each
8-coord byte window — a genuinely secret-dependent quantity. The α=128 designs
contribute zero (constant) and dilute the regression target.

D scout regression on `flip_byte_hw` (the right secret-dependent label):

```text
b64_f128_r1_corr: corr z = +1.54, rMAE z = +1.14   (sub-gate)
```

D scout regression on `mu_delta_byte_hw` (signed difference, also
secret-dependent):

```text
b8_f32_r10_corr: corr z = -0.49, rMAE z = -0.49   (null)
b8_f32_r100_snr (sweep best): corr z = +0.85, rMAE z = +0.21   (null)
```

Z scout regression on `mu_delta_block16_hw` (existing C2.2 result):

```text
corr z = -0.21   (null)
```

### D window scan with `b64_f32_r1_corr`

```text
window         AUROC_pool z   AUROC_meankey z
0:3976         -0.62          -0.30
4096:8072      -2.23          -2.06
8192:12168     -0.30          +0.58
12288:16264    -0.10          -0.11
16384:20360    -1.13          -1.13
20424:24400    -0.21          -0.55
```

No subwindow reproduces the full-window `+1.80`; the (contaminated) signal
spreads across the whole 24400-sample window.

## Z Ablation (NOT attack claim, code path validation only)

Re-ran the new analyzer on existing `s4_c2_z_pair_b15_16_d24n10_k*.npz`.

Sweep best `b32_f64_r100_corr`, full window:

```text
AUROC_pooled = 0.7413 (null 0.6523+/-0.0415, z = +2.15)
AUROC_meankey= 0.7500 (null 0.6487+/-0.0443, z = +2.29)
rMAE z = +1.19   (regression aux)
corr z = +0.99   (regression aux)
```

Same `flip_any` design flaw applies: the AUROC z is mostly classifying
public α via the trace. The regression `corr z = +0.99` is the secret-honest
metric.

## Phase 1 Verdict

**FAIL.** With `c1 = α·X^j` for `α ∈ {64, 128, 192}` and `c2 = (15, 16)`:

- The naive `flip_any` AUROC is contaminated by public α classification.
- The honest secret-dependent regression on `flip_byte_hw` gives D `corr z =
  +1.54`, well below the `+3` gate.
- D and Z signal levels are similar; D does not gain meaningfully from the FO
  downstream because the 24400-sample capture window does not extend past
  `indcpa_dec` activity (CW-Lite buffer cap).

## Next Steps

In rough order of likely value:

1. **Late-D capture (`--adc-offset 24400` or larger)**: capture the part of
   `crypto_kem_dec` after `indcpa_dec`, where SHAKE/cmov/re-encryption
   amplifies μ' differences. The naive expectation is that FO downstream
   provides the "avalanche" structure earlier branches missed.
2. **c2 staircase + α-aware design**: replace `α ∈ {64,128,192}` with
   diverse alphas where `c2` thresholds genuinely depend on `s` at the
   single-coordinate level (e.g., `α=96` plus pairs `(7,8)` and `(23,24)`,
   which probe specifically `s = 0`; `α=64` plus `(15,16)`, which probes
   `s ≠ 0`). Multiple distinct probes per `c1` increase coverage.
3. **NTRU+ pivot**: if both above stay sub-gate, port to NTRU+ which has more
   classical SCA precedent.

The new oracle script and harness are reusable for all three; only the
capture matrix and label choices change.
