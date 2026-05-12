# Branch Y: FO Downstream Replay

Date: 2026-05-06

## Motivation

Branch R showed that isolated `round_t`/pack diagnostics can predict latent
`mu'` byte/block Hamming weight, but the signal did not transfer cleanly into
natural `Z`/`Q` traces. Branch C2 then tried same-`c1`, different-`c2` paired
threshold tomography. The corrected `15 -> 16` threshold pair created
secret-dependent offline labels, but natural `Z` and diagnostic `Q` remained
below the rounded-MAE/correlation gate.

Branch Y tests the next logical possibility: even if direct `mu'`
materialization is weak, FO downstream computation may amplify `mu'` through:

```text
mu' -> G(mu', H(pk)) -> K' || seed' -> re-encryption -> verify/cmov
```

If this path leaks coarse byte Hamming weight of `K'`, `seed'`, or their
combined 64-byte SHAKE output, it could become a candidate-scoring surface.

## Threat Model

The attacker may submit chosen ciphertexts and record traces. Public keys and
chosen ciphertext fields are public. Profiling keys may be used for supervised
training.

The `Y` command is diagnostic only. Its response is a sanity fingerprint and is
not used for inference. A final attack-valid claim would still need to transfer
to natural decapsulation traces, without using target `mu'`, `K'`, `seed'`, or
any secret-dependent response as an oracle.

## Diagnostic Command

Added firmware command:

```text
Y: run indcpa_dec outside the trigger, then trigger FO downstream replay
```

The triggered region computes:

```text
pk_hash = SHA3-256(pk)
kr      = SHAKE256(mu' || pk_hash, 64)
ct_cmp  = indcpa_enc(pk, mu', kr)
fail    = verify(ct_inj, ct_cmp)
alt     = SHAKE256(z || SHA3-256(ct_inj), 64)
cmov(kr[32:64], alt[32:64], fail)
```

The goal is localization, not oracle construction. `indcpa_dec` is outside the
trigger so the trace window emphasizes FO hashing, re-encryption, compare, and
fallback selection instead of the multiplication that dominated earlier `Z`
traces.

## Labels

Added host-side labels:

```text
fo_kr0_byte_hw  = byte HW of SHAKE256(mu' || H(pk))[0:32]
fo_kr1_byte_hw  = byte HW of SHAKE256(mu' || H(pk))[32:64]
fo_kr64_byte_hw = byte HW of all 64 bytes
```

These are computed only from profiling secrets plus public ciphertext design
and public key metadata. Target responses are not used.

## Implementation Notes

Firmware:

```text
firmware/simpleserial-smaug/simpleserial-smaug.c
```

Capture:

```text
scripts/smaug/s2_z_capture_matrix.py
```

- Adds command `Y`.
- Adds command `B` to dump public key chunks for metadata.
- Saves `pk_hex` into matrix captures so FO labels can include `H(pk)`.

Analysis:

```text
scripts/smaug/s2_z_lowdim_analyze.py
scripts/smaug/s2_z_label_window_scan.py
```

- Accepts `Y` matrix captures.
- Computes `fo_kr*` byte-HW labels.
- Propagates public key metadata through sliced datasets.

Validation:

```bash
python3 -m compileall host scripts tests
python3 tests/run_all.py
make PLATFORM=CW308_STM32F4
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
```

Result:

```text
compileall OK
73/73 tests OK
firmware build OK
firmware upload/verify OK
```

## Scout Matrix

Smoke:

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd Y \
  --num-keys 1 \
  -n 2 \
  --samples 24400 \
  --design-mode detector-grid \
  --coefs 0,8 \
  --detector-alphas 128 \
  --c2-mode grid \
  --c2-grid 15,16 \
  --tag s4_y_fo_downstream_smoke
```

Result:

```text
shape=(4, 2, 24400)
```

Main scout:

```bash
python3 -u scripts/smaug/s2_z_capture_matrix.py \
  --cmd Y \
  --num-keys 6 \
  -n 10 \
  --samples 24400 \
  --design-mode detector-grid \
  --coefs 0,8,16,24 \
  --detector-alphas 64,128,192 \
  --c2-mode grid \
  --c2-grid 15,16 \
  --tag s4_y_fo_downstream_d24n10
```

Result:

```text
6 keys completed
each key: shape=(24 designs, 10 traces/design, 24400 samples)
all designs: 10/10 captures
```

## Full-Window Analysis

Command:

```bash
python3 scripts/smaug/s2_z_lowdim_analyze.py \
  --inputs traces/s4_y_fo_downstream_d24n10_k*.npz \
  --label-kinds fo_kr0_byte_hw fo_kr1_byte_hw fo_kr64_byte_hw mu_byte_hw mu_block16_hw \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --n-perm 300 \
  --out-prefix results/smaug/s4_y_fo_downstream_d24n10_full_p300
```

Results:

```text
fo_kr0_byte_hw : exact z +0.64, rMAE z +0.34, corr z -0.74
fo_kr1_byte_hw : exact z +1.52, rMAE z +1.52, corr z +0.55
fo_kr64_byte_hw: exact z +1.19, rMAE z +1.41, corr z -0.39
mu_byte_hw     : exact z -0.76, rMAE z -0.32, corr z +0.28
mu_block16_hw  : exact z -0.45, rMAE z -1.13, corr z -1.71
```

Interpretation: `fo_kr1_byte_hw` is the best full-window label, but it is far
below the success gate. It is a weak hint, not evidence of usable leakage.

## Window Scan

Command:

```bash
python3 scripts/smaug/s2_z_label_window_scan.py \
  --inputs traces/s4_y_fo_downstream_d24n10_k*.npz \
  --label-kind fo_kr1_byte_hw \
  --block 8 \
  --n-features 32 \
  --ridge 10 \
  --feature-mode corr \
  --window 3976 \
  --stride 512 \
  --top-k 8 \
  --n-perm 200 \
  --out results/smaug/s4_y_fo_downstream_window_scan_fo_kr1_w3976_s512_p200.txt
```

Best confirmed windows:

```text
9216:13192  exact z +0.80, rMAE z +1.02, corr z +1.18
8704:12680  exact z +0.86, rMAE z +1.19, corr z +1.02
10240:14216 exact z +0.81, rMAE z +0.80, corr z +1.21
0:3976      exact z -0.18, rMAE z +0.28, corr z -0.01
```

Interpretation: localized windows do not rescue the branch. The best
correlation z-score is only about `+1.21`, and rounded-MAE remains near the
null distribution.

## Current Conclusion

Branch Y does not currently support the hypothesis that FO downstream replay
turns latent `mu'` into a strong trace-only oracle. The best FO label is
`fo_kr1_byte_hw`, but both full-window and localized-window validation remain
well below the strict gate:

```text
rounded-MAE z > +3
corr z > +3
held-out key
permutation null
target response unused
```

This is another controlled negative result. Together with Branch R and C2, it
narrows the remaining explanation: the implementation may expose diagnostic
`mu'` materialization under isolated triggers, but the signal is too weak or
too mixed after natural downstream computation to create recovery pressure with
the current low-dimensional trace model.

Do not scale this branch blindly. A follow-up should only proceed if it changes
the observation model substantially, for example a full natural decapsulation
window with stronger alignment controls, a verified instruction-local FO hash
window, or a model that explicitly scores candidate `mu'`/seed hypotheses
rather than predicting byte-HW independently.
