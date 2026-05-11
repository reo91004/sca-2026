# SCA Workbench

Workspace for evaluating side-channel leakage in KpqC KEM implementations on
ChipWhisperer-Lite + STM32F415.  The repository started as a SMAUG-T workbench;
the current active paper path is the NTRU+768 chosen-ciphertext NTT-domain CPA
line.

Current NTRU+ conclusion, as of 2026-05-11:

- The attack-valid leakage point is natural `crypto_kem_dec` reached through the
  NTRU+ SimpleSerial `D` command after chosen-ciphertext injection (`F/B/I/L/D`).
- Phase 4 multi-victim data gives **2 TOP-1** NTT-coordinate recoveries and
  **17 unique top-100 / 240** cases using M1 baseline and full-stack evidence.
- Phase 4.5/4.6 single-victim multi-lane data is now **3 victims, 88 lanes,
  352 cases**.  The aggregate top-100 counts are M1 baseline `9/352`, M5
  baseline `12/352`, full stack `8/352`, and M1/M5/full union `29/352`
  (channel-mixed upper bound), but candidate-set-size null analysis shows the
  single-victim raw top-100 counts are not statistically meaningful recovery
  evidence.
- A compact G=200 follow-up scout on calibrated lanes completed cleanly
  (`6400` traces, `0` timeouts) but did not improve the single-victim recovery
  claim: M1/full-stack had `0/16` top-100 and M5 had `1/16` top-100.  Treat
  simple G expansion as a negative scout, not a new positive result.  Follow-up
  PoI/window diagnostics showed that apparent oracle-PoI gains are explained by
  self-oracle selection bias, while attack-compatible PoI choices stay at
  `0/16`-`1/16` top-100.  Window reducer sweeps also stayed at `0/16` top-10.
- This is best stated as **partial NTT-coordinate information disclosure**, not
  full secret-key recovery.  Current lattice/full-key recovery remains
  infeasible under the measured recovery rate.
- Latest NTRU+ status lives in `docs/HANDOFF.md`,
  `docs/IDEA.md`, `docs/PAPER_OUTLINE.md`, and
  `results/ntruplus768/phase45/SUMMARY.md`.

SMAUG-T conclusion, as of 2026-05-05:

- The old `8 traces / 9 keypairs / 100%` story was an instrumented `mu'`-label result and is no longer a valid attack claim.
- PCO, MV-PC, DTW, FFT, and cross-key `mu'` extraction pilots are retired. They are negative evidence, not the current research path.
- The active experimental spine is:
  - **S1**: diagnostic `T` wrapper around `poly_mul_acc`. Strong cross-key leakage, sign recovery about 88%, support about 75% on held-out keys.
  - **S2**: chosen-ciphertext `Z` trace of `indcpa_dec`. Leakage is visible in cross-key Welch-t, but current per-coordinate HW models do not recover the sparse key.
  - **S3**: diagnostic `V` wrapper around `vec_vec_mult_add`. Current data shows weak leakage relative to S1/S2; it is a sanity/localization tool, not an attack result.

`T`, `V`, `X`, and `Z` responses are diagnostic instrumentation. Any attack-valid claim must use traces only and must not use returned `mu'` or dumped secret-key bytes for target-key inference. The defensible next step is to improve the S2 trace model or move to a natural `D`-trace window, not to revive old PCO or instrumented-oracle claims.

## Active Files

Core model and firmware:

- `firmware/simpleserial-smaug/simpleserial-smaug.c`
- `host/smaug/poly_mul.py`
- `host/smaug/chosen.py`
- `host/smaug/codec.py`
- `host/smaug/ciphertext.py`
- `host/smaug/params.py`
- `host/analysis/sparse_recover.py`

Active scripts:

- `scripts/s1_t_roundtrip.py`
- `scripts/s1_t_capture_main.py`
- `scripts/s1_u_capture_matrix.py`
- `scripts/s1_t_final_analysis.py`
- `scripts/s1_t_recover_bayes.py`
- `scripts/s2_z_capture_main.py`
- `scripts/s2_z_capture_matrix.py`
- `scripts/s2_z_analyze.py`
- `scripts/s2_z_recover.py`
- `scripts/s2_z_lowdim_analyze.py`
- `scripts/s2_z_design_score.py`
- `scripts/s2_z_leakage_select.py`
- `scripts/s2_z_label_pressure.py`
- `scripts/s2_z_n_sweep.py`
- `scripts/s3_v_roundtrip.py`
- `scripts/s3_v_capture_main.py`
- `scripts/s3_v_lowdim_analyze.py`
- `scripts/s3_v_analyze_2sk.py`
- `scripts/smoke.sh`

The detailed NTRU+ experiment log lives in `docs/IDEA.md`.  Generated traces
and results stay under `traces/` and `results/`, which are ignored by git.

## Reproduce Current Offline Checks

```bash
python3 scripts/s1_t_final_analysis.py --out-prefix results/recheck_s1_final --n-shuffles 50
python3 scripts/s1_t_recover_bayes.py --out-prefix results/recheck_s1_recover_bayes --conservative --margin-thresh 1.0
python3 scripts/s2_z_analyze.py --out-prefix results/recheck_s2_z_final --n-shuffles 50
python3 scripts/s3_v_analyze_2sk.py --sk-a traces/s3_v_skA_a4_n200.npz --sk-b traces/s3_v_skB_a4_n200.npz --out-prefix results/recheck_s3_v_2sk
python3 tests/run_all.py
```

Latest recheck results:

- S1 final: `99.9-pct |rho|` z-score about `+27`, Bonferroni PoIs `3664`.
- S1 Bayes LOO: coordinate `0.7363`, support `0.7466`, sign `0.8778`.
- S2 HW models: all tested single-coordinate HW models remain at shuffle-null level.
- S3 V two-key sanity: `max|t|=36.43`, much weaker than S1 and S2.

## Hardware Capture

Build and flash the firmware before capture:

```bash
make -C firmware/simpleserial-smaug PLATFORM=CW308_STM32F4 SMAUG_LEVEL=1
python3 host/upload.py firmware/simpleserial-smaug/simpleserial-smaug-CW308_STM32F4.hex
```

Capture examples:

```bash
python3 scripts/s1_t_capture_main.py -n 500 --out traces/s1_main_new_c0_j0_a1_n500.npz
python3 scripts/s2_z_capture_main.py -n 200 --alpha 4 --out traces/s2_z_new_a4_n200.npz
python3 scripts/s3_v_capture_main.py -n 200 --alpha 4 --out traces/s3_v_new_a4_n200.npz
```

Keep new claims conservative: diagnostic wrappers can locate leakage, but the paper-grade claim must be trace-only and target a natural decapsulation path.
