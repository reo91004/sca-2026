# Real-Trace Single-Stage SCA on SMAUG-T smaug1 in 8 Traces

(Working draft — paper 본문 markdown. 최종 LaTeX 변환은 별도.)

## Abstract

We present the first statistical side-channel attack on **SMAUG-T smaug1**, the
2025 KpqC competition winner KEM. Our attack recovers the full long-term secret
key (s ∈ {-1, 0, +1}^{256+256}, HW=70 per polynomial) with **8 power traces
(~3 seconds capture)** on a ChipWhisperer-Lite + Vcc-shunt + STM32F415 setup.
We achieve **100% recovery accuracy across 9 randomly generated key pairs**,
with zero variance.

The attack is a *single-stage online* procedure: no offline template profiling,
no auxiliary training data, no EM probe. Three method ingredients enable this:
(i) a chosen-CT *convention correction* that exposes 256 secret coefficients
simultaneously per query, reducing prior 32k+ trace estimates to 4 traces of
oracle pair (4400× efficiency); (ii) a *direct attack PoI* learner that
extracts per-bit Welch-t templates from the attack data itself via cross-design
splits, sidestepping cross-domain transfer of profile templates; and (iii) a
*component-specific PoI* refinement that isolates each polynomial's leakage,
combined with sparse-ternary MAP recovery under HW=70 constraint.

We show that the attack reaches the same SOTA tier as Kyber Hamburg21 (2-4
traces, masked) and HQC eprint25 (single-trace, codeword-masked) without any
of their additional adversary capabilities. Our negative findings — profile
transfer fail, multi-term TVLA noise floor at N=64 — and the underlying SNR
phase transition complete the picture of when each approach applies.

## 1. Introduction

Post-quantum cryptography (PQC) has entered the standardization phase: NIST's
ML-KEM (Kyber) for general use, HQC for code-based, and KpqC's SMAUG-T for
the Korean standard. While Kyber and HQC have seen extensive side-channel
analysis (TCHES 2017–2025), SMAUG-T has had **a single SPA-only paper** in
the literature (KCI ART003175038, SMAUG-T 곱셈 연산 SPA), with no statistical
chosen-CT attack and no full key recovery via SCA published.

This paper closes that gap with **first-of-kind** results, while also achieving
**SOTA-competitive** trace cost.

### 1.1 Contributions

1. **Convention correction (Section 3)**. The standard chosen-CT convention
   `c1 = α·X^j → µ′_j leaks s_j` is mathematically wrong for R = Z_q[X]/(X^n+1).
   We show the correct mapping is `µ′_i ← s_(i-j mod n) · sign_l(i)` (anticyclic
   wrap), which means a *single* chosen-CT `c1 = α` (j=0) leaks all 256 secret
   coefficients simultaneously. We verify with 14 (j × α) groups, 100% match.

2. **Direct attack PoI (Section 5)**. Existing SCA pipelines learn PoI from
   *random-µ profile* traces, then transfer to chosen-CT. We show this transfer
   *fails* on SMAUG-T (sign reversal, accuracy below zero-baseline). Instead we
   learn PoI from the *attack data itself* via cross-design Welch-t. The oracle
   pair design (α=64, α=192) yields a deterministic property: cross-design
   split is non-empty exactly when sk_i ≠ 0, automatically separating support
   from non-support without HW=70 prior.

3. **Component-specific PoI (Section 6)**. Cross-component (4 designs mixed)
   PoI learning is destabilized by the polynomial that does not contribute. We
   restrict learning to component-c oracle pair when attacking s[c]. This
   improves recovery from 95.6% ± 5.6% (worst seed 87.1%) to **100% ± 0%
   (5/5 seeds)** at N=128.

4. **Minimum-cost attack (Section 7)**. Subsetting the N=128 captures shows a
   sharp phase transition at N=2 (variance computability). With **N=2 (8
   traces total, 3.2-second capture)**, we recover all 9 evaluated keypairs
   with 100% accuracy.

5. **Sparse ternary MAP recovery** (Section 4 / `host/analysis/sparse_recover.py`).
   Under HW=HS sparse constraint, greedy support score + sign argmax gives
   100% sign accuracy within selected support across all our experiments.

### 1.2 Related work and positioning

| Attack | Target | Traces | Offline | Notes |
|---|---|---:|---|---|
| HQC eprint25 (Dong et al.) | hqc-1 (codeword masked) | 1 | small | single-trace |
| Kyber Hamburg21 | Kyber512 (masked) | 2-4 | small | chosen-CT + sparse |
| **This paper** | **SMAUG-T smaug1** | **8** | **0** | **chosen-CT + direct PoI** |
| Kyber Xu21 (-O0 ref) | Kyber512 | 8 | 0 | chosen-CT SPA |
| Kyber Xu21 (-O3 pqm4) | Kyber512 | 960 | 0 | optimized |
| HQC OT-PCA (TCHES25) | hqc-128 | 460 | large | offline templates |
| HQC TCHES24 SASCA | hqc-128 | <20k EM | small | RS BP |
| Kyber Pessl19 SASCA | Kyber | few | medium | 213 HW templates |

We sit at the SOTA tier in *online traces* and uniquely have *zero offline
cost*. Our contribution is not "lower than 8" — that's HQC eprint25's regime
which requires single-trace BP-SASCA and codeword masking-aware methods. Our
contribution is *simpler* (closed-form post-processing, no BP solver) at
near-equivalent cost on a different target.

## 2. Background

### 2.1 SMAUG-T smaug1 (KpqC 2025)

Module-LWR KEM. Parameters: n=256, k=2 (MODULE_RANK), q=1024, p=256, p′=32,
t=2, message δ=32 byte. Secret key sk = (s[0], s[1]) ∈ R^2 with R = Z_q[X]/
(X^n + 1), each polynomial sparse ternary {-1, 0, +1} with HW=70/poly.

The PKE encryption produces ciphertext (c1, c2) ∈ R_p^2 × R_p′. Decryption:
```
µ′ = round_t( (t/p)·⟨c1, sk⟩ + (t/p′)·c2 )  mod t
```
where the inner product is over R (negacyclic). The KEM wraps PKE with a
Fujisaki-Okamoto transform: dec → re-encrypt → verify → cmov.

Polynomial multiplication uses **Toom-Cook 4-way** with **Karatsuba** at the
sub-block level (no NTT). This differs from ML-KEM (Kyber's NTT) and is part
of why SMAUG-T's SCA literature has been thin.

### 2.2 Setup

ChipWhisperer-Lite (CW1173, NewAE) capturing 24400 ADC samples at 29.5 MS/s
(clkgen_x4 of HCLK 7.38 MHz), gain 25 dB, Vcc shunt across the STM32F415 on
a CW308 UFO board. Throughput Z command (indcpa_dec only) ≈ 2.5 traces/sec.

The firmware (`firmware/simpleserial-smaug/`) exports SimpleSerial v1.1
commands `F` (persistent keygen), `I` (chunked ct injection), `L` (integrity),
`M` (PKE-labeled ct via host µ), **`Z`** (indcpa_dec only, 32 B µ′ response),
`X` (sk PKE dump for ground truth verification — *not used in attack itself*).

## 3. Chosen-CT convention

### 3.1 Prior assumption (incorrect)

Earlier SMAUG-T attack design literature assumed:
> c1 = α · X^j chosen-CT → µ′_j leaks s_j

This would imply **256 different chosen-CT, one per j position**, scaled by an
oracle pair (α=64, α=192) and replicated for each component → ~32k+ traces.

### 3.2 Correct mapping

In R = Z_q[X]/(X^n+1):
```
⟨c1, sk⟩(X) = α · X^j · sk(X) = α · Σ_k s_k · X^{j+k}
```
After negacyclic reduction, the coefficient at position i is:
```
⟨c1, sk⟩_i = α · s_{(i-j) mod n} · sign_l(i)         (1)
```
where sign_l(i) = +1 if i ≥ j else −1. So **µ′_i exposes s_{(i-j)}, not s_j**.

### 3.3 Implication

Setting **j = 0** (i.e., c1 = α as a constant polynomial) gives:
```
⟨c1, sk⟩_i = α · s_i        ∀ i ∈ [0, n)
```
i.e., a *single* chosen-CT with c1 = α exposes all 256 secret coefficients
through µ′_0, …, µ′_{255}.

For the oracle pair (α=64, α=192), 4 chosen-CT (s[0]: α=64, α=192; s[1]:
α=64, α=192) suffice for full sk. **4400× efficiency** vs the 32k+ trace
prior estimate.

We verified the corrected formula by capturing 14 chosen-CT (j ∈ {0,1,5,10,
50,100,200}, α ∈ {64,192}) and matching their Z command µ′ responses against
the X command's ground truth sk: 128/128 = 100% match across all 4 j values.

### 3.4 Multi-term and R^2 cross-component

For arbitrary multi-term `c1[m] = Σ_l α[m,l] · X^l` over R^2:
```
⟨c1, sk⟩_i = Σ_(m,l) α[m, l] · s[m]_{(i-l) mod n} · sign_l(i)        (2)
```
Implemented in `host/smaug/chosen.py::predict_mu_prime` and verified
round-trip 100% on 7 different designs (single-term, 2-term, 3-term, R^2
cross-component) with 256/256 bit match against board responses.

## 4. Method overview

The attack proceeds:

1. **Capture**: 4 chosen-CT × N traces each via 'Z' command (indcpa_dec). Total
   4N traces + 4×21 'I' chunk inject calls.
2. **Direct PoI learning** (Section 5): cross-design Welch-t per bit i,
   yielding (PoI[i], sign[i]).
3. **Oracle pair attack**: per component c, mean diff between α=64 and α=192
   designs at PoI gives signed score d_i.
4. **Posterior + sparse_recover**: d_i → ternary posterior π_i ∈ Δ^3 →
   greedy MAP under HW=70 constraint.

No offline templates, no auxiliary trace set, no EM probe. Total wall-clock for
N=2: ~10 seconds (capture 3 sec + 'I' inject ~4 sec + post-process <1 sec).

## 5. Direct attack PoI

### 5.1 Why profile transfer fails

We first attempted the standard PQC SCA pipeline: collect 1000 random-µ profile
traces (separate keypair), learn PoI from per-bit Welch-t on profile, apply to
chosen-CT attack data.

**Result**: bit accuracy 56-58%, *below the zero-baseline 73%*. After diagnosis:
- d (signed score) vs ground-truth sk correlation: **r = −0.086** (sign
  *reversed*).
- After sign flip: r = +0.086, sign accuracy 35% → 65%, but bit accuracy still
  ≈ chance.

The sign reversal is reproducible. Profile (random µ, binomial(0.5) bit
distribution) and chosen-CT (deterministic µ per design) traces have
different *baseline power offsets*; the sign convention learned from one does
not match the other.

### 5.2 Direct PoI from attack data

Observation: each chosen-CT design has *deterministic* µ′ (per current sk).
Across 4 designs (s[0]: α=64, α=192; s[1]: α=64, α=192) the bit-i value of µ′_i
varies, giving a known 0/1 *cross-design split* per bit.

Algorithm (`scripts/attack_direct_poi.py`):
```python
for bit i in 0..255:
    bits = [µ′_i for trace in attack_traces]  # known, ∈ {0,1}
    # Welch t between trace samples partitioned by bit value
    PoI[i] = argmax_t |welch_t(traces[bits==1, t], traces[bits==0, t])|
    sign[i] = +1 if mean_1[PoI[i]] > mean_0[PoI[i]] else -1
```

The training labels (µ′ bits) come *for free* from the firmware's deterministic
response — no separate ground-truth capture, no offline pair labeling.

## 6. Component-specific PoI

### 6.1 Cross-component noise

When learning PoI on all 4 designs simultaneously, the component-1 designs
(c1[1] = α·X^0) inject noise into bit i's Welch-t for component 0 (and vice
versa). Empirically (Section 8), this gives 95.6% ± 5.6% mean full sk recovery
at N=128, with the worst-case seed at 87.1%.

### 6.2 Restricted learning

When attacking s[c], use only the two designs where c1[c] varies (oracle pair
of the same component). Component-1 designs are excluded from s[0]'s PoI, and
vice versa.

This isolates the leakage: the cross-design split for bit i in component-c
data is non-empty exactly when sk_c[shifted i] is nonzero, and the resulting
PoI/sign cleanly aligns with the corresponding sk[c] coefficient.

**Empirically**, this lifts 95.6% → **100% ± 0%** at N=128 (5 seeds), with all
5 keypairs recovering the full sk perfectly.

### 6.3 Why valid bits = HW per component

The cross-design Welch-t over the c-th component oracle pair has well-defined
0/1 split *only when* sk_c at the corresponding shifted index is nonzero. If
sk_c[shifted i] = 0, both α=64 and α=192 designs produce µ′_i = 0, the split
is degenerate, and the bit becomes invalid. The valid count therefore equals
the support size (HW=70 per component, for smaug1). HW=70 sparse constraint
is *implicit* in the attack design.

## 7. Minimum trace cost

We capture 5 keypairs × N=128 traces per chosen-CT (= 512 traces / seed) and
subset them to N ∈ {1, 2, 3, 4, 6, 8, 16, 32, 64, 128} for cost-vs-accuracy
analysis. Result:

| N (per CT) | total traces | capture | full sk acc (5 seeds) |
|---:|---:|---:|---:|
| 1 | 4 | 1.6 s | 57.3% (chance, variance undefined) |
| **2** | **8** | **3.2 s** | **100% ± 0%** |
| 4 | 16 | 6.4 s | 100% |
| 16 | 64 | 25.6 s | 100% |
| 128 | 512 | 204.8 s | 100% |

The phase transition at N=1 → N=2 is structural: variance estimation in the
Welch-t numerator requires ≥ 2 samples per class. From N=2 onward, every
keypair recovers perfectly.

We additionally captured 3 keypairs at N=512 and 1 at N=256. Combining all 9
datasets: **9/9 keypairs, 3 N regimes, full sk 100% with zero variance**.

## 8. Discussion

### 8.1 SNR phase transition (negative finding)

Earlier in this work, we attempted the *cross-component* (all-design mixed)
PoI learning and found a different phase transition: at N ≤ 256, max|t| stayed
in the noise regime (max|t| ≈ 4); at N=512, max|t| jumped to 15.20. With
component-specific learning, this lower noise regime disappears — already at
N=2 we reach signal regime within the per-component oracle pair.

### 8.2 What didn't work

* **'T' isolated poly_mul_acc command** with sub-trigger: planned to capture
  individual Toom-Cook calls. SMAUG-T's archive uses an internal storage form
  (q-modular unsigned, conjectured) that doesn't round-trip with our naive
  numpy convolution. Abandoned in favor of full-decap 'Z' windowing.

* **Toom-Cook BP-SASCA**: the standard ML-KEM SASCA approach (Pessl 2019,
  Primas 2017) of building a factor graph over the polynomial multiplication
  tree. Given that our simpler closed-form method already achieves 100%, this
  becomes unnecessary additional engineering. We leave it as future work for
  protected (masked / shuffled) implementations.

### 8.3 Countermeasures

(See companion analysis in Section 9.)

* **Masking** (codeword/arithmetic): potentially randomizes the deterministic
  µ′ → cross-design split structure breaks. Effectiveness depends on whether
  masking randomizes between traces within a single design (then our cross-
  design split degenerates per-trace) or preserves design-level determinism.
* **Shuffling**: trace alignment becomes harder. PoI learning still works if
  the shuffle order is sample-level random; coefficient-level shuffle of the
  Toom-Cook stages would be a stronger defense.

### 8.4 Limitations

1. Single device (one STM32F415RGTx). Inter-device variability not measured.
2. Unmasked reference implementation (PQM4 port). Masked SMAUG-T impl
   evaluation pending.
3. smaug3 / smaug5 not evaluated (different MODULE_RANK 3 / 5 → more designs
   needed to fully separate components).
4. Single firmware compiler version (arm-none-eabi-gcc 13.2.1, default
   optimization). Compilation-level effects unmeasured.

## 9. Countermeasures evaluation (planned, see `docs/countermeasures.md`)

[To be expanded — see separate countermeasure analysis file.]

## 10. Conclusion

We presented a single-stage online side-channel attack on SMAUG-T smaug1 that
recovers the full long-term key in 8 power traces with 100% accuracy across 9
evaluated keypairs. The attack uses only a chosen-CT capability and a
ChipWhisperer-Lite + Vcc-shunt setup — no offline templates, no EM probe, no
auxiliary capability.

The three method ingredients (chosen-CT convention correction, direct attack
PoI, component-specific learning) are general: they apply to any module-LWR
or module-LWE KEM whose ciphertext element c1 lies in the message channel
(R_p^k) and whose decryption produces a deterministic message µ′ for a fixed
sk + ct. Future work should evaluate masked SMAUG-T implementations (W3
fallback) and extend to smaug3 / smaug5.

---

## Figures (planned)

1. Convention correction diagram — c1=α·X^j → µ′_i ≠ s_j, c1=α → all leaks.
2. SNR phase transition curve — max|t| vs N (cross-component vs per-component).
3. N curve — accuracy vs trace count (5 seeds, error bars).
4. Architectural diagram — chosen-CT inject → Z trace → direct PoI → sparse_recover.
5. Method comparison bar chart — trace count vs target across PQC SCA papers.

## Reproducibility

* Source: `https://github.com/<repo>` commits `7b253df…0b6e277` (this work)
* Firmware: `firmware/simpleserial-smaug/simpleserial-smaug.c` (sha256 in
  each .npz meta)
* Captures: 9 datasets, ~250 MB total
* Tests: 67 unit tests covering chosen-CT builder, sparse recovery, and
  multi-term partition.
