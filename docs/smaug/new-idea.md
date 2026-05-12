2026-05-05에 업데이트된 `idea.md` 기준으로, 지금 필요한 전환은 **“현재 결과를 논문으로 포장”이 아니라 “attack-valid한 자연 `D=crypto_kem_dec` 인터페이스에서 비밀 의존 내부값을 관측 가능한 oracle로 바꾸는 것”**입니다. 지금까지의 결과는 실패라기보다, 어떤 종류의 leakage가 논문 공격으로 이어지지 않는지를 꽤 선명하게 잘라냈습니다.  

가장 추천하는 새 방향은 다음입니다.

> **Public-label self-calibrated `D` oracle**
> `c1=0`인 chosen ciphertext로 자연 `crypto_kem_dec` 안의 `μ′`를 공격자가 공개적으로 알 수 있게 만든 뒤, 같은 타깃 장치에서 `μ′` byte/FO-downstream leakage를 먼저 self-calibration하고, 그 모델을 `c1≠0` threshold ciphertext에 적용해 secret-dependent `μ′`를 복구한다.

이건 기존에 했던 “`R`/`Z` diagnostic에서 `μ′` label을 쓰는 것”과 다릅니다. `μ′`를 장치가 반환해서 쓰는 게 아니라, **공격자가 만든 `c1=0,c2` 때문에 내부 `μ′`가 수학적으로 공개값이 되는 자연 `D` trace**를 calibration에 쓰는 겁니다. 따라서 target secret dump도, diagnostic response도 필요 없습니다.

---

## 1. 지금까지의 문제를 다시 정리하면

### 첫 번째 문제: v3.0식 곱셈 SPA 표면이 v4.0에서 사라졌다

기존 SMAUG-T 공격 논문은 sparse ternary 비밀키에 맞춘 다항식 곱셈 구조가 constant-time이 아니고, 비밀키 계수 `1/0/-1`에 따른 전력 차분으로 단일 파형에서 각 계수 class의 개수를 복구하는 흐름이었습니다. ([KCI][1])

그런데 v4.0 changelog는 이 문제를 직접 의식한 변경을 담고 있습니다. v4.0은 Encap/Decap에서 fixed-weight sampler 사용을 피하고, secret/ephemeral distribution과 Hamming weight를 바꾸며, 특히 polynomial multiplication에서 secret index representation 대신 coefficient representation을 쓰도록 바꿨다고 설명합니다. 즉, 기존 v3.0 논문과 같은 “sparse index/control-flow SPA”를 그대로 반복하면 논문적으로도, 실험적으로도 충돌합니다. 

따라서 너의 새 논문은 **“v3.0 곱셈 SPA의 반복”이 아니라 “v4.0 자연 decapsulation에서 chosen ciphertext가 만드는 secret-dependent anchor variable leakage”** 쪽으로 가야 합니다.

### 두 번째 문제: diagnostic leakage는 분명 있는데, attack-valid leakage로 전이되지 않았다

`T`의 isolated `poly_mul_acc`에서는 coefficient-HW CPA와 Bayes recovery가 분명히 보였습니다. 하지만 이건 diagnostic wrapper라서 공격 claim으로 쓸 수 없습니다. `Z`, `V`, `W`, `U`로 옮기면서 Toom/Karatsuba label, low-dimensional label, random multi-term design, leakage-aware design selection, residualization 등을 시도했지만 대부분 permutation null을 넘지 못했습니다. 특히 `Z`/`V`/`W`에서는 “trace에는 key/session variation이 있는데, 그게 secret coefficient label로 안정적으로 매핑되지 않는” 문제가 반복되었습니다.  

이건 중요한 결론입니다. 앞으로는 더 많은 random multi-term `c1`이나 더 큰 ridge sweep을 돌리는 게 아니라, **내부값을 먼저 관측 가능한 oracle로 만드는 calibration 구조**가 필요합니다.

### 세 번째 문제: `D`-pair threshold tomography는 좋은 아이디어였지만, 구현상 raw pair-difference oracle은 실패했다

`D` branch에서 가장 중요한 교훈은 두 가지입니다. 첫째, 자연 공격 claim은 결국 `D=crypto_kem_dec`에서 나와야 합니다. `Z/R/Q/Y/V/W/T/U`는 위치 찾기와 모델 디버깅용 diagnostic입니다. 둘째, `flip_any`는 secret oracle이 아니라 public `α` classifier로 오염될 수 있습니다. 실제로 monomial `c1=αX^j`, `c2=(15,16)`에서 `flip_any`는 secret보다 public `α`에 의해 결정되는 구조가 생겼습니다.  

그리고 더 강한 결론도 있습니다. 자연 `D`에서 early/late window, staircase c2 pairs, random multi-term `c1`을 모두 해도 secret-dependent regression corr z가 gate를 넘지 않았고, paired TVLA도 대체로 max |t|≈3.7–4.6 수준이었습니다. 즉, **현재 CW-Lite 환경에서 raw pair subtraction으로 `μ′` 차이를 직접 보는 방식은 막혔다**고 보는 게 맞습니다.  

하지만 이게 `μ′` 기반 공격 전체의 실패는 아닙니다. 실패한 건 **“비지도 pair 차분으로 바로 secret label을 regression하는 방식”**입니다.

---

## 2. 가장 논리적인 새 공격: public-label self-calibrated `D` oracle

SMAUG-T PKE decryption은 본질적으로

[
\mu'=\left\lfloor {t\over p}\langle c_1,s\rangle+{t\over p'}c_2\right\rceil \in R_t
]

형태입니다. 이 식 자체가 공격 표면입니다. 공격자가 `c1`과 `c2`를 고를 수 있고, `c1=0`이면 내부 `μ′`는 secret과 무관하게 `c2`만으로 정해집니다. 공식 specification의 PKE.Dec도 이 구조를 명시합니다. 

여기서 새 아이디어는 간단합니다.

1. **Calibration phase:** `c1=0`으로 자연 `D`를 호출한다. 이때 `μ′`는 공개적으로 계산 가능하다.
2. 같은 타깃 장치에서 `μ′` byte, prefix-HW staircase, 또는 FO downstream trace model을 학습한다.
3. **Attack phase:** `c1≠0` threshold ciphertext를 보낸다. 이때 `μ′`는 secret-dependent다.
4. calibration한 모델로 attack trace의 `μ′` 또는 `μ′`-derived downstream state를 추정한다.
5. 여러 `j,α,c2` query의 likelihood를 합쳐 secret coefficient의 zero/nonzero, sign, byte-window count, 또는 key-rank를 줄인다.

이 흐름은 기존 실패 원인 세 가지를 동시에 피합니다.

첫째, `D`만 쓰므로 attack-valid입니다. SMAUG-T KEM Decap은 PKE.Dec로 `μ′`를 만들고, `G(μ′,H(pk))`, re-encryption, ciphertext comparison, fallback key selection을 수행합니다. 즉 자연 `crypto_kem_dec` trace 안에 `μ′`와 그 downstream effect가 실제로 존재합니다. 

둘째, target key transfer 문제를 피합니다. 지금까지는 cross-key profiling이나 diagnostic-to-natural transfer가 깨졌습니다. 새 방법은 같은 타깃 장치에서 `c1=0` 공개-label traces로 calibration하므로, key/session identity, probe 위치, amplitude, board drift를 모두 target-specific하게 흡수할 수 있습니다.

셋째, `flip_any α-bug`를 피합니다. binary “flip 여부”를 쓰지 않고, `μ′` byte value, byte HW, prefix-HW sequence, 또는 multi-threshold response curve 전체를 씁니다.

---

## 3. 실험 1순위: `c1=0` public-μ calibration이 자연 `D`에서 가능한지 확인

제일 먼저 해야 할 실험은 secret attack이 아닙니다. **자연 `D`에서 공개적으로 아는 `μ′`를 분류할 수 있는지** 확인해야 합니다.

### 실험 설계

`c1=0`으로 두고, `c2`를 조작해서 원하는 `μ′` bit/byte pattern을 만듭니다. 예를 들어 한 byte만 `0x00, 0x01, 0x02, …, 0xff`로 바꾸고 나머지 byte는 고정하는 식입니다. 이 ciphertext는 보통 KEM-valid ciphertext가 아니어도 됩니다. 중요한 것은 `D=crypto_kem_dec`가 자연 경로로 실행되고, 내부 `μ′`가 `c2`만으로 공개 계산 가능하다는 점입니다.

이 calibration dataset으로 다음 세 모델을 따로 봐야 합니다.

[
P(\mu'_b=v \mid \tau_D)
]

[
P(HW(\mu'_b)=h \mid \tau_D)
]

[
P(\text{prefix-HW staircase of byte }b \mid \tau_D)
]

여기서 세 번째가 특히 중요합니다. 최근 SMAUG-T polynomial-to-message conversion leakage 논문은 repeated `STRB` store 때문에 cumulative Hamming-weight staircase signature가 생기며, STM32F4 조건에서 TVLA, SNR, CPA가 매우 크게 나온다고 보고했습니다. ([Springer][2]) 너의 diagnostic `R` positive control도 이 방향과 일치합니다. 

다만 그 논문과 충돌하지 않으려면 claim을 다르게 잡아야 합니다. 그 논문은 polynomial-to-message conversion 자체의 leakage assessment와 byte classification이 핵심입니다. 너의 새 claim은 **“그 leakage를 public-label calibration과 chosen-ciphertext threshold spectroscopy로 secret information leakage에 연결한다”**가 되어야 합니다.

### 성공 기준

`c1=0` public-μ calibration에서 먼저 다음을 gate로 둡니다.

[
\text{held-out byte accuracy} > 70%\quad \text{or}\quad
\text{HW rMAE/corr z} > +5
]

이게 자연 `D`에서 안 되면, 지금 CW-Lite 파형으로는 `μ′` oracle 자체가 어렵다는 뜻입니다. 그 경우 더 많은 secret attack trace를 모으는 건 의미가 작고, EM probe나 더 큰 bandwidth로 가는 게 맞습니다.

반대로 이게 되면, 드디어 attack-valid path가 생깁니다.

---

## 4. 실험 2순위: pair가 아니라 “threshold spectroscopy”로 secret을 본다

기존 `D`-pair는 `c2=(7,8)`, `(15,16)`, `(23,24)` 같은 인접 pair의 trace 차이를 보려고 했습니다. 그런데 pair subtraction은 SNR이 너무 낮고, `flip_any`는 public classifier로 오염될 수 있었습니다. 

새 방식은 pair가 아니라 **여러 `c2` 값을 훑는 response curve**를 봅니다.

고정된 monomial `c1=αX^j`에 대해 각 coefficient 위치의 내부 bit는

[
\mu'*i(c_2)=
\operatorname{round}\left(
{t\over p}\alpha\cdot s*{i-j}+{t\over p'}c_{2,i}
\right)
]

입니다. `s∈{-1,0,+1}`이면 `c2` threshold curve의 위치가 달라집니다. 따라서 `c2`를 한두 쌍만 보지 말고, 작은 grid 전체로 훑으면 각 secret class가 서로 다른 codeword를 가집니다.

예를 들어 현재 실험에서 `α=64`일 때 `(7,8)`은 `s=0` detector, `(15,16)`은 nonzero detector, `(23,24)`은 alternate zero detector로 작동했습니다. 이걸 pair별 regression으로 보지 말고, `c2∈{0,1,…}`의 전체 response vector로 보면 훨씬 강한 redundancy가 생깁니다. 

### attack inference

공격 trace 하나에서 byte classifier가 완벽할 필요는 없습니다. 각 query가 주는 soft likelihood만 있으면 됩니다.

[
\mathcal{L}(s_i=a)
==================

\sum_{c_2\in G}
\log P\left(
\widehat{\mu'}_{i}(c_2)
=======================

\mu'_i(c_2; s_i=a)
\mid \tau_D(c_2)
\right)
]

이걸 모든 shift `j`, 모든 coefficient, 여러 `α`에 대해 누적하고, 마지막에는 SMAUG-T sparse ternary prior를 넣습니다.

[
s^\star
=======

\arg\max_{s\in{-1,0,+1}^{256},,|s|*0=70}
\sum*{i,a} \mathcal{L}_i(a)
]

full key recovery가 안 되어도 논문 claim은 충분히 만들 수 있습니다.

* byte-window마다 `#0`, `#+`, `#-` count leakage
* support entropy reduction
* sign entropy reduction
* true key rank reduction
* sparse candidate set reduction
* chosen ciphertext 수에 따른 leakage accumulation curve

이건 기존 v3.0의 “single trace로 다항식 내 계수 class 개수 복구”와 다릅니다. v4.0 자연 `D`에서 public-label calibrated `μ′` oracle을 만들고, chosen `c1,c2` spectroscopy로 secret posterior를 누적하는 방식입니다.

---

## 5. 실험 3순위: FO downstream을 “hash amplifier”로 쓰기

`μ′` byte-store가 자연 `D`에서 약하면, 다음은 `μ′` 자체가 아니라 FO downstream을 봐야 합니다.

SMAUG-T KEM Decap은 `μ′`를 얻은 뒤 `G(μ′,H(pk))`로 `K′,seed′`를 만들고, `seed′`로 re-encryption을 한 뒤 `ct`와 `ct′`를 비교합니다.  이건 한 bit의 `μ′` 차이가 SHAKE/G output과 re-encryption trace 전체를 pseudo-random하게 바꾸는 **amplifier**입니다.

기존 Branch Y는 `fo_kr1_byte_hw` 같은 단순 label로 보았고 sub-gate였습니다.  하지만 새로 해야 할 건 scalar label regression이 아니라 **candidate likelihood classification**입니다.

### 구체적인 flow

1. `c1=0` public-label traces로 여러 known `μ′`에 대한 natural `D` late-window trace를 모읍니다.
2. 각 `μ′`에 대해 `G(μ′,H(pk))`, `seed′`, re-encryption `ct′`, verify input `ct⊕ct′`를 모두 공개 계산합니다.
3. late-D trace를 다음 후보별로 scoring합니다.

[
\log P(\tau_{\text{late-D}} \mid \mu'=\mu_a)
]

4. attack ciphertext에서는 한 coefficient guess `s_i∈{-1,0,+1}`가 만들 수 있는 `μ′` 후보가 작습니다. 세 후보의 late-D likelihood를 비교하면 secret class likelihood가 됩니다.

이 방식의 장점은 한 bit/byte의 `μ′` 차이가 downstream에서 수많은 load/store, SHAKE state, sampler, re-encryption multiplication, verify/cmov state 차이로 확산된다는 점입니다. chosen-ciphertext SCA 문헌에서도 공격자가 ciphertext를 구성해 decryption의 anchor variable을 소수 후보로 제한하고, side-channel로 그 값을 분류한 뒤 key recovery로 연결하는 구조가 핵심입니다. ([NIST CSRC][3])

SMAUG-T에서는 이 anchor variable이 `μ′`이고, novelty는 **`c1=0` natural-D public-label calibration + `c1≠0` threshold attack**입니다.

---

## 6. 실험 4순위: verify/cmov distance oracle

FO downstream 안에서도 특히 볼 만한 곳은 `ct`와 `ct′`의 comparison입니다. 설령 comparison이 constant-time이어도, 구현은 보통 `diff |= ct[i] ^ ct_prime[i]` 같은 누적 상태를 갖습니다. 이 누적 OR, XOR byte, Hamming distance, memory access state가 전력으로 보이면 decryption-failure oracle 또는 distance oracle이 됩니다.

이건 기존 Kyber/Frodo류 chosen-ciphertext SCA에서 자주 쓰이는 패턴과도 맞습니다. lattice-based KEM의 chosen-ciphertext SCA는 decrypted message, failure, comparison state 같은 oracle을 만들고 key recovery로 이어지는 구조를 반복적으로 사용해 왔습니다. ([NIST CSRC][3])

SMAUG-T에서는 다음 label을 써야 합니다.

[
HD_j(\mu')
==========

HW\left(
ct[j:j+w]\oplus ct'(\mu')[j:j+w]
\right)
]

또는 prefix 형태로

[
D_j(\mu')
=========

\bigvee_{r\le j}
\left(
ct[r]\oplus ct'(\mu')[r]
\right)
]

입니다.

이것도 `c1=0`이면 `μ′`가 공개값이므로 target-specific calibration이 가능합니다. 이후 attack ciphertext에서는 `s_i=-1,0,+1` 후보별 `ct′(μ'_a)`를 계산하고 late-D verify/cmov trace likelihood를 비교합니다.

이 방향은 raw `μ′` byte oracle보다 더 “hash-amplified”입니다. `μ′` 한 bit 차이가 `seed′`를 바꾸고, `ct′` 전체를 바꿀 수 있기 때문입니다. 단, public `ct` 자체 leakage와 verify loop 위치를 잘 분리해야 합니다.

---

## 7. 순수 곱셈 쪽을 계속한다면: high-level Toom label을 버리고 instruction-local로 내려가야 한다

곱셈 leakage를 완전히 버릴 필요는 없습니다. 다만 지금까지의 결과는 `toom*_conv16_hw`, `kara*_conv8_hw`, `prod64_hw`, `vdelta64_hw` 같은 grouped label이 너무 coarse하다는 걸 말합니다. isolated `U`에서는 약한 Karatsuba 관련 신호가 보였지만 `V/W/Z`로 전이되지 않았고, 실제 `vec_vec_mult_add`에서는 unshifted public input, product shift, add context가 달라졌습니다.  

따라서 이 branch를 살리려면 다음처럼 바꿔야 합니다.

* `a >> mod`
* component별 `poly_mul_acc`
* `product << mod`
* `poly_add`
* final rounding / pack

각각을 **한 coefficient 또는 한 32-bit load/store 단위**로 label해야 합니다. whole-vector HW/HD가 아니라 register transition입니다.

예를 들어:

[
HD(R_{\text{before add}}, R_{\text{after add}})
]

[
HW((c2_i\ll k)+(tmp_i\ll \ell))
]

[
HD(\text{store old halfword},\text{store new halfword})
]

이걸 `W2` 같은 diagnostic trigger로 한 loop iteration만 고립해서 leakage template을 만든 뒤, 자연 `D`에서 같은 template이 어디에 나타나는지 찾아야 합니다. 하지만 이 branch의 우선순위는 `c1=0` public-μ self-calibration보다 낮습니다. 지금까지의 결과상 곱셈 쪽은 “leakage는 있지만 secret oracle로 정렬되지 않는다”는 문제가 강합니다.

---

## 8. fault-assisted branch는 별도 논문으로는 매우 강할 수 있다

순수 SCA가 아니라 FI/SCA hybrid가 허용된다면, 가장 강한 공격은 FO verify/cmov를 fault로 우회하는 것입니다.

자연 Decap은 invalid ciphertext에서 `ct≠ct′`이면 fallback key를 반환합니다.  만약 fault로 comparison 또는 conditional move를 우회해서 invalid ciphertext에도 `K′=G(μ′,H(pk))`가 반환되게 만들 수 있으면, chosen threshold ciphertext에서 `μ′` 후보가 소수이므로 attacker는 후보 `μ′`별 `G(μ′,H(pk))`를 계산해 반환 key와 비교할 수 있습니다.

이건 side-channel 논문이라기보다 **fault-assisted plaintext-checking oracle** 논문입니다. 기존 v3.0 곱셈 SPA와는 충돌하지 않습니다. 다만 범위가 달라지므로, 지금 논문 목표가 “power SCA only”라면 이 branch는 별도 트랙으로 분리하는 게 좋습니다.

---

## 9. 당장 하지 말아야 할 것

더 이상 같은 방식으로 scale하지 않는 게 좋습니다.

`17731:19779` window를 더 키우는 것, random multi-term `c1`을 더 많이 모으는 것, `U`-derived shifted-public Karatsuba label을 다시 키우는 것, `flip_any` AUROC를 다시 보는 것, residualized ridge를 계속 튜닝하는 것은 우선순위가 낮습니다. `idea.md`의 현재 정리도 이 방향을 거의 retire하고 있습니다.  

특히 `flip_any`는 다시 쓰면 안 됩니다. 쓰더라도 반드시 public-α classifier control을 먼저 통과해야 합니다. secret-honest metric은 `μ′` byte value, signed `mu_delta_byte_hw`, threshold response curve likelihood, 또는 final secret entropy reduction이어야 합니다.

---

## 10. 추천 실험 순서

제가 권하는 순서는 명확합니다.

### Phase A — 자연 `D` public-label μ oracle 만들기

`c1=0`, controlled `c2`로 known `μ′` codebook을 만들고, natural `D` trace에서 byte/HW/prefix-HW classifier를 학습합니다. 이때 diagnostic `R`은 window 찾기용으로만 쓰고, 최종 성능은 반드시 `D`에서 봅니다.

성공 기준은 held-out public-μ byte/HW classification이 강하게 나오는 것입니다. 이게 안 되면 현재 CW-Lite setup에서는 `μ′` oracle이 어렵습니다.

### Phase B — threshold spectroscopy로 secret leakage 만들기

`c1=αX^j`, 여러 `c2` threshold grid를 사용해 `s_i∈{-1,0,+1}`별 response curve likelihood를 계산합니다. pair subtraction이 아니라 calibrated byte classifier의 posterior를 씁니다.

목표 claim은 처음부터 full key recovery가 아니라 다음으로 잡습니다.

[
(n_0,n_+,n_-)\text{ per byte window}
]

또는

[
H(S)-H(S\mid \text{traces})
]

즉, secret entropy 감소를 보입니다.

### Phase C — FO hash-amplified likelihood

Phase B의 `μ′` oracle이 약하면, 같은 public-label calibration dataset으로 late-D FO/re-encryption/verify template을 학습합니다. 후보 `μ′`별 re-encryption trace likelihood를 비교해 coefficient posterior를 만듭니다.

### Phase D — instruction-local multiplication only if A–C fail

곱셈으로 돌아가야 한다면, high-level Toom/Karatsuba grouped labels를 버리고 disassembly-level one-loop register/memory transition label로 내려갑니다. 이건 “v4.0 곱셈 leakage” 논문으로는 가능하지만, attack-valid secret recovery까지는 A–C보다 가능성이 낮습니다.

---

## 11. 논문 novelty를 이렇게 잡으면 기존 논문과 충돌하지 않는다

기존 2025년 SMAUG-T SPA 논문은 v3.0 계열의 sparse ternary 곱셈 구현이 constant-time이 아니고, 단일 파형으로 비밀키 계수 class 개수를 복구하는 공격입니다. ([KCI][1])

2026년 SMAUG-T polynomial-to-message conversion leakage 논문은 repeated `STRB` 기반 cumulative HW staircase와 byte classification 취약점을 보입니다. ([Springer][2])

너의 새 논문은 둘과 다르게 잡을 수 있습니다.

> **SMAUG-T v4.0 natural decapsulation에서, 공개-label chosen ciphertext calibration을 이용해 `μ′`/FO downstream leakage oracle을 만들고, threshold spectroscopy로 sparse ternary secret의 zero/nonzero/sign 정보를 누출한다.**

이 claim은 v3.0 곱셈 SPA도 아니고, 단순 polynomial-to-message leakage assessment도 아닙니다. chosen-ciphertext SCA 문헌의 “anchor variable을 소수 후보로 제한하고 side-channel로 분류해 key recovery로 연결한다”는 큰 틀과는 맞지만, SMAUG-T v4.0의 `c1=0` public-label calibration과 threshold response spectroscopy는 충분히 독자적인 공격 flow가 될 수 있습니다. ([Eprint][4])

가장 먼저 할 실험은 하나입니다.

**자연 `D`에서 `c1=0,c2` public-label traces만으로 target-specific `μ′` byte/HW/staircase oracle을 만들 수 있는지 확인하세요.**
이게 성공하면 secret leakage 논문의 길이 열립니다. 이게 실패하면, 현재 setup에서는 더 많은 D-pair나 random multi-term capture가 아니라 측정 장비/EM probe/late-D localization을 바꾸는 게 맞습니다.

[1]: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003175038 "SMAUG-T 곱셈 연산에 대한 단순전력분석 공격 및 대응기법"
[2]: https://link.springer.com/chapter/10.1007/978-981-95-8034-7_11 "Side-Channel Leakage Assessment of SMAUG-T: Exploiting Hamming Weight Patterns in Polynomial-to-Message Conversion | Springer Nature Link"
[3]: https://csrc.nist.gov/CSRC/media/Events/third-pqc-standardization-conference/documents/accepted-papers/ravi-generic-side-channel-pqc2021.pdf "On Generic Side-Channel Assisted Chosen Ciphertext Attacks on Lattice-based PKE/KEMs Towards key recovery attacks on NTRU-based PKE/KEMs"
[4]: https://eprint.iacr.org/2020/912 "Magnifying Side-Channel Leakage of Lattice-Based Cryptosystems with Chosen Ciphertexts: The Case Study of Kyber"
