맞아. 지금 결과는 “논문으로 포장할 재료”가 아니라 **다음 공격을 설계하기 위한 진단 데이터**로 봐야 합니다. 그래서 방향을 다시 잡으면, 핵심은 다음입니다.

**단순 selected-lane CPA를 더 크게 반복하는 건 멈추고, NTRU+의 `c·f → INTT → crepmod3` 구조를 이용해서 secret을 더 낮은 차원의 라벨로 바꾸는 실험을 해야 합니다.** 지금까지는 NTT-domain의 ( \hat f_k \in \mathbb Z_q ) 하나를 3457개 후보 중에서 맞히려고 했습니다. 앞으로는 가능하면 coefficient-domain의 (s_i \in {-1,0,1}) 또는 support/sign을 직접 누출시키는 쪽으로 공격면을 바꾸는 게 논리적으로 더 강합니다.

---

## 1. 지금까지의 문제를 다시 정리하면

### 1.1 문제는 “trace가 없다”가 아니라 “라벨이 너무 어렵고, public nuisance가 너무 컸다”

NTRU+ 공식 사이트는 NTRU+가 KpqC PKE/KEM finalist이고 2026-01-30 final version이 공개되었다고 공지합니다. 공식 GitHub의 NTRU+768 reference code를 보면 secret polynomial (f)는 `poly_cbd1 → poly_triple → f[0]+=1 → poly_ntt`로 NTT-domain에 올라가고, secret key에는 `f`와 `hinv`가 `poly_tobytes`로 저장됩니다. decapsulation에서는 `ct`, `f`, `hinv`를 `poly_frombytes`로 읽은 뒤 `poly_basemul(&m1,&c,&f)`, `poly_invntt`, `poly_crepmod3` 순서로 처리합니다. 즉 지금까지 잡은 NTT-domain selected-lane 공격면 자체는 맞았습니다. ([NTRU+][1])

하지만 그 공격면은 후보가 (q=3457)개인 ( \hat f_k )를 맞히는 문제였습니다. 현재 로그에서 Phase 4는 attack-valid positive를 냈지만, Phase 4.5/4.6의 wide single-victim 결과는 candidate-set null과 거의 구분되지 않았고, G=200 확장도 M1/full-stack 0/16 top-100, M5 1/16 top-100에 그쳤습니다. 즉 “더 많은 γ”가 아니라 **더 좋은 라벨과 더 깨끗한 nuisance 제거**가 필요하다는 결론입니다.  

### 1.2 N을 늘리는 건 거의 막혔다

실험 로그의 핵심 진단은 trace model이 대략

[
T[g,n,t] = \alpha[g,t] + b[t]\cdot L_f(g) + \epsilon[g,n,t]
]

처럼 보인다는 점입니다. 여기서 (\alpha[g,t])는 secret-independent γ/public-design baseline이고, (L_f(g))가 우리가 원하는 secret-dependent label입니다. N을 늘리면 (\epsilon)은 줄지만 (\alpha)는 줄지 않습니다. 그래서 N=32, N=64로 가도 corr가 포화되고, G=200에서도 score sharpness가 살아나지 않았습니다. 

### 1.3 PoI/window를 넓히는 것도 막혔다

Secret-referenced oracle PoI는 좋아 보였지만, self-oracle null에서 null candidate도 거의 top-100처럼 보였습니다. 이건 “숨은 leakage가 있었는데 우리가 못 잡았다”가 아니라, **candidate별/window별 best sample 선택이 multiple-comparison bias를 만든다**는 뜻입니다. 따라서 앞으로의 PoI 선택은 공격자-compatible해야 합니다. 즉 independent profiling victim, public timing prior, firmware-cycle marker, 또는 window marginalization이어야지 true secret 기반 PoI는 쓰면 안 됩니다. 

### 1.4 M1/M5는 진짜 별도 채널이지만, naive fusion은 부족하다

M1은 (HW(\gamma \hat f \bmod q)), M5는 Montgomery reduction output 쪽으로 보입니다. 로그상 M5-only top cases가 있고, (f=882)처럼 M1이 못 잡던 large/moderate coordinate가 M5/full-stack에서 강해졌습니다. 이건 중요한 힌트입니다. 다만 M1과 M5를 그냥 z-score로 더하면 null도 같이 커집니다. 앞으로는 M1/M5를 합치는 게 아니라 **각 채널을 별도 likelihood로 유지하고, 어느 채널이 어떤 좌표군을 보는지 모델링해야 합니다.** 

### 1.5 기존 논문과의 충돌 지점도 분명하다

Askeland/Rønjom은 NTRU secret-key unpacking의 강한 HW leakage를 이용해 single-trace로 secret key의 큰 부분을 회수하고 lattice reduction으로 나머지를 찾는 방향입니다. Ravi et al.은 Streamlined NTRU Prime에 대해 plaintext-checking/decryption-failure oracle류의 chosen-ciphertext SCA를 만들고 full key recovery를 보였습니다. Xu/Pemberton/Oswald/Zheng의 CARDIS 논문은 NTRU-HPS/HRSS 쪽 chosen-ciphertext SCA입니다. 그래서 우리가 가야 할 길은 “unpack leakage”도 아니고 “PC/DF oracle”도 아니어야 합니다. ([NIST CSRC][2])

---

## 2. 앞으로의 핵심 공격 가설: time-domain monomial-pair tomography

내가 가장 먼저 밀고 싶은 새 방향은 이것입니다.

지금까지는 `c_ntt`의 한 lane만 켜서

[
m_{1,\mathrm{NTT}}[k] = \gamma \hat f_k
]

를 만들었습니다. 이러면 (\hat f_k)는 (\mathbb Z_{3457}) 원소라 후보가 너무 많습니다.

대신 host에서 time-domain monomial을 만든 뒤 NTT-domain ciphertext로 직렬화합니다.

[
c_{\mathrm{time}}^{(j,\gamma)} = \gamma X^j,\qquad
c_{\mathrm{NTT}}^{(j,\gamma)} = \mathrm{NTT}(\gamma X^j)
]

그러면 board의 decapsulation은 그대로

[
m_{1,\mathrm{NTT}} = c_{\mathrm{NTT}}\odot f_{\mathrm{NTT}}
]

를 계산하고, `poly_invntt(&m1)` 이후에는

[
m_{1,\mathrm{time}}
= \gamma X^j \cdot f_{\mathrm{time}}
]

가 됩니다.

NTRU+ reference code의 key generation 구조상 (f_{\mathrm{time}})는 대략

[
f_i = 3s_i + \delta_{i,0}, \qquad s_i\in{-1,0,1}
]

형태입니다. 즉 monomial ciphertext를 넣으면 `poly_invntt` 직후의 coefficient-domain intermediate가

[
m_{1,t}
= \gamma(3s_{t-j}+\delta_{t-j,0})
]

가 됩니다.

이게 중요합니다. 이제 공격 목표는 (3457)개 후보 중 ( \hat f_k )를 맞히는 게 아니라, 각 coefficient의 (s_i\in{-1,0,1}), support, sign을 맞히는 문제가 됩니다. 라벨 차원이 극적으로 줄어듭니다.

단, `poly_crepmod3`의 출력은 대체로 public하게 collapse될 가능성이 큽니다. 왜냐하면 (f \equiv 1 \pmod 3)에 가까운 구조라서 (c\cdot f \bmod 3)는 secret이 아니라 chosen (c) 쪽에 의해 결정될 수 있기 때문입니다. 따라서 이 공격은 **crepmod3의 출력 oracle**이 아니라, **`poly_invntt` 마지막 단계와 `poly_crepmod3` 입력 처리 중의 analog leakage**를 노립니다. 이 점이 기존 PC/DF oracle 논문과의 차별점입니다.

---

## 3. 이 공격을 더 강하게 만드는 trick: mod-3 invariant γ-pair

단일 (γ)만 쓰면 dense `c_ntt`의 public byte leakage가 매우 커질 수 있습니다. 그래서 γ를 pair로 설계합니다.

[
\gamma^+ \equiv \gamma^- \pmod 3
]

가 되도록 고르면, `crepmod3` 이후의 public message는 두 ciphertext에서 같거나 매우 유사해집니다. 반면 `poly_invntt` 직후의 analog value는

[
\gamma^+ f_{\mathrm{time}}
\quad\text{vs}\quad
\gamma^- f_{\mathrm{time}}
]

로 달라집니다. 따라서 차분 trace

[
\Delta T_{j,p}
==============

## T(c_{\mathrm{time}}=\gamma^+X^j)

T(c_{\mathrm{time}}=\gamma^-X^j)
]

를 보면, post-crepmod3/SOTP/FO 쪽 public behavior는 줄어들고, pre-crepmod3의 secret-dependent amplitude가 남을 가능성이 있습니다.

γ-pair는 단순히 mod 3만 맞추면 안 됩니다. 기존 실패 원인 때문에 다음 조건으로 고르는 게 좋습니다.

[
\gamma^+ \equiv \gamma^- \pmod 3
]

[
HW(\mathrm{pack12}(\mathrm{NTT}(\gamma^+X^j)))
\approx
HW(\mathrm{pack12}(\mathrm{NTT}(\gamma^-X^j)))
]

[
HD(\text{public byte stream}^+,\text{public byte stream}^-)
\text{ small}
]

[
\Delta L(s)
===========

L(\gamma^+(3s+\delta)) - L(\gamma^-(3s+\delta))
\text{ large for }s\in{-1,0,1}
]

즉 public footprint은 비슷하게, secret label contrast는 크게 만드는 γ-pair를 host-only로 먼저 찾습니다.

이건 지금까지의 selected-lane γ expansion과 완전히 다릅니다. G를 늘리는 게 아니라, **public nuisance를 설계 단계에서 상쇄하는 γ-codebook**을 만드는 겁니다.

---

## 4. 이 방법이 창의적으로 강한 이유

가장 큰 장점은 **shift (j)** 를 바꾸면 같은 secret coefficient (s_i)를 서로 다른 시간 위치로 옮길 수 있다는 점입니다.

[
m_{1,t}^{(j)} = \gamma(3s_{t-j}+\delta_{t-j,0})
]

즉 (j)를 여러 개 쓰면, 하나의 secret coefficient (s_i)가 여러 loop position에서 반복 관측됩니다. 지금까지는 특정 lane/slot의 PoI가 운 나쁘면 끝이었지만, monomial-shift 방식은 같은 (s_i)를 여러 timing context에 흩뿌릴 수 있습니다.

이건 사실상 **rotational tomography**입니다.

* coefficient (s_i)를 여러 shift (j)에서 관측한다.
* 각 관측은 다른 time sample / 다른 local leakage 조건을 갖는다.
* posterior를 coefficient별로 합산한다.
* support/sign 확률을 얻는다.

selected-lane 공격은 한 좌표가 운 나쁘면 SNR이 끝났습니다. monomial-shift 공격은 같은 secret을 여러 위치로 이동시켜 SNR을 모을 수 있습니다.

---

## 5. 실험 계획

### Phase A — host-only algebra verification

먼저 trace 없이 산술부터 검증합니다.

1. known sk에서 (f_{\mathrm{time}})을 복원한다.
2. 여러 (j,\gamma)에 대해 (c_{\mathrm{time}}=\gamma X^j)를 만든다.
3. host에서 (c_{\mathrm{NTT}}=\mathrm{NTT}(c_{\mathrm{time}}))를 12-bit pack한다.
4. board/reference와 동일한 연산으로

[
\mathrm{INTT}(c_{\mathrm{NTT}}\odot f_{\mathrm{NTT}})
=====================================================

\gamma X^j f_{\mathrm{time}}
]

가 맞는지 확인한다.
5. `crepmod3(m1)`이 실제로 secret을 대부분 collapse하는지 확인한다.

이 gate를 통과하면 “이 공격은 crepmod3 output이 아니라 pre-crepmod3 analog leakage를 노려야 한다”는 사실도 명확해집니다.

### Phase B — full decap timeline 재매핑

기존 capture는 basemul 앞부분/중간에 집중되어 있었습니다. 이번 공격은 `poly_invntt` 후반과 `poly_crepmod3` 입력 loop가 중요합니다. 따라서 natural `D` trace를 유지하되 sample offset을 바꿔 full decap timeline을 다시 잡아야 합니다.

diagnostic trigger는 localization에만 써도 됩니다. 최종 claim은 여전히 natural `crypto_kem_dec`의 `D` trace 위에서만 합니다. 이 원칙은 기존 계획 문서와도 일치합니다. 

Gate는 단순 TVLA가 아니라 다음입니다.

* (γ^+), (γ^-) pair의 post-crepmod3 이후 trace가 잘 cancel되는가?
* `invntt → crepmod3 input` window에서만 (s_i) label과 corr/AUROC가 뜨는가?
* public-only label, shift-only label, γ-only label로는 같은 성능이 안 나오는가?

### Phase C — small proof-of-concept capture

처음부터 768 전체를 잡지 말고 작게 갑니다.

권장 design:

[
K=8\text{ keys},\quad
J=16\text{ shifts},\quad
P=8\text{ γ-pairs},\quad
N=2\sim4
]

총 trace 수는 (8\times16\times8\times2\times N)입니다. (N=2)면 4096 traces라 충분히 작습니다.

분석 단위는 q=3457 ranking이 아니라 coefficient label입니다.

* support: (1[s_i\ne0])
* sign: (\mathrm{sign}(s_i))
* ternary class: (s_i\in{-1,0,1})

평가 방식:

[
\log P(s_i\mid T)
=================

\sum_{j,p,t(i,j)}
\log P(\Delta T_{j,p,t}\mid s_i)
]

여기서 (t(i,j))는 shift (j)에서 coefficient (s_i)가 처리되는 예상 sample 위치입니다. 이 위치는 true secret으로 고르면 안 되고, firmware-cycle model 또는 public timing prior로 정해야 합니다.

성공 gate는 다음처럼 잡습니다.

* held-out key에서 support AUROC가 null보다 유의하게 높을 것.
* sign accuracy가 support-known subset에서 chance를 넘을 것.
* coefficient-shuffle null을 통과할 것.
* γ-pair sign을 뒤집었을 때 score sign도 뒤집힐 것.
* 같은 (s_i)가 여러 shift (j)에서 일관된 posterior를 줄 것.

이 마지막 조건이 제일 중요합니다. 한 sample에서만 튀면 또 oracle-PoI 문제입니다. 여러 shift에서 같은 coefficient posterior가 누적되면 진짜 leakage입니다.

### Phase D — single-victim scale-up

Phase C에서 coefficient support/sign이 살아나면, single victim으로 확장합니다.

권장 design:

[
K=1,\quad
J=64\text{ shifts},\quad
P=16\text{ γ-pairs},\quad
N=2
]

이렇게 하면 각 coefficient는 여러 shift에서 반복 관측됩니다. 목표는 full key recovery가 아니라 우선 다음입니다.

* (P(s_i\ne0)>0.9)인 coefficient 수
* sign confidence가 높은 coefficient 수
* entropy reduction
* posterior가 실제 (s_i)와 얼마나 calibrated되는지

현재 selected-lane 방식은 top-100 candidate set이 null과 구분되지 않았습니다. 이 방식은 애초에 후보를 (3457)개에서 3개로 줄입니다. 그래서 성공하면 논문급 결과가 훨씬 강해집니다.

---

## 6. 병행해야 할 보조 실험: balanced selected-lane residual CPA

monomial tomography가 메인 승부수라면, selected-lane은 버리는 게 아니라 **balanced γ residual CPA**로 다시 설계해야 합니다.

현재 selected-lane 실패의 핵심은 (\alpha[g])입니다. 그러므로 γ를 독립 design으로 보지 말고 pair/block으로 봅니다.

[
\Delta T_i = T(\gamma_i^+) - T(\gamma_i^-)
]

[
\Delta L_f(i)
=============

## L(\gamma_i^+ \hat f)

L(\gamma_i^- \hat f)
]

γ-pair 조건은 다음입니다.

* same or near-same byte-HW
* same or near-same byte-HD from zero
* same slot packing footprint
* large (\Delta L_f) variance over candidate (f)
* low pairwise candidate-label correlation

이건 기존 G=200과 다릅니다. G=200은 후보 label 수를 늘렸지만 public nuisance도 같이 늘었습니다. balanced design은 nuisance를 줄이는 쪽입니다.

실험은 작게 하면 됩니다.

[
K=4,\quad L=4\text{ calibrated lanes},\quad P=64\text{ pairs},\quad N=2
]

성공 gate는 top-100 count가 아니라 다음입니다.

* residual (\sigma_\alpha)가 raw 대비 감소
* true-rank distribution이 candidate-set null 밖으로 이동
* M1/M5 중 하나라도 top-10 pressure 증가
* public-design classifier가 pair sign을 못 맞힘

---

## 7. M5는 “보조 채널”이 아니라 micro-operation 모델로 다시 파야 한다

M5는 우연이 아니라고 봅니다. 기존 로그에서 M5-only recovery가 있었고, (f=882), (f=1617) 같은 large/moderate coordinate가 M5에서 강해졌습니다. 문제는 M5를 단순히 `HW(montgomery_reduce(γf))` 하나로 둔 겁니다. 

다음에는 Montgomery reduction을 여러 micro-label로 쪼개야 합니다.

예상 label family:

[
HW(t),\quad t=\gamma f
]

[
HW(u),\quad u=(t\cdot QINV)\bmod 2^{16}
]

[
HW(uq),\quad HW(t+uq)
]

[
carry(t+uq),\quad MSB((t+uq)\gg16)
]

[
HD(\text{previous register},\text{reduction output})
]

그리고 score는 이렇게 둡니다.

[
P(f\mid T)
\propto
P_{M1}(T\mid f)^{w_1}
P_{M5a}(T\mid f)^{w_2}
P_{M5b}(T\mid f)^{w_3}
\cdots
]

가중치 (w)는 attack victim에서 맞추면 안 됩니다. profiling victims 또는 public-only calibration으로 고정해야 합니다. 이걸 하지 않으면 또 PoI/score selection bias가 됩니다.

---

## 8. two-lane nonlinear differential도 준비할 만하다

monomial tomography가 pre-crepmod3 input을 노린다면, 두 번째 창의적 방향은 nonlinear interaction입니다.

두 chosen ciphertext (a,b)를 만들고 네 trace를 찍습니다.

[
T(a+b),\quad T(a),\quad T(b),\quad T(0)
]

그리고

[
I_{a,b}
=======

T(a+b)-T(a)-T(b)+T(0)
]

를 봅니다.

basemul과 public load가 거의 additive라면 이 interaction에서 많이 사라집니다. 반대로 `centered reduction`, `crepmod3`, `SOTP decode` 같은 nonlinear 처리에서만 남는 신호가 생길 수 있습니다.

이 방향은 Ravi/Xu류 oracle 공격과 조심스럽게 구분해야 합니다. 우리는 fail bit, mismatch flag, plaintext-checking oracle을 쓰지 않습니다. 목표는 **analog interaction leakage**입니다. 성공하면 pairwise constraint가 생깁니다.

[
\phi(s_i,s_j)
]

형태의 factor를 만들 수 있고, 이를 factor graph / belief propagation으로 묶을 수 있습니다. 다만 이건 monomial-pair support/sign leakage가 어느 정도 잡힌 뒤에 하는 게 맞습니다. 처음부터 하면 변수와 null이 너무 커집니다.

---

## 9. 실험 우선순위

내가 정하면 순서는 이렇습니다.

1. **Host-only monomial identity 검증**
   (c_{\mathrm{time}}=\gamma X^j)가 실제로 (m_1=\gamma X^j f)를 만드는지 확인합니다. 이건 하루 안에 끝낼 수 있고, 실패하면 방향을 즉시 접을 수 있습니다.

2. **`invntt → crepmod3 input` timeline 찾기**
   기존 basemul window가 아니라 후반 window를 찾아야 합니다. final claim은 natural `D` trace로 하되, diagnostic marker는 위치 찾기에만 씁니다.

3. **mod-3 invariant γ-pair codebook 생성**
   γ-pair를 host-only로 최적화합니다. 조건은 same mod3, matched public byte footprint, high ternary-label contrast입니다.

4. **K=8, J=16, P=8, N=2 small proof**
   q-residue ranking이 아니라 support/sign AUROC를 봅니다. 이게 살아나면 selected-lane보다 훨씬 강한 길입니다.

5. **성공 시 K=1 single-victim tomography**
   J=64, P=16 정도로 한 victim의 (s_i) posterior를 누적합니다.

6. **실패 시 balanced selected-lane residual CPA로 복귀**
   단순 G 확장이 아니라 pair-difference로 (\alpha[g])를 없애는 실험을 합니다.

7. **M5 micro-label model은 병행 개발**
   이건 monomial/selected-lane 양쪽에 모두 쓸 수 있습니다.

---

## 10. 기존 논문과 충돌하지 않게 claim을 잡는 법

논문 claim은 이렇게 가야 합니다.

* Askeland류와 다르게: secret-key unpacking을 공격하지 않는다.
* Ravi류와 다르게: plaintext-checking/decryption-failure oracle을 만들지 않는다.
* Xu류와 다르게: NTRU-HPS/HRSS polynomial modular reduction oracle이 아니라 NTRU+ final implementation의 NTT-domain chosen ciphertext와 pre-crepmod3 analog leakage를 쓴다.
* 기존 selected-lane 결과와 다르게: ( \hat f_k\in\mathbb Z_q ) residue ranking이 아니라 (s_i\in{-1,0,1}) coefficient-domain tomography를 목표로 한다.

가장 좋은 논문 방향은 다음입니다.

> **From NTT Lanes to Ternary Secrets: Chosen-Ciphertext Analog Tomography of NTRU+ Decapsulation**

이 제목의 핵심은 “복구 오라클”이 아니라 “analog tomography”입니다.

---

## 11. 최종 판단

지금까지의 결과가 말해주는 건 명확합니다.

**selected-lane basemul CPA는 진짜 leakage를 봤지만, 그 라벨은 너무 어렵고 public nuisance가 너무 컸습니다.** 그래서 top-100 후보가 생겨도 candidate-set null을 못 깼습니다.

앞으로는 공격면을 바꿔야 합니다. 가장 논리적인 다음 공격은:

[
c_{\mathrm{time}}=\gamma X^j
\Rightarrow
m_{1,\mathrm{time}}=\gamma X^j f_{\mathrm{time}}
]

를 이용하는 **mod-3 invariant monomial-pair tomography**입니다.

이 방법은 NTRU+의 final reference 구조와 맞고, 기존 NTRU SCA 논문들과도 직접 충돌하지 않으며, 지금까지 실패 원인인 (q)-후보 ranking, (\alpha[g]), PoI bias를 동시에 줄일 수 있습니다. 지금은 이 실험을 1순위로 설계하는 게 맞습니다.

[1]: https://www.ntruplus.org/ "NTRU+"
[2]: https://csrc.nist.gov/CSRC/media/Events/third-pqc-standardization-conference/documents/accepted-papers/askeland-sid-channel-pqc2021.pdf "A Side-Channel Assisted Attack on NTRU"
