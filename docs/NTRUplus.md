# NTRU+ SCA — 공격 계획

마지막 업데이트: 2026-05-11.

이 문서는 NTRU+ (KpqC final, 공식 GitHub release v1.0 / 2026-01-30) 에 대한
chosen-CT 사이드채널 공격의 **계획 단일 진실의 원천**이다. 진행 중 실험의
가정·과정·결과 분석은 `docs/IDEA.md` 에 누적한다. 본 문서는 전략·gate·산출물
규약만 담는다.

현재 상태 요약 (2026-05-11): Phase 4 NTT-domain selected-lane CPA 는
attack-valid positive 결과를 확보했다. Multi-victim 결과는 240 cases 에서
2 TOP-1 및 M1 baseline ∪ full-stack 기준 17 top-100 이고, single-victim
multi-lane 확장은 3 victims / 88 lanes / 352 cases 에서 M1 baseline 9 top-100,
M5 baseline 12 top-100, full stack 8 top-100 이다. 최신 수치는
`docs/HANDOFF.md` 와 `results/ntruplus768/phase45/SUMMARY.md` 를 우선한다.
후속 G=200 compact scout 는 clean capture 였지만 M1/full-stack 0/16 top-100,
M5 1/16 top-100 에 그쳐, 단순 γ design 확장만으로는 single-victim score
sharpness 문제가 해결되지 않는 것으로 정리한다.

SMAUG-T 단계의 교훈을 전제로, **natural `crypto_kem_dec` trace 에서만 main
claim 을 만든다**. diagnostic sub-trigger / target dump / mismatch flag 는
라벨링·튜닝에만 사용한다.

---

## 0. 핵심 결정 (one-liner)

```
sk-unpack triage → NTT-domain selected-lane oracle (main)
                 → coefficient threshold + SOTP/FO⊥ validity (fallback)
                 → all claims on natural `crypto_kem_dec` trace only
```

이렇게 잡으면 기존 NTRU SCA (Ravi/Xu/Askeland) 와 정면 충돌하지 않고
**NTRU+ final implementation 의 NTT-friendly 구조** 를 활용한 독립 공격이 된다.

## 1. 위협 모델

- 대상: NTRU+ final, **level ntruplus768** (smallest official final level).
  - 결정 근거 (2026-05-08): 공식 KpqC final 은 768/864/1152 만. 로컬 archive 의
    576 은 final 이 아닌 변종. 768 은 (a) 공식 final 의 가장 작은 보안수준,
    (b) upstream reference C 와 zetas[192] binary-identical 한 archive 보유.
  - hex: `simpleserial-ntruplus-CW308_STM32F4-ntruplus768.hex`.
  - 실패 시 fallback: 864/1152.
- 보드: ChipWhisperer-Lite + CW308T-STM32F4 (STM32F415).
- 공격자 능력: chosen-CT 주입 (`I` × 27 + `L` + `D`), power trace, 공개
  parameters, 공개 pk dump (`B`).
- 금지: target `mu'` / `ss` / `sk` dump 를 inference 에 사용. 이는 전부 IND-CCA
  위반 (SMAUG-T direct attack PoI 의 corrigendum 과 동일 이유).
- attack-valid trace = `cmd_decap_inject` 의 `D`. 다른 trigger 는 localization /
  diagnostic 보조용.

## 2. 공격 표면 우선순위

| 우선순위 | 표면 | 라벨 | 위험 | 비고 |
|---|---|---|---|---|
| A | secret-key unpacking | HW(f_i), sign(f_i), support | novelty 약함 (Askeland 2021 선행) | 빠른 positive control |
| B | **NTT-domain selected-lane** | HW(γ·f̂_k mod q), MSB, reduction-bit, HD | **main contribution** | host-side INTT(γ e_k), 공개-design classifier null 필요 |
| C | coefficient threshold | signed local Δ(SOTP-inv intermediate) | Ravi/Xu 와 겹침 | NTT 가 약하면 fallback |
| D | SOTP / FO⊥ validity | HW(SOTP^-1 a), HD(valid bit) | classical CCA 와 구분 필요 | implementation-level only |

## 3. NTRU+ 결정 구조와 공격 hook

**핵심 관찰 (2026-05-08)**: NTRU+ ciphertext 와 secret 모두 **NTT 도메인** 으로
바이트 직렬화된다 (`poly_tobytes(ct, &c_ntt)`, sk 의 f 영역도 NTT-domain
packed). 따라서 host 가 INTT 를 직접 돌릴 필요가 없다 — c_ntt 의 한 lane 만
채우고 12-bit pack 만 하면 board 가 그대로 `poly_basemul(&c_ntt, &f_ntt)` 한다.

upstream reference (`Reference_Implementation/NTRU+768/kem.c`) 의 decap:

```
1. poly_frombytes(&c,    ct)                    ← bytes → c_ntt
2. poly_frombytes(&f,    sk)                    ← bytes → f_ntt   (sk 0..POLYBYTES)
3. poly_frombytes(&hinv, sk + POLYBYTES)        ← hinv (NTT-domain)
4. poly_basemul(&m1, &c, &f)                    ← Hook B  (selected-lane)
5. poly_invntt(&m1)                             ← Hook B' (per-lane HD)
6. poly_crepmod3(&m1, &m1)                      ← Hook C  (mod-3 message decode)
7. m2 = m1; poly_ntt(&m2); poly_sub(&c,&c,&m2); poly_basemul(&r2,&c,&hinv)  ← FO 재계산
8. hash_g, poly_sotp_decode(msg, &m1, buf2)     ← Hook D  (SOTP)
9. hash_h, poly_cbd1, poly_ntt → buf2 / verify(buf1, buf2, POLYBYTES)       ← Hook D' (verify)
10. KDF (msg / sk[2·POLYBYTES:])
```

selected-lane attack — basemul 한 lane (4-coefficient block, X^4 - zeta_i):

```
chosen ĉ at lane k  : ĉ[4k+0..4k+3] = (γ, 0, 0, 0),   ĉ[else] = 0.
basemul output m1[4k+i] = γ · f_ntt[4k+i]  mod q  (i=0..3),
                m1[else] = 0.
```

따라서 한 chosen-CT design 으로 한 lane 의 4 개 NTT-domain f 값을
γ·f_ntt[4k+i] mod q 형태로 leak 한다. NTRU+ 는 q=3457 이라 fqmul/Montgomery
reduction 이 명시 일어나고, reduction-bit 또는 HW(γ·f_ntt[4k+i] mod q) 가
가장 깔끔한 라벨이다.

NTRU+ 768 의 NTT 는 `X^N - X^(N/2) + 1` 위에서 4-radix split, 192-zetas
twiddle. 로컬 archive (`lib/crypto_kem/ntruplus768.a`) 의 zetas binary 가
upstream reference 와 모든 192 entry 일치 — host 포팅은 upstream
`Reference_Implementation/NTRU+768/{poly.c, ntt.c}` 를 numpy 로 그대로 옮기면
충분.

## 4. Phase plan

### Phase 0 — 보드/host 스켈레톤
- firmware (이미 있음): `k/e/d/p` baseline + `F/B/I/L/D` chosen-CT + `X` sk dump.
- smoke (이미 있음): `tests/smoke_ntruplus.py`.
- 추가 작업:
  1. `host/ntruplus/` 패키지 — params/codec/ntt/poly_basemul/chosen.
  2. `host/upload.py` 가 ntruplus576 hex 를 그대로 플래시하는지 확인.
  3. 산출물 디렉터리 — `traces/ntruplus576_*`, `results/ntruplus576_*`.
- gate: smoke 100 % 통과, decap 응답 속도 측정 (trace/sec).

### Phase 1 — natural D capture map
- 입력: 단일 valid CT 고정. K = 8 keys × N = 20 traces/key. samples = 24400.
- 분석: cross-key Welch-t, time-resolved SNR, valid-CT 반복 trace stability.
- gate: max|t| > 10 또는 key-classification AUROC ≫ permutation null. window
  후보 (NTT/INTT/basemul/SOTP) 식별.
- 위험: NTRU+ decap 이 24400 sample 에 다 안 들어갈 가능성. 안 들어가면 sample
  offset 을 옮겨가며 phase map 을 만들어 어느 phase 가 시간 어디인지 잡는다.

### Phase 2 — sk-unpack triage (positive control)
- 라벨: HW(f_i), 1[f_i ≠ 0], sign(f_i). NTRU+ secret 분포를 NB/CBD1 packed
  bytes 에서 디코드.
- 분석: held-out-key ridge 또는 template. cross-validation (leave-1-key-out).
- gate: z_rMAE > 3 ∧ z_corr > 3. classification 이면 z_AUROC > 3.
- 결과 처리: 성공해도 보조 claim 으로 둔다 (Askeland 의 결과 재현 → 자체로
  주된 novelty 아님).

### Phase 3 — NTT selected-lane oracle (main)
- ciphertext 가 NTT-domain 직렬화이므로 host 는 c_ntt 를 직접 구성. lane k 에
  c_ntt[4k+i] = γ·δ_{i,0}, 나머지 0. poly_tobytes (12-bit pack) 로 1152 bytes.
- design matrix: K=8 keys, L=192 lanes (basemul block 단위, 768/4=192 — 단,
  basemul 은 8-block 안에 ±zeta 페어로 묶이므로 effective lane = 96 페어),
  scout 는 우선 L=16, G=6 (γ ∈ {1,2,4,8,16,32}), N=10 traces.
  총 8 × 16 × 6 × 10 = 7680 traces. CW-Lite 기준 약 2~4 시간.
- 라벨 family:
  - HW(γ·f̂_k mod q),
  - 1{γ·f̂_k mod q > q/2} (MSB),
  - 1{Barrett 또는 Montgomery reduction subtract 발생},
  - HD(v_before, v_after) (per-lane register transition; reference C 에서 추정).
- 분석: window scan 은 diagnostic. 최종 평가는 natural `D` full window.
- **null 가드 (필수)**:
  - public-design classifier null: label 이 (k,γ) 만으로 결정되면 폐기.
  - secret-shuffle-within-design null: 같은 (k,γ) 내 secret 라벨만 섞었을 때
    AUROC ≈ chance 인지 확인.
- gate: z_rMAE > 3 ∧ z_corr > 3 (회귀) 또는 z_AUROC > 3 (분류) — 둘 다 NULL
  대비. 추가로 expected loop position 에서만 보여야 함.

### Phase 4 — oracle-to-key recovery
- per-design likelihood:
  ```
  log p(T_{k,γ} | f̂_k) = -½ ((y_pred - y_true)/σ)^2
  ```
  여러 γ 합산.
- candidate ranking + INTT consistency:
  ```
  f = INTT^{-1}(f̂),  f_i ∈ {-1,0,1} (NTRU+ secret distribution)
  score(f) = Σ log p(T_{k,γ} | f̂_k) + λ log p(f).
  ```
- evaluation: held-out key 에서 true-key percentile, top-K hit, entropy
  reduction (`H(S) - H(S|T)`).

### Phase 5 — coefficient-domain threshold (fallback)
- `c = α X^j + β X^i`, β ∈ τ + [-B, B]. natural D trace.
- pairwise:
  ```
  ΔT_β = T_D(c_{β+1}) - T_D(c_β)
  y_β  = HW(SOTP^{-1}(a)_{β+1}) - HW(SOTP^{-1}(a)_β)   (signed local)
       또는 HD bucket
  ```
- public-(α)-bug 가드 (SMAUG flip_any 교훈). 4 종 null 모두 통과해야 함.

### Phase 6 — SOTP / validity (마지막 fallback)
- target: `poly_sotp_inv` 의 byte-XOR + reduction, FO⊥ verify (`memcmp(ct, ct')`).
- main claim 금지: scheme break 가 아니라 implementation leakage.
- 위 Phase 1~5 가 다 약하면 마지막 카드. 표는 비교용 baseline 으로만.

## 5. 실험 gate (전 phase 공통)

- leakage gate: `z_rMAE > 3 ∧ z_corr > 3` (회귀) / `z_AUROC > 3` (분류).
- localization gate: 같은 라벨 family 가 expected loop position 에서 반복.
  무작위 위치 1 점만 뜨면 overfit.
- recovery-pressure gate: held-out key 에서 true-key percentile > 0.95.
- attack-valid gate: 최종 claim 은 `T_D` 위에서만. trigger sub-window 는
  figure 보조용.

## 6. 산출물 규약

- traces: `traces/ntruplus576/<phase>/<design>/<key>.npz`
  - `traces` (N, T) float32, `responses` (N, R) uint8, `meta` dict.
- results: `results/ntruplus576/<phase>/<analysis>.{npz,md,png}`.
- 모든 .npz meta 에 firmware hex sha256 + git rev 포함 (이미 capture.py 가 함).
- 실험 가정·과정·결과 분석은 `docs/IDEA.md` 에 추기.

## 7. 시간 예산

| 단계 | 시간 |
|---|---|
| Phase 0 skeleton | 2~4 h |
| Phase 1 D map | 30 min capture + 1 h 분석 |
| Phase 2 sk-unpack | 1 h capture + 2 h 분석 |
| Phase 3 selected-lane scout | 4 h capture + 4 h 분석 |
| Phase 3 confirmation | 6 h capture + 4 h 분석 |
| Phase 4 recovery | 2 h |
| Phase 5/6 fallback | 6 h 합계 |

총 약 1.5~2 주 (단일 보드, 단일 사람 기준).

## 8. 예상 결과별 판단

| 시나리오 | 판단 |
|---|---|
| sk-unpack 만 성공 | 보조 결과. 대회 공격으로는 사용 가능, 논문 main 은 약함 |
| selected-lane oracle 성공, 복구 실패 | partial key exposure / entropy reduction claim |
| selected-lane + ranking 성공 | true-key percentile > 0.95 — 강한 결과 |
| full key recovery | 베스트 — 기존 NTRU SCA 와 비교해 “NTRU+ NTT-domain” 로 차별화 |
| 전부 null | SMAUG-T + NTRU+ 결합 “KpqC finalist resistance survey” 로 전환 |

## 9. 체크리스트

```
[ ] Phase 0: ntruplus576 hex 플래시 + smoke.py 통과
[ ] Phase 0: host/ntruplus/ 패키지 (params/codec/ntt/chosen)
[ ] Phase 1: natural D map capture (K=8, N=20)
[ ] Phase 1: cross-key TVLA / SNR 분석
[ ] Phase 2: sk-unpack 라벨 train / held-out predict
[ ] Phase 3: host-side INTT(γ e_k) ciphertext 생성기
[ ] Phase 3: scout matrix capture (8×16×6×10)
[ ] Phase 3: HW/MSB/reduction-bit/HD 라벨링
[ ] Phase 3: public-design / secret-shuffle null
[ ] Phase 4: multi-γ likelihood 합산 + INTT consistency
[ ] Phase 4: held-out percentile / entropy reduction
[ ] Phase 5: coeff-domain threshold pair sweep (필요 시)
[ ] Phase 6: SOTP / validity (필요 시)
```
