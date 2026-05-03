# history/ — 발견 과정 + negative results

이 디렉토리는 **paper main flow 와 직접 무관한** dev/incremental 코드를 보존하는
공간이다. paper 의 *main result* (4 chosen-CT × N=2 = 8 traces, 5/5 keypairs
100% recovery) reproduce 에는 필요없지만, *논문 evidence trail* 로서 가치 있다.

`docs/paper_draft.md` 의 각 section 이 인용한 *수치/통찰* 의 *원천 코드* 들이
여기에 모두 있다. 향후 reviewer 가 negative result reproduction 을 요청하거나,
저자가 따라가는 작업 (masked SMAUG-T 평가 등) 시 직접 사용 가능.

## 디렉토리

```
history/
├── scripts/           # 발견 과정 + dev 도구 + negative result 분석 (12)
├── host_analysis/     # E3-specific helper modules (2)
└── tests/             # 위 모듈의 unit tests (1)
```

## scripts/ — 발견 과정 timeline

| 파일 | Step | 한 줄 설명 | Paper 의 어느 evidence |
|---|---|---|---|
| `capture_step1_collective_leak_n1000.py` | **1** | 라벨된 µ pattern × N=1000 trace TVLA 캡처 | Section 8.1 SNR phase transition 의 *시드 측정* (max\|t\|=8.99) |
| `capture_step2_convention_pilot.py` | **2** | 14 chosen-CT pilot (j × α grid) 캡처 | Section 3 chosen-CT 컨벤션 정정 의 trigger (128/128 매칭 검증) |
| `capture_step3_random_mu_profile.py` | **3** | random-µ × N=200/1000 profile 캡처 | Section 5.1 *profile-PoI cross-domain transfer fail* 의 원천 데이터 |
| `analyze_step4a_profile_PoI_NEGATIVE.py` | **4a** | profile-PoI → chosen-CT 분류 (56% chance) | Section 5.1 *negative baseline* (profile transfer fail) |
| `analyze_step4b_sparse_on_profile_NEGATIVE.py` | **4b** | + sparse_recover (raw 58% → sparse 55%) | Section 5.1 *sparse_recover 한계* (raw 가 chance 면 후처리 도와주지 못함) |
| `analyze_step5_multi_term_initial.py` | **5** | H_attack.npz multi-term 분석 (round-trip 100%) | Section 3.4 multi-term 의 *first analyzer* (super-seded by `analyze_multi_seed.py`) |
| `debug_T_command_NEGATIVE.py` | — | isolated `poly_mul_acc` 'T' 명령 round-trip 검증 (0/6 fail) | Section 8.2 *'T' 명령 negative* (q-modular internal storage form 추정) |
| `util_throughput_bench.py` | — | Z 명령 throughput 측정 (2.65 tr/s) | Setup section 의 throughput 수치 출처 |
| `util_firmware_zx_verify.py` | — | Z/X firmware command 검증 (dev-time) | (no paper relation, dev tool) |
| `util_plot_overview.py` | — | .npz overview (mean ± σ) plot | dev plotting (paper figure 별도) |
| `util_plot_partition.py` | — | partition table heatmap | dev plotting |
| `util_plot_tvla.py` | — | 2-class Welch-t plot | dev plotting |

## host_analysis/

| 파일 | 한 줄 설명 |
|---|---|
| `util_e3c_analysis.py` | E3c single-PoI matched filter helper (super-seded by `host/analysis/sparse_recover.py`) |
| `util_labelmix_helper.py` | E3b fixed-µ TVLA helper (`capture_step1_collective_leak_n1000.py` 보조) |

## tests/

| 파일 | 한 줄 설명 |
|---|---|
| `test_labelmix.py` | `util_labelmix_helper.py` 의 unit tests (legacy) |

## 발견 과정 timeline (paper 흐름)

```
Step 1: collective µ flip 의 trace leak 발견 (max|t|=8.99)
    ↓
Step 2: 14 chosen-CT pilot → "왜 모든 j 가 같은 답?" 의문
    ↓ 수학 재검토
    [컨벤션 정정 — paper Section 3]
    ↓
Step 3: profile-PoI 시도 → cross-domain transfer fail (negative)
    ↓
Step 4a/4b: profile-PoI 의 56% (raw) / 55% (sparse) 측정 (negative)
    ↓
Step 5: multi-term H_attack 첫 분석 → cross-design split idea 발생
    ↓
[Direct attack PoI 발견 — paper Section 5]
[Component-specific PoI 개선 — paper Section 6]
[N curve 측정 — paper Section 7]
    ↓
Main result: 8 traces, 9/9 keypairs 100% (paper Section 7-8)
```

## 사용

`history/` 안 코드는 *deprecated* 표시하지만 *동작* 한다 (의존성 깨졌다면 Edit
필요할 수 있음). 예시:

```bash
# Step 4a profile-PoI baseline reproduce (negative result)
python3 history/scripts/analyze_step4a_profile_PoI_NEGATIVE.py \
    --profile traces/profile_random_mu_n1000.npz \
    --attack traces/attack_const_c1.npz
```

단 `host/analysis/` 의 `e3c.py`, `labelmix.py` 가 history 로 이동했으므로 그
모듈을 import 하는 history 안 코드는 ImportError 가 날 수 있음. 필요시 history
파일의 import 경로 수정 또는 git history 에서 원본 복구.

## paper main flow (참고)

| 파일 | 역할 |
|---|---|
| `scripts/run_attack.py` | MAIN capture (4 chosen-CT × N) |
| `scripts/run_h_attack.py` | Phase H multi-term capture (검증) |
| `scripts/attack_direct_poi.py` | single-seed direct PoI 분석 |
| `scripts/analyze_multi_seed.py` | multi-seed (paper main result generator) |

이 4 파일이 paper Section 4-7 의 *exact reproducer*. `history/` 는 그 *evidence
trail*.
