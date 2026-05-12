# sca-2026 — 문서 hub

ChipWhisperer-Lite + STM32F415 위에서 두 KpqC KEM 의 side-channel
leakage 를 평가한다. 트랙은 서로 독립적이며 각자 단일 진입점을 갖는다.

## 두 트랙

| 트랙 | 진입점 | 한 줄 상태 |
|---|---|---|
| **NTRU+768** chosen-CT NTT-domain CPA | [`ntruplus/README.md`](ntruplus/README.md) | Paper-grade 2 perfect TOP-1 + 17/240 top-100. Phase 4 종료, 4.5/4.6 partial-recovery 보강. |
| **SMAUG-T** smaug{1,3,5} chosen-CT SCA | [`smaug/README.md`](smaug/README.md) | 표준 chosen-CT 는 cross-key fail. PCO oracle 은 sk-universal 이지만 µ′-bit 직접 leak 없음. 트랙 archived. |

각 트랙 문서는 `README.md` (한 페이지 overview + 결과 + 재현) 와 길어진
경우 `experiments.md` (phase 별 상세 로그) 두 파일이다.

## Reference 자료

`paper/` 디렉토리에 HQC 관련 SCA reference PDF (TCHES24/25, eprint25,
2020-912 등). 두 트랙의 chosen-CT SCA 방법론 비교 reference.

## 트랙 외부

루트 [`README.md`](../README.md) — 보드/펌웨어 셋업 + 두 트랙 결론 요약.
