# sca-2026 — 문서 hub

ChipWhisperer-Lite + STM32F415 위에서 두 KpqC KEM 의 side-channel
leakage 를 평가한다. 트랙은 서로 독립적이며 각자 진입점이 따로 있다.

## 두 트랙

| 트랙 | 진입점 | 한 줄 상태 |
|---|---|---|
| **NTRU+768** chosen-CT NTT-domain CPA | [ntruplus/README.md](ntruplus/README.md) | Paper-grade 2 perfect TOP-1 + 17/240 top-100 (Phase 4 종료, 4.5/4.6 partial-recovery 보강) |
| **SMAUG-T** smaug{1,3,5} chosen-CT SCA | [smaug/README.md](smaug/README.md) | 표준 chosen-CT 는 cross-key fail. PCO oracle 은 sk-universal 이지만 µ′-bit 직접 leak 없음 |

각 트랙의 한 눈 timeline 은 `FLOW.md` 에 있다 — 어떤 문제가 있어 어떤
실험을 했고 어떤 결과가 나왔는지 한 표.

- NTRU+ flow: [ntruplus/FLOW.md](ntruplus/FLOW.md)
- SMAUG-T flow: [smaug/FLOW.md](smaug/FLOW.md)

## Reference 자료

`paper/` 디렉토리에 HQC 관련 SCA reference PDF 가 보관됨 (TCHES24/25,
eprint25, 2020-912 등). 두 트랙 모두 chosen-CT SCA 방법론의 reference
literature 로 사용.

## 트랙 외부

- 루트 [`README.md`](../README.md) — 보드/펌웨어 셋업 + 두 트랙 결론 요약.
- `history/` (top-level) — 옛 archived 자료. 현재 트랙 문서와 무관.
