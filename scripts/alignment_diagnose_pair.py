#!/usr/bin/env python3
"""E1 분석 — same-sk session A vs B alignment diagnosis.

세 단계 진단:
1. **In-session sanity** — 각 session 안에서 µ′-PoI 학습으로 full_sk acc.
   100% 면 같은 capture session 안의 trace 들은 정렬됨 (= 기존 가정 확인).
2. **Cross-session A→B** — A 의 µ′-PoI 학습 schedule 을 B 에 적용. 같은 sk
   이므로 µ′ 응답 자체는 같다 (deterministic). 차이는 *trace 쪽* 정렬만.
3. **Cross-correlation** — A vs B 의 oracle-pair |t| trace cross-correlation.
   norm-corr 와 optimal shift 보고.

해석:
- (a) In-session 100% + Cross A→B 100% → 같은 sk 위 alignment 보존. 즉 기존
  9 dataset 의 cross-key fail 은 sk-dependent timing 의 문제.
- (b) In-session 100% + Cross A→B fail → 같은 sk 위에서도 alignment 깨짐.
  hardware-state jitter (ART prefetch 등) 가 indcpa_dec 내부에 있다는 의미.
- (c) In-session 도 fail → 펌웨어/캡처 setup 의 broader issue.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from scripts.analyze_multi_seed import analyze_one, learn_schedule_from_npz  # noqa: E402
from scripts.alignment_diagnose import welch_t_abs, best_shift  # noqa: E402


def main() -> int:
    pa = Path("traces/E1_sessionA.npz")
    pb = Path("traces/E1_sessionB.npz")
    if not pa.exists() or not pb.exists():
        raise SystemExit("[ERROR] E1 sessions missing — run scripts/run_e1_alignment.py first")

    da = np.load(pa, allow_pickle=True)
    db = np.load(pb, allow_pickle=True)
    sk_a = da["meta"].item()["sk_pke_bytes_hex"][:16]
    sk_b = db["meta"].item()["sk_pke_bytes_hex"][:16]
    print(f"[E1] sk_A = {sk_a}, sk_B = {sk_b}, match = {sk_a == sk_b}")
    if sk_a != sk_b:
        raise SystemExit("[ERROR] sk mismatch — E1 invalid")

    Path("results").mkdir(exist_ok=True)
    out_lines = []

    def log(s: str) -> None:
        print(s)
        out_lines.append(s)

    # === 1. In-session sanity ===
    log("\n[1] In-session µ′-PoI baseline (100% expected — same-session aligned)")
    for label, path in [("A", pa), ("B", pb)]:
        r = analyze_one(path, poi_source="mu-prime")
        log(f"    session {label}: full_sk_acc = {r['full_sk_acc']:.4f}, "
            f"valid_bits = {r['n_valid_bits']}, max|t| = {r['max_t']:.2f}")

    # === 2. Cross-session A→B and B→A (mu-prime schedule transfer) ===
    log("\n[2] Cross-session schedule transfer (same sk, different capture session)")
    sched_a = learn_schedule_from_npz(pa)
    sched_b = learn_schedule_from_npz(pb)
    r_ab = analyze_one(pb, poi_source="schedule", schedule=sched_a)
    r_ba = analyze_one(pa, poi_source="schedule", schedule=sched_b)
    log(f"    A→B (apply A schedule to B): full_sk_acc = {r_ab['full_sk_acc']:.4f}")
    log(f"    B→A (apply B schedule to A): full_sk_acc = {r_ba['full_sk_acc']:.4f}")

    # === 3. Cross-correlation of |t| traces ===
    log("\n[3] Cross-correlation of oracle-pair |t| traces")
    Ta, Tb = da["traces"], db["traces"]
    metaA = da["meta"].item(); metaB = db["meta"].item()
    laA = np.asarray(metaA["label_alpha"]); laB = np.asarray(metaB["label_alpha"])
    ap = int(metaA["alpha_pos"]); an = int(metaA["alpha_neg"])
    tA = welch_t_abs(Ta, laA, ap, an)
    tB = welch_t_abs(Tb, laB, ap, an)
    log(f"    |t| max session A: {float(tA.max()):.3f} @ sample {int(np.argmax(tA))}")
    log(f"    |t| max session B: {float(tB.max()):.3f} @ sample {int(np.argmax(tB))}")
    shift, score = best_shift(tA, tB, max_shift=4000)
    log(f"    best shift (B vs A): {shift:+d}  norm-corr = {score:.4f}")

    # post-align cumulative
    n = tA.size
    if shift >= 0:
        tB_aligned = np.zeros_like(tB); tB_aligned[shift:] = tB[:n - shift]
    else:
        tB_aligned = np.zeros_like(tB); tB_aligned[:n + shift] = tB[-shift:]
    cum_mean = (tA + tB_aligned) / 2.0
    cum_max = np.maximum(tA, tB_aligned)
    log(f"    post-align cumulative max|t|: {float(cum_max.max()):.3f}")
    log(f"    post-align cumulative mean|t|: {float(cum_mean.max()):.3f}")

    # === 4. 결론 ===
    log("\n[4] Conclusion")
    if r_ab["full_sk_acc"] >= 0.95 and r_ba["full_sk_acc"] >= 0.95:
        log("    (a) Same-sk cross-session alignment HOLDS.")
        log("        → cross-key fail (어제 결과) 은 sk-dependent timing 또는 keygen RNG.")
        log("        → 다음 단계: key-swap firmware 또는 sub-trigger 추가.")
    elif score < 0.2:
        log("    (b) Same-sk cross-session alignment FAILS (norm-corr < 0.2).")
        log("        → indcpa_dec 자체가 hardware-state-jitter 영향 받음.")
        log("        → 다음 단계: DTW/elastic alignment 또는 implementation-derived schedule.")
    else:
        log(f"    (c) Mixed result. norm-corr={score:.3f}, full_sk A→B={r_ab['full_sk_acc']:.3f}.")
        log("        → 추가 진단 필요.")

    out_path = Path("results/E1_alignment.txt")
    out_path.write_text("\n".join(out_lines))
    print(f"\n[OK] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
