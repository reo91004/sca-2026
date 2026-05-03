#!/usr/bin/env python3
"""[STEP 5] Multi-term chosen-CT 첫 분석 (paper Section 3.4 검증).

Paper Section 3.4 (Multi-term + R^2 cross-component). H_attack.npz (7 designs:
const/2-term/3-term/combined R^2) 의 *first analyzer*. round-trip 100% +
HW gain 1.6-2.2× 측정 후, *cross-design split* idea 발생 → paper Section 5
의 *direct attack PoI* method 도출.

분석:
  1. Round-trip 검증: 보드 µ′ vs host predict_mu_prime(c1, sk)
     → 모든 7 designs 256/256 일치 (paper 의 multi-term 수학 검증)
  2. µ′ HW 분포 비교 (단항 36 → 2-term 58-68 → 3-term 80) — leak amp 1.6-2.2×
  3. per-i Welch t-test (design pair) — N=64 max|t|=4.03 (noise floor 미달)
  4. sparse_recover (Z µ′ oracle) — s[0] 100% (artificial oracle 검증)

Note (history/):
    super-seded by `scripts/analyze_multi_seed.py` (5 seeds × N=128, paper
    main result). 이 스크립트는 H_attack 의 *single-seed first analyzer*.
    Round-trip 100% 검증은 tests/test_chosen.py 에서도 자동 검증.

용법 (legacy):
    history/scripts/analyze_step5_multi_term_initial.py traces/H_attack.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.analysis import sparse_recover  # noqa: E402
from host.smaug import chosen as _chosen  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug import params as _params  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", type=Path, help="run_h_attack.py 의 .npz")
    p.add_argument("--poi", type=Path, default=None,
                   help="profile-PoI .npz (선택). 미지정시 SCA 분류 단계 스킵.")
    p.add_argument("--out-prefix", type=str, default="H",
                   help="결과 PNG/TXT prefix (results/{prefix}_*).")
    p.add_argument("--results-dir", type=Path,
                   default=_REPO / "results")
    return p.parse_args()


def _bits_from_bytes(buf: np.ndarray) -> np.ndarray:
    """uint8 (32,) → bit array (256,) (LSB-first per byte)."""
    if buf.shape != (32,):
        raise ValueError(f"expected (32,), got {buf.shape}")
    bits = np.zeros(256, dtype=np.int64)
    for i in range(32):
        b = int(buf[i])
        for k in range(8):
            bits[i * 8 + k] = (b >> k) & 1
    return bits


def _design_coefs_from_json(spec_json: str) -> dict[tuple[int, int], int]:
    raw = json.loads(spec_json)
    return {(int(m), int(l)): int(a) for m, l, a in raw}


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading {args.npz}")
    d = np.load(args.npz, allow_pickle=True)
    traces = d["traces"]      # (N_total, samples)
    mus = d["mu_prime"]       # (N_total, 32)
    meta = d["meta"].item()
    design_names = list(meta["design_names"])
    design_specs = list(meta["design_specs_json"])
    design_idx = np.array(meta["design_index_per_trace"], dtype=np.int64)
    n_per = int(meta["n_per_design"])
    sk_pke_bytes = bytes.fromhex(meta["sk_pke_bytes_hex"])

    p = _params.get(meta.get("target_level", "smaug1") if "target_level" in meta else "smaug1")
    sk_unpacked = _codec.unpack_sx(sk_pke_bytes)  # (PKE_SECRETKEY_BYTES * 4 = 512,)
    sk = sk_unpacked.reshape(p.module_rank, p.n).astype(np.int64)
    print(f"[INFO] sk shape {sk.shape}, HW per poly = {[int(np.count_nonzero(sk[m])) for m in range(p.module_rank)]}")
    print(f"[INFO] {len(design_names)} 디자인, N/design = {n_per}, total trace = {traces.shape[0]}")

    # =========================================================================
    # 1. Round-trip 검증
    # =========================================================================
    print("\n[Round-trip 검증] 보드 µ′ vs predict_mu_prime(c1, sk)")
    rt_lines = []
    rt_summary = []
    for d_idx, name in enumerate(design_names):
        coefs = _design_coefs_from_json(design_specs[d_idx])
        ct = _chosen.build_multi_term_c1(p, coefs)
        mu_pred = _chosen.predict_mu_prime(p, ct.c1, sk)  # (256,) ∈ {0, 1}
        # 같은 design 의 모든 N trace 의 보드 µ′ 추출 (deterministic 라 모두 같아야 함)
        mask = design_idx == d_idx
        boards = mus[mask]  # (n_per, 32)
        # 각 trace 의 µ′ 가 모두 같은지
        all_same = np.all(boards == boards[0])
        # µ′_predicted 와 µ′_board 비트 비교
        bits_board = _bits_from_bytes(boards[0])
        match = int(np.sum(bits_board == mu_pred))
        line = (f"  [{d_idx}] {name:30s}  match={match}/256  "
                f"board_HW={int(bits_board.sum()):3d}  pred_HW={int(mu_pred.sum()):3d}  "
                f"deterministic={all_same}")
        print(line)
        rt_lines.append(line)
        rt_summary.append((name, match, int(bits_board.sum()), int(mu_pred.sum()), bool(all_same)))

    # 종합
    matches = [s[1] for s in rt_summary]
    all_pass = all(m == 256 for m in matches)
    print(f"\n  [SUMMARY] {sum(1 for m in matches if m == 256)}/{len(matches)} 디자인 round-trip 100%")
    if not all_pass:
        print("  [WARN] 일부 디자인 mismatch — chosen.predict_mu_prime 또는 sk unpacking 또는 컨벤션 재검토 필요")

    # =========================================================================
    # 2. µ′ HW 분포 비교 (per design)
    # =========================================================================
    print("\n[µ′ HW] (보드 응답, 같은 sk 위)")
    hw_lines = []
    for d_idx, name in enumerate(design_names):
        mask = design_idx == d_idx
        # board mu' deterministic 가정 — first sample.
        bits = _bits_from_bytes(mus[mask][0])
        hw = int(bits.sum())
        hw_lines.append(f"  {name:30s}  µ′ HW = {hw:3d}/256")
        print(hw_lines[-1])

    # =========================================================================
    # 3. per-i Welch t-test — 각 design A vs B 의 trace pair
    #    (간단히 첫 두 디자인 간 비교 — H_const_a64 vs H_2term_l5_k50_a64 expected)
    # =========================================================================
    print("\n[per-i TVLA] design pair 비교 (collective µ flip 영역)")
    tvla_summary = []
    if "H_const_a64" in design_names and "H_2term_l5_k50_a64" in design_names:
        a_idx = design_names.index("H_const_a64")
        b_idx = design_names.index("H_2term_l5_k50_a64")
        ta = traces[design_idx == a_idx]
        tb = traces[design_idx == b_idx]
        from host.analysis import tvla as _tvla
        result = _tvla.welch_t(ta, tb)
        max_t = float(np.max(np.abs(result.t)))
        n_leaky = int(np.sum(np.abs(result.t) > 4.5))
        tvla_summary.append(("H_const_a64 vs H_2term_l5_k50_a64", max_t, n_leaky))
        print(f"  H_const_a64 vs H_2term_l5_k50: max|t|={max_t:.2f}, leaky pts (|t|>4.5)={n_leaky}")
    if "H_const_a64" in design_names and "H_3term_l_5_50_200" in design_names:
        a_idx = design_names.index("H_const_a64")
        b_idx = design_names.index("H_3term_l_5_50_200")
        ta = traces[design_idx == a_idx]
        tb = traces[design_idx == b_idx]
        from host.analysis import tvla as _tvla
        result = _tvla.welch_t(ta, tb)
        max_t = float(np.max(np.abs(result.t)))
        n_leaky = int(np.sum(np.abs(result.t) > 4.5))
        tvla_summary.append(("H_const_a64 vs H_3term_l_5_50_200", max_t, n_leaky))
        print(f"  H_const_a64 vs H_3term: max|t|={max_t:.2f}, leaky pts (|t|>4.5)={n_leaky}")

    # =========================================================================
    # 4. sparse_recover 검증 — Z µ′ oracle (artificial) 위에서 sk recovery
    # =========================================================================
    print("\n[Sparse recovery — Z µ′ oracle (idealized)]")
    # H_const_a64 (+det) + H_const_a192 (-det) 의 두 board µ′ 응답을 쓰면
    # E5-Crypto 의 100% 복구가 재현되어야 함.
    sk_hat_lines = []
    if "H_const_a64" in design_names and "H_const_a192" in design_names:
        # 단항 c1 = α (component 0) → ⟨c1, s⟩_i = α · s[0]_i for all i.
        # 보드 µ′ 응답 = predict_mu_prime_bit(α, s[0]_i)
        # 두 응답 → binary_to_ternary_posterior → greedy_sparse_recover.
        a64_idx = design_names.index("H_const_a64")
        a192_idx = design_names.index("H_const_a192")
        # 보드 µ′ deterministic — first sample.
        b64 = _bits_from_bytes(mus[design_idx == a64_idx][0]).astype(np.float64)
        b192 = _bits_from_bytes(mus[design_idx == a192_idx][0]).astype(np.float64)
        # binary_to_ternary 형식: score_pos = P(µ′=1 | α_pos detector).
        # E3a 결과: α=64 가 +1 detector (s=+1 → µ′=1, s=-1 → µ′=0, s=0 → 0)
        #          α=192 가 -1 detector (s=-1 → 1, +1 → 0, 0 → 0)
        post = sparse_recover.binary_to_ternary_posterior(b64, b192)
        s_hat_pke0 = sparse_recover.greedy_sparse_recover(post, hs=p.hs)
        # ground truth: sk[0]
        m = sparse_recover.accuracy(s_hat_pke0, sk[0])
        line = (f"  s[0] via Z oracle (α=64+192): "
                f"bit_acc={m['bit_accuracy']:.3f} "
                f"support_acc={m['support_accuracy']:.3f} "
                f"sign_acc={m['sign_accuracy']:.3f}")
        print(line)
        sk_hat_lines.append(line)
    else:
        print("  [SKIP] H_const_a64 / H_const_a192 디자인이 없음 (oracle pair 못 만듦)")

    # combined cross-component (H_combined_l0_k0): c1[0] = 64, c1[1] = 64 →
    # ⟨c1, s⟩_i = 64 · (s[0]_i + s[1]_i). Single-CT 라 +1 detector 와 동치.
    # 9-class 로 (s[0], s[1]) 이지만 binary µ′ 는 sum-mod-2 정보. 별도 분석 필요.
    if "H_combined_l0_k0" in design_names:
        cidx = design_names.index("H_combined_l0_k0")
        bc = _bits_from_bytes(mus[design_idx == cidx][0])
        # 검증: predict_mu_prime(combined c1, sk) == bc 인지 — round-trip 위에서
        # 이미 확인됨. 추가 분석은 별도.
        print(f"  H_combined_l0_k0 µ′ HW = {int(bc.sum())}/256 — multi-component leak.")

    # =========================================================================
    # 결과 저장
    # =========================================================================
    out_txt = args.results_dir / f"{args.out_prefix}_analysis_summary.txt"
    with out_txt.open("w") as fh:
        fh.write(f"# Phase H 분석 summary — {args.npz.name}\n\n")
        fh.write("## Round-trip 검증\n")
        for line in rt_lines:
            fh.write(line + "\n")
        fh.write(f"\n  total: {sum(1 for m in matches if m == 256)}/{len(matches)} "
                 f"디자인 round-trip 100%\n\n")
        fh.write("## µ′ HW (보드 응답, sample 0 of each design)\n")
        for line in hw_lines:
            fh.write(line + "\n")
        fh.write("\n## per-i TVLA pair 비교\n")
        for tag, mx, nl in tvla_summary:
            fh.write(f"  {tag}: max|t|={mx:.2f}, leaky={nl}\n")
        fh.write("\n## Sparse recovery (Z µ′ oracle)\n")
        for line in sk_hat_lines:
            fh.write(line + "\n")
    print(f"\n[OK] saved {out_txt}")

    # =========================================================================
    # Plot — µ′ HW per design + TVLA
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # HW bar chart
        fig, ax = plt.subplots(figsize=(10, 4))
        hws = []
        for d_idx, name in enumerate(design_names):
            bits = _bits_from_bytes(mus[design_idx == d_idx][0])
            hws.append(int(bits.sum()))
        ax.bar(range(len(design_names)), hws,
               color=["#1f77b4" if "const" in n else "#ff7f0e" if "2term" in n
                      else "#2ca02c" if "3term" in n else "#d62728"
                      for n in design_names])
        ax.axhline(128, ls="--", color="grey", alpha=0.5, label="50% (max info)")
        ax.set_xticks(range(len(design_names)))
        ax.set_xticklabels(design_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("µ′ HW (out of 256)")
        ax.set_title(f"Phase H — µ′ HW per chosen-CT design (sk fixed)")
        ax.legend()
        plt.tight_layout()
        png = args.results_dir / f"{args.out_prefix}_mu_hw_per_design.png"
        plt.savefig(png, dpi=120)
        print(f"[OK] saved {png}")
        plt.close()
    except ImportError:
        print("[SKIP] matplotlib 없음")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
