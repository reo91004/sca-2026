#!/usr/bin/env python3
"""[MAIN ★] Multi-seed evaluation — paper main result generator.

Paper Section 7 (Minimum trace cost) + Section 8.4 (9-seed evaluation).
**paper 의 핵심 표 산출** — 5+ seeds × N regimes 의 mean/std/min/max table.

Method (paper Section 6 — component-specific PoI):
  각 attack_seed*.npz 별:
    1. *Component-specific* direct PoI 학습 (default, paper main):
       s[c] 학습 시 component=c oracle pair (label_comp==c) 만 사용 →
       다른 component 의 trace 영향 제거 → cleaner Welch-t.
       (component_specific=False 옵션이 cross-component baseline)
    2. Oracle pair attack per component (α=64 vs α=192)
    3. sparse_recover.greedy(hs=70) — HW=70 MAP recovery
    4. accuracy(bit, support, sign) per component + full sk

핵심 결과 (paper main, 9 seeds 종합):
  full_sk_acc: mean=100%, std=0% (zero variance across 9 keypairs × 3 N regimes)
  s[0] / s[1] sparse bit + sign: 100% / 100%
  valid_bits: 120 ± 2.5 (= 2 × HW=70 - overlap)

비교 (paper Table):
  Cross-component PoI: mean 95.6% ± 5.6%, min 87.1%
  Component-specific PoI (paper main): mean 100% ± 0%

용법 (paper reproducer):
    # 1. 각 seed 캡처: scripts/run_attack.py -n 128 --out traces/attack_seed${s}.npz
    # 2. 분석:
    scripts/analyze_multi_seed.py traces/attack_seed{1..5}.npz \\
        --out-prefix results/multi_seed_n128

산출물:
    results/<prefix>_summary.txt — per-seed + aggregate mean/std/min/max
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.analysis import sparse_recover  # noqa: E402
from host.smaug.codec import unpack_sx  # noqa: E402
from host.smaug import params as _params  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+", type=Path,
                   help="attack_seed*.npz 경로들 (run_attack.py schema)")
    p.add_argument("--out-prefix", type=str, default="results/multi_seed")
    return p.parse_args()


def _learn_poi_from_subset(traces: np.ndarray, bit_per: np.ndarray,
                           min_class: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """주어진 trace subset 위에서 per-bit PoI/sign 학습.

    bit_per: (N, 256) — each trace 의 µ′ bit array.
    """
    poi = np.full(256, -1, dtype=int)
    t_score = np.zeros(256); sign_v = np.zeros(256, dtype=int)
    for bi in range(256):
        bits = bit_per[:, bi]
        n0 = int((bits == 0).sum()); n1 = int((bits == 1).sum())
        if n0 < min_class or n1 < min_class:
            continue
        m0 = traces[bits == 0].mean(axis=0)
        m1 = traces[bits == 1].mean(axis=0)
        v0 = traces[bits == 0].var(axis=0, ddof=1).clip(min=1e-12)
        v1 = traces[bits == 1].var(axis=0, ddof=1).clip(min=1e-12)
        se = np.sqrt(v0 / n0 + v1 / n1)
        t = (m1 - m0) / se
        poi[bi] = int(np.argmax(np.abs(t)))
        t_score[bi] = float(t[poi[bi]])
        sign_v[bi] = +1 if t_score[bi] > 0 else -1
    return poi, t_score, sign_v


def analyze_one(npz_path: Path, component_specific: bool = True) -> dict:
    """attack_seed{N}.npz 한 파일 분석.

    Schema: traces/mu_prime/meta with label_component, label_alpha,
            sk_pke_bytes_hex, n_per_ct, alpha_pos, alpha_neg.

    component_specific=True (default, paper method): s[c] 학습 시 component=c
    oracle pair (label_comp==c) 만 사용. 다른 component 가 trace 에 미치는
    noise 제거 → cleaner Welch-t.

    component_specific=False: 4 designs all-mixed PoI (cross-component, 더 noisy).
    """
    p_smaug = _params.SMAUG1
    hs = p_smaug.hs

    d = np.load(npz_path, allow_pickle=True)
    T = d["traces"]; mus = d["mu_prime"]; meta = d["meta"].item()
    label_comp = np.asarray(meta["label_component"])
    label_alpha = np.asarray(meta["label_alpha"])
    sk_bytes = bytes.fromhex(meta["sk_pke_bytes_hex"])
    sk = unpack_sx(sk_bytes).reshape(p_smaug.module_rank, p_smaug.n).astype(np.int64)
    n_per = int(meta["n_per_ct"])

    # bit per trace
    N_total = T.shape[0]
    bit_per = np.zeros((N_total, 256), dtype=np.int8)
    for i in range(N_total):
        for k in range(256):
            bit_per[i, k] = (int(mus[i, k // 8]) >> (k % 8)) & 1

    min_class = max(8, n_per // 4)

    # Cross-component PoI (분석 만, 비교용)
    poi_all, t_all, _ = _learn_poi_from_subset(T, bit_per, min_class)
    n_valid_all = int((poi_all >= 0).sum())
    max_t_all = float(np.abs(t_all[poi_all >= 0]).max()) if n_valid_all else 0.0

    out: dict = {
        "path": str(npz_path),
        "sk_hash_short": (sk_bytes.hex())[:16],
        "sk_hw": [int(np.count_nonzero(sk[i])) for i in range(p_smaug.module_rank)],
        "n_per_ct": n_per,
        "n_total_trace": N_total,
        "n_valid_bits": n_valid_all,  # cross-component
        "max_t": max_t_all,
        "method": "component_specific" if component_specific else "cross_component",
    }

    # per component analysis
    full_correct = 0
    for comp in (0, 1):
        mp = (label_comp == comp) & (label_alpha == 64)
        mn = (label_comp == comp) & (label_alpha == 192)
        if not mp.any() or not mn.any():
            out[f"s{comp}"] = None
            continue

        # PoI 학습 — component_specific 옵션
        if component_specific:
            mask_comp = (label_comp == comp)
            T_c = T[mask_comp]
            bit_c = bit_per[mask_comp]
            poi, t_score, sign_v = _learn_poi_from_subset(T_c, bit_c, min_class)
        else:
            poi = poi_all; t_score = t_all
            sign_v = np.where(t_score > 0, +1, -1).astype(int)

        # oracle attack
        diff = T[mp].mean(axis=0) - T[mn].mean(axis=0)
        d_per = np.zeros(256)
        for bi in range(256):
            if poi[bi] >= 0:
                d_per[bi] = float(sign_v[bi]) * float(diff[poi[bi]])
        s_gt = sk[comp].astype(np.int8)
        corr = float(np.corrcoef(d_per, s_gt.astype(float))[0, 1])
        sigma = max(float(d_per.std()), 1e-9)

        # raw threshold (1σ)
        raw = np.where(d_per > sigma, +1,
                       np.where(d_per < -sigma, -1, 0)).astype(np.int8)
        m_raw = sparse_recover.accuracy(raw, s_gt)

        # posterior + sparse
        pos = np.abs(d_per[d_per > 0]); neg = np.abs(d_per[d_per < 0])
        mu = ((pos.mean() if pos.size else 1.0) + (neg.mean() if neg.size else 1.0)) / 2.0
        log_neg = -0.5 * ((d_per + mu) / sigma) ** 2
        log_zer = -0.5 * (d_per / sigma) ** 2
        log_pos = -0.5 * ((d_per - mu) / sigma) ** 2
        L = np.stack([log_neg, log_zer, log_pos], axis=1)
        L -= L.max(axis=1, keepdims=True)
        P = np.exp(L); P /= P.sum(axis=1, keepdims=True)
        sp = sparse_recover.greedy_sparse_recover(P, hs)
        m_sp = sparse_recover.accuracy(sp, s_gt)

        valid_c = poi >= 0
        out[f"s{comp}"] = {
            "valid_bits": int(valid_c.sum()),
            "max_t": float(np.abs(t_score[valid_c]).max()) if valid_c.any() else 0.0,
            "corr": corr,
            "raw_bit": m_raw["bit_accuracy"],
            "raw_support": m_raw["support_accuracy"],
            "raw_sign": m_raw["sign_accuracy"],
            "sparse_bit": m_sp["bit_accuracy"],
            "sparse_support": m_sp["support_accuracy"],
            "sparse_sign": m_sp["sign_accuracy"],
            "sparse_pred_hw": int(np.count_nonzero(sp)),
        }
        full_correct += int((sp == s_gt).sum())

    out["full_sk_acc"] = full_correct / 512.0
    return out


def summarize(results: list[dict]) -> dict:
    """results 의 mean/std/min/max metric 표."""
    metrics = ["full_sk_acc", "n_valid_bits", "max_t"]
    per_comp_metrics = ["corr", "raw_bit", "sparse_bit", "sparse_sign"]

    summary = {"n_seeds": len(results)}
    for m in metrics:
        vals = np.array([r[m] for r in results], dtype=float)
        summary[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
        }
    for comp in (0, 1):
        for pm in per_comp_metrics:
            vals = np.array([r[f"s{comp}"][pm] for r in results
                             if r[f"s{comp}"] is not None], dtype=float)
            if vals.size == 0:
                continue
            summary[f"s{comp}.{pm}"] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "median": float(np.median(vals)),
            }
    return summary


def main() -> int:
    args = parse_args()
    paths = sorted(args.paths)
    print(f"[INFO] {len(paths)} seeds")

    results = []
    for p in paths:
        if not p.exists():
            print(f"  [SKIP] {p} not found")
            continue
        print(f"  analyzing {p.name} ...")
        r = analyze_one(p)
        results.append(r)
        s0 = r.get("s0", {}); s1 = r.get("s1", {})
        print(f"    sk={r['sk_hash_short']}, N/CT={r['n_per_ct']}, valid={r['n_valid_bits']}, "
              f"max|t|={r['max_t']:.2f}")
        print(f"    s[0] sparse bit={s0.get('sparse_bit', 0):.3f} "
              f"sign={s0.get('sparse_sign', 0):.3f}, "
              f"s[1] sparse bit={s1.get('sparse_bit', 0):.3f} "
              f"sign={s1.get('sparse_sign', 0):.3f}, "
              f"full sk={r['full_sk_acc']:.3f}")

    if not results:
        print("[ERROR] no valid seeds")
        return 1

    summary = summarize(results)

    print(f"\n[SUMMARY] {summary['n_seeds']} seeds")
    print(f"  full_sk_acc: mean={summary['full_sk_acc']['mean']:.3f}, "
          f"std={summary['full_sk_acc']['std']:.3f}, "
          f"min={summary['full_sk_acc']['min']:.3f}, "
          f"max={summary['full_sk_acc']['max']:.3f}")
    print(f"  n_valid_bits: mean={summary['n_valid_bits']['mean']:.1f}, "
          f"std={summary['n_valid_bits']['std']:.1f}")
    print(f"  max_t:        mean={summary['max_t']['mean']:.2f}, "
          f"std={summary['max_t']['std']:.2f}")
    for comp in (0, 1):
        sb = summary.get(f"s{comp}.sparse_bit")
        ss = summary.get(f"s{comp}.sparse_sign")
        cc = summary.get(f"s{comp}.corr")
        if sb is None: continue
        print(f"  s[{comp}] corr:        mean={cc['mean']:+.3f} ± {cc['std']:.3f}")
        print(f"  s[{comp}] sparse bit:  mean={sb['mean']:.3f} ± {sb['std']:.3f} "
              f"(min {sb['min']:.3f}, max {sb['max']:.3f})")
        print(f"  s[{comp}] sparse sign: mean={ss['mean']:.3f} ± {ss['std']:.3f}")

    # save summary
    out_txt = Path(args.out_prefix + "_summary.txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w") as fh:
        fh.write(f"# Multi-seed evaluation — {len(results)} seeds\n\n")
        for r in results:
            fh.write(f"## seed {r['sk_hash_short']}, N/CT={r['n_per_ct']}\n")
            fh.write(f"  sk HW: {r['sk_hw']}, valid bits: {r['n_valid_bits']}, "
                     f"max|t|={r['max_t']:.2f}\n")
            for comp in (0, 1):
                if r.get(f"s{comp}") is None: continue
                s = r[f"s{comp}"]
                fh.write(f"  s[{comp}]: corr={s['corr']:+.4f}, "
                         f"raw bit={s['raw_bit']:.3f}, "
                         f"sparse bit={s['sparse_bit']:.3f}, "
                         f"sign={s['sparse_sign']:.3f}, "
                         f"pred_hw={s['sparse_pred_hw']}\n")
            fh.write(f"  full sk: {r['full_sk_acc']:.3f}\n\n")
        fh.write("## Summary\n")
        for k, v in summary.items():
            if k == "n_seeds": fh.write(f"  n_seeds: {v}\n"); continue
            fh.write(f"  {k}: mean={v['mean']:.4f} std={v['std']:.4f} "
                     f"min={v['min']:.4f} max={v['max']:.4f} median={v['median']:.4f}\n")
    print(f"\n[OK] {out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
