#!/usr/bin/env python3
"""[MAIN ★] Multi-seed evaluation — paper main result generator.

Paper Section 7 (Minimum trace cost) + Section 8.4 (9-seed evaluation).
**paper 의 핵심 표 산출** — 5+ seeds × N regimes 의 mean/std/min/max table.

Method (paper Section 6 — component-specific PoI):
  각 attack_seed*.npz 별:
    1. *Component-specific* direct PoI 학습:
       기본 경로는 µ′ 라벨을 쓰지 않고, 공격자가 아는 design label
       (α_pos vs α_neg) 만으로 Welch-t 를 수행한다. 각 coefficient i 는
       trace window i 안에서 PoI 를 고른다.
    2. Oracle pair attack per component (α=64 vs α=192)
    3. sparse_recover.greedy(hs=70) — HW=70 MAP recovery
    4. accuracy(bit, support, sign) per component + full sk

중요한 해석:
  --poi-source mu-prime 은 board 가 반환한 µ′ 라벨로 PoI 를 고르는
  diagnostic/ablation 경로다. 이 경로의 100% 결과는 target µ′ 를 모르는
  공격 모델의 성능으로 주장하면 안 된다.

  --poi-source design-window 는 공격자가 아는 design label(α_pos/α_neg)만
  사용한다. 다만 per-coefficient time window 가 실제 trace layout 과 맞아야
  하므로, attack-valid 결과는 window alignment/calibration 후 별도로 산출한다.

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
    p.add_argument("--poi-source",
                   choices=("design-window", "mu-prime", "schedule"),
                   default="design-window",
                   help="design-window: α_pos/α_neg design labels only (attack-valid, "
                        "needs trace alignment). "
                        "mu-prime: target µ′ labels (instrumented baseline; "
                        "공격 결과 아님). "
                        "schedule: load PoI idx_t from --schedule-source npz "
                        "(profiled SCA — calibration keypair 의 µ′ 로 학습한 schedule).")
    p.add_argument("--window-start", type=int, default=0,
                   help="design-window PoI search start sample.")
    p.add_argument("--window-end", type=int, default=None,
                   help="design-window PoI search end sample. Default: trace end.")
    p.add_argument("--schedule-source", type=Path, default=None,
                   help="poi-source=schedule 시 PoI idx_t 를 학습할 calibration .npz. "
                        "target 평가 시 자기 자신을 사용하면 µ′-leak 와 동등 → 가드.")
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


def _welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sample-wise Welch t for two trace groups."""
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.zeros(a.shape[1], dtype=np.float64)
    ma = a.mean(axis=0); mb = b.mean(axis=0)
    va = a.var(axis=0, ddof=1).clip(min=1e-12)
    vb = b.var(axis=0, ddof=1).clip(min=1e-12)
    return (ma - mb) / np.sqrt(va / a.shape[0] + vb / b.shape[0])


def _learn_poi_from_design_windows(
    traces_pos: np.ndarray,
    traces_neg: np.ndarray,
    *,
    n_bits: int = 256,
    window_start: int = 0,
    window_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unlabeled direct PoI: α_pos vs α_neg design labels only.

    Without µ′ labels, per-bit PoI needs a time-to-coefficient convention. This
    function divides [window_start, window_end) into n_bits equal windows and
    chooses the strongest α_pos-vs-α_neg Welch-t sample inside each window.

    sign_v is intentionally all +1: the signed score is the raw
    mean(α_pos)-mean(α_neg) at the selected point. Multiplying by the t sign would
    take abs(diff) and destroy the secret sign.
    """
    if traces_pos.ndim != 2 or traces_neg.ndim != 2:
        raise ValueError("traces_pos/traces_neg must be 2-D")
    n_samples = traces_pos.shape[1]
    if traces_neg.shape[1] != n_samples:
        raise ValueError("trace length mismatch")
    if window_end is None:
        window_end = n_samples
    if not (0 <= window_start < window_end <= n_samples):
        raise ValueError(
            f"bad window range [{window_start}, {window_end}) for trace length {n_samples}"
        )
    if (window_end - window_start) < n_bits:
        raise ValueError("window range must contain at least one sample per bit")

    t = _welch_t(traces_pos, traces_neg)
    edges = np.linspace(window_start, window_end, n_bits + 1, dtype=int)
    poi = np.full(n_bits, -1, dtype=int)
    t_score = np.zeros(n_bits, dtype=np.float64)
    sign_v = np.ones(n_bits, dtype=int)
    for bi in range(n_bits):
        lo, hi = int(edges[bi]), int(edges[bi + 1])
        if hi <= lo:
            continue
        local = t[lo:hi]
        off = int(np.argmax(np.abs(local)))
        poi[bi] = lo + off
        t_score[bi] = float(local[off])
    return poi, t_score, sign_v


def learn_schedule_from_npz(npz_path: Path) -> dict:
    """Calibration npz 의 µ′ 라벨로 component 별 PoI idx_t (256,) 추출.

    Returns dict: {comp: {"poi": (256,), "sign": (256,), "t": (256,)}} (component-specific).
    """
    d = np.load(npz_path, allow_pickle=True)
    T = d["traces"]; mus = d["mu_prime"]; meta = d["meta"].item()
    level_name = meta.get("level") or {128: "smaug1", 192: "smaug3",
                                         256: "smaug5"}[len(meta["sk_pke_bytes_hex"]) // 2]
    p = _params.get(level_name)
    label_comp = np.asarray(meta["label_component"])
    n_per = int(meta["n_per_ct"])
    min_class = max(8, n_per // 4)

    N = T.shape[0]
    bit_per = np.zeros((N, 256), dtype=np.int8)
    for i in range(N):
        for k in range(256):
            bit_per[i, k] = (int(mus[i, k // 8]) >> (k % 8)) & 1

    out = {"_meta": {"level": level_name, "module_rank": p.module_rank,
                     "sk_hash_short": meta["sk_pke_bytes_hex"][:16],
                     "source": str(npz_path)}}
    for comp in range(p.module_rank):
        mask = (label_comp == comp)
        poi, t_score, sign_v = _learn_poi_from_subset(T[mask], bit_per[mask], min_class)
        out[comp] = {"poi": poi, "sign": sign_v, "t": t_score}
    return out


def analyze_one(
    npz_path: Path,
    component_specific: bool = True,
    *,
    poi_source: str = "design-window",
    window_start: int = 0,
    window_end: int | None = None,
    schedule: dict | None = None,
) -> dict:
    """attack_seed{N}.npz 한 파일 분석.

    Schema: traces/mu_prime/meta with label_component, label_alpha,
            sk_pke_bytes_hex, n_per_ct, alpha_pos, alpha_neg, level (optional).

    component_specific=True (default, paper method): s[c] 학습 시 component=c
    oracle pair (label_comp==c) 만 사용. 다른 component 가 trace 에 미치는
    noise 제거 → cleaner Welch-t.

    component_specific=False: 모든 designs all-mixed PoI (cross-component, 더 noisy).

    level-agnostic — meta 에 'level' 있으면 그 SmaugParams 로, 없으면 smaug1 fallback.
    """
    d = np.load(npz_path, allow_pickle=True)
    T = d["traces"]; mus = d["mu_prime"]; meta = d["meta"].item()

    # level 자동 감지: meta['level'] (신규 schema) 또는 sk_pke 길이로 fallback.
    level_name = meta.get("level")
    if level_name is None:
        sk_len = len(meta["sk_pke_bytes_hex"]) // 2
        # smaug1=128, smaug3=192, smaug5=256
        level_name = {128: "smaug1", 192: "smaug3", 256: "smaug5"}.get(sk_len, "smaug1")
    p_smaug = _params.get(level_name)
    hs = p_smaug.hs
    alpha_pos = int(meta.get("alpha_pos", 64))
    alpha_neg = int(meta.get("alpha_neg", 192))

    label_comp = np.asarray(meta["label_component"])
    label_alpha = np.asarray(meta["label_alpha"])
    sk_bytes = bytes.fromhex(meta["sk_pke_bytes_hex"])
    sk = unpack_sx(sk_bytes).reshape(p_smaug.module_rank, p_smaug.n).astype(np.int64)
    n_per = int(meta["n_per_ct"])

    # Legacy/debug path only. The profile-free design-window path never uses
    # board µ′ labels for PoI learning.
    N_total = T.shape[0]
    bit_per = None
    if poi_source == "mu-prime":
        bit_per = np.zeros((N_total, 256), dtype=np.int8)
        for i in range(N_total):
            for k in range(256):
                bit_per[i, k] = (int(mus[i, k // 8]) >> (k % 8)) & 1

    min_class = max(8, n_per // 4)

    poi_all = np.full(256, -1, dtype=int)
    t_all = np.zeros(256)
    n_valid_all = 0
    max_t_all = 0.0
    if poi_source == "mu-prime":
        assert bit_per is not None
        poi_all, t_all, _ = _learn_poi_from_subset(T, bit_per, min_class)
        n_valid_all = int((poi_all >= 0).sum())
        max_t_all = float(np.abs(t_all[poi_all >= 0]).max()) if n_valid_all else 0.0

    out: dict = {
        "path": str(npz_path),
        "level": level_name,
        "module_rank": p_smaug.module_rank,
        "hs": hs,
        "alpha_pos": alpha_pos,
        "alpha_neg": alpha_neg,
        "sk_hash_short": (sk_bytes.hex())[:16],
        "sk_hw": [int(np.count_nonzero(sk[i])) for i in range(p_smaug.module_rank)],
        "n_per_ct": n_per,
        "n_total_trace": N_total,
        "n_valid_bits": n_valid_all,  # cross-component
        "max_t": max_t_all,
        "method": f"{'component_specific' if component_specific else 'cross_component'}:{poi_source}",
    }

    # per component analysis
    full_correct = 0
    valid_total = 0
    max_t_components = 0.0
    for comp in range(p_smaug.module_rank):
        mp = (label_comp == comp) & (label_alpha == alpha_pos)
        mn = (label_comp == comp) & (label_alpha == alpha_neg)
        if not mp.any() or not mn.any():
            out[f"s{comp}"] = None
            continue

        # PoI 학습 — default는 µ′ 없이 design label만 사용.
        if poi_source == "schedule":
            if schedule is None or comp not in schedule:
                raise ValueError(f"poi_source=schedule needs schedule[{comp}]")
            poi = np.asarray(schedule[comp]["poi"], dtype=int).copy()
            t_score = np.asarray(schedule[comp]["t"], dtype=float).copy()
            sign_v = np.asarray(schedule[comp]["sign"], dtype=int).copy()
        elif poi_source == "design-window":
            poi, t_score, sign_v = _learn_poi_from_design_windows(
                T[mp], T[mn],
                n_bits=p_smaug.n,
                window_start=window_start,
                window_end=window_end,
            )
        elif component_specific:
            assert bit_per is not None
            mask_comp = (label_comp == comp)
            T_c = T[mask_comp]
            bit_c = bit_per[mask_comp]
            poi, t_score, sign_v = _learn_poi_from_subset(T_c, bit_c, min_class)
        else:
            assert bit_per is not None
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
        # Score decomposition (PoI 가 살아있는지 진단 — full_sk 단일 metric 보완).
        # top_k_recall: |d_per| 의 상위 hs 위치에 진짜 nonzero 자리가 몇 개 들어가는가.
        #   pre-sparse rank metric. 1.0 이면 sparse_recover 가 완벽한 input 을 받음.
        # support_pred ∩ support_true / hs : sparse_recover 출력 후 support 일치율 (raw recall).
        # nonzero_corr : nonzero 자리 위에서만 본 d_per vs sk_gt 상관 (sign 정보 보존 여부).
        order = np.argsort(-np.abs(d_per), kind="stable")
        top_k_set = set(order[:hs].tolist())
        true_nz_set = set(np.flatnonzero(s_gt).tolist())
        top_k_recall = len(top_k_set & true_nz_set) / float(hs) if hs > 0 else 0.0
        sp_nz = set(np.flatnonzero(sp).tolist())
        sp_support_recall = len(sp_nz & true_nz_set) / float(hs) if hs > 0 else 0.0
        nz_mask = s_gt != 0
        if nz_mask.any() and float(d_per[nz_mask].std()) > 1e-9:
            nz_corr = float(np.corrcoef(d_per[nz_mask], s_gt[nz_mask].astype(float))[0, 1])
        else:
            nz_corr = float("nan")
        out[f"s{comp}"] = {
            "valid_bits": int(valid_c.sum()),
            "max_t": float(np.abs(t_score[valid_c]).max()) if valid_c.any() else 0.0,
            "corr": corr,
            "nz_corr": nz_corr,
            "top_k_recall": top_k_recall,
            "raw_bit": m_raw["bit_accuracy"],
            "raw_support": m_raw["support_accuracy"],
            "raw_sign": m_raw["sign_accuracy"],
            "sparse_bit": m_sp["bit_accuracy"],
            "sparse_support": m_sp["support_accuracy"],
            "sparse_support_recall": sp_support_recall,
            "sparse_sign": m_sp["sign_accuracy"],
            "sparse_pred_hw": int(np.count_nonzero(sp)),
        }
        valid_total += int(valid_c.sum())
        if valid_c.any():
            max_t_components = max(max_t_components, float(np.abs(t_score[valid_c]).max()))
        full_correct += int((sp == s_gt).sum())

    if poi_source == "design-window":
        out["n_valid_bits"] = valid_total
        out["max_t"] = max_t_components
    out["full_sk_acc"] = full_correct / float(p_smaug.module_rank * p_smaug.n)
    return out


def baseline_full_sk_random(n: int, hs: int) -> float:
    """sparse_recover (HW=hs constraint) 하 random 예측의 기대 full-sk accuracy.

    n 자리 중 hs 자리를 nonzero 로 random 선택 → 위치 일치율 (1-HG): (n-hs)²/n² + hs²/n².
    nonzero 자리 위에서 sign 일치 1/2 → 부분 가중. 정확:
        E[bit_acc] = P(both zero) + P(both nonzero) · 1/2
                   = ((n-hs)/n)² + (hs/n)² · (1/2)
    """
    if n <= 0:
        return 0.0
    pz = (n - hs) / float(n)
    pn = hs / float(n)
    return pz * pz + pn * pn * 0.5


def baseline_full_sk_all_zero(n: int, hs: int) -> float:
    """unconstrained predict-all-zero baseline. sparse_recover 와는 다른 framework."""
    if n <= 0:
        return 0.0
    return (n - hs) / float(n)


def summarize(results: list[dict]) -> dict:
    """results 의 mean/std/min/max metric 표."""
    metrics = ["full_sk_acc", "n_valid_bits", "max_t"]
    per_comp_metrics = ["corr", "nz_corr", "top_k_recall", "raw_bit",
                       "sparse_bit", "sparse_sign", "sparse_support_recall"]

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
    # component count 는 results 의 module_rank 가 다를 수 있으므로 max 로.
    max_comp = max(int(r.get("module_rank", 2)) for r in results)
    for comp in range(max_comp):
        for pm in per_comp_metrics:
            vals = np.array([r[f"s{comp}"][pm] for r in results
                             if r.get(f"s{comp}") is not None], dtype=float)
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

    schedule = None
    schedule_sk_hash = None
    if args.poi_source == "schedule":
        if args.schedule_source is None:
            raise SystemExit("[ERROR] --poi-source schedule requires --schedule-source")
        if not args.schedule_source.exists():
            raise SystemExit(f"[ERROR] schedule source {args.schedule_source} not found")
        print(f"[schedule] learning from {args.schedule_source.name}")
        schedule = learn_schedule_from_npz(args.schedule_source)
        schedule_sk_hash = schedule["_meta"]["sk_hash_short"]
        n_valid_total = sum(int((schedule[c]["poi"] >= 0).sum())
                            for c in schedule if isinstance(c, int))
        print(f"  source sk={schedule_sk_hash}, valid_bits_total={n_valid_total}")

    results = []
    for p in paths:
        if not p.exists():
            print(f"  [SKIP] {p} not found")
            continue
        if args.poi_source == "schedule":
            d = np.load(p, allow_pickle=True)
            tgt_hash = d["meta"].item()["sk_pke_bytes_hex"][:16]
            if tgt_hash == schedule_sk_hash:
                print(f"  [SKIP] {p.name} — same sk as schedule source (would be µ′-leak)")
                continue
        print(f"  analyzing {p.name} ...")
        r = analyze_one(
            p,
            poi_source=args.poi_source,
            window_start=args.window_start,
            window_end=args.window_end,
            schedule=schedule,
        )
        results.append(r)
        print(f"    [{r['level']} k={r['module_rank']} hs={r['hs']}] sk={r['sk_hash_short']}, "
              f"N/CT={r['n_per_ct']}, valid={r['n_valid_bits']}, max|t|={r['max_t']:.2f}")
        per_comp_strs = []
        for comp in range(int(r['module_rank'])):
            sd = r.get(f"s{comp}") or {}
            per_comp_strs.append(
                f"s[{comp}] bit={sd.get('sparse_bit', 0):.3f}/sign={sd.get('sparse_sign', 0):.3f}"
            )
        print(f"    {' | '.join(per_comp_strs)}, full sk={r['full_sk_acc']:.3f}")

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

    n0 = int(results[0]["module_rank"]) * 256
    hs0 = int(results[0]["hs"]) * int(results[0]["module_rank"])
    rand_b = baseline_full_sk_random(n0, hs0)
    az_b = baseline_full_sk_all_zero(n0, hs0)
    print(f"  [baseline] random (HW={hs0}/{n0} constrained) ≈ {rand_b:.3f}")
    print(f"  [baseline] predict-all-zero (unconstrained)   ≈ {az_b:.3f}")

    max_comp = max(int(r.get("module_rank", 2)) for r in results)
    for comp in range(max_comp):
        sb = summary.get(f"s{comp}.sparse_bit")
        ss = summary.get(f"s{comp}.sparse_sign")
        cc = summary.get(f"s{comp}.corr")
        nzc = summary.get(f"s{comp}.nz_corr")
        tk = summary.get(f"s{comp}.top_k_recall")
        ssr = summary.get(f"s{comp}.sparse_support_recall")
        if sb is None: continue
        print(f"  s[{comp}] corr (all):    mean={cc['mean']:+.3f} ± {cc['std']:.3f}")
        if nzc is not None:
            print(f"  s[{comp}] corr (nz):     mean={nzc['mean']:+.3f} ± {nzc['std']:.3f}")
        if tk is not None:
            print(f"  s[{comp}] top-K recall:  mean={tk['mean']:.3f} ± {tk['std']:.3f}")
        if ssr is not None:
            print(f"  s[{comp}] sup recall:    mean={ssr['mean']:.3f} ± {ssr['std']:.3f}")
        print(f"  s[{comp}] sparse bit:  mean={sb['mean']:.3f} ± {sb['std']:.3f} "
              f"(min {sb['min']:.3f}, max {sb['max']:.3f})")
        print(f"  s[{comp}] sparse sign: mean={ss['mean']:.3f} ± {ss['std']:.3f}")

    # save summary
    out_txt = Path(args.out_prefix + "_summary.txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w") as fh:
        fh.write(f"# Multi-seed evaluation — {len(results)} seeds\n\n")
        for r in results:
            fh.write(f"## [{r['level']} k={r['module_rank']} hs={r['hs']}] "
                     f"seed {r['sk_hash_short']}, N/CT={r['n_per_ct']}\n")
            fh.write(f"  sk HW: {r['sk_hw']}, valid bits: {r['n_valid_bits']}, "
                     f"max|t|={r['max_t']:.2f}\n")
            for comp in range(int(r['module_rank'])):
                if r.get(f"s{comp}") is None: continue
                s = r[f"s{comp}"]
                fh.write(f"  s[{comp}]: corr={s['corr']:+.4f} "
                         f"(nz={s.get('nz_corr', float('nan')):+.4f}), "
                         f"top-K rec={s.get('top_k_recall', 0):.3f}, "
                         f"sup rec={s.get('sparse_support_recall', 0):.3f}, "
                         f"raw bit={s['raw_bit']:.3f}, "
                         f"sparse bit={s['sparse_bit']:.3f}, "
                         f"sign={s['sparse_sign']:.3f}, "
                         f"pred_hw={s['sparse_pred_hw']}\n")
            fh.write(f"  full sk: {r['full_sk_acc']:.3f}\n\n")

        n0 = int(results[0]["module_rank"]) * 256
        hs0 = int(results[0]["hs"]) * int(results[0]["module_rank"])
        rand_b = baseline_full_sk_random(n0, hs0)
        az_b = baseline_full_sk_all_zero(n0, hs0)
        fh.write(f"## Baselines (n={n0}, total HW={hs0})\n")
        fh.write(f"  random with HW=hs constraint:  {rand_b:.4f}\n")
        fh.write(f"  predict-all-zero (no HW cstr): {az_b:.4f}\n")
        fh.write("  µ′-leak instrumented (this code w/ --poi-source mu-prime): 1.0000 (어제 결과)\n\n")

        fh.write("## Summary\n")
        for k, v in summary.items():
            if k == "n_seeds": fh.write(f"  n_seeds: {v}\n"); continue
            fh.write(f"  {k}: mean={v['mean']:.4f} std={v['std']:.4f} "
                     f"min={v['min']:.4f} max={v['max']:.4f} median={v['median']:.4f}\n")
    print(f"\n[OK] {out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
