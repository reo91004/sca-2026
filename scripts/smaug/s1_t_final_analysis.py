#!/usr/bin/env python3
"""S1 최종 분석 — 모든 'T'(c=0,j=0,α=1) capture 통합.

흐름 :
  1. 모든 capture 의 per-sk mean trace 계산.
  2. cross-sk Welch-t 검증 (sk 의존 누설 신호 존재 입증).
  3. per-coord HW CPA + shuffle null + Bonferroni-corrected threshold.
  4. LOO sk recovery 시도 — 단일 coord HW 모델이 성능에 도달 가능한지.
  5. 모든 결과를 results/smaug/s1_final_*.{txt,png} 로 저장.

본 스크립트는 sk_A..J (N=500) + sk_11..sk_30 (N=200) 등 capture 명명 규약에
구애받지 않고, ``s1_main_*c0_j0_a1*.npz`` glob 으로 모두 읽는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from host.smaug import codec as _codec  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s1_main_*c0_j0_a1*.npz")),
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s1_final",
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--n-shuffles", type=int, default=200)
    return p.parse_args()


def _load(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    return data["traces"], data["meta"].item()


def hw_int16(x: int) -> int:
    return bin(int(x) & 0xFFFF).count("1")


def pearson_batch(M_mat: np.ndarray, hw_mat: np.ndarray) -> np.ndarray:
    """returns (K, T) Pearson correlation. NaN for K-rows of zero variance."""
    Mc = M_mat - M_mat.mean(axis=0, keepdims=True)
    Hc = hw_mat - hw_mat.mean(axis=0, keepdims=True)
    Hc = Hc.T  # (K, S)
    num = Hc @ Mc
    norm_h = np.linalg.norm(Hc, axis=1, keepdims=True)
    norm_m = np.linalg.norm(Mc, axis=0, keepdims=True)
    denom = norm_h * norm_m
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / denom
    return np.where(np.isfinite(r), r, np.nan)


def main() -> int:
    args = parse_args()
    if not args.inputs:
        print("[FAIL] no inputs")
        return 1

    means: list[np.ndarray] = []
    sk_ternary_per: list[np.ndarray] = []
    sk_pkfp: list[str] = []
    n_traces_per: list[int] = []
    skipped = 0
    for path in args.inputs:
        try:
            traces, meta = _load(path)
        except Exception as e:
            print(f"[WARN] {path.name} 읽기 실패: {e}")
            skipped += 1
            continue
        if "sk_pke_hex" not in meta:
            print(f"[WARN] {path.name} sk_pke_hex 없음")
            skipped += 1
            continue
        means.append(traces.mean(axis=0).astype(np.float64))
        n_traces_per.append(int(traces.shape[0]))
        full = _codec.unpack_sx(bytes.fromhex(meta["sk_pke_hex"])).astype(np.int64)
        coefs = full[args.component * 256 : (args.component + 1) * 256]
        sk_ternary_per.append(coefs)
        sk_pkfp.append(meta["pk_fp16"])

    if len(means) < 4:
        print(f"[FAIL] need ≥4 sk captures, got {len(means)}")
        return 1
    # 중복 sk 제거
    pkfp_unique: dict[str, int] = {}
    keep = []
    for i, fp in enumerate(sk_pkfp):
        if fp not in pkfp_unique:
            pkfp_unique[fp] = i
            keep.append(i)
    means = [means[i] for i in keep]
    sk_ternary_per = [sk_ternary_per[i] for i in keep]
    sk_pkfp = [sk_pkfp[i] for i in keep]
    n_traces_per = [n_traces_per[i] for i in keep]

    M = np.stack(means, axis=0)
    sk_ternary = np.stack(sk_ternary_per, axis=0)
    S, T = M.shape
    K = sk_ternary.shape[1]
    print(f"[INFO] S={S} unique sk, T={T}, K={K}, n_traces_per_sk={n_traces_per}")

    # HW models
    sk_coef_hw = np.zeros((S, K), dtype=np.int64)
    for s in range(S):
        for k in range(K):
            sk_coef_hw[s, k] = hw_int16(int(sk_ternary[s, k]))

    sk_byte_hw = np.zeros((S, K // 4), dtype=np.int64)
    for s in range(S):
        # reconstruct packed bytes from ternary
        packed = _codec.pack_sx(sk_ternary[s].astype(np.int8))
        bytes_for_comp = packed
        for b_i, byte in enumerate(bytes_for_comp[:K // 4]):
            sk_byte_hw[s, b_i] = bin(int(byte)).count("1")

    # additional: chunk-eval HW model (Toom-Cook eval at t=1)
    # eval_t1[i] = sk[i] + sk[i+64] + sk[i+128] + sk[i+192], i ∈ [0, 63]
    # int16 wrap, popcount
    sk_eval_t1_hw = np.zeros((S, 64), dtype=np.int64)
    for s in range(S):
        for i in range(64):
            v = (
                int(sk_ternary[s, i])
                + int(sk_ternary[s, i + 64])
                + int(sk_ternary[s, i + 128])
                + int(sk_ternary[s, i + 192])
            )
            sk_eval_t1_hw[s, i] = bin(v & 0xFFFF).count("1")

    # ============ A) Pairwise Welch-t — 신호 존재 입증 =================
    # 각 (i, j) 쌍 sk 에 대한 각 sample t 에서 Welch-t. max |t| 의 분포.
    print(
        "\n[A] pairwise Welch-t — first 6 pair 의 max |t| 측정 "
        "(N traces per sk 가 다를 수 있어 SE 계산 시 N 에 의존)"
    )
    # 각 capture 의 traces 도 다시 로드해서 Welch-t 정확히 계산
    pair_max_t = []
    for i in range(min(6, S - 1)):
        j = i + 1  # adjacent pair
        Ai = sk_pkfp[i]
        Aj = sk_pkfp[j]
        # 두 sk 의 traces (재로드 필요) — 메모리 절약 하려면 한 번 만 로드하는데
        # 본 분석에선 cross-sk 검증이 목적이라 재로드 OK
        traces_i = None
        traces_j = None
        for p in args.inputs:
            try:
                tr, mt = _load(p)
            except Exception:
                continue
            if mt.get("pk_fp16") == Ai:
                traces_i = tr
            elif mt.get("pk_fp16") == Aj:
                traces_j = tr
            if traces_i is not None and traces_j is not None:
                break
        if traces_i is None or traces_j is None:
            continue
        ni = traces_i.shape[0]
        nj = traces_j.shape[0]
        ma = traces_i.mean(axis=0)
        mb = traces_j.mean(axis=0)
        sa = traces_i.std(axis=0, ddof=1)
        sb = traces_j.std(axis=0, ddof=1)
        se = np.sqrt(sa**2 / ni + sb**2 / nj)
        t_pair = (ma - mb) / np.where(se == 0, 1, se)
        max_t = float(np.abs(t_pair).max())
        pair_max_t.append(max_t)
        print(
            f"  pair (sk_{i}, sk_{j}) ni={ni} nj={nj}: max |Welch-t|={max_t:.1f}"
        )

    # ============ B) HW CPA + shuffle null — per-coord 신호 검증 ==========
    print(f"\n[B] HW CPA — coef-int16 model, shuffle null ({args.n_shuffles} trials)")
    rho_real = pearson_batch(M, sk_coef_hw)
    real_max = float(np.nanmax(np.abs(rho_real)))
    real_999 = float(np.nanpercentile(np.abs(rho_real), 99.9))

    rng = np.random.default_rng(0xCAFEBABE)
    null_max = []
    null_999 = []
    for _ in range(args.n_shuffles):
        perm = rng.permutation(S)
        rho_sh = pearson_batch(M, sk_coef_hw[perm])
        null_max.append(float(np.nanmax(np.abs(rho_sh))))
        null_999.append(float(np.nanpercentile(np.abs(rho_sh), 99.9)))
    null_max_arr = np.array(null_max)
    null_999_arr = np.array(null_999)
    delta_max = real_max - null_max_arr.mean()
    z_max = delta_max / max(null_max_arr.std(), 1e-9)
    delta_999 = real_999 - null_999_arr.mean()
    z_999 = delta_999 / max(null_999_arr.std(), 1e-9)
    print(f"  real max |rho|={real_max:.5f}, 99.9-pct={real_999:.5f}")
    print(
        f"  null max |rho| (mean±std) = "
        f"{null_max_arr.mean():.5f}±{null_max_arr.std():.5f}, "
        f"99.9-pct null = {null_999_arr.mean():.5f}±{null_999_arr.std():.5f}"
    )
    print(f"  z(real - null mean) for max = {z_max:+.2f}, for 99.9-pct = {z_999:+.2f}")
    if z_999 > 3:
        print("  → 신호 존재 (99.9-pct 가 shuffle null 보다 명확히 높음)")
    else:
        print("  → S 부족 — 더 많은 sk capture 필요")

    # 추가 model: chunk-eval-HW
    rho_real_eval = pearson_batch(M, sk_eval_t1_hw)
    print(
        f"  eval_t1_HW model: real max |rho|={np.nanmax(np.abs(rho_real_eval)):.5f}"
    )

    # ============ C) Bonferroni 한계 — coef-int16 PoI 후보 =================
    # Bonferroni: alpha=0.05, T*K_valid tests. df=S-2.
    K_valid = int(np.sum(sk_coef_hw.var(axis=0, ddof=1) > 0))
    n_tests = T * K_valid
    from scipy import stats  # type: ignore
    alpha = 0.05
    bonf_p = alpha / n_tests
    df = S - 2
    bonf_t = float(stats.t.isf(bonf_p / 2, df))  # two-sided
    bonf_r = float(bonf_t / np.sqrt(df + bonf_t**2))
    print(
        f"\n[C] Bonferroni : T*K_valid={n_tests} tests, df={df}, "
        f"|t|={bonf_t:.2f}, |r|={bonf_r:.4f} threshold"
    )
    n_bonf_pois = int(np.sum(np.abs(rho_real) > bonf_r))
    print(
        f"  real PoI (|r| > {bonf_r:.4f}) count = {n_bonf_pois}  "
        + ("(PoI 발견됨)" if n_bonf_pois > 0 else "(PoI 없음)")
    )
    n_null_avg_pois = float(
        np.mean([
            int(np.sum(np.abs(pearson_batch(M, sk_coef_hw[rng.permutation(S)])) > bonf_r))
            for _ in range(min(20, args.n_shuffles))
        ])
    )
    print(f"  shuffle null average PoI count = {n_null_avg_pois:.1f}")

    # ============ D) LOO sk recovery (best-PoI per coord) ==================
    # 각 hold sk 에 대해 train 에서 best PoI 학습 + test 에 적용
    overall_acc = []
    overall_supp = []
    overall_sign = []
    valid_coef_mask = sk_coef_hw.var(axis=0, ddof=1) > 0
    for hold in range(S):
        train_idx = [s for s in range(S) if s != hold]
        Mt = M[train_idx]
        Ht = sk_coef_hw[train_idx]
        Mc = Mt - Mt.mean(axis=0, keepdims=True)
        Mn = np.linalg.norm(Mc, axis=0, keepdims=True)
        poi_t = -np.ones(K, dtype=np.int64)
        slope = np.zeros(K)
        intercept = np.zeros(K)
        for k in range(K):
            if not valid_coef_mask[k]:
                continue
            h = Ht[:, k].astype(np.float64)
            if h.var(ddof=1) <= 0:
                continue
            Hc = h - h.mean()
            denom = (np.linalg.norm(Hc) * Mn[0])
            denom = np.where(denom == 0, 1.0, denom)
            r = (Hc @ Mc) / denom
            t_best = int(np.argmax(np.abs(r)))
            poi_t[k] = t_best
            cov = ((Mc[:, t_best]) * Hc).sum() / (len(train_idx) - 1)
            b = cov / max(h.var(ddof=1), 1e-12)
            a = Mt[:, t_best].mean() - b * h.mean()
            slope[k] = b
            intercept[k] = a

        Mte = M[hold]
        sk_te = sk_ternary[hold]
        ternary_pred = np.zeros(K, dtype=np.int64)
        for k in range(K):
            if poi_t[k] < 0 or abs(slope[k]) < 1e-12:
                ternary_pred[k] = 0
                continue
            hw_p = (Mte[poi_t[k]] - intercept[k]) / slope[k]
            hw_round = int(round(hw_p))
            # nearest-cluster mapping
            ternary_pred[k] = min(
                {0: 0, 1: 1, 16: -1}.items(),
                key=lambda kv: abs(kv[0] - hw_round),
            )[1]

        coord_acc = float((ternary_pred == sk_te).mean())
        supp_mask = sk_te != 0
        if supp_mask.any():
            supp_acc = float(((ternary_pred != 0) == supp_mask).mean())
            sign_mask = supp_mask & (ternary_pred != 0)
            if sign_mask.any():
                sign_acc = float(
                    (np.sign(ternary_pred[sign_mask]) == np.sign(sk_te[sign_mask])).mean()
                )
            else:
                sign_acc = 0.0
        else:
            supp_acc = 1.0
            sign_acc = 1.0
        overall_acc.append(coord_acc)
        overall_supp.append(supp_acc)
        overall_sign.append(sign_acc)

    mean_coord = float(np.mean(overall_acc))
    mean_supp = float(np.mean(overall_supp))
    mean_sign = float(np.mean(overall_sign))

    # baseline: predict all 0 — 정답률 = ratio of true zeros
    zero_baseline = float((sk_ternary == 0).mean())
    print(
        f"\n[D] LOO single-coord HW recovery: "
        f"coord_acc={mean_coord:.4f}  supp_acc={mean_supp:.4f}  "
        f"sign_acc={mean_sign:.4f}  (baseline pred-zero = {zero_baseline:.4f})"
    )

    # ============ E) summary report =================
    summary = [
        f"S1 final analysis report",
        f"",
        f"S={S} unique sk, T={T}, K={K}",
        f"n_traces per sk: min={min(n_traces_per)}, max={max(n_traces_per)}",
        f"",
        f"[A] pairwise Welch-t max:",
    ]
    for i, t in enumerate(pair_max_t):
        summary.append(f"  pair {i}: |t|max = {t:.1f}")
    summary += [
        f"",
        f"[B] cross-sk HW CPA (coef-int16 model):",
        f"  real max |rho| = {real_max:.5f}",
        f"  shuffle null max |rho| (mean±std) = "
        f"{null_max_arr.mean():.5f}±{null_max_arr.std():.5f}",
        f"  z(real - null mean) for max = {z_max:+.2f}",
        f"  z(real - null mean) for 99.9-pct = {z_999:+.2f}",
        f"  → {'신호 존재' if z_999 > 3 else 'S 부족 — 더 많은 sk 필요'}",
        f"",
        f"[C] Bonferroni-corrected PoI:",
        f"  threshold |r| > {bonf_r:.4f}",
        f"  real PoI count = {n_bonf_pois}",
        f"  null PoI count avg = {n_null_avg_pois:.1f}",
        f"",
        f"[D] LOO single-coord HW recovery:",
        f"  mean coord_acc = {mean_coord:.4f}",
        f"  mean supp_acc = {mean_supp:.4f}",
        f"  mean sign_acc = {mean_sign:.4f}",
        f"  baseline predict-zero = {zero_baseline:.4f}",
        f"  → {'복구 ↑' if mean_coord > zero_baseline + 0.05 else '단일 좌표 HW 모델 부족'}",
    ]
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_text = args.out_prefix.with_suffix(".txt")
    out_text.write_text("\n".join(summary) + "\n")
    print(f"\n[OK] {out_text}")

    # plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))
    axes[0].plot(M.T, lw=0.3, alpha=0.6)
    axes[0].set_title(f"per-sk mean traces (S={S})")
    axes[0].set_xlabel("sample")

    max_per_t = np.nanmax(np.abs(rho_real), axis=0)
    axes[1].plot(max_per_t, lw=0.4, label="real max |rho| per t")
    axes[1].axhline(bonf_r, color="r", ls="--", lw=0.6, label=f"Bonf |r|={bonf_r:.3f}")
    axes[1].axhline(null_999_arr.mean(), color="k", ls=":", lw=0.5,
                    label=f"null 99.9-pct mean={null_999_arr.mean():.3f}")
    axes[1].set_title("max |rho| over coefs per sample t")
    axes[1].legend()

    abs_rho = np.abs(rho_real)
    abs_rho = np.where(np.isnan(abs_rho), 0, abs_rho)
    im = axes[2].imshow(
        abs_rho, aspect="auto", cmap="viridis", origin="lower",
        extent=[0, T, 0, K],
    )
    axes[2].set_title("|rho| heatmap (coef k vs sample t)")
    axes[2].set_xlabel("sample")
    axes[2].set_ylabel("sk coef k")
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=120)
    print(f"[OK] {args.out_prefix.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
