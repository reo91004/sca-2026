#!/usr/bin/env python3
"""E5 — FFT-magnitude alignment-invariant design-window.

배경
----
E1+E3 결과: 다른 sk 사이는 raw trace 에서 alignment 깨짐. DTW 후도 cross-key
transfer 0.55-0.58 (random). 단순 shift 가 아닌 layout-level 차이.

가설 (FFT shift theorem)
-----------------------
|FFT(x[t-d])| = |FFT(x[t])| 이므로 raw trace 의 *shift* 는 FFT magnitude 에서
사라진다. 추가로 알고리즘의 *spectral envelope* (Toom-Cook periodicity, rounding
loop frequency) 은 sk-invariant 일 가능성.

따라서:
- design-window PoI 를 raw trace 가 아닌 |FFT(trace)| 위에서 학습/적용.
- per-bin Welch-t (α=64 vs α=192).
- 결과 score d_i 가 sk_i 와 cross-key 상관되면 alignment-invariant SCA path 확보.

평가 (정합한 threat model)
--------------------------
- *Offline*: calibration dataset 1개 (또는 leave-one-out 이어도 됨). target 은
  *제외*. target 의 sk/µ′ 사용 금지.
- *Online*: target dataset 의 (chosen-CT design label, FFT|trace|) 만 사용.
- 평가: held-out target 의 full_sk_acc (random baseline 0.566 대비).

이전 mistake 회피
-----------------
1. PoI 학습 시 µ′ 라벨 사용 금지 → α=pos vs α=neg design label 만.
2. Calibration ≠ target 강제 → 같은 sk 검증 (skip).
3. Random baseline (constrained-HW=hs) 명시 비교.

산출
----
- results/E5_fft_design_window.txt
- results/E5_fft_perbit_corr.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.analysis import sparse_recover  # noqa: E402
from host.smaug import params as _params  # noqa: E402
from host.smaug.codec import unpack_sx  # noqa: E402


def fft_mag(traces: np.ndarray) -> np.ndarray:
    """N×T traces → N×(T//2+1) magnitude."""
    return np.abs(np.fft.rfft(traces, axis=1)).astype(np.float32)


def welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.zeros(a.shape[1], dtype=np.float64)
    ma = a.mean(axis=0); mb = b.mean(axis=0)
    va = a.var(axis=0, ddof=1).clip(min=1e-12)
    vb = b.var(axis=0, ddof=1).clip(min=1e-12)
    return (ma - mb) / np.sqrt(va / a.shape[0] + vb / b.shape[0])


def design_window_fft_poi(F_pos: np.ndarray, F_neg: np.ndarray,
                          n_bits: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """FFT magnitude 위 design-window PoI.

    F_pos, F_neg: (N, n_freq) FFT magnitude per design.
    Returns: poi (n_bits,) — selected freq bin, t_score (n_bits,).
    """
    n_freq = F_pos.shape[1]
    if n_freq < n_bits:
        raise ValueError(f"n_freq {n_freq} < n_bits {n_bits}")
    t = welch_t(F_pos, F_neg)
    edges = np.linspace(0, n_freq, n_bits + 1, dtype=int)
    poi = np.zeros(n_bits, dtype=int)
    score = np.zeros(n_bits, dtype=np.float64)
    for bi in range(n_bits):
        lo, hi = edges[bi], edges[bi + 1]
        if hi <= lo:
            continue
        local = t[lo:hi]
        off = int(np.argmax(np.abs(local)))
        poi[bi] = lo + off
        score[bi] = float(local[off])
    return poi, score


def attack_one(F: np.ndarray, label_alpha: np.ndarray, label_comp: np.ndarray,
               sk: np.ndarray, p, alpha_pos: int, alpha_neg: int) -> dict:
    """FFT-domain design-window attack on a single dataset (per component)."""
    full_correct = 0
    out = {}
    for comp in range(p.module_rank):
        mp = (label_comp == comp) & (label_alpha == alpha_pos)
        mn = (label_comp == comp) & (label_alpha == alpha_neg)
        if not (mp.any() and mn.any()):
            out[f"s{comp}"] = None
            continue
        poi, score = design_window_fft_poi(F[mp], F[mn], n_bits=p.n)
        # signed score per "bit"
        d = np.zeros(p.n)
        diff = F[mp].mean(axis=0) - F[mn].mean(axis=0)
        for bi in range(p.n):
            d[bi] = float(diff[poi[bi]])
        s_gt = sk[comp].astype(np.int8)
        sigma = max(float(d.std()), 1e-9)
        # posterior + sparse_recover
        pos = np.abs(d[d > 0]); neg = np.abs(d[d < 0])
        mu = ((pos.mean() if pos.size else 1.0) + (neg.mean() if neg.size else 1.0)) / 2.0
        ln = -0.5 * ((d + mu) / sigma) ** 2
        lz = -0.5 * (d / sigma) ** 2
        lp = -0.5 * ((d - mu) / sigma) ** 2
        L = np.stack([ln, lz, lp], axis=1)
        L -= L.max(axis=1, keepdims=True)
        P = np.exp(L); P /= P.sum(axis=1, keepdims=True)
        sp = sparse_recover.greedy_sparse_recover(P, p.hs)
        m = sparse_recover.accuracy(sp, s_gt)
        # additional metrics
        order = np.argsort(-np.abs(d), kind="stable")
        true_nz = set(np.flatnonzero(s_gt).tolist())
        top_k = set(order[:p.hs].tolist())
        top_k_recall = len(top_k & true_nz) / float(p.hs)
        corr = float(np.corrcoef(d, s_gt.astype(float))[0, 1]) if d.std() > 0 else float("nan")
        out[f"s{comp}"] = {
            "max_abs_t": float(np.abs(score).max()),
            "n_freq_bins_covered": int((poi > 0).sum()),
            "corr": corr,
            "top_k_recall": top_k_recall,
            "sparse_bit": m["bit_accuracy"],
            "sparse_sign": m["sign_accuracy"],
        }
        full_correct += int((sp == s_gt).sum())
    out["full_sk_acc"] = full_correct / float(p.module_rank * p.n)
    return out


def cross_key_eval(calib_path: Path, target_path: Path) -> dict:
    """schedule (poi per coef per comp) 를 calib 에서 학습 → target 에 적용.

    threat model: target sk/µ′ 사용 안 함. calib sk 만 알려져 있다.
    """
    dc = np.load(calib_path, allow_pickle=True); dt = np.load(target_path, allow_pickle=True)
    Tc = dc["traces"]; Tt = dt["traces"]
    metac = dc["meta"].item(); metat = dt["meta"].item()
    p = _params.get(metac.get("level", "smaug1"))
    ap = int(metac.get("alpha_pos", 64)); an = int(metac.get("alpha_neg", 192))
    lac = np.asarray(metac["label_alpha"]); lcc = np.asarray(metac["label_component"])
    lat = np.asarray(metat["label_alpha"]); lct = np.asarray(metat["label_component"])
    sk_t = unpack_sx(bytes.fromhex(metat["sk_pke_bytes_hex"])
                     ).reshape(p.module_rank, p.n).astype(np.int64)

    Fc = fft_mag(Tc); Ft = fft_mag(Tt)

    full_correct = 0
    out = {}
    for comp in range(p.module_rank):
        mpc = (lcc == comp) & (lac == ap); mnc = (lcc == comp) & (lac == an)
        mpt = (lct == comp) & (lat == ap); mnt = (lct == comp) & (lat == an)
        # Schedule from calib
        poi_c, score_c = design_window_fft_poi(Fc[mpc], Fc[mnc], n_bits=p.n)
        # Apply to target
        diff_t = Ft[mpt].mean(axis=0) - Ft[mnt].mean(axis=0)
        d = np.array([float(diff_t[poi_c[bi]]) for bi in range(p.n)])
        s_gt = sk_t[comp].astype(np.int8)
        # posterior + sparse
        sigma = max(float(d.std()), 1e-9)
        pos = np.abs(d[d > 0]); neg = np.abs(d[d < 0])
        mu = ((pos.mean() if pos.size else 1.0) + (neg.mean() if neg.size else 1.0)) / 2.0
        ln = -0.5 * ((d + mu) / sigma) ** 2
        lz = -0.5 * (d / sigma) ** 2
        lp = -0.5 * ((d - mu) / sigma) ** 2
        L = np.stack([ln, lz, lp], axis=1); L -= L.max(axis=1, keepdims=True)
        P = np.exp(L); P /= P.sum(axis=1, keepdims=True)
        sp = sparse_recover.greedy_sparse_recover(P, p.hs)
        m = sparse_recover.accuracy(sp, s_gt)
        order = np.argsort(-np.abs(d), kind="stable")
        true_nz = set(np.flatnonzero(s_gt).tolist())
        top_k_recall = len(set(order[:p.hs].tolist()) & true_nz) / float(p.hs)
        corr = float(np.corrcoef(d, s_gt.astype(float))[0, 1]) if d.std() > 0 else float("nan")
        out[f"s{comp}"] = {
            "max_abs_t": float(np.abs(score_c).max()),
            "corr": corr,
            "top_k_recall": top_k_recall,
            "sparse_bit": m["bit_accuracy"],
            "sparse_sign": m["sign_accuracy"],
        }
        full_correct += int((sp == s_gt).sum())
    out["full_sk_acc"] = full_correct / float(p.module_rank * p.n)
    return out


def baseline_random(n: int, hs: int) -> float:
    pz = (n - hs) / float(n); pn = hs / float(n)
    return pz * pz + pn * pn * 0.5


def main() -> int:
    paths = sorted(Path("traces").glob("attack_seed*.npz"))
    paths = [p for p in paths if "smaug3" not in p.name and "smaug5" not in p.name]
    paths += [Path("traces/attack_const_c1_n256.npz")]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("[ERROR] no smaug1 attack seeds")

    p_smaug = _params.get("smaug1")
    rand = baseline_random(p_smaug.module_rank * p_smaug.n,
                            p_smaug.hs * p_smaug.module_rank)
    print(f"[E5] {len(paths)} smaug1 datasets, hs={p_smaug.hs}, n={p_smaug.n}")
    print(f"[E5] random baseline (HW=140/512 cstr) ≈ {rand:.4f}\n")

    out_lines = [f"# E5 — FFT-magnitude alignment-invariant design-window\n"]
    out_lines.append(f"random baseline ≈ {rand:.4f}\n\n")

    # === 1. Per-dataset in-key sanity (FFT design-window applied to itself) ===
    print("[1] in-key sanity — FFT design-window self-apply (calib=target=same)")
    out_lines.append("## 1. In-key sanity (calib=target, sk leak path)\n")
    for path in paths:
        d = np.load(path, allow_pickle=True)
        T = d["traces"]; meta = d["meta"].item()
        ap = int(meta.get("alpha_pos", 64)); an = int(meta.get("alpha_neg", 192))
        la = np.asarray(meta["label_alpha"]); lc = np.asarray(meta["label_component"])
        sk = unpack_sx(bytes.fromhex(meta["sk_pke_bytes_hex"])
                       ).reshape(p_smaug.module_rank, p_smaug.n).astype(np.int64)
        F = fft_mag(T)
        r = attack_one(F, la, lc, sk, p_smaug, ap, an)
        sk_short = meta["sk_pke_bytes_hex"][:16]
        line = f"  {path.name:35s} sk={sk_short} full_sk={r['full_sk_acc']:.4f}"
        for c in range(p_smaug.module_rank):
            sd = r.get(f"s{c}")
            if sd:
                line += f" | s[{c}] corr={sd['corr']:+.3f} top-K={sd['top_k_recall']:.2f}"
        print(line)
        out_lines.append(line + "\n")

    # === 2. Cross-key schedule transfer (calib=seed1, target=others) ===
    print("\n[2] cross-key FFT schedule transfer (calib=seed1, target=others)")
    out_lines.append("\n## 2. Cross-key schedule transfer (FFT)\n")
    calib = paths[0]
    out_lines.append(f"calib = {calib.name}\n")
    accs = []
    for tgt in paths[1:]:
        r = cross_key_eval(calib, tgt)
        accs.append(r["full_sk_acc"])
        line = f"  target {tgt.name:35s} full_sk={r['full_sk_acc']:.4f}"
        for c in range(p_smaug.module_rank):
            sd = r.get(f"s{c}")
            if sd:
                line += f" | s[{c}] corr={sd['corr']:+.3f} top-K={sd['top_k_recall']:.2f} sign={sd['sparse_sign']:.2f}"
        print(line)
        out_lines.append(line + "\n")
    print(f"\n  cross-key mean full_sk: {np.mean(accs):.4f} (random ≈ {rand:.4f})")
    out_lines.append(f"\n  cross-key mean full_sk: {np.mean(accs):.4f} "
                     f"(random ≈ {rand:.4f})\n")

    # === 3. Conclusion ===
    print("\n[3] Verdict")
    if np.mean(accs) > rand + 0.05:
        verdict = "✓ FFT magnitude shows non-random cross-key signal — promising"
    elif np.mean(accs) > rand + 0.01:
        verdict = "△ marginal cross-key signal (under noise tolerance)"
    else:
        verdict = "✗ FFT magnitude does not transfer cross-key (= even spectral envelope is sk-dependent)"
    print(f"  {verdict}")
    out_lines.append(f"\n## 3. Verdict\n  {verdict}\n")

    Path("results").mkdir(exist_ok=True)
    Path("results/E5_fft_design_window.txt").write_text("".join(out_lines))
    print(f"\n[OK] results/E5_fft_design_window.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
