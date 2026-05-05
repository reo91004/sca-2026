#!/usr/bin/env python3
"""S2 — 'Z' (chosen-CT indcpa_dec) cross-sk Welch-t + Bonferroni HW CPA.

This script tests candidate leakage models for the attack-valid proxy trace.
Do not treat any model below as an established implementation fact without a
fresh disassembly check. In particular, the current 'V' firmware replay passes
the secret polyvec to vec_vec_mult_add unshifted while shifting c1/c2 buffers, so
the shifted-sk model is retained only as a legacy/alternative hypothesis.

Candidate models:
  M1) HW(sk_int16): raw ternary int16 values {-1, 0, +1}.
  M2) HW(sk_int16 << 11): legacy shifted-secret hypothesis.
  M3) byte-level versions of M2.
  M4) HW of a four-coefficient evaluation sum.

Bonferroni threshold + shuffle null comparison decide whether a candidate model
extracts coordinate-level signal beyond chance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from host.smaug import codec as _codec  # noqa: E402


def _load(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    return data["traces"], data["meta"].item()


def hw_int16(x: int) -> int:
    return bin(int(x) & 0xFFFF).count("1")


def pearson_batch(M_mat: np.ndarray, hw_mat: np.ndarray) -> np.ndarray:
    Mc = M_mat - M_mat.mean(axis=0, keepdims=True)
    Hc = hw_mat - hw_mat.mean(axis=0, keepdims=True)
    Hc = Hc.T
    num = Hc @ Mc
    norm_h = np.linalg.norm(Hc, axis=1, keepdims=True)
    norm_m = np.linalg.norm(Mc, axis=0, keepdims=True)
    denom = norm_h * norm_m
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / denom
    return np.where(np.isfinite(r), r, np.nan)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s2_z_sk*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s2_z_final",
    )
    p.add_argument("--n-shuffles", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    means: list[np.ndarray] = []
    sk_t_per: list[np.ndarray] = []
    pkfps: list[str] = []
    n_traces_per: list[int] = []
    for path in args.inputs:
        try:
            traces, meta = _load(path)
        except Exception as e:
            print(f"[WARN] {path.name} 읽기 fail: {e}")
            continue
        if meta.get("pk_fp16") in pkfps:
            continue  # dedup
        if meta.get("cmd") != "Z":
            continue
        means.append(traces.mean(axis=0).astype(np.float64))
        n_traces_per.append(int(traces.shape[0]))
        full = _codec.unpack_sx(bytes.fromhex(meta["sk_pke_hex"])).astype(np.int64)
        sk_t_per.append(full[args.component * 256 : (args.component + 1) * 256])
        pkfps.append(meta["pk_fp16"])

    if len(means) < 4:
        print(f"[FAIL] need ≥4 unique sks, got {len(means)}")
        return 1
    M = np.stack(means, axis=0)
    sk_ternary = np.stack(sk_t_per, axis=0)
    S, T = M.shape
    K = sk_ternary.shape[1]
    print(f"[INFO] S={S} unique sks, T={T}, K={K}, n_traces avg={np.mean(n_traces_per):.0f}")

    # HW models
    hw_raw = np.zeros((S, K), dtype=np.int64)         # M1
    hw_shift = np.zeros((S, K), dtype=np.int64)       # M2 (indcpa_dec scaling)
    hw_lobyte = np.zeros((S, K), dtype=np.int64)      # M3 low byte
    hw_hibyte = np.zeros((S, K), dtype=np.int64)      # M3 high byte
    for s in range(S):
        for k in range(K):
            v = int(sk_ternary[s, k])
            v_int16 = v & 0xFFFF
            v_shift = (v << 11) & 0xFFFF  # left shift by 11, int16 wrap
            hw_raw[s, k] = bin(v_int16).count("1")
            hw_shift[s, k] = bin(v_shift).count("1")
            hw_lobyte[s, k] = bin(v_shift & 0xFF).count("1")
            hw_hibyte[s, k] = bin((v_shift >> 8) & 0xFF).count("1")

    # eval-t1 model (4-sum chunk eval)
    hw_eval_t1 = np.zeros((S, 64), dtype=np.int64)
    for s in range(S):
        for i in range(64):
            v = (
                int(sk_ternary[s, i]) + int(sk_ternary[s, i + 64])
                + int(sk_ternary[s, i + 128]) + int(sk_ternary[s, i + 192])
            )
            v_shift = (v << 11) & 0xFFFF
            hw_eval_t1[s, i] = bin(v_shift).count("1")

    rng = np.random.default_rng(0xCAFEBABE)
    summary_lines = [
        f"S2 'Z' analysis",
        f"S={S}, T={T}, K={K}, alpha=4 (4·X^0 component 0)",
        f"",
    ]

    def evaluate(hw_mat: np.ndarray, label: str) -> dict:
        valid = hw_mat.var(axis=0, ddof=1) > 0
        if not valid.any():
            print(f"  [{label}] no variance — skipped")
            return {}
        sk_v = hw_mat[:, valid]
        rho_real = pearson_batch(M, sk_v)
        real_max = float(np.nanmax(np.abs(rho_real)))
        real_999 = float(np.nanpercentile(np.abs(rho_real), 99.9))
        # null
        null_max = []
        null_999 = []
        for _ in range(args.n_shuffles):
            perm = rng.permutation(S)
            r = pearson_batch(M, sk_v[perm])
            null_max.append(float(np.nanmax(np.abs(r))))
            null_999.append(float(np.nanpercentile(np.abs(r), 99.9)))
        nm = np.array(null_max)
        n9 = np.array(null_999)
        z_max = (real_max - nm.mean()) / max(nm.std(), 1e-9)
        z_999 = (real_999 - n9.mean()) / max(n9.std(), 1e-9)
        # Bonferroni r threshold
        from scipy import stats  # type: ignore
        n_tests = T * int(valid.sum())
        bonf_p = 0.05 / n_tests
        df = S - 2
        bonf_t = float(stats.t.isf(bonf_p / 2, df))
        bonf_r = float(bonf_t / np.sqrt(df + bonf_t**2))
        n_pois = int(np.sum(np.abs(rho_real) > bonf_r))
        # null PoI count avg
        n_null_avg = float(
            np.mean([
                int(np.sum(np.abs(pearson_batch(M, sk_v[rng.permutation(S)])) > bonf_r))
                for _ in range(min(20, args.n_shuffles))
            ])
        )
        print(
            f"  [{label}] real max={real_max:.4f} 99.9-pct={real_999:.4f}, "
            f"null_max_mean={nm.mean():.4f}, z(max)={z_max:+.2f} z(99.9)={z_999:+.2f}"
        )
        print(
            f"  [{label}] Bonferroni |r|>{bonf_r:.4f}: real PoI={n_pois}, null avg={n_null_avg:.1f}"
        )
        summary_lines.append(
            f"[{label}] real max={real_max:.4f} 99.9-pct={real_999:.4f}, "
            f"z(99.9)={z_999:+.2f}, Bonf-PoI real={n_pois} null={n_null_avg:.1f}"
        )
        return dict(
            rho_real=rho_real, label=label, valid=valid,
            real_max=real_max, real_999=real_999,
            null_max_mean=nm.mean(), z_999=z_999, n_pois=n_pois, bonf_r=bonf_r
        )

    print(f"\n[A] HW model evaluation (shuffle null {args.n_shuffles} trials)")
    res_raw = evaluate(hw_raw, "M1: HW(sk_int16) raw")
    res_shift = evaluate(hw_shift, "M2: HW(sk<<11) legacy hypothesis")
    res_lo = evaluate(hw_lobyte, "M3a: HW(low byte sk<<11) legacy")
    res_hi = evaluate(hw_hibyte, "M3b: HW(high byte sk<<11) legacy")
    res_eval = evaluate(hw_eval_t1, "M4: HW(eval_t1 sum<<11)")

    # plot
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(M.mean(axis=0), lw=0.4, color="C7")
    for s in range(S):
        axes[0].plot(M[s], lw=0.2, alpha=0.4)
    axes[0].set_title(f"per-sk mean traces (S={S}, 'Z' chosen-CT c0=4·X^0)")

    for i, (res, color) in enumerate([
        (res_raw, "C0"),
        (res_shift, "C1"),
        (res_eval, "C2"),
    ]):
        if not res:
            continue
        rho = res["rho_real"]
        max_per_t = np.nanmax(np.abs(rho), axis=0)
        axes[i + 1].plot(max_per_t, lw=0.4, color=color, label=res["label"])
        axes[i + 1].axhline(res["bonf_r"], color="r", ls="--", lw=0.5, alpha=0.5)
        axes[i + 1].set_title(
            f"{res['label']}: max |rho| per t  "
            f"(Bonf={res['bonf_r']:.3f}, PoI={res['n_pois']})"
        )
        axes[i + 1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=120)
    print(f"\n[OK] {args.out_prefix.with_suffix('.png')}")

    args.out_prefix.with_suffix(".txt").write_text("\n".join(summary_lines) + "\n")
    print(f"[OK] {args.out_prefix.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
