#!/usr/bin/env python3
"""S4 paired-trace binary mu'-change oracle.

For a fixed public c1 and adjacent c2 values (c2=a, c2=a+delta), check whether
the per-trace pair distance, evaluated on a held-out target key, classifies
"mu' changed" vs "mu' did not change" above the permutation null.

This is intentionally simpler than the s4_c2_pair_analyze regression: the
classifier only needs to detect *whether* mu' flipped, not predict its byte HW.
The label is a single bit per (key, pair):

    flip_any[k, p] = int(mu'(sk_k, c1_p, c2=a) != mu'(sk_k, c1_p, c2=a+delta))

The flip label is computed entirely from profiling secrets and public ciphertext
metadata. The target key's response bytes are never consulted. Held-out evaluation
uses leave-one-key-out folds; the permutation null shuffles per-key label
vectors across keys.

Primary metric: held-out AUROC. Secondary regression metrics (rMAE/corr against
flip_byte_hw) are reported alongside for context.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from host.smaug.params import SMAUG1  # noqa: E402
from scripts.s2_z_lowdim_analyze import (  # noqa: E402
    Config,
    Dataset,
    fit_predict_ridge,
    label_from_mu_bits,
    load_dataset,
    make_observation_features,
    parse_sample_range,
    select_features,
)


@dataclass(frozen=True)
class PairDataset:
    diff_xmean: np.ndarray   # (S, P, B) per-pair mean trace diff
    xvar: np.ndarray         # (S, P, B)
    counts: np.ndarray       # (S, P)
    sks: np.ndarray          # (S, 256)
    base_terms: list[list[tuple[int, int]]]
    base_c2: list[int]
    delta_c2: list[int]
    pkfps: list[str]


def _terms_key(terms: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(c), int(a)) for c, a in terms)


def build_pair_dataset(
    ds: Dataset,
    cfg: Config,
    c2_delta: int,
    *,
    absolute_diff: bool,
) -> PairDataset:
    xmean, xvar, counts = make_observation_features(ds.traces, cfg.block)
    by_design: dict[tuple[tuple[tuple[int, int], ...], int], int] = {}
    for i, (terms, c2) in enumerate(zip(ds.design_terms, ds.design_c2)):
        by_design[(_terms_key(terms), int(c2))] = i

    pairs: list[tuple[int, int]] = []
    base_terms: list[list[tuple[int, int]]] = []
    base_c2: list[int] = []
    delta_c2: list[int] = []
    for i, (terms, c2) in enumerate(zip(ds.design_terms, ds.design_c2)):
        target_c2 = (int(c2) + int(c2_delta)) % SMAUG1.p2
        j = by_design.get((_terms_key(terms), target_c2))
        if j is None:
            continue
        pairs.append((i, j))
        base_terms.append(terms)
        base_c2.append(int(c2))
        delta_c2.append(target_c2)

    if not pairs:
        raise ValueError(f"no same-c1 c2 pairs found for delta={c2_delta}")

    base_idx = np.asarray([i for i, _ in pairs], dtype=np.int64)
    delta_idx = np.asarray([j for _, j in pairs], dtype=np.int64)
    diff = xmean[:, delta_idx, :] - xmean[:, base_idx, :]
    if absolute_diff:
        diff = np.abs(diff)
    return PairDataset(
        diff_xmean=diff,
        xvar=xvar[:, delta_idx, :] + xvar[:, base_idx, :],
        counts=np.minimum(counts[:, delta_idx], counts[:, base_idx]),
        sks=ds.sks,
        base_terms=base_terms,
        base_c2=base_c2,
        delta_c2=delta_c2,
        pkfps=ds.pkfps,
    )


def labels_flip_any(pds: PairDataset) -> np.ndarray:
    """(S, P) binary: did *any* mu' bit flip between c2=a and c2=b?"""
    out = np.zeros((pds.sks.shape[0], len(pds.base_terms)), dtype=np.float64)
    for k in range(pds.sks.shape[0]):
        for p, (terms, c2_a, c2_b) in enumerate(
            zip(pds.base_terms, pds.base_c2, pds.delta_c2)
        ):
            ba = label_from_mu_bits(pds.sks[k], terms, c2_a, "mu_bit").astype(np.int8)
            bb = label_from_mu_bits(pds.sks[k], terms, c2_b, "mu_bit").astype(np.int8)
            out[k, p] = float(np.any(np.bitwise_xor(ba, bb)))
    return out


def labels_flip_byte_hw(pds: PairDataset) -> np.ndarray:
    """(S, P, 32) byte-HW of XOR(mu'(c2=a), mu'(c2=b)). Used for regression aux."""
    out = np.zeros((pds.sks.shape[0], len(pds.base_terms), 32), dtype=np.float64)
    for k in range(pds.sks.shape[0]):
        for p, (terms, c2_a, c2_b) in enumerate(
            zip(pds.base_terms, pds.base_c2, pds.delta_c2)
        ):
            ba = label_from_mu_bits(pds.sks[k], terms, c2_a, "mu_bit").astype(np.int8)
            bb = label_from_mu_bits(pds.sks[k], terms, c2_b, "mu_bit").astype(np.int8)
            flip = np.bitwise_xor(ba, bb).astype(np.float64)
            out[k, p, :] = flip.reshape(32, 8).sum(axis=1)
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney rank-sum AUROC. Returns NaN if a class is empty."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(np.int64).ravel()
    if s.size != y.size:
        raise ValueError(f"size mismatch: scores={s.size} labels={y.size}")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_oracle(
    pds: PairDataset,
    cfg: Config,
    *,
    key_perm: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    y_bin = labels_flip_any(pds)              # (S, P)
    y_byte = labels_flip_byte_hw(pds)         # (S, P, 32)
    s, p, b = pds.diff_xmean.shape

    y_bin_train_src = y_bin if key_perm is None else y_bin[key_perm]
    y_byte_train_src = y_byte if key_perm is None else y_byte[key_perm]

    bin_preds: list[np.ndarray] = []
    bin_trues: list[np.ndarray] = []
    byte_preds: list[np.ndarray] = []
    byte_trues: list[np.ndarray] = []
    feature_counts = np.zeros(b, dtype=np.int64)
    per_key_auroc = np.full(s, np.nan, dtype=np.float64)

    for hold in range(s):
        train_keys = np.asarray([i for i in range(s) if i != hold], dtype=np.int64)
        x_train = pds.diff_xmean[train_keys].reshape(-1, b)
        v_train = pds.xvar[train_keys].reshape(-1, b)
        c_train = pds.counts[train_keys].reshape(-1)
        y_train_bin = y_bin_train_src[train_keys].reshape(-1, 1)
        y_train_byte = y_byte_train_src[train_keys].reshape(-1, 32)
        y_train_select = np.concatenate([y_train_bin, y_train_byte], axis=1)
        x_test = pds.diff_xmean[[hold]].reshape(p, b)
        y_test_bin = y_bin[[hold]].reshape(p, 1)
        y_test_byte = y_byte[[hold]].reshape(p, 32)

        features = select_features(x_train, v_train, c_train, y_train_select, cfg)
        feature_counts[features] += 1
        pred_bin = fit_predict_ridge(x_train, y_train_bin, x_test, features, cfg.ridge)
        pred_byte = fit_predict_ridge(x_train, y_train_byte, x_test, features, cfg.ridge)

        bin_preds.append(pred_bin)
        bin_trues.append(y_test_bin)
        byte_preds.append(pred_byte)
        byte_trues.append(y_test_byte)

        per_key_auroc[hold] = auroc(pred_bin.ravel(), y_test_bin.ravel())

    yp_bin = np.concatenate(bin_preds, axis=0).ravel()
    yt_bin = np.concatenate(bin_trues, axis=0).ravel()
    yp_byte = np.concatenate(byte_preds, axis=0)
    yt_byte = np.concatenate(byte_trues, axis=0)

    pooled_auroc = auroc(yp_bin, yt_bin)
    valid = ~np.isnan(per_key_auroc)
    per_key_mean = float(np.nanmean(per_key_auroc)) if valid.any() else float("nan")

    lo, hi = float(np.min(yt_byte)), float(np.max(yt_byte))
    pred_round = np.clip(np.rint(yp_byte), lo, hi)
    rounded_mae = float(np.mean(np.abs(pred_round - yt_byte)))
    exact_byte = float(np.mean(pred_round == yt_byte))
    if np.std(yp_byte) > 1e-12 and np.std(yt_byte) > 1e-12:
        corr = float(np.corrcoef(yp_byte.ravel(), yt_byte.ravel())[0, 1])
    else:
        corr = 0.0

    return {
        "pooled_auroc": float(pooled_auroc),
        "mean_key_auroc": per_key_mean,
        "per_key_auroc": per_key_auroc,
        "rounded_mae": rounded_mae,
        "exact_byte": exact_byte,
        "corr": corr,
        "pred_bin": yp_bin,
        "true_bin": yt_bin,
        "pred_byte": yp_byte,
        "true_byte": yt_byte,
        "feature_counts": feature_counts,
    }


def class_balance(y: np.ndarray) -> tuple[float, int, int]:
    yf = np.asarray(y).astype(np.int64).ravel()
    n_pos = int((yf == 1).sum())
    n_neg = int((yf == 0).sum())
    n = n_pos + n_neg
    return (n_pos / n if n else float("nan"), n_pos, n_neg)


def summarize(
    cfg: Config,
    real: dict[str, float | np.ndarray],
    nulls: list[dict[str, float | np.ndarray]],
) -> str:
    null_pooled = np.asarray([n["pooled_auroc"] for n in nulls], dtype=np.float64)
    null_mean_key = np.asarray([n["mean_key_auroc"] for n in nulls], dtype=np.float64)
    null_mae = np.asarray([n["rounded_mae"] for n in nulls], dtype=np.float64)
    null_corr = np.asarray([n["corr"] for n in nulls], dtype=np.float64)

    pooled = float(real["pooled_auroc"])
    mean_key = float(real["mean_key_auroc"])
    mae = float(real["rounded_mae"])
    corr = float(real["corr"])

    def _z(value: float, arr: np.ndarray, *, lower_is_better: bool = False) -> float:
        a = arr[np.isfinite(arr)]
        if a.size < 2 or a.std(ddof=1) < 1e-12:
            return float("nan")
        if lower_is_better:
            return (a.mean() - value) / a.std(ddof=1)
        return (value - a.mean()) / a.std(ddof=1)

    return (
        f"{cfg.name}: "
        f"AUROC_pooled={pooled:.4f} "
        f"(null {np.nanmean(null_pooled):.4f}+/-{np.nanstd(null_pooled, ddof=1):.4f}, "
        f"z={_z(pooled, null_pooled):+.2f}), "
        f"AUROC_meankey={mean_key:.4f} "
        f"(null {np.nanmean(null_mean_key):.4f}+/-{np.nanstd(null_mean_key, ddof=1):.4f}, "
        f"z={_z(mean_key, null_mean_key):+.2f}); "
        f"rMAE={mae:.4f} (null {np.nanmean(null_mae):.4f}+/-{np.nanstd(null_mae, ddof=1):.4f}, "
        f"z={_z(mae, null_mae, lower_is_better=True):+.2f}), "
        f"corr={corr:.4f} (null {np.nanmean(null_corr):.4f}+/-{np.nanstd(null_corr, ddof=1):.4f}, "
        f"z={_z(corr, null_corr):+.2f})"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s4_d_pair_*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--c2-delta", type=int, default=1)
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--n-features", type=int, default=32)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument("--sample-range", default=None)
    p.add_argument("--max-traces", type=int, default=None)
    p.add_argument("--absolute-diff", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--n-perm", type=int, default=300)
    p.add_argument("--seed", type=int, default=0xD0DA)
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s4_pair_oracle",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.inputs, args.component)
    if args.max_traces is not None:
        if not (1 <= args.max_traces <= ds.traces.shape[2]):
            raise ValueError(f"--max-traces outside [1, {ds.traces.shape[2]}]")
        ds = Dataset(
            traces=ds.traces[:, :, : args.max_traces],
            sks=ds.sks,
            design_terms=ds.design_terms,
            design_c2=ds.design_c2,
            pkfps=ds.pkfps,
            pks=ds.pks,
        )
    sample_lo, sample_hi = parse_sample_range(args.sample_range, ds.traces.shape[-1])
    if sample_lo != 0 or sample_hi != ds.traces.shape[-1]:
        ds = Dataset(
            traces=ds.traces[..., sample_lo:sample_hi],
            sks=ds.sks,
            design_terms=ds.design_terms,
            design_c2=ds.design_c2,
            pkfps=ds.pkfps,
            pks=ds.pks,
        )
    if ds.traces.shape[0] < 4:
        print(f"[FAIL] need at least 4 keys, got {ds.traces.shape[0]}")
        return 1

    configs = (
        [
            Config(block, nf, ridge, mode)
            for block in (8, 16, 32, 64)
            for nf in (32, 64, 128)
            for ridge in (1.0, 10.0, 100.0)
            for mode in ("snr", "corr")
        ]
        if args.sweep
        else [Config(args.block, args.n_features, args.ridge, args.feature_mode)]
    )

    real_rows: list[
        tuple[float, float, Config, dict[str, float | np.ndarray], PairDataset]
    ] = []
    pair_count = None
    bal_str = None
    for cfg in configs:
        pds = build_pair_dataset(ds, cfg, args.c2_delta, absolute_diff=args.absolute_diff)
        pair_count = pds.diff_xmean.shape[1]
        if bal_str is None:
            bal_overall = class_balance(labels_flip_any(pds))
            bal_str = (
                f"flip_any class balance: pos_frac={bal_overall[0]:.3f} "
                f"(pos={bal_overall[1]} neg={bal_overall[2]})"
            )
        real = evaluate_oracle(pds, cfg)
        real_rows.append((float(real["pooled_auroc"]), float(real["mean_key_auroc"]), cfg, real, pds))

    real_rows.sort(reverse=True, key=lambda x: (x[0], x[1]))
    best_pooled, best_meankey, best_cfg, best_real, best_pds = real_rows[0]
    print(
        f"[BEST-REAL] {best_cfg.name} pairs={pair_count} "
        f"AUROC_pooled={best_pooled:.4f} AUROC_meankey={best_meankey:.4f} "
        f"rMAE={best_real['rounded_mae']:.4f} corr={best_real['corr']:.4f}"
    )

    nulls: list[dict[str, float | np.ndarray]] = []
    for i in range(args.n_perm):
        nulls.append(
            evaluate_oracle(
                best_pds,
                best_cfg,
                key_perm=rng.permutation(best_pds.sks.shape[0]),
            )
        )
        if (i + 1) % max(1, args.n_perm // 5) == 0:
            print(f"[NULL] {i + 1}/{args.n_perm}")

    summary = summarize(best_cfg, best_real, nulls)
    print("[SUMMARY] " + summary)

    lines = [
        "S4 paired-trace mu'-change oracle",
        f"S={ds.traces.shape[0]}, D={ds.traces.shape[1]}, N={ds.traces.shape[2]}, T={ds.traces.shape[3]}",
        f"P={pair_count}",
        f"sample_range=[{sample_lo}:{sample_hi}]",
        f"c2_delta={args.c2_delta}",
        f"absolute_diff={args.absolute_diff}",
        f"terms={ds.design_terms}",
        f"c2={ds.design_c2}",
        bal_str or "",
        "",
    ]
    if args.sweep:
        lines.append("real-only top configs:")
        for row in real_rows[:5]:
            cfg = row[2]
            real = row[3]
            lines.append(
                f"  {cfg.name}: AUROC_pooled={real['pooled_auroc']:.4f}, "
                f"AUROC_meankey={real['mean_key_auroc']:.4f}, "
                f"rMAE={real['rounded_mae']:.4f}, corr={real['corr']:.4f}"
            )
        lines.append("")
    lines.append(summary)
    per_key = best_real["per_key_auroc"]
    lines.append(
        "per-key AUROC: "
        + ", ".join(
            f"{ds.pkfps[k][:8]}={float(per_key[k]):.3f}"
            if np.isfinite(per_key[k])
            else f"{ds.pkfps[k][:8]}=NaN"
            for k in range(len(ds.pkfps))
        )
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = args.out_prefix.with_suffix(".txt")
    txt_path.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    null_pooled = np.asarray([n["pooled_auroc"] for n in nulls], dtype=np.float64)
    null_meankey = np.asarray([n["mean_key_auroc"] for n in nulls], dtype=np.float64)
    null_corr = np.asarray([n["corr"] for n in nulls], dtype=np.float64)
    panes = [
        (null_pooled, float(best_real["pooled_auroc"]), "AUROC pooled"),
        (null_meankey, float(best_real["mean_key_auroc"]), "AUROC mean-of-keys"),
        (null_corr, float(best_real["corr"]), "byte-HW corr"),
    ]
    for ax, (null_arr, real_val, title) in zip(axes, panes):
        a = null_arr[np.isfinite(null_arr)]
        if a.size > 0:
            ax.hist(a, bins=20, color="0.75", edgecolor="0.35")
        ax.axvline(real_val, color="C3", lw=2)
        ax.set_title(title)
    fig.tight_layout()
    png_path = args.out_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=120)
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
