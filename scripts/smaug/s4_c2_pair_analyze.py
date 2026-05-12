#!/usr/bin/env python3
"""Paired c2 threshold-difference profiling for SMAUG-T matrix captures.

This branch keeps c1 fixed and compares captures with c2=beta and
c2=beta+delta. The goal is to subtract multiplication common mode and test
whether natural traces carry the latent mu' flip Hamming weight.
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
    summarize_features,
)


LABEL_KINDS = (
    "flip_byte_hw",
    "flip_block16_hw",
    "flip_block32_hw",
    "mu_delta_byte_hw",
    "mu_delta_block16_hw",
)


@dataclass(frozen=True)
class PairDataset:
    xmean: np.ndarray       # (S,P,B)
    xvar: np.ndarray        # (S,P,B)
    counts: np.ndarray      # (S,P)
    sks: np.ndarray         # (S,256)
    base_terms: list[list[tuple[int, int]]]
    base_c2: list[int]
    delta_c2: list[int]
    pair_indices: list[tuple[int, int]]
    pkfps: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s4_c2_*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--label-kinds", nargs="+", choices=LABEL_KINDS, default=list(LABEL_KINDS))
    p.add_argument("--c2-delta", type=int, default=1)
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--n-features", type=int, default=32)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument("--sample-range", default=None)
    p.add_argument("--max-traces", type=int, default=None)
    p.add_argument("--absolute-diff", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--seed", type=int, default=0xC2D1FF)
    p.add_argument("--out-prefix", type=Path, default=_REPO / "results" / "s4_c2_pair")
    return p.parse_args()


def _terms_key(terms: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(coef), int(alpha)) for coef, alpha in terms)


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
        xmean=diff,
        xvar=xvar[:, delta_idx, :] + xvar[:, base_idx, :],
        counts=np.minimum(counts[:, delta_idx], counts[:, base_idx]),
        sks=ds.sks,
        base_terms=base_terms,
        base_c2=base_c2,
        delta_c2=delta_c2,
        pair_indices=pairs,
        pkfps=ds.pkfps,
    )


def labels_for_pair(pds: PairDataset, kind: str) -> np.ndarray:
    labels = []
    for key_i in range(pds.sks.shape[0]):
        per_pair = []
        for terms, base_c2, delta_c2 in zip(pds.base_terms, pds.base_c2, pds.delta_c2):
            b0 = label_from_mu_bits(pds.sks[key_i], terms, base_c2, "mu_bit").astype(np.int8)
            b1 = label_from_mu_bits(pds.sks[key_i], terms, delta_c2, "mu_bit").astype(np.int8)
            flip = np.bitwise_xor(b0, b1).astype(np.float64)
            delta = b1.astype(np.float64) - b0.astype(np.float64)
            if kind == "flip_byte_hw":
                per_pair.append(flip.reshape(32, 8).sum(axis=1))
            elif kind == "flip_block16_hw":
                per_pair.append(flip.reshape(16, 16).sum(axis=1))
            elif kind == "flip_block32_hw":
                per_pair.append(flip.reshape(8, 32).sum(axis=1))
            elif kind == "mu_delta_byte_hw":
                per_pair.append(delta.reshape(32, 8).sum(axis=1))
            elif kind == "mu_delta_block16_hw":
                per_pair.append(delta.reshape(16, 16).sum(axis=1))
            else:
                raise ValueError(f"unknown label kind {kind}")
        labels.append(np.stack(per_pair))
    return np.stack(labels)


def evaluate(
    pds: PairDataset,
    kind: str,
    cfg: Config,
    *,
    key_perm: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    y_true = labels_for_pair(pds, kind)
    y_labels = y_true if key_perm is None else y_true[key_perm]
    s, pair_count, b = pds.xmean.shape

    preds = []
    trues = []
    feature_counts = np.zeros(b, dtype=np.int64)
    for hold in range(s):
        train_keys = np.asarray([i for i in range(s) if i != hold], dtype=np.int64)
        x_train = pds.xmean[train_keys].reshape(-1, b)
        v_train = pds.xvar[train_keys].reshape(-1, b)
        c_train = pds.counts[train_keys].reshape(-1)
        y_train = y_labels[train_keys].reshape(-1, y_true.shape[-1])
        x_test = pds.xmean[[hold]].reshape(pair_count, b)
        y_test = y_true[[hold]].reshape(pair_count, y_true.shape[-1])

        features = select_features(x_train, v_train, c_train, y_train, cfg)
        feature_counts[features] += 1
        preds.append(fit_predict_ridge(x_train, y_train, x_test, features, cfg.ridge))
        trues.append(y_test)

    yp = np.concatenate(preds, axis=0)
    yt = np.concatenate(trues, axis=0)
    lo, hi = float(np.min(yt)), float(np.max(yt))
    pred_round = np.clip(np.rint(yp), lo, hi)
    rounded_mae = float(np.mean(np.abs(pred_round - yt)))
    exact = float(np.mean(pred_round == yt))
    mae = float(np.mean(np.abs(yp - yt)))
    if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
        corr = float(np.corrcoef(yp.ravel(), yt.ravel())[0, 1])
    else:
        corr = 0.0
    return {
        "mae": mae,
        "rounded_mae": rounded_mae,
        "exact": exact,
        "corr": corr,
        "pred": yp,
        "true": yt,
        "feature_counts": feature_counts,
    }


def summarize(kind: str, cfg: Config, real: dict[str, float | np.ndarray],
              nulls: list[dict[str, float | np.ndarray]]) -> str:
    null_exact = np.asarray([n["exact"] for n in nulls], dtype=np.float64)
    null_mae = np.asarray([n["rounded_mae"] for n in nulls], dtype=np.float64)
    null_corr = np.asarray([n["corr"] for n in nulls], dtype=np.float64)
    exact = float(real["exact"])
    mae = float(real["rounded_mae"])
    corr = float(real["corr"])
    z_exact = (exact - null_exact.mean()) / max(null_exact.std(ddof=1), 1e-12)
    z_mae = (null_mae.mean() - mae) / max(null_mae.std(ddof=1), 1e-12)
    z_corr = (corr - null_corr.mean()) / max(null_corr.std(ddof=1), 1e-12)
    return (
        f"{kind} {cfg.name}: exact={exact:.4f} "
        f"(null {null_exact.mean():.4f}+/-{null_exact.std(ddof=1):.4f}, z={z_exact:+.2f}), "
        f"rMAE={mae:.4f} (null {null_mae.mean():.4f}+/-{null_mae.std(ddof=1):.4f}, z={z_mae:+.2f}), "
        f"corr={corr:.4f} (null {null_corr.mean():.4f}+/-{null_corr.std(ddof=1):.4f}, z={z_corr:+.2f})"
    )


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
            for nf in (32, 64, 128, 256)
            for ridge in (1.0, 10.0, 100.0)
            for mode in ("snr", "corr")
        ]
        if args.sweep
        else [Config(args.block, args.n_features, args.ridge, args.feature_mode)]
    )

    lines = [
        "S4 C2 paired threshold-difference profiling",
        f"S={ds.traces.shape[0]}, D={ds.traces.shape[1]}, N={ds.traces.shape[2]}, T={ds.traces.shape[3]}",
        f"sample_range=[{sample_lo}:{sample_hi}]",
        f"c2_delta={args.c2_delta}",
        f"absolute_diff={args.absolute_diff}",
        f"terms={ds.design_terms}",
        f"c2={ds.design_c2}",
        "",
    ]
    plot_rows = []
    best_pair_count = None

    for kind in args.label_kinds:
        print(f"[KIND] {kind}")
        real_rows = []
        pair_count_for_kind = None
        for cfg in configs:
            pds = build_pair_dataset(ds, cfg, args.c2_delta, absolute_diff=args.absolute_diff)
            pair_count_for_kind = pds.xmean.shape[1]
            real = evaluate(pds, kind, cfg)
            real_rows.append((float(real["exact"]), -float(real["rounded_mae"]), float(real["corr"]), cfg, real, pds))
        real_rows.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
        best_cfg = real_rows[0][3]
        best_real = real_rows[0][4]
        best_pds = real_rows[0][5]
        best_pair_count = best_pds.xmean.shape[1]
        print(
            f"[BEST-REAL] {kind} {best_cfg.name} pairs={pair_count_for_kind} "
            f"exact={best_real['exact']:.4f} rMAE={best_real['rounded_mae']:.4f} corr={best_real['corr']:.4f}"
        )
        if args.sweep:
            lines.append(f"{kind} real-only top configs:")
            for row in real_rows[:5]:
                cfg = row[3]
                real = row[4]
                lines.append(
                    f"  {cfg.name}: exact={real['exact']:.4f}, "
                    f"rMAE={real['rounded_mae']:.4f}, corr={real['corr']:.4f}"
                )

        nulls = []
        for i in range(args.n_perm):
            nulls.append(
                evaluate(
                    best_pds,
                    kind,
                    best_cfg,
                    key_perm=rng.permutation(best_pds.xmean.shape[0]),
                )
            )
            if (i + 1) % max(1, args.n_perm // 5) == 0:
                print(f"[NULL {kind}] {i + 1}/{args.n_perm}")
        summary = summarize(kind, best_cfg, best_real, nulls)
        print("[SUMMARY] " + summary)
        feature_summary = summarize_features(best_real, best_cfg)
        print("[FEATURES] " + feature_summary)
        lines.append(summary)
        lines.append(feature_summary)
        lines.append("")
        plot_rows.append((kind, best_real, nulls))

    if best_pair_count is not None:
        lines.insert(2, f"P={best_pair_count}")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = args.out_prefix.with_suffix(".txt")
    txt_path.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(len(plot_rows), 3, figsize=(13, 3.2 * len(plot_rows)))
    if len(plot_rows) == 1:
        axes = np.asarray([axes])
    for row_i, (kind, real, nulls) in enumerate(plot_rows):
        null_exact = np.asarray([n["exact"] for n in nulls], dtype=np.float64)
        null_mae = np.asarray([n["rounded_mae"] for n in nulls], dtype=np.float64)
        null_corr = np.asarray([n["corr"] for n in nulls], dtype=np.float64)
        vals = [
            (null_exact, float(real["exact"]), "exact"),
            (null_mae, float(real["rounded_mae"]), "rounded MAE"),
            (null_corr, float(real["corr"]), "corr"),
        ]
        for col_i, (null_arr, real_val, title) in enumerate(vals):
            ax = axes[row_i, col_i]
            ax.hist(null_arr[np.isfinite(null_arr)], bins=20, color="0.75", edgecolor="0.35")
            ax.axvline(real_val, color="C3", lw=2)
            ax.set_title(f"{kind}: {title}")
    fig.tight_layout()
    png_path = args.out_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=120)
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
