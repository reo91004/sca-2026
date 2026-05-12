#!/usr/bin/env python3
"""S2 leakage-aware public-design selection.

This experiment is intentionally nested: for each held-out target key, public
designs are selected using only the remaining profiling keys. The selected
design subset is then used to train a trace-to-label model and evaluate the
held-out key. Target-key labels never participate in selection or fitting.

The goal is to test the S2.12 hypothesis that designs should be selected by
measured trace-to-label transfer, not by synthetic public-label variation.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from scripts.smaug.s2_z_lowdim_analyze import (  # noqa: E402
    Config,
    Dataset,
    cached_labels,
    fit_predict_ridge,
    label_bounds,
    load_dataset,
    make_observation_features,
    parse_sample_range,
    select_features,
)


@dataclass(frozen=True)
class Metrics:
    exact: float
    rounded_mae: float
    corr: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s2_z_matrix_randmt_win17731_d8n20_k*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--label-kind", default="toom2_conv16_hw")
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument("--sample-range", default=None)
    p.add_argument("--max-traces", type=int, default=None)
    p.add_argument("--select-k", type=int, default=4)
    p.add_argument(
        "--selection-metric",
        choices=("corr", "rounded_mae", "exact"),
        default="corr",
    )
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0x1EA61E)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "s2_z_leakage_select.txt",
    )
    return p.parse_args()


def slice_dataset(ds: Dataset, sample_range: str | None, max_traces: int | None) -> tuple[Dataset, tuple[int, int]]:
    if max_traces is not None:
        if not (1 <= max_traces <= ds.traces.shape[2]):
            raise ValueError(f"--max-traces outside [1, {ds.traces.shape[2]}]")
        ds = Dataset(
            traces=ds.traces[:, :, :max_traces],
            sks=ds.sks,
            design_terms=ds.design_terms,
            pkfps=ds.pkfps,
        )
    lo, hi = parse_sample_range(sample_range, ds.traces.shape[-1])
    if lo != 0 or hi != ds.traces.shape[-1]:
        ds = Dataset(
            traces=ds.traces[..., lo:hi],
            sks=ds.sks,
            design_terms=ds.design_terms,
            pkfps=ds.pkfps,
        )
    return ds, (lo, hi)


def score_metrics(pred: np.ndarray, true: np.ndarray, bounds: tuple[float, float]) -> Metrics:
    lo, hi = bounds
    rounded = np.clip(np.rint(pred), lo, hi)
    exact = float(np.mean(rounded == true))
    rounded_mae = float(np.mean(np.abs(rounded - true)))
    if np.std(pred) > 1e-12 and np.std(true) > 1e-12:
        corr = float(np.corrcoef(pred.ravel(), true.ravel())[0, 1])
    else:
        corr = 0.0
    return Metrics(exact=exact, rounded_mae=rounded_mae, corr=corr)


def metric_value(m: Metrics, selection_metric: str) -> float:
    if selection_metric == "corr":
        return m.corr
    if selection_metric == "rounded_mae":
        return -m.rounded_mae
    if selection_metric == "exact":
        return m.exact
    raise ValueError(selection_metric)


def predict_for_subset(
    xmean: np.ndarray,
    xvar: np.ndarray,
    counts: np.ndarray,
    y_fit: np.ndarray,
    train_keys: np.ndarray,
    test_key: int,
    subset: tuple[int, ...],
    cfg: Config,
) -> np.ndarray:
    x_train = xmean[np.ix_(train_keys, subset)].reshape(-1, xmean.shape[-1])
    v_train = xvar[np.ix_(train_keys, subset)].reshape(-1, xvar.shape[-1])
    c_train = counts[np.ix_(train_keys, subset)].reshape(-1)
    y_train = y_fit[np.ix_(train_keys, subset)].reshape(-1, y_fit.shape[-1])
    x_test = xmean[test_key, list(subset)].reshape(len(subset), xmean.shape[-1])
    features = select_features(x_train, v_train, c_train, y_train, cfg)
    return fit_predict_ridge(x_train, y_train, x_test, features, cfg.ridge)


def inner_score_subset(
    xmean: np.ndarray,
    xvar: np.ndarray,
    counts: np.ndarray,
    y_fit: np.ndarray,
    train_pool: np.ndarray,
    subset: tuple[int, ...],
    cfg: Config,
    bounds: tuple[float, float],
    selection_metric: str,
) -> Metrics:
    preds = []
    trues = []
    for val_key in train_pool:
        inner_train = train_pool[train_pool != val_key]
        pred = predict_for_subset(
            xmean,
            xvar,
            counts,
            y_fit,
            inner_train,
            int(val_key),
            subset,
            cfg,
        )
        preds.append(pred)
        trues.append(y_fit[int(val_key), list(subset)])
    metrics = score_metrics(np.concatenate(preds, axis=0), np.concatenate(trues, axis=0), bounds)
    # Keep this function's caller simple while preserving the metric choice in one place.
    _ = metric_value(metrics, selection_metric)
    return metrics


def select_subset_for_hold(
    xmean: np.ndarray,
    xvar: np.ndarray,
    counts: np.ndarray,
    y_fit: np.ndarray,
    hold: int,
    subsets: list[tuple[int, ...]],
    cfg: Config,
    bounds: tuple[float, float],
    selection_metric: str,
) -> tuple[tuple[int, ...], Metrics]:
    train_pool = np.asarray([i for i in range(xmean.shape[0]) if i != hold], dtype=np.int64)
    best_subset = subsets[0]
    best_metrics = Metrics(exact=-np.inf, rounded_mae=np.inf, corr=-np.inf)
    best_score = -np.inf
    for subset in subsets:
        metrics = inner_score_subset(
            xmean,
            xvar,
            counts,
            y_fit,
            train_pool,
            subset,
            cfg,
            bounds,
            selection_metric,
        )
        score = metric_value(metrics, selection_metric)
        if score > best_score:
            best_score = score
            best_subset = subset
            best_metrics = metrics
    return best_subset, best_metrics


def evaluate_nested(
    ds: Dataset,
    label_kind: str,
    cfg: Config,
    select_k: int,
    selection_metric: str,
    key_perm: np.ndarray | None = None,
) -> dict[str, object]:
    xmean, xvar, counts = make_observation_features(ds.traces, cfg.block)
    y_true = cached_labels(ds, label_kind)
    y_fit = y_true if key_perm is None else y_true[key_perm]
    bounds = label_bounds(y_true)
    s, d, _ = xmean.shape
    if not (1 <= select_k <= d):
        raise ValueError(f"--select-k must be in [1, {d}], got {select_k}")
    subsets = list(itertools.combinations(range(d), select_k))

    preds = []
    trues = []
    selected: list[tuple[int, ...]] = []
    inner_metrics: list[Metrics] = []
    for hold in range(s):
        subset, inner = select_subset_for_hold(
            xmean,
            xvar,
            counts,
            y_fit,
            hold,
            subsets,
            cfg,
            bounds,
            selection_metric,
        )
        train_keys = np.asarray([i for i in range(s) if i != hold], dtype=np.int64)
        pred = predict_for_subset(
            xmean,
            xvar,
            counts,
            y_fit,
            train_keys,
            hold,
            subset,
            cfg,
        )
        preds.append(pred)
        trues.append(y_true[hold, list(subset)])
        selected.append(subset)
        inner_metrics.append(inner)

    pred_a = np.concatenate(preds, axis=0)
    true_a = np.concatenate(trues, axis=0)
    metrics = score_metrics(pred_a, true_a, bounds)
    return {
        "metrics": metrics,
        "pred": pred_a,
        "true": true_a,
        "selected": selected,
        "inner_metrics": inner_metrics,
    }


def summarize_null(real: Metrics, nulls: list[Metrics]) -> str:
    exact = np.asarray([m.exact for m in nulls], dtype=np.float64)
    mae = np.asarray([m.rounded_mae for m in nulls], dtype=np.float64)
    corr = np.asarray([m.corr for m in nulls], dtype=np.float64)
    z_exact = (real.exact - exact.mean()) / max(exact.std(ddof=1), 1e-12)
    z_mae = (mae.mean() - real.rounded_mae) / max(mae.std(ddof=1), 1e-12)
    z_corr = (real.corr - corr.mean()) / max(corr.std(ddof=1), 1e-12)
    return (
        f"exact={real.exact:.4f} (null {exact.mean():.4f}+/-{exact.std(ddof=1):.4f}, z={z_exact:+.2f})\n"
        f"rMAE={real.rounded_mae:.4f} (null {mae.mean():.4f}+/-{mae.std(ddof=1):.4f}, z={z_mae:+.2f})\n"
        f"corr={real.corr:.4f} (null {corr.mean():.4f}+/-{corr.std(ddof=1):.4f}, z={z_corr:+.2f})"
    )


def format_selection(selected: list[tuple[int, ...]]) -> list[str]:
    counts: dict[tuple[int, ...], int] = {}
    design_counts: dict[int, int] = {}
    for subset in selected:
        counts[subset] = counts.get(subset, 0) + 1
        for d in subset:
            design_counts[d] = design_counts.get(d, 0) + 1
    lines = ["selected subsets:"]
    for subset, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {subset}: {count}")
    lines.append("design hit counts:")
    for d, count in sorted(design_counts.items()):
        lines.append(f"  d{d}: {count}")
    return lines


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = Config(args.block, args.n_features, args.ridge, args.feature_mode)
    ds = load_dataset(args.inputs, args.component)
    ds, (sample_lo, sample_hi) = slice_dataset(ds, args.sample_range, args.max_traces)
    if ds.traces.shape[0] < 4:
        print(f"[FAIL] need at least 4 keys, got {ds.traces.shape[0]}")
        return 1

    print(
        f"[INFO] S={ds.traces.shape[0]} D={ds.traces.shape[1]} "
        f"N={ds.traces.shape[2]} T={ds.traces.shape[3]} "
        f"label={args.label_kind} cfg={cfg.name} select_k={args.select_k}"
    )
    real = evaluate_nested(
        ds,
        args.label_kind,
        cfg,
        args.select_k,
        args.selection_metric,
    )
    real_metrics = real["metrics"]
    assert isinstance(real_metrics, Metrics)
    print(
        f"[REAL] exact={real_metrics.exact:.4f} "
        f"rMAE={real_metrics.rounded_mae:.4f} corr={real_metrics.corr:.4f}"
    )

    null_metrics: list[Metrics] = []
    for i in range(args.n_perm):
        null = evaluate_nested(
            ds,
            args.label_kind,
            cfg,
            args.select_k,
            args.selection_metric,
            key_perm=rng.permutation(ds.traces.shape[0]),
        )
        m = null["metrics"]
        assert isinstance(m, Metrics)
        null_metrics.append(m)
        if (i + 1) % max(1, args.n_perm // 5) == 0:
            print(f"[NULL] {i + 1}/{args.n_perm}")

    selected = real["selected"]
    assert isinstance(selected, list)
    lines = [
        "S2 leakage-aware design selection",
        f"inputs={[p.name for p in args.inputs]}",
        f"S={ds.traces.shape[0]}, D={ds.traces.shape[1]}, N={ds.traces.shape[2]}, T={ds.traces.shape[3]}",
        f"sample_range=[{sample_lo}:{sample_hi}]",
        f"label={args.label_kind}",
        f"cfg={cfg.name}",
        f"select_k={args.select_k}",
        f"selection_metric={args.selection_metric}",
        "",
        summarize_null(real_metrics, null_metrics),
        "",
    ]
    lines.extend(format_selection(selected))
    lines.append("")
    lines.append("per-hold inner validation metrics:")
    inner_metrics = real["inner_metrics"]
    assert isinstance(inner_metrics, list)
    for hold, (subset, inner) in enumerate(zip(selected, inner_metrics)):
        lines.append(
            f"  hold={hold:02d} subset={subset} "
            f"inner_exact={inner.exact:.4f} inner_rMAE={inner.rounded_mae:.4f} "
            f"inner_corr={inner.corr:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("[SUMMARY]")
    print(summarize_null(real_metrics, null_metrics))
    print(f"[OK] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
