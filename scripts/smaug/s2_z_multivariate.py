#!/usr/bin/env python3
"""S2 — multivariate trace-only profiling on 'Z' captures.

This experiment asks the first attack-relevant question after the S1 diagnostic:
does the full 'Z' trace contain enough key-dependent information for a general
profiler to predict a held-out key better than sparse baselines?

Threat-model discipline:
  * training keys may use known sk labels, as in a profiling attack;
  * held-out inference uses only the target trace mean and public sparse prior;
  * target mu' responses are ignored;
  * feature selection is trace-only and performed inside each fold.

Model:
  1. Average traces into fixed-size sample blocks.
  2. Select high-SNR blocks using training traces only.
  3. Fit a multi-output ridge regressor from selected trace blocks to the 256
     ternary coefficients of one secret polynomial.
  4. Predict the held-out key and enforce SMAUG-T HS=70 sparsity by choosing the
     largest absolute scores as nonzero positions.
  5. Compare against label-permutation null and sparse random baseline.
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

from host.analysis.sparse_recover import accuracy  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug.params import SMAUG1  # noqa: E402


@dataclass(frozen=True)
class Capture:
    path: Path
    pk_fp16: str
    traces: np.ndarray
    sk_poly: np.ndarray


@dataclass(frozen=True)
class Config:
    block: int
    n_features: int
    ridge: float

    @property
    def name(self) -> str:
        return f"b{self.block}_f{self.n_features}_r{self.ridge:g}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s2_z_sk*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--hs", type=int, default=SMAUG1.hs)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate a small predeclared grid in addition to the fixed config.",
    )
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--seed", type=int, default=0x51A2)
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s2_z_multivariate",
    )
    return p.parse_args()


def load_captures(paths: list[Path], component: int) -> list[Capture]:
    out: list[Capture] = []
    seen: set[str] = set()
    for path in paths:
        try:
            data = np.load(path, allow_pickle=True)
            traces = np.asarray(data["traces"], dtype=np.float64)
            meta = data["meta"].item()
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
            continue
        if meta.get("cmd") != "Z":
            continue
        pk = str(meta.get("pk_fp16", ""))
        if not pk or pk in seen:
            continue
        if "sk_pke_hex" not in meta:
            print(f"[WARN] skip {path.name}: sk_pke_hex missing")
            continue
        full = _codec.unpack_sx(bytes.fromhex(meta["sk_pke_hex"])).astype(np.int8)
        lo = component * 256
        hi = lo + 256
        out.append(Capture(path=path, pk_fp16=pk, traces=traces, sk_poly=full[lo:hi]))
        seen.add(pk)
    return out


def block_stats(captures: list[Capture], block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-key block means, variance of block averages, and trace counts."""
    t_min = min(c.traces.shape[1] for c in captures)
    n_blocks = t_min // block
    if n_blocks < 2:
        raise ValueError(f"block={block} leaves only {n_blocks} blocks")
    means = []
    variances = []
    counts = []
    for cap in captures:
        x = cap.traces[:, : n_blocks * block]
        xb = x.reshape(x.shape[0], n_blocks, block).mean(axis=2)
        means.append(xb.mean(axis=0))
        variances.append(xb.var(axis=0, ddof=1))
        counts.append(xb.shape[0])
    return np.stack(means), np.stack(variances), np.asarray(counts, dtype=np.float64)


def select_features(
    block_means: np.ndarray,
    block_vars: np.ndarray,
    counts: np.ndarray,
    train_idx: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Trace-only SNR feature selection using training keys only."""
    train_means = block_means[train_idx]
    between = train_means.var(axis=0, ddof=1)
    mean_noise_of_mean = np.mean(block_vars[train_idx] / counts[train_idx, None], axis=0)
    score = between / np.maximum(mean_noise_of_mean, 1e-12)
    n = min(n_features, score.size)
    idx = np.argsort(-score, kind="stable")[:n]
    return np.sort(idx)


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (x_train - mu) / sd, (x_test - mu) / sd


def fit_predict_ridge(
    block_means: np.ndarray,
    sks: np.ndarray,
    feature_idx: np.ndarray,
    train_idx: np.ndarray,
    test_idx: int,
    ridge: float,
) -> np.ndarray:
    x_train = block_means[train_idx][:, feature_idx]
    x_test = block_means[[test_idx]][:, feature_idx]
    x_train, x_test = standardize_train_test(x_train, x_test)

    y_train = sks[train_idx].astype(np.float64)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_centered = y_train - y_mean

    gram = x_train @ x_train.T
    alpha = np.linalg.solve(
        gram + float(ridge) * np.eye(gram.shape[0]),
        y_centered,
    )
    weights = x_train.T @ alpha
    return (x_test @ weights + y_mean)[0]


def sparse_decode(scores: np.ndarray, hs: int) -> np.ndarray:
    pred = np.zeros(scores.shape[0], dtype=np.int8)
    order = np.argsort(-np.abs(scores), kind="stable")
    support = order[:hs]
    pred[support] = np.where(scores[support] >= 0, 1, -1).astype(np.int8)
    return pred


def evaluate_config(
    captures: list[Capture],
    cfg: Config,
    *,
    component: int,
    hs: int,
    label_permutation: np.ndarray | None = None,
) -> dict[str, object]:
    block_means, block_vars, counts = block_stats(captures, cfg.block)
    sks_true = np.stack([c.sk_poly for c in captures]).astype(np.int8)
    sks_train_labels = sks_true if label_permutation is None else sks_true[label_permutation]
    n_keys = sks_true.shape[0]

    preds = []
    feature_sets = []
    for hold in range(n_keys):
        train_idx = np.asarray([i for i in range(n_keys) if i != hold], dtype=np.int64)
        feature_idx = select_features(
            block_means,
            block_vars,
            counts,
            train_idx,
            cfg.n_features,
        )
        scores = fit_predict_ridge(
            block_means,
            sks_train_labels,
            feature_idx,
            train_idx,
            hold,
            cfg.ridge,
        )
        preds.append(sparse_decode(scores, hs))
        feature_sets.append(feature_idx)

    pred_arr = np.stack(preds)
    per_key = [accuracy(pred_arr[i], sks_true[i]) for i in range(n_keys)]
    coord = np.asarray([m["bit_accuracy"] for m in per_key])
    support = np.asarray([m["support_accuracy"] for m in per_key])
    sign = np.asarray([m["sign_accuracy"] for m in per_key])

    return {
        "config": cfg,
        "pred": pred_arr,
        "coord": coord,
        "support": support,
        "sign": sign,
        "feature_sets": feature_sets,
        "block_count": block_means.shape[1],
        "component": component,
    }


def random_sparse_baseline(
    sks: np.ndarray,
    hs: int,
    rng: np.random.Generator,
    n_trials: int = 200,
) -> dict[str, float]:
    coords = []
    supports = []
    signs = []
    n_keys, n = sks.shape
    for _ in range(n_trials):
        pred = np.zeros_like(sks, dtype=np.int8)
        for i in range(n_keys):
            supp = rng.choice(n, size=hs, replace=False)
            pred[i, supp] = rng.choice(np.array([-1, 1], dtype=np.int8), size=hs)
        metrics = [accuracy(pred[i], sks[i]) for i in range(n_keys)]
        coords.append(np.mean([m["bit_accuracy"] for m in metrics]))
        supports.append(np.mean([m["support_accuracy"] for m in metrics]))
        signs.append(np.mean([m["sign_accuracy"] for m in metrics]))
    return {
        "coord": float(np.mean(coords)),
        "support": float(np.mean(supports)),
        "sign": float(np.nanmean(signs)),
    }


def summarize_result(result: dict[str, object]) -> str:
    cfg = result["config"]
    assert isinstance(cfg, Config)
    coord = np.asarray(result["coord"], dtype=np.float64)
    support = np.asarray(result["support"], dtype=np.float64)
    sign = np.asarray(result["sign"], dtype=np.float64)
    return (
        f"{cfg.name}: coord={coord.mean():.4f}±{coord.std(ddof=1):.4f}, "
        f"support={support.mean():.4f}±{support.std(ddof=1):.4f}, "
        f"sign={np.nanmean(sign):.4f}"
    )


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    captures = load_captures(args.inputs, args.component)
    if len(captures) < 8:
        print(f"[FAIL] need at least 8 unique Z captures, got {len(captures)}")
        return 1

    sks = np.stack([c.sk_poly for c in captures]).astype(np.int8)
    zero_baseline = float(np.mean(sks == 0))
    configs = [Config(args.block, args.n_features, args.ridge)]
    if args.sweep:
        configs = [
            Config(block, n_features, ridge)
            for block in (8, 16, 32, 64)
            for n_features in (16, 32, 64, 128)
            for ridge in (1.0, 10.0, 100.0)
        ]

    print(
        f"[INFO] S={len(captures)} unique Z captures, component={args.component}, "
        f"HS={args.hs}, zero-baseline={zero_baseline:.4f}, configs={len(configs)}"
    )

    results = []
    for cfg in configs:
        result = evaluate_config(
            captures,
            cfg,
            component=args.component,
            hs=args.hs,
        )
        results.append(result)
        print("[REAL] " + summarize_result(result))

    best = max(results, key=lambda r: float(np.mean(np.asarray(r["coord"], dtype=np.float64))))
    best_cfg = best["config"]
    assert isinstance(best_cfg, Config)
    print(f"[INFO] best exploratory config by coord_acc: {best_cfg.name}")

    null_coords = []
    null_supports = []
    null_signs = []
    for i in range(args.n_perm):
        perm = rng.permutation(len(captures))
        null = evaluate_config(
            captures,
            best_cfg,
            component=args.component,
            hs=args.hs,
            label_permutation=perm,
        )
        null_coords.append(float(np.mean(np.asarray(null["coord"], dtype=np.float64))))
        null_supports.append(float(np.mean(np.asarray(null["support"], dtype=np.float64))))
        null_signs.append(float(np.nanmean(np.asarray(null["sign"], dtype=np.float64))))
        if (i + 1) % max(1, args.n_perm // 10) == 0:
            print(f"[NULL] {i + 1}/{args.n_perm}")

    null_coord = np.asarray(null_coords)
    null_support = np.asarray(null_supports)
    null_sign = np.asarray(null_signs)
    best_coord = float(np.mean(np.asarray(best["coord"], dtype=np.float64)))
    best_support = float(np.mean(np.asarray(best["support"], dtype=np.float64)))
    best_sign = float(np.nanmean(np.asarray(best["sign"], dtype=np.float64)))
    z_coord = (best_coord - float(null_coord.mean())) / max(float(null_coord.std(ddof=1)), 1e-12)
    z_support = (best_support - float(null_support.mean())) / max(float(null_support.std(ddof=1)), 1e-12)

    rand = random_sparse_baseline(sks, args.hs, rng)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "S2 Z multivariate trace-only profiling",
        f"S={len(captures)}, component={args.component}, HS={args.hs}",
        f"zero baseline coord={zero_baseline:.4f}",
        (
            "random sparse baseline "
            f"coord={rand['coord']:.4f}, support={rand['support']:.4f}, sign={rand['sign']:.4f}"
        ),
        "",
        "Real configs:",
        *[summarize_result(r) for r in results],
        "",
        f"Best exploratory config: {best_cfg.name}",
        f"best coord={best_coord:.4f}, support={best_support:.4f}, sign={best_sign:.4f}",
        (
            f"permutation null coord={null_coord.mean():.4f}±{null_coord.std(ddof=1):.4f}, "
            f"z={z_coord:+.2f}"
        ),
        (
            f"permutation null support={null_support.mean():.4f}±{null_support.std(ddof=1):.4f}, "
            f"z={z_support:+.2f}"
        ),
        f"permutation null sign={np.nanmean(null_sign):.4f}±{np.nanstd(null_sign, ddof=1):.4f}",
    ]
    txt_path = args.out_prefix.with_suffix(".txt")
    txt_path.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(null_coord, bins=20, color="0.75", edgecolor="0.35")
    axes[0].axvline(best_coord, color="C3", lw=2, label="real")
    axes[0].axvline(zero_baseline, color="C0", lw=1, ls="--", label="zero")
    axes[0].set_title("coord accuracy")
    axes[0].legend()

    axes[1].hist(null_support, bins=20, color="0.75", edgecolor="0.35")
    axes[1].axvline(best_support, color="C3", lw=2)
    axes[1].axvline(rand["support"], color="C0", lw=1, ls="--")
    axes[1].set_title("support accuracy")

    axes[2].hist(null_sign[np.isfinite(null_sign)], bins=20, color="0.75", edgecolor="0.35")
    axes[2].axvline(best_sign, color="C3", lw=2)
    axes[2].axvline(rand["sign"], color="C0", lw=1, ls="--")
    axes[2].set_title("sign accuracy")
    fig.tight_layout()
    png_path = args.out_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=120)

    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {png_path}")
    print("[SUMMARY]")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
