#!/usr/bin/env python3
"""Recovery-pressure check for S2 low-dimensional labels.

This does not claim key recovery. It asks whether trace-predicted intermediate
labels rank the real held-out sparse secret closer than random SMAUG1-sparse
candidate secrets. Target-key labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.smaug.params import SMAUG1  # noqa: E402
from scripts.s2_z_lowdim_analyze import (  # noqa: E402
    Config,
    Dataset,
    cached_labels,
    evaluate,
    label_from_toom,
    load_dataset,
    parse_sample_range,
)


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
    p.add_argument("--sample-range", default=None)
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument("--n-candidates", type=int, default=5000)
    p.add_argument("--score-mode", choices=("raw", "z"), default="z")
    p.add_argument(
        "--prediction-mode",
        choices=("trace", "oracle"),
        default="trace",
        help="trace: use held-out trace predictions; oracle: use exact labels to test label identifiability only.",
    )
    p.add_argument("--seed", type=int, default=0x5CA2026)
    p.add_argument("--out", type=Path, default=_REPO / "results" / "s2_z_label_pressure_toom2.txt")
    return p.parse_args()


def random_sparse_secret(rng: np.random.Generator) -> np.ndarray:
    sk = np.zeros(SMAUG1.n, dtype=np.int8)
    idx = rng.choice(SMAUG1.n, size=SMAUG1.hs, replace=False)
    sk[idx] = rng.choice(np.array([-1, 1], dtype=np.int8), size=SMAUG1.hs)
    return sk


def candidate_labels(sk: np.ndarray, terms: list[list[tuple[int, int]]], kind: str) -> np.ndarray:
    return np.stack([label_from_toom(sk, design_terms, kind) for design_terms in terms])


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.inputs, args.component)
    sample_lo, sample_hi = parse_sample_range(args.sample_range, ds.traces.shape[-1])
    if sample_lo != 0 or sample_hi != ds.traces.shape[-1]:
        ds = Dataset(
            traces=ds.traces[..., sample_lo:sample_hi],
            sks=ds.sks,
            design_terms=ds.design_terms,
            pkfps=ds.pkfps,
        )
    cfg = Config(args.block, args.n_features, args.ridge, args.feature_mode)
    real = evaluate(ds, args.label_kind, cfg)
    y_true = cached_labels(ds, args.label_kind)
    if args.prediction_mode == "trace":
        pred = np.asarray(real["pred"], dtype=np.float64)
        true = np.asarray(real["true"], dtype=np.float64)
    else:
        pred = y_true.reshape(-1, y_true.shape[-1]).astype(np.float64)
        true = pred.copy()
    s, d, l = y_true.shape
    if pred.shape != (s * d, l):
        raise ValueError(f"unexpected pred shape {pred.shape}, expected {(s * d, l)}")

    if args.score_mode == "z":
        scale = y_true.reshape(-1, y_true.shape[-1]).std(axis=0, ddof=1)
        scale = np.where(scale > 1e-9, scale, 1.0)
    else:
        scale = np.ones(y_true.shape[-1], dtype=np.float64)

    lines = [
        "S2 label recovery-pressure check",
        f"label={args.label_kind} cfg={cfg.name}",
        f"S={s}, D={d}, L={l}, candidates/key={args.n_candidates}",
        f"sample_range=[{sample_lo}:{sample_hi}]",
        f"score_mode={args.score_mode}",
        f"prediction_mode={args.prediction_mode}",
        "",
        "key pkfp true_mse random_mean random_best true_percentile exact_label_mse",
    ]

    percentiles = []
    margins = []
    for key_i in range(s):
        p = pred[key_i * d : (key_i + 1) * d]
        t = true[key_i * d : (key_i + 1) * d]
        true_mse = float(np.mean(((p - t) / scale) ** 2))
        exact_mse = float(np.mean(((p - y_true[key_i]) / scale) ** 2))
        cand_scores = np.empty(args.n_candidates, dtype=np.float64)
        for cand_i in range(args.n_candidates):
            cand = random_sparse_secret(rng)
            cy = candidate_labels(cand, ds.design_terms, args.label_kind)
            cand_scores[cand_i] = np.mean(((p - cy) / scale) ** 2)
        percentile = float(np.mean(cand_scores > true_mse))
        best = float(np.min(cand_scores))
        mean = float(np.mean(cand_scores))
        percentiles.append(percentile)
        margins.append(best - true_mse)
        lines.append(
            f"{key_i:02d} {ds.pkfps[key_i]} {true_mse:.6g} {mean:.6g} "
            f"{best:.6g} {percentile:.4f} {exact_mse:.6g}"
        )

    percentiles_a = np.asarray(percentiles)
    margins_a = np.asarray(margins)
    lines.extend(
        [
            "",
            f"mean_true_percentile={percentiles_a.mean():.4f}",
            f"median_true_percentile={np.median(percentiles_a):.4f}",
            f"keys_above_95pct={int(np.sum(percentiles_a >= 0.95))}/{s}",
            f"keys_above_99pct={int(np.sum(percentiles_a >= 0.99))}/{s}",
            f"mean_best_margin={margins_a.mean():.6g}",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
