#!/usr/bin/env python3
"""Rank captured public designs by measured trace-to-label leakage.

This is a leakage-aware complement to synthetic design scoring. It evaluates
each captured design independently with a fixed low-dimensional model and a
key-permutation null. It must be used for experiment planning, not as a final
attack claim, because the same captures are used for ranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from scripts.s2_z_lowdim_analyze import Config, Dataset, evaluate, load_dataset, summarize  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--label-kind", default="toom2_conv16_hw")
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--seed", type=int, default=0xD3516E)
    p.add_argument("--out", type=Path, default=_REPO / "results" / "s2_z_design_leakage_rank.txt")
    return p.parse_args()


def slice_design(ds: Dataset, design_i: int) -> Dataset:
    return Dataset(
        traces=ds.traces[:, design_i : design_i + 1],
        sks=ds.sks,
        design_terms=ds.design_terms[design_i : design_i + 1],
        pkfps=ds.pkfps,
    )


def zscore(real: float, null: np.ndarray, *, larger_is_better: bool = True) -> float:
    if larger_is_better:
        return float((real - null.mean()) / max(null.std(ddof=1), 1e-12))
    return float((null.mean() - real) / max(null.std(ddof=1), 1e-12))


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.inputs, args.component)
    cfg = Config(args.block, args.n_features, args.ridge, args.feature_mode)
    rows = []
    lines = [
        "S2 design leakage rank",
        f"label={args.label_kind} cfg={cfg.name}",
        f"S={ds.traces.shape[0]} D={ds.traces.shape[1]} N={ds.traces.shape[2]} T={ds.traces.shape[3]}",
        "",
    ]
    for design_i, terms in enumerate(ds.design_terms):
        sub = slice_design(ds, design_i)
        real = evaluate(sub, args.label_kind, cfg)
        nulls = []
        for _ in range(args.n_perm):
            nulls.append(
                evaluate(
                    sub,
                    args.label_kind,
                    cfg,
                    key_perm=rng.permutation(sub.traces.shape[0]),
                )
            )
        null_exact = np.asarray([n["exact"] for n in nulls], dtype=np.float64)
        null_mae = np.asarray([n["rounded_mae"] for n in nulls], dtype=np.float64)
        null_corr = np.asarray([n["corr"] for n in nulls], dtype=np.float64)
        row = {
            "design": design_i,
            "terms": terms,
            "exact": float(real["exact"]),
            "rmae": float(real["rounded_mae"]),
            "corr": float(real["corr"]),
            "z_exact": zscore(float(real["exact"]), null_exact),
            "z_mae": zscore(float(real["rounded_mae"]), null_mae, larger_is_better=False),
            "z_corr": zscore(float(real["corr"]), null_corr),
        }
        row["score"] = row["z_exact"] + row["z_mae"] + row["z_corr"]
        rows.append(row)
        lines.append("[RAW] " + summarize(args.label_kind, cfg, real, nulls))
        lines.append(f"[RAW] design={design_i} terms={terms}")
        lines.append("")

    rows.sort(key=lambda r: r["score"], reverse=True)
    lines.append("Ranked designs:")
    for r in rows:
        lines.append(
            f"design={r['design']} score={r['score']:+.2f} "
            f"z_exact={r['z_exact']:+.2f} z_mae={r['z_mae']:+.2f} "
            f"z_corr={r['z_corr']:+.2f} exact={r['exact']:.4f} "
            f"rMAE={r['rmae']:.4f} corr={r['corr']:.4f} terms={r['terms']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
