#!/usr/bin/env python3
"""Scan trace windows for a specific low-dimensional label.

This is an exploratory localization tool. It slides a fixed-size sample window
over an existing matrix capture, evaluates a fixed trace-to-label model in each
window, then runs a key-permutation null only for the top real-only windows.

Window selection is data-driven, so this script is for planning the next capture
or trigger/window experiment, not for final attack claims.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from scripts.s2_z_lowdim_analyze import (  # noqa: E402
    Config,
    Dataset,
    evaluate,
    load_dataset,
    summarize,
)


@dataclass(frozen=True)
class WindowRow:
    lo: int
    hi: int
    exact: float
    rounded_mae: float
    corr: float

    @property
    def score(self) -> tuple[float, float, float]:
        return (self.corr, -self.rounded_mae, self.exact)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--label-kind", default="toom3_kara2_conv8_hw")
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0x51A6E)
    p.add_argument("--out", type=Path, default=_REPO / "results" / "s2_z_label_window_scan.txt")
    return p.parse_args()


def slice_samples(ds: Dataset, lo: int, hi: int) -> Dataset:
    return Dataset(
        traces=ds.traces[..., lo:hi],
        sks=ds.sks,
        design_terms=ds.design_terms,
        pkfps=ds.pkfps,
    )


def window_starts(t: int, window: int, stride: int) -> list[int]:
    if not (1 <= window <= t):
        raise ValueError(f"window={window} outside [1, {t}]")
    if stride < 1:
        raise ValueError(f"stride={stride} < 1")
    starts = list(range(0, t - window + 1, stride))
    if starts[-1] != t - window:
        starts.append(t - window)
    return starts


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.inputs, args.component)
    cfg = Config(args.block, args.n_features, args.ridge, args.feature_mode)
    t = ds.traces.shape[-1]
    rows: list[WindowRow] = []
    print(
        f"[INFO] S={ds.traces.shape[0]} D={ds.traces.shape[1]} "
        f"N={ds.traces.shape[2]} T={t} label={args.label_kind} cfg={cfg.name}"
    )
    starts = window_starts(t, args.window, args.stride)
    for i, lo in enumerate(starts):
        hi = lo + args.window
        real = evaluate(slice_samples(ds, lo, hi), args.label_kind, cfg)
        row = WindowRow(
            lo=lo,
            hi=hi,
            exact=float(real["exact"]),
            rounded_mae=float(real["rounded_mae"]),
            corr=float(real["corr"]),
        )
        rows.append(row)
        print(
            f"[SCAN] {i+1:03d}/{len(starts):03d} {lo}:{hi} "
            f"exact={row.exact:.4f} rMAE={row.rounded_mae:.4f} corr={row.corr:.4f}"
        )

    rows.sort(key=lambda r: r.score, reverse=True)
    top = rows[: args.top_k]
    lines = [
        "S2 label-specific window scan",
        f"inputs={[p.name for p in args.inputs]}",
        f"S={ds.traces.shape[0]}, D={ds.traces.shape[1]}, N={ds.traces.shape[2]}, T={t}",
        f"label={args.label_kind}",
        f"cfg={cfg.name}",
        f"window={args.window}, stride={args.stride}",
        "",
        "Top real-only windows:",
    ]
    for row in top:
        lines.append(
            f"  {row.lo}:{row.hi} exact={row.exact:.4f} "
            f"rMAE={row.rounded_mae:.4f} corr={row.corr:.4f}"
        )
    lines.append("")
    lines.append("Permutation confirmations:")

    for row in top:
        sub = slice_samples(ds, row.lo, row.hi)
        real = evaluate(sub, args.label_kind, cfg)
        nulls = []
        for i in range(args.n_perm):
            nulls.append(
                evaluate(
                    sub,
                    args.label_kind,
                    cfg,
                    key_perm=rng.permutation(sub.traces.shape[0]),
                )
            )
            if (i + 1) % max(1, args.n_perm // 5) == 0:
                print(f"[NULL {row.lo}:{row.hi}] {i + 1}/{args.n_perm}")
        summary = summarize(args.label_kind, cfg, real, nulls)
        lines.append(f"[{row.lo}:{row.hi}] {summary}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("[SUMMARY]")
    print("\n".join(lines))
    print(f"[OK] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
