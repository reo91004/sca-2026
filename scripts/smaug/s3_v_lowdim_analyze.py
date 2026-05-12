#!/usr/bin/env python3
"""S3 V low-dimensional profiling.

This is a diagnostic-only companion to the S2 Z low-dimensional analysis. It
loads existing single-design `V` captures, wraps them as a D=1 matrix dataset,
and reuses the same trace-to-label evaluator and permutation null.

The purpose is not an attack claim. With the current five-key V dataset this is
only a localization sanity check: if `V` does not predict the same Toom labels
more clearly than `Z`, the sub-trigger is probably not isolating the useful
state or the current label/window alignment is wrong.
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
from scripts.smaug.s2_z_lowdim_analyze import (  # noqa: E402
    LABEL_KINDS,
    Config,
    Dataset,
    evaluate,
    parse_sample_range,
    summarize,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s3_v*_a4_n200.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument(
        "--label-kinds",
        nargs="+",
        choices=LABEL_KINDS,
        default=[
            "toom0_conv16_hw",
            "toom1_conv16_hw",
            "toom2_conv16_hw",
            "toom3_conv16_hw",
            "toom4_conv16_hw",
            "toom5_conv16_hw",
            "toom6_conv16_hw",
        ],
    )
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument("--sample-range", default=None)
    p.add_argument("--max-traces", type=int, default=None)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--seed", type=int, default=0x53A2026)
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s3_v_lowdim",
    )
    return p.parse_args()


def load_v_dataset(paths: list[Path], component: int) -> Dataset:
    traces = []
    sks = []
    pkfps = []
    terms_ref: list[tuple[int, int]] | None = None
    seen = set()
    for path in paths:
        try:
            data = np.load(path, allow_pickle=True)
            meta = data["meta"].item()
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
            continue
        if meta.get("cmd") != "V":
            continue
        pk = str(meta.get("pk_fp16", ""))
        if not pk or pk in seen:
            continue
        x = np.asarray(data["traces"], dtype=np.float64)
        if x.ndim != 2:
            print(f"[WARN] skip {path.name}: traces shape {x.shape} != (N,T)")
            continue
        comp = int(meta.get("component", 0))
        coef = int(meta.get("coef_idx", 0))
        alpha = int(meta.get("alpha", 0))
        if comp != component:
            print(f"[WARN] skip {path.name}: component {comp} != requested {component}")
            continue
        terms = [(coef, alpha)]
        if terms_ref is None:
            terms_ref = terms
        elif terms != terms_ref:
            print(f"[WARN] skip {path.name}: design {terms} differs from {terms_ref}")
            continue
        full = _codec.unpack_sx(bytes.fromhex(meta["sk_pke_hex"])).astype(np.int8)
        lo = component * 256
        hi = lo + 256
        traces.append(x)
        sks.append(full[lo:hi])
        pkfps.append(pk)
        seen.add(pk)
    if not traces or terms_ref is None:
        raise ValueError("no usable V captures")
    min_n = min(x.shape[0] for x in traces)
    min_t = min(x.shape[1] for x in traces)
    traces3 = [x[:min_n, :min_t] for x in traces]
    return Dataset(
        traces=np.stack(traces3)[:, None, :, :],
        sks=np.stack(sks),
        design_terms=[terms_ref],
        pkfps=pkfps,
    )


def apply_slices(ds: Dataset, sample_range: str | None, max_traces: int | None) -> tuple[Dataset, tuple[int, int]]:
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


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = load_v_dataset(args.inputs, args.component)
    ds, (sample_lo, sample_hi) = apply_slices(ds, args.sample_range, args.max_traces)
    if ds.traces.shape[0] < 4:
        print(f"[FAIL] need at least 4 keys, got {ds.traces.shape[0]}")
        return 1
    print(
        f"[INFO] S={ds.traces.shape[0]} D={ds.traces.shape[1]} "
        f"N={ds.traces.shape[2]} T={ds.traces.shape[3]} terms={ds.design_terms}"
    )

    if args.sweep:
        configs = [
            Config(block, nf, ridge, mode)
            for block in (8, 16, 32, 64)
            for nf in (16, 32, 64, 128)
            for ridge in (1.0, 10.0, 100.0)
            for mode in ("snr", "corr")
        ]
    else:
        configs = [Config(args.block, args.n_features, args.ridge, args.feature_mode)]

    lines = [
        "S3 V low-dimensional profiling",
        f"S={ds.traces.shape[0]}, D={ds.traces.shape[1]}, N={ds.traces.shape[2]}, T={ds.traces.shape[3]}",
        f"sample_range=[{sample_lo}:{sample_hi}]",
        f"terms={ds.design_terms}",
        "",
    ]
    plot_rows = []
    for kind in args.label_kinds:
        print(f"[KIND] {kind}")
        rows = []
        for cfg in configs:
            real = evaluate(ds, kind, cfg)
            rows.append((float(real["exact"]), -float(real["rounded_mae"]), float(real["corr"]), cfg, real))
        rows.sort(reverse=True, key=lambda row: (row[0], row[1], row[2]))
        best_cfg = rows[0][3]
        best_real = rows[0][4]
        print(
            f"[BEST-REAL] {kind} {best_cfg.name} "
            f"exact={best_real['exact']:.4f} rMAE={best_real['rounded_mae']:.4f} "
            f"corr={best_real['corr']:.4f}"
        )
        if args.sweep:
            lines.append(f"{kind} real-only top configs:")
            for _, _, _, cfg, real in rows[:5]:
                lines.append(
                    f"  {cfg.name}: exact={real['exact']:.4f}, "
                    f"rMAE={real['rounded_mae']:.4f}, corr={real['corr']:.4f}"
                )

        nulls = []
        for i in range(args.n_perm):
            nulls.append(
                evaluate(
                    ds,
                    kind,
                    best_cfg,
                    key_perm=rng.permutation(ds.traces.shape[0]),
                )
            )
            if (i + 1) % max(1, args.n_perm // 5) == 0:
                print(f"[NULL {kind}] {i + 1}/{args.n_perm}")
        summary = summarize(kind, best_cfg, best_real, nulls)
        print("[SUMMARY] " + summary)
        lines.append(summary)
        lines.append("")
        plot_rows.append((kind, best_real, nulls))

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = args.out_prefix.with_suffix(".txt")
    txt_path.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(len(plot_rows), 3, figsize=(13, 3.2 * len(plot_rows)))
    if len(plot_rows) == 1:
        axes = np.asarray([axes])
    for row_i, (kind, real, nulls) in enumerate(plot_rows):
        vals = [
            (np.asarray([n["exact"] for n in nulls], dtype=np.float64), float(real["exact"]), "exact"),
            (np.asarray([n["rounded_mae"] for n in nulls], dtype=np.float64), float(real["rounded_mae"]), "rounded MAE"),
            (np.asarray([n["corr"] for n in nulls], dtype=np.float64), float(real["corr"]), "corr"),
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
