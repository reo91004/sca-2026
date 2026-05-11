#!/usr/bin/env python3
"""Null significance for Phase 4.5/4.6 candidate sets.

Top-k candidate counts can look impressive even when the candidate set itself
is large.  This script compares observed true-in-candidate hits against the
uniform-rank null induced by the actual set sizes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "results/ntruplus768/phase45/candidate_export.npz"
DEFAULT_OUT = ROOT / "results/ntruplus768/phase45/candidate_null"
Q_MINUS_1 = 3456
CHANNELS = ["M1", "M5", "M1_slot", "M5_slot", "Full"]
UNIONS = [
    ("M1∪M5", ["M1", "M5"]),
    ("M1∪Full", ["M1", "Full"]),
    ("M5∪Full", ["M5", "Full"]),
    ("M1∪M5∪Full", ["M1", "M5", "Full"]),
    ("All", CHANNELS),
]
KS = [1, 5, 10, 20, 50, 100]


def candidate_set(row: dict, channels: list[str], k: int) -> set[int]:
    out: set[int] = set()
    for ch in channels:
        out.update(int(x) for x in row["candidates"][ch][:k])
    return out


def normal_sf(z: float) -> float:
    """Survival function for standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def summarize(rows: list[dict], channels: list[str], k: int) -> dict:
    sets = [candidate_set(r, channels, k) for r in rows]
    obs = sum(1 for r, s in zip(rows, sets) if int(r["true_f"]) in s)
    ps = np.array([len(s) / Q_MINUS_1 for s in sets], dtype=np.float64)
    exp = float(ps.sum())
    var = float(np.sum(ps * (1.0 - ps)))
    z = (obs - exp) / math.sqrt(var) if var > 0 else 0.0
    return {
        "obs": obs,
        "exp": exp,
        "avg_size": float(np.mean([len(s) for s in sets])),
        "z": z,
        "p_one_sided": normal_sf(z),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    rows = [dict(r) for r in z["rows"]]

    single_rows = []
    union_rows = []

    md = ["# Phase 4.5/4.6 candidate-set null significance", "",
          f"- input: `{args.input}`",
          f"- rows: {len(rows)}",
          f"- null: true residue uniform in 1..3456 with actual candidate-set sizes",
          ""]

    md.append("## Single Channels")
    md.append("| channel | k | obs | null exp | avg set size | z | one-sided p |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for ch in CHANNELS:
        for k in KS:
            s = summarize(rows, [ch], k)
            single_rows.append((ch, k, s))
            md.append(f"| {ch} | {k} | {s['obs']} | {s['exp']:.2f} | "
                      f"{s['avg_size']:.1f} | {s['z']:.2f} | {s['p_one_sided']:.3g} |")

    md.append("")
    md.append("## Channel Unions")
    md.append("| union | k | obs | null exp | avg set size | z | one-sided p |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, channels in UNIONS:
        for k in KS:
            s = summarize(rows, channels, k)
            union_rows.append((label, k, s))
            md.append(f"| {label} | {k} | {s['obs']} | {s['exp']:.2f} | "
                      f"{s['avg_size']:.1f} | {s['z']:.2f} | {s['p_one_sided']:.3g} |")

    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- Phase 4.5/4.6 single-victim candidate sets do not exceed the simple "
              "uniform-rank null by a meaningful margin at top-100.")
    md.append("- The candidate export is still useful as an engineering artifact, but "
              "paper claims should rely on Phase 4 multi-victim TOP-1/top-rank "
              "cases and mechanistic leakage, not on raw Phase 4.5 top-100 "
              "candidate-set counts.")
    md.append("- For the next experiment, improving score sharpness with a larger G "
              "or a better model is more important than merely unioning more "
              "wide candidate sets.")

    Path(f"{args.out_prefix}.md").write_text("\n".join(md) + "\n")
    np.savez_compressed(
        f"{args.out_prefix}.npz",
        single=np.array(single_rows, dtype=object),
        unions=np.array(union_rows, dtype=object),
    )
    print(f"[OK] wrote {args.out_prefix}.{{md,npz}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
