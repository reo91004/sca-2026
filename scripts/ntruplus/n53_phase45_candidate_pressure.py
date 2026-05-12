#!/usr/bin/env python3
"""Top-k candidate-set pressure from Phase 4.5/4.6 candidate export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "results/ntruplus/phase45/candidate_export.npz"
DEFAULT_OUT = ROOT / "results/ntruplus/phase45/candidate_pressure"
CHANNELS = ["M1", "M5", "M1_slot", "M5_slot", "Full"]
UNIONS = [
    ("M1∪M5", ["M1", "M5"]),
    ("M1∪Full", ["M1", "Full"]),
    ("M5∪Full", ["M5", "Full"]),
    ("M1∪M5∪Full", ["M1", "M5", "Full"]),
    ("All", CHANNELS),
]
KS = [1, 5, 10, 20, 50, 100]


def in_top(row: dict, channel: str, k: int) -> bool:
    cands = row["candidates"][channel]
    if len(cands) == 0:
        return False
    return int(row["true_f"]) in {int(x) for x in cands[:k]}


def union_set(row: dict, channels: list[str], k: int) -> set[int]:
    out: set[int] = set()
    for ch in channels:
        out.update(int(x) for x in row["candidates"][ch][:k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    rows = [dict(r) for r in z["rows"]]
    n = len(rows)

    md = ["# Phase 4.5/4.6 candidate pressure", "",
          f"- input: `{args.input}`",
          f"- rows: {n}", ""]

    md.append("## Single Channels")
    md.append("| channel | " + " | ".join(f"top-{k}" for k in KS) + " |")
    md.append("|---|" + "|".join("---:" for _ in KS) + "|")
    single_summary = []
    for ch in CHANNELS:
        counts = [sum(1 for r in rows if in_top(r, ch, k)) for k in KS]
        single_summary.append((ch, counts))
        md.append("| " + ch + " | " + " | ".join(f"{c}/{n}" for c in counts) + " |")

    md.append("")
    md.append("## Channel Unions")
    md.append("| union | " + " | ".join(f"top-{k}" for k in KS) + " |")
    md.append("|---|" + "|".join("---:" for _ in KS) + "|")
    union_summary = []
    for label, channels in UNIONS:
        counts = []
        avg_sizes = []
        for k in KS:
            sets = [union_set(r, channels, k) for r in rows]
            counts.append(sum(1 for r, s in zip(rows, sets) if int(r["true_f"]) in s))
            avg_sizes.append(float(np.mean([len(s) for s in sets])))
        union_summary.append((label, counts, avg_sizes))
        md.append("| " + label + " | " + " | ".join(f"{c}/{n}" for c in counts) + " |")

    md.append("")
    md.append("## Average Union Candidate Set Size")
    md.append("| union | " + " | ".join(f"top-{k}" for k in KS) + " |")
    md.append("|---|" + "|".join("---:" for _ in KS) + "|")
    for label, _counts, avg_sizes in union_summary:
        md.append("| " + label + " | " + " | ".join(f"{x:.1f}" for x in avg_sizes) + " |")

    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- Top-k pressure is still sparse at k <= 10; most signal is a candidate-set "
              "constraint rather than direct recovery.")
    md.append("- The all-channel union is the right input for downstream pruning, while "
              "single-channel rows remain the right way to report standalone attack strength.")

    Path(f"{args.out_prefix}.md").write_text("\n".join(md) + "\n")
    np.savez_compressed(
        f"{args.out_prefix}.npz",
        ks=np.array(KS, dtype=np.int32),
        single=np.array(single_summary, dtype=object),
        unions=np.array(union_summary, dtype=object),
    )
    print(f"[OK] wrote {args.out_prefix}.{{md,npz}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
