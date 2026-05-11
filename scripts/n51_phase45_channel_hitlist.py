#!/usr/bin/env python3
"""Phase 4.5/4.6 channel hit-list diagnostics.

This capture-free helper answers: which (victim, lane, slot, f) cases are
recovered by each channel, how much the channels overlap, and whether hits
cluster by |f_c|.  It is intended to guide the next capture or model-design
step without creating large trace artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "results/ntruplus768/phase45/combined.npz"
DEFAULT_OUT = ROOT / "results/ntruplus768/phase45/channel_hitlist"

CHANNELS = [
    ("M1", "rk_M1_pred"),
    ("M5", "rk_M5_pred"),
    ("M1_slot", "rk_M1_slot"),
    ("M5_slot", "rk_M5_slot"),
    ("Full", "rk_full"),
]


def hit(row: dict, key: str, thresh: int = 100) -> bool:
    rk = int(row.get(key, -1))
    return 0 <= rk < thresh


def bin_abs_fc(x: int) -> str:
    if x < 16:
        return "<16"
    if x < 64:
        return "16-63"
    if x < 256:
        return "64-255"
    if x < 768:
        return "256-767"
    return ">=768"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=int, default=100)
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    rows = [dict(r) for r in z["rows"]]
    thresh = args.threshold

    hit_sets = {
        name: {
            (r["victim"], int(r["lane"]), int(r["slot"]))
            for r in rows if hit(r, key, thresh)
        }
        for name, key in CHANNELS
    }
    all_union = set().union(*hit_sets.values())

    md: list[str] = []
    md.append("# Phase 4.5/4.6 channel hit-list")
    md.append("")
    md.append(f"- input: `{args.input}`")
    md.append(f"- threshold: top-{thresh}")
    md.append(f"- cases: {len(rows)}")
    md.append("")

    md.append("## Channel Counts")
    md.append("| channel | hits | unique-only hits |")
    md.append("|---|---:|---:|")
    for name, _key in CHANNELS:
        others = set().union(*(hit_sets[n] for n, _ in CHANNELS if n != name))
        md.append(f"| {name} | {len(hit_sets[name])} | {len(hit_sets[name] - others)} |")
    md.append(f"| union | {len(all_union)} | — |")
    md.append("")

    md.append("## Pairwise Overlap")
    md.append("| A | B | overlap |")
    md.append("|---|---|---:|")
    for i, (a, _ka) in enumerate(CHANNELS):
        for b, _kb in CHANNELS[i + 1:]:
            md.append(f"| {a} | {b} | {len(hit_sets[a] & hit_sets[b])} |")
    md.append("")

    md.append("## Hit |f_c| Bins")
    bins = ["<16", "16-63", "64-255", "256-767", ">=768"]
    md.append("| channel | " + " | ".join(bins) + " |")
    md.append("|---|" + "|".join("---:" for _ in bins) + "|")
    for name, key in CHANNELS:
        counts = {b: 0 for b in bins}
        for r in rows:
            if hit(r, key, thresh):
                counts[bin_abs_fc(int(r["abs_f_c"]))] += 1
        md.append("| " + name + " | " + " | ".join(str(counts[b]) for b in bins) + " |")
    md.append("")

    md.append("## Union Hit List")
    md.append("| victim | lane | slot | true f | |f_c| | M1 | M5 | M1_slot | M5_slot | Full |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows_by_id = {
        (r["victim"], int(r["lane"]), int(r["slot"])): r
        for r in rows
    }
    for ident in sorted(all_union, key=lambda x: (x[0], x[1], x[2])):
        r = rows_by_id[ident]
        vals = []
        for _name, key in CHANNELS:
            rk = int(r.get(key, -1))
            vals.append(str(rk) if 0 <= rk < thresh else "")
        md.append(
            f"| {r['victim'][:8]} | {int(r['lane'])} | {int(r['slot'])} "
            f"| {int(r['true_f'])} | {int(r['abs_f_c'])} | "
            + " | ".join(vals)
            + " |"
        )

    out_md = Path(f"{args.out_prefix}.md")
    out_md.write_text("\n".join(md) + "\n")
    np.savez_compressed(
        f"{args.out_prefix}.npz",
        channel_names=np.array([name for name, _ in CHANNELS], dtype=object),
        hit_counts=np.array([len(hit_sets[name]) for name, _ in CHANNELS], dtype=np.int32),
        union_count=np.array(len(all_union), dtype=np.int32),
    )
    print(f"[OK] wrote {args.out_prefix}.{{md,npz}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
