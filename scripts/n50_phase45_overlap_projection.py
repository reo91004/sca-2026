#!/usr/bin/env python3
"""Phase 4.5/4.6 overlap and 192-lane projection summary.

Reads the small combined result table produced by n45/n49 and computes
channel overlap, union top-k counts, per-victim yield, and conservative
192-lane projections.  This is capture-free and only writes a compact
Markdown/NPZ result under results/.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "results/ntruplus768/phase45/combined.npz"
DEFAULT_OUT = ROOT / "results/ntruplus768/phase45/overlap_projection"
Q = 3457
SK_ENTROPY_BITS = 1152.0
FULL_LANES = 192


PIPELINES = [
    ("M1 baseline", "rk_M1_pred"),
    ("M5 baseline", "rk_M5_pred"),
    ("M1 slot-PoI", "rk_M1_slot"),
    ("M5 slot-PoI", "rk_M5_slot"),
    ("Full stack", "rk_full"),
]


def hit(row: dict, key: str, thresh: int) -> bool:
    rk = int(row.get(key, -1))
    return rk >= 0 and rk < thresh


def count_hits(rows: list[dict], key: str, thresh: int) -> int:
    return sum(1 for r in rows if hit(r, key, thresh))


def union_hits(rows: list[dict], keys: list[str], thresh: int) -> int:
    return sum(1 for r in rows if any(hit(r, k, thresh) for k in keys))


def projection(hits: int, lane_visits: int) -> float:
    return FULL_LANES * hits / lane_visits if lane_visits else 0.0


def entropy_pct(coords: float) -> float:
    return 100.0 * coords * math.log2(Q) / SK_ENTROPY_BITS


def fmt_float(x: float) -> str:
    return f"{x:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    rows = [dict(r) for r in z["rows"]]
    victims = z["victims"].item() if "victims" in z.files else {}
    lane_visits = sum(int(info["L"]) for info in victims.values())
    cases = len(rows)

    print(f"[INFO] rows={cases} victims={len(victims)} lane_visits={lane_visits}")

    summary_rows = []
    for label, key in PIPELINES:
        top1 = count_hits(rows, key, 1)
        top10 = count_hits(rows, key, 10)
        top100 = count_hits(rows, key, 100)
        top500 = count_hits(rows, key, 500)
        proj = projection(top100, lane_visits)
        summary_rows.append((label, top1, top10, top100, top500, proj, entropy_pct(proj)))

    unions = [
        ("M1 baseline ∪ M5 baseline", ["rk_M1_pred", "rk_M5_pred"]),
        ("M1 baseline ∪ Full stack", ["rk_M1_pred", "rk_full"]),
        ("M5 baseline ∪ Full stack", ["rk_M5_pred", "rk_full"]),
        ("M1 ∪ M5 ∪ Full", ["rk_M1_pred", "rk_M5_pred", "rk_full"]),
    ]
    union_rows = []
    for label, keys in unions:
        top10 = union_hits(rows, keys, 10)
        top100 = union_hits(rows, keys, 100)
        top500 = union_hits(rows, keys, 500)
        proj = projection(top100, lane_visits)
        union_rows.append((label, top10, top100, top500, proj, entropy_pct(proj)))

    per_victim_rows = []
    for victim, info in victims.items():
        sub = [r for r in rows if r["victim"] == victim]
        L = int(info["L"])
        m1 = count_hits(sub, "rk_M1_pred", 100)
        m5 = count_hits(sub, "rk_M5_pred", 100)
        full = count_hits(sub, "rk_full", 100)
        union_all = union_hits(sub, ["rk_M1_pred", "rk_M5_pred", "rk_full"], 100)
        per_victim_rows.append((victim, L, int(info["N"]), len(sub), m1, m5, full,
                                union_all, projection(union_all, L)))

    md: list[str] = []
    md.append("# Phase 4.5/4.6 overlap and projection")
    md.append("")
    md.append(f"- input: `{args.input}`")
    md.append(f"- victims: {len(victims)}")
    md.append(f"- lane visits: {lane_visits}")
    md.append(f"- cases: {cases}")
    md.append(f"- projection basis: {FULL_LANES} lanes / victim")
    md.append("")
    md.append("## Single Pipelines")
    md.append("| pipeline | top-1 | top-10 | top-100 | top-500 | projected coords | entropy upper % |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, top1, top10, top100, top500, proj, epct in summary_rows:
        md.append(f"| {label} | {top1}/{cases} | {top10}/{cases} | "
                  f"{top100}/{cases} | {top500}/{cases} | "
                  f"{fmt_float(proj)} | {fmt_float(epct)} |")
    md.append("")
    md.append("## Unions")
    md.append("| union | top-10 | top-100 | top-500 | projected coords | entropy upper % |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for label, top10, top100, top500, proj, epct in union_rows:
        md.append(f"| {label} | {top10}/{cases} | {top100}/{cases} | "
                  f"{top500}/{cases} | {fmt_float(proj)} | {fmt_float(epct)} |")
    md.append("")
    md.append("## Per Victim")
    md.append("| victim | L | N | cases | M1 t100 | M5 t100 | full t100 | union t100 | union projected coords |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for victim, L, N, ncases, m1, m5, full, union_all, proj in per_victim_rows:
        md.append(f"| {victim[:8]} | {L} | {N} | {ncases} | {m1} | {m5} | "
                  f"{full} | {union_all} | {fmt_float(proj)} |")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- The most conservative single-pipeline projection is the full-stack "
              "yield; the best current single-channel projection is M5 baseline.")
    md.append("- Union counts are useful for quantifying information disclosure, but "
              "they mix channels and should be stated separately from any one "
              "attack pipeline.")
    md.append("- All entropy percentages are upper bounds because a top-100 rank is "
              "not a perfect coefficient recovery.")

    md_path = Path(f"{args.out_prefix}.md")
    md_path.write_text("\n".join(md) + "\n")
    np.savez_compressed(
        f"{args.out_prefix}.npz",
        summary=np.array(summary_rows, dtype=object),
        unions=np.array(union_rows, dtype=object),
        per_victim=np.array(per_victim_rows, dtype=object),
    )
    print(f"[OK] wrote {args.out_prefix}.{{md,npz}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
