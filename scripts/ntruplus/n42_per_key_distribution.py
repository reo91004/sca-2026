#!/usr/bin/env python3
"""Per-(batch, key) recovery distribution from full_stack.npz.

For Phase 4.5 capture planning: estimate expected #recovered coords per
single victim if we extend lane coverage. Each (batch, key) tuple is a
unique sk (sk_blobs differ across batches per Phase 4.5 sanity check).

Outputs:
  - results/ntruplus768/phase4/per_key_dist.npz
  - results/ntruplus768/phase4/per_key_dist.md (table)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    rows = np.load(ROOT / "results/ntruplus768/phase4/full_stack.npz",
                   allow_pickle=True)["rows"]
    cases = list(rows)
    print(f"[INFO] {len(cases)} (batch, key, slot) cases loaded\n")

    # Group by (batch, key)
    keys = sorted({(r["batch"], r["key"]) for r in cases})
    print(f"[INFO] {len(keys)} unique (batch, key) tuples = unique sk\n")

    # For each unique key, count top-100/top-500 hits among 4 slots
    bucket = {}
    for r in cases:
        bucket.setdefault((r["batch"], r["key"]),
                          {"slots": [], "lane": r["lane"]})
        bucket[(r["batch"], r["key"])]["slots"].append(r)

    rate_summary = []
    for col in ("rk_M1_pred", "rk_M1_slot", "rk_M5_slot", "rk_full"):
        per_key_top100 = []
        per_key_top500 = []
        for k, v in bucket.items():
            t100 = sum(1 for s in v["slots"] if s[col] < 100)
            t500 = sum(1 for s in v["slots"] if s[col] < 500)
            per_key_top100.append(t100)
            per_key_top500.append(t500)
        per_key_top100 = np.array(per_key_top100)
        per_key_top500 = np.array(per_key_top500)
        rate_summary.append({
            "pipeline": col,
            "n_keys": len(keys),
            "mean_top100_per_key": float(per_key_top100.mean()),
            "max_top100_per_key": int(per_key_top100.max()),
            "keys_geq1_top100": int((per_key_top100 >= 1).sum()),
            "keys_geq2_top100": int((per_key_top100 >= 2).sum()),
            "mean_top500_per_key": float(per_key_top500.mean()),
            "max_top500_per_key": int(per_key_top500.max()),
            "keys_geq1_top500": int((per_key_top500 >= 1).sum()),
        })

    print("=== Per-(batch, key) recovery distribution ===\n")
    print(f"  Total unique sks: {len(keys)} (each: 1 lane × 4 slots = 4 cases)\n")

    print(f"{'pipeline':>20} {'mean t100':>10} {'max':>5} "
          f"{'≥1 t100':>8} {'≥2 t100':>8} {'mean t500':>10} {'max':>5} {'≥1 t500':>8}")
    print("-" * 90)
    for s in rate_summary:
        print(f"{s['pipeline']:>20} {s['mean_top100_per_key']:>10.3f} "
              f"{s['max_top100_per_key']:>5} "
              f"{s['keys_geq1_top100']:>3}/{s['n_keys']:<4} "
              f"{s['keys_geq2_top100']:>3}/{s['n_keys']:<4} "
              f"{s['mean_top500_per_key']:>10.3f} "
              f"{s['max_top500_per_key']:>5} "
              f"{s['keys_geq1_top500']:>3}/{s['n_keys']:<4}")

    # Per-batch breakdown (keys vs lanes)
    print(f"\n=== Per-batch (1 batch = 1 lane × 4-8 keys) ===\n")
    print(f"{'batch':>20} {'n_keys':>7} {'lane':>5} "
          f"{'t100/key (M1pred)':>18} {'t100/key (full)':>16}")
    for batch in sorted({r["batch"] for r in cases}):
        sub = [r for r in cases if r["batch"] == batch]
        n_k = len({r["key"] for r in sub})
        lane = sub[0]["lane"]
        t100_M1 = sum(1 for r in sub if r["rk_M1_pred"] < 100) / n_k
        t100_full = sum(1 for r in sub if r["rk_full"] < 100) / n_k
        print(f"{batch:>20} {n_k:>7} {lane:>5} {t100_M1:>17.3f} {t100_full:>15.3f}")

    # Single-victim projection: 1 victim with 192 lanes × 4 slots = 768 cases
    print(f"\n=== Single-victim projection ===\n")
    print(f"  Current data: 1 victim has only 1 lane × 4 slots = 4 cases.")
    print(f"  Mean recovery (top-100, full stack): "
          f"{rate_summary[3]['mean_top100_per_key']:.3f} per (1 lane × 4 slots).")
    print(f"  Extrapolating to 192-lane coverage:")
    full_rate = rate_summary[3]["mean_top100_per_key"]
    M1_rate = rate_summary[0]["mean_top100_per_key"]
    print(f"    M1 baseline:  ~{M1_rate * 192:.1f} top-100 coords / victim "
          f"(192-lane × {M1_rate:.3f}/lane)")
    print(f"    Full stack:   ~{full_rate * 192:.1f} top-100 coords / victim "
          f"(192-lane × {full_rate:.3f}/lane)")
    print(f"    Union M1∪full: ~{(M1_rate + full_rate) * 192 * 0.7:.1f} "
          f"unique top-100 coords / victim (assuming 30% overlap)")

    # Save artifacts
    out_npz = ROOT / "results/ntruplus768/phase4/per_key_dist.npz"
    np.savez_compressed(out_npz,
                        rate_summary=np.array(rate_summary, dtype=object),
                        unique_keys=np.array(keys, dtype=object))
    print(f"\n[OK] saved → {out_npz}")

    # Markdown summary for paper / handoff
    out_md = ROOT / "results/ntruplus768/phase4/per_key_dist.md"
    lines = ["# Per-(batch, key) recovery distribution", "",
             f"From `full_stack.npz` ({len(cases)} cases, "
             f"{len(keys)} unique sks, each 1 lane × 4 slots).", "",
             "## Per-pipeline summary", "",
             "| pipeline | mean t100/key | max | ≥1 t100 | ≥2 t100 "
             "| mean t500/key | max | ≥1 t500 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in rate_summary:
        lines.append(f"| {s['pipeline']} | {s['mean_top100_per_key']:.3f} "
                     f"| {s['max_top100_per_key']} "
                     f"| {s['keys_geq1_top100']}/{s['n_keys']} "
                     f"| {s['keys_geq2_top100']}/{s['n_keys']} "
                     f"| {s['mean_top500_per_key']:.3f} "
                     f"| {s['max_top500_per_key']} "
                     f"| {s['keys_geq1_top500']}/{s['n_keys']} |")
    lines += ["", "## Single-victim 192-lane projection", "",
              f"- M1 baseline: ~{M1_rate * 192:.1f} top-100 coords per victim",
              f"- Full stack: ~{full_rate * 192:.1f} top-100 coords per victim",
              f"- Union (≈30% overlap): "
              f"~{(M1_rate + full_rate) * 192 * 0.7:.1f} unique top-100 / victim"]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[OK] saved → {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
