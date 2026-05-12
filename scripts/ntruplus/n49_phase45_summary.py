#!/usr/bin/env python3
"""Phase 4.5 final summary + figures.

Run after main capture completes. Produces:
  - F10: per-victim recovery summary (top-1/10/100 per victim)
  - F11: lane-by-lane recovery hit rate (heatmap)
  - F12: cumulative coords vs lanes covered
  - results/ntruplus/phase45/SUMMARY.md (paper-ready statistics)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

P45 = ROOT / "results/ntruplus/phase45"
FIGS = P45 / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    combined_npz = P45 / "combined.npz"
    if not combined_npz.exists():
        print(f"[ERR] combined.npz not found at {combined_npz}")
        print("      run scripts/ntruplus/n45_singleVictim_combine.py first")
        return 1
    z = np.load(combined_npz, allow_pickle=True)
    rows = list(z["rows"])
    victims_meta = z["victims"].item() if "victims" in z.files else {}
    print(f"[INFO] {len(rows)} cases over {len(victims_meta)} victims")

    by_victim = {}
    for r in rows:
        by_victim.setdefault(r["victim"], []).append(r)

    # ----- F10: per-victim recovery summary --------------------------
    if len(by_victim) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        victims = list(by_victim.keys())
        x = np.arange(len(victims))
        bw = 0.18

        # Left: top-1, 10, 100, 500 counts per victim (M1 baseline)
        ax = axes[0]
        for j, (label, t) in enumerate([("top-1", 1), ("top-10", 10),
                                         ("top-100", 100), ("top-500", 500)]):
            counts = []
            for v in victims:
                rs = np.array([r["rk_M1_pred"] for r in by_victim[v]])
                counts.append((rs < t).sum())
            ax.bar(x + (j - 1.5) * bw, counts, bw, label=label,
                    alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([v[:8] for v in victims], rotation=45, ha="right",
                            fontsize=8)
        ax.set_ylabel("# cases recovered")
        ax.set_title("M1 baseline (predict_poi, profile-free)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # Right: full stack
        ax = axes[1]
        for j, (label, t) in enumerate([("top-1", 1), ("top-10", 10),
                                         ("top-100", 100), ("top-500", 500)]):
            counts = []
            for v in victims:
                rs = np.array([r["rk_full"] for r in by_victim[v]
                                if r["rk_full"] >= 0])
                counts.append((rs < t).sum() if len(rs) > 0 else 0)
            ax.bar(x + (j - 1.5) * bw, counts, bw, label=label,
                    alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([v[:8] for v in victims], rotation=45, ha="right",
                            fontsize=8)
        ax.set_ylabel("# cases recovered")
        ax.set_title("Full stack (M1+M5 Zsum, calibrated lanes only)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        fig.suptitle("F10: Per-victim Phase 4.5 recovery", y=1.02)
        fig.tight_layout()
        out = FIGS / "F10_per_victim_recovery.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] F10 → {out}")

    # ----- F11: lane × victim heatmap (top-100 hit) ------------------
    all_lanes = sorted(set(r["lane"] for r in rows))
    if len(all_lanes) > 0 and len(by_victim) > 0:
        h_M1 = np.full((len(victims), len(all_lanes)), np.nan)
        h_full = np.full_like(h_M1, np.nan)
        for vi, v in enumerate(victims):
            for li, lane in enumerate(all_lanes):
                sub = [r for r in by_victim[v] if r["lane"] == lane]
                if not sub:
                    continue
                m1_t100 = sum(1 for r in sub if r["rk_M1_pred"] < 100)
                full_t100 = sum(1 for r in sub
                                 if r["rk_full"] >= 0 and r["rk_full"] < 100)
                h_M1[vi, li] = m1_t100
                h_full[vi, li] = full_t100

        fig, axes = plt.subplots(2, 1, figsize=(max(8, len(all_lanes) * 0.25), 4),
                                   sharex=True)
        for ax, h, title in [(axes[0], h_M1, "M1 baseline"),
                              (axes[1], h_full, "Full stack")]:
            im = ax.imshow(h, cmap="Greens", aspect="auto", vmin=0, vmax=4)
            for vi in range(h.shape[0]):
                for li in range(h.shape[1]):
                    v = h[vi, li]
                    if not np.isnan(v):
                        col = "white" if v > 1 else "black"
                        ax.text(li, vi, f"{int(v)}", ha="center", va="center",
                                 color=col, fontsize=7)
            ax.set_yticks(range(len(victims)))
            ax.set_yticklabels([v[:8] for v in victims], fontsize=7)
            ax.set_title(title)
        axes[1].set_xticks(range(len(all_lanes)))
        axes[1].set_xticklabels(all_lanes, rotation=90, fontsize=7)
        axes[1].set_xlabel("lane")
        fig.suptitle("F11: Per-(victim, lane) top-100 hits", y=1.02)
        fig.tight_layout()
        out = FIGS / "F11_lane_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] F11 → {out}")

    # ----- F12: cumulative coords vs lanes coverage ------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in victims:
        sub = sorted(by_victim[v], key=lambda r: r["lane"])
        cum = []
        seen_lanes = set()
        cum_M1 = 0; cum_full = 0
        xs = []; ys_M1 = []; ys_full = []
        for r in sub:
            seen_lanes.add(r["lane"])
            if r["rk_M1_pred"] < 100:
                cum_M1 += 1
            if r["rk_full"] >= 0 and r["rk_full"] < 100:
                cum_full += 1
            xs.append(len(seen_lanes))
            ys_M1.append(cum_M1)
            ys_full.append(cum_full)
        ax.plot(xs, ys_M1, marker="o", linestyle="-", alpha=0.7,
                 label=f"{v[:8]} M1")
        ax.plot(xs, ys_full, marker="s", linestyle="--", alpha=0.5,
                 label=f"{v[:8]} full")
    ax.set_xlabel("# lanes covered")
    ax.set_ylabel("cumulative top-100 coords")
    ax.set_title("F12: Cumulative recovery vs lane coverage")
    # 192-lane projection lines from Phase 4 stats
    ax.plot([0, 192], [0, 192 * 0.167], "k:", alpha=0.4,
             label="M1 projection (0.167/lane)")
    ax.plot([0, 192], [0, 192 * 0.150], "k--", alpha=0.4,
             label="Full proj (0.150/lane)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "F12_cumulative_lanes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F12 → {out}")

    # ----- SUMMARY.md --------------------------------------------------
    md = ["# Phase 4.5 — Single-victim multi-lane summary", "",
          "## Captures", "",
          "| file | victim | L | N | cases |",
          "|---|---|---:|---:|---:|"]
    for v, info in victims_meta.items() if hasattr(victims_meta, "items") else []:
        md.append(f"| {info.get('file', '?')} | {v} | {info.get('L', '?')} "
                  f"| {info.get('N', '?')} | {info.get('cases', '?')} |")

    md += ["", "## Aggregate"]
    md.append("| pipeline | top-1 | top-10 | top-100 | top-500 |")
    md.append("|---|---:|---:|---:|---:|")
    for label, key in [("M1 baseline", "rk_M1_pred"),
                       ("M5 baseline", "rk_M5_pred"),
                       ("M1 slot-PoI (calib)", "rk_M1_slot"),
                       ("M5 slot-PoI (calib)", "rk_M5_slot"),
                       ("Full stack", "rk_full")]:
        rks = np.array([r[key] for r in rows if r[key] >= 0])
        if len(rks) == 0:
            md.append(f"| {label} | — | — | — | — |")
            continue
        md.append(f"| {label} | {(rks==0).sum()}/{len(rks)} "
                  f"| {(rks<10).sum()}/{len(rks)} "
                  f"| {(rks<100).sum()}/{len(rks)} "
                  f"| {(rks<500).sum()}/{len(rks)} |")

    md += ["", "## Per-victim", "",
           "| victim | L | N | cases | M1 t100 | full t100 |",
           "|---|---:|---:|---:|---:|---:|"]
    for v in victims:
        sub = by_victim[v]
        m1_t100 = sum(1 for r in sub if r["rk_M1_pred"] < 100)
        full_t100 = sum(1 for r in sub
                         if r["rk_full"] >= 0 and r["rk_full"] < 100)
        info = victims_meta.get(v, {}) if hasattr(victims_meta, "get") else {}
        md.append(f"| {v[:8]} | {info.get('L','?')} | {info.get('N','?')} "
                  f"| {len(sub)} | {m1_t100} | {full_t100} |")

    md += ["", "## Top-100 hits (M1 baseline)",
           "| victim | lane | slot | true f | |f_c| | rk_M1_pred |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: r["rk_M1_pred"]):
        if r["rk_M1_pred"] < 100:
            md.append(f"| {r['victim'][:8]} | {r['lane']} | {r['slot']} "
                      f"| {r['true_f']} | {r['abs_f_c']} | {r['rk_M1_pred']} |")
    out = P45 / "SUMMARY.md"
    out.write_text("\n".join(md) + "\n")
    print(f"[OK] summary → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
