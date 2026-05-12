#!/usr/bin/env python3
"""F0 (conceptual): chosen-CT NTT layout diagram.

Visualizes how a single chosen-CT design (c_ntt with one nonzero γ in
lane k slot 0) leaks 4 NTT-domain f coefficients via basemul.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "results/ntruplus768/phase4/figures"
FIGS.mkdir(parents=True, exist_ok=True)


def fig0_chosen_ct_layout():
    fig, ax = plt.subplots(figsize=(11, 6))

    # Top: c_ntt array (192 lanes, 4 coords each = 768 coords)
    # Show 16 lanes, with one (lane=k) highlighted
    n_show = 16
    lane_w = 0.5
    coord_w = lane_w / 4
    y_top = 4.5
    y_mid = 2.5
    y_bot = 0.5

    target_lane = 4
    for li in range(n_show):
        x = li * lane_w
        is_target = (li == target_lane)
        for ci in range(4):
            color = "#E74C3C" if (is_target and ci == 0) else \
                    "#FCF3CF" if is_target else "#EAEDED"
            label = "γ" if (is_target and ci == 0) else "0"
            r = Rectangle((x + ci * coord_w, y_top), coord_w, 0.5,
                          facecolor=color, edgecolor="black", linewidth=0.6)
            ax.add_patch(r)
            ax.text(x + ci * coord_w + coord_w / 2, y_top + 0.25,
                     label, ha="center", va="center", fontsize=7,
                     fontweight="bold" if is_target and ci == 0 else "normal")
        ax.text(x + lane_w / 2, y_top + 0.65, f"L{li}", ha="center",
                 fontsize=8)
    ax.text(-0.6, y_top + 0.25, r"$\hat{c}$", fontsize=14, ha="right",
             va="center", fontweight="bold")
    ax.text(n_show * lane_w + 0.4, y_top + 0.25, "...", fontsize=14,
             va="center")

    # f_ntt array (parallel)
    for li in range(n_show):
        x = li * lane_w
        is_target = (li == target_lane)
        for ci in range(4):
            color = "#85C1E9" if is_target else "#D5DBDB"
            label = f"f{ci}" if is_target else ""
            r = Rectangle((x + ci * coord_w, y_mid), coord_w, 0.5,
                          facecolor=color, edgecolor="black", linewidth=0.6)
            ax.add_patch(r)
            if label:
                ax.text(x + ci * coord_w + coord_w / 2, y_mid + 0.25,
                         label, ha="center", va="center", fontsize=7,
                         fontweight="bold")
        ax.text(x + lane_w / 2, y_mid + 0.65, f"L{li}", ha="center",
                 fontsize=8)
    ax.text(-0.6, y_mid + 0.25, r"$\hat{f}$", fontsize=14, ha="right",
             va="center", fontweight="bold")
    ax.text(n_show * lane_w + 0.4, y_mid + 0.25, "...", fontsize=14,
             va="center")

    # basemul output (highlighted lane only)
    for ci in range(4):
        color = "#27AE60" if ci == 0 else "#82E0AA"
        # γ·fc[i]+lane structure
        labels = [r"$\gamma f_0$", r"$\gamma f_1$", r"$\gamma f_2$", r"$\gamma f_3$"]
        x = target_lane * lane_w
        r = Rectangle((x + ci * coord_w, y_bot), coord_w, 0.5,
                      facecolor=color, edgecolor="black", linewidth=0.8)
        ax.add_patch(r)
        ax.text(x + ci * coord_w + coord_w / 2, y_bot + 0.25,
                 labels[ci], ha="center", va="center", fontsize=8,
                 fontweight="bold")
    ax.text(target_lane * lane_w + lane_w / 2, y_bot + 0.65,
             "leak via mod q (M1)\nand montgomery reduce (M5)",
             ha="center", fontsize=9, style="italic")

    # Arrow from highlighted c lane through f lane to output
    # basemul arrow
    x_t = target_lane * lane_w + lane_w / 2
    ax.annotate("", xy=(x_t, y_bot + 0.55), xytext=(x_t, y_mid),
                 arrowprops=dict(arrowstyle="->", color="#A04000",
                                  lw=2))
    ax.text(x_t + 0.7, (y_bot + y_mid) / 2 + 0.1,
             "poly_basemul()\n(basemul_lane)", fontsize=10,
             color="#A04000", fontweight="bold")

    # mark X^4 - zeta_k label
    ax.text(target_lane * lane_w + lane_w / 2, y_bot - 0.3,
             r"in $\mathbb{Z}_q[X] / (X^4 - \zeta_k)$",
             ha="center", fontsize=9)

    # Labels
    ax.text(n_show * lane_w / 2, 5.5, "Chosen ciphertext (NTT-domain bytes)",
             ha="center", fontsize=11, fontweight="bold")
    ax.text(n_show * lane_w / 2, 3.5, "Secret key (NTT-domain, target)",
             ha="center", fontsize=11, fontweight="bold")
    ax.text(n_show * lane_w / 2, 1.5, "basemul output (leak point)",
             ha="center", fontsize=11, fontweight="bold")

    # explanatory text
    fmt = (
        f"Selected-lane chosen-CT: $\\hat{{c}}[4k+i] = \\gamma \\cdot \\delta_{{i,0}}$, "
        f"others 0\n"
        f"Result: 4 NTT coefficients leaked per design via\n"
        f"  $r[i] = \\gamma \\cdot \\hat{{f}}[4k+i]$ mod $q$ "
        f"(after montgomery_reduce)\n"
        f"Wide-γ G=78 (HW=1+HW=2): each $(k, i)$ slot has 78 design samples\n"
        f"to identify $\\hat{{f}}[4k+i] \\in \\mathbb{{Z}}_q$"
    )
    ax.text(0.5, -1.2, fmt, ha="left", va="top", fontsize=9,
             family="serif", bbox=dict(facecolor="#F4F6F7",
                                        edgecolor="black", boxstyle="round"))

    ax.set_xlim(-1, n_show * lane_w + 1.5)
    ax.set_ylim(-2.3, 6.2)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("F0: chosen-CT NTT-domain selected-lane attack layout",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGS / "F0_chosen_ct_layout.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F0 → {out}")


if __name__ == "__main__":
    fig0_chosen_ct_layout()
