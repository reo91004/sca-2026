#!/usr/bin/env python3
"""[UTIL] E3a partition table heatmap plot (host-only).

Paper Section 3.1 (E3a partition table) 의 시각화 dev tool. 256 α × 3 class
ternary partition matrix 의 heatmap. paper Section 3 의 oracle pair 분석
근거.

Note (history/):
    분석 결과는 host/smaug/sk_partition.py 의 build_partition_table /
    find_oracle_pairs 함수가 main flow 에 보존. 이 plot 은 dev visualization.

용법 (legacy):
    history/scripts/util_plot_partition.py
  support only             : ±1 같고 0 만 다름 — zero/nonzero oracle
  trivial                  : 셋 다 같음

사용법:
  scripts/plot_partition.py
  scripts/plot_partition.py --png results/E3a_partition_table.png
  scripts/plot_partition.py --use-negative-alpha
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from host.smaug import params as _params  # noqa: E402
from host.smaug import sk_partition  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--level", default="smaug1")
    p.add_argument("--png", type=Path, default=None,
                   help="기본: results/E3a_partition_<level>.png")
    p.add_argument("--use-negative-alpha", action="store_true",
                   help="α 의 음수 (anticyclic wrap) 도 함께 sweep")
    p.add_argument("--list-sign-separating", action="store_true",
                   help="sign_separating α 값들을 stdout 에 print")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    p = _params.get(args.level)
    stats = sk_partition.build_partition_table(
        p, use_negative_alpha=args.use_negative_alpha,
    )
    print(f"[E3a] {stats.summary()}")

    if args.list_sign_separating:
        print("[E3a] sign-separating α 와 (µ_-1, µ_0, µ_+1):")
        for ai in stats.sign_separating_idx:
            row = tuple(int(x) for x in stats.matrix[ai])
            print(f"       α = {int(stats.alphas[ai]):>5d}  →  {row}")

    png = args.png or _REPO / "results" / f"E3a_partition_{args.level}.png"
    png.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # 위 panel: (α, class) heatmap. matrix.T 라 행 = class, 열 = α.
    im = ax_top.imshow(
        stats.matrix.T,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",  # 0=흰색 1=검정 — message bit 0/1
    )
    ax_top.set_yticks([0, 1, 2])
    ax_top.set_yticklabels([r"$s=-1$", r"$s=0$", r"$s=+1$"])
    ax_top.set_ylabel("secret coefficient")
    ax_top.set_title(
        f"E3a partition table  ({args.level}, c1=α·X^j, c2=0)  "
        f"full-sep={stats.fully_separating_idx.size} "
        f"sign-sep={stats.sign_separating_idx.size} "
        f"support-only={stats.support_only_idx.size} "
        f"trivial={stats.alphas.size - (stats.fully_separating_idx.size + stats.sign_separating_idx.size + stats.support_only_idx.size)}"
    )
    cbar = fig.colorbar(im, ax=ax_top, fraction=0.02, pad=0.02, ticks=[0, 1])
    cbar.set_label("predicted µ′_i bit")

    # 아래 panel: 카테고리별 어디 α 에서 발생하는지 marker.
    cat_color = {
        "full":    ("C2", "fully separating (3-way)"),
        "sign":    ("C0", "sign separating (+1 vs −1)"),
        "support": ("C3", "support only (|s| oracle)"),
    }
    for cat, idx in (
        ("full",    stats.fully_separating_idx),
        ("sign",    stats.sign_separating_idx),
        ("support", stats.support_only_idx),
    ):
        c, lbl = cat_color[cat]
        if idx.size:
            ax_bot.vlines(idx, 0, 1, color=c, lw=0.6, label=lbl)
    ax_bot.set_yticks([])
    ax_bot.set_xlabel("α index in U_p (0..255 for smaug1)")
    ax_bot.set_xlim(-0.5, stats.alphas.size - 0.5)
    ax_bot.legend(loc="upper right", fontsize=8, ncols=3)
    ax_bot.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[VIZ] saved {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
