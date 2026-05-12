"""Matplotlib 기반 PNG 산출.

설계 원칙:
  - 모든 함수는 `Agg` 백엔드 강제 (서버/CI 안전).
  - figure 객체를 반환하지 않고 path 를 받아 저장 + 닫기까지 한다 — 호출
    측이 leak 시킬 위험을 없앤다.
  - 색은 matplotlib 기본 cycle 만 사용 (의존성 추가 X).

함수:
  plot_overview(cap, png)          단일 캡처 평균/표준편차/오버레이.
  plot_tvla(tvla_result, png, ...) Welch-t 곡선 + threshold 음영.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (Agg 강제 후 import)
import numpy as np  # noqa: E402

from .io import Capture  # noqa: E402
from .tvla import TvlaResult  # noqa: E402


_DEFAULT_DPI = 130


def plot_overview(
    cap: Capture,
    png_path: str | Path,
    *,
    n_overlay: int = 5,
    seed: int = 0,
) -> Path:
    """평균 ± 1σ 영역 + 무작위 트레이스 오버레이 + 표준편차 단독 panel."""
    p = Path(png_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    traces = cap.traces
    x = np.arange(traces.shape[1])
    mean = traces.mean(axis=0)
    std = traces.std(axis=0)

    rng = np.random.default_rng(seed)
    k = min(n_overlay, traces.shape[0])
    idx = rng.choice(traces.shape[0], size=k, replace=False) if k else np.array([], dtype=int)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.fill_between(x, mean - std, mean + std, color="C0", alpha=0.25,
                     label=r"$\mu \pm \sigma$")
    ax1.plot(x, mean, color="C0", lw=0.8, label=r"$\mu$")
    for i in idx:
        ax1.plot(x, traces[i], lw=0.4, alpha=0.6, label=f"trace {i}")

    target = cap.meta.get("target", "?")
    cmd = cap.meta.get("cmd", "?")
    gain = cap.meta.get("gain_db", "?")
    ax1.set_ylabel("ADC (norm.)")
    ax1.set_title(
        f"{target} overview  N={traces.shape[0]} S={traces.shape[1]}  "
        f"cmd={cmd}  gain={gain}dB  src={p.stem}"
    )
    ax1.legend(loc="upper right", fontsize=8, ncols=2)
    ax1.grid(alpha=0.3)

    ax2.plot(x, std, color="C3", lw=0.6)
    ax2.set_ylabel(r"$\sigma$")
    ax2.set_xlabel("sample")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(p, dpi=_DEFAULT_DPI)
    plt.close(fig)
    return p


def plot_tvla(
    result: TvlaResult,
    png_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Welch t-statistic 곡선 + ±threshold 가이드라인 + PoI 음영."""
    p = Path(png_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    t = result.t
    x = np.arange(t.size)

    fig, (ax_t, ax_d) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # 위 panel: t-statistic
    ax_t.plot(x, t, color="C0", lw=0.7, label="Welch t")
    ax_t.axhline(result.threshold, color="C3", lw=0.6, ls="--",
                 label=f"±{result.threshold} (TVLA)")
    ax_t.axhline(-result.threshold, color="C3", lw=0.6, ls="--")
    leaky = result.leaky_idx
    if leaky.size:
        # 누출 영역을 얇은 빨간 음영 — 점 단위로 vlines 보다 가벼움.
        ax_t.fill_between(x, -result.threshold, result.threshold,
                          where=np.abs(t) > result.threshold,
                          color="C3", alpha=0.15, step="mid",
                          label="|t|>threshold")
    ax_t.set_ylabel("t-statistic")
    ax_t.set_title(
        title
        or f"Welch-t  N_A={result.n_a} N_B={result.n_b}  "
           f"max|t|={result.max_abs_t:.2f}  leaky={leaky.size}"
    )
    ax_t.legend(loc="upper right", fontsize=8)
    ax_t.grid(alpha=0.3)

    # 아래 panel: 단순 평균 차분 (PoI 시각화 보조)
    diff = result.mu_a - result.mu_b
    ax_d.plot(x, diff, color="C1", lw=0.6, label=r"$\mu_A - \mu_B$")
    ax_d.axhline(0, color="k", lw=0.4)
    ax_d.set_xlabel("sample")
    ax_d.set_ylabel("Δμ")
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(p, dpi=_DEFAULT_DPI)
    plt.close(fig)
    return p
