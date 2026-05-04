#!/usr/bin/env python3
"""Leakage phase scan — broad Welch-t TVLA over the full trace length.

목적
----
Step 0 진단에서 design-window equal-window PoI 가 random level 로 죽었음을 확인.
원인 가설: trace 24400 sample 안에서 leakage 가 *균등* 분포가 아니라 특정
phase (rounding/store loop) 에 모여 있고, 256-등분 window 가 그 phase 를 못 잡음.

본 스크립트는 다음 두 broad TVLA 를 산출한다:
1. **Oracle-pair TVLA**  (α=64 vs α=192, smaug1 attack capture)
   — design label 만 사용. attack-valid. trace 어디에 |t| 가 모이는지 확인.
2. **Multi-term beacon TVLA**  (1-term vs 2-term vs 3-term, H_attack_n256)
   — multi-term 이 µ′ HW 를 키워 같은 phase 의 신호 amplitude 를 키운다는 가설.

산출:
    results/leakage_phase_<tag>.npz   max|t| trace, top-K sample idx
    results/leakage_phase_<tag>.png   plot

사용:
    python3 scripts/leakage_phase_scan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.zeros(a.shape[1], dtype=np.float64)
    ma = a.mean(axis=0); mb = b.mean(axis=0)
    va = a.var(axis=0, ddof=1).clip(min=1e-12)
    vb = b.var(axis=0, ddof=1).clip(min=1e-12)
    return (ma - mb) / np.sqrt(va / a.shape[0] + vb / b.shape[0])


def hot_regions(t: np.ndarray, threshold: float = 4.5,
                merge_gap: int = 50) -> list[tuple[int, int, float]]:
    """|t| > threshold 인 sample 들을 인접 영역으로 merge. 각 region (lo, hi, max|t|)."""
    mask = np.abs(t) > threshold
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    regions = []
    lo = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap:
            regions.append((int(lo), int(prev), float(np.abs(t[lo:prev + 1]).max())))
            lo = i
        prev = i
    regions.append((int(lo), int(prev), float(np.abs(t[lo:prev + 1]).max())))
    return regions


def scan_oracle_pair(npz_paths: list[Path], out_prefix: str) -> dict:
    """Oracle pair α_pos vs α_neg broad TVLA across multiple seeds."""
    print(f"\n[oracle-pair] {len(npz_paths)} seeds")
    t_max_per = None
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        T = d["traces"]
        meta = d["meta"].item()
        ap = int(meta.get("alpha_pos", 64))
        an = int(meta.get("alpha_neg", 192))
        la = np.asarray(meta["label_alpha"])
        # component-merged (attacker labels: alpha only).
        t = welch_t(T[la == ap], T[la == an])
        if t_max_per is None:
            t_max_per = np.abs(t)
        else:
            t_max_per = np.maximum(t_max_per, np.abs(t))
        print(f"  {path.name:45s} max|t|={float(np.abs(t).max()):6.2f}  "
              f"@sample={int(np.argmax(np.abs(t))):>5d}")
    regs = hot_regions(t_max_per, threshold=4.5, merge_gap=50)
    print(f"  cumulative max|t| over {len(npz_paths)} seeds: {float(t_max_per.max()):.2f}")
    print(f"  hot regions (|t|>4.5, merge_gap=50): {len(regs)}")
    for lo, hi, mx in regs[:20]:
        print(f"    [{lo:>5d}, {hi:>5d}]  width={hi-lo:>4d}  max|t|={mx:5.2f}")

    # save
    out_npz = Path(f"{out_prefix}_oracle_pair.npz")
    np.savez(out_npz, t_max=t_max_per, regions=np.array(regs, dtype=object),
             seeds=np.array([p.name for p in npz_paths]))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t_max_per, lw=0.5)
    ax.axhline(4.5, color="r", ls="--", lw=0.5)
    ax.set_xlabel("sample"); ax.set_ylabel("max |t| over seeds")
    ax.set_title(f"Oracle-pair broad TVLA — α=64 vs α=192, {len(npz_paths)} seeds")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_oracle_pair.png", dpi=120)
    plt.close(fig)
    return {"max_t": float(t_max_per.max()), "n_regions": len(regs), "regions": regs}


def scan_multiterm_beacon(npz_path: Path, out_prefix: str) -> dict:
    """Multi-term vs 1-term broad TVLA (H_attack_n256 schema)."""
    d = np.load(npz_path, allow_pickle=True)
    T = d["traces"]
    meta = d["meta"].item()
    didx = np.asarray(meta["design_index_per_trace"])
    names = list(meta["design_names"])
    print(f"\n[beacon] {npz_path.name}: T={T.shape}, designs={names}")

    # 1-term as reference (design idx with the "const" name)
    ref_idx = next((i for i, n in enumerate(names) if "const" in n), 0)
    ref_traces = T[didx == ref_idx]

    out_regions = {}
    cumulative_t = None
    for i, name in enumerate(names):
        if i == ref_idx:
            continue
        beacon_traces = T[didx == i]
        t = welch_t(beacon_traces, ref_traces)
        regs = hot_regions(t, threshold=4.5, merge_gap=50)
        print(f"  {name:35s} (vs {names[ref_idx]}) max|t|={float(np.abs(t).max()):6.2f}, "
              f"regions={len(regs)}")
        for lo, hi, mx in regs[:5]:
            print(f"    [{lo:>5d}, {hi:>5d}]  width={hi-lo:>4d}  max|t|={mx:5.2f}")
        out_regions[name] = regs
        cumulative_t = np.abs(t) if cumulative_t is None else np.maximum(cumulative_t, np.abs(t))

    out_npz = Path(f"{out_prefix}_beacon.npz")
    np.savez(out_npz, t_max=cumulative_t,
             names=np.array(names),
             ref_name=names[ref_idx])
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(cumulative_t, lw=0.5)
    ax.axhline(4.5, color="r", ls="--", lw=0.5)
    ax.set_xlabel("sample"); ax.set_ylabel("max |t| (multi-term vs 1-term)")
    ax.set_title(f"Multi-term beacon TVLA — {npz_path.name}")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_beacon.png", dpi=120)
    plt.close(fig)
    return {"regions": out_regions, "max_t": float(cumulative_t.max())}


def main() -> int:
    out_prefix = "results/leakage_phase"
    Path("results").mkdir(exist_ok=True)

    seed_paths = sorted(Path("traces").glob("attack_seed*.npz"))
    seed_paths = [p for p in seed_paths if "smaug3" not in p.name and "smaug5" not in p.name]
    seed_paths += [Path("traces/attack_const_c1_n256.npz")]
    seed_paths = [p for p in seed_paths if p.exists()]
    scan_oracle_pair(seed_paths, out_prefix)

    beacon_path = Path("traces/H_attack_n256.npz")
    if beacon_path.exists():
        scan_multiterm_beacon(beacon_path, out_prefix)
    else:
        print(f"\n[skip] beacon {beacon_path} not found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
