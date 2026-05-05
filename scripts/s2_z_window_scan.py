#!/usr/bin/env python3
"""Locate candidate multiplication leakage windows in T/V/Z traces.

This is an exploratory localization tool. It computes a key-dependent SNR
envelope for S1 'T', S2 'Z', and S3 'V' captures, then reports high-SNR Z
windows and template-correlation matches from the strongest T/V envelope region.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--t-inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s1_main_*c0_j0_a1*.npz")),
    )
    p.add_argument(
        "--z-inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s2_z_sk*.npz")),
    )
    p.add_argument(
        "--v-inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s3_v_sk*_a4_n200.npz")),
    )
    p.add_argument("--smooth", type=int, default=64)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-prefix", type=Path, default=_REPO / "results" / "s2_z_window_scan")
    return p.parse_args()


def _load_one(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["traces"], dtype=np.float64), data["meta"].item()


def load_key_traces(paths: list[Path], cmd: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    means = []
    vars_ = []
    counts = []
    keys = []
    seen = set()
    for path in paths:
        try:
            traces, meta = _load_one(path)
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
            continue
        if meta.get("cmd") != cmd:
            continue
        pk = str(meta.get("pk_fp16", path.stem))
        if pk in seen:
            continue
        means.append(traces.mean(axis=0))
        vars_.append(traces.var(axis=0, ddof=1))
        counts.append(traces.shape[0])
        keys.append(pk)
        seen.add(pk)
    if len(means) < 2:
        raise ValueError(f"need >=2 {cmd} captures, got {len(means)}")
    t = min(x.shape[0] for x in means)
    return (
        np.stack([x[:t] for x in means]),
        np.stack([x[:t] for x in vars_]),
        np.asarray(counts, dtype=np.float64),
        keys,
    )


def snr_envelope(means: np.ndarray, vars_: np.ndarray, counts: np.ndarray) -> np.ndarray:
    between = means.var(axis=0, ddof=1)
    noise = np.mean(vars_ / counts[:, None], axis=0)
    return between / np.maximum(noise, 1e-12)


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.astype(np.float64)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(x.astype(np.float64), kernel, mode="same")


def top_windows(score: np.ndarray, window: int, k: int) -> list[tuple[int, int, float]]:
    if window >= score.size:
        return [(0, score.size, float(score.sum()))]
    rolling = np.convolve(score, np.ones(window), mode="valid")
    order = np.argsort(-rolling, kind="stable")
    out: list[tuple[int, int, float]] = []
    for start in order:
        end = int(start) + window
        if any(not (end <= a or start >= b) for a, b, _ in out):
            continue
        out.append((int(start), end, float(rolling[int(start)])))
        if len(out) >= k:
            break
    return out


def template_matches(template_source: np.ndarray, target: np.ndarray, window: int, k: int) -> list[tuple[int, int, float]]:
    source_windows = top_windows(template_source, window, 1)
    lo, hi, _ = source_windows[0]
    tmpl = template_source[lo:hi].copy()
    tmpl = (tmpl - tmpl.mean()) / max(tmpl.std(), 1e-12)
    scores = []
    for start in range(0, target.size - window + 1):
        seg = target[start:start + window]
        seg = (seg - seg.mean()) / max(seg.std(), 1e-12)
        scores.append(float(np.mean(tmpl * seg)))
    scores_arr = np.asarray(scores)
    order = np.argsort(-scores_arr, kind="stable")
    out: list[tuple[int, int, float]] = []
    for start in order:
        end = int(start) + window
        if any(not (end <= a or start >= b) for a, b, _ in out):
            continue
        out.append((int(start), end, float(scores_arr[int(start)])))
        if len(out) >= k:
            break
    return out


def main() -> int:
    args = parse_args()
    m_t, v_t, n_t, keys_t = load_key_traces(args.t_inputs, "T")
    m_z, v_z, n_z, keys_z = load_key_traces(args.z_inputs, "Z")
    m_v, v_v, n_v, keys_v = load_key_traces(args.v_inputs, "V")

    snr_t = smooth(snr_envelope(m_t, v_t, n_t), args.smooth)
    snr_z = smooth(snr_envelope(m_z, v_z, n_z), args.smooth)
    snr_v = smooth(snr_envelope(m_v, v_v, n_v), args.smooth)

    z_top = top_windows(snr_z, args.window, args.top_k)
    v_to_z = template_matches(snr_v, snr_z, args.window, args.top_k)
    t_to_z = template_matches(snr_t, snr_z, args.window, args.top_k)

    lines = [
        "S2 Z window scan",
        f"T keys={len(keys_t)}, Z keys={len(keys_z)}, V keys={len(keys_v)}",
        f"smooth={args.smooth}, window={args.window}",
        "",
        "Top Z SNR windows:",
        *[f"  {lo}:{hi} score={score:.6g}" for lo, hi, score in z_top],
        "",
        "V-template matches in Z SNR:",
        *[f"  {lo}:{hi} corr={score:.4f}" for lo, hi, score in v_to_z],
        "",
        "T-template matches in Z SNR:",
        *[f"  {lo}:{hi} corr={score:.4f}" for lo, hi, score in t_to_z],
    ]
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt = args.out_prefix.with_suffix(".txt")
    txt.write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    for a, label, snr, color in [
        (ax[0], "T", snr_t, "C0"),
        (ax[1], "V", snr_v, "C1"),
        (ax[2], "Z", snr_z, "C2"),
    ]:
        a.plot(snr, lw=0.5, color=color)
        a.set_ylabel(f"{label} SNR")
    for lo, hi, _ in z_top[:5]:
        ax[2].axvspan(lo, hi, color="C3", alpha=0.12)
    ax[2].set_xlabel("sample")
    fig.tight_layout()
    png = args.out_prefix.with_suffix(".png")
    fig.savefig(png, dpi=120)

    print("[SUMMARY]")
    for line in lines:
        print(line)
    print(f"[OK] wrote {txt}")
    print(f"[OK] wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
