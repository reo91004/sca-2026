#!/usr/bin/env python3
"""E3 — DTW elastic alignment for cross-session schedule transfer.

가설 (E1 결과 후)
----------------
- 같은 sk + 다른 capture session 사이는 alignment 보존 (E1 100%).
- 다른 sk 사이는 cross-corr 0.02-0.1 → linear shift 로 align 안 됨.
- 단 같은 firmware/code path 면 *operation 순서* 는 같고 *시간축에서의
  warping* 만 다를 가능성. 이 경우 DTW (Dynamic Time Warping) 로 elastic
  alignment 가능.

방법
----
1. 각 session 의 *mean trace* 를 high-SNR feature 로 사용
   (oracle-pair |t| 보다 훨씬 강한 signal).
2. fastdtw(mean_ref, mean_target) → warp path = list of (i, j).
3. warp path 로 target session 의 모든 trace 를 ref session 의 sample axis 로
   resample. → target 의 schedule 이 ref 의 schedule 과 호환됨.
4. resampled target 위에서 ref 의 µ′-profile schedule 적용 → cross-key
   schedule transfer.

산출
----
- traces/aligned_to_<ref>/<seed>.npz: warp 적용된 traces (meta 그대로 + warp path 추가)
- results/dtw_align_summary.txt: 각 (ref, target) 쌍의 cross-key full-sk
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from fastdtw import fastdtw  # noqa: E402

from scripts.analyze_multi_seed import analyze_one, learn_schedule_from_npz  # noqa: E402


def downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """Trace 를 boxcar 로 downsample. fastdtw 비용 감소."""
    n = len(x)
    n2 = (n // factor) * factor
    return x[:n2].reshape(-1, factor).mean(axis=1)


def warp_path_full(ref: np.ndarray, target: np.ndarray, *, radius: int = 100,
                   ds_factor: int = 8) -> list[tuple[int, int]]:
    """fastdtw with downsample. path 는 *full-resolution* sample idx 로 복원."""
    r_ds = downsample(ref, ds_factor)
    t_ds = downsample(target, ds_factor)
    _, path_ds = fastdtw(r_ds, t_ds, radius=radius)
    # upscale: each (i_ds, j_ds) → (i_ds*ds, j_ds*ds), interpolate intermediate
    path: list[tuple[int, int]] = []
    for i_ds, j_ds in path_ds:
        path.append((i_ds * ds_factor, j_ds * ds_factor))
    # ensure monotone ending
    if path[-1][0] < ref.size - 1 or path[-1][1] < target.size - 1:
        path.append((ref.size - 1, target.size - 1))
    return path


def apply_warp_to_traces(traces: np.ndarray, path: list[tuple[int, int]],
                         ref_n: int) -> np.ndarray:
    """Target traces (N, n_target) → (N, ref_n).

    path: list of (i_ref, j_target). For each i_ref, average target[:, j] over
    all j with (i_ref, j) ∈ path. samples not covered → linear interp from
    nearest covered samples (rare for monotone path).
    """
    n_traces, n_target = traces.shape
    out = np.zeros((n_traces, ref_n), dtype=traces.dtype)
    counts = np.zeros(ref_n, dtype=np.int64)
    for i_ref, j_target in path:
        if 0 <= i_ref < ref_n and 0 <= j_target < n_target:
            out[:, i_ref] += traces[:, j_target]
            counts[i_ref] += 1
    # fill empty sample columns by carrying nearest covered (linear interpolate index)
    if (counts == 0).any():
        good_idx = np.flatnonzero(counts > 0)
        all_idx = np.arange(ref_n)
        out_safe = out.copy()
        norm = counts.clip(min=1)
        out_safe = out_safe / norm[None, :]
        # linear interp the missing indices in the time axis using neighbors
        if good_idx.size > 0:
            for k in range(n_traces):
                out[k] = np.interp(all_idx, good_idx, out_safe[k, good_idx])
        return out
    out /= counts[None, :]
    return out


def align_npz_to_ref(target_path: Path, ref_path: Path, out_path: Path,
                     *, radius: int = 100, ds_factor: int = 8) -> dict:
    dr = np.load(ref_path, allow_pickle=True)
    dt = np.load(target_path, allow_pickle=True)
    Tr = dr["traces"]; Tt = dt["traces"]
    ref_mean = Tr.mean(axis=0).astype(np.float64)
    tgt_mean = Tt.mean(axis=0).astype(np.float64)
    # normalize for DTW (zero-mean unit-var)
    rn = (ref_mean - ref_mean.mean()) / max(ref_mean.std(), 1e-9)
    tn = (tgt_mean - tgt_mean.mean()) / max(tgt_mean.std(), 1e-9)

    started = time.time()
    path = warp_path_full(rn, tn, radius=radius, ds_factor=ds_factor)
    dtw_time = time.time() - started

    # apply warp to all target traces
    aligned_T = apply_warp_to_traces(Tt, path, ref_n=Tr.shape[1])
    aligned_T = aligned_T.astype(np.float32)

    # save with original meta + alignment info
    meta_t = dict(dt["meta"].item())
    meta_t["aligned_to"] = ref_path.name
    meta_t["aligned_via"] = "fastdtw"
    meta_t["dtw_radius"] = radius
    meta_t["dtw_ds_factor"] = ds_factor
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        traces=aligned_T,
        mu_prime=dt["mu_prime"],  # mu_prime unchanged
        meta=np.array(meta_t, dtype=object),
    )

    return {
        "target": target_path.name,
        "ref": ref_path.name,
        "out": str(out_path),
        "dtw_time_s": dtw_time,
        "ref_mean_max": float(np.abs(ref_mean).max()),
        "tgt_mean_max": float(np.abs(tgt_mean).max()),
        "n_path": len(path),
    }


def main() -> int:
    paths = sorted(Path("traces").glob("attack_seed*.npz"))
    paths = [p for p in paths
             if "smaug3" not in p.name and "smaug5" not in p.name]
    paths += [Path("traces/attack_const_c1_n256.npz")]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("[ERROR] no smaug1 attack seeds")

    ref_path = paths[0]
    Path("traces/aligned").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    print(f"[E3] reference: {ref_path.name}")
    print(f"[E3] aligning {len(paths) - 1} other seeds + reference baseline")
    info_rows = []

    # Reference: just copy (passthrough)
    aligned_ref = Path("traces/aligned") / f"{ref_path.stem}.npz"
    if not aligned_ref.exists():
        d = np.load(ref_path, allow_pickle=True)
        meta_r = dict(d["meta"].item())
        meta_r["aligned_to"] = ref_path.name
        meta_r["aligned_via"] = "passthrough"
        np.savez_compressed(aligned_ref, traces=d["traces"], mu_prime=d["mu_prime"],
                            meta=np.array(meta_r, dtype=object))
    info_rows.append({"target": ref_path.name, "ref": ref_path.name,
                      "out": str(aligned_ref), "dtw_time_s": 0.0,
                      "ref_mean_max": 0.0, "tgt_mean_max": 0.0, "n_path": 0})

    for p in paths[1:]:
        out = Path("traces/aligned") / f"{p.stem}.npz"
        info = align_npz_to_ref(p, ref_path, out)
        print(f"  {p.name:40s} dtw {info['dtw_time_s']:.1f}s, "
              f"ref|mean|max={info['ref_mean_max']:.4f} tgt={info['tgt_mean_max']:.4f}")
        info_rows.append(info)

    # === Schedule transfer evaluation in aligned space ===
    print(f"\n[E3] cross-key schedule transfer in aligned space (ref schedule applied)")
    aligned_paths = [Path("traces/aligned") / f"{p.stem}.npz" for p in paths]
    sched_ref = learn_schedule_from_npz(aligned_paths[0])
    print(f"  schedule from {aligned_paths[0].name}, "
          f"sk={sched_ref['_meta']['sk_hash_short']}")

    rows = []
    for ap in aligned_paths:
        d = np.load(ap, allow_pickle=True)
        sk_short = d["meta"].item()["sk_pke_bytes_hex"][:16]
        if sk_short == sched_ref["_meta"]["sk_hash_short"]:
            tag = "(ref)"
        else:
            tag = "(target)"
        r = analyze_one(ap, poi_source="schedule", schedule=sched_ref)
        rows.append((ap.name, sk_short, r["full_sk_acc"], tag))
        print(f"  {ap.name:50s} {tag:10s} sk={sk_short} full_sk={r['full_sk_acc']:.4f}")

    # also baseline µ'-leak per file (in-session, sanity)
    print(f"\n[E3] in-session µ′-PoI baseline (aligned space)")
    for ap in aligned_paths[:3]:
        r = analyze_one(ap, poi_source="mu-prime")
        print(f"  {ap.name:50s} full_sk={r['full_sk_acc']:.4f}, "
              f"valid={r['n_valid_bits']}, max|t|={r['max_t']:.2f}")

    out_lines = [f"# E3 DTW alignment summary (ref={ref_path.name})\n"]
    out_lines.append("## warp paths\n")
    for r in info_rows:
        out_lines.append(f"  {r['target']:40s} dtw_time={r['dtw_time_s']:.1f}s "
                         f"path_len={r['n_path']}\n")
    out_lines.append("\n## cross-key schedule transfer (ref schedule on aligned target)\n")
    for name, sk, acc, tag in rows:
        out_lines.append(f"  {name:50s} {tag:10s} sk={sk}  full_sk={acc:.4f}\n")
    Path("results/E3_dtw_align.txt").write_text("".join(out_lines))
    print(f"\n[OK] results/E3_dtw_align.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
