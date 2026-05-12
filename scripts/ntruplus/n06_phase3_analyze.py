#!/usr/bin/env python3
"""Phase 3 scout analysis — per-lane chosen-CT leakage on a single key.

For each NTT lane k tested in `traces/ntruplus768/phase3/scout.npz`, we
have G=6 γ values × N=10 traces = 60 traces. The chosen-CT design has
c_ntt[4*lane+slot] = γ, all other coefficients zero, so the on-board
`poly_basemul(&m1, &c, &f)` computes

    m1[4*lane + i] = γ · f_ntt[4*lane + i]   mod q     (i = 0..3)

with all other m1 lanes equal to zero. The dominant register-level
leakage at the basemul iteration of lane k is therefore a function of
γ · f_ntt[4*lane + slot]; we test the byte-HW labels of that value at
each sample t.

Outputs per-lane PoIs and a comparison against the permutation null.
A pass at this stage is a single-key test and not yet attack-valid;
held-out evaluation comes in Phase 3-confirm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "host"))

from ntruplus.codec import center, from_bytes  # noqa: E402
from ntruplus.params import D, POLYBYTES, Q  # noqa: E402


def hw16(values: np.ndarray) -> np.ndarray:
    """Hamming weight of int values, treating each as 16-bit unsigned."""
    out = np.zeros_like(values, dtype=np.int16)
    v = (values.astype(np.int32)) & 0xFFFF
    for s in range(16):
        out += ((v >> s) & 1).astype(np.int16)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=ROOT / "traces/ntruplus768/phase3/scout.npz")
    p.add_argument("--out-prefix", type=Path,
                   default=ROOT / "results/ntruplus/phase3/scout")
    p.add_argument("--label", choices=("modq_hw", "modq_value"), default="modq_hw",
                   help="label = HW(γ·f mod q) (modq_hw) or γ·f mod q centred (modq_value)")
    p.add_argument("--n-shuffles", type=int, default=300)
    p.add_argument("--seed", type=int, default=2026_05_08)
    args = p.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.input, allow_pickle=True)
    traces = z["traces"].astype(np.float32)        # (L, G, N, T)
    sk_blob = bytes(z["sk_blob"])
    lanes = z["lanes"].astype(int)
    gammas = z["gammas"].astype(int)
    meta = z["meta"].item()
    L, G, N, T = traces.shape
    print(f"[INFO] traces (L, G, N, T) = {traces.shape}  decimate={meta['decimate']}")

    # Recover f_ntt from sk[0..POLYBYTES) (12-bit packed, [0, 0xfff)).
    f_unsigned = from_bytes(sk_blob[:POLYBYTES])
    f_centered = center(f_unsigned)                 # int16 in (-q/2, q/2]
    print(f"[INFO] sk parsed: |f_ntt| max = {np.abs(f_centered).max()}")
    slot = meta.get("slot", 0)

    # Build per-trace label tables: shape (L, G, N).
    # For every lane k, all N traces of design (γ_g, k) share the same
    # label, so labels[l, g, :] is constant in n.
    labels_value = np.zeros((L, G, N), dtype=np.int32)
    for li, lane in enumerate(lanes):
        f_lane = int(f_centered[D * lane + slot])
        for gi, g in enumerate(gammas):
            v = (g * f_lane) % Q
            if v > Q // 2:
                v -= Q
            labels_value[li, gi, :] = v
    if args.label == "modq_hw":
        labels = hw16(labels_value % Q).astype(np.float32)
    else:
        labels = labels_value.astype(np.float32)

    print(f"[INFO] label = {args.label}; example lane=0 across γ "
          f"{[(g, int(labels[0, gi, 0])) for gi, g in enumerate(gammas)]}")

    # Per-lane correlation, using flatten over (G, N) → 60 samples.
    corr_per_lane = np.zeros((L, T), dtype=np.float32)
    for li in range(L):
        flat_tr = traces[li].reshape(G * N, T)
        flat_lab = labels[li].reshape(G * N)
        if flat_lab.std() < 1e-6:
            continue
        tr_z = (flat_tr - flat_tr.mean(axis=0)) / (flat_tr.std(axis=0) + 1e-12)
        lab_z = (flat_lab - flat_lab.mean()) / (flat_lab.std() + 1e-12)
        corr_per_lane[li] = (tr_z * lab_z[:, None]).mean(axis=0)

    abs_corr = np.abs(corr_per_lane)
    real_max_per_lane = abs_corr.max(axis=1)
    real_argmax_per_lane = abs_corr.argmax(axis=1)
    print("\n[STAT] per-lane real max |corr|:")
    for li in range(L):
        print(f"  lane idx={lanes[li]:>4}  max|corr|={real_max_per_lane[li]:.4f}  "
              f"@ sample {real_argmax_per_lane[li]:>5}")

    # PUBLIC-DESIGN CONTROL: same correlation but with label = HW(γ) only
    # (public, no f). If real |corr| ≈ this null, the signal is from γ-byte
    # leakage in poly_frombytes(c, ct) and not from γ·f in basemul/downstream.
    g_hw = np.zeros(G, dtype=np.float32)
    for gi, g in enumerate(gammas):
        g_hw[gi] = bin(int(g) % Q).count("1")     # HW of unsigned 12-bit γ
    if g_hw.std() > 1e-6:
        # repeat to match (G, N) → flat
        g_hw_flat = np.repeat(g_hw[:, None], N, axis=1).reshape(G * N)
        gctrl_per_lane = np.zeros((L, T), dtype=np.float32)
        for li in range(L):
            flat_tr = traces[li].reshape(G * N, T)
            tr_z = (flat_tr - flat_tr.mean(axis=0)) / (flat_tr.std(axis=0) + 1e-12)
            lab_z = (g_hw_flat - g_hw_flat.mean()) / (g_hw_flat.std() + 1e-12)
            gctrl_per_lane[li] = (tr_z * lab_z[:, None]).mean(axis=0)
        gctrl_max = np.abs(gctrl_per_lane).max(axis=1)
        print("\n[CTRL ] public-design null: label = HW(γ) only "
              f"(γ HW range {int(g_hw.min())}..{int(g_hw.max())}):")
        for li in range(L):
            print(f"  lane idx={lanes[li]:>4}  HW(γ)-only max|corr|={gctrl_max[li]:.4f}")
    else:
        print("\n[CTRL ] HW(γ) is constant across γ set — no public-design "
              "leakage possible from γ-byte HW. Skipping HW(γ) control.")
        gctrl_max = np.zeros(L, dtype=np.float32)

    # Permutation null: shuffle labels within (G, N) of each lane independently.
    rng = np.random.default_rng(args.seed)
    null_max = np.zeros((args.n_shuffles, L), dtype=np.float32)
    for s in range(args.n_shuffles):
        for li in range(L):
            flat_tr = traces[li].reshape(G * N, T)
            flat_lab = labels[li].reshape(G * N).copy()
            rng.shuffle(flat_lab)
            if flat_lab.std() < 1e-6:
                null_max[s, li] = 0
                continue
            tr_z = (flat_tr - flat_tr.mean(axis=0)) / (flat_tr.std(axis=0) + 1e-12)
            lab_z = (flat_lab - flat_lab.mean()) / (flat_lab.std() + 1e-12)
            corr = (tr_z * lab_z[:, None]).mean(axis=0)
            null_max[s, li] = float(np.abs(corr).max())
    null_p999_per_lane = np.percentile(null_max, 99.9, axis=0)
    null_mean_per_lane = null_max.mean(axis=0)
    null_std_per_lane = null_max.std(axis=0)

    z_per_lane = (real_max_per_lane - null_mean_per_lane) / np.maximum(null_std_per_lane, 1e-6)
    # gate: PASS only if real > permutation null AND real > public-design (HW(γ)) null.
    # second check kills the "γ-byte read in poly_frombytes(c)" confounder.
    print("\n[STAT] per-lane null + z (gate = z>5 AND real > p99.9 AND real > HW(γ)-ctrl):")
    print(f"  {'lane':>5} {'real':>8} {'null μ':>8} {'null σ':>8} {'p99.9':>8} "
          f"{'γ-ctrl':>8} {'z':>7} verdict")
    for li in range(L):
        passes_perm = z_per_lane[li] > 5 and real_max_per_lane[li] > null_p999_per_lane[li]
        passes_pub  = real_max_per_lane[li] > gctrl_max[li] + 0.05
        verdict = ("PASS" if (passes_perm and passes_pub)
                   else "WEAK" if z_per_lane[li] > 3 else "FAIL")
        print(f"  {lanes[li]:>5} {real_max_per_lane[li]:>8.4f} "
              f"{null_mean_per_lane[li]:>8.4f} {null_std_per_lane[li]:>8.4f} "
              f"{null_p999_per_lane[li]:>8.4f} {gctrl_max[li]:>8.4f} "
              f"{z_per_lane[li]:>7.2f} {verdict}")

    # cross-lane consistency: do PoIs cluster (basemul iteration ≈ linear in lane)?
    # If basemul is a flat O(N) loop over 96 zeta-pairs, each pair processing 8 coeffs,
    # then iteration k of pair (k>>1) takes ~30 cycles → linear position in trace.
    print(f"\n[STAT] PoI vs lane index (linear-fit consistency):")
    for li, lane in enumerate(lanes):
        print(f"  lane {lane:>4}  PoI sample = {int(real_argmax_per_lane[li]):>5}")

    np.savez_compressed(
        f"{args.out_prefix}.npz",
        corr_per_lane=corr_per_lane, abs_corr=abs_corr,
        real_max_per_lane=real_max_per_lane,
        real_argmax_per_lane=real_argmax_per_lane,
        null_max=null_max, null_p999_per_lane=null_p999_per_lane,
        null_mean_per_lane=null_mean_per_lane, null_std_per_lane=null_std_per_lane,
        z_per_lane=z_per_lane,
        lanes=lanes, gammas=gammas,
        labels_value=labels_value, label_kind=args.label,
    )

    md = []
    md.append(f"# Phase 3 scout — selected-lane signal on a single sk")
    md.append("")
    md.append(f"- traces (L, G, N, T) = ({L}, {G}, {N}, {T}), decimate={meta['decimate']}")
    md.append(f"- lanes  = {lanes.tolist()}")
    md.append(f"- gammas = {gammas.tolist()}")
    md.append(f"- label  = `{args.label}`")
    md.append("")
    md.append("## per-lane PoI vs null")
    md.append("| lane | max\\|corr\\| | PoI sample | null p99.9 | z | verdict |")
    md.append("|---:|---:|---:|---:|---:|---|")
    for li in range(L):
        verdict = "PASS" if z_per_lane[li] > 5 and real_max_per_lane[li] > null_p999_per_lane[li] \
                  else "WEAK" if z_per_lane[li] > 3 else "FAIL"
        md.append(f"| {lanes[li]} | {real_max_per_lane[li]:.4f} | "
                  f"{int(real_argmax_per_lane[li])} | {null_p999_per_lane[li]:.4f} | "
                  f"{z_per_lane[li]:.2f} | {verdict} |")
    Path(f"{args.out_prefix}.md").write_text("\n".join(md))
    print(f"[OK] saved → {args.out_prefix}.{{npz, md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
