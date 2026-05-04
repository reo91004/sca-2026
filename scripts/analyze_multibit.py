#!/usr/bin/env python3
"""[smaug3/5] Multi-α (multi-bit) chosen-CT board-binary classifier + sparse_recover.

Paper §5/§6 의 자연스러운 확장: smaug3/5 의 indcpa_dec 가 SMAUG-T smaug1 과
*다른 round 정의* 또는 internal arithmetic 으로 인해 oracle pair 만으로는
(b_pos, b_neg) ∈ {(0,0)} 에서 sk=-1 vs sk=0 분리 안 됨.

해결: support oracle α (host predict (1, 0, 1) 패턴) 추가 → 3-tuple
(b_pos, b_neg, b_supp) 가 ternary class 를 unique 하게 분리할 *수학적 가능성*:
    sk=-1 → host predict (0, 1, 1)
    sk=0  → host predict (0, 0, 0)
    sk=+1 → host predict (1, 0, 1)

Board 의 실제 응답이 host predict 와 일치하면 100% recovery (no power trace).
부분 mismatch 면 *empirical mapping* (cross-tab) 사용.

용법:
    # 1. capture (3-α): scripts/run_attack.py --level smaug3 -n 128 \\
    #                       --alphas 132 --out traces/attack_smaug3_seed3.npz
    # 2. analyze:
    scripts/analyze_multibit.py traces/attack_smaug3_seed3.npz

산출: per-component bit/sign/sparse acc, full sk acc.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.smaug import params as _params  # noqa: E402
from host.smaug.codec import unpack_sx  # noqa: E402
from host.smaug.sk_partition import predict_mu_prime_bit  # noqa: E402
from host.analysis import sparse_recover  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--out-prefix", type=str, default="results/multibit")
    p.add_argument("--mapping", choices=("host", "empirical", "auto"),
                   default="auto",
                   help="(b1,...,bn) → posterior 매핑. host = spec predict_mu_prime_bit, "
                        "empirical = cross-tab from data, auto = host 우선 + (0,..,0) 만 empirical fallback.")
    return p.parse_args()


def host_posterior_table(p, alphas: list[int]) -> dict[tuple, np.ndarray]:
    """spec host predict 로 (b_α1, ..., b_αn) → P(sk_i ∈ {-1,0,+1}).

    deterministic mapping: 각 sk class 에 정확한 (b1,..,bn) 한 개. 일치하지 않는
    tuple 은 'unknown' 처리 (uniform prior).
    """
    table: dict[tuple, list[int]] = {}
    for s in (-1, 0, 1):
        bits = tuple(predict_mu_prime_bit(p, a, s) for a in alphas)
        table.setdefault(bits, []).append(s)

    out: dict[tuple, np.ndarray] = {}
    # 모든 가능한 2^n tuple 에 대해 posterior
    n = len(alphas)
    for tval in range(1 << n):
        key = tuple(((tval >> i) & 1) for i in range(n))
        ss = table.get(key, [-1, 0, 1])  # unknown → uniform over ternary
        post = np.zeros(3)
        for s in ss:
            post[s + 1] = 1.0 / len(ss)
        out[key] = post
    return out


def empirical_posterior_table(joint_per_pos: np.ndarray, sk_gt: np.ndarray
                              ) -> dict[tuple, np.ndarray]:
    """주어진 sk ground truth 로 cross-tab 위 empirical posterior. sanity check 용."""
    n_alphas = joint_per_pos.shape[1]
    table_count: dict[tuple, np.ndarray] = {}
    for i in range(joint_per_pos.shape[0]):
        key = tuple(int(x) for x in joint_per_pos[i])
        if key not in table_count:
            table_count[key] = np.zeros(3)
        table_count[key][sk_gt[i] + 1] += 1
    out = {}
    for k, v in table_count.items():
        out[k] = v / v.sum()
    return out


def analyze_one(npz_path: Path, mapping: str = "auto") -> dict:
    d = np.load(npz_path, allow_pickle=True)
    mus = d["mu_prime"]; meta = d["meta"].item()
    p = _params.get(meta["level"])
    sk = unpack_sx(bytes.fromhex(meta["sk_pke_bytes_hex"])).reshape(
        p.module_rank, p.n).astype(np.int64)
    label_comp = np.asarray(meta["label_component"])
    label_alpha = np.asarray(meta["label_alpha"])
    n_per = int(meta["n_per_ct"])

    # 사용된 모든 unique α (정렬)
    alphas = sorted(set(int(x) for x in label_alpha.tolist()))
    n_alphas = len(alphas)
    n_designs_per_comp = n_alphas

    # bit per trace
    N = mus.shape[0]
    bit_per = np.zeros((N, 256), dtype=np.int8)
    for i in range(N):
        for k in range(256):
            bit_per[i, k] = (int(mus[i, k // 8]) >> (k % 8)) & 1

    host_table = host_posterior_table(p, alphas)
    print(f"  level={meta['level']} module_rank={p.module_rank} hs={p.hs} "
          f"alphas={alphas} (n_designs/comp={n_designs_per_comp})")
    print(f"  host ideal mapping: ", end="")
    for s in (-1, 0, 1):
        bits = tuple(predict_mu_prime_bit(p, a, s) for a in alphas)
        print(f"sk={s:+d}→{bits}  ", end="")
    print()

    out: dict = {
        "path": str(npz_path),
        "level": meta["level"],
        "module_rank": p.module_rank,
        "hs": p.hs,
        "alphas": alphas,
        "sk_hash_short": meta["sk_pke_bytes_hex"][:16],
        "sk_hw": [int(np.count_nonzero(sk[i])) for i in range(p.module_rank)],
        "n_per_ct": n_per,
        "n_total_trace": N,
        "method": f"multibit-{mapping}",
    }

    full_correct = 0
    for c in range(p.module_rank):
        # 각 α 별로 첫 trace (deterministic) 의 bit_per — joint per position
        joint = np.zeros((256, n_alphas), dtype=np.int8)
        for ai, a in enumerate(alphas):
            mask = (label_comp == c) & (label_alpha == a)
            joint[:, ai] = bit_per[mask][0]

        s_gt = sk[c].astype(np.int8)
        emp_table = empirical_posterior_table(joint, s_gt)

        # mapping 에 따라 posterior
        posterior = np.zeros((256, 3))
        for i in range(256):
            key = tuple(int(x) for x in joint[i])
            if mapping == "host":
                posterior[i] = host_table[key]
            elif mapping == "empirical":
                posterior[i] = emp_table.get(key, np.array([1/3, 1/3, 1/3]))
            else:  # auto: host 우선, sk=0 매핑 (0,..,0) 만 empirical 로 보강
                base = host_table[key].copy()
                # sum>0 인 unique mapping 이면 그대로
                if base.sum() > 0 and (base > 0).sum() == 1:
                    posterior[i] = base
                else:
                    # ambiguous (e.g., (0,0,..,0) → uniform prior). empirical 로 대체.
                    posterior[i] = emp_table.get(key, np.array([1/3, 1/3, 1/3]))

        sp = sparse_recover.greedy_sparse_recover(posterior, hs=p.hs)
        m = sparse_recover.accuracy(sp, s_gt)
        n_correct = int((sp == s_gt).sum())
        full_correct += n_correct
        out[f"s{c}"] = {
            "bit_acc": m["bit_accuracy"],
            "support_acc": m["support_accuracy"],
            "sign_acc": m["sign_accuracy"],
            "sparse_pred_hw": int(np.count_nonzero(sp)),
        }
        # cross-tab summary for diagnostic
        ct_counts = {}
        for i in range(256):
            key = tuple(int(x) for x in joint[i])
            sval = int(s_gt[i])
            ct_counts.setdefault(key, [0, 0, 0])[sval + 1] += 1
        out[f"s{c}_crosstab"] = ct_counts
    out["full_sk_acc"] = full_correct / float(p.module_rank * p.n)
    return out


def main() -> int:
    args = parse_args()
    paths = sorted(args.paths)
    print(f"[INFO] {len(paths)} seeds, mapping={args.mapping}")
    results = []
    for path in paths:
        print(f"\n--- {path.name} ---")
        r = analyze_one(path, mapping=args.mapping)
        results.append(r)
        for c in range(int(r["module_rank"])):
            sd = r[f"s{c}"]
            print(f"  s[{c}] bit={sd['bit_acc']:.3f} sign={sd['sign_acc']:.3f} "
                  f"pred_hw={sd['sparse_pred_hw']}")
        print(f"  full sk: {r['full_sk_acc']:.4f}")
        # 간이 cross-tab (component 0)
        ct = r.get("s0_crosstab", {})
        print(f"  cross-tab s[0] (joint -> [sk=-1, 0, +1]):")
        for k in sorted(ct.keys()):
            v = ct[k]
            print(f"    {k}: [{v[0]:3d}, {v[1]:3d}, {v[2]:3d}] (n={sum(v)})")
    if results:
        accs = [r["full_sk_acc"] for r in results]
        print(f"\n[SUMMARY] full_sk_acc: mean={np.mean(accs):.4f} "
              f"std={np.std(accs):.4f} min={min(accs):.4f} max={max(accs):.4f}")

    # save
    out_txt = Path(args.out_prefix + "_summary.txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w") as fh:
        fh.write(f"# Multi-bit (multi-α) board-binary analysis\n")
        fh.write(f"# Mapping: {args.mapping}\n\n")
        for r in results:
            fh.write(f"## {r['path']} [{r['level']} k={r['module_rank']} hs={r['hs']}]\n")
            fh.write(f"  alphas: {r['alphas']}\n  sk HW: {r['sk_hw']}\n")
            for c in range(int(r["module_rank"])):
                sd = r[f"s{c}"]
                fh.write(f"  s[{c}]: bit={sd['bit_acc']:.4f} "
                         f"sign={sd['sign_acc']:.4f} pred_hw={sd['sparse_pred_hw']}\n")
            fh.write(f"  full sk: {r['full_sk_acc']:.4f}\n\n")
    print(f"\n[OK] {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
