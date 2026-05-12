#!/usr/bin/env python3
"""S2 matrix analysis — multi-design Z profiling with held-out keys."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from host.analysis.sparse_recover import accuracy  # noqa: E402
from host.smaug import codec as _codec  # noqa: E402
from host.smaug.params import SMAUG1  # noqa: E402


@dataclass(frozen=True)
class Config:
    block: int
    n_features: int
    ridge: float

    @property
    def name(self) -> str:
        return f"b{self.block}_f{self.n_features}_r{self.ridge:g}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s2_z_matrix_k*_d*_n*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument("--hs", type=int, default=SMAUG1.hs)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--seed", type=int, default=0x5A17)
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s2_z_matrix",
    )
    return p.parse_args()


def load_matrix(paths: list[Path], component: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    traces = []
    sks = []
    metas = []
    seen = set()
    for path in paths:
        try:
            data = np.load(path, allow_pickle=True)
            meta = data["meta"].item()
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
            continue
        if meta.get("capture_kind") != "matrix" or meta.get("cmd") != "Z":
            continue
        pk = meta.get("pk_fp16")
        if pk in seen:
            continue
        x = np.asarray(data["traces"], dtype=np.float64)
        if x.ndim != 3:
            print(f"[WARN] skip {path.name}: traces shape {x.shape} != (D,N,T)")
            continue
        full = _codec.unpack_sx(bytes.fromhex(meta["sk_pke_hex"])).astype(np.int8)
        lo = component * 256
        hi = lo + 256
        traces.append(x)
        sks.append(full[lo:hi])
        metas.append(meta)
        seen.add(pk)
    if not traces:
        raise ValueError("no matrix captures")
    min_d = min(x.shape[0] for x in traces)
    min_t = min(x.shape[2] for x in traces)
    min_n = min(x.shape[1] for x in traces)
    traces = [x[:min_d, :min_n, :min_t] for x in traces]
    return np.stack(traces), np.stack(sks), metas


def block_stats(x: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x=(S,D,N,T). Return feature means/vars over per-trace block means."""
    s, d, n, t = x.shape
    b = t // block
    xb = x[..., : b * block].reshape(s, d, n, b, block).mean(axis=4)
    means = xb.mean(axis=2).reshape(s, d * b)
    vars_ = xb.var(axis=2, ddof=1).reshape(s, d * b)
    counts = np.full(s, n, dtype=np.float64)
    return means, vars_, counts


def select_features(means: np.ndarray, vars_: np.ndarray, counts: np.ndarray,
                    train_idx: np.ndarray, n_features: int) -> np.ndarray:
    between = means[train_idx].var(axis=0, ddof=1)
    noise = np.mean(vars_[train_idx] / counts[train_idx, None], axis=0)
    score = between / np.maximum(noise, 1e-12)
    return np.sort(np.argsort(-score, kind="stable")[: min(n_features, score.size)])


def sparse_decode(scores: np.ndarray, hs: int) -> np.ndarray:
    out = np.zeros(scores.size, dtype=np.int8)
    support = np.argsort(-np.abs(scores), kind="stable")[:hs]
    out[support] = np.where(scores[support] >= 0, 1, -1).astype(np.int8)
    return out


def fit_predict(means: np.ndarray, sk_labels: np.ndarray, features: np.ndarray,
                train_idx: np.ndarray, hold: int, ridge: float) -> np.ndarray:
    xtr = means[train_idx][:, features]
    xte = means[[hold]][:, features]
    mu = xtr.mean(axis=0, keepdims=True)
    sd = xtr.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd
    y = sk_labels[train_idx].astype(np.float64)
    ymu = y.mean(axis=0, keepdims=True)
    yc = y - ymu
    gram = xtr @ xtr.T
    alpha = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), yc)
    w = xtr.T @ alpha
    return (xte @ w + ymu)[0]


def evaluate(x: np.ndarray, sk_true: np.ndarray, cfg: Config, hs: int,
             perm: np.ndarray | None = None) -> dict[str, np.ndarray]:
    means, vars_, counts = block_stats(x, cfg.block)
    labels = sk_true if perm is None else sk_true[perm]
    preds = []
    for hold in range(sk_true.shape[0]):
        train = np.asarray([i for i in range(sk_true.shape[0]) if i != hold], dtype=np.int64)
        feat = select_features(means, vars_, counts, train, cfg.n_features)
        scores = fit_predict(means, labels, feat, train, hold, cfg.ridge)
        preds.append(sparse_decode(scores, hs))
    pred = np.stack(preds)
    metrics = [accuracy(pred[i], sk_true[i]) for i in range(sk_true.shape[0])]
    return {
        "coord": np.asarray([m["bit_accuracy"] for m in metrics]),
        "support": np.asarray([m["support_accuracy"] for m in metrics]),
        "sign": np.asarray([m["sign_accuracy"] for m in metrics]),
        "pred": pred,
    }


def random_sparse_baseline(sks: np.ndarray, hs: int, rng: np.random.Generator) -> dict[str, float]:
    vals = []
    for _ in range(200):
        pred = np.zeros_like(sks, dtype=np.int8)
        for i in range(sks.shape[0]):
            support = rng.choice(sks.shape[1], size=hs, replace=False)
            pred[i, support] = rng.choice(np.array([-1, 1], dtype=np.int8), size=hs)
        metrics = [accuracy(pred[i], sks[i]) for i in range(sks.shape[0])]
        vals.append((
            np.mean([m["bit_accuracy"] for m in metrics]),
            np.mean([m["support_accuracy"] for m in metrics]),
            np.nanmean([m["sign_accuracy"] for m in metrics]),
        ))
    arr = np.asarray(vals)
    return {"coord": float(arr[:, 0].mean()), "support": float(arr[:, 1].mean()), "sign": float(arr[:, 2].mean())}


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    x, sks, metas = load_matrix(args.inputs, args.component)
    cfg = Config(args.block, args.n_features, args.ridge)
    print(f"[INFO] S={x.shape[0]} D={x.shape[1]} N={x.shape[2]} T={x.shape[3]} cfg={cfg.name}")
    real = evaluate(x, sks, cfg, args.hs)
    coord = float(real["coord"].mean())
    support = float(real["support"].mean())
    sign = float(np.nanmean(real["sign"]))
    zero = float(np.mean(sks == 0))
    rand = random_sparse_baseline(sks, args.hs, rng)
    null_coord = []
    null_support = []
    null_sign = []
    for i in range(args.n_perm):
        null = evaluate(x, sks, cfg, args.hs, perm=rng.permutation(x.shape[0]))
        null_coord.append(float(null["coord"].mean()))
        null_support.append(float(null["support"].mean()))
        null_sign.append(float(np.nanmean(null["sign"])))
        if (i + 1) % max(1, args.n_perm // 10) == 0:
            print(f"[NULL] {i + 1}/{args.n_perm}")
    nc = np.asarray(null_coord)
    ns = np.asarray(null_support)
    ng = np.asarray(null_sign)
    zc = (coord - nc.mean()) / max(nc.std(ddof=1), 1e-12)
    zs = (support - ns.mean()) / max(ns.std(ddof=1), 1e-12)
    lines = [
        "S2 Z matrix multi-design profiling",
        f"S={x.shape[0]}, D={x.shape[1]}, N={x.shape[2]}, T={x.shape[3]}, cfg={cfg.name}",
        f"zero baseline coord={zero:.4f}",
        f"random sparse baseline coord={rand['coord']:.4f}, support={rand['support']:.4f}, sign={rand['sign']:.4f}",
        f"real coord={coord:.4f}, support={support:.4f}, sign={sign:.4f}",
        f"perm coord={nc.mean():.4f}±{nc.std(ddof=1):.4f}, z={zc:+.2f}",
        f"perm support={ns.mean():.4f}±{ns.std(ddof=1):.4f}, z={zs:+.2f}",
        f"perm sign={np.nanmean(ng):.4f}±{np.nanstd(ng, ddof=1):.4f}",
    ]
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(nc, bins=20, color="0.75", edgecolor="0.35")
    axes[0].axvline(coord, color="C3", lw=2)
    axes[0].axvline(zero, color="C0", lw=1, ls="--")
    axes[0].set_title("coord")
    axes[1].hist(ns, bins=20, color="0.75", edgecolor="0.35")
    axes[1].axvline(support, color="C3", lw=2)
    axes[1].axvline(rand["support"], color="C0", lw=1, ls="--")
    axes[1].set_title("support")
    axes[2].hist(ng[np.isfinite(ng)], bins=20, color="0.75", edgecolor="0.35")
    axes[2].axvline(sign, color="C3", lw=2)
    axes[2].axvline(rand["sign"], color="C0", lw=1, ls="--")
    axes[2].set_title("sign")
    fig.tight_layout()
    args.out_prefix.with_suffix(".png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=120)
    print("[SUMMARY]")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
