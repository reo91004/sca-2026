#!/usr/bin/env python3
"""Score public c1 designs for the S2 Toom/Karatsuba label branch.

The scorer uses synthetic SMAUG1-sparse secrets only. It does not inspect target
keys. A good public design should make the chosen implementation-state label
vary strongly across plausible sparse secrets and avoid dimensions that are
constant for most candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from host.smaug.params import SMAUG1  # noqa: E402
from scripts.smaug.s2_z_lowdim_analyze import label_from_toom, load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-kind", default="toom2_conv16_hw")
    p.add_argument("--num-secrets", type=int, default=1024)
    p.add_argument("--num-candidates", type=int, default=512)
    p.add_argument("--terms", type=int, default=4)
    p.add_argument("--alpha-choices", default="32,64,96,128,160,192,224")
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--select-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260505)
    p.add_argument("--existing-input", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=_REPO / "results" / "s2_z_scored_designs.json")
    p.add_argument("--out-txt", type=Path, default=_REPO / "results" / "s2_z_design_score.txt")
    return p.parse_args()


def parse_alpha_choices(text: str) -> list[int]:
    return [int(x.strip(), 0) for x in text.split(",") if x.strip()]


def random_sparse_secrets(rng: np.random.Generator, count: int) -> np.ndarray:
    out = np.zeros((count, SMAUG1.n), dtype=np.int8)
    signs = np.array([-1, 1], dtype=np.int8)
    for i in range(count):
        idx = rng.choice(SMAUG1.n, size=SMAUG1.hs, replace=False)
        out[i, idx] = rng.choice(signs, size=SMAUG1.hs)
    return out


def random_design(rng: np.random.Generator, terms: int, alpha_choices: list[int]) -> list[tuple[int, int]]:
    coefs = rng.choice(SMAUG1.n, size=terms, replace=False)
    alphas = rng.choice(np.asarray(alpha_choices, dtype=np.int64), size=terms, replace=True)
    return [(int(c), int(a)) for c, a in zip(coefs, alphas)]


def design_labels(secrets: np.ndarray, design: list[tuple[int, int]], kind: str) -> np.ndarray:
    return np.stack([label_from_toom(sk, design, kind) for sk in secrets]).astype(np.float64)


def score_labels(y: np.ndarray) -> dict[str, float]:
    std = y.std(axis=0, ddof=1)
    active = std > 1e-9
    if not np.any(active):
        return {"score": 0.0, "mean_std": 0.0, "min_std": 0.0, "active": 0.0, "eff_rank": 0.0}
    yc = y[:, active] - y[:, active].mean(axis=0, keepdims=True)
    cov = (yc.T @ yc) / max(1, y.shape[0] - 1)
    vals = np.linalg.eigvalsh(cov)
    vals = np.maximum(vals, 0.0)
    eff_rank = float((vals.sum() ** 2) / max(float(np.sum(vals * vals)), 1e-12))
    mean_std = float(std[active].mean())
    min_std = float(std[active].min())
    active_frac = float(active.mean())
    score = mean_std * (0.5 + 0.5 * active_frac) * np.log1p(eff_rank)
    return {
        "score": float(score),
        "mean_std": mean_std,
        "min_std": min_std,
        "active": active_frac,
        "eff_rank": eff_rank,
    }


def design_key(design: list[tuple[int, int]]) -> str:
    return ",".join(f"{c}:{a}" for c, a in design)


def greedy_select(
    candidates: list[dict],
    label_cache: dict[str, np.ndarray],
    select_k: int,
) -> list[dict]:
    selected: list[dict] = []
    selected_y: list[np.ndarray] = []
    remaining = candidates[:]
    for _ in range(min(select_k, len(remaining))):
        best_i = 0
        best_score = -np.inf
        for i, cand in enumerate(remaining):
            ys = selected_y + [label_cache[design_key(cand["terms"])]]
            y = np.concatenate(ys, axis=1)
            s = score_labels(y)["score"]
            if s > best_score:
                best_score = s
                best_i = i
        chosen = remaining.pop(best_i)
        chosen = dict(chosen)
        chosen["greedy_score"] = float(best_score)
        selected.append(chosen)
        selected_y.append(label_cache[design_key(chosen["terms"])])
    return selected


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    alpha_choices = parse_alpha_choices(args.alpha_choices)
    secrets = random_sparse_secrets(rng, args.num_secrets)

    raw_designs: list[list[tuple[int, int]]] = []
    if args.existing_input is not None:
        ds = load_dataset([args.existing_input], 0)
        raw_designs.extend(ds.design_terms)
    seen = {design_key(d) for d in raw_designs}
    while len(raw_designs) < args.num_candidates + (0 if args.existing_input is None else len(seen)):
        d = random_design(rng, args.terms, alpha_choices)
        k = design_key(d)
        if k in seen:
            continue
        seen.add(k)
        raw_designs.append(d)

    rows = []
    label_cache: dict[str, np.ndarray] = {}
    for design_i, design in enumerate(raw_designs):
        y = design_labels(secrets, design, args.label_kind)
        label_cache[design_key(design)] = y
        metrics = score_labels(y)
        rows.append({"index": design_i, "terms": design, **metrics})

    rows.sort(key=lambda r: r["score"], reverse=True)
    selected = greedy_select(rows, label_cache, args.select_k)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "label_kind": args.label_kind,
                "num_secrets": args.num_secrets,
                "terms": args.terms,
                "alpha_choices": alpha_choices,
                "top": rows[: args.top_k],
                "selected": selected,
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        "S2 public design score",
        f"label={args.label_kind} synthetic_secrets={args.num_secrets}",
        "",
        "Top individual designs:",
    ]
    for r in rows[: args.top_k]:
        lines.append(
            f"score={r['score']:.4f} mean_std={r['mean_std']:.4f} "
            f"eff_rank={r['eff_rank']:.2f} terms={r['terms']}"
        )
    lines.append("")
    lines.append("Greedy selected design set:")
    for r in selected:
        lines.append(
            f"greedy={r['greedy_score']:.4f} indiv={r['score']:.4f} terms={r['terms']}"
        )
    args.out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[OK] wrote {args.out_json}")
    print(f"[OK] wrote {args.out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
