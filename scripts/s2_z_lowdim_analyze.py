#!/usr/bin/env python3
"""S2 matrix low-dimensional profiling.

Full 256-coefficient recovery failed on the small matrix pilot. This script
tests an earlier checkpoint: can Z traces predict lower-dimensional
design-conditioned secret states above permutation null?

Each (key, public design) pair is one observation. Older monomial labels use
the shifted/signed secret vector for ``c1 = alpha * X^j``:

    shifted[i] = sk[(i - j) mod n] * (+1 if i >= j else -1)

Random multi-term labels model firmware-consistent ``c1 << 8`` products and
disassembly-derived Toom/Karatsuba evaluation states. The target key is always
held out as a whole; all design observations for that key are test
observations. Profiling labels are allowed only for training keys.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from host.smaug import codec as _codec  # noqa: E402
from host.smaug.params import SMAUG1  # noqa: E402
from host.smaug.poly_mul import monomial_predict  # noqa: E402


TOOM_MODES = (
    "mul16",
    "mul4",
    "conv16",
    "sec8",
    "kara0_conv8",
    "kara1_conv8",
    "kara2_conv8",
    "kara0_mul4",
    "kara1_mul4",
    "kara2_mul4",
    "kara0_op8",
    "kara1_op8",
    "kara2_op8",
)

TOOM_POINT_LABEL_KINDS = tuple(
    f"{prefix}{i}_{mode}_hw"
    for prefix in ("toom", "vtoom")
    for i in range(7)
    for mode in TOOM_MODES
)

VECADD_LABEL_KINDS = (
    "vtmp16_hw",
    "vtmp64_hw",
    "vtmp16_lohw",
    "vtmp16_hihw",
    "vshift16_hw",
    "vshift64_hw",
    "vshift16_hihw",
    "vadd16_hw",
    "vadd64_hw",
    "vdelta16_hw",
    "vdelta64_hw",
)

MU_LABEL_KINDS = (
    "mu_total_hw",
    "mu_byte_hw",
    "mu_block16_hw",
    "mu_block32_hw",
    "mu_bit",
)

LABEL_KINDS = (
    "support4",
    "sum4",
    "byte_hw",
    "eval64_support",
    "eval64_sum",
    "support16",
    "sum16",
    "prod16_sum",
    "prod16_abs",
    "prod16_hw",
    "prod64_sum",
    "prod64_hw",
    "qprod16_hw",
    "qprod16_hibyte",
    "qprod64_hw",
    "qprod64_hibyte",
    "toom7_mul16_hw",
    "toom7_mul4_hw",
    "toom7_conv16_hw",
) + TOOM_POINT_LABEL_KINDS + VECADD_LABEL_KINDS + MU_LABEL_KINDS

_HW16 = np.fromiter((i.bit_count() for i in range(1 << 16)), dtype=np.uint8, count=1 << 16)
_LABEL_CACHE: dict[tuple[int, str], np.ndarray] = {}


@dataclass(frozen=True)
class Config:
    block: int
    n_features: int
    ridge: float
    feature_mode: str

    @property
    def name(self) -> str:
        return f"b{self.block}_f{self.n_features}_r{self.ridge:g}_{self.feature_mode}"


@dataclass(frozen=True)
class Dataset:
    traces: np.ndarray       # (S,D,N,T)
    sks: np.ndarray          # (S,256)
    design_terms: list[list[tuple[int, int]]]
    design_c2: list[int]
    pkfps: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=sorted((_REPO / "traces").glob("s2_z_matrix_pilot_d4n20_k*.npz")),
    )
    p.add_argument("--component", type=int, default=0)
    p.add_argument(
        "--label-kinds",
        nargs="+",
        choices=LABEL_KINDS,
        default=list(LABEL_KINDS),
    )
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--ridge", type=float, default=10.0)
    p.add_argument("--feature-mode", choices=("snr", "corr"), default="corr")
    p.add_argument(
        "--diff-base",
        type=int,
        default=None,
        help="Use design-difference observations X[d]-X[base] and labels Y[d]-Y[base].",
    )
    p.add_argument(
        "--residualize",
        choices=("none", "two-way"),
        default="none",
        help=(
            "Apply fold-local residualization before fitting. 'two-way' removes "
            "per-key common mode and train-key public-design effect."
        ),
    )
    p.add_argument(
        "--sample-range",
        default=None,
        help="Restrict traces to sample range lo:hi before block features.",
    )
    p.add_argument(
        "--max-designs",
        type=int,
        default=None,
        help="Use only the first K public designs after loading.",
    )
    p.add_argument(
        "--design-offset",
        type=int,
        default=0,
        help="Start design selection at this design index.",
    )
    p.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Use only the first M traces per design after loading.",
    )
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--seed", type=int, default=0xB10C)
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=_REPO / "results" / "s2_z_lowdim",
    )
    return p.parse_args()


def parse_sample_range(text: str | None, t: int) -> tuple[int, int]:
    if text is None:
        return 0, t
    lo_s, hi_s = text.split(":")
    lo = int(lo_s) if lo_s else 0
    hi = int(hi_s) if hi_s else t
    if not (0 <= lo < hi <= t):
        raise ValueError(f"invalid sample range {text!r} for T={t}")
    return lo, hi


def load_dataset(paths: list[Path], component: int) -> Dataset:
    traces = []
    sks = []
    pkfps = []
    terms_ref: list[list[tuple[int, int]]] | None = None
    c2_ref: list[int] | None = None
    seen = set()
    for path in paths:
        try:
            data = np.load(path, allow_pickle=True)
            meta = data["meta"].item()
        except Exception as exc:
            print(f"[WARN] skip {path.name}: {exc}")
            continue
        capture_kind = meta.get("capture_kind")
        cmd = meta.get("cmd")
        if not (
            (capture_kind == "matrix" and cmd == "Z")
            or (capture_kind == "matrix" and cmd == "V")
            or (capture_kind == "matrix" and cmd == "W")
            or (capture_kind == "matrix" and cmd == "R")
            or (capture_kind == "u_matrix" and cmd == "U")
        ):
            continue
        pk = str(meta.get("pk_fp16", ""))
        if not pk or pk in seen:
            continue
        x = np.asarray(data["traces"], dtype=np.float64)
        if x.ndim != 3:
            print(f"[WARN] skip {path.name}: traces shape {x.shape} != (D,N,T)")
            continue
        designs = [dict(d) for d in data["designs"]]
        terms: list[list[tuple[int, int]]] = []
        c2_consts: list[int] = []
        for d in designs:
            raw_terms = d.get("terms")
            if raw_terms is None:
                raw_terms = [(d["coef_idx"], d["alpha"])]
            terms.append([(int(coef), int(alpha)) for coef, alpha in raw_terms])
            c2_consts.append(int(d.get("c2_alpha", meta.get("c2_alpha", 0)) or 0))
        if terms_ref is None:
            terms_ref = terms
            c2_ref = c2_consts
        elif terms != terms_ref:
            print(f"[WARN] skip {path.name}: design list differs")
            continue
        elif c2_consts != c2_ref:
            print(f"[WARN] skip {path.name}: c2 design list differs")
            continue
        full = _codec.unpack_sx(bytes.fromhex(meta["sk_pke_hex"])).astype(np.int8)
        lo = component * 256
        hi = lo + 256
        traces.append(x)
        sks.append(full[lo:hi])
        pkfps.append(pk)
        seen.add(pk)
    if not traces or terms_ref is None or c2_ref is None:
        raise ValueError("no usable matrix captures")
    min_d = min(x.shape[0] for x in traces)
    min_n = min(x.shape[1] for x in traces)
    min_t = min(x.shape[2] for x in traces)
    traces = [x[:min_d, :min_n, :min_t] for x in traces]
    return Dataset(
        traces=np.stack(traces),
        sks=np.stack(sks),
        design_terms=terms_ref[:min_d],
        design_c2=c2_ref[:min_d],
        pkfps=pkfps,
    )


def shifted_secret(sk: np.ndarray, coef: int) -> np.ndarray:
    return monomial_predict(sk.astype(np.int64), int(coef), 1).astype(np.int8)


def product_vector(sk: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(256, dtype=np.int64)
    for coef, alpha in terms:
        out += monomial_predict(sk.astype(np.int64), int(coef), int(alpha))
    return out


def qproduct_vector(sk: np.ndarray, terms: list[tuple[int, int]]) -> np.ndarray:
    """Product state closer to firmware: c1 coefficients are shifted left by 8."""
    out = np.zeros(256, dtype=np.int64)
    for coef, alpha in terms:
        out += monomial_predict(sk.astype(np.int64), int(coef), int(alpha) << 8)
    return out


def public_qpoly_from_terms(terms: list[tuple[int, int]]) -> np.ndarray:
    """Public c1 component as passed to vec_vec_mult_add: coefficients << 8."""
    out = np.zeros(256, dtype=np.int64)
    for coef, alpha in terms:
        out[int(coef)] += int(alpha) << 8
    return out


def public_poly_from_terms(terms: list[tuple[int, int]]) -> np.ndarray:
    """Unshifted public c1 component as used inside vec_vec_mult_add's poly_mul."""
    out = np.zeros(256, dtype=np.int64)
    for coef, alpha in terms:
        out[int(coef)] += int(alpha)
    return out


def int16_center(v: np.ndarray) -> np.ndarray:
    u = np.asarray(v, dtype=np.int64) & 0xFFFF
    return np.where(u >= 0x8000, u - 0x10000, u).astype(np.int64)


def toom4_eval_vectors(poly: np.ndarray) -> np.ndarray:
    """Disassembly-derived Toom-4 evaluation buffers before 7 Karatsuba calls.

    SMAUG's toom_cook_4way splits a 256-coefficient int16 polynomial into four
    64-coefficient chunks and stores seven wrapped int16 evaluation arrays. The
    formulas below follow the load/add/sub/shift sequence in smaug1.a
    toomcook.c.o at toom_cook_4way+0x164..0x47e.
    """
    chunks = int16_center(np.asarray(poly, dtype=np.int64)).reshape(4, 64)
    c0, c1, c2, c3 = chunks
    evals = [
        c3,
        c0 + 2 * c1 + 4 * c2 + 8 * c3,
        c0 + c1 + c2 + c3,
        c0 - c1 + c2 - c3,
        8 * c0 + 4 * c1 + 2 * c2 + c3,
        8 * c0 - 4 * c1 + 2 * c2 - c3,
        c0,
    ]
    return np.stack([int16_center(e) for e in evals])


def grouped_sum(v: np.ndarray, group: int) -> np.ndarray:
    flat = np.asarray(v, dtype=np.float64).reshape(-1)
    rem = flat.size % group
    if rem:
        flat = np.pad(flat, (0, group - rem), constant_values=0)
    return flat.reshape(-1, group).sum(axis=1)


def hw16(v: np.ndarray) -> np.ndarray:
    return _HW16[np.asarray(v, dtype=np.int64) & 0xFFFF].astype(np.float64)


def parse_toom_kind(kind: str) -> tuple[list[int], str]:
    parts = kind.split("_")
    if len(parts) < 3 or parts[-1] != "hw":
        raise ValueError(f"unknown Toom label kind {kind}")
    point_s, mode = parts[0], parts[1]
    if len(parts) > 3:
        mode = "_".join(parts[1:-1])
    if point_s == "toom7":
        points = list(range(7))
    elif point_s.startswith("toom") and point_s[4:].isdigit():
        point = int(point_s[4:])
        if not (0 <= point < 7):
            raise ValueError(f"Toom point outside [0, 6]: {kind}")
        points = [point]
    elif point_s.startswith("vtoom") and point_s[5:].isdigit():
        point = int(point_s[5:])
        if not (0 <= point < 7):
            raise ValueError(f"Toom point outside [0, 6]: {kind}")
        points = [point]
    else:
        raise ValueError(f"unknown Toom label kind {kind}")
    if mode not in TOOM_MODES:
        raise ValueError(f"unknown Toom label kind {kind}")
    return points, mode


def karatsuba32_operands(s_eval: np.ndarray, p_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the three 32-coefficient operand pairs for one Karatsuba split."""
    s0 = int16_center(s_eval[:32])
    s1 = int16_center(s_eval[32:])
    p0 = int16_center(p_eval[:32])
    p1 = int16_center(p_eval[32:])
    return (
        s0,
        p0,
        int16_center(s0 + s1),
        int16_center(p0 + p1),
        s1,
        p1,
    )


def label_from_karatsuba32(s_eval: np.ndarray, p_eval: np.ndarray, mode: str) -> np.ndarray:
    """Lower-level labels for the top 64x64 Karatsuba split.

    The labels intentionally model intermediate operands/subproducts rather than
    the final recombined polynomial. They are diagnostic hypotheses for where the
    measured power might be dominated.
    """
    operands = karatsuba32_operands(s_eval, p_eval)
    pair_i = int(mode[4])
    a = operands[2 * pair_i]
    b = operands[2 * pair_i + 1]
    suffix = mode[6:]
    if suffix == "conv8":
        return grouped_sum(hw16(np.convolve(a, b)), 8)
    if suffix == "mul4":
        prod = a[:, None] * b[None, :]
        return grouped_sum(hw16(prod).sum(axis=1), 4)
    if suffix == "op8":
        return np.concatenate([grouped_sum(hw16(a), 8), grouped_sum(hw16(b), 8)])
    raise ValueError(f"unknown Karatsuba mode {mode}")


def label_from_toom(sk: np.ndarray, terms: list[tuple[int, int]], kind: str) -> np.ndarray:
    sec = toom4_eval_vectors(sk)
    pub = toom4_eval_vectors(public_qpoly_from_terms(terms))
    return label_from_toom_evals(sec, pub, kind, secret_is_first=True)


def label_from_vtoom(sk: np.ndarray, terms: list[tuple[int, int]], kind: str) -> np.ndarray:
    """Toom labels matching V/W's internal poly_mul: public c1 first, unshifted."""
    pub = toom4_eval_vectors(public_poly_from_terms(terms))
    sec = toom4_eval_vectors(sk)
    return label_from_toom_evals(pub, sec, kind, secret_is_first=False)


def label_from_toom_evals(
    first: np.ndarray,
    second: np.ndarray,
    kind: str,
    *,
    secret_is_first: bool,
) -> np.ndarray:
    points, mode = parse_toom_kind(kind)
    rows = []
    for point in points:
        s_eval = first[point]
        p_eval = second[point]
        if mode == "sec8":
            sec_eval = s_eval if secret_is_first else p_eval
            rows.append(grouped_sum(hw16(sec_eval), 8))
        elif mode.startswith("kara"):
            rows.append(label_from_karatsuba32(s_eval, p_eval, mode))
        elif mode == "mul16" or mode == "mul4":
            group = 16 if mode == "mul16" else 4
            prod = s_eval[:, None] * p_eval[None, :]
            rows.append(grouped_sum(hw16(prod).sum(axis=1), group))
        elif mode == "conv16":
            conv = np.convolve(s_eval, p_eval)
            rows.append(grouped_sum(hw16(conv), 16))
        else:
            raise ValueError(f"unknown Toom label kind {kind}")
    return np.concatenate(rows).astype(np.float64)


def label_from_shifted(s: np.ndarray, kind: str) -> np.ndarray:
    if kind == "support4":
        return (s.reshape(64, 4) != 0).sum(axis=1).astype(np.float64)
    if kind == "sum4":
        return s.reshape(64, 4).sum(axis=1).astype(np.float64)
    if kind == "byte_hw":
        packed = _codec.pack_sx(s.astype(np.int8))
        return np.fromiter((int(x).bit_count() for x in packed), dtype=np.float64, count=64)
    if kind == "eval64_support":
        groups = np.stack([s[:64], s[64:128], s[128:192], s[192:256]], axis=1)
        return (groups != 0).sum(axis=1).astype(np.float64)
    if kind == "eval64_sum":
        groups = np.stack([s[:64], s[64:128], s[128:192], s[192:256]], axis=1)
        return groups.sum(axis=1).astype(np.float64)
    if kind == "support16":
        return (s.reshape(16, 16) != 0).sum(axis=1).astype(np.float64)
    if kind == "sum16":
        return s.reshape(16, 16).sum(axis=1).astype(np.float64)
    raise ValueError(f"unknown label kind {kind}")


def label_from_product(v: np.ndarray, kind: str) -> np.ndarray:
    if kind == "prod16_sum":
        return v.reshape(16, 16).sum(axis=1).astype(np.float64)
    if kind == "prod16_abs":
        return np.abs(v.reshape(16, 16)).sum(axis=1).astype(np.float64)
    if kind == "prod16_hw":
        hw = np.fromiter((int(x & 0xFF).bit_count() for x in v), dtype=np.float64, count=256)
        return hw.reshape(16, 16).sum(axis=1)
    if kind == "prod64_sum":
        return v.reshape(64, 4).sum(axis=1).astype(np.float64)
    if kind == "prod64_hw":
        hw = np.fromiter((int(x & 0xFF).bit_count() for x in v), dtype=np.float64, count=256)
        return hw.reshape(64, 4).sum(axis=1)
    if kind == "qprod16_hw":
        hw = np.fromiter((int(x & 0xFFFF).bit_count() for x in v), dtype=np.float64, count=256)
        return hw.reshape(16, 16).sum(axis=1)
    if kind == "qprod16_hibyte":
        hw = np.fromiter((int((x >> 8) & 0xFF).bit_count() for x in v), dtype=np.float64, count=256)
        return hw.reshape(16, 16).sum(axis=1)
    if kind == "qprod64_hw":
        hw = np.fromiter((int(x & 0xFFFF).bit_count() for x in v), dtype=np.float64, count=256)
        return hw.reshape(64, 4).sum(axis=1)
    if kind == "qprod64_hibyte":
        hw = np.fromiter((int((x >> 8) & 0xFF).bit_count() for x in v), dtype=np.float64, count=256)
        return hw.reshape(64, 4).sum(axis=1)
    raise ValueError(f"unknown product label kind {kind}")


def _grouped_hw(v: np.ndarray, group: int, byte: str = "word") -> np.ndarray:
    centered = int16_center(v)
    if byte == "word":
        hw = hw16(centered)
    elif byte == "lo":
        hw = np.fromiter(
            (int(x & 0xFF).bit_count() for x in centered),
            dtype=np.float64,
            count=centered.size,
        )
    elif byte == "hi":
        hw = np.fromiter(
            (int((x >> 8) & 0xFF).bit_count() for x in centered),
            dtype=np.float64,
            count=centered.size,
        )
    else:
        raise ValueError(f"unknown byte selector {byte}")
    return grouped_sum(hw, group)


def label_from_vecadd_state(
    sk: np.ndarray,
    terms: list[tuple[int, int]],
    c2_const: int,
    kind: str,
) -> np.ndarray:
    """Labels for the actual vec_vec_mult_add sequence after poly_mul_acc.

    Disassembly shows:
      1. a_shifted >>= mod
      2. tmp = poly_mul_acc(a_unshifted, sk)
      3. tmp <<= mod
      4. output = poly_add(output, tmp)

    If a capture stores a constant public c2, output before step 4 is c2<<11.
    Older captures do not have c2 metadata and are treated as c2=0.
    """
    tmp = int16_center(product_vector(sk, terms))
    shifted = int16_center(tmp << SMAUG1.log_p)
    before = int16_center(np.full_like(shifted, int(c2_const)) << (SMAUG1.log_q + SMAUG1.log_t))
    after = int16_center(before + shifted)
    delta = np.bitwise_xor(before.astype(np.int64) & 0xFFFF, after.astype(np.int64) & 0xFFFF)

    if kind == "vtmp16_hw":
        return _grouped_hw(tmp, 16)
    if kind == "vtmp64_hw":
        return _grouped_hw(tmp, 64)
    if kind == "vtmp16_lohw":
        return _grouped_hw(tmp, 16, byte="lo")
    if kind == "vtmp16_hihw":
        return _grouped_hw(tmp, 16, byte="hi")
    if kind == "vshift16_hw":
        return _grouped_hw(shifted, 16)
    if kind == "vshift64_hw":
        return _grouped_hw(shifted, 64)
    if kind == "vshift16_hihw":
        return _grouped_hw(shifted, 16, byte="hi")
    if kind == "vadd16_hw":
        return _grouped_hw(after, 16)
    if kind == "vadd64_hw":
        return _grouped_hw(after, 64)
    if kind == "vdelta16_hw":
        return grouped_sum(hw16(delta), 16)
    if kind == "vdelta64_hw":
        return grouped_sum(hw16(delta), 64)
    raise ValueError(f"unknown vecadd label kind {kind}")


def mu_bits_from_component_state(
    sk: np.ndarray,
    terms: list[tuple[int, int]],
    c2_const: int,
) -> np.ndarray:
    """µ′ bits for a single-component chosen-CT design.

    The matrix captures place all public c1 terms in one component and load
    only that component's secret into ``ds.sks``. This is equivalent to the full
    ``chosen.predict_mu_prime`` inner product with other components set to zero.
    """
    inner = product_vector(sk, terms)
    inner_mod = inner % SMAUG1.q
    half = SMAUG1.q // 2
    inner_signed = ((inner_mod + half) % SMAUG1.q) - half
    c2 = np.full(SMAUG1.n, int(c2_const), dtype=np.int64) % SMAUG1.p2
    numerator = 2 * SMAUG1.t * (inner_signed * SMAUG1.p2 + c2 * SMAUG1.p) + SMAUG1.p * SMAUG1.p2
    denom = 2 * SMAUG1.p * SMAUG1.p2
    return (numerator // denom % SMAUG1.t).astype(np.float64)


def label_from_mu_bits(
    sk: np.ndarray,
    terms: list[tuple[int, int]],
    c2_const: int,
    kind: str,
) -> np.ndarray:
    bits = mu_bits_from_component_state(sk, terms, c2_const)
    if kind == "mu_total_hw":
        return np.asarray([bits.sum()], dtype=np.float64)
    if kind == "mu_byte_hw":
        return bits.reshape(32, 8).sum(axis=1).astype(np.float64)
    if kind == "mu_block16_hw":
        return bits.reshape(16, 16).sum(axis=1).astype(np.float64)
    if kind == "mu_block32_hw":
        return bits.reshape(8, 32).sum(axis=1).astype(np.float64)
    if kind == "mu_bit":
        return bits.astype(np.float64)
    raise ValueError(f"unknown mu label kind {kind}")


def make_labels(ds: Dataset, kind: str) -> np.ndarray:
    labels = []
    for key_i in range(ds.sks.shape[0]):
        per_design = []
        for design_i, terms in enumerate(ds.design_terms):
            if kind.startswith("vtoom"):
                per_design.append(label_from_vtoom(ds.sks[key_i], terms, kind))
            elif kind.startswith(("vtmp", "vshift", "vadd", "vdelta")):
                per_design.append(
                    label_from_vecadd_state(
                        ds.sks[key_i],
                        terms,
                        ds.design_c2[design_i],
                        kind,
                    )
                )
            elif kind.startswith("mu_"):
                per_design.append(
                    label_from_mu_bits(
                        ds.sks[key_i],
                        terms,
                        ds.design_c2[design_i],
                        kind,
                    )
                )
            elif kind.startswith("toom"):
                per_design.append(label_from_toom(ds.sks[key_i], terms, kind))
            elif kind.startswith("qprod"):
                per_design.append(label_from_product(qproduct_vector(ds.sks[key_i], terms), kind))
            elif kind.startswith("prod"):
                per_design.append(label_from_product(product_vector(ds.sks[key_i], terms), kind))
            else:
                if len(terms) != 1:
                    raise ValueError(
                        f"{kind} requires monomial designs; got terms={terms}"
                    )
                per_design.append(label_from_shifted(shifted_secret(ds.sks[key_i], terms[0][0]), kind))
        labels.append(np.stack(per_design))
    return np.stack(labels)  # (S,D,L)


def cached_labels(ds: Dataset, kind: str) -> np.ndarray:
    key = (id(ds), kind)
    labels = _LABEL_CACHE.get(key)
    if labels is None:
        labels = make_labels(ds, kind)
        _LABEL_CACHE[key] = labels
    return labels


def make_observation_features(x: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x=(S,D,N,T). Return Xmean=(S,D,B), Xvar=(S,D,B), counts=(S,D)."""
    s, d, n, t = x.shape
    b = t // block
    xb = x[..., : b * block].reshape(s, d, n, b, block).mean(axis=4)
    return xb.mean(axis=2), xb.var(axis=2, ddof=1), np.full((s, d), n, dtype=np.float64)


def select_features(
    x_train: np.ndarray,
    var_train: np.ndarray,
    count_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    if cfg.feature_mode == "snr":
        between = x_train.var(axis=0, ddof=1)
        noise = np.mean(var_train / count_train[:, None], axis=0)
        score = between / np.maximum(noise, 1e-12)
    elif cfg.feature_mode == "corr":
        xc = x_train - x_train.mean(axis=0, keepdims=True)
        yc = y_train - y_train.mean(axis=0, keepdims=True)
        nx = np.linalg.norm(xc, axis=0, keepdims=True)
        ny = np.linalg.norm(yc, axis=0, keepdims=True).T
        denom = np.maximum(ny * nx, 1e-12)
        rho = (yc.T @ xc) / denom
        score = np.nanmax(np.abs(np.where(np.isfinite(rho), rho, 0.0)), axis=0)
    else:
        raise ValueError(cfg.feature_mode)
    return np.sort(np.argsort(-score, kind="stable")[: min(cfg.n_features, score.size)])


def fit_predict_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                      features: np.ndarray, ridge: float) -> np.ndarray:
    xt = x_train[:, features]
    xv = x_test[:, features]
    mu = xt.mean(axis=0, keepdims=True)
    sd = xt.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    xt = (xt - mu) / sd
    xv = (xv - mu) / sd
    ym = y_train.mean(axis=0, keepdims=True)
    yc = y_train - ym
    gram = xt @ xt.T
    alpha = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), yc)
    w = xt.T @ alpha
    return xv @ w + ym


def two_way_residualize_x(
    x: np.ndarray,
    hold: int,
    train_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test residuals for x=(S,D,B), fit using train keys only."""
    x_train = x[train_keys]
    design_mean = x_train.mean(axis=0, keepdims=True)
    global_mean = x_train.mean(axis=(0, 1), keepdims=True)
    train_key_mean = x_train.mean(axis=1, keepdims=True)
    test = x[[hold]]
    test_key_mean = test.mean(axis=1, keepdims=True)
    return (
        x_train - train_key_mean - design_mean + global_mean,
        test - test_key_mean - design_mean + global_mean,
    )


def two_way_residualize_y(
    y_train_source: np.ndarray,
    y_true: np.ndarray,
    hold: int,
    train_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Residualize train labels and real held-out labels in the same fold shape."""
    train = y_train_source[train_keys]
    train_design_mean = train.mean(axis=0, keepdims=True)
    train_global_mean = train.mean(axis=(0, 1), keepdims=True)
    train_key_mean = train.mean(axis=1, keepdims=True)

    true_train = y_true[train_keys]
    true_design_mean = true_train.mean(axis=0, keepdims=True)
    true_global_mean = true_train.mean(axis=(0, 1), keepdims=True)
    true_test = y_true[[hold]]
    true_test_key_mean = true_test.mean(axis=1, keepdims=True)
    return (
        train - train_key_mean - train_design_mean + train_global_mean,
        true_test - true_test_key_mean - true_design_mean + true_global_mean,
    )


def evaluate(ds: Dataset, kind: str, cfg: Config,
             diff_base: int | None = None,
             key_perm: np.ndarray | None = None,
             residualize: str = "none") -> dict[str, float | np.ndarray]:
    xmean, xvar, counts = make_observation_features(ds.traces, cfg.block)
    y_true = cached_labels(ds, kind)

    if diff_base is not None:
        if not (0 <= diff_base < xmean.shape[1]):
            raise ValueError(f"diff_base={diff_base} outside [0, {xmean.shape[1]})")
        keep = [i for i in range(xmean.shape[1]) if i != diff_base]
        xmean = xmean[:, keep, :] - xmean[:, [diff_base], :]
        xvar = xvar[:, keep, :] + xvar[:, [diff_base], :]
        counts = counts[:, keep]
        y_true = y_true[:, keep, :] - y_true[:, [diff_base], :]

    y_labels = y_true if key_perm is None else y_true[key_perm]
    s, d, b = xmean.shape

    preds = []
    trues = []
    feature_counts = np.zeros(b, dtype=np.int64)
    for hold in range(s):
        train_keys = np.asarray([i for i in range(s) if i != hold], dtype=np.int64)
        if residualize == "none":
            x_train_fold = xmean[train_keys]
            x_test_fold = xmean[[hold]]
            y_train_fold = y_labels[train_keys]
            y_test_fold = y_true[[hold]]
        elif residualize == "two-way":
            x_train_fold, x_test_fold = two_way_residualize_x(xmean, hold, train_keys)
            y_train_fold, y_test_fold = two_way_residualize_y(
                y_labels,
                y_true,
                hold,
                train_keys,
            )
        else:
            raise ValueError(f"unknown residualize mode {residualize!r}")

        x_train = x_train_fold.reshape(-1, b)
        v_train = xvar[train_keys].reshape(-1, b)
        c_train = counts[train_keys].reshape(-1)
        y_train = y_train_fold.reshape(-1, y_true.shape[-1])
        x_test = x_test_fold.reshape(d, b)
        features = select_features(x_train, v_train, c_train, y_train, cfg)
        feature_counts[features] += 1
        pred = fit_predict_ridge(x_train, y_train, x_test, features, cfg.ridge)
        preds.append(pred)
        trues.append(y_test_fold.reshape(d, y_true.shape[-1]))

    yp = np.concatenate(preds, axis=0)
    yt = np.concatenate(trues, axis=0)
    lo, hi = float(np.min(yt)), float(np.max(yt))
    pred_round = np.clip(np.rint(yp), lo, hi)
    mae = float(np.mean(np.abs(yp - yt)))
    rounded_mae = float(np.mean(np.abs(pred_round - yt)))
    exact = float(np.mean(pred_round == yt))
    if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
        corr = float(np.corrcoef(yp.ravel(), yt.ravel())[0, 1])
    else:
        corr = 0.0
    return {
        "mae": mae,
        "rounded_mae": rounded_mae,
        "exact": exact,
        "corr": corr,
        "pred": yp,
        "true": yt,
        "feature_counts": feature_counts,
    }


def summarize(kind: str, cfg: Config, real: dict[str, float | np.ndarray],
              nulls: list[dict[str, float | np.ndarray]]) -> str:
    null_exact = np.asarray([n["exact"] for n in nulls], dtype=np.float64)
    null_mae = np.asarray([n["rounded_mae"] for n in nulls], dtype=np.float64)
    null_corr = np.asarray([n["corr"] for n in nulls], dtype=np.float64)
    exact = float(real["exact"])
    mae = float(real["rounded_mae"])
    corr = float(real["corr"])
    z_exact = (exact - null_exact.mean()) / max(null_exact.std(ddof=1), 1e-12)
    z_mae = (null_mae.mean() - mae) / max(null_mae.std(ddof=1), 1e-12)
    z_corr = (corr - null_corr.mean()) / max(null_corr.std(ddof=1), 1e-12)
    return (
        f"{kind} {cfg.name}: exact={exact:.4f} "
        f"(null {null_exact.mean():.4f}±{null_exact.std(ddof=1):.4f}, z={z_exact:+.2f}), "
        f"rMAE={mae:.4f} (null {null_mae.mean():.4f}±{null_mae.std(ddof=1):.4f}, z={z_mae:+.2f}), "
        f"corr={corr:.4f} (null {null_corr.mean():.4f}±{null_corr.std(ddof=1):.4f}, z={z_corr:+.2f})"
    )


def summarize_features(real: dict[str, float | np.ndarray], cfg: Config, limit: int = 16) -> str:
    counts = np.asarray(real["feature_counts"], dtype=np.int64)
    if counts.size == 0:
        return "top selected blocks: none"
    top = np.argsort(-counts, kind="stable")[: min(limit, counts.size)]
    parts = []
    for block_i in top:
        if counts[block_i] <= 0:
            continue
        lo = int(block_i) * cfg.block
        hi = lo + cfg.block
        parts.append(f"{lo}:{hi}({int(counts[block_i])})")
    return "top selected blocks: " + (", ".join(parts) if parts else "none")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(args.inputs, args.component)
    if args.design_offset:
        if not (0 <= args.design_offset < ds.traces.shape[1]):
            raise ValueError(f"--design-offset outside [0, {ds.traces.shape[1]})")
    if args.design_offset or args.max_designs is not None:
        max_designs = (
            ds.traces.shape[1] - args.design_offset
            if args.max_designs is None
            else args.max_designs
        )
        lo = args.design_offset
        hi = lo + max_designs
        if not (lo < hi <= ds.traces.shape[1]):
            raise ValueError(f"design slice {lo}:{hi} outside [0, {ds.traces.shape[1]}]")
        ds = Dataset(
            traces=ds.traces[:, lo:hi],
            sks=ds.sks,
            design_terms=ds.design_terms[lo:hi],
            design_c2=ds.design_c2[lo:hi],
            pkfps=ds.pkfps,
        )
    if args.max_traces is not None:
        if not (1 <= args.max_traces <= ds.traces.shape[2]):
            raise ValueError(f"--max-traces outside [1, {ds.traces.shape[2]}]")
        ds = Dataset(
            traces=ds.traces[:, :, : args.max_traces],
            sks=ds.sks,
            design_terms=ds.design_terms,
            design_c2=ds.design_c2,
            pkfps=ds.pkfps,
        )
    sample_lo, sample_hi = parse_sample_range(args.sample_range, ds.traces.shape[-1])
    if sample_lo != 0 or sample_hi != ds.traces.shape[-1]:
        ds = Dataset(
            traces=ds.traces[..., sample_lo:sample_hi],
            sks=ds.sks,
            design_terms=ds.design_terms,
            design_c2=ds.design_c2,
            pkfps=ds.pkfps,
        )
    if ds.traces.shape[0] < 4:
        print(f"[FAIL] need at least 4 keys, got {ds.traces.shape[0]}")
        return 1
    print(
        f"[INFO] S={ds.traces.shape[0]} D={ds.traces.shape[1]} "
        f"N={ds.traces.shape[2]} T={ds.traces.shape[3]} "
        f"sample_range=[{sample_lo}:{sample_hi}] "
        f"diff_base={args.diff_base} residualize={args.residualize} "
        f"terms={ds.design_terms}"
    )

    if args.sweep:
        configs = [
            Config(block, nf, ridge, mode)
            for block in (8, 16, 32, 64)
            for nf in (32, 64, 128, 256)
            for ridge in (1.0, 10.0, 100.0)
            for mode in ("snr", "corr")
        ]
    else:
        configs = [Config(args.block, args.n_features, args.ridge, args.feature_mode)]

    lines = [
        "S2 Z matrix low-dimensional profiling",
        f"S={ds.traces.shape[0]}, D={ds.traces.shape[1]}, N={ds.traces.shape[2]}, T={ds.traces.shape[3]}",
        f"sample_range=[{sample_lo}:{sample_hi}]",
        f"terms={ds.design_terms}",
        f"diff_base={args.diff_base}",
        f"residualize={args.residualize}",
        "",
    ]
    plot_rows = []

    for kind in args.label_kinds:
        print(f"[KIND] {kind}")
        real_rows = []
        for cfg in configs:
            real = evaluate(
                ds,
                kind,
                cfg,
                diff_base=args.diff_base,
                residualize=args.residualize,
            )
            real_rows.append((float(real["exact"]), -float(real["rounded_mae"]), float(real["corr"]), cfg, real))
        real_rows.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
        best_cfg = real_rows[0][3]
        best_real = real_rows[0][4]
        print(
            f"[BEST-REAL] {kind} {best_cfg.name} "
            f"exact={best_real['exact']:.4f} rMAE={best_real['rounded_mae']:.4f} corr={best_real['corr']:.4f}"
        )
        if args.sweep:
            lines.append(f"{kind} real-only top configs:")
            for row in real_rows[:5]:
                cfg = row[3]
                real = row[4]
                lines.append(
                    f"  {cfg.name}: exact={real['exact']:.4f}, "
                    f"rMAE={real['rounded_mae']:.4f}, corr={real['corr']:.4f}"
                )

        nulls = []
        for i in range(args.n_perm):
            nulls.append(
                evaluate(
                    ds,
                    kind,
                    best_cfg,
                    diff_base=args.diff_base,
                    key_perm=rng.permutation(ds.traces.shape[0]),
                    residualize=args.residualize,
                )
            )
            if (i + 1) % max(1, args.n_perm // 5) == 0:
                print(f"[NULL {kind}] {i + 1}/{args.n_perm}")
        summary = summarize(kind, best_cfg, best_real, nulls)
        print("[SUMMARY] " + summary)
        feature_summary = summarize_features(best_real, best_cfg)
        print("[FEATURES] " + feature_summary)
        lines.append(summary)
        lines.append(feature_summary)
        lines.append("")
        plot_rows.append((kind, best_real, nulls))

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = args.out_prefix.with_suffix(".txt")
    txt_path.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(len(plot_rows), 3, figsize=(13, 3.2 * len(plot_rows)))
    if len(plot_rows) == 1:
        axes = np.asarray([axes])
    for row_i, (kind, real, nulls) in enumerate(plot_rows):
        null_exact = np.asarray([n["exact"] for n in nulls], dtype=np.float64)
        null_mae = np.asarray([n["rounded_mae"] for n in nulls], dtype=np.float64)
        null_corr = np.asarray([n["corr"] for n in nulls], dtype=np.float64)
        vals = [
            (null_exact, float(real["exact"]), "exact"),
            (null_mae, float(real["rounded_mae"]), "rounded MAE"),
            (null_corr, float(real["corr"]), "corr"),
        ]
        for col_i, (null_arr, real_val, title) in enumerate(vals):
            ax = axes[row_i, col_i]
            ax.hist(null_arr[np.isfinite(null_arr)], bins=20, color="0.75", edgecolor="0.35")
            ax.axvline(real_val, color="C3", lw=2)
            ax.set_title(f"{kind}: {title}")
    fig.tight_layout()
    png_path = args.out_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=120)
    print(f"[OK] wrote {txt_path}")
    print(f"[OK] wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
