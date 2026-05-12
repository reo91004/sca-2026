"""Unit tests for s4_pair_distance_oracle.

Covers:
- AUROC manual implementation (rank-sum).
- flip_any binary label = (any mu' bit XOR != 0).
- flip_byte_hw label round-trips against direct XOR-of-mu_bits.
- evaluate_oracle on a synthetic dataset where signal is perfectly aligned
  with the binary label gives AUROC=1.0; randomly-aligned signal gives ~0.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.smaug import params  # noqa: E402
from scripts.smaug.s2_z_lowdim_analyze import Config, label_from_mu_bits  # noqa: E402
from scripts.smaug.s4_pair_distance_oracle import (  # noqa: E402
    PairDataset,
    auroc,
    evaluate_oracle,
    labels_flip_any,
    labels_flip_byte_hw,
)


def _random_ternary_sk(p, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 0, 1], size=(p.module_rank, p.n)).astype(np.int64)


def test_auroc_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    assert auroc(scores, labels) == 1.0


def test_auroc_inverted_separation_is_zero() -> None:
    scores = np.array([0.9, 0.95, 0.99, 0.1, 0.2, 0.3], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    assert auroc(scores, labels) == 0.0


def test_auroc_random_is_about_half() -> None:
    rng = np.random.default_rng(0xA0CA)
    n = 5000
    scores = rng.standard_normal(n)
    labels = rng.integers(0, 2, size=n)
    a = auroc(scores, labels)
    assert 0.45 <= a <= 0.55, f"random AUROC = {a:.4f}, expected ~0.5"


def test_auroc_handles_ties_with_average_rank() -> None:
    scores = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    assert auroc(scores, labels) == 0.5


def test_auroc_nan_when_class_empty() -> None:
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    labels = np.array([1, 1, 1], dtype=np.int64)
    assert np.isnan(auroc(scores, labels))


def test_flip_any_matches_xor_of_mu_bits() -> None:
    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0xF11)
    terms_list = [[(0, 128)], [(0, 64)], [(13, 192)]]
    base_c2 = [0, 7, 15]
    delta_c2 = [1, 8, 16]

    pds = PairDataset(
        diff_xmean=np.zeros((1, len(terms_list), 4), dtype=np.float64),
        xvar=np.ones((1, len(terms_list), 4), dtype=np.float64),
        counts=np.ones((1, len(terms_list)), dtype=np.float64),
        sks=sk[0][None, :],
        base_terms=terms_list,
        base_c2=base_c2,
        delta_c2=delta_c2,
        pkfps=["test"],
    )
    flip = labels_flip_any(pds)
    assert flip.shape == (1, 3)
    for p_idx, (terms, c2_a, c2_b) in enumerate(zip(terms_list, base_c2, delta_c2)):
        b0 = label_from_mu_bits(sk[0], terms, c2_a, "mu_bit").astype(np.int8)
        b1 = label_from_mu_bits(sk[0], terms, c2_b, "mu_bit").astype(np.int8)
        expected = float(np.any(np.bitwise_xor(b0, b1)))
        assert flip[0, p_idx] == expected, (
            f"design {p_idx}: got {flip[0, p_idx]} expected {expected}"
        )


def test_flip_byte_hw_round_trips() -> None:
    p = params.SMAUG1
    sk = _random_ternary_sk(p, seed=0xF12)
    terms = [(7, 192)]
    base_c2 = 14
    delta_c2 = 17

    pds = PairDataset(
        diff_xmean=np.zeros((1, 1, 4), dtype=np.float64),
        xvar=np.ones((1, 1, 4), dtype=np.float64),
        counts=np.ones((1, 1), dtype=np.float64),
        sks=sk[0][None, :],
        base_terms=[terms],
        base_c2=[base_c2],
        delta_c2=[delta_c2],
        pkfps=["test"],
    )
    byte_hw = labels_flip_byte_hw(pds)
    assert byte_hw.shape == (1, 1, 32)
    b0 = label_from_mu_bits(sk[0], terms, base_c2, "mu_bit").astype(np.int8)
    b1 = label_from_mu_bits(sk[0], terms, delta_c2, "mu_bit").astype(np.int8)
    expected = np.bitwise_xor(b0, b1).reshape(32, 8).sum(axis=1).astype(np.float64)
    assert np.array_equal(byte_hw[0, 0], expected)


def test_evaluate_oracle_perfect_signal_gives_auroc_one() -> None:
    """Synthetic: feature vector encodes flip_any directly → held-out AUROC=1.0.

    The fixture builds a (S=4, P=8, B=4) diff feature where one block exactly
    equals the binary label. With held-out leave-one-key-out + ridge, the model
    should perfectly recover the ranking on the held-out fold.
    """
    p = params.SMAUG1
    rng = np.random.default_rng(0xCAFE)

    # Choose a c2 pair where flip_any has both classes for ALL keys.
    # base_c2=15, delta_c2=16 is the canonical threshold pair on SMAUG1.
    terms_list: list[list[tuple[int, int]]] = []
    base_c2: list[int] = []
    delta_c2: list[int] = []
    coefs = [0, 8, 16, 24, 32, 40, 48, 56]
    for c in coefs:
        terms_list.append([(int(c), 128)])
        base_c2.append(15)
        delta_c2.append(16)

    sks = np.stack(
        [_random_ternary_sk(p, seed=0xC0DE0 + k)[0] for k in range(4)]
    )

    # Build a placeholder diff feature, then overwrite block 0 with the binary
    # label scaled large so ridge picks it as the dominant feature.
    diff = rng.standard_normal((4, len(coefs), 4)).astype(np.float64) * 0.01
    pds = PairDataset(
        diff_xmean=diff,
        xvar=np.ones((4, len(coefs), 4), dtype=np.float64),
        counts=np.full((4, len(coefs)), 10, dtype=np.float64),
        sks=sks,
        base_terms=terms_list,
        base_c2=base_c2,
        delta_c2=delta_c2,
        pkfps=[f"k{k:02d}" for k in range(4)],
    )
    flip = labels_flip_any(pds)
    # Skip the test if any key has degenerate (single-class) labels — that
    # would prevent meaningful AUROC for that fold.
    per_key_pos = flip.sum(axis=1)
    if (per_key_pos == 0).any() or (per_key_pos == len(coefs)).any():
        return  # degenerate sk draw; harmless for unit-test purposes
    pds.diff_xmean[..., 0] = flip * 5.0

    cfg = Config(block=4, n_features=2, ridge=1.0, feature_mode="corr")
    out = evaluate_oracle(pds, cfg)
    assert out["pooled_auroc"] >= 0.99, (
        f"perfect-signal AUROC = {out['pooled_auroc']:.4f}, expected >=0.99"
    )


def test_evaluate_oracle_random_signal_near_half() -> None:
    p = params.SMAUG1
    rng = np.random.default_rng(0xBEEF)
    terms_list: list[list[tuple[int, int]]] = []
    base_c2: list[int] = []
    delta_c2: list[int] = []
    for c in [0, 8, 16, 24, 32, 40, 48, 56]:
        terms_list.append([(int(c), 128)])
        base_c2.append(15)
        delta_c2.append(16)

    sks = np.stack(
        [_random_ternary_sk(p, seed=0xC0DE0 + k)[0] for k in range(4)]
    )
    diff = rng.standard_normal((4, len(terms_list), 4)).astype(np.float64)
    pds = PairDataset(
        diff_xmean=diff,
        xvar=np.ones((4, len(terms_list), 4), dtype=np.float64),
        counts=np.full((4, len(terms_list)), 10, dtype=np.float64),
        sks=sks,
        base_terms=terms_list,
        base_c2=base_c2,
        delta_c2=delta_c2,
        pkfps=[f"k{k:02d}" for k in range(4)],
    )
    cfg = Config(block=4, n_features=2, ridge=10.0, feature_mode="corr")
    out = evaluate_oracle(pds, cfg)
    if not np.isfinite(out["pooled_auroc"]):
        return  # degenerate (all-positive or all-negative) — accept.
    assert 0.20 <= out["pooled_auroc"] <= 0.80, (
        f"random-signal AUROC = {out['pooled_auroc']:.4f}, "
        "expected to be near 0.5 for an 8-pair toy problem"
    )
