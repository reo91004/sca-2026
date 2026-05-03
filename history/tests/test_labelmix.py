"""host/analysis/labelmix.py 의 라벨 합치기/split 동작."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from host.analysis import io as a_io  # noqa: E402
from host.analysis import labelmix  # noqa: E402


def _make_npz(tmpdir: Path, name: str, n: int, t: int, label: str,
              resp_byte_value: int = 0) -> Path:
    rng = np.random.default_rng(hash(name) & 0xFFFF)
    traces = rng.normal(0, 1, size=(n, t)).astype(np.float32)
    responses = np.full((n, 1), resp_byte_value, dtype=np.uint8)
    meta = {"target": "smaug", "cmd": "D", "label": label,
            "samples": t, "gain_db": 25.0,
            "ss_ver": "SS_VER_1_1", "baud": 38400,
            "n_timeouts": 0, "n_attempted": n,
            "send_len": 0, "resp_len": 1}
    p = tmpdir / f"{name}.npz"
    a_io.save_capture(p, traces, responses, meta)
    return p


def test_labelmix_basic_pair(tmp_path: Path) -> None:
    a = _make_npz(tmp_path, "a", 100, 800, "zero")
    b = _make_npz(tmp_path, "b", 100, 800, "one")
    mix = labelmix.LabelMix.from_npz_files([(a, "zero"), (b, "one")])
    assert sorted(mix.labels()) == ["one", "zero"]
    assert mix.n("zero") == 100 and mix.n("one") == 100
    ta, tb = mix.group_pair("zero", "one")
    assert ta.shape == (100, 800) and tb.shape == (100, 800)


def test_labelmix_concat_same_label(tmp_path: Path) -> None:
    a = _make_npz(tmp_path, "a", 50, 400, "zero")
    b = _make_npz(tmp_path, "b", 75, 400, "zero")
    mix = labelmix.LabelMix.from_npz_files([(a, "zero"), (b, "zero")])
    assert mix.n("zero") == 125  # 50 + 75 합쳐짐


def test_labelmix_rejects_sample_mismatch(tmp_path: Path) -> None:
    a = _make_npz(tmp_path, "a", 30, 400, "zero")
    b = _make_npz(tmp_path, "b", 30, 800, "one")
    raised = False
    try:
        labelmix.LabelMix.from_npz_files([(a, "zero"), (b, "one")])
    except ValueError as e:
        raised = "sample" in str(e).lower() or "샘플" in str(e) or "불일치" in str(e)
    assert raised


def test_relabel_by_response_byte(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n, t = 100, 200
    traces = rng.normal(size=(n, t)).astype(np.float32)
    resp = np.zeros((n, 1), dtype=np.uint8)
    resp[10:30, 0] = 1
    resp[50:70, 0] = 2
    meta = {"samples": t, "gain_db": 0.0, "target": "smaug", "cmd": "D",
            "send_len": 0, "resp_len": 1, "ss_ver": "SS_VER_1_1",
            "baud": 38400, "n_timeouts": 0, "n_attempted": n}
    p = tmp_path / "x.npz"
    a_io.save_capture(p, traces, resp, meta)
    cap = a_io.load_capture(p)
    groups = labelmix.relabel_by_response_byte(cap, byte_index=0)
    assert set(groups.keys()) == {0, 1, 2}
    assert groups[0].size == 60
    assert groups[1].size == 20
    assert groups[2].size == 20


if __name__ == "__main__":
    import tempfile
    fns = sorted(n for n in globals() if n.startswith("test_"))
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for n in fns:
            print(f"[RUN] {n}")
            globals()[n](tdp)
            print(f"[OK ] {n}")
    print(f"[OK ] {len(fns)} tests")
