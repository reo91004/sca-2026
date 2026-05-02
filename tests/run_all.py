#!/usr/bin/env python3
"""tests/test_*.py 의 모든 test_* 함수를 raw 실행으로 돌린다.

pytest 가 없는 환경에서도 동작. 각 테스트는 raise 시 실패. 모두 통과
하면 exit 0, 하나라도 실패하면 exit 1.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    files = sorted(p for p in here.glob("test_*.py"))
    total = 0
    failed: list[tuple[str, str, str]] = []  # (file, fn, traceback)

    for f in files:
        spec = importlib.util.spec_from_file_location(f.stem, f)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(vars(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            total += 1
            print(f"[RUN] {f.name}::{name}")
            try:
                # pytest-style tmp_path 인자 자동 제공 (TemporaryDirectory).
                sig = inspect.signature(fn)
                if "tmp_path" in sig.parameters:
                    with tempfile.TemporaryDirectory() as td:
                        fn(tmp_path=Path(td))
                else:
                    fn()
            except Exception:
                failed.append((f.name, name, traceback.format_exc()))
                print(f"[FAIL] {f.name}::{name}")
            else:
                print(f"[OK ] {f.name}::{name}")

    print()
    if failed:
        print(f"==== {len(failed)}/{total} FAILED ====")
        for fname, n, tb in failed:
            print(f"\n----- {fname}::{n} -----\n{tb}")
        return 1
    print(f"==== {total}/{total} OK ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
