#!/usr/bin/env python3
"""Run the six CPU self-tests in the public DyPAC-VLA release."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

SELFTEST_MODULES = (
    "dypac_vla_protocol",
    "quantvla_ripa",
    "quantvla_selector",
    "quantvla_metric_protocol",
    "quantvla_hessian_w4",
    "quantvla_dynamic_a8_protocol",
)


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for name in SELFTEST_MODULES:
        try:
            module = importlib.import_module(name)
            module.selftest()
        except BaseException as exc:  # noqa: BLE001 - report all checks in one run
            failures.append((name, exc))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{len(failures)} of {len(SELFTEST_MODULES)} self-tests failed")
        return 1
    print(f"\nall {len(SELFTEST_MODULES)} self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())