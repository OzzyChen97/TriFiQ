#!/usr/bin/env python3
"""Run every self-test that ships with the minimal DyPAC-VLA release."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

SELFTEST_MODULES = (
    "quantvla_table1_bytes",
    "gr00t_func_metrics",
    "quantvla_metric_protocol",
    "quantvla_hessian_w4",
    "quantvla_full_context",
)


def main() -> int:
    failures: list[tuple[str, BaseException]] = []

    for name in SELFTEST_MODULES:
        try:
            module = importlib.import_module(name)
            module.selftest()
        except BaseException as exc:  # noqa: BLE001 - report every failure, keep going
            failures.append((name, exc))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    # quantvla_dynamic_a8_protocol exposes a contract validator rather than a selftest.
    try:
        dynamic = importlib.import_module("quantvla_dynamic_a8_protocol")
        attestation = dynamic.protocol_attestation()
        assert attestation["protocol_sha256"], "empty protocol hash"
        print("[OK]   quantvla_dynamic_a8_protocol: runtime contract validated")
    except BaseException as exc:  # noqa: BLE001
        failures.append(("quantvla_dynamic_a8_protocol", exc))
        print(f"[FAIL] quantvla_dynamic_a8_protocol: {type(exc).__name__}: {exc}")

    print()
    print("NOTE: scripts/tools/quantvla_cross_model_protocol.py imports cleanly and is used by every")
    print("      check above, but its own selftest() additionally verifies the frozen on-policy probe")
    print("      and calibration buffers, which are not redistributed in this minimal release.")

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print(f"\nall {len(SELFTEST_MODULES) + 1} shipped checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
