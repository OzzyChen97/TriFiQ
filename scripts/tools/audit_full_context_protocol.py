#!/usr/bin/env python3
"""Fail-closed audit for full-context plans, scores, and frozen selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantvla_full_context import (
    PROTOCOL,
    protocol_attestation,
    require_protocol_attestation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+")
    args = parser.parse_args()
    expected = protocol_attestation()
    rows = []
    for raw in args.artifact:
        path = Path(raw).expanduser().resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        carrier = value.get("meta") if "full_context_protocol" not in value else value
        require_protocol_attestation(carrier or {}, source=str(path))
        attestation = (carrier or {}).get("full_context_protocol") or {}
        source = value.get("source_sha256") or {}
        shared_source_checks = {
            "metric": attestation.get("metric_core_sha256"),
            "selection_core": attestation.get("selection_core_sha256"),
        }
        for key, expected_sha in shared_source_checks.items():
            if key in source and source[key] != expected_sha:
                raise ValueError(
                    f"{path}: shared {key} source differs from the frozen protocol"
                )
        rendered = json.dumps(value, sort_keys=True).lower()
        if value.get("kind", "").endswith("frozen") or (value.get("meta") or {}).get("frozen"):
            for forbidden in PROTOCOL["deployment"]["forbidden_components"]:
                if f'"{forbidden}": true' in rendered:
                    raise ValueError(f"{path}: forbidden deployed component {forbidden}")
        rows.append(
            {
                "path": str(path),
                "selection_core_sha256": attestation["selection_core_sha256"],
                "metric_core_sha256": attestation["metric_core_sha256"],
                "quick_statistics_sha256": attestation["quick_statistics_sha256"],
                "formal_statistics_sha256": attestation["formal_statistics_sha256"],
            }
        )
    if len({row["selection_core_sha256"] for row in rows}) != 1:
        raise ValueError("selection core differs across model artifacts")
    for key in (
        "metric_core_sha256",
        "quick_statistics_sha256",
        "formal_statistics_sha256",
    ):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"shared {key} differs across model artifacts")
    print(json.dumps({"passed": True, "artifacts": rows}, indent=2))


if __name__ == "__main__":
    main()
