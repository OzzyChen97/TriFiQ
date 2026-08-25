#!/usr/bin/env python3
"""Merge and fail-close the two model-level selector equivalence reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gr00t", required=True)
    parser.add_argument("--pi05", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path, gr_path, pi_path = map(
        lambda value: Path(value).resolve(), (args.manifest, args.gr00t, args.pi05)
    )
    manifest_hash = sha256_file(manifest_path)
    reports = {"gr00t": read(gr_path), "pi05": read(pi_path)}
    errors = []
    for model, report in reports.items():
        if report.get("model") != model:
            errors.append(f"{model}: wrong model id")
        if report.get("manifest_sha256") != manifest_hash:
            errors.append(f"{model}: manifest SHA mismatch")
        if report.get("observations") != 256:
            errors.append(f"{model}: expected 256 observations")
        comparison = report.get("comparison") or {}
        if report.get("complete") is not True:
            errors.append(f"{model}: report incomplete")
        if comparison.get("bitwise_equal_observations") != 256:
            errors.append(f"{model}: not all observations are equal")
        if comparison.get("mismatched_observations") != 0 or comparison.get("max_abs") != 0.0:
            errors.append(f"{model}: nonzero action difference")
    payload = {
        "schema_version": 1,
        "kind": "gdsq_vla_selector_equivalence_gate",
        "complete": not errors,
        "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "reports": {
            model: {"path": str(path), "sha256": sha256_file(path), "result": reports[model]}
            for model, path in (("gr00t", gr_path), ("pi05", pi_path))
        },
        "reuse_decisions": {
            "gr00t_static_official50": not errors and reports["gr00t"].get("complete") is True,
            "pi05_static_ohb_results": False,
        },
        "pi05_formal_selector_rollout_required": True,
        "errors": errors,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": payload["complete"], "errors": errors, "out": str(out)}, indent=2))
    raise SystemExit(0 if payload["complete"] else 1)


if __name__ == "__main__":
    main()
