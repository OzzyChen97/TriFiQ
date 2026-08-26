#!/usr/bin/env python3
"""Audit whether artifacts/results satisfy the GR00T/pi0.5 adapter-only rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import (
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_artifact,
    require_protocol_attestation,
    validate_quant_plan,
    validate_closed_loop_row,
)
from quantvla_model_adapters import validate_calibration_artifact


REPO = Path(__file__).resolve().parents[2]


def check(name: str, function, failures: list[dict[str, str]]) -> None:
    try:
        function()
    except Exception as error:  # audit records every independent failure
        failures.append({"check": name, "error": str(error)})


def first_jsonl_row(paths: list[Path]) -> tuple[Path, int, dict[str, Any]] | None:
    for path in paths:
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                return path, line_number, json.loads(line)
    return None


def validate_all_rows(model: str, paths: list[Path]) -> None:
    violations: list[str] = []
    by_config: dict[str, list[tuple[str, int]]] = {}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                validate_closed_loop_row(row, source=f"{path}:{line_number}")
            except Exception as error:
                violations.append(str(error))
            config = str(row.get("config") or row.get("config_id") or "missing")
            by_config.setdefault(config, []).append(
                (str(row.get("task")), int(row.get("seed", -1)))
            )
    if violations:
        raise ValueError(
            f"{len(violations)} row protocol violation(s); examples={violations[:3]}"
        )
    expected = {
        (task, int(seed))
        for tasks in PROTOCOL["closed_loop"]["tasks"].values()
        for task in tasks
        for seed in PROTOCOL["closed_loop"]["seeds"]
    }
    for config, keys in by_config.items():
        actual = set(keys)
        duplicates = len(keys) - len(actual)
        if actual != expected or duplicates:
            raise ValueError(
                f"{model}/{config}: matrix mismatch rows={len(keys)} unique={len(actual)} "
                f"missing={len(expected - actual)} extra={len(actual - expected)} "
                f"duplicates={duplicates}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", default=str(REPO / "runs/dpac_softfold_15x20_v1")
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).expanduser().resolve()
    failures: list[dict[str, str]] = []

    check("selection_buffer", lambda: protocol_artifact("selection_buffer"), failures)
    check("calibration_buffer", lambda: protocol_artifact("calibration_buffer"), failures)
    calibration_hash = PROTOCOL["data"]["calibration_buffer"]["sha256"]
    plan_paths = {
        "gr00t": [
            REPO / "checkpoints/packs/robocasa365/quantvla_v1_uniform_w4a8.json",
            REPO / "checkpoints/packs/robocasa365/quantvla_v1_uniform_w4a8_composite_seen.json",
            REPO / "checkpoints/packs/robocasa365/quantvla_v1_uniform_w4a8_composite_unseen.json",
        ],
        "pi05": [
            REPO / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
        ],
    }
    for model, paths in plan_paths.items():
        for index, path in enumerate(paths):
            check(
                f"{model}_uniform_w4_plan_{index}",
                lambda value=path, adapter=model: validate_quant_plan(
                    json.loads(value.read_text(encoding="utf-8")),
                    model=adapter,
                    source=str(value),
                ),
                failures,
            )
    gr00t_a8 = [
        root / f"calibration/gr00t/{task_set}/shared_protocol_artifacts/a8_shared_n256.npz"
        for task_set in ("atomic_seen", "composite_seen", "composite_unseen")
    ]
    pi05_a8 = REPO / (
        "runs/pi05_gdsq_gr00t_aligned/a8/"
        "pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz"
    )
    for index, path in enumerate(gr00t_a8):
        check(
            f"gr00t_a8_shared_buffer_{index}",
            lambda value=path: validate_calibration_artifact(
                value, model="gr00t", expected_buffer_sha256=calibration_hash
            ),
            failures,
        )
    check(
        "pi05_a8_shared_buffer",
        lambda: validate_calibration_artifact(
            pi05_a8, model="pi05", expected_buffer_sha256=calibration_hash
        ),
        failures,
    )

    score_files = sorted((root / "calibration").glob("**/scores_merged.json"))
    if not score_files:
        failures.append({"check": "score_artifacts", "error": "no merged scores found"})
    for path in score_files:
        check(
            f"score_protocol:{path.relative_to(root)}",
            lambda value=path: require_protocol_attestation(
                json.loads(value.read_text(encoding="utf-8")), source=str(value)
            ),
            failures,
        )
        check(
            f"score_selection:{path.relative_to(root)}",
            lambda value=path: (
                (_ for _ in ()).throw(
                    ValueError("missing shared quantization-selection attestation")
                )
                if not json.loads(value.read_text(encoding="utf-8")).get(
                    "quantization_selection"
                )
                else None
            ),
            failures,
        )

    model_patterns = {
        "gr00t": list((root / "closed_loop/gr00t").glob("**/softfold_*.jsonl")),
        "pi05": list((root / "closed_loop/pi05/results").glob("**/*.jsonl")),
    }
    for model, paths in model_patterns.items():
        if not paths:
            failures.append({"check": f"{model}_closed_loop", "error": "no row found"})
            continue
        check(
            f"{model}_closed_loop_protocol",
            lambda adapter=model, values=paths: validate_all_rows(adapter, values),
            failures,
        )

    payload = {
        "schema_version": 1,
        "kind": "quantvla_adapter_only_audit",
        "run_root": str(root),
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "compliant": not failures,
        "failures": failures,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        output = Path(args.out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
