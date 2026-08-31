#!/usr/bin/env python3
"""Fail-closed aggregation and registry update for the 4,000-episode Table 6."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs/table6_libero_v1"
RESULTS = RUN_ROOT / "results"
DYPAC_RESULTS = ROOT / "runs/libero_dypac_v1/results"
DYPAC_GR00T_RESULTS = ROOT / "runs/libero_dypac_v1/results_v2"
OUTPUT = RUN_ROOT / "aggregate/summary.json"
REGISTRY = ROOT / "docs/gdsq_vla_iclr2027/experiment_registry.json"
MODELS = ("gr00t", "pi05")
CONFIGS = (
    "fp16",
    "quantvla_w4a8",
    "uniform_w6",
    "omega_qvla_w4a4",
    "gdsq_vla_selector",
)
SUITES = ("goal", "spatial", "object", "long")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def result_path(model: str, config: str, suite: str) -> Path:
    # The paper-method row comes from the corrected benchmark-specific DyPAC
    # evaluations.  Do not reuse the invalidated selector staging rollouts in
    # the Table-6 result root.
    if config == "gdsq_vla_selector":
        root = DYPAC_GR00T_RESULTS if model == "gr00t" else DYPAC_RESULTS
        return root / model / "dypac_vla" / suite / "merged_summary.json"
    return RESULTS / model / config / suite / "merged_summary.json"


def load_cell(model: str, config: str, suite: str) -> tuple[dict, list[dict], Path]:
    path = result_path(model, config, suite)
    require(path.is_file(), f"missing Table 6 cell: {model}/{config}/{suite}")
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("episode_summaries") or []
    keys = [(int(row["task_id"]), int(row["initial_state_index"])) for row in rows]
    expected = {(task, trial) for task in range(10) for trial in range(10, 20)}
    counts = Counter(keys)
    require(len(rows) == 100, f"{model}/{config}/{suite}: {len(rows)} != 100")
    require(set(keys) == expected, f"{model}/{config}/{suite}: episode-key coverage drift")
    require(not [key for key, count in counts.items() if count != 1], f"{model}/{config}/{suite}: duplicates")
    require(int(value.get("total_episodes", -1)) == 100, f"{model}/{config}/{suite}: summary count drift")
    if config == "gdsq_vla_selector" and model == "gr00t":
        require(value.get("policy_backend") == "groot_zmq", f"{model}/{config}/{suite}: backend drift")
        require(int(value.get("executed_replan_steps", -1)) == 5, f"{model}/{config}/{suite}: replan drift")
        require(value.get("paired_action_noise") is True, f"{model}/{config}/{suite}: paired-noise drift")
        require(value.get("action_noise_suite_key") == suite, f"{model}/{config}/{suite}: noise namespace drift")
    return value, rows, path


def main() -> None:
    models: dict[str, dict] = {}
    all_keys: set[tuple[str, str, str, int, int]] = set()
    invalid_cells: list[str] = []
    cell_artifacts = []
    for model in MODELS:
        config_rows = {}
        reference_keys: dict[str, set[tuple[int, int]]] = {}
        for config in CONFIGS:
            suite_metrics = {}
            successes_total = 0
            for suite in SUITES:
                value, rows, path = load_cell(model, config, suite)
                keys = {(int(row["task_id"]), int(row["initial_state_index"])) for row in rows}
                if suite in reference_keys:
                    require(keys == reference_keys[suite], f"shared-state drift: {model}/{config}/{suite}")
                else:
                    reference_keys[suite] = keys
                successes = sum(bool(row["success"]) for row in rows)
                successes_total += successes
                suite_metrics[suite] = 100.0 * successes / 100.0
                all_keys.update((model, config, suite, *key) for key in keys)
                cell_artifacts.append(
                    {
                        "model": model,
                        "config": config,
                        "suite": suite,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path),
                        "episodes": len(rows),
                        "successes": successes,
                    }
                )
            suite_metrics["average"] = sum(suite_metrics.values()) / len(SUITES)
            config_rows[config] = {
                "episodes": 400,
                "successes": successes_total,
                "metrics": suite_metrics,
            }
        models[model] = {"configs": config_rows}

    require(len(all_keys) == 4000, f"joint coverage drift: {len(all_keys)} != 4000")
    payload = {
        "schema_version": 1,
        "kind": "table6_libero_five_configuration_summary",
        "complete": True,
        "formal_result": True,
        "protocol": "runs/gdsq_extension_preregistered_v1/libero_table2_five_config_v2.json",
        "protocol_sha256": sha256_file(
            ROOT / "runs/gdsq_extension_preregistered_v1/libero_table2_five_config_v2.json"
        ),
        "coverage": {
            "expected_episodes": 4000,
            "observed_episodes": 4000,
            "missing_episodes": 0,
            "duplicate_episodes": 0,
            "invalid_cells": invalid_cells,
            "missing_cells": [],
        },
        "models": models,
        "cell_artifacts": cell_artifacts,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(OUTPUT) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiment = registry["experiments"]["libero_table2_five_config_local"]
    experiment.update(
        {
            "status": "complete",
            "main_claim_enabled": True,
            "coverage": payload["coverage"],
            "summary": {
                "path": str(OUTPUT.relative_to(ROOT)),
                "sha256": sha256_file(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
            },
            "notes": (
                "Complete locally audited five-configuration LIBERO comparison with "
                "exact 4,000-episode coverage and no missing or duplicate episode keys."
            ),
        }
    )
    registry_tmp = Path(str(REGISTRY) + ".tmp")
    registry_tmp.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry_tmp.replace(REGISTRY)
    print(json.dumps({"summary": str(OUTPUT), "sha256": sha256_file(OUTPUT), "episodes": 4000}, indent=2))


if __name__ == "__main__":
    main()
