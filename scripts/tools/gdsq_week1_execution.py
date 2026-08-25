#!/usr/bin/env python3
"""Materialize and audit the executable GR00T week-1 experiment chain.

The preregistration owns allocation masks.  This tool only derives
checkpoint-specific deployment plans/specs, records their hashes, and (after
development runs finish) freezes the random/action-only representatives.  It
never reads held-out results when choosing a representative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "runs/gdsq_week1_preregistered_v1"
PREREG = ROOT / "preregistration.json"
PREREG_SHA256 = "216f1b6267b5bc9ff67cb19f9e3502c30836b0aa7e81e10ade522fa7d6104541"
EXECUTION = ROOT / "execution"
CHECKPOINT_ROOT = REPO_ROOT / (
    "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining"
)
PACK_ROOT = REPO_ROOT / "checkpoints/packs/robocasa365"
DEV_TASKS = [
    "CoffeeSetupMug",
    "OpenCabinet",
    "OpenStandMixerHead",
    "PickPlaceDrawerToCounter",
]
ATOMIC_TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]
PRIMARY14 = [task for task in ATOMIC_TASKS if task not in DEV_TASKS]
TASK_SETS = ("atomic_seen", "composite_seen", "composite_unseen")
ALL_GPUS = tuple(range(8))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == rendered, f"frozen artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def pack_for(task_set: str) -> Path:
    suffix = {
        "atomic_seen": "protocolfix",
        "composite_seen": "composite_seen",
        "composite_unseen": "composite_unseen",
    }[task_set]
    return PACK_ROOT / f"duquant_packed_robocasa365_{suffix}_d4_w4a8_b64c32ls015"


def checkpoint_for(task_set: str) -> Path:
    return CHECKPOINT_ROOT / task_set / "checkpoint-60000"


def selected_count(plan: dict[str, Any]) -> int:
    return sum(
        not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
        and int(row.get("bits", 0) or 0) > 0
        for row in (plan.get("layers") or {}).values()
    )


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def server_placement(start_port: int) -> dict[str, Any]:
    return {
        "gpu": 0,
        "port": start_port,
        "replicas": [
            {"gpu": gpu, "port": start_port + gpu} for gpu in ALL_GPUS[1:]
        ],
    }


def uniform_plan(prereg: dict[str, Any], task_set: str) -> tuple[Path, dict[str, Any]]:
    source_record = prereg["models"]["gr00t"]["plans"]["uniform_w6"]
    source_path = Path(source_record["path"])
    require(sha256_file(source_path) == source_record["sha256"], "uniform-W6 plan drift")
    payload = read_json(source_path)
    pack = pack_for(task_set).resolve()
    require(pack.is_dir(), f"missing {task_set} pack: {pack}")
    payload["packdirs"] = {"64": str(pack)}
    payload.setdefault("meta", {}).update(
        {
            "deployment_task_set": task_set,
            "allocation_source_path": str(source_path.resolve()),
            "allocation_source_sha256": source_record["sha256"],
            "checkpoint_specific_pack_only": True,
            "heldout_results_used": False,
        }
    )
    output = EXECUTION / "plans/gr00t/uniform_w6" / f"{task_set}.plan.json"
    frozen_json(output, payload)
    return output, payload


def config_row(
    *,
    config_id: str,
    plan: Path,
    a8: Path,
    expected_wrapped: int,
    gpu: int,
    port: int,
    replicas: list[dict[str, int]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": config_id,
        "gpu": gpu,
        "port": port,
        "plan": str(plan.resolve()),
        "act_scale": str(a8.resolve()),
        "expected_wrapped": expected_wrapped,
        "meta": meta or {},
    }
    if replicas:
        row["replicas"] = replicas
    return row


def materialize() -> dict[str, Any]:
    require(sha256_file(PREREG) == PREREG_SHA256, "week-1 preregistration drift")
    prereg = read_json(PREREG)
    specs: dict[str, dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []

    for offset, task_set in enumerate(TASK_SETS):
        plan_path, plan = uniform_plan(prereg, task_set)
        a8 = EXECUTION / "a8/gr00t" / f"uniform_w6_{task_set}.npz"
        placement = server_placement(19800 + offset * 10)
        spec = {
            "purpose": f"Preregistered GR00T uniform-W6 control on {task_set}",
            "decision": {
                "final_config": "uniform_w6",
                "selection": "none",
                "same_budget_reference": "uniform_w6",
            },
            "comparisons": [],
            "configs": [
                config_row(
                    config_id="uniform_w6",
                    plan=plan_path,
                    a8=a8,
                    expected_wrapped=selected_count(plan),
                    gpu=placement["gpu"],
                    port=placement["port"],
                    replicas=placement["replicas"],
                    meta={
                        "weight_bits": 6,
                        "activation_bits": 8,
                        "allocation_source_sha256": prereg["models"]["gr00t"]["plans"]["uniform_w6"]["sha256"],
                    },
                )
            ],
        }
        spec_path = EXECUTION / "specs" / f"gr00t_uniform_w6_{task_set}_8gpu_v1.json"
        frozen_json(spec_path, spec)
        specs[f"uniform_w6_{task_set}"] = artifact(spec_path)
        commands.append(
            {
                "id": f"gr00t_uniform_w6_{task_set}",
                "priority": "P0",
                "scope": f"{task_set} x 50 seeds",
                "status": "waiting_for_a8_and_gpu",
                "spec": str(spec_path.resolve()),
                "a8": str(a8.resolve()),
            }
        )

    ablation_records = prereg["models"]["gr00t"]["plans"]["ablations"]
    ablation_a8 = {
        "cka_only": PACK_ROOT / "a8_scales_ckaonly_protocolfix_d4.npz",
        "cs_only": PACK_ROOT / "a8_scales_csonly_protocolfix_d4.npz",
        "weights_uniform": EXECUTION / "a8/gr00t/ablation_weights_uniform.npz",
        "no_guards": EXECUTION / "a8/gr00t/ablation_no_guards.npz",
        "no_functional_adjudication": EXECUTION
        / "a8/gr00t/ablation_no_functional_adjudication.npz",
    }
    configs = []
    for index, name in enumerate(
        ("cka_only", "cs_only", "weights_uniform", "no_guards", "no_functional_adjudication")
    ):
        record = ablation_records[name]
        plan_path = Path(record["path"])
        require(sha256_file(plan_path) == record["sha256"], f"{name} plan drift")
        plan = read_json(plan_path)
        replicas = (
            [{"gpu": 5 + index, "port": 19905 + index}]
            if index < 3
            else None
        )
        configs.append(
            config_row(
                config_id=f"ablation_{name}",
                plan=plan_path,
                a8=ablation_a8[name],
                expected_wrapped=selected_count(plan),
                gpu=index,
                port=19900 + index,
                replicas=replicas,
                meta={
                    "ablation": name,
                    "plan_sha256": record["sha256"],
                    "development_tasks_only_for_selection": DEV_TASKS,
                    "formal_failure_on_crash": name == "no_guards",
                },
            )
        )
    ablation_spec = {
        "purpose": "Preregistered GR00T component ablations on Primary14",
        "decision": {
            "final_config": "none",
            "selection_tasks_excluded": DEV_TASKS,
            "formal_tasks": PRIMARY14,
        },
        "comparisons": [],
        "configs": configs,
    }
    # v2 makes the preregistered no-guards crash policy executable.  Keep the
    # earlier generated draft immutable rather than silently replacing it.
    ablation_spec_path = EXECUTION / "specs/gr00t_core_ablations_primary14_v3_8gpu.json"
    frozen_json(ablation_spec_path, ablation_spec)
    specs["core_ablations_primary14"] = artifact(ablation_spec_path)
    commands.append(
        {
            "id": "gr00t_core_ablations_primary14",
            "priority": "P1",
            "scope": "Primary14 x 50 seeds",
            "status": "waiting_for_a8_and_gpu",
            "spec": str(ablation_spec_path.resolve()),
            "formal_failure_policy": "NaN/crash/out-of-range counts as failure",
        }
    )

    for family in ("search_matched_random", "action_only"):
        record = prereg["models"]["gr00t"]["plans"][family]
        commands.append(
            {
                "id": f"gr00t_{family}",
                "priority": "P0",
                "scope": "functional screen -> dev4 x 50 -> Primary14 x 50",
                "status": "waiting_for_functional_screen_and_gpu",
                "candidate_count": record["candidate_count"],
                "functional_candidate_indices": record["functional_candidate_indices"],
                "functional_discrimination_count": record["functional_discrimination_count"],
                "development_representatives": record[
                    "development_representatives_after_functional_ranking"
                ],
                "heldout_feedback_forbidden": True,
            }
        )

    ledger = {
        "schema_version": 1,
        "kind": "gdsq_vla_week1_execution_ledger",
        "result_blind": True,
        "preregistration": artifact(PREREG),
        "gpu_policy": {
            "authorized_gpus": list(ALL_GPUS),
            "server_replication": "use every GPU for pending formal matrices when configuration count permits",
            "launch_after": "pi05 selector strict completion and explicit eight-GPU authorization",
        },
        "development_tasks": DEV_TASKS,
        "primary14": PRIMARY14,
        "specs": specs,
        "jobs": commands,
        "entrypoint": "scripts/run_gr00t_week1.sh",
    }
    ledger_path = EXECUTION / "ledger_v3_8gpu.json"
    frozen_json(ledger_path, ledger)
    return {"ledger": str(ledger_path), "ledger_sha256": sha256_file(ledger_path), "jobs": len(commands)}


def collect_dev_rows(run_dirs: list[Path]) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    grouped: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    expected = {(task, seed) for task in DEV_TASKS for seed in range(50)}
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        require(manifest_path.is_file(), f"missing dev manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        require(manifest.get("dev_tasks") == DEV_TASKS, f"non-preregistered dev tasks: {run_dir}")
        require(manifest.get("heldout_tasks") == [], f"held-out tasks entered dev selection: {run_dir}")
        manifest_sha = sha256_file(manifest_path)
        for config in manifest["configs"]:
            config_id = config["id"]
            rows: dict[tuple[str, int], dict[str, Any]] = {}
            for path_text in config["result_files"]:
                path = Path(path_text)
                require(path.is_file(), f"missing dev result: {path}")
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (str(row["task"]), int(row["seed"]))
                    require(row.get("manifest_sha256") == manifest_sha, f"manifest drift in {path}")
                    require(row.get("config") == config_id, f"foreign config in {path}")
                    require(key not in rows, f"duplicate dev key {config_id}/{key}")
                    rows[key] = row
            require(set(rows) == expected, f"incomplete dev coverage for {config_id}")
            require(config_id not in grouped, f"duplicate dev config: {config_id}")
            grouped[config_id] = rows
    return grouped


def materialize_dev_specs() -> dict[str, Any]:
    reports = {
        "random": read_json(EXECUTION / "screen/gr00t_search_matched_random.json"),
        "action_only": read_json(EXECUTION / "screen/gr00t_action_only.json"),
    }
    rows: list[dict[str, Any]] = []
    for family, report in reports.items():
        require(report.get("complete") is True, f"incomplete {family} screen")
        require(report.get("heldout_results_read") is False, f"tainted {family} screen")
        require(
            len(report["development_representatives"]) == 7,
            f"wrong representative count for {family}",
        )
        for candidate in report["development_representatives"]:
            index = int(candidate["candidate_index"])
            rows.append(
                config_row(
                    config_id=f"{family}_{index:02d}",
                    plan=Path(candidate["plan_path"]),
                    a8=Path(candidate["a8_path"]),
                    expected_wrapped=int(candidate["wrapped_layers"]),
                    gpu=0,
                    port=0,
                    meta={
                        "family": family,
                        "candidate_index": index,
                        "functional_d_func": candidate["d_func"],
                        "functional_screen_only": True,
                        "heldout_feedback_used": False,
                    },
                )
            )
    specs = []
    for wave, start in enumerate(range(0, len(rows), 5), 1):
        configs = rows[start : start + 5]
        for index, row in enumerate(configs):
            row["gpu"] = index
            row["port"] = 20000 + (wave - 1) * 10 + index
            replica_gpu = len(configs) + index
            if replica_gpu < len(ALL_GPUS):
                row["replicas"] = [
                    {
                        "gpu": replica_gpu,
                        "port": 20000 + (wave - 1) * 10 + replica_gpu,
                    }
                ]
        spec = {
            "purpose": f"GR00T same-budget control development selection wave {wave}",
            "decision": {
                "selection_tasks": DEV_TASKS,
                "heldout_tasks_used": False,
                "representative_rule": "max task-macro SR; D_func then index tie-break",
            },
            "comparisons": [],
            "configs": configs,
        }
        path = EXECUTION / "specs" / f"gr00t_controls_dev4_wave{wave}.json"
        frozen_json(path, spec)
        specs.append(artifact(path))
    manifest = {
        "schema_version": 1,
        "kind": "gdsq_vla_gr00t_control_dev_specs",
        "result_blind": True,
        "development_tasks": DEV_TASKS,
        "heldout_tasks": PRIMARY14,
        "screen_reports": {
            family: artifact(EXECUTION / "screen" / (
                "gr00t_search_matched_random.json"
                if family == "random"
                else "gr00t_action_only.json"
            ))
            for family in reports
        },
        "waves": specs,
        "configurations": len(rows),
    }
    output = EXECUTION / "selection/gr00t_control_dev_specs.json"
    frozen_json(output, manifest)
    return {"manifest": str(output), "sha256": sha256_file(output), "waves": len(specs)}


def freeze_dev_selection() -> dict[str, Any]:
    screen_reports = {
        "random": EXECUTION / "screen/gr00t_search_matched_random.json",
        "action_only": EXECUTION / "screen/gr00t_action_only.json",
    }
    run_dirs = sorted((EXECUTION / "runs").glob("gr00t_controls_dev4_wave*"))
    require(run_dirs, "no completed development waves")
    rows = collect_dev_rows(run_dirs)
    selected: dict[str, Any] = {}
    for family, report_path in screen_reports.items():
        report = read_json(report_path)
        candidates = report["development_representatives"]
        scored = []
        for candidate in candidates:
            config_id = f"{family}_{int(candidate['candidate_index']):02d}"
            candidate_rows = rows[config_id]
            per_task = {
                task: sum(bool(candidate_rows[(task, seed)]["success"]) for seed in range(50)) / 50
                for task in DEV_TASKS
            }
            scored.append(
                {
                    **candidate,
                    "config_id": config_id,
                    "dev_task_macro_sr": sum(per_task.values()) / len(per_task),
                    "per_task_sr": per_task,
                }
            )
        winner = min(
            scored,
            key=lambda row: (
                -row["dev_task_macro_sr"],
                row["d_func"],
                row["candidate_index"],
            ),
        )
        selected[family] = {
            "selection_rule": "max dev4 task-macro SR; ties: lower D_func then candidate index",
            "candidates": scored,
            "winner": winner,
        }
    payload = {
        "schema_version": 1,
        "kind": "gdsq_vla_gr00t_same_budget_dev_selection",
        "valid": True,
        "development_tasks": DEV_TASKS,
        "heldout_results_read": False,
        "formal_scope": PRIMARY14,
        "screen_reports": {name: artifact(path) for name, path in screen_reports.items()},
        "development_run_manifests": [artifact(path / "manifest.json") for path in run_dirs],
        "selected": selected,
    }
    output = EXECUTION / "selection/gr00t_controls_dev4.json"
    frozen_json(output, payload)
    return {"selection": str(output), "sha256": sha256_file(output)}


def materialize_formal_control_spec() -> dict[str, Any]:
    selection_path = EXECUTION / "selection/gr00t_controls_dev4.json"
    selection = read_json(selection_path)
    require(selection.get("valid") is True, "development selection is invalid")
    require(selection.get("heldout_results_read") is False, "held-out feedback leak")
    configs = []
    placements = {
        "random": {
            "gpu": 0,
            "port": 20100,
            "replicas": [{"gpu": gpu, "port": 20100 + gpu} for gpu in (1, 2, 3)],
        },
        "action_only": {
            "gpu": 4,
            "port": 20104,
            "replicas": [{"gpu": gpu, "port": 20100 + gpu} for gpu in (5, 6, 7)],
        },
    }
    for family in ("random", "action_only"):
        winner = selection["selected"][family]["winner"]
        placement = placements[family]
        configs.append(
            config_row(
                config_id=("search_matched_random" if family == "random" else "action_only"),
                plan=Path(winner["plan_path"]),
                a8=Path(winner["a8_path"]),
                expected_wrapped=int(winner["wrapped_layers"]),
                gpu=placement["gpu"],
                port=placement["port"],
                replicas=placement["replicas"],
                meta={
                    "family": family,
                    "candidate_index": winner["candidate_index"],
                    "development_task_macro_sr": winner["dev_task_macro_sr"],
                    "selection_sha256": sha256_file(selection_path),
                    "heldout_feedback_used": False,
                },
            )
        )
    spec = {
        "purpose": "Preregistered GR00T same-budget controls on Primary14",
        "decision": {
            "selection_artifact": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "selection_tasks": DEV_TASKS,
            "formal_tasks": PRIMARY14,
            "heldout_feedback_used": False,
        },
        "comparisons": [["search_matched_random", "action_only"]],
        "configs": configs,
    }
    output = EXECUTION / "specs/gr00t_same_budget_controls_primary14_v2_8gpu.json"
    frozen_json(output, spec)
    return {"spec": str(output), "sha256": sha256_file(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "materialize",
            "verify",
            "materialize-dev-specs",
            "freeze-dev-selection",
            "materialize-formal-control-spec",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"materialize", "verify"}:
        result = materialize()
    elif args.command == "materialize-dev-specs":
        result = materialize_dev_specs()
    elif args.command == "materialize-formal-control-spec":
        result = materialize_formal_control_spec()
    else:
        result = freeze_dev_selection()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
