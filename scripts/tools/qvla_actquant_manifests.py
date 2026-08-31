#!/usr/bin/env python3
"""Create and validate frozen calibration manifests for Table 1 reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "scripts" / "qvla_actquant_table1_protocol.json"
TASK_PROTOCOL_PATH = REPO_ROOT / "scripts" / "quantvla_full_context_protocol_v2.json"


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ranked_tasks(tasks: list[str], *, seed: int, tag: str) -> list[str]:
    return sorted(
        tasks,
        key=lambda task: hashlib.sha256(f"{seed}\0{tag}\0{task}".encode()).digest(),
    )


def counts_for(tasks: list[str], total: int, *, seed: int, tag: str) -> dict[str, int]:
    quotient, remainder = divmod(total, len(tasks))
    counts = {task: quotient for task in tasks}
    for task in ranked_tasks(tasks, seed=seed, tag=tag)[:remainder]:
        counts[task] += 1
    assert sum(counts.values()) == total
    return counts


def task_episode_seeds(task: str, count: int, *, selection_seed: int, start: int) -> list[int]:
    candidates = list(range(start, start + 10_000))
    digest = hashlib.sha256(f"{selection_seed}\0episode\0{task}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    generator.shuffle(candidates)
    return sorted(candidates[:count])


def make_unit_manifest(
    *,
    model: str,
    unit: str,
    tasks: list[str],
    checkpoint: str,
    selection_seed: int,
    teacher_seed_start: int,
    qvla_total: int,
    actquant_total: int,
    protocol: dict,
) -> dict:
    q_counts = counts_for(tasks, qvla_total, seed=selection_seed, tag=f"{model}/{unit}/qvla")
    a_counts = counts_for(tasks, actquant_total, seed=selection_seed, tag=f"{model}/{unit}/actquant")
    qvla_episodes = []
    actquant_keys = []
    for task in tasks:
        seeds = task_episode_seeds(
            task,
            q_counts[task],
            selection_seed=selection_seed,
            start=teacher_seed_start,
        )
        actquant_seed_order = sorted(
            seeds,
            key=lambda value: hashlib.sha256(
                f"{selection_seed}\0actquant-subset\0{model}\0{unit}\0{task}\0{value}".encode()
            ).digest(),
        )
        actquant_selected = set(actquant_seed_order[: a_counts[task]])
        for env_seed in seeds:
            key = f"{task}/{env_seed}"
            qvla_episodes.append(
                {
                    "episode_key": key,
                    "task": task,
                    "env_seed": env_seed,
                    "split": "target",
                    "source": protocol["calibration"]["source"],
                    "status": "planned",
                    "actquant_subset": env_seed in actquant_selected,
                }
            )
            if env_seed in actquant_selected:
                actquant_keys.append(key)
    qvla_episodes.sort(key=lambda row: (row["task"], row["env_seed"]))
    actquant_keys.sort()
    checkpoint_path = (REPO_ROOT / checkpoint).resolve()
    result = {
        "schema_version": 1,
        "kind": "qvla_actquant_calibration_manifest",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": canonical_hash(protocol),
        "model": model,
        "model_unit": unit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_files": sorted(
            str(path.relative_to(REPO_ROOT))
            for path in ([checkpoint_path] if checkpoint_path.is_file() else checkpoint_path.glob("**/*"))
            if path.is_file() and path.suffix in (".json", ".safetensors")
        ),
        "calibration_source": protocol["calibration"]["source"],
        "source_protocol_equivalent": False,
        "selection_seed": selection_seed,
        "formal_seeds": list(range(50)),
        "formal_seed_overlap_forbidden": True,
        "frame_policy": protocol["calibration"]["frame_policy"],
        "qvla_episode_count": len(qvla_episodes),
        "actquant_episode_count": len(actquant_keys),
        "qvla_task_counts": q_counts,
        "actquant_task_counts": a_counts,
        "qvla_episodes": qvla_episodes,
        "actquant_episode_keys": actquant_keys,
        "test_results_used": False,
    }
    result["episode_inventory_sha256"] = canonical_hash(qvla_episodes)
    result["actquant_subset_sha256"] = canonical_hash(actquant_keys)
    return result


def make_all(output: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if sha256_file(TASK_PROTOCOL_PATH) != protocol["benchmark"]["task_source_sha256"]:
        raise ValueError("Table 1 task protocol SHA drift")
    table = json.loads(TASK_PROTOCOL_PATH.read_text(encoding="utf-8"))["table1"]
    task_sets = table["tasks"]
    if sum(map(len, task_sets.values())) != 50:
        raise ValueError("Table 1 must contain exactly 50 tasks")
    selection_seed = int(protocol["calibration"]["selection_seed"])
    teacher_seed_start = int(protocol["calibration"]["teacher_seed_start"])
    qvla_total = int(protocol["calibration"]["qvla_episodes_per_model_unit"])
    actquant_total = int(protocol["calibration"]["actquant_episodes_per_model_unit"])
    paths = []
    for unit, checkpoint in protocol["models"]["gr00t"]["units"].items():
        manifest = make_unit_manifest(
            model="gr00t",
            unit=unit,
            tasks=list(task_sets[unit]),
            checkpoint=checkpoint,
            selection_seed=selection_seed,
            teacher_seed_start=teacher_seed_start,
            qvla_total=qvla_total,
            actquant_total=actquant_total,
            protocol=protocol,
        )
        path = output / f"gr00t_{unit}.json"
        atomic_json(path, manifest)
        paths.append(path)
    pi05_checkpoint = protocol["models"]["pi05"]["checkpoint"]
    all_tasks = [task for split in ("atomic_seen", "composite_seen", "composite_unseen") for task in task_sets[split]]
    manifest = make_unit_manifest(
        model="pi05",
        unit="all_target",
        tasks=all_tasks,
        checkpoint=pi05_checkpoint,
        selection_seed=selection_seed,
        teacher_seed_start=teacher_seed_start,
        qvla_total=qvla_total,
        actquant_total=actquant_total,
        protocol=protocol,
    )
    path = output / "pi05_all_target.json"
    atomic_json(path, manifest)
    paths.append(path)
    index = {
        "schema_version": 1,
        "kind": "qvla_actquant_calibration_index",
        "protocol": str(PROTOCOL_PATH),
        "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
        "manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "total_teacher_episodes": sum(
            json.loads(path.read_text(encoding="utf-8"))["qvla_episode_count"] for path in paths
        ),
        "total_actquant_episodes": sum(
            json.loads(path.read_text(encoding="utf-8"))["actquant_episode_count"] for path in paths
        ),
    }
    atomic_json(output / "index.json", index)
    print(json.dumps(index, indent=2))


def validate_manifest(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    qvla = value["qvla_episodes"]
    qkeys = [row["episode_key"] for row in qvla]
    actquant = value["actquant_episode_keys"]
    if len(qvla) != 512 or len(qkeys) != len(set(qkeys)):
        raise ValueError("QVLA manifest must contain 512 unique episodes")
    if len(actquant) != 60 or len(actquant) != len(set(actquant)):
        raise ValueError("ActQuant manifest must contain 60 unique episodes")
    if not set(actquant) <= set(qkeys):
        raise ValueError("ActQuant episodes are not a QVLA subset")
    if any(int(row["env_seed"]) in range(50) for row in qvla):
        raise ValueError("formal seed leaked into calibration")
    if canonical_hash(qvla) != value["episode_inventory_sha256"]:
        raise ValueError("episode inventory SHA mismatch")
    if canonical_hash(actquant) != value["actquant_subset_sha256"]:
        raise ValueError("ActQuant subset SHA mismatch")
    print(json.dumps({"path": str(path), "qvla": len(qvla), "actquant": len(actquant), "status": "valid"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("manifests", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "make":
        make_all(args.output.resolve())
    else:
        for path in args.manifests:
            validate_manifest(path.resolve())


if __name__ == "__main__":
    main()
