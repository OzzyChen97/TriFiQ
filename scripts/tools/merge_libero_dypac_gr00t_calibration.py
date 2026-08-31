#!/usr/bin/env python3
"""Validate and merge the four suite-specific GR00T LIBERO DyPAC shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


SUITES = tuple(PROTOCOL["benchmark"]["suites"])
ARRAY_KEYS = (
    "images",
    "wrist_images",
    "states",
    "prompts",
    "task_ids",
    "suite_ids",
    "task_indices",
    "env_seeds",
    "replan_indices",
    "selection_rows",
    "action_noises",
    "action_noises_b",
    "action_seeds",
    "action_seeds_b",
    "teacher_actions",
)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def validate_sequences(arrays: dict[str, np.ndarray], *, selection_only: bool) -> None:
    expected = tuple(PROTOCOL["benchmark"]["retained_replan_indices"])
    groups: dict[tuple[str, int, int], list[int]] = {}
    for index in range(len(arrays["states"])):
        if selection_only and not bool(arrays["selection_rows"][index]):
            continue
        key = (
            str(arrays["suite_ids"][index]),
            int(arrays["task_indices"][index]),
            int(arrays["env_seeds"][index]),
        )
        groups.setdefault(key, []).append(int(arrays["replan_indices"][index]))
    for key, replans in groups.items():
        if tuple(replans) != expected:
            raise ValueError(f"incomplete ordered GR00T replan sequence {key}: {replans}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-out", required=True)
    args = parser.parse_args()
    shard_dir = Path(args.shard_dir).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    selection_output = Path(args.selection_out).expanduser().resolve()
    if output.exists() or selection_output.exists():
        raise FileExistsError("refusing to overwrite merged GR00T DyPAC artifacts")

    banks = {key: [] for key in ARRAY_KEYS}
    shards = []
    checkpoint_paths = {}
    expected_protocol_sha = sha256_file(PROTOCOL_PATH)
    for suite in SUITES:
        path = shard_dir / f"{suite}.npz"
        sidecar = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
        if sidecar.get("model") != "gr00t" or sidecar.get("protocol_sha256") != expected_protocol_sha:
            raise ValueError(f"{suite}: GR00T protocol drift")
        if sidecar.get("sha256") != sha256_file(path):
            raise ValueError(f"{suite}: shard hash mismatch")
        if sidecar.get("rows") != 64 or sidecar.get("selection_rows") != 36:
            raise ValueError(f"{suite}: unexpected shard coverage")
        runtime = sidecar["source_server_metadata"]
        checkpoint_paths[suite] = runtime.get("model_path")
        with np.load(path, allow_pickle=False) as archive:
            missing = set(ARRAY_KEYS) - set(archive.files)
            if missing:
                raise ValueError(f"{suite}: missing arrays {sorted(missing)}")
            for key in ARRAY_KEYS:
                value = np.asarray(archive[key])
                if len(value) != 64:
                    raise ValueError(f"{suite}/{key}: {len(value)} != 64")
                banks[key].append(value)
        shards.append({"suite": suite, "path": str(path), "sha256": sidecar["sha256"]})

    arrays = {key: np.concatenate(values, axis=0) for key, values in banks.items()}
    if len(arrays["states"]) != 256 or int(arrays["selection_rows"].sum()) != 144:
        raise RuntimeError("merged GR00T calibration coverage drift")
    if arrays["action_noises"].shape[1:] != (16, 32):
        raise RuntimeError("merged GR00T native noise shape drift")
    if set(map(str, arrays["suite_ids"])) != set(SUITES):
        raise RuntimeError("merged GR00T calibration misses a LIBERO suite")
    if len(set(map(str, arrays["task_ids"]))) != 40:
        raise RuntimeError("merged GR00T calibration misses LIBERO tasks")
    if set(map(int, arrays["env_seeds"])) & set(PROTOCOL["benchmark"]["formal_initial_state_indices"]):
        raise RuntimeError("GR00T calibration overlaps formal initial states")
    validate_sequences(arrays, selection_only=False)
    validate_sequences(arrays, selection_only=True)

    atomic_npz(output, arrays)
    mask = arrays["selection_rows"].astype(bool)
    selection = {key: value[mask] for key, value in arrays.items()}
    atomic_npz(selection_output, selection)
    payload = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_fp16_onpolicy_merged",
        "model": "gr00t",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": expected_protocol_sha,
        "rows": 256,
        "selection_rows": 144,
        "tasks": 40,
        "suites": list(SUITES),
        "suite_specific_checkpoints": checkpoint_paths,
        "initial_state_indices": [0, 1, 2],
        "formal_initial_state_indices": PROTOCOL["benchmark"]["formal_initial_state_indices"],
        "overlap_with_formal": False,
        "uses_success_labels": False,
        "uses_test_rollout_feedback": False,
        "shards": shards,
        "npz": str(output),
        "sha256": sha256_file(output),
        "selection_npz": str(selection_output),
        "selection_sha256": sha256_file(selection_output),
    }
    atomic_json(Path(str(output) + ".json"), payload)
    atomic_json(
        Path(str(selection_output) + ".json"),
        {
            **payload,
            "kind": "dypac_libero_gr00t_selection_buffer",
            "rows": 144,
            "npz": str(selection_output),
            "sha256": payload["selection_sha256"],
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
