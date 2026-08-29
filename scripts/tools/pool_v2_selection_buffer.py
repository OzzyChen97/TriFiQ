#!/usr/bin/env python3
"""Pool per-(task, seed) windowed shards into the v2 selection buffer.

Reads the frozen v2 selection context spec and the shard archives written
by the windowed FP16 on-policy collector, verifies exact coverage (4 tasks
per split, 3 seeds per task, 4 replans per sequence from the assigned
window), and emits the canonical cross-model selection archive plus a
provenance manifest. Any missing shard, window drift, or duplicate row is
a hard error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from quantvla_outputimpact import atomic_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_noise_a_rows(shard: Path) -> tuple[dict[str, np.ndarray], int]:
    """Return the noise-A half of a collector archive (rows 0..n-1)."""
    with np.load(shard, allow_pickle=False) as archive:
        required = {
            "images",
            "wrist_images",
            "right_images",
            "states",
            "prompts",
            "task_ids",
            "env_seeds",
            "env_steps",
            "replan_indices",
            "action_noises",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{shard}: missing fields: {missing}")
        count = len(archive["states"])
        if count % 2:
            raise ValueError(f"{shard}: expected duplicated noise-A/B rows, got {count}")
        half = count // 2
        return {
            name: np.asarray(archive[name])[:half]
            for name in ("images", "wrist_images", "right_images", "states", "prompts",
                         "task_ids", "env_seeds", "env_steps", "replan_indices",
                         "action_noises")
        }, half


def pool(spec: dict[str, Any], shard_dir: Path, out: Path) -> dict[str, Any]:
    shard_dir = shard_dir.expanduser().resolve()
    replans = int(spec["replans_per_sequence"])
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    shards = []
    for entry in spec["entries"]:
        split = str(entry["split"])
        task = str(entry["task"])
        for seed_row in entry["seeds"]:
            seed = int(seed_row["seed"])
            window_start = int(seed_row["window_start"])
            shard = shard_dir / f"{task}_{seed}.npz"
            sidecar = Path(str(shard) + ".json")
            if not shard.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"missing v2 selection shard: {shard}")
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("selection") != "window":
                raise ValueError(f"{sidecar}: shard was not captured with --selection window")
            if int(meta.get("window_start", -1)) != window_start:
                raise ValueError(
                    f"{sidecar}: window_start drift {meta.get('window_start')} != {window_start}"
                )
            if int(meta.get("replans_per_trial", -1)) != replans:
                raise ValueError(f"{sidecar}: replans-per-trial drift")
            arrays, half = load_noise_a_rows(shard)
            if half != replans:
                raise ValueError(f"{sidecar}: expected {replans} noise-A rows, got {half}")
            task_ids = [str(value) for value in arrays["task_ids"]]
            if any(value != task for value in task_ids):
                raise ValueError(f"{sidecar}: task id drift")
            if any(int(value) != seed for value in arrays["env_seeds"]):
                raise ValueError(f"{sidecar}: seed drift")
            replan_indices = [int(value) for value in arrays["replan_indices"]]
            if replan_indices != list(range(window_start, window_start + replans)):
                raise ValueError(
                    f"{sidecar}: replan indices {replan_indices} != window {window_start}"
                )
            for index, replan in enumerate(replan_indices):
                key = (task, seed, replan)
                if key in rows:
                    raise ValueError(f"duplicate v2 selection row: {key}")
                rows[key] = {
                    "split": split,
                    "task": task,
                    "seed": seed,
                    "replan": replan,
                    "image": arrays["images"][index],
                    "wrist_image": arrays["wrist_images"][index],
                    "right_image": arrays["right_images"][index],
                    "state": arrays["states"][index],
                    "prompt": str(arrays["prompts"][index]),
                    "env_step": int(arrays["env_steps"][index]),
                    "noise": arrays["action_noises"][index],
                }
            shards.append(
                {
                    "task": task,
                    "seed": seed,
                    "path": str(shard),
                    "sha256": sha256_file(shard),
                    "sidecar_sha256": sha256_file(sidecar),
                }
            )
    expected = spec["total_sequences"] * replans
    if len(rows) != expected:
        raise ValueError(f"coverage drift: pooled {len(rows)} rows, expected {expected}")
    ordered = sorted(
        rows.values(),
        key=lambda row: (row["split"], row["task"], row["seed"], row["replan"]),
    )
    arrays = {
        "images": np.stack([row["image"] for row in ordered]),
        "wrist_images": np.stack([row["wrist_image"] for row in ordered]),
        "right_images": np.stack([row["right_image"] for row in ordered]),
        "states": np.stack(
            [np.asarray(row["state"], dtype=np.float32) for row in ordered]
        ),
        "prompts": np.asarray([row["prompt"] for row in ordered]),
        "task_ids": np.asarray([row["task"] for row in ordered]),
        "env_seeds": np.asarray([row["seed"] for row in ordered], dtype=np.int64),
        "env_steps": np.asarray([row["env_step"] for row in ordered], dtype=np.int64),
        "replan_indices": np.asarray([row["replan"] for row in ordered], dtype=np.int64),
        "action_noises": np.stack(
            [np.asarray(row["noise"], dtype=np.float32) for row in ordered]
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return {
        "kind": "full_context_v2_selection_buffer_manifest",
        "archive": str(out),
        "archive_sha256": sha256_file(out),
        "observations": len(ordered),
        "sequences": spec["total_sequences"],
        "tasks": spec["total_tasks"],
        "spec_sha256": spec["protocol_sha256"],
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).expanduser().resolve().read_text(encoding="utf-8"))
    if spec.get("kind") != "full_context_v2_selection_context_spec":
        raise ValueError(f"{args.spec}: wrong spec kind")
    manifest = pool(spec, Path(args.shard_dir), Path(args.out).expanduser().resolve())
    manifest_path = Path(str(Path(args.out).expanduser().resolve()) + ".json")
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()