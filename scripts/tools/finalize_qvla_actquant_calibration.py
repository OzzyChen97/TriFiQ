#!/usr/bin/env python3
"""Validate teacher-proxy episodes and freeze the frame-level calibration set.

The collector journals are the commit log.  This command never repairs or
silently drops a row: it requires exactly the 512 planned episodes, checks
every archive hash and array, and emits a canonical frozen manifest.  The
optional ActQuant export contains every valid replan frame from the fixed
60-episode subset in the flat format consumed by the released pi0.5 tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_committed_rows(calibration_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    origins: dict[str, list[str]] = {}
    for journal in sorted(calibration_dir.glob("worker_*.jsonl")):
        for line_number, line in enumerate(journal.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["episode_key"])
            origins.setdefault(key, []).append(f"{journal.name}:{line_number}")
            if key in rows and rows[key] != row:
                raise ValueError(f"conflicting committed rows for {key}: {origins[key]}")
            rows[key] = row
    return rows


def validate_archive(path: Path, row: dict[str, Any], model: str) -> dict[str, Any]:
    expected_action_horizon = 16 if model == "gr00t" else 50
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "images", "wrist_images", "right_images", "states", "prompts",
            "teacher_actions", "action_noises", "env_steps", "task", "env_seed",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        arrays = {name: archive[name] for name in required}
        frames = int(arrays["states"].shape[0])
        expected = {
            "images": (frames, 224, 224, 3),
            "wrist_images": (frames, 224, 224, 3),
            "right_images": (frames, 224, 224, 3),
            "states": (frames, 16),
            "prompts": (frames,),
            "teacher_actions": (frames, expected_action_horizon, 12),
            "action_noises": (frames, 50, 32),
            "env_steps": (frames,),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{path}: {name} shape {arrays[name].shape} != {shape}")
        if frames <= 0 or frames != int(row["frame_count"]):
            raise ValueError(f"{path}: invalid frame count {frames}")
        if any(arrays[name].dtype != np.uint8 for name in ("images", "wrist_images", "right_images")):
            raise ValueError(f"{path}: image dtype drift")
        for name in ("states", "teacher_actions", "action_noises"):
            if not np.issubdtype(arrays[name].dtype, np.floating) or not np.isfinite(arrays[name]).all():
                raise ValueError(f"{path}: non-finite/non-floating {name}")
        if str(arrays["task"].item()) != row["task"] or int(arrays["env_seed"].item()) != int(row["env_seed"]):
            raise ValueError(f"{path}: task/seed metadata mismatch")
        if not all(str(value) for value in arrays["prompts"]):
            raise ValueError(f"{path}: empty prompt")
    return {"frames": frames, "action_horizon": expected_action_horizon}


def finalize(
    manifest_path: Path,
    calibration_dir: Path,
    output: Path,
    precision_attestations_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    precision_attestations = json.loads(
        precision_attestations_path.read_text(encoding="utf-8")
    )
    if precision_attestations.get("kind") != "qvla_actquant_teacher_precision_v1":
        raise ValueError("invalid teacher precision attestation inventory")
    precision_by_sha = {
        row["server_metadata_sha256"]: row
        for row in precision_attestations.get("servers", [])
    }
    if manifest["qvla_episode_count"] != 512 or manifest["actquant_episode_count"] != 60:
        raise ValueError("frozen plan must contain exactly 512/60 episodes")
    if set(manifest["formal_seeds"]) != set(range(50)):
        raise ValueError("formal seed freeze drift")
    planned = {row["episode_key"]: row for row in manifest["qvla_episodes"]}
    committed = load_committed_rows(calibration_dir)
    if set(committed) != set(planned):
        missing = sorted(set(planned) - set(committed))
        extra = sorted(set(committed) - set(planned))
        raise ValueError(f"calibration is incomplete/noncanonical: missing={len(missing)} extra={len(extra)}")
    frozen_rows = []
    server_shas = set()
    total_frames = 0
    for key in sorted(planned):
        row = committed[key]
        if row.get("status") != "complete":
            raise ValueError(f"non-complete calibration row: {key}")
        if row.get("source") != "fp16_teacher_proxy" or row.get("source_protocol_equivalent") is not False:
            raise ValueError(f"source semantics drift: {key}")
        if row.get("test_results_used") is not False or int(row["env_seed"]) in range(50):
            raise ValueError(f"formal feedback/seed leakage: {key}")
        if row.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError(f"collector manifest SHA drift: {key}")
        archive = Path(row["archive"]).expanduser().resolve()
        if not archive.is_file() or sha256_file(archive) != row["archive_sha256"]:
            raise ValueError(f"archive missing/corrupt: {key}")
        checked = validate_archive(archive, row, manifest["model"])
        total_frames += checked["frames"]
        server_shas.add(row["server_metadata_sha256"])
        precision = precision_by_sha.get(row["server_metadata_sha256"])
        if (
            not precision
            or precision.get("model") != manifest["model"]
            or precision.get("resolved") != "float16"
            or precision.get("strict_all_linear_conv_fp16") is not True
        ):
            raise ValueError(
                f"teacher precision is not attested FP16 for {key}: "
                f"{row['server_metadata_sha256']}"
            )
        frozen_rows.append({
            "episode_key": key,
            "task": row["task"],
            "env_seed": int(row["env_seed"]),
            "archive": str(archive),
            "archive_sha256": row["archive_sha256"],
            "frames": checked["frames"],
            "success": bool(row["success"]),
            "termination": row["termination"],
            "actquant_subset": key in set(manifest["actquant_episode_keys"]),
        })
    if sum(row["actquant_subset"] for row in frozen_rows) != 60:
        raise ValueError("ActQuant subset cardinality drift")
    frozen = {
        "schema_version": 1,
        "kind": "qvla_actquant_frozen_calibration",
        "model": manifest["model"],
        "model_unit": manifest["model_unit"],
        "checkpoint": manifest["checkpoint"],
        "source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "test_results_used": False,
        "selection_seed": 42,
        "formal_seeds": list(range(50)),
        "planned_manifest": str(manifest_path),
        "planned_manifest_sha256": sha256_file(manifest_path),
        "qvla_episode_count": len(frozen_rows),
        "actquant_episode_count": sum(row["actquant_subset"] for row in frozen_rows),
        "qvla_frame_count": total_frames,
        "server_metadata_sha256": sorted(server_shas),
        "teacher_precision": "float16",
        "teacher_precision_attestations": str(precision_attestations_path),
        "teacher_precision_attestations_sha256": sha256_file(precision_attestations_path),
        "episodes": frozen_rows,
    }
    frozen["episode_rows_sha256"] = canonical_hash(frozen_rows)
    atomic_json(output, frozen)
    return frozen


def export_actquant(frozen_path: Path, output_dir: Path) -> dict[str, Any]:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    episodes = [row for row in frozen["episodes"] if row["actquant_subset"]]
    if len(episodes) != 60:
        raise ValueError("ActQuant export requires exactly 60 episodes")
    total = sum(int(row["frames"]) for row in episodes)
    horizon = 16 if frozen["model"] == "gr00t" else 50
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pixel_values": output_dir / "pixel_values.npy",
        "state": output_dir / "state.npy",
        "gt_actions": output_dir / "gt_actions.npy",
    }
    pixel_values = open_memmap(paths["pixel_values"], mode="w+", dtype=np.uint8, shape=(total, 3, 224, 224, 3))
    states = open_memmap(paths["state"], mode="w+", dtype=np.float32, shape=(total, 16))
    actions = open_memmap(paths["gt_actions"], mode="w+", dtype=np.float32, shape=(total, horizon, 12))
    prompts: list[str] = []
    frames = []
    offset = 0
    for episode_index, row in enumerate(episodes):
        archive_path = Path(row["archive"])
        if sha256_file(archive_path) != row["archive_sha256"]:
            raise ValueError(f"archive drift during ActQuant export: {row['episode_key']}")
        with np.load(archive_path, allow_pickle=False) as archive:
            count = int(row["frames"])
            stop = offset + count
            pixel_values[offset:stop, 0] = archive["images"]
            pixel_values[offset:stop, 1] = archive["wrist_images"]
            pixel_values[offset:stop, 2] = archive["right_images"]
            states[offset:stop] = archive["states"]
            actions[offset:stop] = archive["teacher_actions"]
            episode_prompts = [str(value) for value in archive["prompts"]]
            prompts.extend(episode_prompts)
            for local_index in range(count):
                frames.append({
                    "frame_index": offset + local_index,
                    "episode_index": episode_index,
                    "episode_key": row["episode_key"],
                    "task": row["task"],
                    "env_seed": row["env_seed"],
                    "replan_index": local_index,
                })
            offset = stop
    if offset != total or len(prompts) != total or len(frames) != total:
        raise AssertionError("frame export accounting drift")
    pixel_values.flush(); states.flush(); actions.flush()
    del pixel_values, states, actions
    prompts_path = output_dir / "prompts.json"
    atomic_json(prompts_path, prompts)
    frame_path = output_dir / "frame_manifest.json"
    atomic_json(frame_path, frames)
    metadata = {
        "schema_version": 1,
        "kind": "actquant_flat_calibration",
        "model": frozen["model"],
        "model_unit": frozen["model_unit"],
        "source": "fp16_teacher_proxy",
        "source_protocol_equivalent": False,
        "selection_seed": 42,
        "episode_count": len(episodes),
        "total_frames": total,
        "action_horizon": horizon,
        "action_dim": 12,
        "state_dim": 16,
        "frozen_manifest": str(frozen_path),
        "frozen_manifest_sha256": sha256_file(frozen_path),
        "frame_inventory_sha256": canonical_hash(frames),
        "files": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in {**paths, "prompts": prompts_path, "frame_manifest": frame_path}.items()
        },
    }
    expected_bytes = total * (3 * 224 * 224 * 3 + 16 * 4 + horizon * 12 * 4)
    actual_array_bytes = sum(paths[name].stat().st_size - 128 for name in paths)  # informational only
    metadata["expected_raw_payload_bytes"] = expected_bytes
    metadata["array_file_payload_bytes_approx"] = actual_array_bytes
    atomic_json(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--calibration-dir", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--precision-attestations", type=Path, required=True)
    export = sub.add_parser("export-actquant")
    export.add_argument("--frozen-manifest", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = finalize(
            args.manifest.resolve(),
            args.calibration_dir.resolve(),
            args.output.resolve(),
            args.precision_attestations.resolve(),
        )
    else:
        result = export_actquant(args.frozen_manifest.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
