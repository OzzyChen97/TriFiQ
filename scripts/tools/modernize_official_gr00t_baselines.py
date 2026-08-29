#!/usr/bin/env python3
"""Modernize the official GR00T paired-50 baseline rows for Table-1 reuse.

The official benchmark rows predate the modern ``status`` field convention
but carry full raw-result attestation (``manifest_sha256`` /
``config_sha256``) and the identical paired-noise protocol.  This tool
verifies every row against the frozen official manifests and the current
checkpoints, then emits schema-modernized copies (``status: complete``)
into the canonical baseline directories consumed by the pinned Table-1
aggregator.  No success value is ever changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import sha256_file
from quantvla_full_context import PROTOCOL
from quantvla_outputimpact import atomic_json

SPLITS = {
    "atomic_seen": "robocasa365_official_full_atomic_paired50",
    "composite_seen": "robocasa365_official_full_composite_seen_paired50",
    "composite_unseen": "robocasa365_official_full_composite_unseen_paired50",
}
CONFIG_MAP = {
    "fp16": "fp16",
    "quantvla_w4a8": "w4a8_atmohb",
    "gdsq_vla_main": "cscka_final",
}
REPO = Path("/home1/gyy/vla/QuantVLA")
CKPT = (
    REPO / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining"
)
ACTION_NOISE_SCHEME = "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"


def modernize_split(
    split: str,
    official_dir: Path,
    config_id: str,
    official_config_id: str,
    out_dir: Path,
    registry_tasks: list[str],
    manifest_record: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    manifest_path = official_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configs = {str(row["id"]): row for row in manifest.get("configs") or []}
    official_config = configs.get(official_config_id)
    if official_config is None:
        raise ValueError(f"{manifest_path}: missing config {official_config_id}")
    config_sha = str(official_config.get("config_sha256") or "")
    if not config_sha:
        raise ValueError(f"{manifest_path}: {official_config_id} lacks config_sha256")
    checkpoint_path = Path(manifest.get("checkpoint_path") or "").expanduser().resolve()
    expected_checkpoint = (CKPT / split / "checkpoint-60000").resolve()
    if checkpoint_path != expected_checkpoint:
        raise ValueError(
            f"{manifest_path}: checkpoint drift {checkpoint_path} != {expected_checkpoint}"
        )
    protocol = manifest.get("protocol") or {}
    checks = {
        "paired_noise": protocol.get("paired_action_noise") is True,
        "noise_scheme": protocol.get("action_noise_scheme") == ACTION_NOISE_SCHEME,
        "n_action_steps": int(protocol.get("n_action_steps", -1)) == 16,
        "denoising_steps": int(protocol.get("denoising_steps", -1)) == 4,
        "split": protocol.get("split") == "target",
        "tasks": set(manifest.get("tasks") or []) == set(registry_tasks),
        "seeds": list(manifest.get("seeds") or []) == list(range(50)),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{manifest_path}: official protocol drift: {failed}")

    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for source in sorted(official_dir.glob(f"{official_config_id}_s*.jsonl")):
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            checks = {
                "manifest": row.get("manifest_sha256") == manifest_sha,
                "config": row.get("config_sha256") == config_sha,
                "config_id": row.get("config") == official_config_id,
                "paired": row.get("paired_action_noise") is True,
                "scheme": row.get("action_noise_scheme") == ACTION_NOISE_SCHEME,
                "success": row.get("success") is not None,
                "steps": row.get("steps") is not None,
            }
            bad = [key for key, passed in checks.items() if not passed]
            if bad:
                raise ValueError(
                    f"{source}:{line_number}: unattested official row: {bad}"
                )
            key = (str(row["task"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"{source}: duplicate official key {key}")
            rows[key] = row
    expected = {(task, seed) for task in registry_tasks for seed in range(50)}
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    if missing or extra:
        raise ValueError(
            f"{official_config_id}/{split}: coverage drift missing={len(missing)} extra={len(extra)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    modernized = []
    for index, row in enumerate(ordered):
        row = dict(row)
        row["status"] = "complete"
        row["config"] = config_id
        row["modernized_from_official"] = True
        modernized.append(row)
    output = out_dir / f"{config_id}_{split}_modernized.jsonl"
    with open(output, "w", encoding="utf-8") as handle:
        for row in modernized:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_record[split] = {
        "official_dir": str(official_dir),
        "official_manifest_sha256": manifest_sha,
        "official_config_id": official_config_id,
        "config_sha256": config_sha,
        "rows": len(modernized),
        "out": str(output),
        "out_sha256": sha256_file(output),
    }
    return len(modernized), manifest_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(REPO / "runs/full_context_v2/table1/baselines"),
    )
    parser.add_argument(
        "--manifest-out",
        default=str(REPO / "runs/full_context_v2/table1/baselines/manifest.json"),
    )
    args = parser.parse_args()
    out_root = Path(args.out_dir).expanduser().resolve()
    registry = PROTOCOL["table1"]["tasks"]
    record: dict[str, Any] = {"kind": "official_gr00t_baseline_modernization_manifest"}
    total = 0
    for config_id, official_config_id in CONFIG_MAP.items():
        record.setdefault(config_id, {})
        for split, official_name in SPLITS.items():
            count, _ = modernize_split(
                split,
                REPO / "runs" / official_name,
                config_id,
                official_config_id,
                out_root / config_id / split,
                list(registry[split]),
                record[config_id],
            )
            total += count
    record["total_rows"] = total
    atomic_json(Path(args.manifest_out).expanduser().resolve(), record)
    print(json.dumps({"out": args.out_dir, "manifest": args.manifest_out, "rows": total}, indent=2))


if __name__ == "__main__":
    main()
