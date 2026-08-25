#!/usr/bin/env python3
"""Audit π0.5 checkpoint and quantization artifacts for reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"
DEFAULT_ROOT = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    layers = payload.get("layers") or {}
    wrapped = sum(
        not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) > 0
        for row in layers.values()
    )
    meta = payload.get("meta") or {}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "layers": len(layers),
        "wrapped_layers": wrapped,
        "fp16_layers": len(layers) - wrapped,
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "total_bytes": meta.get("total_bytes", payload.get("total_bytes")),
        "budget_bytes": meta.get("budget_bytes", payload.get("budget_bytes")),
        "kind": meta.get("kind"),
        "mask_sha256": meta.get("mask_sha256"),
    }


def artifact_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        result["sha256"] = sha256_file(path)
    sidecar = Path(str(path) + ".json")
    if sidecar.is_file():
        result["sidecar"] = {
            "path": str(sidecar),
            "sha256": sha256_file(sidecar),
        }
        try:
            meta = read_json(sidecar)
            result["checkpoint_sha256"] = meta.get("checkpoint_sha256")
            result["plan_sha256"] = meta.get("plan_sha256")
            result["calibration_buffer_sha256"] = meta.get("calibration_buffer_sha256")
        except json.JSONDecodeError:
            result["sidecar_error"] = "invalid JSON"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--a8", action="append", default=[])
    parser.add_argument("--atm", action="append", default=[])
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    model_file = checkpoint / "model.safetensors"
    config_file = checkpoint / "config.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_dir": str(checkpoint),
        "checkpoint": {
            "model_safetensors": str(model_file),
            "model_sha256": sha256_file(model_file) if model_file.is_file() else None,
            "config": str(config_file),
            "config_sha256": sha256_file(config_file) if config_file.is_file() else None,
            "config_json": read_json(config_file) if config_file.is_file() else None,
        },
        "plans": [plan_summary(Path(path).expanduser().resolve()) for path in args.plan],
        "a8": [artifact_summary(Path(path).expanduser().resolve()) for path in args.a8],
        "atm": [artifact_summary(Path(path).expanduser().resolve()) for path in args.atm],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        output = Path(args.out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(json.dumps({"out": str(output), "sha256": sha256_file(output)}, indent=2, sort_keys=True))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
