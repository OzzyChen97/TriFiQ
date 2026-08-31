#!/usr/bin/env python3
"""Freeze source, license, patch, build, paper, and checkpoint provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "qvla": {
        "root": REPO_ROOT / "external" / "QVLA",
        "commit": "26cc4821a3be4c003d09d3c7997b38db2a347982",
        "url": "https://github.com/AutoLab-SAI-SJTU/QVLA",
        "paper": REPO_ROOT / "docs/gdsq_vla_iclr2027/third_party/qvla_actquant/qvla-2602.03782v1.pdf",
        "paper_sha256": "e53902d1c8f1c07b555a18c058d64a62d415e6e1c0b581b41fe1f25007af17c4",
        "licenses": {
            "openvla/LICENSE": "MIT",
            "openvla-oft/LICENSE": "MIT",
            "UniVLA/LICENSE": "Apache-2.0",
        },
    },
    "actquant": {
        "root": REPO_ROOT / "external" / "ActQuant",
        "commit": "b64791125070652fe6b554e244fe809c79ef5246",
        "url": "https://github.com/arashakb/ActQuant",
        "paper": REPO_ROOT / "docs/gdsq_vla_iclr2027/third_party/qvla_actquant/actquant-2605.24011v3.pdf",
        "paper_sha256": "8a3371d2668a259508184e4064197ea5c7c0dd2228725ffade028e7c739200b0",
        "licenses": {"LICENSE": "MIT"},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def git_archive_sha256(root: Path, commit: str) -> str:
    digest = hashlib.sha256()
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", commit], cwd=root, stdout=subprocess.PIPE
    )
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
    if process.wait() != 0:
        raise RuntimeError(f"git archive failed: {root}")
    return digest.hexdigest()


def source_record(name: str, config: dict[str, Any], patch: Path) -> dict[str, Any]:
    root = Path(config["root"])
    commit = command("git", "rev-parse", "HEAD", cwd=root)
    if commit != config["commit"]:
        raise ValueError(f"{name} commit drift: {commit}")
    paper = Path(config["paper"])
    if sha256_file(paper) != config["paper_sha256"]:
        raise ValueError(f"{name} paper SHA drift")
    licenses = []
    for relative, identifier in config["licenses"].items():
        path = root / relative
        licenses.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "spdx": identifier,
            "sha256": sha256_file(path),
        })
    dirty = command("git", "status", "--short", cwd=root).splitlines()
    record = {
        "repository": config["url"],
        "checkout": str(root.relative_to(REPO_ROOT)),
        "commit": commit,
        "clean_source_archive_sha256": git_archive_sha256(root, commit),
        "paper": str(paper.relative_to(REPO_ROOT)),
        "paper_sha256": config["paper_sha256"],
        "licenses": licenses,
        "working_tree_status": dirty,
    }
    if name == "actquant":
        live_diff = subprocess.check_output(
            ["git", "diff", "--", "tools/quantize-vision/quantize-vision.cpp"], cwd=root
        )
        if live_diff != patch.read_bytes():
            raise ValueError("ActQuant live quantizer diff is not the frozen replayable patch")
        record["local_patch"] = {
            "path": str(patch.relative_to(REPO_ROOT)),
            "sha256": sha256_file(patch),
            "applies_to_commit": config["commit"],
        }
    elif dirty:
        raise ValueError(f"unexpected dirty upstream source: {root}")
    return record


def checkpoint_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        files = [path]
        base = path.parent
    else:
        base = path
        files = sorted(
            child for child in path.rglob("*")
            if child.is_file() and child.suffix in {".json", ".safetensors"}
        )
    records = []
    for index, child in enumerate(files, 1):
        print(f"[provenance] hashing {path.name}: {index}/{len(files)} {child.name}", flush=True)
        records.append({
            "path": str(child.relative_to(REPO_ROOT)),
            "relative_path": str(child.relative_to(base)),
            "bytes": child.stat().st_size,
            "sha256": sha256_file(child),
        })
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "files": records,
        "bytes": sum(record["bytes"] for record in records),
        "checkpoint_manifest_sha256": canonical_hash(records),
    }


def build_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "runs/qvla_actquant_table1/provenance.json",
    )
    args = parser.parse_args()
    patch = REPO_ROOT / "patches/qvla_actquant/actquant_component_quantizer.patch"
    checkpoints = {
        unit: checkpoint_record(path)
        for unit, path in {
            "gr00t_atomic_seen": REPO_ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000",
            "gr00t_composite_seen": REPO_ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000",
            "gr00t_composite_unseen": REPO_ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000",
            "pi05_all_target": REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors",
        }.items()
    }
    value = {
        "schema_version": 1,
        "kind": "qvla_actquant_source_provenance",
        "sources": {name: source_record(name, config, patch) for name, config in SOURCES.items()},
        "build": {
            "component_quantizer": build_record(REPO_ROOT / "external/ActQuant/build-qvla-actquant/bin/quantize-vision"),
            "llama_quantize": build_record(REPO_ROOT / "external/ActQuant/build-qvla-actquant/bin/llama-quantize"),
            "cmake": command("/home1/gyy/probe/miniforge3/envs/groot_test/bin/cmake", "--version").splitlines()[0],
            "compiler": command("c++", "--version").splitlines()[0],
        },
        "checkpoint_hash_semantics": "SHA256 of canonical sorted per-file path/size/SHA256 records",
        "checkpoints": checkpoints,
        "calibration_download_probe": {
            "url": "https://utexas.box.com/shared/static/6yw8qwkwufg7di063v7e5zbcyhgn8nlf.tar",
            "checked_utc": "2026-08-31T08:14:14Z",
            "http_status": 404,
            "fallback": "fp16_teacher_proxy",
            "source_protocol_equivalent": False,
        },
    }
    value["record_sha256"] = canonical_hash(value)
    atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output.resolve()),
        "record_sha256": value["record_sha256"],
        "checkpoint_manifest_sha256": {
            key: record["checkpoint_manifest_sha256"] for key, record in checkpoints.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
