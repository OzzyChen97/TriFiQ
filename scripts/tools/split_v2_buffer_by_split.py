#!/usr/bin/env python3
"""Split the pooled v2 selection buffer into per-split archives.

GR00T scoring requires one checkpoint per target split, so the pooled
144-row archive is split into three 48-row archives (one per split) for
the anchor re-scoring runs. Row membership comes from the frozen spec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quantvla_outputimpact import atomic_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_buffer(spec: dict, buffer: Path, out_dir: Path) -> dict:
    task_split = {
        entry["task"]: entry["split"] for entry in spec["entries"]
    }
    with np.load(buffer, allow_pickle=False) as archive:
        names = list(archive.files)
        arrays = {name: np.asarray(archive[name]) for name in names}
    indices = {split: [] for split in ("atomic_seen", "composite_seen", "composite_unseen")}
    for index, task in enumerate(arrays["task_ids"]):
        split = task_split[str(task)]
        indices[split].append(index)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"kind": "full_context_v2_per_split_buffer_manifest", "splits": {}}
    for split, rows in indices.items():
        expected = spec["tasks_per_split"] * spec["seeds_per_task"] * spec["replans_per_sequence"]
        if len(rows) != expected:
            raise ValueError(f"{split}: {len(rows)} rows, expected {expected}")
        out = out_dir / f"selection_buffer_{split}.npz"
        np.savez_compressed(
            out,
            **{name: arrays[name][rows] for name in names},
        )
        report["splits"][split] = {
            "archive": str(out),
            "sha256": sha256_file(out),
            "rows": len(rows),
        }
    report["source"] = str(buffer)
    report["source_sha256"] = sha256_file(buffer)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).expanduser().resolve().read_text(encoding="utf-8"))
    if spec.get("kind") != "full_context_v2_selection_context_spec":
        raise ValueError(f"{args.spec}: wrong spec kind")
    report = split_buffer(
        spec, Path(args.buffer).expanduser().resolve(), Path(args.out_dir).expanduser().resolve()
    )
    manifest = Path(args.out_dir).expanduser().resolve() / "manifest.json"
    atomic_json(manifest, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()