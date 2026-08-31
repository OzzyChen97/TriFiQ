#!/usr/bin/env python3
"""Losslessly redistribute resumable GR00T mask scores across more shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from quantvla_libero_dypac import PROTOCOL, atomic_json, sha256_file


SUITES = ("goal", "spatial", "object", "long")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-dir", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard-count", type=int)
    group.add_argument(
        "--shard-counts",
        help="Comma-separated counts in goal,spatial,object,long order",
    )
    parser.add_argument("--backup-dir", required=True)
    args = parser.parse_args()
    if args.shard_counts:
        counts = [int(value) for value in args.shard_counts.split(",")]
        if len(counts) != len(SUITES):
            raise ValueError("--shard-counts requires four values")
    else:
        counts = [int(args.shard_count)] * len(SUITES)
    if any(value < 1 for value in counts):
        raise ValueError("shard counts must be positive")

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL["protocol_id"]:
        raise ValueError("manifest protocol drift")
    identifiers = sorted(manifest["candidates"])
    source_root = Path(args.source_dir).resolve()
    backup_root = Path(args.backup_dir).resolve()
    backup_root.mkdir(parents=True, exist_ok=False)
    report: dict[str, dict] = {}

    for suite_index, suite in enumerate(SUITES):
        target_count = counts[suite_index]
        suite_root = source_root / suite
        sources = sorted(suite_root.glob("shard_*.json"))
        if not sources:
            raise FileNotFoundError(f"no source shards for {suite}")
        templates = []
        scores: dict[str, dict] = {}
        for path in sources:
            value = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "kind": "dypac_libero_gr00t_complete_mask_scores_suite",
                "model": "gr00t",
                "suite": suite,
                "protocol_id": PROTOCOL["protocol_id"],
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
            drift = {
                key: (value.get(key), wanted)
                for key, wanted in expected.items()
                if value.get(key) != wanted
            }
            if drift:
                raise ValueError(f"{path}: incompatible source shard: {drift}")
            templates.append(value)
            for identifier, row in value.get("scores", {}).items():
                if identifier not in manifest["candidates"]:
                    raise ValueError(f"{path}: unknown candidate {identifier}")
                if identifier in scores and scores[identifier] != row:
                    raise ValueError(f"{path}: conflicting duplicate score {identifier}")
                scores[identifier] = row

        suite_backup = backup_root / suite
        suite_backup.mkdir(parents=True)
        for path in sorted(suite_root.glob("shard_*.*")):
            if path.is_file():
                shutil.copy2(path, suite_backup / path.name)

        template = templates[0]
        assigned_sets = []
        written = 0
        for shard_index in range(target_count):
            assigned = [
                identifier
                for index, identifier in enumerate(identifiers)
                if index % target_count == shard_index
            ]
            assigned_sets.append(set(assigned))
            shard_scores = {
                identifier: scores[identifier]
                for identifier in assigned
                if identifier in scores
            }
            payload = dict(template)
            payload.update(
                {
                    "shard_index": shard_index,
                    "shard_count": target_count,
                    "assigned_candidates": assigned,
                    "scores": shard_scores,
                    "complete": set(shard_scores) == set(assigned),
                }
            )
            atomic_json(suite_root / f"shard_{shard_index}.json", payload)
            written += len(shard_scores)

        if set.union(*assigned_sets) != set(identifiers):
            raise RuntimeError(f"{suite}: reshard coverage failure")
        if any(left & right for i, left in enumerate(assigned_sets) for right in assigned_sets[i + 1 :]):
            raise RuntimeError(f"{suite}: reshard overlap failure")
        if written != len(scores):
            raise RuntimeError(f"{suite}: score conservation failure")
        report[suite] = {
            "source_shards": len(sources),
            "preserved_scores": len(scores),
            "target_shards": target_count,
        }

    atomic_json(backup_root / "reshard_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
