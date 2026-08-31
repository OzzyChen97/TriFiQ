#!/usr/bin/env python3
"""Strict result-key and statistical-family tests for the Table-1 extension."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

import qvla_actquant_formal_results as formal  # noqa: E402


class FormalResultsTest(unittest.TestCase):
    def write_arm(self, path: Path, method: str, model: str, value) -> None:
        manifest_path = path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps({"method": method, "model": model, "storage": {"synthetic": True}}),
            encoding="utf-8",
        )
        manifest_sha = formal.sha256_file(manifest_path)
        rows = []
        for split, task, seed in sorted(formal.expected_episode_keys()):
            record = {
                "schema_version": 1,
                "method": method,
                "model": model,
                "task_split": split,
                "task": task,
                "seed": seed,
                "success": bool(value(split, task, seed)),
                "status": "complete",
                "server_metadata_sha256": "1" * 64,
                "arm_manifest_path": str(manifest_path),
                "arm_manifest_sha256": manifest_sha,
                "protocol_sha256": formal.sha256_file(formal.PROTOCOL_PATH),
                "flow_steps": 4,
                "n_action_steps": 16,
                "paired_action_noise": True,
                "source_file": "/synthetic.jsonl",
                "source_line": len(rows) + 1,
            }
            record["record_sha256"] = formal.canonical_sha256(record)
            rows.append(record)
        formal.atomic_jsonl(path, rows)

    def test_complete_family_and_holm_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = []
            for method in ("qvla", "actquant"):
                for model in ("gr00t", "pi05"):
                    path = root / f"{method}_{model}.jsonl"
                    self.write_arm(path, method, model, lambda _s, _t, seed: seed < 30)
                    candidates.append(path)
            baselines = {}
            for model in ("gr00t", "pi05"):
                path = root / f"dypac_{model}.jsonl"
                self.write_arm(path, "dypac", model, lambda _s, _t, seed: seed < 25)
                baselines[model] = path
            output = root / "aggregate.json"
            result = formal.aggregate(candidates, baselines, output, draws=32)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(result["ready_for_table_update"])
            self.assertEqual(payload["formal_episode_count_new"], 10_000)
            self.assertEqual(len(payload["holm_family"]), 4)
            self.assertTrue(all(row["paired_wins"] == 250 for row in payload["comparisons"].values()))
            self.assertTrue(all(row["paired_losses"] == 0 for row in payload["comparisons"].values()))

    def test_partial_and_tampered_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.jsonl"
            self.write_arm(path, "qvla", "gr00t", lambda _s, _t, _seed: False)
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                formal.load_canonical(path)
            row = json.loads(lines[0])
            row["success"] = True
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record SHA mismatch"):
                formal.load_canonical(path)


if __name__ == "__main__":
    unittest.main()
