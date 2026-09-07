#!/usr/bin/env python3
"""Regression tests for durable formal-evaluation supervision."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "tools" / "formal_resume_supervisor.py"
SPEC = importlib.util.spec_from_file_location("formal_resume_supervisor", MODULE_PATH)
assert SPEC and SPEC.loader
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


class FormalResumeSupervisorTest(unittest.TestCase):
    def test_commands_are_hard_limited_to_gpu_3456(self) -> None:
        for name in ("actquant", "daptq"):
            command = supervisor.pipeline_command(name)
            index = command.index("--eligible-gpus")
            self.assertEqual(command[index + 1], "3,4,5,6")
        actquant = supervisor.pipeline_command("actquant")
        self.assertEqual(actquant[actquant.index("--methods") + 1], "actquant")
        self.assertEqual(actquant[actquant.index("--max-workers") + 1], "32")
        self.assertEqual(actquant[actquant.index("--workers-per-gpu") + 1], "8")
        self.assertNotIn("qvla", actquant)

    def test_registered_worker_identity_fails_closed(self) -> None:
        formal = ROOT / "runs" / "daptq_table1" / "formal"
        output = formal / "raw" / "formal" / "daptq" / "pi05" / "job.jsonl"
        row = {"pid": 123, "process_group": 123, "output": str(output)}
        command = (
            f"/python {ROOT}/scripts/run_robocasa365_pi05_eval.py "
            f"--out {output} --egl-device 3"
        )
        self.assertTrue(supervisor.registered_worker_matches(row, command, 123, formal))
        self.assertFalse(supervisor.registered_worker_matches(row, command, 124, formal))
        unrelated = {**row, "output": "/tmp/unrelated.jsonl"}
        self.assertFalse(
            supervisor.registered_worker_matches(unrelated, command, 123, formal)
        )

    def test_legacy_worker_identity_requires_exact_run_and_gpu(self) -> None:
        formal = ROOT / "runs" / "daptq_table1" / "formal"
        output = formal / "raw" / "formal" / "daptq" / "pi05" / "job.jsonl"
        command = (
            f"/python {ROOT}/scripts/run_robocasa365_pi05_eval.py "
            f"--out {output} --egl-device 6"
        )
        self.assertTrue(supervisor.legacy_worker_matches(456, command, 456, formal))
        self.assertFalse(supervisor.legacy_worker_matches(456, command, 457, formal))
        self.assertFalse(
            supervisor.legacy_worker_matches(
                456, command.replace("--egl-device 6", "--egl-device 7"), 456, formal
            )
        )


if __name__ == "__main__":
    unittest.main()
