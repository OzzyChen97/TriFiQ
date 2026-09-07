#!/usr/bin/env python3
"""Focused tests for resumable ActQuant formal finalization."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

import qvla_actquant_formal_pipeline as pipeline  # noqa: E402


class FinalMetadataTest(unittest.TestCase):
    def test_completed_rows_define_server_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            formal = Path(directory)
            output = formal / "raw" / "formal" / "actquant" / "pi05" / "rows.jsonl"
            output.parent.mkdir(parents=True)
            hashes = ["1" * 64, "2" * 64]
            output.write_text(
                "".join(
                    json.dumps({"server_metadata_sha256": hashes[index % 2]}) + "\n"
                    for index in range(2500)
                ),
                encoding="utf-8",
            )
            with mock.patch.object(pipeline, "FORMAL", formal):
                self.assertEqual(
                    pipeline.completed_result_server_hashes("actquant", "pi05"),
                    hashes,
                )

    def test_incomplete_rows_cannot_be_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            formal = Path(directory)
            output = formal / "raw" / "formal" / "actquant" / "gr00t" / "rows.jsonl"
            output.parent.mkdir(parents=True)
            output.write_text(
                json.dumps({"server_metadata_sha256": "1" * 64}) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(pipeline, "FORMAL", formal):
                with self.assertRaisesRegex(ValueError, "requires 2500 rows"):
                    pipeline.completed_result_server_hashes("actquant", "gr00t")


if __name__ == "__main__":
    unittest.main()
