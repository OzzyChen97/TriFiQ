#!/usr/bin/env python3
"""Tests for controlled final-document block replacement."""

from __future__ import annotations

import unittest

import update_pi05_final_doc as updater


class FinalDocReplacementTest(unittest.TestCase):
    def test_replaces_only_marked_block(self) -> None:
        original = f"before\n{updater.BEGIN}\nold\n{updater.END}\nafter\n"
        block = f"{updater.BEGIN}\nnew\n{updater.END}"
        self.assertEqual(
            updater.replace_block(original, block),
            f"before\n{block}\nafter\n",
        )

    def test_missing_or_duplicate_markers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "begin marker count"):
            updater.replace_block("no markers", "block")
        duplicated = f"{updater.BEGIN}{updater.BEGIN}{updater.END}"
        with self.assertRaisesRegex(ValueError, "begin marker count"):
            updater.replace_block(duplicated, "block")


if __name__ == "__main__":
    unittest.main()
