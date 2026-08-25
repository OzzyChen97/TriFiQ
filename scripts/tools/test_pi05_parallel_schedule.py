#!/usr/bin/env python3
"""Unit tests for exact-key coverage in π0.5 resource schedules."""

from __future__ import annotations

import unittest

import pi05_make_parallel_schedule as schedule


SERVERS = [
    {"instance": "fpa", "config_id": "fp16", "gpu": 1},
    {"instance": "fpb", "config_id": "fp16", "gpu": 2},
    {"instance": "w4a", "config_id": "quantvla_w4a8_atmohb", "gpu": 3},
    {"instance": "w4b", "config_id": "quantvla_w4a8_atmohb", "gpu": 4},
    {"instance": "atma", "config_id": "gdsq_vla_atmohb", "gpu": 5},
    {"instance": "atmb", "config_id": "gdsq_vla_atmohb", "gpu": 6},
    {"instance": "gdsq", "config_id": "gdsq_vla", "gpu": 7},
]

LAYOUT = {
    "fp16": ("fpa", "fpb"),
    "quantvla_w4a8_atmohb": ("w4a", "w4b"),
    "gdsq_vla_atmohb": ("atma", "atmb"),
    "gdsq_vla": ("gdsq",),
}


def workers(intervals: tuple[tuple[int, int], ...]) -> list[dict]:
    rows = []
    for config, instances in LAYOUT.items():
        shard_count = len(instances)
        for shard, instance in enumerate(instances):
            for index, (start, end) in enumerate(intervals):
                rows.append(
                    schedule.parse_worker(
                        f"{config}_{shard}_{index},{config},{instance},"
                        f"{shard},{shard_count},{start}-{end}"
                    )
                )
    return rows


class ParallelScheduleCoverageTest(unittest.TestCase):
    def test_two_and_four_way_seed_partitions_are_exact(self) -> None:
        schedule.validate_layout(SERVERS, workers(((0, 24), (25, 49))))
        schedule.validate_layout(
            SERVERS, workers(((0, 12), (13, 24), (25, 37), (38, 49)))
        )

    def test_gap_is_rejected(self) -> None:
        rows = workers(((0, 12), (13, 24), (25, 37), (38, 49)))
        rows = [
            row
            for row in rows
            if not (
                row["config_id"] == "fp16"
                and row["trial_seed_start"] == 13
            )
        ]
        with self.assertRaisesRegex(ValueError, "gap/overlap"):
            schedule.validate_layout(SERVERS, rows)

    def test_overlap_is_rejected(self) -> None:
        rows = workers(((0, 12), (12, 24), (25, 37), (38, 49)))
        with self.assertRaisesRegex(ValueError, "gap/overlap"):
            schedule.validate_layout(SERVERS, rows)

    def test_out_of_range_shard_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid worker spec"):
            schedule.parse_worker("bad,fp16,fp,2,2,0-49")

    def test_duplicate_worker_id_is_rejected(self) -> None:
        rows = workers(((0, 24), (25, 49)))
        rows[1]["worker_id"] = rows[0]["worker_id"]
        with self.assertRaisesRegex(ValueError, "duplicate worker id"):
            schedule.validate_layout(SERVERS, rows)


if __name__ == "__main__":
    unittest.main()
