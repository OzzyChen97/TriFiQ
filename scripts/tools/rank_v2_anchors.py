#!/usr/bin/env python3
"""Rank the four v2 anchors on the cross-context buffer and gate the search.

Reads the per-split anchor score payloads produced by the offline scorer,
computes the per-anchor per-split task-macro D_PAC / D_func scalars (the v2
estimand), and emits the stop decision: if full-W4 ranks as good as or
better than the historical main mask on any split while the closed-loop
quick gate showed main >> full-W4, the fixed teacher-state proxy is dead
and the 296 layer flips must NOT be rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from quantvla_full_context import task_scalars
from quantvla_outputimpact import atomic_json

SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
FULL_W4_ID = "full_w4"
MAIN_ID = "main_mask"


def task_macro(score_doc: dict[str, Any], metric: str) -> float:
    scalars = task_scalars(score_doc)
    values = [value for (key, _task), value in scalars.items() if key == metric]
    return float(np.mean(values))


def rank_anchors(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    anchors = sorted({identifier for payload in payloads.values() for identifier in payload})
    table: dict[str, dict[str, float]] = {}
    for identifier in anchors:
        table[identifier] = {}
        for split in SPLITS:
            score_doc = payloads[split][identifier]
            table[identifier][f"{split}_d_pac"] = task_macro(score_doc, "d_pac")
            table[identifier][f"{split}_d_func"] = task_macro(score_doc, "d_func")
    stop = False
    contradictions = []
    if FULL_W4_ID in table and MAIN_ID in table:
        for split in SPLITS:
            full_w4 = table[FULL_W4_ID][f"{split}_d_pac"]
            main = table[MAIN_ID][f"{split}_d_pac"]
            if full_w4 <= main + 1e-12:
                stop = True
                contradictions.append(split)
    return {
        "kind": "full_context_v2_anchor_ranking",
        "anchors": table,
        "stop_before_layer_flips": stop,
        "offline_contradicts_quick": stop,
        "contradicting_splits": contradictions,
        "decision": (
            "STOP: fixed teacher-state proxy ranks full-W4 no worse than main "
            "while closed loop shows main >> full-W4"
            if stop
            else "proxy_passes: main ranks above full-W4 on every split"
        ),
    }


def parse_payloads(value: str) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for chunk in value.split(","):
        split, _, path = chunk.partition("=")
        if not split or not path:
            raise ValueError(f"payload must be split=/path.json, got {chunk!r}")
        document = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        payloads[split] = document["scores"]
    if set(payloads) != set(SPLITS):
        raise ValueError(f"payloads must cover {SPLITS}")
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payloads",
        required=True,
        help="comma list of split=/path.json score payloads for all three splits",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = rank_anchors(parse_payloads(args.payloads))
    atomic_json(Path(args.out).expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()