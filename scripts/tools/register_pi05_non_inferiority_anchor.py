#!/usr/bin/env python3
"""Register the pi0.5 14/50 W=L candidate as a non-inferiority anchor.

The v1 pi0.5 quick candidate matched main (14 vs 14 successes, W=4/L=4) at
much higher compression (172 W4 layers vs main's 80). It must never be
reported as a success-rate improvement; this tool records it as a frozen
high-compression non-inferiority anchor for the v2 quick to compare
against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_outputimpact import atomic_json


def register(aggregate: dict[str, Any], *, source: str) -> dict[str, Any]:
    if aggregate.get("kind") != "full_context_quick_development_gate":
        raise ValueError(f"{source}: not a full-context quick gate aggregate")
    main = int(aggregate["main_successes"])
    candidate = int(aggregate["candidate_successes"])
    wins = int(aggregate["paired_wins"])
    losses = int(aggregate["paired_losses"])
    if candidate > main or wins > losses:
        raise ValueError(
            f"{source}: candidate is superior to main; refusing to register "
            "a non-inferiority anchor for a superior candidate"
        )
    return {
        "schema_version": 1,
        "kind": "full_context_pi05_non_inferiority_anchor",
        "role": "high_compression_non_inferiority_anchor",
        "status": "not_superior_to_gdsq_main",
        "source": str(source),
        "main_successes": main,
        "candidate_successes": candidate,
        "paired_wins": wins,
        "paired_losses": losses,
        "claim_guard": (
            "success-parity with many more W4 layers is compression parity, "
            "not a success-rate improvement"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-aggregate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.quick_aggregate).expanduser().resolve()
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    report = register(aggregate, source=str(path))
    atomic_json(Path(args.out).expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()