#!/usr/bin/env python3
"""Register the GR00T v2 W=L quick outcome as a non-inferiority anchor.

When the frozen v2 winner (historical main mask on the common runtime)
matches exact gdsq_main in the quick gate (no strict win), the honest
record is a compression/runtime-parity anchor — never a success-rate
improvement.  This tool freezes that registration from the v2 quick
aggregate with an explicit claim guard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_outputimpact import atomic_json


def register(aggregate: dict[str, Any], *, source: str) -> dict[str, Any]:
    if aggregate.get("kind") != "full_context_v2_quick_development_gate":
        raise ValueError(f"{source}: not a v2 quick gate aggregate")
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
        "kind": "full_context_gr00t_v2_non_inferiority_anchor",
        "role": "runtime_portability_non_inferiority_anchor",
        "status": "not_superior_to_gdsq_main",
        "source": str(source),
        "main_successes": main,
        "candidate_successes": candidate,
        "paired_wins": wins,
        "paired_losses": losses,
        "candidate_static_bytes": int(aggregate["table1_total_static_bytes"]),
        "budget_bytes": int(aggregate["table1_total_static_budget_bytes"]),
        "claim_guard": (
            "the v2 frozen mask is the historical main mask ported to the "
            "common runtime: parity with exact main is runtime portability "
            "(compression parity), not a success-rate improvement"
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
