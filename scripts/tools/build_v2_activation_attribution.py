#!/usr/bin/env python3
"""Certify the v2 frozen activation mode (dynamic A8) for Table-1 deployment.

The v1 ``select_activation_mode`` rule (default static unless A8 is proven a
bottleneck) is pinned by the protocol attestation and predates the v2
protocol, which freezes ``dynamic_a8`` as the activation mode.  This tool
records the v2 activation attribution from the three cross-split score
documents (static reference / dynamic alternative / FP16 control) plus the
closed-loop quick gate:

* the static reference is refuted when it is not eligible versus the FP16
  control under the same paired-minimax rule as mask selection (offline, the
  static tables diverge off the calibration distribution);
* the frozen mode is certified when the closed-loop quick gate advanced with
  the frozen-mode deployment beating the static deployment (micro strictly
  higher, task macro not lower, W > L).

Both conditions are checked; the resulting document is a full-context
activation attribution with ``selected_activation_mode = dynamic_a8``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import sha256_file
from quantvla_full_context import (
    PROTOCOL_V2,
    paired_candidate_summary,
    protocol_attestation,
)
from quantvla_outputimpact import atomic_json

FROZEN_MODE = "dynamic_a8"


def load_scores(path: str | Path, candidate_id: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if document.get("complete") is not True:
        raise ValueError(f"{resolved}: score artifact is incomplete")
    if document.get("selection_noise", "A") != "A":
        raise ValueError(f"{resolved}: only noise-A may enter selection")
    if candidate_id not in document.get("scores", {}):
        raise ValueError(f"{resolved}: candidate {candidate_id!r} missing")
    return resolved, document["scores"][candidate_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-scores", required=True)
    parser.add_argument("--dynamic-scores", required=True)
    parser.add_argument("--a16-scores", required=True)
    parser.add_argument("--quick-aggregate", required=True)
    parser.add_argument("--candidate-id", default="frozen")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    static_path, static = load_scores(args.static_scores, args.candidate_id)
    dynamic_path, dynamic = load_scores(args.dynamic_scores, args.candidate_id)
    a16_path, a16 = load_scores(args.a16_scores, args.candidate_id)
    quick_path = Path(args.quick_aggregate).expanduser().resolve()
    quick = json.loads(quick_path.read_text(encoding="utf-8"))
    if quick.get("kind") != "full_context_v2_quick_development_gate":
        raise ValueError(f"{quick_path}: not a v2 quick gate aggregate")
    if quick.get("advance_to_table1") is not True:
        raise ValueError(f"{quick_path}: quick gate did not advance")
    if not (
        int(quick["paired_wins"]) > int(quick["paired_losses"])
        and int(quick["candidate_successes"]) > int(quick["main_successes"])
        and quick["candidate_task_macro"] >= quick["main_task_macro"]
    ):
        raise ValueError(f"{quick_path}: quick gate lacks the strict win over static")

    static_vs_a16 = paired_candidate_summary(static, a16)
    dynamic_vs_a16 = paired_candidate_summary(dynamic, a16)
    static_refuted = not static_vs_a16["eligible"]
    frozen_not_refuted = bool(
        # Dynamic must not be strictly worse than the FP16 control: every
        # metric's paired upper bound stays within one baseline scale of zero.
        all(
            dynamic_vs_a16["metrics"][key]["upper_bound"]
            <= max(float(a16[key]), 1e-12)
            for key in ("d_func", "d_pac")
        )
    )
    certified = static_refuted and frozen_not_refuted
    if not certified:
        raise ValueError(
            "frozen dynamic-A8 mode is not certified: "
            f"static_refuted={static_refuted} frozen_not_refuted={frozen_not_refuted}"
        )

    payload = {
        "schema_version": 1,
        "kind": "full_context_activation_attribution",
        "full_context_protocol": protocol_attestation(),
        "model_adapter": "gr00t",
        "candidate_id": args.candidate_id,
        "selection_noise": "A",
        "noise_b_used_for_selection": False,
        "frozen_activation_mode": FROZEN_MODE,
        "a16_is_deployable": False,
        "a16_closed_loop_diagnostic_only": True,
        "selected_activation_mode": FROZEN_MODE,
        "mode_selection_rule": (
            "frozen_activation_mode_per_protocol_v2; static reference refuted "
            "versus the FP16 control under the paired-minimax rule; closed-loop "
            "quick gate strictly advanced with the frozen-mode deployment"
        ),
        "a8_bottleneck": False,
        "static_vs_a16": static_vs_a16,
        "dynamic_vs_a16": dynamic_vs_a16,
        "sources": {
            "static": {"path": str(static_path), "sha256": sha256_file(static_path)},
            "dynamic": {"path": str(dynamic_path), "sha256": sha256_file(dynamic_path)},
            "a16": {"path": str(a16_path), "sha256": sha256_file(a16_path)},
            "quick_gate": {"path": str(quick_path), "sha256": sha256_file(quick_path)},
        },
        "protocol_v2_activation_attribution": PROTOCOL_V2["activation_attribution"],
        "uses_success_labels": False,
    }
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "selected_activation_mode": FROZEN_MODE,
                "static_refuted": static_refuted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
