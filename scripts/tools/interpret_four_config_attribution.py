#!/usr/bin/env python3
"""Interpret the GR00T four-config attribution (H / M / C / C16).

The four closed-loop configurations share tasks and seeds 50-59 as
development diagnostics:

* H   historical main mask on the historical main runtime (static A8,
     old per-output quantizer, row_rotation=restore, OHB off);
* M   historical main mask on the common runtime (Hessian group64,
     row_rotation=0, dynamic A8, all corrections off);
* C   v1 frozen candidate mask (116 W4 / 0 FP16) on the common runtime;
* C16 same mask as C with FP16 activations (A16 diagnostic).

Interpretation rules (paired over the same task/seed keys):

* H ~= M and M >> C: the protected-mask choice is the problem;
* H >> M:           the historical runtime / rotation / A8 policy is the
                    main contributor, so protection alone cannot explain main;
* C16 ~= C:         A8 is exonerated; the problem is W4 or state drift;
* C16 >> C:         dynamic A8 is unstable on candidate states -> a bounded
                    dynamic A8 would be investigated next;
* offline order ranks C above M but closed loop shows M >> C: the fixed
  teacher-state proxy is dead.

This tool is a diagnostic reporter; it never feeds Table 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantvla_outputimpact import atomic_json

CONFIG_IDS = ("h", "m", "c", "c16")
DELTA_EQUAL_THRESHOLD = 1  # |wins - losses| <= 1 counts as "approx equal"


def load_rows(run_dir: str | Path, config_id: str) -> dict[tuple[str, int], bool]:
    """Load completed episode rows for one config from a run directory."""
    rows: dict[tuple[str, int], bool] = {}
    for path in sorted(Path(run_dir).glob("**/*.jsonl")):
        # Retired run dirs are moved aside with a dot-prefix (".aborted_*");
        # their rows share (task, seed) keys with the live run and must not
        # enter the paired comparison.
        if any(part.startswith(".") for part in path.relative_to(run_dir).parts):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("config") != config_id:
                continue
            if row.get("status") != "complete":
                continue
            key = (str(row["task"]), int(row["seed"]))
            if key in rows:
                raise ValueError(f"duplicate episode key for {config_id}: {key}")
            rows[key] = bool(row["success"])
    return rows


def paired_comparison(
    a: dict[tuple[str, int], bool], b: dict[tuple[str, int], bool]
) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("configs share no paired (task, seed) keys")
    wins = losses = 0
    per_task: dict[str, dict[str, int]] = {}
    for task, seed in common:
        delta = int(bool(a[(task, seed)])) - int(bool(b[(task, seed)]))
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        slot = per_task.setdefault(task, {"a": 0, "b": 0, "n": 0})
        slot["n"] += 1
        slot["a"] += int(bool(a[(task, seed)]))
        slot["b"] += int(bool(b[(task, seed)]))
    return {
        "pairs": len(common),
        "a_successes": sum(int(bool(a[key])) for key in common),
        "b_successes": sum(int(bool(b[key])) for key in common),
        "wins": wins,
        "losses": losses,
        "ties": len(common) - wins - losses,
        "delta": wins - losses,
        "per_task": per_task,
    }


def relation(comparison: dict[str, Any]) -> str:
    """Map a paired comparison to a >> / approx / << relation label."""
    if abs(comparison["delta"]) <= DELTA_EQUAL_THRESHOLD:
        return "approx"
    return "a_gt_b" if comparison["delta"] > 0 else "b_gt_a"


def interpret(
    rows: dict[str, dict[tuple[str, int], bool]],
    offline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply the H/M/C/C16 interpretation rules and return the report."""
    missing = [identifier for identifier in CONFIG_IDS if identifier not in rows]
    if missing:
        raise ValueError(f"missing attribution config rows: {missing}")
    h, m, c, c16 = (rows[identifier] for identifier in CONFIG_IDS)
    pairs = {
        "h_vs_m": paired_comparison(h, m),
        "m_vs_c": paired_comparison(m, c),
        "c16_vs_c": paired_comparison(c16, c),
    }
    conclusions = {
        "h_vs_m": relation(pairs["h_vs_m"]),
        "m_vs_c": relation(pairs["m_vs_c"]),
        "c16_vs_c": relation(pairs["c16_vs_c"]),
    }
    findings: list[str] = []
    if conclusions["h_vs_m"] == "approx" and conclusions["m_vs_c"] == "a_gt_b":
        findings.append("mask_problem: protection choice dominates; historical runtime irrelevant")
    elif conclusions["h_vs_m"] == "a_gt_b":
        findings.append("historical_runtime_dominates: rotation/A8-policy/quantizer carry main")
    if conclusions["m_vs_c"] == "a_gt_b":
        findings.append("main_mask_beats_candidate_mask")
    else:
        findings.append("candidate_mask_not_worse: grow-from-full-W4 direction not refuted")
    if conclusions["c16_vs_c"] == "approx":
        findings.append("a8_exonerated: problem is W4 or state drift, not dynamic A8")
    elif conclusions["c16_vs_c"] == "a_gt_b":
        findings.append("a16_recovers: dynamic A8 unstable on candidate states -> bounded dynamic A8")
    else:
        findings.append("a16_worse: dynamic A8 helps; keep dynamic A8 frozen")

    proxy_dead = False
    if offline:
        order = sorted(
            offline,
            key=lambda identifier: float(offline[identifier]),
        )
        offline_prefers_c = order.index("c") < order.index("m")
        if offline_prefers_c and conclusions["m_vs_c"] == "a_gt_b":
            findings.append("proxy_dead: offline ranks C above M but closed loop shows M >> C")
            proxy_dead = True
    return {
        "kind": "full_context_four_config_attribution_report",
        "paired": pairs,
        "conclusions": conclusions,
        "findings": findings,
        "proxy_dead": proxy_dead,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attribution-root", required=True, help="dir with per-task-set result jsonl"
    )
    parser.add_argument(
        "--offline-json",
        default=None,
        help="optional offline score map {m: obj, c: obj} for the proxy-dead check",
    )
    parser.add_argument("--out", default=None, help="write the report JSON here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = {
        identifier: load_rows(args.attribution_root, identifier)
        for identifier in CONFIG_IDS
    }
    offline = None
    if args.offline_json:
        payload = json.loads(Path(args.offline_json).read_text(encoding="utf-8"))
        offline = payload.get("objectives") or payload.get("scores")
    report = interpret(rows, offline=offline)
    if args.out:
        atomic_json(Path(args.out).expanduser().resolve(), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()