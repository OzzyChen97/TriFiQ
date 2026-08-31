#!/usr/bin/env python3
"""Fail closed unless one Table 6 LIBERO cell has exact episode coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument(
        "--task-ids",
        help="Optional comma-separated task IDs; overrides --tasks range",
    )
    parser.add_argument("--offset", type=int, default=10)
    parser.add_argument("--require-paired-action-noise", action="store_true")
    parser.add_argument("--groot-replan-steps", type=int)
    parser.add_argument("--replan-steps", type=int)
    parser.add_argument("--policy-backend")
    parser.add_argument("--action-noise-generator")
    parser.add_argument("--server-config-id")
    parser.add_argument("--server-flow-steps", type=int)
    args = parser.parse_args()

    if not args.summary.is_file():
        raise SystemExit(f"missing summary: {args.summary}")
    value = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = value.get("episode_summaries") or []
    task_ids = (
        [int(item) for item in args.task_ids.split(",") if item.strip()]
        if args.task_ids
        else list(range(args.tasks))
    )
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise SystemExit("--task-ids must contain unique task IDs")
    expected = {
        (task_id, trial_id)
        for task_id in task_ids
        for trial_id in range(args.offset, args.offset + args.trials)
    }
    keys = [(int(row["task_id"]), int(row["trial_id"])) for row in rows]
    observed = set(keys)
    checks = {
        "episode_count": int(value.get("total_episodes", -1)) == len(expected),
        "episode_rows": len(rows) == len(expected),
        "unique_rows": len(keys) == len(observed),
        "exact_keys": observed == expected,
        "task_rows": len(value.get("task_summaries") or []) == len(task_ids),
        "trials": int(value.get("num_trials_per_task", -1)) == args.trials,
    }
    if args.require_paired_action_noise:
        suite_key = {
            "libero_goal": "goal",
            "libero_spatial": "spatial",
            "libero_object": "object",
            "libero_10": "long",
        }.get(value.get("task_suite_name"))
        checks["paired_action_noise"] = (
            value.get("paired_action_noise") is True
            and bool(value.get("action_noise_protocol_id"))
            and bool(value.get("action_noise_stream"))
            and value.get("action_noise_suite_key") == suite_key
        )
        if value.get("policy_backend") == "openpi_ws":
            server_protocol = value.get("server_runtime_protocol") or {}
            checks["openpi_server_protocol"] = (
                server_protocol.get("protocol_id")
                == value.get("action_noise_protocol_id")
                and int(server_protocol.get("replan_steps", -1))
                == int(value.get("executed_replan_steps", -2))
                and server_protocol.get("paired_noise")
                == "sha256(protocol,suite,task,state,replan,stream)"
            )
    if args.groot_replan_steps is not None:
        checks["groot_replan_steps"] = (
            int(value.get("groot_replan_steps", -1)) == args.groot_replan_steps
        )
    if args.replan_steps is not None:
        checks["executed_replan_steps"] = (
            int(value.get("executed_replan_steps", -1)) == args.replan_steps
        )
    if args.policy_backend is not None:
        checks["policy_backend"] = value.get("policy_backend") == args.policy_backend
    if args.action_noise_generator is not None:
        checks["action_noise_generator"] = (
            value.get("action_noise_generator") == args.action_noise_generator
        )
    if args.server_config_id is not None:
        checks["server_config_id"] = value.get("server_config_id") == args.server_config_id
    if args.server_flow_steps is not None:
        checks["server_flow_steps"] = (
            int((value.get("server_runtime_protocol") or {}).get("flow_steps", -1))
            == args.server_flow_steps
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        missing = sorted(expected - observed)[:20]
        unexpected = sorted(observed - expected)[:20]
        raise SystemExit(
            f"invalid Table 6 cell ({', '.join(failed)}); "
            f"rows={len(rows)} unique={len(observed)} missing={missing} unexpected={unexpected}"
        )
    print(
        f"valid Table 6 cell: episodes={len(rows)} "
        f"successes={int(value.get('total_successes', 0))}"
    )


if __name__ == "__main__":
    main()
