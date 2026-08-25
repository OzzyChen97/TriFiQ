#!/usr/bin/env python3
"""Configuration-level common-key comparison for GDSQ-VLA runtime selector runs.

The runtime selector is applied only to the GDSQ-VLA runtime-selector row.
All static baselines are loaded from their normal evaluation rows.  During an
incomplete run, every static row is restricted to the exact same (task, seed)
keys already completed by the runtime-selector row for that model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTOR = REPO_ROOT / "runs/atmohb_dynamic_selector_v8/selector.json"
DEFAULT_FINAL_ATOMIC_ROOT = REPO_ROOT / "runs/atmohb_seeded_atomic_10x10/final/raw"
DEFAULT_GR00T_ATOMIC_ROOT = DEFAULT_FINAL_ATOMIC_ROOT / "gr00t"
DEFAULT_PI05_ATOMIC_ROOT = DEFAULT_FINAL_ATOMIC_ROOT / "pi05"
SEEDS = set(range(50))
EXPECTED_EPISODES = 18 * 50

CONFIG_TABLE = {
    "gr00t": [
        {
            "name": "FP16",
            "config_id": "fp16",
            "selector_applied": False,
            "mode": "normal_static_eval",
            "roots": [DEFAULT_GR00T_ATOMIC_ROOT],
        },
        {
            "name": "GDSQ-VLA",
            "config_id": "cscka_final",
            "selector_applied": False,
            "mode": "normal_static_eval",
            "roots": [DEFAULT_GR00T_ATOMIC_ROOT],
        },
        {
            "name": "GDSQ-VLA + ATM",
            "config_id": "cscka_final_atm",
            "selector_applied": False,
            "mode": "normal_static_eval",
            "roots": [DEFAULT_GR00T_ATOMIC_ROOT],
        },
        {
            "name": "GDSQ-VLA + Runtime Selector",
            "config_id": "cscka_final_runtime_selector",
            "selector_applied": True,
            "mode": "gdsq_vla_runtime_selector_eval",
            "runtime_subdir": "final/raw/gr00t",
        },
    ],
    "pi05": [
        {
            "name": "FP16",
            "config_id": "fp16",
            "selector_applied": False,
            "mode": "normal_static_eval",
            "roots": [DEFAULT_PI05_ATOMIC_ROOT],
        },
        {
            "name": "GDSQ-VLA",
            "config_id": "gdsq_vla",
            "selector_applied": False,
            "mode": "normal_static_eval",
            "roots": [DEFAULT_PI05_ATOMIC_ROOT],
        },
        {
            "name": "GDSQ-VLA + OHB",
            "config_id": "gdsq_vla_ohb_only",
            "selector_applied": False,
            "mode": "normal_static_eval",
            "roots": [DEFAULT_PI05_ATOMIC_ROOT],
        },
        {
            "name": "GDSQ-VLA + Runtime Selector",
            "config_id": "gdsq_vla_runtime_selector",
            "selector_applied": True,
            "mode": "gdsq_vla_runtime_selector_eval",
            "runtime_subdir": "final/raw/pi05",
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Runtime selector validation root.")
    parser.add_argument("--selector", default=str(DEFAULT_SELECTOR))
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--markdown-out", default=None, help="Optional Markdown summary path.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("**/*.jsonl")):
        if "efficiency" in path.name:
            continue
        if path.name.startswith("gpu_") or path.name.startswith("server_"):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_number}") from exc
            row.setdefault("_source_file", str(path))
            rows.append(row)
    return rows


def selector_payload(selector_path: Path) -> dict[str, Any]:
    payload = read_json(selector_path)
    payload["_sha256"] = sha256_file(selector_path)
    return payload


def selector_tasks(selector: dict[str, Any], model_id: str) -> set[str]:
    tasks = (selector.get("models") or {}).get(model_id, {}).get("tasks") or {}
    if not tasks:
        raise ValueError(f"selector has no tasks for model {model_id}")
    return set(map(str, tasks))


def dedupe_by_task_seed(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    deduped: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("success") is None:
            continue
        task = row.get("task")
        seed = row.get("seed")
        if task is None or seed is None:
            continue
        key = (str(task), int(seed))
        if key in deduped:
            duplicates.append({"task": key[0], "seed": key[1], "source_file": row.get("_source_file")})
            continue
        deduped[key] = row
    return deduped, duplicates


def load_config_map(runtime_root: Path, selector: dict[str, Any], model_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    tasks = selector_tasks(selector, model_id)
    if spec.get("selector_applied"):
        roots = [runtime_root / spec["runtime_subdir"]]
        raw_rows = [row for root in roots for row in read_jsonl_rows(root)]
        rows = [
            row for row in raw_rows
            if row.get("config") == spec["config_id"]
            and str(row.get("task")) in tasks
            and row.get("seed") is not None
            and int(row.get("seed")) in SEEDS
            and row.get("success") is not None
            and row.get("status", "complete") == "complete"
        ]
    else:
        roots = spec["roots"]
        raw_rows = [row for root in roots for row in read_jsonl_rows(root)]
        rows = [
            row for row in raw_rows
            if row.get("config") == spec["config_id"]
            and str(row.get("task")) in tasks
            and row.get("seed") is not None
            and int(row.get("seed")) in SEEDS
            and row.get("success") is not None
        ]
    row_map, duplicates = dedupe_by_task_seed(rows)
    return {
        "spec": spec,
        "roots": roots,
        "row_map": row_map,
        "duplicates": duplicates,
        "loaded_episode_count": len(row_map),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row.get("success")) for row in rows)
    per_task_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_task_buckets[str(row.get("task"))].append(row)
    per_task = {}
    for task, task_rows in sorted(per_task_buckets.items()):
        task_successes = sum(bool(row.get("success")) for row in task_rows)
        per_task[task] = {
            "successes": task_successes,
            "episodes": len(task_rows),
            "sr": task_successes / len(task_rows) if task_rows else None,
        }
    episodes = len(rows)
    return {
        "successes": successes,
        "episodes": episodes,
        "episode_sr": successes / episodes if episodes else None,
        "task_macro_sr": (
            sum(float(row["sr"]) for row in per_task.values()) / len(per_task)
            if per_task
            else None
        ),
        "completed_tasks": len(per_task),
        "per_task": per_task,
    }


def runtime_server_payload(runtime_root: Path, model_id: str) -> dict[str, Any] | None:
    if model_id == "gr00t":
        candidates = [
            runtime_root / "control/gr00t_runtime_info.json",
            runtime_root / "gr00t_v8/control/runtime_info.json",
        ]
    elif model_id == "pi05":
        candidates = [
            runtime_root / "control/pi05_metadata.json",
            runtime_root / "pi05_v8/control/seeded_pi05_v8.runtime.json",
        ]
        alternatives = sorted((runtime_root / "control/pi05").glob("*.runtime.json"))
        candidates.extend(reversed(alternatives))
    else:
        return None
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    return read_json(path) if path is not None else None


def runtime_server_metadata(runtime_root: Path, model_id: str) -> dict[str, Any] | None:
    data = runtime_server_payload(runtime_root, model_id)
    if data is None:
        return None
    if model_id == "gr00t":
        return data.get("runtime_selector")
    if model_id == "pi05":
        return data.get("runtime_selector") or (data.get("openpi_runtime") or {}).get(
            "runtime_selector"
        )
    return None


def server_metadata_sha256(payload: dict[str, Any], model_id: str) -> str:
    if model_id == "gr00t":
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    elif model_id == "pi05":
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:
        raise ValueError(f"unknown model id: {model_id}")
    return hashlib.sha256(encoded).hexdigest()


def summarize_runtime_attestation(
    runtime_rows: list[dict[str, Any]],
    runtime_root: Path,
    selector: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    expected_sha = selector["_sha256"]
    expected_rule = selector.get("rule_name")
    expected_version = (
        expected_rule.split("_", 1)[0]
        if isinstance(expected_rule, str) and expected_rule.startswith("v")
        else None
    )
    row_sha_counts = Counter(row.get("selector_sha256") for row in runtime_rows)
    row_rule_counts = Counter(row.get("selector_rule_name") for row in runtime_rows)
    row_enabled_bad = [
        {"task": row.get("task"), "seed": row.get("seed")}
        for row in runtime_rows
        if row.get("runtime_selector_enabled") is not True
    ]
    row_sha_bad = [
        {"task": row.get("task"), "seed": row.get("seed"), "selector_sha256": row.get("selector_sha256")}
        for row in runtime_rows
        if row.get("selector_sha256") != expected_sha
    ]
    row_rule_bad = [
        {
            "task": row.get("task"),
            "seed": row.get("seed"),
            "selector_rule_name": row.get("selector_rule_name"),
        }
        for row in runtime_rows
        if row.get("selector_rule_name") != expected_rule
    ]
    server_payload = runtime_server_payload(runtime_root, model_id) or {}
    server_meta = runtime_server_metadata(runtime_root, model_id) or {}
    expected_server_metadata_sha256 = (
        server_metadata_sha256(server_payload, model_id) if server_payload else None
    )
    row_server_metadata_bad = [
        {
            "task": row.get("task"),
            "seed": row.get("seed"),
            "server_metadata_sha256": row.get("server_metadata_sha256"),
        }
        for row in runtime_rows
        if row.get("server_metadata_sha256") != expected_server_metadata_sha256
    ]
    alignment_candidates = [
        runtime_root / "control/cross_model_quantization_alignment.json",
        runtime_root / "cross_model_alignment_v8.json",
    ]
    alignment_path = next(
        (candidate for candidate in alignment_candidates if candidate.exists()),
        alignment_candidates[0],
    )
    alignment = read_json(alignment_path) if alignment_path.exists() else {}
    alignment_verified = alignment.get("verified") is True
    alignment_selector_sha_matches = alignment.get("selector_sha256") == expected_sha
    server_sha = server_meta.get("selector_sha256")
    server_rule = server_meta.get("rule_name")
    server_model = server_meta.get("model_id")
    verified = (
        bool(runtime_rows)
        and expected_version is not None
        and not row_enabled_bad
        and not row_sha_bad
        and not row_rule_bad
        and not row_server_metadata_bad
        and server_sha == expected_sha
        and server_rule == expected_rule
        and server_model == model_id
        and alignment_verified
        and alignment_selector_sha_matches
    )
    return {
        "expected_selector_path": selector.get("source_selector_path"),
        "expected_selector_sha256": expected_sha,
        "expected_rule_name": expected_rule,
        "expected_rule_version": expected_version,
        "runtime_row_count": len(runtime_rows),
        "row_selector_sha256_counts": dict(row_sha_counts),
        "row_selector_rule_name_counts": dict(row_rule_counts),
        "row_all_enabled": not row_enabled_bad,
        "row_enabled_bad_count": len(row_enabled_bad),
        "row_all_sha_match_expected": not row_sha_bad,
        "row_sha_bad_count": len(row_sha_bad),
        "row_sha_bad_examples": row_sha_bad[:20],
        "row_all_rule_match_expected": not row_rule_bad,
        "row_rule_bad_count": len(row_rule_bad),
        "row_rule_bad_examples": row_rule_bad[:20],
        "expected_server_metadata_sha256": expected_server_metadata_sha256,
        "row_all_server_metadata_match_expected": not row_server_metadata_bad,
        "row_server_metadata_bad_count": len(row_server_metadata_bad),
        "row_server_metadata_bad_examples": row_server_metadata_bad[:20],
        "server_metadata": server_meta,
        "server_selector_sha256": server_sha,
        "server_selector_sha256_matches_expected": server_sha == expected_sha,
        "server_rule_name": server_rule,
        "server_rule_name_matches_expected": server_rule == expected_rule,
        "server_model_id": server_model,
        "server_model_id_matches": server_model == model_id,
        "cross_model_alignment_path": str(alignment_path),
        "cross_model_alignment_verified": alignment_verified,
        "cross_model_alignment_selector_sha_matches": alignment_selector_sha_matches,
        "runtime_selector_is_verified": verified,
        "runtime_selector_is_verified_v3": verified and expected_version == "v3",
    }


def build_model_payload(runtime_root: Path, selector: dict[str, Any], model_id: str) -> dict[str, Any]:
    loaded = [load_config_map(runtime_root, selector, model_id, spec) for spec in CONFIG_TABLE[model_id]]
    duplicate_configs = {
        item["spec"]["config_id"]: item["duplicates"]
        for item in loaded
        if item["duplicates"]
    }
    if duplicate_configs:
        raise ValueError(f"{model_id}: duplicate (task, seed) rows: {duplicate_configs}")
    key_sets = [set(item["row_map"]) for item in loaded]
    common_keys = sorted(set.intersection(*key_sets)) if key_sets else []
    runtime_item = next(item for item in loaded if item["spec"].get("selector_applied"))
    runtime_all_rows = [runtime_item["row_map"][key] for key in sorted(runtime_item["row_map"])]
    configs = []
    for item in loaded:
        spec = item["spec"]
        common_rows = [item["row_map"][key] for key in common_keys]
        summary = summarize_rows(common_rows)
        selector_variant_counts = Counter()
        selector_selected_config_counts = Counter()
        if spec.get("selector_applied"):
            selector_variant_counts.update(row.get("selected_variant") for row in common_rows)
            selector_selected_config_counts.update(row.get("selected_config_id") for row in common_rows)
        configs.append(
            {
                "name": spec["name"],
                "config_id": spec["config_id"],
                "selector_applied": bool(spec.get("selector_applied")),
                "mode": spec["mode"],
                "source_roots": [str(path) for path in item["roots"]],
                "loaded_episode_count": item["loaded_episode_count"],
                "expected_episodes": EXPECTED_EPISODES,
                "common_task_seed_episode_count": len(common_keys),
                "successes_on_common_subset": summary["successes"],
                "episode_sr_on_common_subset": summary["episode_sr"],
                "task_macro_sr_on_common_subset": summary["task_macro_sr"],
                "completed_tasks_on_common_subset": summary["completed_tasks"],
                "per_task_on_common_subset": summary["per_task"],
                "selector_variant_counts_on_common_subset": dict(selector_variant_counts),
                "selector_selected_config_counts_on_common_subset": dict(selector_selected_config_counts),
                "duplicates": item["duplicates"],
            }
        )
    return {
        "comparison_contract": (
            "All config rows are compared on the exact same common (task, seed) subset. "
            "The common subset is the intersection across the runtime-selector row and all static baselines."
        ),
        "expected_full_episodes": EXPECTED_EPISODES,
        "common_task_seed_episode_count": len(common_keys),
        "common_task_count": len({task for task, _seed in common_keys}),
        "common_seed_minmax": (
            [min(seed for _task, seed in common_keys), max(seed for _task, seed in common_keys)]
            if common_keys
            else None
        ),
        "common_task_seed_examples": [{"task": task, "seed": seed} for task, seed in common_keys[:20]],
        "runtime_selector_attestation": summarize_runtime_attestation(runtime_all_rows, runtime_root, selector, model_id),
        "configs": configs,
    }


def build_payload(runtime_root: Path, selector_path: Path) -> dict[str, Any]:
    selector = selector_payload(selector_path)
    selector["source_selector_path"] = str(selector_path)
    models = {
        model_id: build_model_payload(runtime_root, selector, model_id)
        for model_id in CONFIG_TABLE
    }
    aligned_selector = (
        (selector.get("fit_metadata") or {}).get(
            "cross_model_quantization_config_aligned"
        )
        is True
    )
    if aligned_selector:
        failed = {
            model_id: model["runtime_selector_attestation"]
            for model_id, model in models.items()
            if model["runtime_selector_attestation"]["runtime_row_count"] > 0
            and not model["runtime_selector_attestation"]["runtime_selector_is_verified"]
        }
        if failed:
            raise ValueError(
                "aligned runtime rows failed selector/server/cross-model quantization attestation: "
                f"{sorted(failed)}"
            )
    return {
        "schema_version": 4,
        "kind": "runtime_selector_common_task_seed_config_comparison",
        "runtime_root": str(runtime_root),
        "selector": str(selector_path),
        "selector_sha256": selector["_sha256"],
        "selector_rule_name": selector.get("rule_name"),
        "cross_model_quantization_alignment_required": aligned_selector,
        "models": models,
    }


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Runtime Selector Common Task/Seed Comparison",
        "",
        "Contract: selector is applied only to the `GDSQ-VLA + Runtime Selector` row. "
        "All static baselines are normal evaluation rows. During a partial run, every row below is computed on the exact same common `(task, seed)` set for that model.",
        "",
        f"Runtime root: `{payload['runtime_root']}`",
        f"Selector: `{payload['selector']}`",
        f"Selector rule: `{payload['selector_rule_name']}`",
        f"Selector SHA256: `{payload['selector_sha256']}`",
        "",
    ]
    for model_id, model in payload["models"].items():
        att = model["runtime_selector_attestation"]
        lines.extend([
            f"## {model_id}",
            "",
            f"Common `(task, seed)` subset: **{model['common_task_seed_episode_count']} episodes**, "
            f"{model['common_task_count']} tasks. Static configs are trimmed to this same subset.",
            f"Runtime selector verified as {att['expected_rule_version'] or 'unknown'}: **{'yes' if att['runtime_selector_is_verified'] else 'no'}** "
            f"(server SHA: {att['server_selector_sha256_matches_expected']}, server rule: {att['server_rule_name_matches_expected']}, "
            f"row SHA: {att['row_all_sha_match_expected']}, row rule: {att['row_all_rule_match_expected']}, "
            f"row server metadata: {att['row_all_server_metadata_match_expected']}, rows enabled: {att['row_all_enabled']}, "
            f"cross-model quant alignment: {att['cross_model_alignment_verified']}).",
            "",
            "| Config | Selector applied? | Loaded rows | Common rows | Success on common rows | Episode SR on common rows | Task-macro SR on common rows | Selector choices on common rows |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in model["configs"]:
            choices = "-"
            if row["selector_applied"]:
                choices = f"`{row['selector_variant_counts_on_common_subset']}`"
            lines.append(
                f"| {row['name']} (`{row['config_id']}`) | "
                f"{'yes' if row['selector_applied'] else 'no'} | "
                f"{row['loaded_episode_count']}/{row['expected_episodes']} | "
                f"{row['common_task_seed_episode_count']} | "
                f"{row['successes_on_common_subset']}/{row['common_task_seed_episode_count']} | "
                f"{fmt_pct(row['episode_sr_on_common_subset'])} | "
                f"{fmt_pct(row['task_macro_sr_on_common_subset'])} | "
                f"{choices} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    selector_path = Path(args.selector).resolve()
    payload = build_payload(root, selector_path)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        write_markdown(payload, Path(args.markdown_out).resolve())
    print(json.dumps({"out": str(out), "kind": payload["kind"], "selector_rule_name": payload["selector_rule_name"]}))


if __name__ == "__main__":
    main()