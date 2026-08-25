#!/usr/bin/env python3
"""Materialize task/spec groupings from an ATM/OHB dynamic selector JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONFIG_ID_BY_MODEL_VARIANT = {
    "gr00t": {
        "baseline": "cscka_final",
        "atm": "cscka_final_atm",
        "ohb": "cscka_final_ohb",
        "atmohb": "cscka_final_atmohb",
    },
    "pi05": {
        "baseline": "gdsq_vla",
        "atm": "gdsq_vla_atm_only",
        "ohb": "gdsq_vla_ohb_only",
        "atmohb": "gdsq_vla_atmohb",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="0-9", help="Comma/range seed spec, default: 0-9")
    parser.add_argument("--heldout-only", action="store_true")
    return parser.parse_args()


def parse_seeds(spec: str) -> list[int]:
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid seed range: {part}")
            seeds.update(range(lo, hi + 1))
        else:
            seeds.add(int(part))
    if not seeds:
        raise ValueError("seed list is empty")
    return sorted(seeds)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_specs(selector: dict[str, Any], seeds: list[int], heldout_only: bool) -> dict[str, Any]:
    models_out: dict[str, Any] = {}
    for model_id, model in sorted((selector.get("models") or {}).items()):
        mapping = CONFIG_ID_BY_MODEL_VARIANT.get(model_id)
        if mapping is None:
            raise ValueError(f"unknown model id: {model_id}")
        grouped = {
            variant: {"config_id": config_id, "tasks": [], "seeds": seeds}
            for variant, config_id in mapping.items()
        }
        rows = []
        for task, decision in sorted((model.get("tasks") or {}).items()):
            if heldout_only and not bool(decision.get("is_heldout")):
                continue
            variant = decision.get("selected_variant")
            if variant not in grouped:
                raise ValueError(f"{model_id}/{task}: invalid selected variant {variant!r}")
            config_id = mapping[variant]
            grouped[variant]["tasks"].append(task)
            rows.append({"task": task, "selected_variant": variant, "config_id": config_id})
        models_out[model_id] = {
            "config_id_mapping": mapping,
            "groups": grouped,
            "task_rows": rows,
        }
    return {
        "schema_version": 1,
        "kind": "atmohb_dynamic_selector_specs",
        "source_selector_rule": selector.get("rule_name"),
        "heldout_only": heldout_only,
        "seeds": seeds,
        "models": models_out,
    }


def write_task_lists(out_dir: Path, specs: dict[str, Any]) -> None:
    for model_id, model in specs["models"].items():
        model_dir = out_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        for variant, group in model["groups"].items():
            path = model_dir / f"{variant}_{group['config_id']}_tasks.txt"
            path.write_text("\n".join(group["tasks"]) + ("\n" if group["tasks"] else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    selector_path = Path(args.selector).resolve()
    out_dir = Path(args.out_dir).resolve()
    seeds = parse_seeds(args.seeds)
    specs = build_specs(read_json(selector_path), seeds, args.heldout_only)
    specs["source_selector"] = str(selector_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / ("specs_heldout.json" if args.heldout_only else "specs.json")
    out_json.write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_task_lists(out_dir, specs)
    print(json.dumps({"out": str(out_json), "heldout_only": args.heldout_only, "seeds": seeds}))


if __name__ == "__main__":
    main()