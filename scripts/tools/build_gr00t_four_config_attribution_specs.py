#!/usr/bin/env python3
"""Build frozen four-config attribution specs (H / M / C / C16) for GR00T.

The specs reuse the already-consumed quick tasks and seeds 50-59 as
development diagnostics. H and C are copied verbatim from the frozen
quick manifests; M and C16 are derived:

* H   historical main mask on the historical main runtime (static A8);
* M   historical main mask on the common runtime (Hessian group64,
     row_rotation=0, dynamic A8, corrections off);
* C   v1 frozen candidate mask on the common runtime;
* C16 candidate mask with FP16 activations (A16 diagnostic).

Placement (gpu/port) is intentionally omitted; the launcher rewrites it
at execution time. Specs are frozen: editing a spec after the run
invalidates its manifest lineage.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from quantvla_cross_model_protocol import sha256_file
from quantvla_outputimpact import atomic_json

TASK_SETS = ("atomic_seen", "composite_seen", "composite_unseen")

PLACEMENT_FIELDS = (
    "gpu",
    "port",
    "egl_device",
    "result_files",
    "shard_egl_devices",
    "config_sha256",
)


def strip_placement(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in PLACEMENT_FIELDS
    }


def build_spec(quick_root: Path, task_set: str) -> dict[str, Any]:
    manifest_path = quick_root / task_set / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {str(row["id"]): row for row in manifest["configs"]}
    if set(rows) != {"gdsq_main", "full_context_v1"}:
        raise ValueError(f"{manifest_path}: unexpected quick config inventory")
    h = strip_placement(rows["gdsq_main"])
    c = strip_placement(rows["full_context_v1"])
    if h.get("meta", {}).get("role") != "frozen_gdsq_vla_main":
        raise ValueError(f"{manifest_path}: gdsq_main role drift")
    if c.get("meta", {}).get("role") != "frozen_full_context_candidate":
        raise ValueError(f"{manifest_path}: full_context_v1 role drift")
    h.update({"id": "h", "source_config_id": "gdsq_main"})
    c.update({"id": "c", "source_config_id": "full_context_v1"})

    m = copy.deepcopy(h)
    m.update(
        {
            "id": "m",
            "packdir": copy.deepcopy(c["packdir"]),
            "act_scale": None,
            "hessian_w4": copy.deepcopy(c["hessian_w4"]),
            "activation_mode": "dynamic_a8",
            "atm": None,
            "ohb": False,
            "ohb_only": False,
            "meta": {"role": "historical_main_mask_common_runtime"},
        }
    )
    c16 = copy.deepcopy(c)
    c16.update(
        {
            "id": "c16",
            "activation_mode": "fp16",
            "meta": {"role": "candidate_mask_a16_diagnostic"},
        }
    )
    return {
        "schema_version": 1,
        "kind": "full_context_four_config_attribution_spec",
        "purpose": "h_m_c_c16_attribution",
        "diagnostic_only": True,
        "task_set": task_set,
        "seeds": "50-59",
        "source_quick_manifest": str(manifest_path),
        "source_quick_manifest_sha256": sha256_file(manifest_path),
        "configs": [h, m, c, c16],
        "comparisons": {
            "pairs": [["h", "m"], ["m", "c"], ["c16", "c"]],
            "unit": "paired_task_seed",
        },
        "decision": {
            "role": "development_diagnostic",
            "not_eligible_for_table1": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick-root",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v1/gr00t/quick_real",
    )
    parser.add_argument(
        "--out-dir",
        default="/home1/gyy/vla/QuantVLA/runs/full_context_v1/gr00t/attribution/specs",
    )
    args = parser.parse_args()
    quick_root = Path(args.quick_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_set in TASK_SETS:
        spec = build_spec(quick_root, task_set)
        atomic_json(out_dir / f"{task_set}.json", spec)
    print(json.dumps({"specs": str(out_dir), "task_sets": list(TASK_SETS)}, indent=2))


if __name__ == "__main__":
    main()