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


def flatten_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    """The runner spec schema uses plain paths, not manifest artifact dicts."""
    for key in (
        "plan",
        "act_scale",
        "hessian_w4",
        "errorfold",
        "omega_pack",
        "omega_calibration_manifest",
        "atm",
        "packdir",
    ):
        value = config.get(key)
        if isinstance(value, dict):
            config[key] = value.get("path")
    return config


def build_spec(
    quick_root: Path, task_set: str, a8_scales: dict[str, Path] | None = None
) -> dict[str, Any]:
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
    if a8_scales and task_set in a8_scales:
        scales = a8_scales[task_set]
        h["act_scale"] = {
            "path": str(scales),
            "sha256": sha256_file(scales),
            "bytes": scales.stat().st_size,
            "regenerated": True,
        }
    h = flatten_artifacts(h)
    c = flatten_artifacts(c)

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


def parse_named(value: str) -> tuple[str, Path]:
    key, separator, raw_path = value.partition("=")
    if not separator or not key or not raw_path:
        raise argparse.ArgumentTypeError("override must be task_set=/path.npz")
    return key, Path(raw_path).expanduser().resolve()


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
    parser.add_argument(
        "--a8-scales",
        action="append",
        type=parse_named,
        default=None,
        help="optional regenerated gdsq_main A8 scales: task_set=/path.npz",
    )
    args = parser.parse_args()
    quick_root = Path(args.quick_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    a8_scales = dict(args.a8_scales) if args.a8_scales else None
    if a8_scales and set(a8_scales) != set(TASK_SETS):
        raise ValueError(f"--a8-scales must cover exactly {TASK_SETS}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_set in TASK_SETS:
        spec = build_spec(quick_root, task_set, a8_scales)
        atomic_json(out_dir / f"{task_set}.json", spec)
    print(json.dumps({"specs": str(out_dir), "task_sets": list(TASK_SETS)}, indent=2))


if __name__ == "__main__":
    main()