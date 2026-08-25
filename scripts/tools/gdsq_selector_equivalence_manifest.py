#!/usr/bin/env python3
"""Freeze the v8 selector-equivalence inputs before starting either server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "runs/atmohb_dynamic_selector_v8/selector.json"
EXPECTED_SELECTOR_SHA256 = "0f3178726c2b784898f18dfde248d9a9bae152da6ffcfdc299e8bdc02f0bd871"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def gr00t_buffer_hash() -> str:
    sys.path.insert(0, str(REPO_ROOT / "scripts/tools"))
    from gr00t_v2_common import fixed_calibration_buffer

    _, _, digest = fixed_calibration_buffer(20260823, 256, 16, 12, "robocasa365")
    return digest


def build() -> dict[str, Any]:
    selector_hash = sha256_file(SELECTOR)
    if selector_hash != EXPECTED_SELECTOR_SHA256:
        raise ValueError("frozen v8 selector SHA mismatch")
    gr_root = REPO_ROOT / "checkpoints/packs/robocasa365"
    pi_root = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned"
    return {
        "schema_version": 1,
        "kind": "gdsq_vla_selector_equivalence_preregistration",
        "result_blind": True,
        "frozen_before_server_start": True,
        "selector": artifact(SELECTOR),
        "protocol": {
            "observations_per_model": 256,
            "denoising_steps": 4,
            "paired_noise": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
            "comparison": "numpy.array_equal over every returned action tensor element",
            "pass_rule": "256/256 equal, max_abs=0, no non-finite output",
            "gr00t_reuse_rule": "reuse static full matrix only after pass",
        },
        "frozen_observations": {
            "gr00t": {
                "generator": artifact(REPO_ROOT / "scripts/tools/gr00t_v2_common.py"),
                "seed": 20260823,
                "count": 256,
                "format": "robocasa365",
                "buffer_sha256": gr00t_buffer_hash(),
            },
            "pi05": artifact(
                pi_root / "calibration/pi05_robocasa365_seed0_n256.npz"
            ),
        },
        "configurations": {
            "gr00t_static": {
                "config_id": "cscka_final",
                "selected_variant": "baseline",
                "plan": artifact(
                    gr_root
                    / "gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json"
                ),
                "a8": artifact(gr_root / "a8_scales_cscka_16to1_protocolfix_d4.npz"),
            },
            "gr00t_selector": {
                "config_id": "cscka_final_runtime_selector",
                "expected_selected_config_id": "cscka_final",
                "expected_variant": "baseline",
                "atm_ohb_superset": artifact(
                    gr_root / "atm_alpha_beta_static_cscka_16to1_protocolfix_d4.json"
                ),
            },
            "pi05_static": {
                "config_id": "gdsq_vla_ohb_only",
                "selected_variant": "ohb",
                "plan": artifact(pi_root / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"),
                "a8": artifact(pi_root / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz"),
                "atm_ohb": artifact(
                    pi_root / "atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json"
                ),
            },
            "pi05_selector": {
                "config_id": "gdsq_vla_runtime_selector",
                "expected_selected_config_id": "gdsq_vla_ohb_only",
                "expected_variant": "ohb",
            },
        },
        "source": {
            "equivalence_client": artifact(
                REPO_ROOT / "scripts/tools/gdsq_selector_equivalence.py"
            ),
            "gr00t_selector_runtime": artifact(
                REPO_ROOT / "code/gr00t/atm/runtime_selector.py"
            ),
            "pi05_selector_runtime": artifact(
                REPO_ROOT / "code/pi05/openpi/src/openpi/quant/atm_runtime_selector.py"
            ),
            "gr00t_server_launcher": artifact(
                REPO_ROOT / "scripts/run_robocasa365_quant_serve.sh"
            ),
            "pi05_server_launcher": artifact(REPO_ROOT / "scripts/run_pi05_formal_server.sh"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {out}")
    payload = build()
    atomic_json(out, payload)
    print(json.dumps({"out": str(out), "sha256": sha256_file(out)}, indent=2))


if __name__ == "__main__":
    main()
