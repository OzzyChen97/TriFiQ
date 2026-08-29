#!/usr/bin/env python3
"""Materialize the three GR00T four-config specs from frozen v3 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantvla_cross_model_protocol import PROTOCOL, PROTOCOL_SHA256


REPO = Path(__file__).resolve().parents[2]
PACKS = REPO / "checkpoints/packs/robocasa365"

TASKSET_ARTIFACTS = {
    "atomic_seen": {
        "plan": "quantvla_v1_uniform_w4a8.json",
        "pack": "duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015",
        "a8": "a8_scales_quantvla_v1_w4a8_protocolfix_d4.npz",
        "atm": "atm_alpha_beta_static_quantvla_v1_w4a8_protocolfix_d4.json",
    },
    "composite_seen": {
        "plan": "quantvla_v1_uniform_w4a8_composite_seen.json",
        "pack": "duquant_packed_robocasa365_composite_seen_d4_w4a8_b64c32ls015",
        "a8": "a8_scales_quantvla_v1_w4a8_composite_seen_d4.npz",
        "atm": "atm_alpha_beta_static_quantvla_v1_w4a8_composite_seen_d4.json",
    },
    "composite_unseen": {
        "plan": "quantvla_v1_uniform_w4a8_composite_unseen.json",
        "pack": "duquant_packed_robocasa365_composite_unseen_d4_w4a8_b64c32ls015",
        "a8": "a8_scales_quantvla_v1_w4a8_composite_unseen_d4.npz",
        "atm": "atm_alpha_beta_static_quantvla_v1_w4a8_composite_unseen_d4.json",
    },
}


def build(root: Path, task_set: str) -> dict:
    legacy = TASKSET_ARTIFACTS[task_set]
    calibration = root / "calibration/gr00t" / task_set
    plan = PACKS / legacy["plan"]
    wrapped = len(json.loads(plan.read_text(encoding="utf-8"))["layers"])
    placements = {
        # GPU0 has only ~6 GiB free under a long-running external workload.
        # Keep eight server instances by co-locating one FP16 and one real-W4
        # instance on the otherwise empty GPU3; the shared-memory preflight
        # accounts for both server budgets plus its EGL clients.
        "fp16": (1, 22100, 3, 22101),
        "quantvla_w4a8_paper": (2, 22102, 4, 22103),
        "errorfold_dfunc": (5, 22104, 6, 22105),
        "errorfold_dpac_v2": (7, 22106, 3, 22107),
    }

    def placement(config_id: str) -> dict:
        gpu, port, replica_gpu, replica_port = placements[config_id]
        return {
            "id": config_id,
            "gpu": gpu,
            "port": port,
            "replicas": [{"gpu": replica_gpu, "port": replica_port}],
            "egl_device": replica_gpu,
        }

    fp16 = {**placement("fp16"), "expected_wrapped": 0}
    paper = {
        **placement("quantvla_w4a8_paper"),
        "expected_wrapped": wrapped,
        "plan": str(plan),
        "packdir": str(PACKS / legacy["pack"]),
        # Recalibrated with the paper-faithful static-A8 rule, but on the same
        # frozen 256-observation buffer used by both model adapters and v3.
        "act_scale": str(calibration / "paper_a8_shared_n256.npz"),
        "atm": str(PACKS / legacy["atm"]),
        "ohb": True,
        "meta": {"method": "paper_faithful_quantvla_w4a8"},
    }

    def errorfold(config_id: str, artifact: str, metric: str) -> dict:
        return {
            **placement(config_id),
            "expected_wrapped": wrapped,
            "plan": str(plan),
            "packdir": str(calibration / "identity_pack"),
            "act_scale": str(calibration / "a8_scales.npz"),
            "hessian_w4": str(calibration / "hessian_w4.npz"),
            "errorfold": str(calibration / artifact),
            "atm": str(calibration / artifact),
            "ohb": True,
            "atm_application": "fold_q_weight",
            "ohb_application": "fold_o_weight_perhead",
            "meta": {
                "method": "hessian_w4a8_errorfold",
                "selection_metric": metric,
                "selector_free": True,
            },
        }

    return {
        "schema_version": 3,
        "protocol_sha256": PROTOCOL_SHA256,
        "task_set": task_set,
        "configs": [
            fp16,
            paper,
            errorfold("errorfold_dfunc", "errorfold_dfunc.json", "d_func_v1"),
            errorfold(
                "errorfold_dpac_v2", "errorfold_dpac_v2.json", "d_pac_v2"
            ),
        ],
        "comparisons": [
            ["quantvla_w4a8_paper", "fp16"],
            ["errorfold_dfunc", "quantvla_w4a8_paper"],
            ["errorfold_dpac_v2", "quantvla_w4a8_paper"],
            ["errorfold_dpac_v2", "errorfold_dfunc"],
        ],
        "closed_loop_success_used_for_selection": False,
        "episodes_per_config": PROTOCOL["evaluation_matrix"][
            "episodes_per_model_per_config"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for task_set in TASKSET_ARTIFACTS:
        path = output / f"gr00t_{task_set}.json"
        path.write_text(
            json.dumps(build(root, task_set), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
