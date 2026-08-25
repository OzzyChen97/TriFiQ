#!/usr/bin/env python3
"""Synthetic fail-closed tests for the corrected paper-faithful Table-1 gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from openpi_client.paired_noise import PROTOCOL as NOISE_PROTOCOL

from pi05_audit_corrected_table1_completion import verify_corrected_manifest_invariants
from pi05_make_corrected_table1_manifest import (
    CHECKPOINT_SHA256,
    CONFIGS,
    validate_runtime,
)


HASHES = {
    "checkpoint": CHECKPOINT_SHA256,
    "candidate_inventory": "1" * 64,
    "inventory": "2" * 64,
    "pack_manifest": "3" * 64,
    "calibration_buffer": "4" * 64,
    "full_w4a8_plan": "5" * 64,
    "full_w4a8_a8": "6" * 64,
    "full_w4a8_atm_ohb": "7" * 64,
    "gdsq_plan": "8" * 64,
    "gdsq_a8": "9" * 64,
    "gdsq_atm_ohb": "a" * 64,
}


def runtime(config: str) -> dict:
    expected = CONFIGS[config]
    value = {
        "config_id": config,
        "model_dtype": {
            "resolved": "float16",
            "linear_layers_by_weight_dtype": {"float16": 458},
        },
        "protocol": {
            "action_horizon": 50,
            "replan_steps": 5,
            "flow_steps": 10,
            "split": "pretrain",
            "paired_noise": NOISE_PROTOCOL,
        },
        "duquant": {"enabled": False, "wrapped_layers": 0},
        "atm_ohb": {"enabled": False, "matched_layers": 0},
    }
    if not expected["wrapped"]:
        return value
    is_gdsq = config.startswith("gdsq_vla")
    plan_key = "gdsq_plan" if is_gdsq else "full_w4a8_plan"
    a8_key = "gdsq_a8" if is_gdsq else "full_w4a8_a8"
    atm_key = "gdsq_atm_ohb" if is_gdsq else "full_w4a8_atm_ohb"
    value["duquant"] = {
        "enabled": True,
        "wrapped_layers": expected["wrapped"],
        "act_scales_ready": True,
        "weight_bits": 4,
        "act_bits": 8,
        "block_in": 64,
        "block_out": 64,
        "act_percentile": 99.9,
        "calib_batches": 32,
        "denoising_steps": 10,
        "enable_permute": True,
        "execution_backend": "fake_quant_fp16_gemm",
        "integer_gemm": False,
        "packed_low_bit_residency": False,
        "candidate_inventory_sha256": HASHES["candidate_inventory"],
        "pack_manifest_sha256": HASHES["pack_manifest"],
        "plan_sha256": HASHES[plan_key],
        "act_scale_sha256": HASHES[a8_key],
    }
    if not expected["atm"]:
        return value
    value["atm_ohb"] = {
        "enabled": True,
        "atm_enabled": True,
        "ohb_enabled": True,
        "matched_layers": 18,
        "ohb_layers": 18,
        "atm_application": "fold_q_weight",
        "ohb_mode": "per_layer_post_projection",
        "ohb_application": "fold_o_weight",
        "artifact_sha256": HASHES[atm_key],
        "metadata": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "plan_sha256": HASHES[plan_key],
            "a8_scale_sha256": HASHES[a8_key],
            "calibration_buffer_sha256": HASHES["calibration_buffer"],
            "pack_manifest_sha256": HASHES["pack_manifest"],
            "enable_permute": True,
            "wrapped_layers": expected["wrapped"],
            "frames": 128,
            "flow_steps": 10,
            "scope": "expert",
            "log_clamp": 0.30,
            "alpha_neutral": 0.03,
            "beta_neutral": 0.03,
            "noise_protocol": NOISE_PROTOCOL,
            "atm_application": "fold_q_weight",
            "ohb_mode": "per_layer_post_projection",
            "ohb_application": "fold_o_weight",
            "ohb_capture_point": "post_o_proj_pre_residual",
        },
    }
    return value


def protocol() -> dict:
    return {
        "split": "pretrain",
        "task_sets": {
            "atomic_seen": [f"a{i}" for i in range(18)],
            "composite_seen": [f"s{i}" for i in range(16)],
            "composite_unseen": [f"u{i}" for i in range(16)],
        },
        "trial_seeds": list(range(50)),
        "fresh_environment_per_trial": True,
        "render_enabled": True,
        "official_task_horizon": True,
        "action_horizon": 50,
        "replan_steps": 5,
        "flow_steps": 10,
        "paired_action_noise": True,
        "calibration_source": "real-on-policy-robocasa-pi05",
        "calibration_steps": 128,
        "max_calibration_trials_per_task": 5,
    }


class CorrectedTable1GateTest(unittest.TestCase):
    def test_all_four_canonical_runtimes_are_accepted(self) -> None:
        for config in CONFIGS:
            validate_runtime(config, runtime(config), HASHES)

    def test_wrong_plan_and_a8_are_rejected(self) -> None:
        value = runtime("gdsq_vla")
        value["duquant"]["plan_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "plan hash"):
            validate_runtime("gdsq_vla", value, HASHES)
        value = runtime("gdsq_vla")
        value["duquant"]["act_scale_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "A8 hash"):
            validate_runtime("gdsq_vla", value, HASHES)

    def test_permutation_and_custom_ohb_are_rejected(self) -> None:
        value = runtime("quantvla_w4a8_atmohb")
        value["duquant"]["enable_permute"] = False
        with self.assertRaisesRegex(ValueError, "enable_permute"):
            validate_runtime("quantvla_w4a8_atmohb", value, HASHES)
        value = runtime("quantvla_w4a8_atmohb")
        value["atm_ohb"]["ohb_mode"] = "per_head_pre_projection"
        with self.assertRaisesRegex(ValueError, "ohb_mode"):
            validate_runtime("quantvla_w4a8_atmohb", value, HASHES)

    def test_non_fp16_linear_weights_are_rejected(self) -> None:
        value = runtime("fp16")
        value["model_dtype"]["linear_layers_by_weight_dtype"]["float32"] = 1
        with self.assertRaisesRegex(ValueError, "non-FP16"):
            validate_runtime("fp16", value, HASHES)

    def test_completion_gate_rechecks_shared_gdsq_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / "inventory.json"
            inventory.write_text(
                json.dumps({"candidate_inventory_sha256": HASHES["candidate_inventory"]}),
                encoding="utf-8",
            )
            artifacts = {
                key: {"path": str(inventory), "sha256": value}
                for key, value in HASHES.items()
                if key != "candidate_inventory"
            }
            manifest = {
                "configs": {config: {} for config in CONFIGS},
                "table_1_protocol": protocol(),
                "quantization_protocol": {
                    "weight_bits": 4,
                    "activation_bits": 8,
                    "block_in": 64,
                    "block_out": 64,
                    "enable_permute": True,
                    "lambda_smooth": 0.15,
                    "activation_percentile": 99.9,
                    "activation_calibration_batches": 32,
                    "atm": "per-head alpha folded into action-expert q_proj",
                    "ohb": "per-layer scalar beta folded into o_proj at the pre-residual interface",
                    "execution_backend": "fake_quant_fp16_gemm",
                    "accuracy_scope_only": True,
                    "paper_integer_kernel_efficiency_reproduced": False,
                },
                "servers": [
                    {"config_id": config, "runtime": runtime(config)} for config in CONFIGS
                ],
                "invalidated_predecessor": {
                    "use": "diagnostic only; rows must not be merged into corrected Table 1"
                },
            }
            result = verify_corrected_manifest_invariants(manifest, artifacts)
            self.assertTrue(result["gdsq_shared_plan_and_a8"])
            broken = copy.deepcopy(manifest)
            target = next(row for row in broken["servers"] if row["config_id"] == "gdsq_vla")
            target["runtime"]["duquant"]["act_scale_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "A8 hash"):
                verify_corrected_manifest_invariants(broken, artifacts)


if __name__ == "__main__":
    unittest.main()
