#!/usr/bin/env python3
"""Fail closed unless GR00T and pi0.5 use the same quantization contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SHARED_KEYS = (
    "logical_profile",
    "quantization_method",
    "layer_selection_policy",
    "weight_quantizer",
    "activation_quantizer",
    "execution_backend",
    "integer_gemm",
    "packed_low_bit_residency",
    "weight_bits",
    "activation_bits",
    "block_in",
    "block_out",
    "lambda_smooth",
    "activation_percentile",
    "calibration_policy",
    "calibration_batches",
    "calibration_batch_size",
    "calibration_samples",
    "permutation",
    "row_rotation",
    "static_activation_scales",
    "activation_scales_ready",
    "denoising_steps",
    "n_action_steps",
    "replan_steps",
    "paired_noise",
    "selector_loads_atm_and_ohb_superset",
    "correction_application",
    "atm_application",
    "ohb_application",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-runtime", required=True)
    parser.add_argument("--pi05-runtime", required=True)
    parser.add_argument(
        "--selector",
        default=None,
        help="Expected selector JSON. Defaults to the selector_path attested by both servers.",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contract_subset(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in SHARED_KEYS}


def main() -> None:
    args = parse_args()
    gr_path = Path(args.gr00t_runtime).resolve()
    pi_path = Path(args.pi05_runtime).resolve()
    gr = read_json(gr_path)
    pi_payload = read_json(pi_path)
    pi = pi_payload.get("openpi_runtime") or pi_payload

    gr_contract = gr.get("quantization_contract") or {}
    pi_contract = pi.get("cross_model_quantization_contract") or {}
    missing = {
        "gr00t": [key for key in SHARED_KEYS if key not in gr_contract],
        "pi05": [key for key in SHARED_KEYS if key not in pi_contract],
    }
    mismatches = {
        key: {"gr00t": gr_contract.get(key), "pi05": pi_contract.get(key)}
        for key in SHARED_KEYS
        if gr_contract.get(key) != pi_contract.get(key)
    }
    gr_contract_sha256 = canonical_sha256(gr_contract)
    pi_contract_sha256 = canonical_sha256(pi_contract)
    contract_hash_checks = {
        "gr00t_reported_matches_payload": (
            gr.get("quantization_contract_sha256") == gr_contract_sha256
        ),
        "pi05_reported_matches_payload": (
            pi.get("cross_model_quantization_contract_sha256") == pi_contract_sha256
        ),
        "cross_model_contract_hash_equal": gr_contract_sha256 == pi_contract_sha256,
    }

    gr_selector = gr.get("runtime_selector") or {}
    pi_selector = pi.get("runtime_selector") or {}
    selector_path_value = args.selector or gr_selector.get("selector_path")
    selector_path = Path(selector_path_value).expanduser().resolve() if selector_path_value else None
    selector_payload: dict[str, Any] = {}
    selector_sha256 = None
    if selector_path is not None and selector_path.is_file():
        selector_payload = read_json(selector_path)
        selector_sha256 = sha256_file(selector_path)
    declared_contract = (
        (((selector_payload.get("fit_metadata") or {}).get("deployment") or {}).get(
            "shared_quantization_contract"
        ))
        or {}
    )
    selector_contract_missing = [key for key in SHARED_KEYS if key not in declared_contract]
    selector_contract_mismatches = {
        model_id: {
            key: {
                "selector": declared_contract.get(key),
                "runtime": contract.get(key),
            }
            for key in SHARED_KEYS
            if declared_contract.get(key) != contract.get(key)
        }
        for model_id, contract in (("gr00t", gr_contract), ("pi05", pi_contract))
    }
    selector_checks = {
        "both_enabled": gr_selector.get("enabled") is True and pi_selector.get("enabled") is True,
        "both_strict": gr_selector.get("strict") is True and pi_selector.get("strict") is True,
        "model_ids_correct": (
            gr_selector.get("model_id") == "gr00t" and pi_selector.get("model_id") == "pi05"
        ),
        "sha256_nonempty": bool(gr_selector.get("selector_sha256")),
        "sha256_equal": gr_selector.get("selector_sha256") == pi_selector.get("selector_sha256"),
        "rule_equal": gr_selector.get("rule_name") == pi_selector.get("rule_name"),
        "selector_file_exists": selector_path is not None and selector_path.is_file(),
        "selector_file_sha_matches_gr00t": selector_sha256 == gr_selector.get("selector_sha256"),
        "selector_file_sha_matches_pi05": selector_sha256 == pi_selector.get("selector_sha256"),
        "selector_rule_matches_file": (
            selector_payload.get("rule_name") == gr_selector.get("rule_name")
        ),
        "selector_declares_aligned_runtime": (
            (selector_payload.get("fit_metadata") or {}).get(
                "cross_model_quantization_config_aligned"
            )
            is True
        ),
        "selector_declared_contract_complete": not selector_contract_missing,
        "selector_declared_contract_matches_both_runtimes": not any(
            selector_contract_mismatches.values()
        ),
    }
    runtime_artifact_checks = {
        "gr00t_config_is_runtime_selector": gr.get("config_id")
        in {"cscka_final_runtime_selector", "gdsq_vla_runtime_selector"},
        "pi05_config_is_runtime_selector": pi.get("config_id")
        == "gdsq_vla_runtime_selector",
        "gr00t_quantized_layers_positive": int(gr.get("wrapped_layers") or 0) > 0,
        "pi05_quantized_layers_positive": int((pi.get("duquant") or {}).get("wrapped_layers") or 0)
        > 0,
        "gr00t_plan_attested": bool(gr.get("plan_sha256")),
        "pi05_plan_attested": bool((pi.get("duquant") or {}).get("plan_sha256")),
        "gr00t_static_a8_attested": bool(gr.get("act_scale_sha256")),
        "pi05_static_a8_attested": bool((pi.get("duquant") or {}).get("act_scale_sha256")),
        "gr00t_atmohb_attested": bool(gr.get("atm_artifact_sha256")),
        "pi05_atmohb_attested": bool((pi.get("atm_ohb") or {}).get("artifact_sha256")),
    }
    verified = (
        not any(missing.values())
        and not mismatches
        and all(contract_hash_checks.values())
        and all(selector_checks.values())
        and all(runtime_artifact_checks.values())
    )
    result = {
        "schema_version": 2,
        "kind": "cross_model_quantization_alignment",
        "verified": verified,
        "shared_keys": list(SHARED_KEYS),
        "shared_contract": contract_subset(gr_contract) if verified else None,
        "missing": missing,
        "mismatches": mismatches,
        "contract_hash_checks": contract_hash_checks,
        "runtime_contract_sha256": {
            "gr00t": gr_contract_sha256,
            "pi05": pi_contract_sha256,
        },
        "selector_checks": selector_checks,
        "selector_sha256": selector_sha256,
        "selector_rule_name": selector_payload.get("rule_name"),
        "selector_declared_contract": declared_contract or None,
        "selector_contract_missing": selector_contract_missing,
        "selector_contract_mismatches": selector_contract_mismatches,
        "runtime_artifact_checks": runtime_artifact_checks,
        "architecture_specific": {
            "gr00t": {
                "model_path": gr.get("model_path"),
                "plan": gr.get("plan"),
                "plan_sha256": gr.get("plan_sha256"),
                "act_scale_path": gr.get("act_scale_path"),
                "act_scale_sha256": gr.get("act_scale_sha256"),
                "wrapped_layers": gr.get("wrapped_layers"),
                "attention_layers": gr.get("atm_layers") or gr.get("ohb_layers"),
                "atmohb_artifact_sha256": gr.get("atm_artifact_sha256"),
            },
            "pi05": {
                "plan": (pi.get("duquant") or {}).get("plan_path"),
                "plan_sha256": (pi.get("duquant") or {}).get("plan_sha256"),
                "act_scale_path": (pi.get("duquant") or {}).get("act_scale_path"),
                "act_scale_sha256": (pi.get("duquant") or {}).get("act_scale_sha256"),
                "wrapped_layers": (pi.get("duquant") or {}).get("wrapped_layers"),
                "attention_layers": (pi.get("atm_ohb") or {}).get("matched_layers"),
                "atmohb_artifact_sha256": (pi.get("atm_ohb") or {}).get(
                    "artifact_sha256"
                ),
            },
            "note": (
                "Layer names, selected-layer counts, plans, activation-scale files and ATM/OHB "
                "calibration artifacts are architecture-specific. The logical GDSQ-VLA method, "
                "quantizers, bit widths, grouping, smoothing, calibration/application policy, "
                "execution backend and evaluation protocol must match exactly."
            ),
        },
        "source": {
            "gr00t_runtime": str(gr_path),
            "gr00t_runtime_sha256": sha256_file(gr_path),
            "pi05_runtime": str(pi_path),
            "pi05_runtime_sha256": sha256_file(pi_path),
            "selector": str(selector_path) if selector_path is not None else None,
            "selector_sha256": selector_sha256,
        },
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "verified": verified,
                "mismatches": mismatches,
                "selector_checks": selector_checks,
                "runtime_artifact_checks": runtime_artifact_checks,
            },
            sort_keys=True,
        )
    )
    if not verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()