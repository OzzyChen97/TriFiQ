#!/usr/bin/env python3
"""Freeze the corrected paper-faithful four-config π0.5 Table-1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import robocasa  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from openpi_client.paired_noise import PROTOCOL as NOISE_PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "runs/pi05_quantvla_paper"
CONFIGS = {
    "fp16": {"wrapped": 0, "atm": False},
    "quantvla_w4a8_atmohb": {"wrapped": 180, "atm": True},
    "gdsq_vla_atmohb": {"wrapped": 69, "atm": True},
    "gdsq_vla": {"wrapped": 69, "atm": False},
}
CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def artifact(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--server", action="append", required=True, help="INSTANCE,CONFIG,GPU,PORT,RUNTIME_JSON"
    )
    parser.add_argument("--paper-root", default=str(PAPER_ROOT))
    return parser.parse_args()


def validate_runtime(config: str, runtime: dict, artifact_hashes: dict[str, str]) -> None:
    """Reject a mislabeled server before its metadata can enter the manifest."""
    if config not in CONFIGS:
        raise ValueError(f"unknown corrected Table-1 config: {config}")
    if runtime.get("config_id") != config:
        raise ValueError(f"runtime config mismatch: {runtime.get('config_id')!r} != {config!r}")
    expected = CONFIGS[config]
    duquant = runtime.get("duquant") or {}
    scaling = runtime.get("atm_ohb") or {}
    dtype = runtime.get("model_dtype") or {}
    protocol = runtime.get("protocol") or {}
    if dtype.get("resolved") != "float16":
        raise ValueError(f"{config}: runtime is not strict FP16")
    linear_dtypes = dtype.get("linear_layers_by_weight_dtype") or {}
    if not linear_dtypes.get("float16") or any(
        count for name, count in linear_dtypes.items() if name != "float16"
    ):
        raise ValueError(f"{config}: non-FP16 Linear weights are present")
    for key, value in {
        "action_horizon": 50,
        "replan_steps": 5,
        "flow_steps": 10,
        "split": "pretrain",
        "paired_noise": NOISE_PROTOCOL,
    }.items():
        if protocol.get(key) != value:
            raise ValueError(f"{config}: runtime protocol {key} mismatch")
    if int(duquant.get("wrapped_layers", 0)) != expected["wrapped"]:
        raise ValueError(f"{config}: wrapped-layer mismatch")
    if not expected["wrapped"]:
        if bool(duquant.get("enabled")):
            raise ValueError("fp16: DuQuant is unexpectedly enabled")
        if bool(scaling.get("enabled")):
            raise ValueError("fp16: ATM/OHB is unexpectedly enabled")
        return

    for key, value in {
        "enabled": True,
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
    }.items():
        if duquant.get(key) != value:
            raise ValueError(f"{config}: DuQuant runtime {key} mismatch")
    if duquant.get("candidate_inventory_sha256") != artifact_hashes["candidate_inventory"]:
        raise ValueError(f"{config}: candidate inventory hash mismatch")
    if duquant.get("pack_manifest_sha256") != artifact_hashes["pack_manifest"]:
        raise ValueError(f"{config}: pack manifest hash mismatch")

    is_gdsq = config.startswith("gdsq_vla")
    plan_key = "gdsq_plan" if is_gdsq else "full_w4a8_plan"
    a8_key = "gdsq_a8" if is_gdsq else "full_w4a8_a8"
    atm_key = "gdsq_atm_ohb" if is_gdsq else "full_w4a8_atm_ohb"
    if duquant.get("plan_sha256") != artifact_hashes[plan_key]:
        raise ValueError(f"{config}: plan hash mismatch")
    if duquant.get("act_scale_sha256") != artifact_hashes[a8_key]:
        raise ValueError(f"{config}: A8 hash mismatch")

    if bool(scaling.get("enabled")) != expected["atm"]:
        raise ValueError(f"{config}: ATM/OHB enablement mismatch")
    if not expected["atm"]:
        return
    for key, value in {
        "atm_enabled": True,
        "ohb_enabled": True,
        "matched_layers": 18,
        "ohb_layers": 18,
        "atm_application": "fold_q_weight",
        "ohb_mode": "per_layer_post_projection",
        "ohb_application": "fold_o_weight",
    }.items():
        if scaling.get(key) != value:
            raise ValueError(f"{config}: ATM/OHB runtime {key} mismatch")
    if scaling.get("artifact_sha256") != artifact_hashes[atm_key]:
        raise ValueError(f"{config}: ATM/OHB artifact hash mismatch")
    metadata = scaling.get("metadata") or {}
    for key, value in {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "plan_sha256": artifact_hashes[plan_key],
        "a8_scale_sha256": artifact_hashes[a8_key],
        "calibration_buffer_sha256": artifact_hashes["calibration_buffer"],
        "pack_manifest_sha256": artifact_hashes["pack_manifest"],
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
    }.items():
        if metadata.get(key) != value:
            raise ValueError(f"{config}: ATM/OHB metadata {key} mismatch")


def parse_server(value: str, artifact_hashes: dict[str, str]) -> dict:
    instance, config, gpu_text, port_text, runtime_text = value.split(",", 4)
    runtime_path = Path(runtime_text).resolve()
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime = payload.get("openpi_runtime") or {}
    try:
        validate_runtime(config, runtime, artifact_hashes)
    except ValueError as error:
        raise ValueError(f"{instance}: {error}") from error
    return {
        "instance": instance,
        "config_id": config,
        "gpu": int(gpu_text),
        "port": int(port_text),
        "runtime_path": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": canonical_hash(payload),
        "runtime": runtime,
    }


def atomic_write(path: Path, payload: dict) -> None:
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


def main() -> None:
    args = parse_args()
    paper_root = Path(args.paper_root).resolve()
    artifact_paths = {
        "checkpoint": REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors",
        "checkpoint_config": REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/config.json",
        "norm_stats": REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/assets/pi05_pretrain_human300/norm_stats.json",
        "inventory": REPO_ROOT / "runs/pi05_gdsq_port/plans/pi05_candidate_inventory.json",
        "pack_manifest": paper_root / "packs/pi05_robocasa_block64_permute_w4a8_ls015/manifest.json",
        "calibration_buffer": paper_root / "calibration/robocasa_real_observations_128.npz",
        "calibration_buffer_sidecar": paper_root / "calibration/robocasa_real_observations_128.npz.json",
        "full_w4a8_plan": paper_root / "plans/pi05_quantvla_paper_w4a8.plan.json",
        "full_w4a8_a8": paper_root / "a8/pi05_quantvla_paper_real32_p999_b32.npz",
        "full_w4a8_a8_sidecar": paper_root / "a8/pi05_quantvla_paper_real32_p999_b32.npz.json",
        "full_w4a8_atm_ohb": paper_root / "atm_ohb/pi05_quantvla_paper_real128_scalar.json",
        "gdsq_plan": paper_root / "plans/pi05_gdsq_vla_final_paper.plan.json",
        "gdsq_a8": paper_root / "a8/pi05_gdsq_vla_final_real32_p999_b32.npz",
        "gdsq_a8_sidecar": paper_root / "a8/pi05_gdsq_vla_final_real32_p999_b32.npz.json",
        "gdsq_atm_ohb": paper_root / "atm_ohb/pi05_gdsq_vla_final_real128_scalar.json",
        "parent_gdsq_plan": REPO_ROOT / "runs/pi05_gdsq_final/final/pi05_gdsq_vla_final.plan.json",
        "selector": REPO_ROOT / "scripts/tools/pi05_select_plan.py",
        "functional_metric": REPO_ROOT / "scripts/tools/pi05_func_metrics.py",
        "buffer_collector": REPO_ROOT / "scripts/tools/pi05_collect_real_calibration_buffer.py",
        "a8_calibrator": REPO_ROOT / "scripts/tools/pi05_calibrate_a8.py",
        "atm_ohb_calibrator": REPO_ROOT / "scripts/tools/pi05_calibrate_atm_ohb.py",
        "artifact_auditor": REPO_ROOT / "scripts/tools/pi05_audit_paper_quantvla.py",
        "server_launcher": REPO_ROOT / "scripts/run_pi05_paper_server.sh",
        "evaluator": REPO_ROOT / "scripts/run_robocasa365_pi05_eval.py",
        "aggregator": REPO_ROOT / "scripts/tools/aggregate_pi05_robocasa365.py",
        "manifest_builder": REPO_ROOT / "scripts/tools/pi05_make_corrected_table1_manifest.py",
        "schedule_builder": REPO_ROOT / "scripts/tools/pi05_make_parallel_schedule.py",
        "completion_auditor": REPO_ROOT / "scripts/tools/pi05_audit_corrected_table1_completion.py",
        "reporter": REPO_ROOT / "scripts/tools/render_pi05_table1_report.py",
        "corrected_runner": REPO_ROOT / "scripts/run_pi05_corrected_table1.sh",
    }
    artifacts = {name: artifact(path) for name, path in artifact_paths.items()}
    if artifacts["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash does not match the frozen pi0.5 checkpoint")
    artifact_hashes = {name: row["sha256"] for name, row in artifacts.items()}
    inventory_payload = json.loads(
        Path(artifacts["inventory"]["path"]).read_text(encoding="utf-8")
    )
    artifact_hashes["candidate_inventory"] = inventory_payload.get(
        "candidate_inventory_sha256"
    )
    if not artifact_hashes["candidate_inventory"]:
        raise ValueError("candidate inventory is missing its canonical layer-list hash")
    pack_manifest = json.loads(
        Path(artifacts["pack_manifest"]["path"]).read_text(encoding="utf-8")
    )
    pack_directory = Path(artifacts["pack_manifest"]["path"]).parent
    pack_files = pack_manifest.get("files") or []
    if len(pack_files) != 180:
        raise ValueError("paper pack manifest does not contain exactly 180 files")
    for index, pack_row in enumerate(pack_files):
        record = artifact(pack_directory / pack_row["file"])
        if record["sha256"] != pack_row.get("sha256"):
            raise ValueError(f"paper pack file hash mismatch: {pack_row['file']}")
        record["layer"] = pack_row["layer"]
        artifacts[f"pack_file_{index:03d}"] = record
    servers = [parse_server(value, artifact_hashes) for value in args.server]
    if {row["config_id"] for row in servers} != set(CONFIGS):
        raise ValueError("corrected Table-1 manifest requires all four configs")
    if len({row["instance"] for row in servers}) != len(servers):
        raise ValueError("duplicate server instance")
    payload = {
        "schema_version": 2,
        "immutable": True,
        "kind": "pi05_corrected_paper_faithful_table1",
        "benchmark": "RoboCasa365",
        "table_1_protocol": {
            "split": "pretrain",
            "task_sets": {
                name: list(TASK_SET_REGISTRY[name])
                for name in ("atomic_seen", "composite_seen", "composite_unseen")
            },
            "task_counts": {"atomic_seen": 18, "composite_seen": 16, "composite_unseen": 16},
            "trial_seeds": list(range(50)),
            "fresh_environment_per_trial": True,
            "render_enabled": True,
            "official_task_horizon": True,
            "state_dim": 16,
            "action_dim": 12,
            "action_horizon": 50,
            "replan_steps": 5,
            "flow_steps": 10,
            "paired_action_noise": True,
            "paired_action_noise_protocol": NOISE_PROTOCOL,
            "calibration_source": "real-on-policy-robocasa-pi05",
            "calibration_steps": 128,
            "max_calibration_trials_per_task": 5,
        },
        "configs": {
            "fp16": {"wrapped_layers": 0, "atm_ohb": False},
            "quantvla_w4a8_atmohb": {"wrapped_layers": 180, "atm_ohb": True},
            "gdsq_vla_atmohb": {"wrapped_layers": 69, "atm_ohb": True},
            "gdsq_vla": {"wrapped_layers": 69, "atm_ohb": False},
        },
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
        "artifacts": artifacts,
        "servers": servers,
        "invalidated_predecessor": {
            "path": str((REPO_ROOT / "runs/pi05_gdsq_final/official_pretrain_paired50").resolve()),
            "reason": "full-W4 baseline used permute=false, synthetic calibration and custom per-head OHB",
            "use": "diagnostic only; rows must not be merged into corrected Table 1",
        },
    }
    output = Path(args.out).resolve()
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable corrected manifest changed: {output}")
        print(f"corrected Table-1 manifest verified unchanged: {output}")
        return
    atomic_write(output, payload)
    print(f"corrected Table-1 manifest created: {output}")
    print(f"sha256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
