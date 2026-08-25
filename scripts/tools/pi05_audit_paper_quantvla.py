#!/usr/bin/env python3
"""Fail-closed artifact audit for the paper-faithful π0.5 QuantVLA baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "runs/pi05_quantvla_paper"
CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def audit_a8(path: Path, meta_path: Path, *, plan_hash: str, buffer_hash: str, wrapped: int) -> None:
    sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = sidecar.get("metadata") or {}
    for key, value in {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "calibration_buffer_sha256": buffer_hash,
        "plan_sha256": plan_hash,
        "wrapped_layers": wrapped,
        "act_percentile": 99.9,
        "calib_batches": 32,
        "denoising_steps": 10,
        "enable_permute": True,
    }.items():
        if metadata.get(key) != value:
            raise ValueError(f"A8 metadata {key}={metadata.get(key)!r} != {value!r}")
    if sidecar.get("npz_sha256") != sha256_file(path):
        raise ValueError("A8 file hash mismatch")


def audit_atm(
    path: Path,
    *,
    plan_hash: str,
    buffer_hash: str,
    a8_hash: str,
    pack_hash: str,
    wrapped: int,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("meta") or {}
    for key, value in {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "calibration_buffer_sha256": buffer_hash,
        "plan_sha256": plan_hash,
        "a8_scale_sha256": a8_hash,
        "pack_manifest_sha256": pack_hash,
        "wrapped_layers": wrapped,
        "attention_layers": 18,
        "frames": 128,
        "flow_steps": 10,
        "enable_permute": True,
        "ohb_mode": "per_layer_post_projection",
        "ohb_capture_point": "post_o_proj_pre_residual",
        "atm_application": "fold_q_weight",
        "ohb_application": "fold_o_weight",
        "scope": "expert",
        "log_clamp": 0.30,
        "alpha_neutral": 0.03,
        "beta_neutral": 0.03,
    }.items():
        if metadata.get(key) != value:
            raise ValueError(f"ATM/OHB metadata {key}={metadata.get(key)!r} != {value!r}")
    layers = payload.get("layers") or {}
    if len(layers) != 18:
        raise ValueError("paper ATM/OHB artifact does not contain 18 attention layers")
    for name, row in layers.items():
        if len(row.get("all") or []) != 8 or not isinstance(row.get("beta"), (int, float)):
            raise ValueError(f"invalid scalar paper ATM/OHB entry: {name}")
        if "beta_perhead" in row:
            raise ValueError(f"paper artifact contains custom beta_perhead: {name}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    paths = {
        "buffer": root / "calibration/robocasa_real_observations_128.npz",
        "buffer_meta": root / "calibration/robocasa_real_observations_128.npz.json",
        "pack": root / "packs/pi05_robocasa_block64_permute_w4a8_ls015/manifest.json",
        "plan": root / "plans/pi05_quantvla_paper_w4a8.plan.json",
        "a8": root / "a8/pi05_quantvla_paper_real32_p999_b32.npz",
        "a8_meta": root / "a8/pi05_quantvla_paper_real32_p999_b32.npz.json",
        "atm": root / "atm_ohb/pi05_quantvla_paper_real128_scalar.json",
        "gdsq_plan": root / "plans/pi05_gdsq_vla_final_paper.plan.json",
        "gdsq_a8": root / "a8/pi05_gdsq_vla_final_real32_p999_b32.npz",
        "gdsq_a8_meta": root / "a8/pi05_gdsq_vla_final_real32_p999_b32.npz.json",
        "gdsq_atm": root / "atm_ohb/pi05_gdsq_vla_final_real128_scalar.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    buffer_hash = sha256_file(paths["buffer"])
    buffer_meta = json.loads(paths["buffer_meta"].read_text(encoding="utf-8"))
    if buffer_meta.get("sha256") != buffer_hash:
        raise ValueError("calibration buffer hash mismatch")
    expected_buffer = {
        "kind": "real-on-policy-robocasa-pi05",
        "rows": 128,
        "a8_prefix_rows": 32,
        "trials_per_task": 5,
        "max_trials_per_task": 5,
        "calibration_steps": 128,
        "source_policy": "fp16",
        "state_dim": 16,
    }
    for key, value in expected_buffer.items():
        if buffer_meta.get(key) != value:
            raise ValueError(f"buffer metadata {key}={buffer_meta.get(key)!r} != {value!r}")
    source_metadata = buffer_meta.get("source_server_metadata") or {}
    source_runtime = source_metadata.get("openpi_runtime") or {}
    source_duquant = source_runtime.get("duquant") or {}
    source_dtype = source_runtime.get("model_dtype") or {}
    if source_runtime.get("config_id") != "fp16":
        raise ValueError("calibration source runtime is not FP16")
    if bool(source_duquant.get("enabled")) or int(source_duquant.get("wrapped_layers", 0)) != 0:
        raise ValueError("calibration source runtime contains quantized layers")
    if source_dtype.get("resolved") != "float16":
        raise ValueError("calibration source runtime does not use float16")
    source_linear_dtypes = source_dtype.get("linear_layers_by_weight_dtype") or {}
    if not source_linear_dtypes.get("float16") or any(
        count for name, count in source_linear_dtypes.items() if name != "float16"
    ):
        raise ValueError("calibration source runtime contains non-FP16 Linear weights")
    canonical_source = hashlib.sha256(
        json.dumps(source_metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if buffer_meta.get("source_server_metadata_sha256") != canonical_source:
        raise ValueError("calibration source runtime hash mismatch")
    if set(buffer_meta.get("first32_task_counts", {}).values()) != {8}:
        raise ValueError("first 32 A8 rows are not balanced 8-per-task")
    with np.load(paths["buffer"], allow_pickle=False) as archive:
        if len(archive["states"]) != 128 or archive["states"].shape[1:] != (16,):
            raise ValueError("invalid real calibration state matrix")
        if len(set(map(str, archive["prompts"]))) < 4:
            raise ValueError("real calibration prompts are unexpectedly degenerate")

    pack = json.loads(paths["pack"].read_text(encoding="utf-8"))
    expected_pack = {
        "complete": True,
        "wrapped_layer_count": 180,
        "block_in": 64,
        "block_out": 64,
        "lambda_smooth": 0.15,
        "enable_permute": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    for key, value in expected_pack.items():
        if pack.get(key) != value:
            raise ValueError(f"pack {key}={pack.get(key)!r} != {value!r}")
    if len(pack.get("files") or []) != 180:
        raise ValueError("paper pack does not contain 180 attested files")
    pack_directory = paths["pack"].parent
    pack_names = set()
    pack_layers = set()
    for row in pack["files"]:
        file_name = row.get("file")
        layer_name = row.get("layer")
        if not file_name or file_name in pack_names or not layer_name or layer_name in pack_layers:
            raise ValueError("paper pack contains a missing or duplicate file/layer entry")
        pack_names.add(file_name)
        pack_layers.add(layer_name)
        file_path = (pack_directory / file_name).resolve()
        if file_path.parent != pack_directory.resolve() or not file_path.is_file():
            raise ValueError(f"paper pack file is missing or escapes the pack directory: {file_name}")
        if sha256_file(file_path) != row.get("sha256"):
            raise ValueError(f"paper pack file hash mismatch: {file_name}")

    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    plan_meta = plan.get("meta") or {}
    layers = plan.get("layers") or {}
    if len(layers) != 180 or sum(int(row.get("bits", 0)) == 4 for row in layers.values()) != 180:
        raise ValueError("paper plan is not exactly 180 W4 layers")
    for key, value in {
        "paper_faithful": True,
        "enable_permute": True,
        "atm_mode": "per_head_fold_q_weight",
        "ohb_mode": "per_layer_post_projection",
    }.items():
        if plan_meta.get(key) != value:
            raise ValueError(f"plan {key}={plan_meta.get(key)!r} != {value!r}")
    plan_hash = sha256_file(paths["plan"])

    pack_hash = sha256_file(paths["pack"])
    audit_a8(paths["a8"], paths["a8_meta"], plan_hash=plan_hash, buffer_hash=buffer_hash, wrapped=180)
    audit_atm(
        paths["atm"], plan_hash=plan_hash, buffer_hash=buffer_hash,
        a8_hash=sha256_file(paths["a8"]), pack_hash=pack_hash, wrapped=180,
    )

    gdsq_plan = json.loads(paths["gdsq_plan"].read_text(encoding="utf-8"))
    gdsq_meta = gdsq_plan.get("meta") or {}
    gdsq_layers = gdsq_plan.get("layers") or {}
    gdsq_selected = sum(
        not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
        for row in gdsq_layers.values()
    )
    if len(gdsq_layers) != 180 or gdsq_selected != 69:
        raise ValueError("paper GDSQ plan is not 69 W4 + 111 FP16")
    for key, value in {
        "kind": "gdsq_vla_pi05_faithful_final_paper_calibration_frozen",
        "paper_faithful_duquant": True,
        "enable_permute": True,
        "calibration_buffer_sha256": buffer_hash,
        "pack_manifest_sha256": pack_hash,
        "atm_mode": "per_head_fold_q_weight",
        "ohb_mode": "per_layer_post_projection",
        "frozen_before_corrected_table1_test": True,
    }.items():
        if gdsq_meta.get(key) != value:
            raise ValueError(f"GDSQ plan {key}={gdsq_meta.get(key)!r} != {value!r}")
    gdsq_hash = sha256_file(paths["gdsq_plan"])
    audit_a8(
        paths["gdsq_a8"], paths["gdsq_a8_meta"], plan_hash=gdsq_hash,
        buffer_hash=buffer_hash, wrapped=69,
    )
    audit_atm(
        paths["gdsq_atm"], plan_hash=gdsq_hash, buffer_hash=buffer_hash,
        a8_hash=sha256_file(paths["gdsq_a8"]), pack_hash=pack_hash, wrapped=69,
    )

    print(
        json.dumps(
            {
                "status": "paper_artifacts_valid",
                "root": str(root),
                "buffer_sha256": buffer_hash,
                "pack_manifest_sha256": sha256_file(paths["pack"]),
                "plan_sha256": plan_hash,
                "a8_sha256": sha256_file(paths["a8"]),
                "atm_ohb_sha256": sha256_file(paths["atm"]),
                "gdsq_plan_sha256": gdsq_hash,
                "gdsq_a8_sha256": sha256_file(paths["gdsq_a8"]),
                "gdsq_atm_ohb_sha256": sha256_file(paths["gdsq_atm"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
