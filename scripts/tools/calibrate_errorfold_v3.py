#!/usr/bin/env python3
"""Run the shared v3 FP16 -> Hessian-W4A8 -> paired ErrorFold calibration.

The outer procedure, capture limits, artifact builders and provenance checks
are shared.  Only the two functions that load/run a native model are adapter
specific, matching the frozen cross-model boundary.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
OPENPI = REPO / "code/pi05/openpi"
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages/openpi-client/src"))
sys.path.insert(0, str(REPO / "scripts/tools"))

from build_errorfold_artifact import build as build_errorfold  # noqa: E402
from build_hessian_w4_artifact import build as build_hessian  # noqa: E402
from build_v3_a8_artifact import build as build_a8  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL,
    PROTOCOL_SHA256,
    protocol_artifact,
    protocol_attestation,
    sha256_file,
    validate_quant_plan,
)
from quantvla_model_adapters import load_model_records  # noqa: E402
from quantvla_v3_capture import (  # noqa: E402
    AttentionCapture,
    LayerCapture,
    merge_errorfold_with_attention,
    save_npz,
)


DEFAULTS = {
    "gr00t": {
        "checkpoint": REPO / (
            "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
            "target_posttraining/atomic_seen/checkpoint-60000"
        ),
        "plan": REPO / "checkpoints/packs/robocasa365/quantvla_v1_uniform_w4a8.json",
    },
    "pi05": {
        "checkpoint": REPO / "checkpoints/robocasa/pi05_pretrain_human300_pytorch",
        "plan": REPO / (
            "runs/pi05_gdsq_gr00t_aligned/plans/"
            "pi05_quantvla_uniform_w4a8_d4.plan.json"
        ),
    },
}


def _checkpoint_sha256(path: Path, model: str) -> str:
    if model == "pi05":
        return sha256_file(path / "model.safetensors")
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no checkpoint shards under {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8") + b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _ordered_layer_names(model: torch.nn.Module, plan_names: Sequence[str]) -> list[str]:
    requested = set(plan_names)
    names = [name for name, _ in model.named_modules() if name in requested]
    missing = sorted(requested - set(names))
    if missing or len(names) != len(requested):
        raise ValueError(f"adapter target inventory drift: missing={missing[:5]}")
    return names


def _save_attention(path: Path, capture: AttentionCapture) -> None:
    entries = capture.entries()
    arrays: dict[str, np.ndarray] = {
        "names": np.asarray([name for name, _, _ in entries]),
        "kinds": np.asarray([kind for _, kind, _ in entries]),
    }
    for index, (_, _, value) in enumerate(entries):
        arrays[f"value_{index:04d}"] = value
    save_npz(path, arrays)


def _load_attention(path: Path) -> list[tuple[str, str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        names = [str(value) for value in archive["names"].tolist()]
        kinds = [str(value) for value in archive["kinds"].tolist()]
        return [
            (name, kind, np.asarray(archive[f"value_{index:04d}"]))
            for index, (name, kind) in enumerate(zip(names, kinds))
        ]


def _release(value: Any) -> None:
    del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _capture_gr00t(
    *,
    checkpoint: Path,
    plan: Path,
    pack_dir: Path,
    a8: Path,
    hessian: Path,
    records: list[dict[str, Any]],
    checkpoint_hash: str,
    buffer_hash: str,
    plan_hash: str,
    device: str,
    batch_size: int,
    quantized: bool,
) -> tuple[dict[str, np.ndarray], AttentionCapture, dict[str, Any]]:
    from gr00t.atm import (
        clear_atm_capture,
        ensure_dit_attention_patch,
        register_atm_logits_capture,
        register_errorfold_head_capture,
    )
    from gr00t.quantization.dit_step_context import get_current_dit_step
    from gr00t.quantization.duquant_layers import (
        DuQuantLinear,
        load_act_scales,
    )
    from gr00t_sensitivity_probe import run_activations
    from gr00t_v2_common import (
        DEFAULT_EXCLUDE,
        DEFAULT_INCLUDE,
        load_policy,
        set_quant_env,
        strip_quant_env,
    )
    from quantvla_model_adapters import gr00t_rollout_inputs

    strip_quant_env()
    os.environ.pop("QUANTVLA_ADAPTER_ONLY", None)
    if quantized:
        set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, str(pack_dir), row_rot="0")
        os.environ.update(
            {
                "QUANTVLA_ADAPTER_ONLY": "1",
                "GR00T_DUQUANT_PLAN": str(plan),
                "GR00T_DUQUANT_FUSED": "1",
                "GR00T_DUQUANT_ACT_SCALE_PATH": str(a8),
                "GR00T_DUQUANT_HESSIAN_W4_PATH": str(hessian),
            }
        )
    policy = load_policy(
        str(checkpoint),
        data_config="examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig",
        denoising_steps=4,
        device=device,
    )
    model = policy.model
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    names = _ordered_layer_names(model, list(plan_payload["layers"]))
    if quantized:
        wrapped = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, DuQuantLinear)
        ]
        if [name for name, _ in wrapped] != names:
            raise RuntimeError("GR00T quant wrapper inventory does not match FP16 plan")
        runtime = getattr(model, "_gr00t_duquant_runtime", {})
        if runtime.get("hessian_w4_loaded") != len(names):
            raise RuntimeError(f"GR00T Hessian W4 did not load: {runtime}")
        load_act_scales(
            model,
            str(a8),
            require={
                "plan_sha256": plan_hash,
                "calibration_buffer_sha256": buffer_hash,
            },
        )
    ensure_dit_attention_patch(model, scope="dit")
    linear = LayerCapture(names, step_getter=get_current_dit_step)
    attention = AttentionCapture()
    linear.install(model)
    register_atm_logits_capture(model, attention.record_logits, scope="dit")
    register_errorfold_head_capture(model, attention.record_head_output, scope="dit")
    observations, noises = gr00t_rollout_inputs(records)
    run_activations(model, policy, observations, noises, batch_size)
    linear.remove()
    clear_atm_capture(model)
    arrays = linear.arrays(
        model,
        calibration_buffer_sha256=buffer_hash,
        checkpoint_sha256=checkpoint_hash,
        plan_sha256=plan_hash,
        include_weights=not quantized,
    )
    runtime = getattr(model, "_gr00t_duquant_runtime", {})
    _release(policy)
    return arrays, attention, runtime


def _pi05_run_batches(
    policy: Any,
    records: list[dict[str, Any]],
    *,
    device: str,
    batch_size: int,
) -> None:
    import jax
    from openpi.models import model as model_api

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        transformed = [
            policy._input_transform(copy.deepcopy(row["observation"])) for row in batch
        ]
        stacked = jax.tree.map(
            lambda *values: np.stack([np.asarray(value) for value in values]),
            *transformed,
        )
        tensors = jax.tree.map(
            lambda value: torch.from_numpy(np.asarray(value)).to(device), stacked
        )
        observation = model_api.Observation.from_dict(tensors)
        noise = torch.from_numpy(
            np.stack([row["noises"][0] for row in batch])
        ).to(device)
        with torch.inference_mode():
            policy._model.sample_actions(
                device,
                observation,
                noise=noise,
                num_steps=4,
            )


def _configure_pi05_quant(
    *,
    checkpoint_hash: str,
    plan: Path,
    pack_dir: Path,
    a8: Path,
    hessian: Path,
    buffer_hash: str,
    expected_wrapped: int,
) -> None:
    for key in list(os.environ):
        if key.startswith(("OPENPI_DUQUANT_", "OPENPI_ATM_", "OPENPI_OHB_")):
            os.environ.pop(key, None)
    os.environ.pop("OPENPI_ERRORFOLD_PATH", None)
    os.environ.update(
        {
            "QUANTVLA_ADAPTER_ONLY": "1",
            "TORCHDYNAMO_DISABLE": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OPENPI_MODEL_DTYPE": "float16",
            "OPENPI_CHECKPOINT_SHA256": checkpoint_hash,
            "OPENPI_DUQUANT_PLAN": str(plan),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": str(expected_wrapped),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "0",
            "OPENPI_DUQUANT_ROW_ROT": "0",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": "4",
            "OPENPI_DUQUANT_PACKDIR": str(pack_dir),
            "OPENPI_DUQUANT_ACT_SCALE_PATH": str(a8),
            "OPENPI_DUQUANT_REQUIRE_ACT_SCALE": "1",
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": buffer_hash,
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "1",
            "OPENPI_DUQUANT_HESSIAN_W4_PATH": str(hessian),
            "OPENPI_DUQUANT_QUIET": "1",
        }
    )


def _capture_pi05(
    *,
    checkpoint: Path,
    plan: Path,
    pack_dir: Path,
    a8: Path,
    hessian: Path,
    records: list[dict[str, Any]],
    checkpoint_hash: str,
    buffer_hash: str,
    plan_hash: str,
    device: str,
    batch_size: int,
    quantized: bool,
) -> tuple[dict[str, np.ndarray], AttentionCapture, dict[str, Any]]:
    from openpi.policies import policy_config
    from openpi.quant import enable_duquant_if_configured
    from openpi.quant.atm_pi05 import (
        clear_atm_capture,
        ensure_pi05_attention_patch,
        register_atm_logits_capture,
        register_errorfold_head_capture,
    )
    from openpi.quant.dit_step_context import get_current_dit_step
    from openpi.quant.duquant_layers import DuQuantLinear
    from openpi.training import config

    for key in list(os.environ):
        if key.startswith(("OPENPI_DUQUANT_", "OPENPI_ATM_", "OPENPI_OHB_")):
            os.environ.pop(key, None)
    os.environ.pop("OPENPI_ERRORFOLD_PATH", None)
    os.environ.pop("QUANTVLA_ADAPTER_ONLY", None)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    if quantized:
        _configure_pi05_quant(
            checkpoint_hash=checkpoint_hash,
            plan=plan,
            pack_dir=pack_dir,
            a8=a8,
            hessian=hessian,
            buffer_hash=buffer_hash,
            expected_wrapped=len(plan_payload["layers"]),
        )
    else:
        os.environ.update(
            {
                "TORCHDYNAMO_DISABLE": "1",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "OPENPI_MODEL_DTYPE": "float16",
                "OPENPI_CHECKPOINT_SHA256": checkpoint_hash,
            }
        )
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"), checkpoint, pytorch_device=device
    )
    runtime: dict[str, Any] = {}
    if quantized:
        runtime = enable_duquant_if_configured(policy._model)
        policy._model.to(device)
    model = policy._model
    names = _ordered_layer_names(model, list(plan_payload["layers"]))
    if quantized:
        wrapped_names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, DuQuantLinear)
        ]
        if wrapped_names != names or runtime.get("hessian_w4_loaded") != len(names):
            raise RuntimeError(f"pi0.5 quant inventory/Hessian drift: {runtime}")
    ensure_pi05_attention_patch(model, scope="expert")
    linear = LayerCapture(names, step_getter=get_current_dit_step)
    attention = AttentionCapture()
    linear.install(model)
    register_atm_logits_capture(model, attention.record_logits, scope="expert")
    register_errorfold_head_capture(
        model, attention.record_head_output, scope="expert"
    )
    _pi05_run_batches(policy, records, device=device, batch_size=batch_size)
    linear.remove()
    clear_atm_capture(model)
    arrays = linear.arrays(
        model,
        calibration_buffer_sha256=buffer_hash,
        checkpoint_sha256=checkpoint_hash,
        plan_sha256=plan_hash,
        include_weights=not quantized,
    )
    runtime = getattr(model, "_openpi_duquant_runtime", runtime)
    _release(policy)
    return arrays, attention, runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PROTOCOL["models"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plan", default=None)
    parser.add_argument(
        "--buffer", default=str(protocol_artifact("calibration_buffer", verify=False))
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pack-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hessian-device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    checkpoint = Path(args.checkpoint or DEFAULTS[model_name]["checkpoint"]).resolve()
    plan = Path(args.plan or DEFAULTS[model_name]["plan"]).resolve()
    buffer = Path(args.buffer).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = Path(args.pack_dir).resolve() if args.pack_dir else out_dir / "identity_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    selection = validate_quant_plan(plan_payload, model=model_name, source=str(plan))
    if len(plan_payload["layers"]) != selection["target_layers"]:
        raise ValueError("quant plan inventory is not complete")
    buffer_hash = sha256_file(buffer)
    if buffer_hash != PROTOCOL["data"]["calibration_buffer"]["sha256"]:
        raise ValueError("v3 calibration requires the frozen shared 256-observation buffer")
    records, buffer_provenance = load_model_records(
        buffer,
        int(PROTOCOL["hessian_w4a8"]["calibration_observations"]),
        model=model_name,
    )
    checkpoint_hash = _checkpoint_sha256(checkpoint, model_name)
    plan_hash = sha256_file(plan)
    identity_pack_manifest = {
        "schema_version": 3,
        "kind": "hessian_w4_identity_pack",
        "model_adapter": model_name,
        "checkpoint_sha256": checkpoint_hash,
        "plan_sha256": plan_hash,
        "protocol_sha256": PROTOCOL_SHA256,
        "packed_weights_source": "hessian_w4.npz",
        "permutation": False,
        "row_rotation": "identity",
    }
    identity_pack_path = pack_dir / "manifest.json"
    identity_pack_rendered = (
        json.dumps(identity_pack_manifest, indent=2, sort_keys=True) + "\n"
    )
    if identity_pack_path.exists():
        if identity_pack_path.read_text(encoding="utf-8") != identity_pack_rendered:
            raise ValueError(f"identity-pack provenance drift: {identity_pack_path}")
    else:
        identity_pack_path.write_text(identity_pack_rendered, encoding="utf-8")
    paths = {
        "fp16": out_dir / "fp16_capture.npz",
        "fp16_attention": out_dir / "fp16_attention.npz",
        "a8": out_dir / "a8_scales.npz",
        "hessian": out_dir / "hessian_w4.npz",
        "quant": out_dir / "quant_capture.npz",
        "quant_attention": out_dir / "quant_attention.npz",
        "paired": out_dir / "paired_errorfold_capture.npz",
        "raw": out_dir / "raw_errorfold.json",
        "manifest": out_dir / "calibration_manifest.json",
    }
    if args.force:
        for path in paths.values():
            path.unlink(missing_ok=True)
            Path(str(path) + ".json").unlink(missing_ok=True)
            Path(str(path) + ".meta.json").unlink(missing_ok=True)
    required = [
        paths["fp16"],
        paths["fp16_attention"],
        paths["a8"],
        Path(str(paths["a8"]) + (".meta.json" if model_name == "gr00t" else ".json")),
        paths["hessian"],
        Path(str(paths["hessian"]) + ".json"),
        paths["quant"],
        paths["quant_attention"],
        paths["paired"],
        paths["raw"],
    ]
    if any(path.exists() for path in required) and not all(path.exists() for path in required):
        raise RuntimeError("partial calibration artifacts found; use --force to rebuild atomically")
    runtimes: dict[str, Any] = {}
    if not all(path.exists() for path in required):
        capture = _capture_gr00t if model_name == "gr00t" else _capture_pi05
        fp16_arrays, fp16_attention, runtimes["fp16"] = capture(
            checkpoint=checkpoint,
            plan=plan,
            pack_dir=pack_dir,
            a8=paths["a8"],
            hessian=paths["hessian"],
            records=records,
            checkpoint_hash=checkpoint_hash,
            buffer_hash=buffer_hash,
            plan_hash=plan_hash,
            device=args.device,
            batch_size=args.batch_size,
            quantized=False,
        )
        save_npz(paths["fp16"], fp16_arrays)
        _save_attention(paths["fp16_attention"], fp16_attention)
        build_a8(paths["fp16"], paths["a8"], model_name)
        build_hessian(
            paths["fp16"],
            paths["hessian"],
            model_name,
            device=args.hessian_device or args.device,
        )
        quant_arrays, quant_attention, runtimes["complete_quant"] = capture(
            checkpoint=checkpoint,
            plan=plan,
            pack_dir=pack_dir,
            a8=paths["a8"],
            hessian=paths["hessian"],
            records=records,
            checkpoint_hash=checkpoint_hash,
            buffer_hash=buffer_hash,
            plan_hash=plan_hash,
            device=args.device,
            batch_size=args.batch_size,
            quantized=True,
        )
        save_npz(paths["quant"], quant_arrays)
        _save_attention(paths["quant_attention"], quant_attention)
        merge_errorfold_with_attention(
            paths["fp16"],
            paths["quant"],
            _load_attention(paths["fp16_attention"]),
            _load_attention(paths["quant_attention"]),
            paths["paired"],
        )
        build_errorfold(paths["paired"], paths["raw"], model_name)
    manifest = {
        "schema_version": 3,
        "kind": "errorfold_v3_calibration",
        "cross_model_protocol": protocol_attestation(),
        "model_adapter": model_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "plan": str(plan),
        "plan_sha256": plan_hash,
        "calibration_buffer": buffer_provenance,
        "gradient_updates": False,
        "fp16_weight_updates": False,
        "success_labels_used": False,
        "runtimes": runtimes,
        "artifacts": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
            if key != "manifest" and path.is_file()
        },
        "protocol_sha256": PROTOCOL_SHA256,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
