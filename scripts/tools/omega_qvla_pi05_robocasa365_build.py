#!/usr/bin/env python3
"""Build frozen task-set-specific Omega-QVLA W4A4 packs for pi0.5/RoboCasa365."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
OMEGA = REPO / "external/Omega-QVLA"
ROOT = REPO / "runs/gdsq_extension_preregistered_v1/omega_qvla_pi05_robocasa365_v1"
CHECKPOINT = REPO / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"
SOURCE_BUFFER = (
    REPO
    / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
)
CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
UPSTREAM_COMMIT = "3727e2203568db43fc5fba06ee8686c1b47c044f"
CALIBRATION_SEED = 20260824
CALIBRATION_SAMPLES = 10
TOKEN_CAP = 1024
MANIFEST_VERSION = 3
EXPERT_RE = (
    r".*paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\..*"
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
PALIGEMMA_RE = (
    r".*paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\..*"
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
EXCLUDE_RE = (
    r"(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|"
    r"layernorm|lm_head)(?:\.|$)"
)
TASKS = {
    "atomic_seen": [
        "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor",
        "CoffeeSetupMug", "NavigateKitchen", "OpenCabinet", "OpenDrawer",
        "OpenStandMixerHead", "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter", "PickPlaceToasterToCounter",
        "SlideDishwasherRack", "TurnOffStove", "TurnOnElectricKettle",
        "TurnOnMicrowave", "TurnOnSinkFaucet",
    ],
    "composite_seen": [
        "DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher",
        "PackIdenticalLunches", "PreSoakPan", "PrepareCoffee", "RinseSinkBasin",
        "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation",
        "StackBowlsCabinet", "SteamInMicrowave", "StirVegetables",
        "StoreLeftoversInBowl", "WashLettuce",
    ],
    "composite_unseen": [
        "ArrangeBreadBasket", "ArrangeTea", "BreadSelection",
        "CategorizeCondiments", "CuttingToolSelection", "GarnishPancake",
        "GatherTableware", "HeatKebabSandwich", "MakeIceLemonade",
        "PanTransfer", "PortionHotDogs", "RecycleBottlesByType",
        "SeparateFreezerRack", "WaffleReheat", "WashFruitColander",
        "WeighIngredients",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def camel_prompt(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(OMEGA), *args], text=True).strip()


def upstream_provenance() -> dict[str, Any]:
    diff = subprocess.check_output(
        ["git", "-C", str(OMEGA), "diff", "--binary"], text=False
    )
    commit = git_value("rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise SystemExit(f"Omega-QVLA commit drift: {commit} != {UPSTREAM_COMMIT}")
    return {
        "repository": "https://github.com/ucmp137538/Omega-QVLA",
        "commit": commit,
        "dirty": bool(diff),
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def materialize_observations(task_set: str) -> tuple[Path, str]:
    """Bind the frozen image/state buffer to ten preregistered task prompts."""
    output = ROOT / "calibration" / f"{task_set}.observations.v{MANIFEST_VERSION}.pt"
    source = np.load(SOURCE_BUFFER, allow_pickle=False)
    rng = np.random.default_rng(CALIBRATION_SEED + list(TASKS).index(task_set))
    indices = rng.choice(len(source["images"]), CALIBRATION_SAMPLES, replace=False)
    rows = []
    for index, (task, source_index) in enumerate(
        zip(TASKS[task_set][:CALIBRATION_SAMPLES], indices, strict=True)
    ):
        rows.append({
            "observation/image": np.asarray(source["images"][source_index]).copy(),
            "observation/wrist_image": np.asarray(source["wrist_images"][source_index]).copy(),
            "observation/right_image": np.asarray(source["right_images"][source_index]).copy(),
            "observation/state": np.asarray(
                source["states"][source_index], dtype=np.float32
            ).copy(),
            "prompt": camel_prompt(task),
            "task": task,
            "calibration_index": index,
            "source_index": int(source_index),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        saved = torch.load(output, map_location="cpu", weights_only=False)
        if len(saved) != CALIBRATION_SAMPLES or [r["task"] for r in saved] != [
            r["task"] for r in rows
        ]:
            raise SystemExit(f"immutable calibration observations drift: {output}")
    else:
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        torch.save(rows, temporary)
        temporary.replace(output)
    return output, sha256_file(output)


def manifest_payload(task_set: str) -> dict[str, Any]:
    observations, observations_sha = materialize_observations(task_set)
    source_paths = [
        Path(__file__).resolve(),
        OMEGA / "tools/build_pi05_a2lite_gptq_perstep.py",
        OMEGA / "tools/merge_packs.py",
        OMEGA / "gr00t/quantization/gptq_layers.py",
        REPO / "code/pi05/openpi/src/openpi/models_pytorch/pi0_pytorch.py",
        REPO / "code/pi05/openpi/src/openpi/quant/dit_step_context.py",
    ]
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "kind": "omega_qvla_pi05_robocasa365_pack_preregistration",
        "task_set": task_set,
        "checkpoint": {
            "path": str(CHECKPOINT.resolve()),
            "model_sha256": sha256_file(CHECKPOINT / "model.safetensors"),
        },
        "calibration": {
            "source": "frozen_pi05_robocasa365_data_free_buffer_task_prompt_binding_v1",
            "source_buffer": str(SOURCE_BUFFER.resolve()),
            "source_buffer_sha256": sha256_file(SOURCE_BUFFER),
            "observations": str(observations.resolve()),
            "observations_sha256": observations_sha,
            "seed": CALIBRATION_SEED + list(TASKS).index(task_set),
            "samples": CALIBRATION_SAMPLES,
            "tasks": TASKS[task_set][:CALIBRATION_SAMPLES],
            "test_results_used": False,
        },
        "recipe": {
            "paligemma": "A2-lite SVD-Hadamard rotation + GPTQ",
            "gemma_expert": "A2-lite SVD-Hadamard rotation + RTN + per-step scales",
            "weight_bits": 4,
            "activation_bits": 4,
            "denoising_steps": 4,
            "execute_steps": 16,
            "gptq_block_size": 128,
            "gptq_damp_percent": 0.05,
            "duquant_block_size": 64,
            "duquant_block_out": 64,
            "act_percentile": 99.9,
            "token_cap": TOKEN_CAP,
        },
        "evaluation": {
            "split": "target",
            "tasks": TASKS[task_set],
            "seeds": list(range(50)),
            "paired_action_noise": True,
        },
        "gr00t_alignment": {
            "reference_builder": str(
                (REPO / "scripts/tools/omega_qvla_robocasa365_build.py").resolve()
            ),
            "reference_builder_sha256": sha256_file(
                REPO / "scripts/tools/omega_qvla_robocasa365_build.py"
            ),
            "reference_orchestrator": str(
                (REPO / "scripts/run_omega_qvla_robocasa365.sh").resolve()
            ),
            "reference_orchestrator_sha256": sha256_file(
                REPO / "scripts/run_omega_qvla_robocasa365.sh"
            ),
            "matched_fields": [
                "task_sets", "calibration_seed", "calibration_samples",
                "task_prompt_inventory", "weight_bits", "activation_bits",
                "llm_gptq", "action_rtn_perstep", "token_cap",
                "denoising_steps", "execute_steps", "target_split",
                "paired_action_noise", "tasks", "seeds",
            ],
            "adapter_only_differences": [
                "checkpoint", "observation_adapter", "target_layer_names",
                "policy_server",
            ],
        },
        "upstream": upstream_provenance(),
        "source_sha256": {str(path.resolve()): sha256_file(path) for path in source_paths},
    }
    if payload["checkpoint"]["model_sha256"] != CHECKPOINT_SHA256:
        raise SystemExit("pi0.5 checkpoint SHA256 drift")
    previous = ROOT / "calibration" / f"{task_set}.manifest.v{MANIFEST_VERSION - 1}.json"
    if previous.is_file():
        payload["supersedes"] = {
            "path": str(previous.resolve()),
            "sha256": sha256_file(previous),
            "reason": (
                "v2 passed the namespaced-input adapter check, but its one-layer "
                "diagnostic entered max-autotune compilation and was stopped before "
                "capturing all ten samples. v3 disables torch.compile, matching the "
                "existing formal pi0.5 server and preserving eager model numerics."
            ),
            "previous_produced_complete_stage_pack": False,
            "previous_produced_formal_test_results": False,
        }
    payload["preregistration_sha256"] = canonical_sha(payload)
    return payload


def prepare(task_set: str) -> Path:
    path = ROOT / "calibration" / f"{task_set}.manifest.v{MANIFEST_VERSION}.json"
    proposed = manifest_payload(task_set)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != proposed:
            raise SystemExit(f"immutable calibration manifest mismatch: {path}")
    else:
        atomic_json(path, proposed)
    print(path)
    return path


def clear_quant_env() -> None:
    for key in list(os.environ):
        if key.startswith(("GR00T_GPTQ", "GR00T_DUQUANT_", "GR00T_RTN", "OPENPI_")):
            os.environ.pop(key, None)


def build_stage(task_set: str, stage: str, *, max_layers: int = 0) -> Path:
    manifest_path = prepare(task_set)
    manifest = json.loads(manifest_path.read_text())
    output_dir = ROOT / "packs" / task_set
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".smoke{max_layers}" if max_layers else ""
    output = output_dir / f"{stage}{suffix}.pt"
    attestation = output.with_suffix(".attestation.json")
    if output.is_file() and attestation.is_file():
        saved = json.loads(attestation.read_text())
        if (
            saved.get("pack_sha256") == sha256_file(output)
            and saved.get("calibration_manifest_sha256") == sha256_file(manifest_path)
        ):
            print(f"reuse attested {output}")
            return output
        raise SystemExit(f"existing unattested or drifted pack: {output}")

    clear_quant_env()
    os.environ["OPENPI_MODEL_DTYPE"] = "float16"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    sys.path.insert(0, str(OMEGA))
    sys.path.insert(1, str(REPO / "code/pi05/openpi/src"))
    sys.path.insert(2, str(REPO / "code/pi05/openpi/packages/openpi-client/src"))
    from tools import build_pi05_a2lite_gptq_perstep as builder
    from openpi.quant.dit_step_context import get_current_dit_step

    # The upstream builder imports the GR00T context, while this pi0.5 fork
    # publishes the flow-step in OpenPI's equivalent context.
    builder.get_current_dit_step = get_current_dit_step
    original_loader = builder.load_pi05_policy

    def load_with_four_steps(checkpoint: str, data_config: str, device: str):
        policy, model = original_loader(checkpoint, data_config, device)
        policy._sample_kwargs["num_steps"] = 4
        return policy, model

    builder.load_pi05_policy = load_with_four_steps
    argv = [
        "--checkpoint", str(CHECKPOINT),
        "--data-config", "pi05_pretrain_human300",
        "--obs-path", manifest["calibration"]["observations"],
        "--output", str(output),
        "--max-samples", str(CALIBRATION_SAMPLES),
        "--token-cap", str(TOKEN_CAP),
        "--device", "cuda",
        "--exclude-regex", EXCLUDE_RE,
        "--w-bits", "4",
        "--a-bits", "4",
        "--gptq-block-size", "128",
        "--gptq-damp-percent", "0.05",
        "--duquant-block-size", "64",
        "--duquant-block-out", "64",
        "--act-percentile", "99.9",
        "--save-dtype", "float16",
    ]
    if max_layers:
        argv += ["--max-layers", str(max_layers)]
    if stage == "paligemma":
        argv += ["--include-regex", PALIGEMMA_RE, "--capture-prefix", "--num-steps", "1"]
    elif stage == "expert":
        argv += ["--include-regex", EXPERT_RE, "--use-rtn", "--num-steps", "4"]
    else:
        raise SystemExit(f"unknown stage: {stage}")
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(builder.__file__).resolve()), *argv]
        builder.main()
    finally:
        sys.argv = original_argv
        builder.load_pi05_policy = original_loader
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit(f"builder did not create pack: {output}")
    atomic_json(attestation, {
        "schema_version": 1,
        "kind": "omega_qvla_pi05_robocasa365_stage_pack_attestation",
        "stage": stage,
        "task_set": task_set,
        "max_layers": max_layers,
        "pack_path": str(output.resolve()),
        "pack_sha256": sha256_file(output),
        "pack_bytes": output.stat().st_size,
        "calibration_manifest_path": str(manifest_path.resolve()),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return output


def merge(task_set: str) -> Path:
    manifest_path = prepare(task_set)
    pack_dir = ROOT / "packs" / task_set
    inputs = [pack_dir / "paligemma.pt", pack_dir / "expert.pt"]
    for path in inputs:
        if not path.is_file() or not path.with_suffix(".attestation.json").is_file():
            raise SystemExit(f"missing attested stage pack: {path}")
    output = pack_dir / "omega_qvla_w4a4.pt"
    attestation = output.with_suffix(".attestation.json")
    input_hashes = {path.stem: sha256_file(path) for path in inputs}
    if output.is_file() and attestation.is_file():
        saved = json.loads(attestation.read_text())
        if saved.get("inputs") == input_hashes and saved.get("pack_sha256") == sha256_file(output):
            print(f"reuse attested {output}")
            return output
        raise SystemExit(f"existing merged pack drift: {output}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{OMEGA}:{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [sys.executable, "-m", "tools.merge_packs", "--out", str(output), *map(str, inputs)],
        cwd=OMEGA, env=env, check=True,
    )
    atomic_json(attestation, {
        "schema_version": 1,
        "kind": "omega_qvla_pi05_robocasa365_merged_pack_attestation",
        "task_set": task_set,
        "pack_path": str(output.resolve()),
        "pack_sha256": sha256_file(output),
        "pack_bytes": output.stat().st_size,
        "inputs": input_hashes,
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return output


def verify(task_set: str) -> Path:
    manifest_path = prepare(task_set)
    output = ROOT / "packs" / task_set / "omega_qvla_w4a4.pt"
    attestation = output.with_suffix(".attestation.json")
    if not output.is_file() or not attestation.is_file():
        raise SystemExit(f"missing merged pack: {output}")
    saved = json.loads(attestation.read_text())
    pack = torch.load(output, map_location="cpu", weights_only=False, mmap=True)
    names = [name for name in pack if name != "__meta__"]
    experts = [name for name in names if "gemma_expert" in name]
    paligemma = [name for name in names if ".paligemma." in name]
    checks = [
        saved.get("pack_sha256") == sha256_file(output),
        saved.get("calibration_manifest_sha256") == sha256_file(manifest_path),
        len(names) == 252,
        len(experts) == 126,
        len(paligemma) == 126,
        all(int(pack[name].get("weight_bits", -1)) == 4 for name in names),
        all(int(pack[name].get("a_bits", -1)) == 4 for name in names),
        all(tuple(pack[name]["act_scale_table"].shape[:1]) == (4,) for name in experts),
        all(tuple(pack[name]["act_scale_table"].shape[:1]) == (1,) for name in paligemma),
    ]
    if not all(checks):
        raise SystemExit(f"pack verification failed: {output}")
    print(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "build", "merge", "verify"])
    parser.add_argument("--task-set", required=True, choices=sorted(TASKS))
    parser.add_argument("--stage", choices=["paligemma", "expert"])
    parser.add_argument("--max-layers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.task_set)
    elif args.command == "build":
        if args.stage is None:
            raise SystemExit("build requires --stage")
        print(build_stage(args.task_set, args.stage, max_layers=args.max_layers))
    elif args.command == "merge":
        print(merge(args.task_set))
    else:
        verify(args.task_set)


if __name__ == "__main__":
    main()
