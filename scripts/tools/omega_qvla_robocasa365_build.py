#!/usr/bin/env python3
"""Build and attest model-specific Omega-QVLA packs for RoboCasa365.

The upstream builders are reused without changing their quantization recipe.
Only their LIBERO observation loader is replaced with a frozen RoboCasa365
buffer, and the released rank-0 DiT recipe uses a mathematically identical
fast path that skips an otherwise unused full SVD.
"""

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


REPO = Path(__file__).resolve().parents[2]
OMEGA = REPO / "external/Omega-QVLA"
ROOT = REPO / "runs/gdsq_extension_preregistered_v1/omega_qvla_robocasa365_v1"
DATA_CONFIG = "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig"
CALIBRATION_SEED = 20260824
CALIBRATION_SAMPLES = 10
MANIFEST_VERSION = 3
TOKEN_CAP = 1024
LLM_RE = (
    r".*backbone\.eagle_model\.language_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"
)
DIT_RE = (
    r".*action_head\.model\.transformer_blocks\.\d+\."
    r"(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*"
)
EXCLUDE_RE = (
    r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|"
    r"state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|"
    r"vlln|future_tokens)(?:\.|$)"
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(OMEGA), *args], text=True
    ).strip()


def upstream_provenance() -> dict[str, Any]:
    diff = subprocess.check_output(
        ["git", "-C", str(OMEGA), "diff", "--binary"], text=False
    )
    return {
        "repository": "https://github.com/ucmp137538/Omega-QVLA",
        "commit": git_value("rev-parse", "HEAD"),
        "dirty": bool(diff),
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "rank0_fast_path": (
            "skips full SVD only when requested rank is zero; output low-rank "
            "factors and residual are mathematically unchanged"
        ),
    }


def camel_prompt(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def frozen_samples(task_set: str) -> tuple[list[dict[str, Any]], str]:
    sys.path.insert(0, str(REPO / "scripts/tools"))
    sys.path.insert(0, str(REPO / "code"))
    from gr00t_v2_common import make_obs

    rng = np.random.default_rng(CALIBRATION_SEED)
    samples: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    chosen = TASKS[task_set][:CALIBRATION_SAMPLES]
    for index, task in enumerate(chosen):
        obs = make_obs(rng, "robocasa365")
        prompt = camel_prompt(task)
        obs["annotation.human.task_description"] = [prompt]
        for key in sorted(obs):
            digest.update(key.encode())
            value = obs[key]
            if isinstance(value, np.ndarray):
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes())
            else:
                digest.update(repr(value).encode())
        samples.append({
            "dataset_index": index,
            "trajectory_id": index,
            "base_index": 0,
            "seed": CALIBRATION_SEED + index,
            "obs": obs,
            "task_suite": f"robocasa365_{task_set}",
            "task_id": index,
            "trial_idx": 0,
            "language": prompt,
        })
    return samples, digest.hexdigest()


def manifest_payload(task_set: str, checkpoint: Path) -> dict[str, Any]:
    samples, buffer_sha = frozen_samples(task_set)
    source_paths = [
        Path(__file__).resolve(),
        OMEGA / "tools/build_gptq_weights.py",
        OMEGA / "tools/build_dit_a2lite_svd_gptq_perstep.py",
        OMEGA / "tools/merge_packs.py",
        OMEGA / "gr00t/quantization/gptq_layers.py",
        REPO / "code/gr00t/model/policy.py",
        REPO / "code/gr00t/model/action_head/flow_matching_action_head.py",
        REPO / "code/examples/RoboCasa365/custom_data_config.py",
        REPO / "scripts/inference_service.py",
    ]
    payload = {
        "schema_version": MANIFEST_VERSION,
        "kind": "omega_qvla_robocasa365_pack_preregistration",
        "task_set": task_set,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "data_config": DATA_CONFIG,
        "calibration": {
            "source": "frozen_synthetic_robocasa365_v1",
            "seed": CALIBRATION_SEED,
            "samples": CALIBRATION_SAMPLES,
            "tasks": TASKS[task_set][:CALIBRATION_SAMPLES],
            "buffer_sha256": buffer_sha,
            "test_results_used": False,
            "runtime_llm_scale_warmup_observations": 1,
            "runtime_warmup_before_test_requests": True,
        },
        "recipe": {
            "llm": "DuQuant svd_hadamard rotation + GPTQ",
            "dit": "DuQuant svd_hadamard rotation + RTN residual + per-step scales",
            "weight_bits": 4,
            "activation_bits": 4,
            "denoising_steps": 4,
            "execute_steps": 16,
            "token_cap": TOKEN_CAP,
            "gptq_block_size": 128,
            "gptq_damp_percent": 0.05,
            "duquant_block_size": 64,
            "duquant_block_out": 64,
            "act_percentile": 99.9,
            "svd_rank": 0,
        },
        "upstream": upstream_provenance(),
        "source_sha256": {
            str(path.resolve()): sha256_file(path) for path in source_paths
        },
    }
    payload["preregistration_sha256"] = canonical_sha(payload)
    previous = ROOT / "calibration" / f"{task_set}.manifest.v2.json"
    if not previous.is_file():
        previous = ROOT / "calibration" / f"{task_set}.manifest.json"
    if previous.is_file():
        payload["supersedes"] = {
            "path": str(previous.resolve()),
            "sha256": sha256_file(previous),
            "reason": (
                "The superseded build failed before producing any pack or "
                "rollout because the upstream examples package shadowed the "
                "local RoboCasa365 data config. This version attests the "
                "import-path compatibility fix."
            ),
        }
        payload["preregistration_sha256"] = canonical_sha(
            {key: value for key, value in payload.items() if key != "preregistration_sha256"}
        )
    return payload


def prepare(task_set: str, checkpoint: Path) -> Path:
    path = ROOT / "calibration" / f"{task_set}.manifest.v{MANIFEST_VERSION}.json"
    proposed = manifest_payload(task_set, checkpoint)
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
        if key.startswith(("GR00T_GPTQ", "GR00T_DUQUANT_", "GR00T_RTN")):
            os.environ.pop(key, None)


def build_stage(task_set: str, checkpoint: Path, stage: str) -> Path:
    manifest_path = prepare(task_set, checkpoint)
    manifest = json.loads(manifest_path.read_text())
    output_dir = ROOT / "packs" / task_set
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{stage}.pt"
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

    samples, buffer_sha = frozen_samples(task_set)
    if buffer_sha != manifest["calibration"]["buffer_sha256"]:
        raise SystemExit("frozen calibration buffer drift")
    clear_quant_env()
    sys.path.insert(0, str(OMEGA))
    sys.path.insert(1, str(REPO / "code"))
    sys.path.insert(2, str(REPO / "scripts/tools"))
    # Omega-QVLA ships a regular ``examples`` package, which otherwise hides
    # QuantVLA's RoboCasa365 data config.  Extend that package's search path
    # without changing the upstream builder or the frozen data-config name.
    import examples

    local_examples = str(REPO / "code/examples")
    if local_examples not in examples.__path__:
        examples.__path__.append(local_examples)

    def sample_loader(_args, _data_config):
        return None, None, samples

    common = [
        "--checkpoint", str(checkpoint),
        "--output-path", str(output),
        "--task-suite-name", f"robocasa365_{task_set}",
        "--data-config", DATA_CONFIG,
        "--device", "cuda",
        "--denoising-steps", "4",
        "--num-samples", str(CALIBRATION_SAMPLES),
        "--token-cap", str(TOKEN_CAP),
        "--seed", str(CALIBRATION_SEED),
    ]
    if stage == "llm":
        from tools import build_gptq_weights as builder

        builder.load_libero_samples = sample_loader
        argv = common + [
            "--include-regex", LLM_RE,
            "--exclude-regex", EXCLUDE_RE,
            "--duquant-rotation",
            "--duquant-rot-mode", "svd_hadamard",
            "--weight-bits", "4",
            "--gptq-block-size", "128",
            "--gptq-damp-percent", "0.05",
        ]
    elif stage == "dit":
        from tools import build_dit_a2lite_svd_gptq_perstep as builder

        builder.ensure_libero_runtime = lambda: None
        builder.load_libero_samples = sample_loader
        argv = common + [
            "--num-steps", "4",
            "--svd-rank", "0",
            "--use-rtn",
            "--w-bits", "4",
            "--a-bits", "4",
            "--act-percentile", "99.9",
            "--duquant-block-size", "64",
            "--duquant-block-out", "64",
            "--gptq-block-size", "128",
            "--gptq-damp-percent", "0.05",
        ]
    else:
        raise SystemExit(f"unknown stage: {stage}")
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(builder.__file__).resolve()), *argv]
        builder.main()
    finally:
        sys.argv = original_argv
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit(f"builder did not create pack: {output}")
    atomic_json(attestation, {
        "schema_version": 1,
        "kind": "omega_qvla_robocasa365_pack_attestation",
        "stage": stage,
        "task_set": task_set,
        "pack_path": str(output.resolve()),
        "pack_sha256": sha256_file(output),
        "pack_bytes": output.stat().st_size,
        "calibration_manifest_path": str(manifest_path.resolve()),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return output


def merge(task_set: str, checkpoint: Path) -> Path:
    manifest_path = prepare(task_set, checkpoint)
    pack_dir = ROOT / "packs" / task_set
    llm = pack_dir / "llm.pt"
    dit = pack_dir / "dit.pt"
    for path in (llm, dit):
        if not path.is_file() or not path.with_suffix(".attestation.json").is_file():
            raise SystemExit(f"missing attested stage pack: {path}")
    output = pack_dir / "omega_qvla_w4a4.pt"
    attestation = output.with_suffix(".attestation.json")
    inputs = {"llm": sha256_file(llm), "dit": sha256_file(dit)}
    if output.is_file() and attestation.is_file():
        saved = json.loads(attestation.read_text())
        if saved.get("inputs") == inputs and saved.get("pack_sha256") == sha256_file(output):
            print(f"reuse attested {output}")
            return output
        raise SystemExit(f"existing merged pack drift: {output}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{OMEGA}:{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [sys.executable, "-m", "tools.merge_packs", "--out", str(output),
         str(llm), str(dit)], cwd=OMEGA, env=env, check=True
    )
    atomic_json(attestation, {
        "schema_version": 1,
        "kind": "omega_qvla_robocasa365_merged_pack_attestation",
        "task_set": task_set,
        "pack_path": str(output.resolve()),
        "pack_sha256": sha256_file(output),
        "pack_bytes": output.stat().st_size,
        "inputs": inputs,
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "build", "merge", "verify"])
    parser.add_argument("--task-set", required=True, choices=sorted(TASKS))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=["llm", "dit"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"checkpoint missing config.json: {checkpoint}")
    if args.command == "prepare":
        prepare(args.task_set, checkpoint)
    elif args.command == "build":
        if args.stage is None:
            raise SystemExit("build requires --stage")
        print(build_stage(args.task_set, checkpoint, args.stage))
    elif args.command == "merge":
        print(merge(args.task_set, checkpoint))
    else:
        manifest_path = prepare(args.task_set, checkpoint)
        output = ROOT / "packs" / args.task_set / "omega_qvla_w4a4.pt"
        attestation = output.with_suffix(".attestation.json")
        if not output.is_file() or not attestation.is_file():
            raise SystemExit(f"missing merged pack: {output}")
        saved = json.loads(attestation.read_text())
        checks = [
            saved.get("pack_sha256") == sha256_file(output),
            saved.get("calibration_manifest_sha256") == sha256_file(manifest_path),
            "libero" not in str(output).lower(),
        ]
        if not all(checks):
            raise SystemExit(f"pack attestation failed: {output}")
        print(output)


if __name__ == "__main__":
    main()
