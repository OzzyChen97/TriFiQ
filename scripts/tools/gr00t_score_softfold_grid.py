#!/usr/bin/env python3
"""Score a sharded 9x9 GR00T SoftFold grid against the original FP16 teacher."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from fit_softfold_compensation import GRID, fold_layers  # noqa: E402
from gr00t_func_metrics import aggregate_d_pac_sequences, d_func, d_pac_sequence  # noqa: E402
from gr00t_sensitivity_probe import run_rollouts  # noqa: E402
from gr00t_v2_common import (  # noqa: E402
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    ensure_a8_calibrated,
    ensure_flash_attn_rpath,
    fixed_calibration_buffer,
    load_policy,
    set_quant_env,
    strip_quant_env,
)


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
DEFAULT_PLAN = REPO_ROOT / (
    "checkpoints/packs/robocasa365/"
    "gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json"
)
DEFAULT_PACK = REPO_ROOT / (
    "checkpoints/packs/robocasa365/"
    "duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015"
)
DEFAULT_A8 = REPO_ROOT / (
    "checkpoints/packs/robocasa365/a8_scales_cscka_16to1_protocolfix_d4.npz"
)
DEFAULT_RAW = REPO_ROOT / (
    "checkpoints/packs/robocasa365/"
    "atm_alpha_beta_static_cscka_16to1_protocolfix_d4.json"
)
DEFAULT_DATA_CONFIG = "examples.RoboCasa365.custom_data_config:RoboCasa365DataConfig"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"pack directory contains no files: {root}")
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256_file(candidate).encode("ascii") + b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--pack-dir", default=str(DEFAULT_PACK))
    parser.add_argument("--a8", default=str(DEFAULT_A8))
    parser.add_argument("--raw-correction", default=str(DEFAULT_RAW))
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--grid-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def config_id(gate_atm: float, gate_ohb: float) -> str:
    return f"softfold_a{int(round(gate_atm * 8)):02d}_b{int(round(gate_ohb * 8)):02d}"


def materialize_candidates(raw_path: Path, grid_dir: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_layers = raw.get("layers", raw)
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise ValueError("raw correction artifact has no layers")
    raw_meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    registry: dict[str, dict[str, Any]] = {}
    grid_dir.mkdir(parents=True, exist_ok=True)
    for gate_atm in GRID:
        for gate_ohb in GRID:
            layers, ohb_mode = fold_layers(
                raw_layers,
                gate_atm=gate_atm,
                gate_ohb=gate_ohb,
            )
            identifier = config_id(gate_atm, gate_ohb)
            path = grid_dir / f"{identifier}.json"
            candidate = {
                "schema_version": 1,
                "kind": "softfold_grid_candidate",
                "meta": {
                    **raw_meta,
                    "metric": "d_pac_v1",
                    "ohb_mode": ohb_mode,
                    "atm_application": "fold_q_weight",
                    "ohb_application": "fold_o_weight_perhead",
                    "selector_free": True,
                },
                "gate": {"atm": gate_atm, "ohb": gate_ohb},
                "layers": layers,
                "selection": {
                    "uses_task_labels": False,
                    "uses_rollout_success": False,
                    "status": "grid_candidate_not_selected",
                },
            }
            rendered = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
            if path.exists() and path.read_text(encoding="utf-8") != rendered:
                raise ValueError(f"frozen grid candidate drift: {path}")
            if not path.exists():
                path.write_text(rendered, encoding="utf-8")
            registry[identifier] = {
                "path": path,
                "gate": candidate["gate"],
                "ohb_mode": ohb_mode,
            }
    return registry


def score(args: argparse.Namespace) -> dict[str, Any]:
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid SoftFold grid shard")
    if args.n_obs < 1 or args.batch_size < 1:
        raise ValueError("--n-obs and --batch-size must be positive")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    plan = Path(args.plan).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    a8 = Path(args.a8).expanduser().resolve()
    raw = Path(args.raw_correction).expanduser().resolve()
    grid_dir = Path(args.grid_dir).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    for path in (checkpoint / "config.json", plan, a8, raw):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)

    registry = materialize_candidates(raw, grid_dir)
    all_ids = sorted(registry)
    selected_ids = [
        identifier
        for index, identifier in enumerate(all_ids)
        if index % args.shard_count == args.shard_index
    ]
    if not selected_ids:
        raise ValueError("SoftFold grid shard is empty")

    a8_meta_path = Path(str(a8) + ".meta.json")
    a8_meta = json.loads(a8_meta_path.read_text(encoding="utf-8")) if a8_meta_path.is_file() else {}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gr00t_softfold_grid_score",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint / "config.json"),
        "plan": str(plan),
        "plan_sha256": sha256_file(plan),
        "pack_dir": str(pack_dir),
        "pack_dir_sha256": sha256_tree(pack_dir),
        "a8": str(a8),
        "a8_sha256": sha256_file(a8),
        "artifact_calibration_buffer_sha256": a8_meta.get("buffer_sha256"),
        "raw_correction": str(raw),
        "raw_correction_sha256": sha256_file(raw),
        "selection_metric": "d_pac",
        "teacher": "original_fp16",
        "uses_task_labels_for_selection": False,
        "uses_rollout_success_for_selection": False,
        "n_obs": args.n_obs,
        "gamma": args.gamma,
        "softfold_grid": True,
        "softfold_grid_size": 81,
        "softfold_grid_shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "candidate_count": len(selected_ids),
            "config_ids": selected_ids,
        },
        "source_sha256": {
            "scorer": sha256_file(Path(__file__)),
            "d_pac_metrics": sha256_file(REPO_ROOT / "scripts/tools/gr00t_func_metrics.py"),
            "softfold_fitter": sha256_file(
                REPO_ROOT / "scripts/tools/fit_softfold_compensation.py"
            ),
            "gr00t_atm_runtime": sha256_file(REPO_ROOT / "code/gr00t/atm/dit_atm.py"),
        },
        "scores": {},
    }
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        invariant_keys = (
            "checkpoint_sha256",
            "plan_sha256",
            "a8_sha256",
            "raw_correction_sha256",
            "pack_dir_sha256",
            "n_obs",
            "gamma",
            "softfold_grid_shard",
            "source_sha256",
        )
        if any(previous.get(key) != payload.get(key) for key in invariant_keys):
            raise ValueError(f"existing score shard provenance drift: {output}")
        payload["scores"] = previous.get("scores") or {}

    ensure_flash_attn_rpath()
    strip_quant_env()
    torch.manual_seed(0)
    fp16 = load_policy(
        str(checkpoint),
        data_config=args.data_config,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    horizon = int(fp16.model.action_head.config.action_horizon)
    action_dim = int(fp16.model.action_head.config.action_dim)
    observations, noises, scoring_buffer_sha = fixed_calibration_buffer(
        0,
        args.n_obs,
        horizon,
        action_dim,
        fmt="robocasa365",
    )
    payload["buffer_sha256"] = scoring_buffer_sha
    reference = run_rollouts(fp16.model, fp16, observations, noises, args.batch_size)
    del fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    warm_obs, warm_noises, warm_sha = fixed_calibration_buffer(
        0,
        32 * args.batch_size,
        horizon,
        action_dim,
        fmt="robocasa365",
    )
    payload["a8_reproduction_buffer_sha256"] = warm_sha
    expected_wrapped = sum(
        not bool(row.get("skip", not int(row.get("bits", 0) or 0)))
        and int(row.get("bits", 0) or 0) > 0
        for row in (json.loads(plan.read_text(encoding="utf-8")).get("layers") or {}).values()
    )

    for identifier in selected_ids:
        if identifier in payload["scores"]:
            print(f"[gr00t-softfold] reuse {identifier}", flush=True)
            continue
        strip_quant_env()
        set_quant_env(DEFAULT_INCLUDE, DEFAULT_EXCLUDE, str(pack_dir))
        os.environ.update(
            {
                "GR00T_DUQUANT_PLAN": str(plan),
                "GR00T_DUQUANT_ACT_SCALE_PATH": str(a8),
                "GR00T_ATM_ENABLE": "1",
                "GR00T_OHB_ENABLE": "1",
                "GR00T_ATM_ALPHA_PATH": str(registry[identifier]["path"]),
                "GR00T_ATM_SCOPE": "dit",
                "GR00T_OHB_SCOPE": "dit",
                "GR00T_ATM_PER_STEP": "0",
                "GR00T_ATM_APPLICATION": "fold_q_weight",
                "GR00T_OHB_APPLICATION": "fold_o_weight_perhead",
            }
        )
        started = time.time()
        policy = load_policy(
            str(checkpoint),
            data_config=args.data_config,
            denoising_steps=args.denoising_steps,
            device=args.device,
        )
        ensure_a8_calibrated(
            policy,
            warm_obs,
            warm_noises,
            args.batch_size,
            expected_wrapped=expected_wrapped,
            act_scale_path=str(a8),
        )
        runtime = getattr(policy.model, "_gr00t_atm_runtime", {})
        if not runtime.get("enabled") or not runtime.get("selector_free"):
            raise RuntimeError(f"{identifier}: selector-free SoftFold was not loaded: {runtime}")
        trajectory = run_rollouts(policy.model, policy, observations, noises, args.batch_size)
        func = d_func(reference, trajectory, args.gamma)
        sequences = [
            d_pac_sequence(
                reference[:, index : index + 1],
                trajectory[:, index : index + 1],
                [0],
                executed_actions=min(16, horizon),
                gamma=args.gamma,
            )
            for index in range(args.n_obs)
        ]
        pac = aggregate_d_pac_sequences(sequences)
        payload["scores"][identifier] = {
            "gate": registry[identifier]["gate"],
            "artifact": str(registry[identifier]["path"]),
            "artifact_sha256": sha256_file(registry[identifier]["path"]),
            "wrapped_layers": expected_wrapped,
            "atm_runtime": runtime,
            "d_func": float(func["d_func"]),
            "d_func_summary": func,
            "per_obs": [float(value) for value in func["per_obs"]],
            "d_pac": float(pac["d_pac"]),
            "d_pac_summary": pac,
            "elapsed_s": time.time() - started,
        }
        atomic_json(output, payload)
        print(
            f"[gr00t-softfold] {identifier}: D_func={func['d_func']:.6g} "
            f"D_PAC={pac['d_pac']:.6g}",
            flush=True,
        )
        del policy, trajectory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = min(payload["scores"], key=lambda key: payload["scores"][key]["d_pac"])
    payload["best"] = {
        "config_id": best,
        "metric": "d_pac",
        "value": payload["scores"][best]["d_pac"],
        "d_func": payload["scores"][best]["d_func"],
        "d_pac": payload["scores"][best]["d_pac"],
        "selection_rule": "minimum original-FP16-relative D_PAC on the frozen scoring buffer",
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    args = parse_args()
    payload = score(args)
    output = Path(args.out).expanduser().resolve()
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": sha256_file(output),
                "candidates": len(payload["scores"]),
                "best": payload["best"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
