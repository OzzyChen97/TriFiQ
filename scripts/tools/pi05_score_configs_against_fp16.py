#!/usr/bin/env python3
"""Score π0.5 quantization configs against the FP16 teacher on one buffer.

This is a diagnostic/selection tool, not an A8 calibration tool.  It answers:
given the same observations and paired denoising noise, which quantized config
best preserves the original FP16 policy actions under the paper D_func metric?
"""

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
from collections import defaultdict

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = REPO_ROOT / "code" / "pi05" / "openpi"
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import enable_duquant_if_configured, enable_pi05_atm_if_configured  # noqa: E402
from openpi.quant import sha256_file  # noqa: E402
from openpi.training import config  # noqa: E402
from gr00t_func_metrics import aggregate_d_pac_sequences  # noqa: E402
from pi05_func_metrics import d_func, d_pac_sequence  # noqa: E402
from pi05_sensitivity_probe import load_records, run_records  # noqa: E402
from fit_softfold_compensation import GRID, fold_layers  # noqa: E402


CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"
DEFAULT_PACK = REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
DEFAULT_BUFFER = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/diagnostics/task_reset_probe/task_reset_4x4_target_n32.npz"
)
DEFAULT_ORIGINAL_CALIBRATION_BUFFER = (
    REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
)
ALIGNED_ROOT = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned"


DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "quantvla_w4a8_atmohb": {
        "plan": ALIGNED_ROOT / "plans/pi05_quantvla_uniform_w4a8_d4.plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "atm_ohb/pi05_quantvla_uniform_w4a8_d4_static_perhead.json",
        "atm_enable": True,
        "ohb_enable": True,
        "wrapped": 180,
    },
    "gdsq_vla": {
        "plan": ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        "atm": None,
        "wrapped": 80,
    },
    "gdsq_vla_atmohb": {
        "plan": ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json",
        "atm_enable": True,
        "ohb_enable": True,
        "wrapped": 80,
    },
    "gdsq_vla_atm_only": {
        "plan": ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json",
        "atm_enable": True,
        "ohb_enable": False,
        "wrapped": 80,
    },
    "gdsq_vla_ohb_only": {
        "plan": ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        "a8": ALIGNED_ROOT / "a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json",
        "atm_enable": False,
        "ohb_enable": True,
        "wrapped": 80,
    },
    "expert_protect": {
        "plan": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected.plan.json",
        "a8": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected_p999_b32x8.npz",
        "atm": None,
        "wrapped": 60,
    },
    "expert_protect_atmohb": {
        "plan": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected.plan.json",
        "a8": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected_p999_b32x8.npz",
        "atm": ALIGNED_ROOT / "diagnostics/expert_protect/pi05_gdsq_expert_mlp_protected_static_perhead.json",
        "atm_enable": True,
        "ohb_enable": True,
        "wrapped": 60,
    },
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument("--artifact-calibration-buffer", default=str(DEFAULT_ORIGINAL_CALIBRATION_BUFFER))
    parser.add_argument("--pack-dir", default=str(DEFAULT_PACK))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument(
        "--selection-metric",
        choices=("d_pac", "d_func"),
        default="d_pac",
        help="FP16-relative frozen-buffer metric used to select the best config.",
    )
    parser.add_argument(
        "--pac-overlap-weight",
        type=float,
        default=0.1,
        help="Low-weight pi0.5 16:50 forecast-overlap coefficient in D_PAC.",
    )
    parser.add_argument(
        "--include",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated config ids from the default registry.",
    )
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Require A8/ATM artifacts to match the scoring buffer. Off by default for FP16-guided selection buffers.",
    )
    parser.add_argument(
        "--out",
        default=str(ALIGNED_ROOT / "diagnostics/task_reset_probe/task_reset_action_probe_16obs.json"),
    )
    parser.add_argument(
        "--softfold-grid",
        action="store_true",
        help="Evaluate the complete selector-free 9x9 ATM/OHB gate grid.",
    )
    parser.add_argument(
        "--softfold-raw",
        default=None,
        help="Raw ATM/OHB artifact used to materialize --softfold-grid coefficients.",
    )
    parser.add_argument(
        "--softfold-base",
        choices=tuple(DEFAULT_CONFIGS),
        default="gdsq_vla_atmohb",
        help="Quant plan/A8 configuration held fixed during the SoftFold grid.",
    )
    parser.add_argument(
        "--softfold-grid-dir",
        default=None,
        help="Directory for deterministic effective-coefficient grid artifacts.",
    )
    parser.add_argument(
        "--softfold-grid-shard-index",
        type=int,
        default=0,
        help="Zero-based deterministic shard of the 9x9 SoftFold grid.",
    )
    parser.add_argument(
        "--softfold-grid-shard-count",
        type=int,
        default=1,
        help="Number of disjoint deterministic SoftFold grid shards.",
    )
    return parser.parse_args()


def materialize_softfold_grid(
    *,
    raw_path: Path,
    base_spec: dict[str, Any],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_layers = raw.get("layers", raw)
    raw_meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise ValueError("SoftFold raw artifact has no layers")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry: dict[str, dict[str, Any]] = {}
    for gate_atm in GRID:
        for gate_ohb in GRID:
            layers, ohb_mode = fold_layers(
                raw_layers,
                gate_atm=gate_atm,
                gate_ohb=gate_ohb,
            )
            config_id = f"softfold_a{int(round(gate_atm * 8)):02d}_b{int(round(gate_ohb * 8)):02d}"
            path = output_dir / f"{config_id}.json"
            payload = {
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
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            registry[config_id] = {
                **base_spec,
                "atm": path,
                "atm_enable": True,
                "ohb_enable": True,
                "gate": {"atm": gate_atm, "ohb": gate_ohb},
                "atm_application": "fold_q_weight",
                "ohb_application": "fold_o_weight_perhead",
            }
    return registry


def clear_quant_environment() -> None:
    for key in list(os.environ):
        if key.startswith(("OPENPI_DUQUANT_", "OPENPI_ATM_", "OPENPI_OHB_")):
            os.environ.pop(key, None)


def configure_base() -> None:
    clear_quant_environment()
    os.environ.update(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "OPENPI_MODEL_DTYPE": "float16",
            "OPENPI_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
        }
    )


def configure_quant(
    *,
    spec: dict[str, Any],
    pack_dir: Path,
    artifact_buffer_hash: str,
    strict_artifacts: bool,
) -> None:
    configure_base()
    plan = Path(spec["plan"]).expanduser().resolve()
    a8 = Path(spec["a8"]).expanduser().resolve()
    os.environ.update(
        {
            "OPENPI_DUQUANT_PLAN": str(plan),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": str(int(spec["wrapped"])),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "0",
            "OPENPI_DUQUANT_ROW_ROT": "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": "4",
            "OPENPI_DUQUANT_PACKDIR": str(pack_dir),
            "OPENPI_DUQUANT_ACT_SCALE_PATH": str(a8),
            "OPENPI_DUQUANT_REQUIRE_ACT_SCALE": "1",
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": artifact_buffer_hash,
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1" if strict_artifacts else "0",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "0",
            "OPENPI_DUQUANT_QUIET": "1",
        }
    )
    atm = spec.get("atm")
    if atm:
        atm_enable = bool(spec.get("atm_enable", True))
        ohb_enable = bool(spec.get("ohb_enable", True))
        os.environ.update(
            {
                "OPENPI_ATM_ENABLE": "1" if atm_enable else "0",
                "OPENPI_OHB_ENABLE": "1" if ohb_enable else "0",
                "OPENPI_ATM_ALPHA_PATH": str(Path(atm).expanduser().resolve()),
                "OPENPI_ATM_SCOPE": "expert",
                "OPENPI_OHB_SCOPE": "expert",
                "OPENPI_ATM_STRICT": "1" if strict_artifacts else "0",
                "OPENPI_ATM_EXPECT_LAYERS": "18" if atm_enable or ohb_enable else "0",
                "OPENPI_ATM_APPLICATION": str(
                    spec.get("atm_application", "runtime_query")
                ),
                "OPENPI_OHB_EXPECT_MODE": "per_head_pre_projection",
                "OPENPI_OHB_APPLICATION": str(
                    spec.get("ohb_application", "runtime_output")
                ),
                "OPENPI_ATM_EXPECT_PLAN_SHA256": sha256_file(plan),
                "OPENPI_ATM_EXPECT_BUFFER_SHA256": artifact_buffer_hash,
            }
        )


def load_policy(checkpoint_dir: Path, device: str):
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"),
        checkpoint_dir,
        pytorch_device=device,
    )
    policy._model.to(device)
    return policy


def _mean_by_key(values: list[float], records: list[dict[str, Any]], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for value, record in zip(values, records):
        if key in record:
            groups[str(record[key])].append(float(value))
    return {name: float(np.mean(items)) for name, items in sorted(groups.items())}


def _mean_by_replan_bin(values: list[float], records: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for value, record in zip(values, records):
        if "replan" not in record:
            continue
        replan = int(record["replan"])
        groups[f"{(replan // 4) * 4:02d}-{(replan // 4) * 4 + 3:02d}"].append(float(value))
    return {name: float(np.mean(items)) for name, items in sorted(groups.items())}


def summarize_metrics(metrics: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    per_obs = [float(value) for value in metrics["per_obs"]]
    return {
        "d_func": float(metrics["d_func"]),
        "d_solver": float(metrics["d_solver"]),
        "d_final": float(metrics["d_final"]),
        "d_kin": float(metrics["d_kin"]),
        "d_grip": float(metrics["d_grip"]),
        "tail_cvar90": float(metrics["tail"]["cvar90"]),
        "per_dim": metrics["per_dim"],
        "per_obs": per_obs,
        "by_task": _mean_by_key(per_obs, records, "task"),
        "by_replan": _mean_by_key(per_obs, records, "replan"),
        "by_replan_bin": _mean_by_replan_bin(per_obs, records),
    }


def summarize_d_pac(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    records: list[dict[str, Any]],
    *,
    overlap_weight: float,
    gamma: float,
) -> dict[str, Any]:
    """Group paired trajectories by task/seed and aggregate sequence-level CVaR.

    Buffers without replan metadata are handled conservatively as independent
    one-observation action-prefix sequences; unrelated observations are never
    presented as a synthetic control-time rollout.
    """
    groups: dict[tuple[str, int, int | None], list[int]] = defaultdict(list)
    has_replans = all("replan" in record for record in records)
    for index, record in enumerate(records):
        identity = (str(record.get("task", "unknown")), int(record.get("seed", -1)))
        singleton = None if has_replans else index
        groups[(*identity, singleton)].append(index)

    sequence_rows = []
    for (task, seed, _singleton), positions in sorted(groups.items()):
        replans = [int(records[index].get("replan", 0)) for index in positions]
        selected = torch.tensor(positions, dtype=torch.long)
        result = d_pac_sequence(
            reference.index_select(1, selected),
            candidate.index_select(1, selected),
            replans,
            overlap_weight=overlap_weight,
            gamma=gamma,
        )
        sequence_rows.append(
            {
                "task": task,
                "seed": seed,
                "record_indices": positions,
                **result,
            }
        )
    aggregate = aggregate_d_pac_sequences(sequence_rows, tail_weight=1.0, cvar_alpha=0.9)
    return {
        **aggregate,
        "metric": "d_pac_v1",
        "teacher": "original_fp16",
        "overlap_weight": float(overlap_weight),
        "sequence_grouping": "task_seed_replan" if has_replans else "independent_action_prefix",
        "sequences": sequence_rows,
    }


def digest_path(path: Path | None) -> str | None:
    return sha256_file(path) if path is not None and path.is_file() else None


def main() -> None:
    args = parse_args()
    registry = dict(DEFAULT_CONFIGS)
    if args.softfold_grid:
        if not args.softfold_raw:
            raise ValueError("--softfold-grid requires --softfold-raw")
        grid_dir = (
            Path(args.softfold_grid_dir).expanduser().resolve()
            if args.softfold_grid_dir
            else Path(str(Path(args.out).expanduser().resolve()) + ".softfold_grid")
        )
        registry.update(
            materialize_softfold_grid(
                raw_path=Path(args.softfold_raw).expanduser().resolve(),
                base_spec=DEFAULT_CONFIGS[args.softfold_base],
                output_dir=grid_dir,
            )
        )
        all_grid_config_ids = sorted(key for key in registry if key.startswith("softfold_"))
        if args.softfold_grid_shard_count < 1:
            raise ValueError("--softfold-grid-shard-count must be positive")
        if not 0 <= args.softfold_grid_shard_index < args.softfold_grid_shard_count:
            raise ValueError("invalid --softfold-grid-shard-index")
        config_ids = [
            config_id
            for index, config_id in enumerate(all_grid_config_ids)
            if index % args.softfold_grid_shard_count == args.softfold_grid_shard_index
        ]
        if not config_ids:
            raise ValueError("SoftFold grid shard is empty")
    else:
        if args.softfold_grid_shard_index != 0 or args.softfold_grid_shard_count != 1:
            raise ValueError("SoftFold grid sharding requires --softfold-grid")
        config_ids = [item.strip() for item in args.include.split(",") if item.strip()]
    unknown = sorted(set(config_ids) - set(registry))
    if unknown:
        raise ValueError(f"unknown config ids: {unknown}")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    buffer_path = Path(args.buffer).expanduser().resolve()
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    artifact_buffer_hash = sha256_file(Path(args.artifact_calibration_buffer).expanduser().resolve())
    records = load_records(buffer_path, args.n_obs)

    record_details = [{"task": row["task"], "seed": row["seed"]} for row in records]
    with np.load(buffer_path, allow_pickle=False) as archive:
        if "env_steps" in archive.files:
            env_steps = archive["env_steps"][: args.n_obs]
            for row, env_step in zip(record_details, env_steps):
                row["env_step"] = int(env_step)
        if "replan_indices" in archive.files:
            replans = archive["replan_indices"][: args.n_obs]
            for row, replan in zip(record_details, replans):
                row["replan"] = int(replan)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fp16_guided_quant_config_score",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": sha256_file(checkpoint_dir / "model.safetensors"),
        "buffer": str(buffer_path),
        "buffer_sha256": sha256_file(buffer_path),
        "artifact_calibration_buffer_sha256": artifact_buffer_hash,
        "strict_artifacts": bool(args.strict_artifacts),
        "n_obs": args.n_obs,
        "gamma": args.gamma,
        "selection_metric": args.selection_metric,
        "pac_overlap_weight": args.pac_overlap_weight,
        "softfold_grid": bool(args.softfold_grid),
        "softfold_grid_size": 81 if args.softfold_grid else 0,
        "softfold_grid_shard": (
            {
                "index": args.softfold_grid_shard_index,
                "count": args.softfold_grid_shard_count,
                "candidate_count": len(config_ids),
                "config_ids": config_ids,
            }
            if args.softfold_grid
            else None
        ),
        "quant_plan_sha256": (
            sha256_file(Path(DEFAULT_CONFIGS[args.softfold_base]["plan"]).resolve())
            if args.softfold_grid
            else None
        ),
        "a8_sha256": (
            sha256_file(Path(DEFAULT_CONFIGS[args.softfold_base]["a8"]).resolve())
            if args.softfold_grid
            else None
        ),
        "raw_correction_sha256": (
            sha256_file(Path(args.softfold_raw).expanduser().resolve())
            if args.softfold_grid and args.softfold_raw
            else None
        ),
        "source_sha256": {
            "scorer": sha256_file(Path(__file__)),
            "d_pac_metrics": sha256_file(REPO_ROOT / "scripts/tools/pi05_func_metrics.py"),
            "softfold_fitter": sha256_file(
                REPO_ROOT / "scripts/tools/fit_softfold_compensation.py"
            ),
            "pi05_atm_runtime": sha256_file(
                REPO_ROOT / "code/pi05/openpi/src/openpi/quant/atm_pi05.py"
            ),
        },
        "uses_task_labels_for_selection": False,
        "uses_rollout_success_for_selection": False,
        "records": record_details,
        "scores": {},
    }
    output = Path(args.out).expanduser().resolve()
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        invariant_keys = (
            "checkpoint_sha256",
            "buffer_sha256",
            "artifact_calibration_buffer_sha256",
            "n_obs",
            "gamma",
            "selection_metric",
            "pac_overlap_weight",
            "softfold_grid_shard",
            "quant_plan_sha256",
            "a8_sha256",
            "raw_correction_sha256",
            "source_sha256",
        )
        if any(previous.get(key) != payload.get(key) for key in invariant_keys):
            raise ValueError(f"existing score shard provenance drift: {output}")
        payload["scores"] = previous.get("scores") or {}

    configure_base()
    started = time.time()
    fp16 = load_policy(checkpoint_dir, args.device)
    reference, _actions, timings = run_records(fp16, records, args.device, noise_index=0)
    payload["fp16"] = {
        "latency_mean_s": float(np.mean(timings)),
        "elapsed_s": time.time() - started,
    }
    del fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for config_id in config_ids:
        if config_id in payload["scores"]:
            print(f"[fp16-guided] reuse {config_id}", flush=True)
            continue
        spec = registry[config_id]
        plan = Path(spec["plan"]).expanduser().resolve()
        a8 = Path(spec["a8"]).expanduser().resolve()
        atm = Path(spec["atm"]).expanduser().resolve() if spec.get("atm") else None
        started = time.time()
        configure_quant(
            spec={**spec, "plan": plan, "a8": a8, "atm": atm},
            pack_dir=pack_dir,
            artifact_buffer_hash=artifact_buffer_hash,
            strict_artifacts=args.strict_artifacts,
        )
        policy = load_policy(checkpoint_dir, args.device)
        runtime = enable_duquant_if_configured(policy._model)
        policy._model.to(args.device)
        atm_runtime = {"enabled": False}
        if atm is not None:
            enable_pi05_atm_if_configured(policy._model)
            atm_runtime = getattr(policy._model, "_openpi_atm_runtime", {"enabled": False})
        trajectory, _actions, timings = run_records(policy, records, args.device, noise_index=0)
        metrics = d_func(reference, trajectory, args.gamma)
        pac_metrics = summarize_d_pac(
            reference,
            trajectory,
            record_details,
            overlap_weight=args.pac_overlap_weight,
            gamma=args.gamma,
        )
        payload["scores"][config_id] = {
            "plan": str(plan),
            "plan_sha256": digest_path(plan),
            "a8": str(a8),
            "a8_sha256": digest_path(a8),
            "atm": str(atm) if atm else None,
            "atm_sha256": digest_path(atm),
            "wrapped_layers": int(runtime["wrapped_layers"]),
            "atm_enabled": bool(atm_runtime.get("enabled")),
            "gate": spec.get("gate"),
            **summarize_metrics(metrics, record_details),
            "d_pac": pac_metrics["d_pac"],
            "d_pac_summary": pac_metrics,
            "latency_mean_s": float(np.mean(timings)),
            "elapsed_s": time.time() - started,
        }
        atomic_json(output, payload)
        print(
            f"[fp16-guided] {config_id}: "
            f"D_func={payload['scores'][config_id]['d_func']:.6g} "
            f"D_PAC={payload['scores'][config_id]['d_pac']:.6g}",
            flush=True,
        )
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_key = args.selection_metric
    best = min(payload["scores"], key=lambda key: payload["scores"][key][metric_key])
    payload["best"] = {
        "config_id": best,
        "metric": metric_key,
        "value": payload["scores"][best][metric_key],
        "d_func": payload["scores"][best]["d_func"],
        "d_pac": payload["scores"][best]["d_pac"],
        "selection_rule": f"minimum FP16-teacher {metric_key} on the frozen scoring buffer",
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "out": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "best": payload["best"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
