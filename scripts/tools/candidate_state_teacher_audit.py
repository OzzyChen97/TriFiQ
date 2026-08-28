#!/usr/bin/env python3
"""Candidate-state teacher adjudication for frozen GR00T masks.

Closed-loop candidates visit states the fixed teacher-state buffer never
saw. This tool audits whether each candidate stays action-consistent with
the FP16 teacher on the candidate's *own* states:

1. ``pool`` pools per-candidate observation archives (captured from
   closed-loop development replans) into one canonical archive;
2. ``audit`` runs the FP16 teacher and every candidate mask on the pooled
   observations with identical deterministic paired noise and reports::

       J_candidate_state(M) = max over {D_func, D_PAC} of
           (mean(per-sequence delta) + SE(delta)) / teacher-state baseline scale

Success labels and rewards are never read. This is a selection-stage
diagnostic; it does not feed Table 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from quantvla_outputimpact import atomic_json  # noqa: E402


REQUIRED_FIELDS = (
    "images",
    "wrist_images",
    "right_images",
    "states",
    "prompts",
    "task_ids",
    "env_seeds",
    "action_noises",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_archive(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_FIELDS) - set(archive.files))
        if missing:
            raise ValueError(f"{path}: candidate-state archive lacks fields: {missing}")
        return (
            {name: np.asarray(archive[name]) for name in archive.files},
            list(archive.files),
        )


def pool_archives(buffers: list[tuple[str, Path]], out: Path) -> dict[str, Any]:
    """Pool per-candidate archives into one canonical archive.

    Rows are keyed by (task, env_seed, env_step, replan); duplicate keys
    across sources are a hard error so one source cannot dominate the pool.
    """
    if not buffers:
        raise ValueError("at least one candidate buffer is required")
    rows: dict[str, dict[str, Any]] = {}
    sources = []
    for identifier, path in buffers:
        archive, fields = load_archive(path)
        count = len(archive["states"])
        for index in range(count):
            task = str(archive["task_ids"][index])
            seed = int(archive["env_seeds"][index])
            env_step = (
                int(archive["env_steps"][index])
                if "env_steps" in fields
                else None
            )
            replan = (
                int(archive["replan_indices"][index])
                if "replan_indices" in fields
                else None
            )
            key = f"{task}|{seed}|{env_step}|{replan}"
            if key in rows:
                raise ValueError(f"duplicate pooled candidate-state key: {key}")
            rows[key] = {
                "image": np.asarray(archive["images"][index]),
                "wrist_image": np.asarray(archive["wrist_images"][index]),
                "right_image": np.asarray(archive["right_images"][index]),
                "state": np.asarray(archive["states"][index], dtype=np.float32),
                "prompt": str(archive["prompts"][index]),
                "task": task,
                "seed": seed,
                "env_step": env_step,
                "replan": replan,
                "noise": np.asarray(archive["action_noises"][index], dtype=np.float32),
                "source": identifier,
            }
        sources.append(
            {
                "identifier": identifier,
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": count,
            }
        )
    ordered = sorted(rows.values(), key=lambda row: (row["task"], row["seed"], row["replan"] or -1))
    arrays = {
        "images": np.stack([row["image"] for row in ordered]),
        "wrist_images": np.stack([row["wrist_image"] for row in ordered]),
        "right_images": np.stack([row["right_image"] for row in ordered]),
        "states": np.stack([row["state"] for row in ordered]),
        "prompts": np.array([row["prompt"] for row in ordered]),
        "task_ids": np.array([row["task"] for row in ordered]),
        "env_seeds": np.array([row["seed"] for row in ordered], dtype=np.int64),
        "env_steps": np.array(
            [row["env_step"] if row["env_step"] is not None else -1 for row in ordered],
            dtype=np.int64,
        ),
        "replan_indices": np.array(
            [row["replan"] if row["replan"] is not None else -1 for row in ordered],
            dtype=np.int64,
        ),
        "action_noises": np.stack([row["noise"] for row in ordered]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return {
        "kind": "full_context_candidate_state_pooled_archive",
        "archive": str(out),
        "archive_sha256": sha256_file(out),
        "rows": len(ordered),
        "sources": sources,
    }


def deterministic_noise(task: str, seed: int, env_step: int | None, replan: int | None) -> np.ndarray:
    """Stable paired noise shared by teacher and candidates on one pooled row."""
    key = f"{task}|{seed}|{env_step}|{replan}"
    generator = np.random.default_rng(
        int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    )
    return generator.standard_normal((50, 32), dtype=np.float32)


def j_candidate_state(score_doc: dict[str, Any], baseline_scales: dict[str, float]) -> dict[str, Any]:
    """Normalized paired upper bound from a summarize_pair score document."""
    entries = {}
    for metric in ("d_func", "d_pac"):
        vector = np.asarray(
            score_doc[f"{metric}_summary"]["per_sequence"], dtype=np.float64
        )
        mean = float(vector.mean())
        se = float(vector.std(ddof=1) / np.sqrt(len(vector)))
        scale = float(baseline_scales[metric])
        if scale <= 0:
            raise ValueError(f"non-positive baseline scale for {metric}: {scale}")
        entries[metric] = {
            "mean": mean,
            "se": se,
            "upper_bound": (mean + se) / scale,
        }
    return {
        "entries": entries,
        "j": max(entry["upper_bound"] for entry in entries.values()),
    }


def parse_named_buffer(value: str) -> tuple[str, Path]:
    identifier, separator, raw_path = value.partition("=")
    if not separator or not identifier or not raw_path:
        raise argparse.ArgumentTypeError("buffer must be ID=/path.npz")
    return identifier, Path(raw_path).expanduser().resolve()


def parse_named_plan(value: str) -> tuple[str, Path]:
    identifier, separator, raw_path = value.partition("=")
    if not separator or not identifier or not raw_path:
        raise argparse.ArgumentTypeError("candidate plan must be ID=/path.json")
    return identifier, Path(raw_path).expanduser().resolve()


def cmd_pool(args: argparse.Namespace) -> None:
    report = pool_archives(args.candidate_buffer, Path(args.out).expanduser().resolve())
    manifest = Path(str(Path(args.out).expanduser().resolve()) + ".json")
    atomic_json(manifest, report)
    print(json.dumps(report, indent=2))


def cmd_audit(args: argparse.Namespace) -> None:
    # Heavy imports are deferred so the offline pool path stays dependency-free.
    import os

    import torch

    from gr00t_score_outputimpact_plans import (  # noqa: F401
        DEFAULT_CALIBRATION,
        DEFAULT_CHECKPOINT,
        DEFAULT_DATA_CONFIG,
        DEFAULT_PLAN,
    )
    from quantvla_model_adapters import gr00t_rollout_inputs, record_metadata
    from quantvla_metric_protocol import physical_action_scale, summarize_pair
    from gr00t_sensitivity_probe import run_rollouts
    from gr00t_v2_common import (
        DEFAULT_EXCLUDE,
        DEFAULT_INCLUDE,
        load_policy,
        set_quant_env,
        strip_quant_env,
    )
    from quantvla_cross_model_protocol import sha256_file as cross_sha256_file
    from quantvla_outputimpact import install_fp16_bypass, identity_check
    from gr00t.quantization.duquant_layers import DuQuantLinear

    checkpoint = Path(args.checkpoint or str(DEFAULT_CHECKPOINT)).expanduser().resolve()
    data_config = args.data_config or DEFAULT_DATA_CONFIG
    base_plan_path = Path(args.base_full_w4_plan or str(DEFAULT_PLAN)).expanduser().resolve()
    pack_dir = Path(args.pack_dir or str(DEFAULT_CALIBRATION / "identity_pack")).expanduser().resolve()
    hessian_path = Path(args.hessian_w4 or str(DEFAULT_CALIBRATION / "hessian_w4.npz")).expanduser().resolve()
    pooled = Path(args.pooled_buffer).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    archive, fields = load_archive(pooled)
    if not Path(str(hessian_path) + ".json").is_file():
        raise FileNotFoundError(f"hessian sidecar missing: {hessian_path}.json")

    base_plan = json.loads(base_plan_path.read_text(encoding="utf-8"))
    full_names = {
        name
        for name, row in base_plan["layers"].items()
        if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
    }
    candidates: dict[str, Path] = {}
    quantized_names: dict[str, set[str]] = {}
    for identifier, path in args.candidate_plan:
        document = json.loads(path.read_text(encoding="utf-8"))
        names = {
            name
            for name, row in document["layers"].items()
            if not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
        }
        if not names.issubset(full_names):
            raise ValueError(f"{identifier}: candidate contains out-of-scope layers")
        candidates[identifier] = path
        quantized_names[identifier] = names
    if not candidates:
        raise ValueError("at least one candidate plan is required")

    baseline_scales = None
    if args.baseline_scale_json:
        baseline = json.loads(Path(args.baseline_scale_json).read_text(encoding="utf-8"))
        entry = baseline["scores"][args.baseline_scale_entry]
        baseline_scales = {
            "d_func": float(entry["d_func"]),
            "d_pac": float(entry["d_pac"]),
        }

    records = []
    for index in range(len(archive["states"])):
        noise = deterministic_noise(
            str(archive["task_ids"][index]),
            int(archive["env_seeds"][index]),
            (
                int(archive["env_steps"][index])
                if "env_steps" in fields else None
            ),
            (
                int(archive["replan_indices"][index])
                if "replan_indices" in fields else None
            ),
        )
        records.append(
            {
                "observation": {
                    "image": np.asarray(archive["images"][index]),
                    "wrist_image": np.asarray(archive["wrist_images"][index]),
                    "right_image": np.asarray(archive["right_images"][index]),
                    "state": np.asarray(archive["states"][index], dtype=np.float32),
                    "prompt": str(archive["prompts"][index]),
                },
                "task": str(archive["task_ids"][index]),
                "seed": int(archive["env_seeds"][index]),
                "env_step": (
                    int(archive["env_steps"][index])
                    if "env_steps" in fields else None
                ),
                "replan": (
                    int(archive["replan_indices"][index])
                    if "replan_indices" in fields else None
                ),
                "noises": [noise[: args.flow_steps].copy()],
            }
        )
    details = record_metadata(records)
    observations, noises = gr00t_rollout_inputs(records, noise_index=0)

    strip_quant_env()
    torch.manual_seed(0)
    teacher_policy = load_policy(
        str(checkpoint), data_config=data_config,
        denoising_steps=args.flow_steps, device=args.device,
    )
    _, teacher_actions = run_rollouts(
        teacher_policy.model, teacher_policy, observations, noises,
        args.batch_size, return_physical=True,
    )
    scale = physical_action_scale(teacher_actions)

    strip_quant_env()
    set_quant_env(
        DEFAULT_INCLUDE, DEFAULT_EXCLUDE, str(pack_dir),
        row_rot="0", act_dynamic=args.activation_mode == "dynamic_a8",
    )
    os.environ.update(
        {
            "GR00T_DUQUANT_FUSED": "1",
            "GR00T_DUQUANT_PLAN": str(base_plan_path),
            "GR00T_DUQUANT_HESSIAN_W4_PATH": str(hessian_path),
            "GR00T_ATM_ENABLE": "0",
            "GR00T_OHB_ENABLE": "0",
        }
    )
    if args.activation_mode == "fp16":
        os.environ["GR00T_DUQUANT_ABITS"] = "0"
    policy = load_policy(
        str(checkpoint), data_config=data_config,
        denoising_steps=args.flow_steps, device=args.device,
    )
    layers = install_fp16_bypass(policy.model.named_modules(), module_type=DuQuantLinear)
    if set(layers) != full_names:
        raise RuntimeError("live full-W4 layer inventory mismatch")
    for module in layers.values():
        module._outputimpact_fp16 = True
    _, bypass_actions = run_rollouts(
        policy.model, policy, observations, noises, args.batch_size, return_physical=True
    )
    identity = identity_check(teacher_actions, bypass_actions)

    scores = {}
    j_values = {}
    for identifier, path in candidates.items():
        active = quantized_names[identifier]
        for name, module in layers.items():
            module._outputimpact_fp16 = name not in active
        _, actions = run_rollouts(
            policy.model, policy, observations, noises, args.batch_size, return_physical=True
        )
        pair = summarize_pair(teacher_actions, actions, details, scale=scale)
        scores[identifier] = {
            "d_pac": float(pair["d_pac_summary"]["d_pac"]),
            "d_pac_summary": pair["d_pac_summary"],
            "d_func": float(pair["d_func_summary"]["d_func"]),
            "d_func_summary": pair["d_func_summary"],
        }
        if baseline_scales is not None:
            j_values[identifier] = j_candidate_state(scores[identifier], baseline_scales)

    payload = {
        "schema_version": 1,
        "kind": "full_context_candidate_state_teacher_audit",
        "model_adapter": "gr00t",
        "teacher": "original_fp16",
        "pooled_buffer": str(pooled),
        "pooled_buffer_sha256": cross_sha256_file(pooled),
        "checkpoint": str(checkpoint),
        "activation_mode": args.activation_mode,
        "flow_steps": args.flow_steps,
        "n_observations": len(records),
        "fp16_bypass_identity": identity,
        "candidate_plans": {
            identifier: {"path": str(path), "sha256": cross_sha256_file(path)}
            for identifier, path in candidates.items()
        },
        "uses_success_labels": False,
        "scores": scores,
    }
    if baseline_scales is not None:
        payload["baseline_scales"] = baseline_scales
        payload["j_candidate_state"] = j_values
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pool_parser = sub.add_parser("pool")
    pool_parser.add_argument("--candidate-buffer", action="append", type=parse_named_buffer)
    pool_parser.add_argument("--out", required=True)
    pool_parser.set_defaults(handler=cmd_pool)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--checkpoint", default=None)
    audit_parser.add_argument("--data-config", default=None)
    audit_parser.add_argument("--base-full-w4-plan", default=None)
    audit_parser.add_argument("--pack-dir", default=None)
    audit_parser.add_argument("--hessian-w4", default=None)
    audit_parser.add_argument("--activation-mode", choices=("static_a8", "dynamic_a8", "fp16"), default="dynamic_a8")
    audit_parser.add_argument("--pooled-buffer", required=True)
    audit_parser.add_argument("--candidate-plan", action="append", type=parse_named_plan)
    audit_parser.add_argument("--baseline-scale-json")
    audit_parser.add_argument("--baseline-scale-entry", default="context_base")
    audit_parser.add_argument("--flow-steps", type=int, default=4)
    audit_parser.add_argument("--batch-size", type=int, default=8)
    audit_parser.add_argument("--device", default="cuda")
    audit_parser.add_argument("--out", required=True)
    audit_parser.set_defaults(handler=cmd_audit)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()