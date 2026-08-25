#!/usr/bin/env python3
"""Create or verify the immutable manifest for the formal π0.5 Table-1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import robocasa  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

from openpi_client.paired_noise import PROTOCOL as NOISE_PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGNED_ROOT = REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--server",
        action="append",
        required=True,
        help="INSTANCE,CONFIG,GPU,PORT,RUNTIME_JSON",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def artifact(path: str) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def git_state(path: Path) -> dict:
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    )
    return {"path": str(path), "head": head, "dirty": dirty}


def parse_server(spec: str) -> dict:
    instance, config_id, gpu_text, port_text, runtime_text = spec.split(",", 4)
    runtime_path = Path(runtime_text).resolve()
    runtime_metadata = json.loads(runtime_path.read_text(encoding="utf-8"))
    actual_config = (runtime_metadata.get("openpi_runtime") or {}).get("config_id")
    if actual_config != config_id:
        raise ValueError(f"{instance}: runtime config {actual_config!r} != {config_id!r}")
    protocol = (runtime_metadata.get("openpi_runtime") or {}).get("protocol") or {}
    expected_protocol = {
        "action_horizon": 50,
        "n_action_steps": 16,
        "replan_steps": 16,
        "flow_steps": 4,
        "split": "target",
        "fresh_environment_per_episode": True,
        "official_task_horizon": True,
        "render": True,
        "paired_noise": NOISE_PROTOCOL,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected_protocol.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{instance}: server is not GR00T N1.5 aligned: {mismatches}")
    return {
        "instance": instance,
        "config_id": config_id,
        "gpu": int(gpu_text),
        "port": int(port_text),
        "runtime_file": str(runtime_path),
        "runtime_file_sha256": sha256_file(runtime_path),
        "server_metadata_sha256": canonical_hash(runtime_metadata),
        "runtime": runtime_metadata["openpi_runtime"],
    }


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    args = parse_args()
    def configured_path(variable: str, default: str) -> str:
        return os.environ.get(variable, default)

    paths = {
        "checkpoint": "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors",
        "checkpoint_config": "checkpoints/robocasa/pi05_pretrain_human300_pytorch/config.json",
        "norm_stats": (
            "checkpoints/robocasa/pi05_pretrain_human300_pytorch/assets/"
            "pi05_pretrain_human300/norm_stats.json"
        ),
        "inventory": "runs/pi05_gdsq_gr00t_aligned/plans/pi05_candidate_inventory_d4.json",
        "pack_manifest": configured_path(
            "PI05_PACK_MANIFEST",
            "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015/manifest.json",
        ),
        "calibration_buffer": configured_path(
            "PI05_CALIBRATION_BUFFER",
            "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz",
        ),
        "sensitivity": configured_path(
            "PI05_SENSITIVITY",
            "runs/pi05_gdsq_gr00t_aligned/sensitivity/pi05_sensitivity_action_n16_d4_merged.json",
        ),
        "full_w4a8_plan": configured_path(
            "PI05_FULL_PLAN",
            "runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json",
        ),
        "gdsq_plan": configured_path(
            "PI05_GDSQ_PLAN",
            "runs/pi05_gdsq_gr00t_aligned/plans/pi05_gdsq_cscka_16to1_d4.final_plan.json",
        ),
        "full_w4a8_a8": configured_path(
            "PI05_FULL_A8",
            "runs/pi05_gdsq_gr00t_aligned/a8/pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz",
        ),
        "gdsq_a8": configured_path(
            "PI05_GDSQ_A8",
            "runs/pi05_gdsq_gr00t_aligned/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
        ),
        "full_w4a8_a8_sidecar": configured_path(
            "PI05_FULL_A8_SIDECAR",
            configured_path(
                "PI05_FULL_A8",
                "runs/pi05_gdsq_gr00t_aligned/a8/pi05_quantvla_uniform_w4a8_d4_p999_b32x8.npz",
            ) + ".json",
        ),
        "gdsq_a8_sidecar": configured_path(
            "PI05_GDSQ_A8_SIDECAR",
            configured_path(
                "PI05_GDSQ_A8",
                "runs/pi05_gdsq_gr00t_aligned/a8/pi05_gdsq_cscka_16to1_d4_p999_b32x8.npz",
            ) + ".json",
        ),
        "full_w4a8_atm_ohb": configured_path(
            "PI05_FULL_ATM",
            "runs/pi05_gdsq_gr00t_aligned/atm_ohb/pi05_quantvla_uniform_w4a8_d4_static_perhead.json",
        ),
        "gdsq_atm_ohb": configured_path(
            "PI05_GDSQ_ATM",
            "runs/pi05_gdsq_gr00t_aligned/atm_ohb/pi05_gdsq_cscka_16to1_d4_static_perhead.json",
        ),
        "final_ratio_selection": configured_path(
            "PI05_FINAL_SELECTION",
            "runs/pi05_gdsq_gr00t_aligned/selection/final_ratio_selection.json",
        ),
        "development_summary": configured_path(
            "PI05_DEV_SUMMARY",
            "runs/pi05_gdsq_gr00t_aligned/selection/dev_summary.json",
        ),
        "development_equivalence": configured_path(
            "PI05_DEV_EQUIVALENCE",
            "runs/pi05_gdsq_gr00t_aligned/selection/executable_equivalence.json",
        ),
        "faithful_selector": "scripts/tools/pi05_select_plan.py",
        "faithful_topk_scorer": "scripts/tools/pi05_topk_scorer.py",
        "faithful_functional_metric": "scripts/tools/pi05_func_metrics.py",
        "development_equivalence_tool": "scripts/tools/pi05_transfer_gr00t_selection.py",
        "finalizer": "scripts/finalize_pi05_faithful.sh",
        "wave_orchestrator": "scripts/run_pi05_faithful_waves.sh",
        "server_launcher": "scripts/run_pi05_formal_server.sh",
        "worker_launcher": "scripts/run_pi05_formal_worker_seeded.sh",
        "evaluator": "scripts/run_robocasa365_pi05_eval.py",
        "aggregator": "scripts/tools/aggregate_pi05_robocasa365.py",
        "strict_parser_test": "scripts/tools/test_pi05_strict_parsers.py",
    }
    servers = [parse_server(value) for value in args.server]
    if len({row["instance"] for row in servers}) != len(servers):
        raise ValueError("duplicate server instance")
    gdsq_payload = json.loads(
        Path(paths["gdsq_plan"]).expanduser().resolve().read_text(encoding="utf-8")
    )
    gdsq_wrapped = sum(
        not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) > 0
        for row in gdsq_payload["layers"].values()
    )
    if len(gdsq_payload["layers"]) != 180 or gdsq_wrapped <= 0:
        raise ValueError("formal GDSQ plan has an invalid candidate inventory")
    if os.environ.get("PI05_REQUIRE_FAITHFUL_FINAL", "0") not in ("0", "false", "False", ""):
        meta = gdsq_payload.get("meta") or {}
        if meta.get("kind") != "gdsq_vla_pi05_faithful_final_frozen":
            raise ValueError("formal GDSQ plan is not the frozen faithful-final artifact")
        if meta.get("adjudicated") is not True or meta.get("frozen_before_table1_test") is not True:
            raise ValueError("formal GDSQ plan lacks adjudication/freeze attestations")
        metric_hash = sha256_file(REPO_ROOT / paths["faithful_functional_metric"])
        if meta.get("functional_metric_sha256") != metric_hash:
            raise ValueError("formal GDSQ plan was frozen with a stale functional metric")
        selection_path = Path(paths["final_ratio_selection"]).expanduser().resolve()
        if meta.get("ratio_selection_sha256") != sha256_file(selection_path):
            raise ValueError("formal GDSQ plan/final ratio-selection hash mismatch")
        dev_summary_path = Path(paths["development_summary"]).expanduser().resolve()
        dev_equivalence_path = Path(paths["development_equivalence"]).expanduser().resolve()
        dev_summaries = meta.get("development_summaries") or []
        if not any(
            Path(row.get("path", "")).resolve() == dev_summary_path
            and row.get("sha256") == sha256_file(dev_summary_path)
            for row in dev_summaries
        ):
            raise ValueError("formal GDSQ plan/development-summary provenance mismatch")
        dev_summary = json.loads(dev_summary_path.read_text(encoding="utf-8"))
        if dev_summary.get("equivalence_manifest_sha256") != sha256_file(
            dev_equivalence_path
        ):
            raise ValueError("development summary/equivalence hash mismatch")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        equivalence = json.loads(dev_equivalence_path.read_text(encoding="utf-8"))
        if meta.get("selection_basis") == "transferred_gr00t_n15_final_ratio":
            if selection.get("selection_basis") != meta.get("selection_basis"):
                raise ValueError("frozen plan/ratio-selection basis mismatch")
            if dev_summary.get("selection_basis") != meta.get("selection_basis"):
                raise ValueError("frozen plan/development-summary basis mismatch")
            if equivalence.get("kind") != "pi05_gr00t_n15_fixed_ratio_transfer":
                raise ValueError("invalid GR00T-to-pi0.5 ratio-transfer provenance")
            if dev_summary.get("pi05_ratio_tuning_rollouts_performed") is not False:
                raise ValueError("aligned transfer must not claim pi0.5 ratio-tuning rollouts")
            if equivalence.get("pi05_ratio_tuning_rollouts_performed") is not False:
                raise ValueError("aligned equivalence must not claim pi0.5 ratio tuning")
            if selection.get("selected", {}).get("ratio") != 16:
                raise ValueError("aligned formal selection must transfer CKA:CS=16:1")
            source_path = Path(selection.get("source_selection_path", "")).resolve()
            source_hash = sha256_file(source_path)
            if any(
                payload.get("source_selection_sha256") != source_hash
                for payload in (selection, dev_summary, equivalence)
            ):
                raise ValueError("GR00T source selection provenance mismatch")
        elif meta.get("pi05_ratio_tuning_rollouts_performed") is not True:
            raise ValueError("unknown or incomplete ratio-selection provenance")
    buffer_hash = sha256_file(Path(paths["calibration_buffer"]).expanduser().resolve())
    pack_manifest_hash = sha256_file(Path(paths["pack_manifest"]).expanduser().resolve())
    for prefix, wrapped in (("full_w4a8", 180), ("gdsq", gdsq_wrapped)):
        plan_path = Path(paths[f"{prefix}_plan"] if prefix == "full_w4a8" else paths["gdsq_plan"]).expanduser().resolve()
        a8_path = Path(paths[f"{prefix}_a8"] if prefix == "full_w4a8" else paths["gdsq_a8"]).expanduser().resolve()
        sidecar_path = Path(paths[f"{prefix}_a8_sidecar"] if prefix == "full_w4a8" else paths["gdsq_a8_sidecar"]).expanduser().resolve()
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        metadata = sidecar.get("metadata") or {}
        expected = {
            "plan_sha256": sha256_file(plan_path),
            "checkpoint_sha256": "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c",
            "calibration_buffer_sha256": buffer_hash,
            "wrapped_layers": wrapped,
            "act_percentile": 99.9,
            "calib_batches": 32,
            "denoising_steps": 4,
            "calibration_batch_size": 8,
            "calibration_observations": 256,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if sidecar.get("npz_sha256") != sha256_file(a8_path):
            mismatches["npz_sha256"] = (sidecar.get("npz_sha256"), sha256_file(a8_path))
        if mismatches:
            raise ValueError(f"{prefix} A8 artifact graph mismatch: {mismatches}")
        atm_key = "full_w4a8_atm_ohb" if prefix == "full_w4a8" else "gdsq_atm_ohb"
        atm = json.loads(Path(paths[atm_key]).expanduser().resolve().read_text(encoding="utf-8"))
        atm_meta = atm.get("meta") or {}
        atm_expected = {
            "plan_sha256": sha256_file(plan_path),
            "a8_scale_sha256": sha256_file(a8_path),
            "pack_manifest_sha256": pack_manifest_hash,
            "calibration_buffer_sha256": buffer_hash,
            "wrapped_layers": wrapped,
            "attention_layers": 18,
            "frames": 16,
            "batch_size": 8,
            "flow_steps": 4,
            "ohb_mode": "per_head_pre_projection",
            "pooling": "mean of four per-denoising-step correction ratios",
            "alpha_min": 0.7,
            "alpha_max": 1.4,
            "beta_log_clamp": 0.30,
            "alpha_neutral": 0.02,
            "beta_neutral": 0.03,
        }
        atm_mismatches = {
            key: (atm_meta.get(key), value)
            for key, value in atm_expected.items()
            if atm_meta.get(key) != value
        }
        if atm_mismatches:
            raise ValueError(f"{prefix} ATM/OHB artifact graph mismatch: {atm_mismatches}")
        if not isinstance((atm_meta.get("cv_stats") or {}).get("static_sufficient"), bool):
            raise ValueError(f"{prefix} ATM/OHB artifact lacks the frozen static-CV diagnostic")
    artifact_rows = {name: artifact(str(REPO_ROOT / path)) for name, path in paths.items()}
    payload = {
        "schema_version": 1,
        "immutable": True,
        "benchmark": "RoboCasa365",
        "table_1_protocol": {
            "split": "target",
            "task_sets": {
                key: list(TASK_SET_REGISTRY[key])
                for key in ("atomic_seen", "composite_seen", "composite_unseen")
            },
            "task_counts": {"atomic_seen": 18, "composite_seen": 16, "composite_unseen": 16},
            "trial_seeds": list(range(50)),
            "fresh_environment_per_trial": True,
            "render_enabled": True,
            "official_task_horizon": True,
            "state_dim": 16,
            "action_dim": 12,
            "action_horizon": 50,
            "n_action_steps": 16,
            "replan_steps": 16,
            "flow_steps": 4,
            "paired_action_noise": True,
            "paired_action_noise_protocol": NOISE_PROTOCOL,
            "selection_functional_metric": {
                "formula": "GR00T final: D_final + D_kin + D_grip + 2*CVaR0.9",
                "adapter": "execute16/deployed12/gripper6:7; no extra pi0.5 term",
                "sha256": sha256_file(REPO_ROOT / paths["faithful_functional_metric"]),
            },
        },
        "configs": {
            "fp16": {"wrapped_layers": 0, "atm_ohb": False},
            "quantvla_w4a8_atmohb": {"wrapped_layers": 180, "atm_ohb": True},
            "gdsq_vla_atmohb": {"wrapped_layers": gdsq_wrapped, "atm_ohb": True},
            "gdsq_vla": {"wrapped_layers": gdsq_wrapped, "atm_ohb": False},
        },
        "artifacts": artifact_rows,
        "servers": servers,
        "git": {
            "quantvla": git_state(REPO_ROOT),
            "openpi": git_state(REPO_ROOT / "code/pi05/openpi"),
        },
    }
    output = Path(args.out).resolve()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"immutable manifest already exists with different content: {output}")
        print(f"formal manifest verified unchanged: {output}")
        return
    atomic_write(output, payload)
    print(f"formal manifest created: {output}")
    print(f"formal manifest sha256: {sha256_file(output)}")


if __name__ == "__main__":
    main()
