#!/usr/bin/env python3
"""Fail-closed audit of the pi0.5 port against the GR00T N1.5 final route.

``--phase code`` validates mathematical and protocol definitions before GPU
work.  ``--phase artifacts`` additionally validates the newly frozen
sensitivity/plan/A8/ATM graph.  ``--phase final`` also requires four-config
smoke evidence and the immutable run manifest.  A failed or missing required
check makes the command non-zero; formal evaluation must never bypass it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "scripts" / "tools"
OPENPI_ROOT = REPO_ROOT / "code" / "pi05" / "openpi"
ALIGNED_ROOT = REPO_ROOT / "runs" / "pi05_gdsq_gr00t_aligned"
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("code", "artifacts", "final"), default="code")
    parser.add_argument(
        "--out",
        default=str(ALIGNED_ROOT / "audit/gr00t_final_alignment.json"),
    )
    return parser.parse_args()


class Audit:
    def __init__(self, phase: str):
        self.phase = phase
        self.checks: list[dict[str, Any]] = []

    def check(self, identifier: str, function: Callable[[], Any]) -> None:
        try:
            details = function()
            self.checks.append({"id": identifier, "status": "PASS", "details": details})
        except Exception as error:  # fail-closed report retains every failure
            self.checks.append(
                {
                    "id": identifier,
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(row["status"] == "PASS" for row in self.checks)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def metric_equivalence() -> dict[str, Any]:
    from gr00t_func_metrics import d_func as authority
    from pi05_func_metrics import (
        FUNCTIONAL_FORMULA_ID,
        GR00T_LAYOUT,
        GR00T_WEIGHTS,
        adapt_trajectory,
        d_func,
    )

    generator = torch.Generator().manual_seed(20260820)
    reference = torch.randn(5, 16, 50, 32, generator=generator)
    candidate = reference + 0.03 * torch.randn(5, 16, 50, 32, generator=generator)
    pi_value = d_func(reference, candidate)
    gr_value = authority(
        adapt_trajectory(reference),
        adapt_trajectory(candidate),
        gamma=1.2,
        layout=GR00T_LAYOUT,
        weights=GR00T_WEIGHTS,
    )
    for key in ("d_func", "d_final", "d_kin", "d_grip", "d_solver", "d_mean"):
        require(pi_value[key] == gr_value[key], f"metric mismatch for {key}")
    source = (TOOLS_ROOT / "pi05_func_metrics.py").read_text(encoding="utf-8")
    require(
        "CHUNK_WEIGHT" not in source and "d_exec" not in source.lower(),
        "stale chunk loss remains",
    )
    require(pi_value["adapter"]["formula_id"] == FUNCTIONAL_FORMULA_ID, "formula id mismatch")
    return {
        "authority": str((TOOLS_ROOT / "gr00t_func_metrics.py").resolve()),
        "authority_sha256": sha256_file(TOOLS_ROOT / "gr00t_func_metrics.py"),
        "adapter_sha256": sha256_file(TOOLS_ROOT / "pi05_func_metrics.py"),
        "formula_id": FUNCTIONAL_FORMULA_ID,
        "numeric_exact": True,
        "action_adapter": "H=50->16, D=32->12, gripper=6:7",
    }


def sensitivity_definition() -> dict[str, Any]:
    source = (TOOLS_ROOT / "pi05_sensitivity_probe.py").read_text(encoding="utf-8")
    required = (
        'choices=("final_hidden",)',
        '"cka_location": "action_expert_final_hidden_pre_action_out_proj"',
        '"cs_location": "intervened_target_linear_output"',
        "final_hidden_bank.accumulate_reference(inputs[0])",
        "representation_scores = banks[name].evaluate(q_score_outputs)",
        '"--n-rollout-obs"',
        "GR00T final uses 8 observations x two noises",
        "np.median(wrapper_values)",
        "np.median(wrapper_solver_values)",
    )
    missing = [value for value in required if value not in source]
    require(not missing, f"sensitivity definition missing {missing}")
    require("output[..., :12]" not in source, "CKA still measured at action projection output")
    return {
        "cka": "action expert final hidden state (pre action_out_proj)",
        "cs": "intervened Linear output",
        "importance_aggregation": "8 observations x two paired noises; median",
        "probe_sha256": sha256_file(TOOLS_ROOT / "pi05_sensitivity_probe.py"),
    }


def selector_definition() -> dict[str, Any]:
    import gr00t_select_plan as authority
    import pi05_select_plan as adapter

    require(adapter.final_selector is authority, "selector does not reuse authority module")
    args = argparse.Namespace(cka_only=False, cs_only=False, ratio=16.0)
    identity, cka, cs = adapter._identity_and_lambdas(args)
    require((cka, cs) == (16.0, 1.0), "frozen CKA:CS ratio is not 16:1")
    source_selection = json.loads(
        (REPO_ROOT / "runs/robocasa365_cs_loss/final_selection_official50.json").read_text(
            encoding="utf-8"
        )
    )
    selected = source_selection["selected"]
    require(selected["ratio"] == 16, "GR00T final source ratio changed")
    require(abs(float(selected["dev_task_macro_sr"]) - 0.805) < 1e-12, "source dev SR changed")
    return {
        "authority_sha256": sha256_file(TOOLS_ROOT / "gr00t_select_plan.py"),
        "adapter_sha256": sha256_file(TOOLS_ROOT / "pi05_select_plan.py"),
        "lambda": {"cka": cka, "cs": cs},
        "source_dev_task_macro_sr": selected["dev_task_macro_sr"],
        "identity": identity,
    }


def topk_definition() -> dict[str, Any]:
    source = (TOOLS_ROOT / "pi05_topk_scorer.py").read_text(encoding="utf-8")
    for token in (
        "validate_native_skip",
        "candidate_true_mixed_deployment",
        'scale_dir / f"topk_{index:02d}.npz"',
        "select_final(scored, tol=args.tol, key=\"d_func\")",
        'args.tol, key="d_func"',
    ):
        require(token in source, f"TopK invariant missing: {token}")
    require("tol=0.05" in source, "TopK selftest no longer covers 5% tie rule")
    return {
        "deployment": "true mixed; native nn.Linear for skips",
        "a8": "candidate-specific scales on one fixed buffer",
        "rule": "min D_func; +5% relative tie set; min canonical proxy",
        "sha256": sha256_file(TOOLS_ROOT / "pi05_topk_scorer.py"),
    }


def atm_math() -> dict[str, Any]:
    from pi05_calibrate_atm_ohb import correction_values

    def steps(values: list[float]) -> dict[int, torch.Tensor]:
        return {index: torch.tensor([value]) for index, value in enumerate(values)}

    names = [f"L{index}" for index in range(18)]
    teacher_std = {name: steps([1.0, 4.0, 1.0, 4.0]) for name in names}
    quant_std = {name: steps([2.0, 2.0, 2.0, 2.0]) for name in names}
    teacher_rms = {name: steps([1.0, 4.0, 1.0, 4.0]) for name in names}
    quant_rms = {name: steps([2.0, 2.0, 2.0, 2.0]) for name in names}
    layers, per_step = correction_values(
        teacher_std,
        teacher_rms,
        quant_std,
        quant_rms,
        alpha_min=0.7,
        alpha_max=1.4,
        beta_log_clamp=0.30,
        alpha_neutral=0.02,
        beta_neutral=0.03,
        ohb_mode="per_head_pre_projection",
    )
    expected_alpha = (0.7 + 1.4 + 0.7 + 1.4) / 4
    expected_beta = (math.exp(-0.3) + math.exp(0.3)) / 2
    require(abs(layers["L0"]["all"][0] - expected_alpha) < 1e-6, "ATM math mismatch")
    require(
        abs(layers["L0"]["beta_perhead"][0] - expected_beta) < 1e-6,
        "OHB math mismatch",
    )
    require(len(per_step["L0"]) == 4, "ATM/OHB is not stepwise before pooling")
    return {
        "pooling": "per-step ratio then mean",
        "alpha": {"min": 0.7, "max": 1.4, "neutral": 0.02},
        "beta": {"log_clamp": 0.30, "neutral": 0.03},
        "ohb": "per-head, pre-projection",
        "calibrator_sha256": sha256_file(TOOLS_ROOT / "pi05_calibrate_atm_ohb.py"),
    }


def protocol_definition() -> dict[str, Any]:
    from openpi_client.paired_noise import PROTOCOL, action_noise_seed, paired_action_noise

    require(PROTOCOL == "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1", "noise protocol changed")
    task, seed, replan = "OpenCabinet", 7, 11
    payload = f"quantvla-robocasa365-v1\0{task}\0{seed}\0{replan}".encode()
    expected_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    require(action_noise_seed(task, seed, replan) == expected_seed, "noise seed differs from GR00T")
    generator = torch.Generator(device="cpu").manual_seed(expected_seed)
    expected = torch.randn((50, 32), generator=generator, dtype=torch.float32).numpy()
    require(np.array_equal(paired_action_noise(task, seed, replan), expected), "noise tensor mismatch")
    evaluator = (REPO_ROOT / "scripts/run_robocasa365_pi05_eval.py").read_text(encoding="utf-8")
    for token in (
        'FORMAL_SPLIT = "target"',
        "FORMAL_N_ACTION_STEPS = 16",
        "FORMAL_FLOW_STEPS = 4",
        "enable_render=True",
        "get_task_horizon(task)",
        "env.reset(seed=seed)",
        "convert_action(actions[index])",
        '"fresh_environment": True',
    ):
        require(token in evaluator, f"formal evaluator invariant missing: {token}")
    return {
        "split": "target",
        "seeds": "0-49 explicit",
        "execute": 16,
        "denoising": 4,
        "noise": PROTOCOL,
        "fresh_env": True,
        "render": True,
        "official_horizon": True,
    }


def baseline_definition() -> dict[str, Any]:
    plan_path = ALIGNED_ROOT / "plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    layers = plan["layers"]
    wrapped = sum(not row.get("skip", False) and int(row.get("bits", 0)) == 4 for row in layers.values())
    require(len(layers) == wrapped == 180, "uniform-W4 baseline is not all 180 candidates")
    groups = {int(row["group"]) for row in layers.values()}
    require(groups == {64}, f"uniform-W4 groups changed: {groups}")
    return {
        "wrapped_layers": wrapped,
        "weight_bits": 4,
        "activation_bits": 8,
        "block": 64,
        "ls": 0.15,
        "atm_ohb": "plan-specific static",
        "plan_sha256": sha256_file(plan_path),
    }


def artifact_graph() -> dict[str, Any]:
    from pi05_func_metrics import FUNCTIONAL_FORMULA_ID

    metric_hash = sha256_file(TOOLS_ROOT / "pi05_func_metrics.py")
    sensitivity_path = ALIGNED_ROOT / "sensitivity/pi05_sensitivity_action_n16_d4_merged.json"
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    meta = sensitivity["meta"]
    require(sensitivity.get("complete") is True and len(sensitivity["layers"]) == 180, "sensitivity incomplete")
    require(meta.get("functional_metric_sha256") == metric_hash, "sensitivity metric hash stale")
    require((meta.get("functional_formula") or {}).get("formula_id") == FUNCTIONAL_FORMULA_ID, "formula id stale")
    require(meta.get("cka_location") == "action_expert_final_hidden_pre_action_out_proj", "CKA location stale")
    require(meta.get("cs_location") == "intervened_target_linear_output", "CS location stale")

    plan_path = ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    wrapped = sum(not row.get("skip", False) and int(row.get("bits", 0)) == 4 for row in plan["layers"].values())
    require(len(plan["layers"]) == 180 and 0 < wrapped < 180, "GDSQ layer matrix invalid")
    pmeta = plan["meta"]
    require(pmeta.get("adjudicated") is True and pmeta.get("frozen_before_table1_test") is True, "plan not frozen")
    require(pmeta.get("functional_metric_sha256") == metric_hash, "plan metric hash stale")
    require((pmeta.get("lambda") or {}) == {"cka": 16.0, "cs": 1.0}, "plan ratio differs")

    buffer_hash = sha256_file(ALIGNED_ROOT / "calibration/pi05_robocasa365_seed0_n256.npz")
    records = {}
    for identity, stem, expected_wrapped in (
        ("full_w4a8", "pi05_quantvla_uniform_w4a8_d4_p999_b32x8", 180),
        ("gdsq", "pi05_gdsq_cscka_16to1_d4_p999_b32x8", wrapped),
    ):
        scale = ALIGNED_ROOT / f"a8/{stem}.npz"
        sidecar = json.loads(Path(str(scale) + ".json").read_text(encoding="utf-8"))
        smeta = sidecar["metadata"]
        require(sidecar["npz_sha256"] == sha256_file(scale), f"{identity} A8 hash mismatch")
        require(smeta.get("wrapped_layers") == expected_wrapped, f"{identity} wrapped count mismatch")
        require(smeta.get("calibration_buffer_sha256") == buffer_hash, f"{identity} buffer mismatch")
        require(smeta.get("act_percentile") == 99.9 and smeta.get("calib_batches") == 32, f"{identity} A8 protocol")
        require(smeta.get("denoising_steps") == 4, f"{identity} A8 denoising mismatch")
        records[identity] = {"a8_sha256": sha256_file(scale), "wrapped_layers": expected_wrapped}

    for identity, filename in (
        ("full_w4a8", "pi05_quantvla_uniform_w4a8_d4_static_perhead.json"),
        ("gdsq", "pi05_gdsq_cscka_16to1_d4_static_perhead.json"),
    ):
        artifact_path = ALIGNED_ROOT / "atm_ohb" / filename
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        ameta = artifact["meta"]
        require(len(artifact["layers"]) == 18, f"{identity} ATM layer count")
        require(ameta.get("pooling") == "mean of four per-denoising-step correction ratios", f"{identity} pooling")
        require(ameta.get("alpha_min") == 0.7 and ameta.get("alpha_max") == 1.4, f"{identity} alpha clamp")
        require(ameta.get("beta_log_clamp") == 0.30, f"{identity} beta clamp")
        static_sufficient = (ameta.get("cv_stats") or {}).get("static_sufficient")
        require(isinstance(static_sufficient, bool), f"{identity} missing static CV diagnostic")
        records[identity]["atm_ohb_sha256"] = sha256_file(artifact_path)
        records[identity]["static_sufficient_diagnostic"] = static_sufficient
    return {
        "sensitivity_sha256": sha256_file(sensitivity_path),
        "plan_sha256": sha256_file(plan_path),
        "wrapped_layers": wrapped,
        "artifacts": records,
    }


def final_evidence() -> dict[str, Any]:
    from openpi_client.paired_noise import PROTOCOL as noise_protocol

    smoke = json.loads((ALIGNED_ROOT / "smoke/websocket_smoke.json").read_text(encoding="utf-8"))
    require(smoke.get("complete") is True, "websocket smoke incomplete")
    require(smoke.get("noise_protocol") == noise_protocol, "websocket noise protocol")
    result_root = ALIGNED_ROOT / "smoke/results"
    plan_path = ALIGNED_ROOT / "plans/pi05_gdsq_cscka_16to1_d4.final_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    gdsq_wrapped = sum(
        not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4
        for row in plan["layers"].values()
    )
    expected = {
        "fp16": {"wrapped": 0, "atm": False},
        "quantvla_w4a8_atmohb": {"wrapped": 180, "atm": True},
        "gdsq_vla_atmohb": {"wrapped": gdsq_wrapped, "atm": True},
        "gdsq_vla": {"wrapped": gdsq_wrapped, "atm": False},
    }
    require(set(smoke.get("servers", {})) == set(expected), "websocket smoke config matrix")
    protocol_expected = {
        "action_horizon": 50,
        "n_action_steps": 16,
        "replan_steps": 16,
        "flow_steps": 4,
        "split": "target",
        "fresh_environment_per_episode": True,
        "official_task_horizon": True,
        "render": True,
        "paired_noise": noise_protocol,
    }
    websocket = {}
    for config_id, properties in expected.items():
        server = smoke["servers"][config_id]
        runtime = server.get("runtime") or {}
        require(runtime.get("config_id") == config_id, f"websocket runtime config {config_id}")
        require(server.get("actions_shape") == [50, 12], f"websocket action shape {config_id}")
        require(server.get("finite") is True, f"websocket finite actions {config_id}")
        require(server.get("bitwise_repeat_equal") is True, f"websocket determinism {config_id}")
        require(
            all((runtime.get("protocol") or {}).get(key) == value for key, value in protocol_expected.items()),
            f"websocket protocol {config_id}",
        )
        require(
            (runtime.get("model_dtype") or {}).get("resolved") == "float16",
            f"websocket precision {config_id}",
        )
        require(
            int((runtime.get("duquant") or {}).get("wrapped_layers", 0)) == properties["wrapped"],
            f"websocket wrapper count {config_id}",
        )
        require(
            bool((runtime.get("atm_ohb") or {}).get("enabled")) == properties["atm"],
            f"websocket ATM/OHB state {config_id}",
        )
        websocket[config_id] = {
            "metadata_sha256": server["metadata_sha256"],
            "actions_sha256": server["actions_sha256"],
            "wrapped_layers": properties["wrapped"],
            "atm_ohb": properties["atm"],
        }
    closed_loop = {}
    for config_id in expected:
        files = sorted((result_root / config_id).glob("*.jsonl"))
        require(len(files) == 1, f"closed-loop smoke file matrix {config_id}")
        rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line]
        require(len(rows) == 1 and rows[0].get("status") == "complete", f"invalid smoke row {config_id}")
        row = rows[0]
        require(row.get("config") == config_id, f"closed-loop config mismatch {config_id}")
        require(row.get("task") == "OpenCabinet" and row.get("seed") == 0, f"closed-loop key {config_id}")
        closed_loop_expected = {
            "split": "target",
            "action_horizon": 50,
            "n_action_steps": 16,
            "replan_steps": 16,
            "flow_steps": 4,
            "fresh_environment": True,
            "render_enabled": True,
            "paired_action_noise": True,
            "action_noise_protocol": noise_protocol,
        }
        require(
            all(row.get(key) == value for key, value in closed_loop_expected.items()),
            f"closed-loop protocol mismatch {config_id}",
        )
        require(
            row.get("server_metadata_sha256") == smoke["servers"][config_id]["metadata_sha256"],
            f"closed-loop/websocket server mismatch {config_id}",
        )
        closed_loop[config_id] = {"row": str(files[0]), "sha256": sha256_file(files[0])}
    manifest = ALIGNED_ROOT / "official_target_paired50/manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    require(document.get("immutable") is True, "formal manifest is mutable")
    table = document["table_1_protocol"]
    require(table["trial_seeds"] == list(range(50)), "manifest seeds")
    require(table.get("task_counts") == {"atomic_seen": 18, "composite_seen": 16, "composite_unseen": 16}, "manifest task counts")
    require(set(table.get("task_sets", {})) == {"atomic_seen", "composite_seen", "composite_unseen"}, "manifest task sets")
    require(sum(len(tasks) for tasks in table["task_sets"].values()) == 50, "manifest task matrix")
    manifest_protocol = {
        "split": "target",
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
        "paired_action_noise_protocol": noise_protocol,
    }
    require(
        all(table.get(key) == value for key, value in manifest_protocol.items()),
        "manifest protocol",
    )
    require(document.get("configs") == {
        config_id: {"wrapped_layers": properties["wrapped"], "atm_ohb": properties["atm"]}
        for config_id, properties in expected.items()
    }, "manifest config matrix")
    servers = document.get("servers") or []
    require(len(servers) == 28, "manifest server count")
    for config_id, properties in expected.items():
        replicas = [row for row in servers if row.get("config_id") == config_id]
        require(len(replicas) == 7, f"manifest replica count {config_id}")
        require({row.get("gpu") for row in replicas} == set(range(1, 8)), f"manifest GPUs {config_id}")
        for replica in replicas:
            runtime = replica.get("runtime") or {}
            runtime_file = Path(replica["runtime_file"])
            runtime_document = json.loads(runtime_file.read_text(encoding="utf-8"))
            require(sha256_file(runtime_file) == replica.get("runtime_file_sha256"), f"runtime file hash {replica['instance']}")
            require(canonical_hash(runtime_document) == replica.get("server_metadata_sha256"), f"runtime metadata hash {replica['instance']}")
            require(runtime_document.get("openpi_runtime") == runtime, f"runtime content {replica['instance']}")
            require((runtime.get("model_dtype") or {}).get("resolved") == "float16", f"runtime precision {replica['instance']}")
            require(int((runtime.get("duquant") or {}).get("wrapped_layers", 0)) == properties["wrapped"], f"runtime wrappers {replica['instance']}")
            require(bool((runtime.get("atm_ohb") or {}).get("enabled")) == properties["atm"], f"runtime ATM/OHB {replica['instance']}")
            require(
                all((runtime.get("protocol") or {}).get(key) == value for key, value in protocol_expected.items()),
                f"runtime protocol {replica['instance']}",
            )
    for name, record in document.get("artifacts", {}).items():
        artifact_path = Path(record["path"])
        require(artifact_path.is_file(), f"manifest artifact missing {name}")
        require(sha256_file(artifact_path) == record.get("sha256"), f"manifest artifact hash {name}")
        require(artifact_path.stat().st_size == record.get("bytes"), f"manifest artifact size {name}")
    return {
        "websocket_smoke_sha256": sha256_file(ALIGNED_ROOT / "smoke/websocket_smoke.json"),
        "websocket": websocket,
        "closed_loop": closed_loop,
        "manifest_sha256": sha256_file(manifest),
        "formal_matrix": "4 configs x 50 tasks x 50 explicit seeds = 10000 episodes",
        "server_replicas": len(servers),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    args = parse_args()
    audit = Audit(args.phase)
    for identifier, function in (
        ("metric.exact_authority", metric_equivalence),
        ("sensitivity.cka_cs_importance", sensitivity_definition),
        ("selector.authority_ratio_budget", selector_definition),
        ("topk.true_mixed_adjudication", topk_definition),
        ("atm_ohb.gr00t_final_math", atm_math),
        ("evaluation.protocol_and_noise", protocol_definition),
        ("baseline.uniform_w4a8", baseline_definition),
    ):
        audit.check(identifier, function)
    if args.phase in ("artifacts", "final"):
        audit.check("artifacts.immutable_graph", artifact_graph)
    if args.phase == "final":
        audit.check("final.smoke_and_manifest", final_evidence)
    payload = {
        "schema_version": 1,
        "complete": audit.passed,
        "phase": args.phase,
        "gr00t_final_authority": {
            "metric": str((TOOLS_ROOT / "gr00t_func_metrics.py").resolve()),
            "selector": str((TOOLS_ROOT / "gr00t_select_plan.py").resolve()),
            "topk": str((TOOLS_ROOT / "gr00t_topk_scorer.py").resolve()),
            "atm_ohb": str((TOOLS_ROOT / "calibrate_atm_perstep_gr00t.py").resolve()),
        },
        "checks": audit.checks,
    }
    output = Path(args.out).expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
