#!/usr/bin/env python3
"""π0.5 single-layer W4 sensitivity probe under the GR00T 4-step protocol."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import jax
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = REPO_ROOT / "code" / "pi05" / "openpi"
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))

from openpi.models import model as model_api  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.quant import enable_duquant_if_configured, sha256_file  # noqa: E402
from openpi.quant.duquant_layers import DuQuantLinear, iter_duquant_layers  # noqa: E402
from openpi.training import config  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
sys.path.insert(0, str(REPO_ROOT / "code"))
from quantvla_cross_model_protocol import (  # noqa: E402
    protocol_artifact,
    protocol_attestation,
)
from quantvla_kernel_scores import LayerScoreBank, extract_tensor  # noqa: E402
from quantvla_model_adapters import load_model_records  # noqa: E402
from pi05_func_metrics import (  # noqa: E402
    ACTION_HORIZON,
    EXECUTED_ACTIONS,
    FLOW_STEPS,
    FUNCTIONAL_FORMULA_ID,
    PAC_FORMULA_ID,
    d_func as final_d_func,
    d_pac_sequence as final_d_pac_sequence,
)
from quantvla_metric_protocol import (  # noqa: E402
    aggregate_d_pac_sequences,
)


CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
DEFAULT_PLAN = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_quantvla_uniform_w4a8_d4.plan.json"
)
DEFAULT_PACK = REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
DEFAULT_BUFFER = protocol_artifact("selection_buffer", verify=False)
DEFAULT_A8 = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/a8/pi05_probe_truefp16_d4_p999_b32x8.npz"
)
DEFAULT_INVENTORY = (
    REPO_ROOT / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_candidate_inventory_d4.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"),
    )
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--pack-dir", default=str(DEFAULT_PACK))
    parser.add_argument("--a8-scale", default=str(DEFAULT_A8))
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "runs/pi05_gdsq_gr00t_aligned/sensitivity/pi05_sensitivity_d4_shard.json"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument(
        "--n-rollout-obs",
        type=int,
        default=8,
        help="GR00T final uses 8 observations x two noises for per-layer D_func importance.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--score-location",
        choices=("final_hidden",),
        default="final_hidden",
        help=(
            "GR00T-final-equivalent CKA at the action expert's final hidden "
            "representation (the input to action_out_proj)."
        ),
    )
    parser.add_argument("--guard-tokens-per-call", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=1.2)
    parser.add_argument("--n-noises-per-obs", type=int, default=2, choices=(1, 2))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-layers", type=int, default=0)
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace, buffer_hash: str) -> None:
    os.environ.update(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "OPENPI_MODEL_DTYPE": "float16",
            "OPENPI_DUQUANT_PLAN": str(Path(args.plan).resolve()),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": "180",
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "0",
            "OPENPI_DUQUANT_ROW_ROT": "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": str(FLOW_STEPS),
            "OPENPI_DUQUANT_PACKDIR": str(Path(args.pack_dir).resolve()),
            "OPENPI_DUQUANT_ACT_SCALE_PATH": str(Path(args.a8_scale).resolve()),
            "OPENPI_DUQUANT_REQUIRE_ACT_SCALE": "1",
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": buffer_hash,
            "OPENPI_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "0",
            "OPENPI_DUQUANT_QUIET": "1",
        }
    )


def load_records(path: Path, n_obs: int) -> list[dict]:
    records, _ = load_model_records(path, n_obs, model="pi05")
    if any(len(record["noises"]) < 2 for record in records):
        raise ValueError("formal sensitivity requires two paired noises per observation")
    return records


def block_name(layer_name: str) -> str:
    prefix, suffix = layer_name.split(".layers.", 1)
    index = suffix.split(".", 1)[0]
    return f"{prefix}.layers.{index}"


def score_location(layer_name: str, mode: str = "hybrid") -> str:
    """Use the representation location aligned with each π0.5 pathway.

    Expert MLP perturbations are measured at their decoder-block output, as in
    the transferred GR00T protocol.  PaliGemma language perturbations are
    measured at the target Linear output: its block output is separated from
    the action expert by the KV-cache/cross-path interface and empirically
    erases the single-layer attribution signal.
    """
    if mode == "block":
        return block_name(layer_name)
    return block_name(layer_name) if ".gemma_expert." in layer_name else layer_name


def tensor_rows(output, cap: int) -> torch.Tensor | None:
    tensor = extract_tensor(output)
    if tensor is None:
        return None
    rows = tensor.detach().reshape(-1, tensor.shape[-1])
    if rows.shape[0] > cap:
        indices = torch.linspace(
            0, rows.shape[0] - 1, cap, device=rows.device
        ).round().long()
        rows = rows.index_select(0, indices)
    return rows.to(dtype=torch.float32).contiguous()


class GuardAccumulator:
    def __init__(self, cap_per_call: int, saturation_threshold: float | None = None):
        self.cap_per_call = cap_per_call
        self.saturation_threshold = saturation_threshold
        self.sumsq: torch.Tensor | None = None
        self.count = 0
        self.amax: torch.Tensor | None = None
        self.samples: list[torch.Tensor] = []
        self.saturated: torch.Tensor | None = None
        self.saturation_count = 0
        self.calls = 0

    def observe(self, output) -> None:
        rows = tensor_rows(output, self.cap_per_call)
        if rows is None:
            return
        partial = rows.square().sum(dim=0)
        self.sumsq = partial if self.sumsq is None else self.sumsq + partial
        self.count += rows.shape[0]
        current_amax = rows.abs().max()
        self.amax = current_amax if self.amax is None else torch.maximum(self.amax, current_amax)
        flat = rows.flatten()
        if flat.numel() > 4096:
            indices = torch.linspace(
                0, flat.numel() - 1, 4096, device=flat.device
            ).round().long()
            flat = flat.index_select(0, indices)
        self.samples.append(flat)
        if self.saturation_threshold is not None:
            current_saturated = (flat.abs() > self.saturation_threshold).sum()
            self.saturated = (
                current_saturated
                if self.saturated is None
                else self.saturated + current_saturated
            )
            self.saturation_count += flat.numel()
        self.calls += 1

    def summary(self) -> dict:
        if self.sumsq is None or self.count == 0:
            return {}
        samples = torch.cat(self.samples) if self.samples else torch.zeros(1)
        rms = (self.sumsq / self.count).sqrt().to(device="cpu")
        amax = float(self.amax.to(device="cpu")) if self.amax is not None else 0.0
        p999 = float(torch.quantile(samples.abs(), 0.999).to(device="cpu"))
        saturated = int(self.saturated.to(device="cpu")) if self.saturated is not None else 0
        return {
            "rms": rms,
            "amax": amax,
            "p999": p999,
            "sat_rate": (
                saturated / self.saturation_count if self.saturation_count else None
            ),
            "calls": self.calls,
            "sample_rows": self.count,
        }


class HookGroup:
    def __init__(self):
        self.handles = []

    def add(self, module: torch.nn.Module, function) -> None:
        self.handles.append(module.register_forward_hook(function))

    def add_pre(self, module: torch.nn.Module, function) -> None:
        self.handles.append(module.register_forward_pre_hook(function))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def prepared_observation(policy, raw: dict, device: str):
    transformed = policy._input_transform(copy.deepcopy(raw))
    batched = jax.tree.map(lambda value: np.asarray(value)[None, ...], transformed)
    tensors = jax.tree.map(
        lambda value: torch.from_numpy(np.array(value)).to(device),
        batched,
    )
    return transformed, model_api.Observation.from_dict(tensors)


def run_records(
    policy,
    records: list[dict],
    device: str,
    *,
    noise_index: int = 0,
) -> tuple[torch.Tensor, np.ndarray, list[float]]:
    trajectories = []
    physical_actions = []
    timings = []
    for record in records:
        transformed, observation = prepared_observation(policy, record["observation"], device)
        noise = record["noises"][noise_index]
        noise_tensor = torch.from_numpy(noise)[None, ...].to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            final, trajectory = policy._model.sample_actions(
                device,
                observation,
                noise=noise_tensor,
                num_steps=FLOW_STEPS,
                return_trajectory=True,
            )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        final_numpy = final[0].detach().to(torch.float32).cpu().numpy()
        output = policy._output_transform(
            {"state": np.asarray(transformed["state"]), "actions": final_numpy}
        )
        physical = np.asarray(output["actions"], dtype=np.float64)
        if physical.shape != (50, 12) or not np.isfinite(physical).all():
            raise RuntimeError(f"invalid probe action output: {physical.shape}")
        trajectories.append(trajectory.detach().to(torch.float32).cpu())
        physical_actions.append(physical)
    return torch.cat(trajectories, dim=1), np.stack(physical_actions), timings


def solver_divergence(reference: torch.Tensor, candidate: torch.Tensor, gamma: float) -> float:
    difference = (reference - candidate).square().sum(dim=(-1, -2))
    denominator = reference.square().sum(dim=(-1, -2)).clamp_min(1e-12)
    relative = difference / denominator
    weights = torch.tensor([gamma ** (index + 1) for index in range(relative.shape[0])])
    weights /= weights.sum()
    return float((relative * weights[:, None]).sum(dim=0).mean())


def independent_d_pac(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    gamma: float,
) -> dict[str, Any]:
    """Action-prefix D_PAC for non-sequential calibration observations."""
    sequences = [
        final_d_pac_sequence(
            reference[:, index : index + 1],
            candidate[:, index : index + 1],
            [0],
            gamma=gamma,
        )
        for index in range(candidate.shape[1])
    ]
    return {**aggregate_d_pac_sequences(sequences), "sequences": sequences}


def serializable_guard(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "rms"}


def compare_guards(reference: dict, candidate: dict) -> dict:
    if not reference or not candidate:
        return {"rms_ratio": None, "amax_ratio": None, "sat_rate": None}
    rms_ratio = ((candidate["rms"] + 1e-6) / (reference["rms"] + 1e-6)).log().abs().median()
    return {
        "rms_ratio": float(rms_ratio),
        "amax_ratio": float(candidate["amax"] / max(reference["amax"], 1e-6)),
        "sat_rate": candidate["sat_rate"],
    }


def set_all_bits(model: torch.nn.Module, bits: int) -> None:
    for _, module in iter_duquant_layers(model):
        module.weight_bits = bits


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def selftest() -> None:
    same = final_d_func(
        torch.ones(FLOW_STEPS + 1, 2, 50, 32),
        torch.ones(FLOW_STEPS + 1, 2, 50, 32),
        1.2,
    )
    assert same["d_func"] == 0.0
    doubled = final_d_func(
        torch.ones(FLOW_STEPS + 1, 2, 50, 32),
        torch.ones(FLOW_STEPS + 1, 2, 50, 32) * 2,
        1.2,
    )
    assert abs(doubled["d_solver"] - 1.0) < 1e-6
    assert doubled["d_func"] > 0.0
    assert block_name("a.layers.17.mlp.gate_proj") == "a.layers.17"
    print("[pi05_sensitivity_probe] selftest OK")


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard index")
    if args.n_obs != 16 or args.n_rollout_obs != 8:
        raise ValueError("formal sensitivity requires n_obs=16 and n_rollout_obs=8")
    output_path = Path(args.out).resolve()
    buffer_path = Path(args.buffer).resolve()
    buffer_hash = sha256_file(buffer_path)
    a8_path = Path(args.a8_scale).resolve()
    a8_sidecar = Path(str(a8_path) + ".json")
    if not a8_path.is_file() or not a8_sidecar.is_file():
        raise FileNotFoundError(f"probe A8 artifact is incomplete: {a8_path}")
    a8_metadata = json.loads(a8_sidecar.read_text(encoding="utf-8")).get("metadata", {})
    if a8_metadata.get("calibration_reference") != "exact_native_fp16_layer_inputs":
        raise ValueError(
            "single-layer attribution requires A8 scales calibrated on exact native-FP16 inputs"
        )
    configure_environment(args, buffer_hash)
    records = load_records(buffer_path, args.n_obs)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    candidate_names = [row["name"] for row in inventory["layers"]]
    shard_names = [
        name for index, name in enumerate(candidate_names) if index % args.num_shards == args.shard_index
    ]
    if args.limit_layers:
        shard_names = shard_names[: args.limit_layers]
    existing = json.loads(output_path.read_text(encoding="utf-8")) if output_path.is_file() else None
    completed = dict(existing.get("layers", {})) if existing else {}
    started = time.time()

    policy = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"),
        Path(args.checkpoint_dir).resolve(),
        pytorch_device=args.device,
    )
    model = policy._model
    modules = dict(model.named_modules())
    missing = [name for name in candidate_names if name not in modules]
    if missing:
        raise RuntimeError(f"pure FP16 inventory missing {len(missing)} layers")

    # Populate every deploy-facing score bank directly from the original
    # native FP16 model.  The later weight_bits=0 wrapper pass is equivalence
    # diagnostics only and never becomes a teacher.
    banks = {name: LayerScoreBank(name, max_tokens=args.max_tokens) for name in candidate_names}
    final_hidden_bank = LayerScoreBank(
        "action_out_proj.input.final_hidden", max_tokens=args.max_tokens
    )
    pure_guards = {name: GuardAccumulator(args.guard_tokens_per_call) for name in candidate_names}
    hooks = HookGroup()
    for name, accumulator in pure_guards.items():
        hooks.add(modules[name], lambda module, inputs, output, acc=accumulator: acc.observe(output))
        hooks.add(
            modules[name],
            lambda module, inputs, output, score_bank=banks[name]: score_bank.accumulate_reference(output),
        )
    hooks.add_pre(
        modules["action_out_proj"],
        lambda module, inputs: final_hidden_bank.accumulate_reference(inputs[0]),
    )
    pure_trajectory, pure_actions, pure_timings = run_records(policy, records, args.device)
    pure_trajectory_b = pure_actions_b = None
    if args.n_noises_per_obs == 2:
        pure_trajectory_b, pure_actions_b, _ = run_records(
            policy, records, args.device, noise_index=1
        )
    hooks.close()
    pure_guard_summaries = {name: accumulator.summary() for name, accumulator in pure_guards.items()}
    del pure_guards
    for bank in banks.values():
        bank.finalize()
        if not bank.ready:
            raise RuntimeError(f"native FP16 reference bank is not ready: {bank.name}")
    final_hidden_bank.finalize()
    if not final_hidden_bank.ready:
        raise RuntimeError("native FP16 action-expert final-hidden bank is not ready")

    runtime = enable_duquant_if_configured(model)
    model.to(args.device)
    if runtime["wrapped_layers"] != 180 or not runtime.get("act_scales_ready"):
        raise RuntimeError(f"invalid quant runtime: {runtime}")
    set_all_bits(model, 0)
    modules = dict(model.named_modules())
    wrapper_trajectory, wrapper_actions, wrapper_timings = run_records(policy, records, args.device)
    reference_action_max_abs = float(np.max(np.abs(pure_actions - wrapper_actions)))
    reference_trajectory_max_abs = float(
        torch.max(torch.abs(pure_trajectory - wrapper_trajectory)).item()
    )
    if reference_action_max_abs != 0.0 or reference_trajectory_max_abs != 0.0:
        raise RuntimeError(
            "weight_bits=0 is not an exact native-FP16 bypass: "
            f"action_max_abs={reference_action_max_abs}, "
            f"trajectory_max_abs={reference_trajectory_max_abs}"
        )
    reference_fingerprints = {
        "actions_sha256": hashlib.sha256(
            np.ascontiguousarray(pure_actions).tobytes()
        ).hexdigest(),
        "trajectory_sha256": hashlib.sha256(
            np.ascontiguousarray(pure_trajectory.numpy()).tobytes()
        ).hexdigest(),
    }
    check_bank = banks[candidate_names[0]]
    scaling_response = {}
    for multiplier in (2.0, 4.0, 8.0):
        scaling_response[str(multiplier)] = check_bank.evaluate_aligned_rows(
            check_bank._reference * multiplier
        )
    cross_values = [
        scaling_response[str(multiplier)]["cs_cross"] for multiplier in (2.0, 4.0, 8.0)
    ]
    cs_cross_monotonic = all(
        left < right for left, right in zip(cross_values, cross_values[1:])
    )
    wrapper_trajectory_b = wrapper_actions_b = None
    if args.n_noises_per_obs == 2:
        wrapper_trajectory_b, wrapper_actions_b, _ = run_records(
            policy, records, args.device, noise_index=1
        )
        if not torch.equal(pure_trajectory_b, wrapper_trajectory_b) or not np.array_equal(
            pure_actions_b, wrapper_actions_b
        ):
            raise RuntimeError("noise-set-B wrapper reference is not bitwise native FP16")

    payload = {
        "schema_version": 1,
        "complete": False,
        "cross_model_protocol": protocol_attestation(),
        "meta": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": inventory["config_sha256"],
            "norm_stats_sha256": inventory["norm_stats_sha256"],
            "candidate_inventory_sha256": inventory["candidate_inventory_sha256"],
            "plan": str(Path(args.plan).resolve()),
            "plan_sha256": sha256_file(args.plan),
            "pack_manifest": str((Path(args.pack_dir) / "manifest.json").resolve()),
            "pack_manifest_sha256": sha256_file(Path(args.pack_dir) / "manifest.json"),
            "a8_scale": str(Path(args.a8_scale).resolve()),
            "a8_scale_sha256": sha256_file(args.a8_scale),
            "a8_calibration_reference": a8_metadata["calibration_reference"],
            "buffer": str(buffer_path),
            "buffer_sha256": buffer_hash,
            "n_obs": args.n_obs,
            "n_rollout_obs": args.n_rollout_obs,
            "n_noises_per_obs": args.n_noises_per_obs,
            "noise_protocol": "torch-cpu-sequential-seed0-f32-v1",
            "evaluation_noise_protocol": (
                "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"
            ),
            "flow_steps": FLOW_STEPS,
            "action_horizon": ACTION_HORIZON,
            "execute_actions": EXECUTED_ACTIONS,
            "intervention_bits": 4,
            "reference": "original native FP16 model (unique score-bank and functional teacher)",
            "wrapped_zero_bit_role": "bitwise-equivalence debug diagnostic only",
            "reference_equivalence": {
                "action_max_abs": reference_action_max_abs,
                "trajectory_max_abs": reference_trajectory_max_abs,
                "bitwise_equal": True,
            },
            "reference_fingerprints": reference_fingerprints,
            "guard_reference": "pure FP16",
            "cka_location": "action_expert_final_hidden_pre_action_out_proj",
            "cs_location": "intervened_target_linear_output",
            "functional_formula": {
                "formula_id": FUNCTIONAL_FORMULA_ID,
                "d_func": "d_final + d_kin + d_grip + 2*d_tail_cvar90",
                "authority": "scripts/tools/gr00t_func_metrics.py:d_func",
                "action_layout": "[translation0:3, rotation3:6, gripper6:7, other_deployed7:12]",
                "padded_dimensions": "12:32 excluded because they are not deployed",
                "action_chunk": "first 16 executed actions only",
                "extra_pi05_terms": "none",
                "gamma": args.gamma,
            },
            "d_pac_formula": {
                "formula_id": PAC_FORMULA_ID,
                "teacher": "original_fp16",
                "sequence_scope": "independent action-prefix (calibration rows are not asserted consecutive)",
                "outer_cvar": 0.9,
                "forecast_overlap_weight": 0.0,
                "model_specific_auxiliary_terms": "forbidden",
            },
            "functional_metric_path": str(
                (REPO_ROOT / "scripts/tools/pi05_func_metrics.py").resolve()
            ),
            "functional_metric_sha256": sha256_file(
                REPO_ROOT / "scripts/tools/pi05_func_metrics.py"
            ),
            "cs_in_situ_check": {
                "layer": check_bank.name,
                "scaled_reference": scaling_response,
                "cross_monotonic": cs_cross_monotonic,
                "required_for_nonzero_lambda_cs": True,
            },
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "expected_layers": shard_names,
            "runtime": {key: value for key, value in runtime.items() if key != "wrapped_layer_names"},
            "reference_latency_s": {
                "pure_mean": float(np.mean(pure_timings)),
                "wrapper_mean": float(np.mean(wrapper_timings)),
            },
            "score_banks": {
                **{name: bank.metadata() for name, bank in banks.items()},
                "action_out_proj.input.final_hidden": final_hidden_bank.metadata(),
            },
        },
        "layers": completed,
    }

    for ordinal, name in enumerate(shard_names, start=1):
        if name in completed:
            print(f"[probe {args.shard_index}] reuse {ordinal}/{len(shard_names)} {name}", flush=True)
            continue
        module = modules[name]
        if not isinstance(module, DuQuantLinear):
            raise RuntimeError(f"target is not wrapped: {name}")
        reference_guard = pure_guard_summaries[name]
        q_guard = GuardAccumulator(
            args.guard_tokens_per_call,
            saturation_threshold=reference_guard.get("p999"),
        )
        q_score_outputs: list[Any] = []
        q_final_hidden_outputs: list[Any] = []
        hooks = HookGroup()
        hooks.add(module, lambda mod, inputs, output: q_guard.observe(output))
        hooks.add(
            modules[name],
            lambda mod, inputs, output: q_score_outputs.append(
                extract_tensor(output).detach().to(device="cpu", dtype=torch.float32)
            ),
        )
        hooks.add_pre(
            modules["action_out_proj"],
            lambda mod, inputs: q_final_hidden_outputs.append(
                inputs[0].detach().to(device="cpu", dtype=torch.float32)
            ),
        )
        module.weight_bits = 4
        layer_started = time.time()
        try:
            q_trajectory, q_actions, q_timings = run_records(policy, records, args.device)
        finally:
            module.weight_bits = 0
            hooks.close()
        representation_scores = banks[name].evaluate(q_score_outputs)
        final_hidden_scores = final_hidden_bank.evaluate(q_final_hidden_outputs)
        q_guard_summary = q_guard.summary()
        guards = compare_guards(reference_guard, q_guard_summary)
        wrapper_func = final_d_func(
            wrapper_trajectory[:, : args.n_rollout_obs],
            q_trajectory[:, : args.n_rollout_obs],
            args.gamma,
        )
        pure_func = final_d_func(
            pure_trajectory[:, : args.n_rollout_obs],
            q_trajectory[:, : args.n_rollout_obs],
            args.gamma,
        )
        wrapper_repeats = [wrapper_func]
        pure_repeats = [pure_func]
        pure_pac_sequences = independent_d_pac(
            pure_trajectory[:, : args.n_rollout_obs],
            q_trajectory[:, : args.n_rollout_obs],
            gamma=args.gamma,
        )["sequences"]
        if args.n_noises_per_obs == 2:
            module.weight_bits = 4
            try:
                q_trajectory_b, q_actions_b, _ = run_records(
                    policy,
                    records[: args.n_rollout_obs],
                    args.device,
                    noise_index=1,
                )
            finally:
                module.weight_bits = 0
            wrapper_repeats.append(
                final_d_func(
                    wrapper_trajectory_b[:, : args.n_rollout_obs],
                    q_trajectory_b,
                    args.gamma,
                )
            )
            pure_repeats.append(
                final_d_func(
                    pure_trajectory_b[:, : args.n_rollout_obs],
                    q_trajectory_b,
                    args.gamma,
                )
            )
            pure_pac_sequences.extend(
                independent_d_pac(
                    pure_trajectory_b[:, : args.n_rollout_obs],
                    q_trajectory_b,
                    gamma=args.gamma,
                )["sequences"]
            )
        wrapper_values = [float(value["d_func"]) for value in wrapper_repeats]
        pure_values = [float(value["d_func"]) for value in pure_repeats]
        wrapper_solver_values = [
            float(value)
            for repeat in wrapper_repeats
            for value in repeat["per_obs"]
        ]
        pure_solver_values = [
            float(value)
            for repeat in pure_repeats
            for value in repeat["per_obs"]
        ]
        wrapper_func = dict(wrapper_func)
        pure_func = dict(pure_func)
        wrapper_func["d_func"] = float(np.median(wrapper_values))
        wrapper_func["d_func_std"] = float(np.std(wrapper_values))
        wrapper_func["d_solver"] = float(np.median(wrapper_solver_values))
        wrapper_func["d_solver_std"] = float(np.std(wrapper_solver_values))
        pure_func["d_func"] = float(np.median(pure_values))
        pure_func["d_func_std"] = float(np.std(pure_values))
        pure_func["d_solver"] = float(np.median(pure_solver_values))
        pure_func["d_solver_std"] = float(np.std(pure_solver_values))
        pure_pac = aggregate_d_pac_sequences(pure_pac_sequences)
        row = {
            "block": block_name(name),
            "family": "action_expert_mlp" if ".gemma_expert." in name else "paligemma_language",
            "b4": {
                **representation_scores,
                "cka_action": final_hidden_scores["cka"],
                "cka_dit": final_hidden_scores["cka"],
                **guards,
                "calls": q_guard_summary.get("calls"),
                "latency_s_mean": float(np.mean(q_timings)),
            },
            "functional_vs_wrapper_ref": wrapper_func,
            "functional_vs_fp16": pure_func,
            "functional_repeats_vs_wrapper_ref": wrapper_repeats,
            "functional_repeats_vs_fp16": pure_repeats,
            "d_pac_b4": float(pure_pac["d_pac"]),
            "d_pac_b4_std": float(np.std(pure_pac["per_sequence"])),
            "d_pac_vs_fp16": pure_pac,
            "guard_reference": serializable_guard(reference_guard),
            "elapsed_s": time.time() - layer_started,
        }
        payload["layers"][name] = row
        atomic_json(output_path, payload)
        print(
            f"[probe {args.shard_index}] {ordinal}/{len(shard_names)} {name} "
            f"cka_final_hidden={final_hidden_scores['cka']:.6f} "
            f"cs={representation_scores['cs']:.6g} "
            f"d_func={pure_func['d_func']:.6g} d_pac={pure_pac['d_pac']:.6g} "
            f"elapsed={row['elapsed_s']:.1f}s",
            flush=True,
        )
        del q_trajectory, q_actions, q_score_outputs, q_final_hidden_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["complete"] = set(payload["layers"]) == set(shard_names)
    payload["meta"]["elapsed_s"] = time.time() - started
    payload["meta"]["completed_layers"] = len(payload["layers"])
    atomic_json(output_path, payload)
    if not payload["complete"]:
        raise RuntimeError("probe shard incomplete")
    print(f"[probe {args.shard_index}] complete: {output_path}")


if __name__ == "__main__":
    main()
