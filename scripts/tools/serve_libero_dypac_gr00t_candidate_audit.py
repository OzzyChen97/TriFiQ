#!/usr/bin/env python3
"""Serve paired GR00T candidate/FP16 actions for FCP state auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from gr00t.eval.robot import RobotInferenceServer
from gr00t.quantization.duquant_layers import DuQuantLinear
from gr00t_v2_common import ensure_flash_attn_rpath, load_policy
from probe_libero_dypac_gr00t_outputimpact import DATA_CONFIGS, configure
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, sha256_file
from quantvla_outputimpact import install_fp16_bypass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(DATA_CONFIGS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--plan", required=True, help="Suite all-W4 wrapper plan")
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure(args)
    ensure_flash_attn_rpath()
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    candidate_path = Path(args.candidate_plan).resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    protected = set(candidate["protected_layers"])
    policy = load_policy(
        str(Path(args.checkpoint).resolve()),
        data_config=DATA_CONFIGS[args.suite],
        denoising_steps=10,
        device=args.device,
    )
    runtime = getattr(policy.model, "_gr00t_duquant_runtime", {})
    layers = install_fp16_bypass(policy.model.named_modules(), module_type=DuQuantLinear)
    if list(layers) != names or runtime.get("hessian_w4_loaded") != len(names):
        raise RuntimeError("GR00T candidate-state runtime inventory mismatch")

    def set_mask(values: set[str]) -> None:
        for name, layer in layers.items():
            layer._outputimpact_fp16 = name in values

    def paired(payload: dict) -> dict:
        if not isinstance(payload, dict) or "observations" not in payload or "action_seed" not in payload:
            raise ValueError("paired audit requires {observations, action_seed}")
        seed = int(payload["action_seed"])
        if not 0 <= seed < 2**63:
            raise ValueError("invalid GR00T action seed")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((16, 32), generator=generator, dtype=torch.float32)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        set_mask(protected)
        candidate_actions = policy.get_action(payload["observations"], action_noise=noise)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        set_mask(set(layers))
        teacher_actions = policy.get_action(payload["observations"], action_noise=noise)
        set_mask(protected)
        return {"candidate": candidate_actions, "teacher": teacher_actions}

    metadata = {
        "kind": "dypac_libero_gr00t_candidate_state_paired_server",
        "model": "gr00t",
        "suite": args.suite,
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_id": args.candidate_id,
        "candidate_plan": str(candidate_path),
        "candidate_plan_sha256": sha256_file(candidate_path),
        "checkpoint_sha256": inventory["checkpoints"][args.suite]["sha256"],
        "hessian_sha256": sha256_file(args.hessian),
        "protected_layers": len(protected),
        "quantized_layers": len(names) - len(protected),
        "teacher": "same_original_fp16_checkpoint_on_candidate_states",
        "uses_success_labels": False,
    }
    server = RobotInferenceServer(policy, port=args.port)
    server.register_endpoint("get_action_seeded_paired", paired)
    server.register_endpoint("get_runtime_info", lambda: metadata, requires_input=False)
    server.run()


if __name__ == "__main__":
    main()
