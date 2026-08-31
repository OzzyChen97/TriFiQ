#!/usr/bin/env python3
"""Serve paired candidate/FP16 actions for result-blind FCP state auditing."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / "code/pi05/openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import enable_duquant_if_configured, iter_duquant_layers  # noqa: E402
from openpi.quant.duquant_layers import DuQuantLinear  # noqa: E402
from openpi.serving.websocket_policy_server import WebsocketPolicyServer  # noqa: E402
from openpi.training import config  # noqa: E402
from probe_libero_dypac_pi05_outputimpact import configure  # noqa: E402
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, sha256_file  # noqa: E402
from quantvla_outputimpact import install_fp16_bypass  # noqa: E402


class PairedAuditPolicy:
    def __init__(self, policy, layers: dict, protected: set[str], metadata: dict) -> None:
        self.policy = policy
        self.layers = layers
        self.protected = protected
        self.metadata = metadata

    def _set(self, protected: set[str]) -> None:
        for name, layer in self.layers.items():
            layer._outputimpact_fp16 = name in protected

    def infer(self, obs: dict, *, noise=None) -> dict:
        self._set(self.protected)
        candidate = self.policy.infer(copy.deepcopy(obs), noise=noise)
        self._set(set(self.layers))
        teacher = self.policy.infer(copy.deepcopy(obs), noise=noise)
        self._set(self.protected)
        candidate["teacher_actions"] = teacher["actions"]
        return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--plan", required=True, help="All-W4 wrapper plan")
    parser.add_argument("--candidate-plan", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--hessian", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    configure(args, sha256_file(args.buffer))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    candidate_path = Path(args.candidate_plan).resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    protected = set(candidate["protected_layers"])
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"), Path(args.checkpoint_dir).resolve(), pytorch_device=args.device
    )
    runtime = enable_duquant_if_configured(policy._model)
    policy._model.to(args.device).eval()
    layers = install_fp16_bypass(iter_duquant_layers(policy._model), module_type=DuQuantLinear)
    if list(layers) != names or runtime.get("hessian_w4_loaded") != 180:
        raise RuntimeError("candidate-state audit runtime inventory mismatch")
    metadata = {
        "kind": "dypac_libero_candidate_state_paired_server",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_id": args.candidate_id,
        "candidate_plan": str(candidate_path),
        "candidate_plan_sha256": sha256_file(candidate_path),
        "checkpoint_sha256": args.checkpoint_sha256,
        "hessian_sha256": sha256_file(args.hessian),
        "protected_layers": len(protected),
        "quantized_layers": len(names) - len(protected),
        "teacher": "same_original_fp16_checkpoint_on_candidate_states",
        "uses_success_labels": False,
    }
    server = WebsocketPolicyServer(
        policy=PairedAuditPolicy(policy, layers, protected, metadata),
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
