"""Fast self-tests for formal π0.5 quantization/evaluation invariants."""

from __future__ import annotations

import json
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest
from torch import nn

from openpi.quant import DuQuantConfig, DuQuantLinear, load_act_scales, save_act_scales
from openpi.policies.policy import validate_action_noise
from openpi.serving.websocket_policy_server import ACTION_NOISE_REQUEST_KEY, infer_request
from openpi_client.paired_noise import PROTOCOL, action_noise_seed, paired_action_noise


SERVER_PATH = Path(__file__).resolve().parents[1] / "scripts/serve_pi05_quant_policy.py"
SERVER_SPEC = importlib.util.spec_from_file_location("serve_pi05_quant_policy_test", SERVER_PATH)
assert SERVER_SPEC and SERVER_SPEC.loader
SERVER_MODULE = importlib.util.module_from_spec(SERVER_SPEC)
SERVER_SPEC.loader.exec_module(SERVER_MODULE)


class _ToyModel(nn.Module):
    def __init__(self, pack_dir: str):
        super().__init__()
        cfg = DuQuantConfig(
            weight_bits=4,
            act_bits=8,
            block_size=64,
            block_out_size=64,
            lambda_smooth=0.15,
            enable_permute=False,
            act_percentile=99.9,
            calib_batches=2,
            pack_dir=pack_dir,
            row_rot_mode="restore",
        )
        self.layer = DuQuantLinear(nn.Linear(64, 64), "layer", cfg)
        self._openpi_duquant_runtime = {
            "plan_sha256": "plan-test",
            "wrapped_layers": 1,
            "act_percentile": 99.9,
            "calib_batches": 2,
            "denoising_steps": 4,
            "candidate_inventory_sha256": "inventory-test",
        }


class _CapturePolicy:
    def __init__(self):
        self.calls: list[tuple[dict, np.ndarray | None]] = []

    def infer(self, obs, *, noise=None):
        self.calls.append((obs, noise))
        return {"actions": np.zeros((50, 12), dtype=np.float32)}


def _test_noise_and_dispatch() -> None:
    noise = paired_action_noise("OpenCabinet", 3, 7)
    payload = b"quantvla-robocasa365-v1\0OpenCabinet\0" + b"3\0" + b"7"
    expected_seed = int.from_bytes(__import__("hashlib").sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    assert PROTOCOL == "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1"
    assert action_noise_seed("OpenCabinet", 3, 7) == expected_seed
    generator = torch.Generator(device="cpu").manual_seed(expected_seed)
    assert np.array_equal(
        noise,
        torch.randn((50, 32), generator=generator, dtype=torch.float32).numpy(),
    )
    assert np.array_equal(noise, paired_action_noise("OpenCabinet", 3, 7))
    assert not np.array_equal(noise, paired_action_noise("OpenCabinet", 3, 8))

    policy = _CapturePolicy()
    legacy = {"observation/state": np.zeros(12, dtype=np.float32)}
    infer_request(policy, legacy)
    assert policy.calls[-1][1] is None
    assert policy.calls[-1][0] is not legacy

    paired = {**legacy, ACTION_NOISE_REQUEST_KEY: noise}
    infer_request(policy, paired)
    assert np.array_equal(policy.calls[-1][1], noise)
    assert ACTION_NOISE_REQUEST_KEY not in policy.calls[-1][0]
    assert ACTION_NOISE_REQUEST_KEY in paired

    assert validate_action_noise(noise, action_horizon=50, action_dim=32) is noise
    for invalid in (
        noise.astype(np.float64),
        noise[:49],
        np.full((50, 32), np.nan, dtype=np.float32),
    ):
        try:
            validate_action_noise(invalid, action_horizon=50, action_dim=32)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid deterministic action noise was accepted")


def _test_a8_artifact() -> None:
    torch.manual_seed(11)
    with tempfile.TemporaryDirectory() as temporary:
        pack_dir = str(Path(temporary) / "pack")
        scale_path = Path(temporary) / "scales.npz"
        model = _ToyModel(pack_dir)
        model.layer(torch.randn(3, 64))
        assert not model.layer.act_scale_ready
        model.layer(torch.randn(5, 64))
        assert model.layer.act_scale_ready

        metadata = {
            "checkpoint_sha256": "checkpoint-test",
            "calibration_buffer_sha256": "buffer-test",
        }
        payload = save_act_scales(model, scale_path, metadata)
        assert scale_path.is_file()
        assert Path(str(scale_path) + ".json").is_file()
        assert payload["npz_sha256"]

        restored = _ToyModel(pack_dir)
        loaded = load_act_scales(restored, scale_path, expected_metadata=payload["metadata"])
        assert loaded["npz_sha256"] == payload["npz_sha256"]
        assert restored.layer.act_scale_ready
        assert torch.equal(model.layer._act_scale, restored.layer._act_scale)

        try:
            load_act_scales(restored, scale_path, expected_metadata={"plan_sha256": "wrong"})
        except ValueError as error:
            assert "metadata mismatch" in str(error)
        else:
            raise AssertionError("A8 metadata mismatch was accepted")

        sidecar = Path(str(scale_path) + ".json")
        tampered = json.loads(sidecar.read_text(encoding="utf-8"))
        tampered["npz_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            load_act_scales(restored, scale_path)
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("A8 npz hash mismatch was accepted")

        stale_cfg = DuQuantConfig(
            weight_bits=4,
            act_bits=8,
            block_size=32,
            block_out_size=32,
            lambda_smooth=0.15,
            enable_permute=False,
            pack_dir=pack_dir,
        )
        try:
            DuQuantLinear(nn.Linear(64, 64), "layer", stale_cfg)
        except ValueError as error:
            assert "block_size" in str(error)
        else:
            raise AssertionError("stale block-size pack was accepted")


def test_week1_control_uses_fail_closed_formal_runtime(tmp_path, monkeypatch) -> None:
    plan = tmp_path / "uniform_w6.plan.json"
    plan.write_text(
        json.dumps({"layers": {"layer": {"bits": 6, "skip": False}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENPI_FORMAL_MODE", "1")
    monkeypatch.setenv("OPENPI_FORMAL_WEEK1_CONTROL", "1")
    monkeypatch.setenv("OPENPI_FORMAL_EXPECT_WRAPPED", "1")
    monkeypatch.setenv("OPENPI_CHECKPOINT_SHA256", "checkpoint")
    monkeypatch.setenv("OPENPI_DUQUANT_PLAN", str(plan))
    runtime = {
        "config_id": "uniform_w6",
        "checkpoint_sha256": "checkpoint",
        "protocol": SERVER_MODULE.closed_loop_runtime_protocol(),
        "model_dtype": {
            "resolved": "float16",
            "linear_layers_by_weight_dtype": {"float16": 1},
        },
        "runtime_selector": {"enabled": False},
        "atm_ohb": {"enabled": False, "atm_enabled": False, "ohb_enabled": False},
        "duquant": {
            "enabled": True,
            "act_scales_ready": True,
            "block_in": 64,
            "block_out": 64,
            "act_bits": 8,
            "calib_batches": 32,
            "denoising_steps": 4,
            "wrapped_layers": 1,
            "integer_gemm": False,
            "packed_low_bit_residency": False,
            "plan_path": str(plan.resolve()),
            "plan_sha256": "plan",
            "act_scale_sha256": "a8",
        },
    }
    SERVER_MODULE._validate_formal_runtime(runtime)
    runtime["runtime_selector"] = {"enabled": True}
    with pytest.raises(RuntimeError, match="selector must be disabled"):
        SERVER_MODULE._validate_formal_runtime(runtime)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_rotation_caches_follow_wrapped_weight_device(tmp_path, device: str) -> None:
    cfg = DuQuantConfig(
        weight_bits=4,
        act_bits=8,
        block_size=64,
        block_out_size=64,
        lambda_smooth=0.15,
        enable_permute=True,
        act_percentile=99.9,
        calib_batches=1,
        pack_dir=str(tmp_path / "pack"),
        row_rot_mode="restore",
    )
    layer = DuQuantLinear(nn.Linear(64, 64, device=device), "device_layer", cfg)
    assert layer._weight.device.type == device
    assert layer._perm_cache is None or layer._perm_cache.device == layer._weight.device
    assert all(
        tensor.device == layer._weight.device
        for tensor in layer._get_R_in_cache().values()
    )
    assert all(
        tensor.device == layer._weight.device
        for tensor in layer._get_R_out_cache().values()
    )
    output = layer(torch.randn(2, 64, device=device))
    assert output.device.type == device


def main() -> None:
    _test_noise_and_dispatch()
    _test_a8_artifact()
    print("[pi05 formal protocol] selftest OK")


if __name__ == "__main__":
    main()
