from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import json

import torch


TOOLS = Path(__file__).resolve().parent
REPO_ROOT = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code/pi05/openpi/src"))

from fit_softfold_compensation import (  # noqa: E402
    GRID,
    fit,
    fold_layers,
    sample_split,
    select_one_standard_error,
    softfold_value,
    summarize_candidates,
)
from pi05_func_metrics import d_pac_sequence as pi05_d_pac_sequence  # noqa: E402
from quantvla_cross_model_protocol import (  # noqa: E402
    PROTOCOL,
    protocol_attestation,
    validate_quant_plan,
)
from quantvla_model_adapters import canonical_trajectory  # noqa: E402
from gr00t_func_metrics import aggregate_d_pac_sequences, d_pac_sequence  # noqa: E402
from gr00t_select_plan import (  # noqa: E402
    layer_bytes_fp16,
    plan_total_bytes,
    quantvla_w4_budget,
    quantvla_w4_plan,
)


def _teacher() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(4, 8, 7, generator=generator) * 0.1


def test_d_pac_fp16_identity_is_exact_zero() -> None:
    teacher = _teacher()
    result = d_pac_sequence(teacher, teacher, [3, 1, 2, 0], executed_actions=8)
    assert result["d_pac_sequence"] == 0.0
    assert result["d_prefix"] == 0.0
    assert result["d_stitch"] == 0.0
    assert result["d_grip_time"] == 0.0
    assert result["teacher"] == "original_fp16"


def test_constant_bias_accumulates_more_than_equal_mse_alternating_bias() -> None:
    teacher = _teacher()
    constant = teacher.clone()
    alternating = teacher.clone()
    constant[..., 0] += 0.01
    signs = torch.tensor([1.0, -1.0] * 16).reshape(4, 8)
    alternating[..., 0] += 0.01 * signs
    constant_result = d_pac_sequence(teacher, constant, range(4), executed_actions=8)
    alternating_result = d_pac_sequence(teacher, alternating, range(4), executed_actions=8)
    torch.testing.assert_close(
        torch.mean((constant - teacher) ** 2),
        torch.mean((alternating - teacher) ** 2),
    )
    assert constant_result["d_prefix"] > 10.0 * alternating_result["d_prefix"]


def test_stitch_and_gripper_timing_terms_fire() -> None:
    teacher = torch.zeros(3, 6, 7)
    teacher[..., :6] = torch.linspace(-0.2, 0.2, 18).reshape(3, 6, 1)
    stitch_candidate = teacher.clone()
    stitch_candidate[1, 0, 0] += 0.2
    stitch = d_pac_sequence(teacher, stitch_candidate, range(3), executed_actions=6)
    assert stitch["d_stitch"] > 0.0

    teacher[..., 6] = -1.0
    teacher.reshape(-1, 7)[7:, 6] = 1.0
    delayed = teacher.clone()
    delayed[..., 6] = -1.0
    delayed.reshape(-1, 7)[9:, 6] = 1.0
    grip = d_pac_sequence(teacher, delayed, range(3), executed_actions=6)
    assert grip["d_grip_time"] > 0.0


def test_sequence_cvar_is_applied_once_at_outer_level() -> None:
    result = aggregate_d_pac_sequences([0.0, 1.0, 2.0, 10.0], tail_weight=2.0)
    assert result["mean"] == 3.25
    assert result["cvar"] == 10.0
    assert result["d_pac"] == 23.25


def _grid_rows() -> list[dict]:
    rows = []
    for atm in GRID:
        for ohb in GRID:
            mean = (atm - 0.25) ** 2 + (ohb - 0.625) ** 2
            rows.append(
                {
                    "gate": {"atm": atm, "ohb": ohb},
                    "per_sequence": [mean, mean, mean, mean],
                    "task_name": "must_not_be_used",
                }
            )
    return rows


def test_softfold_grid_and_one_standard_error_selection_ignore_task_metadata() -> None:
    rows = _grid_rows()
    first = select_one_standard_error(summarize_candidates(rows))
    changed = copy.deepcopy(rows)
    for index, row in enumerate(changed):
        row["task_name"] = f"different-{index}"
        row["rollout_success"] = bool(index % 2)
    second = select_one_standard_error(summarize_candidates(changed))
    assert first["selected"]["gate"] == {"atm": 0.25, "ohb": 0.625}
    assert second["selected"]["gate"] == first["selected"]["gate"]


def test_softfold_log_interpolation_and_layer_artifact() -> None:
    assert softfold_value(4.0, 0.0) == 1.0
    assert softfold_value(4.0, 1.0) == 4.0
    assert softfold_value(4.0, 0.5) == 2.0
    layers, mode = fold_layers(
        {"layer": {"all": [4.0, 0.25], "beta_perhead": [9.0, 1.0 / 9.0]}},
        gate_atm=0.5,
        gate_ohb=0.5,
    )
    assert mode == "per_head_pre_projection"
    assert layers["layer"]["all"] == [2.0, 0.5]
    assert math.isclose(layers["layer"]["beta_perhead"][0], 3.0)
    assert math.isclose(layers["layer"]["beta_perhead"][1], 1.0 / 3.0)
    assert sample_split("abc") == sample_split("abc")


def test_gr00t_perhead_ohb_fold_matches_runtime_scaling() -> None:
    from gr00t.atm.dit_atm import _fold_ohb_perhead_into_o_projection

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = 2
            self.to_out = torch.nn.ModuleList(
                [torch.nn.Linear(6, 4, bias=True), torch.nn.Identity()]
            )

    generator = torch.Generator().manual_seed(11)
    module = FakeAttention()
    inputs = torch.randn(3, 5, 6, generator=generator)
    beta = torch.tensor([0.75, 1.25])
    runtime_scale = beta.repeat_interleave(3)
    expected = module.to_out[0](inputs * runtime_scale)
    _fold_ohb_perhead_into_o_projection(module, beta)
    actual = module.to_out[0](inputs)
    torch.testing.assert_close(actual, expected)


def test_exact_quantvla_budget_forces_full_w4_in_binary_space() -> None:
    shapes = {
        "small": {"out": 512, "in": 512, "has_bias": False},
        "large": {"out": 1024, "in": 512, "has_bias": True},
    }
    plan = quantvla_w4_plan(shapes, group=64, row_rot="restore")
    budget = quantvla_w4_budget(shapes, group=64, row_rot="restore")
    assert all(not row["skip"] and row["bits"] == 4 for row in plan.values())
    assert plan_total_bytes(plan, shapes, "restore") == budget
    retained = copy.deepcopy(plan)
    retained["small"] = {"bits": None, "group": 64, "skip": True}
    assert plan_total_bytes(retained, shapes, "restore") > budget
    assert layer_bytes_fp16(512, 512, False) > 0.0


def test_adapter_only_plan_rule_accepts_uniform_w4_and_rejects_retention() -> None:
    valid = {
        "layers": {
            "adapter.layer.a": {"bits": 4, "group": 64, "skip": False},
            "adapter.layer.b": {"bits": 4, "group": 64, "skip": False},
        }
    }
    attestation = validate_quant_plan(valid, model="gr00t", source="test")
    assert attestation["quantized_w4_layers"] == 2
    invalid = copy.deepcopy(valid)
    invalid["layers"]["adapter.layer.b"] = {"bits": None, "group": 64, "skip": True}
    try:
        validate_quant_plan(invalid, model="pi05", source="test")
    except ValueError as error:
        assert "every target W4/group64" in str(error)
    else:
        raise AssertionError("adapter-only protocol must reject FP16 target retention")


def test_model_adapters_map_to_identical_canonical_metric_space() -> None:
    generator = torch.Generator().manual_seed(29)
    shared = torch.randn(5, 3, 16, 12, generator=generator)
    gr00t = torch.zeros(5, 3, 16, 32)
    pi05 = torch.zeros(5, 3, 50, 32)
    gr00t[..., :12] = shared
    pi05[..., :16, :12] = shared
    torch.testing.assert_close(
        canonical_trajectory(gr00t, model="gr00t"),
        canonical_trajectory(pi05, model="pi05"),
    )
    try:
        pi05_d_pac_sequence(pi05, pi05, range(3), overlap_weight=0.1)
    except ValueError as error:
        assert "overlap is forbidden" in str(error)
    else:
        raise AssertionError("pi0.5-only overlap must be rejected")


def test_both_real_quant_backends_use_identical_two_values_per_byte_w4() -> None:
    from gr00t.quantization.duquant_fused import pack_w4_nibbles as pack_gr00t
    from openpi.quant.duquant_triton import pack_w4_nibbles as pack_pi05

    weight = torch.tensor([[-8.0, -7.0, -1.0, 0.0, 1.0, 7.0]])
    scales = torch.ones(1)
    gr00t = pack_gr00t(weight, scales)
    pi05 = pack_pi05(weight, scales)
    assert torch.equal(gr00t, pi05)
    assert gr00t.numel() == weight.numel() // 2
    low = (gr00t & 0x0F).to(torch.int16)
    high = ((gr00t >> 4) & 0x0F).to(torch.int16)
    unpacked = torch.stack((low, high), dim=-1).reshape_as(weight)
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    assert torch.equal(unpacked, weight.to(torch.int16))


def test_softfold_fitter_emits_selector_free_fold_artifact(tmp_path: Path) -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    digest_c = PROTOCOL["data"]["calibration_buffer"]["sha256"]
    raw_path = tmp_path / "raw.json"
    scores_path = tmp_path / "scores.json"
    raw_path.write_text(
        json.dumps(
            {
                "meta": {
                    "checkpoint_sha256": digest_a,
                    "plan_sha256": digest_b,
                    "calibration_buffer_sha256": digest_c,
                    "flow_steps": 4,
                    "frames": 16,
                    "batch_size": 8,
                    "ohb_mode": "per_head_pre_projection",
                },
                "layers": {"layer": {"all": [4.0], "beta_perhead": [9.0]}},
            }
        ),
        encoding="utf-8",
    )
    raw_sha = __import__("hashlib").sha256(raw_path.read_bytes()).hexdigest()
    scores_path.write_text(
        json.dumps(
            {
                "cross_model_protocol": protocol_attestation(),
                "checkpoint_sha256": digest_a,
                "plan_sha256": digest_b,
                "artifact_calibration_buffer_sha256": digest_c,
                "raw_correction_sha256": raw_sha,
                "scores": _grid_rows(),
            }
        ),
        encoding="utf-8",
    )
    payload = fit(
        SimpleNamespace(
            raw_correction=str(raw_path),
            validation_scores=str(scores_path),
            allow_partial_grid=False,
            lambda_identity=0.0,
            lambda_interaction=0.0,
            teacher_checkpoint_sha256=None,
            teacher_checkpoint=None,
            quant_plan_sha256=None,
            quant_plan=None,
            buffer_sha256=None,
            buffer=None,
            metric="d_pac_v1",
        )
    )
    assert payload["gate"] == {"atm": 0.25, "ohb": 0.625}
    assert payload["selection"]["uses_task_labels"] is False
    assert payload["selection"]["uses_rollout_success"] is False
    assert payload["meta"]["atm_application"] == "fold_q_weight"
    assert payload["meta"]["ohb_application"] == "fold_o_weight_perhead"
    assert payload["teacher_checkpoint_sha256"] == digest_a
