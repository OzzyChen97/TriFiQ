from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

import pytest
import torch


TOOLS = Path(__file__).resolve().parent
REPO_ROOT = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code/pi05/openpi/src"))

from fit_softfold_compensation import (  # noqa: E402
    GRID,
    require_grid,
    select_one_standard_error,
    summarize_candidates,
)
from quantvla_cross_model_protocol import PROTOCOL, validate_quant_plan  # noqa: E402
from quantvla_errorfold import (  # noqa: E402
    fit_errorfold,
    fold_dequant_scales,
    fold_linear_parameters,
    gated_affine,
)
from quantvla_hessian_w4 import (  # noqa: E402
    a8_scale_table,
    hessian_aware_w4,
    pack_signed_nibbles,
    unpack_signed_nibbles,
)
from quantvla_metric_protocol import (  # noqa: E402
    aggregate_d_pac_sequences,
    d_pac_sequence,
    physical_action_scale,
    summarize_noise_a_b,
    summarize_pair,
)
from quantvla_model_adapters import (  # noqa: E402
    _resize_uint8_image,
    canonical_physical_chunk,
)
from quantvla_v3_capture import _stratified_rows  # noqa: E402


def _records(count: int = 8) -> list[dict]:
    return [
        {"task": "task", "seed": index // 4, "replan": index % 4}
        for index in range(count)
    ]


def _teacher() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(8, 16, 12, generator=generator) * 0.1


def test_v3_protocol_freezes_physical_action_and_shared_search() -> None:
    assert PROTOCOL["protocol_id"] == "quantvla-gr00t-pi05-errorfold-v3"
    assert PROTOCOL["metrics"]["canonical_action"]["space"].startswith("physical")
    assert PROTOCOL["metrics"]["d_pac"]["formula_id"].startswith("d_pac_v2")
    assert PROTOCOL["metrics"]["d_pac"]["pi05_forecast_overlap"] is False
    assert len(PROTOCOL["softfold"]["grid"]["gate_atm"]) == 9
    assert len(PROTOCOL["softfold"]["grid"]["gate_errorfold"]) == 9


def test_d_pac_v2_fp16_identity_is_strict_zero() -> None:
    teacher = _teacher()
    result = summarize_pair(teacher, teacher, _records())
    assert result["d_func_summary"]["d_func"] == 0.0
    assert result["d_pac_summary"]["d_pac"] == 0.0
    for sequence in result["d_pac_summary"]["sequences"]:
        assert sequence["d_pac_sequence"] == 0.0
        assert all(value == 0.0 for value in sequence["components"].values())


def test_dfunc_one_se_samples_are_task_seed_sequences_not_replans() -> None:
    teacher = _teacher()
    candidate = teacher.clone()
    candidate[0:4, :, 0] += torch.tensor([0.01, 0.02, 0.03, 0.04])[:, None]
    summary = summarize_pair(teacher, candidate, _records())["d_func_summary"]
    assert len(summary["per_obs"]) == 8
    assert len(summary["per_sequence"]) == 2
    assert summary["selection_sample_unit"] == "paired_task_seed_sequence"
    assert summary["per_sequence"][0] == pytest.approx(
        sum(summary["per_obs"][:4]) / 4.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture regression")
def test_v3_capture_stratified_indices_follow_tensor_device() -> None:
    value = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
    rows = _stratified_rows(value, 3)
    assert rows.device.type == "cpu"
    assert rows.shape == (3, 8)


def test_gr00t_adapter_resizes_shared_224_images_deterministically() -> None:
    image = torch.arange(224 * 224 * 3, dtype=torch.int64).remainder(256)
    image = image.to(torch.uint8).reshape(224, 224, 3).numpy()
    first = _resize_uint8_image(image, 256)
    second = _resize_uint8_image(image, 256)
    assert first.shape == (256, 256, 3)
    assert first.dtype == image.dtype
    assert (first == second).all()


def test_global_scale_is_finite_for_static_and_tiny_mad_dimensions() -> None:
    teacher = torch.zeros(8, 16, 12)
    teacher[..., 1] = 1e-10
    teacher[..., 2] = torch.linspace(-1e-8, 1e-8, teacher.shape[0] * 16).reshape(8, 16)
    scale = physical_action_scale(teacher)
    assert torch.isfinite(scale).all()
    assert bool((scale >= 1e-4).all())
    candidate = teacher.clone()
    candidate[..., 0] = 2e-4
    result = summarize_pair(teacher, candidate, _records())
    assert math.isfinite(result["d_pac_summary"]["d_pac"])
    assert result["d_pac_summary"]["d_pac"] < 100.0


def test_constant_bias_accumulates_more_than_equal_mse_alternating_bias() -> None:
    teacher = _teacher()[:4]
    scale = physical_action_scale(_teacher())
    constant = teacher.clone()
    alternating = teacher.clone()
    constant[..., 7] += 0.01
    signs = torch.tensor([1.0, -1.0] * 32).reshape(4, 16)
    alternating[..., 7] += 0.01 * signs
    constant_result = d_pac_sequence(teacher, constant, range(4), scale=scale)
    alternating_result = d_pac_sequence(teacher, alternating, range(4), scale=scale)
    torch.testing.assert_close(
        torch.mean((constant - teacher) ** 2),
        torch.mean((alternating - teacher) ** 2),
    )
    assert constant_result["d_prefix"] > 8.0 * alternating_result["d_prefix"]


def test_stitch_and_gripper_event_terms_are_finite_and_fire() -> None:
    teacher = torch.zeros(3, 16, 12)
    scale = torch.full((12,), 0.1)
    stitch_candidate = teacher.clone()
    stitch_candidate[1, 0, 0] += 0.2
    stitch = d_pac_sequence(teacher, stitch_candidate, range(3), scale=scale)
    assert stitch["d_stitch"] > 0.0

    teacher.reshape(-1, 12)[20:, 6] = 1.0
    delayed = torch.zeros_like(teacher)
    delayed.reshape(-1, 12)[23:, 6] = 1.0
    grip = d_pac_sequence(teacher, delayed, range(3), scale=scale)
    assert math.isfinite(grip["d_grip"]) and grip["d_grip"] > 0.0


def test_different_native_horizons_same_physical_trace_have_same_loss() -> None:
    generator = torch.Generator().manual_seed(29)
    physical = torch.randn(3, 16, 12, generator=generator)
    gr00t = canonical_physical_chunk(physical, model="gr00t")
    pi05_native = torch.zeros(3, 50, 12)
    pi05_native[:, :16] = physical
    pi05 = canonical_physical_chunk(pi05_native, model="pi05")
    torch.testing.assert_close(gr00t, pi05)
    changed = physical.clone()
    changed[..., 0] += 0.01
    scale = physical_action_scale(physical)
    first = d_pac_sequence(gr00t, changed, range(3), scale=scale)
    second = d_pac_sequence(pi05, changed, range(3), scale=scale)
    assert first["d_pac_sequence"] == second["d_pac_sequence"]


def test_sequence_mean_plus_cvar_is_only_outer_tail() -> None:
    result = aggregate_d_pac_sequences([0.0, 1.0, 2.0, 10.0])
    assert result["mean"] == 3.25
    assert result["cvar"] == 10.0
    assert result["d_pac"] == 13.25


def test_noise_b_is_audit_only_and_uses_noise_a_teacher_scale() -> None:
    teacher_a = _teacher()
    teacher_b = teacher_a * 3.0
    result = summarize_noise_a_b(
        teacher_a,
        teacher_a + 0.01,
        teacher_b,
        teacher_b + 0.01,
        _records(),
    )
    assert result["selection_frozen_before_noise_B"] is True
    assert result["dimension_scale_from"].startswith("FP16_noise_A")
    assert result["heldout_noise_B"]["d_pac_summary"]["dimension_scale"] == result[
        "selection_noise_A"
    ]["d_pac_summary"]["dimension_scale"]


def test_errorfold_closed_form_and_reliability_shrinkage() -> None:
    generator = torch.Generator().manual_seed(31)
    quant = torch.randn(128, 6, generator=generator)
    teacher = quant * 1.2 - 0.15
    fitted = fit_errorfold(teacher, quant)
    gain, bias = gated_affine(fitted, 1.0)
    torch.testing.assert_close(quant * gain + bias, teacher, atol=2e-5, rtol=2e-5)
    assert min(fitted["gain_reliability"]) > 0.99

    identity_fit = fit_errorfold(quant, quant)
    assert max(identity_fit["gain_reliability"]) == 0.0
    assert max(identity_fit["bias_reliability"]) == 0.0
    identity_gain, identity_bias = gated_affine(identity_fit, 1.0)
    torch.testing.assert_close(identity_gain, torch.ones_like(identity_gain))
    torch.testing.assert_close(identity_bias, torch.zeros_like(identity_bias))


def test_errorfold_runtime_affine_equals_folded_linear_and_scales() -> None:
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(9, 64, generator=generator)
    weight = torch.randn(7, 64, generator=generator)
    bias = torch.randn(7, generator=generator)
    gain = torch.linspace(0.8, 1.2, 7)
    correction_bias = torch.linspace(-0.1, 0.1, 7)
    runtime = torch.nn.functional.linear(x, weight, bias) * gain + correction_bias
    folded_weight, folded_bias = fold_linear_parameters(
        weight, bias, gain, correction_bias
    )
    folded = torch.nn.functional.linear(x, folded_weight, folded_bias)
    torch.testing.assert_close(runtime, folded, atol=2e-5, rtol=2e-5)
    scales = torch.rand(7, 1, generator=generator) + 0.1
    torch.testing.assert_close(fold_dequant_scales(scales, gain), scales * gain[:, None])


def test_errorfold_supports_direction_reversing_static_fold() -> None:
    generator = torch.Generator().manual_seed(51)
    x = torch.randn(5, 8, generator=generator)
    weight = torch.randn(4, 8, generator=generator)
    bias = torch.randn(4, generator=generator)
    gain = torch.tensor([-0.75, 0.0, 0.5, 1.25])
    correction_bias = torch.linspace(-0.2, 0.2, 4)
    runtime = torch.nn.functional.linear(x, weight, bias) * gain + correction_bias
    folded_weight, folded_bias = fold_linear_parameters(
        weight, bias, gain, correction_bias
    )
    folded = torch.nn.functional.linear(x, folded_weight, folded_bias)
    torch.testing.assert_close(folded, runtime, atol=2e-5, rtol=2e-5)


def test_hessian_w4_is_group64_packed_and_reconstructable() -> None:
    generator = torch.Generator().manual_seed(53)
    weight = torch.randn(10, 128, generator=generator) * 0.1
    inputs = torch.randn(96, 128, generator=generator)
    result = hessian_aware_w4(weight, inputs)
    assert result.scales.shape == (10, 2)
    assert {round(value, 3) for value in result.clipping.unique().tolist()}.issubset(
        {round(value, 3) for value in PROTOCOL["hessian_w4a8"]["clipping_ratios"]}
    )
    packed = pack_signed_nibbles(result.codes)
    assert packed.numel() == weight.numel() // 2
    torch.testing.assert_close(unpack_signed_nibbles(packed, 128), result.codes)
    assert torch.isfinite(result.dequantized).all()


def test_a8_scale_floor_survives_fp16_residency() -> None:
    activations = torch.zeros(32, 64)
    prefix = a8_scale_table(activations)
    flow = a8_scale_table(activations.reshape(4, 8, 64), flow_steps=4)
    assert torch.all(prefix >= 1e-6)
    assert torch.all(flow >= 1e-6)
    assert torch.all(prefix.to(torch.float16) > 0)
    assert torch.all(flow.to(torch.float16) > 0)


def test_a8_has_one_prefix_table_and_four_deterministic_flow_tables() -> None:
    generator = torch.Generator().manual_seed(59)
    prefix = a8_scale_table(torch.randn(8, 12, 64, generator=generator))
    flow = a8_scale_table(
        torch.randn(4, 8, 12, 64, generator=generator), flow_steps=4
    )
    assert prefix.shape == (64,)
    assert flow.shape == (4, 64)
    assert bool((prefix > 0).all()) and bool((flow > 0).all())


def _grid_rows() -> list[dict]:
    rows = []
    for atm in GRID:
        for errorfold in GRID:
            mean = (atm - 0.25) ** 2 + (errorfold - 0.625) ** 2
            rows.append(
                {
                    "gate": {"atm": atm, "errorfold": errorfold},
                    "per_sequence": [mean] * 8,
                    "correction_norm": atm * atm + errorfold * errorfold,
                    "task_name": "must_not_be_used",
                }
            )
    return rows


def test_complete_grid_and_paired_one_se_ignore_task_and_success() -> None:
    rows = _grid_rows()
    summaries = summarize_candidates(rows)
    require_grid(summaries)
    first = select_one_standard_error(summaries)
    changed = copy.deepcopy(rows)
    for index, row in enumerate(changed):
        row["task_name"] = f"other-{index}"
        row["rollout_success"] = bool(index % 2)
    second = select_one_standard_error(summarize_candidates(changed))
    assert first["selected"]["gate"] == {"atm": 0.25, "errorfold": 0.625}
    assert second["selected"]["gate"] == first["selected"]["gate"]


def test_paired_one_se_uses_differences_not_absolute_task_variance() -> None:
    hard_tasks = [100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0]
    rows = [
        {
            "gate": {"atm": 1.0, "errorfold": 1.0},
            "per_sequence": hard_tasks,
            "correction_norm": 2.0,
        },
        {
            "gate": {"atm": 0.0, "errorfold": 0.0},
            "per_sequence": [value + 0.5 for value in hard_tasks],
            "correction_norm": 0.0,
        },
    ]
    selected = select_one_standard_error(summarize_candidates(rows))
    assert selected["eligible_count"] == 1
    assert selected["selected"]["gate"] == {"atm": 1.0, "errorfold": 1.0}


def test_adapter_only_quant_plan_remains_all_w4_group64() -> None:
    valid = {
        "layers": {
            "adapter.a": {"bits": 4, "group": 64, "skip": False},
            "adapter.b": {"bits": 4, "group": 64, "skip": False},
        }
    }
    assert validate_quant_plan(valid, model="gr00t", source="test")["quantized_w4_layers"] == 2
    invalid = copy.deepcopy(valid)
    invalid["layers"]["adapter.b"]["skip"] = True
    try:
        validate_quant_plan(invalid, model="pi05", source="test")
    except ValueError as error:
        assert "every target W4/group64" in str(error)
    else:
        raise AssertionError("FP target retention must be rejected")


def test_both_real_quant_backends_pack_identical_grouped_nibbles() -> None:
    from gr00t.quantization.duquant_fused import pack_w4_nibbles as pack_gr00t
    from openpi.quant.duquant_triton import pack_w4_nibbles as pack_pi05

    weight = torch.linspace(-1.0, 1.0, 128).repeat(2, 1)
    scales = torch.tensor([[0.1, 0.2], [0.08, 0.16]])
    gr00t = pack_gr00t(weight, scales)
    pi05 = pack_pi05(weight, scales)
    assert torch.equal(gr00t, pi05)
    assert gr00t.numel() == weight.numel() // 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton W4 kernel needs CUDA")
def test_directional_head_fold_is_exact_static_dequant_not_requantized() -> None:
    from gr00t.quantization.duquant_fused import (
        fused_linear_w4_nibbles as gr00t_w4,
        pack_w4_nibbles as pack_gr00t,
    )
    from openpi.quant.duquant_triton import _w4_linear as pi05_w4

    generator = torch.Generator().manual_seed(71)
    weight = torch.randn(64, 128, generator=generator, dtype=torch.float32).cuda() * 0.05
    scales = torch.stack(
        [
            weight[:, :64].abs().amax(dim=1) / 7.0,
            weight[:, 64:].abs().amax(dim=1) / 7.0,
        ],
        dim=1,
    ).clamp_min(1e-6)
    packed = pack_gr00t(weight, scales)
    codes = unpack_signed_nibbles(packed.cpu(), 128).cuda().to(torch.float32)
    dequant = codes * scales.repeat_interleave(64, dim=1)
    gain = torch.linspace(0.8, 1.2, 128, device="cuda")
    x = torch.randn(9, 128, generator=generator, dtype=torch.float16).cuda()
    reference = x.float() @ (dequant * gain[None, :]).T
    gr00t = gr00t_w4(x, packed, scales, input_gain=gain).float()
    pi05 = pi05_w4(x, packed, scales, input_gain=gain).float()
    torch.testing.assert_close(gr00t, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(pi05, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton W4 kernel needs CUDA")
def test_pi05_fused_w4_covers_small_output_projection() -> None:
    from openpi.quant.duquant_triton import (
        duquant_linear_fused_w4,
        pack_w4_nibbles,
    )

    generator = torch.Generator().manual_seed(73)
    weight = torch.randn(256, 128, generator=generator, dtype=torch.float32).cuda() * 0.03
    scales = torch.stack(
        [
            weight[:, :64].abs().amax(dim=1) / 7.0,
            weight[:, 64:].abs().amax(dim=1) / 7.0,
        ],
        dim=1,
    ).clamp_min(1e-6)
    packed = pack_w4_nibbles(weight, scales)
    codes = unpack_signed_nibbles(packed.cpu(), 128).cuda().to(torch.float32)
    dequant = codes * scales.repeat_interleave(64, dim=1)
    x = torch.randn(9, 128, generator=generator, dtype=torch.float16).cuda()
    act_scale = torch.full((128,), 1e-6, dtype=torch.float16, device="cuda")
    x_quant = torch.clamp(torch.round(x / act_scale), -128, 127) * act_scale
    expected = (x_quant.float() @ dequant.T).to(torch.float16)
    actual = duquant_linear_fused_w4(
        x,
        packed,
        scales,
        None,
        None,
        None,
        act_scale,
        None,
        B=64,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton W4 kernel needs CUDA")
def test_pi05_w4_masks_ragged_batch_rows() -> None:
    from openpi.quant.duquant_triton import (
        duquant_linear_fused_w4,
        pack_w4_nibbles,
    )

    generator = torch.Generator().manual_seed(79)
    weight = torch.randn(64, 64, generator=generator, dtype=torch.float32).cuda() * 0.03
    scales = (weight.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-6)
    packed = pack_w4_nibbles(weight, scales)
    codes = unpack_signed_nibbles(packed.cpu(), 64).cuda().to(torch.float32)
    dequant = codes * scales
    act_scale = torch.full((64,), 0.01, dtype=torch.float16, device="cuda")
    for rows in (1, 63, 65, 129):
        x = torch.randn(rows, 64, generator=generator, dtype=torch.float16).cuda()
        x_quant = torch.clamp(torch.round(x / act_scale), -128, 127) * act_scale
        expected = (x_quant.float() @ dequant.T).to(torch.float16)
        actual = duquant_linear_fused_w4(
            x,
            packed,
            scales,
            None,
            None,
            None,
            act_scale,
            None,
            B=64,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton W4 kernel needs CUDA")
def test_gr00t_w4_masks_ragged_batch_rows() -> None:
    from gr00t.quantization.duquant_fused import (
        fused_linear_w4_nibbles,
        pack_w4_nibbles,
    )

    generator = torch.Generator().manual_seed(83)
    weight = torch.randn(64, 64, generator=generator, dtype=torch.float32).cuda() * 0.03
    scales = (weight.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-6)
    packed = pack_w4_nibbles(weight, scales)
    codes = unpack_signed_nibbles(packed.cpu(), 64).cuda().to(torch.float32)
    dequant = codes * scales
    for rows in (1, 63, 65, 129):
        x = torch.randn(rows, 64, generator=generator, dtype=torch.float16).cuda()
        expected = (x.float() @ dequant.T).to(torch.float16)
        actual = fused_linear_w4_nibbles(x, packed, scales)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_pi05_finalized_weight_exposes_only_dtype_metadata() -> None:
    from openpi.quant.duquant_layers import DuQuantLinear

    layer = object.__new__(DuQuantLinear)
    torch.nn.Module.__init__(layer)
    layer.name = "test.finalized"
    layer.register_buffer("_weight", None)
    layer.register_buffer(
        "_weight_metadata",
        torch.empty(0, dtype=torch.bfloat16),
        persistent=False,
    )
    layer._inference_only_ready = True
    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.numel() == 0


def test_gr00t_paper_baseline_uses_shared_buffer_a8_artifact(tmp_path: Path) -> None:
    from make_errorfold_v3_gr00t_specs import build

    spec = build(tmp_path, "atomic_seen")
    paper = next(row for row in spec["configs"] if row["id"] == "quantvla_w4a8_paper")
    assert Path(paper["act_scale"]) == (
        tmp_path
        / "calibration/gr00t/atomic_seen/paper_a8_shared_n256.npz"
    )
    server_gpus = [
        gpu
        for row in spec["configs"]
        for gpu in [row["gpu"], *(replica["gpu"] for replica in row["replicas"])]
    ]
    assert 0 not in server_gpus
    assert server_gpus.count(3) == 2
    assert len(server_gpus) == 8


def test_shared_gpu_preflight_accumulates_both_servers_and_clients() -> None:
    from run_robocasa_atomic_matrix import egl_pool_memory_requirements

    manifest = {
        "protocol": {
            "allow_shared_gpus": True,
            "egl_device_pool": [3],
            "shard_egl_devices": {"fp16": [3, 3, 3, 3, 3]},
        },
        "configs": [
            {"gpu": 3, "expected_wrapped": 0},
            {"gpu": 3, "expected_wrapped": 116},
        ],
    }
    assert egl_pool_memory_requirements(manifest) == {
        3: 2048.0 + 5 * 2000.0 + 12288.0 + 16384.0
    }
