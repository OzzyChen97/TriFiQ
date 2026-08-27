#!/usr/bin/env python3
"""CPU unit tests for paper-faithful π0.5 ATM/OHB wiring."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaAttention


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/pi05/openpi/src"))
sys.path.insert(0, str(REPO_ROOT / "scripts/tools"))

from openpi.quant.atm_pi05 import (  # noqa: E402
    clear_atm_capture,
    enable_pi05_atm_if_configured,
    register_ohb_output_capture,
)
from pi05_calibrate_atm_ohb import correction_values  # noqa: E402


LAYER = "paligemma_with_expert.gemma_expert.model.layers.0.self_attn"


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = GemmaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            num_hidden_layers=1,
            attention_bias=False,
        )
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Module()
        layer = nn.Module()
        layer.self_attn = GemmaAttention(config, layer_idx=0)
        self.paligemma_with_expert.gemma_expert.model.layers = nn.ModuleList([layer])


class PaperAtmOhbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_env = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.old_env)

    def test_scalar_capture_is_post_projection(self) -> None:
        model = ToyModel()
        values = {}
        register_ohb_output_capture(model, lambda name, rms: values.setdefault(name, rms))
        attention = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn
        x = torch.randn(2, 3, 8)
        output = attention.o_proj(x)
        self.assertIn(LAYER, values)
        expected = torch.sqrt(torch.mean(output.float() ** 2) + 1e-12)
        torch.testing.assert_close(values[LAYER], expected)
        clear_atm_capture(model)

    def test_teacher_to_student_ratio_has_restorative_direction(self) -> None:
        names = [LAYER.replace(".layers.0.", f".layers.{index}.") for index in range(18)]
        per_step = lambda value: {step: torch.as_tensor(value) for step in range(4)}
        layers, serialized_steps = correction_values(
            {name: per_step([1.20, 0.90]) for name in names},
            {name: per_step(1.10) for name in names},
            {name: per_step([1.00, 1.00]) for name in names},
            {name: per_step(1.00) for name in names},
            alpha_min=0.70,
            alpha_max=1.40,
            beta_log_clamp=0.30,
            alpha_neutral=0.0,
            beta_neutral=0.0,
            ohb_mode="per_layer_post_projection",
        )
        alpha = torch.tensor(layers[LAYER]["all"])
        beta = layers[LAYER]["beta"]
        torch.testing.assert_close(alpha * torch.tensor([1.00, 1.00]), torch.tensor([1.20, 0.90]))
        self.assertAlmostEqual(beta * 1.00, 1.10, places=6)
        self.assertEqual(sorted(serialized_steps[LAYER]), ["0", "1", "2", "3"])

    def test_folded_atm_and_scalar_ohb(self) -> None:
        model = ToyModel()
        attention = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn
        q_before = attention.q_proj.weight.detach().clone()
        o_before = attention.o_proj.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "paper_atm.json"
            artifact.write_text(
                json.dumps(
                    {
                        "meta": {
                            "ohb_mode": "per_layer_post_projection",
                            "atm_application": "fold_q_weight",
                            "ohb_application": "fold_o_weight",
                        },
                        "layers": {LAYER: {"all": [2.0, 0.5], "beta": 1.25}},
                    }
                ),
                encoding="utf-8",
            )
            os.environ.update(
                {
                    "OPENPI_ATM_ENABLE": "1",
                    "OPENPI_OHB_ENABLE": "1",
                    "OPENPI_ATM_ALPHA_PATH": str(artifact),
                    "OPENPI_ATM_SCOPE": "expert",
                    "OPENPI_OHB_SCOPE": "expert",
                    "OPENPI_ATM_STRICT": "1",
                    "OPENPI_ATM_APPLICATION": "fold_q_weight",
                    "OPENPI_OHB_EXPECT_MODE": "per_layer_post_projection",
                    "OPENPI_OHB_APPLICATION": "fold_o_weight",
                }
            )
            enable_pi05_atm_if_configured(model)

        expected_q = q_before.clone()
        expected_q[:4].mul_(2.0)
        expected_q[4:].mul_(0.5)
        torch.testing.assert_close(attention.q_proj.weight, expected_q)
        torch.testing.assert_close(attention.o_proj.weight, o_before * 1.25)
        self.assertIsNone(getattr(attention, "_openpi_ohb_output_hook", None))
        x = torch.randn(2, 3, 8)
        expected_output = torch.nn.functional.linear(x, o_before) * 1.25
        torch.testing.assert_close(attention.o_proj(x), expected_output)
        runtime = model._openpi_atm_runtime
        self.assertEqual(runtime["atm_application"], "fold_q_weight")
        self.assertEqual(runtime["ohb_mode"], "per_layer_post_projection")
        self.assertEqual(runtime["ohb_application"], "fold_o_weight")

    def test_paper_mode_rejects_per_head_artifact(self) -> None:
        model = ToyModel()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "custom_atm.json"
            artifact.write_text(
                json.dumps(
                    {
                        "meta": {"ohb_mode": "per_head_pre_projection"},
                        "layers": {LAYER: {"all": [1.0, 1.0], "beta_perhead": [1.0, 1.0]}},
                    }
                ),
                encoding="utf-8",
            )
            os.environ.update(
                {
                    "OPENPI_ATM_ENABLE": "1",
                    "OPENPI_OHB_ENABLE": "1",
                    "OPENPI_ATM_ALPHA_PATH": str(artifact),
                    "OPENPI_OHB_EXPECT_MODE": "per_layer_post_projection",
                }
            )
            with self.assertRaisesRegex(ValueError, "ohb_mode"):
                enable_pi05_atm_if_configured(model)

    def test_v3_errorfold_uses_static_perhead_fold_without_legacy_ohb_field(self) -> None:
        model = ToyModel()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "errorfold_v3.json"
            artifact.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "kind": "errorfold_grid_candidate",
                        "meta": {
                            "atm_application": "fold_q_weight",
                            "errorfold_application": "fold_affine_into_weight_dequant_scale_and_bias",
                            "selector_free": True,
                            "runtime_branch": False,
                        },
                        "layers": {
                            f"{LAYER}::attention_logits": {
                                "kind": "attention_logits",
                                "effective_gain": [1.0, 1.0],
                                "effective_bias": [0.0, 0.0],
                            },
                            f"{LAYER}::attention_head_output": {
                                "kind": "attention_head_output",
                                "effective_gain": [1.0] * 8,
                                "effective_bias": [0.0] * 8,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            os.environ.update(
                {
                    "OPENPI_ATM_ENABLE": "1",
                    "OPENPI_OHB_ENABLE": "1",
                    "OPENPI_ATM_ALPHA_PATH": str(artifact),
                    "OPENPI_ATM_SCOPE": "expert",
                    "OPENPI_OHB_SCOPE": "expert",
                    "OPENPI_ATM_STRICT": "1",
                    "OPENPI_ATM_EXPECT_LAYERS": "1",
                    "OPENPI_ATM_APPLICATION": "fold_q_weight",
                    "OPENPI_OHB_APPLICATION": "fold_o_weight_perhead",
                }
            )
            enable_pi05_atm_if_configured(model)

        runtime = model._openpi_atm_runtime
        self.assertEqual(runtime["matched_layers"], 1)
        self.assertEqual(runtime["ohb_layers"], 1)
        self.assertEqual(runtime["ohb_application"], "fold_o_weight_perhead")
        attention = model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn
        self.assertTrue(attention._openpi_errorfold_head_affine_folded)

    def test_legacy_per_head_runtime_metadata_is_backward_compatible(self) -> None:
        model = ToyModel()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "legacy_atm.json"
            artifact.write_text(
                json.dumps(
                    {
                        "meta": {},
                        "layers": {LAYER: {"all": [1.0, 1.0], "beta_perhead": [1.0, 1.0]}},
                    }
                ),
                encoding="utf-8",
            )
            os.environ.update(
                {
                    "OPENPI_ATM_ENABLE": "1",
                    "OPENPI_OHB_ENABLE": "1",
                    "OPENPI_ATM_ALPHA_PATH": str(artifact),
                    "OPENPI_ATM_SCOPE": "expert",
                    "OPENPI_OHB_SCOPE": "expert",
                }
            )
            enable_pi05_atm_if_configured(model)
        runtime = model._openpi_atm_runtime
        self.assertNotIn("atm_application", runtime)
        self.assertNotIn("ohb_mode", runtime)


if __name__ == "__main__":
    unittest.main()
