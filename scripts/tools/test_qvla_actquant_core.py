#!/usr/bin/env python3
"""Parity and round-trip gates for the QVLA/ActQuant reproduction core."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from qvla_actquant.core import (  # noqa: E402
    HessianProxy,
    actquant_greedy_l2_allocate,
    apply_actquant_gguf_bundle,
    apply_qvla_pack,
    canonical_hash,
    compute_hsic_record,
    fisher_diagonal,
    hsic_reference_modules,
    qvla_compute_proxies,
    qvla_greedy_allocate,
    qvla_pack_model,
    sha256_file,
    target_inventory,
)
from qvla_actquant.gguf_artifacts import (  # noqa: E402
    inspect_quantized_gguf,
    pi05_official_tensor_maps,
    quantize_component,
    tensor_name_map,
    write_component_gguf,
    write_fisher_gguf,
    write_fixed_component_gguf,
)
from qvla_actquant.runtime import apply_reproduction_artifact_from_env  # noqa: E402


def load_upstream(relative: str, module_name: str):
    path = REPO_ROOT / "external" / "QVLA" / "openvla-oft" / "qvla" / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_local_script(relative: str, module_name: str):
    path = REPO_ROOT / "scripts" / "tools" / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ToyGR00T(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.eagle_model = nn.Module()
        self.backbone.eagle_model.vision_model = nn.Sequential(nn.Linear(6, 5, bias=False))
        self.backbone.eagle_model.language_model = nn.Sequential(nn.Linear(5, 4, bias=False))
        self.backbone.eagle_linear = nn.Linear(4, 4, bias=False)
        self.action_head = nn.Sequential(nn.Linear(4, 3, bias=False))


class ToyPi05(nn.Module):
    """Minimal module tree whose Linear is a frozen pi0.5 target."""

    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Module()
        language = nn.Module()
        self.paligemma_with_expert.paligemma.language_model = language
        language.layers = nn.ModuleList([nn.Module()])
        language.layers[0].input_layernorm = nn.LayerNorm(300, elementwise_affine=False)
        language.layers[0].self_attn = nn.Module()
        language.layers[0].self_attn.q_proj = nn.Linear(300, 32, bias=False)
        self.paligemma_with_expert.gemma_expert = nn.Linear(5, 3, bias=False)


class ReproductionCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up_proxy = load_upstream("sensitivity_hessian_proxy.py", "qvla_upstream_proxy")
        cls.up_allocate = load_upstream("assign_gates_from_sensitivity.py", "qvla_upstream_allocate")
        cls.up_inject = load_upstream("inject_fake_w.py", "qvla_upstream_inject")
        cls.qvla_driver = load_local_script("run_qvla_calibration.py", "qvla_local_driver")
        cls.actquant_driver = load_local_script("run_actquant_calibration.py", "actquant_local_driver")

    def test_method_precision_gates_are_fail_closed(self):
        previous_gr00t = os.environ.get("GR00T_MODEL_DTYPE")
        try:
            os.environ["GR00T_MODEL_DTYPE"] = "bfloat16"
            qvla = self.qvla_driver.require_bf16_model(
                "gr00t", ToyGR00T().to(torch.bfloat16)
            )
            self.assertTrue(qvla["strict_all_linear_conv_bf16"])
            with self.assertRaises(ValueError):
                self.qvla_driver.require_bf16_model(
                    "gr00t", ToyGR00T().to(torch.float16)
                )

            os.environ["GR00T_MODEL_DTYPE"] = "float16"
            actquant = self.actquant_driver.require_fp16_model(
                "gr00t", ToyGR00T().to(torch.float16)
            )
            self.assertTrue(actquant["strict_all_linear_conv_fp16"])
            with self.assertRaises(ValueError):
                self.actquant_driver.require_fp16_model(
                    "gr00t", ToyGR00T().to(torch.bfloat16)
                )
        finally:
            if previous_gr00t is None:
                os.environ.pop("GR00T_MODEL_DTYPE", None)
            else:
                os.environ["GR00T_MODEL_DTYPE"] = previous_gr00t

    def test_qvla_hessian_and_proxy_parity(self):
        torch.manual_seed(7)
        layer = nn.Linear(6, 5, bias=False)
        ours = HessianProxy(layer, "cpu")
        upstream = self.up_proxy._HessianProxy(layer, torch.device("cpu"))
        for value in (torch.randn(2, 3, 6), torch.randn(1, 4, 6)):
            ours.add_batch(value)
            upstream.add_batch(value)
        torch.testing.assert_close(ours.H, upstream.H, rtol=0, atol=0)
        ours_diag = ours.diag_hinv(0.01)
        upstream_diag = upstream.diag_hinv(0.01)
        torch.testing.assert_close(ours_diag, upstream_diag, rtol=1e-6, atol=1e-6)
        ours_proxy = qvla_compute_proxies(layer, ours_diag, (0, 2, 4, 8, 16))
        upstream_proxy = self.up_proxy._compute_proxy_for_bits(
            layer, upstream_diag, (0, 2, 4, 8, 16)
        )
        for bit in ours_proxy:
            torch.testing.assert_close(ours_proxy[bit], upstream_proxy[bit], rtol=1e-6, atol=1e-6)

    def test_disabled_runtime_adapter_is_an_exact_identity(self):
        """An ordinary FP16 load must not be changed when no method is selected."""
        torch.manual_seed(5)
        model = ToyGR00T().to(torch.float16)
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        previous = os.environ.pop("QVLA_ACTQUANT_METHOD", None)
        try:
            result = apply_reproduction_artifact_from_env(model, "gr00t")
        finally:
            if previous is not None:
                os.environ["QVLA_ACTQUANT_METHOD"] = previous
        self.assertEqual(result, {"enabled": False, "method": None})
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_qvla_allocator_parity(self):
        generator = torch.Generator().manual_seed(11)
        proxies = {
            "a": {bit: torch.rand(7, generator=generator) for bit in (0, 2, 4, 8, 16)},
            "b": {bit: torch.rand(5, generator=generator) for bit in (0, 2, 4, 8, 16)},
        }
        for value in proxies.values():
            value[16].zero_()
        ours, stats = qvla_greedy_allocate(proxies, target_average_bits=4.0)
        expected, expected_stats = self.up_allocate.greedy_allocate(
            proxies, [0, 2, 4, 8, 16], 4.0
        )
        self.assertEqual(ours, expected)
        self.assertEqual(stats["final_total_bits"], expected_stats["final_total_bits"])
        self.assertLessEqual(stats["final_average_bits"], 4.0)

    def test_qvla_real_pack_roundtrip_matches_upstream_fake_quant(self):
        torch.manual_seed(19)
        reference = ToyGR00T().to(torch.bfloat16)
        packed_model = ToyGR00T().to(torch.bfloat16)
        packed_model.load_state_dict(reference.state_dict())
        gates = {
            "backbone.eagle_model.vision_model.0": [0, 2, 4, 8, 16],
            "backbone.eagle_model.language_model.0": [2, 4, 8, 16],
        }
        expected = ToyGR00T().to(torch.bfloat16)
        expected.load_state_dict(reference.state_dict())
        modules = dict(expected.named_modules())
        for name, values in gates.items():
            self.up_inject._apply_weight_only_fake_quant(
                modules[name], torch.tensor(values, dtype=torch.int64)
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.qpack"
            metadata = qvla_pack_model(
                packed_model,
                gates,
                path,
                model_family="gr00t",
                artifact_metadata={"test": True},
            )
            self.assertGreater(metadata["payload_bytes"], 0)
            self.assertGreater(metadata["fixed_payload_bytes"], 0)
            with torch.no_grad():
                packed_model.action_head[0].weight.zero_()
            apply_qvla_pack(packed_model, path, model_family="gr00t", expected_sha256=metadata["sha256"])
        for name in gates:
            torch.testing.assert_close(
                dict(packed_model.named_modules())[name].weight,
                dict(expected.named_modules())[name].weight,
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(packed_model.action_head[0].weight, reference.action_head[0].weight)

    def test_actquant_hsic_is_finite(self):
        rng = np.random.default_rng(23)
        values = [rng.normal(size=(12, dim)).astype(np.float32) for dim in (4, 3, 5, 6)]
        record = compute_hsic_record(*values)
        self.assertTrue(all(np.isfinite(value) for value in record.values()))
        self.assertAlmostEqual(record["sens"], record["F_out"] - record["F_in"])

    def test_actquant_hsic_reference_is_first_transformer_input(self):
        model = ToyPi05()
        inventory = target_inventory(model, "pi05")
        references = hsic_reference_modules(model, inventory)
        self.assertEqual(
            references["paligemma_language"],
            "paligemma_with_expert.paligemma.language_model.layers.0.input_layernorm",
        )

    def test_actquant_fisher_is_exact_gradient_square(self):
        torch.manual_seed(24)
        layer = nn.Linear(4, 3, bias=False)
        loss = layer(torch.randn(5, 4)).square().mean()
        expected = torch.autograd.grad(loss, layer.weight, retain_graph=True)[0].detach().square()
        actual = fisher_diagonal(loss, [("weight", layer.weight)])["weight"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_actquant_allocator_budget_and_determinism(self):
        scores = {"blk.0.a.weight": 1.0, "blk.0.b.weight": 2.0, "blk.1.a.weight": 3.0}
        parameters = {name: count for name, count in zip(scores, (1024, 2048, 4096), strict=True)}
        layers = {name: int(name.split(".")[1]) for name in scores}
        first = actquant_greedy_l2_allocate(scores, parameters, layers)
        second = actquant_greedy_l2_allocate(scores, parameters, layers)
        self.assertEqual(first, second)
        self.assertLessEqual(first[1]["achieved_bpw"], 4.0)
        self.assertLessEqual(first[1]["objective"], first[1]["initial_objective"])
        self.assertTrue(set(first[0].values()) <= {"IQ2_XS", "IQ2_S", "Q2_K", "IQ3_XXS", "IQ3_S", "IQ4_XS", "Q4_K"})

    def test_actquant_component_gguf_decode_matches_official_path(self):
        """The runtime adapter must copy exactly the official GGUF dequant result."""
        sys.path.insert(0, str(REPO_ROOT / "external" / "ActQuant" / "gguf-py"))
        from gguf import GGUFReader
        from gguf.quants import dequantize

        torch.manual_seed(29)
        source = ToyPi05().to(torch.bfloat16)
        restored = ToyPi05().to(torch.bfloat16)
        restored.load_state_dict(source.state_dict())
        name = "paligemma_with_expert.paligemma.language_model.layers.0.self_attn.q_proj"
        mapping = tensor_name_map([name])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            component = root / "component-f16.gguf"
            fisher = root / "fisher.gguf"
            quantized = root / "component-q4k.gguf"
            unweighted_quantized = root / "component-q4k-unweighted.gguf"
            fixed_path = root / "fixed-f16.gguf"
            write_component_gguf(
                source,
                [name],
                mapping,
                component,
                model_family="pi05",
                metadata={"test": "true"},
            )
            importance_tensor = torch.rand_like(
                source.state_dict()[name + ".weight"], dtype=torch.float32
            )
            write_fisher_gguf(
                {name: importance_tensor},
                [name],
                mapping,
                fisher,
                num_samples=3,
                calibration_sha256="0" * 64,
            )
            quantize_component(
                component,
                fisher,
                {name: "Q4_K"},
                [name],
                mapping,
                quantized,
                root / "tensor-types.tsv",
                log_path=root / "quantizer.log",
            )
            subprocess.run(
                [
                    str(REPO_ROOT / "external/ActQuant/build-qvla-actquant/bin/quantize-vision"),
                    str(component),
                    str(unweighted_quantized),
                    "F16",
                    "--pad",
                    "--tensor-types",
                    str(root / "tensor-types.tsv"),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            inspected = inspect_quantized_gguf(quantized)
            self.assertEqual(inspected[mapping[name]]["quant_type"], "Q4_K")
            fixed = write_fixed_component_gguf(
                source,
                fixed_path,
                model_family="pi05",
                metadata={"test": "true"},
            )
            self.assertEqual(len(fixed["tensors"]), 1)
            reader = GGUFReader(quantized)
            tensor = next(value for value in reader.tensors if value.name == mapping[name])
            direct = np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32)
            direct = direct[:, :300]
            baseline_reader = GGUFReader(unweighted_quantized)
            baseline_tensor = next(
                value for value in baseline_reader.tensors if value.name == mapping[name]
            )
            baseline_direct = np.asarray(
                dequantize(baseline_tensor.data, baseline_tensor.tensor_type), dtype=np.float32
            )[:, :300]
            original = source.state_dict()[name + ".weight"].float().numpy()
            importance = importance_tensor.numpy()
            optimized_objective = float(np.sum(importance * (original - direct) ** 2))
            unweighted_objective = float(
                np.sum(importance * (original - baseline_direct) ** 2)
            )
            self.assertLessEqual(optimized_objective, unweighted_objective + 1e-7)
            manifest = {
                "format": "actquant_gguf_bundle_v1",
                "method": "actquant",
                "model_family": "pi05",
                "target_inventory_sha256": canonical_hash(target_inventory(restored, "pi05")),
                "files": [{
                    "id": "component-00000",
                    "path": str(quantized),
                    "sha256": sha256_file(quantized),
                    "bytes": quantized.stat().st_size,
                }, {
                    "id": "fixed_fp16",
                    "path": str(fixed_path),
                    "sha256": sha256_file(fixed_path),
                    "bytes": fixed_path.stat().st_size,
                }],
                "tensors": [{
                    "module_name": name,
                    "gguf_name": mapping[name],
                    "file_id": "component-00000",
                    "shape": [32, 300],
                    "quant_type": "Q4_K",
                }],
                "target_tensor_count": 1,
                "fixed_tensors": fixed["tensors"],
                "fixed_tensor_names_sha256": fixed["tensor_names_sha256"],
                "achieved_bpw": 4.5,
            }
            manifest_path = root / "bundle.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with torch.no_grad():
                restored.paligemma_with_expert.gemma_expert.weight.zero_()
            apply_result = apply_actquant_gguf_bundle(
                restored,
                manifest_path,
                model_family="pi05",
            )
            loaded = restored.state_dict()[name + ".weight"].float().numpy()
            direct_native = torch.from_numpy(direct.copy()).to(torch.bfloat16).float().numpy()
            np.testing.assert_array_equal(loaded, direct_native)
            cosine = torch.nn.functional.cosine_similarity(
                torch.from_numpy(loaded).flatten(), torch.from_numpy(direct.copy()).flatten(), dim=0
            )
            self.assertGreaterEqual(float(cosine), 0.999)
            self.assertEqual(apply_result["applied_modules"], 1)
            expected_fixed = source.paligemma_with_expert.gemma_expert.weight.float().to(
                torch.float16
            ).to(torch.bfloat16)
            torch.testing.assert_close(
                restored.paligemma_with_expert.gemma_expert.weight,
                expected_fixed,
                rtol=0,
                atol=0,
            )

    def test_pi05_official_tensor_mapping(self):
        names = [
            "paligemma_with_expert.paligemma.language_model.layers.2.self_attn.o_proj",
            "paligemma_with_expert.paligemma.language_model.layers.2.mlp.gate_proj",
            "paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers.4.self_attn.out_proj",
            "paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers.4.mlp.fc2",
            "paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.patch_embedding",
        ]
        standalone, final = pi05_official_tensor_maps(names)
        self.assertEqual(standalone[names[0]], "blk.2.attn_output.weight")
        self.assertEqual(final[names[0]], "pali.blk.2.attn_o.weight")
        self.assertEqual(final[names[1]], "pali.blk.2.ffn_gate.weight")
        self.assertEqual(final[names[2]], "v.blk.4.attn_out.weight")
        self.assertEqual(final[names[3]], "v.blk.4.ffn_down.weight")
        self.assertEqual(final[names[4]], "v.patch_embd.weight")


if __name__ == "__main__":
    unittest.main()
