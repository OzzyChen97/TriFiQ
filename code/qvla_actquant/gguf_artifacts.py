"""Graph-free component GGUF export and ActQuant AMF packing utilities."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .core import canonical_hash, classify_target, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTQUANT_ROOT = REPO_ROOT / "external" / "ActQuant"
GGUF_PY = ACTQUANT_ROOT / "gguf-py"
COMPONENT_QUANTIZER = ACTQUANT_ROOT / "build-qvla-actquant" / "bin" / "quantize-vision"
LLAMA_QUANTIZER = ACTQUANT_ROOT / "build-qvla-actquant" / "bin" / "llama-quantize"


def _import_gguf():
    if str(GGUF_PY) not in sys.path:
        sys.path.insert(0, str(GGUF_PY))
    import gguf
    return gguf


def tensor_name_map(module_names: Sequence[str]) -> dict[str, str]:
    return {name: f"aq.{index:05d}.weight" for index, name in enumerate(sorted(module_names))}


def pi05_official_tensor_maps(
    module_names: Sequence[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Map live openpi targets to standalone-LLM and final unified GGUF names.

    The final names are exactly those emitted by the released
    ``export_pi05.py`` / ``merge_pi05_llm.py`` path.  Vision tensors have no
    standalone-LLaMA counterpart and are returned only in the final map.
    """
    standalone: dict[str, str] = {}
    final: dict[str, str] = {}
    attention = {
        "q_proj": ("attn_q", "attn_q"),
        "k_proj": ("attn_k", "attn_k"),
        "v_proj": ("attn_v", "attn_v"),
        "o_proj": ("attn_output", "attn_o"),
    }
    vision_attention = {
        "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "out_proj": "attn_out",
    }
    for name in sorted(module_names):
        language_match = re.search(
            r"language_model\.layers\.(\d+)\.(?:self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$",
            name,
        )
        if language_match:
            layer = int(language_match.group(1))
            if language_match.group(2):
                standalone_suffix, final_suffix = attention[language_match.group(2)]
            else:
                suffix = {"gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}[
                    language_match.group(3)
                ]
                standalone_suffix = final_suffix = suffix
            standalone[name] = f"blk.{layer}.{standalone_suffix}.weight"
            final[name] = f"pali.blk.{layer}.{final_suffix}.weight"
            continue
        vision_match = re.search(
            r"vision_tower\.vision_model\.encoder\.layers\.(\d+)\."
            r"(?:self_attn\.(q_proj|k_proj|v_proj|out_proj)|mlp\.(fc1|fc2))$",
            name,
        )
        if vision_match:
            layer = int(vision_match.group(1))
            suffix = (
                vision_attention[vision_match.group(2)]
                if vision_match.group(2)
                else {"fc1": "ffn_up", "fc2": "ffn_down"}[vision_match.group(3)]
            )
            final[name] = f"v.blk.{layer}.{suffix}.weight"
            continue
        if re.search(r"vision_tower\.vision_model\.embeddings\.patch_embedding$", name):
            final[name] = "v.patch_embd.weight"
            continue
        raise ValueError(f"pi0.5 target has no released GGUF tensor mapping: {name}")
    if len(final) != len(module_names) or len(final.values()) != len(set(final.values())):
        raise ValueError("pi0.5 official tensor mapping is incomplete or non-injective")
    return standalone, final


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_component_gguf(
    model: nn.Module,
    module_names: Sequence[str],
    mapping: Mapping[str, str],
    output: str | Path,
    *,
    model_family: str,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    gguf = _import_gguf()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    writer = gguf.GGUFWriter(str(temporary), arch="")
    writer.add_name("actquant_component")
    writer.add_string("actquant.component_format", "graph_free_weight_manifest_v1")
    writer.add_string("actquant.model_family", model_family)
    for key, value in sorted(metadata.items()):
        writer.add_string(f"actquant.{key}", str(value))
    modules = dict(model.named_modules())
    tensors = []
    try:
        for name in sorted(module_names):
            module = modules.get(name)
            if module is None or classify_target(name, module, model_family) is None:
                raise ValueError(f"component export target mismatch: {name}")
            weight = module.weight.detach().float().cpu().flatten(1).numpy().astype(np.float16)
            writer.add_tensor(mapping[name], weight, raw_dtype=gguf.GGMLQuantizationType.F16)
            tensors.append({
                "module_name": name,
                "gguf_name": mapping[name],
                "shape": list(module.weight.shape),
                "matrix_shape": list(weight.shape),
            })
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "tensors": tensors,
    }


def write_fixed_component_gguf(
    model: nn.Module,
    output: str | Path,
    *,
    model_family: str,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    """Write every excluded/non-target parameter as an actual F16 GGUF payload."""
    gguf = _import_gguf()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    writer = gguf.GGUFWriter(str(temporary), arch="")
    writer.add_name("actquant_fixed_fp16_component")
    writer.add_string("actquant.component_format", "graph_free_fixed_fp16_manifest_v1")
    writer.add_string("actquant.model_family", model_family)
    for key, value in sorted(metadata.items()):
        writer.add_string(f"actquant.{key}", str(value))
    target_weight_ids = {
        id(module.weight)
        for name, module in model.named_modules()
        if classify_target(name, module, model_family) is not None
    }
    parameters = [
        (name.removeprefix("module."), parameter)
        for name, parameter in model.named_parameters(remove_duplicate=True)
        if id(parameter) not in target_weight_ids
    ]
    parameters.sort(key=lambda item: item[0])
    tensors = []
    try:
        for index, (name, parameter) in enumerate(parameters):
            gguf_name = f"aq.fixed.{index:05d}.param"
            value = parameter.detach().float().cpu().numpy().astype(np.float16)
            writer.add_tensor(gguf_name, value, raw_dtype=gguf.GGMLQuantizationType.F16)
            tensors.append(
                {
                    "parameter_name": name,
                    "gguf_name": gguf_name,
                    "file_id": "fixed_fp16",
                    "shape": list(parameter.shape),
                    "quant_type": "F16",
                    "payload_bytes": int(value.nbytes),
                }
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "id": "fixed_fp16",
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "tensors": tensors,
        "tensor_names_sha256": canonical_hash([record["parameter_name"] for record in tensors]),
    }


def write_fisher_gguf(
    fisher: Mapping[str, torch.Tensor | np.ndarray],
    module_names: Sequence[str],
    mapping: Mapping[str, str],
    output: str | Path,
    *,
    num_samples: int,
    calibration_sha256: str,
) -> dict[str, Any]:
    gguf = _import_gguf()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    writer = gguf.GGUFWriter(str(temporary), arch="")
    writer.add_string("general.type", "imatrix")
    writer.add_array("imatrix.datasets", [f"sha256:{calibration_sha256}"])
    writer.add_uint32("imatrix.chunk_count", int(num_samples))
    writer.add_uint32("imatrix.chunk_size", 1)
    writer.add_string("fisher.method", "action_mixed_fisher_alpha_1_flow_matching")
    writer.add_float32("fisher.alpha", 1.0)
    writer.add_uint32("fisher.num_samples", int(num_samples))
    floors = {}
    try:
        for name in sorted(module_names):
            value = fisher[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float32).reshape(value.shape[0], -1).copy()
            np.maximum(value, 0.0, out=value)
            positive = value[value > 0]
            floor = float(positive.mean() * 1e-6) if positive.size else 1e-20
            floors[name] = floor
            np.maximum(value, floor, out=value)
            writer.add_tensor(f"{mapping[name]}.in_sum2", value)
            writer.add_tensor(f"{mapping[name]}.counts", np.array([[1.0]], dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "floor_sha256": canonical_hash(floors),
        "tensors": len(module_names),
    }


def quantize_component(
    input_gguf: str | Path,
    fisher_gguf: str | Path,
    assignments: Mapping[str, str],
    module_names: Sequence[str],
    mapping: Mapping[str, str],
    output_gguf: str | Path,
    type_manifest: str | Path,
    *,
    log_path: str | Path,
) -> dict[str, Any]:
    if not COMPONENT_QUANTIZER.is_file():
        raise FileNotFoundError(f"ActQuant component quantizer is not built: {COMPONENT_QUANTIZER}")
    type_manifest = Path(type_manifest).resolve()
    lines = [f"{mapping[name]}\t{assignments[name]}\n" for name in sorted(module_names)]
    atomic_text(type_manifest, "".join(lines))
    output_gguf = Path(output_gguf).resolve()
    temporary = output_gguf.with_name(f".{output_gguf.name}.tmp.{os.getpid()}")
    command = [
        str(COMPONENT_QUANTIZER), str(Path(input_gguf).resolve()), str(temporary), "F16",
        "--pad", "--imatrix", str(Path(fisher_gguf).resolve()),
        "--tensor-types", str(type_manifest),
    ]
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    atomic_text(Path(log_path).resolve(), completed.stdout)
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"component quantizer failed ({completed.returncode}); see {log_path}")
    os.replace(temporary, output_gguf)
    return {
        "path": str(output_gguf),
        "sha256": sha256_file(output_gguf),
        "bytes": output_gguf.stat().st_size,
        "type_manifest": str(type_manifest),
        "type_manifest_sha256": sha256_file(type_manifest),
        "log": str(Path(log_path).resolve()),
        "log_sha256": sha256_file(Path(log_path).resolve()),
    }


def inspect_quantized_gguf(path: str | Path) -> dict[str, dict[str, Any]]:
    gguf = _import_gguf()
    reader = gguf.GGUFReader(Path(path).resolve())
    return {
        tensor.name: {
            "quant_type": tensor.tensor_type.name,
            "shape": [int(value) for value in tensor.shape],
            "payload_bytes": int(tensor.data.nbytes),
        }
        for tensor in reader.tensors
    }


def block_padded_parameters(shape: Sequence[int], block: int = 256) -> int:
    rows = int(shape[0])
    columns = math.prod(int(value) for value in shape[1:])
    return rows * int(math.ceil(columns / block) * block)


def _logged_command(command: Sequence[str], log_path: str | Path, *, env: Mapping[str, str] | None = None) -> None:
    completed = subprocess.run(
        list(command), check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    atomic_text(Path(log_path), completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log_path}")


def export_pi05_official_ggufs(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Run the released unified and standalone-PaliGemma exporters unchanged."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(tokenizer_path).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    staged_tokenizer = output_dir / "tokenizer.model"
    if not staged_tokenizer.exists():
        shutil.copy2(tokenizer_path, staged_tokenizer)
    python_path = os.pathsep.join(
        [str(GGUF_PY), str(ACTQUANT_ROOT / "tools/pi0.5"), os.environ.get("PYTHONPATH", "")]
    )
    env = {**os.environ, "PYTHONPATH": python_path}
    unified = output_dir / "pi05.gguf"
    standalone = output_dir / "pali_llm_bf16.gguf"
    if unified.exists() and standalone.exists():
        return {
            "unified": {"path": str(unified), "sha256": sha256_file(unified), "bytes": unified.stat().st_size},
            "standalone_llm": {
                "path": str(standalone), "sha256": sha256_file(standalone), "bytes": standalone.stat().st_size,
            },
            "tokenizer": {
                "path": str(staged_tokenizer), "sha256": sha256_file(staged_tokenizer),
                "source": str(tokenizer_path), "source_sha256": sha256_file(tokenizer_path),
            },
            "scripts": {
                "unified": str(ACTQUANT_ROOT / "tools/pi0.5/export_pi05.py"),
                "standalone_llm": str(ACTQUANT_ROOT / "tools/pi0.5/export_pi05_llm.py"),
            },
            "reused_complete_stage": True,
        }
    if unified.exists() or standalone.exists():
        raise FileExistsError("partial official pi0.5 export exists; refusing an ambiguous overwrite")
    _logged_command(
        [python_executable, str(ACTQUANT_ROOT / "tools/pi0.5/export_pi05.py"),
         "-d", str(checkpoint_dir), "-o", str(output_dir)],
        output_dir / "export_pi05.log", env=env,
    )
    _logged_command(
        [python_executable, str(ACTQUANT_ROOT / "tools/pi0.5/export_pi05_llm.py"),
         "-d", str(checkpoint_dir), "-o", str(output_dir)],
        output_dir / "export_pi05_llm.log", env=env,
    )
    for path in (unified, standalone):
        if not path.is_file():
            raise RuntimeError(f"released pi0.5 exporter did not create {path}")
    return {
        "unified": {"path": str(unified), "sha256": sha256_file(unified), "bytes": unified.stat().st_size},
        "standalone_llm": {
            "path": str(standalone), "sha256": sha256_file(standalone), "bytes": standalone.stat().st_size,
        },
        "tokenizer": {
            "path": str(staged_tokenizer), "sha256": sha256_file(staged_tokenizer),
            "source": str(tokenizer_path), "source_sha256": sha256_file(tokenizer_path),
        },
        "scripts": {
            "unified": str(ACTQUANT_ROOT / "tools/pi0.5/export_pi05.py"),
            "standalone_llm": str(ACTQUANT_ROOT / "tools/pi0.5/export_pi05_llm.py"),
        },
    }


def quantize_pi05_official_llm(
    input_gguf: str | Path,
    fisher_gguf: str | Path,
    assignments: Mapping[str, str],
    standalone_mapping: Mapping[str, str],
    output_gguf: str | Path,
    *,
    log_path: str | Path,
) -> dict[str, Any]:
    """Use released llama-quantize with exact per-tensor overrides."""
    if not LLAMA_QUANTIZER.is_file():
        raise FileNotFoundError(LLAMA_QUANTIZER)
    if set(assignments) != set(standalone_mapping):
        raise ValueError("standalone pi0.5 allocation/mapping coverage mismatch")
    output_gguf = Path(output_gguf).resolve()
    if output_gguf.exists():
        return {
            "path": str(output_gguf), "sha256": sha256_file(output_gguf), "bytes": output_gguf.stat().st_size,
            "log": str(Path(log_path).resolve()),
            "log_sha256": sha256_file(Path(log_path).resolve()) if Path(log_path).is_file() else None,
            "reused_complete_stage": True,
        }
    command = [
        str(LLAMA_QUANTIZER),
        "--imatrix", str(Path(fisher_gguf).resolve()),
        "--token-embedding-type", "F16",
        "--output-tensor-type", "F16",
    ]
    for name in sorted(assignments):
        # llama-quantize interprets tensor selectors as regular expressions.
        command.extend(["--tensor-type", f"^{re.escape(standalone_mapping[name])}$={assignments[name]}"])
    command.extend([str(Path(input_gguf).resolve()), str(output_gguf), "IQ2_XS"])
    _logged_command(command, log_path)
    return {
        "path": str(output_gguf), "sha256": sha256_file(output_gguf), "bytes": output_gguf.stat().st_size,
        "log": str(Path(log_path).resolve()), "log_sha256": sha256_file(Path(log_path).resolve()),
        "command_sha256": canonical_hash(command),
    }


def merge_pi05_official_llm(
    base_gguf: str | Path,
    llm_gguf: str | Path,
    output_gguf: str | Path,
    *,
    log_path: str | Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Invoke the released merge_pi05_llm.py without source modification."""
    output_gguf = Path(output_gguf).resolve()
    if output_gguf.exists():
        return {
            "path": str(output_gguf), "sha256": sha256_file(output_gguf), "bytes": output_gguf.stat().st_size,
            "log": str(Path(log_path).resolve()),
            "log_sha256": sha256_file(Path(log_path).resolve()) if Path(log_path).is_file() else None,
            "script": str(ACTQUANT_ROOT / "tools/pi0.5/merge_pi05_llm.py"),
            "reused_complete_stage": True,
        }
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(GGUF_PY), os.environ.get("PYTHONPATH", "")])}
    _logged_command(
        [python_executable, str(ACTQUANT_ROOT / "tools/pi0.5/merge_pi05_llm.py"),
         "--base", str(Path(base_gguf).resolve()), "--llm", str(Path(llm_gguf).resolve()),
         "--output", str(output_gguf), "--quant-type", "ACTQUANT_MIXED_4BPW"],
        log_path, env=env,
    )
    return {
        "path": str(output_gguf), "sha256": sha256_file(output_gguf), "bytes": output_gguf.stat().st_size,
        "log": str(Path(log_path).resolve()), "log_sha256": sha256_file(Path(log_path).resolve()),
        "script": str(ACTQUANT_ROOT / "tools/pi0.5/merge_pi05_llm.py"),
    }


def merge_gguf_tensor_overrides(
    base_gguf: str | Path,
    replacement_gguf: str | Path,
    names: Sequence[str],
    output_gguf: str | Path,
) -> dict[str, Any]:
    """Copy a GGUF while replacing an exact frozen tensor inventory.

    This is the local vision-component merge following the released pi0.5 LLM
    merge.  Metadata copying is delegated to the released merge module.
    """
    gguf = _import_gguf()
    merge_script = ACTQUANT_ROOT / "tools/pi0.5/merge_pi05_llm.py"
    spec = importlib.util.spec_from_file_location("actquant_released_pi05_merge", merge_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {merge_script}")
    released = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(released)
    base = gguf.GGUFReader(Path(base_gguf).resolve(), "r")
    replacement = gguf.GGUFReader(Path(replacement_gguf).resolve(), "r")
    wanted = set(names)
    replacements = {tensor.name: tensor for tensor in replacement.tensors if tensor.name in wanted}
    if set(replacements) != wanted:
        raise ValueError(f"vision GGUF replacement coverage mismatch: {len(replacements)} != {len(wanted)}")
    base_names = {tensor.name for tensor in base.tensors}
    if not wanted <= base_names:
        raise ValueError(f"official base GGUF lacks replacements: {sorted(wanted - base_names)}")
    output_gguf = Path(output_gguf).resolve()
    if output_gguf.exists():
        return {
            "path": str(output_gguf), "sha256": sha256_file(output_gguf), "bytes": output_gguf.stat().st_size,
            "replacement_tensor_names_sha256": canonical_hash(sorted(wanted)),
            "reused_complete_stage": True,
        }
    temporary = output_gguf.with_name(f".{output_gguf.name}.tmp.{os.getpid()}")
    writer = gguf.GGUFWriter(str(temporary), "pi05")
    try:
        released.copy_kv_metadata(base, writer, "ACTQUANT_MIXED_4BPW")
        for tensor in base.tensors:
            source = replacements.get(tensor.name, tensor)
            if source.data.dtype == np.uint8:
                writer.add_tensor(tensor.name, source.data, raw_dtype=source.tensor_type)
            else:
                writer.add_tensor(tensor.name, source.data)
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
        os.replace(temporary, output_gguf)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output_gguf), "sha256": sha256_file(output_gguf), "bytes": output_gguf.stat().st_size,
        "replacement_tensor_names_sha256": canonical_hash(sorted(wanted)),
    }
