#!/usr/bin/env python3
"""Model-agnostic bounded hooks for v3 FP16/Hessian/ErrorFold captures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


def _tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(value, Mapping):
        for item in value.values():
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"hook value has no tensor: {type(value)}")


def _stratified_rows(value: Any, rows: int) -> torch.Tensor:
    tensor = _tensor(value).detach().to(torch.float32)
    flat = tensor.reshape(-1, tensor.shape[-1])
    if flat.shape[0] <= rows:
        return flat.cpu()
    indices = torch.linspace(0, flat.shape[0] - 1, rows).round().to(torch.long)
    return flat.index_select(0, indices).cpu()


class LayerCapture:
    """Capture a bounded, deterministic row sample from every call/layer."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        rows_per_call: int = 2,
        max_rows_per_layer: int = 4096,
        step_getter: Callable[[], int | None] | None = None,
    ) -> None:
        self.names = tuple(names)
        self.rows_per_call = int(rows_per_call)
        self.max_rows_per_layer = int(max_rows_per_layer)
        self.step_getter = step_getter or (lambda: None)
        self.inputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.inputs_by_step: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.calls: dict[str, int] = defaultdict(int)
        self.handles: list[Any] = []

    def _append(self, bank: list[torch.Tensor], value: torch.Tensor) -> None:
        current = sum(chunk.shape[0] for chunk in bank)
        if current >= self.max_rows_per_layer:
            return
        bank.append(value[: self.max_rows_per_layer - current])

    def _hook(self, name: str):
        def capture(_module, inputs, output):
            input_rows = _stratified_rows(inputs, self.rows_per_call)
            output_rows = _stratified_rows(output, self.rows_per_call)
            self._append(self.inputs[name], input_rows)
            self._append(self.outputs[name], output_rows)
            step = self.step_getter()
            if step is not None:
                self._append(self.inputs_by_step[name][int(step)], input_rows)
            self.calls[name] += 1

        return capture

    def install(self, model: torch.nn.Module) -> None:
        self.remove()
        modules = dict(model.named_modules())
        missing = [name for name in self.names if name not in modules]
        if missing:
            raise ValueError(f"capture model lacks {len(missing)} layers: {missing[:5]}")
        for name in self.names:
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @staticmethod
    def _cat(bank: list[torch.Tensor], name: str) -> torch.Tensor:
        if not bank:
            raise ValueError(f"capture bank is empty for {name}")
        return torch.cat(bank, dim=0)

    def arrays(
        self,
        model: torch.nn.Module,
        *,
        calibration_buffer_sha256: str,
        checkpoint_sha256: str,
        plan_sha256: str,
        include_weights: bool,
    ) -> dict[str, np.ndarray]:
        modules = dict(model.named_modules())
        arrays: dict[str, np.ndarray] = {
            "layer_names": np.asarray(self.names),
            "calibration_buffer_sha256": np.asarray(calibration_buffer_sha256),
            "checkpoint_sha256": np.asarray(checkpoint_sha256),
            "plan_sha256": np.asarray(plan_sha256),
            "capture_calls": np.asarray([self.calls[name] for name in self.names]),
        }
        for index, name in enumerate(self.names):
            arrays[f"inputs_{index:04d}"] = self._cat(self.inputs[name], name).numpy()
            arrays[f"outputs_{index:04d}"] = self._cat(self.outputs[name], name).numpy()
            if include_weights:
                weight = getattr(modules[name], "weight", None)
                if weight is None:
                    raise ValueError(f"{name} has no capturable FP16 weight")
                arrays[f"weight_{index:04d}"] = weight.detach().to(torch.float32).cpu().numpy()
            step_banks = self.inputs_by_step.get(name) or {}
            if step_banks:
                if set(step_banks) != {0, 1, 2, 3}:
                    raise ValueError(f"{name}: incomplete flow-step calls {sorted(step_banks)}")
                arrays[f"step_inputs_{index:04d}"] = np.stack(
                    [self._cat(step_banks[step], f"{name}/step{step}").numpy() for step in range(4)]
                )
        return arrays


class AttentionCapture:
    """Paired-row capture for attention logits and direction-aware head output."""

    def __init__(self, *, rows_per_call: int = 2, max_rows_per_layer: int = 4096) -> None:
        self.rows_per_call = int(rows_per_call)
        self.max_rows_per_layer = int(max_rows_per_layer)
        self.logits: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.head_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)

    def _append(self, bank: list[torch.Tensor], value: torch.Tensor) -> None:
        current = sum(chunk.shape[0] for chunk in bank)
        if current < self.max_rows_per_layer:
            bank.append(value[: self.max_rows_per_layer - current].cpu())

    def record_logits(self, name: str, value: torch.Tensor) -> None:
        # B,H,Q,K -> observations/tokens as rows, heads as ridge channels.
        tensor = value.detach().to(torch.float32).permute(0, 2, 3, 1)
        self._append(
            self.logits[name], _stratified_rows(tensor, self.rows_per_call)
        )

    def record_head_output(self, name: str, value: torch.Tensor) -> None:
        # B,H,S,D -> B,S,H*D, preserving direction inside every head.
        tensor = value.detach().to(torch.float32).transpose(1, 2).flatten(-2)
        self._append(
            self.head_outputs[name], _stratified_rows(tensor, self.rows_per_call)
        )

    def entries(self) -> list[tuple[str, str, np.ndarray]]:
        result = []
        for name in sorted(self.logits):
            result.append(
                (
                    f"{name}::attention_logits",
                    "attention_logits",
                    LayerCapture._cat(self.logits[name], name).numpy(),
                )
            )
        for name in sorted(self.head_outputs):
            result.append(
                (
                    f"{name}::attention_head_output",
                    "attention_head_output",
                    LayerCapture._cat(self.head_outputs[name], name).numpy(),
                )
            )
        return result


def save_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(output)


def merge_paired_outputs(
    fp16_path: str | Path,
    quant_path: str | Path,
    output_path: str | Path,
) -> None:
    with np.load(fp16_path, allow_pickle=False) as fp16, np.load(
        quant_path, allow_pickle=False
    ) as quant:
        names = [str(value) for value in fp16["layer_names"].tolist()]
        if names != [str(value) for value in quant["layer_names"].tolist()]:
            raise ValueError("paired capture layer inventory mismatch")
        arrays: dict[str, np.ndarray] = {
            "layer_names": np.asarray(names),
            "layer_kinds": np.asarray(["linear"] * len(names)),
            "calibration_buffer_sha256": np.asarray(fp16["calibration_buffer_sha256"]),
            "checkpoint_sha256": np.asarray(fp16["checkpoint_sha256"]),
            "plan_sha256": np.asarray(fp16["plan_sha256"]),
        }
        for index, name in enumerate(names):
            teacher = np.asarray(fp16[f"outputs_{index:04d}"])
            candidate = np.asarray(quant[f"outputs_{index:04d}"])
            if teacher.shape != candidate.shape:
                raise ValueError(f"{name}: paired output shape mismatch {teacher.shape}/{candidate.shape}")
            arrays[f"fp16_{index:04d}"] = teacher
            arrays[f"quant_{index:04d}"] = candidate
    save_npz(output_path, arrays)


def merge_errorfold_with_attention(
    fp16_linear_path: str | Path,
    quant_linear_path: str | Path,
    fp16_attention: Sequence[tuple[str, str, np.ndarray]],
    quant_attention: Sequence[tuple[str, str, np.ndarray]],
    output_path: str | Path,
) -> None:
    """Merge Linear and adapter-bound attention captures into one fit schema."""
    temporary_linear = Path(str(Path(output_path).resolve()) + ".linear.tmp.npz")
    merge_paired_outputs(fp16_linear_path, quant_linear_path, temporary_linear)
    with np.load(temporary_linear, allow_pickle=False) as linear:
        arrays = {key: np.asarray(linear[key]) for key in linear.files}
    temporary_linear.unlink()
    teacher_attention = {name: (kind, value) for name, kind, value in fp16_attention}
    candidate_attention = {name: (kind, value) for name, kind, value in quant_attention}
    if set(teacher_attention) != set(candidate_attention):
        raise ValueError("paired attention capture inventory mismatch")
    names = [str(value) for value in arrays["layer_names"].tolist()]
    kinds = [str(value) for value in arrays["layer_kinds"].tolist()]
    start = len(names)
    for offset, name in enumerate(sorted(teacher_attention)):
        teacher_kind, teacher = teacher_attention[name]
        candidate_kind, candidate = candidate_attention[name]
        if teacher_kind != candidate_kind or teacher.shape != candidate.shape:
            raise ValueError(f"{name}: paired attention capture mismatch")
        index = start + offset
        names.append(name)
        kinds.append(teacher_kind)
        arrays[f"fp16_{index:04d}"] = teacher
        arrays[f"quant_{index:04d}"] = candidate
    arrays["layer_names"] = np.asarray(names)
    arrays["layer_kinds"] = np.asarray(kinds)
    save_npz(output_path, arrays)
