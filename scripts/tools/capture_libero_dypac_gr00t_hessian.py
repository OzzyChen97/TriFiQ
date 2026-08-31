#!/usr/bin/env python3
"""Capture bounded FP16 GR00T layer inputs on one LIBERO suite shard."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from gr00t_v2_common import ensure_flash_attn_rpath, load_policy, strip_quant_env
from quantvla_libero_dypac import (
    PROTOCOL,
    PROTOCOL_PATH,
    load_records,
    run_gr00t_policy,
    sha256_file,
)


DATA_CONFIGS = {
    "goal": "examples.Libero.custom_data_config:LiberoDataConfigMeanStd",
    "spatial": "examples.Libero.custom_data_config:LiberoDataConfig",
    "object": "examples.Libero.custom_data_config:LiberoDataConfig",
    "long": "examples.Libero.custom_data_config:LiberoDataConfig",
}


class InputCapture:
    def __init__(self, names: list[str], max_rows: int) -> None:
        self.names = names
        self.max_rows = int(max_rows)
        self.rows: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)
        self.calls: dict[str, int] = defaultdict(int)
        self.handles = []

    def _hook(self, name: str):
        def capture(_module, inputs):
            self.calls[name] += 1
            if self.counts[name] >= self.max_rows:
                return
            value = inputs[0].detach().to(torch.float32)
            channels = value.shape[-1]
            samples = value.reshape(value.shape[0], -1, channels)
            positions = samples.shape[1]
            per_observation = min(2, positions)
            indices = torch.linspace(
                0, positions - 1, per_observation, device=value.device
            ).round().long()
            selected = samples.index_select(1, indices).reshape(-1, channels)
            remaining = self.max_rows - self.counts[name]
            selected = selected[:remaining].to(device="cpu", dtype=torch.float16)
            self.rows[name].append(selected)
            self.counts[name] += len(selected)

        return capture

    def install(self, model: torch.nn.Module) -> None:
        modules = dict(model.named_modules())
        missing = [name for name in self.names if name not in modules]
        if missing:
            raise RuntimeError(f"model lacks {len(missing)} GR00T candidates: {missing[:5]}")
        for name in self.names:
            self.handles.append(modules[name].register_forward_pre_hook(self._hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(DATA_CONFIGS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--buffer", required=True, help="The suite's 64-row FP16 shard")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-rows-per-layer", type=int, default=512)
    args = parser.parse_args()
    output = Path(args.out).expanduser().resolve()
    sidecar = Path(str(output) + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite GR00T Hessian capture: {output}")
    inventory_path = Path(args.inventory).expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    if len(names) != 116:
        raise ValueError("GR00T candidate inventory drift")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    expected_checkpoint = Path(inventory["checkpoints"][args.suite]["path"]).resolve()
    if checkpoint != expected_checkpoint:
        raise ValueError(f"checkpoint/inventory mismatch: {checkpoint} != {expected_checkpoint}")
    buffer = Path(args.buffer).expanduser().resolve()
    records = load_records(buffer, model="gr00t")
    if len(records) != 64 or {row["suite"] for row in records} != {args.suite}:
        raise ValueError("GR00T Hessian capture requires one complete 64-row suite shard")

    os.environ.update({"TORCHDYNAMO_DISABLE": "1", "TORCH_COMPILE_DISABLE": "1"})
    strip_quant_env()
    ensure_flash_attn_rpath()
    policy = load_policy(
        str(checkpoint),
        data_config=DATA_CONFIGS[args.suite],
        denoising_steps=10,
        device=args.device,
    )
    capture = InputCapture(names, args.max_rows_per_layer)
    capture.install(policy.model)
    started = time.monotonic()
    try:
        completed = 0
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            run_gr00t_policy(policy, batch, batch_size=args.batch_size)
            completed += len(batch)
            print(f"[GR00T Hessian capture/{args.suite}] {completed}/64", flush=True)
    finally:
        capture.close()
    empty = [name for name in names if not capture.rows[name]]
    if empty:
        raise RuntimeError(f"empty GR00T candidate captures: {empty[:5]}")
    arrays: dict[str, np.ndarray] = {
        "layer_names": np.asarray(names),
        "capture_rows": np.asarray([capture.counts[name] for name in names], dtype=np.int64),
        "capture_calls": np.asarray([capture.calls[name] for name in names], dtype=np.int64),
        "calibration_buffer_sha256": np.asarray(sha256_file(buffer)),
        "checkpoint_sha256": np.asarray(inventory["checkpoints"][args.suite]["sha256"]),
        "protocol_sha256": np.asarray(sha256_file(PROTOCOL_PATH)),
        "flow_steps": np.asarray(10),
    }
    for index, name in enumerate(names):
        arrays[f"inputs_{index:04d}"] = torch.cat(capture.rows[name], dim=0).numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(output)
    metadata = {
        "schema_version": 1,
        "kind": "dypac_libero_gr00t_fp16_hessian_capture",
        "model": "gr00t",
        "suite": args.suite,
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": inventory["checkpoints"][args.suite]["sha256"],
        "calibration_buffer": str(buffer),
        "calibration_buffer_sha256": sha256_file(buffer),
        "inventory": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "layers": 116,
        "rows_per_layer_min": min(capture.counts.values()),
        "rows_per_layer_max": max(capture.counts.values()),
        "flow_steps": 10,
        "uses_success_labels": False,
        "npz": str(output),
        "sha256": sha256_file(output),
        "elapsed_s": time.monotonic() - started,
    }
    temporary_sidecar = Path(str(sidecar) + f".tmp.{os.getpid()}")
    temporary_sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_sidecar.replace(sidecar)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
