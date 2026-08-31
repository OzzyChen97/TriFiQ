#!/usr/bin/env python3
"""Capture bounded FP16 layer inputs on the 256-row LIBERO DyPAC buffer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OPENPI = ROOT / "code/pi05/openpi"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages/openpi-client/src"))
sys.path.insert(0, str(ROOT / "scripts/tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.training import config  # noqa: E402
from quantvla_libero_dypac import (  # noqa: E402
    FLOW_STEPS,
    PROTOCOL,
    PROTOCOL_PATH,
    iter_batches,
    load_records,
    prepare_batch,
    sha256_file,
)


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
            if self.counts[name] >= self.max_rows:
                self.calls[name] += 1
                return
            value = inputs[0].detach().to(torch.float32)
            channels = value.shape[-1]
            if value.ndim >= 3:
                samples = value.reshape(value.shape[0], -1, channels)
                positions = samples.shape[1]
                indices = torch.linspace(0, positions - 1, 1, device=value.device).round().long()
                selected = samples.index_select(1, indices).reshape(-1, channels)
            else:
                flat = value.reshape(-1, channels)
                count = min(8, len(flat))
                indices = torch.linspace(0, len(flat) - 1, count, device=value.device).round().long()
                selected = flat.index_select(0, indices)
            remaining = self.max_rows - self.counts[name]
            selected = selected[:remaining].to(device="cpu", dtype=torch.float16)
            self.rows[name].append(selected)
            self.counts[name] += len(selected)
            self.calls[name] += 1
        return capture

    def install(self, model: torch.nn.Module) -> None:
        modules = dict(model.named_modules())
        missing = [name for name in self.names if name not in modules]
        if missing:
            raise RuntimeError(f"model lacks {len(missing)} candidate layers: {missing[:5]}")
        for name in self.names:
            self.handles.append(modules[name].register_forward_pre_hook(self._hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-rows-per-layer", type=int, default=512)
    args = parser.parse_args()
    output = Path(args.out).expanduser().resolve()
    sidecar = Path(str(output) + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite Hessian capture: {output}")
    inventory_path = Path(args.inventory).expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    names = [row["name"] for row in inventory["layers"]]
    if len(names) != 180 or inventory["checkpoint_sha256"] != args.checkpoint_sha256:
        raise ValueError("π0.5 candidate inventory drift")
    buffer = Path(args.buffer).expanduser().resolve()
    records = load_records(buffer)
    if len(records) != 256:
        raise ValueError(f"Hessian capture requires 256 observations, got {len(records)}")

    os.environ.update({"TORCHDYNAMO_DISABLE": "1", "OPENPI_MODEL_DTYPE": "float16"})
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"),
        Path(args.checkpoint_dir).expanduser().resolve(),
        pytorch_device=args.device,
    )
    model = policy._model
    model.to(args.device).eval()
    capture = InputCapture(names, args.max_rows_per_layer)
    capture.install(model)
    started = time.monotonic()
    try:
        completed = 0
        for batch in iter_batches(records, args.batch_size):
            _, observation = prepare_batch(policy, batch, args.device)
            noise = torch.from_numpy(np.stack([row["noises"][0] for row in batch])).to(args.device)
            with torch.inference_mode():
                model.sample_actions(
                    args.device, observation, noise=noise, num_steps=FLOW_STEPS
                )
            completed += len(batch)
            print(f"[pi05 Hessian capture] {completed}/256", flush=True)
    finally:
        capture.close()
    empty = [name for name in names if not capture.rows[name]]
    if empty:
        raise RuntimeError(f"empty candidate captures: {empty[:5]}")
    arrays: dict[str, np.ndarray] = {
        "layer_names": np.asarray(names),
        "capture_rows": np.asarray([capture.counts[name] for name in names], dtype=np.int64),
        "capture_calls": np.asarray([capture.calls[name] for name in names], dtype=np.int64),
        "calibration_buffer_sha256": np.asarray(sha256_file(buffer)),
        "checkpoint_sha256": np.asarray(args.checkpoint_sha256),
        "protocol_sha256": np.asarray(sha256_file(PROTOCOL_PATH)),
        "flow_steps": np.asarray(FLOW_STEPS),
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
        "kind": "dypac_libero_pi05_fp16_hessian_capture",
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "checkpoint_sha256": args.checkpoint_sha256,
        "calibration_buffer": str(buffer),
        "calibration_buffer_sha256": sha256_file(buffer),
        "inventory": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "layers": 180,
        "rows_per_layer_min": min(capture.counts.values()),
        "rows_per_layer_max": max(capture.counts.values()),
        "flow_steps": FLOW_STEPS,
        "uses_success_labels": False,
        "npz": str(output),
        "sha256": sha256_file(output),
        "elapsed_s": time.monotonic() - started,
    }
    temporary_sidecar = Path(str(sidecar) + f".tmp.{os.getpid()}")
    temporary_sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_sidecar.replace(sidecar)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
