#!/usr/bin/env python3
"""Build a resumable GPU shard of LIBERO DyPAC Hessian-aware group-64 W4."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tools"))
from quantvla_hessian_w4 import artifact_arrays, hessian_aware_w4  # noqa: E402
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file  # noqa: E402


def valid(path: Path, name: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return str(archive["layer_name"].item()) == name and archive["packed_u4"].dtype == np.uint8
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite", help="Required when --checkpoint is a sharded directory")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid Hessian shard")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    rows = inventory["layers"]
    selected = [(index, row) for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    capture = np.load(Path(args.capture).expanduser().resolve(), allow_pickle=False)
    capture_names = [str(value) for value in capture["layer_names"].tolist()]
    if capture_names != [row["name"] for row in rows]:
        raise ValueError("Hessian capture/inventory mismatch")
    journal = []
    started = time.monotonic()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    with ExitStack() as stack:
        if checkpoint.is_dir():
            if not args.suite:
                raise ValueError("--suite is required for a sharded checkpoint directory")
            filenames = sorted(
                {row["checkpoint_files"][args.suite] for _, row in selected}
            )
            archives = {
                filename: stack.enter_context(
                    safe_open(checkpoint / filename, framework="pt", device="cpu")
                )
                for filename in filenames
            }
        else:
            archives = {"__single__": stack.enter_context(safe_open(checkpoint, framework="pt", device="cpu"))}
        for ordinal, (index, row) in enumerate(selected, start=1):
            path = output / f"layer_{index:04d}.npz"
            if valid(path, row["name"]):
                journal.append({"index": index, "name": row["name"], "status": "reused"})
                print(f"[Hessian {args.shard_index}] reuse {ordinal}/{len(selected)} {row['name']}", flush=True)
                continue
            layer_started = time.monotonic()
            archive = (
                archives[row["checkpoint_files"][args.suite]]
                if checkpoint.is_dir()
                else archives["__single__"]
            )
            weight = archive.get_tensor(row["checkpoint_key"])
            inputs = torch.from_numpy(np.asarray(capture[f"inputs_{index:04d}"]))
            result = hessian_aware_w4(weight, inputs, device=args.device)
            values = artifact_arrays(result)
            temporary = Path(str(path) + f".tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                np.savez(handle, layer_name=np.asarray(row["name"]), **values)
            temporary.replace(path)
            elapsed = time.monotonic() - layer_started
            journal.append({"index": index, "name": row["name"], "status": "built", "elapsed_s": elapsed, "sha256": sha256_file(path)})
            print(f"[Hessian {args.shard_index}] built {ordinal}/{len(selected)} {row['name']} {elapsed:.1f}s", flush=True)
            del result, weight, inputs
            torch.cuda.empty_cache()
    capture.close()
    atomic_json(
        output / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json",
        {
            "schema_version": 1,
            "kind": "dypac_libero_hessian_shard",
            "model": inventory.get("model", "pi05"),
            "suite": args.suite,
            "protocol_id": PROTOCOL["protocol_id"],
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "layers": journal,
            "elapsed_s": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
