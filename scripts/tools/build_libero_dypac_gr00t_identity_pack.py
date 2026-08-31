#!/usr/bin/env python3
"""Build the identity transform pack required by GR00T Hessian-W4 DyPAC."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open

from gr00t.quantization.duquant_preprocess import pack_weight, sanitize_name, save_pack
from quantvla_libero_dypac import PROTOCOL, PROTOCOL_PATH, atomic_json, sha256_file


def valid(path: Path, name: str) -> bool:
    if not path.is_file():
        return False
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["meta"].tolist())
        return (
            meta.get("layer_name") == name
            and meta.get("identity_for_hessian_w4") is True
            and meta.get("block_size") == 64
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid identity-pack shard")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    rows = inventory["layers"]
    selected = [
        row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index
    ]
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if checkpoint != Path(inventory["checkpoints"][args.suite]["path"]).resolve():
        raise ValueError("GR00T identity-pack checkpoint/inventory mismatch")
    output = Path(args.out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["GR00T_DUQUANT_HESSIAN_W4_PATH"] = "identity-pack-build"
    started = time.monotonic()
    journal = []
    filenames = sorted({row["checkpoint_files"][args.suite] for row in selected})
    with ExitStack() as stack:
        archives = {
            filename: stack.enter_context(
                safe_open(checkpoint / filename, framework="pt", device="cpu")
            )
            for filename in filenames
        }
        for ordinal, row in enumerate(selected, start=1):
            path = output / f"{sanitize_name(row['name'])}.npz"
            if valid(path, row["name"]):
                journal.append({"name": row["name"], "status": "reused"})
                print(
                    f"[GR00T identity pack {args.shard_index}] reuse "
                    f"{ordinal}/{len(selected)} {row['name']}",
                    flush=True,
                )
                continue
            weight = archives[row["checkpoint_files"][args.suite]].get_tensor(
                row["checkpoint_key"]
            ).to(torch.float32)
            pack = pack_weight(
                weight,
                block_size=64,
                block_out_size=64,
                enable_permute=False,
                lambda_smooth=0.0,
            )
            pack.meta.update(
                {
                    "schema_version": 1,
                    "layer_name": row["name"],
                    "suite": args.suite,
                    "checkpoint_sha256": inventory["checkpoints"][args.suite]["sha256"],
                    "protocol_id": PROTOCOL["protocol_id"],
                    "protocol_sha256": sha256_file(PROTOCOL_PATH),
                }
            )
            save_pack(row["name"], pack, str(output))
            journal.append({"name": row["name"], "status": "built"})
            print(
                f"[GR00T identity pack {args.shard_index}] built "
                f"{ordinal}/{len(selected)} {row['name']}",
                flush=True,
            )
            del weight, pack
    atomic_json(
        output / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json",
        {
            "schema_version": 1,
            "kind": "dypac_libero_gr00t_identity_pack_shard",
            "model": "gr00t",
            "suite": args.suite,
            "protocol_id": PROTOCOL["protocol_id"],
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "checkpoint_sha256": inventory["checkpoints"][args.suite]["sha256"],
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "layers": journal,
            "elapsed_s": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
