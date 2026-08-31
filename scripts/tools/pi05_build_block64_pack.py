#!/usr/bin/env python3
"""Build and attest the formal π0.5 DuQuant block64 pack in GPU shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from safetensors import safe_open
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_SRC = REPO_ROOT / "code" / "pi05" / "openpi" / "src"
sys.path.insert(0, str(OPENPI_SRC))

from openpi.quant.duquant_preprocess import pack_weight, sanitize_name, save_pack  # noqa: E402
from openpi.quant.plan import sha256_file  # noqa: E402


CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
PATTERNS = (
    re.compile(
        r"^paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
    ),
    re.compile(
        r"^paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
        r"mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT
            / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--lambda-smooth", type=float, default=0.15)
    parser.add_argument("--permute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument(
        "--rehash-checkpoint",
        action="store_true",
        help="Re-read the 14 GB checkpoint in this shard (finalize always rehashes once).",
    )
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def candidate_keys(checkpoint: Path) -> list[str]:
    with safe_open(checkpoint, framework="pt", device="cpu") as archive:
        keys = sorted(key for key in archive.keys() if any(pattern.match(key) for pattern in PATTERNS))
    if len(keys) != 180:
        raise RuntimeError(f"expected 180 π0.5 candidate weights, found {len(keys)}")
    return keys


def inventory_sha256(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def expected_pack_path(out: Path, layer_name: str) -> Path:
    return out / f"{sanitize_name(layer_name)}.npz"


def read_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(archive["meta"].tolist())


def valid_existing(
    path: Path,
    *,
    layer_name: str,
    checkpoint_sha256: str,
    block: int,
    lambda_smooth: float,
    permute: bool,
) -> bool:
    if not path.is_file():
        return False
    try:
        meta = read_meta(path)
    except Exception:
        return False
    return (
        meta.get("layer_name") == layer_name
        and meta.get("checkpoint_sha256") == checkpoint_sha256
        and meta.get("block_size") == block
        and meta.get("block_out_size") == block
        and abs(float(meta.get("lambda_smooth")) - lambda_smooth) <= 1e-12
        and bool(meta.get("enable_permute")) == permute
        and bool(meta.get("weight_sha256"))
    )


def build(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    actual_checkpoint_sha256 = (
        sha256_file(checkpoint) if args.rehash_checkpoint else args.checkpoint_sha256
    )
    if actual_checkpoint_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint hash changed: {actual_checkpoint_sha256} != {args.checkpoint_sha256}"
        )
    keys = candidate_keys(checkpoint)
    names = [key.removesuffix(".weight") for key in keys]
    selected = [
        (key, name)
        for index, (key, name) in enumerate(zip(keys, names, strict=True))
        if index % args.num_shards == args.shard_index
    ]
    rows = []
    started = time.time()
    with safe_open(checkpoint, framework="pt", device="cpu") as archive:
        for ordinal, (key, name) in enumerate(selected, start=1):
            path = expected_pack_path(out, name)
            if not args.force and valid_existing(
                path,
                layer_name=name,
                checkpoint_sha256=actual_checkpoint_sha256,
                block=args.block,
                lambda_smooth=args.lambda_smooth,
                permute=args.permute,
            ):
                print(f"[pack shard {args.shard_index}] reuse {ordinal}/{len(selected)} {name}", flush=True)
                rows.append({"layer": name, "path": str(path), "status": "reused"})
                continue
            weight = archive.get_tensor(key).to(torch.float32)
            weight_hash = tensor_sha256(weight)
            layer_started = time.time()
            pack = pack_weight(
                weight,
                block_size=args.block,
                block_out_size=args.block,
                enable_permute=args.permute,
                lambda_smooth=args.lambda_smooth,
            )
            pack.meta.update(
                {
                    "schema_version": 2,
                    "layer_name": name,
                    "checkpoint_sha256": actual_checkpoint_sha256,
                    "weight_sha256": weight_hash,
                    "weight_dtype_for_pack": "float32",
                }
            )
            save_pack(name, pack, str(out))
            elapsed = time.time() - layer_started
            print(
                f"[pack shard {args.shard_index}] built {ordinal}/{len(selected)} {name} "
                f"shape={tuple(weight.shape)} elapsed={elapsed:.1f}s",
                flush=True,
            )
            rows.append(
                {"layer": name, "path": str(path), "status": "built", "elapsed_s": elapsed}
            )
            del weight, pack
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    journal = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "candidate_inventory_sha256": inventory_sha256(names),
        "block_in": args.block,
        "block_out": args.block,
        "lambda_smooth": args.lambda_smooth,
        "enable_permute": args.permute,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "layers": rows,
        "elapsed_s": time.time() - started,
    }
    journal_path = out / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    journal_path.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[pack shard {args.shard_index}] complete: {journal_path}", flush=True)


def finalize(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    actual_checkpoint_sha256 = sha256_file(checkpoint)
    keys = candidate_keys(checkpoint)
    names = [key.removesuffix(".weight") for key in keys]
    files = []
    errors = []
    for name in names:
        path = expected_pack_path(out, name)
        if not valid_existing(
            path,
            layer_name=name,
            checkpoint_sha256=actual_checkpoint_sha256,
            block=args.block,
            lambda_smooth=args.lambda_smooth,
            permute=args.permute,
        ):
            errors.append(f"missing or invalid: {name}")
            continue
        meta = read_meta(path)
        files.append(
            {
                "layer": name,
                "file": path.name,
                "sha256": sha256_file(path),
                "weight_sha256": meta["weight_sha256"],
                "in_features": meta["in_features"],
                "out_features": meta["out_features"],
            }
        )
    extra = sorted(
        path.name
        for path in out.glob("*.npz")
        if path.name not in {expected_pack_path(out, name).name for name in names}
    )
    if extra:
        errors.append(f"extra npz files: {extra[:5]}")
    manifest = {
        "schema_version": 1,
        "complete": not errors and len(files) == 180,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "candidate_inventory_sha256": inventory_sha256(names),
        "wrapped_layer_count": len(files),
        "block_in": args.block,
        "block_out": args.block,
        "lambda_smooth": args.lambda_smooth,
        "enable_permute": args.permute,
        "files": files,
        "errors": errors,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))
    if not manifest["complete"]:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.finalize:
        finalize(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
