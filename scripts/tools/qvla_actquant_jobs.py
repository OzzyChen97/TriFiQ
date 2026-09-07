#!/usr/bin/env python3
"""Generate immutable dynamic-GPU job manifests for calibration stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GROOT_PYTHON = "/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
OPENPI_PYTHON = "/home1/gyy/probe/miniforge3/envs/openpi/bin/python"
UNITS = {
    "gr00t_atomic_seen": {
        "family": "gr00t",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000",
    },
    "gr00t_composite_seen": {
        "family": "gr00t",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000",
    },
    "gr00t_composite_unseen": {
        "family": "gr00t",
        "checkpoint": ROOT / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000",
    },
    "pi05_all_target": {
        "family": "pi05",
        "checkpoint": ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch",
    },
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def implementation_digest() -> tuple[str, list[dict[str, str]]]:
    paths = (
        ROOT / "code/qvla_actquant/core.py",
        ROOT / "code/qvla_actquant/model_adapters.py",
        ROOT / "code/qvla_actquant/gguf_artifacts.py",
        ROOT / "code/qvla_actquant/runtime.py",
        ROOT / "scripts/tools/run_actquant_calibration.py",
    )
    records = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    return canonical_hash(records), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=(
            "qvla", "qvla-pack", "actquant-hsic", "actquant-fisher",
            "actquant-allocate", "actquant-pack",
        ),
        required=True,
    )
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--eligible-gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    provenance = json.loads(args.provenance.resolve().read_text(encoding="utf-8"))
    implementation_sha256, implementation_files = implementation_digest()
    jobs = []
    for unit, config in UNITS.items():
        python = GROOT_PYTHON if config["family"] == "gr00t" else OPENPI_PYTHON
        frozen = (args.frozen_dir / f"{unit}.json").resolve()
        if not frozen.is_file():
            raise FileNotFoundError(frozen)
        checkpoint_sha = provenance["checkpoints"][unit]["checkpoint_manifest_sha256"]
        stage_dir = (args.output_root / args.phase / unit).resolve()
        indices = range(args.shards) if args.phase in {
            "qvla", "actquant-hsic", "actquant-fisher"
        } else range(1)
        for index in indices:
            if args.phase == "qvla":
                qvla_batch_size = "4" if config["family"] == "gr00t" else "2"
                output = stage_dir / f"proxy_{index:03d}_of_{args.shards:03d}.pt"
                argv = [
                    python, str(ROOT / "scripts/tools/run_qvla_calibration.py"), "collect",
                    "--model-family", config["family"], "--checkpoint", str(config["checkpoint"]),
                    "--checkpoint-sha256", checkpoint_sha, "--frozen-manifest", str(frozen),
                    "--output", str(output), "--shard-index", str(index), "--shard-count", str(args.shards),
                    "--device", "cuda:0", "--batch-size", qvla_batch_size,
                ]
                expected = [str(output), str(output) + ".json"]
                # Measured peak for the two-shard GR00T layout is 15.7 GiB;
                # keep a rounded 16 GiB launch bound plus the scheduler's
                # separate 4 GiB device reserve.  pi0.5's 10.7--11.1 GiB
                # Hessian shard brings its live peak to about 20 GiB.
                required = 16_000 if config["family"] == "gr00t" else 20_000
            elif args.phase in {"actquant-hsic", "actquant-fisher"}:
                mode = "collect-hsic" if args.phase == "actquant-hsic" else "collect-fisher"
                # HSIC stores one pooled activation/action row per sample, so
                # batching is algebraically equivalent and amortizes model
                # transforms.  Fisher must stay at one: squaring a batch-mean
                # gradient is not the empirical sum of per-sample grad^2.
                batch_size = "4" if mode == "collect-hsic" else "1"
                suffix = "json" if mode == "collect-hsic" else "pt"
                output = stage_dir / f"{mode}_{index:03d}_of_{args.shards:03d}.{suffix}"
                argv = [
                    python, str(ROOT / "scripts/tools/run_actquant_calibration.py"), mode,
                    "--model-family", config["family"], "--checkpoint", str(config["checkpoint"]),
                    "--checkpoint-sha256", checkpoint_sha, "--frozen-manifest", str(frozen),
                    "--output", str(output), "--shard-index", str(index), "--shard-count", str(args.shards),
                    "--device", "cuda:0", "--batch-size", batch_size,
                ]
                expected = [str(output)] if mode == "collect-hsic" else [str(output), str(output) + ".json"]
                if mode == "collect-hsic":
                    required = 16_000 if config["family"] == "pi05" else 15_000
                else:
                    # Fisher keeps each parameter shard's FP32 grad^2 sum on
                    # device and performs one final D2H transfer.  Budget the
                    # additional 2--3 GiB explicitly instead of falling back
                    # to a per-frame PCIe/CPU accumulation bottleneck.
                    # The scheduler separately preserves 4 GiB.  Four-way
                    # GR00T Fisher sharding keeps only one quarter of the
                    # FP32 accumulator/gradient inventory resident, so use a
                    # 17 GiB launch bound and verify the live peak before
                    # retaining it.  pi0.5 remains at the conservative bound.
                    required = 24_000 if config["family"] == "pi05" else 17_000
            elif args.phase == "qvla-pack":
                output_root = (args.output_root / "packs" / "qvla" / unit).resolve()
                allocation = output_root / "allocation.json"
                pack = output_root / f"{unit}.qpack"
                manifest_output = output_root / "manifest.json"
                shards = sorted(
                    (args.output_root / "qvla" / unit).resolve().glob(
                        f"proxy_*_of_{args.shards:03d}.pt"
                    )
                )
                if len(shards) != args.shards:
                    raise ValueError(
                        f"{unit}: expected {args.shards} QVLA shards, found {len(shards)}"
                    )
                argv = [
                    python, str(ROOT / "scripts/tools/run_qvla_calibration.py"), "merge-pack",
                    "--model-family", config["family"], "--checkpoint", str(config["checkpoint"]),
                    "--checkpoint-sha256", checkpoint_sha, "--frozen-manifest", str(frozen),
                    "--shards", *map(str, shards), "--shard-count", str(args.shards),
                    "--allocation-output", str(allocation), "--pack-output", str(pack),
                    "--manifest-output", str(manifest_output), "--device", "cuda:0",
                ]
                expected = [str(allocation), str(pack), str(manifest_output)]
                required = 20_000 if config["family"] == "gr00t" else 22_000
            elif args.phase == "actquant-allocate":
                output_root = (args.output_root / "packs" / "actquant" / unit).resolve()
                allocation = output_root / "allocation.json"
                shards = sorted(
                    (args.output_root / "actquant-hsic" / unit).resolve().glob(
                        f"collect-hsic_*_of_{args.shards:03d}.json"
                    )
                )
                if len(shards) != args.shards:
                    raise ValueError(
                        f"{unit}: expected {args.shards} HSIC shards, found {len(shards)}"
                    )
                argv = [
                    python, str(ROOT / "scripts/tools/run_actquant_calibration.py"), "allocate",
                    "--model-family", config["family"], "--checkpoint", str(config["checkpoint"]),
                    "--checkpoint-sha256", checkpoint_sha, "--frozen-manifest", str(frozen),
                    "--hsic-shards", *map(str, shards), "--shard-count", str(args.shards),
                    "--output", str(allocation), "--device", "cuda:0",
                ]
                expected = [str(allocation)]
                required = 16_000 if config["family"] == "gr00t" else 18_000
            else:
                output_root = (args.output_root / "packs" / "actquant" / unit).resolve()
                allocation = output_root / "allocation.json"
                manifest_output = output_root / "manifest.json"
                fisher_shards = sorted(
                    (args.output_root / "actquant-fisher" / unit).resolve().glob(
                        f"collect-fisher_*_of_{args.shards:03d}.pt"
                    )
                )
                if not allocation.is_file():
                    raise FileNotFoundError(allocation)
                if len(fisher_shards) != args.shards:
                    raise ValueError(
                        f"{unit}: expected {args.shards} Fisher shards, found {len(fisher_shards)}"
                    )
                argv = [
                    python, str(ROOT / "scripts/tools/run_actquant_calibration.py"), "pack",
                    "--model-family", config["family"], "--checkpoint", str(config["checkpoint"]),
                    "--checkpoint-sha256", checkpoint_sha, "--frozen-manifest", str(frozen),
                    "--allocation", str(allocation), "--fisher-shards", *map(str, fisher_shards),
                    "--shard-count", str(args.shards), "--output-dir", str(output_root),
                    "--manifest-output", str(manifest_output), "--device", "cuda:0",
                ]
                expected = [str(manifest_output)]
                required = 20_000 if config["family"] == "gr00t" else 24_000
            calibration_dtype = (
                "bfloat16" if args.phase.startswith("qvla") else "float16"
            )
            dtype_variable = (
                "GR00T_MODEL_DTYPE"
                if config["family"] == "gr00t"
                else "OPENPI_MODEL_DTYPE"
            )
            jobs.append({
                "id": (
                    f"{args.phase}__{unit}__{index:03d}"
                    if args.phase in {"qvla", "actquant-hsic", "actquant-fisher"}
                    else f"{args.phase}__{unit}"
                ),
                "argv": argv,
                "cwd": str(ROOT),
                # Precision is part of the immutable job identity.  Never let
                # a shell default silently turn ActQuant into a BF16
                # calibration or QVLA into an FP16 calibration.
                "env": {
                    "PYTHONUNBUFFERED": "1",
                    # Each GPU job is independent.  Bound CPU math pools so
                    # concurrent shards use the 128-thread host without each
                    # spawning a machine-wide OpenMP/BLAS pool.
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "NUMEXPR_NUM_THREADS": "4",
                    "TOKENIZERS_PARALLELISM": "false",
                    dtype_variable: calibration_dtype,
                    "QVLA_ACTQUANT_CALIBRATION_DTYPE": calibration_dtype,
                    # Bind every newly generated ActQuant job identity to the
                    # exact collector/adapter implementation.  QVLA jobs that
                    # predate this field retain their already frozen receipts.
                    **(
                        {"QVLA_ACTQUANT_JOB_IMPLEMENTATION_SHA256": implementation_sha256}
                        if args.phase.startswith("actquant-")
                        else {}
                    ),
                },
                "required_mib": required,
                "expected_outputs": expected,
            })
    manifest = {
        "schema_version": 1,
        "kind": "qvla_actquant_gpu_jobs_v1",
        "phase": args.phase,
        "shard_count_per_model_unit": args.shards,
        "eligible_gpus": [int(value) for value in args.eligible_gpus.split(",") if value.strip()],
        "provenance": str(args.provenance.resolve()),
        "provenance_sha256": hashlib.sha256(args.provenance.resolve().read_bytes()).hexdigest(),
        "jobs": jobs,
        "implementation_sha256": implementation_sha256,
        "implementation_files": implementation_files,
    }
    manifest["jobs_sha256"] = canonical_hash(jobs)
    atomic_json(args.manifest_output.resolve(), manifest)
    print(json.dumps({
        "output": str(args.manifest_output.resolve()), "phase": args.phase,
        "jobs": len(jobs), "jobs_sha256": manifest["jobs_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
