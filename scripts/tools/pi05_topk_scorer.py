#!/usr/bin/env python3
"""True-mixed-deployment TopK D_PAC adjudicator for the π0.5 final port."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from collections import defaultdict

# Must be set before importing torch/openpi.  TopK mutates the module graph for
# every candidate; compiling those transient graphs is both wasted work and a
# source of candidate-dependent autotuning noise.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = REPO_ROOT / "code" / "pi05" / "openpi"
sys.path.insert(0, str(OPENPI_ROOT / "src"))
sys.path.insert(0, str(OPENPI_ROOT / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))

from openpi.policies import policy_config  # noqa: E402
from openpi.quant import (  # noqa: E402
    enable_duquant_if_configured,
    save_act_scales,
    sha256_file,
    static_scales_ready,
)
from openpi.quant.duquant_layers import DuQuantLinear, iter_duquant_layers  # noqa: E402
from openpi.training import config  # noqa: E402

from gr00t_select_plan import select_final  # noqa: E402
from pi05_batched_policy import (  # noqa: E402
    CALIBRATION_BATCHES,
    CALIBRATION_BATCH_SIZE,
    FLOW_STEPS,
    iter_batches,
    load_records as load_calibration_records,
    sample_batch,
)
from gr00t_func_metrics import aggregate_d_pac_sequences  # noqa: E402
from pi05_func_metrics import d_func, d_pac_sequence  # noqa: E402
from pi05_sensitivity_probe import (  # noqa: E402
    load_records as load_sensitivity_records,
    run_records,
    solver_divergence,
)


CHECKPOINT_SHA256 = "4174133479c6a51d79cac90d6a1739f32f928624eb529bf791cd5be942afdf1c"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch"
DEFAULT_BUFFER = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/calibration/pi05_robocasa365_seed0_n256.npz"
)
DEFAULT_PACK = REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"


def parse_indices(value: str | None, size: int) -> list[int]:
    if not value:
        return list(range(size))
    indices: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            indices.extend(range(int(left), int(right) + 1))
        else:
            indices.append(int(token))
    if len(indices) != len(set(indices)) or any(index < 0 or index >= size for index in indices):
        raise ValueError(f"invalid candidate indices for TopK size {size}: {indices}")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=False)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER))
    parser.add_argument("--pack-dir", default=str(DEFAULT_PACK))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-obs", type=int, default=16)
    parser.add_argument("--candidate-indices", default=None)
    parser.add_argument("--out", required=False)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--merge-journal", action="append", default=[])
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sequence_d_pac(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task/seed/replan grouping with singleton fallback for non-sequences."""
    has_replans = all("replan" in record for record in records)
    groups: dict[tuple[str, int, int | None], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        singleton = None if has_replans else index
        groups[(str(record.get("task", "unknown")), int(record.get("seed", -1)), singleton)].append(index)
    sequences = []
    for (task, seed, _), positions in sorted(groups.items()):
        selected = torch.tensor(positions, dtype=torch.long)
        result = d_pac_sequence(
            reference.index_select(1, selected),
            candidate.index_select(1, selected),
            [int(records[index].get("replan", 0)) for index in positions],
        )
        sequences.append({"task": task, "seed": seed, **result})
    return {
        **aggregate_d_pac_sequences(sequences),
        "sequences": sequences,
        "grouping": "task_seed_replan" if has_replans else "independent_action_prefix",
    }


def safe_source(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "candidate"))


def materialize_topk(
    selector_plan: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_names = list(selector_plan["layers"])
    group = int(selector_plan["meta"]["group"])
    files: list[dict[str, Any]] = []
    for index, candidate in enumerate(selector_plan["topk"]):
        skipped = set(candidate["skip_layers"])
        if not skipped <= set(all_names):
            raise ValueError(f"TopK {index} contains unknown skip layers")
        layers = {
            name: {
                "bits": 0 if name in skipped else 4,
                "group": group,
                "skip": name in skipped,
            }
            for name in all_names
        }
        payload = {
            "schema_version": 2,
            "meta": {
                "kind": "pi05_faithful_topk_candidate",
                "parent_selector_sha256": selector_plan["meta"].get("selector_plan_sha256"),
                "algorithm_authority": selector_plan["meta"]["algorithm_authority"],
                "algorithm_authority_sha256": selector_plan["meta"]["algorithm_authority_sha256"],
                "architecture_adapter": selector_plan["meta"]["architecture_adapter"],
                "architecture_adapter_sha256": selector_plan["meta"]["architecture_adapter_sha256"],
                "checkpoint_sha256": selector_plan["meta"]["checkpoint_sha256"],
                "config_sha256": selector_plan["meta"]["config_sha256"],
                "norm_stats_sha256": selector_plan["meta"]["norm_stats_sha256"],
                "candidate_inventory_sha256": selector_plan["meta"]["candidate_inventory_sha256"],
                "sensitivity_sha256": selector_plan["meta"]["sensitivity_sha256"],
                "calibration_buffer_sha256": selector_plan["meta"]["calibration_buffer_sha256"],
                "pack_manifest_sha256": selector_plan["meta"]["pack_manifest_sha256"],
                "lambda": selector_plan["meta"]["lambda"],
                "cka_to_cs_ratio": selector_plan["meta"]["cka_to_cs_ratio"],
                "cka_location": selector_plan["meta"]["cka_location"],
                "cs_location": selector_plan["meta"]["cs_location"],
                "guard_thresholds": selector_plan["meta"]["guard_thresholds"],
                "budget_reference": selector_plan["meta"]["budget_reference"],
                "budget_bytes": selector_plan["budget_bytes"],
                "fp16_total_bytes": selector_plan["fp16_total_bytes"],
                "total_bytes": candidate["bytes"],
                "topk_index": index,
                "topk_source": candidate["source"],
                "canonical_proxy": candidate["objective"],
                "expected_wrapped_layers": len(all_names) - len(skipped),
                "act_bits": 8,
                "act_percentile": 99.9,
                "calibration_batches": 32,
                "denoising_steps": FLOW_STEPS,
                "evaluation_split": "target",
                "n_action_steps": 16,
                "skip_semantics": "native torch.nn.Linear FP16",
                "adjudicated": False,
            },
            "packdirs": selector_plan.get("packdirs", {}),
            "layers": layers,
        }
        path = output_dir / f"topk_{index:02d}_{safe_source(candidate['source'])}.plan.json"
        atomic_json(path, payload)
        files.append(
            {
                "index": index,
                "path": path,
                "sha256": sha256_file(path),
                "candidate": candidate,
                "expected_wrapped": len(all_names) - len(skipped),
            }
        )
    return files


def clear_quant_environment() -> None:
    for key in list(os.environ):
        if key.startswith(("OPENPI_DUQUANT_", "OPENPI_ATM_", "OPENPI_OHB_")):
            os.environ.pop(key, None)


def configure_quant_environment(
    *,
    plan_path: Path,
    pack_dir: Path,
    scale_path: Path,
    buffer_hash: str,
    expected_wrapped: int,
) -> None:
    clear_quant_environment()
    os.environ.update(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "OPENPI_MODEL_DTYPE": "float16",
            "OPENPI_DUQUANT_PLAN": str(plan_path),
            "OPENPI_DUQUANT_PLAN_STRICT": "1",
            "OPENPI_DUQUANT_WBITS_DEFAULT": "4",
            "OPENPI_DUQUANT_ABITS": "8",
            "OPENPI_DUQUANT_BLOCK": "64",
            "OPENPI_DUQUANT_BLOCK_OUT": "64",
            "OPENPI_DUQUANT_EXPECT_BLOCK": "64",
            "OPENPI_DUQUANT_EXPECT_WRAPPED": str(expected_wrapped),
            "OPENPI_DUQUANT_LS": "0.15",
            "OPENPI_DUQUANT_PERMUTE": "0",
            "OPENPI_DUQUANT_ROW_ROT": "restore",
            "OPENPI_DUQUANT_ACT_PCT": "99.9",
            "OPENPI_DUQUANT_CALIB_STEPS": "32",
            "OPENPI_DUQUANT_DENOISING_STEPS": str(FLOW_STEPS),
            "OPENPI_DUQUANT_PACKDIR": str(pack_dir),
            "OPENPI_DUQUANT_ACT_SCALE_PATH": str(scale_path),
            "OPENPI_DUQUANT_CALIB_BUFFER_SHA256": buffer_hash,
            "OPENPI_CHECKPOINT_SHA256": CHECKPOINT_SHA256,
            "OPENPI_DUQUANT_STRICT_ARTIFACTS": "1",
            "OPENPI_DUQUANT_PRECACHE_WEIGHTS": "1",
            "OPENPI_DUQUANT_TRITON": "0",
            "OPENPI_DUQUANT_QUIET": "1",
        }
    )


def _parent_and_attribute(model: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def unwrap_all(model: torch.nn.Module) -> int:
    """Restore every DuQuantLinear to an actual native torch.nn.Linear."""
    wrapped = list(iter_duquant_layers(model))
    for name, module in wrapped:
        parent, attribute = _parent_and_attribute(model, name)
        base = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module._weight.device,
            dtype=module._weight.dtype,
        )
        base.weight = torch.nn.Parameter(module._weight.detach(), requires_grad=False)
        if module.bias is not None:
            base.bias = torch.nn.Parameter(module.bias.detach(), requires_grad=False)
        setattr(parent, attribute, base)
    setattr(
        model,
        "_openpi_duquant_runtime",
        {"enabled": False, "candidate_layers": 180, "wrapped_layers": 0},
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if iter_duquant_layers(model):
        raise RuntimeError("failed to restore native FP16 model after TopK candidate")
    return len(wrapped)


def calibrate_candidate(
    *,
    policy,
    records: list[dict],
    device: str,
    scale_path: Path,
    plan_path: Path,
    pack_dir: Path,
    buffer_path: Path,
    buffer_hash: str,
) -> dict[str, Any]:
    if static_scales_ready(policy._model):
        sidecar = Path(str(scale_path) + ".json")
        if not scale_path.is_file() or not sidecar.is_file():
            raise RuntimeError("static scales are ready without a complete persisted artifact")
        return json.loads(sidecar.read_text(encoding="utf-8"))
    timings: list[float] = []
    for index, batch in enumerate(iter_batches(records, CALIBRATION_BATCH_SIZE)):
        started = time.perf_counter()
        actions = sample_batch(policy, batch, device, num_steps=FLOW_STEPS)
        if actions.shape != (len(batch), 50, 32) or not torch.isfinite(actions).all():
            raise RuntimeError(f"invalid A8 calibration output at batch {index}")
        timings.append(time.perf_counter() - started)
        if index + 1 < CALIBRATION_BATCHES and static_scales_ready(policy._model):
            raise RuntimeError(f"A8 scales froze early after {index + 1} calibration batches")
    if (
        len(records) != CALIBRATION_BATCHES * CALIBRATION_BATCH_SIZE
        or not static_scales_ready(policy._model)
    ):
        raise RuntimeError("A8 scales did not freeze after exactly 32 batches of 8")
    payload = save_act_scales(
        policy._model,
        scale_path,
        {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "calibration_buffer_sha256": buffer_hash,
            "calibration_buffer_path": str(buffer_path),
            "plan_path": str(plan_path),
            "pack_dir": str(pack_dir),
            "calibration_reference": "candidate_true_mixed_deployment",
            "calibration_seed": 0,
            "calibration_batch_size": CALIBRATION_BATCH_SIZE,
            "calibration_observations": len(records),
            "request_latency_mean_s": float(np.mean(timings)),
        },
    )
    return payload


def validate_native_skip(model: torch.nn.Module, selector_plan: dict[str, Any], candidate: dict[str, Any]) -> None:
    modules = dict(model.named_modules())
    skipped = set(candidate["skip_layers"])
    for name in selector_plan["layers"]:
        module = modules.get(name)
        if name in skipped:
            if type(module) is not torch.nn.Linear:
                raise RuntimeError(f"skip layer is not native nn.Linear: {name} ({type(module)})")
        elif not isinstance(module, DuQuantLinear):
            raise RuntimeError(f"quantized layer is not DuQuantLinear: {name} ({type(module)})")


def load_journal(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def score(args: argparse.Namespace) -> Path:
    selector_path = Path(args.plan).expanduser().resolve()
    selector_plan = json.loads(selector_path.read_text(encoding="utf-8"))
    selector_hash = sha256_file(selector_path)
    selector_plan["meta"]["selector_plan_sha256"] = selector_hash
    if selector_plan["meta"].get("adjudicated") is not False:
        raise ValueError("input must be a non-adjudicated selector plan")
    if selector_plan["meta"].get("algorithm_authority_sha256") != sha256_file(
        REPO_ROOT / "scripts/tools/gr00t_select_plan.py"
    ):
        raise ValueError("selector algorithm authority hash changed")
    if selector_plan["meta"].get("architecture_adapter_sha256") != sha256_file(
        REPO_ROOT / "scripts/tools/pi05_select_plan.py"
    ):
        raise ValueError("π0.5 selector architecture adapter hash changed")
    metric_path = REPO_ROOT / "scripts/tools/pi05_func_metrics.py"
    if selector_plan["meta"].get("functional_metric_sha256") != sha256_file(metric_path):
        raise ValueError("selector was produced by a stale π0.5 functional metric")
    if args.n_obs != 16:
        raise ValueError("faithful TopK adjudication requires n_obs=16")
    output_base = Path(args.out).expanduser().resolve()
    journal_path = Path(str(output_base) + ".journal.json")
    candidate_dir = Path(str(output_base) + ".topk")
    scale_dir = Path(str(output_base) + ".topk_a8")
    files = materialize_topk(selector_plan, candidate_dir)
    selected_indices = parse_indices(args.candidate_indices, len(files))

    buffer_path = Path(args.buffer).expanduser().resolve()
    buffer_hash = sha256_file(buffer_path)
    if buffer_hash != selector_plan["meta"]["calibration_buffer_sha256"]:
        raise ValueError("TopK scorer buffer differs from selector provenance")
    records = load_calibration_records(
        buffer_path, CALIBRATION_BATCHES * CALIBRATION_BATCH_SIZE
    )
    score_records = load_sensitivity_records(buffer_path, args.n_obs)
    pack_dir = Path(args.pack_dir).expanduser().resolve()
    if sha256_file(pack_dir / "manifest.json") != selector_plan["meta"]["pack_manifest_sha256"]:
        raise ValueError("TopK scorer pack differs from selector provenance")

    clear_quant_environment()
    os.environ["OPENPI_MODEL_DTYPE"] = "float16"
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_pretrain_human300"),
        Path(args.checkpoint_dir).expanduser().resolve(),
        pytorch_device=args.device,
    )
    model = policy._model
    model.to(args.device)
    if iter_duquant_layers(model):
        raise RuntimeError("FP16 reference unexpectedly contains DuQuantLinear")
    reference_trajectory, reference_actions, reference_timings = run_records(
        policy, score_records, args.device, noise_index=0
    )
    reference_fingerprint = hashlib.sha256(
        np.ascontiguousarray(reference_trajectory.numpy()).tobytes()
    ).hexdigest()

    existing = load_journal(journal_path)
    if existing and existing.get("selector_sha256") != selector_hash:
        raise ValueError("existing TopK journal belongs to a different selector plan")
    scored_by_index = {
        int(row["index"]): row for row in existing.get("scored", [])
    }
    journal: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "selector_path": str(selector_path),
        "selector_sha256": selector_hash,
        "buffer_path": str(buffer_path),
        "buffer_sha256": buffer_hash,
        "pack_dir": str(pack_dir),
        "pack_manifest_sha256": sha256_file(pack_dir / "manifest.json"),
        "reference_trajectory_sha256": reference_fingerprint,
        "functional_metric_sha256": sha256_file(metric_path),
        "reference_latency_mean_s": float(np.mean(reference_timings)),
        "requested_indices": selected_indices,
        "scored": list(scored_by_index.values()),
    }
    atomic_json(journal_path, journal)

    for index in selected_indices:
        descriptor = files[index]
        if index in scored_by_index:
            row = scored_by_index[index]
            if row.get("plan_sha256") != descriptor["sha256"]:
                raise ValueError(f"journal plan hash mismatch at candidate {index}")
            print(f"[pi05 topk] reuse {index}/{len(files)} {row['source']}", flush=True)
            continue
        scale_path = scale_dir / f"topk_{index:02d}.npz"
        configure_quant_environment(
            plan_path=descriptor["path"],
            pack_dir=pack_dir,
            scale_path=scale_path,
            buffer_hash=buffer_hash,
            expected_wrapped=descriptor["expected_wrapped"],
        )
        started = time.time()
        runtime = enable_duquant_if_configured(model)
        model.to(args.device)
        if runtime["wrapped_layers"] != descriptor["expected_wrapped"]:
            raise RuntimeError("TopK wrapped-layer count mismatch")
        validate_native_skip(model, selector_plan, descriptor["candidate"])
        scale_payload = calibrate_candidate(
            policy=policy,
            records=records,
            device=args.device,
            scale_path=scale_path,
            plan_path=descriptor["path"],
            pack_dir=pack_dir,
            buffer_path=buffer_path,
            buffer_hash=buffer_hash,
        )
        candidate_trajectory, _candidate_actions, timings = run_records(
            policy, score_records, args.device, noise_index=0
        )
        functional = d_func(reference_trajectory, candidate_trajectory, 1.2)
        pac = sequence_d_pac(reference_trajectory, candidate_trajectory, score_records)
        full32_solver = solver_divergence(reference_trajectory, candidate_trajectory, 1.2)
        row = {
            "index": index,
            "source": descriptor["candidate"]["source"],
            "proxy": float(descriptor["candidate"]["objective"]),
            "d_func": float(functional["d_func"]),
            "d_pac": float(pac["d_pac"]),
            "d_solver": float(functional["d_solver"]),
            "d_solver_full32": float(full32_solver),
            "d_func_components": {key: value for key, value in functional.items() if key != "per_obs"},
            "per_obs": functional["per_obs"],
            "d_pac_per_sequence": pac["per_sequence"],
            "d_pac_components": pac,
            "n_wrapped": runtime["wrapped_layers"],
            "n_native_skip": len(descriptor["candidate"]["skip_layers"]),
            "plan_path": str(descriptor["path"]),
            "plan_sha256": descriptor["sha256"],
            "a8_scale_path": str(scale_path),
            "a8_scale_sha256": scale_payload["npz_sha256"],
            "latency_mean_s": float(np.mean(timings)),
            "elapsed_s": time.time() - started,
        }
        scored_by_index[index] = row
        journal["scored"] = [scored_by_index[key] for key in sorted(scored_by_index)]
        atomic_json(journal_path, journal)
        restored = unwrap_all(model)
        if restored != runtime["wrapped_layers"]:
            raise RuntimeError("native model restoration count mismatch")
        clear_quant_environment()
        print(
            f"[pi05 topk] {index + 1}/{len(files)} {row['source']} "
            f"D_PAC={row['d_pac']:.6g} D_func={row['d_func']:.6g} wrapped={row['n_wrapped']} "
            f"elapsed={row['elapsed_s']:.1f}s",
            flush=True,
        )

    journal["complete"] = set(scored_by_index) == set(selected_indices)
    journal["scored"] = [scored_by_index[key] for key in sorted(scored_by_index)]
    atomic_json(journal_path, journal)
    return journal_path


def merged_rows(args: argparse.Namespace, own_journal: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = [Path(value).expanduser().resolve() for value in args.merge_journal]
    if own_journal is not None:
        paths.append(own_journal)
    if not paths:
        default = Path(str(Path(args.out).expanduser().resolve()) + ".journal.json")
        paths = [default]
    reference: dict[str, Any] | None = None
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        comparable = {
            key: payload[key]
            for key in (
                "selector_sha256",
                "buffer_sha256",
                "pack_manifest_sha256",
                "reference_trajectory_sha256",
                "functional_metric_sha256",
            )
        }
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError(f"incompatible TopK journal: {path}")
        for row in payload["scored"]:
            index = int(row["index"])
            if index in rows:
                raise ValueError(f"duplicate TopK candidate {index} across journals")
            rows[index] = row
    return reference or {}, [rows[index] for index in sorted(rows)]


def finalize(args: argparse.Namespace, own_journal: Path | None = None) -> tuple[Path, Path]:
    selector_path = Path(args.plan).expanduser().resolve()
    selector_plan = json.loads(selector_path.read_text(encoding="utf-8"))
    selector_hash = sha256_file(selector_path)
    provenance, scored = merged_rows(args, own_journal)
    expected = set(range(len(selector_plan["topk"])))
    actual = {int(row["index"]) for row in scored}
    if actual != expected:
        raise ValueError(f"incomplete TopK matrix: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    if provenance["selector_sha256"] != selector_hash:
        raise ValueError("TopK journals do not match selector")

    final = select_final(scored, tol=args.tol, key="d_pac")
    if final is None:
        raise RuntimeError("5% tie-set selection returned no candidate")
    sorted_rows = sorted(scored, key=lambda row: float(row["d_pac"]))
    bootstrap = None
    if len(sorted_rows) >= 2:
        best, runner = sorted_rows[:2]
        best_values = np.asarray(best["d_pac_per_sequence"], dtype=np.float64)
        runner_values = np.asarray(runner["d_pac_per_sequence"], dtype=np.float64)
        generator = np.random.default_rng(123)
        indices = generator.integers(0, len(best_values), size=(10000, len(best_values)))
        differences = (best_values[indices] - runner_values[indices]).mean(axis=1)
        bootstrap = {
            "best": best["source"],
            "runner_up": runner["source"],
            "mean_difference": float(differences.mean()),
            "ci95": [
                float(np.quantile(differences, 0.025)),
                float(np.quantile(differences, 0.975)),
            ],
            "p_best_wins": float((differences <= 0).mean()),
            "draws": 10000,
        }

    chosen_plan = json.loads(Path(final["plan_path"]).read_text(encoding="utf-8"))
    chosen_plan["meta"].update(
        {
            "kind": "gdsq_vla_pi05_faithful_topk_adjudicated",
            "adjudicated": True,
            "parent_selector_path": str(selector_path),
            "parent_selector_sha256": selector_hash,
            "final_source": final["source"],
            "final_metric": "d_pac_v1",
            "final_d_pac": final["d_pac"],
            "final_d_func": final["d_func"],
            "final_d_solver": final["d_solver"],
            "tie_tolerance": args.tol,
            "tie_rule": "minimum D_PAC; candidates within relative tolerance; minimum canonical proxy",
            "calibration_buffer_sha256": provenance["buffer_sha256"],
            "pack_manifest_sha256": provenance["pack_manifest_sha256"],
            "reference_trajectory_sha256": provenance["reference_trajectory_sha256"],
            "functional_metric_sha256": provenance["functional_metric_sha256"],
            "requires_fresh_plan_specific_a8": True,
        }
    )
    output_base = Path(args.out).expanduser().resolve()
    final_plan_path = Path(str(output_base) + ".final_plan.json")
    report_path = Path(str(output_base) + ".report.json")
    atomic_json(final_plan_path, chosen_plan)
    report = {
        "schema_version": 1,
        "complete": True,
        "meta": {
            "selector_path": str(selector_path),
            "selector_sha256": selector_hash,
            "metric": "d_pac_v1",
            "n_obs": args.n_obs,
            "tol": args.tol,
            "selection_rule": "minimum original-FP16-relative D_PAC; relative tie set; canonical proxy tie-break",
            "true_mixed_deployment": True,
            "skip_semantics": "native FP16 torch.nn.Linear",
            "shared_calibration_buffer_sha256": provenance["buffer_sha256"],
            "paired_noise": True,
            "functional_metric_sha256": provenance["functional_metric_sha256"],
            "paired_bootstrap": bootstrap,
            "final_source": final["source"],
        },
        "scored": scored,
        "final": {key: value for key, value in final.items() if key not in ("per_obs",)},
        "final_plan_path": str(final_plan_path),
        "final_plan_sha256": sha256_file(final_plan_path),
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "final_source": final["source"],
                "final_d_pac": final["d_pac"],
                "final_d_func": final["d_func"],
                "final_proxy": final["proxy"],
                "final_plan": str(final_plan_path),
                "final_plan_sha256": sha256_file(final_plan_path),
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return final_plan_path, report_path


def selftest() -> None:
    selected = select_final(
        [
            {"source": "a", "d_pac": 1.0, "proxy": 2.0},
            {"source": "b", "d_pac": 1.04, "proxy": 1.0},
            {"source": "c", "d_pac": 1.06, "proxy": 0.0},
        ],
        tol=0.05,
        key="d_pac",
    )
    assert selected["source"] == "b"
    assert parse_indices("0,2-3", 4) == [0, 2, 3]
    print("[pi05_topk_scorer] selftest OK")


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    if not args.plan or not args.out:
        raise SystemExit("--plan and --out are required")
    own_journal = None
    if not args.finalize_only:
        own_journal = score(args)
    if args.finalize_only or not args.candidate_indices:
        finalize(args, own_journal)


if __name__ == "__main__":
    main()
