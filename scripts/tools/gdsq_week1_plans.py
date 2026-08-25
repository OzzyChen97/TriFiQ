#!/usr/bin/env python3
"""Freeze the same-budget plans and search controls for GDSQ-VLA week 1.

This script is deliberately result-blind.  It consumes only the already frozen
sensitivity, inventory, checkpoint, and final-plan artifacts.  It writes plan
files plus a preregistration manifest; no RoboCasa success result is read and
no threshold is fitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "scripts/tools"
sys.path.insert(0, str(TOOLS_ROOT))

import gr00t_select_plan as selector  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "runs/gdsq_week1_preregistered_v1"
GR00T_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/robocasa365/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
GR00T_SENSITIVITY = (
    REPO_ROOT
    / "checkpoints/packs/robocasa365/"
    "sensitivity_robocasa365_atomic_protocolfix_d4_g64_b4_6_dit.json"
)
GR00T_FINAL = (
    REPO_ROOT
    / "checkpoints/packs/robocasa365/"
    "gr00t_quant_plan_robocasa365_cscka_16to1_adjudicated.final_plan.json"
)
GR00T_PARENT = (
    REPO_ROOT
    / "checkpoints/packs/robocasa365/"
    "gr00t_quant_plan_robocasa365_cscka_16to1.json"
)
GR00T_PACK = (
    REPO_ROOT
    / "checkpoints/packs/robocasa365/"
    "duquant_packed_robocasa365_protocolfix_d4_w4a8_b64c32ls015"
)
GR00T_CKA_ONLY = (
    REPO_ROOT
    / "checkpoints/packs/robocasa365/"
    "gr00t_quant_plan_robocasa365_protocolfix_d4_ckaonly_adjudicated.final_plan.json"
)
GR00T_CS_ONLY = (
    REPO_ROOT
    / "checkpoints/packs/robocasa365/"
    "gr00t_quant_plan_robocasa365_protocolfix_d4_csonly_adjudicated.final_plan.json"
)

PI05_INVENTORY = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/plans/pi05_candidate_inventory_d4.json"
)
PI05_SENSITIVITY = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/sensitivity/"
    "pi05_sensitivity_action_n16_d4_merged.json"
)
PI05_FINAL = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/plans/"
    "pi05_gdsq_cscka_16to1_d4.final_plan.json"
)
PI05_PARENT = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/plans/"
    "pi05_gdsq_cscka_16to1_d4.selector.json"
)
PI05_PACK = (
    REPO_ROOT / "runs/pi05_gdsq_port/packs/pi05_robocasa_block64_w4a8_ls015"
)
PI05_BUFFER = (
    REPO_ROOT
    / "runs/pi05_gdsq_gr00t_aligned/calibration/"
    "pi05_robocasa365_seed0_n256.npz"
)

SELECTOR_PATH = REPO_ROOT / "runs/atmohb_dynamic_selector_v8/selector.json"
SELECTOR_SHA256 = "0f3178726c2b784898f18dfde248d9a9bae152da6ffcfdc299e8bdc02f0bd871"
RANDOM_DRAW_SEED = 2026082301
FUNCTIONAL_SUBSET_SEED = 2026082302
SEARCH_CANDIDATE_COUNT = 26
FUNCTIONAL_COUNTS = {"gr00t": 9, "pi05": 7}
DEVELOPMENT_REPRESENTATIVES = 7
BYTE_TOLERANCE = 0.005
GROUP = 64
ROW_ROT = "restore"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def checkpoint_artifact(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"checkpoint contains no model safetensors: {directory}")
    rows = [artifact(path) for path in files]
    return {
        "path": str(directory.resolve()),
        "files": rows,
        "canonical_sha256": canonical_sha(
            [{"name": Path(row["path"]).name, "sha256": row["sha256"]} for row in rows]
        ),
    }


def directory_artifact(directory: Path, pattern: str) -> dict[str, Any]:
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} artifacts under {directory}")
    rows = [artifact(path) for path in files]
    return {
        "path": str(directory.resolve()),
        "pattern": pattern,
        "file_count": len(rows),
        "files": rows,
        "canonical_sha256": canonical_sha(
            [{"name": Path(row["path"]).name, "sha256": row["sha256"]} for row in rows]
        ),
    }


def gr00t_shapes(candidate_names: list[str]) -> dict[str, dict[str, Any]]:
    """Read only safetensors headers/slices, never materialize checkpoint weights."""
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required; run in the groot_test env") from error

    needed = set(candidate_names)
    shapes: dict[str, dict[str, Any]] = {}
    bias_names: set[str] = set()
    files = sorted(GR00T_CHECKPOINT.glob("model*.safetensors"))
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            keys = set(handle.keys())
            bias_names.update(key[:-5] for key in keys if key.endswith(".bias"))
            for name in sorted(needed - set(shapes)):
                key = f"{name}.weight"
                if key not in keys:
                    continue
                shape = tuple(int(value) for value in handle.get_slice(key).get_shape())
                if len(shape) != 2:
                    raise ValueError(f"candidate is not a matrix: {name} {shape}")
                shapes[name] = {
                    "out": shape[0],
                    "in": shape[1],
                    "has_bias": name in bias_names or f"{name}.bias" in keys,
                }
    missing = sorted(needed - set(shapes))
    if missing:
        raise ValueError(f"GR00T checkpoint is missing {len(missing)} candidates: {missing[:5]}")
    return {name: shapes[name] for name in candidate_names}


def pi05_shapes(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = inventory.get("layers") or []
    if len(rows) != 180:
        raise ValueError(f"π0.5 inventory must contain 180 candidates, got {len(rows)}")
    shapes = {
        str(row["name"]): {
            "out": int(row["out_features"]),
            "in": int(row["in_features"]),
            "has_bias": False,
        }
        for row in rows
    }
    if len(shapes) != len(rows):
        raise ValueError("π0.5 inventory contains duplicate names")
    return shapes


def fp16_entry(model: str) -> dict[str, Any]:
    return {
        "bits": None if model == "gr00t" else 0,
        "group": GROUP,
        "skip": True,
    }


def normalize_plan(model: str, plan: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for name, row in plan.items():
        if bool(row.get("skip")):
            normalized[name] = fp16_entry(model)
        else:
            normalized[name] = {
                "bits": int(row["bits"]),
                "group": int(row.get("group", GROUP)),
                "skip": False,
            }
    return normalized


def random_masks(
    *,
    shapes: dict[str, dict[str, Any]],
    protected: set[str],
    n_w4: int,
    target_bytes: float,
    reference_plan: dict[str, Any],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Draw size-stratified masks with exactly the reference byte count.

    Candidate layers fall into a small number of matrix-shape buckets.  Holding
    the number of W4 layers in each byte-savings bucket fixed gives exact byte
    matching without using a rollout or proxy score, while randomizing the
    identities inside every bucket.
    """
    eligible = [name for name in shapes if name not in protected]
    if n_w4 > len(eligible):
        raise ValueError(f"cannot quantize {n_w4}/{len(eligible)} eligible layers")
    savings = {
        name: selector.layer_bytes_fp16(
            shapes[name]["out"], shapes[name]["in"], shapes[name]["has_bias"]
        )
        - selector.layer_bytes_quant(
            shapes[name]["out"],
            shapes[name]["in"],
            shapes[name]["has_bias"],
            4,
            GROUP,
            row_rot=ROW_ROT,
        )
        for name in eligible
    }
    buckets: dict[float, list[str]] = {}
    for name in eligible:
        buckets.setdefault(savings[name], []).append(name)
    desired_counts = {
        saving: sum(
            not bool(reference_plan[name].get("skip")) for name in bucket_names
        )
        for saving, bucket_names in buckets.items()
    }
    if any(
        not bool(reference_plan[name].get("skip")) for name in protected
    ):
        raise ValueError("reference plan quantizes a guard-protected layer")
    if sum(desired_counts.values()) != n_w4:
        raise ValueError("reference plan W4 count differs from requested random-mask count")
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(results) < count and attempts < count * 100:
        attempts += 1
        chosen: set[str] = set()
        for saving, bucket_names in sorted(buckets.items()):
            chosen.update(rng.sample(bucket_names, desired_counts[saving]))
        mask = tuple(sorted(chosen))
        if mask in seen:
            continue
        seen.add(mask)
        plan = {
            name: (
                {"bits": 4, "group": GROUP, "skip": False}
                if name in chosen
                else {"bits": None, "group": GROUP, "skip": True}
            )
            for name in shapes
        }
        actual = selector.plan_total_bytes(plan, shapes, ROW_ROT)
        if abs(actual - target_bytes) / target_bytes > 1e-12:
            raise AssertionError("size-stratified random mask is not exact-byte matched")
        results.append(plan)
    if len(results) != count:
        raise RuntimeError(
            f"generated only {len(results)}/{count} unique byte-matched masks"
        )
    return results


def action_objective(plan: dict[str, Any], weights: dict[str, float]) -> float:
    return sum(
        float(weights[name])
        for name, row in plan.items()
        if not bool(row.get("skip"))
    )


def source_plan_from_topk(
    parent: dict[str, Any], shapes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    topk = parent.get("topk") or []
    if not topk:
        raise ValueError("parent selector contains no TopK candidates")
    first = min(topk, key=lambda row: float(row.get("objective", math.inf)))
    skipped = set(first.get("skip_layers") or [])
    if not skipped:
        raise ValueError("parent TopK candidate lacks skip_layers")
    return {
        name: (
            {"bits": None, "group": GROUP, "skip": True}
            if name in skipped
            else {"bits": 4, "group": GROUP, "skip": False}
        )
        for name in shapes
    }


def plan_document(
    *,
    model: str,
    plan_id: str,
    plan: dict[str, Any],
    shapes: dict[str, dict[str, Any]],
    pack_dir: Path,
    budget: float,
    inputs: dict[str, Any],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_plan(model, plan)
    total = selector.plan_total_bytes(normalized, shapes, ROW_ROT)
    meta = {
        "kind": "gdsq_week1_preregistered_plan",
        "model": model,
        "plan_id": plan_id,
        "frozen_before_formal_rollouts": True,
        "budget_reference": "uniform_w6_static_bytes",
        "budget_bytes": budget,
        "budget_semantics": "theoretical tightly packed static candidate weights",
        "within_budget": total <= budget + 1e-3,
        "group": GROUP,
        "row_rotation": ROW_ROT,
        "weight_activation_profile": "plan-selected Wbits / static A8",
        "activation_calibration": {
            "observations": 256,
            "batches": 32,
            "batch_size": 8,
            "percentile": 99.9,
            "denoising_steps": 4,
        },
        "inputs": inputs,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    meta.update(extra_meta or {})
    return {
        "schema_version": 1,
        "meta": meta,
        "budget_bytes": budget,
        "total_bytes": total,
        "packdirs": {str(GROUP): str(pack_dir.resolve())},
        "layers": normalized,
    }


def write_plan(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    atomic_json(path, document)
    return {
        **artifact(path),
        "plan_id": document["meta"]["plan_id"],
        "total_bytes": document["total_bytes"],
        "budget_bytes": document["budget_bytes"],
        "quantized_layers": sum(
            not bool(row.get("skip")) for row in document["layers"].values()
        ),
    }


def model_inputs(model: str) -> dict[str, Any]:
    if model == "gr00t":
        return {
            "sensitivity": artifact(GR00T_SENSITIVITY),
            "final_plan": artifact(GR00T_FINAL),
            "parent_selector": artifact(GR00T_PARENT),
            "checkpoint": checkpoint_artifact(GR00T_CHECKPOINT),
            "pack_inventory": directory_artifact(GR00T_PACK, "*.npz"),
        }
    return {
        "sensitivity": artifact(PI05_SENSITIVITY),
        "final_plan": artifact(PI05_FINAL),
        "parent_selector": artifact(PI05_PARENT),
        "inventory": artifact(PI05_INVENTORY),
        "checkpoint": artifact(
            REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/model.safetensors"
        ),
        "checkpoint_config": artifact(
            REPO_ROOT / "checkpoints/robocasa/pi05_pretrain_human300_pytorch/config.json"
        ),
        "pack_manifest": artifact(PI05_PACK / "manifest.json"),
        "calibration_buffer": artifact(PI05_BUFFER),
    }


def generate_model(model: str, root: Path) -> dict[str, Any]:
    if model == "gr00t":
        sensitivity_path, final_path, parent_path, pack_dir = (
            GR00T_SENSITIVITY,
            GR00T_FINAL,
            GR00T_PARENT,
            GR00T_PACK,
        )
        final = read_json(final_path)
        shapes = gr00t_shapes(list(final["layers"]))
    else:
        sensitivity_path, final_path, parent_path, pack_dir = (
            PI05_SENSITIVITY,
            PI05_FINAL,
            PI05_PARENT,
            PI05_PACK,
        )
        final = read_json(final_path)
        shapes = pi05_shapes(read_json(PI05_INVENTORY))
    sensitivity = read_json(sensitivity_path)
    parent = read_json(parent_path)
    if set(shapes) != set(final["layers"]):
        raise ValueError(f"{model}: final plan and candidate inventory differ")
    names = list(shapes)
    selector.BITS_ORDER[:] = [4]
    scores = selector.build_scores(
        sensitivity, names, 16.0, 1.0, cka_field="cka_dit"
    )
    weights, weight_log = selector.build_weights_with_log(
        sensitivity, names, metric="d_func"
    )
    thresholds = (sensitivity.get("meta") or {}).get("guard_thresholds") or {}
    tau_rms, tau_sat = thresholds.get("tau_rms"), thresholds.get("tau_sat")
    if tau_rms is None or tau_sat is None:
        tau_rms, tau_sat = selector.estimate_guard_thresholds(
            sensitivity, names, bit=4, margin=1.5
        )
    filtered_scores, removed = selector.filter_guarded(
        scores, sensitivity, names, tau_rms, tau_sat, bit=4
    )
    protected = {
        name for name in names if filtered_scores.get(name, {}).get(4) is None
    }
    removed_names = {row["layer"] for row in removed}
    if not removed_names <= protected:
        raise AssertionError("guard-filtered layers did not leave the search space")

    uniform_w6 = {
        name: {"bits": 6, "group": GROUP, "skip": False} for name in names
    }
    budget = selector.plan_total_bytes(uniform_w6, shapes, ROW_ROT)
    final_total = selector.plan_total_bytes(final["layers"], shapes, ROW_ROT)
    final_w4 = sum(not bool(row.get("skip")) for row in final["layers"].values())
    if final_total > budget + 1e-3:
        raise ValueError(f"{model}: frozen final plan exceeds uniform-W6 budget")
    inputs = model_inputs(model)
    out_dir = root / "plans" / model
    plans: dict[str, Any] = {}

    uniform_doc = plan_document(
        model=model,
        plan_id=f"{model}_uniform_w6",
        plan=uniform_w6,
        shapes=shapes,
        pack_dir=pack_dir,
        budget=budget,
        inputs=inputs,
        extra_meta={"baseline": "uniform_w6", "all_candidate_weight_bits": 6},
    )
    plans["uniform_w6"] = write_plan(out_dir / "uniform_w6.plan.json", uniform_doc)

    masks = random_masks(
        shapes=shapes,
        protected=protected,
        n_w4=final_w4,
        target_bytes=final_total,
        reference_plan=final["layers"],
        count=SEARCH_CANDIDATE_COUNT,
        seed=RANDOM_DRAW_SEED + (0 if model == "gr00t" else 1000),
    )
    ordering = list(range(SEARCH_CANDIDATE_COUNT))
    random.Random(FUNCTIONAL_SUBSET_SEED + (0 if model == "gr00t" else 1000)).shuffle(
        ordering
    )
    functional_indices = sorted(ordering[: FUNCTIONAL_COUNTS[model]])
    random_files = []
    for index, mask in enumerate(masks):
        document = plan_document(
            model=model,
            plan_id=f"{model}_random_mask_{index:02d}",
            plan=mask,
            shapes=shapes,
            pack_dir=pack_dir,
            budget=budget,
            inputs=inputs,
            extra_meta={
                "baseline": "search_matched_random",
                "candidate_index": index,
                "candidate_draw_seed": RANDOM_DRAW_SEED
                + (0 if model == "gr00t" else 1000),
                "functional_discrimination_preregistered": index in functional_indices,
                "target_final_plan_bytes": final_total,
                "byte_relative_error": abs(
                    selector.plan_total_bytes(mask, shapes, ROW_ROT) - final_total
                )
                / final_total,
                "guard_protected_layers": sorted(protected),
                "selection_blinding": (
                    "functional subset frozen before dev/held-out rollouts; held-out results "
                    "must not select a representative"
                ),
            },
        )
        random_files.append(
            write_plan(out_dir / "random" / f"mask_{index:02d}.plan.json", document)
        )
    plans["search_matched_random"] = {
        "candidate_count": SEARCH_CANDIDATE_COUNT,
        "functional_discrimination_count": FUNCTIONAL_COUNTS[model],
        "functional_candidate_indices": functional_indices,
        "development_representatives_after_functional_ranking": DEVELOPMENT_REPRESENTATIVES,
        "files": random_files,
    }

    action_candidates = list(masks)
    action_scores = {name: {4: 1.0} for name in names}
    for name in protected:
        action_scores[name] = {}
    action_base, action_base_objective = selector.milp_binary_plan(
        shapes, action_scores, weights, budget, ROW_ROT, group=GROUP
    )
    if action_base is None:
        action_base, action_base_objective = selector.greedy_plan(
            shapes, action_scores, weights, budget, ROW_ROT
        )
    action_candidates[0] = action_base
    ranked_action = sorted(
        range(len(action_candidates)),
        key=lambda index: (action_objective(action_candidates[index], weights), index),
    )
    action_functional = sorted(ranked_action[: FUNCTIONAL_COUNTS[model]])
    action_files = []
    for index, plan in enumerate(action_candidates):
        document = plan_document(
            model=model,
            plan_id=f"{model}_action_only_{index:02d}",
            plan=plan,
            shapes=shapes,
            pack_dir=pack_dir,
            budget=budget,
            inputs=inputs,
            extra_meta={
                "baseline": "action_only_allocator",
                "candidate_index": index,
                "allocator_signal": "normalized per-layer d_func weight only",
                "representation_similarity_used": False,
                "guards_enabled": True,
                "action_objective": action_objective(plan, weights),
                "functional_discrimination_preregistered": index in action_functional,
                "guard_protected_layers": sorted(protected),
            },
        )
        action_files.append(
            write_plan(out_dir / "action_only" / f"candidate_{index:02d}.plan.json", document)
        )
    plans["action_only"] = {
        "candidate_count": SEARCH_CANDIDATE_COUNT,
        "functional_discrimination_count": FUNCTIONAL_COUNTS[model],
        "functional_candidate_indices": action_functional,
        "development_representatives_after_functional_ranking": DEVELOPMENT_REPRESENTATIVES,
        "exact_allocator_objective": action_base_objective,
        "files": action_files,
    }

    if model == "gr00t":
        ablations: dict[str, Any] = {
            "cka_only": artifact(GR00T_CKA_ONLY),
            "cs_only": artifact(GR00T_CS_ONLY),
        }
        uniform_weights = {name: 1.0 for name in names}
        no_weights, no_weights_objective = selector.milp_binary_plan(
            shapes, filtered_scores, uniform_weights, budget, ROW_ROT, group=GROUP
        )
        if no_weights is None:
            no_weights, no_weights_objective = selector.greedy_plan(
                shapes, filtered_scores, uniform_weights, budget, ROW_ROT
            )
        no_guards, no_guards_objective = selector.milp_binary_plan(
            shapes, scores, weights, budget, ROW_ROT, group=GROUP
        )
        if no_guards is None:
            no_guards, no_guards_objective = selector.greedy_plan(
                shapes, scores, weights, budget, ROW_ROT
            )
        proxy_only = source_plan_from_topk(parent, shapes)
        for plan_id, plan, extra in (
            (
                "weights_uniform",
                no_weights,
                {
                    "ablation": "w_i_equals_1",
                    "action_weights_enabled": False,
                    "proxy_objective": no_weights_objective,
                },
            ),
            (
                "no_guards",
                no_guards,
                {
                    "ablation": "no_guards",
                    "guards_enabled": False,
                    "formal_failures": "NaN/crash/out-of-range count as failures",
                    "proxy_objective": no_guards_objective,
                },
            ),
            (
                "no_functional_adjudication",
                proxy_only,
                {
                    "ablation": "no_functional_discrimination",
                    "selection": "minimum canonical proxy TopK candidate",
                },
            ),
        ):
            document = plan_document(
                model=model,
                plan_id=f"gr00t_ablation_{plan_id}",
                plan=plan,
                shapes=shapes,
                pack_dir=pack_dir,
                budget=budget,
                inputs=inputs,
                extra_meta=extra,
            )
            ablations[plan_id] = write_plan(
                out_dir / "ablations" / f"{plan_id}.plan.json", document
            )
        plans["ablations"] = ablations

    return {
        "model": model,
        "candidate_layers": len(shapes),
        "final_plan": {
            **artifact(final_path),
            "total_bytes_recomputed": final_total,
            "quantized_layers": final_w4,
        },
        "uniform_w6_budget_bytes": budget,
        "fp16_bytes": sum(
            selector.layer_bytes_fp16(row["out"], row["in"], row["has_bias"])
            for row in shapes.values()
        ),
        "guard_thresholds": {"tau_rms": tau_rms, "tau_sat": tau_sat},
        "guard_filtered_layers": sorted(removed_names),
        "unavailable_or_guard_protected_layers": sorted(protected),
        "action_weight_log": weight_log,
        "plans": plans,
    }


def environment_artifacts() -> dict[str, Any]:
    paths = {
        "groot_conda": REPO_ROOT / "environments/groot_env.yml",
        "groot_requirements": REPO_ROOT / "environments/requirements_gr00t.txt",
        "openpi_pyproject": REPO_ROOT / "code/pi05/openpi/pyproject.toml",
        "openpi_lock": REPO_ROOT / "code/pi05/openpi/uv.lock",
    }
    return {name: artifact(path) for name, path in paths.items()}


def generate(root: Path) -> dict[str, Any]:
    if sha256_file(SELECTOR_PATH) != SELECTOR_SHA256:
        raise ValueError("frozen v8 selector SHA mismatch")
    models = {model: generate_model(model, root) for model in ("gr00t", "pi05")}
    manifest = {
        "schema_version": 1,
        "kind": "gdsq_vla_week1_preregistration",
        "created_date": "2026-08-23",
        "result_blind": True,
        "immutable_after_formal_rollout_start": True,
        "runtime_selector": {
            **artifact(SELECTOR_PATH),
            "expected_sha256": SELECTOR_SHA256,
            "selection_scope": "model_level_absolute_mechanism_gate",
            "decisions": {"gr00t": "baseline", "pi05": "ohb"},
            "allowed_outputs": ["baseline", "atm", "ohb"],
            "combined_atmohb_allowed": False,
            "uses_task_metadata": False,
            "uses_rollout_labels": False,
            "uses_runtime_success_feedback": False,
        },
        "protocol": {
            "checkpoint_and_scene_pairing": "identical within each model contrast",
            "action_noise": "sha256(task,env_seed,replan_index)/torch-cpu-normal-v1",
            "denoising_steps": 4,
            "execute_actions": 16,
            "activation_bits": 8,
            "activation_calibration": "static p99.9, 256 observations, 32x8",
            "storage_accounting": "theoretical tightly packed candidate weights/scales/rotation artifacts",
            "random_candidate_count": SEARCH_CANDIDATE_COUNT,
            "random_draw_seed": RANDOM_DRAW_SEED,
            "functional_subset_seed": FUNCTIONAL_SUBSET_SEED,
            "development_tasks": [
                "CoffeeSetupMug",
                "OpenCabinet",
                "OpenStandMixerHead",
                "PickPlaceDrawerToCounter",
            ],
            "heldout_feedback_forbidden": True,
            "no_guard_failure_policy": "NaN, crash, or out-of-range action is a formal failure",
            "statistics": {
                "estimand": "task-macro success rate",
                "bootstrap": "10,000 task-cluster resamples",
                "test": "paired task-level sign-flip",
                "multiplicity": "Holm over preregistered contrasts",
            },
        },
        "formal_scopes": {
            "selector": {
                "pi05": "50 tasks x 50 seeds",
                "gr00t": "reuse static 50x50 only after 256-observation bitwise equivalence; otherwise rerun 50x50",
            },
            "uniform_w6": {"gr00t": "50x50", "pi05": "50x50"},
            "search_matched_random": {
                "gr00t": "Primary14 x 50 seeds",
                "pi05": "held-out46 x 50 seeds",
            },
            "action_only": {
                "gr00t": "Primary14 x 50 seeds",
                "pi05": "held-out46 x 50 seeds",
            },
            "gr00t_ablations": "Primary14 x 50 seeds",
        },
        "models": models,
        "source": {
            "generator": artifact(Path(__file__)),
            "selector_authority": artifact(TOOLS_ROOT / "gr00t_select_plan.py"),
            "gr00t_evaluator": artifact(REPO_ROOT / "scripts/run_robocasa365_gr00t_eval.py"),
            "pi05_evaluator": artifact(REPO_ROOT / "scripts/run_robocasa365_pi05_eval.py"),
        },
        "environments": environment_artifacts(),
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    manifest_path = root / "preregistration.json"
    atomic_json(manifest_path, manifest)
    return {**manifest, "_manifest": artifact(manifest_path)}


def validate(root: Path) -> dict[str, Any]:
    manifest_path = root / "preregistration.json"
    manifest = read_json(manifest_path)
    errors: list[str] = []
    if manifest.get("result_blind") is not True:
        errors.append("preregistration is not result-blind")
    selector_row = manifest.get("runtime_selector") or {}
    if selector_row.get("sha256") != SELECTOR_SHA256:
        errors.append("selector SHA differs from frozen v8")
    checked = 0
    for model, row in (manifest.get("models") or {}).items():
        budget = float(row["uniform_w6_budget_bytes"])
        plan_groups = row["plans"]
        file_rows: list[dict[str, Any]] = [plan_groups["uniform_w6"]]
        file_rows += plan_groups["search_matched_random"]["files"]
        file_rows += plan_groups["action_only"]["files"]
        if model == "gr00t":
            file_rows += [
                value
                for key, value in plan_groups["ablations"].items()
                if key not in ("cka_only", "cs_only")
            ]
        for file_row in file_rows:
            path = Path(file_row["path"])
            if not path.is_file() or sha256_file(path) != file_row["sha256"]:
                errors.append(f"{model}: missing/drifted plan {path}")
                continue
            document = read_json(path)
            if float(document["total_bytes"]) > budget + 1e-3:
                errors.append(f"{model}: over-budget plan {path.name}")
            if document["meta"].get("frozen_before_formal_rollouts") is not True:
                errors.append(f"{model}: plan not frozen {path.name}")
            checked += 1
        random_group = plan_groups["search_matched_random"]
        masks = []
        for file_row in random_group["files"]:
            document = read_json(Path(file_row["path"]))
            masks.append(
                tuple(
                    sorted(
                        name
                        for name, value in document["layers"].items()
                        if not value.get("skip")
                    )
                )
            )
        if len(masks) != SEARCH_CANDIDATE_COUNT or len(set(masks)) != len(masks):
            errors.append(f"{model}: random masks are missing or duplicated")
        if len(random_group["functional_candidate_indices"]) != FUNCTIONAL_COUNTS[model]:
            errors.append(f"{model}: wrong functional discrimination count")
    return {
        "valid": not errors,
        "manifest": artifact(manifest_path),
        "plans_checked": checked,
        "errors": errors,
    }


def selftest() -> None:
    shapes = {
        f"L{index}": {"out": 512, "in": 512, "has_bias": False}
        for index in range(20)
    }
    reference = {
        name: {
            "bits": 4 if index < 12 else None,
            "group": GROUP,
            "skip": index >= 12,
        }
        for index, name in enumerate(shapes)
    }
    target = selector.plan_total_bytes(reference, shapes, ROW_ROT)
    masks = random_masks(
        shapes=shapes,
        protected={"L19"},
        n_w4=12,
        target_bytes=target,
        reference_plan=reference,
        count=8,
        seed=7,
    )
    assert len({tuple(name for name, row in plan.items() if not row["skip"]) for plan in masks}) == 8
    assert all(
        abs(selector.plan_total_bytes(plan, shapes, ROW_ROT) - target) / target
        <= BYTE_TOLERANCE
        for plan in masks
    )
    assert all(plan["L19"]["skip"] for plan in masks)
    print("[gdsq-week1-plans] selftest OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest()
        return
    root = Path(args.out_dir).expanduser().resolve()
    if args.validate_only:
        result = validate(root)
    else:
        if (root / "preregistration.json").exists():
            raise SystemExit(
                f"refusing to overwrite frozen preregistration: {root / 'preregistration.json'}"
            )
        generated = generate(root)
        result = validate(root)
        result["generated_manifest"] = generated["_manifest"]
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
