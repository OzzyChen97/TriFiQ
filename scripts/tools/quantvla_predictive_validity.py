#!/usr/bin/env python3
"""Frozen mechanics for the GR00T D-PAC predictive-validity experiment."""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import threading
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "scripts/quantvla_dpac_predictive_protocol.json"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "gr00t-dpac-predictive-validity-v1":
        raise ValueError(f"unexpected predictive-validity protocol: {value.get('protocol_id')}")
    return value


PROTOCOL = _load_protocol()
PROTOCOL_SHA256 = canonical_hash(PROTOCOL)
PROTOCOL_FILE_SHA256 = sha256_file(PROTOCOL_PATH)


def protocol_attestation() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_file": str(PROTOCOL_PATH),
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
    }


def require_protocol_attestation(value: Mapping[str, Any], *, source: str) -> None:
    actual = value.get("predictive_validity_protocol") or {}
    expected = protocol_attestation()
    drift = {
        key: (actual.get(key), wanted)
        for key, wanted in expected.items()
        if actual.get(key) != wanted
    }
    if drift:
        raise ValueError(f"{source}: predictive-validity protocol drift: {drift}")


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def atomic_json(path: str | Path, payload: Mapping[str, Any], *, immutable: bool = False) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if immutable and output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"immutable predictive-validity artifact drift: {output}")
        return
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def is_w4(row: Mapping[str, Any]) -> bool:
    return not bool(row.get("skip", False)) and int(row.get("bits", 0) or 0) == 4


def _set_state(row: dict[str, Any], *, fp16: bool, reason: str) -> None:
    if fp16:
        row.update({"bits": None, "skip": True, "reason": reason})
    else:
        row.update({"bits": 4, "group": 64, "skip": False, "reason": reason})
    for forbidden in ("errorfold", "blocksoftfold", "atm", "ohb"):
        row.pop(forbidden, None)


def generate_masks(
    protected: set[str],
    byte_rows: Mapping[str, Mapping[str, int]],
    *,
    seed: int,
    swap_counts: Sequence[int],
    per_count: int,
) -> list[dict[str, Any]]:
    """Generate metric- and success-independent exact-byte matched swaps."""
    if set(protected) - set(byte_rows):
        raise ValueError("reference mask contains layers outside the byte inventory")
    by_group: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: {"fp16": [], "w4": []}
    )
    for name, row in byte_rows.items():
        key = int(row["extra_fp16_bytes"])
        by_group[key]["fp16" if name in protected else "w4"].append(name)
    for value in by_group.values():
        value["fp16"].sort()
        value["w4"].sort()

    rng = random.Random(int(seed))
    seen = {frozenset(protected)}
    result: list[dict[str, Any]] = []
    reference_extra = sum(int(byte_rows[name]["extra_fp16_bytes"]) for name in protected)
    for swap_count in swap_counts:
        made = 0
        attempts = 0
        while made < per_count:
            attempts += 1
            if attempts > 100_000:
                raise RuntimeError(f"unable to generate unique k={swap_count} masks")
            removed = rng.sample(sorted(protected), int(swap_count))
            removed_by_group: dict[int, int] = defaultdict(int)
            for name in removed:
                removed_by_group[int(byte_rows[name]["extra_fp16_bytes"])] += 1
            added: list[str] = []
            for group, count in sorted(removed_by_group.items()):
                pool = by_group[group]["w4"]
                if count > len(pool):
                    raise ValueError(f"byte group {group} cannot supply {count} replacements")
                added.extend(rng.sample(pool, count))
            mask = frozenset((protected - set(removed)) | set(added))
            if mask in seen:
                continue
            if len(mask) != len(protected):
                raise AssertionError("matched swap changed the FP16 layer count")
            candidate_extra = sum(
                int(byte_rows[name]["extra_fp16_bytes"]) for name in mask
            )
            if candidate_extra != reference_extra:
                raise AssertionError("matched swap changed the exact byte budget")
            seen.add(mask)
            identifier = f"k{int(swap_count):02d}_m{made:02d}"
            result.append(
                {
                    "candidate_id": identifier,
                    "swap_count": int(swap_count),
                    "hamming_distance": 2 * int(swap_count),
                    "removed_fp16_layers": sorted(removed),
                    "added_fp16_layers": sorted(added),
                    "protected_layers": sorted(mask),
                }
            )
            made += 1
    expected = len(swap_counts) * per_count
    if len(result) != expected or len({row["candidate_id"] for row in result}) != expected:
        raise AssertionError("predictive mask inventory is incomplete or duplicated")
    return result


def physical_action_mse_summary(
    reference: Any,
    candidate: Any,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Raw physical-action MSE with the same task-seed sequence unit as D-PAC."""
    from quantvla_metric_protocol import canonical_physical_actions

    ref = canonical_physical_actions(reference, "mse_reference").to(torch.float64)
    quant = canonical_physical_actions(candidate, "mse_candidate").to(torch.float64)
    if ref.ndim != 3 or quant.shape != ref.shape or ref.shape[0] != len(records):
        raise ValueError("physical MSE expects paired (observation,16,12) chunks")
    squared = (quant - ref).square()
    groups: dict[tuple[str, int, int | None], list[int]] = defaultdict(list)
    has_replans = bool(records) and all("replan" in record for record in records)
    for index, record in enumerate(records):
        singleton = None if has_replans else index
        groups[(str(record.get("task")), int(record.get("seed", -1)), singleton)].append(index)
    sequences = []
    by_task: dict[str, list[float]] = defaultdict(list)
    for (task, seed, _singleton), positions in sorted(groups.items()):
        value = float(squared[positions].mean())
        sequences.append(
            {
                "task": task,
                "seed": seed,
                "record_indices": positions,
                "mse_sequence": value,
            }
        )
        by_task[task].append(value)
    per_task = {task: float(np.mean(values)) for task, values in sorted(by_task.items())}
    task_macro = float(np.mean(list(per_task.values()))) if per_task else 0.0
    return {
        "mse": task_macro,
        "global_elementwise_mse": float(squared.mean()) if squared.numel() else 0.0,
        "per_sequence": [row["mse_sequence"] for row in sequences],
        "sequences": sequences,
        "per_task": per_task,
        "n_sequences": len(sequences),
        "selection_sample_unit": "paired_task_seed_sequence",
        "aggregation": "unweighted_task_macro_of_sequence_raw_physical_action_mse",
        "formula_id": "raw_physical_action_mse_execute16_deployed12_v1",
    }


def combine_mse_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot combine an empty MSE sequence inventory")
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["mse_sequence"])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("invalid physical-action MSE sequence")
        by_task[str(row["task"])].append(value)
    per_task = {task: float(np.mean(values)) for task, values in sorted(by_task.items())}
    return {
        "mse": float(np.mean(list(per_task.values()))),
        "per_sequence": [float(row["mse_sequence"]) for row in rows],
        "sequences": [dict(row) for row in rows],
        "per_task": per_task,
        "n_sequences": len(rows),
        "selection_sample_unit": "paired_task_seed_sequence",
        "aggregation": "unweighted_task_macro_of_sequence_raw_physical_action_mse",
        "formula_id": "raw_physical_action_mse_execute16_deployed12_v1",
        "combined_across_splits": True,
    }


def validate_predictive_response(
    response: Any,
    *,
    candidate_id: str,
    plan_sha256: str,
    library_sha256: str,
) -> dict[str, Any]:
    expected = {
        "candidate_id": candidate_id,
        "plan_sha256": plan_sha256,
        "library_sha256": library_sha256,
        "predictive_validity_protocol": protocol_attestation(),
    }
    if not isinstance(response, Mapping):
        raise ValueError("predictive mask response is missing")
    drift = {
        key: (response.get(key), value)
        for key, value in expected.items()
        if response.get(key) != value
    }
    if drift or response.get("enabled") is not True or response.get("evaluation_only") is not True:
        raise ValueError(f"predictive mask response drift: {drift or response}")
    return dict(response)


class PredictiveMaskRuntime:
    """Evaluation-only explicit mask router for one serialized GR00T server."""

    def __init__(self, manifest_path: str | Path, named_layers: Mapping[str, Any]):
        from quantvla_cross_model_protocol import validate_quant_plan

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        require_protocol_attestation(self.manifest, source=str(self.manifest_path))
        if self.manifest.get("kind") != "gr00t_dpac_predictive_mask_library":
            raise ValueError("predictive mask runtime received the wrong manifest kind")
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.layers = dict(named_layers)
        if not self.layers:
            raise ValueError("predictive mask runtime has no switchable layers")
        candidates: dict[str, dict[str, Any]] = {}
        for row in self.manifest.get("candidates") or []:
            identifier = str(row["candidate_id"])
            path = Path(row["path"]).expanduser().resolve()
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"predictive mask plan drift: {identifier}")
            plan = json.loads(path.read_text(encoding="utf-8"))
            validate_quant_plan(plan, model="gr00t", source=str(path))
            protected = set(plan.get("protected_layers") or [])
            if set(plan.get("layers") or {}) != set(self.layers):
                raise ValueError(f"{identifier}: live/plan inventory mismatch")
            if protected - set(self.layers):
                raise ValueError(f"{identifier}: unknown FP16 bypass layers")
            candidates[identifier] = {
                "candidate_id": identifier,
                "plan_sha256": row["sha256"],
                "protected_layers": protected,
                "swap_count": int(row["swap_count"]),
            }
        expected = int(PROTOCOL["mask_library"]["candidate_count"])
        if len(candidates) != expected:
            raise ValueError(f"predictive mask runtime inventory {len(candidates)} != {expected}")
        self.candidates = candidates
        self._lock = threading.RLock()
        self._current: str | None = None

    def activate(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(metadata, Mapping):
            raise ValueError("predictive mask request lacks eval_metadata")
        identifier = metadata.get("predictive_mask_id")
        if not isinstance(identifier, str) or identifier not in self.candidates:
            raise ValueError(f"unknown or missing predictive_mask_id: {identifier!r}")
        row = self.candidates[identifier]
        with self._lock:
            protected = row["protected_layers"]
            for name, layer in self.layers.items():
                layer._outputimpact_fp16 = name in protected
            self._current = identifier
        return self.response_payload(identifier)

    def response_payload(self, identifier: str | None = None) -> dict[str, Any]:
        chosen = identifier or self._current
        if chosen is None or chosen not in self.candidates:
            raise ValueError("predictive mask runtime has no active candidate")
        row = self.candidates[chosen]
        return {
            "enabled": True,
            "evaluation_only": True,
            "candidate_id": chosen,
            "plan_sha256": row["plan_sha256"],
            "library_sha256": self.manifest_sha256,
            "predictive_validity_protocol": protocol_attestation(),
            "swap_count": row["swap_count"],
            "uses_task_or_success_for_selection": False,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "evaluation_only": True,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "candidate_count": len(self.candidates),
            "candidate_ids": sorted(self.candidates),
            "candidate_plan_sha256s": {
                key: self.candidates[key]["plan_sha256"]
                for key in sorted(self.candidates)
            },
            "predictive_validity_protocol": protocol_attestation(),
            "explicit_request_mask_only": True,
            "uses_task_or_success_for_selection": False,
        }


def configure_predictive_mask_runtime(policy: Any) -> PredictiveMaskRuntime | None:
    path = os.environ.get("GR00T_PREDICTIVE_MASK_MANIFEST")
    if not path:
        return None
    from gr00t.quantization.duquant_layers import DuQuantLinear
    from quantvla_outputimpact import install_fp16_bypass

    layers = install_fp16_bypass(policy.model.named_modules(), module_type=DuQuantLinear)
    runtime = PredictiveMaskRuntime(path, layers)
    setattr(policy.model, "_gr00t_predictive_mask_runtime", runtime)
    return runtime


def get_predictive_mask_runtime(policy: Any) -> PredictiveMaskRuntime | None:
    return getattr(policy.model, "_gr00t_predictive_mask_runtime", None)
