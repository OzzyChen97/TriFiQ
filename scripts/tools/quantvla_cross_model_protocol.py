#!/usr/bin/env python3
"""Single source of truth for GR00T N1.5 / pi0.5 adapter-only comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "scripts/quantvla_cross_model_protocol.json"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if value.get("protocol_id") != "quantvla-gr00t-pi05-errorfold-v3":
        raise ValueError(f"unexpected cross-model protocol: {value.get('protocol_id')!r}")
    return value


PROTOCOL = load_protocol()
PROTOCOL_SHA256 = canonical_hash(PROTOCOL)
PROTOCOL_FILE_SHA256 = sha256_file(PROTOCOL_PATH)
MODELS = tuple(PROTOCOL["models"])


def protocol_artifact(name: str, *, verify: bool = True) -> Path:
    row = PROTOCOL["data"][name]
    path = (REPO_ROOT / row["path"]).resolve()
    if verify:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ValueError(f"{name} SHA256 drift: {actual} != {row['sha256']}")
    return path


def protocol_attestation() -> dict[str, Any]:
    """The exact block that every scorer, artifact and runtime must carry."""
    return {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_file": str(PROTOCOL_PATH),
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "only_allowed_model_difference": "model_adapter",
    }


def require_protocol_attestation(value: Mapping[str, Any], *, source: str) -> None:
    expected = protocol_attestation()
    actual = value.get("cross_model_protocol") or {}
    mismatches = {
        key: (actual.get(key), expected_value)
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{source}: cross-model protocol drift: {mismatches}")


def adapter_attestation(model: str, *, native_action_horizon: int) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"unknown model adapter: {model!r}")
    return {
        "model": model,
        "boundary": "model_adapter",
        "native_action_horizon": int(native_action_horizon),
        "canonical_action_horizon": int(
            PROTOCOL["metrics"]["canonical_action"]["horizon"]
        ),
        "canonical_action_dimension": int(
            PROTOCOL["metrics"]["canonical_action"]["dimension"]
        ),
        "allowed_fields": list(
            PROTOCOL["only_allowed_model_difference"]["fields"]
        ),
    }


def closed_loop_row_protocol() -> dict[str, Any]:
    row = PROTOCOL["closed_loop"]
    return {
        "split": row["split"],
        "action_horizon": int(row["canonical_action_horizon"]),
        "n_action_steps": int(row["n_action_steps"]),
        "replan_steps": int(row["replan_steps"]),
        "flow_steps": int(row["flow_steps"]),
        "paired_action_noise": bool(row["paired_action_noise"]),
        "action_noise_protocol": row["action_noise_protocol"],
        "environment_seed_protocol": row["environment_seed_protocol"],
        "fresh_environment": bool(row["fresh_environment_per_episode"]),
        "official_task_horizon": bool(row["official_task_horizon"]),
        "render": bool(row["render"]),
        "cross_model_protocol_sha256": PROTOCOL_SHA256,
    }


def closed_loop_runtime_protocol() -> dict[str, Any]:
    """Model-agnostic server contract; native shapes live in the adapter block."""
    row = PROTOCOL["closed_loop"]
    return {
        "action_horizon": int(row["canonical_action_horizon"]),
        "n_action_steps": int(row["n_action_steps"]),
        "replan_steps": int(row["replan_steps"]),
        "flow_steps": int(row["flow_steps"]),
        "split": row["split"],
        "fresh_environment_per_episode": bool(row["fresh_environment_per_episode"]),
        "official_task_horizon": bool(row["official_task_horizon"]),
        "render": bool(row["render"]),
        "paired_noise": row["action_noise_protocol"],
        "cross_model_protocol_sha256": PROTOCOL_SHA256,
    }


def validate_closed_loop_row(row: Mapping[str, Any], *, source: str) -> None:
    expected = closed_loop_row_protocol()
    mismatches = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{source}: closed-loop protocol drift: {mismatches}")


def validate_softfold_runtime(runtime: Mapping[str, Any], *, source: str) -> None:
    expected = PROTOCOL["deployment"]
    selector = runtime.get("runtime_selector") or {}
    correction = runtime.get("errorfold") or runtime.get("atm_ohb") or runtime
    checks = {
        "selector_free": not bool(selector.get("enabled", False)),
        "atm_application": correction.get("atm_application")
        == expected["atm_application"],
        "errorfold_application": correction.get("errorfold_application")
        == expected["errorfold_application"],
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{source}: deployment protocol drift: {failed}")


def validate_quant_plan(
    value: Mapping[str, Any], *, model: str, source: str
) -> dict[str, Any]:
    """Enforce the shared QuantVLA-budget selection rule after adapter binding.

    Layer names and target counts are architecture-specific adapter output.  The
    decision applied to that inventory is not: every bound target must be W4,
    with no FP16 retention or mixed-bit choice.
    """
    if model not in MODELS:
        raise ValueError(f"{source}: unknown model adapter {model!r}")
    layers = value.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError(f"{source}: quant plan has no adapter-bound layers")
    invalid: list[tuple[str, Any, Any, Any]] = []
    for name, raw in layers.items():
        row = raw if isinstance(raw, Mapping) else {}
        bits = row.get("bits")
        group = row.get("group", row.get("block_in", 64))
        skip = row.get("skip", not bool(bits))
        if bool(skip) or int(bits or 0) != 4 or int(group or 0) != 64:
            invalid.append((str(name), bits, group, skip))
    if invalid:
        raise ValueError(
            f"{source}: adapter-only selection requires every target W4/group64; "
            f"invalid={invalid[:5]} (total {len(invalid)})"
        )
    return {
        "model": model,
        "rule": PROTOCOL["quantization_selection"]["rule"],
        "budget_reference": PROTOCOL["quantization_selection"]["budget_reference"],
        "target_layers": len(layers),
        "quantized_w4_layers": len(layers),
        "retained_fp16_target_layers": 0,
        "protocol_sha256": PROTOCOL_SHA256,
    }


def selftest() -> None:
    assert set(MODELS) == {"gr00t", "pi05"}
    assert PROTOCOL["metrics"]["canonical_action"]["space"].startswith("physical")
    assert PROTOCOL["metrics"]["d_pac"]["pi05_forecast_overlap"] is False
    assert PROTOCOL["quantization_selection"]["retained_fp16_target_layers"] == 0
    assert len(PROTOCOL["softfold"]["grid"]["gate_atm"]) == 9
    assert len(PROTOCOL["softfold"]["grid"]["gate_errorfold"]) == 9
    assert len(PROTOCOL["closed_loop"]["seeds"]) == 20
    assert sum(len(value) for value in PROTOCOL["closed_loop"]["tasks"].values()) == 15
    protocol_artifact("selection_buffer")
    protocol_artifact("calibration_buffer")
    print(f"[cross-model-protocol] selftest OK {PROTOCOL_SHA256}")


if __name__ == "__main__":
    selftest()
