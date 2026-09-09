#!/usr/bin/env python3
"""Shared DyRange-A8 v5 amendment layered over the frozen v4 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from quantvla_cross_model_protocol import PROTOCOL_SHA256


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "scripts/quantvla_dynamic_a8_protocol.json"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
if PROTOCOL.get("method_id") != "quantvla-dpac-errorfold-dyrange-v5":
    raise ValueError("unexpected DyRange-A8 protocol id")
BASE_PROTOCOL_SHA256 = (PROTOCOL.get("base_cross_model_protocol") or {}).get(
    "protocol_sha256"
)
PROTOCOL_SHA256 = _canonical_hash(PROTOCOL)
PROTOCOL_FILE_SHA256 = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def _require_base_protocol() -> None:
    from quantvla_cross_model_protocol import PROTOCOL_SHA256 as active_base_sha256

    if BASE_PROTOCOL_SHA256 != active_base_sha256:
        raise ValueError("DyRange-A8 base protocol drift")


def protocol_attestation() -> dict[str, Any]:
    _require_base_protocol()
    return {
        "method_id": PROTOCOL["method_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_file": str(PROTOCOL_PATH),
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "base_protocol_sha256": PROTOCOL["base_cross_model_protocol"][
            "protocol_sha256"
        ],
        "only_allowed_model_difference": "model_adapter",
    }


def require_protocol_attestation(value: Mapping[str, Any], *, source: str) -> None:
    _require_base_protocol()
    actual = value.get("dynamic_a8_protocol") or value
    expected = protocol_attestation()
    mismatches = {
        key: (actual.get(key), expected_value)
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{source}: DyRange-A8 protocol drift: {mismatches}")


def validate_runtime(contract: Mapping[str, Any], *, source: str) -> None:
    _require_base_protocol()
    checks = {
        "activation_bits": int(contract.get("activation_bits", -1)) == 8,
        "dynamic": contract.get("static_activation_scales") is False,
        "policy": contract.get("calibration_policy")
        == "online_dynamic_per_forward_per_channel_amax",
        "selector_free": not bool(contract.get("runtime_selector", False)),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{source}: invalid DyRange-A8 runtime fields: {failed}")
