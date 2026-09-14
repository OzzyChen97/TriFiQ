#!/usr/bin/env python3
"""Load and validate the public DyPAC-VLA Method contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "scripts" / "dypac_vla_protocol.json"
EXPECTED_METHOD_ID = "dypac_vla_paper_method_v1"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if value.get("method_id") != EXPECTED_METHOD_ID:
        raise ValueError(f"unexpected DyPAC-VLA method id: {value.get('method_id')!r}")
    return value


PROTOCOL = load_protocol()
PROTOCOL_SHA256 = canonical_hash(PROTOCOL)
RIPA_ALPHA = (
    float(PROTOCOL["ripa"]["alpha"]["numerator"])
    / float(PROTOCOL["ripa"]["alpha"]["denominator"])
)
GROUP_SIZE = int(PROTOCOL["weight_quantization"]["group_size"])


def protocol_attestation() -> dict[str, Any]:
    return {
        "method_id": PROTOCOL["method_id"],
        "protocol_sha256": PROTOCOL_SHA256,
    }


def require_protocol_attestation(value: Mapping[str, Any], *, source: str) -> None:
    actual = value.get("dypac_vla_protocol") or value
    expected = protocol_attestation()
    mismatches = {
        key: (actual.get(key), expected_value)
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{source}: DyPAC-VLA protocol drift: {mismatches}")


def selftest() -> None:
    assert PROTOCOL["paper_title"].startswith("DyPAC-VLA: Representation-to-Action")
    assert math.isclose(RIPA_ALPHA, 16.0 / 17.0, rel_tol=0.0, abs_tol=0.0)
    assert GROUP_SIZE == 64
    assert PROTOCOL["weight_mask"]["layer_count_proxy_allowed"] is False
    assert PROTOCOL["d_pac"]["unexecuted_chunk_suffix_used"] is False
    assert PROTOCOL["d_pac"]["tie_tolerance"] == 1e-10
    assert PROTOCOL["d_pac"]["controlled_libero_bounds"] == {
        "delta_R": 0.05,
        "delta_A": 0.05,
    }
    assert PROTOCOL["dyrange_a8"]["recompute"] == "every_forward_call"
    assert PROTOCOL["dyrange_a8"]["scale_granularity"] == "input_channel"
    assert PROTOCOL["dyrange_a8"]["runtime_selector"] is False
    print(f"[dypac-vla-protocol] selftest OK {PROTOCOL_SHA256}")


if __name__ == "__main__":
    selftest()