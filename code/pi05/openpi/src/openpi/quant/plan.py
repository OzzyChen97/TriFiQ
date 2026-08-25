"""Immutable mixed-precision plan handling for the OpenPI QuantVLA port."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class QuantPlan:
    path: Path
    sha256: str
    layers: dict[str, dict[str, Any]]
    meta: dict[str, Any]

    @property
    def quantized_bits(self) -> dict[str, int]:
        selected: dict[str, int] = {}
        for name, entry in self.layers.items():
            bits = int(entry.get("bits", 0) or 0)
            if not bool(entry.get("skip", False)) and bits > 0:
                selected[name] = bits
        return selected

    @property
    def skipped_layers(self) -> set[str]:
        return set(self.layers) - set(self.quantized_bits)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_entry(name: str, raw: Any) -> dict[str, Any]:
    if isinstance(raw, bool):
        return {"bits": 4 if raw else 0, "skip": not raw, "group": 64}
    if isinstance(raw, int):
        return {"bits": int(raw), "skip": int(raw) <= 0, "group": 64}
    if not isinstance(raw, dict):
        raise ValueError(f"plan layer {name!r} must be an object or integer")
    entry = dict(raw)
    bits = int(entry.get("bits", 0) or 0)
    entry["bits"] = bits
    entry["skip"] = bool(entry.get("skip", bits <= 0))
    if not entry["skip"] and bits not in (4, 6, 8, 16):
        raise ValueError(f"plan layer {name!r} has unsupported bits={bits}")
    entry.setdefault("group", 64)
    return entry


def load_quant_plan(path: str | Path) -> QuantPlan:
    plan_path = Path(path).expanduser().resolve()
    with plan_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    layers_raw = payload.get("layers")
    if not isinstance(layers_raw, dict) or not layers_raw:
        raise ValueError(f"quant plan {plan_path} must contain a non-empty layers object")
    layers = {str(name): _normalize_entry(str(name), raw) for name, raw in layers_raw.items()}
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError(f"quant plan {plan_path} meta must be an object")
    return QuantPlan(
        path=plan_path,
        sha256=sha256_file(plan_path),
        layers=layers,
        meta=dict(meta),
    )


def validate_plan_inventory(
    plan: QuantPlan,
    candidate_names: list[str],
    *,
    require_complete: bool = True,
) -> None:
    candidates = set(candidate_names)
    planned = set(plan.layers)
    unknown = sorted(planned - candidates)
    missing = sorted(candidates - planned)
    if unknown:
        raise ValueError(f"quant plan contains {len(unknown)} unknown layers: {unknown[:5]}")
    if require_complete and missing:
        raise ValueError(f"quant plan is missing {len(missing)} candidate layers: {missing[:5]}")
    selected = set(plan.quantized_bits)
    if not selected:
        raise ValueError("quant plan selects zero quantized layers")


def plan_selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plan.json"
        path.write_text(
            json.dumps(
                {
                    "layers": {
                        "a": {"bits": 4, "skip": False, "group": 64},
                        "b": {"bits": 0, "skip": True, "group": 64},
                    },
                    "meta": {"adjudicated": True},
                }
            ),
            encoding="utf-8",
        )
        plan = load_quant_plan(path)
        assert plan.quantized_bits == {"a": 4}
        assert plan.skipped_layers == {"b"}
        validate_plan_inventory(plan, ["a", "b"])
        try:
            validate_plan_inventory(plan, ["a", "b", "c"])
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete inventory was not rejected")


if __name__ == "__main__":
    plan_selftest()
    print("[openpi.quant.plan] selftest OK")
