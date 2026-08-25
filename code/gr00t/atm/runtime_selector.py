"""Frozen calibration-gated correction selector for GR00T serving.

The v8 paper selector makes one decision per model from calibration statistics.
Request metadata is retained only for result attestation; it does not influence
the selected correction.  Legacy task-mapped selector files remain supported
for old diagnostic artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import contextvars
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

RUNTIME_SELECTOR_PATH_ENV = "GR00T_RUNTIME_SELECTOR_PATH"
RUNTIME_SELECTOR_MODEL_ENV = "GR00T_RUNTIME_SELECTOR_MODEL"
RUNTIME_SELECTOR_STRICT_ENV = "GR00T_RUNTIME_SELECTOR_STRICT"

VARIANT_FLAGS = {
    "baseline": (False, False),
    "atm": (True, False),
    "ohb": (False, True),
    "atmohb": (True, True),
}


@dataclass(frozen=True)
class RuntimeSelectorDecision:
    enabled: bool
    task_name: str | None
    selected_variant: str
    selected_config_id: str
    atm_enabled: bool
    ohb_enabled: bool
    selector_sha256: str | None
    selector_rule_name: str | None
    model_id: str | None

    def response_payload(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeSelector:
    def __init__(
        self,
        *,
        selector_path: Path,
        selector: dict[str, Any],
        model_id: str,
        strict: bool = True,
    ) -> None:
        self.selector_path = selector_path.resolve()
        self.selector = selector
        self.model_id = model_id
        self.strict = strict
        raw = self.selector_path.read_bytes()
        self.selector_sha256 = hashlib.sha256(raw).hexdigest()
        self.rule_name = selector.get("rule_name")
        fit_metadata = selector.get("fit_metadata") or {}
        self.selection_scope = (
            selector.get("selection_scope")
            or fit_metadata.get("selection_scope")
            or "task_mapping"
        )
        model = (selector.get("models") or {}).get(model_id)
        if not isinstance(model, dict):
            raise ValueError(f"selector has no model block for {model_id!r}")
        self.variant_config_ids = dict(model.get("variant_config_ids") or {})
        if not self.variant_config_ids:
            raise ValueError(f"selector model {model_id!r} has no variant_config_ids")
        self.tasks: dict[str, dict[str, Any]] = {}
        for task, decision in sorted((model.get("tasks") or {}).items()):
            variant = decision.get("selected_variant")
            if variant not in VARIANT_FLAGS:
                raise ValueError(f"{model_id}/{task}: invalid selected_variant={variant!r}")
            config_id = decision.get("selected_config_id") or self.variant_config_ids.get(variant)
            if not config_id:
                raise ValueError(f"{model_id}/{task}: missing selected_config_id")
            self.tasks[str(task)] = {
                "selected_variant": str(variant),
                "selected_config_id": str(config_id),
            }
        if not self.tasks:
            raise ValueError(f"selector model {model_id!r} has no task decisions")
        self.uses_task_metadata = self.selection_scope != "model_level_absolute_mechanism_gate"
        self.model_decision: dict[str, str] | None = None
        if not self.uses_task_metadata:
            decisions = {
                (row["selected_variant"], row["selected_config_id"])
                for row in self.tasks.values()
            }
            if len(decisions) != 1:
                raise ValueError(
                    f"{model_id}: model-level selector contains task-varying decisions"
                )
            variant, config_id = next(iter(decisions))
            if variant == "atmohb":
                raise ValueError("model-level paper selector forbids combined ATM+OHB")
            model_fit = model.get("fit") or {}
            if model_fit.get("task_metadata_used_for_variant_selection") is not False:
                raise ValueError(
                    f"{model_id}: model-level selector lacks no-task-metadata attestation"
                )
            self.model_decision = {
                "selected_variant": variant,
                "selected_config_id": config_id,
            }

    @classmethod
    def from_env(cls) -> "RuntimeSelector | None":
        path_text = os.environ.get(RUNTIME_SELECTOR_PATH_ENV)
        if not path_text:
            return None
        path = Path(path_text).expanduser().resolve()
        selector = json.loads(path.read_text(encoding="utf-8"))
        model_id = os.environ.get(RUNTIME_SELECTOR_MODEL_ENV, "gr00t")
        strict = os.environ.get(RUNTIME_SELECTOR_STRICT_ENV, "1") not in (
            "0", "false", "False", ""
        )
        return cls(selector_path=path, selector=selector, model_id=model_id, strict=strict)

    def metadata(self) -> dict[str, Any]:
        mapping = {
            task: {
                **decision,
                "atm_enabled": VARIANT_FLAGS[decision["selected_variant"]][0],
                "ohb_enabled": VARIANT_FLAGS[decision["selected_variant"]][1],
            }
            for task, decision in sorted(self.tasks.items())
        }
        encoded_mapping = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return {
            "enabled": True,
            "kind": "atmohb_runtime_selector",
            "model_id": self.model_id,
            "selector_path": str(self.selector_path),
            "selector_sha256": self.selector_sha256,
            "rule_name": self.rule_name,
            "strict": self.strict,
            "selection_scope": self.selection_scope,
            "uses_task_metadata_for_selection": self.uses_task_metadata,
            "model_decision": self.model_decision,
            "variant_config_ids": self.variant_config_ids,
            "task_count": len(mapping),
            "task_mapping_sha256": hashlib.sha256(encoded_mapping).hexdigest(),
            "task_mapping": mapping,
        }

    def select(self, metadata: dict[str, Any] | None) -> RuntimeSelectorDecision:
        metadata = metadata or {}
        task_name = (
            metadata.get("task_name")
            or metadata.get("task")
            or metadata.get("env_name")
            or metadata.get("task_id")
        )
        task_name = str(task_name) if task_name is not None else None
        if self.model_decision is not None:
            variant = self.model_decision["selected_variant"]
            config_id = self.model_decision["selected_config_id"]
        elif not task_name or task_name not in self.tasks:
            if self.strict:
                raise ValueError(f"runtime selector missing decision for task={task_name!r}")
            variant = "baseline"
            config_id = self.variant_config_ids.get("baseline", "baseline")
        else:
            decision = self.tasks[task_name]
            variant = decision["selected_variant"]
            config_id = decision["selected_config_id"]
        atm_enabled, ohb_enabled = VARIANT_FLAGS[variant]
        return RuntimeSelectorDecision(
            enabled=True,
            task_name=task_name,
            selected_variant=variant,
            selected_config_id=config_id,
            atm_enabled=atm_enabled,
            ohb_enabled=ohb_enabled,
            selector_sha256=self.selector_sha256,
            selector_rule_name=self.rule_name,
            model_id=self.model_id,
        )


_CURRENT_DECISION: contextvars.ContextVar[RuntimeSelectorDecision | None] = contextvars.ContextVar(
    "gr00t_runtime_selector_decision", default=None
)
_RUNTIME_SELECTOR: RuntimeSelector | None = None


def configure_runtime_selector_from_env() -> RuntimeSelector | None:
    global _RUNTIME_SELECTOR
    _RUNTIME_SELECTOR = RuntimeSelector.from_env()
    return _RUNTIME_SELECTOR


def set_runtime_selector(selector: RuntimeSelector | None) -> None:
    global _RUNTIME_SELECTOR
    _RUNTIME_SELECTOR = selector


def get_runtime_selector() -> RuntimeSelector | None:
    return _RUNTIME_SELECTOR


def runtime_selector_enabled() -> bool:
    return _RUNTIME_SELECTOR is not None


def current_decision() -> RuntimeSelectorDecision | None:
    return _CURRENT_DECISION.get()


@contextmanager
def runtime_selector_context(metadata: dict[str, Any] | None) -> Iterator[RuntimeSelectorDecision | None]:
    selector = get_runtime_selector()
    if selector is None:
        yield None
        return
    decision = selector.select(metadata)
    token = _CURRENT_DECISION.set(decision)
    try:
        yield decision
    finally:
        _CURRENT_DECISION.reset(token)


def atm_enabled_for_current_request() -> bool:
    decision = current_decision()
    if decision is not None:
        return bool(decision.atm_enabled)
    if runtime_selector_enabled():
        return False
    return True


def ohb_enabled_for_current_request() -> bool:
    decision = current_decision()
    if decision is not None:
        return bool(decision.ohb_enabled)
    if runtime_selector_enabled():
        return False
    return True
