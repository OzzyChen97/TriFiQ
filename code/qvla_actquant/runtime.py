"""Environment-bound loader shared by the GR00T and OpenPI services."""

from __future__ import annotations

import os
from pathlib import Path

from torch import nn

from .core import apply_actquant_gguf_bundle, apply_qvla_pack


FALSE_VALUES = ("", "0", "false", "False")


def apply_reproduction_artifact_from_env(model: nn.Module, model_family: str) -> dict:
    method = os.environ.get("QVLA_ACTQUANT_METHOD", "").strip().lower()
    if not method:
        return {"enabled": False, "method": None}
    if method not in ("qvla", "actquant"):
        raise RuntimeError(f"unsupported QVLA_ACTQUANT_METHOD: {method!r}")
    # Fail closed if a reproduction row is accidentally combined with any
    # existing quantizer, selector, or correction.
    forbidden = [
        key
        for key, value in os.environ.items()
        if value not in FALSE_VALUES
        and (
            key.startswith("GR00T_DUQUANT_")
            or key.startswith("OPENPI_DUQUANT_")
            or key.startswith("GR00T_GPTQ")
            or key.startswith("OPENPI_OMEGA_")
            or key.startswith("GR00T_ATM_")
            or key.startswith("OPENPI_ATM_")
            or key.startswith("GR00T_OHB_")
            or key.startswith("OPENPI_OHB_")
            or key in ("OPENPI_ERRORFOLD_PATH", "QUANTVLA_ADAPTER_ONLY")
        )
    ]
    if forbidden:
        raise RuntimeError(f"QVLA/ActQuant row has forbidden mixed runtime variables: {forbidden}")
    artifact = Path(os.environ.get("QVLA_ACTQUANT_PACK", "")).expanduser()
    if not artifact.is_file():
        raise RuntimeError(f"QVLA/ActQuant artifact is missing: {artifact}")
    expected_sha = os.environ.get("QVLA_ACTQUANT_PACK_SHA256") or None
    if method == "qvla":
        return apply_qvla_pack(
            model,
            artifact,
            model_family=model_family,
            expected_sha256=expected_sha,
        )
    return apply_actquant_gguf_bundle(
        model,
        artifact,
        model_family=model_family,
        expected_manifest_sha256=expected_sha,
        actquant_root=os.environ.get("ACTQUANT_ROOT") or None,
    )
