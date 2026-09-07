"""Environment-bound DA-PTQ artifact loader."""

from __future__ import annotations

import os
from pathlib import Path

from torch import nn

from .core import apply_artifact


FALSE_VALUES = ("", "0", "false", "False")


def apply_daptq_from_env(model: nn.Module, model_family: str) -> dict:
    if os.environ.get("DAPTQ_ENABLE", "0") in FALSE_VALUES:
        return {"enabled": False, "method": None}
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
            or key.startswith("QVLA_ACTQUANT_")
            or key in ("OPENPI_ERRORFOLD_PATH", "QUANTVLA_ADAPTER_ONLY")
        )
    ]
    if forbidden:
        raise RuntimeError(f"DA-PTQ cannot be mixed with other quantizers: {forbidden}")
    manifest = Path(os.environ.get("DAPTQ_MANIFEST", "")).expanduser()
    if not manifest.is_file():
        raise RuntimeError(f"DA-PTQ manifest is missing: {manifest}")
    return apply_artifact(
        model,
        manifest,
        model_family=model_family,
        expected_sha256=os.environ.get("DAPTQ_MANIFEST_SHA256") or None,
    )
