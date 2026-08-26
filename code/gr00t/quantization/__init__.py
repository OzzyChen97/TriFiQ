"""GR00T DuQuant W4A8 fake quantization module."""

from .duquant_layers import (
    DuQuantConfig,
    DuQuantLinear,
    enable_duquant_if_configured,
    finalize_real_quant,
    load_hessian_w4,
    apply_errorfold,
    select_targets,
    wrap_duquant,
)
from .kernel_scores import LayerScoreBank, pool_samples, selftest

__all__ = [
    "DuQuantConfig",
    "DuQuantLinear",
    "enable_duquant_if_configured",
    "finalize_real_quant",
    "load_hessian_w4",
    "apply_errorfold",
    "select_targets",
    "wrap_duquant",
    "LayerScoreBank",
    "pool_samples",
    "selftest",
]
