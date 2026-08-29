"""OpenPI (pi0.5) DuQuant W4A8 + ATM/OHB quantization (QuantVLA method).

Ported from QuantVLA (https://github.com/AIoT-MLSys-Lab/QuantVLA) gr00t/quantization
for OpenPI pi0.5 model quantization. Independent from the GR00T codebase.
"""

from .duquant_layers import (
    DuQuantConfig,
    DuQuantLinear,
    enable_duquant_if_configured,
    finalize_real_quant,
    iter_duquant_layers,
    load_act_scales,
    load_hessian_w4,
    apply_errorfold,
    save_act_scales,
    select_targets,
    static_scales_ready,
    wrap_duquant,
)
from .plan import QuantPlan, load_quant_plan, sha256_file, validate_plan_inventory
from .kernel_scores import LayerScoreBank, guard_metrics, reference_guard_stats
from .atm_pi05 import (
    enable_pi05_atm_if_configured,
    register_atm_capture,
    register_atm_logits_capture,
    register_errorfold_head_capture,
    reset_errorfold_attention_folds,
    register_ohb_output_capture,
    register_ohb_perhead_capture,
)
from .atm_runtime_selector import (
    configure_runtime_selector_from_env,
    current_decision,
    get_runtime_selector,
    runtime_selector_context,
    runtime_selector_enabled,
)

__all__ = [
    "DuQuantConfig",
    "DuQuantLinear",
    "enable_duquant_if_configured",
    "finalize_real_quant",
    "iter_duquant_layers",
    "load_act_scales",
    "load_hessian_w4",
    "apply_errorfold",
    "save_act_scales",
    "select_targets",
    "static_scales_ready",
    "wrap_duquant",
    "QuantPlan",
    "load_quant_plan",
    "sha256_file",
    "validate_plan_inventory",
    "LayerScoreBank",
    "guard_metrics",
    "reference_guard_stats",
    "enable_pi05_atm_if_configured",
    "register_atm_capture",
    "register_atm_logits_capture",
    "register_errorfold_head_capture",
    "reset_errorfold_attention_folds",
    "register_ohb_output_capture",
    "register_ohb_perhead_capture",
    "configure_runtime_selector_from_env",
    "current_decision",
    "get_runtime_selector",
    "runtime_selector_context",
    "runtime_selector_enabled",
]
