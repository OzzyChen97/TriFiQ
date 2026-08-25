"""ATM (attention temperature matching) + OHB (output head balancing) for OpenPI pi0.5.

Ported from QuantVLA gr00t/atm/dit_atm.py. pi0.5 attention (PaliGemma LLM and Gemma
action expert) goes through transformers.models.gemma.modeling_gemma.eager_attention_forward,
so we monkey-patch that function at runtime. No changes to transformers files.
"""

import json
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch import nn

from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import repeat_kv

from openpi.quant.atm_runtime_selector import (
    atm_enabled_for_current_request,
    ohb_enabled_for_current_request,
    runtime_selector_enabled,
)

ATM_ENABLE_ENV = "OPENPI_ATM_ENABLE"
ATM_ALPHA_ENV = "OPENPI_ATM_ALPHA_PATH"
ATM_SCOPE_ENV = "OPENPI_ATM_SCOPE"

OHB_ENABLE_ENV = "OPENPI_OHB_ENABLE"
OHB_SCOPE_ENV = "OPENPI_OHB_SCOPE"
OHB_FALLBACK_ENV = "OPENPI_OHB_FALLBACK"
ATM_STRICT_ENV = "OPENPI_ATM_STRICT"
ATM_APPLICATION_ENV = "OPENPI_ATM_APPLICATION"
OHB_EXPECT_MODE_ENV = "OPENPI_OHB_EXPECT_MODE"
OHB_APPLICATION_ENV = "OPENPI_OHB_APPLICATION"

_ATM_PATCH_FLAG = "_openpi_atm_processor_patched"

# Default scope: Gemma action expert (DiT) attention layers
EXPERT_SCOPE_PREFIX = "paligemma_with_expert.gemma_expert.model.layers."


def _is_dit_attention(name: str, module: nn.Module, scope: str = "expert") -> bool:
    """pi0.5 attention layers are GemmaAttention modules."""
    if not isinstance(module, modeling_gemma.GemmaAttention):
        return False
    if scope == "expert":
        return EXPERT_SCOPE_PREFIX in name
    if scope == "all":
        return True
    return scope in name


def _compute_logits_std(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_dim: int,
    *,
    return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float32
    scale = 1.0 / math.sqrt(max(float(head_dim), 1.0))
    logits = torch.matmul(query.to(dtype), key.to(dtype).transpose(-1, -2)) * scale

    if attention_mask is not None:
        # attention mask is additive, with large negative entries for masked positions
        valid = attention_mask >= -1e4
        # broadcast (batch, 1, seq, seq) -> (batch, heads, seq, seq)
        valid = valid.expand(-1, logits.shape[1], -1, -1)
    else:
        valid = torch.ones_like(logits, dtype=torch.bool)

    valid = valid.to(dtype)
    count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
    mean = (logits * valid).sum(dim=(-1, -2)) / count
    mean = mean.unsqueeze(-1).unsqueeze(-1)
    var = ((logits - mean) ** 2 * valid).sum(dim=(-1, -2)) / count
    std = torch.sqrt(var.clamp_min(1e-12))
    std = std.detach()
    if return_logits:
        return std, logits.detach()
    return std


def _compute_rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor.detach().to(torch.float32) ** 2) + 1e-12)


def _compute_rms_per_head(tensor: torch.Tensor) -> torch.Tensor:
    """RMS per head. tensor: (batch, heads, seq, head_dim) attention output before reshape."""
    t = tensor.detach().to(torch.float32)
    return torch.sqrt(torch.mean(t ** 2, dim=(0, 2, 3)) + 1e-12)  # (heads,)


def _patched_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # ---- ATM: capture per-head logits std before scaling ----
    capture_cb = getattr(module, "_atm_capture_callback", None)
    if capture_cb is not None:
        std = _compute_logits_std(query, key_states, attention_mask, module.head_dim)
        capture_cb(module, std)

    # ---- ATM: per-head alpha scaling ----
    alpha = getattr(module, "_atm_alpha_all", None)
    if alpha is not None and atm_enabled_for_current_request():
        alpha = alpha.to(dtype=query.dtype, device=query.device).view(1, -1, 1, 1)
        query = query * alpha

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)

    # ---- OHB: capture per-head output RMS + per-head beta (before reshape) ----
    ohb_perhead_capture_cb = getattr(module, "_atm_ohb_perhead_capture_callback", None)
    if ohb_perhead_capture_cb is not None:
        ohb_perhead_capture_cb(module, _compute_rms_per_head(attn_output))
    beta_perhead = getattr(module, "_ohb_beta_perhead", None)
    if beta_perhead is not None and ohb_enabled_for_current_request():
        # attn_output shape here: (batch, heads, seq, head_dim)
        beta_perhead = beta_perhead.to(dtype=attn_output.dtype, device=attn_output.device).view(1, -1, 1, 1)
        attn_output = attn_output * beta_perhead

    attn_output = attn_output.transpose(1, 2).contiguous()

    # Legacy scalar capture point. Paper-faithful scalar OHB is captured and
    # applied by an o_proj output hook below, at the residual interface.
    ohb_capture_cb = getattr(module, "_atm_ohb_capture_callback", None)
    if ohb_capture_cb is not None:
        ohb_capture_cb(module, _compute_rms(attn_output))

    return attn_output, attn_weights


def ensure_pi05_attention_patch(model: nn.Module, scope: str = "expert") -> None:
    """Monkey-patch modeling_gemma.eager_attention_forward (idempotent)."""
    if not getattr(modeling_gemma, _ATM_PATCH_FLAG, False):
        modeling_gemma.eager_attention_forward = _patched_eager_attention_forward
        setattr(modeling_gemma, _ATM_PATCH_FLAG, True)
    # Mark matched attention modules so we can find them later
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, _ATM_PATCH_FLAG, True)


def register_atm_capture(
    model: nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "expert",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, "_atm_capture_callback", lambda attn, std, layer=name: callback(layer, std))
            setattr(module, "_atm_capture_name", name)


def register_ohb_capture(
    model: nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "expert",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, "_atm_ohb_capture_callback", lambda attn, rms, layer=name: callback(layer, rms))
            setattr(module, "_atm_ohb_capture_name", name)


def register_ohb_perhead_capture(
    model: nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "expert",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_ohb_perhead_capture_callback",
                lambda attn, rms, layer=name: callback(layer, rms),
            )
            setattr(module, "_atm_ohb_perhead_capture_name", name)


def _ensure_ohb_output_hook(module: nn.Module, name: str) -> None:
    """Install one idempotent post-o_proj hook for scalar paper OHB."""
    if getattr(module, "_openpi_ohb_output_hook", None) is not None:
        return

    def hook(_projection: nn.Module, _inputs, output: torch.Tensor):
        capture_cb = getattr(module, "_atm_ohb_output_capture_callback", None)
        if capture_cb is not None:
            capture_cb(module, _compute_rms(output))
        beta = getattr(module, "_ohb_beta_scalar", None)
        if beta is not None and beta != 1.0 and ohb_enabled_for_current_request():
            return output * float(beta)
        return output

    handle = module.o_proj.register_forward_hook(hook)
    setattr(module, "_openpi_ohb_output_hook", handle)
    setattr(module, "_openpi_ohb_output_name", name)


def register_ohb_output_capture(
    model: nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "expert",
) -> None:
    """Capture one scalar RMS per layer after o_proj, before residual addition."""
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_ohb_output_capture_callback",
                lambda attn, rms, layer=name: callback(layer, rms),
            )
            _ensure_ohb_output_hook(module, name)


def clear_atm_capture(model: nn.Module) -> None:
    for _, module in model.named_modules():
        for attr in (
            "_atm_capture_callback",
            "_atm_capture_name",
            "_atm_ohb_capture_callback",
            "_atm_ohb_capture_name",
            "_atm_ohb_perhead_capture_callback",
            "_atm_ohb_perhead_capture_name",
            "_atm_ohb_output_capture_callback",
        ):
            if hasattr(module, attr):
                delattr(module, attr)


def _fold_atm_into_q_projection(module: nn.Module, alpha: torch.Tensor) -> None:
    """Fold per-head ATM into the FP16 action-expert q_proj weights."""
    if getattr(module, "_openpi_atm_q_weight_folded", False):
        raise RuntimeError("ATM q_proj weights were already folded")
    projection = module.q_proj
    weight = getattr(projection, "weight", None)
    if weight is None:
        raise TypeError("fold_q_weight requires an unwrapped q_proj weight")
    heads = int(module.config.num_attention_heads)
    head_dim = int(module.head_dim)
    if weight.shape[0] != heads * head_dim:
        raise ValueError(
            f"unexpected q_proj rows {weight.shape[0]} for {heads} heads x {head_dim} dimensions"
        )
    with torch.no_grad():
        scale = alpha.to(device=weight.device, dtype=weight.dtype).repeat_interleave(head_dim)
        weight.mul_(scale[:, None])
        bias = getattr(projection, "bias", None)
        if bias is not None:
            bias.mul_(scale)
    setattr(module, "_openpi_atm_q_weight_folded", True)


def _fold_ohb_into_o_projection(module: nn.Module, beta: float) -> None:
    """Fold scalar post-projection OHB into o_proj without a runtime operator."""
    if getattr(module, "_openpi_ohb_o_weight_folded", False):
        raise RuntimeError("OHB o_proj weights were already folded")
    projection = module.o_proj
    weight = getattr(projection, "weight", None)
    if weight is None:
        raise TypeError("fold_o_weight requires an o_proj weight")
    with torch.no_grad():
        weight.mul_(float(beta))
        bias = getattr(projection, "bias", None)
        if bias is not None:
            bias.mul_(float(beta))
    setattr(module, "_openpi_ohb_o_weight_folded", True)


def _fold_ohb_perhead_into_o_projection(module: nn.Module, beta: torch.Tensor) -> None:
    """Fold pre-projection per-head OHB into o_proj input columns."""
    if getattr(module, "_openpi_ohb_o_perhead_weight_folded", False):
        raise RuntimeError("OHB per-head o_proj weights were already folded")
    projection = module.o_proj
    weight = getattr(projection, "weight", None)
    if weight is None:
        raise TypeError("fold_o_weight_perhead requires an o_proj weight")
    heads = int(module.config.num_attention_heads)
    head_dim = int(module.head_dim)
    if beta.numel() != heads or weight.shape[1] != heads * head_dim:
        raise ValueError(
            f"unexpected o_proj shape={tuple(weight.shape)} for {heads} heads x {head_dim}, beta={beta.numel()}"
        )
    scale = beta.to(device=weight.device, dtype=weight.dtype).repeat_interleave(head_dim)
    with torch.no_grad():
        weight.mul_(scale[None, :])
    setattr(module, "_openpi_ohb_o_perhead_weight_folded", True)


def _require_uniform_selector_variant(expected_variant: str) -> None:
    selector_path = os.environ.get("OPENPI_RUNTIME_SELECTOR_PATH")
    if not selector_path:
        return
    payload = json.loads(Path(selector_path).read_text(encoding="utf-8"))
    model_id = os.environ.get("OPENPI_RUNTIME_SELECTOR_MODEL", "pi05")
    model = (payload.get("models") or {}).get(model_id) or {}
    variants = {
        str(row.get("selected_variant"))
        for row in (model.get("tasks") or {}).values()
        if isinstance(row, dict)
    }
    if variants != {expected_variant}:
        raise ValueError(
            f"weight-fused {expected_variant} requires a uniform runtime selector; found {sorted(variants)}"
        )


@dataclass
class _AlphaSummary:
    matched_layers: int = 0
    total_heads: int = 0


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enable_pi05_atm_if_configured(model: nn.Module) -> None:
    atm_flag = os.environ.get(ATM_ENABLE_ENV, "0")
    ohb_flag = os.environ.get(OHB_ENABLE_ENV, "0")
    atm_enabled = atm_flag not in ("0", "false", "False", "")
    ohb_enabled = ohb_flag not in ("0", "false", "False", "")
    if not atm_enabled and not ohb_enabled:
        setattr(model, "_openpi_atm_runtime", {"enabled": False, "matched_layers": 0})
        return

    strict = os.environ.get(ATM_STRICT_ENV, "0") not in ("0", "false", "False", "")

    alpha_path = os.environ.get(ATM_ALPHA_ENV)
    if not alpha_path:
        message = "Scaling requested but OPENPI_ATM_ALPHA_PATH not set"
        if strict:
            raise RuntimeError(message)
        print(f"[OPENPI-ATM] {message}; skipping.")
        return
    if not os.path.exists(alpha_path):
        message = f"Alpha JSON not found at {alpha_path}"
        if strict:
            raise FileNotFoundError(message)
        print(f"[OPENPI-ATM] {message}; skipping ATM.")
        return

    with open(alpha_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    alpha_data = payload.get("layers", payload)
    metadata = payload.get("meta", {}) if isinstance(payload, dict) else {}
    if not isinstance(alpha_data, dict):
        raise ValueError("ATM/OHB artifact layers must be an object")
    expected_metadata = {
        "checkpoint_sha256": os.environ.get("OPENPI_CHECKPOINT_SHA256"),
        "plan_sha256": os.environ.get("OPENPI_ATM_EXPECT_PLAN_SHA256"),
        "calibration_buffer_sha256": os.environ.get("OPENPI_ATM_EXPECT_BUFFER_SHA256"),
    }
    for key, expected in expected_metadata.items():
        if expected and metadata.get(key) != expected:
            raise ValueError(
                f"ATM/OHB metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )

    scope = os.environ.get(ATM_SCOPE_ENV, "expert")
    ohb_scope = os.environ.get(OHB_SCOPE_ENV, None) or scope
    ohb_fallback = float(os.environ.get(OHB_FALLBACK_ENV, "1.0"))
    atm_application = os.environ.get(ATM_APPLICATION_ENV, "runtime_query")
    if atm_application not in ("runtime_query", "fold_q_weight"):
        raise ValueError(f"unsupported ATM application: {atm_application!r}")
    expected_ohb_mode = os.environ.get(OHB_EXPECT_MODE_ENV)
    artifact_ohb_mode = metadata.get("ohb_mode")
    if expected_ohb_mode and artifact_ohb_mode != expected_ohb_mode:
        raise ValueError(
            f"ATM/OHB metadata mismatch for ohb_mode: {artifact_ohb_mode!r} != "
            f"{expected_ohb_mode!r}"
        )
    ohb_application = os.environ.get(OHB_APPLICATION_ENV, "runtime_output")
    if ohb_application not in ("runtime_output", "fold_o_weight", "fold_o_weight_perhead"):
        raise ValueError(f"unsupported OHB application: {ohb_application!r}")
    if runtime_selector_enabled():
        if atm_application == "fold_q_weight":
            _require_uniform_selector_variant("atm")
        if ohb_application in {"fold_o_weight", "fold_o_weight_perhead"}:
            _require_uniform_selector_variant("ohb")
    expected_ohb_application = os.environ.get(OHB_APPLICATION_ENV)
    artifact_ohb_application = metadata.get("ohb_application")
    fold_equivalent = (
        expected_ohb_application in {"fold_o_weight", "fold_o_weight_perhead"}
        and artifact_ohb_application == "runtime_output"
    )
    if expected_ohb_application and artifact_ohb_application != expected_ohb_application and not fold_equivalent:
        raise ValueError(
            f"ATM/OHB metadata mismatch for ohb_application: "
            f"{artifact_ohb_application!r} != {expected_ohb_application!r}"
        )

    summary = _AlphaSummary()
    ohb_layers = 0

    ensure_pi05_attention_patch(model, scope=scope)

    for name, module in model.named_modules():
        if not _is_dit_attention(name, module, scope=scope):
            continue
        alpha_entry = alpha_data.get(name)
        if not alpha_entry:
            beta_value = None
            alpha_values = None
        else:
            alpha_values = alpha_entry.get("all") or alpha_entry.get("alpha")
            beta_value = alpha_entry.get("beta")

        if atm_enabled and alpha_values:
            alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32)
            expected_heads = int(module.config.num_attention_heads)
            if alpha_tensor.numel() != expected_heads:
                raise ValueError(
                    f"{name}: ATM alpha has {alpha_tensor.numel()} heads, expected {expected_heads}"
                )
            if atm_application == "fold_q_weight":
                _fold_atm_into_q_projection(module, alpha_tensor)
            else:
                setattr(module, "_atm_alpha_all", alpha_tensor)
            summary.matched_layers += 1
            summary.total_heads += len(alpha_values)

        if ohb_enabled and _is_dit_attention(name, module, scope=ohb_scope):
            beta_perhead_values = alpha_entry.get("beta_perhead") if alpha_entry else None
            if beta_perhead_values is not None:
                beta_tensor = torch.tensor(beta_perhead_values, dtype=torch.float32)
                expected_heads = int(module.config.num_attention_heads)
                if beta_tensor.numel() != expected_heads:
                    raise ValueError(
                        f"{name}: OHB beta has {beta_tensor.numel()} heads, expected {expected_heads}"
                    )
                if ohb_application == "fold_o_weight_perhead":
                    _fold_ohb_perhead_into_o_projection(module, beta_tensor)
                elif ohb_application == "fold_o_weight":
                    raise ValueError(
                        "per-head OHB artifact requires fold_o_weight_perhead, not scalar fold_o_weight"
                    )
                else:
                    setattr(module, "_ohb_beta_perhead", beta_tensor)
                ohb_layers += 1
            else:
                beta = float(beta_value) if beta_value is not None else ohb_fallback
                if ohb_application == "fold_o_weight":
                    _fold_ohb_into_o_projection(module, beta)
                elif ohb_application == "fold_o_weight_perhead":
                    raise ValueError("fold_o_weight_perhead requires beta_perhead in every matched layer")
                else:
                    setattr(module, "_ohb_beta_scalar", beta)
                    _ensure_ohb_output_hook(module, name)
                ohb_layers += 1

    if summary.matched_layers == 0 and atm_enabled:
        print(f"[OPENPI-ATM] No attention layers matched alpha JSON ({alpha_path}).")
    elif atm_enabled:
        print(
            f"[OPENPI-ATM] ATM enabled for {summary.matched_layers} layers "
            f"({summary.total_heads} heads) using {alpha_path}"
        )

    if ohb_enabled:
        if ohb_layers == 0:
            print(f"[OPENPI-ATM] OHB requested but no layers found (scope={ohb_scope}); fallback beta={ohb_fallback}")
        else:
            print(f"[OPENPI-ATM] OHB enabled for {ohb_layers} layers using {alpha_path}")
    expected_layers = os.environ.get("OPENPI_ATM_EXPECT_LAYERS")
    effective_layers = summary.matched_layers if atm_enabled else ohb_layers
    if expected_layers is not None and effective_layers != int(expected_layers):
        raise RuntimeError(
            f"ATM/OHB matched {effective_layers} layers, expected {int(expected_layers)}"
        )
    runtime = {
        "enabled": True,
        "atm_enabled": atm_enabled,
        "ohb_enabled": ohb_enabled,
        "scope": scope,
        "artifact_path": str(Path(alpha_path).resolve()),
        "artifact_sha256": _sha256_file(alpha_path),
        "matched_layers": summary.matched_layers,
        "total_heads": summary.total_heads,
        "ohb_layers": ohb_layers,
        "metadata": metadata,
    }
    if (
        os.environ.get(ATM_APPLICATION_ENV) is not None
        or expected_ohb_mode
        or artifact_ohb_mode
        or expected_ohb_application
        or artifact_ohb_application
    ):
        runtime["atm_application"] = atm_application
        runtime["ohb_mode"] = artifact_ohb_mode or (
            "per_head_pre_projection"
            if any("beta_perhead" in entry for entry in alpha_data.values() if isinstance(entry, dict))
            else "per_layer_post_projection"
        )
        runtime["ohb_application"] = ohb_application
    setattr(model, "_openpi_atm_runtime", runtime)
