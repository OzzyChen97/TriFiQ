"""
GR00T DuQuant W4A8 Fake Quantization Layers

Adapted from OpenPI duquant implementation for GR00T model quantization.
Supports quantization of LLM (Eagle VLM) and DiT (action transformer) layers.
"""

import os
import re
from dataclasses import dataclass
import json
from pathlib import Path

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn

from .duquant_preprocess import (
    PackResult,
    PercentileCalibrator,
    apply_input_transform,
    apply_output_restore,
    apply_bias_row_rot,
    fake_quantize_sym,
    load_pack,
    pack_weight,
    qmax,
    save_pack,
    transform_weight_for_forward,
)


@dataclass
class DuQuantConfig:
    """DuQuant configuration matching OpenPI parameters.

    NOTE: Default values are set to None and resolved in __post_init__ to ensure
    environment variables are read at instantiation time, not at module import time.
    """
    weight_bits: Optional[int] = None
    act_bits: Optional[int] = None
    block_size: Optional[int] = None
    lambda_smooth: Optional[float] = None
    enable_permute: Optional[bool] = None
    act_percentile: Optional[float] = None
    calib_batches: Optional[int] = None
    pack_dir: Optional[str] = None
    row_rot_mode: Optional[str] = None
    block_out_size: Optional[int] = None
    # QuantVLA v2: on-the-fly min-max activation scale (Q-DiT 5.2 style)
    act_dynamic: Optional[bool] = None
    # v1.4 fast-path probe: Triton fused W4-dequant matmul (duquant_fused)
    use_fused: Optional[bool] = None

    def __post_init__(self):
        """Read environment variables at instantiation time."""
        if self.use_fused is None:
            self.use_fused = os.environ.get("GR00T_DUQUANT_FUSED", "0") not in ("0", "false", "False", "")
        if self.weight_bits is None:
            self.weight_bits = int(os.environ.get("GR00T_DUQUANT_WBITS_DEFAULT", 4))
        if self.act_bits is None:
            self.act_bits = int(os.environ.get("GR00T_DUQUANT_ABITS", 8))
        if self.block_size is None:
            self.block_size = int(os.environ.get("GR00T_DUQUANT_BLOCK", 16))
        if self.lambda_smooth is None:
            self.lambda_smooth = float(os.environ.get("GR00T_DUQUANT_LS", 0.15))
        if self.enable_permute is None:
            self.enable_permute = os.environ.get("GR00T_DUQUANT_PERMUTE", "1") not in ("0", "false", "False")
        if self.act_percentile is None:
            self.act_percentile = float(os.environ.get("GR00T_DUQUANT_ACT_PCT", 99.9))
        if self.calib_batches is None:
            self.calib_batches = int(os.environ.get("GR00T_DUQUANT_CALIB_STEPS", 32))
        if self.pack_dir is None:
            self.pack_dir = os.environ.get("GR00T_DUQUANT_PACKDIR", None)
        if self.row_rot_mode is None:
            self.row_rot_mode = os.environ.get("GR00T_DUQUANT_ROW_ROT", "restore")
        if self.block_out_size is None:
            self.block_out_size = int(os.environ.get("GR00T_DUQUANT_BLOCK_OUT", os.environ.get("GR00T_DUQUANT_BLOCK", 16)))
        if self.act_dynamic is None:
            self.act_dynamic = os.environ.get("GR00T_DUQUANT_ACT_DYNAMIC", "0") not in ("0", "false", "False")


def _parse_per_layer_wbits(env_val: Optional[str]) -> Dict[str, int]:
    """Parse per-layer weight bits from environment variable."""
    if not env_val:
        return {}
    result: Dict[str, int] = {}
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            pass
    return result


class DuQuantLinear(nn.Module):
    """DuQuant quantized linear layer with W4A8 fake quantization."""

    def __init__(self, base: nn.Linear, name: str, cfg: DuQuantConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.register_buffer("_weight", base.weight.detach().clone())
        self.register_buffer(
            "_errorfold_base_bias",
            base.bias.detach().clone() if base.bias is not None else None,
        )

        # Config
        self.cfg = cfg
        self.weight_bits = cfg.weight_bits if weight_bits is None else int(weight_bits)

        # Load or compute packing
        pack = load_pack(self.name, cfg.pack_dir)
        if pack is None:
            pack = pack_weight(
                self._weight,
                block_size=cfg.block_size,
                block_out_size=cfg.block_out_size,
                enable_permute=cfg.enable_permute,
                lambda_smooth=cfg.lambda_smooth,
            )
            pack.meta["layer_name"] = self.name
            checkpoint_sha256 = os.environ.get("GR00T_CHECKPOINT_SHA256")
            if checkpoint_sha256:
                pack.meta["checkpoint_sha256"] = checkpoint_sha256
            save_pack(self.name, pack, cfg.pack_dir)
        self.pack: PackResult = pack

        # Cache rotation matrices as torch tensors
        if pack.perm is not None:
            self.register_buffer("_perm_cache", torch.from_numpy(pack.perm).long())
        else:
            self._perm_cache = None

        # Cache input rotation matrices
        self._R_in_block_indices: List[int] = []
        if pack.R_in_blocks:
            for b, R in pack.R_in_blocks.items():
                buffer_name = f"_R_in_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_in_block_indices.append(b)

        # Cache output rotation matrices
        self._R_out_block_indices: List[int] = []
        if pack.R_out_blocks:
            for b, R in pack.R_out_blocks.items():
                buffer_name = f"_R_out_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_out_block_indices.append(b)

        # Store metadata
        self._block_size = int(pack.meta.get("block_size", 16))
        self._block_out_size = int(pack.meta.get("block_out_size", self._block_size))

        # Calibrator for activation (skipped in dynamic mode: scale is computed
        # on the fly per forward, Q-DiT 5.2 style)
        self.calibrator = (
            PercentileCalibrator(percentile=cfg.act_percentile, max_batches=cfg.calib_batches)
            if (self.cfg.act_bits > 0 and not self.cfg.act_dynamic)
            else None
        )
        self.register_buffer("_act_scale", None)
        self._act_scale_initialized = False

        # Cache transformed weight
        self._cached_weight_key: Optional[Tuple[str, torch.dtype]] = None
        self.register_buffer(
            "_W_t", None if self.cfg.use_fused else torch.zeros_like(self._weight)
        )
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype))

        # Pre-cache quantized weights
        self._precache_weight = (
            not self.cfg.use_fused
            and os.environ.get("GR00T_DUQUANT_PRECACHE_WEIGHTS", "1")
            not in ("0", "false", "False")
        )
        if self._precache_weight:
            self.register_buffer("_W_t_quantized", torch.zeros_like(self._weight))
        else:
            self._W_t_quantized = None
        self.register_buffer(
            "_W_packed_u4",
            torch.zeros(
                self.out_features, (self.in_features + 1) // 2, dtype=torch.uint8
            ),
        )
        # Static column multiplier used by direction-aware head ErrorFold.
        # Keeping it separate from the group-64 scale preserves the original
        # W4 codes and avoids a second lossy requantization.
        self.register_buffer(
            "_w_input_gain",
            None,
        )
        self._fused_ready = False
        self._inference_only_ready = False
        self._hessian_w4_loaded = False
        self._weight_quantized_cached = False

        self._bias_rot: Optional[torch.Tensor] = None
        self._debug_enabled = os.environ.get("GR00T_DUQUANT_DEBUG", "0") not in ("0", "false", "False")
        self._debug_forward_logged = False

    def _get_R_in_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_in rotation matrices on the correct device."""
        if not hasattr(self, '_R_in_cache_dict'):
            self._R_in_cache_dict = {}
        for b in self._R_in_block_indices:
            self._R_in_cache_dict[b] = getattr(self, f"_R_in_{b}")
        return self._R_in_cache_dict

    def _get_R_out_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_out rotation matrices on the correct device."""
        if not hasattr(self, '_R_out_cache_dict'):
            self._R_out_cache_dict = {}
        for b in self._R_out_block_indices:
            self._R_out_cache_dict[b] = getattr(self, f"_R_out_{b}")
        return self._R_out_cache_dict

    @property
    def weight(self) -> torch.Tensor:
        """Expose the foldable FP weight until real-quant finalization."""
        if self._weight is None:
            raise RuntimeError(f"{self.name}: FP weight was released for real-quant inference")
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        if self._weight is None:
            raise RuntimeError(f"{self.name}: cannot mutate a finalized real-quant layer")
        with torch.no_grad():
            self._weight.copy_(value)

    def _maybe_update_weight_cache(self) -> None:
        if self._inference_only_ready:
            return
        if self._hessian_w4_loaded:
            return
        apply_row = (self.cfg.row_rot_mode != "0")
        key = (str(self._weight.device), self._weight.dtype, int(self.weight_bits), int(apply_row))
        if self._cached_weight_key == key:
            return

        from .duquant_preprocess import transform_weight_for_forward_optimized

        W_t, scales = transform_weight_for_forward_optimized(
            self._weight,
            self.pack,
            weight_bits=self.weight_bits,
            apply_row_rot=apply_row,
            perm_cache=self._perm_cache,
            R_in_cache=self._get_R_in_cache(),
            R_out_cache=self._get_R_out_cache(),
            block_size=self._block_size,
            block_out_size=self._block_out_size,
        )
        if self._W_t is not None:
            self._W_t.copy_(W_t)
        self._w_scales.copy_(scales)

        # Pre-quantize weights if enabled
        if self._precache_weight and self.weight_bits > 0:
            with torch.no_grad():
                self._W_t_quantized.copy_(
                    fake_quantize_sym(W_t, scales[:, None], self.weight_bits, label="weight_prequant")
                )
            self._weight_quantized_cached = True
        else:
            self._weight_quantized_cached = False

        # v1.4 fast-path probe: packed int8 weights for the Triton fused kernel
        if self.cfg.use_fused and self.weight_bits == 4:
            from .duquant_fused import pack_w4_nibbles

            with torch.no_grad():
                self._W_packed_u4.copy_(pack_w4_nibbles(W_t, scales))
            self._fused_ready = True
        else:
            self._fused_ready = False

        self._cached_weight_key = key
        if self.bias is not None:
            if self.cfg.row_rot_mode == "propagate" and self.pack.R_out_blocks is not None:
                with torch.no_grad():
                    from .duquant_preprocess import apply_bias_row_rot_optimized
                    self._bias_rot = apply_bias_row_rot_optimized(
                        self.bias.detach(), self.pack, self._get_R_out_cache(), self._block_out_size
                    )
            else:
                self._bias_rot = None
        if self._debug_enabled:
            import logging
            logging.info(
                f"[GR00T-DUQUANT][CACHE] {self.name} device={self._weight.device} dtype={self._weight.dtype} "
                f"Wbits={self.weight_bits} Abits={self.cfg.act_bits} block_in={self.cfg.block_size} "
                f"permute={self.pack.perm is not None} row_rot={self.cfg.row_rot_mode}"
            )
            if self._weight_quantized_cached:
                logging.info(f"[GR00T-DUQUANT][CACHE] {self.name} pre-quantized weights cached")

    def set_hessian_w4(self, packed_u4: torch.Tensor, scales: torch.Tensor) -> None:
        """Install frozen GPTQ-feedback group-64 codes before finalization."""
        if self._inference_only_ready:
            raise RuntimeError(f"{self.name}: cannot replace finalized W4 codes")
        if self.pack.perm is not None or self.pack.R_in_blocks or self.pack.R_out_blocks:
            raise RuntimeError(
                f"{self.name}: v3 Hessian W4 requires the identity transform pack"
            )
        expected_packed = (self.out_features, (self.in_features + 1) // 2)
        expected_scales = (self.out_features, (self.in_features + 63) // 64)
        if tuple(packed_u4.shape) != expected_packed or packed_u4.dtype != torch.uint8:
            raise ValueError(f"{self.name}: packed W4 {packed_u4.shape} != {expected_packed}")
        if tuple(scales.shape) != expected_scales:
            raise ValueError(f"{self.name}: group scales {scales.shape} != {expected_scales}")
        if not torch.isfinite(scales).all() or bool((scales <= 0).any()):
            raise ValueError(f"{self.name}: group scales must be positive and finite")
        self._W_packed_u4 = packed_u4.detach().to(self._weight.device).contiguous()
        self._w_scales = scales.detach().to(
            device=self._weight.device, dtype=self._weight.dtype
        ).contiguous()
        self._hessian_base_scales = self._w_scales.clone()
        self._hessian_w4_loaded = True
        self._fused_ready = True
        self._cached_weight_key = ("hessian_w4_group64",)
        self._w_input_gain = None
        with torch.no_grad():
            if self._errorfold_base_bias is None:
                self.bias = None
            elif self.bias is None:
                self.bias = torch.nn.Parameter(self._errorfold_base_bias.clone())
            else:
                self.bias.copy_(self._errorfold_base_bias)

    def reset_errorfold(self) -> None:
        """Restore the frozen Hessian codes/scales before another offline gate."""
        if not self._hessian_w4_loaded or self._inference_only_ready:
            raise RuntimeError(f"{self.name}: cannot reset ErrorFold in current state")
        self._w_scales.copy_(self._hessian_base_scales)
        self._w_input_gain = None
        with torch.no_grad():
            if self._errorfold_base_bias is None:
                self.bias = None
            elif self.bias is None:
                self.bias = torch.nn.Parameter(self._errorfold_base_bias.clone())
            else:
                self.bias.copy_(self._errorfold_base_bias)

    def fold_errorfold(self, gain: torch.Tensor, correction_bias: torch.Tensor) -> None:
        """Fold an output affine into signed group scales and native bias."""
        if not self._hessian_w4_loaded or self._inference_only_ready:
            raise RuntimeError(f"{self.name}: ErrorFold requires loaded, unfinalized Hessian W4")
        gain = gain.detach().to(device=self._w_scales.device, dtype=self._w_scales.dtype).reshape(-1)
        correction_bias = correction_bias.detach().to(
            device=self._w_scales.device, dtype=self._w_scales.dtype
        ).reshape(-1)
        if gain.numel() != self.out_features or correction_bias.numel() != self.out_features:
            raise ValueError(f"{self.name}: ErrorFold channel mismatch")
        if not torch.isfinite(gain).all():
            raise ValueError(f"{self.name}: ErrorFold gain must be finite")
        self._w_scales.mul_(gain[:, None])
        with torch.no_grad():
            if self.bias is None:
                self.bias = torch.nn.Parameter(correction_bias.clone())
            else:
                self.bias.mul_(gain).add_(correction_bias)

    def fold_input_errorfold(
        self, gain: torch.Tensor, correction_bias: torch.Tensor
    ) -> None:
        """Fold an input-channel affine into packed W4 codes and output bias."""
        if not self._hessian_w4_loaded or self._inference_only_ready:
            raise RuntimeError(f"{self.name}: input ErrorFold requires loaded Hessian W4")
        gain = gain.detach().to(device=self._w_scales.device, dtype=torch.float32).reshape(-1)
        correction_bias = correction_bias.detach().to(
            device=self._w_scales.device, dtype=torch.float32
        ).reshape(-1)
        if gain.numel() != self.in_features or correction_bias.numel() != self.in_features:
            raise ValueError(f"{self.name}: input ErrorFold channel mismatch")
        packed = self._W_packed_u4
        low = (packed & 0x0F).to(torch.int16)
        high = ((packed >> 4) & 0x0F).to(torch.int16)
        codes = torch.stack((low, high), dim=-1).reshape(self.out_features, -1)
        codes = torch.where(codes >= 8, codes - 16, codes)[:, : self.in_features]
        expanded = self._w_scales.to(torch.float32).repeat_interleave(64, dim=1)[
            :, : self.in_features
        ]
        weight = codes.to(torch.float32) * expanded
        additive = weight @ correction_bias
        if self._w_input_gain is None:
            self._w_input_gain = gain.to(self._w_scales.dtype).clone()
        else:
            self._w_input_gain.mul_(gain.to(self._w_input_gain.dtype))
        with torch.no_grad():
            if self.bias is None:
                self.bias = torch.nn.Parameter(additive.to(self._w_scales.dtype))
            else:
                self.bias.add_(additive.to(self.bias.dtype))

    def finalize_real_quant(self) -> int:
        """Freeze packed W4 residency and release all FP weight-sized buffers."""
        if self._inference_only_ready:
            return int(self._W_packed_u4.numel())
        if not self.cfg.use_fused or self.weight_bits != 4:
            raise RuntimeError(f"{self.name}: real-quant finalization requires fused W4")
        if (
            self.cfg.act_bits > 0
            and not self.cfg.act_dynamic
            and not self._act_scale_initialized
        ):
            raise RuntimeError(f"{self.name}: static A8 scale is not ready")
        self._maybe_update_weight_cache()
        if not self._fused_ready:
            raise RuntimeError(f"{self.name}: packed W4 cache is not ready")
        self._weight = None
        self._W_t = None
        self._W_t_quantized = None
        # Candidate reuse is calibration-only.  A frozen deployment no longer
        # needs the identity-reset copies, so release them before reporting
        # inference residency and static bytes.
        if hasattr(self, "_hessian_base_scales"):
            self._hessian_base_scales = None
        self._errorfold_base_bias = None
        self.calibrator = None
        self._inference_only_ready = True
        return int(self._W_packed_u4.numel())

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        # QuantVLA v2: on-the-fly min-max activation scale (Q-DiT 5.2 style).
        # Computed per forward call; no calibration buffer needed, so the scale
        # adapts to per-sample / per-denoise-step dynamic range drift.
        if self.cfg.act_dynamic:
            with torch.no_grad():
                x_abs = torch.abs(x.detach().to(torch.float32))
                x2d = x_abs.reshape(-1, x_abs.shape[-1])
                amax = x2d.amax(dim=0)
                max_q = qmax(self.cfg.act_bits)
                scale = torch.clamp(amax / max_q, min=1e-6)
                return scale.to(dtype=x.dtype, device=x.device)

        if self._act_scale_initialized:
            if self._act_scale.ndim == 1:
                return self._act_scale
            if self._act_scale.ndim != 2 or self._act_scale.shape != (
                4, self.in_features
            ):
                raise RuntimeError(f"{self.name}: invalid per-step A8 table {self._act_scale.shape}")
            from .dit_step_context import get_current_dit_step, get_total_dit_steps

            step = get_current_dit_step()
            total = get_total_dit_steps()
            if step is None or total != 4 or not 0 <= int(step) < 4:
                raise RuntimeError(
                    f"{self.name}: four-row DiT A8 table requires deterministic step context"
                )
            return self._act_scale[int(step)]

        with torch.no_grad():
            if self.calibrator is not None:
                # P0-2 (correctness review): the scale must NOT freeze on the
                # first batch. Keep observing until the calibrator is full
                # (cfg.calib_batches batches); until then return a PROVISIONAL
                # per-forward scale computed from the current batch WITHOUT
                # setting _act_scale_initialized. Only a full calibration
                # finalizes and freezes the scale.
                self.calibrator.observe(x)
                if self.calibrator.is_full():
                    p_vec = self.calibrator.finalize()
                    max_q = qmax(self.cfg.act_bits)
                    scale = torch.clamp(p_vec / max_q, min=1e-6)
                    scale = scale.to(dtype=x.dtype, device=x.device).clone()
                    if self._act_scale is None:
                        self._act_scale = scale
                    else:
                        self._act_scale.copy_(scale)
                    self._act_scale_initialized = True
                    return self._act_scale
                # provisional scale (not frozen)
                x_abs = torch.abs(x.detach().to(torch.float32))
                C = x_abs.shape[-1]
                x2d = x_abs.reshape(-1, C)
                p_vec = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
                max_q = qmax(self.cfg.act_bits)
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                return scale.to(dtype=x.dtype, device=x.device).clone()

            # no calibrator (should not happen in static mode, kept as fallback)
            if not self._act_scale_initialized:
                x_abs = torch.abs(x.detach().to(torch.float32))
                C = x_abs.shape[-1]
                x2d = x_abs.reshape(-1, C)
                p_vec = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
                max_q = qmax(self.cfg.act_bits)
                scale = torch.clamp(p_vec / max_q, min=1e-6)
                scale = scale.to(dtype=x.dtype, device=x.device).clone()
                if self._act_scale is None:
                    self._act_scale = scale
                else:
                    self._act_scale.copy_(scale)
                self._act_scale_initialized = True

        return self._act_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply optimized per-block input transform
        from .duquant_preprocess import apply_input_transform_optimized
        x_t = apply_input_transform_optimized(
            x, self.pack, self._perm_cache, self._get_R_in_cache(), self._block_size
        )

        # Fake-quantize activations if enabled
        if self.cfg.act_bits > 0:
            s_a = self._get_act_scale(x_t)
            x_t = fake_quantize_sym(x_t, s_a, self.cfg.act_bits, label="activation_forward")

        # Transform and fake-quantize weights
        self._maybe_update_weight_cache()

        # Use pre-quantized weights
        if self.cfg.use_fused and self.weight_bits == 4 and self._fused_ready:
            # v1.4 fast-path probe: Triton fused W4-dequant matmul (same math
            # as the eager fake-quant path; fp32 accumulation in-kernel)
            from .duquant_fused import fused_linear_w4_nibbles

            y_lin = fused_linear_w4_nibbles(
                x_t,
                self._W_packed_u4,
                self._w_scales,
                input_gain=self._w_input_gain,
            )
        elif self._weight_quantized_cached:
            y_lin = torch.nn.functional.linear(x_t, self._W_t_quantized, None)
        elif self.weight_bits > 0:
            y_lin = torch.nn.functional.linear(
                x_t,
                fake_quantize_sym(
                    self._W_t,
                    self._w_scales[:, None],
                    self.weight_bits,
                    label="weight_fallback",
                ),
                None
            )
        else:
            y_lin = torch.nn.functional.linear(x_t, self._W_t, None)

        # Apply row restore if requested
        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            from .duquant_preprocess import apply_output_restore_optimized
            y_lin = apply_output_restore_optimized(
                y_lin, self.pack, self._get_R_out_cache(), self._block_out_size
            )
            if self.bias is not None:
                y_lin = y_lin + self.bias
        else:
            if self.bias is not None:
                bias_to_add = (
                    self._bias_rot
                    if self.cfg.row_rot_mode == "propagate" and self._bias_rot is not None
                    else self.bias
                )
                y_lin = y_lin + bias_to_add
        if self._debug_enabled and not self._debug_forward_logged:
            import logging
            logging.info(
                f"[GR00T-DUQUANT][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y_lin.shape)} "
                f"weight_bits={self.weight_bits} act_bits={self.cfg.act_bits}"
            )
            self._debug_forward_logged = True
        return y_lin


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def select_targets(
    model: nn.Module,
    *,
    include_regex: str = r".*(q_proj|k_proj|v_proj|out_proj|fc1|fc2|up_proj|down_proj|gate_proj).*",
    exclude_regex: str = r"(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)",
    scope_prefix: Optional[str] = None,
    whitelist: Optional[Iterable[str]] = None,
    blacklist: Optional[Iterable[str]] = None,
) -> List[Tuple[str, nn.Linear]]:
    """Select linear layers to quantize based on regex patterns."""
    inc = re.compile(include_regex)
    exc = re.compile(exclude_regex)
    wl = set(whitelist or [])
    bl = set(blacklist or [])
    results: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if scope_prefix is not None and not name.startswith(scope_prefix):
            continue
        if name in bl:
            continue
        if wl and name not in wl:
            continue
        if not wl and (not inc.search(name) or exc.search(name)):
            continue
        results.append((name, mod))
    return results


def _build_layer_cfg(
    cfg: DuQuantConfig,
    overrides: Optional[Dict[str, Any]] = None,
    pack_dir: Optional[str] = None,
) -> DuQuantConfig:
    """Return a per-layer config copy with optional field overrides (QuantVLA v2)."""
    import dataclasses

    if not overrides and not pack_dir:
        return cfg
    kwargs: Dict[str, Any] = {}
    if overrides:
        kwargs.update(overrides)
    if pack_dir:
        kwargs["pack_dir"] = pack_dir
    return dataclasses.replace(cfg, **kwargs)


def wrap_duquant(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: DuQuantConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    per_layer_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> None:
    """Wrap selected layers with DuQuant quantization.

    QuantVLA v2 extensions:
    - per_layer_cfg: {name: {block_size, block_out_size, act_percentile,
      lambda_smooth, pack_dir}} per-layer overrides on top of the global cfg.
      NOTE: pack content (R rotations / permutation) depends on block_size and
      ls, so layers with different (block, ls) MUST use separate pack_dir
      values (see enable_duquant_if_configured / GR00T_DUQUANT_PLAN).
    """
    per_layer_wbits = per_layer_wbits or {}
    per_layer_cfg = per_layer_cfg or {}
    replaced = 0
    listed = 0
    for name in layer_names:
        # Skip action head by default unless explicitly requested
        if os.environ.get("GR00T_DUQUANT_INCLUDE_ACTION_HEAD", "0") in ("0", "false", "False"):
            is_action_head = "action_head" in name and not name.startswith("action_head.model.")
            if (
                name.endswith("action_out_proj")
                or ".action_out_proj" in name
                or is_action_head
            ):
                continue
        parent, attr = _get_parent_module_and_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        layer_cfg = _build_layer_cfg(cfg, per_layer_cfg.get(name))
        if dry_run:
            msg = (
                f"[GR00T-DUQUANT][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{layer_cfg.act_bits} perm={layer_cfg.enable_permute} "
                f"block_in={layer_cfg.block_size} block_out={layer_cfg.block_out_size} row_rot={layer_cfg.row_rot_mode}"
            )
            print(msg)
            listed += 1
            continue
        dq = DuQuantLinear(mod, name=name, cfg=layer_cfg, weight_bits=wbits)
        setattr(parent, attr, dq)
        # Use actual block sizes from pack (not cfg defaults)
        actual_block_in = dq._block_size
        actual_block_out = dq._block_out_size
        print(
            f"[GR00T-DUQUANT][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) -> DuQuantLinear "
            f"W{wbits} A{layer_cfg.act_bits} perm={layer_cfg.enable_permute} block_in={actual_block_in} block_out={actual_block_out} row_rot={layer_cfg.row_rot_mode}"
        )
        replaced += 1
    if dry_run:
        print(f"[GR00T-DUQUANT] Dry-run total layers listed: {listed}")
    else:
        print(f"[GR00T-DUQUANT] Total layers replaced: {replaced}")


def enable_duquant_if_configured(model: nn.Module) -> None:
    """
    Entry point to enable DuQuant based on environment variables.

    Activation conditions:
    - If GR00T_DUQUANT_DRYRUN is set => dry-run listing only
    - Or if any GR00T_DUQUANT_* variable (other than PACKDIR) is set => perform replacement
    - Otherwise do nothing
    """
    env = os.environ
    keys = [k for k in env.keys() if k.startswith("GR00T_DUQUANT_")]
    activate = any(k not in ("GR00T_DUQUANT_PACKDIR",) for k in keys)
    if not activate:
        return

    # Scope defaults to empty (search entire model)
    scope = env.get("GR00T_DUQUANT_SCOPE", "")
    whitelist = env.get("GR00T_DUQUANT_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None

    # Default: quantize LLM + DiT MLP layers (matching OpenPI pattern)
    # Include LLM attention+MLP and DiT MLP projections
    inc = env.get(
        "GR00T_DUQUANT_INCLUDE",
        (
            r".*(?:"
            r"backbone\.eagle_model\.language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            r"|"
            r"action_head\.model\..*(?:attn1\.to_(?:q|k|v)|attn1\.to_out\.0|ff\.net\.(?:0\.proj|2))"
            r").*"
        ),
    )
    # Exclude vision encoder, embeddings, auxiliary projectors
    exc = env.get(
        "GR00T_DUQUANT_EXCLUDE",
        (
            r"(?:^|\.)"
            r"(?:vision_model|vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|future_tokens|vl_self_attention)"
            r"(?:\.|$)"
        ),
    )

    per_layer_wbits = _parse_per_layer_wbits(env.get("GR00T_DUQUANT_WBITS"))
    dry_run = env.get("GR00T_DUQUANT_DRYRUN", "0") not in ("0", "false", "False")

    cfg = DuQuantConfig()

    # ---- QuantVLA v2: mixed-precision plan (GR00T_DUQUANT_PLAN=<json>) ----
    # Plan schema (docs/quantvla_v2_design.md §6.3.4):
    #   {"layers": {"<name>": {"bits": 4, "group": 64, "skip": false, ...}}, ...,
    #    "packdirs": {"64": "/path/b64", "128": "/path/b128"}, ...}
    # The plan takes precedence over GR00T_DUQUANT_WBITS_DEFAULT/WBITS for the
    # listed layers; skipped layers are left untouched (FP16).
    per_layer_cfg: Dict[str, Dict[str, Any]] = {}
    plan = None
    plan_path = env.get("GR00T_DUQUANT_PLAN")
    if plan_path:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
        packdirs = plan.get("packdirs") or {}
        for name, entry in (plan.get("layers") or {}).items():
            if entry.get("skip"):
                continue
            bits = entry.get("bits")
            if bits is not None:
                per_layer_wbits[name] = int(bits)
            override: Dict[str, Any] = {}
            group = entry.get("group")
            if group is not None:
                override["block_size"] = int(group)
                override["block_out_size"] = int(group)
            if entry.get("ls") is not None:
                override["lambda_smooth"] = float(entry["ls"])
            # v3 Hessian codes are defined in the identity basis.  Legacy
            # plans may still name rotation-pack directories; ignore those
            # adapter-local cache paths and use the explicitly supplied
            # identity pack directory for Hessian deployment.
            pack_dir = None
            if not env.get("GR00T_DUQUANT_HESSIAN_W4_PATH"):
                pack_dir = entry.get("packdir") or packdirs.get(str(group))
            if pack_dir:
                override["pack_dir"] = pack_dir
            if override:
                per_layer_cfg[name] = override
        print(f"[GR00T-DUQUANT][PLAN] loaded {len(plan.get('layers') or {})} layer entries from {plan_path}")

    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    if plan is not None:
        skipped = {
            name
            for name, entry in (plan.get("layers") or {}).items()
            if entry.get("skip")
        }
        kept = [n for n in layer_names if n not in skipped]
        print(f"[GR00T-DUQUANT][PLAN] skipping {len(skipped)} layers, wrapping {len(kept)}")
        layer_names = kept
    print(f"[GR00T-DUQUANT] SCOPE filter: '{scope}'")
    print(f"[GR00T-DUQUANT] Matched Linear layers: {len(layer_names)}")

    if len(layer_names) == 0 and scope:
        # Debug: print some layer names to help diagnose
        all_linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        print(f"[GR00T-DUQUANT] DEBUG: Total Linear layers in model: {len(all_linears)}")
        print(f"[GR00T-DUQUANT] DEBUG: First 10 Linear layer names:")
        for name, _ in all_linears[:10]:
            print(f"[GR00T-DUQUANT] DEBUG:   {name}")
        if scope:
            matching_prefix = [n for n, _ in all_linears if n.startswith(scope.rstrip('.'))]
            print(f"[GR00T-DUQUANT] DEBUG: Layers matching prefix '{scope.rstrip('.')}': {len(matching_prefix)}")
            if matching_prefix:
                for name in matching_prefix[:5]:
                    print(f"[GR00T-DUQUANT] DEBUG:   {name}")

    if dry_run:
        wrap_duquant(model, layer_names, cfg, per_layer_wbits, per_layer_cfg, dry_run=True)
        return
    wrap_duquant(model, layer_names, cfg, per_layer_wbits, per_layer_cfg, dry_run=False)
    hessian_path = env.get("GR00T_DUQUANT_HESSIAN_W4_PATH")
    if hessian_path:
        load_hessian_w4(
            model,
            hessian_path,
            errorfold_path=env.get("GR00T_ERRORFOLD_PATH"),
        )


# --------------------------------------------------------------------------- #
# v1.3 helpers: calibration lifecycle (P0-2)
# --------------------------------------------------------------------------- #
def calibration_progress(model: nn.Module) -> Tuple[int, int]:
    """(fully_calibrated, total) act calibrators across DuQuantLinear layers."""
    total = full = 0
    for m in model.modules():
        if isinstance(m, DuQuantLinear) and m.calibrator is not None:
            total += 1
            if m.calibrator.is_full():
                full += 1
    return full, total


def all_calibrated(model: nn.Module) -> bool:
    full, total = calibration_progress(model)
    return total > 0 and full == total


def static_calibrators_required(model: nn.Module) -> bool:
    """True when the model has at least one STATIC-A8 DuQuantLinear.

    (dynamic-act mode has calibrator=None everywhere -> False, so 0/0 is NOT
    treated as an incomplete static calibration; review round 2, item 4.)"""
    return any(
        isinstance(m, DuQuantLinear) and m.calibrator is not None
        for m in model.modules()
    )


def static_scales_ready(model: nn.Module) -> bool:
    """True when every STATIC-A8 layer has a usable frozen scale installed.

    Review round 3, item 1: `all_calibrated()` checks calibrator.is_full(), but
    a scale loaded from disk never re-runs the calibrator — so the loaded state
    must be judged by `_act_scale_initialized`, not by the calibrator counter.
    """
    static_layers = [
        m for m in model.modules()
        if isinstance(m, DuQuantLinear) and m.calibrator is not None
    ]
    return bool(static_layers) and all(
        m._act_scale_initialized and m._act_scale is not None for m in static_layers
    )


def finalize_real_quant(model: nn.Module) -> Dict[str, int | bool]:
    """Finalize every DuQuant layer into inference-only packed W4 residency."""
    layers = [m for m in model.modules() if isinstance(m, DuQuantLinear)]
    if not layers:
        raise RuntimeError("real-quant finalization found no DuQuant layers")
    packed_bytes = sum(layer.finalize_real_quant() for layer in layers)
    dequant_scale_bytes = sum(
        layer._w_scales.numel() * layer._w_scales.element_size() for layer in layers
    )
    input_gain_bytes = sum(
        layer._w_input_gain.numel() * layer._w_input_gain.element_size()
        for layer in layers
        if layer._w_input_gain is not None
    )
    activation_scale_bytes = sum(
        layer._act_scale.numel() * layer._act_scale.element_size()
        for layer in layers
        if layer._act_scale is not None
    )
    bias_bytes = sum(
        layer.bias.numel() * layer.bias.element_size()
        for layer in layers
        if layer.bias is not None
    )
    if not all(layer._inference_only_ready for layer in layers):
        raise RuntimeError("real-quant finalization is incomplete")
    return {
        "layers": len(layers),
        "packed_weight_bytes": int(packed_bytes),
        "dequant_scale_bytes": int(dequant_scale_bytes),
        "input_gain_bytes": int(input_gain_bytes),
        "activation_scale_bytes": int(activation_scale_bytes),
        "bias_bytes": int(bias_bytes),
        "auxiliary_static_bytes": int(
            dequant_scale_bytes
            + input_gain_bytes
            + activation_scale_bytes
            + bias_bytes
        ),
        "fp_weight_sized_buffers": 0,
        "packed_low_bit_residency": True,
    }


def load_hessian_w4(
    model: nn.Module,
    path: str | Path,
    *,
    errorfold_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Load one frozen, inventory-exact group-64 W4 artifact for GR00T."""
    import hashlib

    import numpy as np

    artifact = Path(path).expanduser().resolve()
    sidecar = Path(str(artifact) + ".json")
    if not artifact.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Hessian W4 artifact is incomplete: {artifact}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 3 or metadata.get("group_size") != 64:
        raise ValueError("unsupported Hessian W4 artifact schema/group")
    if metadata.get("protocol_id") != "quantvla-gr00t-pi05-errorfold-v4":
        raise ValueError("Hessian W4 artifact protocol drift")
    actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if metadata.get("npz_sha256") != actual_hash:
        raise ValueError("Hessian W4 artifact hash mismatch")
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, DuQuantLinear)]
    names = [name for name, _ in layers]
    if metadata.get("layer_names") != names:
        raise ValueError("Hessian W4 layer inventory does not match wrapped GR00T model")
    correction_layers: Mapping[str, Any] = {}
    if errorfold_path is not None:
        correction = json.loads(Path(errorfold_path).expanduser().resolve().read_text(encoding="utf-8"))
        if correction.get("schema_version") != 3 or correction.get("kind") not in {
            "errorfold_compensation", "errorfold_grid_candidate"
        }:
            raise ValueError("unsupported ErrorFold deployment artifact")
        correction_layers = correction.get("layers") or {}
    with np.load(artifact, allow_pickle=False) as archive:
        stored_names = [str(value) for value in archive["layer_names"].tolist()]
        if stored_names != names:
            raise ValueError("Hessian W4 NPZ inventory mismatch")
        for index, (name, module) in enumerate(layers):
            module.set_hessian_w4(
                torch.from_numpy(np.asarray(archive[f"packed_{index:04d}"])),
                torch.from_numpy(np.asarray(archive[f"scales_{index:04d}"])),
            )
            if errorfold_path is not None:
                entry = correction_layers.get(name)
                if not entry or entry.get("kind") != "linear":
                    raise ValueError(f"ErrorFold lacks Linear correction for {name}")
                module.fold_errorfold(
                    torch.as_tensor(entry["effective_gain"]),
                    torch.as_tensor(entry["effective_bias"]),
                )
    runtime = getattr(model, "_gr00t_duquant_runtime", {})
    runtime.update(
        {
            "hessian_w4_path": str(artifact),
            "hessian_w4_sha256": actual_hash,
            "hessian_group_size": 64,
            "hessian_w4_loaded": len(layers),
            "errorfold_path": str(Path(errorfold_path).resolve()) if errorfold_path else None,
            "selector_free": True,
        }
    )
    setattr(model, "_gr00t_duquant_runtime", runtime)
    return runtime


def apply_errorfold(model: nn.Module, path: str | Path) -> Dict[str, Any]:
    """Apply one materialized gate without reloading model or packed codes."""
    correction_path = Path(path).expanduser().resolve()
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    if correction.get("schema_version") != 3 or correction.get("kind") not in {
        "errorfold_compensation",
        "errorfold_grid_candidate",
    }:
        raise ValueError("unsupported ErrorFold artifact")
    correction_layers = correction.get("layers") or {}
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DuQuantLinear)
    ]
    for name, module in layers:
        entry = correction_layers.get(name)
        if not entry or entry.get("kind") != "linear":
            raise ValueError(f"ErrorFold lacks Linear correction for {name}")
        module.reset_errorfold()
        module.fold_errorfold(
            torch.as_tensor(entry["effective_gain"]),
            torch.as_tensor(entry["effective_bias"]),
        )
    runtime = getattr(model, "_gr00t_duquant_runtime", {})
    runtime.update(
        {
            "errorfold_path": str(correction_path),
            "selector_free": True,
            "offline_candidate_reuse": True,
        }
    )
    setattr(model, "_gr00t_duquant_runtime", runtime)
    return runtime


def save_act_scales(model: nn.Module, path: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """Persist frozen per-layer static A8 scales + a metadata sidecar.

    Review round 3, item 4: the sidecar records the experiment identity
    (plan/checkpoint/buffer/data-config/act settings) so a stale scale file can
    never be silently reused for a different plan or calibration buffer.
    """
    import hashlib
    import json

    import numpy as np

    scales: Dict[str, np.ndarray] = {}
    for name, m in model.named_modules():
        if isinstance(m, DuQuantLinear) and m.calibrator is not None:
            if not m._act_scale_initialized or m._act_scale is None:
                raise RuntimeError(f"layer {name} has no frozen act scale to save")
            scales[name] = m._act_scale.detach().float().cpu().numpy()
    np.savez(path, **scales)
    meta = dict(meta or {})
    meta.setdefault("n_static_layers", len(scales))
    meta.setdefault("layer_names", sorted(scales))
    sidecar = Path(str(path) + ".meta.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_act_scales(
    model: nn.Module, path: str, require: Optional[Dict[str, Any]] = None
) -> None:
    """Restore frozen static A8 scales; every static layer must be covered.

    Review round 3, item 1: after loading, the per-layer calibrators are marked
    full so BOTH `all_calibrated()` (calibrator counter) and
    `static_scales_ready()` (frozen-scale state) agree — the previous version
    left calibrator.is_full() == False and the load path aborted with
    'calibration incomplete' on the second server start.
    `require` is a dict of sidecar fields that must match (e.g. buffer sha256,
    plan hash) — a mismatch raises instead of silently reusing stale scales.
    """
    import json

    import numpy as np

    if require:
        sidecar = Path(str(path) + ".meta.json")
        if not sidecar.exists():
            raise RuntimeError(f"act-scale sidecar missing: {sidecar}")
        with open(sidecar, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for k, v in require.items():
            if meta.get(k) != v:
                raise RuntimeError(
                    f"act-scale sidecar mismatch on '{k}': saved {meta.get(k)!r} != expected {v!r}"
                )

    data = np.load(path, allow_pickle=False)
    missing = []
    for name, m in model.named_modules():
        if isinstance(m, DuQuantLinear) and m.calibrator is not None:
            if name not in data:
                missing.append(name)
                continue
            scale = torch.from_numpy(data[name].astype(np.float32)).to(
                dtype=m._weight.dtype, device=m._weight.device
            )
            # A model (and therefore its registered buffers) may have been
            # constructed under ``torch.inference_mode()``.  Such buffers are
            # inference tensors and PyTorch rejects an in-place update made
            # outside inference mode, even though loading a frozen scale is a
            # non-gradient state restore.  Keep the update in inference mode
            # so both ordinary and inference-tensor buffers are supported.
            with torch.inference_mode():
                if m._act_scale is None:
                    m._act_scale = scale
                else:
                    m._act_scale.copy_(scale)
            m._act_scale_initialized = True
            # mark the calibrator full: the frozen scale supersedes any future
            # observation (and _get_act_scale short-circuits on initialized)
            m.calibrator._seen = m.calibrator.max_batches
    if missing:
        raise RuntimeError(f"act-scale file missing {len(missing)} static layers: {missing[:5]}")


def selftest() -> None:
    """P0 regression tests (CPU): weight immutability + bit-order invariance
    (P0-1) and act-scale calibration lifecycle (P0-2).

    Run: python -m gr00t.quantization.duquant_layers
    """
    import tempfile
    from pathlib import Path

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        base = nn.Linear(64, 64, bias=False)
        base.weight.data.normal_()
        cfg = DuQuantConfig(
            weight_bits=4, act_bits=0, block_size=16, block_out_size=16,
            enable_permute=False, pack_dir=tmp, row_rot_mode="restore",
            act_dynamic=False, calib_batches=4, lambda_smooth=0.15,
        )
        layer = DuQuantLinear(base, "selftest_w", cfg)
        w0 = layer._weight.clone()
        x = torch.randn(8, 64)

        # P0-1a: weight buffer must never change under bit switching
        layer.weight_bits = 4
        y4a = layer(x).clone()
        assert torch.equal(layer._weight, w0), "weight buffer mutated after bits=4"
        layer.weight_bits = 8
        y8 = layer(x).clone()
        assert torch.equal(layer._weight, w0), "weight buffer mutated after bits=8"
        layer.weight_bits = 4
        y4b = layer(x).clone()
        assert torch.equal(layer._weight, w0), "weight buffer mutated after bits=4 (second pass)"
        # P0-1b: bit-order invariance — second W4 forward equals the first
        assert torch.allclose(y4a, y4b, atol=1e-5), "bit-order invariance violated (4->8->4)"
        # P0-1c: weight_bits=0 twice identical
        layer.weight_bits = 0
        y0a = layer(x).clone()
        layer.weight_bits = 4
        layer.weight_bits = 0
        y0b = layer(x).clone()
        assert torch.allclose(y0a, y0b, atol=1e-5), "weight_bits=0 path not repeatable"

        # P0-2: static A8 scale must accumulate cfg.calib_batches batches and
        # only freeze afterwards; provisional scales must NOT freeze.
        base2 = nn.Linear(64, 64, bias=False)
        base2.weight.data.normal_()
        cfg2 = DuQuantConfig(
            weight_bits=0, act_bits=8, block_size=16, block_out_size=16,
            enable_permute=False, pack_dir=tmp, row_rot_mode="restore",
            act_dynamic=False, calib_batches=4, act_percentile=99.9, lambda_smooth=0.15,
        )
        layer2 = DuQuantLinear(base2, "selftest_a8", cfg2)
        layer2(x)
        layer2(x * 2.0)  # batches 1-2 observed, NOT frozen
        assert not layer2._act_scale_initialized, "act scale froze too early (P0-2)"
        s1 = layer2._get_act_scale(x)  # batch 3 -> provisional, still not frozen
        assert not layer2._act_scale_initialized, "provisional scale must not freeze"
        s2 = layer2._get_act_scale(x * 3.0)  # batch 4 -> calibrator full -> freeze
        assert layer2.calibrator.is_full() and layer2._act_scale_initialized, (
            "scale did not freeze after full calibration"
        )
        assert not torch.allclose(s1, s2), "frozen scale should reflect all 4 batches"
        s_frozen = layer2._act_scale.clone()
        layer2(torch.randn(8, 64) * 100.0)
        assert torch.equal(layer2._act_scale, s_frozen), "frozen scale changed after finalize"

        # ---- round-3 regression: scale persistence round-trip ----
        # A calibrated model saves its frozen scales; a FRESH model loads them
        # with NO warmup and must be judged ready by BOTH all_calibrated() and
        # static_scales_ready(), and produce identical outputs.
        out_a = layer2(x).clone()  # frozen-scale forward on model A
        scale_path = Path(tmp) / "act_scales.npz"
        save_act_scales(layer2, str(scale_path), meta={"buffer_sha256": "test", "calib_batches": 4})
        base3 = nn.Linear(64, 64, bias=False)
        base3.weight.data.copy_(base2.weight.data)
        layer3 = DuQuantLinear(base3, "selftest_a8_b", cfg2)
        assert not static_scales_ready(layer3)
        load_act_scales(layer3, str(scale_path))
        assert static_scales_ready(layer3), "loaded scales must be ready"
        assert all_calibrated(layer3), "loaded scales must satisfy all_calibrated"
        out_b = layer3(x).clone()
        assert torch.allclose(out_a, out_b, atol=1e-6), "round-trip outputs differ"
        # Models loaded by the inference service can own inference-tensor
        # buffers.  Frozen-scale restore must also work when called later from
        # ordinary Python code (the ATM/OHB calibration path does this).
        base4 = nn.Linear(64, 64, bias=False)
        base4.weight.data.copy_(base2.weight.data)
        layer4 = DuQuantLinear(base4, "selftest_a8_inference_buffer", cfg2)
        with torch.inference_mode():
            layer4._act_scale = torch.zeros_like(s_frozen)
        load_act_scales(layer4, str(scale_path))
        assert torch.equal(layer4._act_scale, s_frozen), (
            "scale restore into inference-tensor buffer failed"
        )
        # sidecar mismatch must raise
        try:
            load_act_scales(layer3, str(scale_path), require={"buffer_sha256": "other"})
            raise AssertionError("sidecar mismatch was not detected")
        except RuntimeError:
            pass
        print("  scale round-trip OK (save -> load -> ready -> identical outputs; sidecar checked)")

    print("[duquant_layers] selftest OK (P0-1 weight immutability + bit-order invariance, "
          "P0-2 act-scale calibration lifecycle, round-3 scale persistence)")


if __name__ == "__main__":
    selftest()
