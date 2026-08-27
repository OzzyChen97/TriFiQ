"""
GR00T DuQuant W4A8 Fake Quantization Layers

Adapted from OpenPI duquant implementation for GR00T model quantization.
Supports quantization of LLM (Eagle VLM) and DiT (action transformer) layers.
"""

import json
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
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
from .plan import load_quant_plan, sha256_file, validate_plan_inventory


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
    # Shared v5 activation policy: deterministic per-forward per-channel A8.
    act_dynamic: Optional[bool] = None

    def __post_init__(self):
        """Read environment variables at instantiation time."""
        if self.weight_bits is None:
            self.weight_bits = int(os.environ.get("OPENPI_DUQUANT_WBITS_DEFAULT", 4))
        if self.act_bits is None:
            self.act_bits = int(os.environ.get("OPENPI_DUQUANT_ABITS", 8))
        if self.block_size is None:
            self.block_size = int(os.environ.get("OPENPI_DUQUANT_BLOCK", 16))
        if self.lambda_smooth is None:
            self.lambda_smooth = float(os.environ.get("OPENPI_DUQUANT_LS", 0.15))
        if self.enable_permute is None:
            self.enable_permute = os.environ.get("OPENPI_DUQUANT_PERMUTE", "1") not in ("0", "false", "False")
        if self.act_percentile is None:
            self.act_percentile = float(os.environ.get("OPENPI_DUQUANT_ACT_PCT", 99.9))
        if self.calib_batches is None:
            self.calib_batches = int(os.environ.get("OPENPI_DUQUANT_CALIB_STEPS", 32))
        if self.pack_dir is None:
            self.pack_dir = os.environ.get("OPENPI_DUQUANT_PACKDIR", None)
        if self.row_rot_mode is None:
            self.row_rot_mode = os.environ.get("OPENPI_DUQUANT_ROW_ROT", "restore")
        if self.block_out_size is None:
            self.block_out_size = int(os.environ.get("OPENPI_DUQUANT_BLOCK_OUT", os.environ.get("OPENPI_DUQUANT_BLOCK", 16)))
        if self.act_dynamic is None:
            self.act_dynamic = os.environ.get("OPENPI_DUQUANT_ACT_DYNAMIC", "0") not in (
                "0", "false", "False", ""
            )


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


def _validate_pack_for_layer(
    pack: PackResult,
    *,
    name: str,
    in_features: int,
    out_features: int,
    cfg: DuQuantConfig,
) -> None:
    """Reject stale or mislabeled packs before they can affect evaluation."""
    expected = {
        "in_features": int(in_features),
        "out_features": int(out_features),
        "block_size": int(cfg.block_size),
        "block_out_size": int(cfg.block_out_size),
        "enable_permute": bool(cfg.enable_permute),
    }
    for key, value in expected.items():
        if pack.meta.get(key) != value:
            raise ValueError(
                f"{name}: pack metadata {key}={pack.meta.get(key)!r} != requested {value!r}"
            )
    stored_layer_name = pack.meta.get("layer_name")
    if stored_layer_name is not None and stored_layer_name != name:
        raise ValueError(f"{name}: pack belongs to layer {stored_layer_name!r}")
    expected_checkpoint = os.environ.get("OPENPI_CHECKPOINT_SHA256")
    if expected_checkpoint and pack.meta.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError(
            f"{name}: pack checkpoint hash {pack.meta.get('checkpoint_sha256')!r} "
            f"!= {expected_checkpoint!r}"
        )
    stored_lambda = pack.meta.get("lambda_smooth")
    if stored_lambda is None or abs(float(stored_lambda) - float(cfg.lambda_smooth)) > 1e-12:
        raise ValueError(
            f"{name}: pack lambda_smooth={stored_lambda!r} != requested {cfg.lambda_smooth!r}"
        )
    if tuple(pack.weight_scale.shape) != (out_features,):
        raise ValueError(
            f"{name}: pack weight_scale shape {pack.weight_scale.shape} != ({out_features},)"
        )
    if cfg.enable_permute:
        if pack.perm is None or tuple(pack.perm.shape) != (in_features,):
            raise ValueError(f"{name}: enabled permutation is missing or malformed")
    elif pack.perm is not None:
        raise ValueError(f"{name}: pack contains a permutation but permutation is disabled")


class DuQuantLinear(nn.Module):
    """DuQuant quantized linear layer with W4A8 fake quantization."""

    def __init__(self, base: nn.Linear, name: str, cfg: DuQuantConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.register_buffer("_weight", base.weight.detach().clone())
        # Some upstream policy code inspects ``linear.weight.dtype`` to choose
        # an autocast path.  Preserve only zero-sized dtype/device metadata so
        # that inspection remains valid after the FP weight itself is released.
        self.register_buffer(
            "_weight_metadata",
            base.weight.detach().new_empty(0),
            persistent=False,
        )
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
            checkpoint_sha256 = os.environ.get("OPENPI_CHECKPOINT_SHA256")
            if checkpoint_sha256:
                pack.meta["checkpoint_sha256"] = checkpoint_sha256
            save_pack(self.name, pack, cfg.pack_dir)
        _validate_pack_for_layer(
            pack,
            name=self.name,
            in_features=self.in_features,
            out_features=self.out_features,
            cfg=cfg,
        )
        self.pack: PackResult = pack

        # Cache rotation matrices as torch tensors
        if pack.perm is not None:
            self.register_buffer(
                "_perm_cache",
                torch.from_numpy(pack.perm).to(device=self._weight.device, dtype=torch.long),
            )
        else:
            self._perm_cache = None

        # Cache input rotation matrices
        self._R_in_block_indices: List[int] = []
        if pack.R_in_blocks:
            for b, R in pack.R_in_blocks.items():
                buffer_name = f"_R_in_{b}"
                self.register_buffer(
                    buffer_name,
                    torch.from_numpy(R).to(
                        device=self._weight.device, dtype=self._weight.dtype
                    ),
                )
                self._R_in_block_indices.append(b)

        # Cache output rotation matrices
        self._R_out_block_indices: List[int] = []
        if pack.R_out_blocks:
            for b, R in pack.R_out_blocks.items():
                buffer_name = f"_R_out_{b}"
                self.register_buffer(
                    buffer_name,
                    torch.from_numpy(R).to(
                        device=self._weight.device, dtype=self._weight.dtype
                    ),
                )
                self._R_out_block_indices.append(b)

        # Store metadata
        self._block_size = int(pack.meta.get("block_size", 16))
        self._block_out_size = int(pack.meta.get("block_out_size", self._block_size))

        # Calibrator for activation
        # The language/VLM path runs once per policy request, while the action
        # expert runs once per flow step.  Scale ``max_batches`` accordingly so
        # "32 calibration batches" means 32 complete policy requests for both.
        denoising_steps = int(os.environ.get("OPENPI_DUQUANT_DENOISING_STEPS", 10))
        calls_per_batch = denoising_steps if ".gemma_expert." in name else 1
        self._calibration_calls_per_batch = calls_per_batch
        self.calibrator = (
            PercentileCalibrator(
                percentile=cfg.act_percentile,
                max_batches=cfg.calib_batches * calls_per_batch,
            )
            if self.cfg.act_bits > 0 and not self.cfg.act_dynamic
            else None
        )
        self.register_buffer("_act_scale", None)
        self._act_scale_initialized = False

        self._triton_enabled = os.environ.get("OPENPI_DUQUANT_TRITON", "0") not in (
            "0", "false", "False"
        )

        # Cache transformed weight only for the eager diagnostic path.  The
        # real-quant path packs one layer at a time and never allocates a
        # model-sized FP transformed-weight cache.
        self._cached_weight_key: Optional[Tuple[str, torch.dtype]] = None
        self.register_buffer(
            "_W_t", None if self._triton_enabled else torch.zeros_like(self._weight)
        )
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype))

        # Pre-cache quantized weights
        self._precache_weight = (
            not self._triton_enabled
            and os.environ.get("OPENPI_DUQUANT_PRECACHE_WEIGHTS", "1")
            not in ("0", "false", "False")
        )
        if self._precache_weight:
            self.register_buffer("_W_t_quantized", torch.zeros_like(self._weight))
        else:
            self._W_t_quantized = None
        self._weight_quantized_cached = False

        self._bias_rot: Optional[torch.Tensor] = None
        self._debug_enabled = os.environ.get("OPENPI_DUQUANT_DEBUG", "0") not in ("0", "false", "False")
        self._debug_forward_logged = False

        # Triton fused fast path (QuantVLA "fast" variant). Math is identical
        # to the eager path; the kernel is used only after act-scale
        # calibration has frozen (see forward()).
        self._triton_cache_key = None
        self._triton_perm32 = None
        self._triton_rin = None
        self._triton_rout = None
        self.register_buffer(
            "_W_packed_u4",
            torch.zeros(
                self.out_features,
                (self.in_features + 1) // 2,
                dtype=torch.uint8,
                device=self._weight.device,
            ),
        )
        self.register_buffer(
            "_w_input_gain",
            None,
        )
        self._fused_ready = False
        self._inference_only_ready = False
        self._hessian_w4_loaded = False

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
        """Expose FP weight before finalization, then zero-sized metadata only."""
        if self._weight is None:
            if self._inference_only_ready:
                return self._weight_metadata
            raise RuntimeError(f"{self.name}: FP weight is unavailable")
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        if self._weight is None:
            raise RuntimeError(f"{self.name}: cannot mutate a finalized real-quant layer")
        with torch.no_grad():
            self._weight.copy_(value)

    @property
    def act_scale_ready(self) -> bool:
        return (
            self.cfg.act_bits <= 0
            or bool(self.cfg.act_dynamic)
            or bool(self._act_scale_initialized)
        )

    def set_act_scale(self, scale: torch.Tensor) -> None:
        expected = self.in_features
        if tuple(scale.shape) not in ((expected,), (4, expected)):
            raise ValueError(
                f"{self.name}: activation scale shape {tuple(scale.shape)} "
                f"must be ({expected},) or (4,{expected})"
            )
        value = scale.detach().to(device=self._weight.device, dtype=self._weight.dtype).clone()
        if not torch.isfinite(value).all() or torch.any(value <= 0):
            raise ValueError(f"{self.name}: activation scale must be positive and finite")
        if self._act_scale is None:
            self._act_scale = value
        else:
            self._act_scale.copy_(value)
        self._act_scale_initialized = True
        if self.calibrator is not None:
            self.calibrator.mark_full()

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

        if self._triton_enabled and self.weight_bits == 4:
            from .duquant_triton import pack_w4_nibbles

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
                f"[OPENPI-DUQUANT][CACHE] {self.name} device={self._weight.device} dtype={self._weight.dtype} "
                f"Wbits={self.weight_bits} Abits={self.cfg.act_bits} block_in={self.cfg.block_size} "
                f"permute={self.pack.perm is not None} row_rot={self.cfg.row_rot_mode}"
            )
            if self._weight_quantized_cached:
                logging.info(f"[OPENPI-DUQUANT][CACHE] {self.name} pre-quantized weights cached")

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
                self.bias = nn.Parameter(self._errorfold_base_bias.clone())
            else:
                self.bias.copy_(self._errorfold_base_bias)

    def reset_errorfold(self) -> None:
        if not self._hessian_w4_loaded or self._inference_only_ready:
            raise RuntimeError(f"{self.name}: cannot reset ErrorFold in current state")
        self._w_scales.copy_(self._hessian_base_scales)
        self._w_input_gain = None
        with torch.no_grad():
            if self._errorfold_base_bias is None:
                self.bias = None
            elif self.bias is None:
                self.bias = nn.Parameter(self._errorfold_base_bias.clone())
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

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        if self.cfg.act_dynamic:
            with torch.no_grad():
                x_abs = torch.abs(x.detach().to(torch.float32))
                x2d = x_abs.reshape(-1, x_abs.shape[-1])
                scale = torch.clamp(
                    x2d.amax(dim=0) / qmax(self.cfg.act_bits), min=1e-6
                )
                return scale.to(dtype=x.dtype, device=x.device)

        if self._act_scale_initialized:
            if self._act_scale.ndim == 1:
                return self._act_scale
            from .dit_step_context import get_current_dit_step, get_total_dit_steps

            step = get_current_dit_step()
            total = get_total_dit_steps()
            if step is None or total != 4 or not 0 <= int(step) < 4:
                raise RuntimeError(
                    f"{self.name}: four-row DiT A8 table requires deterministic step context"
                )
            return self._act_scale[int(step)]

        with torch.no_grad():
            if self.calibrator is not None and not self.calibrator.is_full():
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

            if not self._act_scale_initialized:
                # Provisional scale for this forward only.  Do not cache/freeze it:
                # formal static A8 calibration requires all configured activation
                # batches before the persistent scale becomes ready.
                x_abs = torch.abs(x.detach().to(torch.float32))
                channels = x_abs.shape[-1]
                x2d = x_abs.reshape(-1, channels)
                p_vec = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
                max_q = qmax(self.cfg.act_bits)
                return torch.clamp(p_vec / max_q, min=1e-6).to(
                    dtype=x.dtype, device=x.device
                )

        return self._act_scale

    def _prepare_triton_caches(self) -> bool:
        """Build (once) the stacked rotation/permutation tensors for the
        fused kernel. Returns False if the layer shape is unsupported."""
        if self._inference_only_ready and self._triton_cache_key is not None:
            return True
        key = (str(self._weight.device), self._weight.dtype, int(self._block_size))
        if self._triton_cache_key == key:
            return True
        try:
            B = self._block_size
            perm32 = None
            if self.pack.perm is not None:
                perm32 = self._perm_cache.to(torch.int32).contiguous()
            rin = None
            if self.pack.R_in_blocks:
                if len(self._R_in_block_indices) != self.in_features // B:
                    return False
                rin = torch.stack(
                    [getattr(self, f"_R_in_{b}") for b in range(self.in_features // B)]
                ).contiguous()
            rout = None
            if self.pack.R_out_blocks and self.cfg.row_rot_mode == "restore":
                if len(self._R_out_block_indices) != self.out_features // B:
                    return False
                rout = torch.stack(
                    [getattr(self, f"_R_out_{b}") for b in range(self.out_features // B)]
                ).contiguous()
            self._triton_perm32 = perm32
            self._triton_rin = rin
            self._triton_rout = rout
            self._triton_cache_key = key
            return True
        except Exception:
            return False

    def _triton_fast_ok(self, x: torch.Tensor) -> bool:
        if not self._triton_enabled or not x.is_cuda:
            return False
        if x.dtype not in (torch.float16, torch.bfloat16):
            return False
        B = self._block_size
        if self.in_features % B != 0 or self.out_features % B != 0:
            return False
        # The fused GEMM uses 64-wide tiles.
        if self.in_features % 64 != 0 or self.out_features % 64 != 0:
            return False
        if not self._fused_ready:
            return False
        # During act-scale calibration the eager path must observe the
        # rotated input; the fused path only handles the frozen static scale.
        if (
            self.cfg.act_bits > 0
            and not self.cfg.act_dynamic
            and not self._act_scale_initialized
        ):
            return False
        return self._prepare_triton_caches()

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        from .duquant_triton import duquant_linear_fused_w4

        x2 = x.reshape(-1, self.in_features)
        sa = self._get_act_scale(x) if self.cfg.act_bits > 0 else None
        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            rout = self._triton_rout
            bias_arg = self.bias
        else:
            rout = None
            bias_arg = (
                self._bias_rot
                if self.cfg.row_rot_mode == "propagate" and self._bias_rot is not None
                else self.bias
            )
        y = duquant_linear_fused_w4(
            x2,
            self._W_packed_u4,
            self._w_scales,
            bias_arg,
            self._triton_perm32,
            self._triton_rin,
            sa,
            rout,
            weight_input_gain=self._w_input_gain,
            B=self._block_size,
        )
        return y.reshape(*x.shape[:-1], self.out_features)

    def finalize_real_quant(self) -> int:
        """Freeze true W4 residency and discard all FP weight-sized buffers."""
        if self._inference_only_ready:
            return int(self._W_packed_u4.numel())
        if not self._triton_enabled or self.weight_bits != 4:
            raise RuntimeError(f"{self.name}: real-quant finalization requires Triton W4")
        if (
            self.cfg.act_bits > 0
            and not self.cfg.act_dynamic
            and not self._act_scale_initialized
        ):
            raise RuntimeError(f"{self.name}: static A8 scale is not ready")
        if self.in_features % 64 or self.out_features % 64:
            raise RuntimeError(f"{self.name}: real-quant kernel requires 64-aligned shapes")
        self._maybe_update_weight_cache()
        if not self._prepare_triton_caches() or not self._fused_ready:
            raise RuntimeError(f"{self.name}: packed W4 caches are not ready")
        self._weight = None
        self._W_t = None
        self._W_t_quantized = None
        self._weight_quantized_cached = False
        if hasattr(self, "_hessian_base_scales"):
            self._hessian_base_scales = None
        self._errorfold_base_bias = None
        self.calibrator = None
        self._inference_only_ready = True
        return int(self._W_packed_u4.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Probe-only true skip path.  A mixed deployment leaves skipped layers
        # as native nn.Linear modules, so weight_bits=0 must be mathematically
        # identical to that native FP16 path: no rotation, no A8, and no
        # transformed-weight cache.  Without this bypass, single-layer probes
        # compare against an all-A8 reference that can never occur in the
        # executable mixed plan.
        if self.weight_bits == 0:
            return torch.nn.functional.linear(x, self._weight, self.bias)

        # Weight cache must be current for both paths (idempotent call).
        self._maybe_update_weight_cache()

        # QuantVLA "fast" variant: single Triton kernel, math identical to
        # the eager path below.
        if self._triton_fast_ok(x):
            return self._forward_triton(x)
        if self._inference_only_ready:
            raise RuntimeError(f"{self.name}: finalized real-quant layer cannot fall back")

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
        if self._weight_quantized_cached:
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
                f"[OPENPI-DUQUANT][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y_lin.shape)} "
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


def wrap_duquant(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: DuQuantConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    dry_run: bool = False,
) -> List[str]:
    """Wrap selected layers with DuQuant quantization."""
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
    listed = 0
    wrapped_names: List[str] = []
    expected_block = os.environ.get("OPENPI_DUQUANT_EXPECT_BLOCK")
    quiet = os.environ.get("OPENPI_DUQUANT_QUIET", "0") not in ("0", "false", "False", "")
    for name in layer_names:
        # Skip action head by default unless explicitly requested
        if os.environ.get("OPENPI_DUQUANT_INCLUDE_ACTION_HEAD", "0") in ("0", "false", "False"):
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
        if dry_run:
            msg = (
                f"[OPENPI-DUQUANT][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} "
                f"block_in={cfg.block_size} block_out={cfg.block_out_size} row_rot={cfg.row_rot_mode}"
            )
            print(msg)
            listed += 1
            continue
        dq = DuQuantLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, dq)
        # Use actual block sizes from pack (not cfg defaults)
        actual_block_in = dq._block_size
        actual_block_out = dq._block_out_size
        if expected_block is not None:
            expected = int(expected_block)
            if actual_block_in != expected or actual_block_out != expected:
                raise RuntimeError(
                    f"{name}: pack block {actual_block_in}/{actual_block_out} != expected {expected}"
                )
        if not quiet:
            print(
                f"[OPENPI-DUQUANT][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) -> DuQuantLinear "
                f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} block_in={actual_block_in} block_out={actual_block_out} row_rot={cfg.row_rot_mode}"
            )
        replaced += 1
        wrapped_names.append(name)
    if dry_run:
        print(f"[OPENPI-DUQUANT] Dry-run total layers listed: {listed}")
    else:
        print(f"[OPENPI-DUQUANT] Total layers replaced: {replaced}")
    return wrapped_names


def enable_duquant_if_configured(model: nn.Module) -> dict:
    """
    Entry point to enable DuQuant based on environment variables.

    Activation conditions:
    - If OPENPI_DUQUANT_DRYRUN is set => dry-run listing only
    - Or if any OPENPI_DUQUANT_* variable (other than PACKDIR) is set => perform replacement
    - Otherwise do nothing
    """
    env = os.environ
    keys = [k for k in env.keys() if k.startswith("OPENPI_DUQUANT_")]
    activate = any(k not in ("OPENPI_DUQUANT_PACKDIR",) for k in keys)
    if not activate:
        runtime = {"enabled": False, "candidate_layers": 0, "wrapped_layers": 0}
        setattr(model, "_openpi_duquant_runtime", runtime)
        return runtime

    # Scope defaults to empty (search entire model)
    scope = env.get("OPENPI_DUQUANT_SCOPE", "")
    whitelist = env.get("OPENPI_DUQUANT_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None

    # Default: quantize LLM + DiT MLP layers (pi0.5: PaliGemma LLM all linear + Gemma expert MLP)
    # LLM: paligemma_with_expert.paligemma.model.language_model.layers.<i>.(self_attn.(q|k|v|o)_proj | mlp.(gate|up|down)_proj) -> 126
    # Expert MLP: paligemma_with_expert.gemma_expert.model.layers.<i>.mlp.(gate|up|down)_proj -> 54
    inc = env.get(
        "OPENPI_DUQUANT_INCLUDE",
        (
            r".*(?:"
            r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
            r"|"
            r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)"
            r").*"
        ),
    )
    # Exclude vision encoder, embeddings, auxiliary projectors
    exc = env.get(
        "OPENPI_DUQUANT_EXCLUDE",
        (
            r"(?:^|\.)"
            r"(?:vision_tower|vision|radio|norm|ln|layernorm|embed_tokens|lm_head|action_in_proj|action_out_proj|time_mlp)"
            r"(?:\.|$)"
        ),
    )

    per_layer_wbits = _parse_per_layer_wbits(env.get("OPENPI_DUQUANT_WBITS"))
    dry_run = env.get("OPENPI_DUQUANT_DRYRUN", "0") not in ("0", "false", "False")

    cfg = DuQuantConfig()

    candidate_targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=None,
        blacklist=None,
    )
    candidate_names = [name for name, _ in candidate_targets]
    candidate_inventory_sha256 = hashlib.sha256(
        ("\n".join(sorted(candidate_names)) + "\n").encode("utf-8")
    ).hexdigest()
    plan_path = env.get("OPENPI_DUQUANT_PLAN")
    plan = None
    if plan_path:
        if whitelist_list:
            raise ValueError("OPENPI_DUQUANT_PLAN and OPENPI_DUQUANT_LAYERS are mutually exclusive")
        plan = load_quant_plan(plan_path)
        strict = env.get("OPENPI_DUQUANT_PLAN_STRICT", "1") not in ("0", "false", "False")
        validate_plan_inventory(plan, candidate_names, require_complete=strict)
        whitelist_list = sorted(plan.quantized_bits)
        per_layer_wbits = dict(plan.quantized_bits)

    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    print(f"[OPENPI-DUQUANT] SCOPE filter: '{scope}'")
    print(f"[OPENPI-DUQUANT] Candidate Linear layers: {len(candidate_names)}")
    print(f"[OPENPI-DUQUANT] Matched Linear layers: {len(layer_names)}")

    if len(layer_names) == 0 and scope:
        # Debug: print some layer names to help diagnose
        all_linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        print(f"[OPENPI-DUQUANT] DEBUG: Total Linear layers in model: {len(all_linears)}")
        print(f"[OPENPI-DUQUANT] DEBUG: First 10 Linear layer names:")
        for name, _ in all_linears[:10]:
            print(f"[OPENPI-DUQUANT] DEBUG:   {name}")
        if scope:
            matching_prefix = [n for n, _ in all_linears if n.startswith(scope.rstrip('.'))]
            print(f"[OPENPI-DUQUANT] DEBUG: Layers matching prefix '{scope.rstrip('.')}': {len(matching_prefix)}")
            if matching_prefix:
                for name in matching_prefix[:5]:
                    print(f"[OPENPI-DUQUANT] DEBUG:   {name}")

    if dry_run:
        wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=True)
        return {
            "enabled": False,
            "dry_run": True,
            "candidate_layers": len(candidate_names),
            "planned_layers": len(layer_names),
            "plan_sha256": plan.sha256 if plan is not None else None,
        }

    wrapped_names = wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=False)
    expected_wrapped = env.get("OPENPI_DUQUANT_EXPECT_WRAPPED")
    if expected_wrapped is not None and len(wrapped_names) != int(expected_wrapped):
        raise RuntimeError(
            f"wrapped layer count {len(wrapped_names)} != expected {int(expected_wrapped)}"
        )

    wrapped_modules = [
        module for module in model.modules() if isinstance(module, DuQuantLinear)
    ]
    triton_values = {bool(module._triton_enabled) for module in wrapped_modules}
    if len(triton_values) != 1:
        raise RuntimeError("OpenPI aligned quantization cannot mix eager and Triton execution")
    triton_enabled = next(iter(triton_values))
    runtime = {
        "enabled": True,
        "candidate_layers": len(candidate_names),
        "wrapped_layers": len(wrapped_names),
        "wrapped_layer_names": wrapped_names,
        "plan_path": str(plan.path) if plan is not None else None,
        "plan_sha256": plan.sha256 if plan is not None else None,
        "weight_bits": cfg.weight_bits,
        "act_bits": cfg.act_bits,
        "act_dynamic": bool(cfg.act_dynamic),
        "block_in": cfg.block_size,
        "block_out": cfg.block_out_size,
        "act_percentile": cfg.act_percentile,
        "calib_batches": cfg.calib_batches,
        "denoising_steps": int(env.get("OPENPI_DUQUANT_DENOISING_STEPS", 10)),
        "candidate_inventory_sha256": candidate_inventory_sha256,
        "execution_backend": (
            "triton_w4_nibble_dequant_fp16_gemm"
            if triton_enabled else "fake_quant_fp16_gemm"
        ),
        "integer_gemm": False,
        "packed_low_bit_residency": False,
        "packed_weight_bytes": 0,
        "fp_weight_sized_buffers": len(wrapped_modules),
    }
    if env.get("OPENPI_DUQUANT_PACK_MANIFEST_SHA256"):
        runtime["pack_manifest_sha256"] = env["OPENPI_DUQUANT_PACK_MANIFEST_SHA256"]
    if env.get("OPENPI_DUQUANT_ATTEST_PERMUTE", "0") not in ("0", "false", "False", ""):
        runtime["enable_permute"] = bool(cfg.enable_permute)
    setattr(model, "_openpi_duquant_runtime", runtime)

    scale_path = env.get("OPENPI_DUQUANT_ACT_SCALE_PATH")
    if scale_path and Path(scale_path).is_file():
        expected_scale_metadata = {
            "plan_sha256": runtime["plan_sha256"],
            "wrapped_layers": runtime["wrapped_layers"],
            "act_percentile": runtime["act_percentile"],
            "calib_batches": runtime["calib_batches"],
            "denoising_steps": runtime["denoising_steps"],
            "candidate_inventory_sha256": runtime["candidate_inventory_sha256"],
        }
        if env.get("OPENPI_DUQUANT_EXPECT_PERMUTE_METADATA", "0") not in (
            "0", "false", "False", ""
        ):
            expected_scale_metadata["enable_permute"] = runtime["enable_permute"]
        for metadata_key, env_key in (
            ("checkpoint_sha256", "OPENPI_CHECKPOINT_SHA256"),
            ("calibration_buffer_sha256", "OPENPI_DUQUANT_CALIB_BUFFER_SHA256"),
        ):
            if env.get(env_key):
                expected_scale_metadata[metadata_key] = env[env_key]
        scale_artifact = load_act_scales(
            model,
            scale_path,
            expected_metadata=expected_scale_metadata,
        )
        runtime["act_scale_path"] = str(Path(scale_path).resolve())
        runtime["act_scale_sha256"] = scale_artifact["npz_sha256"]
        runtime["act_scales_ready"] = True
    else:
        runtime["act_scale_path"] = str(Path(scale_path).resolve()) if scale_path else None
        runtime["act_scales_ready"] = static_scales_ready(model)
        require_scale = env.get("OPENPI_DUQUANT_REQUIRE_ACT_SCALE", "0") not in (
            "0", "false", "False", "",
        )
        if require_scale and not cfg.act_dynamic:
            raise RuntimeError(f"required static A8 scale file is missing: {scale_path}")
    hessian_path = env.get("OPENPI_DUQUANT_HESSIAN_W4_PATH")
    if hessian_path:
        runtime.update(
            load_hessian_w4(
                model,
                hessian_path,
                errorfold_path=env.get("OPENPI_ERRORFOLD_PATH"),
            )
        )
    return runtime


def iter_duquant_layers(model: nn.Module) -> List[Tuple[str, DuQuantLinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, DuQuantLinear)]


def finalize_real_quant(model: nn.Module) -> Dict[str, int | bool]:
    """Finalize all OpenPI DuQuant layers to inference-only packed W4."""
    layers = [module for _, module in iter_duquant_layers(model)]
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
    runtime = getattr(model, "_openpi_duquant_runtime", {})
    runtime.update(
        {
            "packed_low_bit_residency": True,
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
            "execution_backend": "triton_w4_nibble_dequant_fp16_gemm",
        }
    )
    setattr(model, "_openpi_duquant_runtime", runtime)
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
    """Load one frozen, inventory-exact group-64 W4 artifact for pi0.5."""
    artifact = Path(path).expanduser().resolve()
    sidecar = Path(str(artifact) + ".json")
    if not artifact.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Hessian W4 artifact is incomplete: {artifact}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 3 or metadata.get("group_size") != 64:
        raise ValueError("unsupported Hessian W4 artifact schema/group")
    if metadata.get("protocol_id") != "quantvla-gr00t-pi05-errorfold-v4":
        raise ValueError("Hessian W4 artifact protocol drift")
    actual_hash = sha256_file(artifact)
    if metadata.get("npz_sha256") != actual_hash:
        raise ValueError("Hessian W4 artifact hash mismatch")
    layers = iter_duquant_layers(model)
    names = [name for name, _ in layers]
    if metadata.get("layer_names") != names:
        raise ValueError("Hessian W4 layer inventory does not match wrapped pi0.5 model")
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
    runtime = getattr(model, "_openpi_duquant_runtime", {})
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
    setattr(model, "_openpi_duquant_runtime", runtime)
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
    layers = iter_duquant_layers(model)
    for name, module in layers:
        entry = correction_layers.get(name)
        if not entry or entry.get("kind") != "linear":
            raise ValueError(f"ErrorFold lacks Linear correction for {name}")
        module.reset_errorfold()
        module.fold_errorfold(
            torch.as_tensor(entry["effective_gain"]),
            torch.as_tensor(entry["effective_bias"]),
        )
    runtime = getattr(model, "_openpi_duquant_runtime", {})
    runtime.update(
        {
            "errorfold_path": str(correction_path),
            "selector_free": True,
            "offline_candidate_reuse": True,
        }
    )
    setattr(model, "_openpi_duquant_runtime", runtime)
    return runtime


def static_scales_ready(model: nn.Module) -> bool:
    layers = iter_duquant_layers(model)
    return bool(layers) and all(module.act_scale_ready for _, module in layers)


def _scale_sidecar(path: str | Path) -> Path:
    scale_path = Path(path)
    return Path(str(scale_path) + ".json")


def save_act_scales(model: nn.Module, path: str | Path, metadata: Optional[dict] = None) -> dict:
    layers = iter_duquant_layers(model)
    if not layers:
        raise ValueError("cannot save A8 scales: model has zero DuQuantLinear layers")
    missing = [name for name, module in layers if not module.act_scale_ready]
    if missing:
        raise RuntimeError(f"cannot save A8 scales: {len(missing)} layers are not ready")
    scale_path = Path(path).expanduser().resolve()
    scale_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"layer_names": np.asarray([name for name, _ in layers])}
    for index, (_, module) in enumerate(layers):
        arrays[f"scale_{index:04d}"] = module._act_scale.detach().to(torch.float32).cpu().numpy()
    tmp_path = Path(str(scale_path) + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp_path.replace(scale_path)
    runtime = getattr(model, "_openpi_duquant_runtime", {})
    artifact_metadata = {
        "plan_sha256": runtime.get("plan_sha256"),
        "wrapped_layers": len(layers),
        "act_percentile": runtime.get("act_percentile"),
        "calib_batches": runtime.get("calib_batches"),
        "denoising_steps": runtime.get("denoising_steps"),
        "candidate_inventory_sha256": runtime.get("candidate_inventory_sha256"),
        **dict(metadata or {}),
    }
    if os.environ.get("OPENPI_DUQUANT_STRICT_ARTIFACTS", "0") not in ("0", "false", "False", ""):
        required = (
            "plan_sha256",
            "checkpoint_sha256",
            "calibration_buffer_sha256",
            "wrapped_layers",
            "act_percentile",
            "calib_batches",
            "denoising_steps",
            "candidate_inventory_sha256",
        )
        missing_metadata = [key for key in required if artifact_metadata.get(key) in (None, "")]
        if missing_metadata:
            raise ValueError(f"A8 scale metadata missing required fields: {missing_metadata}")
    payload = {
        "schema_version": 1,
        "layer_names": [name for name, _ in layers],
        "npz_sha256": sha256_file(scale_path),
        "metadata": artifact_metadata,
    }
    sidecar = _scale_sidecar(scale_path)
    sidecar_tmp = Path(str(sidecar) + ".tmp")
    sidecar_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar_tmp.replace(sidecar)
    if isinstance(runtime, dict):
        runtime["act_scale_path"] = str(scale_path)
        runtime["act_scales_ready"] = True
    return payload


def load_act_scales(
    model: nn.Module,
    path: str | Path,
    *,
    expected_metadata: Optional[dict] = None,
) -> dict:
    scale_path = Path(path).expanduser().resolve()
    sidecar = _scale_sidecar(scale_path)
    if not scale_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"A8 scale or sidecar is missing: {scale_path}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported A8 scale schema")
    metadata = payload.get("metadata") or {}
    stored_hash = payload.get("npz_sha256")
    actual_hash = sha256_file(scale_path)
    if stored_hash != actual_hash:
        raise ValueError(f"A8 npz hash mismatch: {actual_hash} != {stored_hash}")
    for key, expected in (expected_metadata or {}).items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"A8 scale metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )
    layers = iter_duquant_layers(model)
    names = [name for name, _ in layers]
    if payload.get("layer_names") != names:
        raise ValueError("A8 scale layer inventory does not match wrapped model")
    with np.load(scale_path, allow_pickle=False) as archive:
        stored_names = [str(value) for value in archive["layer_names"].tolist()]
        if stored_names != names:
            raise ValueError("A8 npz layer inventory does not match sidecar")
        for index, (name, module) in enumerate(layers):
            key = f"scale_{index:04d}"
            if key not in archive:
                raise ValueError(f"A8 npz is missing {key} for {name}")
            module.set_act_scale(torch.from_numpy(np.asarray(archive[key])))
    return payload
