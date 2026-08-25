"""Triton-fused forward for DuQuantLinear (QuantVLA "fast" variant).

Math is IDENTICAL to the eager path in duquant_layers.py (restore mode):

    x_t   = x @ P @ R_in              (permutation + per-16-block rotation)
    x_tq  = fake_quant_sym(x_t, s_a)  (static per-channel int8 symmetric)
    y_lin = x_tq @ W_tq^T             (fp16/bf16 GEMM, fp32 accumulate)
    y     = y_lin @ R_out + bias      (per-16-block restore)

Implemented as a three-kernel pipeline (the single-kernel K=16 design was
~7x slower than eager because of tiny tensor-core dots):

  1. _input_transform_quant_kernel : x -> x_tq   (memory-bound, ~2 passes)
  2. torch F.linear                : x_tq @ W_tq^T (bit-identical to eager)
  3. _output_restore_kernel        : y_lin @ R_out + bias (memory-bound)

The kernels are only used AFTER activation-scale calibration has frozen
(before that, _get_act_scale observes the rotated input, so the layer stays
on the eager path). Enable with OPENPI_DUQUANT_TRITON=1 on the quant server.
"""

import torch
import triton
import triton.language as tl

try:  # IEEE round-half-even (matches torch.round semantics)
    from triton.language.extra import libdevice as _ld
    _rint = _ld.rint
except Exception:  # pragma: no cover
    _rint = tl.math.round


@triton.jit
def _cast(x, DTYPE: tl.constexpr):
    if DTYPE == "fp16":
        return x.to(tl.float16)
    else:
        return x.to(tl.bfloat16)


@triton.jit
def _input_transform_quant_kernel(
    x_ptr, y_ptr, perm_ptr, rin_ptr, sa_ptr,
    M, I: tl.constexpr, B: tl.constexpr, BM: tl.constexpr, DTYPE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    HAS_PERM: tl.constexpr, HAS_RIN: tl.constexpr, HAS_SA: tl.constexpr,
):
    pid = tl.program_id(0)   # over M
    pid_g = tl.program_id(1)  # over groups of BLOCKS_PER_PROG input blocks
    rm = pid * BM + tl.arange(0, BM)
    rmask = rm < M
    for bb in tl.static_range(BLOCKS_PER_PROG):
        pid_b = pid_g * BLOCKS_PER_PROG + bb
        if HAS_PERM:
            src = tl.load(perm_ptr + pid_b * B + tl.arange(0, B))
        else:
            src = pid_b * B + tl.arange(0, B)
        x = tl.load(x_ptr + rm[:, None] * I + src[None, :], mask=rmask[:, None], other=0.0)
        if HAS_RIN:
            roff = tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :]
            rin = tl.load(rin_ptr + pid_b * B * B + roff)
            xr = tl.dot(x.to(tl.float32), rin.to(tl.float32))  # [BM,16] @ [16,16]
        else:
            xr = x.to(tl.float32)
        # Mirror the eager path's dtype rounding chain (bf16/fp16 x_t, dtype
        # division, then quantization).
        xr = _cast(xr, DTYPE)
        if HAS_SA:
            s = tl.load(sa_ptr + pid_b * B + tl.arange(0, B))
            s = tl.maximum(s, 1e-8)
            q = _rint(_cast(xr / s[None, :], DTYPE).to(tl.float32))
            q = tl.minimum(tl.maximum(q, -128.0), 127.0)
            xq = _cast(q * s[None, :], DTYPE)
        else:
            xq = xr
        tl.store(y_ptr + rm[:, None] * I + (pid_b * B + tl.arange(0, B))[None, :],
                 xq, mask=rmask[:, None])


@triton.jit
def _output_restore_kernel(
    y_ptr, out_ptr, bias_ptr, rout_ptr,
    M, O: tl.constexpr, B: tl.constexpr, BM: tl.constexpr, DTYPE: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
    HAS_ROUT: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rmask = rm < M
    for bb in tl.static_range(BLOCKS_PER_PROG):
        pid_b = pid_g * BLOCKS_PER_PROG + bb
        rn = pid_b * B + tl.arange(0, B)
        y = tl.load(y_ptr + rm[:, None] * O + rn[None, :], mask=rmask[:, None], other=0.0)
        if HAS_ROUT:
            roff = tl.arange(0, B)[:, None] * B + tl.arange(0, B)[None, :]
            rout = tl.load(rout_ptr + pid_b * B * B + roff)
            y = tl.dot(y.to(tl.float32), rout.to(tl.float32))
            y = _cast(y, DTYPE)
        if HAS_BIAS:
            b = tl.load(bias_ptr + rn)
            y = _cast(y + b[None, :].to(tl.float32), DTYPE)
        tl.store(out_ptr + rm[:, None] * O + rn[None, :], y, mask=rmask[:, None])


def duquant_linear_fused(
    x: torch.Tensor,
    W_tq: torch.Tensor,
    bias: torch.Tensor,
    perm32,
    rin_stack,
    sa,
    rout_stack,
    B: int = 16,
    BM: int = 64,
):
    """Fused DuQuantLinear forward (three-kernel pipeline). CUDA tensors only.

    x:         [M, I] fp16/bf16
    W_tq:      [O, I] precached quantized weight WITH row rotations applied
    bias:      [O] (already row-rotated for propagate mode)
    perm32:    [I] int32 permutation or None
    rin_stack: [I//B, B, B] input rotations or None
    sa:        [I] static per-channel activation scale or None
    rout_stack: [O//B, B, B] output restore rotations or None
    """
    M, I = x.shape
    O = W_tq.shape[0]
    x = x.contiguous()
    BPG_IN = 16  # 16-blocks handled per program (launch-overhead reduction)
    BPG_OUT = 16
    x_tq = torch.empty_like(x)
    _input_transform_quant_kernel[(triton.cdiv(M, BM), I // (B * BPG_IN))](
        x, x_tq,
        perm32 if perm32 is not None else x,
        rin_stack if rin_stack is not None else x,
        sa if sa is not None else x,
        M, I=I, B=B, BM=BM, DTYPE="fp16" if x.dtype == torch.float16 else "bf16",
        BLOCKS_PER_PROG=BPG_IN,
        HAS_PERM=perm32 is not None,
        HAS_RIN=rin_stack is not None,
        HAS_SA=sa is not None,
    )
    # GEMM via torch (bit-identical to the eager path; the Triton kernels
    # only replace the per-block transform/quantize/restore overhead).
    y_lin = torch.nn.functional.linear(x_tq, W_tq, None)
    y = torch.empty((M, O), dtype=x.dtype, device=x.device)
    _output_restore_kernel[(triton.cdiv(M, BM), O // (B * BPG_OUT))](
        y_lin, y,
        bias if bias is not None else y_lin,
        rout_stack if rout_stack is not None else y_lin,
        M, O=O, B=B, BM=BM, DTYPE="fp16" if x.dtype == torch.float16 else "bf16",
        BLOCKS_PER_PROG=BPG_OUT,
        HAS_ROUT=rout_stack is not None,
        HAS_BIAS=bias is not None,
    )
    return y
