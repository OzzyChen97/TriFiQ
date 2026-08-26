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

import math

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


@triton.jit
def _w4_nibble_dequant_matmul_kernel(
    A, WQ, WS, W_INPUT_GAIN, Y,
    M, N, K,
    stride_am, stride_ak, stride_wn, stride_wb,
    stride_wsn, stride_wsg, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SCALE_GROUP_SIZE: tl.constexpr,
    HAS_INPUT_GAIN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_k in range(0, tl.cdiv(K, BLOCK_K)):
        k = block_k * BLOCK_K + offs_k
        a = tl.load(a_ptrs, mask=k[None, :] < K, other=0.0)
        packed = tl.load(
            WQ + offs_n[:, None] * stride_wn + (k[None, :] // 2) * stride_wb,
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0,
        )
        shift = (k & 1) * 4
        q_u4 = (packed >> shift[None, :]) & 0x0F
        q = tl.where(q_u4 >= 8, q_u4.to(tl.int32) - 16, q_u4.to(tl.int32))
        scale = tl.load(
            WS
            + offs_n[:, None] * stride_wsn
            + (k[None, :] // SCALE_GROUP_SIZE) * stride_wsg,
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=1.0,
        )
        if HAS_INPUT_GAIN:
            input_gain = tl.load(
                W_INPUT_GAIN + k,
                mask=k < K,
                other=1.0,
            )
            weight = q.to(tl.float32) * scale * input_gain[None, :]
        else:
            weight = q.to(tl.float32) * scale
        acc += tl.dot(a, tl.trans(weight).to(a.dtype), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(Y.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def pack_w4_nibbles(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack signed DuQuant W4 values two per uint8 byte."""
    if scales.ndim == 1:
        scale_table = scales[:, None]
    elif scales.ndim == 2 and scales.shape == (
        weight.shape[0], math.ceil(weight.shape[1] / 64)
    ):
        scale_table = scales.repeat_interleave(64, dim=1)[:, : weight.shape[1]]
    else:
        raise ValueError(f"invalid W4 scale table {tuple(scales.shape)}")
    quant = torch.clamp(torch.round(weight / scale_table), -8, 7).to(torch.int8)
    if quant.shape[1] & 1:
        quant = torch.nn.functional.pad(quant, (0, 1))
    unsigned = torch.bitwise_and(quant, 0x0F).to(torch.uint8)
    return torch.bitwise_or(unsigned[:, 0::2], unsigned[:, 1::2] << 4).contiguous()


def _w4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    input_gain: torch.Tensor | None = None,
) -> torch.Tensor:
    m, k = x.shape
    n = packed_weight.shape[0]
    if packed_weight.shape[1] != (k + 1) // 2:
        raise ValueError("packed W4 shape does not match the input dimension")
    if scales.ndim == 1:
        scale_table = scales.reshape(n, 1).contiguous()
        scale_group_size = k
    elif scales.ndim == 2 and scales.shape == (n, math.ceil(k / 64)):
        scale_table = scales.contiguous()
        scale_group_size = 64
    else:
        raise ValueError(f"invalid W4 scale table {tuple(scales.shape)} for N={n}, K={k}")
    if input_gain is not None:
        if input_gain.ndim != 1 or input_gain.numel() != k:
            raise ValueError(f"invalid W4 input gain {tuple(input_gain.shape)} for K={k}")
        input_gain = input_gain.to(device=x.device, dtype=scales.dtype).contiguous()
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_m, block_n, block_k, group_m = 64, 64, 64, 4
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _w4_nibble_dequant_matmul_kernel[grid](
        x, packed_weight, scale_table,
        input_gain if input_gain is not None else x,
        output,
        m, n, k,
        x.stride(0), x.stride(1),
        packed_weight.stride(0), packed_weight.stride(1),
        scale_table.stride(0), scale_table.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        SCALE_GROUP_SIZE=scale_group_size,
        HAS_INPUT_GAIN=input_gain is not None,
    )
    return output


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


def duquant_linear_fused_w4(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor,
    perm32,
    rin_stack,
    sa,
    rout_stack,
    weight_input_gain=None,
    B: int = 16,
    BM: int = 64,
):
    """DuQuant fused transforms with true nibble-packed W4 residency."""
    M, I = x.shape
    O = packed_weight.shape[0]
    x = x.contiguous()
    blocks_per_program = 16
    x_tq = torch.empty_like(x)
    _input_transform_quant_kernel[
        (triton.cdiv(M, BM), I // (B * blocks_per_program))
    ](
        x, x_tq,
        perm32 if perm32 is not None else x,
        rin_stack if rin_stack is not None else x,
        sa if sa is not None else x,
        M, I=I, B=B, BM=BM,
        DTYPE="fp16" if x.dtype == torch.float16 else "bf16",
        BLOCKS_PER_PROG=blocks_per_program,
        HAS_PERM=perm32 is not None,
        HAS_RIN=rin_stack is not None,
        HAS_SA=sa is not None,
    )
    y_lin = _w4_linear(
        x_tq,
        packed_weight,
        weight_scales,
        input_gain=weight_input_gain,
    )
    y = torch.empty((M, O), dtype=x.dtype, device=x.device)
    _output_restore_kernel[
        (triton.cdiv(M, BM), O // (B * blocks_per_program))
    ](
        y_lin, y,
        bias if bias is not None else y_lin,
        rout_stack if rout_stack is not None else y_lin,
        M, O=O, B=B, BM=BM,
        DTYPE="fp16" if x.dtype == torch.float16 else "bf16",
        BLOCKS_PER_PROG=blocks_per_program,
        HAS_ROUT=rout_stack is not None,
        HAS_BIAS=bias is not None,
    )
    return y
