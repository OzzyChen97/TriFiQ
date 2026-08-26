#!/usr/bin/env python3
"""Triton fused W4-dequant matmul for the GR00T DuQuant path (v1.4 fast-path probe).

The eager DuQuantLinear computes, every forward:

    y = linear(x_t, fake_quantize_sym(W_t, w_scales[:, None], 4))

i.e. weights are dequantized to float on the fly and multiplied in fp16/bf16
tensor cores. This module provides the SAME numerical semantics as a single
fused Triton kernel:

    * W_t is rounded/clamped ONCE into int8 per output row (identical
      rounding to torch.round in fake_quantize_sym),
    * the kernel dequantizes inside (w = w_q * scale) and runs one
      tl.dot accumulation,
    * activation A8 fake-quant, input rotations (R_in), output restore (R_out)
      and bias stay EXACTLY as the eager path — so the A/B difference isolates
      the matmul path only.

Accumulation is fp32 in the kernel (Triton dot semantics); the eager path uses
cuBLAS tensor-core accumulation, so bit-equality is not expected — the A/B
script measures the actual divergence on the real model.

Usage:
    python -m gr00t.quantization.duquant_fused            # selftest (CPU/GPU)
    GR00T_DUQUANT_FUSED=1 ...                              # enable in deployment
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _w4_dequant_matmul_kernel(
    A,  # (M, K) fp16/bf16 — already input-transformed + A8 fake-quantized
    WQ,  # (N, K) int8 quantized weights
    WS,  # (N,) per-output-row scales
    Y,  # (M, N) out
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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
    wq_ptrs = WQ + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - _ * BLOCK_K, other=0.0)
        wq = tl.load(wq_ptrs, mask=offs_k[None, :] < K - _ * BLOCK_K, other=0)
        ws = tl.load(WS + offs_n, mask=offs_n < N, other=1.0)
        w = wq.to(tl.float32) * ws[:, None]
        acc += tl.dot(a, tl.trans(w).to(a.dtype), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        wq_ptrs += BLOCK_K * stride_wk

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def pack_w4_int8(w_t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Round+clamp W_t/scale to the same grid fake_quantize_sym uses, as int8.

    fake_quantize_sym clamps to [-max_q-1, max_q] with max_q = 7 for 4 bits,
    i.e. [-8, 7] — the asymmetric negative bucket is preserved exactly.
    """
    max_q = 7
    q = torch.clamp(torch.round(w_t / scales[:, None]), -max_q - 1, max_q)
    return q.to(torch.int8).contiguous()


def pack_w4_nibbles(w_t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack signed W4 values two-per-byte for inference-only residency.

    The low nibble stores the even input channel and the high nibble stores
    the odd channel.  Signed values use two's-complement representation, so
    the exact DuQuant range [-8, 7] is preserved without an FP weight copy.
    """
    if w_t.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(w_t.shape)}")
    if scales.ndim not in (1, 2) or scales.shape[0] != w_t.shape[0]:
        raise ValueError(
            f"scale shape {tuple(scales.shape)} does not match {w_t.shape[0]} rows"
        )
    if scales.ndim == 1:
        expanded_scales = scales[:, None]
    else:
        if scales.shape[1] != math.ceil(w_t.shape[1] / 64):
            raise ValueError("grouped W4 scales must be one value per group-64")
        expanded_scales = scales.repeat_interleave(64, dim=1)[:, : w_t.shape[1]]
    q = torch.clamp(torch.round(w_t / expanded_scales), -8, 7).to(torch.int8)
    if q.shape[1] & 1:
        q = torch.nn.functional.pad(q, (0, 1))
    q_u4 = torch.bitwise_and(q, 0x0F).to(torch.uint8)
    return torch.bitwise_or(q_u4[:, 0::2], q_u4[:, 1::2] << 4).contiguous()


@triton.jit
def _w4_nibble_dequant_matmul_kernel(
    A,
    WQ,  # (N, ceil(K / 2)) uint8, two signed W4 values per byte
    WS,
    W_INPUT_GAIN,
    Y,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wb,
    stride_wsn,
    stride_wsg,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
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
        ws = tl.load(
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
            w = q.to(tl.float32) * ws * input_gain[None, :]
        else:
            w = q.to(tl.float32) * ws
        acc += tl.dot(a, tl.trans(w).to(a.dtype), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(
        y_ptrs,
        acc.to(Y.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_linear_w4_nibbles(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    input_gain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``x @ dequant(W4)^T`` from a true two-values-per-byte W4 tensor."""
    if w_q.dtype != torch.uint8 or w_q.ndim != 2:
        raise ValueError("packed W4 weight must be a 2-D uint8 tensor")
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    wq2 = w_q.contiguous()
    m, k = x2.shape
    n = wq2.shape[0]
    if wq2.shape[1] != (k + 1) // 2:
        raise ValueError(
            f"packed W4 shape {tuple(wq2.shape)} is incompatible with K={k}"
        )
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
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    block_m, block_n, block_k, group_m = 64, 64, 64, 4
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _w4_nibble_dequant_matmul_kernel[grid](
        x2,
        wq2,
        scale_table,
        input_gain if input_gain is not None else x2,
        y,
        m,
        n,
        k,
        x2.stride(0),
        x2.stride(1),
        wq2.stride(0),
        wq2.stride(1),
        scale_table.stride(0),
        scale_table.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        SCALE_GROUP_SIZE=scale_group_size,
        HAS_INPUT_GAIN=input_gain is not None,
    )
    return y.reshape(*x.shape[:-1], n)


def fused_linear_w4(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """x @ dequant(w_q, scales)^T — Triton fused, fp32 accumulation."""
    orig_dtype = x.dtype
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    wq2 = w_q.contiguous()
    m, k = x2.shape
    n = wq2.shape[0]
    y = torch.empty((m, n), dtype=orig_dtype, device=x.device)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 4
    if k % BLOCK_K != 0:
        # pad-free fallback: run the kernel with masked K (works but slower)
        pass
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    _w4_dequant_matmul_kernel[grid](
        x2, wq2, scales, y, m, n, k,
        x2.stride(0), x2.stride(1), wq2.stride(0), wq2.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return y.reshape(*x.shape[:-1], n)


def eager_linear_w4(x: torch.Tensor, w_t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reference eager path (same math as DuQuantLinear's weight_bits>0 branch)."""
    max_q = 7
    scale_table = (
        scales[:, None]
        if scales.ndim == 1
        else scales.repeat_interleave(64, dim=1)[:, : w_t.shape[1]]
    )
    w_scaled = w_t / scale_table
    w_deq = torch.clamp(torch.round(w_scaled), -max_q - 1, max_q) * scale_table
    return torch.nn.functional.linear(x, w_deq.to(x.dtype), None)


def selftest() -> None:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("[duquant_fused] no CUDA — selftest skipped (kernel needs GPU)")
        return
    for (m, n, k, dt) in ((256, 512, 1536, torch.float16), (256, 512, 1536, torch.bfloat16),
                          (37, 6144, 1536, torch.float16)):
        x = torch.randn(m, k, dtype=dt, device=dev) * 0.1
        w_t = torch.randn(n, k, dtype=torch.float32, device=dev) * 0.05
        scales = torch.rand(n, dtype=torch.float32, device=dev) * 0.1 + 0.01
        w_q = pack_w4_int8(w_t, scales)
        w_q_nibbles = pack_w4_nibbles(w_t, scales)
        y_e = eager_linear_w4(x, w_t, scales)
        y_f = fused_linear_w4(x, w_q, scales)
        y_n = fused_linear_w4_nibbles(x, w_q_nibbles, scales)
        # fp32 reference: the exact dequant matmul in float32
        max_q = 7
        w_deq32 = torch.clamp(torch.round(w_t / scales[:, None]), -max_q - 1, max_q) * scales[:, None]
        y_ref = (x.float() @ w_deq32.T.float())
        denom = y_ref.abs() + 1e-2
        err_e = ((y_e.float() - y_ref).abs() / denom).max().item()
        err_f = ((y_f.float() - y_ref).abs() / denom).max().item()
        err_n = ((y_n.float() - y_ref).abs() / denom).max().item()
        print(f"[duquant_fused] M={m} N={n} K={k} {dt}: eager-vs-fp32ref={err_e:.3e} "
              f"int8-container={err_f:.3e} nibble-packed={err_n:.3e}")
        # the meaningful gate: the fused path must be at least as accurate as
        # the eager tensor-core path (both vs the fp32 reference)
        assert err_f <= err_e * 1.2 + 1e-3, f"fused less accurate than eager: {err_f} vs {err_e}"
        assert err_f < 0.5, f"fused error grossly large: {err_f}"
        assert err_n <= err_e * 1.2 + 1e-3, f"nibble fused less accurate: {err_n} vs {err_e}"
        assert err_n < 0.5, f"nibble fused error grossly large: {err_n}"
    # timing
    x = torch.randn(256, 1536, dtype=torch.float16, device=dev) * 0.1
    w_t = torch.randn(512, 1536, dtype=torch.float32, device=dev) * 0.05
    scales = torch.rand(512, dtype=torch.float32, device=dev) * 0.1 + 0.01
    w_q = pack_w4_int8(w_t, scales)
    import time

    for _ in range(3):
        eager_linear_w4(x, w_t, scales)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        eager_linear_w4(x, w_t, scales)
    torch.cuda.synchronize()
    t_e = (time.time() - t0) / 50
    for _ in range(3):
        fused_linear_w4(x, w_q, scales)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        fused_linear_w4(x, w_q, scales)
    torch.cuda.synchronize()
    t_f = (time.time() - t0) / 50
    print(f"[duquant_fused] eager {t_e*1e3:.2f} ms vs fused {t_f*1e3:.2f} ms "
          f"({t_e/t_f:.2f}x speedup)")
    print("[duquant_fused] selftest OK")


if __name__ == "__main__":
    selftest()
