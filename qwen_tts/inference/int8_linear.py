# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Int8-weight linear layers for batch-size-1 decoding (W8A16).

Every decode step of the talker and the code predictor multiplies a single
activation vector with every weight matrix, so the step time is set by how
fast the weights can be read. On an RTX 4070 Ti cuBLAS already reads bf16
weights at ~500 GB/s, i.e. the memory bandwidth; storing them as int8 with one
scale per output row halves the bytes.

The matrix-vector product is a small Triton kernel with a fixed launch
configuration. torchao's int8 path reaches similar speed only after Inductor's
coordinate-descent autotuning, which took minutes at every server start.
Inputs with more than a few rows (long prompts) dequantize and use cuBLAS.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Up to this many rows (tokens) the kernel reads the weights once per row;
# beyond it a dequantized cuBLAS matmul is cheaper
MAX_GEMV_ROWS = 16
BLOCK_N = 4
BLOCK_K = 512


@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """y[m, n] = (sum_k x[m, k] * w[n, k]) * s[n]; one program per (row block, input row)."""
    pid_n = tl.program_id(0)
    m = tl.program_id(1)
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        col_ok = cols < K
        x = tl.load(x_ptr + m * K + cols, mask=col_ok, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rows[:, None] * K + cols[None, :],
                    mask=row_ok[:, None] & col_ok[None, :], other=0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    scale = tl.load(s_ptr + rows, mask=row_ok, other=0.0).to(tl.float32)
    tl.store(y_ptr + m * N + rows, (acc * scale).to(y_ptr.dtype.element_ty), mask=row_ok)


@torch.library.custom_op("qwen_tts::w8a16_linear", mutates_args=())
def w8a16_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x [..., K] (bf16) @ weight [N, K] (int8) * scale [N] -> [..., N]."""
    rows = x.numel() // x.shape[-1]
    if rows > MAX_GEMV_ROWS:
        return F.linear(x, weight.to(x.dtype) * scale.unsqueeze(1).to(x.dtype))
    n, k = weight.shape
    x2 = x.reshape(rows, k).contiguous()
    y = torch.empty(rows, n, dtype=x.dtype, device=x.device)
    # One configuration for all layer shapes of the 1.7B model: in a sweep on an
    # RTX 4070 Ti (graph-replayed, BLOCK_N 4..32, BLOCK_K 128..512, 2..8 warps)
    # it had the lowest total time, 39 us per pass over all shapes vs. 107 us bf16.
    _w8a16_kernel[(triton.cdiv(n, BLOCK_N), rows)](x2, weight, scale, y, n, k,
                                                   BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    return y.reshape(*x.shape[:-1], n)


@w8a16_linear.register_fake
def _(x, weight, scale):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


class Int8Linear(torch.nn.Module):
    """Drop-in for a bias-free nn.Linear with symmetric per-row int8 weights."""

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("Int8Linear expects a bias-free linear layer")
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        self.register_buffer("weight", torch.round(w / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(linear.weight.dtype))
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w8a16_linear(x, self.weight, self.scale)


def quantize_linears_int8(module: torch.nn.Module) -> int:
    """Replace every bias-free nn.Linear below `module` with Int8Linear. Returns the count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear) and child.bias is None:
            setattr(module, name, Int8Linear(child))
            count += 1
        else:
            count += quantize_linears_int8(child)
    return count
