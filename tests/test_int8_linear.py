# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the int8-weight linear layer (Triton kernel). Skipped without CUDA.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def layer():
    from qwen_tts.inference.int8_linear import Int8Linear

    torch.manual_seed(0)
    linear = torch.nn.Linear(1024, 3072, bias=False, device="cuda", dtype=torch.bfloat16)
    return linear, Int8Linear(linear)


def _dequantized(q):
    return q.weight.to(torch.bfloat16) * q.scale.unsqueeze(1)


@pytest.mark.parametrize("rows", [1, 10, 40])  # 40 rows takes the dequantized cuBLAS path
def test_matches_dequantized_matmul(layer, rows):
    _, q = layer
    x = torch.randn(1, rows, 1024, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x.float(), _dequantized(q).float())
    got = q(x)
    assert got.shape == (1, rows, 3072) and got.dtype == torch.bfloat16
    assert torch.allclose(got.float(), expected, atol=0.05, rtol=0.02)


def test_quantization_error_is_small(layer):
    linear, q = layer
    rel = (_dequantized(q).float() - linear.weight.float()).norm() / linear.weight.float().norm()
    assert rel < 0.01


def test_quantize_linears_replaces_only_bias_free_linears():
    from qwen_tts.inference.int8_linear import Int8Linear, quantize_linears_int8

    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64, bias=False), torch.nn.Linear(64, 64, bias=True), torch.nn.LayerNorm(64)
    ).cuda().to(torch.bfloat16)
    assert quantize_linears_int8(model) == 1
    assert isinstance(model[0], Int8Linear) and isinstance(model[1], torch.nn.Linear)


def test_works_inside_compile_and_cuda_graph(layer):
    _, q = layer
    compiled = torch.compile(lambda x: torch.nn.functional.silu(q(x)), dynamic=False)
    x = torch.randn(1, 1, 1024, device="cuda", dtype=torch.bfloat16)
    expected = compiled(x).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = compiled(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
