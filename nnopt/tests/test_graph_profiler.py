"""Smoke/correctness tests for nnopt.profiler.graph_profiler (README Sec 2.2).

Validates against a small hand-built ONNX graph (see make_synthetic_model.py)
covering the three operator shapes seen in real transformer exports:
Gemm-as-Linear, MatMul-as-Linear (3-D activation, 2-D weight), and
MatMul-as-attention (batched, no weight, both operands runtime tensors).
"""

from __future__ import annotations

import math

import pytest

from make_synthetic_model import build_synthetic_model
from nnopt.hw.cache_topology import CacheInstance
from nnopt.profiler.graph_profiler import (
    ShapeResolutionError,
    evaluate_against_cache,
    profile_onnx_model,
)


@pytest.fixture(scope="module")
def synthetic_model_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("models") / "synthetic_test_model.onnx"
    build_synthetic_model(str(p))
    return str(p)


def _by_name(profiles, name):
    matches = [p for p in profiles if p.name == name]
    assert matches, f"operator '{name}' not found among {[p.name for p in profiles]}"
    return matches[0]


def test_missing_free_dims_raises(synthetic_model_path):
    with pytest.raises(ShapeResolutionError):
        profile_onnx_model(synthetic_model_path, free_dims={}, on_error="raise")


def test_gemm_linear_shapes_and_bytes(synthetic_model_path):
    profiles = profile_onnx_model(
        synthetic_model_path, free_dims={"batch": 2, "seq": 10}, on_error="raise"
    )
    g = _by_name(profiles, "linear_gemm")
    assert g.op_type == "Gemm"
    assert (g.batch, g.m, g.k, g.n) == (1, 8, 16, 32)
    assert g.M_W == 16 * 32 * 4 + 32 * 4  # weight + bias, fp32
    assert g.M_X == 8 * 16 * 4
    assert g.M_Y == 8 * 32 * 4
    assert g.flops == 2 * 8 * 16 * 32
    assert math.isclose(g.arithmetic_intensity, g.flops / g.bytes_io)


def test_matmul_linear_batched_activation(synthetic_model_path):
    profiles = profile_onnx_model(
        synthetic_model_path, free_dims={"batch": 2, "seq": 10}, on_error="raise"
    )
    mm = _by_name(profiles, "linear_matmul")
    assert mm.op_type == "MatMul"
    assert (mm.batch, mm.m, mm.k, mm.n) == (2, 10, 32, 64)
    assert mm.weight_initializer == "w2"
    assert mm.M_W == 32 * 64 * 4
    assert mm.M_X == 2 * 10 * 32 * 4
    assert mm.M_Y == 2 * 10 * 64 * 4
    assert mm.flops == 2 * 2 * 10 * 32 * 64


def test_matmul_attention_no_weight(synthetic_model_path):
    profiles = profile_onnx_model(
        synthetic_model_path, free_dims={"batch": 2, "seq": 10}, on_error="raise"
    )
    attn = _by_name(profiles, "attn_qk_matmul")
    assert attn.op_type == "MatMul"
    assert attn.weight_initializer is None, "attention QK has no initializer weight"
    assert (attn.batch, attn.m, attn.k, attn.n) == (8, 10, 16, 10)
    assert attn.M_W == 0
    # both q and k counted as activations since neither is a weight
    assert attn.M_X == 8 * 10 * 16 * 4 + 8 * 16 * 10 * 4
    assert attn.M_Y == 8 * 10 * 10 * 4
    assert attn.flops == 2 * 8 * 10 * 16 * 10


def test_cache_fit_evaluation(synthetic_model_path):
    profiles = profile_onnx_model(
        synthetic_model_path, free_dims={"batch": 2, "seq": 10}, on_error="raise"
    )
    g = _by_name(profiles, "linear_gemm")

    tiny_cache = CacheInstance(
        level=2, size_bytes=1024, line_size=64, associativity=8,
        cache_type="unified", group=0, core_ids=frozenset({0, 1}),
    )
    result = evaluate_against_cache(g, tiny_cache, alpha=0.7)
    assert result.is_critical  # M_total (3712B) >> 0.7*1024B
    assert result.k_cache > 1.0
    assert result.d_abs > 0

    huge_cache = CacheInstance(
        level=3, size_bytes=64 * 1024 * 1024, line_size=64, associativity=16,
        cache_type="unified", group=0, core_ids=frozenset(range(16)),
    )
    result2 = evaluate_against_cache(g, huge_cache, alpha=0.7)
    assert not result2.is_critical
    assert result2.k_cache < 1.0
    assert result2.d_abs == 0


def test_m_eff_blocked_is_bounded_by_upper(synthetic_model_path):
    profiles = profile_onnx_model(
        synthetic_model_path, free_dims={"batch": 2, "seq": 10}, on_error="raise"
    )
    mm = _by_name(profiles, "linear_matmul")
    blocked = mm.m_eff_blocked(k_block=8, n_block=8)
    assert 0 < blocked <= mm.m_eff_upper
