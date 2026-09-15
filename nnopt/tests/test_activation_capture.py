"""Tests for nnopt.calibrator.activation_capture."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from nnopt.calibrator.activation_capture import (
    ActivationCapture,
    active_mask_from_lengths,
    build_response_vectors,
)


@pytest.fixture(scope="module")
def gemm_relu_model_path(tmp_path_factory):
    """Gemm -> Relu chain where the Gemm output is an *internal* tensor,
    not a declared graph output -- exercises real intermediate-tensor
    capture (as opposed to trivially reading a declared output/input)."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    w = helper.make_tensor(
        "w", TensorProto.FLOAT, [6, 8], np.random.default_rng(0).standard_normal((6, 8)).astype(np.float32).flatten()
    )
    b = helper.make_tensor("b", TensorProto.FLOAT, [6], np.zeros(6, dtype=np.float32))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 6])

    gemm = helper.make_node("Gemm", ["x", "w", "b"], ["gemm_out"], transB=1, name="gemm")
    relu = helper.make_node("Relu", ["gemm_out"], ["y"], name="relu")

    graph = helper.make_graph([gemm, relu], "gemm_relu", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    p = tmp_path_factory.mktemp("models") / "gemm_relu.onnx"
    onnx.save(model, str(p))
    return str(p)


def test_captured_tensor_is_provably_pre_activation(gemm_relu_model_path):
    """Stronger correctness check: gemm_out + relu(gemm_out) relationship
    must hold, proving we captured the *internal* tensor and not the
    (potentially identical-shaped) final output by accident."""
    capture_internal = ActivationCapture(gemm_relu_model_path, tensor_names=["gemm_out"])
    capture_final = ActivationCapture(gemm_relu_model_path, tensor_names=["y"])
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 8)).astype(np.float32) * 5.0  # scaled up to make negative values likely

    gemm_out = capture_internal.run_batch({"x": x})["gemm_out"]
    y = capture_final.run_batch({"x": x})["y"]

    assert (gemm_out < 0).any(), "test setup should produce some negative pre-activation values"
    assert np.allclose(np.maximum(gemm_out, 0), y)


def test_run_calibration_set_collects_all_batches(gemm_relu_model_path):
    capture = ActivationCapture(gemm_relu_model_path, tensor_names=["gemm_out"])
    rng = np.random.default_rng(3)
    feeds = [{"x": rng.standard_normal((4, 8)).astype(np.float32)} for _ in range(3)]

    result = capture.run_calibration_set(feeds)
    assert "gemm_out" in result
    assert len(result["gemm_out"].batches) == 3
    concatenated = result["gemm_out"].concatenate(axis=0)
    assert concatenated.shape == (12, 6)


def test_build_response_vectors_masks_padding_positions():
    # batch=2, seq=3, hidden=4
    activations = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    active_mask = np.array([[True, True, False], [True, False, False]])  # 3 active positions total

    h = build_response_vectors(activations, active_mask=active_mask, node_axis=-1)
    assert h.shape == (4, 3)  # 4 hidden nodes, 3 active (batch,seq) positions

    # manually verify node 0's values match the active positions in order
    expected_node0 = np.array([activations[0, 0, 0], activations[0, 1, 0], activations[1, 0, 0]])
    assert np.array_equal(h[0], expected_node0)


def test_build_response_vectors_without_mask_keeps_all_positions():
    activations = np.random.default_rng(4).standard_normal((2, 5, 7))
    h = build_response_vectors(activations, active_mask=None)
    assert h.shape == (7, 2 * 5)


def test_active_mask_from_lengths_shape_and_content():
    mask = active_mask_from_lengths([3, 1, 5], max_len=5)
    assert mask.shape == (3, 5)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].tolist() == [True, False, False, False, False]
    assert mask[2].tolist() == [True] * 5


def test_active_mask_from_lengths_rejects_overlong_lengths():
    with pytest.raises(ValueError, match="greater than max_len"):
        active_mask_from_lengths([3, 9], max_len=5)


def test_zero_padding_is_geometrically_harmless():
    """Worth pinning explicitly, because it is counter-intuitive: appending
    exactly-zero positions changes neither ||h_j||, nor <h_i, h_j>, nor the
    cosine between nodes -- so IF padding activations were truly zero, the
    mask would be optional. It is the next test that shows why they are not.
    """
    rng = np.random.default_rng(5)
    batch, max_len, hidden = 3, 6, 4
    lengths = [6, 2, 4]

    activations = rng.standard_normal((batch, max_len, hidden))
    for b, real_len in enumerate(lengths):
        activations[b, real_len:, :] = 0.0

    mask = active_mask_from_lengths(lengths, max_len)
    h_masked = build_response_vectors(activations, active_mask=mask)
    h_unmasked = build_response_vectors(activations, active_mask=None)

    assert np.isclose(np.linalg.norm(h_masked[0]), np.linalg.norm(h_unmasked[0]))
    cos_masked = np.dot(h_masked[0], h_masked[1]) / (
        np.linalg.norm(h_masked[0]) * np.linalg.norm(h_masked[1])
    )
    cos_unmasked = np.dot(h_unmasked[0], h_unmasked[1]) / (
        np.linalg.norm(h_unmasked[0]) * np.linalg.norm(h_unmasked[1])
    )
    assert np.isclose(cos_masked, cos_unmasked)


def test_nonzero_padding_corrupts_h_j_when_mask_is_omitted():
    """Fig 2.5's "To'ldiruvchi pozitsiyalar hisobga olinmaydi" matters
    because real padding positions are NOT zero: a transformer still runs
    LayerNorm, positional embeddings and attention over pad slots, and
    Whisper in particular pads every clip to 30 s of "silence" that mel +
    conv turn into perfectly non-zero activations. This test uses non-zero
    pad activations (the realistic case) and shows the resulting h_j
    differs -- in norm AND in inter-node cosine, which is exactly the
    quantity functional grouping thresholds on.
    """
    rng = np.random.default_rng(6)
    batch, max_len, hidden = 3, 6, 4
    lengths = [6, 2, 4]

    activations = rng.standard_normal((batch, max_len, hidden))
    for b, real_len in enumerate(lengths):
        # realistic garbage on pad slots: different distribution, non-zero mean
        activations[b, real_len:, :] = 3.0 + 0.5 * rng.standard_normal((max_len - real_len, hidden))

    mask = active_mask_from_lengths(lengths, max_len)
    h_masked = build_response_vectors(activations, active_mask=mask)
    h_unmasked = build_response_vectors(activations, active_mask=None)

    assert h_masked.shape == (hidden, sum(lengths))
    assert h_unmasked.shape == (hidden, batch * max_len)
    assert not np.isclose(np.linalg.norm(h_masked[0]), np.linalg.norm(h_unmasked[0]))

    cos_masked = np.dot(h_masked[0], h_masked[1]) / (
        np.linalg.norm(h_masked[0]) * np.linalg.norm(h_masked[1])
    )
    cos_unmasked = np.dot(h_unmasked[0], h_unmasked[1]) / (
        np.linalg.norm(h_unmasked[0]) * np.linalg.norm(h_unmasked[1])
    )
    assert not np.isclose(cos_masked, cos_unmasked, atol=1e-3), (
        f"pad positions must change the inter-node cosine that grouping "
        f"thresholds on: masked={cos_masked:.4f} unmasked={cos_unmasked:.4f}"
    )
