"""Builds a small synthetic ONNX graph exercising the three operator shapes
the profiler must handle correctly:

  1. Gemm with a 2-D weight initializer, transB=1 (PyTorch nn.Linear export
     convention: weight stored as (out_features, in_features)).
  2. MatMul with a 2-D weight initializer and a 3-D runtime activation
     (batch, seq, hidden) -- the common "Linear-as-MatMul" export shape.
  3. MatMul between two *runtime* tensors with a batch dimension and no
     initializer at all -- the attention QK^T pattern (no weight to speak
     of; both operands are activations).

This lets graph_profiler be validated against hand-computed expected
values before touching the (much larger, slower to iterate on) real
Whisper export.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper


def build_synthetic_model(path: str) -> None:
    batch = "batch"
    seq = "seq"

    # --- Node 1: Gemm, simulating a 2-D Linear layer -----------------
    # x1: (M=8, K=16) fp32 runtime input
    # W1: (N=32, K=16) fp32 initializer, transB=1  ->  y1 = x1 @ W1^T + b1 : (8, 32)
    x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [8, 16])
    w1 = helper.make_tensor(
        "w1", TensorProto.FLOAT, [32, 16], np.random.randn(32, 16).astype(np.float32).flatten()
    )
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [32], np.zeros(32, dtype=np.float32))
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [8, 32])
    gemm_node = helper.make_node(
        "Gemm", inputs=["x1", "w1", "b1"], outputs=["y1"], transB=1, name="linear_gemm"
    )

    # --- Node 2: MatMul, simulating a 3-D Linear layer ----------------
    # x2: (batch, seq, 32) fp32 runtime input (== y1 broadcast conceptually,
    # but kept independent here for a clean isolated shape test)
    # W2: (32, 64) fp32 initializer -> y2 = x2 @ W2 : (batch, seq, 64)
    x2 = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [batch, seq, 32])
    w2 = helper.make_tensor(
        "w2", TensorProto.FLOAT, [32, 64], np.random.randn(32, 64).astype(np.float32).flatten()
    )
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [batch, seq, 64])
    matmul_linear_node = helper.make_node(
        "MatMul", inputs=["x2", "w2"], outputs=["y2"], name="linear_matmul"
    )

    # --- Node 3: MatMul, simulating attention Q @ K^T (no weight) -----
    # q: (batch, heads=4, seq, d=16), k: (batch, heads=4, d=16, seq)
    # -> attn: (batch, 4, seq, seq)   (both operands are runtime activations)
    heads = 4
    d = 16
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, [batch, heads, seq, d])
    k = helper.make_tensor_value_info("k", TensorProto.FLOAT, [batch, heads, d, seq])
    attn = helper.make_tensor_value_info("attn", TensorProto.FLOAT, [batch, heads, seq, seq])
    matmul_attn_node = helper.make_node(
        "MatMul", inputs=["q", "k"], outputs=["attn"], name="attn_qk_matmul"
    )

    graph = helper.make_graph(
        nodes=[gemm_node, matmul_linear_node, matmul_attn_node],
        name="synthetic_profiler_test",
        inputs=[x1, x2, q, k],
        outputs=[y1, y2, attn],
        initializer=[w1, b1, w2],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="nnopt-synthetic-test",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


if __name__ == "__main__":
    import sys

    out_path = sys.argv[1] if len(sys.argv) > 1 else "synthetic_test_model.onnx"
    build_synthetic_model(out_path)
    print(f"wrote {out_path}")
