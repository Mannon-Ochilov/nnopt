"""End-to-end pipeline validation on a REAL Whisper operator: the decoder's
vocabulary projection (proj_out, 1024 -> 51865), identified by
profile_whisper_decoder.py as the single most resource-critical operator
(K_L3 ~= 12x over the real detected L3 cache budget).

IMPORTANT HONESTY NOTE: input_features / input_ids here are synthetic
placeholder noise, NOT real Uzbek speech. This script validates that the
full pipeline (activation capture -> functional grouping -> CUR -> scale
refinement -> cascade decision) runs correctly end-to-end against a real
model's real weight matrix and real ONNX graph -- it does NOT validate
that the resulting quality numbers (E_loc) reflect genuine speech
calibration statistics. Swapping in real Uzbek audio through the actual
WhisperFeatureExtractor is a follow-up step (needs a speech corpus), not
an architecture change: only the feed-generation function below changes.
"""

import numpy as np
import onnx

from nnopt.calibrator.activation_capture import ActivationCapture, build_response_vectors
from nnopt.cascade.operator_cascade import OperatorContext, run_cascade
from nnopt.hw.cache_topology import detect_cache_topology

DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"


def find_proj_out_activation_input(model_path: str):
    """proj_out uses Whisper's tied embedding weight (the real weight
    initializer is one hop upstream of a Transpose node, not a direct
    input to the MatMul node) -- traced manually once via
    scratch_find_embed.py rather than a generic graph-walking heuristic."""
    model = onnx.load(model_path)
    for node in model.graph.node:
        if node.output[0] == "logits" or "proj_out" in (node.name or ""):
            for inp in node.input:
                if "layer_norm" in inp or "LayerNormalization" in inp:
                    return node, inp
    raise RuntimeError("proj_out node / its activation input not found")


def main():
    node, act_name = find_proj_out_activation_input(DECODER_PATH)
    print(f"proj_out node: {node.name}, op_type={node.op_type}, activation_input={act_name}")

    model = onnx.load(DECODER_PATH)
    init = next(i for i in model.graph.initializer if i.name == "model.decoder.embed_tokens.weight")
    w = onnx.numpy_helper.to_array(init).astype(np.float64)
    # Embedding table convention [vocab_size, hidden] == exactly our OperatorContext
    # convention (m=out_features, n=in_features) for Y = W @ X -- no transpose needed.
    print("W for cascade (m=out_features, n=in_features):", w.shape)

    # --- capture real (synthetic-fed) activation_input across a few "calibration batches" ---
    capture = ActivationCapture(DECODER_PATH, tensor_names=[act_name])
    rng = np.random.default_rng(42)
    seq_len = 16
    feed_batches = []
    for i in range(6):
        input_ids = rng.integers(0, 51865, size=(1, seq_len)).astype(np.int64)
        encoder_hidden_states = (rng.standard_normal((1, 1500, 1024)) * 0.3).astype(np.float32)
        feed_batches.append({"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states})

    captured = capture.run_calibration_set(feed_batches)
    activations = captured[act_name].concatenate(axis=0)  # (6, seq_len, 1024)
    print("captured activation_input shape:", activations.shape)

    x_calib = build_response_vectors(activations, active_mask=None).T  # (batch*seq, 1024)
    print("x_calib for cascade:", x_calib.shape)

    topo = detect_cache_topology()
    l3 = topo.by_level(3)[0]
    print(f"target cache: L3, {l3.size_bytes/1024/1024:.1f} MiB")

    ctx = OperatorContext(
        name="decoder.proj_out",
        w=w,
        x_calib=x_calib,
        dtype_bits_initial=32,
        cache=l3,
        alpha=0.7,
        delta_l=0.1,
    )
    print(f"baseline M_total: {w.size*4/1024/1024:.1f} MiB, required reduction: "
          f"{(w.size*4)/(0.7*l3.size_bytes):.2f}x")

    result = run_cascade(ctx)
    print()
    print("=== CASCADE RESULT ===")
    print("status:", result.status)
    for v in result.variants:
        marker = " <== FINAL" if v is result.final_variant else ""
        print(
            f"  {v.stage:16s} bits={v.bits:3d} rank={str(v.rank):>5s} "
            f"e_loc={v.e_loc:8.4f} k_cache={v.k_cache:7.3f} accepted={v.accepted}{marker}"
        )


if __name__ == "__main__":
    main()
