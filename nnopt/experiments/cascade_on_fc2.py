"""Full cascade on decoder layer 0 fc2 -- the operator where README Sec 8.3.2
showed functional grouping actually HAS headroom (post-GELU input: 965 of
4096 nodes have a merge partner at cos >= 0.8, 195 at cos >= 0.98), unlike
proj_out where grouping was a complete no-op.

fc2 is also genuinely cache-critical per the Sec 2.2 profiling run:
K_L2 ~= 19 (1024x4096 fp32 = 16 MiB vs 0.7*1.25 MiB of L2), so targeting L2
here is a real constraint rather than a contrived one -- and 289 of the
decoder's 337 matmul operators are L2-critical, making this representative.
"""

import io

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from nnopt.calibrator.activation_capture import ActivationCapture, build_response_vectors
from nnopt.cascade.operator_cascade import OperatorContext, run_cascade
from nnopt.hw.cache_topology import detect_cache_topology

MODEL_DIR = "models/hh"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
LAYER = 0
ACT_NAME = f"/model/decoder/layers.{LAYER}/activation_fn/Mul_1_output_0"
WEIGHT_INPUT_NAME = "onnx::MatMul_5447"  # fc2 weight initializer for layer 0
N_CALIB_SAMPLES = 16
TARGET_SR = 16000


def capture_fc2_input():
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    decoder_start_token_id = 50258
    prompt_ids = [tok for _, tok in tokenizer.get_decoder_prompt_ids(language="uz", task="transcribe")]

    enc_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    capture = ActivationCapture(DECODER_PATH, tensor_names=[ACT_NAME])

    chunks = []
    for i, row in enumerate(ds):
        if i >= N_CALIB_SAMPLES:
            break
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = resample_poly(data, TARGET_SR, sr)
        input_features = feature_extractor(
            data.astype(np.float32), sampling_rate=TARGET_SR, return_tensors="np"
        ).input_features.astype(np.float32)
        (enc_hidden,) = enc_session.run(None, {"input_features": input_features})
        text_tokens = tokenizer(row["text"], add_special_tokens=False).input_ids
        input_ids = np.array([[decoder_start_token_id, *prompt_ids, *text_tokens]], dtype=np.int64)
        captured = capture.run_batch(
            {"input_ids": input_ids, "encoder_hidden_states": enc_hidden.astype(np.float32)}
        )
        chunks.append(build_response_vectors(captured[ACT_NAME], active_mask=None))
        print(f"  [{i+1}/{N_CALIB_SAMPLES}]")
    return np.concatenate(chunks, axis=1).T  # (positions, 4096)


def main():
    print("capturing real post-GELU fc2 input activations...")
    x_calib = capture_fc2_input()
    print("x_calib:", x_calib.shape)

    model = onnx.load(DECODER_PATH)
    init = next(i for i in model.graph.initializer if i.name == WEIGHT_INPUT_NAME)
    w_stored = onnx.numpy_helper.to_array(init).astype(np.float64)
    print("fc2 weight as stored:", w_stored.shape)
    # ONNX MatMul: y = x @ W with W (4096, 1024); OperatorContext wants Y = W @ X
    # with W (out_features, in_features) = (1024, 4096).
    w = w_stored.T if w_stored.shape[0] == 4096 else w_stored
    print("W for cascade (m=out, n=in):", w.shape)

    topo = detect_cache_topology()
    # README session finding: L2 on this machine is only shared in pairs (8
    # instances of 2 cores each); without explicit thread-affinity pinning,
    # the OS scheduler could place this operator's threads on ANY core pair,
    # so L2 is not a SAFE target. global_shared_cache() is the level
    # guaranteed shared no matter which cores get used -- L3 here.
    target_cache = topo.global_shared_cache()
    print(f"target cache: L{target_cache.level} (global-shared), {target_cache.size_bytes/1024:.0f} KiB")

    ctx = OperatorContext(
        name=f"decoder.layers.{LAYER}.fc2",
        w=w,
        x_calib=x_calib,
        dtype_bits_initial=32,
        cache=target_cache,
        alpha=0.7,
        delta_l=0.1,
        tau_soft=0.9,  # Sec 8.3.2: real merge headroom starts around here for post-GELU
        eps_threshold_soft=0.2,
    )
    print(f"baseline: {w.size*4/1024/1024:.1f} MiB, required reduction: "
          f"{(w.size*4)/(0.7*target_cache.size_bytes):.2f}x")

    result = run_cascade(ctx)
    print()
    print("=== CASCADE RESULT (fc2, real Uzbek speech calibration) ===")
    print("status:", result.status)
    for v in result.variants:
        marker = " <== FINAL" if v is result.final_variant else ""
        print(
            f"  {v.stage:16s} bits={v.bits:3d} rank={str(v.rank):>5s} "
            f"e_loc={v.e_loc:8.4f} k_cache={v.k_cache:8.3f} accepted={v.accepted}{marker}"
        )


if __name__ == "__main__":
    main()
