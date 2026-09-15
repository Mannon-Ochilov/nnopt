"""Does functional grouping (README Sec 2.3) have ANY headroom in this model?

README Sec 8.3.1 established that on decoder.proj_out no two hidden nodes
reach cos >= 0.8 across real Uzbek-speech calibration, so grouping is a
complete no-op there. That input is a LayerNorm output, whose channels are
decorrelated by construction -- so the finding might be specific to that
position in the block rather than general.

This script measures the pairwise-cosine distribution of the activation
input of FOUR structurally different operator positions, in both an early
and a late decoder layer:

  * fc1        <- final_layer_norm output      (post-LayerNorm, like proj_out)
  * fc2        <- activation_fn (GELU) output  (post-GELU: sign-skewed, the
                                                most likely place for genuine
                                                inter-channel correlation)
  * self_attn.out_proj    <- attention-weighted value mixture
  * encoder_attn.out_proj <- cross-attention-weighted value mixture

Output: for each tensor, how many nodes have at least one partner above
each tau threshold -- i.e. how many nodes functional grouping could
actually merge.
"""

import io

import numpy as np
import onnxruntime as ort
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from nnopt.calibrator.activation_capture import ActivationCapture, build_response_vectors

MODEL_DIR = "models/hh"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
N_CALIB_SAMPLES = 16
TARGET_SR = 16000

TENSORS = {}
for layer in (0, 11, 23):
    TENSORS[f"L{layer}.fc1_in (post-LayerNorm)"] = f"/model/decoder/layers.{layer}/final_layer_norm/LayerNormalization_output_0"
    TENSORS[f"L{layer}.fc2_in (post-GELU)"] = f"/model/decoder/layers.{layer}/activation_fn/Mul_1_output_0"
    TENSORS[f"L{layer}.self_attn.out_proj_in"] = f"/model/decoder/layers.{layer}/self_attn/Reshape_3_output_0"
    TENSORS[f"L{layer}.enc_attn.out_proj_in"] = f"/model/decoder/layers.{layer}/encoder_attn/Reshape_3_output_0"
TENSORS["proj_out_in (reference, known no-op)"] = "/model/decoder/layer_norm/LayerNormalization_output_0"

TAUS = (0.98, 0.9, 0.8, 0.7, 0.6, 0.5)


def main():
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    decoder_start_token_id = 50258
    prompt_ids = [tok for _, tok in tokenizer.get_decoder_prompt_ids(language="uz", task="transcribe")]

    enc_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    capture = ActivationCapture(DECODER_PATH, tensor_names=list(TENSORS.values()))

    chunks = {name: [] for name in TENSORS}
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
        (encoder_hidden_states,) = enc_session.run(None, {"input_features": input_features})

        text_tokens = tokenizer(row["text"], add_special_tokens=False).input_ids
        input_ids = np.array([[decoder_start_token_id, *prompt_ids, *text_tokens]], dtype=np.int64)

        captured = capture.run_batch(
            {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states.astype(np.float32)}
        )
        for label, tname in TENSORS.items():
            act = captured[tname]
            if act.ndim == 3:
                # single un-padded sample -> no mask needed
                chunks[label].append(build_response_vectors(act, active_mask=None))
        print(f"  captured sample {i+1}/{N_CALIB_SAMPLES}")

    print()
    header = f"{'tensor':40s} {'nodes':>6s} {'maxcos':>7s} {'meanmax':>8s} " + " ".join(f"t{t:.2f}" for t in TAUS)
    print(header)
    print("-" * len(header))
    for label in TENSORS:
        if not chunks[label]:
            continue
        h = np.concatenate(chunks[label], axis=1)  # (nodes, positions)
        h_unit = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-12)
        cos = h_unit @ h_unit.T
        np.fill_diagonal(cos, -np.inf)
        per_node_max = cos.max(axis=1)
        counts = " ".join(f"{int((per_node_max >= t).sum()):5d}" for t in TAUS)
        print(
            f"{label:40s} {h.shape[0]:6d} {cos.max():7.4f} {per_node_max.mean():8.4f} {counts}"
        )
    print()
    print("columns tXX = how many nodes have at least one partner at cos >= XX")
    print("(a node with no partner can never be merged -> grouping is a no-op for it)")


if __name__ == "__main__":
    main()
