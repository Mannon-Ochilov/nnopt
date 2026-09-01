"""Real Uzbek-speech calibration for the decoder's proj_out operator.

Replaces the synthetic-noise placeholder used in cascade_on_real_proj_out.py
with an actual data pipeline:

    Common Voice uz audio (32kHz) --resample--> 16kHz
        --WhisperFeatureExtractor--> input_features (1, 80, 3000)
        --encoder.onnx--> real encoder_hidden_states
    real transcript text --WhisperTokenizer--> real input_ids (teacher-forced)
        --decoder.onnx--> real activation_input to proj_out (post layer_norm)

This is the "swap the feed-generation function" step promised as a
follow-up in cascade_on_real_proj_out.py -- everything downstream
(functional grouping, CUR, quantization, the cascade decision itself) is
unchanged.
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
N_CALIB_SAMPLES = 24
TARGET_SR = 16000


def load_real_samples(n: int):
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    samples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = resample_poly(data, TARGET_SR, sr)
        samples.append((data.astype(np.float32), row["text"]))
    return samples


def main():
    print(f"Loading {N_CALIB_SAMPLES} real Common Voice uz samples (streaming)...")
    samples = load_real_samples(N_CALIB_SAMPLES)
    print(f"loaded {len(samples)} samples, e.g. text[0]={samples[0][1]!r}")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    decoder_start_token_id = 50258
    prompt_ids = [tok for _, tok in tokenizer.get_decoder_prompt_ids(language="uz", task="transcribe")]

    enc_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])

    dec_model = onnx.load(DECODER_PATH)
    act_name = "/model/decoder/layer_norm/LayerNormalization_output_0"
    dec_capture = ActivationCapture(DECODER_PATH, tensor_names=[act_name])

    h_chunks = []
    for i, (audio, text) in enumerate(samples):
        input_features = feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="np"
        ).input_features.astype(np.float32)

        (encoder_hidden_states,) = enc_session.run(None, {"input_features": input_features})

        text_tokens = tokenizer(text, add_special_tokens=False).input_ids
        input_ids = np.array([[decoder_start_token_id, *prompt_ids, *text_tokens]], dtype=np.int64)

        captured = dec_capture.run_batch(
            {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states.astype(np.float32)}
        )
        activation = captured[act_name]  # (1, seq_len, 1024)
        h_chunk = build_response_vectors(activation, active_mask=None)  # (1024, seq_len)
        h_chunks.append(h_chunk)
        print(f"  [{i+1}/{len(samples)}] seq_len={activation.shape[1]} text={text[:40]!r}")

    x_calib = np.concatenate(h_chunks, axis=1).T  # (total_positions, 1024)
    print("x_calib shape (real speech-derived):", x_calib.shape)

    init = next(i for i in dec_model.graph.initializer if i.name == "model.decoder.embed_tokens.weight")
    w = onnx.numpy_helper.to_array(init).astype(np.float64)

    topo = detect_cache_topology()
    l3 = topo.by_level(3)[0]
    ctx = OperatorContext(
        name="decoder.proj_out", w=w, x_calib=x_calib, dtype_bits_initial=32, cache=l3, alpha=0.7, delta_l=0.1
    )
    print(f"required reduction: {(w.size*4)/(0.7*l3.size_bytes):.2f}x")

    result = run_cascade(ctx)
    print()
    print("=== CASCADE RESULT (REAL Uzbek speech calibration) ===")
    print("status:", result.status)
    for v in result.variants:
        marker = " <== FINAL" if v is result.final_variant else ""
        print(
            f"  {v.stage:16s} bits={v.bits:3d} rank={str(v.rank):>5s} "
            f"e_loc={v.e_loc:8.4f} k_cache={v.k_cache:7.3f} accepted={v.accepted}{marker}"
        )


if __name__ == "__main__":
    main()
