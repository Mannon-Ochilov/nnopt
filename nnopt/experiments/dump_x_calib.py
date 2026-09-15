"""Capture and persist the real proj_out calibration activations once, so
diagnostics can re-analyse them without re-running encoder+decoder over the
whole calibration set each time.

Produces models/x_calib_proj_out.npy of shape (total_active_positions, 1024).
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
ACT_NAME = "/model/decoder/layer_norm/LayerNormalization_output_0"
N_CALIB_SAMPLES = 24
TARGET_SR = 16000
OUT_PATH = "models/x_calib_proj_out.npy"


def main():
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    decoder_start_token_id = 50258
    prompt_ids = [tok for _, tok in tokenizer.get_decoder_prompt_ids(language="uz", task="transcribe")]

    enc_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    dec_capture = ActivationCapture(DECODER_PATH, tensor_names=[ACT_NAME])

    h_chunks = []
    for i, row in enumerate(ds):
        if i >= N_CALIB_SAMPLES:
            break
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = resample_poly(data, TARGET_SR, sr)
        audio = data.astype(np.float32)

        input_features = feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="np"
        ).input_features.astype(np.float32)
        (encoder_hidden_states,) = enc_session.run(None, {"input_features": input_features})

        text_tokens = tokenizer(row["text"], add_special_tokens=False).input_ids
        input_ids = np.array([[decoder_start_token_id, *prompt_ids, *text_tokens]], dtype=np.int64)

        captured = dec_capture.run_batch(
            {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states.astype(np.float32)}
        )
        # single sample fed at its exact length -> no padding, mask not needed
        h_chunks.append(build_response_vectors(captured[ACT_NAME], active_mask=None))
        print(f"  [{i+1}/{N_CALIB_SAMPLES}] seq_len={captured[ACT_NAME].shape[1]}")

    x_calib = np.concatenate(h_chunks, axis=1).T
    np.save(OUT_PATH, x_calib)
    print("saved", OUT_PATH, x_calib.shape)


if __name__ == "__main__":
    main()
