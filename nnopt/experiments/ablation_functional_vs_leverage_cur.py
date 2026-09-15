"""Core ablation for the dissertation's central claim (README Sec 7.4):
does functional-clustering-guided CUR column selection (Fig 2.7/2.8 --
calibration-response clustering + representative selection) beat generic
leverage-score CUR (no functional grouping, no calibration input at all)?

Deliberately DECOUPLED from any cache-fit/accept-reject framing -- this is
not about whether an operator's compression "solves" a hardware budget, it
is about whether the proposed column-selection method is a better CUR than
the standard baseline, at MATCHED rank budgets, on whichever real layers it
can be applied to.

Two variants compared at each rank:
  * baseline: columns AND rows both chosen by raw leverage score on the
    ORIGINAL W (fully generic CUR, README Sec 1.3.2's classical approach).
  * proposed: functional grouping -> compensated W~ -> functional-priority
    columns from the representative pool (Fig 2.8) + leverage-score rows
    on W~'s own spectrum (Fig 2.7/2.8's full pipeline, README Sec 2.3).

Measures BOTH the weight reconstruction error (||W-CUR||_F/||W||_F) and the
real-speech-calibration output error E_loc = ||Y-Y'||/||Y||, across several
structurally different real Whisper decoder operators and rank levels.
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
from nnopt.cur.svd_cur import (
    analyze_spectrum,
    build_cur,
    select_cur_columns,
    select_cur_columns_by_leverage,
    select_cur_rows,
)
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group

MODEL_DIR = "models/hh"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
N_CALIB_SAMPLES = 16
TARGET_SR = 16000

# (label, activation_tensor_name, weight_initializer_name)
OPERATORS = [
    ("L0.fc1", "/model/decoder/layers.0/final_layer_norm/LayerNormalization_output_0", "onnx::MatMul_5446"),
    ("L0.fc2", "/model/decoder/layers.0/activation_fn/Mul_1_output_0", "onnx::MatMul_5447"),
    ("L0.self_attn.out_proj", "/model/decoder/layers.0/self_attn/Reshape_3_output_0", "onnx::MatMul_5428"),
    ("L0.encoder_attn.out_proj", "/model/decoder/layers.0/encoder_attn/Reshape_3_output_0", "onnx::MatMul_5445"),
]

RANK_FRACTIONS = (0.15, 0.35)


def capture_calibration_data():
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    decoder_start_token_id = 50258
    prompt_ids = [tok for _, tok in tokenizer.get_decoder_prompt_ids(language="uz", task="transcribe")]

    enc_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    tensor_names = [act_name for _, act_name, _ in OPERATORS]
    capture = ActivationCapture(DECODER_PATH, tensor_names=tensor_names)

    chunks = {name: [] for name in tensor_names}
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
        for name in tensor_names:
            chunks[name].append(build_response_vectors(captured[name], active_mask=None))
        print(f"  captured sample {i+1}/{N_CALIB_SAMPLES}")

    return {name: np.concatenate(arrs, axis=1).T for name, arrs in chunks.items()}  # (positions, hidden)


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


def baseline_leverage_cur(w, rank):
    spectrum = analyze_spectrum(w)
    col_idx = select_cur_columns_by_leverage(spectrum, rank=rank, c=rank)
    row_idx = select_cur_rows(spectrum, rank=rank, r=rank)
    return build_cur(w, col_idx, row_idx).reconstruct()


def proposed_functional_cur(w, x_calib, rank, tau=0.9, eps_threshold=0.2):
    w_col_norms = np.linalg.norm(w, axis=0)
    y_ref = x_calib @ w.T
    y_norm = float(np.linalg.norm(y_ref))
    grouping = greedy_group(x_calib.T, w_col_norms, y_norm, tau=tau, eps_threshold=eps_threshold)
    w_tilde = build_compensated_weight(w, grouping)
    representative_cols = grouping.representative_indices()
    h_norms = np.linalg.norm(x_calib, axis=0)
    col_priority = {g.representative: float(g.size) * float(h_norms[g.representative]) for g in grouping.groups}
    spectrum = analyze_spectrum(w_tilde)
    c = min(rank, len(representative_cols))
    col_idx = select_cur_columns(representative_cols, col_priority, c=c)
    row_idx = select_cur_rows(spectrum, rank=rank, r=rank)
    return build_cur(w_tilde, col_idx, row_idx).reconstruct(), len(grouping.groups)


def main():
    print(f"capturing real Uzbek-speech calibration for {len(OPERATORS)} operators...")
    x_calib_by_tensor = capture_calibration_data()

    dec_model = onnx.load(DECODER_PATH)
    initializers = {i.name: i for i in dec_model.graph.initializer}

    print()
    header = f"{'operator':26s} {'rank':>5s} {'m x n':>12s} {'ngrp':>5s}  {'W-err leverage':>15s} {'W-err proposed':>15s}  {'E_loc leverage':>15s} {'E_loc proposed':>15s}  winner"
    print(header)
    print("-" * len(header))

    for label, act_name, weight_name in OPERATORS:
        w_stored = onnx.numpy_helper.to_array(initializers[weight_name]).astype(np.float64)
        x_calib = x_calib_by_tensor[act_name]
        w = w_stored if w_stored.shape[1] == x_calib.shape[1] else w_stored.T
        m, n = w.shape
        y_ref = x_calib @ w.T

        for frac in RANK_FRACTIONS:
            rank = max(1, int(frac * min(m, n)))

            w_leverage = baseline_leverage_cur(w, rank)
            w_proposed, n_groups = proposed_functional_cur(w, x_calib, rank)

            werr_leverage = relative_error(w, w_leverage)
            werr_proposed = relative_error(w, w_proposed)
            eloc_leverage = relative_error(y_ref, x_calib @ w_leverage.T)
            eloc_proposed = relative_error(y_ref, x_calib @ w_proposed.T)

            winner = "proposed" if eloc_proposed < eloc_leverage else "leverage"
            print(
                f"{label:26s} {rank:5d} {f'{m}x{n}':>12s} {n_groups:5d}  "
                f"{werr_leverage:15.4f} {werr_proposed:15.4f}  "
                f"{eloc_leverage:15.4f} {eloc_proposed:15.4f}  {winner}"
            )


if __name__ == "__main__":
    main()
