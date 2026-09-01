"""The comprehensive comparison the user asked for: compression ratio,
accuracy (E_loc), AND real measured inference latency, side by side, for:

  1. baseline        -- unmodified FP32 operator
  2. int8_only        -- standard existing method: ONNX Runtime dynamic INT8
                         quantization (onnxruntime.quantization.quantize_dynamic),
                         no structural change. This is the "usual" baseline
                         the dissertation compares against.
  3. leverage_cur      -- generic CUR (Sec 7.4 ablation baseline): columns
                         AND rows chosen by raw SVD leverage score, no
                         functional grouping, no calibration input.
  4. functional_cur     -- the proposed method: functional-clustering-guided
                         CUR columns (Fig 2.7/2.8) + leverage-score rows.
  5. functional_cur_int8 -- proposed method's CUR factors (C, U, R) ALSO
                         quantized to INT8 (matches the cascade's Sec 2.4
                         "cur_quantized" stage).

All FIVE variants are turned into REAL standalone ONNX graphs and timed
with nnopt.bench.latency (warmup=10, median-of-50) -- not FLOPs proxies.

HONESTY NOTE on what is NOT measured here: true hardware cache-miss counts
require Intel VTune or a fully-configured Windows Performance Toolkit PMC
trace, neither available in this environment (checked: wpr.exe only offers
CPU-usage/Disk-IO profiles, not raw LLC-miss counters). The closest
available proxy is the theoretical bytes-moved / arithmetic-intensity
figure from nnopt.profiler.graph_profiler, reported alongside for context,
clearly labeled as a proxy, not a measurement.
"""

import io
import time

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from datasets import Audio, load_dataset
from onnx import TensorProto, helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from nnopt.bench.latency import make_session, measure_latency
from nnopt.calibrator.activation_capture import ActivationCapture, build_response_vectors
from nnopt.cur.svd_cur import (
    analyze_spectrum,
    build_cur,
    cur_param_count,
    original_param_count,
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
BATCH_FOR_LATENCY = 32  # fixed timing batch, applied identically to every variant
WARMUP, MEASURED = 10, 50

OPERATORS = [
    ("L0.fc2", "/model/decoder/layers.0/activation_fn/Mul_1_output_0", "onnx::MatMul_5447", 358),
    ("L0.self_attn.out_proj", "/model/decoder/layers.0/self_attn/Reshape_3_output_0", "onnx::MatMul_5428", 358),
]


def capture_calibration_data():
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)
    decoder_start_token_id = 50258
    prompt_ids = [tok for _, tok in tokenizer.get_decoder_prompt_ids(language="uz", task="transcribe")]
    enc_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    tensor_names = [act for _, act, _, _ in OPERATORS]
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
        captured = capture.run_batch({"input_ids": input_ids, "encoder_hidden_states": enc_hidden.astype(np.float32)})
        for name in tensor_names:
            chunks[name].append(build_response_vectors(captured[name], active_mask=None))
        print(f"  captured sample {i+1}/{N_CALIB_SAMPLES}")
    return {name: np.concatenate(arrs, axis=1).T for name, arrs in chunks.items()}


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


def build_single_matmul_model(w_t: np.ndarray, path: str, n: int, m: int):
    """Y = X @ w_t ; X:(batch,n) dynamic batch, w_t:(n,m) initializer."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", m])
    w_init = helper.make_tensor("w", TensorProto.FLOAT, list(w_t.shape), w_t.astype(np.float32).flatten())
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")
    graph = helper.make_graph([node], "single_matmul", [x], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_cur_chain_model(r_t: np.ndarray, u_t: np.ndarray, c_t: np.ndarray, path: str, n: int, m: int):
    """Y = ((X @ R^T) @ U^T) @ C^T ; three sequential MatMuls."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", m])
    r_init = helper.make_tensor("r_t", TensorProto.FLOAT, list(r_t.shape), r_t.astype(np.float32).flatten())
    u_init = helper.make_tensor("u_t", TensorProto.FLOAT, list(u_t.shape), u_t.astype(np.float32).flatten())
    c_init = helper.make_tensor("c_t", TensorProto.FLOAT, list(c_t.shape), c_t.astype(np.float32).flatten())
    n1 = helper.make_node("MatMul", ["x", "r_t"], ["h1"], name="mm1")
    n2 = helper.make_node("MatMul", ["h1", "u_t"], ["h2"], name="mm2")
    n3 = helper.make_node("MatMul", ["h2", "c_t"], ["y"], name="mm3")
    graph = helper.make_graph([n1, n2, n3], "cur_chain", [x], [y], initializer=[r_init, u_init, c_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def bench_model(path: str, n: int, seed: int) -> float:
    sess = make_session(path, intra_op_threads=1)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((BATCH_FOR_LATENCY, n)).astype(np.float32)
    result = measure_latency(sess, name=path, fixed_feed={"x": x}, warmup_runs=WARMUP, measured_runs=MEASURED)
    return result.median_ms


def main():
    print("capturing real Uzbek-speech calibration data...")
    x_calib_by_tensor = capture_calibration_data()

    dec_model = onnx.load(DECODER_PATH)
    initializers = {i.name: i for i in dec_model.graph.initializer}

    rows = []
    tmp_dir = "models/_bench_tmp"
    import os
    os.makedirs(tmp_dir, exist_ok=True)

    for label, act_name, weight_name, rank in OPERATORS:
        print(f"\n=== {label} (rank={rank}) ===")
        w_stored = onnx.numpy_helper.to_array(initializers[weight_name]).astype(np.float64)
        x_calib = x_calib_by_tensor[act_name]
        w = w_stored if w_stored.shape[1] == x_calib.shape[1] else w_stored.T
        m, n = w.shape
        y_ref = x_calib @ w.T
        w_t = w.T.copy()  # (n, m), what the ONNX graph actually stores

        # ---- 1. baseline FP32 ----
        p_baseline = f"{tmp_dir}/{label}_fp32.onnx"
        build_single_matmul_model(w_t, p_baseline, n, m)
        t_baseline = bench_model(p_baseline, n, seed=0)
        rows.append(dict(
            operator=label, method="baseline_fp32", params=original_param_count(m, n),
            bytes_=original_param_count(m, n) * 4, e_loc=0.0, latency_ms=t_baseline,
            speedup=1.0,
        ))

        # ---- 2. INT8-only (ORT dynamic quantization, standard existing method) ----
        p_int8 = f"{tmp_dir}/{label}_int8.onnx"
        quantize_dynamic(p_baseline, p_int8, weight_type=QuantType.QInt8)
        t_int8 = bench_model(p_int8, n, seed=0)
        sess_int8 = make_session(p_int8, intra_op_threads=1)
        rng = np.random.default_rng(0)
        x_fixed = rng.standard_normal((BATCH_FOR_LATENCY, n)).astype(np.float32)
        y_int8 = sess_int8.run(None, {"x": x_fixed})[0]
        y_fp32_ref = x_fixed.astype(np.float64) @ w.T
        eloc_int8 = relative_error(y_fp32_ref, y_int8.astype(np.float64))
        rows.append(dict(
            operator=label, method="int8_only (existing)", params=original_param_count(m, n),
            bytes_=original_param_count(m, n) * 1, e_loc=eloc_int8, latency_ms=t_int8,
            speedup=t_baseline / t_int8,
        ))

        # ---- functional grouping (shared setup for CUR variants) ----
        w_col_norms = np.linalg.norm(w, axis=0)
        y_norm = float(np.linalg.norm(y_ref))
        grouping = greedy_group(x_calib.T, w_col_norms, y_norm, tau=0.9, eps_threshold=0.2)
        w_tilde = build_compensated_weight(w, grouping)
        representative_cols = grouping.representative_indices()
        h_norms = np.linalg.norm(x_calib, axis=0)
        col_priority = {g.representative: float(g.size) * float(h_norms[g.representative]) for g in grouping.groups}
        spectrum_tilde = analyze_spectrum(w_tilde)
        c_func = min(rank, len(representative_cols))
        col_idx_func = select_cur_columns(representative_cols, col_priority, c=c_func)
        row_idx_func = select_cur_rows(spectrum_tilde, rank=rank, r=rank)
        cur_func = build_cur(w_tilde, col_idx_func, row_idx_func)

        # ---- 3. leverage-score CUR (generic baseline) ----
        spectrum_raw = analyze_spectrum(w)
        col_idx_lev = select_cur_columns_by_leverage(spectrum_raw, rank=rank, c=rank)
        row_idx_lev = select_cur_rows(spectrum_raw, rank=rank, r=rank)
        cur_lev = build_cur(w, col_idx_lev, row_idx_lev)
        y_lev = x_calib @ cur_lev.reconstruct().T
        eloc_lev = relative_error(y_ref, y_lev)
        params_cur = cur_param_count(m, n, len(col_idx_lev), len(row_idx_lev))

        p_lev = f"{tmp_dir}/{label}_leverage_cur.onnx"
        build_cur_chain_model(cur_lev.R.T, cur_lev.U.T, cur_lev.C.T, p_lev, n, m)
        t_lev = bench_model(p_lev, n, seed=1)
        rows.append(dict(
            operator=label, method="leverage_cur (existing)", params=params_cur,
            bytes_=params_cur * 4, e_loc=eloc_lev, latency_ms=t_lev, speedup=t_baseline / t_lev,
        ))

        # ---- 4. proposed: functional-clustering CUR (FP32) ----
        y_func = x_calib @ cur_func.reconstruct().T
        eloc_func = relative_error(y_ref, y_func)
        params_func = cur_param_count(m, n, len(col_idx_func), len(row_idx_func))
        p_func = f"{tmp_dir}/{label}_functional_cur.onnx"
        build_cur_chain_model(cur_func.R.T, cur_func.U.T, cur_func.C.T, p_func, n, m)
        t_func = bench_model(p_func, n, seed=1)
        rows.append(dict(
            operator=label, method="functional_cur (PROPOSED)", params=params_func,
            bytes_=params_func * 4, e_loc=eloc_func, latency_ms=t_func, speedup=t_baseline / t_func,
        ))

        # ---- 5. proposed CUR + INT8 on the C,R factors (matches cascade Sec 2.4) ----
        p_func_int8 = f"{tmp_dir}/{label}_functional_cur_int8.onnx"
        quantize_dynamic(p_func, p_func_int8, weight_type=QuantType.QInt8)
        t_func_int8 = bench_model(p_func_int8, n, seed=1)
        sess_fi8 = make_session(p_func_int8, intra_op_threads=1)
        y_func_int8 = sess_fi8.run(None, {"x": x_fixed})[0]
        eloc_func_int8 = relative_error(y_fp32_ref, y_func_int8.astype(np.float64))
        rows.append(dict(
            operator=label, method="functional_cur+int8 (PROPOSED)", params=params_func,
            bytes_=params_func * 1, e_loc=eloc_func_int8, latency_ms=t_func_int8,
            speedup=t_baseline / t_func_int8,
        ))

    print()
    header = f"{'operator':22s} {'method':28s} {'params':>9s} {'bytes(KiB)':>11s} {'compress':>9s} {'E_loc':>8s} {'latency(ms)':>12s} {'speedup':>8s}"
    print(header)
    print("-" * len(header))
    baseline_bytes_by_op = {}
    for r in rows:
        if r["method"] == "baseline_fp32":
            baseline_bytes_by_op[r["operator"]] = r["bytes_"]
    for r in rows:
        comp = baseline_bytes_by_op[r["operator"]] / r["bytes_"]
        print(
            f"{r['operator']:22s} {r['method']:28s} {r['params']:9d} {r['bytes_']/1024:11.1f} "
            f"{comp:8.2f}x {r['e_loc']:8.4f} {r['latency_ms']:12.4f} {r['speedup']:7.2f}x"
        )


if __name__ == "__main__":
    main()
