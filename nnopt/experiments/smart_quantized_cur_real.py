"""Properly answering "can a combination beat INT8?": use OUR OWN
calibration-aware quantizer.scale_refine (alternating minimization + grid
search, README Sec 2.4) for C and R -- not ORT's naive per-tensor min/max
dynamic quantization, which the previous script showed is NOT good enough
(E_loc stayed catastrophic even with U excluded).

U stays FP32 (README Sec 2.4: "zarur holatda yuqoriroq aniqlikda saqlanadi").
The quantized C, R are embedded as fake-quantized FP32 constants first (to
verify accuracy matches the cascade's own numpy-level finding), and the
real INT8-typed ONNX QuantizeLinear/DequantizeLinear pattern is built for
the real latency measurement.
"""

import io
import os

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from datasets import Audio, load_dataset
from onnx import TensorProto, helper
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from nnopt.bench.latency import make_session, measure_latency
from nnopt.calibrator.activation_capture import ActivationCapture, build_response_vectors
from nnopt.cur.svd_cur import analyze_spectrum, build_cur, cur_param_count, original_param_count, select_cur_columns, select_cur_rows
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.quantizer.scale_refine import dequantize, quantize_codes, refine_scale

MODEL_DIR = "models/hh"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
ACT_NAME = "/model/decoder/layers.0/activation_fn/Mul_1_output_0"
WEIGHT_NAME = "onnx::MatMul_5447"
N_CALIB_SAMPLES = 16
TARGET_SR = 16000
BATCH = 32
WARMUP, MEASURED = 10, 50
RANK = 358
Q_MAX = 127
TMP = "models/_bench_tmp"


def capture():
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    dst = 50258
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    enc = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    cap = ActivationCapture(DECODER_PATH, tensor_names=[ACT_NAME])
    chunks = []
    for i, row in enumerate(ds):
        if i >= N_CALIB_SAMPLES:
            break
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = resample_poly(data, TARGET_SR, sr)
        feats = fe(data.astype(np.float32), sampling_rate=TARGET_SR, return_tensors="np").input_features.astype(np.float32)
        (eh,) = enc.run(None, {"input_features": feats})
        tt = tok(row["text"], add_special_tokens=False).input_ids
        ids = np.array([[dst, *prompt_ids, *tt]], dtype=np.int64)
        out = cap.run_batch({"input_ids": ids, "encoder_hidden_states": eh.astype(np.float32)})
        chunks.append(build_response_vectors(out[ACT_NAME], active_mask=None))
        print(f"  sample {i+1}/{N_CALIB_SAMPLES}")
    return np.concatenate(chunks, axis=1).T


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


def quantize_with_our_scale_refine(mat: np.ndarray) -> tuple[np.ndarray, float]:
    """README Sec 2.4: alternating minimization for the scale (no
    layer_response_fn here -- pure weight-reconstruction refinement,
    matching what the cascade's _cur_candidate uses for C, R)."""
    res = refine_scale(mat, Q_MAX)
    q = quantize_codes(mat, res.scale, Q_MAX)
    return dequantize(q, res.scale), res.scale


def build_qdq_int8_matmul_node(name_prefix, x_name, out_name, w_int8, scale, node_name):
    """One MatMul preceded by DequantizeLinear(INT8 const, scale) -> real
    INT8 storage, FP32 compute after dequant (weight-only quantization
    pattern -- gives real memory-bandwidth benefit; see note in results)."""
    w_init = helper.make_tensor(f"{name_prefix}_w_i8", TensorProto.INT8, list(w_int8.shape), w_int8.astype(np.int8).flatten())
    scale_init = helper.make_tensor(f"{name_prefix}_scale", TensorProto.FLOAT, [], [float(scale)])
    dq = helper.make_node(
        "DequantizeLinear", [f"{name_prefix}_w_i8", f"{name_prefix}_scale"], [f"{name_prefix}_w_fp32"],
        name=f"{name_prefix}_dq",
    )
    mm = helper.make_node("MatMul", [x_name, f"{name_prefix}_w_fp32"], [out_name], name=node_name)
    return [dq, mm], [w_init, scale_init]


def main():
    os.makedirs(TMP, exist_ok=True)
    print("capturing real Uzbek-speech calibration for fc2...")
    x_calib = capture()

    dec = onnx.load(DECODER_PATH)
    init = {i.name: i for i in dec.graph.initializer}[WEIGHT_NAME]
    w_stored = onnx.numpy_helper.to_array(init).astype(np.float64)
    w = w_stored if w_stored.shape[1] == x_calib.shape[1] else w_stored.T
    m, n = w.shape
    y_ref = x_calib @ w.T

    w_col_norms = np.linalg.norm(w, axis=0)
    y_norm = float(np.linalg.norm(y_ref))
    grouping = greedy_group(x_calib.T, w_col_norms, y_norm, tau=0.9, eps_threshold=0.2)
    w_tilde = build_compensated_weight(w, grouping)
    reps = grouping.representative_indices()
    h_norms = np.linalg.norm(x_calib, axis=0)
    prio = {g_.representative: float(g_.size) * float(h_norms[g_.representative]) for g_ in grouping.groups}
    spec = analyze_spectrum(w_tilde)
    c_idx = select_cur_columns(reps, prio, c=min(RANK, len(reps)))
    r_idx = select_cur_rows(spec, rank=RANK, r=RANK)
    cur = build_cur(w_tilde, c_idx, r_idx)
    c_count, r_count = len(c_idx), len(r_idx)
    params_cur = cur_param_count(m, n, c_count, r_count)

    # --- numpy-level accuracy check FIRST (cheap) with OUR quantizer ---
    r_deq, r_scale = quantize_with_our_scale_refine(cur.R)
    c_deq, c_scale = quantize_with_our_scale_refine(cur.C)
    y_smart_np = x_calib @ (c_deq @ cur.U @ r_deq).T
    eloc_smart_np = relerr(y_ref, y_smart_np)
    print(f"\n[numpy check] our-quantizer C,R (U fp32): E_loc = {eloc_smart_np:.4f}")
    print(f"[numpy check] (for reference) 3-factor FP32 CUR: E_loc = {relerr(y_ref, cur.reconstruct().T @ x_calib.T if False else x_calib @ cur.reconstruct().T):.4f}")

    # --- build the REAL ONNX QDQ graph: R,C as real INT8 storage + DequantizeLinear; U plain FP32 ---
    r_q = quantize_codes(cur.R, r_scale, Q_MAX).astype(np.int8)  # (r, n)
    c_q = quantize_codes(cur.C, c_scale, Q_MAX).astype(np.int8)  # (m, c)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", m])
    nodes, inits = [], []
    # stage 1: X @ R^T  (R^T stored INT8, shape (n, r))
    n1, i1 = build_qdq_int8_matmul_node("R", "x", "h1", r_q.T, r_scale, "mm1_R")
    nodes += n1; inits += i1
    # stage 2: h1 @ U^T (U stays FP32, no quant)
    u_init = helper.make_tensor("u_t", TensorProto.FLOAT, list(cur.U.T.shape), cur.U.T.astype(np.float32).flatten())
    n2 = helper.make_node("MatMul", ["h1", "u_t"], ["h2"], name="mm2_U")
    nodes.append(n2); inits.append(u_init)
    # stage 3: h2 @ C^T (C^T stored INT8, shape (c, m))
    n3, i3 = build_qdq_int8_matmul_node("C", "h2", "y", c_q.T, c_scale, "mm3_C")
    nodes += n3; inits += i3

    g = helper.make_graph(nodes, "cur_smart_qdq", [x], [y], initializer=inits)
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mdl.ir_version = 8
    onnx.checker.check_model(mdl)
    p_smart = f"{TMP}/fc2_smart_qdq.onnx"
    onnx.save(mdl, p_smart)

    sess = make_session(p_smart, intra_op_threads=1)
    rng = np.random.default_rng(0)
    x_fixed = rng.standard_normal((BATCH, n)).astype(np.float32)
    result = measure_latency(sess, name=p_smart, fixed_feed={"x": x_fixed}, warmup_runs=WARMUP, measured_runs=MEASURED)
    y_onnx = sess.run(None, {"x": x_fixed})[0]
    y_fp32_ref = x_fixed.astype(np.float64) @ w.T
    eloc_onnx = relerr(y_fp32_ref, y_onnx.astype(np.float64))

    bytes_smart = (r_count * n * 1) + (c_count * r_count * 4) + (m * c_count * 1)

    print()
    print("=== fc2: properly-quantized (our scale_refine) CUR, U kept FP32 ===")
    print(f"{'variant':40s} {'bytes(KiB)':>11s} {'E_loc(numpy)':>13s} {'E_loc(ONNX)':>12s} {'latency(ms)':>12s}")
    print(f"{'CUR + our-scale-refine int8 (C,R); U fp32':40s} {bytes_smart/1024:11.1f} {eloc_smart_np:13.4f} {eloc_onnx:12.4f} {result.median_ms:12.4f}")
    print()
    print("Reference from earlier runs (same fc2, rank=358, batch=32, same machine):")
    print("  baseline FP32           : bytes=16384.0 KiB  E_loc=0.0000  latency=2.4548 ms")
    print("  INT8 only (existing)    : bytes= 4096.0 KiB  E_loc=0.0332  latency=0.6308 ms")
    print("  3-factor CUR FP32       : bytes= 7660.6 KiB  E_loc=0.2694  latency=1.0478 ms")
    print("  naive full int8 CUR     : bytes= 2290.6 KiB  E_loc=1.2591  latency=0.3999 ms")


if __name__ == "__main__":
    main()
