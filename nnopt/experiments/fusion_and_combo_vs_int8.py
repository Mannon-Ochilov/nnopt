"""Two follow-up questions from the user, both about decoder.layers.0.fc2:

Q1: Can a COMBINATION of methods (functional CUR + selective/smart
    quantization -- keep the sensitive U factor at FP32, quantize only
    C and R) beat plain INT8 alone, where naive full quantization
    (functional_cur+int8 in the previous table) catastrophically failed?

Q2: Does graph-level fusion of the CUR factors reduce computation? Tested
    both analytically and empirically: pre-multiplying R^T@U^T into one
    (n x c) matrix collapses the 3-MatMul chain to 2 MatMuls. Whether this
    HELPS depends on the relationship between r, c, n -- with c=r=rank (as
    used throughout this session), the algebra says the "fused" 2-factor
    form is actually CHEAPER than the "unfused" 3-factor form, because
    tying c=r makes U square (r x r), adding pure overhead with no
    bottleneck benefit. Verified here both by FLOPs count and real
    ONNX Runtime latency.

All numbers are REAL: real Whisper decoder weight, real Common Voice
calibration, real ONNX graphs, real onnxruntime timing (warmup=10,
median-of-50).
"""

import io
import os

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
from nnopt.cur.svd_cur import analyze_spectrum, build_cur, cur_param_count, original_param_count, select_cur_columns, select_cur_rows
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group

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


def build_3factor_model(r_t, u_t, c_t, path, n):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", c_t.shape[1]])
    ri = helper.make_tensor("r_t", TensorProto.FLOAT, list(r_t.shape), r_t.astype(np.float32).flatten())
    ui = helper.make_tensor("u_t", TensorProto.FLOAT, list(u_t.shape), u_t.astype(np.float32).flatten())
    ci = helper.make_tensor("c_t", TensorProto.FLOAT, list(c_t.shape), c_t.astype(np.float32).flatten())
    n1 = helper.make_node("MatMul", ["x", "r_t"], ["h1"], name="mm1_R")
    n2 = helper.make_node("MatMul", ["h1", "u_t"], ["h2"], name="mm2_U")
    n3 = helper.make_node("MatMul", ["h2", "c_t"], ["y"], name="mm3_C")
    g = helper.make_graph([n1, n2, n3], "cur3", [x], [y], initializer=[ri, ui, ci])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)


def build_2factor_fused_model(m1, c_t, path, n):
    """R,U pre-fused offline into m1 = R^T @ U^T, shape (n, c)."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", n])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", c_t.shape[1]])
    m1i = helper.make_tensor("m1", TensorProto.FLOAT, list(m1.shape), m1.astype(np.float32).flatten())
    ci = helper.make_tensor("c_t", TensorProto.FLOAT, list(c_t.shape), c_t.astype(np.float32).flatten())
    n1 = helper.make_node("MatMul", ["x", "m1"], ["h1"], name="mm1_fused_RU")
    n2 = helper.make_node("MatMul", ["h1", "c_t"], ["y"], name="mm2_C")
    g = helper.make_graph([n1, n2], "cur2fused", [x], [y], initializer=[m1i, ci])
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mdl.ir_version = 8
    onnx.checker.check_model(mdl)
    onnx.save(mdl, path)


def bench(path, n, seed=1):
    sess = make_session(path, intra_op_threads=1)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((BATCH, n)).astype(np.float32)
    r = measure_latency(sess, name=path, fixed_feed={"x": x}, warmup_runs=WARMUP, measured_runs=MEASURED)
    return r.median_ms, x


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

    # baseline FP32 single matmul, for the speedup denominator
    p_base = f"{TMP}/fc2_fusion_fp32base.onnx"
    x_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", n])
    y_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", m])
    w_init = helper.make_tensor("w", TensorProto.FLOAT, list(w.T.shape), w.T.astype(np.float32).flatten())
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")
    g = helper.make_graph([node], "base", [x_t], [y_t], initializer=[w_init])
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mdl.ir_version = 8
    onnx.save(mdl, p_base)
    t_base, x_fixed = bench(p_base, n, seed=0)
    y_fp32_ref = x_fixed.astype(np.float64) @ w.T

    # INT8-only baseline (existing method, for comparison)
    p_int8 = f"{TMP}/fc2_fusion_int8base.onnx"
    quantize_dynamic(p_base, p_int8, weight_type=QuantType.QInt8)
    t_int8, _ = bench(p_int8, n, seed=0)
    y_int8 = make_session(p_int8, intra_op_threads=1).run(None, {"x": x_fixed})[0]
    eloc_int8 = relerr(y_fp32_ref, y_int8.astype(np.float64))
    bytes_int8 = original_param_count(m, n) * 1

    # functional CUR factors (proposed method)
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

    # ---- 3-factor FP32 CUR (already known from previous table, recomputed for a clean seed-matched comparison) ----
    p_3f = f"{TMP}/fc2_3factor_fp32.onnx"
    build_3factor_model(cur.R.T, cur.U.T, cur.C.T, p_3f, n)
    t_3f, _ = bench(p_3f, n, seed=1)
    y_3f = x_calib @ cur.reconstruct().T
    eloc_3f = relerr(y_ref, y_3f)

    # ---- Q1: SMART combo -- quantize ONLY C and R (mm1_R, mm3_C), leave U (mm2_U) at FP32 ----
    p_3f_smart_int8 = f"{TMP}/fc2_3factor_smartint8.onnx"
    quantize_dynamic(p_3f, p_3f_smart_int8, weight_type=QuantType.QInt8, nodes_to_exclude=["mm2_U"])
    t_3f_smart, _ = bench(p_3f_smart_int8, n, seed=1)
    y_3f_smart = make_session(p_3f_smart_int8, intra_op_threads=1).run(None, {"x": x_fixed})[0]
    eloc_3f_smart = relerr(y_fp32_ref, y_3f_smart.astype(np.float64))
    bytes_3f_smart = (c_count * n + r_count * c_count * 4 + m * c_count) if False else None
    # bytes: R (n x r=r_count*n) INT8=1B, U (c x r) FP32=4B, C (m x c) INT8=1B  -- shapes: R:(r,n), U:(c,r), C:(m,c)
    bytes_3f_smart = (r_count * n * 1) + (c_count * r_count * 4) + (m * c_count * 1)

    # ---- Q2: 2-factor GRAPH-FUSED CUR (pre-multiply R,U offline), FP32 ----
    m1 = cur.R.T @ cur.U.T  # (n, c)
    p_2f = f"{TMP}/fc2_2factor_fused_fp32.onnx"
    build_2factor_fused_model(m1, cur.C.T, p_2f, n)
    t_2f, _ = bench(p_2f, n, seed=1)
    y_2f = x_calib @ (cur.C @ (cur.U @ cur.R)).T  # mathematically identical reconstruction to cur.reconstruct()
    eloc_2f = relerr(y_ref, y_2f)
    params_2f = n * c_count + c_count * m

    # ---- Q2 bonus: 2-factor fused + INT8 on both remaining factors ----
    p_2f_int8 = f"{TMP}/fc2_2factor_fused_int8.onnx"
    quantize_dynamic(p_2f, p_2f_int8, weight_type=QuantType.QInt8)
    t_2f_int8, _ = bench(p_2f_int8, n, seed=1)
    y_2f_int8 = make_session(p_2f_int8, intra_op_threads=1).run(None, {"x": x_fixed})[0]
    eloc_2f_int8 = relerr(y_fp32_ref, y_2f_int8.astype(np.float64))
    bytes_2f_int8 = params_2f * 1

    print()
    print("=== fc2 (1024x4096), rank=358 -- fusion & smart-quantization follow-up ===")
    header = f"{'variant':32s} {'params':>10s} {'bytes(KiB)':>11s} {'compress':>9s} {'E_loc':>8s} {'latency(ms)':>12s} {'speedup':>8s}"
    print(header)
    print("-" * len(header))
    baseline_bytes = original_param_count(m, n) * 4

    def row(name, params, bytes_, eloc, t):
        comp = baseline_bytes / bytes_
        print(f"{name:32s} {params:10d} {bytes_/1024:11.1f} {comp:8.2f}x {eloc:8.4f} {t:12.4f} {t_base/t:7.2f}x")

    row("baseline FP32", original_param_count(m, n), baseline_bytes, 0.0, t_base)
    row("INT8 only (existing)", original_param_count(m, n), bytes_int8, eloc_int8, t_int8)
    row("3-factor CUR FP32 (proposed)", params_cur, params_cur * 4, eloc_3f, t_3f)
    row("3-factor CUR + SMART int8 (C,R only)", params_cur, bytes_3f_smart, eloc_3f_smart, t_3f_smart)
    row("2-factor FUSED CUR FP32 (R*U offline)", params_2f, params_2f * 4, eloc_2f, t_2f)
    row("2-factor FUSED CUR + int8", params_2f, bytes_2f_int8, eloc_2f_int8, t_2f_int8)

    print()
    print("FLOPs (theoretical, batch=32) sanity check:")
    print(f"  3-factor:  {2*BATCH*(RANK*(n+c_count)+c_count*m):,}")
    print(f"  2-factor:  {2*BATCH*(n*c_count+c_count*m):,}")


if __name__ == "__main__":
    main()
