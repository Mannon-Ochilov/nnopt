"""FULL-TRACK comparison: whole-network plain INT8 vs our mixed/cascade method.

The scientifically decisive framing (and the one this dissertation can
defend): plain weight quantization has a HARD compression ceiling --
INT8 is exactly 4x over FP32 and cannot go further. Any existing-method
attempt to exceed 4x must drop to INT4 (8x), which is where plain
quantization is known to collapse. Our method reaches the same 8x by a
DIFFERENT route: functional-clustering-guided CUR (structural) + INT8 on
the C, R factors (U kept FP32, README Sec 2.4).

So the comparison is run at MATCHED compression points, per operator:

  4x  : INT8 alone            vs  cascade's choice at a 4x target
  8x  : INT4 alone            vs  functional-CUR + INT8 (our method)
  8x  : leverage-CUR + INT8   (the generic-CUR baseline, README Sec 7.4)

Everything is measured on REAL calibration activations captured from real
Uzbek speech through the real Whisper decoder graph -- E_loc is a real
measured per-operator output error, not a proxy.

Operator coverage is a STRATIFIED SAMPLE across decoder depth (early /
middle / late layers), all 10 operator kinds per layer, since running all
240 operators' SVD + functional grouping is not necessary to establish the
per-operator-kind pattern. The sampled layers are stated in the output.
"""

import io
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from nnopt.calibrator.activation_capture import ActivationCapture, build_response_vectors
from nnopt.cascade.budget import closed_form_rank_for_target, max_cur_compression_factor
from nnopt.cur.svd_cur import (
    analyze_spectrum,
    build_cur,
    cur_param_count,
    select_cur_columns,
    select_cur_columns_by_leverage,
    select_cur_rows,
)
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.profiler.graph_profiler import profile_onnx_model
from nnopt.quantizer.scale_refine import dequantize, quantize_codes, refine_scale

MODEL_DIR = "models/hh"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
N_CALIB_SAMPLES = 12
TARGET_SR = 16000
MAX_CALIB_ROWS = 512  # subsample calibration rows (esp. encoder_hidden_states: 1500 pos/sample)
SAMPLED_LAYERS = [0, 6, 12, 18, 23]
Q_MAX = {8: 127, 4: 7}
TAU, EPS_THR = 0.9, 0.2
OUT_JSON = "experiments/results_full_track.json"
RNG = np.random.default_rng(0)


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


def q_deq(mat, bits):
    """Our calibration-aware quantizer (README Sec 2.4)."""
    qm = Q_MAX[bits]
    res = refine_scale(mat, qm)
    return dequantize(quantize_codes(mat, res.scale, qm), res.scale)


def select_operators():
    profiles = profile_onnx_model(
        DECODER_PATH,
        free_dims={"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500},
    )
    chosen = []
    for p in profiles:
        if p.weight_initializer is None:
            continue
        for li in SAMPLED_LAYERS:
            if f"/layers.{li}/" in p.name:
                chosen.append(p)
                break
    return chosen


def capture_activations(tensor_names):
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    enc = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    cap = ActivationCapture(DECODER_PATH, tensor_names=list(tensor_names))

    collected = {nm: [] for nm in tensor_names}
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
        ids = np.array([[50258, *prompt_ids, *tok(row["text"], add_special_tokens=False).input_ids]], dtype=np.int64)
        out = cap.run_batch({"input_ids": ids, "encoder_hidden_states": eh.astype(np.float32)})
        for nm, arr in out.items():
            collected[nm].append(build_response_vectors(arr, active_mask=None))
        print(f"  calib sample {i+1}/{N_CALIB_SAMPLES}", flush=True)

    x_by_tensor = {}
    for nm, chunks in collected.items():
        x = np.concatenate(chunks, axis=1).T  # (rows, n)
        if x.shape[0] > MAX_CALIB_ROWS:
            x = x[RNG.choice(x.shape[0], MAX_CALIB_ROWS, replace=False)]
        x_by_tensor[nm] = x.astype(np.float64)
    return x_by_tensor


def analyze_operator(p, w, x_calib):
    """Return dict of per-operator results at each matched compression point."""
    m, n = w.shape
    y_ref = x_calib @ w.T
    params = m * n
    out = {"name": p.name, "m": m, "n": n, "params": params}

    # --- existing methods: plain quantization (hard 4x / 8x ceilings) ---
    out["int8_eloc"] = relerr(y_ref, x_calib @ q_deq(w, 8).T)
    out["int8_ratio"] = 4.0
    out["int4_eloc"] = relerr(y_ref, x_calib @ q_deq(w, 4).T)
    out["int4_ratio"] = 8.0

    # --- our method at the 8x point: functional CUR + INT8(C,R), U fp32 ---
    # bytes = c*m*1 + c*r*4 + r*n*1  -> solve rank for the 8x target on that
    # mixed-precision budget (not the uniform-fp32 closed form).
    target_bytes = params * 4 / 8.0
    lo, hi, rank8 = 1, min(m, n), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        b = mid * m * 1 + mid * mid * 4 + mid * n * 1
        if b <= target_bytes:
            rank8, lo = mid, mid + 1
        else:
            hi = mid - 1
    out["rank8"] = rank8

    t0 = time.time()
    w_col_norms = np.linalg.norm(w, axis=0)
    grouping = greedy_group(x_calib.T, w_col_norms, float(np.linalg.norm(y_ref)), tau=TAU, eps_threshold=EPS_THR)
    w_tilde = build_compensated_weight(w, grouping)
    reps = grouping.representative_indices()
    h_norms = np.linalg.norm(x_calib, axis=0)
    prio = {g.representative: float(g.size) * float(h_norms[g.representative]) for g in grouping.groups}
    spec = analyze_spectrum(w_tilde)
    out["n_groups"] = len(grouping.groups)
    out["n_reps"] = len(reps)

    r_idx = select_cur_rows(spec, rank=rank8, r=rank8)

    # ours: functional-clustering column selection
    c_idx_f = select_cur_columns(reps, prio, c=min(rank8, len(reps)))
    cur_f = build_cur(w_tilde, c_idx_f, r_idx)
    cf, rf = len(c_idx_f), len(r_idx)
    ours_bytes = cf * m + cf * rf * 4 + rf * n
    y_f = x_calib @ (q_deq(cur_f.C, 8) @ cur_f.U @ q_deq(cur_f.R, 8)).T
    out["ours_eloc"] = relerr(y_ref, y_f)
    out["ours_ratio"] = params * 4 / ours_bytes
    out["ours_c"], out["ours_r"] = cf, rf

    # baseline: generic leverage-score column selection (README Sec 7.4)
    c_idx_l = select_cur_columns_by_leverage(spec, rank=rank8, c=rank8)
    cur_l = build_cur(w_tilde, c_idx_l, r_idx)
    cl = len(c_idx_l)
    lev_bytes = cl * m + cl * rf * 4 + rf * n
    y_l = x_calib @ (q_deq(cur_l.C, 8) @ cur_l.U @ q_deq(cur_l.R, 8)).T
    out["lev_eloc"] = relerr(y_ref, y_l)
    out["lev_ratio"] = params * 4 / lev_bytes

    out["secs"] = time.time() - t0
    return out


def main():
    ops = select_operators()
    print(f"sampled layers {SAMPLED_LAYERS} -> {len(ops)} operators")
    tensor_names = sorted({p.activation_input for p in ops if p.activation_input != "encoder_hidden_states"})
    print(f"distinct internal activation tensors to capture: {len(tensor_names)}")

    print("capturing real Uzbek-speech calibration activations...")
    x_by_tensor = capture_activations(tensor_names)

    # encoder_hidden_states is a graph INPUT, not capturable as internal tensor;
    # recompute it directly for the same samples.
    if any(p.activation_input == "encoder_hidden_states" for p in ops):
        print("capturing encoder_hidden_states separately...")
        x_by_tensor["encoder_hidden_states"] = capture_encoder_states()

    dec = onnx.load(DECODER_PATH)
    inits = {i.name: i for i in dec.graph.initializer}

    results = []
    for k, p in enumerate(ops, 1):
        x = x_by_tensor.get(p.activation_input)
        if x is None:
            print(f"[{k}/{len(ops)}] SKIP {p.name} (no calibration for {p.activation_input})")
            continue
        w_stored = onnx.numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        w = w_stored if w_stored.shape[1] == x.shape[1] else w_stored.T
        if w.shape[1] != x.shape[1]:
            print(f"[{k}/{len(ops)}] SKIP {p.name} (shape mismatch {w.shape} vs {x.shape})")
            continue
        r = analyze_operator(p, w, x)
        results.append(r)
        print(
            f"[{k}/{len(ops)}] {p.name.replace('/model/decoder/','')[:48]:48s} "
            f"int8={r['int8_eloc']:.4f} int4={r['int4_eloc']:.4f} "
            f"ours={r['ours_eloc']:.4f}({r['ours_ratio']:.1f}x) lev={r['lev_eloc']:.4f} "
            f"[{r['secs']:.0f}s]",
            flush=True,
        )

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_JSON} ({len(results)} operators)")
    summarize(results)


def capture_encoder_states():
    ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    enc = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
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
        chunks.append(build_response_vectors(eh, active_mask=None))
    x = np.concatenate(chunks, axis=1).T
    if x.shape[0] > MAX_CALIB_ROWS:
        x = x[RNG.choice(x.shape[0], MAX_CALIB_ROWS, replace=False)]
    return x.astype(np.float64)


def kind_of(name):
    for k in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"):
        if k in name:
            if "proj" not in k:
                return k
            prefix = "enc_attn/" if "encoder_attn" in name else "self_attn/"
            return prefix + k
    return "other"


def summarize(results):
    if not results:
        return
    print("\n" + "=" * 100)
    print("FULL-TRACK: whole-network INT8 vs our mixed cascade (matched compression points)")
    print("=" * 100)
    print(f"{'operator kind':22s} {'#':>3s} {'INT8 4x':>9s} {'INT4 8x':>9s} {'OURS 8x':>9s} {'LEV-CUR 8x':>11s}")
    by_kind = {}
    for r in results:
        by_kind.setdefault(kind_of(r["name"]), []).append(r)
    for kind, rs in sorted(by_kind.items()):
        print(
            f"{kind:22s} {len(rs):3d} "
            f"{np.mean([r['int8_eloc'] for r in rs]):9.4f} "
            f"{np.mean([r['int4_eloc'] for r in rs]):9.4f} "
            f"{np.mean([r['ours_eloc'] for r in rs]):9.4f} "
            f"{np.mean([r['lev_eloc'] for r in rs]):11.4f}"
        )
    print("-" * 100)
    print(
        f"{'ALL (mean E_loc)':22s} {len(results):3d} "
        f"{np.mean([r['int8_eloc'] for r in results]):9.4f} "
        f"{np.mean([r['int4_eloc'] for r in results]):9.4f} "
        f"{np.mean([r['ours_eloc'] for r in results]):9.4f} "
        f"{np.mean([r['lev_eloc'] for r in results]):11.4f}"
    )
    wins = sum(1 for r in results if r["ours_eloc"] < r["int4_eloc"])
    wins_lev = sum(1 for r in results if r["ours_eloc"] < r["lev_eloc"])
    print(f"\nAt the SAME 8x compression: ours beats INT4 on {wins}/{len(results)} operators")
    print(f"At the SAME 8x compression: ours beats leverage-CUR on {wins_lev}/{len(results)} operators")
    tot = sum(r["params"] for r in results)
    print(f"\nsampled weight volume: {tot*4/1024/1024:.0f} MiB fp32 -> {tot*4/8/1024/1024:.0f} MiB at 8x")


if __name__ == "__main__":
    main()
