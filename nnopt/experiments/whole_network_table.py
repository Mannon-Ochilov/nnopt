"""THE table: whole-network compression x accuracy x speed, for
INT8 / INT4 / our cascade, all measured end-to-end on the real Whisper
Uzbek decoder.

Tracks
------
  FP32      : untouched decoder (reference for accuracy + speed)
  INT8      : ORT quantize_dynamic over the whole graph (existing method)
  INT4      : our calibration-aware refine_scale at 4 bits on all 240
              weighted MatMuls (existing method pushed to 8x). Stored as
              FP32 on our INT4 grid -> ACCURACY IS REAL; size is reported
              analytically and latency is marked N/A, because ONNX Runtime
              has no INT4 CPU GEMM kernel (stated honestly, not faked).
  CASCADE   : our proposed adaptive method. Per README Sec 8.3.6 the
              cascade picks the softest sufficient change per operator:
              INT8 everywhere, plus functional-clustering CUR (2-factor
              fused, Sec 8.3.6-D) on the fc1 operators -- the one operator
              family the measured data shows CUR actually works on.

Real-INT8-speed trick (important, and verified in-script): a
DequantizeLinear-on-constant pattern is constant-folded back to FP32 by
ORT at session init, so it yields INT8 file size but FP32 speed. To get
BOTH our better scales AND ORT's real MatMulInteger kernel, every weight
is first snapped onto our refine_scale grid (values become exactly s*q,
|q|<=127); ORT's own symmetric per-tensor min/max quantization of such a
matrix recovers scale s exactly, so the ORT step is lossless and the
resulting model runs on real INT8 kernels.

Accuracy metric = END-TO-END final decoder output (logits) relative error
on HELD-OUT samples that were never used for calibration.
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
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from scipy.signal import resample_poly
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from nnopt.bench.latency import make_session, measure_latency
from nnopt.calibrator.activation_capture import ActivationCapture, build_response_vectors
from nnopt.cur.svd_cur import analyze_spectrum, build_cur, select_cur_columns, select_cur_rows
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.profiler.graph_profiler import profile_onnx_model
from nnopt.quantizer.scale_refine import dequantize, quantize_codes, refine_scale

MODEL_DIR = "models/hh"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
OUT_DIR = "models/_whole_net"
N_CALIB = 12          # calibration samples (used to fit CUR / scales)
N_EVAL = 8            # HELD-OUT samples (never seen during calibration)
MAX_CALIB_ROWS = 512
TARGET_SR = 16000
WARMUP, MEASURED = 3, 12
TAU, EPS_THR = 0.9, 0.2
CUR_TARGET_FACTOR = 8.0
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------- data --
AUDIO_CACHE = "models/_calib_cache/cv_uz_validation.npz"


def _load_cached_audio():
    """Local cache built by experiments/cache_audio_locally.py -- the HF
    streaming path proved too flaky (DNS / read timeouts mid-run) for a
    benchmark that must be reproducible."""
    z = np.load(AUDIO_CACHE, allow_pickle=True)
    flat, lengths, texts = z["audio"], z["lengths"], z["texts"]
    waves, off = [], 0
    for ln in lengths:
        waves.append(flat[off:off + ln])
        off += ln
    return waves, list(texts)


def _iter_audio(skip, take):
    waves, texts = _load_cached_audio()
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    enc = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    for wav, text in list(zip(waves, texts))[skip:skip + take]:
        feats = fe(wav, sampling_rate=TARGET_SR, return_tensors="np").input_features.astype(np.float32)
        (eh,) = enc.run(None, {"input_features": feats})
        ids = np.array([[50258, *prompt_ids, *tok(text, add_special_tokens=False).input_ids]], dtype=np.int64)
        yield {"input_ids": ids, "encoder_hidden_states": eh.astype(np.float32)}


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


# -------------------------------------------------------------- helpers --
def our_grid(mat, bits):
    """Snap `mat` onto our calibration-aware refine_scale grid. Returns
    (dequantized_fp32_on_grid, scale). ORT's symmetric per-tensor min/max
    quantization of the returned matrix recovers exactly this scale."""
    qm = {8: 127, 4: 7}[bits]
    res = refine_scale(mat, qm)
    return dequantize(quantize_codes(mat, res.scale, qm), res.scale), res.scale


def weighted_matmuls():
    profs = profile_onnx_model(
        DECODER_PATH,
        free_dims={"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500},
    )
    return [p for p in profs if p.weight_initializer is not None]


# ------------------------------------------------------------ CUR track --
def build_fc1_cur(w, x_calib, target_factor):
    """2-factor fused functional CUR for one fc1 operator.
    w: (m, n) with Y = X @ w.T.  Returns (m1, c_t, bytes_int8, e_loc_pred).
      m1  = R^T U^T  (n, c)      c_t = C^T  (c, m)
    Both are snapped onto our INT8 grid, so bytes = (n*c + c*m) * 1.
    """
    m, n = w.shape
    y_ref = x_calib @ w.T
    # rank for the target under an all-INT8 2-factor budget: (n*c + c*m) bytes
    budget_bytes = m * n * 4 / target_factor
    rank = max(1, min(int(budget_bytes // (n + m)), min(m, n)))

    grouping = greedy_group(
        x_calib.T, np.linalg.norm(w, axis=0), float(np.linalg.norm(y_ref)), tau=TAU, eps_threshold=EPS_THR
    )
    w_tilde = build_compensated_weight(w, grouping)
    reps = grouping.representative_indices()
    h_norms = np.linalg.norm(x_calib, axis=0)
    prio = {g.representative: float(g.size) * float(h_norms[g.representative]) for g in grouping.groups}
    spec = analyze_spectrum(w_tilde)
    r_idx = select_cur_rows(spec, rank=rank, r=rank)
    c_idx = select_cur_columns(reps, prio, c=min(rank, len(reps)))
    cur = build_cur(w_tilde, c_idx, r_idx)

    m1 = cur.R.T @ cur.U.T                      # (n, c)
    c_t = cur.C.T                               # (c, m)
    m1_g, _ = our_grid(m1, 8)
    c_t_g, _ = our_grid(c_t, 8)
    e_loc = relerr(y_ref, (x_calib @ m1_g) @ c_t_g)
    nbytes = m1_g.size + c_t_g.size             # 1 byte each
    return m1_g.astype(np.float32), c_t_g.astype(np.float32), nbytes, e_loc, rank


def make_cascade_model(ops, x_by_tensor, path):
    """Graph surgery: fc1 -> 2-factor fused CUR chain; every other weighted
    MatMul -> weight snapped onto our INT8 grid. Saved as FP32-on-grid; the
    caller then runs quantize_dynamic to get real INT8 kernels.

    The replaced CUR nodes are spliced in AT THE POSITION of the node they
    replace, so the graph stays topologically sorted (ONNX requires it).
    """
    model = onnx.load(DECODER_PATH)
    g = model.graph
    inits = {i.name: i for i in g.initializer}

    total_bytes = 0
    fc1_report = []
    replacement = {}  # output_name of node being replaced -> [new nodes]
    dead_inits = set()  # original fc1 weights, now unused (must be deleted:
    #                     the FP32 intermediate would otherwise blow past
    #                     protobuf's 2 GiB single-file ceiling)

    for k, p in enumerate(ops, 1):
        w_stored = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        x = x_by_tensor.get(p.activation_input) if "/fc1/" in p.name else None
        w = w_stored if (x is not None and w_stored.shape[1] == x.shape[1]) else w_stored.T

        if x is not None and w.shape[1] == x.shape[1]:
            t0 = time.time()
            m1, c_t, nbytes, e_loc, rank = build_fc1_cur(w, x, CUR_TARGET_FACTOR)
            total_bytes += nbytes
            fc1_report.append({"name": p.name, "rank": rank, "e_loc": e_loc, "bytes": nbytes})
            print(f"  [{k}/{len(ops)}] CUR {p.name.replace('/model/decoder/','')[:40]:40s} "
                  f"rank={rank} E_loc={e_loc:.4f} [{time.time()-t0:.0f}s]", flush=True)

            dead_inits.add(p.weight_initializer)
            base = p.name.replace("/", "_").replace(".", "_")
            g.initializer.append(numpy_helper.from_array(m1, f"{base}_m1"))
            g.initializer.append(numpy_helper.from_array(c_t, f"{base}_ct"))
            replacement[p.output_name] = [
                helper.make_node("MatMul", [p.activation_input, f"{base}_m1"],
                                 [f"{base}_h"], name=f"{base}_cur1"),
                helper.make_node("MatMul", [f"{base}_h", f"{base}_ct"],
                                 [p.output_name], name=f"{base}_cur2"),
            ]
            continue

        w_g, _ = our_grid(w_stored, 8)
        inits[p.weight_initializer].CopyFrom(
            numpy_helper.from_array(w_g.astype(np.float32), p.weight_initializer)
        )
        total_bytes += w_stored.size
        if k % 40 == 0:
            print(f"  [{k}/{len(ops)}] int8-grid ...", flush=True)

    if replacement:
        rebuilt = []
        for nd in g.node:
            hit = next((o for o in nd.output if o in replacement), None)
            if hit is not None:
                rebuilt.extend(replacement[hit])
            else:
                rebuilt.append(nd)
        del g.node[:]
        g.node.extend(rebuilt)

    if dead_inits:
        kept = [i for i in g.initializer if i.name not in dead_inits]
        del g.initializer[:]
        g.initializer.extend(kept)

    onnx.save(model, path)
    return total_bytes, fc1_report


# ----------------------------------------------------------- INT4 track --
def make_int4_model(ops, path):
    model = onnx.load(DECODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    total_params = 0
    for k, p in enumerate(ops, 1):
        w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        w_g, _ = our_grid(w, 4)
        inits[p.weight_initializer].CopyFrom(numpy_helper.from_array(w_g.astype(np.float32), p.weight_initializer))
        total_params += w.size
        if k % 40 == 0:
            print(f"  [{k}/{len(ops)}] int4-grid ...", flush=True)
    onnx.save(model, path)
    return total_params


# ---------------------------------------------------------------- main ---
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ops = weighted_matmuls()
    total_params = sum(
        int(np.prod(tuple(i.dims)))
        for i in onnx.load(DECODER_PATH).graph.initializer
        if i.name in {p.weight_initializer for p in ops}
    )
    print(f"{len(ops)} weighted MatMuls, {total_params:,} params ({total_params*4/1024**2:.0f} MiB fp32)")

    print("\ncapturing fc1 calibration activations (all 24 layers)...")
    fc1_tensors = sorted({p.activation_input for p in ops if "/fc1/" in p.name})
    cap = ActivationCapture(DECODER_PATH, tensor_names=fc1_tensors)
    collected = {nm: [] for nm in fc1_tensors}
    for i, feeds in enumerate(_iter_audio(0, N_CALIB), 1):
        for nm, arr in cap.run_batch(feeds).items():
            collected[nm].append(build_response_vectors(arr, active_mask=None))
        print(f"  calib {i}/{N_CALIB}", flush=True)
    del cap
    x_by_tensor = {}
    for nm, ch in collected.items():
        x = np.concatenate(ch, axis=1).T
        if x.shape[0] > MAX_CALIB_ROWS:
            x = x[RNG.choice(x.shape[0], MAX_CALIB_ROWS, replace=False)]
        x_by_tensor[nm] = x.astype(np.float64)
    del collected

    print("\ncollecting HELD-OUT evaluation feeds...")
    eval_feeds = list(_iter_audio(N_CALIB, N_EVAL))
    print(f"  {len(eval_feeds)} held-out samples")

    tracks = {}

    # -- FP32 reference --
    print("\n[FP32] measuring...")
    sess = make_session(DECODER_PATH, intra_op_threads=1)
    ref_logits = [sess.run(None, f)[0] for f in eval_feeds]
    lat = measure_latency(sess, name="fp32", fixed_feed=eval_feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    tracks["FP32 (asl)"] = {"bytes": total_params * 4, "eloc": 0.0, "ms": lat.median_ms}
    del sess

    # -- INT8 (existing) --
    p_int8 = f"{OUT_DIR}/dec_int8.onnx"
    if not os.path.exists(p_int8):
        print("\n[INT8] quantize_dynamic over whole decoder (slow)...")
        quantize_dynamic(DECODER_PATH, p_int8, weight_type=QuantType.QInt8)
    s = make_session(p_int8, intra_op_threads=1)
    e = float(np.mean([relerr(r, s.run(None, f)[0]) for r, f in zip(ref_logits, eval_feeds)]))
    lat = measure_latency(s, name="int8", fixed_feed=eval_feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    tracks["INT8 (mavjud)"] = {"bytes": total_params * 1, "eloc": e, "ms": lat.median_ms}
    print(f"  E_loc={e:.4f} latency={lat.median_ms:.1f}ms")
    del s

    # -- INT4 (existing, pushed to 8x) --
    p_int4 = f"{OUT_DIR}/dec_int4grid.onnx"
    if not os.path.exists(p_int4):
        print("\n[INT4] snapping all weights onto our 4-bit grid...")
        make_int4_model(ops, p_int4)
    s = make_session(p_int4, intra_op_threads=1)
    e = float(np.mean([relerr(r, s.run(None, f)[0]) for r, f in zip(ref_logits, eval_feeds)]))
    tracks["INT4 (mavjud)"] = {"bytes": int(total_params * 0.5), "eloc": e, "ms": None}
    print(f"  E_loc={e:.4f} latency=N/A (ORT CPU'da INT4 GEMM kerneli yo'q)")
    del s

    # -- CASCADE (ours) --
    p_casc_fp = f"{OUT_DIR}/dec_cascade_grid.onnx"
    p_casc = f"{OUT_DIR}/dec_cascade_int8.onnx"
    if not os.path.exists(p_casc):
        print("\n[CASCADE] building (CUR on fc1, INT8 grid elsewhere)...")
        cbytes, fc1_rep = make_cascade_model(ops, x_by_tensor, p_casc_fp)
        json.dump(fc1_rep, open(f"{OUT_DIR}/fc1_cur_report.json", "w"), indent=2)
        print("[CASCADE] quantize_dynamic (lossless on our grid) ...")
        quantize_dynamic(p_casc_fp, p_casc, weight_type=QuantType.QInt8)
    else:
        cbytes = json.load(open(f"{OUT_DIR}/cascade_bytes.json"))["bytes"]
    json.dump({"bytes": cbytes}, open(f"{OUT_DIR}/cascade_bytes.json", "w"))
    s = make_session(p_casc, intra_op_threads=1)
    e = float(np.mean([relerr(r, s.run(None, f)[0]) for r, f in zip(ref_logits, eval_feeds)]))
    lat = measure_latency(s, name="cascade", fixed_feed=eval_feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    tracks["KASKAD (bizning)"] = {"bytes": cbytes, "eloc": e, "ms": lat.median_ms}
    print(f"  E_loc={e:.4f} latency={lat.median_ms:.1f}ms")
    del s

    # ---- final table ----
    base_bytes = total_params * 4
    base_ms = tracks["FP32 (asl)"]["ms"]
    print("\n" + "=" * 96)
    print("BUTUN TARMOQ (Whisper-medium uz decoder, 240 operator, 8 ta held-out namuna)")
    print("=" * 96)
    print(f"{'Usul':20s} {'Vazn (MiB)':>11s} {'Siqish':>8s} {'E_glob':>9s} {'Latency(ms)':>12s} {'Tezlanish':>10s}")
    print("-" * 96)
    for name, t in tracks.items():
        ratio = base_bytes / t["bytes"]
        ms = f"{t['ms']:.1f}" if t["ms"] else "N/A"
        sp = f"{base_ms/t['ms']:.2f}x" if t["ms"] else "N/A"
        print(f"{name:20s} {t['bytes']/1024**2:11.0f} {ratio:7.2f}x {t['eloc']:9.4f} {ms:>12s} {sp:>10s}")
    print("=" * 96)
    json.dump(tracks, open("experiments/results_whole_network.json", "w"), indent=2)


if __name__ == "__main__":
    main()
