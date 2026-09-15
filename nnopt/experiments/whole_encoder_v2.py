"""Whole-encoder low-rank, with the calibration bug fixed.

The first attempt (whole_encoder_lowrank.py) fit each factorization on 512
calibration rows for ranks up to 409 -- a rows/rank ratio of 1.3, where the
activation-aware solution simply interpolates the calibration set -- and
then reported E_loc on those same rows. calibration_size_sweep.py measured
the damage directly: at rank 409 with 256 rows the fit error is 0.00000
while held-out error is 0.04355, a gap of six orders of magnitude. The
reported per-operator 0.0033 was meaningless and the end-to-end WER blow-up
(0.0667 -> 0.2929) followed from it.

Fixes here:
  * rows/rank >= 10 (the sweep's safe threshold), so rank 409 gets >= 4096
    calibration rows instead of 512;
  * activations captured in float32 and processed in LAYER GROUPS, because
    holding 8192 x 4096 rows for all 24 fc2 operators at once needs several
    GB;
  * E_loc reported on held-out rows the factorization never saw.

Encoder utterances give 1500 positions each, so thousands of rows are cheap
-- the old 512 cap was carried over from decoder scripts, where one
utterance yields only ~16 token positions.
"""

import gc
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import WhisperTokenizer

from calib_utils import ENCODER_PATH, MODEL_DIR, capture_activations, encoder_feeds, load_audio
from nnopt.bench.latency import make_session, measure_latency
from nnopt.cur.lowrank_baselines import activation_aware_svd, output_relative_error
from nnopt.profiler.graph_profiler import profile_onnx_model
from wer_cer_whole_network import error_rate, greedy_decode, normalize

OUT_DIR = "models/_enc_v2"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
N_CALIB_UTT = 8          # 8 x 1500 = 12000 positions available
FIT_ROWS = 4096          # rows/rank = 10.0 at rank 409  (sweep's safe floor)
EVAL_ROWS = 1024
LAYERS_PER_GROUP = 6     # memory: 24 fc2 tensors at 4096x4096 fp32 would be ~6 GB
N_EVAL_UTT = 8
WARMUP, MEASURED = 3, 10
RANKS = [409, 200]
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
OUT_JSON = "experiments/results_whole_encoder_v2.json"


def ffn_ops(profs):
    return [p for p in profs if p.weight_initializer and ("/fc1/" in p.name or "/fc2/" in p.name)]


def layer_of(name):
    import re
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def factorize_all(rank, ops, feeds_cal):
    """Return {op_name: (a, b, eloc_heldout)} processing layers in groups so
    the captured activations never all live at once."""
    out = {}
    enc_model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in enc_model.graph.initializer}
    layers = sorted({layer_of(p.name) for p in ops})

    for gi in range(0, len(layers), LAYERS_PER_GROUP):
        group = layers[gi:gi + LAYERS_PER_GROUP]
        gops = [p for p in ops if layer_of(p.name) in group]
        tensors = sorted({p.activation_input for p in gops})
        print(f"    qatlamlar {group[0]}-{group[-1]}: {len(tensors)} tenzor yig'ilmoqda...", flush=True)
        x_by = capture_activations(ENCODER_PATH, tensors, feeds_cal,
                                   max_rows=FIT_ROWS + EVAL_ROWS)
        for p in gops:
            x = x_by[p.activation_input].astype(np.float32)
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            if w.shape[1] != x.shape[1]:
                w = w.T
            x_fit = x[:FIT_ROWS].astype(np.float64)
            x_eval = x[FIT_ROWS:FIT_ROWS + EVAL_ROWS].astype(np.float64)
            r = min(rank, min(w.shape))
            lr = activation_aware_svd(w, x_fit, r)
            u, s, vt = np.linalg.svd(lr, full_matrices=False)
            rr = min(r, len(s))
            sq = np.sqrt(s[:rr])
            a = u[:, :rr] * sq
            b = sq[:, None] * vt[:rr, :]
            out[p.name] = (a, b, output_relative_error(w, a @ b, x_eval))
            del x, x_fit, x_eval, w, lr, u, s, vt
        del x_by
        gc.collect()
    return out


def build_model(factors, ops, path_fp, path_q):
    model = onnx.load(ENCODER_PATH)
    g = model.graph
    replacement, dead, total_bytes = {}, set(), 0
    for p in ops:
        if p.name not in factors:
            continue
        a, b, _ = factors[p.name]
        dead.add(p.weight_initializer)
        total_bytes += a.size + b.size
        base = p.name.replace("/", "_").replace(".", "_")
        g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{base}_B"))
        g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{base}_A"))
        replacement[p.output_name] = [
            helper.make_node("MatMul", [p.activation_input, f"{base}_B"], [f"{base}_h"], name=f"{base}_lr1"),
            helper.make_node("MatMul", [f"{base}_h", f"{base}_A"], [p.output_name], name=f"{base}_lr2"),
        ]
    rebuilt = []
    for nd in g.node:
        hit = next((o for o in nd.output if o in replacement), None)
        rebuilt.extend(replacement[hit] if hit else [nd])
    del g.node[:]
    g.node.extend(rebuilt)
    kept = [i for i in g.initializer if i.name not in dead]
    del g.initializer[:]
    g.initializer.extend(kept)
    onnx.save(model, path_fp)
    quantize_dynamic(path_fp, path_q, weight_type=QuantType.QInt8)
    os.remove(path_fp)
    return total_bytes


def score(enc_path, feeds_eval, texts, tok, prompt_ids, dec_sess):
    lat = measure_latency(make_session(enc_path, intra_op_threads=1), name=enc_path,
                          fixed_feed=feeds_eval[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(enc_path, sess_options=so, providers=["CPUExecutionProvider"])
    wers, cers = [], []
    for feed, ref in zip(feeds_eval, texts):
        states = enc.run(None, feed)[0].astype(np.float32)
        ids = greedy_decode(dec_sess, states, prompt_ids)
        hyp = normalize(tok.decode(ids, skip_special_tokens=True))
        ref_n = normalize(ref)
        wers.append(error_rate(ref_n.split(), hyp.split()))
        cers.append(error_rate(list(ref_n), list(hyp)))
    del enc
    return lat.median_ms, float(np.mean(wers)), float(np.mean(cers))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    profs = profile_onnx_model(ENCODER_PATH, free_dims=ENC_DIMS)
    ops = ffn_ops(profs)
    all_w = [p for p in profs if p.weight_initializer]
    enc0 = onnx.load(ENCODER_PATH, load_external_data=False)
    dims = {i.name: tuple(i.dims) for i in enc0.graph.initializer}
    total_params = sum(int(np.prod(dims[p.weight_initializer])) for p in all_w)
    ffn_params = sum(int(np.prod(dims[p.weight_initializer])) for p in ops)
    print(f"encoder: {len(all_w)} operator / {total_params:,} parametr; "
          f"FFN: {len(ops)} operator / {ffn_params:,} parametr")
    print(f"kalibrlash: {FIT_ROWS} fit + {EVAL_ROWS} eval qator "
          f"(rank 409 uchun nisbat {FIT_ROWS/409:.1f}x)")

    feeds_cal = encoder_feeds(0, N_CALIB_UTT)
    feeds_eval = encoder_feeds(12, N_EVAL_UTT)
    _, texts = load_audio(12, N_EVAL_UTT)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    dec_sess = ort.InferenceSession(DECODER_INT8, sess_options=so, providers=["CPUExecutionProvider"])

    rows = []

    print("\n[FP32 encoder]")
    ms, wer, cer = score(ENCODER_PATH, feeds_eval, texts, tok, prompt_ids, dec_sess)
    rows.append({"variant": "FP32", "bytes": total_params * 4, "ms": ms, "wer": wer, "cer": cer, "eloc": 0.0})
    print(f"  {ms:.1f} ms  WER={wer:.4f}")

    p_int8 = f"{OUT_DIR}/enc_int8.onnx"
    if not os.path.exists(p_int8):
        print("[INT8] kvantlash...")
        quantize_dynamic(ENCODER_PATH, p_int8, weight_type=QuantType.QInt8)
    print("[INT8 (majburiy)]")
    ms, wer, cer = score(p_int8, feeds_eval, texts, tok, prompt_ids, dec_sess)
    rows.append({"variant": "INT8", "bytes": total_params, "ms": ms, "wer": wer, "cer": cer, "eloc": None})
    print(f"  {ms:.1f} ms  WER={wer:.4f}")

    for rank in RANKS:
        print(f"\n[INT8 + SVD r={rank}]")
        t0 = time.time()
        factors = factorize_all(rank, ops, feeds_cal)
        elocs = [v[2] for v in factors.values()]
        p_q = f"{OUT_DIR}/enc_lr{rank}_int8.onnx"
        ffn_bytes = build_model(factors, ops, f"{OUT_DIR}/_tmp_{rank}.onnx", p_q)
        del factors
        gc.collect()
        print(f"  qurildi [{time.time()-t0:.0f}s]  held-out E_loc: "
              f"o'rtacha={np.mean(elocs):.5f} max={np.max(elocs):.5f}")
        ms, wer, cer = score(p_q, feeds_eval, texts, tok, prompt_ids, dec_sess)
        rows.append({"variant": f"INT8 + SVD r={rank}",
                     "bytes": (total_params - ffn_params) + ffn_bytes,
                     "ms": ms, "wer": wer, "cer": cer,
                     "eloc": float(np.mean(elocs)), "eloc_max": float(np.max(elocs))})
        print(f"  {ms:.1f} ms  WER={wer:.4f}  CER={cer:.4f}")

    json.dump(rows, open(OUT_JSON, "w"), indent=2)

    b0, ms0, wer0 = rows[0]["bytes"], rows[0]["ms"], rows[0]["wer"]
    ms8 = rows[1]["ms"]
    print("\n" + "=" * 100)
    print("BUTUN ENCODER (tuzatilgan kalibrlash bilan)")
    print("=" * 100)
    print(f"{'Variant':20s} {'Vazn(MiB)':>10s} {'Siqish':>8s} {'ms':>9s} {'FP32 ga':>9s} "
          f"{'INT8 ga':>9s} {'WER':>8s} {'CER':>8s} {'dWER':>8s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['variant']:20s} {r['bytes']/1024**2:10.0f} {b0/r['bytes']:7.2f}x "
              f"{r['ms']:9.1f} {ms0/r['ms']:8.2f}x {ms8/r['ms']:8.2f}x "
              f"{r['wer']:8.4f} {r['cer']:8.4f} {r['wer']-wer0:+8.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
