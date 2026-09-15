"""Whole-encoder proof of the cascade's case-3 claim.

cache_cliff_encoder.py showed, on ONE operator, that adding low-rank on top
of the mandatory INT8 buys 1.6x-6.9x more speed for a small accuracy cost,
and that activation-aware SVD is ~14x more accurate than CUR at equal rank
on this operator. This scales that to the whole encoder and measures the
thing that actually matters: end-to-end encoder latency and final WER/CER.

Applied to all 48 FFN operators (24 fc1 + 24 fc2) -- the ones
find_cur_regime.py flagged as over budget. Attention projections are left
at INT8, matching the cascade's "softest sufficient change" rule.

Pipeline per variant: low-rank factorization (fit on calibration) ->
2-MatMul chain spliced into the graph -> quantize_dynamic INT8 over the
whole encoder -> measure latency, then decode with the INT8 decoder and
score WER/CER against the reference transcripts.
"""

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
from nnopt.cur.lowrank_baselines import activation_aware_svd
from nnopt.profiler.graph_profiler import profile_onnx_model
from wer_cer_whole_network import EOT, SOT, error_rate, greedy_decode, normalize

OUT_DIR = "models/_enc_lowrank"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
N_CALIB = 4
N_EVAL = 8
MAX_ROWS = 512
SEQ = 1500
WARMUP, MEASURED = 3, 10
RANKS = [409, 200]
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": SEQ}
OUT_JSON = "experiments/results_whole_encoder.json"


def ffn_ops(profs):
    return [p for p in profs if p.weight_initializer and ("/fc1/" in p.name or "/fc2/" in p.name)]


def build_variant(rank, x_by_tensor, ops, path_fp, path_q):
    """Splice a 2-MatMul low-rank chain in place of every FFN MatMul."""
    model = onnx.load(ENCODER_PATH)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replacement, dead = {}, set()
    total_bytes, elocs = 0, []

    for k, p in enumerate(ops, 1):
        x = x_by_tensor.get(p.activation_input)
        if x is None:
            continue
        w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        if w.shape[1] != x.shape[1]:
            continue
        m, n = w.shape
        r = min(rank, min(m, n))

        lr = activation_aware_svd(w, x, r)
        u, s, vt = np.linalg.svd(lr, full_matrices=False)
        rr = min(r, len(s))
        sq = np.sqrt(s[:rr])
        a = (u[:, :rr] * sq)              # (m, r)
        b = (sq[:, None] * vt[:rr, :])    # (r, n)

        y_ref = x @ w.T
        elocs.append(float(np.linalg.norm(y_ref - x @ (a @ b).T) / (np.linalg.norm(y_ref) + 1e-9)))
        total_bytes += a.size + b.size     # int8 after quantization

        dead.add(p.weight_initializer)
        base = p.name.replace("/", "_").replace(".", "_")
        g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{base}_B"))
        g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{base}_A"))
        replacement[p.output_name] = [
            helper.make_node("MatMul", [p.activation_input, f"{base}_B"], [f"{base}_h"], name=f"{base}_lr1"),
            helper.make_node("MatMul", [f"{base}_h", f"{base}_A"], [p.output_name], name=f"{base}_lr2"),
        ]
        if k % 12 == 0:
            print(f"    {k}/{len(ops)} operator...", flush=True)

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
    return total_bytes, float(np.mean(elocs)) if elocs else 0.0


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
    print(f"encoder: {len(all_w)} vaznli operator, {total_params:,} parametr; "
          f"FFN: {len(ops)} operator, {ffn_params:,} parametr")

    print("kalibrlash faollashuvlarini yig'ish...")
    tensors = sorted({p.activation_input for p in ops})
    feeds_cal = encoder_feeds(0, N_CALIB)
    x_by = capture_activations(ENCODER_PATH, tensors, feeds_cal, max_rows=MAX_ROWS)

    feeds_eval = encoder_feeds(12, N_EVAL)
    _, texts = load_audio(12, N_EVAL)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    dec_sess = ort.InferenceSession(DECODER_INT8, sess_options=so, providers=["CPUExecutionProvider"])

    rows = []

    print("\n[FP32 encoder]")
    ms, wer, cer = score(ENCODER_PATH, feeds_eval, texts, tok, prompt_ids, dec_sess)
    rows.append({"variant": "FP32 encoder", "bytes": total_params * 4, "ms": ms,
                 "wer": wer, "cer": cer, "eloc": 0.0})
    print(f"  {ms:.1f} ms  WER={wer:.4f}")

    p_int8 = f"{OUT_DIR}/enc_int8.onnx"
    if not os.path.exists(p_int8):
        print("[INT8 encoder] kvantlash...")
        quantize_dynamic(ENCODER_PATH, p_int8, weight_type=QuantType.QInt8)
    print("[INT8 encoder]")
    ms, wer, cer = score(p_int8, feeds_eval, texts, tok, prompt_ids, dec_sess)
    rows.append({"variant": "INT8 (majburiy)", "bytes": total_params, "ms": ms,
                 "wer": wer, "cer": cer, "eloc": None})
    print(f"  {ms:.1f} ms  WER={wer:.4f}")

    for rank in RANKS:
        print(f"\n[INT8 + SVD r={rank}] qurilmoqda...")
        t0 = time.time()
        p_fp = f"{OUT_DIR}/enc_lr{rank}_fp32.onnx"
        p_q = f"{OUT_DIR}/enc_lr{rank}_int8.onnx"
        if not os.path.exists(p_q):
            ffn_bytes, eloc = build_variant(rank, x_by, ops, p_fp, p_q)
        else:
            ffn_bytes, eloc = 0, 0.0
        nbytes = (total_params - ffn_params) + ffn_bytes
        print(f"  qurildi [{time.time()-t0:.0f}s], o'rtacha operator E_loc={eloc:.4f}")
        ms, wer, cer = score(p_q, feeds_eval, texts, tok, prompt_ids, dec_sess)
        rows.append({"variant": f"INT8 + SVD r={rank}", "bytes": nbytes, "ms": ms,
                     "wer": wer, "cer": cer, "eloc": eloc})
        print(f"  {ms:.1f} ms  WER={wer:.4f}  CER={cer:.4f}")

    json.dump(rows, open(OUT_JSON, "w"), indent=2)

    base_b, base_ms = rows[0]["bytes"], rows[0]["ms"]
    int8_ms = rows[1]["ms"]
    print("\n" + "=" * 100)
    print("BUTUN ENCODER: INT8 ustiga past-rank qo'shishning uchdan-uchgacha ta'siri")
    print("=" * 100)
    print(f"{'Variant':22s} {'Vazn(MiB)':>10s} {'Siqish':>8s} {'Latency(ms)':>12s} "
          f"{'FP32 ga':>9s} {'INT8 ga':>9s} {'WER':>8s} {'CER':>8s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['variant']:22s} {r['bytes']/1024**2:10.0f} {base_b/r['bytes']:7.2f}x "
              f"{r['ms']:12.1f} {base_ms/r['ms']:8.2f}x {int8_ms/r['ms']:8.2f}x "
              f"{r['wer']:8.4f} {r['cer']:8.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
