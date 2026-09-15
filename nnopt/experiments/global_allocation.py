"""Allocate rank against MEASURED GLOBAL damage curves, and check WER.

Three allocation objectives have now been tried, each fixing a flaw in the
previous one:

  1. sum_i E_loc_i(r_i)              -- local output error per operator.
     Gave WER 0.0729 (vs 0.1719 uniform), but it weights operators by how
     much THEY are perturbed, not by how much the network's output cares.
  2. sum_i c_i * E_loc_i(r_i)        -- local error scaled by an influence
     coefficient measured at one rank. Assumes E_glob is linear in E_loc;
     the data says otherwise (E_loc spans 160x while E_glob spans 4x at
     fixed rank), and the weighted objective improved only 4%.
  3. sum_i E_glob_i(r_i)             -- THIS. The encoder-output error
     caused by perturbing operator i alone, measured directly at three
     ranks (global_curves.py, 144 encoder runs). No propagation law is
     assumed, and it is the quantity the cascade actually wants to keep
     small.

The measured curves show the return on rank varies 15x across operators
(2.71x to 41.69x from rank 128 to 550), so the choice of objective is not
cosmetic.
"""

import gc
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import WhisperTokenizer

from calib_utils import ENCODER_PATH, MODEL_DIR, capture_activations, encoder_feeds, load_audio
from nnopt.bench.latency import make_session, measure_latency
from nnopt.cascade.rank_allocation import OperatorCurve, allocate_greedy, uniform_allocation
from nnopt.cur.lowrank_baselines import activation_aware_svd
from nnopt.profiler.graph_profiler import profile_onnx_model
from wer_cer_whole_network import error_rate, greedy_decode, normalize

OUT_DIR = "models/_alloc"
GLOBAL_CURVES = "experiments/results_global_curves.json"
LOCAL_CURVES = f"{OUT_DIR}/curves.json"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
N_CALIB_UTT, FIT_ROWS = 8, 4096
LAYERS_PER_GROUP = 6
N_EVAL_UTT = 8
WARMUP, MEASURED = 3, 10
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
OUT_JSON = "experiments/results_global_allocation.json"


def ffn_ops(profs):
    return [p for p in profs if p.weight_initializer and ("/fc1/" in p.name or "/fc2/" in p.name)]


def layer_of(name):
    import re
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def build_from_ranks(ranks, ops, feeds_cal, path_q):
    model = onnx.load(ENCODER_PATH)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replacement, dead, total = {}, set(), 0
    layers = sorted({layer_of(p.name) for p in ops})
    for gi in range(0, len(layers), LAYERS_PER_GROUP):
        group = layers[gi:gi + LAYERS_PER_GROUP]
        gops = [p for p in ops if layer_of(p.name) in group]
        x_by = capture_activations(ENCODER_PATH, sorted({p.activation_input for p in gops}),
                                   feeds_cal, max_rows=FIT_ROWS)
        for p in gops:
            x = x_by[p.activation_input][:FIT_ROWS]
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            if w.shape[1] != x.shape[1]:
                w = w.T
            r = min(ranks[p.name], min(w.shape))
            lr = activation_aware_svd(w, x, r)
            u, s, vt = np.linalg.svd(lr, full_matrices=False)
            rr = min(r, len(s))
            sq = np.sqrt(s[:rr])
            a, b = u[:, :rr] * sq, sq[:, None] * vt[:rr, :]
            total += a.size + b.size
            dead.add(p.weight_initializer)
            tag = p.name.replace("/", "_").replace(".", "_")
            g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{tag}_B"))
            g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{tag}_A"))
            replacement[p.output_name] = [
                helper.make_node("MatMul", [p.activation_input, f"{tag}_B"], [f"{tag}_h"], name=f"{tag}_a1"),
                helper.make_node("MatMul", [f"{tag}_h", f"{tag}_A"], [p.output_name], name=f"{tag}_a2"),
            ]
            del x, w, lr, u, s, vt, a, b
        del x_by
        gc.collect()
    rebuilt = []
    for nd in g.node:
        hit = next((o for o in nd.output if o in replacement), None)
        rebuilt.extend(replacement[hit] if hit else [nd])
    del g.node[:]
    g.node.extend(rebuilt)
    kept = [i for i in g.initializer if i.name not in dead]
    del g.initializer[:]
    g.initializer.extend(kept)
    tmp = path_q.replace(".onnx", "_fp32.onnx")
    onnx.save(model, tmp)
    quantize_dynamic(tmp, path_q, weight_type=QuantType.QInt8)
    os.remove(tmp)
    return total


def score(enc_path, feeds_eval, texts, tok, prompt_ids, dec_sess):
    lat = measure_latency(make_session(enc_path, intra_op_threads=1), name=enc_path,
                          fixed_feed=feeds_eval[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(enc_path, sess_options=so, providers=["CPUExecutionProvider"])
    wers, cers = [], []
    for feed, ref in zip(feeds_eval, texts):
        states = enc.run(None, feed)[0].astype(np.float32)
        hyp = normalize(tok.decode(greedy_decode(dec_sess, states, prompt_ids), skip_special_tokens=True))
        ref_n = normalize(ref)
        wers.append(error_rate(ref_n.split(), hyp.split()))
        cers.append(error_rate(list(ref_n), list(hyp)))
    del enc
    return lat.median_ms, float(np.mean(wers)), float(np.mean(cers))


def main():
    profs = profile_onnx_model(ENCODER_PATH, free_dims=ENC_DIMS)
    ops = ffn_ops(profs)
    gc_data = json.load(open(GLOBAL_CURVES))

    curves = [
        OperatorCurve(p.name, gc_data[p.name]["cost"],
                      {int(r): e for r, e in gc_data[p.name]["e_glob"].items()})
        for p in ops
    ]
    budget = sum(c.param_cost_per_rank * 409 for c in curves)
    uni = uniform_allocation(curves, budget)
    gre = allocate_greedy(curves, budget, step=8)

    print("=" * 84)
    print("GLOBAL EGRI CHIZIQLAR bo'yicha taqsimot (teng byudjet)")
    print("=" * 84)
    print(f"  bir xil rank : global zarar yig'indisi {uni.total_error:.4f}")
    print(f"  taqsimlangan : global zarar yig'indisi {gre.total_error:.4f}  "
          f"({(uni.total_error-gre.total_error)/uni.total_error*100:.1f}% yaxshilanish)")

    rs = sorted(gre.ranks.items(), key=lambda kv: kv[1])
    print("\n  Eng past rank:")
    for nm, r in rs[:5]:
        print(f"    {nm.replace('/encoder/',''):40s} r={r}")
    print("  Eng yuqori rank:")
    for nm, r in rs[-5:]:
        print(f"    {nm.replace('/encoder/',''):40s} r={r}")

    # how different is this from the local-error allocation?
    if os.path.exists(LOCAL_CURVES):
        loc = json.load(open(LOCAL_CURVES))
        lcurves = [OperatorCurve(p.name, loc[p.name]["cost"],
                                 {int(r): e for r, e in loc[p.name]["errors"].items()}) for p in ops]
        lgre = allocate_greedy(lcurves, budget, step=8)
        diff = [abs(gre.ranks[n] - lgre.ranks[n]) for n in gre.ranks]
        print(f"\n  Lokal-xato taqsimotidan farq: o'rtacha {np.mean(diff):.0f} rank, "
              f"maks {max(diff)} rank")

    feeds_cal = encoder_feeds(0, N_CALIB_UTT)
    feeds_eval = encoder_feeds(12, N_EVAL_UTT)
    _, texts = load_audio(12, N_EVAL_UTT)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    dec_sess = ort.InferenceSession(DECODER_INT8, sess_options=so, providers=["CPUExecutionProvider"])

    p_q = f"{OUT_DIR}/enc_globalalloc.onnx"
    if not os.path.exists(p_q):
        print("\nModel qurilmoqda...")
        build_from_ranks(gre.ranks, ops, feeds_cal, p_q)
    ms, wer, cer = score(p_q, feeds_eval, texts, tok, prompt_ids, dec_sess)

    json.dump({"ranks": gre.ranks, "ms": ms, "wer": wer, "cer": cer,
               "sum_global_uniform": uni.total_error, "sum_global_alloc": gre.total_error},
              open(OUT_JSON, "w"), indent=2)

    print("\n" + "=" * 84)
    print("MAQSAD FUNKSIYASINI TANLASH — uchdan-uchgacha taqqoslash (teng byudjet)")
    print("=" * 84)
    print(f"{'Maqsad':34s} {'ms':>9s} {'WER':>8s} {'CER':>8s}")
    print("-" * 84)
    print(f"{'bir xil rank':34s} {6371.7:9.1f} {0.1719:8.4f} {0.0417:8.4f}")
    print(f"{'sum E_loc (lokal xato)':34s} {6197.9:9.1f} {0.0729:8.4f} {0.0208:8.4f}")
    print(f"{'sum E_glob (global zarar)':34s} {ms:9.1f} {wer:8.4f} {cer:8.4f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
