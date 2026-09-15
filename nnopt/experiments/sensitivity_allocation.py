"""Sensitivity-driven rank allocation vs uniform rank, at equal budget.

whole_encoder_v2.py gave every FFN operator the same rank and produced
held-out errors from 0.02 to 0.20 -- a 10x spread. That is direct evidence
that uniform rank is misallocating capacity: sensitive operators are being
damaged while tolerant ones hold rank they do not need.

This measures each operator's error-vs-rank curve on held-out activations,
feeds those curves to nnopt.cascade.rank_allocation (greedy = exact for
separable convex objectives), and compares the resulting non-uniform
assignment against the uniform one AT THE SAME TOTAL PARAMETER BUDGET.

Output: per-operator ranks chosen, total error under each scheme, and the
built ONNX models scored end-to-end (latency + WER/CER), so the comparison
lands on the metric that decides.
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
from nnopt.cascade.rank_allocation import OperatorCurve, allocate_greedy, uniform_allocation
from nnopt.cur.lowrank_baselines import activation_aware_svd, output_relative_error
from nnopt.profiler.graph_profiler import profile_onnx_model
from wer_cer_whole_network import error_rate, greedy_decode, normalize

OUT_DIR = "models/_alloc"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
N_CALIB_UTT = 8
FIT_ROWS, EVAL_ROWS = 4096, 1024
LAYERS_PER_GROUP = 6
N_EVAL_UTT = 8
WARMUP, MEASURED = 3, 10
PROBE_RANKS = [64, 128, 200, 300, 409, 550]
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
OUT_JSON = "experiments/results_allocation.json"


def ffn_ops(profs):
    return [p for p in profs if p.weight_initializer and ("/fc1/" in p.name or "/fc2/" in p.name)]


def layer_of(name):
    import re
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def measure_curves(ops, feeds_cal):
    """Per-operator held-out error at each probe rank."""
    curves, meta = {}, {}
    enc_model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in enc_model.graph.initializer}
    layers = sorted({layer_of(p.name) for p in ops})

    for gi in range(0, len(layers), LAYERS_PER_GROUP):
        group = layers[gi:gi + LAYERS_PER_GROUP]
        gops = [p for p in ops if layer_of(p.name) in group]
        tensors = sorted({p.activation_input for p in gops})
        print(f"  qatlamlar {group[0]}-{group[-1]}...", flush=True)
        x_by = capture_activations(ENCODER_PATH, tensors, feeds_cal, max_rows=FIT_ROWS + EVAL_ROWS)
        for p in gops:
            x = x_by[p.activation_input]
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            if w.shape[1] != x.shape[1]:
                w = w.T
            m, n = w.shape
            x_fit, x_eval = x[:FIT_ROWS], x[FIT_ROWS:FIT_ROWS + EVAL_ROWS]
            errs = {}
            for r in PROBE_RANKS:
                rr = min(r, min(m, n))
                errs[rr] = output_relative_error(w, activation_aware_svd(w, x_fit, rr), x_eval)
            curves[p.name] = OperatorCurve(p.name, m + n, errs)
            meta[p.name] = (m, n)
            print(f"    {p.name.split('/')[-2]:>4s} L{layer_of(p.name):<2d} "
                  + " ".join(f"r{r}={errs[min(r,min(m,n))]:.4f}" for r in PROBE_RANKS), flush=True)
            del x, w, x_fit, x_eval
        del x_by
        gc.collect()
    return curves, meta


def build_from_ranks(ranks, ops, feeds_cal, path_q):
    """Factorize each operator at its assigned rank and splice into the graph."""
    model = onnx.load(ENCODER_PATH)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replacement, dead, total = {}, set(), 0
    layers = sorted({layer_of(p.name) for p in ops})

    for gi in range(0, len(layers), LAYERS_PER_GROUP):
        group = layers[gi:gi + LAYERS_PER_GROUP]
        gops = [p for p in ops if layer_of(p.name) in group]
        tensors = sorted({p.activation_input for p in gops})
        x_by = capture_activations(ENCODER_PATH, tensors, feeds_cal, max_rows=FIT_ROWS)
        for p in gops:
            x = x_by[p.activation_input]
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            if w.shape[1] != x.shape[1]:
                w = w.T
            r = min(ranks[p.name], min(w.shape))
            lr = activation_aware_svd(w, x[:FIT_ROWS], r)
            u, s, vt = np.linalg.svd(lr, full_matrices=False)
            rr = min(r, len(s))
            sq = np.sqrt(s[:rr])
            a, b = u[:, :rr] * sq, sq[:, None] * vt[:rr, :]
            total += a.size + b.size
            dead.add(p.weight_initializer)
            base = p.name.replace("/", "_").replace(".", "_")
            g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{base}_B"))
            g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{base}_A"))
            replacement[p.output_name] = [
                helper.make_node("MatMul", [p.activation_input, f"{base}_B"], [f"{base}_h"], name=f"{base}_l1"),
                helper.make_node("MatMul", [f"{base}_h", f"{base}_A"], [p.output_name], name=f"{base}_l2"),
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
    feeds_cal = encoder_feeds(0, N_CALIB_UTT)

    # Measuring the curves costs ~11 min, so cache them: the allocation
    # itself is cheap and gets re-run while tuning.
    CURVE_CACHE = f"{OUT_DIR}/curves.json"
    if os.path.exists(CURVE_CACHE):
        print("1) Sezgirlik egri chiziqlari keshdan o'qilmoqda")
        raw = json.load(open(CURVE_CACHE))
        curves = {k: OperatorCurve(k, v["cost"], {int(r): e for r, e in v["errors"].items()})
                  for k, v in raw.items()}
    else:
        print("1) Sezgirlik egri chiziqlarini o'lchash")
        t0 = time.time()
        curves, _meta = measure_curves(ops, feeds_cal)
        print(f"   [{time.time()-t0:.0f}s]")
        json.dump({k: {"cost": c.param_cost_per_rank, "errors": c.errors}
                   for k, c in curves.items()}, open(CURVE_CACHE, "w"), indent=2)

    clist = [curves[p.name] for p in ops]
    # budget = the uniform-rank-409 configuration, so both schemes spend the same
    budget = sum(c.param_cost_per_rank * 409 for c in clist)

    uni = uniform_allocation(clist, budget)
    gre = allocate_greedy(clist, budget, step=8)
    print(f"\n2) Byudjet {budget:,} parametr")
    print(f"   bir xil rank : jami xato {uni.total_error:.4f}, parametr {uni.total_params:,}")
    print(f"   sezgirlikka  : jami xato {gre.total_error:.4f}, parametr {gre.total_params:,}")
    print(f"   nazariy yaxshilanish: {(uni.total_error-gre.total_error)/uni.total_error*100:.1f}%")

    ranks_sorted = sorted(gre.ranks.items(), key=lambda kv: kv[1])
    print("\n   Tanlangan ranklar (eng kichikdan):")
    for nm, r in ranks_sorted[:5]:
        print(f"     {nm.replace('/encoder/','')[:44]:44s} r={r}")
    print("     ...")
    for nm, r in ranks_sorted[-5:]:
        print(f"     {nm.replace('/encoder/','')[:44]:44s} r={r}")

    feeds_eval = encoder_feeds(12, N_EVAL_UTT)
    _, texts = load_audio(12, N_EVAL_UTT)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    dec_sess = ort.InferenceSession(DECODER_INT8, sess_options=so, providers=["CPUExecutionProvider"])

    rows = []
    for label, alloc in (("bir xil rank", uni), ("sezgirlikka asoslangan", gre)):
        print(f"\n3) [{label}] model qurilmoqda...")
        p_q = f"{OUT_DIR}/enc_{'uniform' if alloc is uni else 'greedy'}.onnx"
        if not os.path.exists(p_q):
            build_from_ranks(alloc.ranks, ops, feeds_cal, p_q)
        ms, wer, cer = score(p_q, feeds_eval, texts, tok, prompt_ids, dec_sess)
        rows.append({"scheme": label, "total_error": alloc.total_error,
                     "params": alloc.total_params, "ms": ms, "wer": wer, "cer": cer,
                     "ranks": alloc.ranks})
        print(f"   {ms:.1f} ms  WER={wer:.4f}  CER={cer:.4f}")

    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    print("\n" + "=" * 88)
    print("RANK TAQSIMOTI: bir xil vs sezgirlikka asoslangan (TENG byudjet)")
    print("=" * 88)
    print(f"{'Sxema':26s} {'Parametr':>14s} {'Jami xato':>11s} {'ms':>9s} {'WER':>8s} {'CER':>8s}")
    print("-" * 88)
    for r in rows:
        print(f"{r['scheme']:26s} {r['params']:14,d} {r['total_error']:11.4f} "
              f"{r['ms']:9.1f} {r['wer']:8.4f} {r['cer']:8.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
