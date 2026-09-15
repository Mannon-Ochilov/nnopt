"""Measure each operator's INFLUENCE on the encoder output, then use it.

Why this exists (README Sec 8.3.10-F): allocating rank to minimize the raw
sum of per-operator errors cut that sum by 12.8% while WER improved 58%.
The sum is not what the output responds to. Some operators' local error is
amplified on the way to the encoder output, others' is absorbed, and the
allocator was treating them as interchangeable.

Method: perturb ONE operator at a time (low-rank at a fixed probe rank,
everything else untouched), run the real encoder, and record the relative
error of the FINAL encoder output. The ratio

    c_i = E_glob(only i perturbed) / E_loc(i)

is that operator's influence coefficient. Sessions are built from
serialized bytes rather than files -- 48 full encoder writes to a USB disk
would dominate the runtime.

Then: re-run the rank allocation minimizing sum_i c_i * E_i(r_i) instead of
sum_i E_i(r_i), and compare end-to-end.
"""

import gc
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from calib_utils import ENCODER_PATH, capture_activations, encoder_feeds
from nnopt.cascade.error_propagation import fit_aggregation, influence_coefficients
from nnopt.cascade.rank_allocation import OperatorCurve, allocate_greedy, uniform_allocation
from nnopt.cur.lowrank_baselines import activation_aware_svd, output_relative_error
from nnopt.profiler.graph_profiler import profile_onnx_model

OUT_DIR = "models/_alloc"
CURVE_CACHE = f"{OUT_DIR}/curves.json"
OUT_JSON = "experiments/results_influence.json"
N_CALIB_UTT = 8
FIT_ROWS = 4096
LAYERS_PER_GROUP = 6
N_PROBE_UTT = 4          # utterances used to measure encoder-output error
PROBE_RANK = 200         # perturbation strength for the influence probe
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}


def ffn_ops(profs):
    return [p for p in profs if p.weight_initializer and ("/fc1/" in p.name or "/fc2/" in p.name)]


def layer_of(name):
    import re
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def splice_one(base_model, p, a, b):
    """Return a copy of the graph with ONLY operator p replaced by a@b."""
    model = onnx.ModelProto()
    model.CopyFrom(base_model)
    g = model.graph
    base = p.name.replace("/", "_").replace(".", "_")
    g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{base}_B"))
    g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{base}_A"))
    new_nodes = [
        helper.make_node("MatMul", [p.activation_input, f"{base}_B"], [f"{base}_h"], name=f"{base}_i1"),
        helper.make_node("MatMul", [f"{base}_h", f"{base}_A"], [p.output_name], name=f"{base}_i2"),
    ]
    rebuilt = []
    for nd in g.node:
        if p.output_name in nd.output:
            rebuilt.extend(new_nodes)
        else:
            rebuilt.append(nd)
    del g.node[:]
    g.node.extend(rebuilt)
    kept = [i for i in g.initializer if i.name != p.weight_initializer]
    del g.initializer[:]
    g.initializer.extend(kept)
    return model


def encoder_outputs(model_bytes, feeds):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(model_bytes, sess_options=so, providers=["CPUExecutionProvider"])
    outs = [sess.run(None, f)[0].astype(np.float64) for f in feeds]
    del sess
    return outs


def rel_err(ref_list, hyp_list):
    num = sum(float(np.linalg.norm(r - h) ** 2) for r, h in zip(ref_list, hyp_list))
    den = sum(float(np.linalg.norm(r) ** 2) for r in ref_list)
    return float(np.sqrt(num / (den + 1e-18)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    profs = profile_onnx_model(ENCODER_PATH, free_dims=ENC_DIMS)
    ops = ffn_ops(profs)
    feeds_cal = encoder_feeds(0, N_CALIB_UTT)
    feeds_probe = encoder_feeds(0, N_PROBE_UTT)

    print("FP32 encoder chiqishi (baza)...")
    base_model = onnx.load(ENCODER_PATH)
    ref_out = encoder_outputs(base_model.SerializeToString(), feeds_probe)
    print(f"  {len(ref_out)} namuna, chiqish shakli {ref_out[0].shape}")

    curves = json.load(open(CURVE_CACHE)) if os.path.exists(CURVE_CACHE) else {}
    inits = {i.name: i for i in base_model.graph.initializer}
    layers = sorted({layer_of(p.name) for p in ops})

    e_glob, e_loc = {}, {}
    t_start = time.time()
    for gi in range(0, len(layers), LAYERS_PER_GROUP):
        group = layers[gi:gi + LAYERS_PER_GROUP]
        gops = [p for p in ops if layer_of(p.name) in group]
        tensors = sorted({p.activation_input for p in gops})
        print(f"\nqatlamlar {group[0]}-{group[-1]}: faollashuv yig'ilmoqda...", flush=True)
        x_by = capture_activations(ENCODER_PATH, tensors, feeds_cal, max_rows=FIT_ROWS)
        for p in gops:
            x = x_by[p.activation_input]
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            if w.shape[1] != x.shape[1]:
                w = w.T
            r = min(PROBE_RANK, min(w.shape))
            lr = activation_aware_svd(w, x[:FIT_ROWS], r)
            u, s, vt = np.linalg.svd(lr, full_matrices=False)
            rr = min(r, len(s))
            sq = np.sqrt(s[:rr])
            a, b = u[:, :rr] * sq, sq[:, None] * vt[:rr, :]

            e_loc[p.name] = output_relative_error(w, a @ b, x[:FIT_ROWS])
            perturbed = splice_one(base_model, p, a, b)
            out = encoder_outputs(perturbed.SerializeToString(), feeds_probe)
            e_glob[p.name] = rel_err(ref_out, out)
            del perturbed, out, w, lr, u, s, vt, a, b
            gc.collect()
            print(f"  {p.name.split('/')[-2]:>4s} L{layer_of(p.name):<2d} "
                  f"E_loc={e_loc[p.name]:.4f}  E_glob={e_glob[p.name]:.4f}  "
                  f"c={e_glob[p.name]/max(e_loc[p.name],1e-12):.2f}", flush=True)
        del x_by
        gc.collect()
    print(f"\n[{time.time()-t_start:.0f}s]")

    coeffs = influence_coefficients(e_glob, e_loc)
    json.dump({"e_glob": e_glob, "e_loc": e_loc, "coefficients": coeffs},
              open(OUT_JSON, "w"), indent=2)

    vals = np.array(list(coeffs.values()))
    print("\n" + "=" * 84)
    print("TA'SIR KOEFFITSIENTLARI c_i = E_glob(faqat i) / E_loc(i)")
    print("=" * 84)
    print(f"  min={vals.min():.2f}  median={np.median(vals):.2f}  max={vals.max():.2f}  "
          f"tarqoqlik={vals.max()/max(vals.min(),1e-9):.1f}x")
    top = sorted(coeffs.items(), key=lambda kv: -kv[1])
    print("\n  Eng ta'sirchan 5 ta:")
    for nm, c in top[:5]:
        print(f"    {nm.replace('/encoder/',''):44s} c={c:.2f}")
    print("  Eng kam ta'sirchan 5 ta:")
    for nm, c in top[-5:]:
        print(f"    {nm.replace('/encoder/',''):44s} c={c:.2f}")

    # ---- re-allocate using influence-weighted errors ----
    if curves:
        clist = []
        for p in ops:
            c = curves[p.name]
            w_i = coeffs.get(p.name, 1.0)
            clist.append(OperatorCurve(p.name, c["cost"],
                                       {int(r): e * w_i for r, e in c["errors"].items()}))
        budget = sum(c.param_cost_per_rank * 409 for c in clist)
        uni = uniform_allocation(clist, budget)
        gre = allocate_greedy(clist, budget, step=8)
        print("\n" + "=" * 84)
        print("TA'SIR bilan tortilgan taqsimot (teng byudjet)")
        print("=" * 84)
        print(f"  bir xil rank : tortilgan xato {uni.total_error:.4f}")
        print(f"  ta'sir bilan : tortilgan xato {gre.total_error:.4f}  "
              f"({(uni.total_error-gre.total_error)/uni.total_error*100:.1f}% yaxshilanish)")
        json.dump({"ranks_influence_weighted": gre.ranks},
                  open("experiments/results_influence_ranks.json", "w"), indent=2)
        rs = sorted(gre.ranks.items(), key=lambda kv: kv[1])
        print("\n  Eng past rank olganlar:")
        for nm, r in rs[:5]:
            print(f"    {nm.replace('/encoder/',''):44s} r={r}")
        print("  Eng yuqori rank olganlar:")
        for nm, r in rs[-5:]:
            print(f"    {nm.replace('/encoder/',''):44s} r={r}")


if __name__ == "__main__":
    main()
