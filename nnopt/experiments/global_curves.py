"""Measure GLOBAL damage curves E_glob_i(r) directly, without assuming a
propagation law.

influence_measurement.py probed each operator at a single rank and formed
c_i = E_glob/E_loc, then rescaled the whole local-error curve by it. That
assumes E_glob is linear in E_loc. The measured data contradicts the
assumption: across 48 operators E_loc spans 160x (0.0014 to 0.225) while
E_glob stays inside 4x (0.012 to 0.047), i.e. the encoder absorbs local
perturbations strongly and sub-linearly. Unsurprisingly the rescaled
allocation only improved the weighted objective by 4%.

So drop the law and measure the thing the allocator actually needs: for
each operator, the encoder-output error as a function of ITS rank, with
every other operator untouched. The allocator then minimizes

    sum_i E_glob_i(r_i)      subject to   sum_i c_i * r_i <= B

which is the same separable-convex problem solved exactly by the existing
greedy allocator -- but with the correct objective and no fitted model in
between.

Cost is the reason this was not done first: 48 operators x 3 ranks = 144
full encoder runs. Probe utterances are kept small for that reason; the
metric is a relative error on the encoder output, which is stable across
utterances.
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
from nnopt.cur.lowrank_baselines import activation_aware_svd, output_relative_error
from nnopt.profiler.graph_profiler import profile_onnx_model

OUT_DIR = "models/_alloc"
OUT_JSON = "experiments/results_global_curves.json"
N_CALIB_UTT = 8
FIT_ROWS = 4096
LAYERS_PER_GROUP = 6
N_PROBE_UTT = 2
PROBE_RANKS = [128, 300, 550]
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}


def ffn_ops(profs):
    return [p for p in profs if p.weight_initializer and ("/fc1/" in p.name or "/fc2/" in p.name)]


def layer_of(name):
    import re
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def splice_one(base_model, p, a, b):
    model = onnx.ModelProto()
    model.CopyFrom(base_model)
    g = model.graph
    tag = p.name.replace("/", "_").replace(".", "_")
    g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{tag}_B"))
    g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{tag}_A"))
    new_nodes = [
        helper.make_node("MatMul", [p.activation_input, f"{tag}_B"], [f"{tag}_h"], name=f"{tag}_g1"),
        helper.make_node("MatMul", [f"{tag}_h", f"{tag}_A"], [p.output_name], name=f"{tag}_g2"),
    ]
    rebuilt = []
    for nd in g.node:
        rebuilt.extend(new_nodes if p.output_name in nd.output else [nd])
    del g.node[:]
    g.node.extend(rebuilt)
    kept = [i for i in g.initializer if i.name != p.weight_initializer]
    del g.initializer[:]
    g.initializer.extend(kept)
    return model


def run_encoder(model_bytes, feeds):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(model_bytes, sess_options=so, providers=["CPUExecutionProvider"])
    outs = [sess.run(None, f)[0].astype(np.float64) for f in feeds]
    del sess
    return outs


def rel_err(ref, hyp):
    num = sum(float(np.linalg.norm(r - h) ** 2) for r, h in zip(ref, hyp))
    den = sum(float(np.linalg.norm(r) ** 2) for r in ref)
    return float(np.sqrt(num / (den + 1e-18)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    profs = profile_onnx_model(ENCODER_PATH, free_dims=ENC_DIMS)
    ops = ffn_ops(profs)
    feeds_cal = encoder_feeds(0, N_CALIB_UTT)
    feeds_probe = encoder_feeds(0, N_PROBE_UTT)

    base_model = onnx.load(ENCODER_PATH)
    ref_out = run_encoder(base_model.SerializeToString(), feeds_probe)
    inits = {i.name: i for i in base_model.graph.initializer}
    layers = sorted({layer_of(p.name) for p in ops})

    results = {}
    t0 = time.time()
    done = 0
    total = len(ops) * len(PROBE_RANKS)
    for gi in range(0, len(layers), LAYERS_PER_GROUP):
        group = layers[gi:gi + LAYERS_PER_GROUP]
        gops = [p for p in ops if layer_of(p.name) in group]
        tensors = sorted({p.activation_input for p in gops})
        print(f"\nqatlamlar {group[0]}-{group[-1]}...", flush=True)
        x_by = capture_activations(ENCODER_PATH, tensors, feeds_cal, max_rows=FIT_ROWS)
        for p in gops:
            x = x_by[p.activation_input][:FIT_ROWS]
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            if w.shape[1] != x.shape[1]:
                w = w.T
            entry = {"cost": int(w.shape[0] + w.shape[1]), "e_glob": {}, "e_loc": {}}
            for rank in PROBE_RANKS:
                r = min(rank, min(w.shape))
                lr = activation_aware_svd(w, x, r)
                u, s, vt = np.linalg.svd(lr, full_matrices=False)
                rr = min(r, len(s))
                sq = np.sqrt(s[:rr])
                a, b = u[:, :rr] * sq, sq[:, None] * vt[:rr, :]
                entry["e_loc"][r] = output_relative_error(w, a @ b, x)
                mdl = splice_one(base_model, p, a, b)
                entry["e_glob"][r] = rel_err(ref_out, run_encoder(mdl.SerializeToString(), feeds_probe))
                del mdl, lr, u, s, vt, a, b
                gc.collect()
                done += 1
            results[p.name] = entry
            gl = entry["e_glob"]
            print(f"  {p.name.split('/')[-2]:>4s} L{layer_of(p.name):<2d} "
                  + "  ".join(f"r{r}:{gl[min(r,min(w.shape))]:.4f}" for r in PROBE_RANKS)
                  + f"   [{done}/{total}, {time.time()-t0:.0f}s]", flush=True)
            del x, w
        del x_by
        gc.collect()

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}  [{time.time()-t0:.0f}s]")

    print("\n" + "=" * 82)
    print("GLOBAL ZARAR EGRI CHIZIQLARI — rank oshgach E_glob qanchalik tushadi?")
    print("=" * 82)
    lo, hi = PROBE_RANKS[0], PROBE_RANKS[-1]
    drops = []
    for nm, e in results.items():
        ks = sorted(int(k) for k in e["e_glob"])
        a, b = e["e_glob"][ks[0]], e["e_glob"][ks[-1]]
        drops.append((nm, a, b, a / max(b, 1e-12)))
    drops.sort(key=lambda z: -z[3])
    print(f"{'operator':44s} {'r=128':>9s} {'r=550':>9s} {'nisbat':>8s}")
    for nm, a, b, ratio in drops[:6]:
        print(f"{nm.replace('/encoder/',''):44s} {a:9.4f} {b:9.4f} {ratio:7.2f}x")
    print("  ...")
    for nm, a, b, ratio in drops[-6:]:
        print(f"{nm.replace('/encoder/',''):44s} {a:9.4f} {b:9.4f} {ratio:7.2f}x")
    arr = np.array([d[3] for d in drops])
    print(f"\nrank 128->550 da E_glob kamayishi: min={arr.min():.2f}x  "
          f"median={np.median(arr):.2f}x  max={arr.max():.2f}x")
    print("Katta nisbat = rank berish shu operatorda ko'p foyda beradi (allokator shularga byudjet beradi).")


if __name__ == "__main__":
    main()
