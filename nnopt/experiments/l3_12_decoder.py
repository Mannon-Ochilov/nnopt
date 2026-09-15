"""The decoder half of a 12 MiB cascade -- and the rank the target implies.

The decoder layer is 64 MiB, half of it attention the cascade does not touch.
At L3 = 12 MiB the layer needs 7.62x, INT8 gives 4.00x, and the residual
1.90x must come entirely out of the 32 MiB of FFN. Solving for the rank that
leaves the layer inside budget:

    keep = total / 1.90            per-layer bytes allowed after INT8
    ffn_budget = keep - attention  what is left for the two FFN matrices
    r = ffn_budget / (d_model + d_ff) / 2     both matrices factored at rank r

On this model that lands near rank 41 of a possible 1024. For scale, the
uniform-aggressive arm of Sec 4.9c used rank 409 -- ten times more generous --
and already cost 0.43 WER. Building this one is therefore not a search for a
good configuration; it is measuring how far outside the feasible region a
12 MiB target puts this model, with a number rather than an extrapolation.
"""

import gc
import os
import re
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from calib_utils import DECODER_PATH, capture_activations, decoder_feeds, weighted_matmul_profiles
from nnopt.cur.lowrank_baselines import activation_aware_svd

OUT = "models/_l3_12/dec_l3_12_int8.onnx"
DEC_DIMS = {"batch_size": 1, "decoder_sequence_length": 16,
            "encoder_sequence_length": 1500}
ALPHA, L3_MIB, INT8 = 0.7, 12.0, 4.0
MIB = 1024.0 ** 2
N_CALIB = 8
MAX_ROWS = 2048


def layer_index(name):
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def derived_rank(profs):
    """Rank the cache target implies, from the layer's own byte breakdown."""
    model = onnx.load(DECODER_PATH, load_external_data=False)
    shapes = {i.name: tuple(i.dims) for i in model.graph.initializer}
    ffn, other, dims = {}, {}, []
    for p in profs:
        li = layer_index(p.name)
        if li < 0 or p.weight_initializer not in shapes:
            continue
        shp = shapes[p.weight_initializer]
        n = int(np.prod(shp))
        if "/fc1/" in p.name or "/fc2/" in p.name:
            ffn[li] = ffn.get(li, 0) + n
            dims.append(shp)
        else:
            other[li] = other.get(li, 0) + n

    li = max(ffn, key=lambda k: ffn[k] + other.get(k, 0))
    total_b = (ffn[li] + other.get(li, 0)) * 4
    need = total_b / (ALPHA * L3_MIB * MIB)
    resid = need / INT8
    allowed = total_b / resid
    ffn_budget = allowed - other.get(li, 0) * 4
    # Two matrices, each (d_model x d_ff), factored at the same rank: the
    # factored pair costs r * (d_model + d_ff) parameters per matrix.
    d_model, d_ff = sorted(dims[0])[0], sorted(dims[0])[1]
    r = int(ffn_budget / 4 / (2 * (d_model + d_ff)))
    print(f"qatlam {total_b/MIB:.1f} MiB (FFN {ffn[li]*4/MIB:.1f} + attn "
          f"{other.get(li,0)*4/MIB:.1f}), talab {need:.2f}x, INT8 dan keyin "
          f"{resid:.2f}x")
    print(f"FFN uchun qoladigan byudjet {ffn_budget/MIB:.2f} MiB "
          f"-> rank {r} / {min(d_model, d_ff)}")
    return max(1, r)


def main():
    if os.path.exists(OUT):
        print(f"mavjud: {OUT}  {os.path.getsize(OUT)/MIB:.0f} MiB")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    profs = weighted_matmul_profiles(DECODER_PATH, DEC_DIMS)
    rank = derived_rank(profs)
    ops = [p for p in profs if "/fc1/" in p.name or "/fc2/" in p.name]
    print(f"\n{len(ops)} ta FFN operatori, rank {rank}\n")

    feeds = decoder_feeds(0, N_CALIB)
    model = onnx.load(DECODER_PATH)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replacement, dead = {}, set()
    t0 = time.time()

    for k, p in enumerate(ops, 1):
        x_by = capture_activations(DECODER_PATH, [p.activation_input], feeds,
                                   max_rows=MAX_ROWS)
        x = x_by.get(p.activation_input)
        if x is None:
            continue
        w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        if w.shape[1] != x.shape[1]:
            continue
        r = min(rank, min(w.shape))
        lr = activation_aware_svd(w, x, r)
        u, s, vt = np.linalg.svd(lr, full_matrices=False)
        rr = max(1, min(r, int(np.sum(s > 0))))
        sq = np.sqrt(s[:rr])
        a, b = (u[:, :rr] * sq), (sq[:, None] * vt[:rr, :])

        dead.add(p.weight_initializer)
        base = p.name.replace("/", "_").replace(".", "_")
        g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{base}_B"))
        g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{base}_A"))
        replacement[p.output_name] = [
            helper.make_node("MatMul", [p.activation_input, f"{base}_B"],
                             [f"{base}_h"], name=f"{base}_lr1"),
            helper.make_node("MatMul", [f"{base}_h", f"{base}_A"],
                             [p.output_name], name=f"{base}_lr2"),
        ]
        if k % 8 == 0:
            print(f"  {k}/{len(ops)} [{time.time()-t0:.0f}s]", flush=True)
        del x_by, x, w, lr
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

    tmp = OUT.replace(".onnx", "_fp32.onnx")
    onnx.save(model, tmp, save_as_external_data=True,
              location=os.path.basename(tmp) + ".data", size_threshold=1024)
    quantize_dynamic(tmp, OUT, weight_type=QuantType.QInt8, per_channel=True)
    for f in (tmp, tmp + ".data"):
        if os.path.exists(f):
            os.remove(f)
    print(f"\nsaqlandi: {OUT}  {os.path.getsize(OUT)/MIB:.0f} MiB")


if __name__ == "__main__":
    main()
