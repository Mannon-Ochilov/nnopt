"""Wanda as published: unstructured, per-output-row, no weight update.

Sec 4.9d compared against a Wanda-SCORED baseline that removed whole
channels. That adaptation was deliberate -- putting structured removal
against unstructured zeroing would have confounded the selection criterion
with the granularity, and the question there was about the criterion. But it
is not the published method, and the difference is not cosmetic: the
published form leaves the tensor SHAPE untouched, which on dense CPU kernels
means it buys neither bytes nor arithmetic.

The published method (Sun et al., ICLR 2024, arXiv:2306.11695) scores each
individual weight

    S_ij = |W_ij| * ||X_j||_2

and, crucially, ranks within each OUTPUT ROW rather than globally, zeroing
the smallest fraction in every row. There is no weight update afterwards;
that is precisely what distinguishes it from SparseGPT.

Sparsity here is matched to the PARAMETER fraction our structural arm removes
in the same layer, so the two are nominally equally compressed. What differs
is what that nominal figure buys:

    ours       tensor shrinks   -> fewer bytes AND fewer MACs
    published  tensor unchanged -> zeros are still stored and still multiplied

Both arms are then GPTQ-quantized, matching our pipeline order (prune, then
quantize), so the quantizer is not the variable.
"""

import gc
import glob
import os
import time

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from calib_utils import (
    ENCODER_PATH,
    capture_activations,
    encoder_feeds,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import layer_of
from nnopt.quantizer.baselines import gptq_quantize

OUT_DIR = "models/_wanda_unstr"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 12
MAX_ROWS = 4096
Q8 = 127
LAYERS_PER_GROUP = 4
TAUS = {0.99: "models/_prune/prune_L{li}_tau0.99.npz",
        0.95: "models/_prune/prune_L{li}_tau0.95.npz"}


def sparsity_targets(tau):
    """{layer: fraction of FFN parameters our structural arm removes there}."""
    out = {}
    for f in sorted(glob.glob(f"models/_prune/prune_L*_tau{tau}.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        keep = np.load(f, allow_pickle=True)["keep"]
        if len(keep) == 4096:
            continue
        out[li] = 1.0 - len(keep) / 4096.0
    return out


def wanda_mask(w, x_norm, sparsity):
    """Zero the lowest-scoring `sparsity` fraction of each OUTPUT ROW.

    w is (out, in) and x_norm is the per-input-feature activation norm, so the
    score matrix is |W| * ||X_j|| broadcast along rows -- and the ranking is
    done inside each row, which is the part of the method that matters and the
    part a global threshold would get wrong.
    """
    if sparsity <= 0:
        return w
    scores = np.abs(w) * x_norm[None, :]
    k = int(round(sparsity * w.shape[1]))
    if k <= 0:
        return w
    idx = np.argpartition(scores, k - 1, axis=1)[:, :k]
    out = w.copy()
    np.put_along_axis(out, idx, 0.0, axis=1)
    return out


def build(tau, path_q):
    """Wanda-prune the FFN operators, then GPTQ-quantize the whole encoder."""
    targets = sparsity_targets(tau)
    mean_sp = float(np.mean(list(targets.values()))) * 100
    print(f"  {len(targets)} qatlam, o'rtacha siyraklik {mean_sp:.1f}% "
          f"(bizning strukturaviy arm bilan bir xil)")

    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    ffn = [p for p in profs if "/fc1/" in p.name or "/fc2/" in p.name]
    feeds = encoder_feeds(0, N_CALIB)

    zeroed = 0
    t0 = time.time()
    for gi in range(0, len(ffn), LAYERS_PER_GROUP):
        grp = ffn[gi:gi + LAYERS_PER_GROUP]
        x_by = capture_activations(ENCODER_PATH,
                                   sorted({p.activation_input for p in grp}),
                                   feeds, max_rows=MAX_ROWS)
        for p in grp:
            sp = targets.get(layer_of(p.name), 0.0)
            x = x_by.get(p.activation_input)
            if x is None or sp <= 0:
                continue
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            transposed = w.shape[1] != x.shape[1]
            if transposed:
                w = w.T
            if w.shape[1] != x.shape[1]:
                continue
            masked = wanda_mask(w, np.linalg.norm(x, axis=0), sp)
            zeroed += int(np.sum(masked == 0) - np.sum(w == 0))
            out = masked.T if transposed else masked
            inits[p.weight_initializer].CopyFrom(
                numpy_helper.from_array(out.astype(np.float32),
                                        p.weight_initializer))
            del w, masked, out
        del x_by
        gc.collect()
        print(f"    {min(gi+LAYERS_PER_GROUP, len(ffn))}/{len(ffn)} operator "
              f"[{time.time()-t0:.0f}s]", flush=True)

    tmp = path_q.replace(".onnx", "_fp32.onnx")
    onnx.save(model, tmp)
    print(f"  {zeroed:,} vazn nolga aylantirildi; GPTQ qo'llanmoqda...",
          flush=True)

    # GPTQ over every weighted operator, exactly as the structural arm gets.
    model = onnx.load(tmp)
    inits = {i.name: i for i in model.graph.initializer}
    ops = [p for p in weighted_matmul_profiles(tmp, ENC_DIMS)
           if p.weight_initializer]
    done = 0
    for gi in range(0, len(ops), LAYERS_PER_GROUP * 2):
        grp = ops[gi:gi + LAYERS_PER_GROUP * 2]
        x_by = capture_activations(tmp,
                                   sorted({p.activation_input for p in grp}),
                                   feeds, max_rows=MAX_ROWS)
        for p in grp:
            x = x_by.get(p.activation_input)
            if x is None:
                continue
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            transposed = w.shape[1] != x.shape[1]
            if transposed:
                w = w.T
            if w.shape[1] != x.shape[1]:
                continue
            wq = gptq_quantize(w, x, Q8)
            out = wq.T if transposed else wq
            inits[p.weight_initializer].CopyFrom(
                numpy_helper.from_array(out.astype(np.float32),
                                        p.weight_initializer))
            done += 1
            del w, wq, out
        del x_by
        gc.collect()
        print(f"    {done} operator GPTQ bilan kvantlandi "
              f"[{time.time()-t0:.0f}s]", flush=True)

    onnx.save(model, tmp)
    quantize_dynamic(tmp, path_q, weight_type=QuantType.QInt8, per_channel=True)
    os.remove(tmp)
    return mean_sp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for tau in sorted(TAUS, reverse=True):
        path = f"{OUT_DIR}/enc_wanda_unstr_tau{tau}_gptq.onnx"
        if os.path.exists(path):
            print(f"[tau={tau}] mavjud, "
                  f"{os.path.getsize(path)/1024**2:.0f} MiB")
            continue
        print(f"\n[tau={tau}] Wanda (asl, strukturasiz) quriladi...")
        sp = build(tau, path)
        print(f"  saqlandi: {path}  "
              f"{os.path.getsize(path)/1024**2:.0f} MiB  "
              f"(o'rtacha {sp:.1f}% siyraklik, shakl O'ZGARMAGAN)")

    print("\nTaqqoslash uchun bizning strukturaviy arm: tau=0.99 -> 267 MiB, "
          "tau=0.95 -> 254 MiB.")
    print("Strukturasiz variant ~300 MiB da qoladi: nollar ham saqlanadi va "
          "ko'paytiriladi.")


if __name__ == "__main__":
    main()
