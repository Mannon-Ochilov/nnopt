"""FLAP: structured pruning with bias compensation -- the closest competitor.

Every baseline so far was missing one of the two things our own method does.
Magnitude ignores the calibration data; Wanda uses it but discards the removed
channels outright. FLAP (An et al., AAAI 2024, arXiv:2312.11983) has both,
which makes it the informative test: if the mechanism we identified is what
matters, a method that shares it should land where we land.

It differs from ours in HOW it compensates, and its selection criterion is
matched to that choice:

  ours   a removed channel is folded into a REPRESENTATIVE channel, so the
         part of its signal that varies is preserved. The criterion therefore
         looks for collinearity: only a channel that duplicates another can be
         folded into it.

  FLAP   a removed channel is replaced by its MEAN contribution, added into
         the output bias. Only the constant part survives, so the criterion
         instead looks for channels that barely fluctuate -- their varying
         part is small enough to discard.

That is why FLAP scores by fluctuation rather than by magnitude:

    S_j = ||W2[:, j]||^2 * Var(h_j)

and compensates by

    b_out += sum over removed j of  mean(h_j) * W2[:, j]

One deviation is deliberate and must be read with the results: FLAP also
allocates its sparsity across layers adaptively, and here the per-layer counts
are taken from OUR grouping instead, so every arm removes the same number of
channels in the same layer. That isolates criterion plus compensation, which
is the question, but it does mean the published method's third component is
not exercised.
"""

import gc
import glob
import os
import time

import numpy as np
import onnx
from onnx import numpy_helper

from calib_utils import (
    ENCODER_PATH,
    capture_activations,
    encoder_feeds,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import layer_of
from structural_baselines import bias_name_for

OUT_DIR = "models/_flap"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 12
FIT_ROWS = 4096
TAUS = (0.99, 0.95)


def out_bias_name(model, fc2_profile):
    """The Add node consuming fc2's output carries the block output bias."""
    for nd in model.graph.node:
        if nd.op_type == "Add" and fc2_profile.output_name in nd.input:
            for inp in nd.input:
                if inp != fc2_profile.output_name:
                    return inp
    return None


def counts_for(tau):
    out = {}
    for f in sorted(glob.glob(f"models/_prune/prune_L*_tau{tau}.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        keep = np.load(f, allow_pickle=True)["keep"]
        if len(keep) == 4096:
            continue
        out[li] = int(len(keep))
    return out


def build_map(tau):
    """FLAP keep sets plus the bias correction each layer needs."""
    counts = counts_for(tau)
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}

    feeds = encoder_feeds(0, N_CALIB)
    out = {}
    t0 = time.time()
    for li in sorted(counts):
        p1, p2 = fc1.get(li), fc2.get(li)
        if p1 is None or p2 is None:
            continue
        x_by = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                   max_rows=FIT_ROWS)
        x = x_by[p2.activation_input][:FIT_ROWS]                 # (rows, 4096)
        w2s = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T         # (1024, 4096)
        w1 = numpy_helper.to_array(inits[p1.weight_initializer]).astype(np.float64)
        if w1.shape[1] != w2.shape[1]:
            w1 = w1.T
        bname = bias_name_for(model, p1)
        bias = (numpy_helper.to_array(inits[bname]).astype(np.float64)
                if bname else None)
        obname = out_bias_name(model, p2)
        obias = (numpy_helper.to_array(inits[obname]).astype(np.float64)
                 if obname else None)

        score = (np.linalg.norm(w2, axis=0) ** 2) * np.var(x, axis=0)
        k = counts[li]
        keep = np.array(sorted(np.argsort(-score)[:k]))
        drop = np.setdiff1d(np.arange(w2.shape[1]), keep)

        # Mean contribution of the removed channels, folded into the bias.
        correction = w2[:, drop] @ np.mean(x[:, drop], axis=0)
        out[li] = {
            "keep": keep,
            "w1": w1[:, keep],
            "bias": None if bias is None else bias[keep],
            "w2": w2[:, keep].T,
            "bias_name": bname,
            "w1_init": p1.weight_initializer,
            "w2_init": p2.weight_initializer,
            "out_bias_name": obname,
            "out_bias": None if obias is None else obias + correction,
        }
        print(f"  L{li:<2d} saqlanadi {k:5d}/4096   bias tuzatmasi normasi "
              f"{np.linalg.norm(correction):.4f}   [{time.time()-t0:.0f}s]",
              flush=True)
        del x_by, x, w1, w2
        gc.collect()
    return out


def apply_and_quantize(pm, path_q, tag):
    """Splice FLAP's shapes AND its bias correction, then GPTQ as usual."""
    from gptq_plus_pruning import build_gptq_model

    # build_gptq_model handles the shape change and the fc1 bias; the output
    # bias is FLAP-specific, so it is written in afterwards on the quantized
    # graph, where the initializer survives untouched by quantization.
    tmp = f"{OUT_DIR}/_tmp_{tag}.onnx"
    build_gptq_model(tmp, path_q, pm, tag)

    model = onnx.load(path_q)
    inits = {i.name: i for i in model.graph.initializer}
    written = 0
    for d in pm.values():
        nm, val = d.get("out_bias_name"), d.get("out_bias")
        if nm and val is not None and nm in inits:
            inits[nm].CopyFrom(numpy_helper.from_array(
                val.astype(np.float32), nm))
            written += 1
    onnx.save(model, path_q)
    return written


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for tau in TAUS:
        path = f"{OUT_DIR}/enc_flap_tau{tau}_gptq.onnx"
        if os.path.exists(path):
            print(f"[tau={tau}] mavjud, {os.path.getsize(path)/1024**2:.0f} MiB")
            continue
        print(f"\n[tau={tau}] FLAP quriladi (fluktuatsiya mezoni + bias "
              f"kompensatsiyasi)")
        pm = build_map(tau)
        n = apply_and_quantize(pm, path, f"flap{tau}")
        print(f"  {n} qatlamda chiqish biasi tuzatildi")
        print(f"  saqlandi: {path}  {os.path.getsize(path)/1024**2:.0f} MiB")
        del pm
        gc.collect()

    print("\nTaqqoslash: bizniki tau=0.99 -> 267 MiB / 0.1833, "
          "tau=0.95 -> 254 MiB / 0.2006.")


if __name__ == "__main__":
    main()
