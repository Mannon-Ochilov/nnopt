"""What someone does when they don't derive anything: pick a ratio and prune.

The framework spends hours choosing a configuration, so it owes an answer to
the obvious objection -- that a practitioner could reach the same place in ten
minutes by picking a round number. These are that practitioner's models.

Blindness has three separate axes, and the baselines here are blind on all of
them at once, which is what makes them the honest comparison:

  ratio        a round number (30%, 50%) rather than a figure derived from L3
  allocation   the same fraction from every layer, ignoring where redundancy is
  criterion    weight magnitude, the default in every structured-pruning tool

One deliberate concession: these baselines GET our compensation. A library
user would not compensate, and uncompensated removal collapses the model --
we measured 1.3393 at a comparable budget. Handing the baseline the better
treatment makes it harder to beat, and any advantage that survives is
therefore an advantage over the strongest version of the naive approach
rather than over a straw man.

Quantization is the same GPTQ pass as every other arm, so the comparison
isolates the structural decision.
"""

import gc
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
from ffn_prune_endtoend import bias_name_for, layer_of

OUT_DIR = "models/_blind"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 6
FIT_ROWS = 4096
N_CHANNELS = 4096
RATIOS = (0.30, 0.50)


def build_maps(removal):
    """Uniform magnitude pruning with compensation to the nearest survivor.

    Compensation needs a target for each dropped channel. Magnitude pruning
    has no notion of a group, so each dropped channel is folded into the
    surviving channel its calibration response is closest to, with the same
    least-squares coefficient our own method uses. That is the most generous
    reading of "magnitude pruning plus compensation" available.
    """
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    layers = sorted(li for li in fc2 if li in fc1)
    keep_n = int(round(N_CHANNELS * (1.0 - removal)))

    feeds = encoder_feeds(0, N_CALIB)
    out, t0 = {}, time.time()
    for li in layers:
        p1, p2 = fc1[li], fc2[li]
        x = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                max_rows=FIT_ROWS)[p2.activation_input]
        w2s = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
        w1 = numpy_helper.to_array(inits[p1.weight_initializer]).astype(np.float64)
        if w1.shape[1] != w2.shape[1]:
            w1 = w1.T
        bname = bias_name_for(model, p1)
        bias = (numpy_helper.to_array(inits[bname]).astype(np.float64)
                if bname else None)

        scores = np.linalg.norm(w2, axis=0)          # the magnitude criterion
        keep = np.sort(np.argsort(-scores)[:keep_n])
        drop = np.setdiff1d(np.arange(N_CHANNELS), keep)

        h = x.T
        norms = np.linalg.norm(h, axis=1) + 1e-12
        w2c = w2.copy()
        keep_dirs = h[keep] / norms[keep][:, None]
        for j in drop:
            cos = keep_dirs @ (h[j] / norms[j])
            p = keep[int(np.argmax(cos))]
            gamma = float(np.dot(h[j], h[p]) / (np.dot(h[p], h[p]) + 1e-12))
            w2c[:, p] += gamma * w2[:, j]

        out[li] = {"keep": keep, "w1": w1[:, keep],
                   "bias": None if bias is None else bias[keep],
                   "w2": w2c[:, keep].T, "bias_name": bname,
                   "w1_init": p1.weight_initializer,
                   "w2_init": p2.weight_initializer}
        print(f"  L{li:<2d} {keep_n}/{N_CHANNELS} saqlandi "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del x, w1, w2, w2c, h
        gc.collect()
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from gptq_plus_pruning import build_gptq_model

    for removal in RATIOS:
        path = f"{OUT_DIR}/enc_magnitude_r{removal:.2f}_gptq.onnx"
        if os.path.exists(path):
            print(f"mavjud: {path}  "
                  f"{os.path.getsize(path)/1024**2:.0f} MiB")
            continue
        print(f"\n[ko'r-ko'rona {removal*100:.0f}%] magnitude, bir xil "
              f"taqsimot, kompensatsiya bilan")
        pm = build_maps(removal)
        print("  GPTQ...", flush=True)
        build_gptq_model(f"{OUT_DIR}/_tmp.onnx", path, pm,
                         f"blind{removal:.2f}")
        del pm
        gc.collect()
        print(f"  saqlandi: {path}  "
              f"{os.path.getsize(path)/1024**2:.0f} MiB")


if __name__ == "__main__":
    main()
