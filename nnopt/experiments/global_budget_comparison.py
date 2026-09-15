"""Comparing the criteria without giving any of them our layer allocation.

Every structural comparison so far imposed OUR per-layer channel counts on
the baselines. That isolates the criterion cleanly, but it also hands each
baseline an allocation it did not choose -- and one result depends on it:
magnitude collapsed at the aggressive budget while being asked to strip 66-73%
from layers 0-4, which its own scores might never have selected. If that
collapse is an artifact of our allocation, the paper is making a claim it
cannot support.

Here only the TOTAL budget is fixed. Each method distributes it across layers
by its own rule:

  ours        tau drives it; the per-layer counts fall out of the grouping
  magnitude   one global ranking over all channels of all layers
  Wanda       uniform sparsity per layer, as published
  FLAP        its own adaptive allocation, via a per-layer normalised score

The totals are taken from our tau = 0.99 and tau = 0.95 maps so the arms land
on the same 267 MiB and 254 MiB artifacts already measured, which keeps the
new numbers comparable to every table produced so far.

A caveat worth stating: global ranking compares raw scores across layers whose
scales differ, and that is the method's own assumption rather than something
imposed here. Uniform allocation makes the opposite assumption. Neither is
obviously right, which is the reason to let each method use the one it was
published with.
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

OUT_DIR = "models/_global_budget"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 12
FIT_ROWS = 4096
N_CHANNELS = 4096

# (method, allocation) -- allocation is the rule the method was published with.
ARMS = [("magnitude", "global"), ("wanda", "uniform"), ("flap", "adaptive")]
TAUS = (0.99, 0.95)


def total_removed(tau):
    """Channels our grouping removes in total, and the layers it touches."""
    per_layer, total = {}, 0
    for f in sorted(glob.glob(f"models/_prune/prune_L*_tau{tau}.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        kept = len(np.load(f, allow_pickle=True)["keep"])
        per_layer[li] = N_CHANNELS - kept
        total += N_CHANNELS - kept
    return total, per_layer


def collect_scores(layers):
    """Per-layer channel scores for all three criteria, plus what each needs."""
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}

    feeds = encoder_feeds(0, N_CALIB)
    data = {}
    t0 = time.time()
    for li in sorted(layers):
        if li not in fc1 or li not in fc2:
            continue
        p1, p2 = fc1[li], fc2[li]
        x_by = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                   max_rows=FIT_ROWS)
        x = x_by[p2.activation_input][:FIT_ROWS]
        w2s = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
        w1 = numpy_helper.to_array(inits[p1.weight_initializer]).astype(np.float64)
        if w1.shape[1] != w2.shape[1]:
            w1 = w1.T
        bname = bias_name_for(model, p1)
        bias = (numpy_helper.to_array(inits[bname]).astype(np.float64)
                if bname else None)

        wn = np.linalg.norm(w2, axis=0)
        flap = (wn ** 2) * np.var(x, axis=0)
        data[li] = {
            "magnitude": wn,
            "wanda": wn * np.linalg.norm(x, axis=0),
            # FLAP normalises per layer before comparing across them, which is
            # what makes its allocation adaptive rather than raw-score global.
            "flap": flap / (np.linalg.norm(flap) + 1e-12),
            "w1": w1, "w2": w2, "bias": bias, "bias_name": bname,
            "w1_init": p1.weight_initializer, "w2_init": p2.weight_initializer,
        }
        print(f"  L{li:<2d} ballar hisoblandi [{time.time()-t0:.0f}s]",
              flush=True)
        del x_by, x
        gc.collect()
    return data


def allocate(data, method, mode, budget, per_layer_ref):
    """{layer: keep indices} for one method under its own allocation rule."""
    if mode == "uniform":
        # Same fraction everywhere, over the layers our method touches.
        frac = budget / (len(per_layer_ref) * N_CHANNELS)
        return {li: np.sort(np.argsort(-data[li][method])[
            :N_CHANNELS - int(round(frac * N_CHANNELS))])
            for li in per_layer_ref if li in data}

    # global / adaptive: one ranking across every channel of every layer.
    flat = [(data[li][method][j], li, j)
            for li in per_layer_ref if li in data
            for j in range(N_CHANNELS)]
    flat.sort(key=lambda t: t[0])
    drop = {}
    for _, li, j in flat[:budget]:
        drop.setdefault(li, []).append(j)
    return {li: np.setdiff1d(np.arange(N_CHANNELS),
                             np.array(drop.get(li, []), dtype=int))
            for li in per_layer_ref if li in data}


def to_map(data, keeps):
    out = {}
    for li, keep in keeps.items():
        d = data[li]
        out[li] = {"keep": keep, "w1": d["w1"][:, keep],
                   "bias": None if d["bias"] is None else d["bias"][keep],
                   "w2": d["w2"][:, keep].T, "bias_name": d["bias_name"],
                   "w1_init": d["w1_init"], "w2_init": d["w2_init"]}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for tau in TAUS:
        budget, per_layer_ref = total_removed(tau)
        pending = [(m, a) for m, a in ARMS
                   if not os.path.exists(f"{OUT_DIR}/enc_{m}_{a}_t{tau}.onnx")]
        if not pending:
            print(f"[tau={tau}] barcha armlar mavjud")
            continue

        print(f"\n[tau={tau}] umumiy byudjet {budget:,} kanal "
              f"({len(per_layer_ref)} qatlamda), taqsimot har bir usulning "
              f"o'ziniki")
        data = collect_scores(per_layer_ref)

        from gptq_plus_pruning import build_gptq_model
        for method, mode in pending:
            path = f"{OUT_DIR}/enc_{method}_{mode}_t{tau}.onnx"
            keeps = allocate(data, method, mode, budget, per_layer_ref)
            spread = sorted((N_CHANNELS - len(k)) for k in keeps.values())
            print(f"\n  [{method}/{mode}] qatlam bo'yicha olib tashlash: "
                  f"min {spread[0]}, mediana {spread[len(spread)//2]}, "
                  f"maks {spread[-1]}", flush=True)
            build_gptq_model(f"{OUT_DIR}/_tmp.onnx", path,
                             to_map(data, keeps), f"{method}-{mode}")
            print(f"    saqlandi: {path}  "
                  f"{os.path.getsize(path)/1024**2:.0f} MiB")
        del data
        gc.collect()

    print("\nBizning arm (mos yozuvlar): tau=0.99 -> 267 MiB / 0.1833, "
          "tau=0.95 -> 254 MiB / 0.2006.")


if __name__ == "__main__":
    main()
