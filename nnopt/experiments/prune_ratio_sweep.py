"""Does the selection criterion matter once the budget gets aggressive?

At the cascade's own budget (tau = 0.99, 17.1% of channels removed on
average) the three structural criteria are indistinguishable end to end:
functional grouping 0.1833, magnitude 0.1837, Wanda 0.1850, every paired
interval covering zero. That is a real negative result, but it is a result
about ONE operating point, and a weak one: if the network has that much
slack, any criterion that removes 17% will find removable channels.

The criterion can only be shown to matter -- or shown not to -- where the
slack runs out. This rebuilds all three arms at tau = 0.95, which on the
redundancy sweep removes 68% of layer 0's channels instead of 43% and
raises the network-wide budget correspondingly.

The comparison stays exactly as before: the baselines match our per-layer
channel count exactly, all three are quantized by the same GPTQ pass, and
only the choice of which channels to drop differs.

Two outcomes, both informative:
  criteria diverge  -> the contribution holds, with a measured condition on
                       when it applies
  criteria tie      -> the claim must be rewritten around the budget rather
                       than the criterion
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
from ffn_prune_endtoend import layer_of, prune_layers
from structural_baselines import bias_name_for

TAU = float(os.environ.get("TAU", "0.95"))
# The baseline arms exist to answer "is our criterion better"; when the sweep
# is used to trace OUR quality-vs-tau curve they are dead weight, so they can
# be switched off and roughly two thirds of the build time with them.
WITH_BASELINES = os.environ.get("WITH_BASELINES", "1") != "0"
OUT_DIR = "models/_ratio_sweep"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 12
FIT_ROWS = 4096
# Which baselines to build. Selectable because tracing the criterion gap as a
# function of budget needs Wanda at several tau values but gains nothing from
# rebuilding magnitude, which already collapsed at tau = 0.95 and would only
# collapse harder further out.
BASELINES = tuple(os.environ.get("BASELINES", "magnitude,wanda").split(","))


def baseline_maps(counts):
    """Per-layer keep sets for the baselines, matched to `counts` exactly."""
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}

    feeds = encoder_feeds(0, N_CALIB)
    maps = {m: {} for m in BASELINES}
    t0 = time.time()
    for li in sorted(counts):
        p1, p2 = fc1.get(li), fc2.get(li)
        if p1 is None or p2 is None:
            continue
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

        w_norm = np.linalg.norm(w2, axis=0)
        scores = {"magnitude": w_norm,
                  "wanda": w_norm * np.linalg.norm(x, axis=0)}
        k = counts[li]
        for m in BASELINES:
            keep = np.array(sorted(np.argsort(-scores[m])[:k]))
            maps[m][li] = {"keep": keep, "w1": w1[:, keep],
                           "bias": None if bias is None else bias[keep],
                           "w2": w2[:, keep].T, "bias_name": bname,
                           "w1_init": p1.weight_initializer,
                           "w2_init": p2.weight_initializer}
        print(f"  L{li:<2d} saqlanadi {k:5d}/4096 "
              f"({(1-k/4096)*100:4.1f}% olib tashlandi) [{time.time()-t0:.0f}s]",
              flush=True)
        del x_by, x, w1, w2
        gc.collect()
    return maps


def map_path(method, li, tau=None):
    """Baseline maps are tau-SPECIFIC: they match our per-layer channel counts,
    which change with tau. An earlier version omitted tau from the filename,
    so a cached tau = 0.95 map was silently reused for tau = 0.97 and 0.93 and
    produced three artifacts of identical size instead of the 261/254/248 MiB
    the matched budgets require.
    """
    return f"{OUT_DIR}/{method}_tau{tau if tau is not None else TAU}_L{li}.npz"


def save(maps):
    os.makedirs(OUT_DIR, exist_ok=True)
    for m, per_layer in maps.items():
        for li, d in per_layer.items():
            np.savez_compressed(
                map_path(m, li), keep=d["keep"], w1=d["w1"], w2=d["w2"],
                bias=d["bias"] if d["bias"] is not None else np.array([]),
                bias_name=str(d["bias_name"]), w1_init=d["w1_init"],
                w2_init=d["w2_init"])


def load(method):
    import glob
    out = {}
    for f in sorted(glob.glob(f"{OUT_DIR}/{method}_tau{TAU}_L*.npz")):
        li = int(f.split("_L")[1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        bn = str(z["bias_name"])
        out[li] = {"keep": z["keep"], "w1": z["w1"], "w2": z["w2"],
                   "bias": z["bias"] if bn != "None" else None,
                   "bias_name": bn, "w1_init": str(z["w1_init"]),
                   "w2_init": str(z["w2_init"])}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    pairs = [(fc1[li], fc2[li]) for li in sorted(fc2) if li in fc1]
    print(f"{len(pairs)} ta FFN juftligi, tau = {TAU}\n")

    print("[bizniki] funksional guruhlash (sekin bosqich, keshlanadi)...")
    ours, stats = prune_layers(pairs, encoder_feeds(0, N_CALIB), tau=TAU)
    counts = {li: int(len(d["keep"])) for li, d in ours.items()
              if len(d["keep"]) < 4096}
    removed = sum(4096 - c for c in counts.values())
    mean_frac = np.mean([1 - c / 4096 for c in counts.values()]) * 100
    print(f"\n{len(counts)} qatlamda qisqartirish, jami {removed} kanal, "
          f"o'rtacha {mean_frac:.1f}%\n")
    del ours
    gc.collect()

    if WITH_BASELINES:
        have = all(os.path.exists(map_path(m, li))
                   for m in BASELINES for li in counts)
        if have:
            print("bazalar keshda")
        else:
            print("[bazalar] mos byudjetda quriladi...")
            maps = baseline_maps(counts)
            save(maps)
            del maps
            gc.collect()

    from ffn_prune_endtoend import _cache_path
    from gptq_plus_pruning import build_gptq_model

    arms = [("bizniki", None)]
    if WITH_BASELINES:
        arms += [(m, m) for m in BASELINES]
    for label, key in arms:
        path = f"{OUT_DIR}/enc_{label}_tau{TAU}_gptq.onnx"
        if os.path.exists(path):
            print(f"[{label}] mavjud, {os.path.getsize(path)/1024**2:.0f} MiB")
            continue
        if key is None:
            pm = {}
            for li in counts:
                z = np.load(_cache_path(li, TAU), allow_pickle=True)
                bn = str(z["bias_name"])
                pm[li] = {"keep": z["keep"], "w1": z["w1"], "w2": z["w2"],
                          "bias": z["bias"] if bn != "None" else None,
                          "bias_name": bn, "w1_init": str(z["w1_init"]),
                          "w2_init": str(z["w2_init"])}
        else:
            pm = load(key)
        print(f"\n[{label}] GPTQ bilan kvantlanmoqda...", flush=True)
        build_gptq_model(f"{OUT_DIR}/_tmp_{label}.onnx", path, pm, label)
        del pm
        gc.collect()
        print(f"  saqlandi: {path}  {os.path.getsize(path)/1024**2:.0f} MiB")

    print(f"\nUchala variant tayyor (tau={TAU}, o'rtacha {mean_frac:.1f}% "
          f"olib tashlangan).")
    print("WER uchun final_wer_testsplit.py ga qo'shiladi.")
    print("tau=0.99 dagi natija: bizniki 0.1833, magnitude 0.1837, "
          "wanda 0.1850 — farqlanmagan.")


if __name__ == "__main__":
    main()
