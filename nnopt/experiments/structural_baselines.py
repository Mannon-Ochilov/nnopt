"""Is functional grouping better than the standard channel-pruning criteria?

The cascade's own contribution is the STRUCTURAL axis, and until now it had no
published competitor: the comparisons in Sec 4.9a were against quantizers
(GPTQ, AWQ, RTN) and the ones in Sec 4.5 against low-rank factorizations
(SVD, CUR). Nothing tested whether the redundancy criterion itself -- group
channels whose calibration responses are collinear, then fold each into its
representative -- beats simply deleting the channels that look least
important.

Three criteria, everything else held fixed. Each layer removes EXACTLY the
number of channels our method removed there, so the budget is identical
layer by layer and only the choice of which channels differs:

  ours        group by cos(h_j, h_p) >= tau, keep representatives, and
              COMPENSATE: W2[:, p] += gamma_j * W2[:, j]
  magnitude   keep the largest ||W2[:, j]||, delete the rest outright
  wanda       keep the largest ||W2[:, j]|| * ||h_j||, delete the rest

W2 is the down-projection, so its column j is the weight vector that carries
intermediate channel j into the block output -- the same quantity our
grouping is already scored against, which keeps the comparison about the
CRITERION rather than about which matrix one chooses to look at. Magnitude
and Wanda differ from each other by exactly the activation term, so the pair
also isolates what calibration data buys a pruning criterion.

Neither baseline compensates, because neither defines a representative to
fold into; that is a property of the published methods, not a handicap
imposed here, and it is the part our criterion is claimed to add.

All three arms are then quantized with GPTQ, the cascade's operating point
(Sec 4.9a), and scored by final_wer_testsplit.py on 300 test utterances.
"""

import gc
import glob
import json
import os
import re
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

PRUNE_DIR = "models/_prune"
OUT_DIR = "models/_struct_base"
OUT_JSON = "experiments/results_structural_baselines.json"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 12
FIT_ROWS = 4096
METHODS = ("magnitude", "wanda")


def layer_of(name):
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def our_keep_counts():
    """{layer: how many channels our method kept} -- the budget to match."""
    out = {}
    for f in sorted(glob.glob(f"{PRUNE_DIR}/prune_L*_tau0.99.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        z = np.load(f, allow_pickle=True)
        keep = z["keep"]
        if len(keep) == 4096:          # nothing removed in this layer
            continue
        out[li] = int(len(keep))
    return out


def bias_name_for(model, fc1_profile):
    """The Add node consuming fc1's output carries the FFN bias."""
    for nd in model.graph.node:
        if nd.op_type == "Add" and fc1_profile.output_name in nd.input:
            for inp in nd.input:
                if inp != fc1_profile.output_name:
                    return inp
    return None


def build_maps(counts):
    """Per-layer keep sets for each baseline criterion, matched to `counts`."""
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}

    feeds = encoder_feeds(0, N_CALIB)
    maps = {m: {} for m in METHODS}
    t0 = time.time()

    for li in sorted(counts):
        p1, p2 = fc1.get(li), fc2.get(li)
        if p1 is None or p2 is None:
            continue
        x_by = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                   max_rows=FIT_ROWS)
        x = x_by[p2.activation_input][:FIT_ROWS]          # (rows, 4096) = h

        w2_stored = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w2 = w2_stored if w2_stored.shape[1] == x.shape[1] else w2_stored.T   # (1024, 4096)
        w1 = numpy_helper.to_array(inits[p1.weight_initializer]).astype(np.float64)
        if w1.shape[1] != w2.shape[1]:
            w1 = w1.T                                     # (1024, 4096)
        bname = bias_name_for(model, p1)
        bias = (numpy_helper.to_array(inits[bname]).astype(np.float64)
                if bname else None)

        w_norm = np.linalg.norm(w2, axis=0)               # ||W2[:, j]||
        h_norm = np.linalg.norm(x, axis=0)                # ||h_j||
        scores = {"magnitude": w_norm, "wanda": w_norm * h_norm}

        k = counts[li]
        for m in METHODS:
            keep = np.array(sorted(np.argsort(-scores[m])[:k]))
            maps[m][li] = {
                "keep": keep,
                "w1": w1[:, keep],
                "bias": None if bias is None else bias[keep],
                "w2": w2[:, keep].T,                      # no compensation
                "bias_name": bname,
                "w1_init": p1.weight_initializer,
                "w2_init": p2.weight_initializer,
            }
        overlap = len(set(maps["magnitude"][li]["keep"].tolist())
                      & set(maps["wanda"][li]["keep"].tolist())) / k * 100
        print(f"  L{li:<2d} saqlanadi {k:5d}/4096  "
              f"magnitude/wanda kesishuvi {overlap:5.1f}%  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del x_by, x, w1, w2
        gc.collect()
    return maps


def save_maps(maps):
    os.makedirs(OUT_DIR, exist_ok=True)
    for m, per_layer in maps.items():
        for li, d in per_layer.items():
            np.savez_compressed(
                f"{OUT_DIR}/{m}_L{li}.npz", keep=d["keep"], w1=d["w1"],
                w2=d["w2"],
                bias=d["bias"] if d["bias"] is not None else np.array([]),
                bias_name=str(d["bias_name"]), w1_init=d["w1_init"],
                w2_init=d["w2_init"])


def load_map(method):
    """One method's prune map. Loaded on demand rather than all at once: the
    float64 weights run to roughly a gigabyte per method, and holding two of
    them alongside the ONNX copies was enough to kill the process."""
    out = {}
    for f in sorted(glob.glob(f"{OUT_DIR}/{method}_L*.npz")):
        li = int(f.split("_L")[1].split(".")[0])
        z = np.load(f, allow_pickle=True)
        bn = str(z["bias_name"])
        out[li] = {"keep": z["keep"], "w1": z["w1"], "w2": z["w2"],
                   "bias": z["bias"] if bn != "None" else None,
                   "bias_name": bn, "w1_init": str(z["w1_init"]),
                   "w2_init": str(z["w2_init"])}
    return out


def main():
    counts = our_keep_counts()
    total_removed = sum(4096 - c for c in counts.values())
    print(f"{len(counts)} qatlamda qisqartirish, jami {total_removed} kanal "
          f"olib tashlanadi — bazalar uchun byudjet AYNAN shu\n")

    have = all(os.path.exists(f"{OUT_DIR}/{m}_L{li}.npz")
               for m in METHODS for li in counts)
    if have:
        print("xaritalar keshda")
    else:
        maps = build_maps(counts)
        save_maps(maps)
        del maps
        gc.collect()

    from gptq_plus_pruning import build_gptq_model
    os.makedirs(OUT_DIR, exist_ok=True)
    sizes = {}
    for m in METHODS:
        path = f"{OUT_DIR}/enc_{m}_gptq.onnx"
        if os.path.exists(path):
            sizes[m] = os.path.getsize(path) / 1024 ** 2
            print(f"[{m}] mavjud, {sizes[m]:.0f} MiB")
            continue
        print(f"\n[{m}] GPTQ bilan kvantlanmoqda...", flush=True)
        pm = load_map(m)
        build_gptq_model(f"{OUT_DIR}/_tmp_{m}.onnx", path, pm, m)
        del pm
        gc.collect()
        sizes[m] = os.path.getsize(path) / 1024 ** 2
        print(f"  saqlandi: {path}  {sizes[m]:.0f} MiB")

    json.dump({"keep_counts": counts, "mib": sizes}, open(OUT_JSON, "w"), indent=2)
    print("\nModellar tayyor. WER uchun final_wer_testsplit.py ishga tushiriladi.")
    print("Taqqoslash bazasi: 'qisqartirish + GPTQ' (bizniki) = 0.1833, "
          "267 MiB.")


if __name__ == "__main__":
    main()
