"""Does the structural axis compose with ANY quantizer, or only with GPTQ?

Sec 4.9b showed that removing redundant channels costs nothing on top of
GPTQ: 11% less memory, dWER = -0.0014 with an interval covering zero. That
establishes orthogonality against the strongest quantizer, but not that the
property is general -- GPTQ redistributes each column's error into the
columns it has not reached yet, so one could argue it simply absorbs
whatever the pruning step leaves behind, and that a quantizer without that
machinery would not.

This script completes the 2x2. The pruning decision, the calibration data,
the export path and the evaluation set are all held fixed; the only thing
that varies is the quantizer:

                        no pruning      + structural pruning
    GPTQ                A (measured)    B (measured)
    round-to-nearest    C (here)        D (here)

If B - A and D - C are both indistinguishable from zero, orthogonality is a
property of the structural axis rather than a property of GPTQ.

Granularity is held at per-output-channel for all four. Note that the
existing "INT8" baseline in Table 13 is NOT usable as C: whole_encoder_v2.py
built it with quantize_dynamic's default, which is per-TENSOR, and Sec 4.4
showed per-tensor collapses after compensation widens the row ranges 188x.
Mixing granularities would confound the comparison, so C is rebuilt here.

RTN needs no calibration activations, so unlike the GPTQ build there is no
capture pass -- both models are a prune-then-export away.
"""

import glob
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from calib_utils import ENCODER_PATH

PRUNE_DIR = "models/_prune"
OUT_DIR = "models/_rtn"
OUT_JSON = "experiments/results_rtn_pruning.json"
FULL_FFN = 4096


def load_prune_map():
    """{layer: pruned weights} from the cached functional-grouping run --
    the SAME decision used for the GPTQ arm, so the arms differ only in the
    quantizer."""
    out = {}
    for f in sorted(glob.glob(f"{PRUNE_DIR}/prune_L*_tau0.99.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        z = np.load(f, allow_pickle=True)
        if len(z["keep"]) == FULL_FFN:      # nothing removed in this layer
            continue
        out[li] = {
            "keep": z["keep"], "w1": z["w1"], "w2": z["w2"], "bias": z["bias"],
            "bias_name": str(z["bias_name"]), "w1_init": str(z["w1_init"]),
            "w2_init": str(z["w2_init"]),
        }
    return out


def build(path_q, prune_map=None):
    """Export one arm: optionally prune, then per-channel INT8 round-to-nearest.

    quantize_dynamic with per_channel=True IS round-to-nearest at
    per-output-channel granularity, so no separate rounding step is needed.
    """
    tmp = path_q.replace(".onnx", "_tmp_fp32.onnx")
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}

    if prune_map:
        for d in prune_map.values():
            inits[d["w1_init"]].CopyFrom(
                numpy_helper.from_array(d["w1"].astype(np.float32), d["w1_init"]))
            inits[d["w2_init"]].CopyFrom(
                numpy_helper.from_array(d["w2"].astype(np.float32), d["w2_init"]))
            if d["bias_name"] != "None":
                inits[d["bias_name"]].CopyFrom(
                    numpy_helper.from_array(d["bias"].astype(np.float32),
                                            d["bias_name"]))

    onnx.save(model, tmp)
    quantize_dynamic(tmp, path_q, weight_type=QuantType.QInt8, per_channel=True)
    os.remove(tmp)
    return os.path.getsize(path_q) / (1024 * 1024)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prune_map = load_prune_map()
    print(f"{len(prune_map)} qatlam uchun qisqartirish qarori keshdan o'qildi "
          f"(GPTQ tarmog'i bilan aynan bir xil)\n")

    arms = [("C: RTN per-channel yolg'iz", f"{OUT_DIR}/enc_rtn_only.onnx", None),
            ("D: qisqartirish + RTN", f"{OUT_DIR}/enc_rtn_pruned.onnx", prune_map)]

    sizes = {}
    for label, path, pm in arms:
        if os.path.exists(path):
            sizes[label] = os.path.getsize(path) / (1024 * 1024)
            print(f"[{label}] mavjud, {sizes[label]:.0f} MiB")
            continue
        print(f"[{label}] quriladi...", flush=True)
        sizes[label] = build(path, pm)
        print(f"  saqlandi: {path}  {sizes[label]:.0f} MiB")

    json.dump(sizes, open(OUT_JSON, "w"), indent=2)
    print(f"\nModellar tayyor. Sifatni o'lchash uchun final_wer_testsplit.py "
          f"ishga tushiriladi\n(ENCODERS ro'yxatiga qo'shilgan, TEST splitining "
          f"300 namunasi).")


if __name__ == "__main__":
    main()
