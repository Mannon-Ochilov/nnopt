"""Why does our calibrated scale collapse at INT4 while winning at INT8?

The WikiText-2 run gave a result that contradicts every operator-level
measurement made so far:

    FP32 7.547 | INT8 RTN 7.550 | INT4 RTN 8.583 | INT4 AWQ 8.222
                                | INT4 GPTQ 8.646 | INT4 ours 12.799

At INT8 our calibrated scale beat min/max by 65-74% on local error. At INT4
it is nearly twice as bad as doing nothing clever at all. Before that goes
into the paper as a property of the method, it has to be separated from a
bug, so this script measures the mechanism directly on real Llama weights.

The hypothesis under test is CLIPPING. Both our phases minimize a squared
error against the weights: phase 1 solves s = <W,q>/<q,q>, which fits the
BULK of the weight distribution and therefore chooses a scale SMALLER than
min/max, deliberately trading a few clipped outliers for finer resolution on
the majority. With 255 levels that trade is nearly free. With 15 levels the
resolution gained is small while the clipped outliers are the entries that
carry the layer's output, so the same trade should invert.

If that is the mechanism, three things must hold at INT4 and not at INT8:
  (a) our scale is markedly below the min/max scale,
  (b) a non-trivial fraction of weights saturates at +-q_max,
  (c) our OUTPUT error exceeds RTN's even though our WEIGHT error is lower
      -- the signature of optimizing the wrong objective, not of a bug.
"""

import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnopt.quantizer.baselines import output_relative_error, rtn_quantize
from nnopt.quantizer.per_channel import (
    initial_scales_minmax,
    quantize_codes_pc,
    refine_scales_alternating_pc,
    refine_scales_per_channel,
    quantize_weight_per_channel,
)

MODEL_DIR = os.environ.get("LLAMA_MODEL", "models/open_llama_3b")
WIKI_CACHE = "models/_calib_cache/wikitext2_test.npz"
SEQ_LEN = 2048
LAYERS = (0, 12, 25)
FFN = ("gate_proj", "up_proj", "down_proj")
BITS = {"INT4": 7, "INT8": 127}


def calib_rows(model, tok, layer_idx):
    z = np.load(WIKI_CACHE, allow_pickle=True)
    ids = tok(str(z["calib"][0]), return_tensors="pt").input_ids[0][:2 * SEQ_LEN]
    ids = ids.view(2, SEQ_LEN)
    store = {n: [] for n in FFN}
    handles = []

    def mk(n):
        def hook(mod, inputs, output):
            store[n].append(inputs[0].detach().to(torch.float32)
                            .reshape(-1, inputs[0].shape[-1]))
        return hook

    mlp = model.model.layers[layer_idx].mlp
    for n in FFN:
        handles.append(getattr(mlp, n).register_forward_hook(mk(n)))
    with torch.no_grad():
        for i in range(len(ids)):
            model(input_ids=ids[i : i + 1])
    for h in handles:
        h.remove()
    return {k: torch.cat(v, 0).numpy().astype(np.float64) for k, v in store.items()}


def main():
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()

    print(f"{'qatlam/operator':22s} {'bit':5s} {'s_biz/s_minmax':>14s} "
          f"{'kesilgan %':>11s} {'W xato biz':>11s} {'W xato RTN':>11s} "
          f"{'Y xato biz':>11s} {'Y xato RTN':>11s}")
    print("-" * 100)

    for li in LAYERS:
        x_by = calib_rows(model, tok, li)
        mlp = model.model.layers[li].mlp
        for nm in FFN:
            w = getattr(mlp, nm).weight.detach().numpy().astype(np.float64)
            x = x_by[nm]
            for bit_name, q_max in BITS.items():
                s_mm = initial_scales_minmax(w, q_max)
                res = refine_scales_per_channel(w, q_max, x_calib=x)
                s_ours = res.scales

                # Saturation fraction under OUR scale: entries the grid cannot
                # represent because they were clipped away.
                codes = np.round(w / s_ours)
                clipped = float(np.mean(np.abs(codes) > q_max)) * 100.0

                w_ours = quantize_codes_pc(w, s_ours, q_max) * s_ours
                w_rtn = rtn_quantize(w, q_max)

                wn = np.linalg.norm(w)
                print(f"L{li:<2d} {nm:<18s} {bit_name:5s} "
                      f"{float(np.mean(s_ours / s_mm)):14.4f} "
                      f"{clipped:10.3f}% "
                      f"{np.linalg.norm(w - w_ours)/wn:11.5f} "
                      f"{np.linalg.norm(w - w_rtn)/wn:11.5f} "
                      f"{output_relative_error(w, w_ours, x):11.5f} "
                      f"{output_relative_error(w, w_rtn, x):11.5f}", flush=True)
            del w, x
        del x_by
    print("-" * 100)
    print("s_biz/s_minmax < 1 -> masshtab kichraytirilgan (kesish evaziga aniqlik)")
    print("W xato biz < W xato RTN, lekin Y xato biz > Y xato RTN bo'lsa -> "
          "maqsad funksiyasi noto'g'ri, xato emas")


if __name__ == "__main__":
    main()
