"""Can OUR quantizer, with all of our own corrections, beat AWQ at INT4?

Standing admission in the paper: GPTQ beats our calibrated scale at the
operator level, and at INT4 our weight-domain scale loses badly end to end
until the output-domain rescale repairs it to 8.246 -- close to AWQ's 8.222
but not ahead. That leaves our quantization contribution phrased as "we lose,
so the cascade delegates to GPTQ".

One of our own components has never been added to that stack end to end:
additive bias correction, worth ~6% of operator error at INT4 for our scale.
This measures the FULL own-method stack

    calibrated per-channel scale  ->  output-domain (LS) rescale
                                  ->  b += (W - What) mean(X)

on WikiText-2 against the cached AWQ/GPTQ/RTN INT4 rows. Llama's FFN has no
bias, so one is ADDED per corrected matrix -- d_out floats per operator,
priced against the model as in Sec 4.14 (the structural stage already pays
this cost when it needs to).

Pre-registered readings:
  ppl <= 8.22  -> our stack is competitive-or-better than AWQ, and the
                  paper's quantization claim strengthens accordingly
  ppl in (8.22, 8.25) -> no change to the claim; correction is negligible
  ppl > 8.25   -> bias correction HURTS at network level (would echo the
                  mBERT accuracy/PPL divergence) and is reported as such
"""

import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wikitext2_int4 import (
    MODEL_DIR,
    WEIGHT_CACHE,
    FFN,
    LAYERS_PER_GROUP,
    capture_group,
    load_segments,
    perplexity,
)

OUT_JSON = "experiments/results_int4_full_stack.json"
METHOD = "ours_ls"          # cached INT4 weights of scale + LS rescale
THREADS = 8


def main():
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    test, calib = load_segments(tok)

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    n_layers = model.config.num_hidden_layers

    # Bias corrections need the ORIGINAL weights and the quantized ones
    # together, plus calibration activations captured from the FP32 model --
    # the same protocol the operator-level measurement used.
    print("bias tuzatishlar hisoblanmoqda (FP32 faolliklarda)...", flush=True)
    corrections = {}
    for start in range(0, n_layers, LAYERS_PER_GROUP):
        group = list(range(start, min(start + LAYERS_PER_GROUP, n_layers)))
        x_by = capture_group(model, calib, group)
        for li in group:
            z = np.load(f"{WEIGHT_CACHE}/int4_{METHOD}_L{li}.npz")
            mlp = model.model.layers[li].mlp
            for nm in FFN:
                w = getattr(mlp, nm).weight.detach().numpy().astype(np.float64)
                what = z[nm].astype(np.float64)
                mu = x_by[(li, nm)].mean(axis=0)
                corrections[(li, nm)] = ((w - what) @ mu).astype(np.float32)
        del x_by
        gc.collect()
        print(f"  {group[-1]+1}/{n_layers} qatlam", flush=True)

    print("kvantlangan vaznlar + bias qo'llanmoqda...", flush=True)
    for li in range(n_layers):
        z = np.load(f"{WEIGHT_CACHE}/int4_{METHOD}_L{li}.npz")
        mlp = model.model.layers[li].mlp
        for nm in FFN:
            lin = getattr(mlp, nm)
            with torch.no_grad():
                lin.weight.copy_(torch.from_numpy(z[nm]).float())
                if lin.bias is None:
                    lin.bias = torch.nn.Parameter(
                        torch.zeros(lin.weight.shape[0]))
                lin.bias += torch.from_numpy(corrections[(li, nm)])

    print("perplexity hisoblanmoqda...", flush=True)
    ppl = perplexity(model, test)
    ref = {"AWQ": 8.222, "RTN": 8.583, "GPTQ": 8.646, "ours_ls (biassiz)": 8.246}
    print("\n" + "=" * 60)
    print(f"ours_ls + bias tuzatish  INT4  PPL = {ppl:.3f}")
    for k, v in ref.items():
        print(f"  {k:22s} {v:.3f}   farq {ppl - v:+.3f}")
    print("=" * 60)
    json.dump({"ppl": float(ppl), "method": METHOD + "+biascorr",
               "reference": ref}, open(OUT_JSON, "w"), indent=2)
    print(f"saqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
