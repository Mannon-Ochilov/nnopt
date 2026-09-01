"""WikiText-2 perplexity: the standard benchmark, so results are comparable
to the literature.

Everything reported for the language model so far used Uzbek transcripts,
which keeps the domain consistent with the ASR work but leaves the numbers
uncheckable against published results. GPTQ, AWQ, SVD-LLM and SliceGPT all
report WikiText-2 perplexity, so evaluating here places our measurements on
the same axis as theirs -- including the FP32 baseline, which acts as a
sanity check on the whole pipeline (open_llama_3b should land near 7-8).

Protocol follows those papers: the test split is joined, tokenized once, and
scored on non-overlapping segments of 2048 tokens. Segment length drives
comparability so it is kept at 2048; the number of segments is reduced
because a single forward pass costs ~42 s on this CPU, and the count actually
used is reported alongside the numbers.

Calibration segments come from the TRAIN split, never from test.

Quantization is applied to the feed-forward operators, which is where the
cascade's decisions apply and where the methods are being compared. GPTQ is
our reimplementation (official package requires CUDA).
"""

import gc
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnopt.quantizer.baselines import awq_quantize, gptq_quantize, rtn_quantize
from nnopt.quantizer.per_channel import quantize_weight_per_channel

MODEL_DIR = os.environ.get("LLAMA_MODEL", "models/open_llama_3b")
WIKI_CACHE = "models/_calib_cache/wikitext2_test.npz"
SEQ_LEN = 2048
N_EVAL_SEGMENTS = 24          # 24 x 2048 = 49152 tokens
N_CALIB_SEGMENTS = 8          # 8 x 2048 = 16384 calibration rows per operator
THREADS = 8
Q8 = 127
FFN = ("gate_proj", "up_proj", "down_proj")
OUT_JSON = "experiments/results_wikitext2.json"


def load_segments(tok):
    z = np.load(WIKI_CACHE, allow_pickle=True)
    test_ids = tok(str(z["text"][0]), return_tensors="pt").input_ids[0]
    calib_ids = tok(str(z["calib"][0]), return_tensors="pt").input_ids[0]

    n_test = min(N_EVAL_SEGMENTS, len(test_ids) // SEQ_LEN)
    test = test_ids[: n_test * SEQ_LEN].view(n_test, SEQ_LEN)
    n_cal = min(N_CALIB_SEGMENTS, len(calib_ids) // SEQ_LEN)
    calib = calib_ids[: n_cal * SEQ_LEN].view(n_cal, SEQ_LEN)
    return test, calib


def perplexity(model, segments):
    """Standard sliding-free perplexity over non-overlapping segments."""
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(len(segments)):
            ids = segments[i : i + 1]
            logits = model(input_ids=ids).logits[:, :-1]
            tgt = ids[:, 1:]
            total_nll += float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum"))
            total_tok += int(tgt.numel())
            if (i + 1) % 8 == 0:
                print(f"      {i+1}/{len(segments)} segment", flush=True)
    return float(np.exp(total_nll / max(total_tok, 1)))


def capture_ffn_inputs(model, calib_segments, layer_idx):
    """Inputs of one layer's FFN projections, collected on calibration text."""
    store = {name: [] for name in FFN}
    handles = []

    def mk(name):
        def hook(mod, inputs, output):
            store[name].append(inputs[0].detach().to(torch.float32)
                               .reshape(-1, inputs[0].shape[-1]))
        return hook

    mlp = model.model.layers[layer_idx].mlp
    for name in FFN:
        handles.append(getattr(mlp, name).register_forward_hook(mk(name)))
    with torch.no_grad():
        for i in range(len(calib_segments)):
            model(input_ids=calib_segments[i : i + 1])
    for h in handles:
        h.remove()
    return {k: torch.cat(v, 0).numpy().astype(np.float64) for k, v in store.items()}


def quantize_model(model, method, calib_segments):
    """Apply `method` to every FFN projection, layer by layer."""
    n_layers = model.config.num_hidden_layers
    t0 = time.time()
    for li in range(n_layers):
        x_by = capture_ffn_inputs(model, calib_segments, li) if method != "rtn" else {}
        mlp = model.model.layers[li].mlp
        for name in FFN:
            mod = getattr(mlp, name)
            w = mod.weight.detach().numpy().astype(np.float64)
            if method == "rtn":
                wq = rtn_quantize(w, Q8)
            elif method == "gptq":
                wq = gptq_quantize(w, x_by[name], Q8)
            elif method == "awq":
                wq, _ = awq_quantize(w, x_by[name], Q8)
            elif method == "ours":
                wq, _ = quantize_weight_per_channel(w, Q8, x_calib=x_by[name])
            else:
                raise ValueError(method)
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(wq).float())
            del w, wq
        del x_by
        gc.collect()
        if (li + 1) % 6 == 0:
            print(f"    {li+1}/{n_layers} qatlam [{time.time()-t0:.0f}s]", flush=True)


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                             low_cpu_mem_usage=True)
    m.eval()
    return m


def main():
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    test, calib = load_segments(tok)
    print(f"WikiText-2: {len(test)} baholash segmenti x {SEQ_LEN} token "
          f"= {len(test)*SEQ_LEN:,} token")
    print(f"kalibrlash: {len(calib)} segment (train dan)\n")

    rows = {}

    print("[FP32] baza")
    m = fresh_model()
    ppl = perplexity(m, test)
    rows["FP32"] = ppl
    print(f"  PPL = {ppl:.3f}")
    del m
    gc.collect()

    for method, label in (("rtn", "RTN INT8"),
                          ("awq", "AWQ INT8"),
                          ("gptq", "GPTQ INT8"),
                          ("ours", "bizning kalibrlangan INT8")):
        print(f"\n[{label}] FFN operatorlari kvantlanmoqda...")
        m = fresh_model()
        quantize_model(m, method, calib)
        ppl = perplexity(m, test)
        rows[label] = ppl
        print(f"  PPL = {ppl:.3f}  (FP32 ga nisbatan {ppl/rows['FP32']:.3f}x)")
        del m
        gc.collect()

    json.dump({"segments": len(test), "seq_len": SEQ_LEN, "ppl": rows},
              open(OUT_JSON, "w"), indent=2)

    base = rows["FP32"]
    print("\n" + "=" * 76)
    print(f"WIKITEXT-2 PERPLEXITY (open_llama_3b, FFN da INT8, "
          f"{len(test)}x{SEQ_LEN} token)")
    print("=" * 76)
    print(f"{'Usul':32s} {'PPL':>10s} {'FP32 ga':>10s}")
    print("-" * 76)
    for k, v in rows.items():
        print(f"{k:32s} {v:10.3f} {v/base:9.3f}x")
    print("=" * 76)
    print("\nEslatma: GPTQ va AWQ — qayta amalga oshirilgan (rasmiy paketlar CUDA "
          "talab qiladi).")


if __name__ == "__main__":
    main()
