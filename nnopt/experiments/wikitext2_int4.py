"""WikiText-2 comparison at the bit width where the methods actually differ.

Two findings from the INT8 run reshaped this experiment.

First, the FP32 baseline came out at 7.547, matching published open_llama_3b
values (~7.6-7.9). The protocol is therefore sound and the numbers here can
be read against GPTQ/AWQ/SVD-LLM papers.

Second, INT8 leaves so much headroom that every method ties: plain
round-to-nearest already scored 7.550, i.e. 1.000x of FP32. That is why the
published works report INT4/INT3 rather than INT8 -- at 8 bits there is
nothing to separate. It also explains why our operator-level gap (GPTQ 54%
lower E_loc) vanished end-to-end: the network absorbs a difference that
small, exactly as the error-absorption measurement predicts.

So the comparison is run at INT4. INT8 is kept only as the single already
measured round-to-nearest reference: once the CHEAPEST method reproduces
FP32 to three digits, no other method can separate from it at that width,
so re-running the remaining three would cost hours to confirm a tie that is
already established.

Efficiency note: the previous version re-ran the whole model once per layer
to gather calibration activations, costing 26x more than necessary. Here a
single pass per layer group feeds ALL methods, so calibration is captured
once rather than once per method.

The first INT4 pass then produced a result that reshaped the experiment
again: our calibrated scale came last by a wide margin (12.799 vs RTN's
8.583) despite being 26% MORE accurate than RTN on every individual
operator. The cause is bias, not error size -- fitting the scale to the
weights shrinks each channel by ~1%, always in the same direction, so the
attenuation compounds over the 78 feed-forward operators (0.9896^78 = 0.44)
instead of cancelling. Two output-domain rescales that keep the integer
codes untouched are therefore evaluated alongside it:

    ours_ls        least-squares optimal scale for the output
    ours_unbiased  gain-matched scale, per-channel gain exactly 1

See README Sec 8.3.17 and experiments/int4_bias_and_fix.py.

Calibration comes from the train split; evaluation from test. GPTQ and AWQ
are our reimplementations (official packages require CUDA).
"""

import gc
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnopt.quantizer.baselines import awq_quantize, gptq_quantize, rtn_quantize
from nnopt.quantizer.per_channel import (
    gram_factor,
    quantize_codes_pc,
    refine_scales_per_channel,
    rescale_output_domain,
)

MODEL_DIR = os.environ.get("LLAMA_MODEL", "models/open_llama_3b")
WIKI_CACHE = "models/_calib_cache/wikitext2_test.npz"
WEIGHT_CACHE = "models/_wiki_qweights"
SEQ_LEN = 2048
N_EVAL_SEGMENTS = 24
N_CALIB_SEGMENTS = 2          # 4096 rows; ample for per-channel scales and Hessians
LAYERS_PER_GROUP = 2
THREADS = 8
FFN = ("gate_proj", "up_proj", "down_proj")
ALL_METHODS = ("rtn", "awq", "gptq", "ours", "ours_ls", "ours_unbiased")
# The three "ours*" variants share the whole scale search and differ only in
# the final per-channel multiplier, so they are produced together.
OURS_FAMILY = ("ours", "ours_ls", "ours_unbiased")
# Both are overridable so a single missing cell can be filled without
# rebuilding the whole table -- e.g. METHODS=ours BITS=int8 to close the
# cascade's actual operating point.
METHODS = tuple(os.environ.get("METHODS", ",".join(ALL_METHODS)).split(","))
BITS = {"int4": 7, "int8": 127}
# Every method is measured at both widths.
#
# The tempting shortcut is to skip the INT8 baselines because round-to-nearest
# already reproduces FP32 to three digits there. That argument is wrong, and
# this experiment is where it was caught: RTN's result bounds the other
# methods from ABOVE and says nothing about a method that deviates DOWNWARD.
# Two measurements here did deviate downward -- our weight-domain scale at
# INT4 (1.696x) and GPTQ at INT4, which landed BELOW plain rounding
# (8.646 vs 8.583). Having seen both, assuming a tie for any method at any
# width is not defensible, so nothing is assumed.
METHODS_BY_BITS = {"int4": METHODS, "int8": METHODS}
OUT_JSON = "experiments/results_wikitext2_int4.json"


def load_segments(tok):
    z = np.load(WIKI_CACHE, allow_pickle=True)
    test_ids = tok(str(z["text"][0]), return_tensors="pt").input_ids[0]
    calib_ids = tok(str(z["calib"][0]), return_tensors="pt").input_ids[0]
    n_test = min(N_EVAL_SEGMENTS, len(test_ids) // SEQ_LEN)
    n_cal = min(N_CALIB_SEGMENTS, len(calib_ids) // SEQ_LEN)
    return (test_ids[: n_test * SEQ_LEN].view(n_test, SEQ_LEN),
            calib_ids[: n_cal * SEQ_LEN].view(n_cal, SEQ_LEN))


def perplexity(model, segments):
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(len(segments)):
            ids = segments[i : i + 1]
            logits = model(input_ids=ids).logits[:, :-1]
            tgt = ids[:, 1:]
            total_nll += float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum"))
            total_tok += int(tgt.numel())
    return float(np.exp(total_nll / max(total_tok, 1)))


def capture_group(model, calib, layer_ids):
    store = {(li, nm): [] for li in layer_ids for nm in FFN}
    handles = []

    def mk(li, nm):
        def hook(mod, inputs, output):
            store[(li, nm)].append(inputs[0].detach().to(torch.float32)
                                   .reshape(-1, inputs[0].shape[-1]))
        return hook

    for li in layer_ids:
        mlp = model.model.layers[li].mlp
        for nm in FFN:
            handles.append(getattr(mlp, nm).register_forward_hook(mk(li, nm)))
    with torch.no_grad():
        for i in range(len(calib)):
            model(input_ids=calib[i : i + 1])
    for h in handles:
        h.remove()
    return {k: torch.cat(v, 0).numpy().astype(np.float64) for k, v in store.items()}


def quantize_missing(methods, w, x, q_max):
    """Quantized weights for exactly the requested methods.

    Only the missing ones are computed, so adding a variant does not force a
    re-run of the baselines already cached -- GPTQ alone costs ~20 s per wide
    operator.
    """
    out = {}
    if "rtn" in methods:
        out["rtn"] = rtn_quantize(w, q_max)
    if "gptq" in methods:
        out["gptq"] = gptq_quantize(w, x, q_max)
    if "awq" in methods:
        out["awq"] = awq_quantize(w, x, q_max)[0]

    wanted = [m for m in OURS_FAMILY if m in methods]
    if wanted:
        # One scale search feeds all three: same integer codes throughout,
        # only the final multiplier differs (README Sec 8.3.17).
        res = refine_scales_per_channel(w, q_max, x_calib=x)
        codes = quantize_codes_pc(w, res.scales, q_max)
        if "ours" in wanted:
            out["ours"] = codes * res.scales
        if len(wanted) > (1 if "ours" in wanted else 0):
            f = gram_factor(x)
            if "ours_ls" in wanted:
                out["ours_ls"] = codes * rescale_output_domain(w, codes, f, "ls")
            if "ours_unbiased" in wanted:
                out["ours_unbiased"] = codes * rescale_output_domain(
                    w, codes, f, "unbiased")
    return out


def build_all_weights(bits_name, q_max):
    """One calibration pass produces quantized weights for every method."""
    os.makedirs(WEIGHT_CACHE, exist_ok=True)
    wanted_methods = METHODS_BY_BITS[bits_name]
    done = all(os.path.exists(f"{WEIGHT_CACHE}/{bits_name}_{m}_L{li}.npz")
               for m in wanted_methods for li in range(26))
    if done:
        print(f"  [{bits_name}] vaznlar keshdan olinadi")
        return

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _, calib = load_segments(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    t0 = time.time()

    def missing(li):
        return [m for m in wanted_methods
                if not os.path.exists(f"{WEIGHT_CACHE}/{bits_name}_{m}_L{li}.npz")]

    for start in range(0, n_layers, LAYERS_PER_GROUP):
        group = list(range(start, min(start + LAYERS_PER_GROUP, n_layers)))
        pending = [li for li in group if missing(li)]
        if not pending:
            continue
        x_by = capture_group(model, calib, pending)
        for li in pending:
            need = missing(li)
            mlp = model.model.layers[li].mlp
            payload = {m: {} for m in need}
            for nm in FFN:
                w = getattr(mlp, nm).weight.detach().numpy().astype(np.float64)
                x = x_by[(li, nm)]
                for m, wq in quantize_missing(need, w, x, q_max).items():
                    payload[m][nm] = wq.astype(np.float32)
                del w, x
            for m in need:
                np.savez_compressed(f"{WEIGHT_CACHE}/{bits_name}_{m}_L{li}.npz",
                                    **payload[m])
            del payload
        del x_by
        gc.collect()
        print(f"    {group[-1]+1}/{n_layers} qatlam [{time.time()-t0:.0f}s]", flush=True)

    del model
    gc.collect()


def apply_weights(model, bits_name, method):
    for li in range(model.config.num_hidden_layers):
        f = f"{WEIGHT_CACHE}/{bits_name}_{method}_L{li}.npz"
        z = np.load(f)
        mlp = model.model.layers[li].mlp
        for nm in FFN:
            with torch.no_grad():
                getattr(mlp, nm).weight.copy_(torch.from_numpy(z[nm]).float())


def load_results():
    """Completed perplexities persist across runs; each evaluation costs ~17
    minutes and sessions here have been interrupted repeatedly."""
    if not os.path.exists(OUT_JSON):
        return {}
    try:
        # utf-8-sig tolerates a BOM, which PowerShell's Out-File adds by
        # default and json.load otherwise rejects.
        with open(OUT_JSON, encoding="utf-8-sig") as f:
            return json.load(f).get("ppl", {})
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ogohlantirish: natija keshi o'qilmadi ({type(exc).__name__}), "
              f"noldan boshlanadi")
        return {}


def save_results(results, n_test, n_calib):
    json.dump({"segments": n_test, "seq_len": SEQ_LEN,
               "calib_segments": n_calib, "ppl": results},
              open(OUT_JSON, "w"), indent=2)


def main():
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    test, calib = load_segments(tok)
    print(f"WikiText-2: {len(test)} segment x {SEQ_LEN} = {len(test)*SEQ_LEN:,} token")
    print(f"kalibrlash: {len(calib)} segment x {SEQ_LEN} (train dan)\n")

    results = load_results()
    if results:
        print(f"keshdan {len(results)} natija o'qildi: {', '.join(results)}\n")

    if "FP32" not in results:
        print("[FP32] baza")
        m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
        m.eval()
        results["FP32"] = perplexity(m, test)
        save_results(results, len(test), len(calib))
        print(f"  PPL = {results['FP32']:.3f}")
        del m
        gc.collect()

    for bits_name, q_max in BITS.items():
        print(f"\n=== {bits_name.upper()} ===")
        bit_methods = METHODS_BY_BITS[bits_name]
        if any(f"{bits_name} {m}" not in results for m in bit_methods):
            build_all_weights(bits_name, q_max)
        for method in bit_methods:
            key = f"{bits_name} {method}"
            if key in results:
                print(f"  {method:6s} PPL = {results[key]:8.3f}  (keshdan)")
                continue
            m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                     low_cpu_mem_usage=True)
            m.eval()
            apply_weights(m, bits_name, method)
            ppl = perplexity(m, test)
            results[key] = ppl
            save_results(results, len(test), len(calib))
            print(f"  {method:6s} PPL = {ppl:8.3f}  "
                  f"({ppl/results['FP32']:.3f}x)", flush=True)
            del m
            gc.collect()

    save_results(results, len(test), len(calib))

    base = results["FP32"]
    print("\n" + "=" * 80)
    print(f"WIKITEXT-2 PERPLEXITY — open_llama_3b, FFN operatorlari, "
          f"{len(test)}x{SEQ_LEN} token")
    print("=" * 80)
    print(f"{'Usul':30s} {'PPL':>10s} {'FP32 ga':>10s}")
    print("-" * 80)
    print(f"{'FP32':30s} {base:10.3f} {1.0:9.3f}x")
    for bits_name in BITS:
        for method in METHODS_BY_BITS[bits_name]:
            k = f"{bits_name} {method}"
            if k in results:
                print(f"{k:30s} {results[k]:10.3f} {results[k]/base:9.3f}x")
    print("=" * 80)
    print("Barcha usullar ikkala bit kengligida ham o'lchandi — hech biri "
          "taxmin qilinmadi.")
    print("\nGPTQ va AWQ — qayta amalga oshirilgan (rasmiy paketlar CUDA talab qiladi).")


if __name__ == "__main__":
    main()
