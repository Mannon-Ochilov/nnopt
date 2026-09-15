"""Two open questions about quantization, answered by one measurement.

The first came out of a refuted recommendation. GPTQ's output gain is 1.0000 --
it carries no multiplicative attenuation for depth to compound -- so the fix
that rescued our own INT4 scale has nothing to repair there. But gain is only
one of the two ways an output can be systematically wrong. The other is an
ADDITIVE shift: quantization error need not average to zero, and GPTQ's
objective, a weighted sum of squared errors, never asks it to. Any constant
component of (W - W_hat)X can be moved into the bias vector the operator
already has, at no cost in bits, codes or memory layout -- the same identity
the structural stage uses when it folds a discarded channel's mean into the
bias.

The second question is whether GPTQ's advantage is real or in-sample. It fits
its Hessian on the calibration rows and its reported error is measured on
those same rows, which is a fit statistic, not a generalisation one. On this
model the Hessian comes from two segments for a three-billion-parameter
network, and the low-rank work in this project already shows what a thin fit
set does: fit error 0.00000, held-out error 0.04355.

So: means are estimated on calibration, every error is reported on held-out
activations, and the corrected arm adds the estimated offset. If the offset
is negligible the correction is pointless and that is the answer; if GPTQ's
held-out error rises to meet the others, its INT4 result stops being a
paradox.
"""

import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from int4_bias_and_fix import FFN, LAYERS, MODEL_DIR, Q_MAX, SEQ_LEN, WIKI_CACHE
from nnopt.quantizer.baselines import gptq_quantize, rtn_quantize
from nnopt.quantizer.per_channel import quantize_codes_pc, refine_scales_per_channel

# Calibration size is the variable the GPTQ result now hangs on. Its Hessian
# is n x n in the operator's INPUT width, and with two segments (4096 rows)
# open_llama_3b's down_proj has n = 8640 -- fewer rows than the matrix it has
# to estimate, so the Hessian is rank-deficient and damping carries it. That
# operator also shows the largest fit-to-held-out jump (3.6x against 2.2x for
# the others), which is what makes rows-per-n the thing to vary.
N_FIT_SEG = int(os.environ.get("N_FIT_SEG", "2"))
N_HO_SEG = int(os.environ.get("N_HO_SEG", "2"))
OUT_JSON = os.environ.get(
    "OUT_JSON",
    f"experiments/results_quant_bias_int{4 if Q_MAX == 7 else 8}"
    f"{'' if N_FIT_SEG == 2 else f'_fit{N_FIT_SEG}'}.json")


def rows_for(model, tok, layer_idx, seg_from, seg_to):
    """Activations into each FFN operator over a chosen span of segments."""
    z = np.load(WIKI_CACHE, allow_pickle=True)
    ids = tok(str(z["calib"][0]), return_tensors="pt").input_ids[0]
    need = seg_to * SEQ_LEN
    if len(ids) < need:                       # fall back to the test text
        ids = tok(str(z["text"][0]), return_tensors="pt").input_ids[0]
    ids = ids[:need].view(seg_to, SEQ_LEN)[seg_from:seg_to]

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
            model(input_ids=ids[i:i + 1])
    for h in handles:
        h.remove()
    return {k: torch.cat(v, 0).numpy().astype(np.float64)
            for k, v in store.items()}


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def main():
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    bits = 4 if Q_MAX == 7 else 8
    print(f"INT{bits}. Xato HELD-OUT da; o'rtacha kalibrlashda baholanadi.")
    print(f"Kalibrlash {N_FIT_SEG} segment = {N_FIT_SEG * SEQ_LEN} satr; "
          f"held-out {N_HO_SEG} segment.\n")
    print(f"{'operator':22s} {'usul':>6s} {'siljish':>9s} {'xato(fit)':>10s} "
          f"{'xato(h/o)':>10s} {'xato(h/o+bias)':>15s}")
    print("-" * 78)

    rows = []
    for li in LAYERS:
        x_fit_by = rows_for(model, tok, li, 0, N_FIT_SEG)
        x_ho_by = rows_for(model, tok, li, N_FIT_SEG, N_FIT_SEG + N_HO_SEG)
        mlp = model.model.layers[li].mlp
        for nm in FFN:
            w = getattr(mlp, nm).weight.detach().numpy().astype(np.float64)
            x_fit, x_ho = x_fit_by[nm], x_ho_by[nm]
            mu = x_fit.mean(axis=0)
            y_fit, y_ho = x_fit @ w.T, x_ho @ w.T

            res = refine_scales_per_channel(w, Q_MAX, x_calib=x_fit)
            variants = {
                "RTN": rtn_quantize(w, Q_MAX),
                "biz": quantize_codes_pc(w, res.scales, Q_MAX) * res.scales,
                "GPTQ": gptq_quantize(w, x_fit, Q_MAX),
            }
            for name, wq in variants.items():
                offset = (w - wq) @ mu
                rel_off = float(np.linalg.norm(offset)
                                / (np.linalg.norm(w @ mu) + 1e-12))
                e_fit = rel(y_fit, x_fit @ wq.T)
                e_ho = rel(y_ho, x_ho @ wq.T)
                e_ho_b = rel(y_ho, x_ho @ wq.T + offset)
                rows.append({"layer": li, "op": nm, "method": name,
                             "rel_offset": rel_off, "e_fit": e_fit,
                             "e_ho": e_ho, "e_ho_bias": e_ho_b})
                print(f"L{li:<2d} {nm:<18s} {name:>6s} {rel_off:9.4f} "
                      f"{e_fit:10.5f} {e_ho:10.5f} {e_ho_b:15.5f}", flush=True)
            del w, x_fit, x_ho, variants
            gc.collect()
        del x_fit_by, x_ho_by
        gc.collect()
        json.dump(rows, open(OUT_JSON, "w"), indent=2)

    print("-" * 78)
    for name in ("RTN", "biz", "GPTQ"):
        sel = [r for r in rows if r["method"] == name]
        off = np.mean([r["rel_offset"] for r in sel])
        ef = np.mean([r["e_fit"] for r in sel])
        eh = np.mean([r["e_ho"] for r in sel])
        eb = np.mean([r["e_ho_bias"] for r in sel])
        print(f"{name:>6s}  siljish {off:.4f}   xato fit {ef:.5f} -> "
              f"held-out {eh:.5f}   bias bilan {eb:.5f} "
              f"({(eh-eb)/eh*100:+.1f}%)")
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
