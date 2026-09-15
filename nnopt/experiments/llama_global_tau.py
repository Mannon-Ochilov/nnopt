"""The per-layer redundancy profile of open_llama_3b, on the same tau grid.

Whisper's encoder and mBERT have now been measured on one global tau across
every layer, and they disagree in a way worth pinning down: Whisper carries
its redundancy in the EARLY layers (42-61% in L0-L5, ~0% from L12) while
mBERT carries it in the LATE ones (2-6% in L0-L3, 34-44% in L9-L11, and
DistilBERT's pre-registered result agreed). Llama is the third architecture
and the one whose refusal the cascade already argues for, so its profile
decides whether "where the redundancy sits" is a property of modality, of
depth, or of neither.

Llama needs no padding mask. Calibration is contiguous WikiText-2 segments of
a fixed length, so every position is real -- the defect found in the Whisper
and mBERT paths cannot occur here, and there is nothing to compare against.

The reducible axis is the gated intermediate,

    act = silu(gate_proj(x)) * up_proj(x)   ->  8640 channels
    out = down_proj(act)

so the criterion sees exactly the tensor entering down_proj, as in
llama_structural_refusal.py.

One forward pass, all layers. The existing helper hooks a single layer and
re-runs the model for each, which on a 3B float32 model costs minutes per
layer for no reason: the hooks are independent and can all be live at once.
The model is then released before grouping starts, since 12 GB of weights are
useless once the activations are captured.

Usage:  python experiments/llama_global_tau.py
"""

import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from nnopt.grouping.functional_grouping import greedy_group
from wikitext2_int4 import MODEL_DIR, load_segments

TAUS = (0.99, 0.95, 0.90, 0.85, 0.80, 0.75)
EPS_THRESHOLD = 0.5
CALIB_ROWS = 2048
N_SEGMENTS = 2
THREADS = 8
# The signed gate is the default and is what the encoders are measured with.
# Llama is the model the codebase runs with abs_cosine=True, because SwiGLU's
# gated product is two-signed and an anti-collinear pair is as mergeable as a
# collinear one (gamma is fitted signed). Both are reported: the signed run
# says how much ordinary collinearity exists, the absolute run says how much
# the relaxed gate adds, and the difference is exactly the anti-collinear mass
# the geometry argument predicts for a two-signed activation.
ABS_COSINE = os.environ.get("ABS_COSINE", "0") != "0"
OUT_JSON = ("experiments/results_llama_global_tau"
            + ("_abs" if ABS_COSINE else "") + ".json")


def capture_all_layers(model, segments, max_rows=CALIB_ROWS):
    """down_proj inputs for every layer, from a single forward per segment."""
    layers = model.model.layers
    store = {i: [] for i in range(len(layers))}
    handles = []

    def make_hook(i):
        def hook(mod, inputs, output):
            store[i].append(inputs[0].detach().to(torch.float32)
                            .reshape(-1, inputs[0].shape[-1]))
        return hook

    for i, lyr in enumerate(layers):
        handles.append(lyr.mlp.down_proj.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        for k, seg in enumerate(segments, 1):
            model(input_ids=seg)
            print(f"  segment {k}/{len(segments)}", flush=True)
    for h in handles:
        h.remove()

    rng = np.random.default_rng(0)
    out = {}
    for i, chunks in store.items():
        x = torch.cat(chunks, 0).numpy()
        if x.shape[0] > max_rows:
            x = x[rng.choice(x.shape[0], max_rows, replace=False)]
        out[i] = x.astype(np.float64)
        store[i] = None
    return out


def main():
    torch.set_num_threads(THREADS)
    print("model yuklanmoqda...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    n_layers = len(model.model.layers)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _, calib = load_segments(tok)
    segments = [calib[i:i + 1] for i in range(min(N_SEGMENTS, len(calib)))]
    print(f"{n_layers} qatlam, {len(segments)} segment", flush=True)

    acts = capture_all_layers(model, segments)
    # down_proj weights are all that is needed from here on; the rest of the
    # 12 GB checkpoint is dead weight while grouping runs.
    weights = {i: model.model.layers[i].mlp.down_proj.weight.detach()
               .to(torch.float64).numpy() for i in range(n_layers)}
    del model
    gc.collect()
    print("model bo'shatildi, guruhlash boshlanmoqda\n", flush=True)

    rows = []
    for li in range(n_layers):
        x = acts[li]
        w2 = weights[li]                       # (hidden, F)
        if w2.shape[1] != x.shape[1]:
            w2 = w2.T
        h = x.T
        wn = np.linalg.norm(w2, axis=0)
        y_norm = float(np.linalg.norm(x @ w2.T))
        line = f"  L{li:<2d}"
        for tau in TAUS:
            g = greedy_group(h, wn, y_norm, tau=tau,
                             eps_threshold=EPS_THRESHOLD,
                             abs_cosine=ABS_COSINE)
            merged = h.shape[0] - len(g.groups)
            rows.append({"layer": li, "tau": tau,
                         "channels": int(h.shape[0]), "merged": int(merged),
                         "share": merged / h.shape[0],
                         "rows": int(x.shape[0])})
            line += f"  t{tau}: {merged / h.shape[0] * 100:5.2f}%"
        print(line, flush=True)
        acts[li] = None
        weights[li] = None
        del x, w2, h
        gc.collect()
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
    report(rows, n_layers)


def report(rows, n_layers):
    print("\n" + "=" * 78)
    print("open_llama_3b — QATLAMLAR BO'YICHA OLIB TASHLASH, %")
    print("=" * 78)
    print(f"{'Qatlam':>7s}" + "".join(f"{'t=' + str(t):>11s}" for t in TAUS))
    for li in range(n_layers):
        line = f"{'L' + str(li):>7s}"
        for tau in TAUS:
            r = next((r for r in rows if r["layer"] == li
                      and r["tau"] == tau), None)
            line += f"{r['share'] * 100:10.2f}%" if r else " " * 11
        print(line)
    print("-" * 78)
    line = f"{'GLOBAL':>7s}"
    for tau in TAUS:
        sel = [r for r in rows if r["tau"] == tau]
        line += (f"{sum(r['merged'] for r in sel) / sum(r['channels'] for r in sel) * 100:10.2f}%"
                 if sel else " " * 11)
    print(line)

    # Where the redundancy sits is the question this run exists to answer, so
    # report it as a split rather than leaving it to be eyeballed.
    print("\n" + "=" * 78)
    print("CHUQURLIK TAQSIMOTI")
    print("=" * 78)
    third = max(1, n_layers // 3)
    for tau in TAUS:
        sel = [r for r in rows if r["tau"] == tau]
        if not sel or not sum(r["merged"] for r in sel):
            print(f"tau={tau}: birlashma yo'q")
            continue
        tot = sum(r["merged"] for r in sel)
        early = sum(r["merged"] for r in sel if r["layer"] < third)
        late = sum(r["merged"] for r in sel if r["layer"] >= n_layers - third)
        print(f"tau={tau}: birinchi uchdan bir {early / tot * 100:5.1f}%, "
              f"oxirgi uchdan bir {late / tot * 100:5.1f}%")
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
