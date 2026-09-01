"""Is the cascade right to refuse channel removal on open_llama_3b?

For mBERT that refusal was measured: forcing 20% of the channels out costs
0.028 of masked-token accuracy against a quantization step that costs nothing
(Sec 4.11). For open_llama_3b it has only ever been ARGUED, from the
redundancy diagnostic -- 0.6% of channels at tau = 0.99 against Whisper's
17.1%. Argued is weaker than measured, and this model is the interesting case
of the three: its redundancy is six times mBERT's, so the answer is not
obviously the same.

The structure differs from the encoders too. A Llama feed-forward block has
TWO expanding projections into one gated activation,

    act = silu(gate_proj(x)) * up_proj(x)     ->  8640 channels
    out = down_proj(act)

so a single channel decision removes a row from each of gate_proj and
up_proj and a column from down_proj. The reducible axis is the same object
the criterion works on; only the number of matrices it touches changes.

Two criteria are compared at the same 20% budget, as on mBERT: our cosine
grouping forced down to the budget, and the fluctuation score with its mean
folded into the bias. Perplexity is WikiText-2, where this paper's other
Llama quantization results already sit (FP32 7.547, INT8 7.550), so the
structural cost can be read against the quantization cost directly.
"""

import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnopt.grouping.functional_grouping import (
    build_compensated_weight,
    greedy_group,
    trim_to_budget,
)
from wikitext2_int4 import (
    MODEL_DIR,
    SEQ_LEN,
    load_segments,
    perplexity,
)

OUT_JSON = "experiments/results_llama_structural.json"
REMOVAL = float(os.environ.get("REMOVAL", "0.20"))
CRITERION = os.environ.get("CRITERION", "cosine")
CALIB_ROWS = int(os.environ.get("CALIB_ROWS", "2048"))
EPS_THRESHOLD = 0.5
TAU_GRID = (0.99, 0.9, 0.7, 0.5, 0.3, 0.0, -1.0)
THREADS = 8


def capture_down_input(model, calib_segments, layer_idx, max_rows=CALIB_ROWS):
    """Activations entering down_proj: the gated 8640-wide intermediate."""
    store = []

    def hook(mod, inputs, output):
        store.append(inputs[0].detach().to(torch.float32)
                     .reshape(-1, inputs[0].shape[-1]))

    h = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(hook)
    with torch.no_grad():
        for seg in calib_segments:
            model(input_ids=seg)
    h.remove()
    x = torch.cat(store, 0).numpy().astype(np.float64)
    if x.shape[0] > max_rows:
        idx = np.random.default_rng(0).choice(x.shape[0], max_rows,
                                              replace=False)
        x = x[idx]
    return x


def keep_at_tau(x, w_down, tau):
    """What the criterion itself endorses removing at this tau, with no budget.

    This is the operating point the method is actually about, and it differs
    from `keep_set` in the thing that matters: the number of channels removed
    is an OUTPUT, not an input. On this model the difference is large and
    per-layer -- at tau = 0.90 the first block yields 25% and the twenty-first
    yields 0.3% -- so a ladder of uniform ratios asks the twenty-first block
    to give up thirty times what the criterion judges redundant there.
    """
    eps = EPS_THRESHOLD if tau >= 0.0 else float("inf")
    # Signed vs absolute matters HERE and not on Whisper, which is why it is
    # set per model rather than globally. This architecture's reducible axis
    # carries a gated product whose sign varies freely (50.1% positive), so
    # anti-collinear pairs exist and the signed gate throws them away: the
    # largest |cos| is 0.8488 against 0.7681 signed, and at tau = 0.70 the
    # absolute form reaches 6.5x as many channels. Whisper's axis reads a
    # GELU output that is ~97% ONE-signed, so there the two agree exactly at
    # the operating point.
    g = greedy_group(x.T, np.linalg.norm(w_down, axis=0),
                     float(np.linalg.norm(x @ w_down.T)), tau=tau,
                     eps_threshold=eps, abs_cosine=True)
    keep = np.array(sorted(gr.representative for gr in g.groups))
    return keep, build_compensated_weight(w_down, g), None


def keep_set(x, w_down, want, criterion):
    """Which channels survive, and the correction their removal implies."""
    if criterion == "fluctuation":
        mu = x.mean(axis=0)
        score = (np.linalg.norm(w_down, axis=0) ** 2) * np.var(x, axis=0)
        keep = np.sort(np.argsort(score)[w_down.shape[1] - want:])
        corr = w_down @ mu - w_down[:, keep] @ mu[keep]
        return keep, w_down, corr, float("nan")

    chosen, tau_used = None, None
    for tau in TAU_GRID:
        eps = EPS_THRESHOLD if tau >= 0.0 else float("inf")
        g = greedy_group(x.T, np.linalg.norm(w_down, axis=0),
                         float(np.linalg.norm(x @ w_down.T)), tau=tau,
                         eps_threshold=eps)
        chosen, tau_used = g, tau
        if len(g.groups) <= want:
            break
    trim_to_budget(chosen, want)
    keep = np.array(sorted(gr.representative for gr in chosen.groups))
    return keep, build_compensated_weight(w_down, chosen), None, tau_used


def prune_model(model, removal, criterion, calib_t=None, verbose=True,
                tau=None):
    """Remove feed-forward channels from every layer, in place.

    Two modes, and the difference is the point of the method. With `tau` set,
    each layer gives up exactly what the criterion endorses THERE, so the
    removal varies by layer and the total is whatever it turns out to be --
    this is the "mildest sufficient change" the cascade is built around. With
    `removal` set instead, every layer gives up the same fraction whether its
    channels are redundant or not, which is the forced comparison.

    Returns the per-layer removal fractions actually applied.
    """
    if calib_t is None:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        _, calib = load_segments(tok)
        calib_t = [calib[i:i + 1] for i in range(len(calib))]

    n_layers = len(model.model.layers)
    width = model.model.layers[0].mlp.down_proj.weight.shape[1]
    want = None if tau is not None else int(round(width * (1.0 - removal)))
    if verbose:
        if tau is not None:
            print(f"{n_layers} qatlam, kenglik {width}, tau={tau} — "
                  f"har bir qatlam mezon tasdiqlagancha qisqaradi\n")
        else:
            print(f"{n_layers} qatlam, oraliq kenglik {width} -> {want} "
                  f"({removal*100:.0f}% olib tashlanadi), mezon: {criterion}\n")

    taus = []
    for li in range(n_layers):
        mlp = model.model.layers[li].mlp
        x = capture_down_input(model, calib_t, li)
        w_down = mlp.down_proj.weight.detach().numpy().astype(np.float64)
        if tau is not None:
            keep, w_new, corr = keep_at_tau(x, w_down, tau)
        else:
            keep, w_new, corr, _ = keep_set(x, w_down, want, criterion)
        taus.append(1.0 - len(keep) / width)

        with torch.no_grad():
            mlp.down_proj.weight = torch.nn.Parameter(
                torch.tensor(w_new[:, keep], dtype=torch.float32))
            mlp.down_proj.in_features = len(keep)
            for nm in ("gate_proj", "up_proj"):
                p = getattr(mlp, nm)
                p.weight = torch.nn.Parameter(p.weight.detach()[keep, :].clone())
                p.out_features = len(keep)
            # A Llama feed-forward carries no bias, so the correction the
            # fluctuation criterion produces has nowhere to go. Dropping it
            # would apply half the method -- the half that removes channels,
            # without the half that puts back what their average contributed.
            # A bias vector is therefore ADDED where the architecture lacks
            # one, which is a real change and is priced: 3200 floats per
            # layer, 333 KiB over the model, against the 87 MiB the 20%
            # removal saves.
            if corr is not None:
                if mlp.down_proj.bias is None:
                    mlp.down_proj.bias = torch.nn.Parameter(
                        torch.zeros(w_down.shape[0], dtype=torch.float32))
                mlp.down_proj.bias += torch.tensor(corr, dtype=torch.float32)
        if verbose:
            print(f"  L{li:<2d} {width} -> {len(keep)}  "
                  f"({taus[-1]*100:.2f}% olindi)", flush=True)
        del x, w_down, w_new
        gc.collect()
    if verbose and tau is not None:
        print(f"\n  tau={tau} bo'yicha jami: o'rtacha "
              f"{sum(taus)/len(taus)*100:.2f}% "
              f"(eng kam {min(taus)*100:.2f}%, eng ko'p {max(taus)*100:.2f}%)")
    return taus


def main():
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    test, calib = load_segments(tok)
    calib_t = [calib[i:i + 1] for i in range(len(calib))]

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    width = model.model.layers[0].mlp.down_proj.weight.shape[1]
    want = int(round(width * (1.0 - REMOVAL)))
    taus = prune_model(model, REMOVAL, CRITERION, calib_t)

    print("\nperplexity hisoblanmoqda...", flush=True)
    ppl = perplexity(model, test)
    print(f"\nWikiText-2 perplexity = {ppl:.3f}   (FP32 7.547, INT8 7.550)")

    rows = {}
    if os.path.exists(OUT_JSON):
        try:
            rows = json.load(open(OUT_JSON, encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            rows = {}
    rows[f"{int(REMOVAL*100)}% {CRITERION}"] = {
        "ppl": float(ppl), "removal": REMOVAL, "criterion": CRITERION,
        "width": int(width), "kept": int(want),
        "taus": [None if t != t else float(t) for t in taus]}
    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    print(f"saqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
