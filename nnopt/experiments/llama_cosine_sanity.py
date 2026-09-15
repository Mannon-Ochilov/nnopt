"""Is "no collinear channels" a property of the model, or a bug in the call?

greedy_group reports 0.00% merged at tau = 0.99 on most Llama layers. That is
either true or a mistake in how it is being invoked, and the two are
distinguishable without trusting the grouping code at all: the criterion is
cos(h_i, h_j) >= tau, so a brute-force scan of the pairwise cosines says how
many pairs COULD merge, and the grouping can then be checked against it.

Three things are separated here.

  1. The cosine distribution itself, so "none above 0.99" can be seen rather
     than inferred.
  2. What the eps criterion rejects. A pair can clear tau and still be
     refused for impact, and that would look identical from outside.
  3. Whisper's encoder for contrast, since the same code reports 17.1% there
     -- if the Llama number were a calling error, the Whisper number obtained
     the same way should be wrong too.

The likely honest answer is architectural: Whisper's fc2 reads a GELU output,
which is almost entirely non-negative, so its channel vectors sit in a
positive cone where cosines are high by construction. A Llama down_proj reads
silu(gate) * up, a product that changes sign freely, so its channel vectors
spread over the whole sphere and near-collinear pairs are rare. If that is
what is happening, the cosine histograms will differ in exactly that way.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnopt.grouping.functional_grouping import greedy_group
from wikitext2_int4 import MODEL_DIR, load_segments

LAYER = int(__import__("os").environ.get("LAYER", "8"))
ROWS = 2048
TAUS = (0.99, 0.95, 0.90, 0.70, 0.50)
EPS_THRESHOLD = 0.5
THREADS = 2          # the operator sweep owns the rest


def capture(model, seg, layer_idx):
    store = []
    h = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(
        lambda m, i, o: store.append(i[0].detach().to(torch.float32)
                                     .reshape(-1, i[0].shape[-1])))
    with torch.no_grad():
        model(input_ids=seg)
    h.remove()
    return torch.cat(store, 0).numpy().astype(np.float64)[:ROWS]


def main():
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _, calib = load_segments(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    x = capture(model, calib[0:1], LAYER)
    w = model.model.layers[LAYER].mlp.down_proj.weight.detach() \
        .numpy().astype(np.float64)
    del model

    h = x.T                                     # (channels, rows)
    n = h.shape[0]
    print(f"L{LAYER}: {n} kanal, {h.shape[1]} qator\n")

    # Sign of the activations: the architectural question above.
    frac_pos = float((x > 0).mean())
    print(f"musbat faolliklar ulushi: {frac_pos*100:.1f}%  "
          f"(GELU chiqishi uchun ~100% kutiladi, gated ko'paytma uchun ~50%)")

    norms = np.linalg.norm(h, axis=1)
    hn = h / (norms[:, None] + 1e-30)
    print("\nkosinus matritsasi hisoblanmoqda "
          f"({n}x{n}, ~{n*n*8/1e9:.1f} GB float64 -> float32 da yarmi)...")
    c = (hn @ hn.T).astype(np.float32)
    np.fill_diagonal(c, -2.0)                   # exclude self-pairs

    best = c.max(axis=1)                        # each channel's nearest neighbour
    print("\nHar bir kanalning ENG YAQIN qo'shnisi bilan kosinusi:")
    for q in (50, 90, 99, 99.9, 100):
        print(f"  {q:5.1f}-protsentil : {np.percentile(best, q):.4f}")

    print("\nBerilgan tau dan yuqori kosinusga ega kanallar soni:")
    for t in TAUS:
        cnt = int((best >= t).sum())
        pairs = int((c >= t).sum() // 2)
        print(f"  tau={t:.2f}: {cnt:5d} kanal ({cnt/n*100:5.2f}%), "
              f"{pairs} juftlik")

    # What the grouping actually does, and what eps costs on top of tau.
    print("\ngreedy_group natijasi (eps bilan va epssiz):")
    wn = np.linalg.norm(w, axis=0)
    y_norm = float(np.linalg.norm(x @ w.T))
    for t in TAUS:
        g_eps = greedy_group(h, wn, y_norm, tau=t, eps_threshold=EPS_THRESHOLD)
        g_inf = greedy_group(h, wn, y_norm, tau=t, eps_threshold=float("inf"))
        r_eps = 1.0 - len(g_eps.groups) / n
        r_inf = 1.0 - len(g_inf.groups) / n
        print(f"  tau={t:.2f}: eps=0.5 -> {r_eps*100:5.2f}% birlashdi, "
              f"eps=inf -> {r_inf*100:5.2f}%")

    print("\nIzoh: 'tau dan yuqori kanallar' ustuni bilan 'eps=inf' ustuni "
          "mos kelsa, algoritm to'g'ri ishlaydi va 0% modelning xossasi.")


if __name__ == "__main__":
    main()
