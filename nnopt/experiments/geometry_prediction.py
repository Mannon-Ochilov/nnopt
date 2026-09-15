"""The geometry law, tested on models it has never seen.

Sections 4.15/4.17 of the full manuscript observe that the SIGN STRUCTURE of
the activation entering the reducible axis decides which redundancy a model
has: a one-signed GELU output puts channel vectors in a cone where pairwise
cosines are structurally high (Whisper: 17.1% at tau = 0.99), a free-signed
gated product spreads them over the sphere and pairwise collinearity
vanishes (open_llama_3b: largest cosine 0.7681 across 37M pairs).

Three models make an observation; a PREDICTION needs models chosen before
measurement. Two are added here, picked only by their activation type:

  qwen2.5-0.5b   SwiGLU decoder  -> predicted: ~50% positive, no pair above
                                    tau = 0.90, nearest-neighbour cosines in
                                    the Llama range
  distilbert     GELU encoder    -> predicted: strongly one-signed, and a
                                    nearest-neighbour cosine distribution
                                    clearly ABOVE the gated models'

The second prediction is deliberately about the DISTRIBUTION, not about a
17% removal share: mBERT already shows that a GELU encoder can carry high
cosine levels yet little redundancy at tau = 0.99, so the honest claim is
'one-signed lifts the cosine floor', not 'one-signed gives Whisper's 17%'.

Both models are evaluated with the same protocol as the L8 sanity check on
Llama: capture the input of the contracting FFN projection at three depths,
compute the sign fraction, the exact nearest-neighbour |cosine| percentiles
and the share of channels above each tau.
"""

import json
import os

import numpy as np
import torch

MODEL = os.environ.get("MODEL", "qwen")   # qwen | distilbert
ROWS = 2048
TAUS = (0.99, 0.95, 0.90, 0.70, 0.50)
THREADS = 8
OUT_JSON = f"experiments/results_geometry_{MODEL}.json"


def nearest_abs(hn):
    c = hn @ hn.T
    np.fill_diagonal(c, 0.0)
    return np.abs(c).max(axis=1)


def analyze(x, tag):
    h = x.T
    n = h.shape[0]
    frac_pos = float((x > 0).mean())
    norms = np.linalg.norm(h, axis=1)
    hn = (h / (norms[:, None] + 1e-30)).astype(np.float32)
    best = nearest_abs(hn)
    row = {"tag": tag, "channels": int(n), "rows": int(h.shape[1]),
           "frac_pos": frac_pos,
           "pct": {str(q): float(np.percentile(best, q))
                   for q in (50, 90, 99, 100)},
           "above": {str(t): float((best >= t).mean()) for t in TAUS}}
    print(f"  {tag}: {n} kanal, musbat {frac_pos*100:5.1f}%  "
          f"med {row['pct']['50']:.3f}  max {row['pct']['100']:.3f}  "
          + "  ".join(f"t{t}:{row['above'][str(t)]*100:.2f}%"
                      for t in (0.90, 0.70, 0.50)), flush=True)
    return row


def run_qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()

    z = np.load("models/_calib_cache/wikitext2_test.npz", allow_pickle=True)
    ids = tok(str(z["calib"][0]), return_tensors="pt",
              truncation=True, max_length=8 * 1024).input_ids
    seg = ids[:, :2048]

    layers = model.model.layers
    picks = [len(layers) // 4, len(layers) // 2, 3 * len(layers) // 4]
    rows = []
    for li in picks:
        store = []
        hnd = layers[li].mlp.down_proj.register_forward_hook(
            lambda m, i, o: store.append(i[0].detach().to(torch.float32)
                                         .reshape(-1, i[0].shape[-1])))
        with torch.no_grad():
            model(input_ids=seg)
        hnd.remove()
        x = torch.cat(store, 0).numpy().astype(np.float64)[:ROWS]
        rows.append(analyze(x, f"L{li} down_proj kirishi (SwiGLU)"))
    return {"model": name, "activation": "SwiGLU (gated)",
            "prediction": "~50% musbat, tau>=0.90 da 0%", "layers": rows}


def run_distilbert():
    from transformers import AutoModel, AutoTokenizer
    name = "distilbert-base-multilingual-cased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()

    from mbert_task_metric import load_texts
    texts, _ = load_texts(0, 400)
    enc = tok(texts[:256], return_tensors="pt", padding=True,
              truncation=True, max_length=64)

    layers = model.transformer.layer
    picks = [1, len(layers) // 2, len(layers) - 1]
    rows = []
    for li in picks:
        store = []
        hnd = layers[li].ffn.lin2.register_forward_hook(
            lambda m, i, o: store.append(i[0].detach().to(torch.float32)
                                         .reshape(-1, i[0].shape[-1])))
        with torch.no_grad():
            model(**enc)
        hnd.remove()
        x = torch.cat(store, 0).numpy().astype(np.float64)
        # Padding positions would dilute the statistics with zeros.
        mask = enc["attention_mask"].reshape(-1).bool().numpy()
        x = x[mask][:ROWS]
        rows.append(analyze(x, f"L{li} lin2 kirishi (GELU)"))
    return {"model": name, "activation": "GELU (bir ishorali kutiladi)",
            "prediction": "kuchli bir ishorali, kosinus poli gated'dan yuqori",
            "layers": rows}


def main():
    torch.set_num_threads(THREADS)
    print(f"model: {MODEL}\n", flush=True)
    out = run_qwen() if MODEL == "qwen" else run_distilbert()
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
