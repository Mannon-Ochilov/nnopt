"""From distribution to outcome: does DistilBERT's high cosine floor buy
cheap channel removal?

The prediction test (geometry_prediction.py) established the DISTRIBUTION:
DistilBERT's GELU axis is one-signed and its nearest-neighbour cosines sit
far above the gated models'. The geometry claim says that should translate
into cheap criterion-endorsed removal — the "column-selection candidate"
cell of the family table. This measures it end to end, with the predictions
stated first:

  1. At tau = 0.90 the criterion endorses a NON-TRIVIAL removal (well above
     the ~0.1% of the gated models).
  2. The criterion-endorsed removal costs little masked-LM accuracy.
  3. A RANDOM removal of the same size costs measurably more — i.e. the
     criterion's choice, not just the budget, carries the effect.

Protocol mirrors the mBERT task metric: Uzbek text, deterministic masking,
top-1 accuracy on masked positions, all arms scored on the SAME positions so
comparisons are paired. Model runs in PyTorch (no ONNX export needed);
compensation folds removed channels into representatives exactly as in the
main pipeline, bias rows dropped with their channels.
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from nnopt.grouping.functional_grouping import (
    build_compensated_weight,
    greedy_group,
)
from mbert_task_metric import load_texts

MODEL_NAME = "distilbert-base-multilingual-cased"
TAU = 0.90
EPS_THRESHOLD = 0.5
N_EVAL_TEXT = 400
MASK_FRACTION = 0.15
SEED = 7
OUT_JSON = "experiments/results_distilbert_e2e.json"
THREADS = 8


def capture_lin2_inputs(model, enc):
    layers = model.distilbert.transformer.layer
    store = {i: [] for i in range(len(layers))}
    hs = []
    for i, layer in enumerate(layers):
        hs.append(layer.ffn.lin2.register_forward_hook(
            lambda m, inp, out, i=i: store[i].append(
                inp[0].detach().reshape(-1, inp[0].shape[-1]))))
    with torch.no_grad():
        model(**enc)
    for h in hs:
        h.remove()
    mask = enc["attention_mask"].reshape(-1).bool()
    return {i: torch.cat(v, 0)[mask].numpy().astype(np.float64)
            for i, v in store.items()}


def prune_layer(layer, keep, w2_comp):
    with torch.no_grad():
        layer.ffn.lin2.weight = torch.nn.Parameter(
            torch.tensor(w2_comp[:, keep], dtype=torch.float32))
        layer.ffn.lin2.in_features = len(keep)
        layer.ffn.lin1.weight = torch.nn.Parameter(
            layer.ffn.lin1.weight.detach()[keep, :].clone())
        layer.ffn.lin1.bias = torch.nn.Parameter(
            layer.ffn.lin1.bias.detach()[keep].clone())
        layer.ffn.lin1.out_features = len(keep)


def masked_eval(model, tok, texts):
    """Top-1 hits on deterministically masked positions."""
    rng = np.random.default_rng(SEED)
    hits = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            enc = tok(texts[start:start + 16], return_tensors="pt",
                      padding=True, truncation=True, max_length=64)
            ids = enc["input_ids"].clone()
            can_mask = (enc["attention_mask"].bool()
                        & (ids != tok.cls_token_id)
                        & (ids != tok.sep_token_id)
                        & (ids != tok.pad_token_id))
            mask_flags = (torch.tensor(rng.random(ids.shape)) < MASK_FRACTION) \
                & can_mask
            if not mask_flags.any():
                continue
            masked = ids.clone()
            masked[mask_flags] = tok.mask_token_id
            logits = model(input_ids=masked,
                           attention_mask=enc["attention_mask"]).logits
            pred = logits.argmax(-1)
            hits.extend((pred[mask_flags] == ids[mask_flags])
                        .float().tolist())
    return hits


def paired(a, b, n=2000, seed=1):
    a, b = np.array(a, float), np.array(b, float)
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


def load_model():
    m = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    m.eval()
    return m


def main():
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    calib_texts, eval_texts = load_texts(0, 400, n_eval=N_EVAL_TEXT)
    eval_texts = eval_texts[:N_EVAL_TEXT]

    model = load_model()
    enc = tok(calib_texts[:256], return_tensors="pt", padding=True,
              truncation=True, max_length=64)
    print("faolliklar olinmoqda...", flush=True)
    acts = capture_lin2_inputs(model, enc)

    # Criterion-endorsed removal at a fixed tau: the count is an OUTPUT.
    layers = model.distilbert.transformer.layer
    keeps, removed = {}, []
    rng = np.random.default_rng(3)
    for i, layer in enumerate(layers):
        x = acts[i]
        w2 = layer.ffn.lin2.weight.detach().numpy().astype(np.float64)
        g = greedy_group(x.T, np.linalg.norm(w2, axis=0),
                         float(np.linalg.norm(x @ w2.T)), tau=TAU,
                         eps_threshold=EPS_THRESHOLD)
        keep = np.array(sorted(gr.representative for gr in g.groups))
        keeps[i] = (keep, build_compensated_weight(w2, g))
        removed.append(1.0 - len(keep) / w2.shape[1])
        print(f"  L{i}: {w2.shape[1]} -> {len(keep)} "
              f"({removed[-1]*100:.2f}% olindi)", flush=True)
    print(f"tau={TAU} o'rtacha olib tashlash: "
          f"{np.mean(removed)*100:.2f}%\n", flush=True)

    print("[FP32] baholanmoqda...", flush=True)
    hits_fp32 = masked_eval(model, tok, eval_texts)

    print("[mezon] qo'llanmoqda va baholanmoqda...", flush=True)
    for i, layer in enumerate(layers):
        keep, w2c = keeps[i]
        prune_layer(layer, keep, w2c)
    hits_crit = masked_eval(model, tok, eval_texts)

    print("[tasodifiy, teng hajm] baholanmoqda...", flush=True)
    model2 = load_model()
    for i, layer in enumerate(model2.distilbert.transformer.layer):
        n = layer.ffn.lin2.weight.shape[1]
        want = len(keeps[i][0])
        keep = np.sort(rng.choice(n, want, replace=False))
        # Random arm gets NO compensation on purpose: it models the blind
        # baseline a practitioner applies; the criterion arm's advantage is
        # allowed to include its compensation, which is part of the method.
        w2 = layer.ffn.lin2.weight.detach().numpy().astype(np.float64)
        prune_layer(layer, keep, w2)
    hits_rand = masked_eval(model2, tok, eval_texts)

    acc = lambda h: float(np.mean(h))
    d_c, lo_c, hi_c = paired(hits_fp32, hits_crit)
    d_r, lo_r, hi_r = paired(hits_fp32, hits_rand)
    print("\n" + "=" * 72)
    print(f"FP32 aniqlik            = {acc(hits_fp32):.4f}  "
          f"({len(hits_fp32)} pozitsiya)")
    print(f"mezon (tau={TAU})        = {acc(hits_crit):.4f}   "
          f"farq {d_c:+.4f} [{lo_c:+.4f}, {hi_c:+.4f}]")
    print(f"tasodifiy (teng hajm)   = {acc(hits_rand):.4f}   "
          f"farq {d_r:+.4f} [{lo_r:+.4f}, {hi_r:+.4f}]")
    print("=" * 72)

    json.dump({"tau": TAU, "removed_mean": float(np.mean(removed)),
               "removed_per_layer": [float(r) for r in removed],
               "n_positions": len(hits_fp32),
               "acc_fp32": acc(hits_fp32), "acc_criterion": acc(hits_crit),
               "acc_random": acc(hits_rand),
               "delta_criterion": [d_c, lo_c, hi_c],
               "delta_random": [d_r, lo_r, hi_r]},
              open(OUT_JSON, "w"), indent=2)
    print(f"saqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
