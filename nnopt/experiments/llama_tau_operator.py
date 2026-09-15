"""What each operating point costs at operator level, on held-out activations.

The end-to-end walk answers this properly but costs a day. This answers the
SHAPE of it in minutes, and it answers two further questions the same
measurement is already set up for.

1. What does tau itself buy?  The cascade's claim is that removal should be
   chosen by the criterion, not as a global percentage. That claim is only
   meaningful if a tau-chosen set of channels is CHEAPER than an equally
   large arbitrary set. Both are measured here at matched size, so the
   comparison is like for like rather than "less removal costs less".

2. Would a bias help what is left?  Group compensation folds a removed
   channel into its representative through a scalar gamma, which is exact
   only where the two are truly collinear. Whatever it misses is a residual,
   and if that residual has a non-zero MEAN then a bias vector -- free in bit
   width, one vector per layer -- removes it. Whether the mean is actually
   large is not obvious and has never been measured here, so the residual is
   decomposed into its constant part and the rest.

3. The same decomposition is applied to INT8 quantization error, which is the
   other place a bias could be added and is currently not.

Errors are measured on activations the grouping never saw. Fitting and
scoring on the same rows is exactly how the low-rank expansion once reported
0.00000 fit error against 0.04355 held out (Sec 4.6).
"""

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
from nnopt.quantizer.per_channel import (
    quantize_codes_pc,
    refine_scales_per_channel,
)
from wikitext2_int4 import MODEL_DIR, load_segments

OUT_JSON = "experiments/results_llama_tau_operator.json"
LAYERS = (0, 4, 8, 12, 16, 20, 24)
TAUS = (0.99, 0.95, 0.90)
RATIOS = (0.10, 0.20, 0.30)
EPS_THRESHOLD = 0.5
ROWS = 2048
THREADS = 4          # the end-to-end walk holds the rest of the machine


def capture(model, seg_fit, seg_ho, layers):
    """down_proj inputs for several layers, fit and held-out, in two passes."""
    store = {li: [] for li in layers}

    def mk(li):
        def hook(mod, inputs, output):
            store[li].append(inputs[0].detach().to(torch.float32)
                             .reshape(-1, inputs[0].shape[-1]))
        return hook

    hs = [model.model.layers[li].mlp.down_proj.register_forward_hook(mk(li))
          for li in layers]
    out = {}
    with torch.no_grad():
        for name, seg in (("fit", seg_fit), ("ho", seg_ho)):
            for li in layers:
                store[li] = []
            model(input_ids=seg)
            for li in layers:
                x = torch.cat(store[li], 0).numpy().astype(np.float64)
                out[(li, name)] = x[:ROWS]
    for h in hs:
        h.remove()
    return out


def split_residual(y, y_hat):
    """Relative error, and how much of it a constant vector could remove.

    Returns (E_loc, E_loc after subtracting the residual's mean, the share of
    residual energy that the mean accounts for). The third is the number that
    decides whether a bias is worth adding: a residual that is mostly
    zero-mean noise cannot be fixed by any constant.
    """
    r = y - y_hat
    den = np.linalg.norm(y)
    mu = r.mean(axis=0, keepdims=True)
    e = float(np.linalg.norm(r) / den)
    e_centered = float(np.linalg.norm(r - mu) / den)
    share = float((np.linalg.norm(mu) ** 2 * r.shape[0])
                  / max(np.linalg.norm(r) ** 2, 1e-30))
    return e, e_centered, share


def cosine_keep(x_fit, w, want=None, tau=None):
    """Channels the criterion keeps, and the compensated weight.

    With `tau` the count is an output; with `want` the criterion is forced
    down until it fits that count. The two are the arms being compared.
    """
    if tau is not None:
        eps = EPS_THRESHOLD if tau >= 0.0 else float("inf")
        g = greedy_group(x_fit.T, np.linalg.norm(w, axis=0),
                         float(np.linalg.norm(x_fit @ w.T)), tau=tau,
                         eps_threshold=eps)
    else:
        g = None
        for t in (0.99, 0.9, 0.7, 0.5, 0.3, 0.0, -1.0):
            eps = EPS_THRESHOLD if t >= 0.0 else float("inf")
            g = greedy_group(x_fit.T, np.linalg.norm(w, axis=0),
                             float(np.linalg.norm(x_fit @ w.T)), tau=t,
                             eps_threshold=eps)
            if len(g.groups) <= want:
                break
        trim_to_budget(g, want)
    keep = np.array(sorted(gr.representative for gr in g.groups))
    return keep, build_compensated_weight(w, g)


def fluctuation_keep(x_fit, w, want):
    score = (np.linalg.norm(w, axis=0) ** 2) * np.var(x_fit, axis=0)
    return np.sort(np.argsort(score)[w.shape[1] - want:]), w


def main():
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _, calib = load_segments(tok)
    if len(calib) < 2:
        raise SystemExit("kamida 2 kalibrlash segmenti kerak (fit + held-out)")

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    print(f"faolliklar olinmoqda ({len(LAYERS)} qatlam, fit + held-out)...",
          flush=True)
    acts = capture(model, calib[0:1], calib[1:2], LAYERS)

    rows = []
    done = set()
    if os.path.exists(OUT_JSON):
        try:
            rows = json.load(open(OUT_JSON, encoding="utf-8-sig"))
            done = {r["layer"] for r in rows}
            print(f"keshdan {len(done)} qatlam o'qildi: {sorted(done)}")
        except (json.JSONDecodeError, OSError):
            rows = []

    for li in LAYERS:
        if li in done:
            continue
        x_fit, x_ho = acts[(li, "fit")], acts[(li, "ho")]
        w = model.model.layers[li].mlp.down_proj.weight.detach() \
            .numpy().astype(np.float64)
        width = w.shape[1]
        y_ho = x_ho @ w.T

        arms = [(f"tau={t}", dict(tau=t)) for t in TAUS]
        for li_ratio in RATIOS:
            arms.append((f"{int(li_ratio*100)}% majburiy kosinus",
                         dict(want=int(round(width * (1 - li_ratio))))))
            arms.append((f"{int(li_ratio*100)}% fluktuatsiya",
                         dict(want=int(round(width * (1 - li_ratio))),
                              fluct=True)))

        for name, kw in arms:
            if kw.pop("fluct", False):
                keep, w_c = fluctuation_keep(x_fit, w, kw["want"])
            else:
                keep, w_c = cosine_keep(x_fit, w, **kw)
            y_hat = x_ho[:, keep] @ w_c[:, keep].T
            e, e_c, share = split_residual(y_ho, y_hat)
            rows.append({"layer": li, "arm": name,
                         "removed": 1.0 - len(keep) / width,
                         "e_loc": e, "e_loc_centered": e_c,
                         "mean_share": share})
            print(f"  L{li:<2d} {name:24s} olindi {rows[-1]['removed']*100:5.2f}%  "
                  f"E={e:.4f}  bias'dan keyin {e_c:.4f}  "
                  f"(o'rtacha ulushi {share*100:.2f}%)", flush=True)

        # The other place a bias could go: INT8 error on the untouched matrix.
        res = refine_scales_per_channel(w, 127, x_calib=x_fit)
        w_q = quantize_codes_pc(w, res.scales, 127) * res.scales
        e, e_c, share = split_residual(y_ho, x_ho @ w_q.T)
        rows.append({"layer": li, "arm": "INT8 (kesishsiz)", "removed": 0.0,
                     "e_loc": e, "e_loc_centered": e_c, "mean_share": share})
        print(f"  L{li:<2d} {'INT8 (kesishsiz)':24s} {'':13s}  E={e:.4f}  "
              f"bias'dan keyin {e_c:.4f}  (o'rtacha ulushi {share*100:.2f}%)\n",
              flush=True)
        # Written per layer, not at the end: this run has been killed twice by
        # session boundaries, and each time every completed layer was lost.
        json.dump(rows, open(OUT_JSON, "w"), indent=2)

    print("=" * 78)
    print(f"{'Arm':26s} {'olindi':>8s} {'E_loc':>8s} {'bias bilan':>11s} "
          f"{'o`rtacha ulushi':>16s}")
    print("-" * 78)
    for name in [f"tau={t}" for t in TAUS] + \
                [f"{int(r*100)}% {k}" for r in RATIOS
                 for k in ("majburiy kosinus", "fluktuatsiya")] + \
                ["INT8 (kesishsiz)"]:
        sel = [r for r in rows if r["arm"] == name]
        if not sel:
            continue
        print(f"{name:26s} {np.mean([r['removed'] for r in sel])*100:7.2f}% "
              f"{np.mean([r['e_loc'] for r in sel]):8.4f} "
              f"{np.mean([r['e_loc_centered'] for r in sel]):11.4f} "
              f"{np.mean([r['mean_share'] for r in sel])*100:15.2f}%")
    print("=" * 78)
    print(f"saqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
