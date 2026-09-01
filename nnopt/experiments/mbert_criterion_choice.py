"""Which removal criterion does mBERT actually reward?

Forcing the cosine criterion down to tau = 0.70 to meet a 20% budget was never
the only option, and on this model it is the least likely to work: mBERT has
almost no collinear channels at any threshold (0.08% at tau = 0.99), so a
cosine-driven cut is choosing among channels its own criterion says are NOT
redundant.

The two-stage method (Sec 4.9d) was built for exactly this shape of problem.
Stage 1 keeps cosine strict and merges only what is genuinely collinear;
stage 2 spends the rest of the budget on the survivors that barely vary,
scored by ||W2_comp[:, p]||^2 * Var(h_p), and sweeps both stages' discarded
means into the output bias. On mBERT stage 1 should contribute almost nothing,
which makes the comparison sharp: if the two-stage arm wins, the win is the
fluctuation criterion, not the cosine one, and the framework's ladder for this
model should say so.

Four arms at the same 20% budget, held-out, with the bias correction included
where the method calls for it:

  kosinus majburiy    cosine down to whatever tau reaches the budget
  ikki bosqichli      cosine strict, then fluctuation, then bias
  sof fluktuatsiya    fluctuation only, then bias  (FLAP-shaped)
  rank                activation-aware low-rank at matched parameters
"""

import gc
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer

from mbert_analysis import (
    MBERT_DIR,
    MBERT_ONNX,
    build_text_batches,
    capture_ffn_activations,
)
from mbert_task_metric import FREE_DIMS, load_texts
from nnopt.cur.lowrank_baselines import activation_aware_svd
from nnopt.grouping.functional_grouping import (
    build_compensated_weight,
    greedy_group,
    trim_to_budget,
)
from nnopt.profiler.blocks import find_reducible_pairs
from nnopt.profiler.graph_profiler import profile_onnx_model

# The first run of this comparison captured activations without the attention
# mask, and that is not neutral between the arms: padding inflates cosine
# similarity, so the cosine arm was choosing channels that agree about [PAD].
# The fluctuation arm scores variance and is far less affected. The measured
# gap therefore had a bias in it, in the direction of the conclusion drawn.
# MASKED=0 reproduces the original run for comparison.
# The base filename holds the run the article reports, which is the unmasked
# one; the masked re-measurement is stored beside it rather than over it. An
# earlier edit had this the other way round and overwrote the article's data.
MASKED = os.environ.get("MASKED", "1") != "0"
OUT_JSON = ("experiments/results_mbert_criterion"
            + ("_masked" if MASKED else "") + ".json")
REMOVAL = 0.20
STRICT_TAU = 0.99
EPS_THRESHOLD = 0.5
TAU_GRID = (0.99, 0.9, 0.7, 0.5, 0.3, 0.0, -1.0)
FIT_BATCHES, HO_BATCHES = 25, 25
MAX_ROWS = 6144


def e_loc(y, y_hat):
    return float(np.linalg.norm(y - y_hat) / (np.linalg.norm(y) + 1e-12))


def group_to_budget(h, w_norms, y_norm, target):
    """Grouping pushed to `target` survivors, and the tau it needed."""
    chosen, tau_used = None, None
    for tau in TAU_GRID:
        eps = EPS_THRESHOLD if tau >= 0.0 else float("inf")
        g = greedy_group(h, w_norms, y_norm, tau=tau, eps_threshold=eps)
        chosen, tau_used = g, tau
        if len(g.groups) <= target:
            break
    trim_to_budget(chosen, target)
    return chosen, tau_used


def main():
    tok = AutoTokenizer.from_pretrained(MBERT_DIR)
    calib_texts, eval_texts = load_texts()
    fit_batches = build_text_batches(tok, calib_texts)[:FIT_BATCHES]
    ho_batches = build_text_batches(tok, eval_texts)[:HO_BATCHES]

    model = onnx.load(MBERT_ONNX)
    profs = [p for p in profile_onnx_model(MBERT_ONNX, free_dims=FREE_DIMS)
             if p.weight_initializer]
    pairs = find_reducible_pairs(model, profs)
    by_name = {p.name: p for p in profs}
    inits = {i.name: i for i in model.graph.initializer}
    print(f"{len(pairs)} juftlik, {REMOVAL*100:.0f}% teng byudjet, held-out\n")
    print(f"{'L':>3s} {'tau':>6s} {'1-bosq.':>8s} {'kosinus':>9s} "
          f"{'ikki bos.':>10s} {'fluktua.':>9s} {'rank':>9s}  yutuvchi")
    print("-" * 74)

    rows = []
    for pr in sorted(pairs, key=lambda p: p.layer):
        con = by_name[pr.contract]
        x_fit = capture_ffn_activations([con.activation_input], fit_batches,
                                        max_rows=MAX_ROWS,
                                        masked=MASKED)[con.activation_input]
        x_ho = capture_ffn_activations([con.activation_input], ho_batches,
                                       max_rows=MAX_ROWS,
                                       masked=MASKED)[con.activation_input]
        w2s = numpy_helper.to_array(inits[con.weight_initializer]).astype(np.float64)
        w2 = w2s if w2s.shape[1] == x_fit.shape[1] else w2s.T
        width = w2.shape[1]
        want = int(round(width * (1.0 - REMOVAL)))
        rank = int(round(w2.shape[0] * want / (w2.shape[0] + width)))
        wn = np.linalg.norm(w2, axis=0)
        y_norm = float(np.linalg.norm(x_fit @ w2.T))
        mean_h = x_fit.mean(axis=0)
        y = x_ho @ w2.T

        # --- cosine forced all the way to the budget -------------------
        g_forced, tau_used = group_to_budget(x_fit.T, wn, y_norm, want)
        keep_f = np.array(sorted(gr.representative for gr in g_forced.groups))
        w2_f = build_compensated_weight(w2, g_forced)[:, keep_f]
        e_cos = e_loc(y, x_ho[:, keep_f] @ w2_f.T)

        # --- stage 1: strict cosine, however little it finds ------------
        g1 = greedy_group(x_fit.T, wn, y_norm, tau=STRICT_TAU,
                          eps_threshold=EPS_THRESHOLD)
        keep1 = np.array(sorted(gr.representative for gr in g1.groups))
        w2_1 = build_compensated_weight(w2, g1)[:, keep1]
        stage1_removed = width - len(keep1)

        def fluctuation_arm(keep_in, w2_in):
            """Spend the remaining budget on the least varying survivors,
            then correct the output bias for everything discarded."""
            n_extra = max(0, len(keep_in) - want)
            score = (np.linalg.norm(w2_in, axis=0) ** 2) \
                * np.var(x_fit[:, keep_in], axis=0)
            sel = np.sort(np.argsort(score)[n_extra:]) if n_extra else \
                np.arange(len(keep_in))
            keep_o, w2_o = keep_in[sel], w2_in[:, sel]
            correction = w2 @ mean_h - w2_o @ mean_h[keep_o]
            return e_loc(y, x_ho[:, keep_o] @ w2_o.T + correction)

        e_two = fluctuation_arm(keep1, w2_1)
        e_flap = fluctuation_arm(np.arange(width), w2)

        lr = activation_aware_svd(w2, x_fit, rank)
        e_rank = e_loc(y, x_ho @ lr.T)

        scores = {"kosinus": e_cos, "ikki bosqichli": e_two,
                  "fluktuatsiya": e_flap, "rank": e_rank}
        win = min(scores, key=scores.get)
        rows.append({"layer": pr.layer, "tau": tau_used,
                     "stage1_removed": int(stage1_removed), **scores})
        print(f"{pr.layer:3d} {tau_used:6.2f} {stage1_removed:8d} "
              f"{e_cos:9.4f} {e_two:10.4f} {e_flap:9.4f} {e_rank:9.4f}  {win}",
              flush=True)
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
        del x_fit, x_ho, w2, lr, y
        gc.collect()

    print("-" * 74)
    keys = ("kosinus", "ikki bosqichli", "fluktuatsiya", "rank")
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    wins = {k: sum(1 for r in rows
                   if min(r[m] for m in keys) == r[k]) for k in keys}
    for k in keys:
        print(f"{k:16s} o'rtacha E = {means[k]:.4f}   g'alaba {wins[k]}/{len(rows)}")
    best = min(means, key=means.get)
    print(f"\nEng yaxshisi: {best} ({means[best]:.4f}); kosinus majburiy "
          f"{means['kosinus']:.4f}, ya'ni {means['kosinus']/means[best]:.2f}x.")
    print(f"1-bosqich o'rtacha {np.mean([r['stage1_removed'] for r in rows]):.0f} "
          f"kanal olib tashlaydi ({np.mean([r['stage1_removed'] for r in rows])/3072*100:.2f}%)")


if __name__ == "__main__":
    main()
