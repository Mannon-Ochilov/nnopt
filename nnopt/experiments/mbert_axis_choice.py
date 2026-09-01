"""Which structural axis does mBERT offer, if any?

The axis diagnostic on Whisper found a sharp crossover: channels win while the
criterion still endorses them (tau near 0.99) and rank wins once tau has to
collapse. mBERT is the extreme case of the second regime -- its criterion
finds 0.08% of channels at tau = 0.99 and 8.3% even at 0.90, four times less
than Whisper at every threshold -- so the prediction is that rank should win
in essentially every layer, and that the ratio ladder the framework currently
gives this model is forcing the wrong axis.

The comparison is the same one used for Whisper: equal parameter count, on
held-out activations. An operator reduced by channels costs d_model * k; the
same operator factored at rank r costs r * (d_model + d_ff), so k = 2458
(the 20% rung) pairs with r = 491.

Held-out matters as much here as there. At this rank the row/rank ratio is
about 6, below the 10-20 band Sec 4.6 requires, so the rank arm is at risk of
memorising its calibration -- which is a reason to report the ratio alongside
the result rather than to leave the comparison out.
"""

import gc
import json

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

OUT_JSON = "experiments/results_mbert_axis.json"
REMOVAL = 0.20
EPS_THRESHOLD = 0.5
TAU_GRID = (0.99, 0.9, 0.7, 0.5, 0.3, 0.0, -1.0)
FIT_BATCHES, HO_BATCHES = 25, 25
MAX_ROWS = 6144


def e_loc(y, y_hat):
    return float(np.linalg.norm(y - y_hat) / (np.linalg.norm(y) + 1e-12))


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
    print(f"{len(pairs)} juftlik, {REMOVAL*100:.0f}% byudjetda taqqoslash\n")

    rows = []
    for pr in sorted(pairs, key=lambda p: p.layer):
        con = by_name[pr.contract]
        x_fit = capture_ffn_activations([con.activation_input], fit_batches,
                                        max_rows=MAX_ROWS)[con.activation_input]
        x_ho = capture_ffn_activations([con.activation_input], ho_batches,
                                       max_rows=MAX_ROWS)[con.activation_input]
        w2s = numpy_helper.to_array(inits[con.weight_initializer]).astype(np.float64)
        w2 = w2s if w2s.shape[1] == x_fit.shape[1] else w2s.T

        k = int(round(w2.shape[1] * (1.0 - REMOVAL)))
        rank = int(round(w2.shape[0] * k / (w2.shape[0] + w2.shape[1])))

        chosen = None
        for tau in TAU_GRID:
            eps = EPS_THRESHOLD if tau >= 0.0 else float("inf")
            g = greedy_group(x_fit.T, np.linalg.norm(w2, axis=0),
                             float(np.linalg.norm(x_fit @ w2.T)), tau=tau,
                             eps_threshold=eps)
            chosen, tau_used = g, tau
            if len(g.groups) <= k:
                break
        trim_to_budget(chosen, k)
        keep = np.array(sorted(gr.representative for gr in chosen.groups))
        w2c = build_compensated_weight(w2, chosen)

        y = x_ho @ w2.T
        e_chan = e_loc(y, x_ho[:, keep] @ w2c[:, keep].T)
        lr = activation_aware_svd(w2, x_fit, rank)
        e_rank = e_loc(y, x_ho @ lr.T)

        win = "kanal" if e_chan < e_rank else "RANK"
        rows.append({"layer": pr.layer, "tau": tau_used, "k": int(k),
                     "rank": rank, "e_channels": e_chan, "e_rank": e_rank,
                     "rows": int(x_fit.shape[0]),
                     "row_rank_ratio": x_fit.shape[0] / rank})
        print(f"  L{pr.layer:<2d} tau={tau_used:5.2f} k={k} rank={rank}  "
              f"E_kanal={e_chan:.4f}  E_rank={e_rank:.4f}  -> {win}",
              flush=True)
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
        del x_fit, x_ho, w2, w2c, lr, y
        gc.collect()

    print("\n" + "=" * 70)
    wins = sum(1 for r in rows if r["e_rank"] < r["e_channels"])
    print(f"rank yutgan qatlamlar: {wins}/{len(rows)}")
    print(f"o'rtacha E_kanal = {np.mean([r['e_channels'] for r in rows]):.4f}, "
          f"E_rank = {np.mean([r['e_rank'] for r in rows]):.4f}")
    print(f"qator/rank nisbati: {min(r['row_rank_ratio'] for r in rows):.1f} "
          f"— 4.6-bo'limdagi 10-20 talabidan past bo'lsa, rank armi "
          f"kalibrlashni yodlab olayotgan bo'lishi mumkin.")


if __name__ == "__main__":
    main()
