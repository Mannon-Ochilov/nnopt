"""Is there real functional redundancy in the FFN intermediate channels?

Rationale. Every comparison so far pitted CUR against SVD at the SAME job --
low-rank approximation -- where Eckart-Young guarantees SVD wins. That is
the wrong role for the dissertation's actual contribution. The core idea is
FUNCTIONAL GROUPING: using real calibration activations to find channels
that carry the same response and merge them with compensation. That is not
approximation, it is exact structural removal when h_j = alpha * h_p:

    W[:, p] <- W[:, p] + alpha * W[:, j],   then drop column j

In a transformer FFN this pays twice. The intermediate width (4096 here) is
fc1's OUTPUT and fc2's INPUT, so removing k intermediate channels shrinks
    fc1: (4096 x 1024) -> (4096-k x 1024)
    fc2: (1024 x 4096) -> (1024 x 4096-k)
from a single decision. Low-rank cannot do this: it reduces rank while n and
m stay put. The two axes therefore COMPOSE rather than compete -- pruning k
channels frees budget for a higher rank at equal parameter count:

    r' * (m + n - k) = r * (m + n)   =>   r' > r

This script answers the precondition: how much genuine redundancy is there?
For each encoder layer's intermediate activation it measures, at several
similarity thresholds, how many channels are mergeable and what the
compensated weight costs in output error.
"""

import gc
import json

import numpy as np
import onnx
from onnx import numpy_helper

from calib_utils import ENCODER_PATH, capture_activations, encoder_feeds
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.profiler.graph_profiler import profile_onnx_model

N_CALIB_UTT = 6
FIT_ROWS, EVAL_ROWS = 3072, 1024
LAYERS = [0, 4, 8, 12, 16, 20, 23]
TAUS = [0.99, 0.95, 0.90, 0.80, 0.70]
EPS_THRESHOLD = 0.5          # generous: we want to see the redundancy ceiling
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
OUT_JSON = "experiments/results_ffn_redundancy.json"


def rel_err(y_ref, y_hat):
    return float(np.linalg.norm(y_ref - y_hat) / (np.linalg.norm(y_ref) + 1e-12))


def main():
    profs = profile_onnx_model(ENCODER_PATH, free_dims=ENC_DIMS)
    # fc2 consumes the intermediate: its input dim is the FFN width
    fc2s = [p for p in profs if p.weight_initializer and "/fc2/" in p.name
            and any(f"/layers.{li}/" in p.name for li in LAYERS)]
    print(f"{len(fc2s)} ta fc2 operatori tekshiriladi (oraliq kengligi = fc2 kirishi)")

    feeds = encoder_feeds(0, N_CALIB_UTT)
    enc = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in enc.graph.initializer}

    results = {}
    for p in fc2s:
        x_by = capture_activations(ENCODER_PATH, [p.activation_input], feeds,
                                   max_rows=FIT_ROWS + EVAL_ROWS)
        x = x_by[p.activation_input]
        w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        m, n = w.shape
        x_fit, x_eval = x[:FIT_ROWS], x[FIT_ROWS:FIT_ROWS + EVAL_ROWS]
        y_ref = x_eval @ w.T
        y_norm_fit = float(np.linalg.norm(x_fit @ w.T))
        col_norms = np.linalg.norm(w, axis=0)

        entry = {"m": m, "n": n, "taus": {}}
        print(f"\n{p.name.replace('/encoder/','')}  W={m}x{n}")
        print(f"  {'tau':>6s} {'guruhlar':>9s} {'olib tashlanadi':>16s} {'ulush':>7s} {'E_loc':>9s}")
        for tau in TAUS:
            g = greedy_group(x_fit.T, col_norms, y_norm_fit, tau=tau,
                             eps_threshold=EPS_THRESHOLD)
            n_groups = len(g.groups)
            removed = n - n_groups
            w_comp = build_compensated_weight(w, g)
            # compensated matrix: non-representative columns are zeroed, so
            # dropping them is exact for the compensated operator
            e = rel_err(y_ref, x_eval @ w_comp.T)
            entry["taus"][str(tau)] = {"groups": n_groups, "removed": int(removed),
                                       "fraction": removed / n, "e_loc": e}
            print(f"  {tau:6.2f} {n_groups:9d} {removed:16d} {removed/n:6.1%} {e:9.4f}")
            del w_comp, g
            gc.collect()
        results[p.name] = entry
        del x, x_by, w, x_fit, x_eval, y_ref
        gc.collect()

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")

    print("\n" + "=" * 80)
    print("XULOSA: oraliq kanallarda funksional ortiqchalik bormi?")
    print("=" * 80)
    print(f"{'tau':>6s} {'o''rtacha olib tashlanadi':>24s} {'o''rtacha E_loc':>15s}")
    for tau in TAUS:
        fr = [results[k]["taus"][str(tau)]["fraction"] for k in results]
        el = [results[k]["taus"][str(tau)]["e_loc"] for k in results]
        print(f"{tau:6.2f} {np.mean(fr):23.1%} {np.mean(el):15.4f}")
    print("\nAgar ulush sezilarli (>10%) va E_loc kichik bo'lsa -> guruhlash o'qi ishlaydi")
    print("va u past-rank bilan QO'SHILADI (byudjet bo'shatib yuqoriroq rank beradi).")


if __name__ == "__main__":
    main()
