"""Does the contribution gate of Eq. (4) ever actually reject anything?

Eq. (4) screens a merge candidate by the relative residual it would inject
into the operator output,

    eps_j = ||W_out[:,j]|| * ||h_j|| * sin(theta_jp) / (||Y||_F + xi) <= eps_thr

with eps_thr = 0.5 in every run (the code comment that set it reads
"generous: we want to see the redundancy ceiling" -- it was chosen for a
diagnostic sweep and then inherited by the main pipeline).

There is an arithmetic reason to doubt it binds. The residual of one merge
is rank-1, so its Frobenius norm is exactly ||w_j|| ||h_j|| sin(theta),
while the denominator is the norm of the WHOLE operator output, to which
roughly F channels contribute. If contributions are comparable and not
aligned, ||Y||_F grows like sqrt(F) times a typical channel term, so

    eps_j ~ sin(theta) / sqrt(F)

and at F = 4096 with tau = 0.99 (sin <= 0.141) that is about 0.002 -- some
two hundred times below the threshold. If so, the "second criterion" the
paper describes is inactive, and the selection is effectively cosine-only.

This measures the actual distribution rather than trusting the estimate:
for the merges the criterion proposes at the paper's operating point, what
are the eps values, how close do they come to 0.5, and what threshold would
be needed for the gate to reject anything at all.
"""

import json

import numpy as np
import onnx
from onnx import numpy_helper

from calib_utils import (
    ENCODER_PATH,
    CalibSet,
    capture_activations,
    feeds_for,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import layer_of
from nnopt.grouping.functional_grouping import greedy_group

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
FIT_ROWS = 3072
TAUS = (0.99, 0.95, 0.90)
EPS_THRESHOLD = 0.5          # the value used throughout the paper
LAYERS = (0, 2, 5, 8, 12, 16, 23)
OUT_JSON = "experiments/results_eps_gate_audit.json"


def main():
    calib = CalibSet(split="validation", skip=0, n=6)
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    feeds = feeds_for(calib)

    rows = []
    for li in LAYERS:
        p2 = fc2[li]
        x = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                max_rows=FIT_ROWS)[p2.activation_input]
        x = x.reshape(-1, x.shape[-1])[:FIT_ROWS].astype(np.float64)
        w2s = numpy_helper.to_array(inits[p2.weight_initializer]) \
            .astype(np.float64)
        w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T   # (d_out, F)
        h = x.T                                             # (F, rows)
        wn = np.linalg.norm(w2, axis=0)                     # ||W_out[:,j]||
        hn = np.linalg.norm(h, axis=1)                      # ||h_j||
        y_norm = float(np.linalg.norm(x @ w2.T))            # ||Y||_F
        F = h.shape[0]

        # The largest eps ANY single channel could produce, even if its
        # response were orthogonal to its representative (sin = 1). This
        # bounds the gate from above independently of which merges happen.
        eps_ceiling = float((wn * hn).max() / y_norm)

        for tau in TAUS:
            g = greedy_group(h, wn, y_norm, tau=tau,
                             eps_threshold=EPS_THRESHOLD)
            eps_vals = [e for gr in g.groups
                        for e in gr.eps_to_representative.values()]
            if not eps_vals:
                continue
            v = np.array(eps_vals)
            merged = len(v)
            rows.append({"layer": li, "tau": tau, "merges": merged,
                         "eps_max": float(v.max()),
                         "eps_p99": float(np.percentile(v, 99)),
                         "eps_median": float(np.median(v)),
                         "eps_ceiling_any_channel": eps_ceiling,
                         "headroom_vs_thr": EPS_THRESHOLD / float(v.max())})
            print(f"  L{li:<2d} tau={tau:.2f}  birlashma {merged:5d}  "
                  f"eps: mediana {np.median(v):.5f}  maks {v.max():.5f}  "
                  f"chegaraga {EPS_THRESHOLD / v.max():7.1f}x zaxira",
                  flush=True)
        del x, h
    if not rows:
        raise SystemExit("guruh obyektida eps qiymatlari topilmadi")

    allmax = max(r["eps_max"] for r in rows)
    ceil = max(r["eps_ceiling_any_channel"] for r in rows)
    print("\n" + "=" * 74)
    print(f"Barcha birlashmalar bo'yicha eng katta eps      : {allmax:.5f}")
    print(f"Ishlatilgan chegara eps_thr                     : {EPS_THRESHOLD}")
    print(f"Zaxira (necha barobar past)                     : "
          f"{EPS_THRESHOLD / allmax:.0f}x")
    print(f"Har qanday kanal bera oladigan eng katta eps    : {ceil:.5f}")
    print(f"  (sin = 1 bo'lganda ham; ya'ni chegara {EPS_THRESHOLD} da "
          f"HECH QACHON ishlay olmaydi)" if ceil < EPS_THRESHOLD else "")
    print("=" * 74)
    json.dump({"eps_thr": EPS_THRESHOLD, "rows": rows,
               "max_eps_observed": allmax,
               "max_eps_possible": ceil}, open(OUT_JSON, "w"), indent=2)
    print(f"saqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
