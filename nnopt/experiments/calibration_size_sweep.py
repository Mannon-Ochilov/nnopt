"""How much calibration data does activation-aware SVD actually need?

Motivation (a real bug found in whole_encoder_lowrank.py): that script fit
the factorization on `x` and then reported E_loc on the SAME `x`, with only
512 calibration rows for ranks up to 409. When rank approaches the number
of calibration rows, the activation-aware solution can essentially
interpolate them: the Gram matrix X^T X has rank <= n_rows, so a rank-409
fit against 512 rows is nearly unconstrained. The reported 0.0033 was
therefore training error, and the end-to-end WER blow-up (0.067 -> 0.293)
is consistent with that, not with a genuine failure of the method.

This sweeps calibration size against rank and reports BOTH errors, so the
overfitting gap is visible and a safe rows/rank ratio can be chosen.

Encoder gives 1500 positions per utterance, so thousands of rows are cheap
-- the 512 cap was an arbitrary carry-over from the decoder scripts, where
each utterance yields only ~16 token positions.
"""

import json

import numpy as np
import onnx

from calib_utils import ENCODER_PATH, capture_activations, encoder_feeds
from nnopt.cur.lowrank_baselines import activation_aware_svd, output_relative_error, truncated_svd
from nnopt.profiler.graph_profiler import profile_onnx_model

N_CALIB_UTT = 12
EVAL_ROWS = 4096
FIT_SIZES = [256, 512, 1024, 2048, 4096, 8192]
RANKS = [409, 200, 128]
TARGETS = ["/layers.0/fc1/", "/layers.0/fc2/"]
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
OUT_JSON = "experiments/results_calib_sweep.json"


def main():
    profs = profile_onnx_model(ENCODER_PATH, free_dims=ENC_DIMS)
    targets = [next(p for p in profs if s in p.name and p.weight_initializer) for s in TARGETS]
    tensors = sorted({p.activation_input for p in targets})

    print(f"{N_CALIB_UTT} ta namuna x 1500 pozitsiya = {N_CALIB_UTT*1500} qator mavjud")
    feeds = encoder_feeds(0, N_CALIB_UTT)
    need = max(FIT_SIZES) + EVAL_ROWS
    x_by = capture_activations(ENCODER_PATH, tensors, feeds, max_rows=need)

    enc = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in enc.graph.initializer}

    rows = []
    for p in targets:
        x = x_by[p.activation_input]
        w = onnx.numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        m, n = w.shape
        x_eval = x[-EVAL_ROWS:]
        print(f"\n=== {p.name}  W={m}x{n},  jami {x.shape[0]} qator, eval {x_eval.shape[0]} ===")
        print(f"{'rank':>6s} {'fit qator':>10s} {'qator/rank':>11s} {'fit E_loc':>11s} "
              f"{'held-out':>11s} {'bo''shliq':>9s} {'plain SVD':>10s}")

        for rank in RANKS:
            e_plain = output_relative_error(w, truncated_svd(w, rank), x_eval)
            for k in FIT_SIZES:
                if k + EVAL_ROWS > x.shape[0]:
                    continue
                x_fit = x[:k]
                approx = activation_aware_svd(w, x_fit, rank)
                e_fit = output_relative_error(w, approx, x_fit)
                e_eval = output_relative_error(w, approx, x_eval)
                gap = e_eval / (e_fit + 1e-12)
                rows.append({"op": p.name, "rank": rank, "fit_rows": k,
                             "eloc_fit": e_fit, "eloc_eval": e_eval,
                             "ratio_rows_rank": k / rank, "eloc_plain_svd": e_plain})
                print(f"{rank:6d} {k:10d} {k/rank:11.1f} {e_fit:11.5f} {e_eval:11.5f} "
                      f"{gap:9.1f}x {e_plain:10.5f}")

    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")

    print("\n" + "=" * 78)
    print("XULOSA: xavfsiz qator/rank nisbati")
    print("=" * 78)
    for rank in RANKS:
        rs = [r for r in rows if r["rank"] == rank]
        if not rs:
            continue
        best = min(rs, key=lambda z: z["eloc_eval"])
        print(f"  rank={rank:4d}: eng yaxshi held-out E_loc={best['eloc_eval']:.5f} "
              f"({best['fit_rows']} qator, nisbat {best['ratio_rows_rank']:.1f}x)")


if __name__ == "__main__":
    main()
