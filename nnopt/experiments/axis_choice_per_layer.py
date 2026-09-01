"""Which axis should each layer be reduced along -- channels or rank?

Building the L3 = 12 MiB encoder exposed something the cascade has never had
to confront. A per-layer cache target asks the same 45% of every layer, but
the redundancy our criterion detects is concentrated in the early ones: tau
stayed near 0.99 through layer 4 and had fallen to 0.30 by layer 19. Below
some tau the grouping is no longer merging collinear responses; it is ranking
which non-collinear channel to sacrifice, and the budget is met by force
rather than by structure.

That is a statement about ONE kind of redundancy. Functional grouping exploits
collinearity between channel responses. Low-rank factorization exploits
spectral decay, which is a different property of the same matrix, and nothing
measured so far says the deep layers lack it -- only that they lack the first
kind. If they have the second, the right per-layer decision is not "how far do
I lower tau" but "which axis does this layer actually offer".

Existing evidence points the other way and is worth stating before measuring:
whole-encoder low-rank scored 0.3056 at 192 MiB against 0.1833 for channel
removal at 267 MiB (Sec 4.9). But that comparison applied one axis uniformly
across all layers at an unmatched budget, so it cannot answer the question
here, which is per-layer and parameter-matched.

The comparison is on fc2, at equal parameter count, on HELD-OUT activations:

  channels   W2[:, keep] with compensation folded in     -> 1024 * k params
  rank       activation-aware low-rank of W2 at rank r   -> r * (1024 + 4096)

so k = 2253 (the 45% budget) pairs with r = 450. Held-out matters here more
than usual: at r = 450 against the rows available, the row/rank ratio is near
the floor where calibration-fitted low-rank starts memorising (Sec 4.6), and a
fit-set comparison would flatter the rank arm precisely where it is weakest.
"""

import gc
import json
import os
import time

import numpy as np
import onnx
from onnx import numpy_helper

from calib_utils import (
    ENCODER_PATH,
    CalibSet,
    capture_activations,
    encoder_feeds,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import layer_of
from l3_12_cascade import DEFAULT_CALIB, map_path
from nnopt.cur.lowrank_baselines import activation_aware_svd

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_FIT, N_HOLDOUT = 6, 6          # utterances; fit set matches the built maps
FIT_ROWS = 8192
N_CHANNELS = 4096
D_MODEL = 1024
OUT_JSON = "experiments/results_axis_choice.json"


def e_loc(y_ref, y_hat):
    return float(np.linalg.norm(y_ref - y_hat) / (np.linalg.norm(y_ref) + 1e-12))


def main():
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}

    layers = sorted(li for li in fc2 if os.path.exists(map_path(li)))
    if not layers:
        raise SystemExit("45% xaritalari topilmadi; avval l3_12_cascade.py")
    print(f"{len(layers)} qatlam, teng parametr byudjetida taqqoslash "
          f"(fit {N_FIT} namuna, held-out {N_HOLDOUT})\n")

    feeds_fit = encoder_feeds(0, N_FIT)
    feeds_ho = encoder_feeds(N_FIT, N_HOLDOUT)

    # Each layer costs about a minute of activation capture, so a restart
    # should not redo finished ones.
    rows = []
    if os.path.exists(OUT_JSON):
        try:
            rows = json.load(open(OUT_JSON, encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            rows = []
    done = {r["layer"] for r in rows}
    if done:
        print(f"keshdan {len(done)} qatlam\n")

    t0 = time.time()
    for li in layers:
        if li in done:
            continue
        p2 = fc2[li]
        x_fit = capture_activations(ENCODER_PATH, [p2.activation_input],
                                    feeds_fit, max_rows=FIT_ROWS)[p2.activation_input]
        x_ho = capture_activations(ENCODER_PATH, [p2.activation_input],
                                   feeds_ho, max_rows=FIT_ROWS)[p2.activation_input]
        w2s = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w2 = w2s if w2s.shape[1] == x_fit.shape[1] else w2s.T   # (1024, 4096)

        z = np.load(map_path(li), allow_pickle=True)
        keep = z["keep"]
        w2_kept = z["w2"].T                                     # (1024, k)
        k = len(keep)
        rank = int(round(D_MODEL * k / (D_MODEL + N_CHANNELS)))

        y_ho = x_ho @ w2.T
        y_chan = x_ho[:, keep] @ w2_kept.T

        lr = activation_aware_svd(w2, x_fit, rank)
        y_rank = x_ho @ lr.T

        r_chan, r_rank = e_loc(y_ho, y_chan), e_loc(y_ho, y_rank)
        tau = float(z["tau"])
        rows.append({"layer": li, "tau": tau, "k": int(k), "rank": rank,
                     "e_channels": r_chan, "e_rank": r_rank,
                     "rows": int(x_fit.shape[0]),
                     "row_rank_ratio": x_fit.shape[0] / rank})
        win = "kanal" if r_chan < r_rank else "RANK"
        print(f"  L{li:<2d} tau={tau:5.2f}  k={k}  rank={rank}  "
              f"E_kanal={r_chan:.4f}  E_rank={r_rank:.4f}  -> {win} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
        del x_fit, x_ho, w2, lr, y_ho, y_chan, y_rank
        gc.collect()

    print("\n" + "=" * 78)
    print("QAYSI O'Q QAYSI QATLAMDA (teng parametr, held-out E_loc)")
    print("=" * 78)
    rank_wins = [r for r in rows if r["e_rank"] < r["e_channels"]]
    print(f"rank yutgan qatlamlar: {len(rank_wins)}/{len(rows)}")
    if rank_wins:
        print("  " + ", ".join(f"L{r['layer']}" for r in rank_wins))
    deep = [r for r in rows if r["tau"] <= 0.60]
    if deep:
        dr = sum(1 for r in deep if r["e_rank"] < r["e_channels"])
        print(f"\nMezon tugagan qatlamlarda (tau <= 0.60): "
              f"rank {dr}/{len(deep)} tasida yutdi")
    print(f"\nqator/rank nisbati: {min(r['row_rank_ratio'] for r in rows):.1f} "
          f"— 4.6-bo'limdagi 10-20 talabidan pastda bo'lsa, rank armi "
          f"kalibrlashni yodlab olayotgan bo'lishi mumkin.")


if __name__ == "__main__":
    main()
