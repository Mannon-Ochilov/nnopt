"""WHY does the magnitude criterion collapse? (turning a report into a cause)

The paper reports that magnitude pruning collapses at the aggressive budget
(WER 2.74 with our layer allocation, 0.42 with its own) while both
calibration-using criteria degrade smoothly. Reported, but not explained --
"activation-blind" names the difference, not the mechanism.

Hypothesis, stated before measuring: on this encoder the column norm of the
downstream weight is essentially UNCORRELATED with the channel's functional
contribution ||w_j||*||h_j||, because activation energy varies over orders
of magnitude across channels. If so, magnitude ranking removes a fraction
of HIGH-contribution channels roughly at chance rate, and at aggressive
budgets a few of those are catastrophic. The measurable signatures:

  1. Spearman correlation between ||W2[:,j]|| and ||h_j|| is weak.
  2. The overlap between magnitude's removal set and the criterion's removal
     set is near the chance level.
  3. Among the channels magnitude removes, the top percentile of functional
     contribution ||w_j||*||h_j|| is orders of magnitude above the largest
     contribution our criterion removes -- these are the channels whose loss
     is unrecoverable without compensation.

All three are cheap operator-level statistics on already-cached activations.
"""

import json

import numpy as np
import onnx
from onnx import numpy_helper
from scipy.stats import spearmanr

from calib_utils import (
    ENCODER_PATH,
    CalibSet,
    capture_activations,
    feeds_for,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import layer_of

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
LAYERS = (0, 2, 5, 8, 16, 23)
REMOVAL = 0.24                     # the tau=0.95 budget where magnitude broke
FIT_ROWS = 3072
OUT_JSON = "experiments/results_magnitude_cause.json"


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
        w2 = numpy_helper.to_array(inits[p2.weight_initializer]) \
            .astype(np.float64)
        if w2.shape[0] != x.shape[1]:
            w2 = w2.T                      # (F, d) -> rows index channels
        wn = np.linalg.norm(w2, axis=1)    # ||W2 row_j|| per channel
        hn = np.linalg.norm(x, axis=0)     # ||h_j||
        contrib = wn * hn                  # functional contribution proxy

        n = len(wn)
        k = int(round(n * REMOVAL))
        rm_mag = set(np.argsort(wn)[:k])          # magnitude removes small ||w||
        rm_ctr = set(np.argsort(contrib)[:k])     # contribution-aware removal

        rho, _ = spearmanr(wn, hn)
        overlap = len(rm_mag & rm_ctr) / k
        # The damage signature: the largest functional contribution each
        # ranking is willing to destroy.
        worst_mag = float(np.max(contrib[list(rm_mag)]))
        worst_ctr = float(np.max(contrib[list(rm_ctr)]))
        med = float(np.median(contrib))
        rows.append({"layer": li, "spearman_wn_hn": float(rho),
                     "overlap": float(overlap), "chance": REMOVAL,
                     "worst_removed_mag": worst_mag,
                     "worst_removed_ctr": worst_ctr,
                     "median_contrib": med})
        print(f"L{li:<2d} Spearman(||w||,||h||) = {rho:+.3f}   "
              f"kesishma {overlap*100:5.1f}% (tasodif {REMOVAL*100:.0f}%)   "
              f"eng katta yo'qotilgan hissa: mag {worst_mag/med:7.1f}x med, "
              f"mezon {worst_ctr/med:5.1f}x med", flush=True)

    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
