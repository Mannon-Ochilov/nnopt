"""How much does the calibration set decide, and was it being read correctly?

Two defects were found in the calibration path and this measures both.

1. Padding was never masked. capture_activations passed active_mask=None,
   which build_response_vectors' own docstring forbids for padded input.
   Whisper pads every clip to a 30 s window (1500 encoder positions) and the
   Common Voice uz clips average 5.1 s, so about 83% of the positions feeding
   h_j were silence the feature extractor invented. Padding responses are
   near-constant across positions, so they should inflate apparent
   collinearity -- two channels look alike because they agree about silence.
   That biases the grouping toward merges the speech does not justify.

2. Six utterances is thin. 6 x 1500 = 9000 positions before masking, but only
   ~1530 after, and FIT_ROWS subsamples to 3072. An FFN layer has 4096
   channels, so the Gram matrix is estimated from fewer samples than it has
   dimensions -- rank-deficient by construction. GPTQ and AWQ calibrate on
   128 sequences; this work used 6.

The experiment holds tau fixed and varies only the calibration, so any change
in the number of accepted merges is attributable to the calibration itself:

    unmasked n=6      what every published number in the paper was built from
    masked   n=6      same audio, padding removed -- isolates defect 1
    masked   n=12,24,48   more audio -- isolates defect 2

If the merge counts move a lot, the paper's operating point was an artifact of
its calibration set and the affected numbers must be rebuilt. If they barely
move, the same measurement becomes a robustness result: the decisions do not
depend on a calibration choice a reviewer could otherwise call arbitrary.

Layers are sampled across depth rather than swept, because the grouping pass
is the expensive part and five layers spanning the encoder answer the
question. All tensors are captured in one pass per configuration, so the cost
is one encoder forward per utterance, not one per layer.

Usage:  python experiments/calib_sensitivity.py
"""

import gc
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper

from calib_utils import (
    ENCODER_PATH,
    CalibSet,
    capture_activations,
    encoder_positions_for,
    feeds_for,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import layer_of
from nnopt.grouping.functional_grouping import greedy_group

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
LAYERS = (0, 5, 11, 17, 23)
TAUS = (0.99, 0.95, 0.90)
EPS_THRESHOLD = 0.5
# Above the row count any configuration can reach, so the number of rows is
# set by the audio rather than silently clipped -- otherwise the n sweep would
# compare identical 3072-row matrices and measure nothing.
FIT_ROWS = 16384
CONFIGS = [
    ("niqobsiz n=6  (maqoladagi)", CalibSet(n=6, masked=False)),
    ("niqobli  n=6", CalibSet(n=6)),
    ("niqobli  n=12", CalibSet(n=12)),
    ("niqobli  n=24", CalibSet(n=24)),
    ("niqobli  n=48", CalibSet(n=48)),
]
OUT_JSON = "experiments/results_calib_sensitivity.json"


def main():
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    names = [fc2[li].activation_input for li in LAYERS]

    rows = []
    for label, calib in CONFIGS:
        pos = encoder_positions_for(calib)
        print(f"\n=== {label} ===")
        print(f"  {calib.n} yozuv, real pozitsiyalar: {sum(pos)} "
              f"(1500 x {calib.n} = {1500 * calib.n} dan)", flush=True)

        x_by = capture_activations(
            ENCODER_PATH, names, feeds_for(calib), max_rows=FIT_ROWS,
            active_positions=pos if calib.masked else None)

        for li in LAYERS:
            p2 = fc2[li]
            x = x_by[p2.activation_input]
            w2s = numpy_helper.to_array(inits[p2.weight_initializer]) \
                .astype(np.float64)
            w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
            h = x.T
            wn = np.linalg.norm(w2, axis=0)
            y_norm = float(np.linalg.norm(x @ w2.T))

            for tau in TAUS:
                g = greedy_group(h, wn, y_norm, tau=tau,
                                 eps_threshold=EPS_THRESHOLD)
                kept = len(g.groups)
                merged = h.shape[0] - kept
                rows.append({"config": label, "masked": calib.masked,
                             "n": calib.n, "real_positions": int(sum(pos)),
                             "rows": int(x.shape[0]), "layer": li, "tau": tau,
                             "channels": int(h.shape[0]), "kept": int(kept),
                             "merged": int(merged),
                             "removal": merged / h.shape[0]})
                print(f"    L{li:<2d} tau={tau:.2f}  birlashma {merged:5d}  "
                      f"({merged / h.shape[0] * 100:5.2f}%)  qatorlar "
                      f"{x.shape[0]}", flush=True)
            del x, w2, h
            gc.collect()
        del x_by
        gc.collect()
        json.dump(rows, open(OUT_JSON, "w"), indent=2)

    report(rows)


def report(rows):
    labels = [c[0] for c in CONFIGS]
    print("\n" + "=" * 96)
    print("KALIBRLASH SEZGIRLIGI -- tau qat'iy, faqat kalibrlash o'zgaradi")
    print("=" * 96)
    for tau in TAUS:
        print(f"\ntau = {tau}   (olib tashlangan kanallar ulushi, %)")
        print(f"  {'qatlam':>7s}" + "".join(f"{l:>22s}" for l in labels))
        for li in LAYERS:
            cells = ""
            for lab in labels:
                r = next((r for r in rows if r["config"] == lab
                          and r["layer"] == li and r["tau"] == tau), None)
                cells += f"{r['removal'] * 100:21.2f}%" if r else " " * 22
            print(f"  {'L' + str(li):>7s}{cells}")
        means = []
        for lab in labels:
            sel = [r["removal"] for r in rows
                   if r["config"] == lab and r["tau"] == tau]
            means.append(sum(sel) / len(sel) if sel else float("nan"))
        print(f"  {'o‘rtacha':>7s}" + "".join(f"{m * 100:21.2f}%" for m in means))

    # The two questions, answered as ratios against the paper's own setting.
    base = labels[0]
    print("\n" + "=" * 96)
    print("XULOSA")
    print("=" * 96)
    for tau in TAUS:
        b = [r["removal"] for r in rows
             if r["config"] == base and r["tau"] == tau]
        m6 = [r["removal"] for r in rows
              if r["config"] == labels[1] and r["tau"] == tau]
        if not b or not m6:
            continue
        bm, mm = sum(b) / len(b), sum(m6) / len(m6)
        print(f"tau={tau}: padding niqobi olib tashlanganda birlashmalar "
              f"{bm * 100:.2f}% -> {mm * 100:.2f}% "
              f"({'kamaydi' if mm < bm else 'oshdi'}, "
              f"{abs(mm - bm) * 100:.2f} f.p.)")
        seq = [(lab, sum(v) / len(v)) for lab in labels[1:]
               if (v := [r["removal"] for r in rows
                         if r["config"] == lab and r["tau"] == tau])]
        if len(seq) > 1:
            vals = [v for _, v in seq]
            print(f"          n 6->48 bo'ylab tarqoqlik: "
                  f"{(max(vals) - min(vals)) * 100:.2f} f.p. "
                  f"(min {min(vals) * 100:.2f}%, maks {max(vals) * 100:.2f}%)")
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
