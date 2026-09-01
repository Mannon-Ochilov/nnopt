"""The padding effect across every encoder layer, not a five-layer sample.

calib_sensitivity.py established two things on five layers: masking padding
out of the calibration changes the merge counts a great deal, and the number
of calibration utterances barely matters at all. The size question is settled
and is not repeated here. What the sample could not give is the shape of the
distortion over depth -- and the sample hinted that shape is the interesting
part, because the ratio between unmasked and masked grew from ~1.5x in the
early layers to 6.45x at L11.

That matters for the cascade specifically. Removal is monotone in tau, so
reaching a 45% budget means bisecting tau downward, and the distortion is
worst exactly where the bisection ends up: low tau, deep layers. A per-layer
table is what shows whether that is a local quirk of L11 or a trend.

Only tau = 0.95 and 0.90 are run. tau = 0.99 is already known to be the mildest
case (average ratio 1.31x) and each pass on the unmasked matrix costs ~24 s,
so the two taus that actually bracket the operating region are worth more than
a third point at the top.

Layers are captured in chunks: holding all 24 activation matrices at 9000 rows
would need ~7 GB, and there is no reason to hold more than one chunk at a time.

Usage:  python experiments/calib_padding_all_layers.py
"""

import gc
import json

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
TAUS = (0.95, 0.90)
EPS_THRESHOLD = 0.5
FIT_ROWS = 16384          # above any configuration's row count: never clips
CHUNK = 6                 # layers captured at once; ~1.8 GB at 9000 rows
CONFIGS = [("niqobsiz", CalibSet(n=6, masked=False)),
           ("niqobli", CalibSet(n=6))]
OUT_JSON = "experiments/results_calib_padding_all_layers.json"


def main():
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    layers = sorted(fc2)
    print(f"{len(layers)} qatlam, tau {TAUS}\n")

    rows = []
    for label, calib in CONFIGS:
        pos = encoder_positions_for(calib)
        print(f"=== {label} (n={calib.n}, real pozitsiya {sum(pos)}) ===",
              flush=True)
        for start in range(0, len(layers), CHUNK):
            chunk = layers[start:start + CHUNK]
            names = [fc2[li].activation_input for li in chunk]
            x_by = capture_activations(
                ENCODER_PATH, names, feeds_for(calib), max_rows=FIT_ROWS,
                active_positions=pos if calib.masked else None)
            for li in chunk:
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
                    merged = h.shape[0] - len(g.groups)
                    rows.append({"config": label, "layer": li, "tau": tau,
                                 "channels": int(h.shape[0]),
                                 "merged": int(merged),
                                 "share": merged / h.shape[0],
                                 "rows": int(x.shape[0])})
                    print(f"  L{li:<2d} tau={tau:.2f}  {merged:5d}  "
                          f"({merged / h.shape[0] * 100:5.2f}%)", flush=True)
                del x, w2, h
                gc.collect()
            del x_by
            gc.collect()
            json.dump(rows, open(OUT_JSON, "w"), indent=2)
    report(rows, layers)


def report(rows, layers):
    def get(cfg, li, tau):
        r = next((r for r in rows if r["config"] == cfg and r["layer"] == li
                  and r["tau"] == tau), None)
        return r["share"] * 100 if r else None

    for tau in TAUS:
        print("\n" + "=" * 62)
        print(f"BIRLASHTIRILGAN KANALLAR ULUSHI, %   (tau = {tau})")
        print("=" * 62)
        print(f"{'Qatlam':>7s} {'Niqobsiz':>10s} {'Niqobli':>10s} "
              f"{'Farq':>9s} {'Nisbat':>9s}")
        print("-" * 62)
        acc = [[], []]
        for li in layers:
            a, b = get("niqobsiz", li, tau), get("niqobli", li, tau)
            if a is None or b is None:
                continue
            acc[0].append(a)
            acc[1].append(b)
            ratio = f"{a / b:8.2f}x" if b > 0 else ("       —" if a == 0
                                                   else "       ∞")
            print(f"{'L' + str(li):>7s} {a:9.2f}% {b:9.2f}% "
                  f"{b - a:8.2f} {ratio}")
        if acc[0]:
            ma = sum(acc[0]) / len(acc[0])
            mb = sum(acc[1]) / len(acc[1])
            print("-" * 62)
            print(f"{'O‘rtacha':>7s} {ma:9.2f}% {mb:9.2f}% "
                  f"{mb - ma:8.2f} {ma / mb if mb else float('nan'):8.2f}x")
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
