"""What a single global tau removes, across the whole encoder.

The cascade currently bisects tau PER LAYER to hit a uniform 45% removal in
every layer. That inverts the method's own claim: the claim is that tau
selects functionally redundant channels, so tau should be the fixed criterion
and the removal share should be whatever the geometry of each layer allows.
Forcing a uniform share does the opposite -- it fixes the share and lets tau
drift down until the layer yields it, which in a layer with no redundancy
means merging channels that are not similar at all.

The five-layer sample showed how uneven the geometry is once padding is
masked out of the calibration: at tau = 0.90, L0 offers 52% and L17 offers
0%. A uniform budget cannot be right against a distribution like that.

This measures the alternative directly. One tau is applied to every layer,
and the question is what the ENCODER as a whole then gives up -- the global
removal share, which is what a byte budget actually cares about. Three taus
bracket the usable region on masked data; the paper's old rungs (0.99/0.95/
0.90) were calibrated against padding-inflated similarity and sit too high,
since masked tau = 0.95 reproduces what unmasked tau = 0.99 used to report.

Masked only: the unmasked comparison is already done, and its 9000-row
matrices cost five times as much per grouping pass.

Usage:  python experiments/global_tau_range.py
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
# Extended downward rather than made finer. Removal is monotone in tau and the
# masked data is smooth between rungs, so extra points between 0.99 and 0.90
# would resolve nothing; the open question is how far down tau has to go before
# the encoder gives up a budget-sized share, and whether a tau that low still
# means "similar" in any useful sense. Bisection remains the right tool for
# PICKING an operating tau -- this grid is for characterising the curve.
TAUS = (0.99, 0.95, 0.90, 0.85, 0.80, 0.75)
EPS_THRESHOLD = 0.5
FIT_ROWS = 16384
CHUNK = 6
# 48 utterances -> 11790 real positions, comfortably above the 4096 channels
# of an FFN layer, so the Gram matrix is full-rank rather than estimated from
# fewer samples than it has dimensions. calib_sensitivity.py showed the merge
# counts barely move between n=6 and n=48 (<=0.64 f.p.), so this buys
# defensibility rather than a different answer.
CALIB = CalibSet(n=48)         # masked
OUT_JSON = "experiments/results_global_tau_range.json"


def main():
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    layers = sorted(fc2)
    pos = encoder_positions_for(CALIB)
    print(f"{len(layers)} qatlam, niqobli n={CALIB.n} "
          f"({sum(pos)} real pozitsiya), tau {TAUS}\n", flush=True)

    rows = []
    for start in range(0, len(layers), CHUNK):
        chunk = layers[start:start + CHUNK]
        x_by = capture_activations(
            ENCODER_PATH, [fc2[li].activation_input for li in chunk],
            feeds_for(CALIB), max_rows=FIT_ROWS, active_positions=pos)
        for li in chunk:
            p2 = fc2[li]
            x = x_by[p2.activation_input]
            w2s = numpy_helper.to_array(inits[p2.weight_initializer]) \
                .astype(np.float64)
            w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
            h = x.T
            wn = np.linalg.norm(w2, axis=0)
            y_norm = float(np.linalg.norm(x @ w2.T))
            line = f"  L{li:<2d}"
            for tau in TAUS:
                g = greedy_group(h, wn, y_norm, tau=tau,
                                 eps_threshold=EPS_THRESHOLD)
                merged = h.shape[0] - len(g.groups)
                rows.append({"layer": li, "tau": tau,
                             "channels": int(h.shape[0]),
                             "merged": int(merged),
                             "share": merged / h.shape[0]})
                line += f"   tau={tau}: {merged:5d} ({merged/h.shape[0]*100:5.2f}%)"
            print(line, flush=True)
            del x, w2, h
            gc.collect()
        del x_by
        gc.collect()
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
    report(rows, layers)


def report(rows, layers):
    print("\n" + "=" * 74)
    print("BITTA GLOBAL TAU: QATLAMLAR BO'YICHA OLIB TASHLASH, %")
    print("=" * 74)
    print(f"{'Qatlam':>7s}" + "".join(f"{'tau=' + str(t):>14s}" for t in TAUS))
    print("-" * 74)
    for li in layers:
        cells = ""
        for tau in TAUS:
            r = next((r for r in rows if r["layer"] == li and r["tau"] == tau),
                     None)
            cells += f"{r['share'] * 100:13.2f}%" if r else " " * 14
        print(f"{'L' + str(li):>7s}{cells}")

    print("-" * 74)
    # The global share is the only number a byte budget can use: layers differ
    # in what they can give, so the per-layer mean is not what the encoder
    # actually sheds.
    for name, fn in (("O‘rtacha", None), ("GLOBAL", None)):
        cells = ""
        for tau in TAUS:
            sel = [r for r in rows if r["tau"] == tau]
            if not sel:
                cells += " " * 14
                continue
            if name == "GLOBAL":
                v = sum(r["merged"] for r in sel) / sum(r["channels"]
                                                        for r in sel)
            else:
                v = sum(r["share"] for r in sel) / len(sel)
            cells += f"{v * 100:13.2f}%"
        print(f"{name:>7s}{cells}")

    print("\n" + "=" * 74)
    print("TAQSIMOT — bir xil byudjet nega noto'g'ri")
    print("=" * 74)
    for tau in TAUS:
        sel = sorted((r for r in rows if r["tau"] == tau),
                     key=lambda r: -r["share"])
        if not sel:
            continue
        nz = [r for r in sel if r["merged"] > 0]
        top = sum(r["merged"] for r in sel[:5])
        tot = sum(r["merged"] for r in sel)
        print(f"tau={tau}: {len(nz)}/{len(sel)} qatlam biror narsa beradi; "
              f"eng yuqori 5 qatlam jami olib tashlashning "
              f"{top / tot * 100:.1f}% ini beradi" if tot else
              f"tau={tau}: hech bir qatlam hech narsa bermaydi")
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
