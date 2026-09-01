"""Does dropping the sign from the criterion change anything on Whisper?

The gate in greedy_group is `cos >= tau`, which refuses a pair with
cos = -0.95 even though it is as mergeable as +0.95: gamma may be negative
and build_compensated_weight already applies a signed gamma, so only the gate
stands in the way. On open_llama_3b that refusal costs real pairs -- the
largest |cos| is 0.8488 against a largest signed cos of 0.7681, and at
tau = 0.70 the absolute criterion finds 6.5x as many channels.

Whisper should behave differently, and the reason is the same activation
geometry that explains the Llama result: fc2 reads a GELU output, which is
almost entirely non-negative, so channel vectors sit in a positive cone and
anti-collinear pairs should barely exist. If that holds, |cos| is a no-op
here and the change is safe to make unconditionally.

That is a PREDICTION, and the point of this script is that it could fail. If
Whisper does carry anti-collinear pairs, the change would alter the operating
points behind the paper's headline numbers, and it would then have to be an
option rather than a replacement.
"""

import os

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

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
FIT_ROWS = 3072
LAYERS = tuple(int(v) for v in
               os.environ.get("LAYERS", "0,8,16,23").split(","))
TAUS = (0.99, 0.95, 0.90, 0.70, 0.50)


def nearest(mat, signed):
    np.fill_diagonal(mat, -2.0 if signed else 0.0)
    return (mat if signed else np.abs(mat)).max(axis=1)


def main():
    calib = CalibSet(split="validation", skip=0, n=6)
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    feeds = feeds_for(calib)

    print(f"Whisper enkoder, kalibrlash {calib.tag}, {FIT_ROWS} qator\n")
    for li in LAYERS:
        if li not in fc2:
            print(f"  L{li}: fc2 topilmadi, o'tkazib yuborildi")
            continue
        p2 = fc2[li]
        x = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                max_rows=FIT_ROWS)[p2.activation_input][:FIT_ROWS]
        h = x.T.astype(np.float64)
        n = h.shape[0]
        frac_pos = float((x > 0).mean())
        norms = np.linalg.norm(h, axis=1)
        hn = (h / (norms[:, None] + 1e-30)).astype(np.float32)
        g = hn @ hn.T
        b_signed = nearest(g.copy(), True)
        b_abs = nearest(g, False)

        # How much of the similarity is a shared offset rather than a shared
        # DIRECTION. The activations here turned out to be ~97% negative, i.e.
        # most units sit in GELU's "off" regime and share a near-constant
        # floor. A floor common to every channel raises every cosine without
        # any functional relationship being present, so the centred version
        # says how much of the redundancy survives removing it.
        mu = h.mean(axis=1)
        centred = h - mu[:, None]
        cnorms = np.linalg.norm(centred, axis=1)
        cn = (centred / (cnorms[:, None] + 1e-30)).astype(np.float32)
        b_corr = nearest(cn @ cn.T, True)
        drop = 1.0 - (cnorms / (norms + 1e-30)).mean()

        print(f"L{li}: {n} kanal, musbat faolliklar {frac_pos*100:.1f}%, "
              f"markazlashtirish normani {drop*100:.1f}% kamaytiradi")
        print(f"  eng katta:  ishorali {b_signed.max():.4f}   "
              f"|cos| {b_abs.max():.4f}   korr {b_corr.max():.4f}")
        for t in TAUS:
            a, c = (b_signed >= t).mean(), (b_abs >= t).mean()
            r = (b_corr >= t).mean()
            flag = "  <-- FARQ" if c - a > 0.002 else ""
            print(f"  tau={t:.2f}: ishorali {a*100:6.2f}%   "
                  f"|cos| {c*100:6.2f}%   korr {r*100:6.2f}%{flag}")
        print()

    print("Bashorat: GELU chiqishi musbat konusda yotgani uchun ikkala "
          "ustun ham deyarli bir xil bo'lishi kerak. Farq chiqsa, "
          "bashorat noto'g'ri va o'zgarish Whisper ish nuqtalarini "
          "siljitadi -- u holda uni almashtirish emas, variant qilish kerak.")


if __name__ == "__main__":
    main()
