"""Operator-level comparison against GPTQ and AWQ.

The evaluation so far compared the proposed quantizer only against library
min/max rounding, which is a 2018-era baseline. This adds the two published
post-training methods in the same class -- weight-only, no retraining --
so the comparison is against current practice rather than against a straw
man.

All four methods quantize to symmetric INT8 with per-output-channel scales,
so what is being compared is the ALGORITHM, not bit width or granularity:

    RTN    round-to-nearest, no calibration
    GPTQ   Hessian-guided sequential quantization with error compensation
    AWQ    activation-aware input-channel rescaling before quantization
    ours   alternating minimization + calibration-refined per-channel scale

GPTQ and AWQ are reimplementations (nnopt.quantizer.baselines); the official
packages require CUDA and cannot run here. This is stated wherever the
numbers are used.

Fitting uses calibration rows; every error is reported on held-out rows.
"""

import json
import time

import numpy as np
import onnx

from calib_utils import (
    DECODER_PATH,
    ENCODER_PATH,
    capture_activations,
    decoder_feeds,
    encoder_feeds,
    weight_for_operator,
    weighted_matmul_profiles,
)
from nnopt.quantizer.baselines import awq_quantize, gptq_quantize, output_relative_error, rtn_quantize
from nnopt.quantizer.per_channel import quantize_weight_per_channel

N_CALIB = 8
MAX_ROWS = 4096
FIT_FRACTION = 0.75
Q8 = 127
OUT_JSON = "experiments/results_vs_gptq_awq.json"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
DEC_DIMS = {"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500}
ENC_LAYERS = [0, 6, 12, 18, 23]
DEC_LAYERS = [0, 12, 23]


def collect(path, dims, layers, feeds_fn, tag):
    profs = weighted_matmul_profiles(path, dims)
    ops = [p for p in profs if any(f"/layers.{li}/" in p.name for li in layers)]
    tensors = sorted({p.activation_input for p in ops
                      if p.activation_input != "encoder_hidden_states"})
    feeds = feeds_fn(0, N_CALIB)
    x_by = capture_activations(path, tensors, feeds, max_rows=MAX_ROWS)
    if any(p.activation_input == "encoder_hidden_states" for p in ops):
        x_by["encoder_hidden_states"] = np.concatenate(
            [f["encoder_hidden_states"].reshape(-1, 1024) for f in feeds], axis=0
        )[:MAX_ROWS].astype(np.float64)
    model = onnx.load(path)
    inits = {i.name: i for i in model.graph.initializer}
    return ops, x_by, model, inits, tag


def run(path, dims, layers, feeds_fn, tag, rows):
    print(f"\n=== {tag} ===", flush=True)
    ops, x_by, model, inits, _ = collect(path, dims, layers, feeds_fn, tag)
    print(f"{len(ops)} operator, kalibrlash yig'ildi")
    print(f"{'operator':40s} {'RTN':>9s} {'GPTQ':>9s} {'AWQ':>9s} {'bizniki':>9s} {'s':>5s}")

    for p in ops:
        x = x_by.get(p.activation_input)
        if x is None:
            continue
        w = weight_for_operator(model, inits, p, x)
        if w.shape[1] != x.shape[1]:
            continue
        split = int(x.shape[0] * FIT_FRACTION)
        x_fit, x_eval = x[:split], x[split:]
        t0 = time.time()

        e = {}
        e["rtn"] = output_relative_error(w, rtn_quantize(w, Q8), x_eval)
        e["gptq"] = output_relative_error(w, gptq_quantize(w, x_fit, Q8), x_eval)
        w_awq, alpha = awq_quantize(w, x_fit, Q8)
        e["awq"] = output_relative_error(w, w_awq, x_eval)
        w_ours, _ = quantize_weight_per_channel(w, Q8, x_calib=x_fit)
        e["ours"] = output_relative_error(w, w_ours, x_eval)

        rows.append({"model": tag, "op": p.name, "m": int(w.shape[0]),
                     "n": int(w.shape[1]), "alpha_awq": float(alpha), **e})
        short = p.name.replace("/model/encoder/", "").replace("/model/decoder/", "")
        print(f"{short[:40]:40s} {e['rtn']:9.5f} {e['gptq']:9.5f} {e['awq']:9.5f} "
              f"{e['ours']:9.5f} {time.time()-t0:5.0f}", flush=True)
        del w, x_fit, x_eval

    del x_by, model
    return rows


def summarize(rows):
    methods = ["rtn", "gptq", "awq", "ours"]
    print("\n" + "=" * 88)
    print("OPERATOR DARAJASIDA TAQQOSLASH (held-out E_loc, INT8 per-channel)")
    print("=" * 88)
    for tag in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == tag]
        print(f"\n{tag} ({len(rs)} operator)")
        print(f"  {'usul':10s} {'o''rtacha':>10s} {'mediana':>10s} {'maks':>10s} "
              f"{'RTN ga nisbatan':>16s}")
        base = np.mean([r["rtn"] for r in rs])
        for meth in methods:
            v = np.array([r[meth] for r in rs])
            gain = (base - v.mean()) / base * 100
            print(f"  {meth:10s} {v.mean():10.5f} {np.median(v):10.5f} {v.max():10.5f} "
                  f"{gain:15.1f}%")

    print("\n" + "=" * 88)
    print("JUFTLIK G'ALABALAR (operator darajasida)")
    print("=" * 88)
    for a, b in [("ours", "rtn"), ("ours", "gptq"), ("ours", "awq"),
                 ("gptq", "rtn"), ("awq", "rtn")]:
        wins = sum(1 for r in rows if r[a] < r[b])
        print(f"  {a:6s} > {b:6s} : {wins:3d}/{len(rows)}")

    alphas = [r["alpha_awq"] for r in rows]
    print(f"\nAWQ tanlagan alpha: o'rtacha {np.mean(alphas):.2f}, "
          f"diapazon [{min(alphas):.2f}, {max(alphas):.2f}]")


def main():
    rows = []
    run(ENCODER_PATH, ENC_DIMS, ENC_LAYERS, encoder_feeds, "Whisper encoder", rows)
    run(DECODER_PATH, DEC_DIMS, DEC_LAYERS, decoder_feeds, "Whisper decoder", rows)
    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")
    summarize(rows)


if __name__ == "__main__":
    main()
