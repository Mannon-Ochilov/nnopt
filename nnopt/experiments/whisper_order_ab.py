"""Order A/B where the order argument actually lives: Whisper, 188x regime.

On mBERT the same A/B showed no order effect (+0.0009 [-0.0092, +0.0119]) --
consistent with the mechanism, because with near-zero collinearity the gammas
carry almost no mass and compensation barely changes the weights. The 188x
row-range expansion was measured on WHISPER fc2, so that is where
quantize-before-fold should hurt.

Arms (channel selection and gammas computed once, shared):

  A: fold gamma-compensation into FP32 weights -> per-channel RTN INT8
  B: per-channel RTN first (dequantized) -> fold the SAME gammas -> RTN again

Quantizer is RTN (not GPTQ) so the effect of ORDER is isolated from GPTQ's
error compensation, and the A arm reproduces the measured pipeline of the
existing prune+RTN row (0.1943 on TEST). Evaluation: TEST 300, INT8 decoder,
paired.

Pre-registered readings:
  B - A > 0 with CI excluding 0 -> order is load-bearing where compensation
      carries mass; together with the mBERT null this gives the full law
      ("order matters in proportion to the mass the compensation moves").
  B ~ A here too -> the order claim is NOT supported and both papers'
      wording must be weakened to the closure argument only.
"""

import gc
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from calib_utils import (
    ENCODER_PATH,
    CalibSet,
    capture_activations,
    feeds_for,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import bias_name_for, layer_of
from nnopt.grouping.functional_grouping import (
    build_compensated_weight,
    greedy_group,
    trim_to_budget,
)

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
FIT_ROWS = 3072
TAU = 0.99
EPS_THRESHOLD = 0.5
OUT_DIR = "models/_order_ab"
OUT_JSON = "experiments/results_whisper_order_ab.json"
Q_MAX = 127


def rtn_rows(w):
    s = np.max(np.abs(w), axis=1, keepdims=True) / Q_MAX
    s[s == 0] = 1.0
    return np.round(w / s) * s


def main():
    calib = CalibSet(split="validation", skip=0, n=6)
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    feeds = feeds_for(calib)

    # Shared per-layer decision at tau = 0.99, exactly the paper's operating
    # point; layers where nothing merges are left untouched.
    decisions = {}
    for li in sorted(fc2):
        p2 = fc2[li]
        x = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                max_rows=FIT_ROWS)[p2.activation_input]
        x = x.reshape(-1, x.shape[-1])[:FIT_ROWS].astype(np.float64)
        w2s = numpy_helper.to_array(inits[p2.weight_initializer]) \
            .astype(np.float64)
        w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
        g = greedy_group(x.T, np.linalg.norm(w2, axis=0),
                         float(np.linalg.norm(x @ w2.T)), tau=TAU,
                         eps_threshold=EPS_THRESHOLD)
        keep = np.array(sorted(gr.representative for gr in g.groups))
        if len(keep) == w2.shape[1]:
            continue
        decisions[li] = (keep, g)
        print(f"  L{li:<2d}: {w2.shape[1]} -> {len(keep)}", flush=True)
        del x
        gc.collect()

    def build_arm(tag, quant_first):
        dst = f"{OUT_DIR}/enc_orderAB_{tag}.onnx"
        if os.path.exists(dst):
            return dst
        m = onnx.load(ENCODER_PATH)
        ii = {i.name: i for i in m.graph.initializer}
        for li, (keep, g) in decisions.items():
            p1, p2 = fc1[li], fc2[li]
            w2s = numpy_helper.to_array(ii[p2.weight_initializer]) \
                .astype(np.float64)
            # orientation: (d_model, width) needed
            tr2 = w2s.shape[1] != 4096
            w2 = w2s.T if tr2 else w2s
            w1s = numpy_helper.to_array(ii[p1.weight_initializer]) \
                .astype(np.float64)
            tr1 = w1s.shape[1] != 4096
            w1 = w1s.T if tr1 else w1s

            if quant_first:
                w2 = rtn_rows(w2)
                w1 = rtn_rows(w1)
            w2c = build_compensated_weight(w2, g)
            new2, new1 = w2c[:, keep], w1[:, keep]
            ii[p2.weight_initializer].CopyFrom(numpy_helper.from_array(
                (new2.T if tr2 else new2).astype(np.float32),
                p2.weight_initializer))
            ii[p1.weight_initializer].CopyFrom(numpy_helper.from_array(
                (new1.T if tr1 else new1).astype(np.float32),
                p1.weight_initializer))
            bname = bias_name_for(m, p1)
            if bname and bname != "None":
                b = numpy_helper.to_array(ii[bname]).astype(np.float64)
                if b.shape[0] == 4096:
                    ii[bname].CopyFrom(numpy_helper.from_array(
                        b[keep].astype(np.float32), bname))
        tmp = dst.replace(".onnx", "_fp32.onnx")
        onnx.save(m, tmp)
        quantize_dynamic(tmp, dst, weight_type=QuantType.QInt8,
                         per_channel=True)
        os.remove(tmp)
        return dst

    os.makedirs(OUT_DIR, exist_ok=True)
    from cascade_runner import DEC_INT8, evaluate, paired_delta

    results = {}
    if os.path.exists(OUT_JSON):
        try:
            results = json.load(open(OUT_JSON, encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            results = {}
    for tag, qf in (("pruneFirst", False), ("quantFirst", True)):
        if tag in results:
            print(f"[{tag}] keshdan WER={results[tag]['wer']:.4f}")
            continue
        print(f"[{tag}] qurilmoqda...", flush=True)
        path = build_arm(tag, qf)
        print(f"[{tag}] baholanmoqda (test, 300)...", flush=True)
        r = evaluate(path, DEC_INT8, 300, "test")
        results[tag] = {"wer": r["wer"], "per_sample": r["per_sample_wer"]}
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        print(f"  WER={r['wer']:.4f}", flush=True)

    a = np.array(results["pruneFirst"]["per_sample"], float)
    b = np.array(results["quantFirst"]["per_sample"], float)
    d, lo, hi = paired_delta(b, a)
    print("\n" + "=" * 70)
    print(f"A kesish->kvantlash : WER {results['pruneFirst']['wer']:.4f}")
    print(f"B kvantlash->kesish : WER {results['quantFirst']['wer']:.4f}")
    print(f"farq (B - A) = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("=" * 70)
    results["delta"] = [d, lo, hi]
    json.dump(results, open(OUT_JSON, "w"), indent=2)


if __name__ == "__main__":
    main()
