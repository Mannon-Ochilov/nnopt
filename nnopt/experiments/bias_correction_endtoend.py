"""Does quantization bias correction survive to WER? (closing C-band)

Operator level it is measured: folding the constant part of the quantization
error into the existing bias vector cuts held-out output error by 3.8% for
per-channel RTN at INT8 (and 0.0% for GPTQ, whose own compensation removes
the offset already). Whether 3.8% of operator error is WORTH anything at the
task level is exactly what the absorption law (E_loc 160x -> E_glob 4x) says
it might not be, so the honest expectation going in is "no measurable WER
change" -- and this measurement exists to say so with a number rather than
an argument.

Construction: the ONNX Runtime dynamic quantizer with per_channel=True IS
symmetric per-output-channel round-to-nearest, so its dequantized weight
What is reproducible offline. The correction

    b <- b + (W - What) mean(X)

is applied to the FP32 graph BEFORE quantize_dynamic; the quantizer then
produces the same integer codes (W unchanged) while the corrected bias rides
along in FP32, which dynamic quantization keeps. Operators without a bias
initializer are skipped and counted -- inventing a bias tensor would change
the graph contract this experiment wants to hold fixed.

Both arms (plain RTN, RTN + bias correction) are evaluated on the TEST
split's 300 samples with the INT8 decoder, and compared PAIRED.
"""

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
from ffn_prune_endtoend import bias_name_for

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
OUT_DIR = "models/_rtn"
PLAIN = f"{OUT_DIR}/enc_rtn_only.onnx"
CORRECTED = f"{OUT_DIR}/enc_rtn_biascorr.onnx"
OUT_JSON = "experiments/results_bias_correction_e2e.json"
MEAN_ROWS = 3072
CHUNK = 12
Q_MAX = 127


def rtn_dequant(w, axis):
    """Symmetric per-channel RTN along `axis`, dequantized."""
    s = np.max(np.abs(w), axis=axis, keepdims=True) / Q_MAX
    s[s == 0] = 1.0
    return np.round(w / s) * s


def build_corrected():
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = [p for p in weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
             if p.weight_initializer is not None]
    feeds = feeds_for(CalibSet(split="validation", skip=100, n=6))

    done, skipped_no_bias, skipped_shape = 0, 0, 0
    for start in range(0, len(profs), CHUNK):
        group = profs[start:start + CHUNK]
        x_by = capture_activations(ENCODER_PATH,
                                   [p.activation_input for p in group],
                                   feeds, max_rows=MEAN_ROWS)
        for p in group:
            x = x_by[p.activation_input]
            mu = x.reshape(-1, x.shape[-1]).mean(axis=0).astype(np.float64)
            bname = bias_name_for(model, p)
            if not bname or bname == "None":
                skipped_no_bias += 1
                continue
            w = numpy_helper.to_array(inits[p.weight_initializer]) \
                .astype(np.float64)
            # Orientation: y = x @ W needs W as (in, out); the transposed
            # layout appears too, so it is resolved against the activation
            # width rather than assumed.
            if w.ndim != 2:
                skipped_shape += 1
                continue
            if w.shape[0] == mu.shape[0]:
                w_io, axis = w, 0            # (in, out): per-out = per-column
            elif w.shape[1] == mu.shape[0]:
                w_io, axis = w.T, 1          # stored (out, in)
            else:
                skipped_shape += 1
                continue
            what = rtn_dequant(w_io, axis=0)
            corr = mu @ (w_io - what)        # (out,)
            b = numpy_helper.to_array(inits[bname]).astype(np.float64)
            if b.shape != corr.shape:
                skipped_shape += 1
                continue
            inits[bname].CopyFrom(numpy_helper.from_array(
                (b + corr).astype(np.float32), bname))
            done += 1
        print(f"  {min(start+CHUNK, len(profs))}/{len(profs)} operator, "
              f"tuzatildi {done}", flush=True)

    print(f"jami: tuzatildi {done}, biassiz {skipped_no_bias}, "
          f"shakl mos kelmadi {skipped_shape}")
    tmp = CORRECTED.replace(".onnx", "_tmp_fp32.onnx")
    onnx.save(model, tmp)
    quantize_dynamic(tmp, CORRECTED, weight_type=QuantType.QInt8,
                     per_channel=True)
    os.remove(tmp)
    return done


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(PLAIN):
        raise SystemExit(f"{PLAIN} yo'q; avval rtn_plus_pruning.py ni yuriting")
    if not os.path.exists(CORRECTED):
        print("tuzatilgan enkoder qurilmoqda...")
        build_corrected()

    from cascade_runner import DEC_INT8, evaluate, paired_delta

    results = {}
    if os.path.exists(OUT_JSON):
        try:
            results = json.load(open(OUT_JSON, encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            results = {}

    for name, path in (("rtn", PLAIN), ("rtn+bias", CORRECTED)):
        if name in results:
            print(f"[{name}] keshdan WER={results[name]['wer']:.4f}")
            continue
        print(f"[{name}] baholanmoqda (test, 300)...", flush=True)
        r = evaluate(path, DEC_INT8, 300, "test")
        results[name] = {"wer": r["wer"], "cer": r.get("cer"),
                         "per_sample": r["per_sample_wer"]}
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        print(f"  WER={r['wer']:.4f}")

    a = np.array(results["rtn"]["per_sample"], dtype=float)
    b = np.array(results["rtn+bias"]["per_sample"], dtype=float)
    d, lo, hi = paired_delta(b, a)
    print("\n" + "=" * 70)
    print(f"RTN            WER = {results['rtn']['wer']:.4f}")
    print(f"RTN + bias     WER = {results['rtn+bias']['wer']:.4f}")
    print(f"juftlik farqi (bias - rtn) = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("=" * 70)
    results["delta"] = {"d": d, "lo": lo, "hi": hi}
    json.dump(results, open(OUT_JSON, "w"), indent=2)


if __name__ == "__main__":
    main()
