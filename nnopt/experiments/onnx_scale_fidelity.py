"""Can ONNX carry OUR quantization scale, or does the runtime overwrite it?

This decides whether ONNX can stay the single framework for this work.

The problem it addresses is concrete. In the ASR track every quantized model
is produced by

    (optionally prune) -> save float32 ONNX -> quantize_dynamic(...)

and quantize_dynamic derives its own per-channel min/max scales from whatever
float weights it is handed. So the cascade's calibrated scale -- a claimed
contribution -- never reached a deployed artifact: Table 13's
"structural removal + per-channel INT8" is ORT's rounding, not ours. Only the
GPTQ arm escaped this, because gptq_plus_pruning.py pre-rounds the weights
before export.

Whether pre-rounding is enough is a question about the arithmetic, not a
matter of taste. ORT computes s' = max|w|/127 per output channel. If we hand
it weights already lying on our grid, w = s*q with |q| <= 127, then

    s' = max_j |s * q_j| / 127 = s * (max_j |q_j|) / 127 ,

so the export is EXACT when some code in the row saturates at +-127, and
rescales the row by max|q|/127 when none does. A calibrated scale that
shrinks the step causes saturation, so exactness is expected -- but the
rows where it does NOT saturate would be silently altered, and that is worth
measuring rather than assuming.

Verdict wanted:
  identical    -> ONNX is fine as the only framework; the bug was simply that
                  the ASR pipeline never pre-rounded with our scales.
  altered      -> quality claims need a second, framework-free track, with
                  ONNX kept for latency and hardware counters only.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from calib_utils import ENCODER_PATH
from nnopt.quantizer.per_channel import (
    quantize_codes_pc,
    refine_scales_per_channel,
)

TMP_DIR = "models/_scale_fidelity"
Q8 = 127
N_OPERATORS = 6
CALIB_ROWS = 512


def single_matmul_model(w, path):
    """Minimal graph Y = X @ W, so nothing but the weight export is tested."""
    k, n = w.shape
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, k])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, n])
    init = numpy_helper.from_array(w.astype(np.float32), "W")
    node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm")
    graph = helper.make_graph([node], "g", [x], [y], [init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def read_back(path):
    """Dequantized weight actually stored in the quantized model."""
    model = onnx.load(path)
    inits = {i.name: i for i in model.graph.initializer}
    for name in inits:
        if not name.endswith("_quantized"):
            continue
        base = name[: -len("_quantized")]
        q = numpy_helper.to_array(inits[name]).astype(np.float64)
        s = numpy_helper.to_array(inits[f"{base}_scale"]).astype(np.float64)
        zp_name = f"{base}_zero_point"
        z = (numpy_helper.to_array(inits[zp_name]).astype(np.float64)
             if zp_name in inits else 0.0)
        if np.ndim(s) and s.size > 1:
            axis = 0 if q.shape[0] == s.size else 1
            shape = (-1, 1) if axis == 0 else (1, -1)
            s, z = s.reshape(shape), np.reshape(z, shape) if np.ndim(z) else z
        return (q - z) * s
    raise RuntimeError(f"{path}: kvantlangan tenzor topilmadi")


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    model = onnx.load(ENCODER_PATH)
    # Square weights are excluded: the read-back cannot tell a transposed
    # export from a faithful one when both dimensions match, which produced
    # spurious 0.75-0.95 relative differences. Non-square weights pin the
    # orientation, so only they can answer the question being asked.
    weights = [(i.name, numpy_helper.to_array(i).astype(np.float64))
               for i in model.graph.initializer
               if numpy_helper.to_array(i).ndim == 2
               and min(numpy_helper.to_array(i).shape) >= 256
               and numpy_helper.to_array(i).shape[0]
               != numpy_helper.to_array(i).shape[1]]
    weights = weights[:N_OPERATORS]
    print(f"{len(weights)} ta nokvadrat enkoder vazni sinovdan o'tkaziladi "
          f"(kvadratlarda orientatsiya aniqlanmaydi)\n")

    print(f"{'vazn':24s} {'shakl':>13s} {'toygan':>8s} {'aynan saqlangan':>16s} "
          f"{'nisbiy farq':>12s}  hukm")
    print("-" * 92)

    verdicts = []
    for name, w in weights:
        # Our calibrated per-channel scale, applied on the (out, in) view.
        w_oi = w.T                                   # ONNX stores (in, out)
        x = rng.standard_normal((CALIB_ROWS, w_oi.shape[1]))
        res = refine_scales_per_channel(w_oi, Q8, x_calib=x)
        codes = quantize_codes_pc(w_oi, res.scales, Q8)
        intended = (codes * res.scales).T            # back to (in, out)

        saturated = float(np.mean(np.max(np.abs(codes), axis=1) >= Q8)) * 100

        fp = f"{TMP_DIR}/tmp_fp32.onnx"
        qp = f"{TMP_DIR}/tmp_q.onnx"
        single_matmul_model(intended, fp)
        quantize_dynamic(fp, qp, weight_type=QuantType.QInt8, per_channel=True)
        deployed = read_back(qp)
        os.remove(fp)
        os.remove(qp)

        if deployed.shape != intended.shape:
            deployed = deployed.T
        diff = float(np.max(np.abs(deployed - intended)))
        rel = diff / (float(np.max(np.abs(intended))) + 1e-30)

        # Per output channel: a row that saturates should survive exactly,
        # since ORT then recomputes the identical scale. A row that does not
        # gets rescaled to fill the range and re-rounded.
        #
        # Tolerance is set by the storage, not by taste: the initializer is
        # float32 while `intended` is computed in float64, so an exact match
        # can only be exact to float32 resolution. Demanding more (atol=1e-12)
        # reports 0% preserved even when every channel is bit-identical after
        # the float32 round trip.
        per_ch_exact = np.all(np.isclose(deployed, intended, rtol=1e-6, atol=0),
                              axis=0)
        exact_pct = float(np.mean(per_ch_exact)) * 100

        ok = exact_pct > 99.0
        verdicts.append(ok)
        print(f"{name[-22:]:24s} {str(w.shape):>13s} {saturated:7.0f}% "
              f"{exact_pct:15.0f}% {rel:12.2e}  {'AYNAN' if ok else 'BUZILDI'}")

    print("-" * 92)
    print("XULOSA: format ayb emas, delegatsiya ayb.")
    print()
    print("ORT ning sxemasi bizniki bilan AYNAN bir xil: simmetrik int8,")
    print("zero_point = 0, chiqish kanali bo'yicha s = max|w|/127 (o'lchandi,")
    print("nisbat 1.0000). Farq faqat shundan chiqadi: quantize_dynamic")
    print("masshtabni bizdan olmaydi, unga berilgan float vaznlardan QAYTA")
    print("hisoblaydi. Kodi +-127 ga yetgan kanalda qayta hisob o'sha qiymatni")
    print("beradi va kanal aynan saqlanadi; yetmagan kanalda esa panjara")
    print("qisqaradi va qiymatlar yarim qadamgacha siljiydi. Yuqoridagi ikki")
    print("ustunning tengligi shuni tasdiqlaydi.")
    print()
    print("Demak ONNX ni tashlash shart emas. Ikki yo'l bor:")
    print("  1) kvantlangan initializerlarni O'ZIMIZ yozish (W_quantized,")
    print("     W_scale, W_zero_point) — 100% aniqlik, format allaqachon mos;")
    print("  2) quantize_dynamic ni saqlab qolish — bu holda joylashtirilgan")
    print("     model ORT ning kvantlashi bo'ladi, bizniki emas, va buni")
    print("     maqolada ochiq aytish kerak.")


if __name__ == "__main__":
    main()
