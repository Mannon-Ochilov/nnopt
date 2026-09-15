"""The same question for the kernel the models actually run.

overflow_regime.py found no cache-residency penalty, but it measured fp32
MatMul. Every artifact in this work runs MatMulInteger after dynamic
quantization, which is a different kernel with its own blocking, and a
negative result on one says little about the other. Since the conclusion being
drawn is a negative one about the framework's own premise, the kernel that
matters has to be the one tested.

Two things change with INT8. A weight of d x d now costs d^2 bytes rather than
4d^2, so the dimensions have to double to span the same budget; and the graph
gains quantize/dequantize nodes whose cost does not scale with d^2, which
inflates ns/MAC at small d and has to be read as an artifact rather than as a
cache effect.

Measurement is interleaved, for the reason overflow_interleaved.py documents:
an ordered sweep on this machine produced a convincing 1.7x knee that was
machine drift, and interleaving is what exposed it.
"""

import json
import os

import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic

from nnopt.bench.latency import make_session, measure_latency
from nnopt.hw.cache_topology import detect_cache_topology
from overflow_regime import MIB, build_matmul

OUT_DIR = "models/_overflow"
OUT_JSON = "experiments/results_overflow_int8.json"
ALPHA = 0.7
ROUNDS = 7
ROWS = 1500
THREADS = int(os.environ.get("THREADS", "1"))
WARMUP, MEASURED = 2, 8
# INT8 weights are d^2 bytes: 4, 16, 25, 36 MiB.
DIMS = (2048, 4096, 5120, 6144)


def int8_path(d):
    path = f"{OUT_DIR}/sq{d}_int8.onnx"
    if os.path.exists(path):
        return path
    src = f"{OUT_DIR}/sq{d}.onnx"
    if not os.path.exists(src):
        build_matmul(src, d)
    quantize_dynamic(src, path, weight_type=QuantType.QInt8, per_channel=False)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = detect_cache_topology().global_shared_cache()
    budget = ALPHA * g.size_bytes
    print(f"L{g.level} = {g.size_bytes/MIB:.0f} MiB, byudjet = "
          f"{budget/MIB:.1f} MiB, INT8, {THREADS} oqim, {ROUNDS} raund\n")

    sessions, feeds = {}, {}
    for d in DIMS:
        sessions[d] = make_session(int8_path(d), intra_op_threads=THREADS)
        feeds[d] = {"x": np.random.RandomState(1).randn(ROWS, d).astype(np.float32)}
        print(f"  d={d}: vazn {d*d/MIB:6.2f} MiB "
              f"{'(byudjetdan tashqari)' if d*d > budget else '(ichida)'}")

    samples = {d: [] for d in DIMS}
    print()
    for rnd in range(ROUNDS):
        line = []
        for d in DIMS:
            lat = measure_latency(sessions[d], name=f"sq{d}i8",
                                  fixed_feed=feeds[d], warmup_runs=WARMUP,
                                  measured_runs=MEASURED)
            ns = lat.median_ms * 1e6 / (ROWS * d * d)
            samples[d].append(ns)
            line.append(f"d={d}: {ns:.4f}")
        print(f"  raund {rnd+1}/{ROUNDS}  " + "  ".join(line), flush=True)

    print("\n" + "=" * 78)
    print(f"INT8 YADROSI, navbatma-navbat ({THREADS} oqim)")
    print("=" * 78)
    print(f"{'d':>6s} {'vazn MiB':>9s} {'mediana':>9s} {'tarqoqlik':>10s}  byudjet")
    print("-" * 78)
    for d in DIMS:
        a = np.array(samples[d])
        print(f"{d:6d} {d*d/MIB:9.2f} {np.median(a):9.4f} "
              f"{(a.max()-a.min())/a.min():9.0%}  "
              f"{'tashqarida' if d*d > budget else 'ichida'}")

    inside = [d for d in DIMS if d * d <= budget]
    outside = [d for d in DIMS if d * d > budget]
    ref = max(inside)
    print("\nRaund ichidagi nisbatlar (drift qisqaradi):")
    for d in outside:
        ratios = np.array(samples[d]) / np.array(samples[ref])
        lo, hi = np.percentile(ratios, [10, 90])
        verdict = "TIZZA" if lo > 1.10 else "tizza yo'q"
        print(f"  d={d} ({d*d/MIB:.0f}M) / d={ref} ({ref*ref/MIB:.0f}M): "
              f"mediana {np.median(ratios):.2f}x  [10-90%: {lo:.2f}-{hi:.2f}]"
              f"  -> {verdict}")

    json.dump({str(d): samples[d] for d in DIMS}, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
