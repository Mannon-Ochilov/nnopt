"""Settling the knee question with a design that drift cannot fake.

A first sweep appeared to show a sharp penalty once an operator's weights
passed alpha*L3 under eight threads -- 1.7x to 2.2x in time per MAC. A plain
replication did not reproduce it: the same 25 MiB operator measured 0.0055 and
then 0.0032 ns/MAC, and every point in the second run was uniformly faster
than in the first. That pattern is machine load, not cache behaviour. A sweep
that walks sizes in order cannot tell the two apart, because size and time
both advance together with whatever else the machine is doing.

Interleaving separates them. Sizes are measured in alternation, A B C A B C,
so any drift lands on all of them equally, and the comparison is made within
rounds rather than across the run. Reporting the spread across rounds alongside
the medians makes it visible when a difference is smaller than the noise it
would have to beat.

Three sizes: far inside the budget, just inside, and just outside. If cache
residency of the whole weight matrix matters under multi-core pressure, the
third should separate from the first two consistently, round after round.
"""

import json
import os

import numpy as np

from nnopt.bench.latency import make_session, measure_latency
from nnopt.hw.cache_topology import detect_cache_topology
from overflow_regime import MIB, build_matmul

OUT_DIR = "models/_overflow"
OUT_JSON = "experiments/results_overflow_interleaved.json"
ALPHA = 0.7
ROUNDS = 7
ROWS = 1500
THREADS = int(os.environ.get("THREADS", "8"))
WARMUP, MEASURED = 2, 8
DIMS = (1024, 2048, 2560, 3072)     # 4, 16, 25, 36 MiB


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = detect_cache_topology().global_shared_cache()
    budget = ALPHA * g.size_bytes
    print(f"L{g.level} = {g.size_bytes/MIB:.0f} MiB, byudjet = "
          f"{budget/MIB:.1f} MiB, {THREADS} oqim, {ROUNDS} raund\n")

    sessions, feeds = {}, {}
    for d in DIMS:
        path = f"{OUT_DIR}/sq{d}.onnx"
        if not os.path.exists(path):
            build_matmul(path, d)
        sessions[d] = make_session(path, intra_op_threads=THREADS)
        feeds[d] = {"x": np.random.RandomState(1).randn(ROWS, d).astype(np.float32)}

    samples = {d: [] for d in DIMS}
    for rnd in range(ROUNDS):
        line = []
        for d in DIMS:
            lat = measure_latency(sessions[d], name=f"sq{d}", fixed_feed=feeds[d],
                                  warmup_runs=WARMUP, measured_runs=MEASURED)
            ns = lat.median_ms * 1e6 / (ROWS * d * d)
            samples[d].append(ns)
            line.append(f"d={d}: {ns:.4f}")
        print(f"  raund {rnd+1}/{ROUNDS}  " + "  ".join(line), flush=True)

    print("\n" + "=" * 78)
    print("NAVBATMA-NAVBAT O'LCHOV: MAC boshiga ns")
    print("=" * 78)
    print(f"{'d':>6s} {'vazn MiB':>9s} {'mediana':>9s} {'min':>9s} {'maks':>9s} "
          f"{'tarqoqlik':>10s}  byudjet")
    print("-" * 78)
    med = {}
    for d in DIMS:
        a = np.array(samples[d])
        med[d] = float(np.median(a))
        print(f"{d:6d} {d*d*4/MIB:9.2f} {med[d]:9.4f} {a.min():9.4f} "
              f"{a.max():9.4f} {(a.max()-a.min())/a.min():9.0%}  "
              f"{'tashqarida' if d*d*4 > budget else 'ichida'}")

    # Within-round ratios: drift cancels, so this is the comparison that counts.
    print("\nRaund ichidagi nisbatlar (drift qisqaradi):")
    inside = [d for d in DIMS if d * d * 4 <= budget]
    outside = [d for d in DIMS if d * d * 4 > budget]
    ref = max(inside)
    for d in outside:
        ratios = np.array(samples[d]) / np.array(samples[ref])
        lo, hi = np.percentile(ratios, [10, 90])
        verdict = "TIZZA" if lo > 1.10 else "tizza yo'q"
        print(f"  d={d} ({d*d*4/MIB:.0f}M) / d={ref} ({ref*ref*4/MIB:.0f}M): "
              f"mediana {np.median(ratios):.2f}x  [10-90%: {lo:.2f}-{hi:.2f}]"
              f"  -> {verdict}")

    json.dump({str(d): samples[d] for d in DIMS}, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
