"""Proving the CAUSE of the missing cache knee: it is the kernel's blocking.

Sec 4.10a reports that crossing the alpha*L3 budget costs at most 2% per MAC
on ONNX Runtime kernels, and ATTRIBUTES this to blocked GEMM: what must stay
resident is a weight TILE, not the matrix. Attributed -- not proven. The
attribution makes a falsifiable prediction: run the SAME weight-size sweep
through a kernel that does NOT tile the weight matrix, and the knee must
APPEAR, because such a kernel re-streams the whole matrix per output row
block and residency of the full matrix starts to matter.

The naive kernel here is a row-panel matmul written so each pass over the
weight matrix touches all of it before returning (numpy dot on the full
matrix per input chunk, chunks small enough that the weight cannot stay hot
between chunks unless it fits in cache). Interleaved A-B-A-B measurement,
same discipline as the original experiment -- an ordered sweep already
fooled us once.

Pre-registered readings:
  in/out-of-budget ratio >= 1.3x on the naive kernel -> cause PROVEN:
      the knee exists and blocking is what removes it; the miss objective's
      overflow term is valid for naive kernels and void for blocked ones.
  ratio ~1.0x also on naive -> the attribution is WRONG and the paper's
      explanation must be weakened to an open question.
"""

import json
import time

import numpy as np

BUDGET_MIB = 16.8
# Square FP32 weights straddling the budget, as in the original sweep.
SIZES_MIB = [4.0, 9.0, 16.0, 25.0, 36.0, 64.0]
N_POS = 1500            # activation positions per pass, as in the encoder
CHUNK = 8               # positions per chunk: forces weight re-streaming
ROUNDS = 5
OUT_JSON = "experiments/results_naive_knee.json"


def make_op(mib):
    n = int(np.sqrt(mib * 1024 * 1024 / 4))
    rng = np.random.default_rng(n)
    w = rng.standard_normal((n, n)).astype(np.float32)
    x = rng.standard_normal((N_POS, n)).astype(np.float32)
    return w, x


def naive_pass(w, x):
    """Chunked matmul: the full weight matrix is traversed once per chunk,
    so with many chunks the matrix must be re-read unless it fits in cache.
    BLAS still does the inner product, but the working set per call is the
    ENTIRE matrix -- the tiling that saves the blocked kernel happens inside
    one call and cannot help across calls."""
    out = np.empty((x.shape[0], w.shape[1]), dtype=np.float32)
    for i in range(0, x.shape[0], CHUNK):
        out[i:i + CHUNK] = x[i:i + CHUNK] @ w
    return out


def main():
    ops = {mib: make_op(mib) for mib in SIZES_MIB}
    # Warm-up every operator once.
    for w, x in ops.values():
        naive_pass(w, x)

    times = {mib: [] for mib in SIZES_MIB}
    for r in range(ROUNDS):
        for mib in SIZES_MIB:            # interleaved: A B C ... per round
            w, x = ops[mib]
            t0 = time.perf_counter()
            naive_pass(w, x)
            dt = time.perf_counter() - t0
            macs = w.shape[0] * w.shape[1] * N_POS
            times[mib].append(dt / macs * 1e9)
        print(f"raund {r+1}/{ROUNDS} tugadi", flush=True)

    med = {mib: float(np.median(v)) for mib, v in times.items()}
    inside = [med[m] for m in SIZES_MIB if m <= BUDGET_MIB]
    outside = [med[m] for m in SIZES_MIB if m > BUDGET_MIB]
    ratio = float(np.median(outside) / np.median(inside))

    print("\n" + "=" * 64)
    print(f"{'Vazn (MiB)':>10s} {'ns/MAC (mediana)':>18s} {'byudjet':>10s}")
    for mib in SIZES_MIB:
        mark = "ichida" if mib <= BUDGET_MIB else "TASHQARIDA"
        print(f"{mib:10.1f} {med[mib]:18.4f} {mark:>10s}")
    print(f"\ntashqari/ichkari nisbati (sodda yadro): {ratio:.2f}x")
    print("(bloklangan ONNX Runtime yadrosida bu nisbat 1.02x edi)")
    print("=" * 64)
    json.dump({"budget_mib": BUDGET_MIB, "chunk": CHUNK,
               "median_ns_per_mac": med, "ratio": ratio,
               "per_round": {str(k): v for k, v in times.items()}},
              open(OUT_JSON, "w"), indent=2)
    print(f"saqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
