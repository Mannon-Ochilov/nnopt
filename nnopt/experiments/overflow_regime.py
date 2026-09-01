"""Testing the miss model where it actually bites, without a second machine.

Every cache prediction in this work so far was taken on one machine in one
regime. Whisper's encoder layer is 12 MiB after INT8 against a 16.8 MiB
budget, so it never overflows, and in that regime the miss expression reduces
to plain byte count -- which is why the latency check in Sec 4.9f could only
establish that fewer bytes means less time, not that the CACHE is the reason.
The overflow term, which is what produces the 1.81x prediction at 12 MiB and
the whole ordering at 2 MiB, was never exercised.

It can be exercised here. The regime is a property of the OPERATOR, not only
of the machine: an operator whose weights exceed alpha*L3 on this machine is
in exactly the regime a smaller cache would put Whisper in. open_llama_3b's
feed-forward weights are about 26 MiB per matrix at INT8, well past the
16.8 MiB budget, while Whisper's are 4 MiB and well inside it. So a sweep of
operator sizes spanning the budget reproduces both regimes on the hardware
available.

The measurement is time per multiply-accumulate against weight footprint. Two
predictions differ sharply:

  FLOP-bound      ns/MAC is flat across the budget; size buys nothing that
                  is not already counted in the arithmetic
  cache-bound     ns/MAC has a knee at alpha*L3, because past it the weight
                  tile stops being reused out of cache

A flat curve refutes the mechanism the miss objective assumes, and that
result would be worth reporting as it stands rather than explained away.
Reuse is varied too, since the overflow term is multiplied by it.
"""

import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper

from nnopt.bench.latency import make_session, measure_latency
from nnopt.hw.cache_topology import detect_cache_topology

MIB = 1024.0 ** 2
ALPHA = 0.7
OUT_DIR = "models/_overflow"
OUT_JSON = os.environ.get("OUT_JSON",
                          "experiments/results_overflow_regime.json")
WARMUP, MEASURED = 3, 10
# Single-threaded first, because it isolates one core's behaviour. But the
# cascade's premise is about a cache SHARED between cores, and one thread
# applies a fraction of the pressure the shared budget was derived for -- so a
# flat single-threaded curve does not settle the question on its own.
THREAD_COUNTS = tuple(int(t) for t in
                      os.environ.get("THREADS", "1,8").split(","))

# Square operators so weight bytes (4d^2) grow quadratically while the
# activations that stream past them grow linearly -- weights dominate at the
# large end, which is the regime under test.
DIMS = tuple(int(d) for d in os.environ.get(
    "DIMS", "512,768,1024,1536,2048,2560,3072,4096").split(","))
ROWS = (256, 1500)          # positions per pass, i.e. the reuse factor
ANCHORS = {1024: "Whisper enkoder fc1 (1024 x 4096 ga yaqin)",
           3072: "open_llama_3b FFN ga yaqin"}


def build_matmul(path, d):
    """One MatMul with a constant weight: X(rows, d) @ W(d, d)."""
    w = np.random.RandomState(0).randn(d, d).astype(np.float32) * 0.02
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="op")],
        f"sq{d}",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                              ["rows", d])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT,
                                               ["rows", d])],
        initializer=[helper.make_tensor("w", TensorProto.FLOAT, [d, d],
                                        w.flatten())])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = detect_cache_topology().global_shared_cache()
    budget = ALPHA * g.size_bytes
    print(f"L{g.level} = {g.size_bytes/MIB:.0f} MiB, byudjet = "
          f"{budget/MIB:.1f} MiB\n")

    rows_data = []
    for d in DIMS:
        path = f"{OUT_DIR}/sq{d}.onnx"
        if not os.path.exists(path):
            build_matmul(path, d)
        w_bytes = d * d * 4
        for rows in ROWS:
            x = np.random.RandomState(1).randn(rows, d).astype(np.float32)
            for threads in THREAD_COUNTS:
                lat = measure_latency(
                    make_session(path, intra_op_threads=threads),
                    name=f"sq{d}_r{rows}_t{threads}", fixed_feed={"x": x},
                    warmup_runs=WARMUP, measured_runs=MEASURED)
                macs = rows * d * d
                ns_per_mac = lat.median_ms * 1e6 / macs
                rows_data.append({"d": d, "rows": rows, "threads": threads,
                                  "w_mib": w_bytes / MIB,
                                  "ms": float(lat.median_ms), "macs": macs,
                                  "ns_per_mac": ns_per_mac,
                                  "over_budget": w_bytes > budget})
                print(f"  d={d:5d} rows={rows:5d} t={threads:2d}  vazn "
                      f"{w_bytes/MIB:7.2f}M  {lat.median_ms:8.2f} ms  "
                      f"{ns_per_mac:7.4f} ns/MAC"
                      f"{'  [byudjetdan tashqari]' if w_bytes > budget else ''}",
                      flush=True)
        json.dump(rows_data, open(OUT_JSON, "w"), indent=2)

    print("\n" + "=" * 88)
    print("MAC BOSHIGA VAQT vs VAZN HAJMI")
    print("=" * 88)
    for rows in ROWS:
      for threads in THREAD_COUNTS:
        sel = [r for r in rows_data
               if r["rows"] == rows and r["threads"] == threads]
        inside = [r for r in sel if not r["over_budget"]]
        outside = [r for r in sel if r["over_budget"]]
        if not inside or not outside:
            continue
        mi = float(np.median([r["ns_per_mac"] for r in inside]))
        mo = float(np.median([r["ns_per_mac"] for r in outside]))
        print(f"\nrows = {rows}, {threads} oqim:")
        print(f"  byudjet ichida ({len(inside)} nuqta): mediana "
              f"{mi:.4f} ns/MAC")
        print(f"  byudjetdan tashqari ({len(outside)} nuqta): mediana "
              f"{mo:.4f} ns/MAC")
        print(f"  nisbat: {mo/mi:.2f}x  -> "
              f"{'TIZZA BOR' if mo/mi > 1.15 else 'tizza yo`q, tekis'}")

    print("\nIzoh: tekis chiziq miss maqsadi tayanadigan MEXANIZMNI rad "
          "etadi —\nu holda vaqt FLOP bilan belgilanadi va kesh-bog'langan "
          "had ortiqcha bo'ladi.")
    for d, name in ANCHORS.items():
        b = d * d * 4 / MIB
        print(f"  d={d}: {b:.1f} MiB — {name}")


if __name__ == "__main__":
    main()
