"""Quantization GRANULARITY on the whole decoder (README Sec 8.3.8 follow-up).

Sec 8.3.7 measured per-tensor INT8 at E_glob = 0.2275, far worse than INT8
post-training quantization should cost. The per-operator spread there
(0.0045 on fc1 vs 0.0576 on encoder_attn.out_proj) points at per-channel
weight-scale variance being forced through one tensor-wide scale.

This measures that hypothesis end-to-end:

  FP32                 reference
  INT8 per-tensor      one scale per weight matrix   (what Sec 8.3.7 used)
  INT8 per-channel     one scale per output channel  (+m scales per op)

Accuracy = final decoder output (logits) relative error on HELD-OUT samples
never used for calibration. Latency = real ORT, warmup + median.

Scale-storage overhead of per-channel is reported explicitly so the
compression column stays honest: it is m extra fp32 scales per operator.
"""

import json
import os

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from calib_utils import DECODER_PATH, decoder_feeds
from nnopt.bench.latency import make_session, measure_latency
from nnopt.profiler.graph_profiler import profile_onnx_model

OUT_DIR = "models/_granularity"
INT8_PT = "models/_whole_net/dec_int8.onnx"          # per-tensor, built in Sec 8.3.7
INT8_PC = f"{OUT_DIR}/dec_int8_perchannel.onnx"
N_CALIB, N_EVAL = 12, 8
WARMUP, MEASURED = 3, 12
FREE_DIMS = {"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500}


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feeds = decoder_feeds(N_CALIB, N_EVAL)
    print(f"{len(feeds)} held-out samples")

    profs = profile_onnx_model(DECODER_PATH, free_dims=FREE_DIMS)
    ops = [p for p in profs if p.weight_initializer is not None]
    dec = onnx.load(DECODER_PATH, load_external_data=False)
    dims = {i.name: tuple(i.dims) for i in dec.graph.initializer}
    total_params = sum(int(np.prod(dims[p.weight_initializer])) for p in ops)
    # per-channel keeps one fp32 scale per OUTPUT channel of each operator
    scale_counts = sum(max(dims[p.weight_initializer]) for p in ops)
    print(f"{len(ops)} ops, {total_params:,} params; per-channel scales: {scale_counts:,} "
          f"(+{scale_counts*4/1024**2:.2f} MiB)")

    rows = {}

    print("\n[FP32] ...", flush=True)
    s = make_session(DECODER_PATH, intra_op_threads=1)
    ref = [s.run(None, f)[0] for f in feeds]
    lat = measure_latency(s, name="fp32", fixed_feed=feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    rows["FP32"] = (total_params * 4, 0.0, lat.median_ms)
    print(f"  {lat.median_ms:.1f} ms")
    del s

    print("[INT8 per-tensor] ...", flush=True)
    s = make_session(INT8_PT, intra_op_threads=1)
    e = float(np.mean([relerr(r, s.run(None, f)[0]) for r, f in zip(ref, feeds)]))
    lat = measure_latency(s, name="pt", fixed_feed=feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    rows["INT8 per-tensor"] = (total_params, e, lat.median_ms)
    print(f"  E_glob {e:.4f}  {lat.median_ms:.1f} ms")
    del s

    if not os.path.exists(INT8_PC):
        print("[INT8 per-channel] quantizing (slow)...", flush=True)
        quantize_dynamic(DECODER_PATH, INT8_PC, weight_type=QuantType.QInt8, per_channel=True)
    print("[INT8 per-channel] ...", flush=True)
    s = make_session(INT8_PC, intra_op_threads=1)
    e = float(np.mean([relerr(r, s.run(None, f)[0]) for r, f in zip(ref, feeds)]))
    lat = measure_latency(s, name="pc", fixed_feed=feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    rows["INT8 per-channel"] = (total_params + scale_counts * 4, e, lat.median_ms)
    print(f"  E_glob {e:.4f}  {lat.median_ms:.1f} ms")
    del s

    base_b, base_ms = rows["FP32"][0], rows["FP32"][2]
    print("\n" + "=" * 86)
    print("KVANTLASH GRANULYARLIGI -- butun decoder, held-out namunalar")
    print("=" * 86)
    print(f"{'Usul':20s} {'Vazn(MiB)':>10s} {'Siqish':>8s} {'E_glob':>9s} {'Latency(ms)':>12s} {'Tezlanish':>10s}")
    print("-" * 86)
    for k, (b, e, ms) in rows.items():
        print(f"{k:20s} {b/1024**2:10.0f} {base_b/b:7.2f}x {e:9.4f} {ms:12.1f} {base_ms/ms:9.2f}x")
    print("=" * 86)
    pt, pc = rows["INT8 per-tensor"][1], rows["INT8 per-channel"][1]
    print(f"\nper-channel aniqlik yutug'i: {pt:.4f} -> {pc:.4f}  ({(pt-pc)/pt*100:.1f}% yaxshilanish)")

    json.dump({k: {"bytes": v[0], "eloc": v[1], "ms": v[2]} for k, v in rows.items()},
              open("experiments/results_granularity.json", "w"), indent=2)


if __name__ == "__main__":
    main()
