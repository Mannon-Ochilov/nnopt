"""Fixed ONNX inference workload, launched under VTune.

Kept deliberately minimal: one session, one fixed input, a timed loop long
enough for event-based sampling to gather statistics. Anything else (model
building, calibration) happens outside so the profile contains only the
GEMM work under study.

Usage:  python vtune_runner.py <model.onnx> <n_in> [seconds]
"""

import sys
import time

import numpy as np
import onnxruntime as ort

SEQ = 1500


def main():
    model_path = sys.argv[1]
    n_in = int(sys.argv[2])
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    x = rng.standard_normal((SEQ, n_in)).astype(np.float32)
    feed = {"x": x}

    for _ in range(3):  # warm up outside the measured window
        sess.run(None, feed)

    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        sess.run(None, feed)
        n += 1
    dt = time.perf_counter() - t0
    print(f"{model_path}: {n} iteratsiya, {dt/n*1000:.3f} ms/iter")


if __name__ == "__main__":
    main()
