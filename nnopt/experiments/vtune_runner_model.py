"""Fixed whole-model ONNX workload, launched under VTune.

Deliberately minimal, like vtune_runner.py: one session, one fixed feed, a
timed loop long enough for event-based sampling. The point is that encoder
and decoder are driven in their NATURAL regimes, because that asymmetry is
the thing being measured:

    encoder   one pass over 1500 positions  -> each weight is loaded once and
                                               reused 1500 times
    decoder   one autoregressive step       -> each weight is loaded once and
                                               used once, then 23 other layers
                                               evict it before it is needed
                                               again

Feeds are built from the graph's declared inputs by name, so the same runner
serves the plain models and the low-rank / quantized rewrites of them.

Usage:  python vtune_runner_model.py <model.onnx> <encoder|decoder> [seconds]
"""

import sys
import time

import numpy as np
import onnxruntime as ort

ENC_SEQ = 1500          # encoder positions per pass
DEC_SEQ = 1             # one generated token: the autoregressive regime
N_MELS, MEL_FRAMES = 80, 3000
D_MODEL = 1024
VOCAB_SAFE = 1000       # any valid token id


def build_feed(sess, kind):
    rng = np.random.default_rng(0)
    feed = {}
    for inp in sess.get_inputs():
        name = inp.name
        if "input_features" in name:
            feed[name] = rng.standard_normal(
                (1, N_MELS, MEL_FRAMES)).astype(np.float32)
        elif "input_ids" in name:
            feed[name] = rng.integers(0, VOCAB_SAFE, (1, DEC_SEQ)).astype(np.int64)
        elif "encoder_hidden_states" in name or "encoder_outputs" in name:
            feed[name] = rng.standard_normal(
                (1, ENC_SEQ, D_MODEL)).astype(np.float32)
        else:
            raise SystemExit(f"kutilmagan kirish: {name} (shape {inp.shape})")
    if not feed:
        raise SystemExit("kirishlar topilmadi")
    return feed


def main():
    model_path, kind = sys.argv[1], sys.argv[2]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(model_path, sess_options=so,
                                providers=["CPUExecutionProvider"])
    feed = build_feed(sess, kind)

    for _ in range(2):          # warm up outside the measured window
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
