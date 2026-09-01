"""Final measurement pass for the whole-network table: FP32 / INT8 / CASCADE
measured back-to-back under identical conditions on HELD-OUT samples.

INT4's end-to-end error is carried over from the earlier run of
whole_network_table.py (identical held-out samples, identical code path);
its model file was deleted to free disk space and it has no ORT CPU kernel,
so it contributes an accuracy row only.
"""

import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from nnopt.bench.latency import make_session, measure_latency
from nnopt.profiler.graph_profiler import profile_onnx_model

MODEL_DIR = "models/hh"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
INT8_PATH = "models/_whole_net/dec_int8.onnx"
CASCADE_PATH = "models/_whole_net_ext/dec_cascade_int8.onnx"
FC1_REPORT = "models/_whole_net/fc1_cur_report.json"
AUDIO_CACHE = "models/_calib_cache/cv_uz_validation.npz"
N_CALIB, N_EVAL = 12, 8
TARGET_SR = 16000
WARMUP, MEASURED = 3, 12
INT4_ELOC_CARRIED = 1.5132  # from the earlier identical run


def eval_feeds():
    z = np.load(AUDIO_CACHE, allow_pickle=True)
    flat, lengths, texts = z["audio"], z["lengths"], z["texts"]
    waves, off = [], 0
    for ln in lengths:
        waves.append(flat[off:off + ln])
        off += ln
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    enc = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    feeds = []
    for wav, text in list(zip(waves, texts))[N_CALIB:N_CALIB + N_EVAL]:
        f = fe(wav, sampling_rate=TARGET_SR, return_tensors="np").input_features.astype(np.float32)
        (eh,) = enc.run(None, {"input_features": f})
        ids = np.array([[50258, *prompt_ids, *tok(text, add_special_tokens=False).input_ids]], dtype=np.int64)
        feeds.append({"input_ids": ids, "encoder_hidden_states": eh.astype(np.float32)})
    return feeds


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9))


def main():
    feeds = eval_feeds()
    print(f"{len(feeds)} held-out samples (never used for calibration)")

    profs = profile_onnx_model(
        DECODER_PATH,
        free_dims={"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500},
    )
    ops = [p for p in profs if p.weight_initializer is not None]
    dec = onnx.load(DECODER_PATH, load_external_data=False)
    dims = {i.name: tuple(i.dims) for i in dec.graph.initializer}
    total_params = sum(int(np.prod(dims[p.weight_initializer])) for p in ops)
    fc1_params = sum(int(np.prod(dims[p.weight_initializer])) for p in ops if "/fc1/" in p.name)

    fc1_rep = json.load(open(FC1_REPORT))
    cur_bytes = sum(r["bytes"] for r in fc1_rep)
    cascade_bytes = (total_params - fc1_params) * 1 + cur_bytes
    print(f"240 ops, {total_params:,} params; fc1 {fc1_params:,} -> CUR {cur_bytes:,} bytes "
          f"(mean rank {np.mean([r['rank'] for r in fc1_rep]):.0f})")

    rows = {}

    print("\n[FP32] ...", flush=True)
    s = make_session(DECODER_PATH, intra_op_threads=1)
    ref = [s.run(None, f)[0] for f in feeds]
    lat = measure_latency(s, name="fp32", fixed_feed=feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    rows["FP32 (asl model)"] = (total_params * 4, 0.0, lat.median_ms)
    print(f"  latency {lat.median_ms:.1f} ms")
    del s

    print("[INT8] ...", flush=True)
    s = make_session(INT8_PATH, intra_op_threads=1)
    e = float(np.mean([relerr(r, s.run(None, f)[0]) for r, f in zip(ref, feeds)]))
    lat = measure_latency(s, name="int8", fixed_feed=feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    rows["INT8 (mavjud usul)"] = (total_params * 1, e, lat.median_ms)
    print(f"  E_glob {e:.4f}  latency {lat.median_ms:.1f} ms")
    del s

    rows["INT4 (mavjud usul)"] = (int(total_params * 0.5), INT4_ELOC_CARRIED, None)

    print("[CASCADE] ...", flush=True)
    s = make_session(CASCADE_PATH, intra_op_threads=1)
    e = float(np.mean([relerr(r, s.run(None, f)[0]) for r, f in zip(ref, feeds)]))
    lat = measure_latency(s, name="cascade", fixed_feed=feeds[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    # NOT the cascade's actual decision: per-operator E_loc for these fc1 CUR
    # variants was 0.31-0.43, far above delta_L=0.05, so the real cascade
    # REJECTS them and falls back to INT8. This row exists to show what that
    # rejection is protecting against. See README Sec 8.3.7.
    rows["fc1'ga CUR majburlangan"] = (cascade_bytes, e, lat.median_ms)
    print(f"  E_glob {e:.4f}  latency {lat.median_ms:.1f} ms")
    del s

    base_bytes = total_params * 4
    base_ms = rows["FP32 (asl model)"][2]
    print("\n" + "=" * 92)
    print("BUTUN TARMOQ: Whisper-medium uz decoder (240 operator), 8 ta held-out namuna")
    print("=" * 92)
    print(f"{'Usul':24s} {'Vazn(MiB)':>10s} {'Siqish':>8s} {'E_glob':>9s} {'Latency(ms)':>12s} {'Tezlanish':>10s}")
    print("-" * 92)
    for name, (b, e, ms) in rows.items():
        msf = f"{ms:.1f}" if ms else "N/A"
        spf = f"{base_ms/ms:.2f}x" if ms else "N/A"
        print(f"{name:24s} {b/1024**2:10.0f} {base_bytes/b:7.2f}x {e:9.4f} {msf:>12s} {spf:>10s}")
    print("=" * 92)

    json.dump({k: {"bytes": v[0], "eloc": v[1], "ms": v[2]} for k, v in rows.items()},
              open("experiments/results_whole_network.json", "w"), indent=2)


if __name__ == "__main__":
    main()
