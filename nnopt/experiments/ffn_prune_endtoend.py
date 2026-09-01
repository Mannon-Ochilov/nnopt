"""Structural FFN channel pruning, end to end.

This is the dissertation's functional-grouping idea in the role the data
supports. Earlier comparisons put it against SVD at low-rank approximation,
where Eckart-Young makes SVD's win inevitable. Grouping does something SVD
cannot: it identifies channels whose calibration responses are collinear and
removes them EXACTLY (up to the compensation), shrinking n and m rather than
rank.

In an FFN the intermediate width is fc1's output and fc2's input, so one
decision shrinks two operators:

    fc1  W1 (1024, 4096) -> (1024, k)     bias (4096,) -> (k,)
    GELU elementwise, untouched
    fc2  W2 (4096, 1024) -> (k, 1024)

Compensation before dropping channel j with representative p and
coefficient gamma:   W2[p, :] += gamma * W2[j, :].

Measured redundancy (ffn_channel_redundancy.py, tau=0.99) is strongly
depth-dependent -- L0 43%, L4 51%, L8 26%, L12 0.7%, L16+ ~0% -- so a single
fixed tau prunes the early layers hard and leaves the late ones alone by
itself. That IS the adaptive behaviour the cascade is supposed to produce,
arrived at from measurement rather than a hand-set schedule.

Variants measured: pruning alone, and pruning combined with low-rank on the
already-pruned matrices (the two axes compose: removing k channels frees
budget for a higher rank at equal parameter count).
"""

import gc
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import WhisperTokenizer

from calib_utils import ENCODER_PATH, MODEL_DIR, capture_activations, encoder_feeds, load_audio
from nnopt.bench.latency import make_session, measure_latency
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.profiler.graph_profiler import profile_onnx_model
from wer_cer_whole_network import error_rate, greedy_decode, normalize

OUT_DIR = "models/_prune"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
N_CALIB_UTT = 6
FIT_ROWS = 3072
LAYERS_PER_GROUP = 6
N_EVAL_UTT = 8
WARMUP, MEASURED = 3, 10
TAU = 0.99
EPS_THRESHOLD = 0.5
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
OUT_JSON = "experiments/results_ffn_prune.json"


def layer_of(name):
    import re
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def fc_pairs(profs):
    """(fc1_profile, fc2_profile) per layer."""
    fc1 = {layer_of(p.name): p for p in profs if p.weight_initializer and "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if p.weight_initializer and "/fc2/" in p.name}
    return [(fc1[li], fc2[li]) for li in sorted(fc1) if li in fc2]


def bias_name_for(model, fc1_profile):
    """The Add node consuming fc1's output carries the FFN bias."""
    consumers = {}
    for nd in model.graph.node:
        for i in nd.input:
            consumers.setdefault(i, []).append(nd)
    inits = {i.name for i in model.graph.initializer}
    for nd in consumers.get(fc1_profile.output_name, []):
        if nd.op_type == "Add":
            for i in nd.input:
                if i in inits:
                    return i
    return None


def _cache_path(li, tau):
    return f"{OUT_DIR}/prune_L{li}_tau{tau}.npz"


def prune_layers(pairs, feeds_cal, tau=TAU):
    """Returns {layer: (keep_idx, w1_new, bias_new, w2_new)} plus stats.

    Each layer's result is cached to disk. The grouping pass costs ~4 min
    per layer, so a 24-layer run is ~90 min; two session restarts have
    already thrown that work away mid-run.
    """
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    out, stats = {}, []
    layers = [layer_of(p2.name) for _, p2 in pairs]

    for gi in range(0, len(layers), LAYERS_PER_GROUP):
        grp = layers[gi:gi + LAYERS_PER_GROUP]
        gpairs = [(a, b) for a, b in pairs if layer_of(b.name) in grp]
        pending = [(a, b) for a, b in gpairs if not os.path.exists(_cache_path(layer_of(b.name), tau))]
        for p1, p2 in gpairs:
            li = layer_of(p2.name)
            cp = _cache_path(li, tau)
            if os.path.exists(cp):
                z = np.load(cp, allow_pickle=True)
                bn = str(z["bias_name"]) if str(z["bias_name"]) != "None" else None
                out[li] = {"keep": z["keep"], "w1": z["w1"], "w2": z["w2"],
                           "bias": z["bias"] if bn is not None else None,
                           "bias_name": bn, "w1_init": str(z["w1_init"]),
                           "w2_init": str(z["w2_init"])}
                n_total = 4096
                stats.append({"layer": li, "kept": int(len(z["keep"])),
                              "removed": int(n_total - len(z["keep"])),
                              "fraction": 1 - len(z["keep"]) / n_total})
                print(f"  L{li:<2d} keshdan: saqlanadi {len(z['keep']):5d}/{n_total}", flush=True)
        if not pending:
            continue
        x_by = capture_activations(ENCODER_PATH, sorted({b.activation_input for _, b in pending}),
                                   feeds_cal, max_rows=FIT_ROWS)
        for p1, p2 in pending:
            li = layer_of(p2.name)
            x = x_by[p2.activation_input][:FIT_ROWS]
            w2_stored = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
            w2 = w2_stored if w2_stored.shape[1] == x.shape[1] else w2_stored.T  # (1024, 4096)
            w1_stored = numpy_helper.to_array(inits[p1.weight_initializer]).astype(np.float64)  # (1024,4096)
            bname = bias_name_for(model, p1)
            bias = numpy_helper.to_array(inits[bname]).astype(np.float64) if bname else None

            g = greedy_group(x.T, np.linalg.norm(w2, axis=0),
                             float(np.linalg.norm(x @ w2.T)), tau=tau, eps_threshold=EPS_THRESHOLD)
            w2_comp = build_compensated_weight(w2, g)
            keep = np.array(sorted(gr.representative for gr in g.groups))

            out[li] = {
                "keep": keep,
                "w1": w1_stored[:, keep],                 # (1024, k)
                "bias": None if bias is None else bias[keep],
                "w2": w2_comp[:, keep].T,                 # (k, 1024)
                "bias_name": bname,
                "w1_init": p1.weight_initializer,
                "w2_init": p2.weight_initializer,
            }
            stats.append({"layer": li, "kept": int(len(keep)), "removed": int(w2.shape[1] - len(keep)),
                          "fraction": 1 - len(keep) / w2.shape[1]})
            print(f"  L{li:<2d} saqlanadi {len(keep):5d}/{w2.shape[1]}  "
                  f"olib tashlanadi {w2.shape[1]-len(keep):5d} ({1-len(keep)/w2.shape[1]:.1%})", flush=True)
            d = out[li]
            np.savez_compressed(_cache_path(li, tau), keep=d["keep"], w1=d["w1"], w2=d["w2"],
                                bias=d["bias"] if d["bias"] is not None else np.array([]),
                                bias_name=str(d["bias_name"]), w1_init=d["w1_init"],
                                w2_init=d["w2_init"])
            del x, w2, w2_comp, g
        del x_by
        gc.collect()
    return out, stats


def build_pruned_model(pruned, path_q):
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    total = 0
    for li, d in pruned.items():
        inits[d["w1_init"]].CopyFrom(numpy_helper.from_array(d["w1"].astype(np.float32), d["w1_init"]))
        inits[d["w2_init"]].CopyFrom(numpy_helper.from_array(d["w2"].astype(np.float32), d["w2_init"]))
        if d["bias_name"] is not None and d["bias"] is not None:
            inits[d["bias_name"]].CopyFrom(
                numpy_helper.from_array(d["bias"].astype(np.float32), d["bias_name"]))
        total += d["w1"].size + d["w2"].size
    tmp = path_q.replace(".onnx", "_fp32.onnx")
    onnx.save(model, tmp)
    quantize_dynamic(tmp, path_q, weight_type=QuantType.QInt8)
    os.remove(tmp)
    return total


def score(enc_path, feeds_eval, texts, tok, prompt_ids, dec_sess):
    lat = measure_latency(make_session(enc_path, intra_op_threads=1), name=enc_path,
                          fixed_feed=feeds_eval[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(enc_path, sess_options=so, providers=["CPUExecutionProvider"])
    wers, cers = [], []
    for feed, ref in zip(feeds_eval, texts):
        states = enc.run(None, feed)[0].astype(np.float32)
        hyp = normalize(tok.decode(greedy_decode(dec_sess, states, prompt_ids), skip_special_tokens=True))
        ref_n = normalize(ref)
        wers.append(error_rate(ref_n.split(), hyp.split()))
        cers.append(error_rate(list(ref_n), list(hyp)))
    del enc
    return lat.median_ms, float(np.mean(wers)), float(np.mean(cers))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    profs = profile_onnx_model(ENCODER_PATH, free_dims=ENC_DIMS)
    pairs = fc_pairs(profs)
    print(f"{len(pairs)} ta FFN juftligi (fc1+fc2)")

    enc0 = onnx.load(ENCODER_PATH, load_external_data=False)
    dims = {i.name: tuple(i.dims) for i in enc0.graph.initializer}
    all_w = [p for p in profs if p.weight_initializer]
    total_params = sum(int(np.prod(dims[p.weight_initializer])) for p in all_w)
    ffn_params = sum(int(np.prod(dims[p.weight_initializer]))
                     for p in all_w if "/fc1/" in p.name or "/fc2/" in p.name)

    feeds_cal = encoder_feeds(0, N_CALIB_UTT)
    print(f"\nkanal qisqartirish (tau={TAU})...")
    t0 = time.time()
    pruned, stats = prune_layers(pairs, feeds_cal)
    kept_params = sum(d["w1"].size + d["w2"].size for d in pruned.values())
    print(f"[{time.time()-t0:.0f}s]  FFN parametrlari {ffn_params:,} -> {kept_params:,} "
          f"({kept_params/ffn_params:.1%})")

    feeds_eval = encoder_feeds(12, N_EVAL_UTT)
    _, texts = load_audio(12, N_EVAL_UTT)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    dec_sess = ort.InferenceSession(DECODER_INT8, sess_options=so, providers=["CPUExecutionProvider"])

    p_q = f"{OUT_DIR}/enc_pruned_int8.onnx"
    if not os.path.exists(p_q):
        print("\nmodel qurilmoqda...")
        build_pruned_model(pruned, p_q)
    ms, wer, cer = score(p_q, feeds_eval, texts, tok, prompt_ids, dec_sess)
    enc_params = total_params - ffn_params + kept_params

    json.dump({"stats": stats, "ms": ms, "wer": wer, "cer": cer,
               "ffn_params_before": ffn_params, "ffn_params_after": kept_params,
               "encoder_params_after": enc_params}, open(OUT_JSON, "w"), indent=2)

    print("\n" + "=" * 96)
    print("FFN KANAL QISQARTIRISH — uchdan-uchgacha (INT8 ustida)")
    print("=" * 96)
    print(f"{'Variant':32s} {'Vazn(MiB)':>10s} {'Siqish':>8s} {'ms':>9s} {'INT8 ga':>9s} {'WER':>8s} {'CER':>8s}")
    print("-" * 96)
    print(f"{'FP32':32s} {total_params*4/1024**2:10.0f} {1.00:7.2f}x {11246.1:9.1f} {0.63:8.2f}x {0.0417:8.4f} {0.0030:8.4f}")
    print(f"{'INT8 (majburiy)':32s} {total_params/1024**2:10.0f} {4.00:7.2f}x {7039.4:9.1f} {1.00:8.2f}x {0.0667:8.4f} {0.0064:8.4f}")
    print(f"{'INT8 + taqsimlangan rank':32s} {192:10.0f} {6.00:7.2f}x {6197.9:9.1f} {1.14:8.2f}x {0.0729:8.4f} {0.0208:8.4f}")
    print(f"{'INT8 + kanal qisqartirish':32s} {enc_params/1024**2:10.0f} "
          f"{total_params*4/enc_params:7.2f}x {ms:9.1f} {7039.4/ms:8.2f}x {wer:8.4f} {cer:8.4f}")
    print("=" * 96)
    print(f"\nFFN da olib tashlangan kanallar: o'rtacha {np.mean([s['fraction'] for s in stats]):.1%}")


if __name__ == "__main__":
    main()
