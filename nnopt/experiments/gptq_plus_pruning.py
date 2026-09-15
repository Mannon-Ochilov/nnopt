"""The decisive comparison: GPTQ alone vs GPTQ combined with our structural
pruning, end to end.

Operator-level results (compare_vs_gptq_awq.py) showed GPTQ's Hessian-guided
error compensation beats our calibrated per-channel scale by a wide margin
(54.3% vs 12.9% error reduction over RTN on the encoder, winning 47/60
operators). That settles the quantization component: the cascade should use
GPTQ at its mandatory quantization step rather than our own scale search.

But GPTQ is a quantization method only. It does not remove channels, does not
consult a cache budget and does not allocate rank. Our structural axis is
therefore orthogonal to it, and the question this experiment answers is
whether the two compose:

    A  GPTQ alone                     -- current practice, 4.00x
    B  structural pruning + GPTQ      -- proposed, 4.32x

Both variants quantize every weighted operator with GPTQ, so the only
difference is whether functionally redundant FFN channels were removed
first. Pruning decisions are reloaded from models/_prune/ (tau = 0.99).

GPTQ here is our reimplementation (nnopt.quantizer.baselines); the official
package needs CUDA.
"""

import gc
import glob
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
from nnopt.profiler.graph_profiler import profile_onnx_model
from nnopt.quantizer.baselines import gptq_quantize
from wer_cer_whole_network import error_rate, greedy_decode, normalize

OUT_DIR = "models/_gptq"
PRUNE_DIR = "models/_prune"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
N_CALIB = 8
MAX_ROWS = 4096
LAYERS_PER_GROUP = 4
N_EVAL_UTT = 80
WARMUP, MEASURED = 3, 10
Q8 = 127
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
OUT_JSON = "experiments/results_gptq_pruning.json"


def layer_of(name):
    import re
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def load_prune_map():
    """{layer: (keep, w1_pruned, bias, w2_pruned, names)} from the cached run."""
    out = {}
    for f in sorted(glob.glob(f"{PRUNE_DIR}/prune_L*_tau0.99.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        z = np.load(f, allow_pickle=True)
        if len(z["keep"]) == 4096:
            continue
        out[li] = {
            "keep": z["keep"], "w1": z["w1"], "w2": z["w2"], "bias": z["bias"],
            "bias_name": str(z["bias_name"]), "w1_init": str(z["w1_init"]),
            "w2_init": str(z["w2_init"]),
        }
    return out


def build_gptq_model(path_fp32, path_q, prune_map=None, tag="",
                     src_path=ENCODER_PATH):
    """Apply GPTQ to every weighted operator, optionally on pruned weights.

    `src_path` lets a caller quantize a graph that has already been rewritten
    -- a factored FFN, say -- rather than only the stock encoder. GPTQ then
    sees the operators that will actually run, which is what its Hessian is
    supposed to be built from.
    """
    model = onnx.load(src_path)
    inits = {i.name: i for i in model.graph.initializer}
    profs = profile_onnx_model(src_path, free_dims=ENC_DIMS)
    ops = [p for p in profs if p.weight_initializer]

    # apply pruning first so GPTQ sees the final shapes
    pruned_inits = {}
    if prune_map:
        for li, d in prune_map.items():
            inits[d["w1_init"]].CopyFrom(
                numpy_helper.from_array(d["w1"].astype(np.float32), d["w1_init"]))
            inits[d["w2_init"]].CopyFrom(
                numpy_helper.from_array(d["w2"].astype(np.float32), d["w2_init"]))
            if d["bias_name"] != "None":
                inits[d["bias_name"]].CopyFrom(
                    numpy_helper.from_array(d["bias"].astype(np.float32), d["bias_name"]))
            pruned_inits[d["w1_init"]] = li
            pruned_inits[d["w2_init"]] = li
        onnx.save(model, path_fp32)  # pruned graph; activations captured from it
        capture_from = path_fp32
    else:
        capture_from = src_path

    feeds = encoder_feeds(0, N_CALIB)
    layers = sorted({layer_of(p.name) for p in ops if layer_of(p.name) >= 0})
    total_bytes, done = 0, 0
    t0 = time.time()

    for gi in range(0, len(layers), LAYERS_PER_GROUP):
        grp = layers[gi:gi + LAYERS_PER_GROUP]
        gops = [p for p in ops if layer_of(p.name) in grp]
        x_by = capture_activations(capture_from,
                                   sorted({p.activation_input for p in gops}),
                                   feeds, max_rows=MAX_ROWS)
        for p in gops:
            x = x_by.get(p.activation_input)
            if x is None:
                continue
            w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
            if w.shape[1] != x.shape[1]:
                w = w.T
                transposed = True
            else:
                transposed = False
            if w.shape[1] != x.shape[1]:
                continue
            wq = gptq_quantize(w, x, Q8)
            out = wq.T if transposed else wq
            inits[p.weight_initializer].CopyFrom(
                numpy_helper.from_array(out.astype(np.float32), p.weight_initializer))
            total_bytes += w.size
            done += 1
            del w, wq, out, x
        del x_by
        gc.collect()
        print(f"    {tag} {done} operator GPTQ bilan kvantlandi "
              f"[{time.time()-t0:.0f}s]", flush=True)

    onnx.save(model, path_fp32)
    quantize_dynamic(path_fp32, path_q, weight_type=QuantType.QInt8, per_channel=True)
    os.remove(path_fp32)
    return total_bytes


def score(enc_path, feeds_eval, texts, tok, prompt_ids, dec_sess):
    lat = measure_latency(make_session(enc_path, intra_op_threads=1), name=enc_path,
                          fixed_feed=feeds_eval[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    enc = ort.InferenceSession(enc_path, sess_options=so, providers=["CPUExecutionProvider"])
    wers, cers = [], []
    for feed, ref in zip(feeds_eval, texts):
        states = enc.run(None, feed)[0].astype(np.float32)
        hyp = normalize(tok.decode(greedy_decode(dec_sess, states, prompt_ids),
                                   skip_special_tokens=True))
        ref_n = normalize(ref)
        wers.append(error_rate(ref_n.split(), hyp.split()))
        cers.append(error_rate(list(ref_n), list(hyp)))
    del enc
    return lat.median_ms, wers, cers


def bootstrap_ci(vals, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prune_map = load_prune_map()
    print(f"{len(prune_map)} qatlam uchun qisqartirish qarori keshdan o'qildi\n")

    p_a = f"{OUT_DIR}/enc_gptq_only.onnx"
    p_b = f"{OUT_DIR}/enc_gptq_pruned.onnx"

    if not os.path.exists(p_a):
        print("[A] GPTQ yolg'iz")
        build_gptq_model(f"{OUT_DIR}/_tmp_a.onnx", p_a, None, "A")
    if not os.path.exists(p_b):
        print("\n[B] qisqartirish + GPTQ")
        build_gptq_model(f"{OUT_DIR}/_tmp_b.onnx", p_b, prune_map, "B")

    feeds_eval = encoder_feeds(12, N_EVAL_UTT)
    _, texts = load_audio(12, N_EVAL_UTT)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    dec = ort.InferenceSession(DECODER_INT8, sess_options=so, providers=["CPUExecutionProvider"])

    rows = {}
    for tag, path in (("A: GPTQ yolg'iz", p_a), ("B: qisqartirish + GPTQ", p_b)):
        print(f"\n{tag} baholanmoqda...", flush=True)
        ms, wers, cers = score(path, feeds_eval, texts, tok, prompt_ids, dec)
        lo, hi = bootstrap_ci(wers)
        rows[tag] = {"ms": ms, "wer": float(np.mean(wers)), "cer": float(np.mean(cers)),
                     "wer_lo": lo, "wer_hi": hi, "mib": os.path.getsize(path) / 1024**2,
                     "per_sample_wer": wers}
        print(f"  WER={np.mean(wers):.4f} [{lo:.4f}, {hi:.4f}]  CER={np.mean(cers):.4f}  "
              f"{ms:.1f} ms  {rows[tag]['mib']:.0f} MiB")

    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_sample_wer"}
               for k, v in rows.items()}, open(OUT_JSON, "w"), indent=2)

    print("\n" + "=" * 96)
    print("GPTQ vs GPTQ + STRUKTURAVIY QISQARTIRISH (Whisper encoder, 80 held-out)")
    print("=" * 96)
    print(f"{'Variant':28s} {'MiB':>7s} {'ms':>9s} {'WER':>8s} {'95% IO':>20s} {'CER':>8s}")
    print("-" * 96)
    print(f"{'FP32 (avvalgi o''lchov)':28s} {1152:7.0f} {11246.1:9.1f} {0.1007:8.4f} "
          f"{'[0.0629, 0.1444]':>20s} {0.0150:8.4f}")
    for k, v in rows.items():
        print(f"{k:28s} {v['mib']:7.0f} {v['ms']:9.1f} {v['wer']:8.4f} "
              f"{('['+format(v['wer_lo'],'.4f')+', '+format(v['wer_hi'],'.4f')+']'):>20s} "
              f"{v['cer']:8.4f}")
    print("=" * 96)

    ks = list(rows)
    if len(ks) == 2:
        d = np.array(rows[ks[1]]["per_sample_wer"]) - np.array(rows[ks[0]]["per_sample_wer"])
        rng = np.random.default_rng(1)
        means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
        verdict = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
        print(f"\nJuftlik bootstrap (B - A): dWER={d.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  -> {verdict}")


if __name__ == "__main__":
    main()
