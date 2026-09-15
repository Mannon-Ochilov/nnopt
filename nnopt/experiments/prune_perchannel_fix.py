"""Pruning + per-channel quantization: the two contributions are coupled.

Diagnosis behind this script. Structural pruning alone is excellent: with
19 of 24 FFN layers pruned (17% of FFN parameters removed) the FP32 encoder
output error is 0.0298, and it does NOT compound with depth (one layer
alone already costs 0.0209). But quantizing that pruned model per-tensor to
INT8 blows the error up to 0.7546 and WER to 1.0.

The cause is the compensation step itself. Folding gamma * W[:, j] into the
representative column W[:, p] makes representative columns accumulate many
contributions, so their magnitudes grow well beyond the rest. A single
tensor-wide INT8 scale must then cover those outliers, and every other
column loses precision.

That is precisely what per-channel scales fix (README Sec 8.3.8 measured
66-74% error reduction from that change alone). So the dissertation's two
elements are not independent: functional grouping CREATES the condition
that calibrated per-channel quantization is needed to handle. This script
measures that interaction directly.
"""

import glob
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import WhisperTokenizer

from calib_utils import ENCODER_PATH, MODEL_DIR, encoder_feeds, load_audio
from nnopt.bench.latency import make_session, measure_latency
from wer_cer_whole_network import error_rate, greedy_decode, normalize

PRUNE_DIR = "models/_prune"
OUT_DIR = "models/_prune_pc"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
N_EVAL_UTT = 8
WARMUP, MEASURED = 3, 10
OUT_JSON = "experiments/results_prune_perchannel.json"


def build_pruned_fp32(path):
    base = onnx.load(ENCODER_PATH)
    mi = {i.name: i for i in base.graph.initializer}
    kept = 0
    for f in sorted(glob.glob(f"{PRUNE_DIR}/prune_L*_tau0.99.npz")):
        z = np.load(f, allow_pickle=True)
        if len(z["keep"]) == 4096:
            continue
        bname, w1i, w2i = str(z["bias_name"]), str(z["w1_init"]), str(z["w2_init"])
        mi[w1i].CopyFrom(numpy_helper.from_array(z["w1"].astype(np.float32), w1i))
        mi[w2i].CopyFrom(numpy_helper.from_array(z["w2"].astype(np.float32), w2i))
        mi[bname].CopyFrom(numpy_helper.from_array(z["bias"].astype(np.float32), bname))
        kept += 1
    onnx.save(base, path)
    return kept


def dynamic_range_report():
    """Show what the compensation does to per-column magnitudes."""
    base = onnx.load(ENCODER_PATH, load_external_data=False)
    full = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in full.graph.initializer}
    print("\nKompensatsiya vazn diapazoniga qanday ta'sir qiladi (fc2, L2):")
    z = np.load(f"{PRUNE_DIR}/prune_L2_tau0.99.npz", allow_pickle=True)
    w2i = str(z["w2_init"])
    orig = numpy_helper.to_array(inits[w2i]).astype(np.float64)      # (4096, 1024)
    comp = z["w2"].astype(np.float64)                                 # (k, 1024)
    for name, arr in (("asl", orig), ("kompensatsiyalangan", comp)):
        rn = np.linalg.norm(arr, axis=1)
        print(f"  {name:22s} max|w|={np.abs(arr).max():8.4f}  "
              f"satr normasi: median={np.median(rn):7.4f} max={rn.max():8.4f}  "
              f"nisbat={rn.max()/np.median(rn):6.1f}x")


def score(enc_path, feeds_eval, texts, tok, prompt_ids, dec_sess, ref_out=None):
    lat = measure_latency(make_session(enc_path, intra_op_threads=1), name=enc_path,
                          fixed_feed=feeds_eval[0], warmup_runs=WARMUP, measured_runs=MEASURED)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    enc = ort.InferenceSession(enc_path, sess_options=so, providers=["CPUExecutionProvider"])
    wers, cers, errs = [], [], []
    for k, (feed, ref) in enumerate(zip(feeds_eval, texts)):
        states = enc.run(None, feed)[0].astype(np.float32)
        if ref_out is not None:
            errs.append(float(np.linalg.norm(ref_out[k] - states) / np.linalg.norm(ref_out[k])))
        hyp = normalize(tok.decode(greedy_decode(dec_sess, states.astype(np.float32), prompt_ids),
                                   skip_special_tokens=True))
        ref_n = normalize(ref)
        wers.append(error_rate(ref_n.split(), hyp.split()))
        cers.append(error_rate(list(ref_n), list(hyp)))
    del enc
    return lat.median_ms, float(np.mean(wers)), float(np.mean(cers)), \
        (float(np.mean(errs)) if errs else None)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dynamic_range_report()

    p_fp32 = f"{OUT_DIR}/enc_pruned_fp32.onnx"
    if not os.path.exists(p_fp32):
        n = build_pruned_fp32(p_fp32)
        print(f"\n{n} ta qatlam qisqartirildi -> {p_fp32}")

    p_pt = f"{OUT_DIR}/enc_pruned_pertensor.onnx"
    p_pc = f"{OUT_DIR}/enc_pruned_perchannel.onnx"
    for path, per_ch in ((p_pt, False), (p_pc, True)):
        if not os.path.exists(path):
            print(f"kvantlash (per_channel={per_ch})...", flush=True)
            quantize_dynamic(p_fp32, path, weight_type=QuantType.QInt8, per_channel=per_ch)

    feeds_eval = encoder_feeds(12, N_EVAL_UTT)
    _, texts = load_audio(12, N_EVAL_UTT)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    dec_sess = ort.InferenceSession(DECODER_INT8, sess_options=so, providers=["CPUExecutionProvider"])

    fp32_enc = ort.InferenceSession(ENCODER_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    ref_out = [fp32_enc.run(None, f)[0].astype(np.float32) for f in feeds_eval]
    del fp32_enc

    rows = []
    for label, path in (("qisqartirilgan FP32", p_fp32),
                        ("qisqartirilgan + INT8 per-tensor", p_pt),
                        ("qisqartirilgan + INT8 per-channel", p_pc)):
        ms, wer, cer, err = score(path, feeds_eval, texts, tok, prompt_ids, dec_sess, ref_out)
        size = os.path.getsize(path) / 1024**2
        rows.append({"variant": label, "mib": size, "ms": ms, "wer": wer, "cer": cer, "e_glob": err})
        print(f"  {label:34s} {size:7.0f} MiB  {ms:8.1f} ms  E_glob={err:.4f}  "
              f"WER={wer:.4f}  CER={cer:.4f}", flush=True)

    json.dump(rows, open(OUT_JSON, "w"), indent=2)

    print("\n" + "=" * 100)
    print("KANAL QISQARTIRISH + KVANTLASH GRANULYARLIGI (encoder, held-out)")
    print("=" * 100)
    print(f"{'Variant':36s} {'MiB':>7s} {'ms':>9s} {'INT8 ga':>9s} {'E_glob':>9s} {'WER':>8s} {'CER':>8s}")
    print("-" * 100)
    print(f"{'FP32 (asl)':36s} {1152:7.0f} {11246.1:9.1f} {0.63:8.2f}x {0.0000:9.4f} {0.0417:8.4f} {0.0030:8.4f}")
    print(f"{'INT8 (majburiy)':36s} {288:7.0f} {7039.4:9.1f} {1.00:8.2f}x {'-':>9s} {0.0667:8.4f} {0.0064:8.4f}")
    for r in rows:
        print(f"{r['variant']:36s} {r['mib']:7.0f} {r['ms']:9.1f} {7039.4/r['ms']:8.2f}x "
              f"{r['e_glob']:9.4f} {r['wer']:8.4f} {r['cer']:8.4f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
