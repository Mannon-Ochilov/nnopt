"""Does the cascade's DECISION-MAKING pay off for the whole model?

Everything measured so far varied one axis at a time: encoder variants with
the decoder pinned, or decoder variants with the encoder pinned. That answers
"is this operator change safe", not "is the cascade worth having". The
cascade's actual product is a per-operator DECISION, and on this model it
makes two opposite calls:

    decoder      derived requirement 3.81x, INT8 gives 4.00x  ->  stop here
    encoder FFN  still over budget after INT8                 ->  go further

A decision is only valuable if BOTH directions are wrong to override. So the
comparison is against the two uniform policies that ignore it:

  A  FP32                          reference
  B  uniform mild    INT8 everywhere, no structural change
                     -> under-treats the encoder, which the cascade says
                        needs more
  C  cascade         INT8 decoder + structurally reduced encoder
  D  uniform aggressive  INT8 + low-rank on encoder AND decoder
                     -> over-treats the decoder, which the cascade says
                        needs nothing

If the cascade is doing real work, B should cost memory it did not need to
spend and D should cost accuracy it did not need to lose, while C sits at the
good corner of both.

Arm D needs a low-rank decoder, built here with activation-aware SVD on the
decoder's feed-forward operators -- the same factorization used for the
encoder arm, so the two uniform policies really are the same recipe applied
everywhere.

Quantization is GPTQ on the encoder arms (Sec 4.9a) and ORT INT8 on the
decoder, identically across B, C and D, so the only thing that varies is
WHERE structural change was applied.
"""

import gc
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from calib_utils import (
    DECODER_PATH,
    ENCODER_PATH,
    MODEL_DIR,
    TARGET_SR,
    capture_activations,
    decoder_feeds,
    weighted_matmul_profiles,
)
from nnopt.cur.lowrank_baselines import activation_aware_svd
from wer_cer_whole_network import error_rate, greedy_decode, normalize

TEST_CACHE = "models/_calib_cache/cv_uz_test.npz"
DEC_FP32 = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
DEC_INT8 = "models/_whole_net/dec_int8.onnx"
DEC_LOWRANK = "models/_lowrank_dec/dec_lr_int8.onnx"
ENC_GPTQ = "models/_gptq/enc_gptq_only.onnx"
ENC_GPTQ_PRUNED = "models/_gptq/enc_gptq_pruned.onnx"
ENC_LOWRANK = "models/_alloc/enc_greedy.onnx"

N_EVAL = int(os.environ.get("N_EVAL", "300"))
THREADS = 8
BOOTSTRAP = 2000
RANK = 409
N_CALIB = 8
MAX_ROWS = 2048
DEC_DIMS = {"batch_size": 1, "decoder_sequence_length": 16,
            "encoder_sequence_length": 1500}
OUT_JSON = "experiments/results_whole_model_cascade.json"

VARIANTS = [
    ("A: FP32", ENCODER_PATH, DEC_FP32),
    ("B: bir xil yumshoq (hamma joyda INT8)", ENC_GPTQ, DEC_INT8),
    ("C: kaskad", ENC_GPTQ_PRUNED, DEC_INT8),
    ("D: bir xil agressiv (hamma joyda past-rank)", ENC_LOWRANK, DEC_LOWRANK),
]


def session(path, threads=THREADS):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def model_mib(path):
    total = os.path.getsize(path)
    ext = path + ".data"
    if os.path.exists(ext):
        total += os.path.getsize(ext)
    return total / (1024 * 1024)


def build_lowrank_decoder():
    """Activation-aware SVD on the decoder FFN, then INT8 -- arm D's decoder.

    This is deliberately the change the cascade REFUSES to make, so that the
    cost of overriding it can be measured rather than argued.
    """
    if os.path.exists(DEC_LOWRANK):
        print(f"[D dekoderi] mavjud, {model_mib(DEC_LOWRANK):.0f} MiB")
        return
    os.makedirs(os.path.dirname(DEC_LOWRANK), exist_ok=True)
    print("[D dekoderi] past-rank dekoder quriladi (kaskad rad etgan o'zgarish)")

    profs = weighted_matmul_profiles(DECODER_PATH, DEC_DIMS)
    ops = [p for p in profs if "/fc1/" in p.name or "/fc2/" in p.name]
    print(f"  {len(ops)} ta FFN operatori, rank {RANK}")

    feeds = decoder_feeds(0, N_CALIB)
    model = onnx.load(DECODER_PATH)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replacement, dead = {}, set()
    t0 = time.time()

    for k, p in enumerate(ops, 1):
        x_by = capture_activations(DECODER_PATH, [p.activation_input], feeds,
                                   max_rows=MAX_ROWS)
        x = x_by.get(p.activation_input)
        if x is None:
            continue
        w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        if w.shape[1] != x.shape[1]:
            continue
        r = min(RANK, min(w.shape))
        lr = activation_aware_svd(w, x, r)
        u, s, vt = np.linalg.svd(lr, full_matrices=False)
        rr = max(1, min(r, int(np.sum(s > 0))))
        sq = np.sqrt(s[:rr])
        a, b = (u[:, :rr] * sq), (sq[:, None] * vt[:rr, :])

        dead.add(p.weight_initializer)
        base = p.name.replace("/", "_").replace(".", "_")
        g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{base}_B"))
        g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{base}_A"))
        replacement[p.output_name] = [
            helper.make_node("MatMul", [p.activation_input, f"{base}_B"],
                             [f"{base}_h"], name=f"{base}_lr1"),
            helper.make_node("MatMul", [f"{base}_h", f"{base}_A"],
                             [p.output_name], name=f"{base}_lr2"),
        ]
        if k % 8 == 0:
            print(f"    {k}/{len(ops)} [{time.time()-t0:.0f}s]", flush=True)
        del x_by, x, w, lr
        gc.collect()

    rebuilt = []
    for nd in g.node:
        hit = next((o for o in nd.output if o in replacement), None)
        rebuilt.extend(replacement[hit] if hit else [nd])
    del g.node[:]
    g.node.extend(rebuilt)
    kept = [i for i in g.initializer if i.name not in dead]
    del g.initializer[:]
    g.initializer.extend(kept)

    tmp = DEC_LOWRANK.replace(".onnx", "_fp32.onnx")
    onnx.save(model, tmp, save_as_external_data=True,
              location=os.path.basename(tmp) + ".data", size_threshold=1024)
    quantize_dynamic(tmp, DEC_LOWRANK, weight_type=QuantType.QInt8,
                     per_channel=True)
    for f in (tmp, tmp + ".data"):
        if os.path.exists(f):
            os.remove(f)
    print(f"  saqlandi: {DEC_LOWRANK}  {model_mib(DEC_LOWRANK):.0f} MiB")


def load_test_audio(take):
    z = np.load(TEST_CACHE, allow_pickle=True)
    flat, lengths, texts = z["audio"], z["lengths"], z["texts"]
    waves, off = [], 0
    for ln in lengths:
        waves.append(flat[off:off + int(ln)])
        off += int(ln)
    return waves[:take], list(texts)[:take]


def bootstrap_ci(vals, n=BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    m = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_ci(a, b, n=BOOTSTRAP, seed=1):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_cache():
    if not os.path.exists(OUT_JSON):
        return {}
    try:
        with open(OUT_JSON, encoding="utf-8-sig") as f:
            return {r["variant"]: r for r in json.load(f)}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


# Arms A, B and C are (encoder, decoder) pairs that final_wer_testsplit.py
# already decoded, on the same 300 test utterances with the same greedy
# protocol: its decoder axis pinned the encoder at FP32, giving arm A, and its
# encoder axis pinned the decoder at INT8, giving arms B and C. Re-decoding
# them would cost about two hours to reproduce numbers that already exist, so
# they are imported by matching the model paths rather than recomputed. Only
# arm D, whose low-rank decoder no previous run built, is decoded here.
IMPORT_FROM = "experiments/results_final_wer_testsplit.json"
IMPORT_MAP = {
    "A: FP32": "decoder FP32",
    "B: bir xil yumshoq (hamma joyda INT8)": "GPTQ yolg'iz",
    "C: kaskad": "qisqartirish + GPTQ",
}


def import_existing(rows, n_eval):
    if not os.path.exists(IMPORT_FROM):
        return rows
    try:
        with open(IMPORT_FROM, encoding="utf-8-sig") as f:
            src = {r["variant"]: r for r in json.load(f) if r.get("n") == n_eval}
    except (json.JSONDecodeError, OSError, KeyError):
        return rows
    for label, enc_path, dec_path in VARIANTS:
        if label in rows or label not in IMPORT_MAP:
            continue
        r = src.get(IMPORT_MAP[label])
        if not r:
            continue
        rows[label] = {"variant": label, "enc": enc_path, "dec": dec_path,
                       "mib": model_mib(enc_path) + model_mib(dec_path),
                       "n": r["n"], "wer": r["wer"], "cer": r["cer"],
                       "wer_lo": r["wer_lo"], "wer_hi": r["wer_hi"],
                       "per_sample_wer": r["per_sample_wer"],
                       "imported_from": IMPORT_MAP[label]}
        print(f"  [{label}] oldingi yugurishdan olindi "
              f"('{IMPORT_MAP[label]}', WER={r['wer']:.4f})")
    return rows


def main():
    build_lowrank_decoder()

    waves, texts = load_test_audio(N_EVAL)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in
                  tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    feats = [fe(w, sampling_rate=TARGET_SR, return_tensors="np")
             .input_features.astype(np.float32) for w in waves]
    print(f"\n{len(feats)} namuna, TEST split\n")

    rows = load_cache()
    if rows:
        print(f"keshdan {len(rows)} variant")
    rows = import_existing(rows, len(feats))
    print()

    for label, enc_path, dec_path in VARIANTS:
        if label in rows:
            print(f"[{label}] keshdan WER={rows[label]['wer']:.4f}")
            continue
        if not (os.path.exists(enc_path) and os.path.exists(dec_path)):
            print(f"[{label}] SKIP — model yo'q")
            continue
        mib = model_mib(enc_path) + model_mib(dec_path)
        print(f"[{label}] {mib:.0f} MiB — kodlanmoqda...", flush=True)

        enc = session(enc_path)
        states = [enc.run(None, {"input_features": f})[0].astype(np.float32)
                  for f in feats]
        del enc
        gc.collect()

        dec = session(dec_path)
        wers, cers = [], []
        t0 = time.time()
        for i, (st, ref) in enumerate(zip(states, texts), 1):
            hyp = normalize(tok.decode(greedy_decode(dec, st, prompt_ids),
                                       skip_special_tokens=True))
            ref_n = normalize(ref)
            wers.append(error_rate(ref_n.split(), hyp.split()))
            cers.append(error_rate(list(ref_n), list(hyp)))
            if i % 100 == 0:
                el = time.time() - t0
                print(f"      {i}/{len(states)} [{el:.0f}s, "
                      f"~{el/i*(len(states)-i):.0f}s qoldi]", flush=True)
        del dec, states
        gc.collect()

        lo, hi = bootstrap_ci(wers)
        rows[label] = {"variant": label, "enc": enc_path, "dec": dec_path,
                       "mib": mib, "n": len(wers),
                       "wer": float(np.mean(wers)), "cer": float(np.mean(cers)),
                       "wer_lo": lo, "wer_hi": hi, "per_sample_wer": wers}
        json.dump(list(rows.values()), open(OUT_JSON, "w"), indent=2)
        print(f"  WER={np.mean(wers):.4f} [{lo:.4f}, {hi:.4f}]  "
              f"CER={np.mean(cers):.4f}", flush=True)

    print("\n" + "=" * 96)
    print(f"BUTUN MODEL: kaskad qarori bir xil siyosatlarga qarshi "
          f"({N_EVAL} namuna, TEST split)")
    print("=" * 96)
    print(f"{'Variant':44s} {'MiB':>7s} {'siqish':>8s} {'WER':>8s} {'CER':>8s}")
    print("-" * 96)
    base = rows.get("A: FP32")
    for label, _, _ in VARIANTS:
        r = rows.get(label)
        if not r:
            continue
        comp = base["mib"] / r["mib"] if base else float("nan")
        print(f"{label:44s} {r['mib']:7.0f} {comp:7.2f}x {r['wer']:8.4f} "
              f"{r['cer']:8.4f}")

    if base:
        print("\nFP32 ga nisbatan (juftlik bootstrap):")
        for label, _, _ in VARIANTS[1:]:
            r = rows.get(label)
            if not r:
                continue
            d, lo, hi = paired_ci(r["per_sample_wer"], base["per_sample_wer"])
            v = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
            print(f"  {label:44s} dWER={d:+.4f} [{lo:+.4f}, {hi:+.4f}]  {v}")

    c, b, d_ = rows.get("C: kaskad"), rows.get(VARIANTS[1][0]), rows.get(VARIANTS[3][0])
    if c and b:
        dd, lo, hi = paired_ci(c["per_sample_wer"], b["per_sample_wer"], seed=2)
        v = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
        print(f"\nKaskad vs bir xil YUMSHOQ: dWER={dd:+.4f} [{lo:+.4f}, {hi:+.4f}] {v}"
              f"   xotira {b['mib']:.0f} -> {c['mib']:.0f} MiB")
    if c and d_:
        dd, lo, hi = paired_ci(d_["per_sample_wer"], c["per_sample_wer"], seed=3)
        v = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
        print(f"Bir xil AGRESSIV vs kaskad: dWER={dd:+.4f} [{lo:+.4f}, {hi:+.4f}] {v}"
              f"   xotira {c['mib']:.0f} -> {d_['mib']:.0f} MiB")


if __name__ == "__main__":
    main()
