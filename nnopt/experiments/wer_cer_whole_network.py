"""WER / CER -- the metric that actually decides whether a compression is
acceptable, and the one every table so far has been missing.

Everything measured up to now (README Sec 8.3.6-8.3.8) is a proxy: E_loc is
a per-operator output error, E_glob a relative error on the decoder's final
logits. Neither answers the only question that matters for ASR: does the
transcription change? A 20% relative logit error can leave every argmax
untouched, or wreck the output -- the number alone cannot tell you.

So this runs real greedy autoregressive decoding through each compressed
decoder and scores the produced text against the reference transcript.

Note on cost: this decoder export has no KV-cache branch, so each generated
token re-runs the full prefix (O(n^2) in tokens). That is why the sample
count is small and the utterances are short -- honest, but not a throughput
benchmark. The latency columns in Sec 8.3.7/granularity tables remain the
speed measurement; this script is purely about quality.
"""

import json

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from calib_utils import ENCODER_PATH, MODEL_DIR, TARGET_SR, load_audio

DECODER_FP32 = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
DECODER_INT8_PT = "models/_whole_net/dec_int8.onnx"
DECODER_INT8_PC = "models/_granularity/dec_int8_perchannel.onnx"
DECODER_CASCADE = "models/_whole_net_ext/dec_cascade_int8.onnx"

N_CALIB = 12          # skip these: used for calibration
N_EVAL = 8            # held-out utterances
MAX_NEW_TOKENS = 96
EOT = 50257           # <|endoftext|>
SOT = 50258


def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def error_rate(ref_units, hyp_units):
    if not ref_units:
        return 0.0 if not hyp_units else 1.0
    return levenshtein(ref_units, hyp_units) / len(ref_units)


def normalize(text: str) -> str:
    keep = [c.lower() for c in text if c.isalnum() or c.isspace() or c in "'’"]
    return " ".join("".join(keep).split())


def greedy_decode(sess, enc_states, prompt_ids, max_new=MAX_NEW_TOKENS):
    ids = [SOT, *prompt_ids]
    for _ in range(max_new):
        logits = sess.run(None, {
            "input_ids": np.array([ids], dtype=np.int64),
            "encoder_hidden_states": enc_states,
        })[0]
        nxt = int(np.argmax(logits[0, -1]))
        if nxt == EOT:
            break
        ids.append(nxt)
    return ids[1 + len(prompt_ids):]


def main():
    waves, texts = load_audio(N_CALIB, N_EVAL)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    enc = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])

    print(f"encoding {len(waves)} held-out utterances...")
    enc_states = []
    for w in waves:
        f = fe(w, sampling_rate=TARGET_SR, return_tensors="np").input_features.astype(np.float32)
        enc_states.append(enc.run(None, {"input_features": f})[0].astype(np.float32))
    del enc

    variants = [
        ("FP32 (asl)", DECODER_FP32),
        ("INT8 per-tensor", DECODER_INT8_PT),
        ("INT8 per-channel", DECODER_INT8_PC),
        ("KASKAD (fc1 CUR + INT8)", DECODER_CASCADE),
    ]

    results = {}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    for label, path in variants:
        print(f"\n[{label}] decoding...", flush=True)
        try:
            sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as exc:
            print(f"  SKIP ({type(exc).__name__}: {str(exc)[:120]})")
            continue
        wers, cers, hyps = [], [], []
        for i, (states, ref) in enumerate(zip(enc_states, texts), 1):
            out_ids = greedy_decode(sess, states, prompt_ids)
            hyp = normalize(tok.decode(out_ids, skip_special_tokens=True))
            ref_n = normalize(ref)
            wers.append(error_rate(ref_n.split(), hyp.split()))
            cers.append(error_rate(list(ref_n), list(hyp)))
            hyps.append(hyp)
            print(f"  {i}/{len(texts)} WER={wers[-1]:.3f} CER={cers[-1]:.3f}", flush=True)
        results[label] = {
            "wer": float(np.mean(wers)), "cer": float(np.mean(cers)),
            "per_sample_wer": wers, "hyps": hyps,
        }
        del sess

    print("\n" + "=" * 74)
    print("WER / CER -- butun decoder, 8 ta held-out namuna, greedy dekodlash")
    print("=" * 74)
    print(f"{'Usul':28s} {'WER':>10s} {'CER':>10s} {'WER o''sishi':>14s}")
    print("-" * 74)
    base = results.get("FP32 (asl)", {}).get("wer")
    for label, r in results.items():
        delta = "" if base is None or label == "FP32 (asl)" else f"{(r['wer']-base)*100:+.1f} p.p."
        print(f"{label:28s} {r['wer']:10.4f} {r['cer']:10.4f} {delta:>14s}")
    print("=" * 74)

    print("\nNamuna transkripsiyalar (1-namuna):")
    print(f"  REF : {normalize(texts[0])[:100]}")
    for label, r in results.items():
        print(f"  {label[:20]:20s}: {r['hyps'][0][:100]}")

    json.dump({k: {"wer": v["wer"], "cer": v["cer"]} for k, v in results.items()},
              open("experiments/results_wer_cer.json", "w"), indent=2)


if __name__ == "__main__":
    main()
