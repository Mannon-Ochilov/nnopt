"""Re-measure every headline variant on a statistically usable eval set.

Why. Every WER/CER number reported so far came from 8 held-out utterances
(README Sec 8.3.11-D). At that size the metric could not separate 0.0729
from 0.0729, so several "no difference" verdicts were unresolvable rather
than measured -- in particular the comparison between local-error and
global-damage rank allocation. This re-runs the headline variants on ~80
held-out utterances and reports a bootstrap confidence interval, so
"different" and "indistinguishable" become claims the data can support.

Speed note: this decoder export has no KV-cache branch, so generation is
O(n^2) in tokens. Latency is NOT measured here (the Sec 8.3.x tables remain
the speed measurement, taken single-threaded); this run is quality-only and
therefore uses all cores.
"""

import json
import os

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from calib_utils import ENCODER_PATH, MODEL_DIR, TARGET_SR, load_audio
from wer_cer_whole_network import error_rate, greedy_decode, normalize

DECODER_FP32 = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
DECODER_INT8_PC = "models/_granularity/dec_int8_perchannel.onnx"

N_CALIB_SKIP = 12          # utterances 0..11 were used for calibration
N_EVAL = 80
THREADS = 8
BOOTSTRAP = 2000
OUT_JSON = "experiments/results_final_wer.json"

ENCODERS = [
    ("FP32", ENCODER_PATH),
    ("INT8", "models/_enc_v2/enc_int8.onnx"),
    ("INT8 + taqsimlangan rank", "models/_alloc/enc_greedy.onnx"),
    ("qisqartirish + INT8 per-channel", "models/_prune_pc/enc_pruned_perchannel.onnx"),
]
DECODERS = [
    ("decoder FP32", DECODER_FP32),
    ("decoder INT8", DECODER_INT8),
    ("decoder INT8 per-channel", DECODER_INT8_PC),
]


def session(path, threads=THREADS):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def bootstrap_ci(per_sample, n=BOOTSTRAP, seed=0):
    """Percentile CI over utterances -- the unit of variation here."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_sample, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(enc_path, dec_path, states_cache, texts, tok, prompt_ids):
    dec = session(dec_path)
    wers, cers = [], []
    for st, ref in zip(states_cache, texts):
        ids = greedy_decode(dec, st, prompt_ids)
        hyp = normalize(tok.decode(ids, skip_special_tokens=True))
        ref_n = normalize(ref)
        wers.append(error_rate(ref_n.split(), hyp.split()))
        cers.append(error_rate(list(ref_n), list(hyp)))
    del dec
    return wers, cers


def main():
    waves, texts = load_audio(N_CALIB_SKIP, N_EVAL)
    print(f"{len(waves)} ta held-out namuna (kalibrlashda ishlatilmagan)")
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in tok.get_decoder_prompt_ids(language="uz", task="transcribe")]

    feats = [fe(w, sampling_rate=TARGET_SR, return_tensors="np").input_features.astype(np.float32)
             for w in waves]

    rows = []

    # --- encoder variants, decoder held at INT8 ---
    for label, path in ENCODERS:
        if not os.path.exists(path):
            print(f"[{label}] SKIP — {path} yo'q")
            continue
        print(f"\n[encoder: {label}] kodlanmoqda...", flush=True)
        enc = session(path)
        states = [enc.run(None, {"input_features": f})[0].astype(np.float32) for f in feats]
        del enc
        wers, cers = evaluate(path, DECODER_INT8, states, texts, tok, prompt_ids)
        lo, hi = bootstrap_ci(wers)
        rows.append({"axis": "encoder", "variant": label, "wer": float(np.mean(wers)),
                     "cer": float(np.mean(cers)), "wer_lo": lo, "wer_hi": hi,
                     "per_sample_wer": wers})
        print(f"  WER={np.mean(wers):.4f} [{lo:.4f}, {hi:.4f}]  CER={np.mean(cers):.4f}")
        del states

    # --- decoder variants, encoder held at FP32 ---
    print("\n[FP32 encoder holatlari tayyorlanmoqda...]", flush=True)
    enc = session(ENCODER_PATH)
    states_fp32 = [enc.run(None, {"input_features": f})[0].astype(np.float32) for f in feats]
    del enc
    for label, path in DECODERS:
        if not os.path.exists(path):
            print(f"[{label}] SKIP — {path} yo'q")
            continue
        print(f"[decoder: {label}]", flush=True)
        wers, cers = evaluate(ENCODER_PATH, path, states_fp32, texts, tok, prompt_ids)
        lo, hi = bootstrap_ci(wers)
        rows.append({"axis": "decoder", "variant": label, "wer": float(np.mean(wers)),
                     "cer": float(np.mean(cers)), "wer_lo": lo, "wer_hi": hi,
                     "per_sample_wer": wers})
        print(f"  WER={np.mean(wers):.4f} [{lo:.4f}, {hi:.4f}]  CER={np.mean(cers):.4f}")

    json.dump([{k: v for k, v in r.items() if k != "per_sample_wer"} for r in rows],
              open(OUT_JSON, "w"), indent=2)

    print("\n" + "=" * 92)
    print(f"YAKUNIY WER/CER — {len(waves)} ta held-out namuna, 95% bootstrap ishonch oralig'i")
    print("=" * 92)
    for axis in ("encoder", "decoder"):
        sub = [r for r in rows if r["axis"] == axis]
        if not sub:
            continue
        print(f"\n{axis.upper()} variantlari:")
        print(f"{'Variant':36s} {'WER':>8s} {'95% CI':>20s} {'CER':>8s}")
        print("-" * 92)
        for r in sub:
            print(f"{r['variant']:36s} {r['wer']:8.4f} "
                  f"[{r['wer_lo']:.4f}, {r['wer_hi']:.4f}]".rjust(20) + f" {r['cer']:8.4f}")

    # pairwise significance against the first variant of each axis
    print("\nJuftlik taqqoslash (paired bootstrap, farq 0 dan farq qiladimi):")
    for axis in ("encoder", "decoder"):
        sub = [r for r in rows if r["axis"] == axis]
        if len(sub) < 2:
            continue
        base = sub[0]
        for other in sub[1:]:
            d = np.array(other["per_sample_wer"]) - np.array(base["per_sample_wer"])
            rng = np.random.default_rng(1)
            means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTSTRAP)]
            lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
            verdict = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
            print(f"  {axis:8s} {other['variant']:34s} dWER={d.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
