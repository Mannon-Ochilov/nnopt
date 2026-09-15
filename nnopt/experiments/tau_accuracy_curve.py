"""Choosing tau by an accuracy budget instead of fixing it by hand.

The cascade is currently one-sided. The cache budget says how much
compression is REQUIRED, but nothing says how much is PERMITTED: tau = 0.99
is a fixed hyperparameter, neither derived nor bounded by quality. Sec 4.9d
made the gap visible -- tau = 0.95 removes far more and costs WER 0.1833 ->
0.2006, and there is no principle in the method that would have stopped it.

Closing the loop needs the other side of the bracket:

    cache budget      -> minimum compression required
    accuracy budget   -> maximum compression permitted
    tau               -> searched between them

Two design constraints follow from this work's own results.

First, the selection signal has to be the TASK metric. Sec 4.7 measured
operator error varying 160x while network output error moved 4x, and Sec
4.9a found GPTQ's 57% operator-level advantage vanishing entirely in WER, so
E_loc and E_glob cannot stand in for quality here. tau must be chosen against
measured WER.

Second, tau must be chosen on data that is NOT the test split. Sec 4.9 showed
what happens when selection and evaluation share a distribution: the ordering
of the variants reversed. Selection therefore runs on VALIDATION utterances,
disjoint from both the calibration set and the reported test set.

Protocol: scan tau on N_SCAN validation utterances to locate the knee, then
confirm the chosen tau on the 300-utterance test split. Scan size is set to
100 rather than 50 because the paired interval scales as 1/sqrt(n): at 300 it
is about +-0.011, so 100 gives roughly +-0.019 and 50 about +-0.027. The
cliff (0.20 -> 2.74) is obvious at any size, but the gentle part of the curve
that decides the knee is not resolvable at 50.
"""

import gc
import json
import os
import time

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from calib_utils import ENCODER_PATH, MODEL_DIR, TARGET_SR, load_audio
from wer_cer_whole_network import error_rate, greedy_decode, normalize

DEC_INT8 = "models/_whole_net/dec_int8.onnx"
SWEEP_DIR = "models/_ratio_sweep"
OUT_JSON = "experiments/results_tau_curve.json"

# Calibration used utterances 0..11 of validation; the scan starts after them
# so the selection set is disjoint from calibration as well as from test.
N_CALIB_SKIP = 12
N_SCAN = int(os.environ.get("N_SCAN", "100"))
THREADS = 8
BOOTSTRAP = 2000

# tau -> encoder artifact. 0.99 and 0.95 already exist from Sec 4.9d; the
# intermediate values are what locate the knee.
TAUS = {
    0.99: "models/_gptq/enc_gptq_pruned.onnx",
    0.97: f"{SWEEP_DIR}/enc_bizniki_tau0.97_gptq.onnx",
    0.95: f"{SWEEP_DIR}/enc_bizniki_tau0.95_gptq.onnx",
    0.93: f"{SWEEP_DIR}/enc_bizniki_tau0.93_gptq.onnx",
    # Built as tau0.9, not tau0.90: the builder formats the float directly.
    0.90: f"{SWEEP_DIR}/enc_bizniki_tau0.9_gptq.onnx",
}
# Two references, because they answer different questions.
#
# FP32 is what a practitioner can actually state a budget against: they know
# their original model's quality and can say "no more than 2% worse than
# that". An ABSOLUTE tolerance does not transfer -- 0.02 WER is a 40%
# degradation on a model at 0.05 and a 4% one on a model at 0.50 -- so the
# user-facing budget is relative: WER_allowed = WER_FP32 * (1 + eps).
#
# The quantized-only arm is the second reference, and it isolates what the
# STRUCTURAL step costs on top of the quantization the cache budget already
# forced. Both are measured on the same utterances so the ratios are valid.
REF_FP32 = ("FP32 (mos yozuvlar)", ENCODER_PATH)
REF_QUANT = ("GPTQ yolg'iz (qisqartirishsiz)", "models/_gptq/enc_gptq_only.onnx")
EPSILONS = (0.01, 0.02, 0.05)


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def paired_ci(a, b, n=BOOTSTRAP, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load_cache():
    if not os.path.exists(OUT_JSON):
        return {}
    try:
        with open(OUT_JSON, encoding="utf-8-sig") as f:
            return {r["key"]: r for r in json.load(f)}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


def main():
    waves, texts = load_audio(N_CALIB_SKIP, N_SCAN)
    print(f"{len(waves)} namuna, VALIDATION splitidan (kalibrlashdan keyingi), "
          f"TEST emas\n")

    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in
                  tok.get_decoder_prompt_ids(language="uz", task="transcribe")]
    feats = [fe(w, sampling_rate=TARGET_SR, return_tensors="np")
             .input_features.astype(np.float32) for w in waves]

    rows = load_cache()
    todo = [REF_FP32, REF_QUANT] + [(f"tau={t:.2f}", p)
                                    for t, p in sorted(TAUS.items(), reverse=True)]

    for label, path in todo:
        key = f"{label}|n{len(feats)}"
        if key in rows:
            print(f"[{label}] keshdan WER={rows[key]['wer']:.4f}")
            continue
        if not os.path.exists(path):
            print(f"[{label}] SKIP — {path} yo'q")
            continue
        print(f"[{label}] kodlanmoqda...", flush=True)
        enc = session(path)
        states = [enc.run(None, {"input_features": f})[0].astype(np.float32)
                  for f in feats]
        del enc
        gc.collect()

        dec = session(DEC_INT8)
        wers, cers = [], []
        t0 = time.time()
        for i, (st, ref) in enumerate(zip(states, texts), 1):
            hyp = normalize(tok.decode(greedy_decode(dec, st, prompt_ids),
                                       skip_special_tokens=True))
            ref_n = normalize(ref)
            wers.append(error_rate(ref_n.split(), hyp.split()))
            cers.append(error_rate(list(ref_n), list(hyp)))
            if i % 50 == 0:
                el = time.time() - t0
                print(f"      {i}/{len(states)} [{el:.0f}s]", flush=True)
        del dec, states
        gc.collect()

        rows[key] = {"key": key, "variant": label, "path": path,
                     "n": len(wers), "mib": os.path.getsize(path) / 1024 ** 2,
                     "wer": float(np.mean(wers)), "cer": float(np.mean(cers)),
                     "per_sample_wer": wers}
        json.dump(list(rows.values()), open(OUT_JSON, "w"), indent=2)
        print(f"  WER={np.mean(wers):.4f}  CER={np.mean(cers):.4f}  "
              f"{rows[key]['mib']:.0f} MiB", flush=True)

    fp32 = rows.get(f"{REF_FP32[0]}|n{len(feats)}")
    quant = rows.get(f"{REF_QUANT[0]}|n{len(feats)}")

    print("\n" + "=" * 96)
    print(f"TAU EGRI CHIZIG'I — {len(feats)} validation namunasi "
          f"(tanlov uchun; hisobot TEST da)")
    print("=" * 96)
    print(f"{'Variant':30s} {'MiB':>6s} {'WER':>8s} {'FP32 ga':>9s} "
          f"{'kvantlanganga nisbatan dWER':>30s}")
    print("-" * 96)
    for label, _ in todo:
        r = rows.get(f"{label}|n{len(feats)}")
        if not r:
            continue
        ratio = f"{r['wer']/fp32['wer']:8.3f}x" if fp32 else "—"
        if quant and r is not quant:
            d, lo, hi = paired_ci(r["per_sample_wer"], quant["per_sample_wer"])
            delta = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        else:
            delta = "—"
        print(f"{r['variant']:30s} {r['mib']:6.0f} {r['wer']:8.4f} {ratio:>9s} "
              f"{delta:>30s}")

    if not fp32:
        return
    print("\nFOYDALANUVCHI BYUDJETI: 'FP32 dan eps dan ko'p yomonlashmasin'")
    print(f"  (FP32 WER = {fp32['wer']:.4f} shu to'plamda; "
          f"ruxsat = FP32 x (1 + eps))")
    for eps in EPSILONS:
        allowed = fp32["wer"] * (1 + eps)
        ok = []
        for t in sorted(TAUS, reverse=True):
            r = rows.get(f"tau={t:.2f}|n{len(feats)}")
            if r and r["wer"] <= allowed:
                ok.append((t, r["mib"], r["wer"]))
        if ok:
            t, mib, w = min(ok, key=lambda x: x[1])
            print(f"  eps = {eps:4.0%}  ruxsat {allowed:.4f}  ->  "
                  f"tau = {t:.2f}, {mib:.0f} MiB, WER {w:.4f} "
                  f"({w/fp32['wer']:.3f}x)")
        else:
            print(f"  eps = {eps:4.0%}  ruxsat {allowed:.4f}  ->  "
                  f"hech bir tau mos kelmadi (kvantlashning o'zi ham oshib "
                  f"ketishi mumkin)")
    if quant:
        print(f"\n  Eslatma: kvantlashning o'zi {quant['wer']/fp32['wer']:.3f}x "
              f"beradi, ya'ni undan past eps ni\n  strukturaviy o'q emas, "
              f"kvantlash bosqichining o'zi qondira olmaydi.")


if __name__ == "__main__":
    main()
