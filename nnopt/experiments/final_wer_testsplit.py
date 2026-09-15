"""Headline WER/CER on the Common Voice TEST split, at a size that can
actually resolve the differences being claimed.

final_wer_evaluation.py measured the same variants on 80 utterances of the
validation split. That was enough to call INT8 significantly worse than
FP32, but the central claim -- that structural removal plus per-channel INT8
is INDISTINGUISHABLE from FP32 -- rested on an interval of
[-0.0020, +0.0240]. An interval that wide cannot separate "no damage" from
"damage up to 0.024 WER", so the claim was directionally supported rather
than established. Here the evaluation set is ~600 utterances from the test
split (see cache_cv_test_split.py), which narrows the interval by about
2.7x and moves the numbers onto the split the literature reports.

Calibration is untouched: it still comes from the first 12 utterances of the
VALIDATION split, so calibration and evaluation are now drawn from different
splits rather than merely disjoint slices of one.

Results are written per variant as they complete. Two background runs in
this project have been killed mid-flight by session teardown, and a full
sweep here is hours of autoregressive decoding, so losing everything on an
interruption is not acceptable. Re-running resumes from the cache.

Speed note: this decoder export has no KV-cache branch, so generation is
O(n^2) in tokens. Latency is NOT measured here -- the Sec 8.3.x tables
remain the speed measurement, taken single-threaded. This run is
quality-only and therefore uses all cores.
"""

import json
import os
import time

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from calib_utils import ENCODER_PATH, MODEL_DIR, TARGET_SR
from wer_cer_whole_network import error_rate, greedy_decode, normalize

TEST_CACHE = "models/_calib_cache/cv_uz_test.npz"
DECODER_FP32 = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
DECODER_INT8 = "models/_whole_net/dec_int8.onnx"
DECODER_INT8_PC = "models/_granularity/dec_int8_perchannel.onnx"

N_EVAL = int(os.environ.get("N_EVAL", "600"))
THREADS = 8
BOOTSTRAP = 2000
OUT_JSON = "experiments/results_final_wer_testsplit.json"

ENCODERS = [
    ("FP32", ENCODER_PATH),
    ("INT8", "models/_enc_v2/enc_int8.onnx"),
    ("INT8 + taqsimlangan rank", "models/_alloc/enc_greedy.onnx"),
    ("qisqartirish + INT8 per-channel", "models/_prune_pc/enc_pruned_perchannel.onnx"),
    # The two variants above test the ORIGINAL positioning, where the cascade
    # supplied its own quantization scale. Sec 4.9a retired that: GPTQ beat our
    # scale on 47 of 60 operators, so the cascade now delegates quantization to
    # GPTQ and claims only the ORTHOGONAL structural axis. The pair below is
    # therefore the paper's actual headline (Table 16c) -- both GPTQ-quantized,
    # differing only in whether redundant channels were removed first -- and it
    # had only ever been measured on 80 validation utterances.
    ("GPTQ yolg'iz", "models/_gptq/enc_gptq_only.onnx"),
    ("qisqartirish + GPTQ", "models/_gptq/enc_gptq_pruned.onnx"),
    # Same two arms with the quantizer swapped for plain round-to-nearest,
    # completing the 2x2 of {GPTQ, RTN} x {no pruning, pruning}. If the
    # structural axis is free on top of RTN as well, orthogonality belongs to
    # the axis rather than to GPTQ's error redistribution. Per-channel, since
    # Sec 4.4 already showed per-tensor collapses after compensation.
    ("RTN per-channel yolg'iz", "models/_rtn/enc_rtn_only.onnx"),
    ("qisqartirish + RTN per-channel", "models/_rtn/enc_rtn_pruned.onnx"),
    # Sec 4.8's allocation claim -- greedy budget-optimal ranks beat a uniform
    # split at EQUAL budget -- was measured only on the 80 validation
    # utterances, i.e. the calibration-adjacent protocol Sec 4.9 retired. Both
    # artifacts are 203 MB, so the budget really is matched; the greedy arm is
    # already scored above as "INT8 + taqsimlangan rank", and this adds the
    # uniform arm it has to be compared against.
    ("INT8 + bir xil rank", "models/_alloc/enc_uniform.onnx"),
    # The structural axis is this work's own contribution, and until now it had
    # no published competitor. These two remove EXACTLY the same number of
    # channels per layer as the cascade does, and are quantized with the same
    # GPTQ pass, so the only thing that varies is which channels go: smallest
    # weight norm (magnitude) or smallest weight norm times activation norm
    # (Wanda). Neither compensates, which is what our criterion adds.
    ("qisqartirish magnitude + GPTQ", "models/_struct_base/enc_magnitude_gptq.onnx"),
    ("qisqartirish wanda + GPTQ", "models/_struct_base/enc_wanda_gptq.onnx"),
    # Same three criteria at a much more aggressive budget (tau = 0.95, up to
    # 73% of a layer's channels removed instead of 43%). At tau = 0.99 they
    # were indistinguishable, which may only mean the network had slack to
    # spare; a criterion can only be shown to matter where the slack is gone.
    ("t95 bizniki + GPTQ", "models/_ratio_sweep/enc_bizniki_tau0.95_gptq.onnx"),
    ("t95 magnitude + GPTQ", "models/_ratio_sweep/enc_magnitude_tau0.95_gptq.onnx"),
    ("t95 wanda + GPTQ", "models/_ratio_sweep/enc_wanda_tau0.95_gptq.onnx"),
    # Same channels our grouping picked, with the compensation step switched
    # off: separates "which channels go" from "what happens to them".
    ("t95 bizniki, kompensatsiyasiz",
     "models/_ratio_sweep/enc_nocomp_tau0.95_gptq.onnx"),
    # Stage 2 of the accuracy-bounded selection: tau = 0.97 is what the
    # eps = 5% budget picked on the 100-utterance validation scan, so it is
    # confirmed here on the independent test split. Selection and reporting
    # deliberately use different splits (Sec 4.9).
    ("t97 bizniki + GPTQ (eps=5% tanlovi)",
     "models/_ratio_sweep/enc_bizniki_tau0.97_gptq.onnx"),
    # The criterion gap looked budget-dependent: 0.0017 at tau = 0.99 and
    # 0.0196 at tau = 0.95, neither individually significant. Two points
    # cannot establish a trend, so all three criteria are traced across four
    # budgets. A gap that widens monotonically across four matched-budget
    # comparisons is evidence chance does not produce, even where the
    # individual intervals cover zero.
    ("t97 magnitude + GPTQ", "models/_ratio_sweep/enc_magnitude_tau0.97_gptq.onnx"),
    ("t97 wanda + GPTQ", "models/_ratio_sweep/enc_wanda_tau0.97_gptq.onnx"),
    ("t93 bizniki + GPTQ", "models/_ratio_sweep/enc_bizniki_tau0.93_gptq.onnx"),
    ("t93 magnitude + GPTQ", "models/_ratio_sweep/enc_magnitude_tau0.93_gptq.onnx"),
    ("t93 wanda + GPTQ", "models/_ratio_sweep/enc_wanda_tau0.93_gptq.onnx"),
    # tau = 0.90 was built but is not scored: the four budgets above already
    # separate the criteria (magnitude collapses at 0.95, the others degrade
    # smoothly), so a fifth point costs an hour of decoding to sharpen a
    # conclusion that is not in doubt.
    # Wanda as PUBLISHED: unstructured, per-output-row, shape unchanged. The
    # arms above use its scoring rule at channel granularity, which is our
    # adaptation and must not be attributed to the original. Matched to our
    # per-layer parameter fraction, so the nominal compression is the same
    # while the artifact stays 300 MiB and 455 GMAC.
    ("Wanda asl (strukturasiz) t99",
     "models/_wanda_unstr/enc_wanda_unstr_tau0.99_gptq.onnx"),
    ("Wanda asl (strukturasiz) t95",
     "models/_wanda_unstr/enc_wanda_unstr_tau0.95_gptq.onnx"),
    # FLAP: the only baseline that, like ours, both uses calibration data AND
    # compensates for what it removes. It differs in how -- a removed channel
    # is replaced by its mean contribution in the output bias rather than
    # folded into a representative -- so its fluctuation criterion follows
    # from its compensation, just as our collinearity criterion follows from
    # ours. Its adaptive layer allocation is disabled here; the per-layer
    # counts are ours, so the comparison isolates criterion plus compensation.
    ("FLAP t99", "models/_flap/enc_flap_tau0.99_gptq.onnx"),
    ("FLAP t95", "models/_flap/enc_flap_tau0.95_gptq.onnx"),
    # Our compensation plus FLAP's, suggested by the comparison itself: the
    # projection onto a representative leaves a residual whose MEAN is simply
    # discarded, and that mean goes into the output bias here. Weights, channel
    # choice and quantizer are untouched, so the difference isolates the bias
    # term. Its size grows with the budget -- residual-mean norms run 0.001 to
    # 0.008 at tau = 0.99 but 0.01 to 0.44 at tau = 0.95.
    ("gibrid (bizniki + bias) t99", "models/_hybrid/enc_hybrid_tau0.99_gptq.onnx"),
    ("gibrid (bizniki + bias) t95", "models/_hybrid/enc_hybrid_tau0.95_gptq.onnx"),
    # Same 254 MiB reached a different way: the cosine stage stays strict at
    # tau = 0.99 and the rest of the budget is taken by removing the survivors
    # that barely fluctuate, with their means swept into the bias. Answers
    # whether relaxing the redundancy criterion or adding a second mechanism
    # is the better way to reach a smaller model.
    ("ikki bosqichli (kosinus + fluktuatsiya)",
     "models/_two_stage/enc_two_stage_gptq.onnx"),
    # tau-free comparison: only the TOTAL budget is fixed and each method
    # spreads it across layers by its own published rule -- magnitude by one
    # global ranking, Wanda by uniform per-layer sparsity, FLAP by its
    # normalised adaptive score. This is what tests whether magnitude's
    # collapse was its own or an artifact of being handed our allocation.
    ("global: magnitude t99", "models/_global_budget/enc_magnitude_global_t0.99.onnx"),
    ("global: wanda uniform t99", "models/_global_budget/enc_wanda_uniform_t0.99.onnx"),
    ("global: FLAP adaptiv t99", "models/_global_budget/enc_flap_adaptive_t0.99.onnx"),
    ("global: magnitude t95", "models/_global_budget/enc_magnitude_global_t0.95.onnx"),
    ("global: wanda uniform t95", "models/_global_budget/enc_wanda_uniform_t0.95.onnx"),
    ("global: FLAP adaptiv t95", "models/_global_budget/enc_flap_adaptive_t0.95.onnx"),
]
DECODERS = [
    ("decoder FP32", DECODER_FP32),
    ("decoder INT8", DECODER_INT8),
    ("decoder INT8 per-channel", DECODER_INT8_PC),
]


def load_test_audio(take):
    z = np.load(TEST_CACHE, allow_pickle=True)
    flat, lengths, texts = z["audio"], z["lengths"], z["texts"]
    waves, off = [], 0
    for ln in lengths:
        waves.append(flat[off:off + int(ln)])
        off += int(ln)
    split = str(z["split"][0]) if "split" in z else "?"
    return waves[:take], list(texts)[:take], split


def session(path, threads=THREADS):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def bootstrap_ci(per_sample, n=BOOTSTRAP, seed=0):
    """Percentile CI over utterances -- the unit of variation here."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_sample, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_ci(a, b, n=BOOTSTRAP, seed=1):
    """CI of the per-utterance difference: the paired test is what decides
    'indistinguishable', since utterance difficulty dominates the spread."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_cache():
    if not os.path.exists(OUT_JSON):
        return {}
    try:
        with open(OUT_JSON, encoding="utf-8-sig") as f:
            return {r["key"]: r for r in json.load(f)}
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        print(f"  ogohlantirish: kesh o'qilmadi ({type(exc).__name__}), noldan")
        return {}


def save_cache(rows):
    json.dump(list(rows.values()), open(OUT_JSON, "w"), indent=2)


def decode_all(dec_path, states, texts, tok, prompt_ids, label):
    dec = session(dec_path)
    wers, cers = [], []
    t0 = time.time()
    for i, (st, ref) in enumerate(zip(states, texts)):
        ids = greedy_decode(dec, st, prompt_ids)
        hyp = normalize(tok.decode(ids, skip_special_tokens=True))
        ref_n = normalize(ref)
        wers.append(error_rate(ref_n.split(), hyp.split()))
        cers.append(error_rate(list(ref_n), list(hyp)))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"      {label}: {i+1}/{len(states)} "
                  f"[{el:.0f}s, ~{el/(i+1)*(len(states)-i-1):.0f}s qoldi]", flush=True)
    del dec
    return wers, cers


def main():
    waves, texts, split = load_test_audio(N_EVAL)
    print(f"{len(waves)} namuna, '{split}' split "
          f"(kalibrlash 'validation' dan — kesishmaydi)")
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    tok = WhisperTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = [t for _, t in
                  tok.get_decoder_prompt_ids(language="uz", task="transcribe")]

    feats = [fe(w, sampling_rate=TARGET_SR, return_tensors="np")
             .input_features.astype(np.float32) for w in waves]
    print(f"{len(feats)} spektrogramma tayyor\n")

    rows = load_cache()
    if rows:
        print(f"keshdan {len(rows)} variant: {', '.join(rows)}\n")

    # The evaluation size is part of the cache identity: a run at a larger
    # N_EVAL must not silently reuse a smaller run's numbers.
    def cache_key(axis, label):
        return f"{axis}|{label}|n{N_EVAL}"

    def record(key, axis, label, wers, cers):
        lo, hi = bootstrap_ci(wers)
        rows[key] = {"key": key, "axis": axis, "variant": label,
                     "n": len(wers), "split": split,
                     "wer": float(np.mean(wers)), "cer": float(np.mean(cers)),
                     "wer_lo": lo, "wer_hi": hi, "per_sample_wer": wers}
        save_cache(rows)
        print(f"  WER={np.mean(wers):.4f} [{lo:.4f}, {hi:.4f}]  "
              f"CER={np.mean(cers):.4f}", flush=True)

    # --- encoder variants, decoder held at INT8 ---
    for label, path in ENCODERS:
        key = cache_key("encoder", label)
        if key in rows:
            print(f"[encoder: {label}] keshdan WER={rows[key]['wer']:.4f}")
            continue
        if not os.path.exists(path):
            print(f"[encoder: {label}] SKIP — {path} yo'q")
            continue
        print(f"[encoder: {label}] kodlanmoqda...", flush=True)
        enc = session(path)
        states = [enc.run(None, {"input_features": f})[0].astype(np.float32)
                  for f in feats]
        del enc
        wers, cers = decode_all(DECODER_INT8, states, texts, tok, prompt_ids, label)
        record(key, "encoder", label, wers, cers)
        del states

    # --- decoder variants, encoder held at FP32 ---
    need_dec = [(l, p) for l, p in DECODERS
                if cache_key("decoder", l) not in rows and os.path.exists(p)]
    if need_dec:
        print("\n[FP32 encoder holatlari tayyorlanmoqda...]", flush=True)
        enc = session(ENCODER_PATH)
        states_fp32 = [enc.run(None, {"input_features": f})[0].astype(np.float32)
                       for f in feats]
        del enc
        for label, path in need_dec:
            print(f"[decoder: {label}]", flush=True)
            wers, cers = decode_all(path, states_fp32, texts, tok, prompt_ids, label)
            record(cache_key("decoder", label), "decoder", label, wers, cers)
        del states_fp32

    # ===================== hisobot =====================
    print("\n" + "=" * 96)
    print(f"YAKUNIY WER/CER — {N_EVAL} namuna, '{split}' split, "
          f"95% bootstrap ishonch oralig'i")
    print("=" * 96)
    # Only this run's size; the cache may also hold rows from other N_EVAL.
    ordered = [r for r in rows.values() if r["n"] == len(waves)]
    for axis in ("encoder", "decoder"):
        sub = [r for r in ordered if r["axis"] == axis]
        if not sub:
            continue
        print(f"\n{axis.upper()} variantlari:")
        print(f"{'Variant':36s} {'WER':>8s} {'95% CI':>22s} {'CER':>8s}")
        print("-" * 96)
        for r in sub:
            ci = f"[{r['wer_lo']:.4f}, {r['wer_hi']:.4f}]"
            print(f"{r['variant']:36s} {r['wer']:8.4f} {ci:>22s} {r['cer']:8.4f}")

    print("\nJuftlik taqqoslash (paired bootstrap, bazaga nisbatan):")
    for axis in ("encoder", "decoder"):
        sub = [r for r in ordered if r["axis"] == axis]
        if len(sub) < 2:
            continue
        base = sub[0]
        for other in sub[1:]:
            d, lo, hi = paired_ci(other["per_sample_wer"], base["per_sample_wer"])
            verdict = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
            print(f"  {axis:8s} {other['variant']:34s} dWER={d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")

    # The paper's actual claim (Sec 4.9b, Table 16c) is not a comparison
    # against FP32 at all: it is whether the structural axis costs anything
    # ON TOP OF the best available quantizer. That is a B-vs-A test, so it
    # needs its own paired comparison rather than two separate FP32 deltas.
    by_variant = {r["variant"]: r for r in ordered}
    alloc = (by_variant.get("INT8 + bir xil rank"),
             by_variant.get("INT8 + taqsimlangan rank"))
    if all(alloc):
        u, g = alloc
        d, lo, hi = paired_ci(g["per_sample_wer"], u["per_sample_wer"])
        verdict = "SEZILARLI" if (lo > 0 or hi < 0) else "FARQLANMAYDI"
        print("\nRank taqsimoti (4.8-bo'lim) — teng byudjetda (203 MB):")
        print(f"  bir xil rank      WER {u['wer']:.4f}")
        print(f"  byudjet-optimal   WER {g['wer']:.4f}")
        print(f"  farq = {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")
        print(f"  (validation 80 da: 0.1719 -> 0.0729)")

    # Structural criteria at identical budget and identical quantizer, at two
    # pruning ratios. The second is the test that matters: at the mild budget
    # any criterion can find removable channels, so a tie there is weak
    # evidence; at the aggressive one the slack is gone.
    for title, keys in (
        ("mo'tadil byudjet, tau=0.99 (267 MiB, o'rtacha 17.1% olib tashlangan)",
         (("bizniki (funksional guruhlash)", "qisqartirish + GPTQ"),
          ("magnitude", "qisqartirish magnitude + GPTQ"),
          ("wanda", "qisqartirish wanda + GPTQ"))),
        ("agressiv byudjet, tau=0.95 (254 MiB, qatlamlarda 73% gacha)",
         (("bizniki (guruhlash + kompensatsiya)", "t95 bizniki + GPTQ"),
          ("bizniki, kompensatsiyasiz", "t95 bizniki, kompensatsiyasiz"),
          ("magnitude", "t95 magnitude + GPTQ"),
          ("wanda", "t95 wanda + GPTQ"))),
        ("tau=0.97 (261 MiB)",
         (("bizniki", "t97 bizniki + GPTQ (eps=5% tanlovi)"),
          ("magnitude", "t97 magnitude + GPTQ"),
          ("wanda", "t97 wanda + GPTQ"))),
        ("tau=0.93 (248 MiB)",
         (("bizniki", "t93 bizniki + GPTQ"),
          ("magnitude", "t93 magnitude + GPTQ"),
          ("wanda", "t93 wanda + GPTQ"))),
        ("tau=0.90 (237 MiB)",
         (("bizniki", "t90 bizniki + GPTQ"),
          ("magnitude", "t90 magnitude + GPTQ"),
          ("wanda", "t90 wanda + GPTQ"))),
        ("Wanda: asl (strukturasiz, 300 MiB) va kanal darajasiga "
         "moslashtirilgan variant",
         (("bizniki, tau=0.99 (267 MiB)", "qisqartirish + GPTQ"),
          ("Wanda mezoni, kanal (267 MiB)", "qisqartirish wanda + GPTQ"),
          ("Wanda asl, strukturasiz (300 MiB)", "Wanda asl (strukturasiz) t99"),
          ("bizniki, tau=0.95 (254 MiB)", "t95 bizniki + GPTQ"),
          ("Wanda mezoni, kanal (254 MiB)", "t95 wanda + GPTQ"),
          ("Wanda asl, strukturasiz (300 MiB)", "Wanda asl (strukturasiz) t95"))),
        ("kompensatsiya qiladigan usullar, B1 = 267 MiB",
         (("bizniki (vakilga qo'shish)", "qisqartirish + GPTQ"),
          ("gibrid (vakilga + bias)", "gibrid (bizniki + bias) t99"),
          ("FLAP (bias)", "FLAP t99"))),
        ("kompensatsiya qiladigan usullar, B2 = 254 MiB",
         (("bizniki (vakilga qo'shish)", "t95 bizniki + GPTQ"),
          ("gibrid (vakilga + bias)", "gibrid (bizniki + bias) t95"),
          ("FLAP (bias)", "FLAP t95"))),
        ("254 MiB ga ikki xil yo'l: mezonni bo'shatish yoki mexanizm qo'shish",
         (("tau ni 0.99 -> 0.95 bo'shatish", "t95 bizniki + GPTQ"),
          ("tau=0.99 qat'iy + fluktuatsiya bosqichi",
           "ikki bosqichli (kosinus + fluktuatsiya)"),
          ("(mos yozuvlar uchun) FLAP", "FLAP t95"))),
        # tau-free arms, with ALL THREE of our variants shown. The earlier
        # version compared each baseline's best configuration against our
        # plainest one, which understated us; showing the progression is both
        # fairer and more informative. Note the last of ours incorporates
        # FLAP's mechanism, so it supports "both mechanisms beat either alone"
        # rather than any claim of beating FLAP.
        ("TAU-DAN XOLI: 267 MiB, taqsimot har bir usulning o'ziniki",
         (("bizniki: asl (kosinus + vakil)", "qisqartirish + GPTQ"),
          ("bizniki: + bias tuzatmasi", "gibrid (bizniki + bias) t99"),
          ("magnitude (global tartiblash)", "global: magnitude t99"),
          ("Wanda (uniform, nashr etilgan)", "global: wanda uniform t99"),
          ("FLAP (moslashuvchan)", "global: FLAP adaptiv t99"))),
        ("TAU-DAN XOLI: 254 MiB, taqsimot har bir usulning o'ziniki",
         (("bizniki: asl (kosinus + vakil)", "t95 bizniki + GPTQ"),
          ("bizniki: + bias tuzatmasi", "gibrid (bizniki + bias) t95"),
          ("bizniki: ikki bosqichli (kompozitsiya)",
           "ikki bosqichli (kosinus + fluktuatsiya)"),
          ("magnitude (global tartiblash)", "global: magnitude t95"),
          ("Wanda (uniform, nashr etilgan)", "global: wanda uniform t95"),
          ("FLAP (moslashuvchan)", "global: FLAP adaptiv t95"))),
    ):
        crit = [(n, by_variant[v]) for n, v in keys if v in by_variant]
        if len(crit) < 2:
            continue
        base_c = crit[0][1]
        print(f"\nStrukturaviy mezonlar — {title}:")
        for name, r in crit:
            if r is base_c:
                print(f"  {name:32s} WER {r['wer']:.4f}   (mos yozuvlar)")
                continue
            d, lo, hi = paired_ci(r["per_sample_wer"], base_c["per_sample_wer"])
            v = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
            print(f"  {name:32s} WER {r['wer']:.4f}   bizdan farq "
                  f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]  {v}")

    # The trend itself: does the gap to each baseline widen with the budget?
    curve = [(0.99, 267, "qisqartirish + GPTQ", "qisqartirish magnitude + GPTQ",
              "qisqartirish wanda + GPTQ"),
             (0.97, 261, "t97 bizniki + GPTQ (eps=5% tanlovi)",
              "t97 magnitude + GPTQ", "t97 wanda + GPTQ"),
             (0.95, 254, "t95 bizniki + GPTQ", "t95 magnitude + GPTQ",
              "t95 wanda + GPTQ"),
             (0.93, 248, "t93 bizniki + GPTQ", "t93 magnitude + GPTQ",
              "t93 wanda + GPTQ"),
             (0.90, 237, "t90 bizniki + GPTQ", "t90 magnitude + GPTQ",
              "t90 wanda + GPTQ")]
    have = [(t, mib, by_variant[o], by_variant[m], by_variant[w])
            for t, mib, o, m, w in curve
            if o in by_variant and m in by_variant and w in by_variant]
    if have:
        print("\nMEZON FARQINING BYUDJETGA BOG'LIQLIGI (teng byudjet, "
              "bir xil GPTQ):")
        print(f"  {'tau':>5s} {'MiB':>5s} {'bizniki':>9s} {'magnitude':>10s} "
              f"{'wanda':>9s} {'wanda-biz':>10s} {'magn-biz':>10s}")
        for t, mib, o, m, w in have:
            print(f"  {t:5.2f} {mib:5d} {o['wer']:9.4f} {m['wer']:10.4f} "
                  f"{w['wer']:9.4f} {w['wer']-o['wer']:+10.4f} "
                  f"{m['wer']-o['wer']:+10.4f}")
        gaps = [w["wer"] - o["wer"] for _, _, o, _, w in have]
        mono = all(b >= a for a, b in zip(gaps, gaps[1:]))
        print(f"\n  wanda-bizniki farqi byudjet bilan monoton kengayadimi: "
              f"{'HA' if mono else 'YO`Q'}")

    pairs = [("GPTQ", "GPTQ yolg'iz", "qisqartirish + GPTQ"),
             ("RTN ", "RTN per-channel yolg'iz",
              "qisqartirish + RTN per-channel")]
    shown = [(t, by_variant[x], by_variant[y]) for t, x, y in pairs
             if x in by_variant and y in by_variant]
    if shown:
        print("\nMarkaziy da'vo (16c-jadval) — strukturaviy o'q kvantlash "
              "ustiga tekinga qo'shiladimi:")
        print("  (har juftlikda kvantlagichdan boshqa hamma narsa bir xil, "
              "xotira 300 -> 267 MiB)")
        for tag, a, b in shown:
            d, lo, hi = paired_ci(b["per_sample_wer"], a["per_sample_wer"])
            verdict = "SEZILARLI" if (lo > 0 or hi < 0) else "FARQLANMAYDI"
            print(f"  {tag}: qisqartirilgan - yolg'iz = {d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  {verdict}   "
                  f"(WER {a['wer']:.4f} -> {b['wer']:.4f})")

    n = len(waves)
    print(f"\nEslatma: 80 namunali oldingi o'lchovda markaziy da'voning oralig'i "
          f"[-0.0087, +0.0114] edi va u VALIDATION splitidan olingan — "
          f"kalibrlash\nbilan bir xil taqsimotdan. Bu yerda n={n}, mustaqil TEST "
          f"splitida, ya'ni oraliq taxminan {np.sqrt(n/80):.1f}x tor.")


if __name__ == "__main__":
    main()
