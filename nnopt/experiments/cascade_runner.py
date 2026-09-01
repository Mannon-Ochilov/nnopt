"""Execution behind the planner: build a rung, score it, decide whether to go on.

`optimize.py` decides WHAT to try and in what order. This is what actually
produces an artifact for a rung and puts a number on it, and it is separated
from the planner for a practical reason: planning is arithmetic and takes
milliseconds, while a single rung costs about an hour to build and half an
hour to score. Anything that expensive needs its own caching, its own restart
behaviour, and its own honest account of what it cannot do.

Selection is scored on the VALIDATION split, never on TEST. This is not
hygiene for its own sake: an earlier round of this work chose between variants
on the same utterances it later reported, and the ordering it produced was not
the ordering the held-out split gave -- calibration-dependent methods looked
better than they were. The framework therefore picks on validation and leaves
TEST for the single final number.

What is wired, stated plainly so nothing here is mistaken for more:

  encoder, quantization      GPTQ (Sec 4.9a picks it over our calibrated scale)
  encoder, structural        functional grouping to a channel budget
  decoder, quantization      ORT dynamic INT8, per-channel
  decoder, structural        NOT wired -- the decoder's FFN is reduced by
                             low-rank factorization in this codebase, not by
                             channel removal, so a rung asking for it raises
                             rather than silently substituting a different
                             treatment

The walk stops at the first rung whose WER exceeds the budget and returns the
rung below it. Stopping is sound because the ladder is monotone in
aggressiveness: nothing above a failing rung can recover the accuracy it lost.
"""

import gc
import json
import os
import time

import numpy as np

MODELS = "models/_framework"
RESULTS = "experiments/results_cascade_runner.json"
ENC_GPTQ_ONLY = "models/_gptq/enc_gptq_only.onnx"
ENC_FP32 = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DEC_FP32 = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
DEC_INT8 = "models/_whole_net/dec_int8.onnx"
N_CHANNELS = 4096


class NotWired(RuntimeError):
    """A rung asks for a treatment this codebase does not implement."""


def model_mib(path):
    # A part can be a directory rather than a file: the Llama profile's
    # untouched rung is the source checkpoint, and getsize on a directory
    # returns the entry's own size, i.e. a few kilobytes reported as the size
    # of a 6 GiB model.
    if os.path.isdir(path):
        total = sum(os.path.getsize(os.path.join(d, f))
                    for d, _, fs in os.walk(path) for f in fs)
        return total / (1024 * 1024)
    total = os.path.getsize(path)
    if os.path.exists(path + ".data"):
        total += os.path.getsize(path + ".data")
    return total / (1024 * 1024)


# The criterion's own operating points, with the total share of FFN channels
# each retains and the artifact already built for it. Realising a structural
# rung from these rather than by cutting a uniform fraction is what the
# planner's `structural_ladder` exists for; the measured reason is on
# `Treatment` in nnopt.cascade.cache_planner.
TAU_LADDER = (
    ("tau=0.99", 0.830, "models/_gptq/enc_gptq_pruned.onnx"),
    ("tau=0.97", 0.799, "models/_ratio_sweep/enc_bizniki_tau0.97_gptq.onnx"),
    ("tau=0.95", 0.764, "models/_ratio_sweep/enc_bizniki_tau0.95_gptq.onnx"),
    ("tau=0.93", 0.729, "models/_ratio_sweep/enc_bizniki_tau0.93_gptq.onnx"),
    ("tau=0.90", 0.674, "models/_l3_12/enc_soft_tau0.9_gptq.onnx"),
)
TAU_BY_TAG = {tag: path for tag, _, path in TAU_LADDER}


def structural_ladder():
    """(tag, keep) pairs for the planner, mildest first."""
    return tuple((tag, keep) for tag, keep, _ in TAU_LADDER)


def build_encoder(bits, keep, prunable_share=1.0, calib=None, tag=""):
    """Encoder artifact for one treatment, cached on disk by its parameters.

    `keep` is the fraction of PRUNABLE bytes retained, which is what the
    planner speaks in; the grouping works in channels, and on these models the
    prunable block is the FFN, so the two coincide up to `prunable_share`.

    The calibration set is part of the cache key, not just of the build: two
    runs asking for the same keep ratio from different utterances produce
    different channel maps, and a filename that cannot tell them apart would
    hand back the wrong artifact without any error.
    """
    from calib_utils import CalibSet

    calib = calib or CalibSet()
    if bits == 32 and keep == 1.0:
        return ENC_FP32
    if bits != 8:
        raise NotWired(f"enkoder uchun INT{bits} ulanmagan (faqat 32 va 8)")
    if keep >= 1.0:
        if not os.path.exists(ENC_GPTQ_ONLY):
            raise NotWired(f"{ENC_GPTQ_ONLY} yo'q; avval GPTQ bazasini quring")
        return ENC_GPTQ_ONLY

    if tag:
        path = TAU_BY_TAG.get(tag)
        if path is None:
            raise NotWired(f"strukturaviy nuqta {tag!r} ulanmagan; "
                           f"mavjud: {sorted(TAU_BY_TAG)}")
        if not os.path.exists(path):
            raise NotWired(f"{tag} uchun artefakt yo'q: {path}")
        return path

    removal = round((1.0 - keep) * prunable_share, 2)
    os.makedirs(MODELS, exist_ok=True)
    path = f"{MODELS}/enc_int8_keep{keep:.2f}_{calib.tag}.onnx"
    if os.path.exists(path):
        return path

    from gptq_plus_pruning import build_gptq_model
    from l3_12_cascade import build_maps, load_maps

    print(f"  [enkoder] {removal*100:.0f}% kanal olib tashlanmoqda, "
          f"kalibrlash {calib.tag} (bu bosqich ~2 soat)", flush=True)
    stats = build_maps(removal=removal, out_dir=MODELS, calib=calib)
    pm = load_maps([s["layer"] for s in stats], removal=removal,
                   out_dir=MODELS, calib=calib)
    print("  [enkoder] GPTQ...", flush=True)
    build_gptq_model(f"{MODELS}/_tmp_enc.onnx", path, pm, f"keep{keep:.2f}")
    del pm
    gc.collect()
    return path


def build_decoder(bits, keep):
    if keep < 1.0:
        raise NotWired(
            "dekoder kanal bo'yicha qisqartirish ulanmagan: bu kodda dekoder "
            "FFN si past-rank yoyilma bilan kichraytiriladi, kanal olib "
            "tashlash bilan emas (experiments/l3_12_decoder.py ga qarang)")
    if bits == 32:
        return DEC_FP32
    if bits != 8:
        raise NotWired(f"dekoder uchun INT{bits} ulanmagan")
    if not os.path.exists(DEC_INT8):
        raise NotWired(f"{DEC_INT8} yo'q; avval INT8 dekoderni quring")
    return DEC_INT8


def build_rung_profile(rung, profile, calib):
    """Artifacts for one rung, one part at a time, through the profile."""
    return {name: profile.build(name, t.bits, t.keep, t.tag, calib)
            for name, t in rung.treatments.items()}


def build_rung(rung, prunable_share=1.0, calib=None):
    """Paths for one rung, or a NotWired explaining exactly what is missing.

    Parts are looked up by name rather than unpacked positionally, so an
    encoder-only model plans and builds through the same path -- it simply has
    no decoder entry, and gets None where the encoder-decoder case gets a
    second graph.
    """
    enc_t = rung.treatments.get("enkoder")
    if enc_t is None:
        raise NotWired(f"'enkoder' qismi yo'q; mavjud: "
                       f"{sorted(rung.treatments)}")
    enc = build_encoder(enc_t.bits, enc_t.keep, prunable_share, calib,
                        enc_t.tag)

    dec_t = rung.treatments.get("dekoder")
    dec = None if dec_t is None else build_decoder(dec_t.bits, dec_t.keep)
    return enc, dec


SPLIT_CACHE = {"validation": "models/_calib_cache/cv_uz_validation.npz",
               "test": "models/_calib_cache/cv_uz_test.npz"}


def load_split(split, take):
    """Utterances and references from a cached split."""
    path = SPLIT_CACHE.get(split)
    if path is None:
        raise ValueError(f"noma'lum split {split!r}")
    if not os.path.exists(path):
        raise NotWired(f"{path} yo'q; avval cache_cv_test_split.py ni yuriting")
    z = np.load(path, allow_pickle=True)
    flat, lengths, texts = z["audio"], z["lengths"], z["texts"]
    waves, off = [], 0
    for ln in lengths:
        waves.append(flat[off:off + int(ln)])
        off += int(ln)
    return waves[:take], list(texts)[:take]


def evaluate(enc_path, dec_path, n_eval, split="validation", bootstrap=2000,
             adapter=None):
    """WER with a percentile bootstrap interval over utterances.

    The encoder is run to completion before the decoder session opens: holding
    both graphs plus their activations at once is what has driven this machine
    into swap on the larger variants.
    """
    from asr_adapters import get_adapter
    from final_wer_testsplit import session
    from wer_cer_whole_network import error_rate

    adapter = adapter or get_adapter("whisper")
    waves, texts = load_split(split, n_eval)

    enc = session(enc_path)
    states = [enc.run(None, adapter.encoder_feed(w))[0].astype(np.float32)
              for w in waves]
    del enc
    gc.collect()

    dec = session(dec_path) if (adapter.has_decoder and dec_path) else None
    wers, cers = [], []
    for st, ref in zip(states, texts):
        hyp = adapter.normalize(adapter.transcribe(dec, st))
        ref_n = adapter.normalize(ref)
        wers.append(error_rate(ref_n.split(), hyp.split()))
        cers.append(error_rate(list(ref_n), list(hyp)))
    del dec, states
    gc.collect()

    a = np.asarray(wers, dtype=float)
    rng = np.random.default_rng(0)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(bootstrap)]
    return {"wer": float(a.mean()), "cer": float(np.mean(cers)),
            "wer_lo": float(np.percentile(means, 2.5)),
            "wer_hi": float(np.percentile(means, 97.5)),
            "n": len(wers), "split": split,
            "per_sample_wer": list(map(float, wers))}


def load_cache():
    if not os.path.exists(RESULTS):
        return {}
    try:
        with open(RESULTS, encoding="utf-8-sig") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def paired_delta(a, b, n=2000, seed=1):
    """Mean per-utterance difference and its percentile interval."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def check_disjoint(calib, split, n_eval, eval_skip=0):
    """Refuse to score a build on the utterances that shaped it.

    This is the one failure this pipeline has actually produced: selecting
    variants on the same data used for calibration made calibration-dependent
    methods look better than they were, and the ordering did not survive an
    independent split. It is cheap to check and expensive to discover late, so
    it raises rather than warns.
    """
    if calib.overlaps(split, eval_skip, n_eval):
        raise ValueError(
            f"kalibrlash to'plami ({calib.tag}) baholash namunalari bilan "
            f"kesishadi ({split}[{eval_skip}:{eval_skip+n_eval}]). "
            f"Boshqa split tanlang yoki --calib-skip ni "
            f"{eval_skip + n_eval} dan katta qiling.")


def walk(plan, wer_max=None, n_eval=100, split="validation", prunable_share=1.0,
         use_upper_bound=True, calib=None, adapter=None, budget_from_base=None,
         profile=None):
    """Climb the ladder until the accuracy budget breaks; return the last rung
    that held.

    The stopping test is PAIRED against the first rung, not absolute. An
    earlier version compared each rung's upper confidence limit against a
    ceiling derived from the baseline's point estimate, and running it end to
    end showed why that cannot work: on 100 utterances the interval is about
    +/-0.035 wide while a 5% margin is 0.005, so the very first rung -- the
    untouched model -- failed the budget derived from itself. Comparing a
    bound against a point estimate is comparing unlike things.

    A paired difference is the like-for-like comparison, and it is what the
    rest of this work uses to decide "indistinguishable": utterance difficulty
    cancels, so the interval on the DIFFERENCE is far tighter than on either
    WER alone. The rule is therefore: stop when the upper limit of
    (rung - baseline) exceeds the allowed margin.

    `budget_from_base` turns a RELATIVE budget into a ceiling using the
    baseline this run actually measures, rather than one supplied from
    elsewhere. It matters because the ladder's first rung IS the untouched
    model: taking its WER as the base makes "5% worse" mean 5% worse on the
    split being selected on. Passing a constant instead lets a baseline
    measured on one split silently set the ceiling for another -- here the
    difference between the test and validation baselines is larger than the
    gap between neighbouring rungs.

    `profile` supplies the model-specific builders and scorer, and with them
    the DIRECTION of the metric. Whisper's word error rate improves downwards
    and mBERT's masked-token accuracy improves upwards; the comparison is
    written for the first, so for the second every difference is negated
    before it meets the margin. Without that, a budget rule would accept
    precisely the rungs it exists to reject.
    """
    from calib_utils import CalibSet

    if profile is None:
        from model_profiles import get_profile
        profile = get_profile("whisper")
    sign = -1.0 if profile.higher_is_better else 1.0
    calib = calib or CalibSet()
    # How calibration and evaluation data are kept apart is a property of the
    # model's data, not of the runner: Whisper indexes both into one cached
    # split and needs range arithmetic, while mBERT splits its text upstream.
    # The profile owns the check so neither is silently skipped.
    profile.check_data_split(calib, split, n_eval)
    cache = load_cache()
    accepted, results = None, []
    baseline, margin = None, None

    for rung in plan.rungs:
        # Resolve the artifacts BEFORE consulting the cache, and key on them.
        # Keying on the treatment encoding instead ties a measurement to how
        # the planner happened to describe it, so a refactor that produced
        # byte-identical models -- renaming a field, adding a tag -- silently
        # discarded hours of decoding. An artifact path is what was actually
        # measured. Resolution is cheap whenever the file exists, which is
        # exactly when a cached measurement could exist.
        try:
            paths = build_rung_profile(rung, profile, calib)
        except NotWired as e:
            print(f"[{rung.index:2d}] {rung.step:32s} ULANMAGAN — {e}")
            print("     zinapoya shu yerda to'xtaydi.")
            break
        enc = paths.get("enkoder")
        dec = paths.get("dekoder")
        key = calib.tag + "|" + "|".join(f"{k}={paths[k]}"
                                         for k in sorted(paths))
        if key in cache:
            r = cache[key]
            print(f"[{rung.index:2d}] {rung.step:32s} keshdan "
                  f"{profile.metric}={r['score']:.4f}", flush=True)
        else:
            mib = sum(model_mib(p) for p in paths.values() if p)
            print(f"[{rung.index:2d}] {rung.step:32s} {mib:6.0f} MiB — "
                  f"baholanmoqda ({split}, {n_eval})...", flush=True)
            t0 = time.time()
            r = profile.evaluate(paths, n_eval, split, calib)
            # split/n are filled in here rather than by each profile: the
            # runner is what knows them, and a profile that forgot one used to
            # surface as a KeyError in the final report, after the work.
            r.update({"enc": enc, "dec": dec, "paths": paths, "mib": mib,
                      "step": rung.step, "index": rung.index,
                      "calib": calib.tag, "metric": profile.metric,
                      "split": split, "n": n_eval,
                      "seconds": time.time() - t0})
            cache[key] = r
            json.dump(cache, open(RESULTS, "w"), indent=2)
            print(f"     {profile.metric}={r['score']:.4f}  "
                  f"[{r['seconds']:.0f}s]", flush=True)

        results.append((rung, r))
        if baseline is None:
            baseline = r
            if wer_max is None:
                if budget_from_base is None:
                    raise ValueError("wer_max yoki budget_from_base kerak")
                wer_max = budget_from_base(r["score"])
            # The margin must live in the SAME units as the paired
            # per-sample differences. For WER and accuracy they coincide
            # with the score. Perplexity does not: the score is
            # exp(mean NLL) while the resampled quantity is per-segment
            # NLL, so a ceiling of base*1.05 is a margin of ln(1.05) in
            # NLL space, not 0.05*base in perplexity space. Comparing the
            # two ACCEPTED a rung that was over budget -- forced 10% on
            # Llama scored 8.0687 against a 7.9240 ceiling and passed,
            # because its NLL delta 0.0669 was read against a perplexity
            # margin of 0.3773. Profiles whose score is a nonlinear
            # summary provide the conversion.
            if hasattr(profile, "paired_margin"):
                margin = profile.paired_margin(r["score"], wer_max)
            else:
                margin = sign * (wer_max - r["score"])
            print(f"     tayanch: {r['score']:.4f}, ruxsat etilgan "
                  f"yomonlashuv {margin:+.4f} ({profile.metric} chegarasi "
                  f"{wer_max:.4f})", flush=True)
            accepted = (rung, r)
            continue

        # Signed so that a positive difference always means WORSE, whichever
        # direction the metric improves in.
        d, lo, hi = paired_delta(
            [sign * v for v in r["per_sample"]],
            [sign * v for v in baseline["per_sample"]])
        r["dwer"], r["dwer_lo"], r["dwer_hi"] = d, lo, hi
        judged = hi if use_upper_bound else d
        print(f"     tayanchga nisbatan yomonlashuv={d:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]", flush=True)

        # Warn once, on the first real comparison, if the budget is finer than
        # the comparison can resolve. It has to be measured on the PAIRED
        # difference: an earlier version used the spread of absolute WER,
        # which is several times wider, and duly announced that nothing could
        # be proven while the walk went on accepting rungs.
        if len(results) == 2:
            half = 0.5 * (hi - lo)
            if margin < half:
                print(f"     DIQQAT: zaxira {margin:.4f} juftlik "
                      f"taqqoslashning ajratish chegarasidan ({half:.4f}) "
                      f"kichik — hech bir pog'ona byudjet ichida ekani "
                      f"ishonchli isbotlanmaydi. Byudjetni kengaytiring "
                      f"yoki namunani ko'paytiring (chegara ~ 1/sqrt(n)).",
                      flush=True)
        if judged > margin:
            print(f"     byudjetdan chiqdi ({judged:+.4f} > {margin:+.4f}) — "
                  f"to'xtatildi")
            break
        accepted = (rung, r)

    return accepted, results
