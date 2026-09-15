"""The framework's entry point: model + cache size + accuracy budget -> plan.

    python experiments/optimize.py --l3 12 --wer-max 0.20

Everything the cascade needs about a model can be read off its graph: how big
a layer is, how much of that layer the structural stage is allowed to touch,
and how often a weight is reused in one pass. This derives those three
numbers per part, hands them to the planner, and prints what the planner
decides -- what fitting would demand, whether it is reachable at all, and the
ordered ladder of configurations to try.

Two things are deliberately separate here.

`--l3` is not read from the machine by default; it is an argument. The point
of the framework is to answer "what would this model need on THAT machine",
and being able to ask about hardware you are not sitting in front of is most
of the value. Passing nothing uses the local L3, which is the special case.

Planning is instant and execution is not. Each rung costs roughly an hour to
build and half an hour to score, so `--run` walks the ladder lazily, from the
mildest rung up, and stops at the first configuration whose WER exceeds the
budget: every rung above it is strictly more aggressive and cannot recover.
The rung below the stop is the answer. Without `--run` this prints the plan
and builds nothing, which is the mode worth using while deciding.
"""

import argparse

from calib_utils import DECODER_PATH, ENCODER_PATH
from nnopt.cascade.cache_planner import MIB, PartSpec, feasibility, plan
from nnopt.hw.cache_topology import detect_cache_topology
from nnopt.profiler.blocks import breakdown
from nnopt.profiler.graph_profiler import profile_onnx_model

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
DEC_DIMS = {"batch_size": 1, "decoder_sequence_length": 16,
            "encoder_sequence_length": 1500}
# Whisper's encoder sees the whole 30 s mel window in one pass, so every weight
# is reused once per position; the decoder at batch 1 touches each weight once
# per token and has evicted it before the next token returns to that layer.
ENC_REUSE, DEC_REUSE = 1500, 1


def resolve_budget(base, wer_max=None, wer_delta=None, wer_eps=None,
                   metric="WER", higher_is_better=False):
    """Turn whichever form of accuracy budget was given into one ceiling.

    The forms are not interchangeable and the difference is large enough to
    change the answer: against a 0.1761 baseline, "0.05 worse" reads as 0.2261
    absolutely and 0.1849 relatively, which are different rungs of the ladder.
    So more than one is refused rather than silently ranked, and the returned
    text says which reading produced the number.

    "Worse" depends on the metric. A tolerance of 0.05 moves the ceiling UP
    from a word error rate and DOWN from an accuracy, so the sign follows
    `higher_is_better`; the returned text names the metric so the printed
    ceiling cannot be read against the wrong one.
    """
    given = [n for n, v in (("--wer-max", wer_max), ("--wer-delta", wer_delta),
                            ("--wer-eps", wer_eps)) if v is not None]
    if len(given) > 1:
        raise ValueError(f"aniqlik byudjetini bitta shaklda bering, "
                         f"{len(given)} tasi berildi: {', '.join(given)}")
    sgn = -1.0 if higher_is_better else 1.0
    rel = "<=" if not higher_is_better else ">="
    if wer_delta is not None:
        cap = base + sgn * wer_delta
        return cap, (f"{metric} {rel} {cap:.4f}  "
                     f"({base:.4f} {'-' if higher_is_better else '+'} "
                     f"{wer_delta:.4f} mutlaq)")
    if wer_eps is not None:
        cap = base * (1.0 + sgn * wer_eps)
        return cap, (f"{metric} {rel} {cap:.4f}  "
                     f"({base:.4f} x {1 + sgn*wer_eps:.4f} nisbiy)")
    if wer_max is not None:
        return wer_max, (f"{metric} {rel} {wer_max:.4f}  (qat'iy chegara, "
                         f"tayanch {base:.4f}, zaxira "
                         f"{sgn*(wer_max - base):+.4f})")
    return None, None


def derive_spec(name, path, dims, reuse, verbose=True, structural=True):
    """Read per-layer size and reducible share straight off the graph.

    Nothing here matches an operator NAME. The reducible block is found by the
    property the structural stage needs -- an expanding axis shared between two
    matrix operators -- so a transformer ASR model this code has never seen is
    described correctly as long as it has one. A model with no such block is
    reported with zero reducible bytes rather than a nearest guess, and the
    planner then honestly has only quantization to offer.
    """
    profs = [p for p in profile_onnx_model(path, free_dims=dims)
             if p.weight_initializer is not None]
    b = breakdown(path, profs)
    if verbose:
        widths = sorted({pr.width for pr in b.pairs})
        print(f"  {name}: {b.n_layers} qatlam, eng kattasi L{b.largest_layer}, "
              f"{len(b.pairs)} ta qisqartiriladigan juftlik"
              + (f", kenglik {widths}" if widths else " (topilmadi)"))
    if b.reducible_bytes == 0 and verbose:
        print(f"  DIQQAT: {name} da kengayadigan umumiy o'q topilmadi — "
              f"strukturaviy bosqich bu qismga hech narsa taklif qila olmaydi.")
    return PartSpec(name=name,
                    per_layer_bytes=b.per_layer_bytes,
                    n_layers=b.n_layers,
                    prunable_bytes=b.reducible_bytes,
                    reuse=reuse,
                    structural_supported=structural)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="whisper",
                    help="model profili (whisper | mbert); qismlar, "
                         "quruvchilar va metrika shundan olinadi")
    ap.add_argument("--encoder", default=ENCODER_PATH,
                    help="enkoder ONNX yo'li (profil qiymatini bekor qiladi)")
    ap.add_argument("--decoder", default=DECODER_PATH,
                    help="dekoder ONNX yo'li")
    # A flag rather than an empty --decoder: some shells drop the empty
    # string, which turned "encoder-only" into an argument-parsing error.
    ap.add_argument("--no-decoder", action="store_true",
                    help="enkoder-only (CTC) model sifatida rejalashtirish")
    ap.add_argument("--adapter", default="whisper",
                    help="baholash adapteri; hozircha faqat 'whisper'")
    ap.add_argument("--uniform-ladder", action="store_true",
                    help="strukturaviy pog'onalarni mezon nuqtalari o'rniga "
                         "bir xil nisbatlar bilan sanash (o'lchov bo'yicha "
                         "yomonroq; taqqoslash uchun saqlangan)")
    # Reuse is a property of the serving loop, not of the graph: it is how
    # many times a weight is revisited before it leaves the cache. An encoder
    # that consumes a whole utterance per pass reuses each weight once per
    # position; an autoregressive decoder at batch 1 does not reuse it at all.
    # Neither is inferable from the file, so both are arguments.
    ap.add_argument("--enc-reuse", type=int, default=ENC_REUSE,
                    help="enkoder vaznining bir o'tishdagi qayta ishlatilishi")
    ap.add_argument("--dec-reuse", type=int, default=DEC_REUSE,
                    help="dekoder vaznining qayta ishlatilishi (batch=1 da 1)")
    ap.add_argument("--l3", type=float, default=None,
                    help="maqsadli L3 hajmi, MiB (standart: shu mashinaniki)")
    ap.add_argument("--alpha", type=float, default=0.7)
    # Three ways to say the same thing, because the natural phrasing differs
    # by context and guessing between them changes which rung is chosen. The
    # paper states the budget relatively (eq. 20), a user asking for "0.05
    # worse" usually means absolutely, and a hard cap is sometimes just known.
    ap.add_argument("--wer-max", type=float, default=None,
                    help="ruxsat etilgan eng yuqori WER (mutlaq chegara)")
    ap.add_argument("--wer-delta", type=float, default=None,
                    help="tayanchdan ruxsat etilgan MUTLAQ farq, mas. 0.05")
    ap.add_argument("--wer-eps", type=float, default=None,
                    help="tayanchdan ruxsat etilgan NISBIY farq, mas. 0.05 "
                         "= 5%% (maqoladagi (20) tenglama)")
    ap.add_argument("--wer-base", type=float, default=None,
                    help="FP32 tayanch qiymati; berilmasa model profilidan "
                         "olinadi, --measure-base bilan esa o'lchanadi")
    ap.add_argument("--measure-base", action="store_true",
                    help="tayanchni zinapoyaning birinchi pog'onasidan "
                         "(o'zgartirilmagan model) o'lchash — nisbiy va "
                         "mutlaq byudjetlar o'shanda tanlov splitiga "
                         "nisbatan hisoblanadi")
    ap.add_argument("--run", action="store_true",
                    help="rejani qurish va o'lchash bilan bajarish")
    ap.add_argument("--n-eval", type=int, default=100,
                    help="tanlov uchun namunalar soni")
    ap.add_argument("--split", default="validation",
                    choices=("validation", "test"),
                    help="tanlov splitli; TEST ni yakuniy son uchun saqlang")
    ap.add_argument("--calib-split", default="validation",
                    choices=("validation", "test"),
                    help="kalibrlash to'plamining spliti")
    ap.add_argument("--calib-n", type=int, default=6,
                    help="kalibrlash namunalari soni")
    ap.add_argument("--calib-skip", type=int, default=0,
                    help="kalibrlashda tashlab ketiladigan namunalar")
    args = ap.parse_args()
    from model_profiles import get_profile
    try:
        profile = get_profile(args.model)
    except ValueError as e:
        raise SystemExit(str(e))

    # Each model has its own baseline; carrying Whisper's 0.1761 into an mBERT
    # run would set a ceiling that means nothing there.
    base = args.wer_base if args.wer_base is not None else profile.baseline_hint
    try:
        wer_max, budget_text = resolve_budget(
            base, wer_max=args.wer_max, wer_delta=args.wer_delta,
            wer_eps=args.wer_eps, metric=profile.metric,
            higher_is_better=profile.higher_is_better)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.l3 is None:
        g = detect_cache_topology().global_shared_cache()
        l3_bytes = float(g.size_bytes)
        print(f"L3 shu mashinadan olindi: {l3_bytes/MIB:.0f} MiB\n")
    else:
        l3_bytes = args.l3 * MIB

    from model_profiles import get_profile
    profile = get_profile(args.model)

    print(f"Model profili: {profile.name} (metrika: {profile.metric}, "
          f"{'katta yaxshi' if profile.higher_is_better else 'kichik yaxshi'})")
    print("Tuzilma grafikdan o'qilmoqda:")
    specs = []
    for src in profile.parts():
        if args.no_decoder and src.name == "dekoder":
            print("  dekoder o'tkazib yuborildi (--no-decoder)")
            continue
        # A profile may know its own sizes without a graph. Llama does: it has
        # no ONNX export here, but a decoder block's shapes are fully given by
        # its config, so reading them there is exact rather than a fallback.
        if hasattr(profile, "spec"):
            specs.append(profile.spec(src))
            print(f"  {src.name}: o'lcham model konfiguratsiyasidan olindi "
                  f"(ONNX grafigi emas)")
        else:
            specs.append(derive_spec(src.name, src.path, src.dims, src.reuse,
                                     structural=src.structural_supported))
        if not src.structural_supported:
            print(f"  {src.name}: strukturaviy o'q ulanmagan — zinapoyaga "
                  f"kiritilmaydi (bajarilish tekshiruvida qoladi)")
    print()

    # The structural rungs come from the model's own operating points, not
    # from round fractions -- see Treatment in cache_planner.
    rungs = None if args.uniform_ladder else profile.structural_ladder()
    p = plan(specs, l3_bytes, alpha=args.alpha, structural_ladder=rungs)
    print(p.summary())
    if rungs:
        print(f"  strukturaviy pog'onalar: "
              + ", ".join(f"{t} ({(1-k)*100:.0f}%)" for t, k in rungs))

    print("\nBajarilish tekshiruvi (INT8 dan keyin):")
    for v in feasibility(specs, l3_bytes, alpha=args.alpha):
        mark = " " if v.feasible else "!"
        print(f" {mark} {v.part:9s} talab {v.required:5.2f}x -> "
              f"{v.after_quant:5.2f}x, {v.note}")

    print(f"\nNomzodlar zinapoyasi ({len(p.rungs)} pog'ona, yumshoqdan qattiqqa):")
    print(f"{'#':>3s} {'qadam':34s} {'vazn':>9s} {'miss':>11s}  sig'adi")
    for r in p.rungs:
        fit = "ha" if r.all_fit else ", ".join(
            n for n, ok in r.fits.items() if not ok) + " yo'q"
        print(f"{r.index:3d} {r.step:34s} {r.total_bytes/MIB:8.0f}M "
              f"{r.miss/MIB:10.0f}M  {fit}")

    if budget_text:
        print(f"\nAniqlik byudjeti: {budget_text}")
    if not args.run:
        print("\nHech narsa qurilmadi. Bajarish uchun --run bering.")
        return
    if wer_max is None and not args.measure_base:
        raise SystemExit(
            "--run uchun aniqlik byudjeti kerak (--wer-max, --wer-delta yoki "
            "--wer-eps): to'xtash mezoni bo'lmasa zinapoya oxirigacha "
            "yuriladi va bu bir necha kun.")

    from asr_adapters import get_adapter
    from calib_utils import CalibSet
    from cascade_runner import walk

    calib = CalibSet(split=args.calib_split, skip=args.calib_skip,
                     n=args.calib_n)
    adapter = get_adapter(args.adapter)
    # The planner speaks in fractions of the PRUNABLE bytes; the grouping
    # removes channels from the FFN, which is that same block, so the two
    # coincide on these models. Passed explicitly so a model where they do not
    # coincide cannot be scored under a silent assumption.
    prunable_share = 1.0
    print(f"\nKalibrlash: {calib.tag} ({calib.path})")
    print(f"Zinapoya bo'ylab yurilmoqda ({args.split}, {args.n_eval} namuna). "
          f"Har bir pog'ona ~1.5 soat.\n")
    budget_from_base = None
    if args.measure_base:
        if args.wer_delta is None and args.wer_eps is None:
            raise SystemExit("--measure-base uchun --wer-delta yoki "
                             "--wer-eps kerak: mutlaq chegara tayanchga "
                             "bog'liq emas.")
        # The ceiling has to move the way the metric worsens. Written without
        # the sign it added the tolerance to an ACCURACY baseline, producing a
        # ceiling above the untouched model and a negative allowance, so every
        # rung was rejected and the walk returned the uncompressed model --
        # with no error, just a wrong answer.
        delta, eps, sgn = args.wer_delta, args.wer_eps, \
            (-1.0 if profile.higher_is_better else 1.0)
        budget_from_base = (lambda b: b + sgn * delta) if delta is not None \
            else (lambda b: b * (1.0 + sgn * eps))
        wer_max = None
        print("Tayanch zinapoyaning birinchi pog'onasidan o'lchanadi.")

    accepted, results = walk(p, wer_max, n_eval=args.n_eval,
                             split=args.split, prunable_share=prunable_share,
                             calib=calib, adapter=adapter,
                             budget_from_base=budget_from_base,
                             profile=profile)

    print("\n" + "=" * 78)
    if accepted is None:
        print("Hech bir konfiguratsiya byudjetga sig'madi.")
        if results:
            best = min(results, key=lambda rr: rr[1]["wer"])
            print(f"Eng yaxshisi: {best[0].step} — WER {best[1]['wer']:.4f}")
    else:
        rung, r = accepted
        print(f"TANLANDI: pog'ona {rung.index} — {rung.step}")
        print(f"  vazn {rung.total_bytes/MIB:.0f} MiB, "
              f"miss {rung.miss/MIB:.0f} MiB, keshga sig'ish: "
              f"{'ha' if rung.all_fit else 'yo`q'}")
        print(f"  {profile.metric} {r['score']:.4f} "
              f"({r.get('split', args.split)}, "
              f"{r.get('n', args.n_eval)} namuna)")
        for name, path in sorted(r.get("paths", {}).items()):
            print(f"  {name}: {path}")
        print("\nBu tanlov VALIDATION da qilindi. Maqolaga beriladigan yakuniy "
              "sonni TEST splitida alohida o'lchang.")


if __name__ == "__main__":
    main()
