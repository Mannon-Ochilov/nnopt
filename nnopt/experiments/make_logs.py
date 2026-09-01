"""Emit the raw measurement logs from the stored result files.

The published record is the log, not a narrative. Every configuration that was
measured appears, with every field that was recorded, in the order the source
file holds it -- including runs that were later superseded. A result file that
exists is a measurement that happened; leaving it out of the log because a
later run disagreed would hide the disagreement rather than document it.

The latency library is the case that makes this matter. It was measured twice
under different protocols and the two runs disagree on one row, so both are
written out with their protocol stated, and the reader can see which is which
and why they differ.

Usage:  python experiments/make_logs.py
"""

import datetime
import json
import os

EXP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(EXP), "logs")

MACHINE = ("Intel Core i7-11850H (Tiger Lake H, 8C/16T)\n"
           "L1d 48 KiB/core  L2 1.25 MiB/core  L3 24 MiB shared\n"
           "64 GiB RAM, Windows 11, onnxruntime 1.28.0 CPUExecutionProvider\n"
           "Python 3.12.8, numpy 2.4.6, torch 2.13.0+cpu, transformers 4.57.6")


def load(name):
    p = os.path.join(EXP, name)
    if not os.path.exists(p):
        return None, None
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f), datetime.datetime.fromtimestamp(
            os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")


def header(fh, title, source, when, protocol):
    fh.write("=" * 78 + "\n")
    fh.write(title + "\n")
    fh.write("=" * 78 + "\n")
    fh.write(f"source   : experiments/{source}\n")
    fh.write(f"recorded : {when}\n")
    fh.write("machine  :\n")
    for line in MACHINE.splitlines():
        fh.write(f"           {line}\n")
    fh.write("protocol :\n")
    for line in protocol.strip().splitlines():
        fh.write(f"           {line.strip()}\n")
    fh.write("=" * 78 + "\n\n")


def w(path, fn):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, path)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        if fn(fh) is False:
            fh.close()
            os.remove(p)
            print(f"  o'tkazib yuborildi: {path} (manba yo'q)")
            return
    print(f"  {os.path.relpath(p, os.path.dirname(EXP))}")


# ---------------------------------------------------------------- latency ---

def latency_blocked(fh):
    d, when = load("results_latency_library.json")
    if d is None:
        return False
    header(fh, "ENCODER LATENCY LIBRARY -- RUN 1, BLOCKED DESIGN",
           "results_latency_library.json", when, """
           Each configuration measured to completion before moving to the
           next (blocked by configuration), sessions cached between repeats.
           Single thread. This is the run reported in Table 8 of the article.
           Superseded by run 2; both are kept, see logs/latency_library_
           interleaved.log for the difference and its cause.
           """)
    fh.write(f"{'key':<8} {'label':<40} {'MiB':>9} {'ms':>10}\n")
    fh.write("-" * 70 + "\n")
    rows = sorted(d.values(), key=lambda r: -r["mib"])
    for r in rows:
        fh.write(f"{r['key']:<8} {r['label'][:40]:<40} "
                 f"{r['mib']:9.2f} {r['ms']:10.2f}\n")
    base = d.get("int8")
    if base:
        fh.write("\nratios against blind INT8 baseline "
                 f"({base['ms']:.1f} ms):\n")
        for r in rows:
            fh.write(f"  {r['key']:<8} {base['ms'] / r['ms']:6.2f}x\n")


def latency_interleaved(fh):
    d, when = load("results_latency_library_interleaved.json")
    if d is None:
        return False
    p = d.get("protokol", {})
    header(fh, "ENCODER LATENCY LIBRARY -- RUN 2, INTERLEAVED DESIGN",
           "results_latency_library_interleaved.json", when, f"""
           Configurations cycled A-B-C-A-B-C for {p.get('raundlar', '?')} rounds,
           {p.get('qizdirish', '?')} warm-up passes, {p.get('oqimlar', '?')} thread,
           fresh session per measurement.
           Re-measurement of run 1. Eleven of twelve rows reproduce within
           run-to-run spread; the blind INT8 row does not (8658.2 ms in run 1,
           6981.2 ms here, -19.4%). Ratios computed against that baseline
           differ correspondingly between the two runs.
           """)
    rows = sorted(d["natijalar"].values(), key=lambda r: -r["mib"])
    fh.write(f"{'key':<8} {'label':<38} {'MiB':>9} {'ms':>10} "
             f"{'spread%':>8} {'vs INT8':>8}\n")
    fh.write("-" * 86 + "\n")
    for r in rows:
        fh.write(f"{r['key']:<8} {r['label'][:38]:<38} {r['mib']:9.2f} "
                 f"{r['ms']:10.2f} {r.get('spread_pct', 0):8.2f} "
                 f"{r.get('speedup_vs_int8', 0):7.2f}x\n")
    fh.write("\nper-round measurements (ms):\n")
    for r in rows:
        runs = " ".join(f"{v:8.1f}" for v in r.get("runs_ms", []))
        fh.write(f"  {r['key']:<8} {runs}\n")


# -------------------------------------------------------------------- WER ---

def wer_all(fh):
    when = None
    header(fh, "WORD ERROR RATE -- ALL MEASURED CONFIGURATIONS",
           "results_*.json (several)", "see per-block source", """
           Uzbek Common Voice. Calibration and evaluation utterances disjoint.
           Confidence intervals: percentile bootstrap over utterances, 2000
           resamples, seed 0.
           WER is the mean of per-utterance error rates (macro average), so
           the resampling unit is the utterance.
           Blocks differ in evaluation set and are NOT comparable across
           blocks: the same FP32 encoder scores 0.0417 on the 24-utterance
           smoke set, 0.0931 on 100 validation utterances and 0.1793 on the
           300-utterance test split.
           """)

    def block(title, source, rows):
        fh.write(f"\n--- {title} ---\n")
        fh.write(f"source: experiments/{source}\n\n")
        fh.write(f"{'configuration':<46} {'MiB':>8} {'WER':>8} "
                 f"{'wer_lo':>8} {'wer_hi':>8} {'CER':>8} {'n':>5}\n")
        fh.write("-" * 96 + "\n")
        for r in rows:
            def f(v, w=8, d=4):
                return f"{v:{w}.{d}f}" if isinstance(v, (int, float)) \
                    else " " * (w - 1) + "-"
            fh.write(f"{r[0][:46]:<46} {f(r[1], 8, 1)} {f(r[2])} "
                     f"{f(r[3])} {f(r[4])} {f(r[5])} "
                     f"{r[6] if r[6] else '-':>5}\n")

    d, when = load("results_config_library.json")
    if d:
        block("ENCODER CONFIGURATION LIBRARY (decoder fixed at INT8; "
              "sizes are whole-model)", "results_config_library.json",
              [(k, v.get("mib"), v["wer"], v.get("wer_lo"), v.get("wer_hi"),
                v.get("cer"), v.get("n"))
               for k, v in d.items() if isinstance(v, dict) and "wer" in v])

    d, _ = load("results_whole_model_cascade.json")
    if isinstance(d, list):
        block("WHOLE MODEL, ENCODER + DECODER",
              "results_whole_model_cascade.json",
              [(r.get("variant", "?"), r.get("mib"), r["wer"], r.get("wer_lo"),
                r.get("wer_hi"), r.get("cer"), r.get("n")) for r in d])

    d, _ = load("results_final_wer_testsplit.json")
    if isinstance(d, list):
        block("ALL SCORED VARIANTS, TEST SPLIT",
              "results_final_wer_testsplit.json",
              [(f"{r.get('axis','')}: {r.get('variant','')}".strip(": "),
                r.get("mib"), r["wer"], r.get("wer_lo"), r.get("wer_hi"),
                r.get("cer"), r.get("n")) for r in d])

    d, _ = load("results_l3_12_eval.json")
    if isinstance(d, dict):
        block("L3 = 12 MiB SCENARIO", "results_l3_12_eval.json",
              [(k, v.get("mib"), v["wer"], v.get("wer_lo"), v.get("wer_hi"),
                v.get("cer"), v.get("n"))
               for k, v in d.items() if isinstance(v, dict) and "wer" in v])

    for title, src, names in (
        ("STAGE ORDER A/B", "results_whisper_order_ab.json", None),
        ("BIAS CORRECTION ABLATION", "results_bias_correction_e2e.json", None),
        ("PRUNING x QUANTIZER (2x2)", "results_gptq_pruning.json", None),
    ):
        d, _ = load(src)
        if isinstance(d, dict):
            block(title, src,
                  [(k, v.get("mib"), v["wer"], v.get("wer_lo"),
                    v.get("wer_hi"), v.get("cer"), v.get("n"))
                   for k, v in d.items()
                   if isinstance(v, dict) and "wer" in v])


# --------------------------------------------------------------- counters ---

def counters(fh):
    d, when = load("results_vtune_whole_model.json")
    if not isinstance(d, list):
        return False
    header(fh, "HARDWARE COUNTERS -- WHOLE MODEL, uarch-exploration",
           "results_vtune_whole_model.json", when, """
           Intel VTune uarch-exploration, single thread, 8 s sampling window.
           Encoder driven at 1500 positions per pass, decoder at one
           autoregressive step -- each in its natural regime, since that
           asymmetry is the thing being measured.
           Memory Bound and the L1/L2/L3/DRAM Bound figures are shares of
           pipeline slots, not miss counts.
           """)
    fh.write(f"{'variant':<40} {'MiB':>8} {'ms/iter':>10} {'Mem%':>6} "
             f"{'L1%':>5} {'L2%':>5} {'L3%':>5} {'DRAM%':>6} {'CPI':>6}\n")
    fh.write("-" * 100 + "\n")
    for r in d:
        def f(k, w=6, dd=1):
            v = r.get(k)
            return f"{v:{w}.{dd}f}" if isinstance(v, (int, float)) \
                else " " * (w - 1) + "-"
        fh.write(f"{r['variant'][:40]:<40} {r['mib']:8.1f} "
                 f"{f('ms_per_iter', 10, 2)} {f('memory_bound')} "
                 f"{f('l1_bound', 5)} {f('l2_bound', 5)} {f('l3_bound', 5)} "
                 f"{f('dram_bound')} {f('cpi', 6, 3)}\n")

    d2, when2 = load("results_llc_miss_count.json")
    if isinstance(d2, list) and d2:
        fh.write("\n\n" + "=" * 78 + "\n")
        fh.write("LAST-LEVEL-CACHE MISS COUNTS, PER FORWARD PASS\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"source   : experiments/results_llc_miss_count.json\n")
        fh.write(f"recorded : {when2}\n")
        fh.write("protocol : VTune runsa, event-config MEM_LOAD_RETIRED.L3_MISS,\n"
                 "           MEM_LOAD_RETIRED.L3_HIT, LONGEST_LAT_CACHE.MISS.\n"
                 "           Each variant profiled twice -- once with the timed\n"
                 "           loop, once with seconds=0 (same model load and\n"
                 "           warm-up, no measured iterations) -- and the\n"
                 "           difference divided by the iteration count, so\n"
                 "           setup traffic is removed from the absolute counts.\n\n")
        fh.write(f"{'variant':<32} {'MiB':>8} {'LLC-load-miss':>15} "
                 f"{'miss rate':>10} {'all L3 miss':>14} {'ms/iter':>9}\n")
        fh.write("-" * 92 + "\n")
        for r in d2:
            fh.write(f"{r['variant'][:32]:<32} {r['mib']:8.1f} "
                     f"{r['llc_load_misses_per_iter']:15,.0f} "
                     f"{r['llc_load_miss_rate']:10.3f} "
                     f"{r['llc_all_misses_per_iter']:14,.0f} "
                     f"{r['ms_per_iter']:9.1f}\n".replace(",", " "))


# ------------------------------------------------------------- redundancy ---

def redundancy_reported(fh):
    """The runs behind Tables 11 and 12 of the article."""
    header(fh, "FUNCTIONAL REDUNDANCY -- AS REPORTED IN THE ARTICLE",
           "results_ffn_prune.json, results_mbert.json, results_llama.json",
           "see per-block source", """
           Share of FFN channels the cosine criterion endorses merging, with
           the contribution gate at eps_thr = 0.5.
           These are the runs quoted in Tables 11 and 12. Calibration
           activations were captured WITHOUT an attention/padding mask, so
           padded positions enter the response vectors: Whisper pads every
           clip to a 30 s window and mBERT to 128 tokens. See
           redundancy_masked.log for a re-measurement with the mask applied
           and for how much it moves each figure.
           """)

    d, when = load("results_ffn_prune.json")
    if d:
        s = d["stats"] if isinstance(d, dict) else d
        fr = [r["fraction"] * 100 for r in s]
        fh.write("--- Whisper-medium encoder, tau = 0.99, all 24 layers ---\n")
        fh.write(f"source: experiments/results_ffn_prune.json   recorded: {when}\n\n")
        fh.write(f"{'layer':>7} {'kept':>8} {'removed':>9} {'share':>9}\n")
        for r in s:
            fh.write(f"{'L' + str(r['layer']):>7} {r['kept']:8d} "
                     f"{r['removed']:9d} {r['fraction'] * 100:8.2f}%\n")
        fh.write(f"\nmean {sum(fr) / len(fr):.2f}%   peak {max(fr):.2f}% "
                 f"(L{fr.index(max(fr))})\n")
        fh.write("-> Table 11: '17.1%, peak 58.0%'\n\n")

    d, when = load("results_ffn_redundancy.json")
    if d:
        taus = ["0.99", "0.95", "0.9", "0.8", "0.7"]
        fh.write("--- Whisper encoder, tau sweep on a 7-layer sample ---\n")
        fh.write(f"source: experiments/results_ffn_redundancy.json   "
                 f"recorded: {when}\n\n")
        fh.write(f"{'operator':<26}" + "".join(f"{'t=' + t:>10}" for t in taus)
                 + "\n")
        for name, v in d.items():
            short = name.replace("/MatMul", "").lstrip("/")
            row = f"{short[:26]:<26}"
            for t in taus:
                e = v["taus"].get(t)
                row += f"{e['fraction'] * 100:9.2f}%" if e else " " * 10
            fh.write(row + "\n")
        fh.write("\n")

    d, when = load("results_mbert.json")
    if d:
        taus = ["0.99", "0.95", "0.9"]
        fr = [v["0.99"]["fraction"] * 100 for v in d.values()]
        fh.write("--- mBERT, 12 layers, 3072 FFN channels ---\n")
        fh.write(f"source: experiments/results_mbert.json   recorded: {when}\n\n")
        fh.write(f"{'operator':<44}" + "".join(f"{'t=' + t:>10}" for t in taus)
                 + "\n")
        for name, v in d.items():
            row = f"{name.replace('/MatMul', '').lstrip('/')[:44]:<44}"
            for t in taus:
                e = v.get(t)
                row += f"{e['fraction'] * 100:9.2f}%" if e else " " * 10
            fh.write(row + "\n")
        fh.write(f"\nmean {sum(fr) / len(fr):.2f}%   peak {max(fr):.2f}% "
                 f"at tau = 0.99\n")
        fh.write("-> Table 11: '0.1%, peak 0.7%'\n\n")

    d, when = load("results_llama.json")
    if d:
        red = d["redundancy"]
        taus = ["0.99", "0.95", "0.9"]
        fr = [v["0.99"]["fraction"] * 100 for v in red.values()]
        fh.write("--- open_llama_3b, gated FFN, 8640 channels ---\n")
        fh.write(f"source: experiments/results_llama.json   recorded: {when}\n")
        fh.write("note: calibrated on Uzbek transcripts padded to 128 tokens\n\n")
        fh.write(f"{'layer':<12}" + "".join(f"{'t=' + t:>10}" for t in taus)
                 + f"{'e_loc(0.99)':>13}\n")
        for name, v in red.items():
            row = f"{name:<12}"
            for t in taus:
                e = v.get(t)
                row += f"{e['fraction'] * 100:9.2f}%" if e else " " * 10
            row += f"{v['0.99'].get('e_loc', 0):13.5f}"
            fh.write(row + "\n")
        fh.write(f"\nmean {sum(fr) / len(fr):.2f}%   peak {max(fr):.2f}% "
                 f"at tau = 0.99\n")
        fh.write("-> Table 11: '0.6%, peak 3.4%'\n\n")
        fh.write("cache decision per operator (same file):\n")
        for k, v in d.get("cache", {}).items():
            fh.write(f"  {k:<22} {v['params']:>11,} params  "
                     f"INT8 {v['int8_mib']:7.2f} MiB  need {v['need']:5.2f}x  "
                     f"{v['case']}\n".replace(",", " "))


def redundancy(fh):
    wrote = False
    header(fh, "FUNCTIONAL REDUNDANCY -- POST-SUBMISSION RE-MEASUREMENT",
           "results_global_tau_range.json and others", "see per-block source",
           """
           Share of FFN channels the cosine criterion endorses merging at a
           given tau, with the contribution gate at eps_thr = 0.5.
           One global tau applied to every layer: the removal share is the
           output, not an input.
           These runs apply the attention/padding mask that the runs in
           redundancy_reported.log omit, and they cover every layer rather
           than a sample. They were measured after the article was submitted
           and are not the figures it quotes; both are published so the
           difference is visible rather than implied.
           """)

    for title, src, chan in (
        ("WHISPER-MEDIUM ENCODER (24 layers, 4096 FFN channels)",
         "results_global_tau_range.json", 4096),
        ("open_llama_3b, SIGNED COSINE (26 layers, 8640 channels)",
         "results_llama_global_tau.json", 8640),
        ("open_llama_3b, ABSOLUTE COSINE (anti-collinear pairs accepted)",
         "results_llama_global_tau_abs.json", 8640),
        ("mBERT (12 layers, 3072 channels)",
         "results_mbert_padding_tau.json", 3072),
    ):
        d, when = load(src)
        if not isinstance(d, list) or not d:
            continue
        wrote = True
        taus = sorted({r["tau"] for r in d}, reverse=True)
        cfgs = sorted({r.get("config", "-") for r in d})
        fh.write(f"\n--- {title} ---\n")
        fh.write(f"source: experiments/{src}   recorded: {when}\n\n")
        for cfg in cfgs:
            if cfg != "-":
                fh.write(f"[{cfg}]\n")
            rows = [r for r in d if r.get("config", "-") == cfg]
            layers = sorted({r["layer"] for r in rows})
            fh.write(f"{'layer':>7}" + "".join(f"{'t=' + str(t):>11}"
                                               for t in taus) + "\n")
            for li in layers:
                line = f"{'L' + str(li):>7}"
                for t in taus:
                    r = next((x for x in rows if x["layer"] == li
                              and x["tau"] == t), None)
                    line += f"{r['share'] * 100:10.2f}%" if r else " " * 11
                fh.write(line + "\n")
            line = f"{'GLOBAL':>7}"
            for t in taus:
                sel = [r for r in rows if r["tau"] == t]
                line += (f"{sum(r['merged'] for r in sel) / sum(r['channels'] for r in sel) * 100:10.2f}%"
                         if sel else " " * 11)
            fh.write(line + "\n\n")
    return wrote or False


def criterion_masked(fh):
    return _criterion(fh, "results_mbert_criterion_masked.json",
                      "REMOVAL CRITERION COMPARISON -- mBERT, MASKED "
                      "CALIBRATION (post-submission)",
                      "Re-measurement with the attention mask applied. Not "
                      "the run the article quotes; see "
                      "criterion_comparison.log for that one.")


def criterion(fh):
    return _criterion(fh, "results_mbert_criterion.json",
                      "REMOVAL CRITERION COMPARISON -- mBERT, EQUAL 20% BUDGET",
                      "This is the run reported in the article. Calibration "
                      "activations captured without a padding mask.")


def _criterion(fh, src, title, note):
    d, when = load(src)
    if not isinstance(d, list) or not d:
        return False
    header(fh, title, src, when, f"""
           {note}
           """ + """
           Four arms at the same budget, scored on held-out text by relative
           output error E_loc = ||Y - Y'|| / ||Y||. Lower is better.
             kosinus        cosine grouping forced down to the budget
             ikki bosqichli strict cosine, then fluctuation, then bias
             fluktuatsiya   fluctuation only, then bias (FLAP-shaped)
             rank           activation-aware low-rank at matched parameters
           st1 is how many channels strict cosine (tau = 0.99) found before
           the second stage ran.
           """)
    arms = ["kosinus", "ikki bosqichli", "fluktuatsiya", "rank"]
    fh.write(f"{'layer':>6} {'tau':>6}" + "".join(f"{a:>16}" for a in arms)
             + f"{'st1':>6}\n")
    fh.write("-" * 84 + "\n")
    for r in d:
        fh.write(f"{'L' + str(r['layer']):>6} {r['tau']:6.2f}"
                 + "".join(f"{r[a]:16.4f}" for a in arms)
                 + f"{r['stage1_removed']:6d}\n")
    fh.write("-" * 84 + "\n")
    fh.write(f"{'MEAN':>13}"
             + "".join(f"{sum(r[a] for r in d) / len(d):16.4f}" for a in arms)
             + "\n")


def main():
    print("loglar yozilmoqda:")
    # Named so the pairing is visible in a directory listing: for each
    # quantity the article reports, the run it reports and any later
    # re-measurement sit next to each other.
    w("latency_library_blocked.log", latency_blocked)
    w("latency_library_interleaved.log", latency_interleaved)
    w("wer_all_configurations.log", wer_all)
    w("hardware_counters.log", counters)
    w("redundancy_reported.log", redundancy_reported)
    w("redundancy_masked.log", redundancy)
    w("criterion_comparison.log", criterion)
    w("criterion_comparison_masked.log", criterion_masked)


if __name__ == "__main__":
    main()
