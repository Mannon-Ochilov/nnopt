"""Raw last-level-cache miss COUNTS, not stall shares.

The manuscript currently supports the memory-wall chain with top-down "bound"
metrics: Memory Bound 12.8% -> 10.1%, L3 Bound 2.4% -> 1.0%, and absolute
memory stall time 1759 -> 731 ms. Those measure how much TIME is lost to
memory, which is one step removed from the claim the framework actually
makes. The claim is about traffic: weights are read once per forward pass,
so the bytes a pass must move equal the model's size, and compressing the
model should cut the number of lines fetched from DRAM in proportion.

That prediction is sharp and has never been tested:

    misses(cascade) / misses(FP32)  ~=  MiB(FP32) / MiB(cascade)

For the whole model that ratio is 2915 / 704.9 = 4.14x. Memory stall time
fell only 2.41x, so if misses do fall ~4x the gap is exposure, not traffic --
a smaller number of misses is less well hidden by the prefetcher. If instead
misses fall ~2.4x, the traffic model is wrong and the byte argument needs
qualifying. Either outcome is worth knowing; only the second is damaging.

Method. VTune's sampling driver counts three events:

    MEM_LOAD_RETIRED.L3_MISS   retired loads served from beyond L3 -- the
                               closest equivalent of perf's LLC-load-misses
    MEM_LOAD_RETIRED.L3_HIT    the same loads served by L3, for a miss rate
    LONGEST_LAT_CACHE.MISS     every core-originated request that missed L3,
                               including prefetch and RFO, so it bounds the
                               lines that actually crossed to DRAM

The counter is process-wide, and the process does more than the timed loop:
it loads a multi-hundred-megabyte ONNX file and runs two warm-up passes.
For a percentage that contamination is minor, but for an absolute count it
is not -- reading 1.2 GB off disk generates its own misses. So each variant
is profiled twice, once with the timed loop and once with seconds=0, which
executes exactly the same load and warm-up and nothing else. The difference
divided by the iteration count is one forward pass, with setup removed.

Usage:  python experiments/llc_miss_count.py
"""

import json
import os
import re
import shutil
import subprocess

VTUNE = r"C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
PYTHON = r".venv\Scripts\python.exe"
RUNNER = "experiments/vtune_runner_model.py"
RESULT_ROOT = r"D:\tmp\vtune_llc_runs"
OUT_JSON = "experiments/results_llc_miss_count.json"

EVENTS = ("MEM_LOAD_RETIRED.L3_MISS", "MEM_LOAD_RETIRED.L3_HIT",
          "LONGEST_LAT_CACHE.MISS")

# Long enough that the timed loop dominates the difference, but not so long
# that a full sweep stops being practical. The encoder is ~12 s per pass and
# the decoder ~0.5 s, so equal wall-clock would give the decoder 25x more
# iterations than it needs.
SECONDS = {"encoder": 45.0, "decoder": 20.0}

VARIANTS = [
    ("enkoder FP32", "encoder", "models/uzbek_stt_v1_onnx/encoder_model.onnx"),
    ("enkoder INT8 [300]", "encoder", "models/_enc_v2/enc_int8.onnx"),
    ("enkoder kaskad [267]", "encoder", "models/_gptq/enc_gptq_pruned.onnx"),
    ("enkoder past-rank [203]", "encoder", "models/_alloc/enc_greedy.onnx"),
    ("dekoder FP32", "decoder", "models/uzbek_stt_v1_onnx/decoder_model.onnx"),
    ("dekoder INT8 [438]", "decoder", "models/_whole_net/dec_int8.onnx"),
    ("dekoder INT8+past-rank [343]", "decoder",
     "models/_lowrank_dec/dec_lr_int8.onnx"),
]


def model_bytes(path):
    total = os.path.getsize(path)
    ext = path + ".data"
    if os.path.exists(ext):
        total += os.path.getsize(ext)
    return total


def run(path, kind, seconds, result_dir):
    """One profiled execution; returns (event totals, iterations, ms/iter)."""
    shutil.rmtree(result_dir, ignore_errors=True)
    cmd = [VTUNE, "-collect-with", "runsa",
           "-knob", "event-config=" + ",".join(EVENTS),
           "-r", result_dir,
           "--", PYTHON, RUNNER, path, kind, str(seconds)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")

    counts = {}
    for ev in EVENTS:
        m = re.search(re.escape(ev) + r"\s+(\d+)\s", text)
        counts[ev] = int(m.group(1)) if m else None
    it = re.search(r"(\d+) iteratsiya", text)
    ms = re.search(r"([\d.]+) ms/iter", text)
    if any(v is None for v in counts.values()):
        print(f"    OGOHLANTIRISH: hodisalar o'qilmadi -- {text[-300:]}")
    return (counts,
            int(it.group(1)) if it else 0,
            float(ms.group(1)) if ms else None)


def load_cache():
    if not os.path.exists(OUT_JSON):
        return {}
    try:
        with open(OUT_JSON, encoding="utf-8-sig") as f:
            return {r["variant"]: r for r in json.load(f)}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)
    rows = load_cache()
    if rows:
        print(f"keshdan {len(rows)} variant\n")

    for label, kind, path in VARIANTS:
        if label in rows:
            print(f"[{label}] keshdan")
            continue
        if not os.path.exists(path):
            print(f"[{label}] SKIP -- {path} topilmadi")
            continue

        tag = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        print(f"\n[{label}]", flush=True)

        print("  bazaviy yurish (yuklash + isitish, o'lchov sikli yo'q)...",
              flush=True)
        base, _, _ = run(path, kind, 0.0,
                         os.path.join(RESULT_ROOT, tag + "_base"))

        print(f"  o'lchov yurishi ({SECONDS[kind]:.0f} s)...", flush=True)
        full, iters, ms = run(path, kind, SECONDS[kind],
                              os.path.join(RESULT_ROOT, tag + "_full"))

        if iters < 1 or any(full[e] is None or base[e] is None for e in EVENTS):
            print("  XATO: yurish natijasiz, o'tkazib yuborildi")
            continue

        per_iter = {e: (full[e] - base[e]) / iters for e in EVENTS}
        miss = per_iter["MEM_LOAD_RETIRED.L3_MISS"]
        hit = per_iter["MEM_LOAD_RETIRED.L3_HIT"]
        rows[label] = {
            "variant": label, "kind": kind, "path": path,
            "mib": model_bytes(path) / 1024 ** 2,
            "ms_per_iter": ms, "iterations": iters,
            "llc_load_misses_per_iter": miss,
            "llc_load_hits_per_iter": hit,
            "llc_all_misses_per_iter": per_iter["LONGEST_LAT_CACHE.MISS"],
            "llc_load_miss_rate": miss / (miss + hit) if miss + hit > 0 else None,
            "raw_full": full, "raw_base": base,
        }
        json.dump(list(rows.values()), open(OUT_JSON, "w"), indent=2)
        print(f"  {iters} iteratsiya | LLC-load-miss/o'tish "
              f"{miss:,.0f} | miss rate {rows[label]['llc_load_miss_rate']:.3f}"
              .replace(",", " "))

    report(rows)


def fmt(x):
    return f"{x:,.0f}".replace(",", " ")


def report(rows):
    print("\n" + "=" * 100)
    print("OXIRGI DARAJA KESH MISSLARI -- bitta o'tishga, yuklash ayirilgan")
    print("=" * 100)
    print(f"{'Variant':30s} {'MiB':>8s} {'LLC-load-miss':>15s} "
          f"{'miss rate':>10s} {'barcha miss':>14s} {'ms/iter':>9s}")
    print("-" * 100)
    for label, _, _ in VARIANTS:
        r = rows.get(label)
        if not r:
            continue
        print(f"{r['variant']:30s} {r['mib']:8.0f} "
              f"{fmt(r['llc_load_misses_per_iter']):>15s} "
              f"{r['llc_load_miss_rate']:10.3f} "
              f"{fmt(r['llc_all_misses_per_iter']):>14s} "
              f"{r['ms_per_iter']:9.1f}")

    # The prediction under test: traffic is proportional to model size, so
    # the miss ratio should track the byte ratio. Printed per component,
    # because encoder and decoder differ in reuse and could differ here too.
    print("\n" + "=" * 100)
    print("BASHORAT SINOVI: misslar bayt hajmiga proporsionalmi?")
    print("=" * 100)
    for kind, base_label in (("encoder", "enkoder FP32"),
                             ("decoder", "dekoder FP32")):
        b = rows.get(base_label)
        if not b:
            continue
        print(f"\n{kind}  (asos: {base_label})")
        print(f"  {'variant':30s} {'bayt nisbati':>13s} "
              f"{'miss nisbati':>13s} {'vaqt nisbati':>13s}")
        for label, k, _ in VARIANTS:
            r = rows.get(label)
            if not r or k != kind or label == base_label:
                continue
            print(f"  {label:30s} "
                  f"{b['mib'] / r['mib']:12.2f}x "
                  f"{b['llc_load_misses_per_iter'] / r['llc_load_misses_per_iter']:12.2f}x "
                  f"{b['ms_per_iter'] / r['ms_per_iter']:12.2f}x")

    have = [r for r in rows.values() if r.get("llc_load_misses_per_iter")]
    if len(have) >= 3:
        import numpy as np
        x = np.array([r["mib"] for r in have])
        y = np.array([r["llc_load_misses_per_iter"] for r in have])
        print(f"\nMiB va LLC-load-miss korrelyatsiyasi (barcha variantlar): "
              f"r = {np.corrcoef(x, y)[0, 1]:+.3f}")
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
