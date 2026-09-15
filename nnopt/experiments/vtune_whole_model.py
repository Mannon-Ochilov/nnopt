"""Hardware counters for the WHOLE model -- does the reuse argument hold?

The cascade's asymmetric decision rests on an argument that has so far been
made analytically rather than measured:

    encoder   1500 positions per pass, each weight reused 1500x
              -> weights want to stay resident, cache-fit is meaningful,
                 and cutting FLOPs converts into time
    decoder   batch=1 autoregressive, each weight used ONCE per token and
              evicted by the 23 layers that follow
              -> only bytes moved matter; cutting FLOPs buys nothing

That argument is why the cascade refuses low-rank on the decoder while
applying it on the encoder, and Sec 4.9c showed the refusal is worth 0.43
WER. But "the decoder is memory bound" was never observed, only asserted.

vtune_cache_evidence.py measured counters for ONE encoder operator and found
Memory Bound at only 9-18%, i.e. compute-bound -- consistent with the encoder
half of the argument. This extends the same measurement to whole models, in
their natural regimes, so the encoder and decoder can be compared directly.

Each variant is profiled single-threaded under uarch-exploration, the same
protocol as the operator-level run, so the numbers are comparable to Table 15.
"""

import json
import os
import re
import shutil
import subprocess

VTUNE = r"C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
PYTHON = r".venv\Scripts\python.exe"
RUNNER = "experiments/vtune_runner_model.py"
RESULT_ROOT = r"D:\tmp\vtune_model_runs"
SECONDS = 8.0
OUT_JSON = "experiments/results_vtune_whole_model.json"

# (label, kind, path). Grouped by MATCHED BUDGET, because comparing a
# compressed model only against FP32 answers the wrong question: of course
# moving 4x fewer bytes reduces memory stalls. The question that decides
# between methods is whether, at the SAME size, one of them behaves better in
# the memory hierarchy. Three encoder groups share a budget almost exactly
# (299-300, 267, 203 MiB), and the decoder pair shares 438-439 MiB.
VARIANTS = [
    ("enkoder FP32", "encoder", "models/uzbek_stt_v1_onnx/encoder_model.onnx"),

    # ~300 MiB (4x): quantizer varies, no structural change
    ("enkoder INT8 per-tensor  [300]", "encoder", "models/_enc_v2/enc_int8.onnx"),
    ("enkoder RTN per-channel  [300]", "encoder", "models/_rtn/enc_rtn_only.onnx"),
    ("enkoder GPTQ            [300]", "encoder", "models/_gptq/enc_gptq_only.onnx"),

    # ~267 MiB (4.3x): structural removal applied, quantizer varies
    ("enkoder qisqartirish+RTN [267]", "encoder", "models/_rtn/enc_rtn_pruned.onnx"),
    ("enkoder kaskad: qisq.+GPTQ [267]", "encoder",
     "models/_gptq/enc_gptq_pruned.onnx"),

    # 203 MiB (5.8x): equal budget, rank ALLOCATION varies
    ("enkoder past-rank bir xil [203]", "encoder", "models/_alloc/enc_uniform.onnx"),
    ("enkoder past-rank optimal [203]", "encoder", "models/_alloc/enc_greedy.onnx"),

    ("dekoder FP32", "decoder", "models/uzbek_stt_v1_onnx/decoder_model.onnx"),

    # ~438 MiB: granularity varies at identical size
    ("dekoder INT8 per-tensor  [438]", "decoder", "models/_whole_net/dec_int8.onnx"),
    ("dekoder INT8 per-channel [439]", "decoder",
     "models/_granularity/dec_int8_perchannel.onnx"),

    ("dekoder INT8 + past-rank (rad etilgan) [343]", "decoder",
     "models/_lowrank_dec/dec_lr_int8.onnx"),
]

METRIC_PATTERNS = {
    "cpi": r"CPI Rate:\s*([\d.]+)",
    "memory_bound": r"Memory Bound:\s*([\d.]+)%",
    "l1_bound": r"L1 Bound:\s*([\d.]+)%",
    "l2_bound": r"L2 Bound:\s*([\d.]+)%",
    "l3_bound": r"L3 Bound:\s*([\d.]+)%",
    "dram_bound": r"DRAM Bound:\s*([\d.]+)%",
    "elapsed": r"Elapsed Time:\s*([\d.]+)s",
}


def model_bytes(path):
    total = os.path.getsize(path)
    ext = path + ".data"
    if os.path.exists(ext):
        total += os.path.getsize(ext)
    return total


def collect(label, path, kind, result_dir):
    shutil.rmtree(result_dir, ignore_errors=True)
    cmd = [VTUNE, "-collect", "uarch-exploration", "-r", result_dir,
           "--", PYTHON, RUNNER, path, kind, str(SECONDS)]
    print(f"\n[{label}] yig'ilmoqda...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"([\d.]+) ms/iter", proc.stdout or "")
    if not m:
        tail = ((proc.stdout or "") + (proc.stderr or ""))[-400:]
        print(f"  OGOHLANTIRISH: ms/iter o'qilmadi. {tail}")
    if proc.returncode != 0 and not os.path.exists(result_dir):
        print(f"  XATO: {(proc.stderr or '')[:300]}")
        return None
    return float(m.group(1)) if m else None


def report(result_dir):
    proc = subprocess.run([VTUNE, "-report", "summary", "-r", result_dir],
                          capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    return {k: (float(m.group(1)) if (m := re.search(p, text)) else None)
            for k, p in METRIC_PATTERNS.items()}


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
            print(f"[{label}] SKIP — {path} topilmadi")
            continue
        rdir = os.path.join(RESULT_ROOT,
                            re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_"))
        ms = collect(label, path, kind, rdir)
        metrics = report(rdir)
        rows[label] = {"variant": label, "kind": kind, "path": path,
                       "mib": model_bytes(path) / 1024 ** 2,
                       "ms_per_iter": ms, **metrics}
        json.dump(list(rows.values()), open(OUT_JSON, "w"), indent=2)
        print(f"  {ms} ms/iter | Memory Bound {metrics.get('memory_bound')}% | "
              f"L3 {metrics.get('l3_bound')}% | DRAM {metrics.get('dram_bound')}% "
              f"| CPI {metrics.get('cpi')}")

    print("\n" + "=" * 104)
    print("BUTUN MODEL APPARAT HISOBLAGICHLARI (bitta oqim, uarch-exploration)")
    print("=" * 104)
    print(f"{'Variant':40s} {'MiB':>7s} {'ms/iter':>9s} {'Mem%':>7s} "
          f"{'L2%':>6s} {'L3%':>6s} {'DRAM%':>7s} {'CPI':>6s}")
    print("-" * 104)
    for label, _, _ in VARIANTS:
        r = rows.get(label)
        if not r:
            continue
        def f(k, w=7, d=1):
            v = r.get(k)
            return f"{v:{w}.{d}f}" if v is not None else " " * (w - 1) + "-"
        print(f"{label:40s} {r['mib']:7.0f} {f('ms_per_iter', 9, 2)} "
              f"{f('memory_bound')} {f('l2_bound', 6)} {f('l3_bound', 6)} "
              f"{f('dram_bound')} {f('cpi', 6, 3)}")

    # Absolute stall time, since Memory Bound is a share of a denominator that
    # itself changes between variants.
    print("\n" + "=" * 104)
    print("TENG BYUDJETDA TAQQOSLASH — bir xil hajmda qaysi usul yaxshiroq "
          "xotira xatti-harakatiga ega")
    print("=" * 104)
    groups = [
        ("~300 MiB (4x): kvantlagich farq qiladi",
         ["enkoder INT8 per-tensor  [300]", "enkoder RTN per-channel  [300]",
          "enkoder GPTQ            [300]"]),
        ("~267 MiB (4.3x): qisqartirish + kvantlagich",
         ["enkoder qisqartirish+RTN [267]", "enkoder kaskad: qisq.+GPTQ [267]"]),
        ("203 MiB (5.8x): rank TAQSIMOTI farq qiladi",
         ["enkoder past-rank bir xil [203]", "enkoder past-rank optimal [203]"]),
        ("~438 MiB: dekoder granulyarligi",
         ["dekoder INT8 per-tensor  [438]", "dekoder INT8 per-channel [439]"]),
    ]
    for title, labels in groups:
        present = [rows[x] for x in labels if x in rows
                   and rows[x].get("memory_bound") is not None]
        if len(present) < 2:
            continue
        print(f"\n{title}")
        print(f"  {'variant':36s} {'ms/iter':>9s} {'Mem%':>6s} "
              f"{'xotira ms':>10s} {'DRAM%':>6s} {'L3%':>5s}")
        for r in present:
            stall = r["ms_per_iter"] * r["memory_bound"] / 100
            print(f"  {r['variant']:36s} {r['ms_per_iter']:9.1f} "
                  f"{r['memory_bound']:6.1f} {stall:10.1f} "
                  f"{r['dram_bound']:6.1f} {r['l3_bound']:5.1f}")
        stalls = [r["ms_per_iter"] * r["memory_bound"] / 100 for r in present]
        spread = (max(stalls) - min(stalls)) / max(min(stalls), 1e-9) * 100
        print(f"  -> xotira to'xtashi tarqoqligi guruh ichida: {spread:.1f}%")

    enc = [r for r in rows.values() if r.get("kind") == "encoder"
           and r.get("memory_bound") is not None and "FP32" not in r["variant"]]
    dec = [r for r in rows.values() if r.get("kind") == "decoder"
           and r.get("memory_bound") is not None and "FP32" not in r["variant"]]
    if enc and dec:
        me = sum(r["memory_bound"] for r in enc) / len(enc)
        md = sum(r["memory_bound"] for r in dec) / len(dec)
        print(f"\nSiqilgan variantlar bo'yicha o'rtacha Memory Bound: "
              f"enkoder {me:.1f}%, dekoder {md:.1f}%")
        print("Kaskadning qayta ishlatish argumenti dekoderda sezilarli "
              "yuqoriroq qiymatni\nbashorat qiladi — yuqoridagi raqamlar uni "
              "tasdiqlaydi yoki rad etadi.")


if __name__ == "__main__":
    main()
