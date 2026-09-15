"""DIRECT cache-miss evidence via Intel VTune hardware counters.

Until now every cache claim in this project was indirect: the machine
exposed no PMC access, so cache behaviour had to be inferred from latency
versus a FLOP-only prediction (README Sec 8.3.x). VTune 2026.4 is now
installed and its event-based sampling DRIVER is active on this Tiger Lake
H part, so the counters can be read directly.

What this settles: cache_cliff_encoder.py found that shrinking encoder fc1's
weight from 4 MiB (dense INT8) to 0.6 MiB (low-rank + INT8) produced LESS
speedup than the FLOP reduction alone predicted (surplus 0.56-0.99, never
above 1.0). That is evidence against a cache-residency mechanism, but it is
inference, not observation. If the mechanism really is absent, the measured
L3/DRAM Bound metrics must NOT improve materially as the weight shrinks.

Metrics collected per variant (uarch-exploration):
    Memory Bound %, L1/L2/L3 Bound %, DRAM Bound %, CPI Rate
"""

import json
import os
import re
import subprocess
import sys

VTUNE = r"C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune.exe"
PYTHON = r".venv\Scripts\python.exe"
RUNNER = "experiments/vtune_runner.py"
RESULT_ROOT = r"D:\tmp\vtune_runs"
MODEL_DIR = "models/_cliff"
N_IN = 1024          # encoder fc1 input dim
SECONDS = 8.0
OUT_JSON = "experiments/results_vtune_cache.json"

VARIANTS = [
    ("dense FP32", "dense_fp32.onnx"),
    ("dense INT8", "dense_int8.onnx"),
    ("SVD r=409 + INT8", "svd_actaw_r409_int8.onnx"),
    ("SVD r=200 + INT8", "svd_actaw_r200_int8.onnx"),
    ("SVD r=128 + INT8", "svd_actaw_r128_int8.onnx"),
    ("SVD r=80 + INT8", "svd_actaw_r80_int8.onnx"),
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


def collect(label, model_path, result_dir):
    if os.path.exists(result_dir):
        import shutil
        shutil.rmtree(result_dir, ignore_errors=True)
    cmd = [VTUNE, "-collect", "uarch-exploration", "-r", result_dir,
           "--", PYTHON, RUNNER, model_path, str(N_IN), str(SECONDS)]
    print(f"\n[{label}] yig'ilmoqda...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ms = None
    m = re.search(r"([\d.]+) ms/iter", proc.stdout or "")
    if m:
        ms = float(m.group(1))
    if proc.returncode != 0 and not os.path.exists(result_dir):
        print(f"  XATO: {(proc.stderr or '')[:300]}")
        return None
    return ms


def report(result_dir):
    proc = subprocess.run([VTUNE, "-report", "summary", "-r", result_dir],
                          capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    out = {}
    for key, pat in METRIC_PATTERNS.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    return out


def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)
    rows = []
    for label, fname in VARIANTS:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"[{label}] SKIP — {path} topilmadi")
            continue
        rdir = os.path.join(RESULT_ROOT, fname.replace(".onnx", ""))
        ms = collect(label, path, rdir)
        metrics = report(rdir)
        wbytes = os.path.getsize(path)
        rows.append({"variant": label, "file": fname, "file_bytes": wbytes,
                     "ms_per_iter": ms, **metrics})
        print(f"  {ms} ms/iter | Memory Bound {metrics.get('memory_bound')}% | "
              f"L3 {metrics.get('l3_bound')}% | DRAM {metrics.get('dram_bound')}% | "
              f"CPI {metrics.get('cpi')}")

    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    summarize(rows)


def summarize(rows):
    if not rows:
        return
    print("\n" + "=" * 104)
    print("VTUNE APPARAT HISOBLAGICHLARI — kesh bosimi past-rank bilan kamayadimi?")
    print("=" * 104)
    print(f"{'Variant':20s} {'ms/iter':>9s} {'Mem Bound':>10s} {'L1':>7s} {'L2':>7s} "
          f"{'L3':>7s} {'DRAM':>7s} {'CPI':>7s}")
    print("-" * 104)
    for r in rows:
        def f(key, width=6):
            v = r.get(key)
            return "-".rjust(width) if v is None else f"{v:{width}.1f}"

        ms = "-" if r["ms_per_iter"] is None else f"{r['ms_per_iter']:.2f}"
        cpi = "-" if r.get("cpi") is None else f"{r['cpi']:.2f}"
        print(f"{r['variant']:20s} {ms:>9s} {f('memory_bound', 9)}% {f('l1_bound')}% "
              f"{f('l2_bound')}% {f('l3_bound')}% {f('dram_bound')}% {cpi:>7s}")
    print("-" * 104)
    print("Agar past-rank kesh orqali yordam bersa: L3/DRAM Bound sezilarli KAMAYISHI kerak.")
    print("Agar ular deyarli o'zgarmasa -> mexanizm kesh emas, to'g'ridan-to'g'ri FLOPs.")


if __name__ == "__main__":
    main()
