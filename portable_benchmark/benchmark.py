"""Portable latency benchmark for a second platform.

One command, two dependencies (numpy + onnxruntime), no network and no audio
corpus. It reports a machine passport and the latency of each encoder
configuration, so the cascade's choice can be checked on hardware other than
the one it was developed on.

    python benchmark.py --quick     # 2 rounds, run this first
    python benchmark.py             # full measurement, 7 rounds
    python benchmark.py --threads 4 # optional multi-threaded run

WER is deliberately NOT measured here. It is a property of the artifact, not
of the machine: the same ONNX file produces the same transcription anywhere.
The measured WER of each configuration is carried in the table below for
reference, which is also why no audio is needed and the run takes minutes.

Configurations are measured INTERLEAVED (A-B-C-A-B-C over several rounds)
rather than one at a time to completion. Blocking a benchmark by
configuration lets the machine's thermal and load drift be read as a size
effect; that happened once in this project and the interleaved design is the
correction.

Output: results/benchmark_<machine>_<date>.json
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not found. Install with:  "
             "pip install onnxruntime numpy")

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "results")

# (file, label, MiB, measured WER). The WER is the article's 300-utterance
# TEST figure and is not re-measured here.
CONFIGS = [
    ("enc_fp32.onnx", "FP32 (control)", 1172, 0.1793),
    ("enc_gptq_only.onnx", "blind INT8", 300, 0.1847),
    ("enc_tau099.onnx", "cascade tau=0.99", 267, 0.1833),
    ("enc_tau095.onnx", "tau=0.95", 254, 0.2006),
    ("enc_tau090.onnx", "tau=0.90", 237, 0.2365),
]

N_POS = 1500          # encoder input: the 30 s mel window
N_MELS = 80
WARMUP = 3
ROUNDS = 7


def detect_l3():
    """Guaranteed shared L3 in MiB, or None.

    None is returned rather than a guess: an empty cell in the results is
    honest, an invented number is not.
    """
    system = platform.system()

    if system == "Linux":
        # sysfs is the reliable source: the size of the cache whose level is 3.
        base = "/sys/devices/system/cpu/cpu0/cache"
        try:
            for idx in sorted(os.listdir(base)):
                d = os.path.join(base, idx)
                if open(os.path.join(d, "level")).read().strip() != "3":
                    continue
                size = open(os.path.join(d, "size")).read().strip()
                if size.upper().endswith("K"):
                    return round(int(size[:-1]) / 1024, 2)
                if size.upper().endswith("M"):
                    return float(size[:-1])
        except OSError:
            pass
        try:                                    # lscpu as a fallback
            out = subprocess.run(["lscpu"], capture_output=True, text=True,
                                 timeout=10).stdout
            for line in out.splitlines():
                if line.lower().startswith("l3 cache"):
                    field = line.split(":", 1)[1].strip()
                    num = float(field.split()[0].rstrip("KMG"))
                    return round(num / 1024, 2) if "K" in field.upper() else num
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        return None

    if system == "Windows":
        # Only all-digit lines are accepted. The command also echoes its own
        # header, and parsing "L3CacheSize" as a value once yielded 0.0.
        for cmd in (
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).L3CacheSize"],
            ["wmic", "cpu", "get", "L3CacheSize"],
        ):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=25).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            for line in out.splitlines():
                s = line.strip()
                if s.isdigit() and int(s) > 0:
                    return round(int(s) / 1024, 2)          # KB -> MiB
        return None

    if system == "Darwin":
        # Apple Silicon does not expose hw.l3cachesize; None is the correct
        # answer there rather than a substituted L2 figure.
        try:
            out = subprocess.run(["sysctl", "-n", "hw.l3cachesize"],
                                 capture_output=True, text=True,
                                 timeout=10).stdout.strip()
            if out.isdigit() and int(out) > 0:
                return round(int(out) / 1024 ** 2, 2)       # bytes -> MiB
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def memory_config():
    """(modules, speed in MT/s) for the installed DIMMs, or (None, None).

    Capacity alone says little about a memory-bound measurement: single- and
    dual-channel DDR4-3200 differ by a factor of two in peak bandwidth, and
    the DRAM-bound share of a run cannot be read without knowing which it was.
    Speed is reported as the configured rate where the platform distinguishes
    it from the rated one.

    As with the cache size, an unreadable value is returned as None rather
    than guessed.
    """
    system = platform.system()
    try:
        if system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_PhysicalMemory | "
                 "ForEach-Object { $_.ConfiguredClockSpeed }"],
                capture_output=True, text=True, timeout=30).stdout
            speeds = [int(s) for s in out.split() if s.isdigit() and int(s) > 0]
            return (len(speeds) or None, max(speeds) if speeds else None)

        if system == "Linux":
            # Requires DMI access; without it the fields are simply absent.
            out = subprocess.run(["dmidecode", "-t", "memory"],
                                 capture_output=True, text=True,
                                 timeout=30).stdout
            speeds, modules = [], 0
            for line in out.splitlines():
                s = line.strip()
                if s.startswith("Size:") and "No Module" not in s:
                    modules += 1
                if s.startswith("Configured Memory Speed:") or \
                        s.startswith("Configured Clock Speed:"):
                    tok = s.split(":", 1)[1].strip().split()
                    if tok and tok[0].isdigit():
                        speeds.append(int(tok[0]))
            return (modules or None, max(speeds) if speeds else None)

        if system == "Darwin":
            out = subprocess.run(["system_profiler", "SPMemoryDataType"],
                                 capture_output=True, text=True,
                                 timeout=40).stdout
            speeds = [int(line.split(":", 1)[1].strip().rstrip("MHz ").strip())
                      for line in out.splitlines()
                      if line.strip().startswith("Speed:")
                      and any(c.isdigit() for c in line)]
            modules = sum(1 for line in out.splitlines()
                          if line.strip().startswith("Size:"))
            return (modules or None, max(speeds) if speeds else None)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return None, None


def total_ram_gb():
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = MemoryStatusEx()
            ms.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.ullTotalPhys / 1024 ** 3, 1)
        return round(os.sysconf("SC_PAGE_SIZE")
                     * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (OSError, ValueError, AttributeError):
        return None


def passport(threads):
    """Machine description, filled automatically wherever possible."""
    p = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine_name": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "intra_op_threads": threads,
        "L3_MiB": None,
        "RAM_GB": None,
        "memory_modules": None,
        "memory_speed_MTs": None,
        "memory_peak_GBs": None,
    }
    try:                                        # a fuller name on Linux
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    p["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    p["L3_MiB"] = detect_l3()
    p["RAM_GB"] = total_ram_gb()
    p["memory_modules"], p["memory_speed_MTs"] = memory_config()
    if p["memory_modules"] and p["memory_speed_MTs"]:
        # Peak = channels x transfers/s x 8 bytes. Channel count is INFERRED
        # from the number of populated modules, which is the usual arrangement
        # but not guaranteed, so this is a ceiling rather than a measurement.
        p["memory_peak_GBs"] = round(
            p["memory_modules"] * p["memory_speed_MTs"] * 8 / 1000.0, 1)
    if p["L3_MiB"]:
        p["alpha_L3_budget_MiB"] = round(0.7 * p["L3_MiB"], 2)
    return p


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def make_input(sess):
    """One synthetic mel window.

    Latency depends on the input's SHAPE, not its values, which is why no
    audio ships with the kit. Symbolic dimensions are resolved by name where
    the model provides one, so a graph that orders its axes differently is
    still fed correctly.
    """
    inp = sess.get_inputs()[0]
    defaults = {"batch": 1, "batch_size": 1, "channel": N_MELS,
                "n_mels": N_MELS, "feature": N_MELS,
                "sequence": N_POS, "seq": N_POS, "time": N_POS,
                "encoder_sequence_length": N_POS, "nb_max_frames": N_POS * 2}
    shape = []
    for axis, dim in enumerate(inp.shape):
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
            continue
        key = str(dim).lower()
        match = next((v for k, v in defaults.items() if k in key), None)
        if match is None:                       # fall back to axis position
            match = 1 if axis == 0 else (N_MELS if axis == 1 else N_POS)
        shape.append(match)
    rng = np.random.default_rng(0)
    return {inp.name: rng.standard_normal(shape).astype(np.float32)}


def main():
    ap = argparse.ArgumentParser(
        description="Portable latency benchmark for encoder configurations.")
    ap.add_argument("--quick", action="store_true",
                    help="short check: 2 rounds (run this first)")
    ap.add_argument("--threads", type=int, default=1,
                    help="intra-op threads (default 1)")
    args = ap.parse_args()
    rounds = 2 if args.quick else ROUNDS

    os.makedirs(OUT_DIR, exist_ok=True)
    p = passport(args.threads)
    print("=" * 66)
    print("PLATFORM PASSPORT")
    for k, v in p.items():
        print(f"  {k:24s} {v}")
    print("=" * 66)
    if p["L3_MiB"] is None:
        print("  note: L3 size could not be read on this platform; the field\n"
              "        is left empty rather than estimated.")
    if p["memory_speed_MTs"] is None:
        print("  note: memory speed unreadable (on Linux this needs DMI\n"
              "        access, e.g. sudo); the field is left empty.")
    elif p["memory_peak_GBs"]:
        print(f"  note: memory_peak_GBs assumes one channel per populated\n"
              f"        module ({p['memory_modules']}); it is a ceiling, not a\n"
              f"        measured bandwidth.")

    ready = []
    for fn, label, mib, wer in CONFIGS:
        path = os.path.join(MODELS, fn)
        if os.path.exists(path):
            ready.append((path, label, mib, wer))
        else:
            print(f"  skipped (file missing): {fn}")
    if len(ready) < 2:
        sys.exit("At least two models are required. Check the models/ folder.")

    print(f"\n{len(ready)} configurations, {rounds} interleaved rounds, "
          f"{args.threads} thread(s)\n")

    sessions, feeds = {}, {}
    for path, label, _, _ in ready:
        print(f"  loading: {label} ...", flush=True)
        s = session(path, args.threads)
        sessions[label] = s
        feeds[label] = make_input(s)
        for _ in range(WARMUP):
            s.run(None, feeds[label])

    times = {label: [] for _, label, _, _ in ready}
    for r in range(rounds):
        for _, label, _, _ in ready:            # A-B-C, then A-B-C again
            t0 = time.perf_counter()
            sessions[label].run(None, feeds[label])
            times[label].append((time.perf_counter() - t0) * 1000.0)
        print(f"  round {r + 1}/{rounds} done", flush=True)

    base = next((l for _, l, _, _ in ready if "INT8" in l), ready[0][1])
    base_med = statistics.median(times[base])

    rows = []
    print("\n" + "=" * 84)
    print(f"{'Configuration':24s} {'MiB':>6s} {'ms (median)':>13s} "
          f"{'spread %':>12s} {'speedup':>10s} {'WER*':>8s}")
    print("-" * 84)
    for _, label, mib, wer in ready:
        v = times[label]
        med = statistics.median(v)
        spread = (max(v) - min(v)) / med * 100 if med else 0.0
        rows.append({"configuration": label, "MiB": mib, "ms_median": med,
                     "spread_percent": spread,
                     "speedup_vs_blind_int8": base_med / med if med else None,
                     "WER_measured": wer, "all_ms": v})
        print(f"{label:24s} {mib:6d} {med:13.1f} {spread:11.1f}% "
              f"{base_med / med:9.2f}x {wer:8.4f}")
    print("=" * 84)
    print("* WER is a property of the artifact (article's TEST/300 figure), "
          "not re-measured here.")

    tag = "".join(c if c.isalnum() else "_" for c in p["machine_name"])[:24]
    out = os.path.join(OUT_DIR,
                       f"benchmark_{tag}_{datetime.now():%Y%m%d_%H%M}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"passport": p,
                   "settings": {"rounds": rounds, "warmup": WARMUP,
                                "threads": args.threads, "positions": N_POS},
                   "results": rows}, fh, indent=2, ensure_ascii=False)
    print(f"\nSAVED: {out}")
    print("Return this file — the cross-platform tables are filled from it.")


if __name__ == "__main__":
    main()
