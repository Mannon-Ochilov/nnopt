"""Re-measure the whole configuration library with an INTERLEAVED design.

The existing latency_library.py walks the library one configuration at a time
(ten runs, next key) and CACHES each row, so entries can come from different
sessions. That is a blocked design, and Sec 4.7 of this work documents why a
blocked design cannot be trusted here: it once produced a convincing 1.7-2.2x
"cache knee" that a plain replication destroyed, because size and machine
drift move together in a blocked sweep.

The blocked numbers now look suspect on their own terms. The same artifact,
models/_gptq/enc_gptq_only.onnx, reads 8658 ms in that library and 7081 /
7417 ms in the VTune sweep, while the paired tau=0.99 artifact reads
6704 / 6728 / 7211 ms. The library's blind-INT8 row stands alone as slow, and
the 1.29x speedup quoted from it is 1.03-1.07x in every other measurement of
the same two files.

This script measures every configuration in ROUNDS -- A B C ... A B C ... --
so drift affects all arms equally and cancels in the ratio. Nothing is
cached: a partial run is discarded rather than mixed with a later one.

    python experiments/latency_library_interleaved.py            # 7 rounds
    ROUNDS=11 python experiments/latency_library_interleaved.py  # tighter
"""

import json
import os
import statistics
import time

from calib_utils import encoder_feeds
from config_library import LIBRARY
from nnopt.bench.latency import make_session

OUT_JSON = "experiments/results_latency_library_interleaved.json"
WARMUP = 3
ROUNDS = int(os.environ.get("ROUNDS", "7"))
THREADS = 1


def main():
    feed = encoder_feeds(0, 1)[0]
    arms = []
    for key, label, path, family in LIBRARY:
        if not os.path.exists(path):
            print(f"[{key:6s}] SKIP — {path} yo'q")
            continue
        arms.append((key, label, path, family))
    if len(arms) < 2:
        raise SystemExit("kamida ikkita konfiguratsiya kerak")

    print(f"{len(arms)} konfiguratsiya, {ROUNDS} navbatlashgan raund, "
          f"{THREADS} oqim\n")

    sessions = {}
    for key, label, path, _ in arms:
        print(f"  yuklanmoqda: {key:6s} {label}", flush=True)
        s = make_session(path, intra_op_threads=THREADS)
        sessions[key] = s
        for _ in range(WARMUP):
            s.run(None, feed)

    times = {key: [] for key, _, _, _ in arms}
    t_start = time.time()
    for r in range(ROUNDS):
        for key, _, _, _ in arms:
            t0 = time.perf_counter()
            sessions[key].run(None, feed)
            times[key].append((time.perf_counter() - t0) * 1000.0)
        print(f"  raund {r + 1}/{ROUNDS}  [{time.time() - t_start:.0f}s]",
              flush=True)

    med = {k: statistics.median(v) for k, v in times.items()}
    base = med["int8"] if "int8" in med else med[arms[0][0]]

    rows = {}
    print("\n" + "=" * 88)
    print(f"{'key':7s} {'label':38s} {'MiB':>7s} {'ms':>9s} "
          f"{'tarq.%':>8s} {'INT8 ga':>9s}")
    print("-" * 88)
    for key, label, path, family in arms:
        v = times[key]
        m = med[key]
        spread = (max(v) - min(v)) / m * 100
        mib = os.path.getsize(path) / (1024 * 1024)
        rows[key] = {"key": key, "label": label, "family": family,
                     "path": path, "mib": mib, "ms": m,
                     "spread_pct": spread, "speedup_vs_int8": base / m,
                     "runs_ms": v}
        print(f"{key:7s} {label[:38]:38s} {mib:7.0f} {m:9.0f} "
              f"{spread:7.1f}% {base / m:8.2f}x")
    print("=" * 88)

    json.dump({"protokol": {"raundlar": ROUNDS, "qizdirish": WARMUP,
                            "oqimlar": THREADS, "dizayn": "interleaved"},
               "natijalar": rows}, open(OUT_JSON, "w"), indent=2)
    print(f"saqlandi: {OUT_JSON}")

    old_path = "experiments/results_latency_library.json"
    if os.path.exists(old_path):
        old = json.load(open(old_path, encoding="utf-8-sig"))
        print("\nBLOKLI (eski) va NAVBATLASHGAN (yangi) taqqoslash:")
        print(f"{'key':7s} {'blokli ms':>11s} {'navbat ms':>11s} {'farq %':>9s}")
        for key in rows:
            if key in old:
                o, n = old[key]["ms"], rows[key]["ms"]
                print(f"{key:7s} {o:11.0f} {n:11.0f} {(n - o) / o * 100:8.1f}%")
        if "int8" in old and "t99" in old:
            r_old = old["int8"]["ms"] / old["t99"]["ms"]
            r_new = rows["int8"]["ms"] / rows["t99"]["ms"]
            print(f"\ntau=0.99 ning INT8 ga tezlanishi: "
                  f"blokli {r_old:.2f}x -> navbatlashgan {r_new:.2f}x")


if __name__ == "__main__":
    main()
