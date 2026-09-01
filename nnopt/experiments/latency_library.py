"""Does the miss objective predict anything real, or only itself?

The framework ranks configurations by an analytic miss count. That number is
derived, not observed, and a derived objective that fails to track measured
time is decoration. So this measures encoder latency for every configuration
in the library and asks whether the ordering survives.

The test it can perform is narrower than the claim it is checking, and saying
so precisely matters. This machine has a 24 MiB L3, where every configuration
in the library already fits the budget; in that regime the miss expression
collapses to plain byte count, so what is really being tested is whether
FEWER BYTES MEANS LESS TIME. The interesting predictions -- the 1.81x at
12 MiB, the ordering at 2 MiB -- involve overflow terms that cannot be
produced on hardware whose cache does not overflow. They stay untested until a
second machine is available, and no amount of measurement here substitutes for
one.

Latency is measured single-threaded to keep memory behaviour legible: with
eight threads the encoder is bound by whichever core finishes last and the
per-configuration differences wash into scheduling noise.
"""

import json
import os

import numpy as np

from calib_utils import encoder_feeds
from config_library import LIBRARY
from nnopt.bench.latency import make_session, measure_latency

OUT_JSON = "experiments/results_latency_library.json"
LIB_JSON = "experiments/results_config_library.json"
WARMUP, MEASURED = 3, 10
THREADS = 1


def main():
    lib = {}
    if os.path.exists(LIB_JSON):
        lib = json.load(open(LIB_JSON, encoding="utf-8-sig"))

    rows = {}
    if os.path.exists(OUT_JSON):
        try:
            rows = json.load(open(OUT_JSON, encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            rows = {}

    feed = encoder_feeds(0, 1)[0]
    for key, label, path, family in LIBRARY:
        if key in rows:
            print(f"[{key:6s}] keshdan {rows[key]['ms']:.0f} ms")
            continue
        if not os.path.exists(path):
            print(f"[{key:6s}] SKIP — {path} yo'q")
            continue
        print(f"[{key:6s}] {label} — o'lchanmoqda...", flush=True)
        lat = measure_latency(make_session(path, intra_op_threads=THREADS),
                              name=path, fixed_feed=feed,
                              warmup_runs=WARMUP, measured_runs=MEASURED)
        rows[key] = {"key": key, "label": label, "ms": float(lat.median_ms),
                     "mib": os.path.getsize(path) / (1024 * 1024)}
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
        print(f"  {lat.median_ms:.0f} ms", flush=True)

    print("\n" + "=" * 82)
    print(f"ENKODER LATENCY ({THREADS} oqim, mediana {MEASURED} yurishdan)")
    print("=" * 82)
    print(f"{'kalit':7s} {'konfiguratsiya':38s} {'MiB':>7s} {'ms':>9s} "
          f"{'INT8 ga':>9s} {'WER':>8s}")
    print("-" * 82)
    base = rows.get("int8", {}).get("ms")
    ordered = sorted(rows.values(), key=lambda r: -r["mib"])
    for r in ordered:
        rel = base / r["ms"] if base else float("nan")
        wer = lib.get(r["key"], {}).get("wer", float("nan"))
        print(f"{r['key']:7s} {r['label']:38s} {r['mib']:7.0f} {r['ms']:9.0f} "
              f"{rel:8.2f}x {wer:8.4f}")

    # The check the objective has to pass in this regime: with everything
    # fitting, miss is byte count, so bytes and time must move together.
    quant = [r for r in rows.values() if r["key"] != "fp32"]
    if len(quant) > 2:
        mib = np.array([r["mib"] for r in quant])
        ms = np.array([r["ms"] for r in quant])
        rho = float(np.corrcoef(mib, ms)[0, 1])
        order_mib = [r["key"] for r in sorted(quant, key=lambda r: r["mib"])]
        order_ms = [r["key"] for r in sorted(quant, key=lambda r: r["ms"])]
        print(f"\nbayt va vaqt korrelyatsiyasi (kvantlangan armlar, "
              f"n={len(quant)}): r = {rho:+.3f}")
        print(f"  hajm bo'yicha tartib: {' < '.join(order_mib)}")
        print(f"  vaqt bo'yicha tartib: {' < '.join(order_ms)}")
        print(f"  tartiblar {'MOS' if order_mib == order_ms else 'MOS EMAS'}")
    print("\nDIQQAT: bu mashinada L3 = 24 MiB va barcha armlar byudjetga "
          "sig'adi,\nya'ni bu yerda miss = bayt hisobi. 12 MiB dagi 1.81x "
          "kabi bashoratlar\noverflow hadiga tayanadi va ikkinchi mashinasiz "
          "tekshirilmaydi.")


if __name__ == "__main__":
    main()
