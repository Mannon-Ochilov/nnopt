"""Is the cascade's decision robust to alpha, or does it hinge on one constant?

The cache budget is alpha * L3 with alpha = 0.7. That constant carries real
weight: it is the only place where "the cache also holds things that are not
this operator's weights" enters the model. The budget deliberately counts
WEIGHT bytes only -- weights are the quantity the cascade decides about, and
they are the quantity a blocked GEMM wants resident while activation tiles
stream past -- so alpha is the derating that stands in for everything else.

A single hand-picked constant is only harmless if the DECISIONS it produces
are stable. This sweeps alpha and reports, for each value, the derived
requirement and which cascade case each granularity lands in:

    case 2  requirement <= 4.00x  ->  INT8 suffices, stop
    case 3  requirement >  4.00x  ->  add low-rank

The decision boundary sits at alpha* = size / (4 * L3). If alpha = 0.7 is
comfortably inside a case, the constant does not matter. If it sits near a
boundary, the paper is resting a measured conclusion on an unmeasured
number, and that is worth saying plainly.
"""

import numpy as np

from nnopt.hw.cache_topology import detect_cache_topology

INT8_FACTOR = 4.0
ALPHAS = (0.4, 0.5, 0.6, 0.667, 0.7, 0.8, 0.9, 1.0)

# Weight footprints measured by cache_anchored_targets.py (fp32, MiB).
FOOTPRINTS = {
    "dekoder, per-layer": 64.0,
    "dekoder, eng katta operator": 16.0,
    "enkoder, per-layer": 48.0,
    "enkoder, eng katta operator": 16.0,
}


def main():
    topo = detect_cache_topology()
    g = topo.global_shared_cache()
    l3 = g.size_bytes / 1024 ** 2
    print(f"Kafolatlangan umumiy kesh: L{g.level}, {l3:.0f} MiB, "
          f"{len(g.core_ids)} yadro")
    print(f"INT8 beradigan siqish: {INT8_FACTOR:.2f}x\n")

    print(f"{'granulyarlik':30s} " + " ".join(f"{a:>7.3f}" for a in ALPHAS))
    print("-" * (30 + 8 * len(ALPHAS)))
    for name, mib in FOOTPRINTS.items():
        cells = []
        for a in ALPHAS:
            need = mib / (a * l3)
            cells.append("sig'adi" if need <= 1.0 else f"{need:.2f}x")
        print(f"{name:30s} " + " ".join(f"{c:>7s}" for c in cells))

    print()
    print(f"{'granulyarlik':30s} " + " ".join(f"{a:>7.3f}" for a in ALPHAS))
    print("-" * (30 + 8 * len(ALPHAS)))
    for name, mib in FOOTPRINTS.items():
        cells = []
        for a in ALPHAS:
            need = mib / (a * l3)
            cells.append("1" if need <= 1.0 else ("2" if need <= INT8_FACTOR else "3"))
        print(f"{name:30s} " + " ".join(f"{c:>7s}" for c in cells) + "   <- holat")

    print("\nQaror chegaralari (holat 2 -> holat 3 o'tish nuqtasi):")
    for name, mib in FOOTPRINTS.items():
        a_star = mib / (INT8_FACTOR * l3)
        if 0.0 < a_star <= 1.0:
            dist = abs(0.7 - a_star)
            flag = "  <-- 0.7 GA YAQIN" if dist < 0.1 else ""
            print(f"  {name:30s} alpha* = {a_star:.3f}   "
                  f"|0.7 - alpha*| = {dist:.3f}{flag}")
        else:
            print(f"  {name:30s} alpha* = {a_star:.3f}   "
                  f"(oraliqdan tashqarida — qaror alphaga bog'liq emas)")

    print("\nIzoh: alpha* oraliqdan tashqarida bo'lsa, alphaning har qanday "
          "maqbul qiymati\nbir xil qarorni beradi va konstanta ahamiyatsiz "
          "bo'ladi.")


if __name__ == "__main__":
    main()
