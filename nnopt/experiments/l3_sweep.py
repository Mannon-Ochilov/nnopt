"""Does the derivation respond to the hardware, or only to this machine?

The cache-anchored target is the paper's central claim, and it rests on one
number measured on one machine: L3 = 24 MiB. A second platform would be the
direct test, but the arithmetic itself can already answer a weaker and still
useful question -- whether the derivation PRODUCES DIFFERENT DECISIONS when
the cache changes, and whether those decisions are the ones the measurements
say are correct.

For each candidate L3 the required per-granularity compression is

    rho = M_eff / (alpha * L3) ,

and the cascade's case follows: rho <= 1 needs nothing, rho <= 4 is met by
INT8 alone, rho > 4 requires structural or spectral reduction on top.

What makes the sweep more than bookkeeping is that several of the cells have
already been measured end to end, so the table carries consequences rather
than only labels:

  decoder pushed into case 3   measured at 0.6101 WER (Sec 4.9c, arm D) --
                               the change the cascade refuses at 24 MiB
  decoder left at case 2       measured at 0.1793, indistinguishable from FP32
  encoder in case 3            measured at 0.1833 with structural removal,
                               free relative to quantization alone

So where the sweep flips a decision, the cost of the flip is known.
"""

import numpy as np

from nnopt.hw.cache_topology import detect_cache_topology

ALPHA = 0.7
INT8 = 4.0
MIB = 1024.0 ** 2

# Weight footprints measured by cache_anchored_targets.py, in MiB (fp32).
FOOTPRINTS = [
    ("dekoder, per-layer", 64.0),
    ("enkoder, per-layer", 48.0),
    ("dekoder, eng katta operator", 16.0),
    ("enkoder, eng katta operator", 16.0),
]
L3_VALUES = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)

# What the measurements say each case costs, where we ran it.
EVIDENCE = {
    ("dekoder, per-layer", 3): "past-rank kerak -> o'lchangan WER 0.6101",
    ("dekoder, per-layer", 2): "INT8 yetarli -> o'lchangan WER 0.1793 (FP32 dan farqsiz)",
    ("enkoder, per-layer", 3): "qisqartirish kerak -> o'lchangan WER 0.1833 (tekin)",
    ("enkoder, per-layer", 2): "INT8 yetarli -> qisqartirishsiz 0.1847",
}


def case_of(rho):
    if rho <= 1.0:
        return 1
    return 2 if rho <= INT8 else 3


def main():
    topo = detect_cache_topology()
    g = topo.global_shared_cache()
    actual = g.size_bytes / MIB
    print(f"Ushbu mashina: L{g.level} = {actual:.0f} MiB, {len(g.core_ids)} yadro. "
          f"alpha = {ALPHA}, INT8 {INT8:.0f}x beradi.\n")

    print(f"{'granulyarlik':30s} " + " ".join(f"{v:>7.0f}" for v in L3_VALUES))
    print("-" * (30 + 8 * len(L3_VALUES)))
    for name, mib in FOOTPRINTS:
        cells = []
        for l3 in L3_VALUES:
            rho = mib / (ALPHA * l3)
            cells.append("sig'adi" if rho <= 1.0 else f"{rho:.2f}x")
        print(f"{name:30s} " + " ".join(f"{c:>7s}" for c in cells))

    print(f"\n{'granulyarlik':30s} " + " ".join(f"{v:>7.0f}" for v in L3_VALUES)
          + "   <- kaskad holati")
    print("-" * (30 + 8 * len(L3_VALUES)))
    flips = {}
    for name, mib in FOOTPRINTS:
        cases = [case_of(mib / (ALPHA * l3)) for l3 in L3_VALUES]
        flips[name] = cases
        print(f"{name:30s} " + " ".join(f"{c:>7d}" for c in cases))

    print("\nQaror o'zgaradigan chegaralar (L3, MiB):")
    for name, mib in FOOTPRINTS:
        for case, label in ((1, "hech narsa kerak emas"), (2, "INT8 yetarli")):
            # rho <= threshold  <=>  L3 >= mib / (alpha * threshold)
            thr = 1.0 if case == 1 else INT8
            l3_star = mib / (ALPHA * thr)
            if 1.0 <= l3_star <= 256.0:
                near = "  <-- SHU MASHINAGA YAQIN" if abs(l3_star - actual) < 4 else ""
                print(f"  {name:30s} {label:22s} L3 >= {l3_star:6.1f} MiB{near}")

    print("\nO'lchangan oqibatlar (qaror o'zgarganda nima yuz beradi):")
    for (name, case), text in EVIDENCE.items():
        where = [f"{l3:.0f}" for l3, c in zip(L3_VALUES, flips[name]) if c == case]
        if where:
            print(f"  {name:30s} holat {case} @ L3 = {', '.join(where)} MiB")
            print(f"  {'':30s}   {text}")

    dec = flips["dekoder, per-layer"]
    idx = L3_VALUES.index(actual) if actual in L3_VALUES else None
    if idx is not None:
        print(f"\nUshbu mashinada dekoder {dec[idx]}-holatda. "
              f"L3 {64.0/(ALPHA*INT8):.1f} MiB dan kichik bo'lsa u 3-holatga "
              f"o'tadi\nva kaskad past-rank buyuradi — 16-jadvalga ko'ra "
              f"bu WER ni 0.1833 dan 0.6101 ga ko'taradi.")


if __name__ == "__main__":
    main()
