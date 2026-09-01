"""The deliverable: what to ship on each machine, and whether choosing paid.

Every configuration in the library was measured once, because WER belongs to
the artifact rather than to the machine. What the machine changes is which
artifact is worth shipping and what it costs in memory traffic, and both of
those are arithmetic over the measured table -- so six cache sizes cost no
extra decoding at all.

Three policies are compared on each machine:

  ko'r-ko'rona INT8      quantize everything, ship it. The actual default.
  ko'r-ko'rona 30/50%    pick a round pruning ratio, uniform over layers.
  freymvork              among configurations meeting the accuracy budget,
                         the one with the fewest cache misses.

The framework's rule is deliberately NOT "fit in L3". Cache residency is the
objective, not a gate: when no configuration that fits also meets the accuracy
budget -- which is most machines below 24 MiB -- the answer is the least-miss
configuration that does meet it, not the smallest one that fits.

Per-layer bytes are read from each artifact's own initializers at their own
dtype rather than divided out of the file size, so a quantized or factored
model is accounted for as it actually is.

Cache sizes are checked against vendor documentation; only the 24 MiB entry
was measured on the machine here. One of them is not a plain number -- see the
note on MACHINES about EPYC's per-die L3, which is a condition rather than a
size.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import onnx

MIB = 1024.0 ** 2
ALPHA = 0.7
ENC_REUSE, DEC_REUSE = 1500, 1
DEC_INT8 = "models/_whole_net/dec_int8.onnx"
LIB_JSON = "experiments/results_config_library.json"
OUT_JSON = "experiments/results_machine_table.json"
WER_BASE = 0.1761

# (label, L3 MiB, note). Sizes checked against vendor documentation; only the
# 24 MiB row was measured here. The EPYC row carries a condition rather than a
# number alone: its L3 is not globally shared. Each of the eight core complex
# dies has its own 96 MiB (32 MiB plus 64 MiB of stacked V-Cache), and the
# 768 MiB headline figure is their sum, not a pool any core can draw on. Since
# this method is anchored on the cache a set of cores is GUARANTEED to share,
# 96 MiB is the right number only when the working threads sit on one die --
# and on a chiplet CPU "the L3" is not one thing.
MACHINES = [
    ("Raspberry Pi 5 (BCM2712)", 2.0, "4 x Cortex-A76, umumiy"),
    ("Intel N100 (Alder Lake-N)", 6.0, "4 yadro, umumiy"),
    ("Core i5-1235U", 12.0, "2P + 8E, umumiy"),
    ("Tiger Lake H (bizniki)", 24.0, "16 mantiqiy yadro, o'lchangan"),
    ("Ryzen 7 5800X", 32.0, "8 yadro, bitta CCD, umumiy"),
    ("EPYC 7773X (Milan-X)", 96.0, "CCD boshiga; jami 768 MiB, global emas"),
]
BUDGETS = [("nisbiy eps=0.05", WER_BASE * 1.05),
           ("mutlaq d=0.05", WER_BASE + 0.05)]
BLIND_KEYS = ("int8", "mag30", "mag50")

DTYPE_BYTES = {1: 4, 2: 1, 3: 1, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8,
               16: 2}


def layer_index(name):
    m = re.search(r"(?:^|[./])layers?[./](\d+)[./]", name)
    return int(m.group(1)) if m else -1


def per_layer_bytes(path):
    """Bytes of weight per transformer layer, at the artifact's own dtypes.

    Attribution goes through the NODES, not the initializer names. Dynamic
    quantization renames the weight tensors -- `layers.0.fc1.weight` becomes
    `onnx::MatMul_2508_quantized` -- so a name-based sum silently finds only
    the biases and layer norms and reports layers a tenth of their real size.
    The consuming node keeps its path (`/layers.0/self_attn/q_proj/...`), so
    each initializer is charged to the layer of the node that reads it.
    """
    model = onnx.load(path, load_external_data=False)
    sizes = {i.name: (int(np.prod(i.dims)) if i.dims else 1)
             * DTYPE_BYTES.get(i.data_type, 4)
             for i in model.graph.initializer}

    by_layer = defaultdict(int)
    charged = set()
    for node in model.graph.node:
        li = layer_index(node.name)
        if li < 0:
            for t in list(node.output) + list(node.input):
                li = layer_index(t)
                if li >= 0:
                    break
        if li < 0:
            continue
        for inp in node.input:
            if inp in sizes and inp not in charged:
                by_layer[li] += sizes[inp]
                charged.add(inp)          # never count a shared weight twice

    if not by_layer:
        raise ValueError(f"{path}: qatlamli operatorlar topilmadi")
    counted = sum(by_layer.values())
    total = sum(sizes.values())
    if counted < 0.5 * total:
        raise ValueError(f"{path}: qatlamlarga atigi {counted/total:.0%} "
                         f"vazn taqsimlandi — atribusiya ishonchsiz")
    return dict(by_layer)


def miss_bytes(layers, budget, reuse):
    return sum(b + max(0.0, b - budget) * (reuse - 1) for b in layers.values())


def main():
    if not os.path.exists(LIB_JSON):
        raise SystemExit(f"{LIB_JSON} yo'q; avval config_library.py")
    lib = json.load(open(LIB_JSON, encoding="utf-8-sig"))
    dec_layers = per_layer_bytes(DEC_INT8)

    # Per-layer profile of every configuration, read from the artifact.
    prof = {}
    for key, rec in lib.items():
        try:
            prof[key] = per_layer_bytes(rec["enc"])
        except (ValueError, OSError) as e:
            print(f"  ogohlantirish: {key} o'tkazib yuborildi ({e})")
    print(f"{len(prof)} konfiguratsiya, {len(dec_layers)} dekoder qatlami\n")

    print(f"{'kalit':7s} {'konfiguratsiya':38s} {'enk/qatlam':>11s} {'WER':>8s}")
    print("-" * 70)
    for key in prof:
        big = max(prof[key].values())
        print(f"{key:7s} {lib[key]['label']:38s} {big/MIB:10.2f}M "
              f"{lib[key]['wer']:8.4f}")

    out = {}
    for bname, cap in BUDGETS:
        print("\n" + "=" * 100)
        print(f"ANIQLIK BYUDJETI: {bname}  ->  WER <= {cap:.4f}")
        print("=" * 100)
        eligible = [k for k in prof if lib[k]["wer"] <= cap]
        print(f"byudjetni qanoatlantiradi: "
              f"{', '.join(eligible) if eligible else 'hech biri'}\n")
        print(f"{'mashina':24s} {'L3':>5s} {'byudjet':>8s}  "
              f"{'freymvork tanlovi':32s} {'miss':>10s}  vs ko'r-ko'rona INT8")
        print("-" * 100)

        for mname, l3, note in MACHINES:
            budget = ALPHA * l3 * MIB
            dec_miss = miss_bytes(dec_layers, budget, DEC_REUSE)

            def total_miss(k):
                return miss_bytes(prof[k], budget, ENC_REUSE) + dec_miss

            row = {"machine": mname, "l3": l3, "note": note,
                   "budget_mib": budget / MIB}
            if eligible:
                pick = min(eligible, key=total_miss)
                m_pick = total_miss(pick)
                m_int8 = total_miss("int8") if "int8" in prof else float("nan")
                fits = max(prof[pick].values()) <= budget
                gain = m_int8 / m_pick if m_pick else float("nan")
                blind_ok = "int8" in eligible
                note = f"{gain:6.2f}x kam miss" if gain == gain else "—"
                if pick == "int8":
                    note = "bir xil tanlov"
                print(f"{mname:24s} {l3:5.0f} {budget/MIB:7.1f}M  "
                      f"{lib[pick]['label'][:32]:32s} {m_pick/MIB:9.0f}M  "
                      f"{note}"
                      + ("" if blind_ok else "   [INT8 byudjetdan chiqadi]")
                      + ("" if fits else "   [sig'maydi]"))
                row.update({"pick": pick, "pick_wer": lib[pick]["wer"],
                            "pick_miss_mib": m_pick / MIB,
                            "int8_miss_mib": m_int8 / MIB,
                            "gain": gain, "fits": bool(fits),
                            "blind_int8_within_budget": blind_ok})
            else:
                print(f"{mname:24s} {l3:5.0f} {budget/MIB:7.1f}M  "
                      f"{'hech bir konfiguratsiya byudjetni qanoatlantirmaydi':32s}")
                row["pick"] = None
            out.setdefault(bname, []).append(row)

        # What the blind ratios would have shipped, budget ignored.
        print(f"\n  ko'r-ko'rona nisbatlar (byudjetga qaramay):")
        for k in BLIND_KEYS:
            if k in prof:
                ok = "byudjet ichida" if lib[k]["wer"] <= cap else "BYUDJETDAN CHIQADI"
                print(f"    {lib[k]['label']:38s} WER={lib[k]['wer']:.4f}  {ok}")

    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
