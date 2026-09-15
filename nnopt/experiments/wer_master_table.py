"""Every measured WER in one place, grouped so incomparable rows cannot be read
as if they were comparable.

WER results in this project were produced across several evaluation settings
and the numbers are NOT interchangeable between them: the same FP32 encoder
scores 0.0417 on the 24-utterance smoke set, 0.0931 on 100 validation
utterances and 0.1793 on the 300-utterance test split. Putting them in one
column would invent differences that are only the evaluation set changing, so
every row here carries its split and n, and the table is grouped by them.

Sizes come from the artifacts where a result file records them, and from the
interleaved latency library otherwise -- never from the blocked one, whose
timings were withdrawn (Sec 4.7).

Usage:  python experiments/wer_master_table.py
"""

import json
import os

EXP = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(EXP, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def sizes_from_latency():
    d = load("results_latency_library_interleaved.json")
    if not d:
        return {}
    return {k: v.get("mib") for k, v in d.get("natijalar", {}).items()}


LIB_LABELS = {
    "fp32": "FP32 (tayanch)",
    "int8": "INT8 (GPTQ)",
    "t99": "tau = 0.99 + INT8",
    "t97": "tau = 0.97 + INT8",
    "t95": "tau = 0.95 + INT8",
    "t93": "tau = 0.93 + INT8",
    "t90": "tau = 0.90 + INT8",
    "r45": "45% kanal (majburiy) + INT8",
    "mag30": "magnitude 30% + INT8",
    "mag50": "magnitude 50% + INT8",
    "lr": "low-rank + INT8",
    "cur": "CUR + INT8",
    "axis": "o'q gibridi (kanal L0-5, rank L6+)",
    "r45b": "45% kanal, muqobil taqsimot + INT8",
}


def rows_config_library(mib_by_key):
    d = load("results_config_library.json")
    if not d:
        return []
    out = []
    for k, v in d.items():
        if not isinstance(v, dict) or "wer" not in v:
            continue
        out.append({
            # The library varies the ENCODER and holds the decoder at INT8,
            # but scores the whole pipeline, so both the size and the WER are
            # whole-model figures: FP32 sits at 1610.1 MiB = 1172.2 encoder +
            # 437.9 INT8 decoder, not at the encoder's own 1172.2.
            "guruh": f"Enkoder konfiguratsiyalari (dekoder INT8 da qat'iy) — "
                     f"butun model, {v.get('split', 'test')}, "
                     f"n={v.get('n', 300)}",
            "konfiguratsiya": LIB_LABELS.get(k, k),
            "mib": v.get("mib") or mib_by_key.get(k),
            "wer": v["wer"], "cer": v.get("cer"),
            "lo": v.get("wer_lo"), "hi": v.get("wer_hi"),
        })
    return out


def rows_testsplit():
    d = load("results_final_wer_testsplit.json")
    if not isinstance(d, list):
        return []
    return [{
        "guruh": f"Yakuniy baholash — {r.get('split', 'test')}, n={r.get('n')}",
        "konfiguratsiya": f"{r.get('axis', '')}: {r.get('variant', '')}".strip(": "),
        "mib": r.get("mib"), "wer": r["wer"], "cer": r.get("cer"),
        "lo": r.get("wer_lo"), "hi": r.get("wer_hi"),
    } for r in d if "wer" in r]


def rows_whole_model():
    d = load("results_whole_model_cascade.json")
    if not isinstance(d, list):
        return []
    return [{
        "guruh": f"Butun model (enkoder+dekoder) — test, n={r.get('n')}",
        "konfiguratsiya": r.get("variant", "?"),
        "mib": r.get("mib"), "wer": r["wer"], "cer": r.get("cer"),
        "lo": r.get("wer_lo"), "hi": r.get("wer_hi"),
    } for r in d if "wer" in r]


def rows_baselines():
    d = load("results_structural_baselines.json")
    if not d:
        return []
    out = []
    stack = [("", d)]
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, dict):
            if "wer" in node:
                out.append({
                    "guruh": "Teng byudjetda baseline taqqoslash — test, n=300",
                    "konfiguratsiya": prefix or "?",
                    "mib": node.get("mib"), "wer": node["wer"],
                    "cer": node.get("cer"), "lo": node.get("wer_lo"),
                    "hi": node.get("wer_hi"),
                })
            else:
                for k, v in node.items():
                    stack.append((f"{prefix} / {k}".strip(" /"), v))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                stack.append((f"{prefix}[{i}]", v))
    return out


def rows_l3_12():
    d = load("results_l3_12_eval.json")
    if not isinstance(d, dict):
        return []
    return [{
        "guruh": f"L3 = 12 MiB stsenariysi — {v.get('split', 'test')}, "
                 f"n={v.get('n', 300)}",
        "konfiguratsiya": k, "mib": v.get("mib"), "wer": v["wer"],
        "cer": v.get("cer"), "lo": v.get("wer_lo"), "hi": v.get("wer_hi"),
    } for k, v in d.items() if isinstance(v, dict) and "wer" in v]


def rows_order():
    d = load("results_whisper_order_ab.json")
    if not isinstance(d, dict):
        return []
    names = {"pruneFirst": "qirqish -> kvantlash (freymvork tartibi)",
             "quantFirst": "kvantlash -> qirqish"}
    return [{
        "guruh": "Bosqichlar tartibi A/B — test",
        "konfiguratsiya": names.get(k, k), "mib": v.get("mib"),
        "wer": v["wer"], "cer": v.get("cer"),
        "lo": v.get("wer_lo"), "hi": v.get("wer_hi"),
    } for k, v in d.items() if isinstance(v, dict) and "wer" in v]


def rows_bias():
    d = load("results_bias_correction_e2e.json")
    if not isinstance(d, dict):
        return []
    names = {"rtn": "RTN (bias tuzatishsiz)", "rtn+bias": "RTN + bias tuzatish"}
    return [{
        "guruh": "Bias tuzatish ablatsiyasi — test",
        "konfiguratsiya": names.get(k, k), "mib": v.get("mib"),
        "wer": v["wer"], "cer": v.get("cer"),
        "lo": v.get("wer_lo"), "hi": v.get("wer_hi"),
    } for k, v in d.items() if isinstance(v, dict) and "wer" in v]


def rows_gptq():
    d = load("results_gptq_pruning.json")
    if not isinstance(d, dict):
        return []
    return [{
        "guruh": "Qirqish + GPTQ (2x2 tajribadan) — validation",
        "konfiguratsiya": k, "mib": v.get("mib"), "wer": v["wer"],
        "cer": v.get("cer"), "lo": v.get("wer_lo"), "hi": v.get("wer_hi"),
    } for k, v in d.items() if isinstance(v, dict) and "wer" in v]


def fmt(v, w, d=4, suffix=""):
    return f"{v:{w}.{d}f}{suffix}" if isinstance(v, (int, float)) else \
        " " * (w - len(suffix)) + "—" + suffix


def main():
    mib_by_key = sizes_from_latency()
    rows = (rows_config_library(mib_by_key) + rows_whole_model()
            + rows_baselines() + rows_l3_12() + rows_order() + rows_bias()
            + rows_gptq() + rows_testsplit())
    if not rows:
        raise SystemExit("WER natijalari topilmadi")

    groups = {}
    for r in rows:
        groups.setdefault(r["guruh"], []).append(r)

    for guruh, items in groups.items():
        items.sort(key=lambda r: r["wer"])
        print("\n" + "=" * 96)
        print(guruh)
        print("=" * 96)
        print(f"{'Konfiguratsiya':44s} {'MiB':>8s} {'WER':>8s} "
              f"{'95% CI':>19s} {'CER':>8s}")
        print("-" * 96)
        base = next((r for r in items
                     if "FP32" in r["konfiguratsiya"].upper()), None)
        for r in items:
            ci = (f"[{r['lo']:.4f}, {r['hi']:.4f}]"
                  if isinstance(r.get("lo"), (int, float)) else "—")
            print(f"{r['konfiguratsiya'][:44]:44s} {fmt(r['mib'], 8, 1)} "
                  f"{r['wer']:8.4f} {ci:>19s} {fmt(r['cer'], 8)}")
        if base and base["mib"]:
            print("-" * 96)
            print(f"  (tayanch: {base['konfiguratsiya']}, "
                  f"{base['mib']:.1f} MiB, WER {base['wer']:.4f})")
    print(f"\n{len(rows)} qator, {len(groups)} guruh. Guruhlar orasida "
          f"WER qiymatlarini taqqoslash mumkin emas — baholash to'plami farq qiladi.")


if __name__ == "__main__":
    main()
