"""One registry of every encoder configuration, measured once.

The machine table needs WER for a dozen configurations across six cache sizes,
and the naive way to get it is to walk a ladder per machine -- which would be
days of decoding. It is unnecessary, because of a fact worth stating plainly:

    WER is a property of the ARTIFACT, not of the machine.

L3 changes which artifact you should pick and what it costs in memory traffic;
it does not change how well a given artifact transcribes. So every
configuration is measured once here, on the same 300 test utterances with the
same decoder, and every cache size is then answered by arithmetic over this
one table.

Results already produced by earlier scripts are imported rather than
re-decoded, matched by the encoder path they were measured with. Re-running
them would cost half an hour each to reproduce numbers that exist.
"""

import json
import os

import numpy as np

from cascade_runner import evaluate, model_mib

DEC_INT8 = "models/_whole_net/dec_int8.onnx"
OUT_JSON = "experiments/results_config_library.json"
N_EVAL = int(os.environ.get("N_EVAL", "300"))
SPLIT = os.environ.get("SPLIT", "test")

# (key, label, encoder path, family)
#
# Ordered cheapest-first on purpose. Everything importable resolves in
# milliseconds and lands in the JSON immediately; the unquantized FP32 encoder
# is the slowest arm by far and goes last, because an earlier run put it first
# and was interrupted forty minutes in with nothing written.
LIBRARY = [
    ("int8", "ko'r-ko'rona INT8 (qisqartirishsiz)",
     "models/_gptq/enc_gptq_only.onnx", "blind"),
    ("mag30", "ko'r-ko'rona magnitude 30%",
     "models/_blind/enc_magnitude_r0.30_gptq.onnx", "blind"),
    ("mag50", "ko'r-ko'rona magnitude 50%",
     "models/_blind/enc_magnitude_r0.50_gptq.onnx", "blind"),
    ("t99", "bizniki tau=0.99 (17%)", "models/_gptq/enc_gptq_pruned.onnx", "tau"),
    ("t97", "bizniki tau=0.97 (20%)",
     "models/_ratio_sweep/enc_bizniki_tau0.97_gptq.onnx", "tau"),
    ("t95", "bizniki tau=0.95 (24%)",
     "models/_ratio_sweep/enc_bizniki_tau0.95_gptq.onnx", "tau"),
    ("t93", "bizniki tau=0.93 (27%)",
     "models/_ratio_sweep/enc_bizniki_tau0.93_gptq.onnx", "tau"),
    ("t90", "bizniki tau=0.90 (33%)",
     "models/_l3_12/enc_soft_tau0.9_gptq.onnx", "tau"),
    ("r45", "kesh-majburiy 45% (tau panjarasi + trim)",
     "models/_l3_12/enc_l3_12_gptq.onnx", "ratio"),
    ("r45b", "kesh-majburiy 45% (uzluksiz tau bisektsiyasi)",
     "models/_l3_12/enc_l3_12_bisect_gptq.onnx", "ratio"),
    ("axis", "o'q gibridi (kanal L0-5, rank L6+)",
     "models/_hybrid/enc_axis_hybrid_gptq.onnx", "hybrid"),
    ("fp32", "FP32 enkoder (nazorat nuqtasi)",
     "models/uzbek_stt_v1_onnx/encoder_model.onnx", "reference"),
]

# Measurements from earlier runs, keyed by the encoder they were taken with.
IMPORT_FROM = [
    ("experiments/results_final_wer_testsplit.json", "variant"),
    ("experiments/results_l3_12_eval.json", None),
]
IMPORT_BY_LABEL = {
    "int8": ("experiments/results_final_wer_testsplit.json", "GPTQ yolg'iz"),
    "t99": ("experiments/results_final_wer_testsplit.json", "qisqartirish + GPTQ"),
    "t97": ("experiments/results_final_wer_testsplit.json",
            "t97 bizniki + GPTQ (eps=5% tanlovi)"),
    "t95": ("experiments/results_final_wer_testsplit.json", "t95 bizniki + GPTQ"),
    "t93": ("experiments/results_final_wer_testsplit.json", "t93 bizniki + GPTQ"),
    "t90": ("experiments/results_l3_12_eval.json",
            "L3=12 yumshoq (tau=0.90, 33%)"),
    "r45": ("experiments/results_l3_12_eval.json",
            "L3=12 qat'iy kanal (45%/qatlam)"),
    "axis": ("experiments/results_l3_12_eval.json",
             "L3=12 o'q gibridi (kanal L0-5, rank L6+)"),
}


def load(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return None


def import_measured(key):
    """A previously measured WER for this configuration, if one exists."""
    ref = IMPORT_BY_LABEL.get(key)
    if not ref:
        return None
    data = load(ref[0])
    if data is None:
        return None
    rec = None
    if isinstance(data, dict):
        rec = data.get(ref[1])
    else:
        rec = next((r for r in data if r.get("variant") == ref[1]), None)
    if not rec or rec.get("n") not in (None, N_EVAL):
        return None
    return {k: rec[k] for k in
            ("wer", "cer", "wer_lo", "wer_hi", "per_sample_wer") if k in rec}


def main():
    rows = load(OUT_JSON) or {}

    for key, label, path, family in LIBRARY:
        if key in rows:
            print(f"[{key:6s}] keshdan WER={rows[key]['wer']:.4f}")
            continue
        if not os.path.exists(path):
            print(f"[{key:6s}] SKIP — {path} yo'q")
            continue

        # Sizes are read ONCE, before the expensive part. An earlier version
        # looked them up again after evaluate() to fill in the record, and a
        # transient filesystem error on that second lookup threw away a WER
        # that had just taken half an hour to decode. Nothing that can fail
        # belongs after the measurement.
        try:
            enc_mib = model_mib(path)
            total_mib = enc_mib + model_mib(DEC_INT8)
        except OSError as e:
            print(f"[{key:6s}] hajm o'qilmadi ({e}); nan bilan davom etiladi")
            enc_mib = total_mib = float("nan")

        rec = import_measured(key)
        src = "oldingi yugurishdan"
        if rec is None:
            print(f"[{key:6s}] {label} — {total_mib:.0f} MiB, baholanmoqda "
                  f"({SPLIT}, {N_EVAL})...", flush=True)
            rec = evaluate(path, DEC_INT8, N_EVAL, SPLIT)
            src = "shu yerda o'lchandi"

        rec.update({"key": key, "label": label, "enc": path, "family": family,
                    "enc_mib": enc_mib, "mib": total_mib, "source": src})
        rows[key] = rec
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
        print(f"  WER={rec['wer']:.4f} "
              f"[{rec.get('wer_lo', float('nan')):.4f}, "
              f"{rec.get('wer_hi', float('nan')):.4f}]  ({src})", flush=True)

    print("\n" + "=" * 92)
    print(f"KONFIGURATSIYALAR KUTUBXONASI ({N_EVAL} namuna, '{SPLIT}' split)")
    print("=" * 92)
    print(f"{'kalit':7s} {'oila':9s} {'konfiguratsiya':38s} {'enk MiB':>8s} "
          f"{'WER':>8s} {'CER':>8s}")
    print("-" * 92)
    for key, label, _, family in LIBRARY:
        r = rows.get(key)
        if r:
            print(f"{key:7s} {family:9s} {label:38s} {r['enc_mib']:8.0f} "
                  f"{r['wer']:8.4f} {r.get('cer', float('nan')):8.4f}")


if __name__ == "__main__":
    main()
