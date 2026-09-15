"""What a 12 MiB cache target costs, measured rather than argued.

The encoder built by l3_12_cascade.py meets the derived budget exactly -- 45%
of the channels gone from every layer -- and the tau each layer had to descend
to is the interesting part of the build log: 0.99 in the shallow layers, 0.30
by layer 19. Where tau collapses the criterion is no longer selecting
redundant channels; it is ranking which non-redundant channel to sacrifice.
This puts a WER on that.

Three arms, all sharing the INT8 decoder and the same GPTQ pass, so the only
thing that varies is how much the cache target demanded of the encoder:

  L3 = 24, kaskad     tau = 0.99, 17% mean removal, 267 MiB   [measured, 0.1833]
  L3 = 12, qat'iy     45% every layer, tau descended, 213 MiB [built here]
  L3 = 12, yumshoq    tau = 0.90, 33% mean, criterion respected

The soft arm is the policy this work actually adopts: cache residency is a
target, not a gate, so where the budget outruns the criterion the cascade
takes what the criterion endorses and stops. The hard arm exists to price the
alternative.
"""

import json
import os

import numpy as np

from cascade_runner import evaluate, model_mib

DEC_INT8 = "models/_whole_net/dec_int8.onnx"
OUT_JSON = "experiments/results_l3_12_eval.json"
N_EVAL = int(os.environ.get("N_EVAL", "300"))
SPLIT = os.environ.get("SPLIT", "test")

ARMS = [
    ("L3=12 qat'iy kanal (45%/qatlam)", "models/_l3_12/enc_l3_12_gptq.onnx"),
    ("L3=12 o'q gibridi (kanal L0-5, rank L6+)",
     "models/_hybrid/enc_axis_hybrid_gptq.onnx"),
    ("L3=12 yumshoq (tau=0.90, 33%)", "models/_l3_12/enc_soft_tau0.9_gptq.onnx"),
]
# The tau = 0.99 cascade at 267 MiB is the reference point and was already
# measured on these same 300 test utterances (Sec 4.9b); re-decoding it would
# cost half an hour to reproduce a number we have.
REFERENCE = ("L3=24 kaskad (tau=0.99, 17%)", 0.1833, 705.0)


def paired_ci(a, b, n=2000, seed=1):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rows = {}
    if os.path.exists(OUT_JSON):
        try:
            rows = json.load(open(OUT_JSON, encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            rows = {}

    for label, enc in ARMS:
        if label in rows:
            print(f"[{label}] keshdan WER={rows[label]['wer']:.4f}")
            continue
        if not os.path.exists(enc):
            print(f"[{label}] SKIP — {enc} yo'q")
            continue
        mib = model_mib(enc) + model_mib(DEC_INT8)
        print(f"[{label}] {mib:.0f} MiB — baholanmoqda ({SPLIT}, {N_EVAL})...",
              flush=True)
        r = evaluate(enc, DEC_INT8, N_EVAL, SPLIT)
        r.update({"enc": enc, "mib": mib})
        rows[label] = r
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
        print(f"  WER={r['wer']:.4f} [{r['wer_lo']:.4f}, {r['wer_hi']:.4f}]  "
              f"CER={r['cer']:.4f}", flush=True)

    print("\n" + "=" * 84)
    print(f"L3 = 12 MiB MAQSADINING NARXI ({N_EVAL} namuna, '{SPLIT}' split)")
    print("=" * 84)
    print(f"{'Arm':42s} {'MiB':>7s} {'WER':>8s} {'95% IO':>20s}")
    print("-" * 84)
    rlab, rwer, rmib = REFERENCE
    print(f"{rlab:42s} {rmib:7.0f} {rwer:8.4f}   (oldingi o'lchov)")
    for label, _ in ARMS:
        r = rows.get(label)
        if r:
            print(f"{label:42s} {r['mib']:7.0f} {r['wer']:8.4f} "
                  f"  [{r['wer_lo']:.4f}, {r['wer_hi']:.4f}]")

    # The decisive pair: same 213 MiB, same GPTQ, same budget -- only the
    # choice of axis in the deep layers differs.
    hard, hybrid = rows.get(ARMS[0][0]), rows.get(ARMS[1][0])
    if hard and hybrid:
        d, lo, hi = paired_ci(hybrid["per_sample_wer"], hard["per_sample_wer"])
        v = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
        print(f"\nTeng byudjetda o'q tanlovi (gibrid - qat'iy kanal): "
              f"dWER={d:+.4f} [{lo:+.4f}, {hi:+.4f}]  {v}")
        print(f"  ikkalasi ham {hard['mib']:.0f} MiB, farq faqat "
              f"L6+ qatlamlarda kanal o'rniga rank")

    soft = rows.get(ARMS[2][0])
    if soft:
        for label in (ARMS[0][0], ARMS[1][0]):
            r = rows.get(label)
            if not r:
                continue
            d, lo, hi = paired_ci(r["per_sample_wer"], soft["per_sample_wer"],
                                  seed=3)
            v = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
            print(f"\n{label} vs yumshoq (tau=0.90): "
                  f"dWER={d:+.4f} [{lo:+.4f}, {hi:+.4f}] {v}")
            print(f"  xotira {soft['mib']:.0f} -> {r['mib']:.0f} MiB")


if __name__ == "__main__":
    main()
