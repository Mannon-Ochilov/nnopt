"""Selecting tau against the confidence bound instead of the point estimate.

Sec 4.9e proposed this as a correction but did not run it. The correction
matters because the point-estimate rule demonstrably failed: the validation
scan picked tau = 0.97 for a 5% budget, and on the independent test split
that configuration came in at 1.069x FP32 -- outside the budget it was
selected to satisfy. tau = 0.99 would have satisfied it, at 1.022x.

The failure is not that the scan was wrong about the numbers; it is that the
scan was asked a question it could not answer. At n = 100 the paired interval
is about +-0.019 while the spread across tau in {0.99, 0.97, 0.95} is 0.0046,
so the ordering inside that group was noise. A selection rule that reads only
the point estimate has no way to notice this and will confidently return the
luckiest candidate.

Reading the interval instead makes the rule state its own uncertainty. For
budget eps, tau is admissible only if

    upper bound of (WER_tau - WER_FP32)  <=  eps * WER_FP32

so a candidate whose interval reaches past the budget is refused even when
its point estimate looks comfortable. Two consequences follow, and both are
desirable: the rule becomes conservative, and when NO candidate qualifies it
is reporting that the scan is too small rather than that compression is
impossible.

This runs on the stored per-utterance errors from the existing scan -- no
model is rebuilt and nothing is decoded again -- and checks the selection it
produces against what the test split already established.
"""

import json
import os

import numpy as np

SCAN_JSON = "experiments/results_tau_curve.json"
TEST_JSON = "experiments/results_final_wer_testsplit.json"
BOOT = 10000
EPSILONS = (0.01, 0.02, 0.05, 0.10)

# tau -> (scan label, test label). The test labels are what Sec 4.9e already
# measured on 300 utterances, used here only to score the selection rule.
TAUS = {
    0.99: ("tau=0.99", "qisqartirish + GPTQ"),
    0.97: ("tau=0.97", "t97 bizniki + GPTQ (eps=5% tanlovi)"),
    0.95: ("tau=0.95", "t95 bizniki + GPTQ"),
    0.93: ("tau=0.93", None),
    0.90: ("tau=0.90", None),
}
MIB = {0.99: 267, 0.97: 261, 0.95: 254, 0.93: 248, 0.90: 237}


def paired(a, b, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(path, key):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {r[key]: r for r in json.load(f)}


def main():
    scan = load(SCAN_JSON, "variant")
    test = {k: v for k, v in load(TEST_JSON, "variant").items()}
    test = {k: v for k, v in test.items() if v.get("n") == 300}

    fp32_scan = scan.get("FP32 (mos yozuvlar)")
    fp32_test = test.get("FP32")
    if not fp32_scan:
        print("skanerlash natijalari topilmadi")
        return

    print(f"Skanerlash: n = {fp32_scan['n']}, FP32 WER = {fp32_scan['wer']:.4f}")
    if fp32_test:
        print(f"Test      : n = {fp32_test['n']}, FP32 WER = {fp32_test['wer']:.4f}")
    print()

    stats = {}
    print(f"{'tau':>6s} {'MiB':>5s} {'WER':>8s} {'dWER':>9s} "
          f"{'95% IO yuqori chegara':>23s}")
    print("-" * 56)
    for t in sorted(TAUS, reverse=True):
        label, _ = TAUS[t]
        r = scan.get(label)
        if not r:
            continue
        d, lo, hi = paired(r["per_sample_wer"], fp32_scan["per_sample_wer"])
        stats[t] = {"wer": r["wer"], "d": d, "lo": lo, "hi": hi}
        print(f"{t:6.2f} {MIB[t]:5d} {r['wer']:8.4f} {d:+9.4f} {hi:+23.4f}")

    print("\n" + "=" * 78)
    print("TANLOV QOIDASI: nuqtaviy baho va ishonch chegarasi")
    print("=" * 78)
    for eps in EPSILONS:
        allow = eps * fp32_scan["wer"]
        point = [t for t in stats if stats[t]["d"] <= allow]
        ucb = [t for t in stats if stats[t]["hi"] <= allow]
        pick_p = min(point, key=lambda t: MIB[t]) if point else None
        pick_u = min(ucb, key=lambda t: MIB[t]) if ucb else None
        print(f"\neps = {eps:4.0%}  (ruxsat etilgan dWER <= {allow:.4f})")
        print(f"  nuqtaviy baho bo'yicha : "
              f"{f'tau = {pick_p:.2f}, {MIB[pick_p]} MiB' if pick_p else 'yo`q'}")
        print(f"  ishonch chegarasi b-cha: "
              f"{f'tau = {pick_u:.2f}, {MIB[pick_u]} MiB' if pick_u else 'yo`q'}")

        # Score both against what the test split actually showed.
        for name, pick in (("nuqtaviy", pick_p), ("chegara", pick_u)):
            if pick is None or not fp32_test:
                continue
            tlabel = TAUS[pick][1]
            tr = test.get(tlabel) if tlabel else None
            if not tr:
                print(f"    {name}: test natijasi yo'q")
                continue
            ratio = tr["wer"] / fp32_test["wer"]
            ok = "byudjet bajarildi" if ratio <= 1 + eps else "BYUDJET BUZILDI"
            print(f"    {name} tanlov testda {ratio:.3f}x  ->  {ok}")


if __name__ == "__main__":
    main()
