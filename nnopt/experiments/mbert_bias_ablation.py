"""Mechanism proof for the metric-type law: ablate the bias fold-in on mBERT.

The law (three architectures): additive bias correction improves argmax
metrics and degrades likelihood metrics, because it shifts logit MEANS. On
mBERT the two effects were observed together -- fluctuation+bias beats the
cosine arm on accuracy while its pseudo-perplexity is WORSE (109.91 vs
103.22) -- but the attribution to the bias term was an argument.

The ablation isolates it: same criterion, same kept channels, bias fold-in
withheld. Pre-registered predictions:

  accuracy(nobias)  <  accuracy(with bias)      (the argmax gain came from
                                                 the bias)
  pseudo-PPL(nobias) <  pseudo-PPL(with bias)   (the likelihood damage came
                                                 from the bias)

If instead both move together, the mechanism is NOT the mean shift and the
law's explanation must be withdrawn to an observation.
"""

import json

import numpy as np
from transformers import AutoTokenizer

from mbert_analysis import MBERT_DIR
from mbert_task_metric import (
    FORCED_REMOVAL,
    build_pruned,
    load_texts,
    masked_batches,
    score,
)

OUT_JSON = "experiments/results_mbert_bias_ablation.json"


def paired(a, b, n=2000, seed=1):
    a, b = np.array(a, float), np.array(b, float)
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


def main():
    tok = AutoTokenizer.from_pretrained(MBERT_DIR)
    calib_texts, eval_texts = load_texts(0, 400)
    batches = masked_batches(tok, eval_texts[:400])

    arms = {}
    for name, kw in (("bias bilan", dict(apply_bias=True)),
                     ("biassiz", dict(apply_bias=False))):
        print(f"[{name}] qurilmoqda/baholanmoqda...", flush=True)
        path = build_pruned(calib_texts, tok, removal=FORCED_REMOVAL,
                            calib_tag="s0n400", criterion="fluctuation", **kw)
        arms[name] = score(path, batches)
        print(f"  aniqlik={arms[name]['acc']:.4f}  "
              f"pseudo-PPL={arms[name]['ppl']:.2f}", flush=True)

    d, lo, hi = paired(arms["biassiz"]["hits"], arms["bias bilan"]["hits"])
    print("\n" + "=" * 70)
    print(f"{'arm':14s} {'aniqlik':>10s} {'pseudo-PPL':>12s}")
    for name, r in arms.items():
        print(f"{name:14s} {r['acc']:10.4f} {r['ppl']:12.2f}")
    print(f"\naniqlik farqi (bias - biassiz) = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("=" * 70)
    json.dump({k: {"acc": v["acc"], "ppl": v["ppl"]}
               for k, v in arms.items()}
              | {"delta_acc": [d, lo, hi]},
              open(OUT_JSON, "w"), indent=2)
    print(f"saqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()

