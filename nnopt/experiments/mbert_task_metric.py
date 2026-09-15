"""Giving mBERT a task metric, so the cross-architecture claim rests on quality.

The third architecture has so far been described only through diagnostics:
how much FFN redundancy the criterion finds, and what the cache target
demands. Whisper is judged by word error rate and open_llama_3b by perplexity,
but mBERT by neither -- so the claim that the cascade's VERDICT transfers has
had nothing to stand on for that model. The coverage audit (Sec 5.2) records
this gap; this closes it.

The metric is masked-language-model accuracy on held-out Uzbek text, with the
pseudo-perplexity of the same predictions alongside. It needs no labelled task
data and no fine-tuned head, and it is the direct analogue of the perplexity
used for Llama: the model is asked to fill positions it did not see, and the
score is how often it is right.

Three arms answer the question the cascade actually poses for this model. Its
verdict on mBERT is that the structural axis has nothing to offer -- the
criterion finds essentially no collinear channels at any depth (Sec 4.11) --
so quantization should be free and forcing channels out should cost. Both
halves of that prediction are testable:

  FP32                 reference
  INT8                 what the cascade prescribes
  20% kanal + INT8     what it refuses, priced

Calibration text is disjoint from the evaluation text, and the masking pattern
is fixed by seed so every arm is scored on identical positions.
"""

import gc
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

from mbert_analysis import MBERT_DIR, MBERT_ONNX, SEQ_LEN, build_text_batches
from nnopt.grouping.functional_grouping import (
    build_compensated_weight,
    greedy_group,
    trim_to_budget,
)
from nnopt.profiler.blocks import find_reducible_pairs
from nnopt.profiler.graph_profiler import profile_onnx_model

OUT_DIR = "models/_mbert_arms"
OUT_JSON = "experiments/results_mbert_task.json"
TEXT_CACHE = "models/_calib_cache/uz_text.npz"
FREE_DIMS = {"batch": 8, "seq": SEQ_LEN}
N_CALIB_TEXT = 400          # matches mbert_analysis.py
N_EVAL_TEXT = 800
MASK_RATE = 0.15
FORCED_REMOVAL = 0.20
EPS_THRESHOLD = 0.5
TAU_GRID = (0.99, 0.9, 0.7, 0.5, 0.3, 0.0, -1.0)
MAX_ROWS = 3072
THREADS = 8


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def load_texts(skip=0, n_calib=N_CALIB_TEXT, n_eval=N_EVAL_TEXT):
    """Calibration and evaluation slices of the Uzbek corpus.

    The slice is addressable rather than fixed. It was fixed at the first 400
    sentences, which made the calibration set invisible: two pruned models
    built from different text would carry the same filename and the same
    cached measurement. That is the failure the tau filenames already
    produced once, and the fix is the same -- make the set an argument and put
    its identity in the name.

    Evaluation always starts after the calibration slice, so the two cannot
    overlap however the slice is moved.
    """
    z = np.load(TEXT_CACHE, allow_pickle=True)
    texts = [str(t) for t in z["texts"]]
    calib = texts[skip:skip + n_calib]
    start = skip + n_calib
    return calib, texts[start:start + n_eval]


def masked_batches(tok, texts, seed=0):
    """Batches with 15% of real tokens masked, plus the targets."""
    rng = np.random.default_rng(seed)
    out = []
    for feed in build_text_batches(tok, texts):
        ids = feed["input_ids"].copy()
        special = np.isin(ids, tok.all_special_ids)
        eligible = (feed["attention_mask"] == 1) & ~special
        pick = eligible & (rng.random(ids.shape) < MASK_RATE)
        if not pick.any():
            continue
        targets = ids[pick]
        ids[pick] = tok.mask_token_id
        out.append(({"input_ids": ids,
                     "attention_mask": feed["attention_mask"],
                     "token_type_ids": feed["token_type_ids"]},
                    pick, targets))
    return out


def score(path, batches):
    """Top-1 accuracy and pseudo-perplexity over the masked positions.

    Per-position outcomes are kept, not just the totals: every arm is scored
    on the SAME masked positions, so the comparison that decides whether two
    arms differ is paired, and a paired interval is far tighter than one built
    from the two accuracies separately.
    """
    sess = session(path)
    hits, nlls = [], []
    for feed, pick, targets in batches:
        logits = sess.run(None, feed)[0][pick]
        logits -= logits.max(axis=-1, keepdims=True)
        lse = np.log(np.exp(logits).sum(axis=-1))
        nlls.extend((lse - logits[np.arange(len(targets)), targets]).tolist())
        hits.extend((logits.argmax(axis=-1) == targets).astype(int).tolist())
    del sess
    gc.collect()
    return {"acc": float(np.mean(hits)), "ppl": float(np.exp(np.mean(nlls))),
            "n_masked": len(hits), "hits": hits}


def paired_delta(a, b, n=2000, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return (float(d.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)))


def build_int8(src, dst):
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=True)
    return dst


def build_pruned(calib_texts, tok, removal=FORCED_REMOVAL, calib_tag="s0n400",
                 criterion="cosine", apply_bias=True):
    """Force `removal` of the FFN channels out of every block, then INT8.

    The reducible pairs are located structurally, not by name: mBERT spells
    its feed-forward `intermediate.dense` / `output.dense`, nothing like
    Whisper's fc1/fc2, and this is the first use of that detector on an
    architecture it was not written against.

    `calib_tag` reaches the filename because the channels kept are a function
    of the calibration text: without it, two models built from different
    sentences are indistinguishable on disk and the second silently returns
    the first.
    """
    # `apply_bias=False` is the ablation arm for the metric-type law: same
    # channels kept, the mean-contribution fold-in withheld. It isolates the
    # bias correction's own effect on accuracy vs pseudo-perplexity.
    btag = "" if apply_bias else "_nobias"
    dst = (f"{OUT_DIR}/mbert_pruned{int(removal*100)}_{criterion}{btag}"
           f"_{calib_tag}_int8.onnx")
    if os.path.exists(dst):
        return dst
    os.makedirs(OUT_DIR, exist_ok=True)

    model = onnx.load(MBERT_ONNX)
    profs = [p for p in profile_onnx_model(MBERT_ONNX, free_dims=FREE_DIMS)
             if p.weight_initializer]
    pairs = find_reducible_pairs(model, profs)
    if not pairs:
        raise SystemExit("mBERT da qisqartiriladigan juftlik topilmadi")
    by_name = {p.name: p for p in profs}
    print(f"  {len(pairs)} ta juftlik topildi, kenglik "
          f"{sorted({p.width for p in pairs})}")

    inits = {i.name: i for i in model.graph.initializer}

    def bias_for(profile, width):
        """The 1-D initializer the Add after `profile` applies, if it is the
        FFN bias. It is indexed by the very axis being shrunk, so leaving it
        at full length would produce a shape mismatch at load time rather
        than a wrong answer -- but only after the build had been paid for."""
        for nd in model.graph.node:
            if nd.op_type != "Add" or profile.output_name not in nd.input:
                continue
            for i in nd.input:
                init = inits.get(i)
                if init is not None and list(init.dims) == [width]:
                    return i
        return None

    from mbert_analysis import capture_ffn_activations
    batches = build_text_batches(tok, calib_texts)
    contract_inputs = sorted({by_name[pr.contract].activation_input
                              for pr in pairs})
    x_by = capture_ffn_activations(contract_inputs, batches, max_rows=MAX_ROWS)

    for pr in pairs:
        exp_p, con_p = by_name[pr.expand], by_name[pr.contract]
        x = x_by[con_p.activation_input]
        w2s = numpy_helper.to_array(inits[con_p.weight_initializer]).astype(np.float64)
        w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
        w1s = numpy_helper.to_array(inits[exp_p.weight_initializer]).astype(np.float64)
        w1 = w1s if w1s.shape[1] == x.shape[1] else w1s.T   # (out, width)

        need_keep = int(round(w2.shape[1] * (1.0 - removal)))
        correction = None
        if criterion == "fluctuation":
            # Keep the channels that vary; the discarded ones are close to
            # constant, so their average contribution is folded into the
            # output bias instead of being dropped. Measured per operator on
            # this model, this beats forcing the cosine criterion down to the
            # budget in every layer (Sec 4.11), which is unsurprising: the
            # cosine criterion looks for collinearity and mBERT has almost
            # none, so forcing it selects among channels it does not consider
            # redundant in the first place.
            mean_h = x.mean(axis=0)
            score = (np.linalg.norm(w2, axis=0) ** 2) * np.var(x, axis=0)
            keep = np.sort(np.argsort(score)[w2.shape[1] - need_keep:])
            w2c = w2
            if apply_bias:
                correction = w2 @ mean_h - w2[:, keep] @ mean_h[keep]
        else:
            chosen = None
            for tau in TAU_GRID:
                eps = EPS_THRESHOLD if tau >= 0.0 else float("inf")
                g = greedy_group(x.T, np.linalg.norm(w2, axis=0),
                                 float(np.linalg.norm(x @ w2.T)), tau=tau,
                                 eps_threshold=eps)
                chosen = g
                if len(g.groups) <= need_keep:
                    break
            trim_to_budget(chosen, need_keep)
            keep = np.array(sorted(gr.representative for gr in chosen.groups))
            w2c = build_compensated_weight(w2, chosen)

        new2 = w2c[:, keep]
        new1 = w1[:, keep]
        inits[con_p.weight_initializer].CopyFrom(numpy_helper.from_array(
            (new2 if w2s.shape[1] == x.shape[1] else new2.T).astype(np.float32),
            con_p.weight_initializer))
        inits[exp_p.weight_initializer].CopyFrom(numpy_helper.from_array(
            (new1 if w1s.shape[1] == x.shape[1] else new1.T).astype(np.float32),
            exp_p.weight_initializer))

        bname = bias_for(exp_p, pr.width)
        if bname is None:
            raise SystemExit(f"{pr.expand}: FFN bias topilmadi — kanal "
                             f"olib tashlash shakl mos kelmasligiga olib "
                             f"keladi")
        b = numpy_helper.to_array(inits[bname]).astype(np.float64)
        inits[bname].CopyFrom(numpy_helper.from_array(
            b[keep].astype(np.float32), bname))

        if correction is not None:
            # The discarded channels were near-constant, not absent: their
            # average contribution belongs in the operator's OWN output bias,
            # which is indexed by d_model rather than by the width being cut.
            obname = bias_for(con_p, w2.shape[0])
            if obname is None:
                raise SystemExit(f"{pr.contract}: chiqish biasi topilmadi — "
                                 f"fluktuatsiya mezoni tuzatishni bias siz "
                                 f"qo'llay olmaydi")
            ob = numpy_helper.to_array(inits[obname]).astype(np.float64)
            inits[obname].CopyFrom(numpy_helper.from_array(
                (ob + correction).astype(np.float32), obname))

        print(f"  L{pr.layer:<2d}: {w2.shape[1]} -> {len(keep)} kanal "
              f"(bias ham qisqartirildi)", flush=True)
        del x, w1, w2, w2c
        gc.collect()

    tmp = f"{OUT_DIR}/_tmp_fp32.onnx"
    onnx.save(model, tmp, save_as_external_data=True,
              location=os.path.basename(tmp) + ".data", size_threshold=1024)
    quantize_dynamic(tmp, dst, weight_type=QuantType.QInt8, per_channel=True)
    for f in (tmp, tmp + ".data"):
        if os.path.exists(f):
            os.remove(f)
    return dst


def main():
    if not os.path.exists(MBERT_ONNX):
        raise SystemExit(f"{MBERT_ONNX} yo'q — avval mbert_prepare.py")
    tok = AutoTokenizer.from_pretrained(MBERT_DIR)
    calib_texts, eval_texts = load_texts()
    batches = masked_batches(tok, eval_texts)
    n_masked = sum(len(t) for _, _, t in batches)
    print(f"{len(batches)} batch, {n_masked} niqoblangan pozitsiya "
          f"(kalibrlash matni bilan kesishmaydi)\n")

    rows = {}
    if os.path.exists(OUT_JSON):
        try:
            rows = json.load(open(OUT_JSON, encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            rows = {}

    arms = [("FP32", MBERT_ONNX)]
    arms.append(("INT8", build_int8(MBERT_ONNX,
                                    f"{OUT_DIR}/mbert_int8.onnx")))
    print(f"\n[{int(FORCED_REMOVAL*100)}% kanal] kaskad RAD ETADIGAN "
          f"o'zgarish quriladi", flush=True)
    arms.append((f"{int(FORCED_REMOVAL*100)}% kanal (kosinus) + INT8",
                 build_pruned(calib_texts, tok)))
    # The operator-level comparison says the fluctuation criterion beats the
    # forced cosine one in every layer of this model. Whether that survives to
    # the task metric is a separate question -- on Whisper an equally clear
    # operator-level advantage did not (Sec 4.9f) -- so it is measured.
    print(f"\n[{int(FORCED_REMOVAL*100)}% kanal, fluktuatsiya mezoni]",
          flush=True)
    arms.append((f"{int(FORCED_REMOVAL*100)}% kanal (fluktuatsiya) + INT8",
                 build_pruned(calib_texts, tok, criterion="fluctuation")))

    print()
    for label, path in arms:
        if label in rows:
            print(f"[{label}] keshdan acc={rows[label]['acc']:.4f}")
            continue
        mib = os.path.getsize(path) / (1024 * 1024)
        print(f"[{label}] {mib:.0f} MiB — baholanmoqda...", flush=True)
        r = score(path, batches)
        r["mib"] = mib
        rows[label] = r
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
        print(f"  aniqlik={r['acc']:.4f}  pseudo-PPL={r['ppl']:.3f}")

    print("\n" + "=" * 72)
    print(f"mBERT MASKED-LM, o'zbek matni ({n_masked} pozitsiya)")
    print("=" * 72)
    print(f"{'Arm':24s} {'MiB':>7s} {'aniqlik':>9s} {'pseudo-PPL':>11s}")
    print("-" * 72)
    base = rows.get("FP32")
    for label, _ in arms:
        r = rows.get(label)
        if r:
            d = f"  ({r['acc']-base['acc']:+.4f})" if base else ""
            print(f"{label:24s} {r['mib']:7.0f} {r['acc']:9.4f} "
                  f"{r['ppl']:11.3f}{d}")

    if base and "hits" in base:
        print("\nFP32 ga nisbatan (juftlik bootstrap, bir xil pozitsiyalar):")
        for label, _ in arms[1:]:
            r = rows.get(label)
            if not r or "hits" not in r:
                continue
            d, lo, hi = paired_delta(r["hits"], base["hits"])
            v = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
            print(f"  {label:24s} d_aniqlik={d:+.4f} [{lo:+.4f}, {hi:+.4f}] {v}")
        pr = rows.get(f"{int(FORCED_REMOVAL*100)}% kanal + INT8")
        i8 = rows.get("INT8")
        if pr and i8 and "hits" in pr and "hits" in i8:
            d, lo, hi = paired_delta(pr["hits"], i8["hits"], seed=2)
            v = "SEZILARLI" if (lo > 0 or hi < 0) else "farqlanmaydi"
            print(f"\nQisqartirish INT8 ustiga: d_aniqlik={d:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] {v}")
            print(f"  qo'shimcha tejash: {i8['mib']-pr['mib']:.0f} MiB")


if __name__ == "__main__":
    main()
