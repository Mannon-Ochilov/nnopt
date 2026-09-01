"""Is the cascade's stage ORDER load-bearing? A direct A/B, nothing else varied.

The paper argues prune-then-quantize is the right order because the quantizer
must see the compensated weights (the 188x row-range expansion is what GPTQ /
per-channel scales adapt to). The evidence so far is indirect -- the chain in
Sec 4.4 and the 2x2. This is the direct test:

  A (cascade order):  fold gamma-compensation into FP32 weights
                          -> per-channel INT8
  B (reverse order):  per-channel INT8 first (weights replaced by their
                      dequantized values) -> fold the SAME gammas into the
                      already-quantized weights -> per-channel INT8 again

Arm B is the honest executable version of "quantize first": compensation is
a linear combination of INT8-representable values and is NOT representable
in INT8 (the closure argument), so a deployable artifact must re-quantize --
and that second quantization acts on weights the first quantizer never saw.
Channel selection and gamma are computed ONCE on FP32 activations and shared
by both arms, so the only difference is the order.

Model: mBERT, forced 20% cosine removal (the regime where compensation
carries the load). Metric: masked-LM accuracy on identical positions, paired.

Pre-registered readings:
  B worse (CI excludes 0)  -> order is load-bearing, measured directly.
  A ~ B                    -> the order argument rests on the 188x chain
                              alone and the paper must say so.
"""

import json
import os

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from mbert_analysis import MBERT_DIR, MBERT_ONNX, FREE_DIMS
from mbert_task_metric import (
    MAX_ROWS,
    TAU_GRID,
    EPS_THRESHOLD,
    build_text_batches,
    load_texts,
    masked_batches,
    paired_delta,
    score,
)
from nnopt.grouping.functional_grouping import (
    build_compensated_weight,
    greedy_group,
    trim_to_budget,
)
from nnopt.profiler.blocks import find_reducible_pairs
from nnopt.profiler.graph_profiler import profile_onnx_model

REMOVAL = 0.20
OUT_DIR = "models/_mbert_metric"
OUT_JSON = "experiments/results_order_ab.json"
Q_MAX = 127


def rtn_pc(w):
    """Per-output-channel symmetric RTN, dequantized. Output channels are
    rows for the (out, in) layout and columns otherwise; resolved by the
    caller passing weights in (out, in)."""
    s = np.max(np.abs(w), axis=1, keepdims=True) / Q_MAX
    s[s == 0] = 1.0
    return np.round(w / s) * s


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MBERT_DIR)
    calib_texts, eval_texts = load_texts(0, 400)

    model = onnx.load(MBERT_ONNX)
    profs = [p for p in profile_onnx_model(MBERT_ONNX, free_dims=FREE_DIMS)
             if p.weight_initializer]
    pairs = find_reducible_pairs(model, profs)
    by_name = {p.name: p for p in profs}
    inits = {i.name: i for i in model.graph.initializer}

    from mbert_analysis import capture_ffn_activations
    batches = build_text_batches(tok, calib_texts)
    contract_inputs = sorted({by_name[pr.contract].activation_input
                              for pr in pairs})
    x_by = capture_ffn_activations(contract_inputs, batches,
                                   max_rows=MAX_ROWS)

    # One shared decision per layer: keep set + grouping (gammas).
    decisions = {}
    for pr in pairs:
        con_p = by_name[pr.contract]
        x = x_by[con_p.activation_input]
        w2s = numpy_helper.to_array(inits[con_p.weight_initializer]) \
            .astype(np.float64)
        w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
        need_keep = int(round(w2.shape[1] * (1.0 - REMOVAL)))
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
        decisions[pr.expand] = (keep, chosen)
        print(f"  {pr.expand}: {w2.shape[1]} -> {len(keep)}", flush=True)

    def build_arm(tag, quant_first):
        dst = f"{OUT_DIR}/mbert_orderAB_{tag}_int8.onnx"
        if os.path.exists(dst):
            return dst
        m = onnx.load(MBERT_ONNX)
        ii = {i.name: i for i in m.graph.initializer}
        for pr in pairs:
            exp_p, con_p = by_name[pr.expand], by_name[pr.contract]
            x = x_by[con_p.activation_input]
            keep, chosen = decisions[pr.expand]

            w2s = numpy_helper.to_array(ii[con_p.weight_initializer]) \
                .astype(np.float64)
            tr2 = w2s.shape[1] != x.shape[1]
            w2 = w2s.T if tr2 else w2s          # (d_model, width)
            w1s = numpy_helper.to_array(ii[exp_p.weight_initializer]) \
                .astype(np.float64)
            tr1 = w1s.shape[1] != x.shape[1]
            w1 = w1s.T if tr1 else w1s          # (out, width)

            if quant_first:
                # Per-output-channel: output channels are the rows in both
                # (d_model, width) and (out, width) layouts used here.
                w2 = rtn_pc(w2)
                w1 = rtn_pc(w1)
            w2c = build_compensated_weight(w2, chosen)

            new2, new1 = w2c[:, keep], w1[:, keep]
            ii[con_p.weight_initializer].CopyFrom(numpy_helper.from_array(
                (new2.T if tr2 else new2).astype(np.float32),
                con_p.weight_initializer))
            ii[exp_p.weight_initializer].CopyFrom(numpy_helper.from_array(
                (new1.T if tr1 else new1).astype(np.float32),
                exp_p.weight_initializer))

            for nd in m.graph.node:              # FFN bias shrink (as before)
                if nd.op_type != "Add" or exp_p.output_name not in nd.input:
                    continue
                for i_ in nd.input:
                    init = ii.get(i_)
                    if init is not None and list(init.dims) == [pr.width]:
                        b = numpy_helper.to_array(init).astype(np.float64)
                        init.CopyFrom(numpy_helper.from_array(
                            b[keep].astype(np.float32), i_))
        tmp = dst.replace(".onnx", "_fp32.onnx")
        onnx.save(m, tmp)
        quantize_dynamic(tmp, dst, weight_type=QuantType.QInt8,
                         per_channel=True)
        os.remove(tmp)
        return dst

    os.makedirs(OUT_DIR, exist_ok=True)
    ev = masked_batches(tok, eval_texts[:400])
    results = {}
    for tag, qf in (("pruneFirst", False), ("quantFirst", True)):
        print(f"[{tag}] qurilmoqda/baholanmoqda...", flush=True)
        path = build_arm(tag, qf)
        results[tag] = score(path, ev)
        print(f"  aniqlik={results[tag]['acc']:.4f}  "
              f"ppl={results[tag]['ppl']:.2f}", flush=True)

    d, lo, hi = paired_delta(results["quantFirst"]["hits"],
                             results["pruneFirst"]["hits"])
    print("\n" + "=" * 70)
    print(f"A kesish->kvantlash : aniqlik {results['pruneFirst']['acc']:.4f}"
          f"  ppl {results['pruneFirst']['ppl']:.2f}")
    print(f"B kvantlash->kesish : aniqlik {results['quantFirst']['acc']:.4f}"
          f"  ppl {results['quantFirst']['ppl']:.2f}")
    print(f"farq (B - A) = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("=" * 70)
    json.dump({t: {"acc": r["acc"], "ppl": r["ppl"]}
               for t, r in results.items()} | {"delta": [d, lo, hi]},
              open(OUT_JSON, "w"), indent=2)


if __name__ == "__main__":
    main()

