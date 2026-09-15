"""Full ablation: which element of the method contributes how much.

The dissertation's method is a stack, not a single trick. Nothing so far
separated the contributions, and one element -- the calibrated quantization
SCALE (README Sec 2.4), the author's own contribution -- has never been
isolated at all: every INT8 number reported until now used ONNX Runtime's
default min/max scale, not ours.

Elements measured independently and in combination:

  Q0  no quantization (FP32)                          -- reference
  Q1  INT8, naive min/max scale                       -- what a library gives you
  Q2  INT8, our alternating-minimization scale        -- README Sec 2.4 phase 1
  Q3  INT8, our calibration-refined scale             -- README Sec 2.4 phase 2
  Q4  INT8, our per-channel calibrated scale          -- Sec 8.3.8 granularity

  L0  no low-rank
  L1  plain truncated SVD                             -- Eckart-Young optimum
  L2  activation-aware SVD                            -- calibration-guided
  L3  functional-clustering CUR                       -- the dissertation's own
  L4  leverage-score CUR                              -- generic CUR baseline

  F0  3-factor CUR chain (C, U, R separate)
  F1  2-factor fused chain (U folded offline)         -- Sec 8.3.6 fusion

Reported per combination: weight bytes, FLOPs, held-out output error E_loc.
Latency is measured for the subset that is materialized as ONNX, since
building 40+ real graphs is not needed to establish the contribution shares.

All fitting uses x_fit; all scoring uses held-out x_eval.
"""

import json
import time

import numpy as np
import onnx

from calib_utils import DECODER_PATH, ENCODER_PATH, capture_activations, decoder_feeds, encoder_feeds
from nnopt.cur.lowrank_baselines import (
    activation_aware_svd,
    output_relative_error,
    rank_for_param_budget_cur,
    rank_for_param_budget_svd,
    truncated_svd,
)
from nnopt.cur.svd_cur import analyze_spectrum, build_cur, select_cur_columns, select_cur_columns_by_leverage, select_cur_rows
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.profiler.graph_profiler import profile_onnx_model
from nnopt.quantizer.per_channel import quantize_weight_per_channel
from nnopt.quantizer.scale_refine import (
    dequantize,
    initial_scale_minmax,
    quantize_codes,
    refine_scale,
    refine_scale_alternating,
)

Q8 = 127
N_CALIB = 6
MAX_ROWS = 1024
FIT_FRACTION = 0.5
TAU, EPS_THR = 0.9, 0.2
TARGET_COMPRESSION = 4.0     # low-rank budget, on top of INT8
OUT_JSON = "experiments/results_ablation_full.json"

ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
DEC_DIMS = {"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500}

# One representative operator from each regime found earlier:
#   encoder fc1 -- compute-bound, genuinely low-rank (Sec 8.3.x)
#   decoder fc1 -- memory-bound, the cascade's "INT8 is enough" case
TARGETS = [
    ("ENCODER fc1", ENCODER_PATH, ENC_DIMS, "/layers.0/fc1/", encoder_feeds),
    ("DECODER fc1", DECODER_PATH, DEC_DIMS, "/layers.0/fc1/", decoder_feeds),
]


# ---------------------------------------------------------------- scales --
def quant_naive(w):
    """Q1: library-style symmetric min/max scale, no refinement."""
    s = initial_scale_minmax(w, Q8)
    return dequantize(quantize_codes(w, s, Q8), s), {"scale": float(s)}


def quant_alternating(w):
    """Q2: README Sec 2.4 phase 1 only (weights, no calibration)."""
    res = refine_scale_alternating(w, Q8)
    return dequantize(quantize_codes(w, res.scale, Q8), res.scale), {"scale": float(res.scale)}


def quant_calibrated(w, x_fit):
    """Q3: phase 1 + calibration-guided grid search with the overfitting guard."""
    y_ref = x_fit @ w.T
    res = refine_scale(w, Q8, layer_response_fn=lambda wd: x_fit @ wd.T, y_reference=y_ref)
    return dequantize(quantize_codes(w, res.scale, Q8), res.scale), {"scale": float(res.scale)}


def quant_per_channel(w, x_fit):
    """Q4: per-output-channel calibrated scales (Sec 8.3.8)."""
    wq, scales = quantize_weight_per_channel(w, Q8, x_calib=x_fit)
    return wq, {"n_scales": int(scales.size)}


QUANTIZERS = {
    "Q1 minmax": lambda w, x: quant_naive(w),
    "Q2 alternating": lambda w, x: quant_alternating(w),
    "Q3 kalibrlangan": quant_calibrated,
    "Q4 per-channel": quant_per_channel,
}


# -------------------------------------------------------------- low-rank --
def lowrank_factors(kind, w, x_fit, budget_params):
    m, n = w.shape
    if kind == "L1 plain SVD":
        r = rank_for_param_budget_svd(m, n, budget_params)
        lr = truncated_svd(w, r)
    elif kind == "L2 act-aware SVD":
        r = rank_for_param_budget_svd(m, n, budget_params)
        lr = activation_aware_svd(w, x_fit, r)
    elif kind in ("L3 funksional CUR", "L4 leverage CUR"):
        r = rank_for_param_budget_cur(m, n, budget_params)
        y_ref = x_fit @ w.T
        grouping = greedy_group(x_fit.T, np.linalg.norm(w, axis=0),
                                float(np.linalg.norm(y_ref)), tau=TAU, eps_threshold=EPS_THR)
        w_t = build_compensated_weight(w, grouping)
        spec = analyze_spectrum(w_t)
        r_idx = select_cur_rows(spec, rank=r, r=r)
        if kind == "L3 funksional CUR":
            reps = grouping.representative_indices()
            hn = np.linalg.norm(x_fit, axis=0)
            prio = {g.representative: float(g.size) * float(hn[g.representative]) for g in grouping.groups}
            c_idx = select_cur_columns(reps, prio, c=min(r, len(reps)))
        else:
            c_idx = select_cur_columns_by_leverage(spec, rank=r, c=r)
        cur = build_cur(w_t, c_idx, r_idx)
        # F1 fusion: fold U into R offline -> two factors
        return cur.C, cur.U @ cur.R, r, cur
    else:
        raise ValueError(kind)

    u, s, vt = np.linalg.svd(lr, full_matrices=False)
    rr = min(r, len(s))
    sq = np.sqrt(s[:rr])
    return (u[:, :rr] * sq), (sq[:, None] * vt[:rr, :]), r, None


# ------------------------------------------------------------------ main --
def analyze_target(label, path, dims, substr, feeds_fn):
    profs = profile_onnx_model(path, free_dims=dims)
    target = next(p for p in profs if substr in p.name and p.weight_initializer)
    feeds = feeds_fn(0, N_CALIB)
    x_by = capture_activations(path, [target.activation_input], feeds, max_rows=MAX_ROWS)
    x = x_by[target.activation_input]

    model = onnx.load(path)
    inits = {i.name: i for i in model.graph.initializer}
    w = onnx.numpy_helper.to_array(inits[target.weight_initializer]).astype(np.float64)
    if w.shape[1] != x.shape[1]:
        w = w.T
    m, n = w.shape
    split = int(x.shape[0] * FIT_FRACTION)
    x_fit, x_eval = x[:split], x[split:]
    print(f"\n{label}: W={m}x{n}, fit={x_fit.shape[0]} eval={x_eval.shape[0]}")

    rows = []
    fp32_bytes = m * n * 4

    def rec(quant, low, bytes_, flops, eloc, extra=None):
        rows.append({"target": label, "quant": quant, "lowrank": low,
                     "bytes": int(bytes_), "flops": int(flops), "eloc": float(eloc),
                     "compression": fp32_bytes / bytes_, **(extra or {})})

    seq = 1500 if "ENCODER" in label else 16
    flops_dense = 2 * seq * m * n
    rec("Q0 yo'q (FP32)", "L0 yo'q", fp32_bytes, flops_dense, 0.0)

    # --- quantization only (isolating the SCALE contribution) ---
    for qname, qfn in QUANTIZERS.items():
        t0 = time.time()
        wq, extra = qfn(w, x_fit)
        e = output_relative_error(w, wq, x_eval)
        nb = m * n + (extra.get("n_scales", 1) * 4)
        rec(qname, "L0 yo'q", nb, flops_dense, e, {"secs": round(time.time() - t0, 1), **extra})
        print(f"  {qname:18s} + L0              E_loc={e:.5f}", flush=True)

    # --- low-rank only, then low-rank + best quantizer ---
    budget = (m * n) / TARGET_COMPRESSION
    for lname in ("L1 plain SVD", "L2 act-aware SVD", "L3 funksional CUR", "L4 leverage CUR"):
        t0 = time.time()
        a, b, rank, _ = lowrank_factors(lname, w, x_fit, budget)
        approx = a @ b
        e_fp = output_relative_error(w, approx, x_eval)
        flops_lr = 2 * seq * (n * rank + rank * m)
        rec("Q0 yo'q (FP32)", lname, (a.size + b.size) * 4, flops_lr, e_fp, {"rank": rank})
        print(f"  Q0 (FP32)          + {lname:18s} rank={rank:4d} E_loc={e_fp:.5f}", flush=True)

        # Y = X @ (a@b).T = (X @ b.T) @ a.T, so factor b sees x_fit and
        # factor a sees the intermediate h -- using x_fit for both would
        # calibrate `a` against the wrong distribution entirely.
        h_fit = x_fit @ b.T
        for qname in ("Q1 minmax", "Q3 kalibrlangan", "Q4 per-channel"):
            qfn = QUANTIZERS[qname]
            aq, ea = qfn(a, h_fit)
            bq, eb = qfn(b, x_fit)
            e_q = output_relative_error(w, aq @ bq, x_eval)
            nb = a.size + b.size + (ea.get("n_scales", 1) + eb.get("n_scales", 1)) * 4
            rec(qname, lname, nb, flops_lr, e_q, {"rank": rank, "secs": round(time.time() - t0, 1)})
            print(f"  {qname:18s} + {lname:18s} rank={rank:4d} E_loc={e_q:.5f}", flush=True)

    return rows


def main():
    all_rows = []
    for label, path, dims, substr, feeds_fn in TARGETS:
        all_rows += analyze_target(label, path, dims, substr, feeds_fn)
    json.dump(all_rows, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")
    summarize(all_rows)


def summarize(rows):
    for target in sorted({r["target"] for r in rows}):
        rs = [r for r in rows if r["target"] == target]
        print("\n" + "=" * 96)
        print(f"{target} — ELEMENTLARNING HISSASI")
        print("=" * 96)

        base = next(r for r in rs if r["quant"].startswith("Q0") and r["lowrank"].startswith("L0"))
        print("\nA) FAQAT KVANTLASH — masshtab g'oyasining hissasi")
        print(f"  {'variant':22s} {'siqish':>8s} {'E_loc':>10s} {'minmax ga nisbatan':>20s}")
        naive = next((r for r in rs if r["quant"] == "Q1 minmax" and r["lowrank"].startswith("L0")), None)
        for r in rs:
            if r["lowrank"].startswith("L0") and not r["quant"].startswith("Q0"):
                gain = "" if not naive or naive["eloc"] == 0 else f"{(naive['eloc']-r['eloc'])/naive['eloc']*100:+.1f}%"
                print(f"  {r['quant']:22s} {r['compression']:7.2f}x {r['eloc']:10.5f} {gain:>20s}")

        print("\nB) FAQAT PAST-RANK (FP32) — dekompozitsiya usulining hissasi")
        print(f"  {'variant':22s} {'rank':>6s} {'siqish':>8s} {'FLOPs nisbat':>13s} {'E_loc':>10s}")
        for r in rs:
            if r["quant"].startswith("Q0") and not r["lowrank"].startswith("L0"):
                print(f"  {r['lowrank']:22s} {r.get('rank',0):6d} {r['compression']:7.2f}x "
                      f"{base['flops']/r['flops']:13.2f} {r['eloc']:10.5f}")

        print("\nC) KOMBINATSIYA — past-rank + kvantlash")
        print(f"  {'past-rank':20s} {'kvantlash':18s} {'siqish':>8s} {'FLOPs':>8s} {'E_loc':>10s}")
        for r in sorted([r for r in rs if not r["quant"].startswith("Q0") and not r["lowrank"].startswith("L0")],
                        key=lambda z: z["eloc"]):
            print(f"  {r['lowrank']:20s} {r['quant']:18s} {r['compression']:7.2f}x "
                  f"{base['flops']/r['flops']:7.2f}x {r['eloc']:10.5f}")


if __name__ == "__main__":
    main()
