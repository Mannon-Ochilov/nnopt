"""CUR's last defensible advantage, tested: does it survive quantization
better than SVD?

Sec 8.3.9's result is blunt -- at equal parameter budget, activation-aware
SVD beats functional CUR on held-out output error 135/135. But that
comparison is in FP32, and in practice low-rank is never deployed alone; it
is combined with INT8. That opens the one structural argument CUR still
has:

    C and R are literal columns and rows of W, so they inherit W's value
    distribution. SVD factors (U*S, V^T) do not -- singular values span
    orders of magnitude, so the factor entries have a far wider dynamic
    range, which is exactly what a fixed-point grid handles badly.

If that matters, CUR should lose less accuracy when both are quantized, and
the FP32 gap should narrow or close. This measures it at EQUAL TOTAL BYTES
(quantized factor bytes + fp32 scale bytes), which is the only fair budget
once the two schemes store different things.

Methods, all scored on HELD-OUT calibration rows:
  svd_actaw_fp32 / svd_actaw_int8   activation-aware SVD, before/after
  cur_func_fp32  / cur_func_int8    functional CUR,       before/after

Quantization uses our own per-channel calibrated scales
(nnopt.quantizer.per_channel), i.e. the dissertation's own Sec 2.4
mechanism, not a library default.
"""

import json
import time

import numpy as np
import onnx

from calib_utils import DECODER_PATH, capture_activations, decoder_feeds, weight_for_operator, weighted_matmul_profiles
from nnopt.cur.lowrank_baselines import (
    activation_aware_svd,
    output_relative_error,
    rank_for_param_budget_cur,
    rank_for_param_budget_svd,
)
from nnopt.cur.svd_cur import analyze_spectrum, build_cur, select_cur_columns, select_cur_rows
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.quantizer.per_channel import quantize_weight_per_channel

N_CALIB = 12
MAX_ROWS = 1024
FIT_FRACTION = 0.5
SAMPLED_LAYERS = [0, 12, 23]
CACHE_ANCHORED = 3.81
COMPRESSIONS = [2.0, CACHE_ANCHORED, 8.0]
TAU, EPS_THR = 0.9, 0.2
Q8 = 127
OUT_JSON = "experiments/results_lowrank_quant.json"
FREE_DIMS = {"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500}
SKIP_KINDS = ("/fc2/",)


def svd_factors(w, x_fit, rank):
    """Recover explicit (A, B) with W ~= A @ B from the activation-aware
    rank-r solution, so the two factors can be quantized separately the
    same way CUR's C and R are."""
    w_lr = activation_aware_svd(w, x_fit, rank)
    u, s, vt = np.linalg.svd(w_lr, full_matrices=False)
    r = min(rank, len(s))
    # split the spectrum evenly between the factors: keeps their dynamic
    # ranges comparable, which is the fairest possible setup for SVD here.
    sq = np.sqrt(s[:r])
    return (u[:, :r] * sq), (sq[:, None] * vt[:r, :])


def cur_factors(w, x_fit, rank):
    y_ref = x_fit @ w.T
    grouping = greedy_group(
        x_fit.T, np.linalg.norm(w, axis=0), float(np.linalg.norm(y_ref)),
        tau=TAU, eps_threshold=EPS_THR,
    )
    w_tilde = build_compensated_weight(w, grouping)
    reps = grouping.representative_indices()
    h_norms = np.linalg.norm(x_fit, axis=0)
    prio = {g.representative: float(g.size) * float(h_norms[g.representative]) for g in grouping.groups}
    spec = analyze_spectrum(w_tilde)
    r_idx = select_cur_rows(spec, rank=rank, r=rank)
    c_idx = select_cur_columns(reps, prio, c=min(rank, len(reps)))
    cur = build_cur(w_tilde, c_idx, r_idx)
    # fuse U into R once, offline (README Sec 8.3.6 fusion result): two
    # factors, directly comparable to SVD's two.
    return cur.C, cur.U @ cur.R


def bytes_for(factors_int8, scale_counts):
    return sum(f.size for f in factors_int8) + scale_counts * 4


def main():
    profs = weighted_matmul_profiles(DECODER_PATH, FREE_DIMS)
    ops = [
        p for p in profs
        if any(f"/layers.{li}/" in p.name for li in SAMPLED_LAYERS)
        and not any(k in p.name for k in SKIP_KINDS)
    ]
    print(f"{len(ops)} operators across layers {SAMPLED_LAYERS}")

    tensors = sorted({p.activation_input for p in ops if p.activation_input != "encoder_hidden_states"})
    feeds = decoder_feeds(0, N_CALIB)
    x_by_tensor = capture_activations(DECODER_PATH, tensors, feeds, max_rows=MAX_ROWS)
    if any(p.activation_input == "encoder_hidden_states" for p in ops):
        x_by_tensor["encoder_hidden_states"] = np.concatenate(
            [f["encoder_hidden_states"].reshape(-1, 1024) for f in feeds], axis=0
        )[:MAX_ROWS].astype(np.float64)

    dec = onnx.load(DECODER_PATH)
    inits = {i.name: i for i in dec.graph.initializer}

    results = []
    for k, p in enumerate(ops, 1):
        x = x_by_tensor.get(p.activation_input)
        if x is None:
            continue
        w = weight_for_operator(dec, inits, p, x)
        if w.shape[1] != x.shape[1]:
            continue
        m, n = w.shape
        split = int(x.shape[0] * FIT_FRACTION)
        x_fit, x_eval = x[:split], x[split:]

        for comp in COMPRESSIONS:
            budget = m * n / comp
            t0 = time.time()

            a_s, b_s = svd_factors(w, x_fit, rank_for_param_budget_svd(m, n, budget))
            c_c, r_c = cur_factors(w, x_fit, rank_for_param_budget_cur(m, n, budget))

            a_q, a_sc = quantize_weight_per_channel(a_s, Q8)
            b_q, b_sc = quantize_weight_per_channel(b_s, Q8)
            c_q, c_sc = quantize_weight_per_channel(c_c, Q8)
            r_q, r_sc = quantize_weight_per_channel(r_c, Q8)

            e_svd_fp = output_relative_error(w, a_s @ b_s, x_eval)
            e_svd_q = output_relative_error(w, a_q @ b_q, x_eval)
            e_cur_fp = output_relative_error(w, c_c @ r_c, x_eval)
            e_cur_q = output_relative_error(w, c_q @ r_q, x_eval)

            row = {
                "name": p.name, "compression": comp,
                "bytes_svd": bytes_for([a_s, b_s], a_sc.size + b_sc.size),
                "bytes_cur": bytes_for([c_c, r_c], c_sc.size + r_sc.size),
                "eloc": {
                    "svd_fp32": e_svd_fp, "svd_int8": e_svd_q,
                    "cur_fp32": e_cur_fp, "cur_int8": e_cur_q,
                },
                # how much each method LOSES by being quantized -- the
                # actual hypothesis under test
                "quant_damage": {
                    "svd": e_svd_q - e_svd_fp,
                    "cur": e_cur_q - e_cur_fp,
                },
                "secs": time.time() - t0,
            }
            results.append(row)
            print(
                f"[{k}/{len(ops)}] {p.name.replace('/model/decoder/','')[:34]:34s} {comp:.2f}x "
                f"svd {e_svd_fp:.4f}->{e_svd_q:.4f} (+{row['quant_damage']['svd']:.4f})  "
                f"cur {e_cur_fp:.4f}->{e_cur_q:.4f} (+{row['quant_damage']['cur']:.4f}) "
                f"[{row['secs']:.0f}s]",
                flush=True,
            )

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")
    summarize(results)


def summarize(results):
    if not results:
        return
    print("\n" + "=" * 92)
    print("PAST-RANK + INT8 -- kvantlashga chidamlilik (held-out E_loc)")
    print("=" * 92)
    print(f"{'siqish':>8s} {'#':>4s} {'svd fp32':>10s} {'svd int8':>10s} {'cur fp32':>10s} {'cur int8':>10s}"
          f" {'svd zarar':>11s} {'cur zarar':>11s}")
    for comp in sorted({r["compression"] for r in results}):
        rs = [r for r in results if r["compression"] == comp]
        g = lambda key, sub: np.mean([r[key][sub] for r in rs])
        print(
            f"{comp:7.2f}x {len(rs):4d} "
            f"{g('eloc','svd_fp32'):10.4f} {g('eloc','svd_int8'):10.4f} "
            f"{g('eloc','cur_fp32'):10.4f} {g('eloc','cur_int8'):10.4f} "
            f"{g('quant_damage','svd'):11.4f} {g('quant_damage','cur'):11.4f}"
        )
    print("\nGipoteza: CUR kvantlashdan KAMROQ zarar ko'rishi kerak (C, R asl vazn taqsimotini saqlaydi).")
    less = sum(1 for r in results if r["quant_damage"]["cur"] < r["quant_damage"]["svd"])
    print(f"  CUR zarari < SVD zarari : {less}/{len(results)}")
    wins = sum(1 for r in results if r["eloc"]["cur_int8"] < r["eloc"]["svd_int8"])
    print(f"  CUR+int8 mutlaq g'alaba : {wins}/{len(results)}")


if __name__ == "__main__":
    main()
