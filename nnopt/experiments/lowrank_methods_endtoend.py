"""CUR vs SVD at the TASK level -- the dissertation's core idea, end to end.

svd_vs_cur_decisive.py compared four low-rank methods across 135 decoder
operators and found functional-clustering CUR beating leverage-score CUR in
134 of them. That is the central claim of the original design, and it has
only ever been measured as E_loc. Sec 4.7 established that operator error and
task error can diverge by 40x, so the comparison has to be repeated where the
claim actually lives: word error rate.

Four encoder variants are built, differing ONLY in how the rank-r subspace is
chosen:

  svd_plain   truncated SVD                -- weight-optimal, no calibration
  svd_actaw   activation-aware SVD         -- output-optimal at its rank
  cur_func    functional-clustering CUR    -- ours
  cur_lev     leverage-score CUR           -- generic CUR baseline

Cost is matched by DEPLOYMENT, not by storage. CUR's C U R product is folded
to (C U) R before export, so every variant ships two matrices of exactly
r(m+n) parameters and the r^2 term never reaches the graph. That is the
favourable framing for CUR -- the penalty the paper charges it in Table 9
disappears -- and it is also what one would actually deploy, so any remaining
difference is attributable to the SUBSPACE each method picks rather than to
its bookkeeping.

Applied to the 48 FFN operators the cascade puts in case 3 (24 fc1 + 24 fc2),
attention left at INT8, rank 409 to match the existing 203 MB low-rank
artifact. Quality is then scored by final_wer_testsplit.py on 300 utterances
of the independent test split.
"""

import gc
import json
import os
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from calib_utils import (
    ENCODER_PATH,
    capture_activations,
    encoder_feeds,
    weighted_matmul_profiles,
)
from nnopt.cur.lowrank_baselines import activation_aware_svd, truncated_svd
from nnopt.cur.svd_cur import (
    analyze_spectrum,
    build_cur,
    select_cur_columns,
    select_cur_columns_by_leverage,
    select_cur_rows,
)
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group

OUT_DIR = "models/_lr_methods"
OUT_JSON = "experiments/results_lowrank_methods.json"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
RANK = 409
N_CALIB = 8
MAX_ROWS = 2048
TAU, EPS_THR = 0.9, 0.2
METHODS = ("svd_plain", "svd_actaw", "cur_func", "cur_lev")


def ffn_ops(profs):
    return [p for p in profs if "/fc1/" in p.name or "/fc2/" in p.name]


def factor_two_matmul(w_approx, rank):
    """Any dense rank-r matrix -> (A, B) with W ~= A B, r(m+n) parameters.

    Folding through an SVD of the approximation is exact: it re-expresses the
    same matrix, so no method gains or loses accuracy here. It only equalises
    the deployed form so the four variants ship identical shapes.
    """
    u, s, vt = np.linalg.svd(w_approx, full_matrices=False)
    rr = max(1, min(rank, int(np.sum(s > 0))))
    sq = np.sqrt(s[:rr])
    return (u[:, :rr] * sq), (sq[:, None] * vt[:rr, :])


def approximations(w, x, rank):
    """All four rank-`rank` approximations of one operator.

    The two CUR flavours share the compensated matrix and the row choice;
    only COLUMN selection differs, which is precisely the claimed
    contribution, so this keeps everything else identical between them.
    """
    out = {"svd_plain": truncated_svd(w, rank),
           "svd_actaw": activation_aware_svd(w, x, rank)}

    y_ref = x @ w.T
    grouping = greedy_group(x.T, np.linalg.norm(w, axis=0),
                            float(np.linalg.norm(y_ref)),
                            tau=TAU, eps_threshold=EPS_THR)
    w_tilde = build_compensated_weight(w, grouping)
    reps = grouping.representative_indices()
    h_norms = np.linalg.norm(x, axis=0)
    prio = {g.representative: float(g.size) * float(h_norms[g.representative])
            for g in grouping.groups}
    spec = analyze_spectrum(w_tilde)
    r_idx = select_cur_rows(spec, rank=rank, r=rank)

    out["cur_func"] = build_cur(
        w_tilde, select_cur_columns(reps, prio, c=min(rank, len(reps))),
        r_idx).reconstruct()
    out["cur_lev"] = build_cur(
        w_tilde, select_cur_columns_by_leverage(spec, rank=rank, c=rank),
        r_idx).reconstruct()
    return out, len(grouping.groups)


def splice(method, factors, path_fp, path_q):
    """Replace every FFN MatMul with the method's two-matrix chain."""
    model = onnx.load(ENCODER_PATH)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replacement, dead = {}, set()

    for name, (a, b, w_init, act_in, out_name) in factors.items():
        dead.add(w_init)
        base = name.replace("/", "_").replace(".", "_")
        g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{base}_B"))
        g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{base}_A"))
        replacement[out_name] = [
            helper.make_node("MatMul", [act_in, f"{base}_B"], [f"{base}_h"],
                             name=f"{base}_lr1"),
            helper.make_node("MatMul", [f"{base}_h", f"{base}_A"], [out_name],
                             name=f"{base}_lr2"),
        ]

    rebuilt = []
    for nd in g.node:
        hit = next((o for o in nd.output if o in replacement), None)
        rebuilt.extend(replacement[hit] if hit else [nd])
    del g.node[:]
    g.node.extend(rebuilt)
    kept = [i for i in g.initializer if i.name not in dead]
    del g.initializer[:]
    g.initializer.extend(kept)

    onnx.save(model, path_fp)
    quantize_dynamic(path_fp, path_q, weight_type=QuantType.QInt8, per_channel=True)
    os.remove(path_fp)
    return os.path.getsize(path_q) / (1024 * 1024)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {m: f"{OUT_DIR}/enc_{m}_r{RANK}_int8.onnx" for m in METHODS}
    if all(os.path.exists(p) for p in paths.values()):
        print("barcha variantlar mavjud:")
        for m, p in paths.items():
            print(f"  {m:12s} {os.path.getsize(p)/(1024*1024):.0f} MiB")
        return

    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    ops = ffn_ops(profs)
    print(f"{len(ops)} ta FFN operatori, rank {RANK}\n")

    feeds = encoder_feeds(0, N_CALIB)
    factors = {m: {} for m in METHODS}
    elocs = {m: [] for m in METHODS}
    t0 = time.time()

    for k, p in enumerate(ops, 1):
        x_by = capture_activations(ENCODER_PATH, [p.activation_input], feeds,
                                   max_rows=MAX_ROWS)
        x = x_by.get(p.activation_input)
        if x is None:
            continue
        w = numpy_helper.to_array(
            {i.name: i for i in onnx.load(ENCODER_PATH).graph.initializer}
            [p.weight_initializer]).astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        if w.shape[1] != x.shape[1]:
            continue

        approx, n_groups = approximations(w, x, RANK)
        y_ref = x @ w.T
        yn = np.linalg.norm(y_ref) + 1e-9
        for m, wa in approx.items():
            elocs[m].append(float(np.linalg.norm(y_ref - x @ wa.T) / yn))
            a, b = factor_two_matmul(wa, RANK)
            factors[m][p.name] = (a, b, p.weight_initializer,
                                  p.activation_input, p.output_name)
        print(f"  {k}/{len(ops)} {p.name.split('/')[-2]:>4s}  "
              f"guruh={n_groups:5d}  " +
              "  ".join(f"{m}={elocs[m][-1]:.4f}" for m in METHODS)
              + f"  [{time.time()-t0:.0f}s]", flush=True)
        del x_by, x, w, approx
        gc.collect()

    print()
    sizes = {}
    for m in METHODS:
        print(f"[{m}] eksport qilinmoqda...", flush=True)
        sizes[m] = splice(m, factors[m], f"{OUT_DIR}/_tmp.onnx", paths[m])
        print(f"  {paths[m]}  {sizes[m]:.0f} MiB")

    summary = {m: {"eloc_mean": float(np.mean(elocs[m])),
                   "eloc_max": float(np.max(elocs[m])),
                   "mib": sizes[m], "rank": RANK,
                   "wins_vs_cur_lev": int(sum(
                       1 for a, b in zip(elocs[m], elocs["cur_lev"]) if a < b))}
               for m in METHODS}
    json.dump(summary, open(OUT_JSON, "w"), indent=2)

    print("\n" + "=" * 78)
    print(f"OPERATOR DARAJASI ({len(elocs['cur_lev'])} operator, rank {RANK})")
    print("=" * 78)
    print(f"{'usul':14s} {'ort. E_loc':>12s} {'maks E_loc':>12s} "
          f"{'MiB':>6s} {'cur_lev dan ustun':>19s}")
    for m in METHODS:
        s = summary[m]
        print(f"{m:14s} {s['eloc_mean']:12.5f} {s['eloc_max']:12.5f} "
              f"{s['mib']:6.0f} {s['wins_vs_cur_lev']:14d}/{len(elocs[m])}")
    print("\nWER uchun: final_wer_testsplit.py (ENCODERS ro'yxatiga qo'shilgan).")


if __name__ == "__main__":
    main()
