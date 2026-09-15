"""THE decisive low-rank comparison (README Sec 7.4, extended after 8.3.7).

Question the dissertation must answer before a reviewer asks it: given that
truncated SVD is provably optimal in Frobenius weight error AND cheaper per
rank than CUR (r(m+n) vs r(m+n)+r^2), what does CUR actually buy?

Four methods, compared at a MATCHED PARAMETER BUDGET (each picks the highest
rank it can afford, so CUR is not flattered by a matched-rank comparison):

  svd_plain   truncated SVD               -- weight-optimal, calibration-free
  svd_actaw   activation-aware SVD        -- OUTPUT-optimal at its rank; the
                                             strongest honest competitor
  cur_func    functional-clustering CUR   -- ours (README Sec 2.3, Fig 2.7/2.8)
  cur_lev     leverage-score CUR          -- generic CUR baseline

Reported per operator: output error E_loc (what matters) AND weight error
(to show the two orderings genuinely differ -- README Sec 8.3.5's point).
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
    truncated_svd,
    weight_relative_error,
)
from nnopt.cur.svd_cur import (
    analyze_spectrum,
    build_cur,
    select_cur_columns,
    select_cur_columns_by_leverage,
    select_cur_rows,
)
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group

N_CALIB = 12
# Captured rows are split in half: methods FIT on x_fit, and E_loc is
# measured on x_eval, which no method has seen.
#
# This is not optional bookkeeping. activation_aware_svd solves a least
# squares fit against the calibration activations, so scoring it on those
# same rows measures training error: a first run showed aa-svd E_loc =
# 0.0000 at 2x on several operators -- it had simply interpolated the 512
# calibration rows. Functional CUR also consumes calibration (grouping +
# column priority), so it needs the same discipline; plain SVD and
# leverage CUR use only W and are unaffected either way.
MAX_ROWS = 1024
FIT_FRACTION = 0.5
SAMPLED_LAYERS = [0, 6, 12, 18, 23]
# Operating points. 3.81x is NOT arbitrary: it is the cache-anchored target
# the cascade actually derives for this machine + model (README Sec 2.2) --
# one decoder layer's weights (64 MiB fp32) against the guaranteed globally
# shared cache alpha*L3 = 0.7 * 24 MiB = 16.8 MiB. See
# experiments/cache_anchored_targets.py. The 2x/8x points bracket it so the
# trend either side of the real target is visible.
CACHE_ANCHORED = 3.81
COMPRESSIONS = [2.0, CACHE_ANCHORED, 8.0]
TAU, EPS_THR = 0.9, 0.2
OUT_JSON = "experiments/results_svd_vs_cur.json"
FREE_DIMS = {"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500}
# fc2 (n=4096) makes greedy_group ~4 min/op; excluded to keep the sweep
# (5 layers x 3 compressions) tractable. fc1 covers the FFN family.
SKIP_KINDS = ("/fc2/",)


def build_cur_variants(w, x_fit, rank):
    """Both CUR flavours share the compensated matrix and the SVD row
    choice; only COLUMN selection differs -- that is exactly the
    dissertation's claimed contribution. Fitting uses x_fit only."""
    x_calib = x_fit
    y_ref = x_calib @ w.T
    grouping = greedy_group(
        x_calib.T, np.linalg.norm(w, axis=0), float(np.linalg.norm(y_ref)),
        tau=TAU, eps_threshold=EPS_THR,
    )
    w_tilde = build_compensated_weight(w, grouping)
    reps = grouping.representative_indices()
    h_norms = np.linalg.norm(x_calib, axis=0)
    prio = {g.representative: float(g.size) * float(h_norms[g.representative]) for g in grouping.groups}
    spec = analyze_spectrum(w_tilde)
    r_idx = select_cur_rows(spec, rank=rank, r=rank)

    c_func = select_cur_columns(reps, prio, c=min(rank, len(reps)))
    c_lev = select_cur_columns_by_leverage(spec, rank=rank, c=rank)
    return (
        build_cur(w_tilde, c_func, r_idx).reconstruct(),
        build_cur(w_tilde, c_lev, r_idx).reconstruct(),
        len(grouping.groups),
    )


def main():
    profs = weighted_matmul_profiles(DECODER_PATH, FREE_DIMS)
    ops = [
        p for p in profs
        if any(f"/layers.{li}/" in p.name for li in SAMPLED_LAYERS)
        and not any(k in p.name for k in SKIP_KINDS)
    ]
    print(f"{len(ops)} operators across layers {SAMPLED_LAYERS}")

    tensors = sorted({p.activation_input for p in ops if p.activation_input != "encoder_hidden_states"})
    print("capturing calibration activations...")
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
            print(f"  SKIP {p.name} (shape mismatch)")
            continue
        m, n = w.shape
        split = int(x.shape[0] * FIT_FRACTION)
        x_fit, x_eval = x[:split], x[split:]
        for comp in COMPRESSIONS:
            budget = m * n / comp
            r_svd = rank_for_param_budget_svd(m, n, budget)
            r_cur = rank_for_param_budget_cur(m, n, budget)
            t0 = time.time()

            # every method FITS on x_fit only ...
            w_svd = truncated_svd(w, r_svd)
            w_aa = activation_aware_svd(w, x_fit, r_svd)
            w_cf, w_cl, n_groups = build_cur_variants(w, x_fit, r_cur)

            row = {
                "name": p.name, "m": m, "n": n, "compression": comp,
                "rank_svd": r_svd, "rank_cur": r_cur, "n_groups": n_groups,
                "n_fit": int(x_fit.shape[0]), "n_eval": int(x_eval.shape[0]),
                # ... and is SCORED on x_eval, which none of them has seen.
                "eloc": {
                    "svd_plain": output_relative_error(w, w_svd, x_eval),
                    "svd_actaw": output_relative_error(w, w_aa, x_eval),
                    "cur_func": output_relative_error(w, w_cf, x_eval),
                    "cur_lev": output_relative_error(w, w_cl, x_eval),
                },
                # training error, kept only to expose overfitting gaps
                "eloc_fit": {
                    "svd_actaw": output_relative_error(w, w_aa, x_fit),
                    "cur_func": output_relative_error(w, w_cf, x_fit),
                },
                "werr": {
                    "svd_plain": weight_relative_error(w, w_svd),
                    "svd_actaw": weight_relative_error(w, w_aa),
                    "cur_func": weight_relative_error(w, w_cf),
                    "cur_lev": weight_relative_error(w, w_cl),
                },
                "secs": time.time() - t0,
            }
            results.append(row)
            e = row["eloc"]
            print(
                f"[{k}/{len(ops)}] {p.name.replace('/model/decoder/','')[:36]:36s} {comp:.2f}x "
                f"svd={e['svd_plain']:.4f} aa-svd={e['svd_actaw']:.4f}"
                f"(fit {row['eloc_fit']['svd_actaw']:.4f}) "
                f"cur={e['cur_func']:.4f} lev={e['cur_lev']:.4f} [{row['secs']:.0f}s]",
                flush=True,
            )

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")
    summarize(results)


def summarize(results):
    if not results:
        return
    keys = ["svd_plain", "svd_actaw", "cur_func", "cur_lev"]
    print("\n" + "=" * 88)
    print("SVD vs CUR -- teng PARAMETR byudjetida, chiqish xatosi (E_loc)")
    print("=" * 88)
    def _tag(c):
        return f"{c:.2f}x*" if abs(c - CACHE_ANCHORED) < 1e-6 else f"{c:.2f}x "

    print(f"{'siqish':>8s} {'#':>4s} " + " ".join(f"{k:>12s}" for k in keys))
    for comp in sorted({r["compression"] for r in results}):
        rs = [r for r in results if r["compression"] == comp]
        print(f"{_tag(comp):>8s} {len(rs):4d} " + " ".join(f"{np.mean([r['eloc'][k] for r in rs]):12.4f}" for k in keys))
    print("  (* = kesh-bog'langan maqsad: bitta decoder qatlami alpha*L3 ga sig'ishi uchun)")

    print("\nVazn xatosi (Frobenius) -- taqqoslash uchun:")
    print(f"{'siqish':>8s} {'#':>4s} " + " ".join(f"{k:>12s}" for k in keys))
    for comp in sorted({r["compression"] for r in results}):
        rs = [r for r in results if r["compression"] == comp]
        print(f"{_tag(comp):>8s} {len(rs):4d} " + " ".join(f"{np.mean([r['werr'][k] for r in rs]):12.4f}" for k in keys))

    print("\nOverfitting bo'shlig'i (fit -> held-out E_loc):")
    print(f"{'siqish':>8s} {'aa-svd fit':>12s} {'aa-svd eval':>12s} {'cur fit':>10s} {'cur eval':>10s}")
    for comp in sorted({r["compression"] for r in results}):
        rs = [r for r in results if r["compression"] == comp]
        print(
            f"{_tag(comp):>8s} "
            f"{np.mean([r['eloc_fit']['svd_actaw'] for r in rs]):12.4f} "
            f"{np.mean([r['eloc']['svd_actaw'] for r in rs]):12.4f} "
            f"{np.mean([r['eloc_fit']['cur_func'] for r in rs]):10.4f} "
            f"{np.mean([r['eloc']['cur_func'] for r in rs]):10.4f}"
        )

    print("\nG'alabalar (E_loc bo'yicha, operator-darajasida):")
    for a, b in [("cur_func", "cur_lev"), ("cur_func", "svd_plain"), ("cur_func", "svd_actaw"),
                 ("svd_actaw", "svd_plain")]:
        wins = sum(1 for r in results if r["eloc"][a] < r["eloc"][b])
        print(f"  {a:10s} > {b:10s} : {wins}/{len(results)}")


if __name__ == "__main__":
    main()
