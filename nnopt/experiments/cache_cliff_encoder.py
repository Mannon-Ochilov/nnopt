"""The cascade's actual claim, tested where it can actually be tested.

Author's cascade logic (2026-08-12):
    1. FP32 fits alpha*cache          -> do nothing
    2. INT8 (mandatory) makes it fit  -> low-rank NOT considered
    3. still does not fit after INT8  -> low-rank + INT8, and the payoff
                                         must appear as real speed, via
                                         fewer cache misses
Low-rank may be CUR or SVD -- whichever helps is acceptable.

Why the encoder, and why the WEIGHT footprint:

  find_cur_regime.py showed the decoder never reaches case 3 (0/240), while
  encoder fc1/fc2 exceed the L3 budget 1.98x. But that used M_eff = M_total
  (the profiler's explicit UPPER bound), where the excess comes from
  activations (M_X+M_Y = 29.3 MiB), which low-rank cannot shrink.

  The physically meaningful quantity for a 1500-position encoder pass is
  different: each weight element is reused 1500 times, so the WEIGHT wants
  to stay resident while activations stream past. That is the quantity
  low-rank does reduce, and it produces a testable prediction on this
  machine:

      FP32 weight            16.0 MiB  -> L3 only
      INT8 weight             4.0 MiB  -> L3 only  (L2 is 1.25 MiB)
      INT8 + low-rank r~200   1.0 MiB  -> fits L2

  So low-rank should move fc1 from L3-resident to L2-resident. If the
  cascade's premise is right, measured speedup at that point must EXCEED
  what the FLOP reduction alone predicts -- the surplus is the cache
  effect. That surplus is the whole claim, and it is what this measures.

Honest limits: this machine exposes no PMC/cache-miss counters, so cache
behaviour is inferred from measured latency versus a FLOP-only prediction,
not observed directly. Stated plainly rather than dressed up.
"""

import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from calib_utils import ENCODER_PATH, capture_activations, encoder_feeds
from nnopt.bench.latency import make_session, measure_latency
from nnopt.cur.lowrank_baselines import activation_aware_svd, output_relative_error, truncated_svd
from nnopt.cur.svd_cur import analyze_spectrum, build_cur, select_cur_columns, select_cur_rows
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.hw.cache_topology import detect_cache_topology

OUT_DIR = "models/_cliff"
N_CALIB = 6
MAX_ROWS = 1024
FIT_FRACTION = 0.5
SEQ = 1500          # encoder positions per pass -- the reuse factor
WARMUP, MEASURED = 5, 25
TAU, EPS_THR = 0.9, 0.2
RANKS = [819, 512, 409, 300, 200, 128, 80]
OUT_JSON = "experiments/results_cache_cliff.json"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": SEQ}
TARGET_SUBSTR = "/layers.0/fc1/"


def build_dense_onnx(w_t, path, n_in):
    """Y = X @ w_t, w_t shape (n_in, m_out)."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["pos", n_in])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["pos", w_t.shape[1]])
    init = numpy_helper.from_array(w_t.astype(np.float32), "W")
    node = helper.make_node("MatMul", ["x", "W"], ["y"], name="mm")
    g = helper.make_graph([node], "dense", [x], [y], initializer=[init])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def build_lowrank_onnx(a_t, b_t, path, n_in):
    """Y = (X @ a_t) @ b_t -- two matmuls, no fused middle factor."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["pos", n_in])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["pos", b_t.shape[1]])
    i1 = numpy_helper.from_array(a_t.astype(np.float32), "A")
    i2 = numpy_helper.from_array(b_t.astype(np.float32), "B")
    n1 = helper.make_node("MatMul", ["x", "A"], ["h"], name="mm1")
    n2 = helper.make_node("MatMul", ["h", "B"], ["y"], name="mm2")
    g = helper.make_graph([n1, n2], "lowrank", [x], [y], initializer=[i1, i2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def bench(path, n_in, seq=SEQ, seed=0):
    sess = make_session(path, intra_op_threads=1)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((seq, n_in)).astype(np.float32)
    r = measure_latency(sess, name=path, fixed_feed={"x": x},
                        warmup_runs=WARMUP, measured_runs=MEASURED)
    del sess
    return r.median_ms


def q_int8(path_in, path_out):
    quantize_dynamic(path_in, path_out, weight_type=QuantType.QInt8)


def cur_factors(w, x_fit, rank):
    y_ref = x_fit @ w.T
    grouping = greedy_group(x_fit.T, np.linalg.norm(w, axis=0),
                            float(np.linalg.norm(y_ref)), tau=TAU, eps_threshold=EPS_THR)
    w_t = build_compensated_weight(w, grouping)
    reps = grouping.representative_indices()
    hn = np.linalg.norm(x_fit, axis=0)
    prio = {g.representative: float(g.size) * float(hn[g.representative]) for g in grouping.groups}
    spec = analyze_spectrum(w_t)
    r_idx = select_cur_rows(spec, rank=rank, r=rank)
    c_idx = select_cur_columns(reps, prio, c=min(rank, len(reps)))
    cur = build_cur(w_t, c_idx, r_idx)
    return cur.C, cur.U @ cur.R          # (m,c) and (c,n)


def svd_factors(w, x_fit, rank, aware=True):
    lr = activation_aware_svd(w, x_fit, rank) if aware else truncated_svd(w, rank)
    u, s, vt = np.linalg.svd(lr, full_matrices=False)
    r = min(rank, len(s))
    sq = np.sqrt(s[:r])
    return (u[:, :r] * sq), (sq[:, None] * vt[:r, :])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    topo = detect_cache_topology()
    l3 = topo.global_shared_cache()
    l2 = topo.by_level(2)[0]
    print(f"L2 = {l2.size_bytes/1024**2:.2f} MiB (juft yadro), L3 = {l3.size_bytes/1024**2:.0f} MiB (16 yadro)")

    from nnopt.profiler.graph_profiler import profile_onnx_model
    profs = profile_onnx_model(ENCODER_PATH, free_dims=ENC_DIMS)
    target = next(p for p in profs if TARGET_SUBSTR in p.name and p.weight_initializer)
    print(f"target: {target.name}")

    feeds = encoder_feeds(0, N_CALIB)
    x_by = capture_activations(ENCODER_PATH, [target.activation_input], feeds, max_rows=MAX_ROWS)
    x = x_by[target.activation_input]

    enc = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in enc.graph.initializer}
    w_stored = numpy_helper.to_array(inits[target.weight_initializer]).astype(np.float64)
    w = w_stored if w_stored.shape[1] == x.shape[1] else w_stored.T
    m, n = w.shape
    split = int(x.shape[0] * FIT_FRACTION)
    x_fit, x_eval = x[:split], x[split:]
    print(f"W = {m} x {n}  ({m*n*4/1024**2:.1f} MiB fp32, {m*n/1024**2:.1f} MiB int8)")
    print(f"kalibrlash: fit={x_fit.shape[0]} eval={x_eval.shape[0]} qator\n")

    rows = []

    # ---------- reference: dense FP32 and dense INT8 ----------
    p_fp32 = f"{OUT_DIR}/dense_fp32.onnx"
    build_dense_onnx(w.T, p_fp32, n)
    t_fp32 = bench(p_fp32, n)
    p_int8 = f"{OUT_DIR}/dense_int8.onnx"
    q_int8(p_fp32, p_int8)
    t_int8 = bench(p_int8, n)
    flops_dense = 2 * SEQ * m * n

    for label, wbytes, t, eloc in [
        ("dense FP32", m * n * 4, t_fp32, 0.0),
        ("dense INT8", m * n * 1, t_int8, None),
    ]:
        rows.append({"variant": label, "rank": None, "weight_bytes": wbytes,
                     "flops": flops_dense, "ms": t, "eloc": eloc})
        print(f"{label:26s} vazn={wbytes/1024**2:6.2f}M  {t:8.3f} ms")

    # dense INT8 accuracy on held-out
    sess = make_session(p_int8, intra_op_threads=1)
    y_q = sess.run(None, {"x": x_eval.astype(np.float32)})[0]
    rows[1]["eloc"] = float(np.linalg.norm(x_eval @ w.T - y_q) / np.linalg.norm(x_eval @ w.T))
    del sess
    print(f"{'':26s} E_loc(int8) = {rows[1]['eloc']:.4f}\n")

    # ---------- low-rank variants ----------
    for method in ("svd_actaw", "cur_func"):
        for rank in RANKS:
            if method == "svd_actaw":
                a, b = svd_factors(w, x_fit, rank)
            else:
                a, b = cur_factors(w, x_fit, rank)
            approx = a @ b
            eloc = output_relative_error(w, approx, x_eval)

            # graph orientation: Y = X @ (a@b).T = X @ b.T @ a.T
            p_fp = f"{OUT_DIR}/{method}_r{rank}_fp32.onnx"
            build_lowrank_onnx(b.T, a.T, p_fp, n)
            p_q = f"{OUT_DIR}/{method}_r{rank}_int8.onnx"
            q_int8(p_fp, p_q)
            t = bench(p_q, n)

            wbytes = a.size + b.size          # int8, 1 byte each
            flops = 2 * SEQ * (n * rank + rank * m)
            rows.append({"variant": f"{method} r={rank}", "rank": rank,
                         "weight_bytes": wbytes, "flops": flops, "ms": t, "eloc": eloc})
            fits = "L2" if wbytes <= 0.7 * l2.size_bytes else ("L3" if wbytes <= 0.7 * l3.size_bytes else "-")
            print(f"{method+' r='+str(rank):26s} vazn={wbytes/1024**2:6.2f}M  {t:8.3f} ms  "
                  f"E_loc={eloc:.4f}  sig'adi:{fits}")

    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    summarize(rows, t_int8, flops_dense, l2, l3)


def summarize(rows, t_int8, flops_dense, l2, l3):
    print("\n" + "=" * 104)
    print("KESH CHEGARASI: past-rank INT8 ustiga qo'shilganda tezlik FLOPs bashoratidan oshadimi?")
    print("=" * 104)
    print(f"{'variant':24s} {'vazn(MiB)':>10s} {'sig''adi':>7s} {'FLOPs nisbat':>13s} "
          f"{'kutilgan x':>11s} {'real x':>9s} {'ORTIQCHA':>10s} {'E_loc':>8s}")
    print("-" * 104)
    for r in rows:
        if r["variant"].startswith("dense FP32"):
            continue
        flop_ratio = flops_dense / r["flops"]
        expected = flop_ratio                       # if time were pure FLOPs
        actual = t_int8 / r["ms"]
        surplus = actual / expected
        wb = r["weight_bytes"]
        fits = "L2" if wb <= 0.7 * l2.size_bytes else ("L3" if wb <= 0.7 * l3.size_bytes else "-")
        el = "-" if r["eloc"] is None else f"{r['eloc']:.4f}"
        print(f"{r['variant']:24s} {wb/1024**2:10.2f} {fits:>7s} {flop_ratio:13.2f} "
              f"{expected:11.2f} {actual:9.2f} {surplus:10.2f} {el:>8s}")
    print("-" * 104)
    print("ORTIQCHA > 1.00 => tezlanish FLOPs kamayishidan KO'PROQ, ya'ni kesh effekti.")
    print("(Bu mashinada PMC/kesh-miss hisoblagichlari yo'q; kesh effekti real vaqt va")
    print(" FLOPs bashorati orasidagi farqdan bilvosita xulosa qilinadi.)")


if __name__ == "__main__":
    main()
