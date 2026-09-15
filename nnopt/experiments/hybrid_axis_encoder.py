"""One axis per layer, chosen by measurement instead of applied uniformly.

The cascade has always claimed two structural axes -- remove channels where
responses are collinear, reduce rank where the spectrum decays -- but which
axis to use where was never decided by evidence. The L3 = 12 MiB build forced
the channel axis onto every layer, and the cost showed up as tau collapsing
with depth: 0.99 in the shallow layers, 0.30 by layer 19. Below some tau the
grouping is no longer finding collinearity; it is picking which non-collinear
channel to sacrifice.

axis_choice_per_layer.py measured the alternative at equal parameter count on
held-out activations, and the crossover is sharp:

    L0-L5    channels win, by 1.6x to 400x   (tau still 0.95-0.99)
    L6-L23   rank wins, by 1.0x to 1.7x      (tau 0.90 down to 0.30)

Rank won every one of the nine layers where the criterion had run out. So this
builds the encoder that follows that measurement: channels through L5, rank
from L6, at the same 45% parameter budget either way. The row/rank ratio at
this rank is 18.2, inside the 10-20 band Sec 4.6 requires, so the rank arm is
not being flattered by a memorised calibration set.

Quantization is the same GPTQ pass as every other arm, applied AFTER the
rewrite so its Hessians are built from the operators that will actually run.
"""

import gc
import os
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper

from calib_utils import (
    ENCODER_PATH,
    capture_activations,
    encoder_feeds,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import layer_of
from l3_12_cascade import load_maps, map_path
from nnopt.cur.lowrank_baselines import activation_aware_svd

OUT_DIR = "models/_hybrid"
# "axis" is in the name deliberately: models/_hybrid already holds
# enc_hybrid_tau*.onnx from the COMPENSATION hybrid (weight plus bias), which
# is an unrelated experiment. Two different things called hybrid in one
# directory is how the tau-less baseline filenames silently collided before.
FP32_PATH = f"{OUT_DIR}/enc_axis_hybrid_fp32.onnx"
OUT_PATH = f"{OUT_DIR}/enc_axis_hybrid_gptq.onnx"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 6
FIT_ROWS = 8192
CUTOVER = 6            # first layer that uses the rank axis
D_MODEL, D_FF = 1024, 4096
KEEP_CHANNELS = 2253   # the 45% budget, matched between the two axes


def matched_rank(keep_channels=KEEP_CHANNELS):
    """Rank costing the same parameters as keeping `keep_channels` columns.

    An operator reduced by channels costs d_model * k; factored at rank r it
    costs r * (d_model + d_ff). Floor rather than round, so the hybrid never
    spends more than the arm it is compared against.
    """
    return int(D_MODEL * keep_channels / (D_MODEL + D_FF))


def build_fp32(rank):
    """Encoder with channels below the cutover and rank at or above it."""
    model = onnx.load(ENCODER_PATH)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    ffn = [p for p in profs if "/fc1/" in p.name or "/fc2/" in p.name]

    chan_layers = sorted({layer_of(p.name) for p in ffn
                          if 0 <= layer_of(p.name) < CUTOVER})
    rank_ops = [p for p in ffn if layer_of(p.name) >= CUTOVER]
    print(f"kanal o'qi: L0-L{CUTOVER-1} ({len(chan_layers)} qatlam), "
          f"rank o'qi: L{CUTOVER}+ ({len(rank_ops)} operator, rank {rank})\n")

    # --- channel axis: reuse the maps already built at this budget ---
    pm = load_maps(chan_layers)
    for li, d in pm.items():
        inits[d["w1_init"]].CopyFrom(
            numpy_helper.from_array(d["w1"].astype(np.float32), d["w1_init"]))
        inits[d["w2_init"]].CopyFrom(
            numpy_helper.from_array(d["w2"].astype(np.float32), d["w2_init"]))
        if d["bias_name"] != "None" and d["bias"] is not None:
            inits[d["bias_name"]].CopyFrom(
                numpy_helper.from_array(d["bias"].astype(np.float32),
                                        d["bias_name"]))
        print(f"  L{li:<2d} kanal: {len(d['keep'])} saqlandi")

    # --- rank axis: factor each operator on its own activations ---
    feeds = encoder_feeds(0, N_CALIB)
    replacement, dead = {}, set()
    t0 = time.time()
    for k, p in enumerate(rank_ops, 1):
        x = capture_activations(ENCODER_PATH, [p.activation_input], feeds,
                                max_rows=FIT_ROWS)[p.activation_input]
        w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        if w.shape[1] != x.shape[1]:
            print(f"  {p.name}: shakl mos kelmadi, o'tkazib yuborildi")
            continue
        r = min(rank, min(w.shape))
        lr = activation_aware_svd(w, x, r)
        u, s, vt = np.linalg.svd(lr, full_matrices=False)
        rr = max(1, min(r, int(np.sum(s > 0))))
        sq = np.sqrt(s[:rr])
        a, b = (u[:, :rr] * sq), (sq[:, None] * vt[:rr, :])

        dead.add(p.weight_initializer)
        base = p.name.replace("/", "_").replace(".", "_")
        g.initializer.append(numpy_helper.from_array(b.T.astype(np.float32), f"{base}_B"))
        g.initializer.append(numpy_helper.from_array(a.T.astype(np.float32), f"{base}_A"))
        replacement[p.output_name] = [
            helper.make_node("MatMul", [p.activation_input, f"{base}_B"],
                             [f"{base}_h"], name=f"{base}_lr1"),
            helper.make_node("MatMul", [f"{base}_h", f"{base}_A"],
                             [p.output_name], name=f"{base}_lr2"),
        ]
        if k % 6 == 0:
            print(f"  rank {k}/{len(rank_ops)} [{time.time()-t0:.0f}s]", flush=True)
        del x, w, lr, u, s, vt
        gc.collect()

    rebuilt = []
    for nd in g.node:
        hit = next((o for o in nd.output if o in replacement), None)
        rebuilt.extend(replacement[hit] if hit else [nd])
    del g.node[:]
    g.node.extend(rebuilt)
    kept = [i for i in g.initializer if i.name not in dead]
    del g.initializer[:]
    g.initializer.extend(kept)

    os.makedirs(OUT_DIR, exist_ok=True)
    onnx.save(model, FP32_PATH)
    print(f"\nFP32 gibrid saqlandi: {FP32_PATH} "
          f"{os.path.getsize(FP32_PATH)/1024**2:.0f} MiB")


def main():
    if os.path.exists(OUT_PATH):
        print(f"mavjud: {OUT_PATH}  "
              f"{os.path.getsize(OUT_PATH)/1024**2:.0f} MiB")
        return
    missing = [li for li in range(CUTOVER) if not os.path.exists(map_path(li))]
    if missing:
        raise SystemExit(f"kanal xaritalari yo'q: {missing}; "
                         f"avval l3_12_cascade.py")

    rank = matched_rank()
    if not os.path.exists(FP32_PATH):
        build_fp32(rank)

    from gptq_plus_pruning import build_gptq_model
    print("\nGPTQ bilan kvantlanmoqda (qayta yozilgan graf ustida)...", flush=True)
    build_gptq_model(f"{OUT_DIR}/_tmp.onnx", OUT_PATH, None, "gibrid",
                     src_path=FP32_PATH)
    print(f"  saqlandi: {OUT_PATH}  "
          f"{os.path.getsize(OUT_PATH)/1024**2:.0f} MiB")


if __name__ == "__main__":
    main()
