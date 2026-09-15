"""Two removal mechanisms instead of one loosened threshold.

Reaching a smaller budget currently has only one lever: lower tau, which makes
the cosine grouping itself more permissive. That works but it degrades -- WER
runs 0.1833, 0.1916, 0.2006, 0.2179 as tau goes 0.99, 0.97, 0.95, 0.93 --
because a looser cosine admits channels that are not really duplicated.

The comparison against FLAP suggests a different lever. A channel can be
removable for two independent reasons, and each needs its own treatment:

    redundant        duplicated by another channel   -> fold into it (gamma)
    nearly constant  barely varies across inputs     -> absorb mean into bias

Our criterion only sees the first. So instead of loosening it, this keeps the
cosine stage exactly as it is at tau = 0.99 and adds a SECOND pass over the
survivors, removing the ones whose response barely fluctuates:

    score_p = ||W2_comp[:, p]||^2 * Var(h_p)

Both stages' discarded means are then swept into the output bias with the
same aggregate identity used in hybrid_compensation.py, which needs no group
bookkeeping and covers stage 1 and stage 2 at once:

    correction = W2_full @ mean(h) - W2_final @ mean(h)[final_keep]

The per-layer counts are taken from the tau = 0.95 map, so the result lands on
exactly the same 254 MiB budget as the loosened-threshold arm. That makes the
comparison a clean answer to one question: at a fixed budget, is it better to
relax the redundancy criterion, or to keep it strict and remove the rest by a
different mechanism?

This is a composition rather than a new algorithm -- stage 1 is ours, stage 2
is FLAP's mechanism applied to what stage 1 left -- and it is presented that
way. What is new is the organising principle the earlier results forced:
match the treatment to the reason a channel is removable.
"""

import gc
import glob
import os
import time

import numpy as np
import onnx
from onnx import numpy_helper

from calib_utils import (
    ENCODER_PATH,
    capture_activations,
    encoder_feeds,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import layer_of
from flap_baseline import out_bias_name

OUT_DIR = "models/_two_stage"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 12
FIT_ROWS = 4096
STAGE1_TAU = 0.99      # cosine stage stays strict
TARGET_TAU = 0.95      # budget to match (254 MiB)


def keep_counts(tau):
    out = {}
    for f in sorted(glob.glob(f"models/_prune/prune_L*_tau{tau}.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        keep = np.load(f, allow_pickle=True)["keep"]
        out[li] = int(len(keep))
    return out


def build_map():
    """Stage 1 from the cached cosine grouping, stage 2 computed here."""
    target = keep_counts(TARGET_TAU)
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}

    feeds = encoder_feeds(0, N_CALIB)
    out = {}
    t0 = time.time()
    for f in sorted(glob.glob(f"models/_prune/prune_L*_tau{STAGE1_TAU}.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        z = np.load(f, allow_pickle=True)
        keep1 = z["keep"]                       # stage-1 survivors
        want = target.get(li, len(keep1))
        if li not in fc1 or li not in fc2:
            continue

        p1, p2 = fc1[li], fc2[li]
        x_by = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                   max_rows=FIT_ROWS)
        x = x_by[p2.activation_input][:FIT_ROWS]
        mean_h = np.mean(x, axis=0)

        w2s = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w2_full = w2s if w2s.shape[1] == x.shape[1] else w2s.T      # (1024, 4096)
        w2_1 = np.asarray(z["w2"], dtype=np.float64).T               # (1024, keep1)
        w1_1 = np.asarray(z["w1"], dtype=np.float64)                 # (1024, keep1)
        bias1 = z["bias"] if str(z["bias_name"]) != "None" else None

        n_extra = max(0, len(keep1) - want)
        if n_extra > 0:
            # Stage 2: among survivors, the ones that barely vary. The weight
            # column is the COMPENSATED one, since that is what multiplies
            # h_p once stage 1 has folded its group into it.
            score = (np.linalg.norm(w2_1, axis=0) ** 2) * np.var(x[:, keep1], axis=0)
            order = np.argsort(score)            # smallest fluctuation first
            drop_local = np.sort(order[:n_extra])
            sel = np.setdiff1d(np.arange(len(keep1)), drop_local)
        else:
            sel = np.arange(len(keep1))

        keep2 = keep1[sel]
        w2_2 = w2_1[:, sel]
        w1_2 = w1_1[:, sel]

        # One correction covers what BOTH stages discarded on average.
        correction = w2_full @ mean_h - w2_2 @ mean_h[keep2]
        obname = out_bias_name(model, p2)
        obias = (numpy_helper.to_array(inits[obname]).astype(np.float64)
                 if obname else None)

        out[li] = {
            "keep": keep2, "w1": w1_2, "w2": w2_2.T,
            "bias": None if bias1 is None else np.asarray(bias1)[sel],
            "bias_name": str(z["bias_name"]),
            "w1_init": str(z["w1_init"]), "w2_init": str(z["w2_init"]),
            "out_bias_name": obname,
            "out_bias": None if obias is None else obias + correction,
        }
        print(f"  L{li:<2d} 1-bosqich {len(keep1):5d} -> 2-bosqich "
              f"{len(keep2):5d} ({n_extra} ta fluktuatsiya bo'yicha)   "
              f"bias normasi {np.linalg.norm(correction):.4f}   "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del x_by, x, w2_full, w2_1, w1_1
        gc.collect()
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/enc_two_stage_gptq.onnx"
    if os.path.exists(path):
        print(f"mavjud: {path}  {os.path.getsize(path)/1024**2:.0f} MiB")
        return

    print(f"1-bosqich: kosinus guruhlash tau = {STAGE1_TAU} (qat'iy, "
          f"o'zgarmaydi)")
    print(f"2-bosqich: fluktuatsiya bo'yicha, tau = {TARGET_TAU} byudjetigacha "
          f"(254 MiB)\n")
    pm = build_map()

    from flap_baseline import apply_and_quantize
    n = apply_and_quantize(pm, path, "twostage")
    print(f"\n  {n} qatlamda chiqish biasi tuzatildi")
    print(f"  saqlandi: {path}  {os.path.getsize(path)/1024**2:.0f} MiB")
    print("\nTaqqoslash (254 MiB): tau=0.95 yolg'iz 0.2006, gibrid 0.1967, "
          "FLAP 0.1925.")


if __name__ == "__main__":
    main()
