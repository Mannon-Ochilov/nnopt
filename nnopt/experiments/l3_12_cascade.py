"""Build the model a 12 MiB L3 would actually demand, and measure what it costs.

The sweep says the derivation responds to the hardware; this asks the harder
question -- when it responds, is the answer any good? On L3 = 12 MiB the
encoder layer needs 5.71x, INT8 supplies 4.00x, and the residual 1.43x has to
come out of the FFN alone, which works out to 45% of the FFN channels in
EVERY layer (l3_12_feasibility.py).

That last word is what makes this more than a rerun. Our criterion's budget
is not something we set; it falls out of tau, and the measured redundancy is
strongly depth-dependent -- at tau = 0.99 layer 0 gives up 43% of its channels
while layers 14-23 give up essentially none. A per-LAYER cache target does not
care: the constraint binds on every layer equally. So a 12 MiB machine asks
the late layers for 45% of channels that, by our own criterion, are not
redundant at any threshold worth the name.

The build here is therefore budget-driven rather than tau-driven: each layer
descends the tau grid until it reaches the required removal, and the tau it
had to fall to is recorded. Where that tau collapses, the criterion is no
longer selecting redundant channels -- it is merely ranking which
non-redundant channel to sacrifice, and the WER is the price of that.

Arms, all quantized identically (GPTQ encoder, ORT INT8 decoder) so the only
thing that varies is the structural demand the cache target creates:

  L3=24 (kaskad)      tau = 0.99, 17% mean removal          [already measured]
  L3=12 enkoder       45% every layer, tau descended per layer
  L3=12 dekoder       FFN low-rank at the derived rank
  L3=12 to'liq        both together -- the whole cascade at 12 MiB
"""

import gc
import json
import os
import time

import numpy as np
import onnx
from onnx import numpy_helper

from calib_utils import (
    ENCODER_PATH,
    CalibSet,
    capture_activations,
    feeds_for,
    weighted_matmul_profiles,
)
from ffn_prune_endtoend import bias_name_for, layer_of
from nnopt.grouping.functional_grouping import (
    build_compensated_weight,
    greedy_group,
    trim_to_budget,
)

OUT_DIR = "models/_l3_12"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 6
FIT_ROWS = 3072
N_CHANNELS = 4096
TARGET_REMOVAL = 0.45          # derived in l3_12_feasibility.py for L3 = 12 MiB
DEFAULT_CALIB = CalibSet(split="validation", skip=0, n=N_CALIB)
EPS_THRESHOLD = 0.5
# Coarse on purpose: trim_to_budget corrects any overshoot exactly, so the
# grid only has to bracket the crossing point, and every extra step costs a
# full grouping pass (~90 s on a 4096-channel layer).
TAU_GRID = (0.99, 0.95, 0.90, 0.80, 0.60, 0.30, -1.0)
# Continuous search instead of the grid. tau is a real number, and the grid
# was an economy rather than a principle: before the grouping pass was fixed
# it cost minutes, so every extra evaluation mattered. At 13 s a pass a
# bisection to 1e-3 is affordable, and it removes two artefacts at once --
# the arbitrary rungs, and the trim step that had to repair the overshoot the
# rungs caused. Set BISECT=0 to fall back to the grid.
BISECT = os.environ.get("BISECT", "1") != "0"
TAU_TOL = 1e-3
TAU_LO, TAU_HI = -1.0, 0.999
STATS_JSON = "experiments/results_l3_12_maps.json"


def map_path(li, removal=TARGET_REMOVAL, out_dir=OUT_DIR, calib=None,
             bisect=None):
    # Budget, calibration set AND search method are all part of the filename,
    # because a map is only valid for the combination it was built from. The
    # grid and the bisection land on different taus and therefore different
    # channel sets at the same budget, so a name that cannot tell them apart
    # would hand back the wrong artifact. An earlier version of the baseline
    # pipeline left the analogous key out and silently reused one budget's
    # maps for another; the artifacts came out identically sized when they
    # should have differed, which is how it was caught.
    tag = (calib or DEFAULT_CALIB).tag
    method = "bisect" if (BISECT if bisect is None else bisect) else "grid"
    return f"{out_dir}/prune_r{removal:.2f}_{tag}_{method}_L{li}.npz"


def build_maps(removal=TARGET_REMOVAL, out_dir=OUT_DIR, calib=None):
    """Per-layer keep sets meeting a removal budget, with the tau each needed."""
    calib = calib or DEFAULT_CALIB
    os.makedirs(out_dir, exist_ok=True)
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}
    layers = sorted(li for li in fc2 if li in fc1)

    need_keep = int(round(N_CHANNELS * (1.0 - removal)))
    print(f"{len(layers)} qatlam, har birida {N_CHANNELS} -> {need_keep} kanal "
          f"({removal*100:.0f}% olib tashlash) talab qilinadi\n")

    stats = []
    feeds = feeds_for(calib)
    t0 = time.time()
    for li in layers:
        if os.path.exists(map_path(li, removal, out_dir, calib)):
            z = np.load(map_path(li, removal, out_dir, calib), allow_pickle=True)
            print(f"  L{li:<2d} keshdan: {len(z['keep'])} kanal, "
                  f"tau={float(z['tau']):.2f}")
            stats.append({"layer": li, "kept": int(len(z["keep"])),
                          "tau": float(z["tau"]), "cached": True})
            continue

        p1, p2 = fc1[li], fc2[li]
        x_by = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                   max_rows=FIT_ROWS)
        x = x_by[p2.activation_input][:FIT_ROWS]
        w2s = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
        w1 = numpy_helper.to_array(inits[p1.weight_initializer]).astype(np.float64)
        if w1.shape[1] != w2.shape[1]:
            w1 = w1.T
        bname = bias_name_for(model, p1)
        bias = (numpy_helper.to_array(inits[bname]).astype(np.float64)
                if bname else None)

        h = x.T
        wn = np.linalg.norm(w2, axis=0)
        y_norm = float(np.linalg.norm(x @ w2.T))

        # Highest tau that meets the budget. Removal is monotone in tau, so
        # either search works; the difference is what they can land on.
        def group_at(tau_i):
            eps_thr = EPS_THRESHOLD if tau_i >= 0.0 else float("inf")
            gg = greedy_group(h, wn, y_norm, tau=tau_i, eps_threshold=eps_thr)
            return gg, len(gg.groups), tau_i

        chosen, best = None, None
        if BISECT:
            # Bisect on tau itself. The grid could only stop at rungs someone
            # chose in advance, so it routinely overshot the budget and needed
            # trimming to repair -- layer 7 landed at 57.5% removal where 45%
            # was asked for. Searching the real interval lands on a tau whose
            # OWN grouping meets the budget, which keeps every merge above the
            # threshold that justified it.
            lo, hi = TAU_LO, TAU_HI
            g_hi, k_hi, _ = group_at(hi)
            if k_hi <= need_keep:
                chosen = (g_hi, k_hi, hi)       # gentlest tau already suffices
            else:
                best = (g_hi, k_hi, hi)
                while hi - lo > TAU_TOL:
                    mid = 0.5 * (lo + hi)
                    gg, kept_i, tau_i = group_at(mid)
                    if kept_i <= need_keep:
                        chosen = (gg, kept_i, tau_i)
                        lo = mid                # a stricter tau may also do
                    else:
                        hi = mid
                    if best is None or kept_i < best[1]:
                        best = (gg, kept_i, tau_i)
        else:
            lo, hi = 0, len(TAU_GRID) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                gg, kept_i, tau_i = group_at(TAU_GRID[mid])
                if best is None or kept_i < best[1]:
                    best = (gg, kept_i, tau_i)
                if kept_i <= need_keep:
                    chosen = (gg, kept_i, tau_i)
                    hi = mid - 1
                else:
                    lo = mid + 1

        if chosen is None:
            chosen = best
            print(f"  L{li:<2d} BYUDJETGA YETMADI: eng yaxshisi {chosen[1]} "
                  f"kanal (tau={chosen[2]:.3f})", flush=True)
        g, kept, tau = chosen
        if kept < need_keep:
            released = trim_to_budget(g, need_keep)
            kept = len(g.groups)
            # Removal is a step function of tau, so even an exact tau lands
            # just past the budget; the release is what puts it exactly on it.
            print(f"  L{li:<2d} byudjetdan oshib ketdi, {released} kanal "
                  f"qaytarildi -> {kept}", flush=True)

        w2_comp = build_compensated_weight(w2, g)
        keep = np.array(sorted(gr.representative for gr in g.groups))
        np.savez_compressed(
            map_path(li, removal, out_dir, calib), keep=keep,
            w1=w1[:, keep], w2=w2_comp[:, keep].T,
            bias=bias[keep] if bias is not None else np.array([]),
            bias_name=str(bname), w1_init=p1.weight_initializer,
            w2_init=p2.weight_initializer, tau=float(tau))
        print(f"  L{li:<2d} saqlanadi {kept:5d}/{N_CHANNELS} "
              f"({(1-kept/N_CHANNELS)*100:4.1f}% olib tashlandi) "
              f"tau={tau:6.3f}  [{time.time()-t0:.0f}s]", flush=True)
        stats.append({"layer": li, "kept": int(kept), "tau": float(tau),
                      "cached": False})
        del x_by, x, w1, w2, w2_comp, g, h
        gc.collect()

    json.dump(stats, open(STATS_JSON, "w"), indent=2)
    return stats


def load_maps(layers, removal=TARGET_REMOVAL, out_dir=OUT_DIR, calib=None):
    out = {}
    for li in layers:
        z = np.load(map_path(li, removal, out_dir, calib), allow_pickle=True)
        bn = str(z["bias_name"])
        out[li] = {"keep": z["keep"], "w1": z["w1"], "w2": z["w2"],
                   "bias": z["bias"] if bn != "None" else None,
                   "bias_name": bn, "w1_init": str(z["w1_init"]),
                   "w2_init": str(z["w2_init"])}
    return out


def main():
    stats = build_maps()

    taus = [s["tau"] for s in stats]
    kept = [s["kept"] for s in stats]
    mean_rm = np.mean([1 - k / N_CHANNELS for k in kept]) * 100
    met = sum(1 for k in kept if k <= int(round(N_CHANNELS * (1 - TARGET_REMOVAL))))
    print(f"\n{met}/{len(kept)} qatlam byudjetga yetdi, o'rtacha olib tashlash "
          f"{mean_rm:.1f}%")
    print(f"tau diapazoni: {max(taus):.3f} ... {min(taus):.3f}")
    print("Qatlam bo'yicha tau:")
    for s in stats:
        print(f"  L{s['layer']:<2d} tau={s['tau']:6.3f} "
              f"saqlandi {s['kept']:5d} ({(1-s['kept']/N_CHANNELS)*100:4.1f}% olib tashlandi)")

    # The search method is in the artifact name for the same reason it is in
    # the map name: the grid and the bisection produce different models at the
    # same budget, and a shared name would quietly return whichever was built
    # first instead of the one that was asked for.
    method = "bisect" if BISECT else "grid"
    path = f"{OUT_DIR}/enc_l3_12_{method}_gptq.onnx"
    if os.path.exists(path):
        print(f"\n[enkoder] mavjud, {os.path.getsize(path)/1024**2:.0f} MiB")
        return
    from gptq_plus_pruning import build_gptq_model
    pm = load_maps([s["layer"] for s in stats])
    print("\n[enkoder] GPTQ bilan kvantlanmoqda...", flush=True)
    build_gptq_model(f"{OUT_DIR}/_tmp.onnx", path, pm, f"l3-12-{method}")
    print(f"  saqlandi: {path}  {os.path.getsize(path)/1024**2:.0f} MiB")


if __name__ == "__main__":
    main()
