"""Which half of the structural method is doing the work?

At the aggressive budget (tau = 0.95) the three criteria separate sharply:
weight-only magnitude selection destroys the model (WER 2.7378) while both
activation-aware routes survive -- ours at 0.2006 and Wanda at 0.2202, a gap
too small to resolve at n = 300.

But "ours" bundles two independent ideas:

  1. WHICH channels go   -- those collinear with a channel that stays
  2. WHAT happens then   -- the removed channel is folded into its
                            representative, W2[:, p] += gamma_j * W2[:, j]

Wanda has neither, yet survives, which already suggests compensation is not
what saves the model. That is an inference from a different method, not a
measurement of ours, so this isolates it directly: the SAME channels our
grouping selected, with the compensation step simply switched off.

  ours              grouping + compensation      (measured: 0.2006)
  ours-no-comp      grouping, raw slice          (this script)

If the two agree, the contribution is the selection criterion and the
compensation machinery -- along with the per-channel quantization it forces
(Sec 4.4) -- can be dropped. If they diverge, compensation is load-bearing
and its coupling constraint is the price of a real benefit.
"""

import gc
import glob
import os

import numpy as np
import onnx
from onnx import numpy_helper

from calib_utils import ENCODER_PATH, weighted_matmul_profiles
from ffn_prune_endtoend import layer_of
from structural_baselines import bias_name_for

TAU = 0.95
PRUNE_DIR = "models/_prune"
OUT_DIR = "models/_ratio_sweep"
OUT_PATH = f"{OUT_DIR}/enc_nocomp_tau{TAU}_gptq.onnx"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}


def uncompensated_map():
    """Our keep sets, but with W2 taken straight from the original graph.

    Compensation only ever modified W2 (it folds a removed channel's output
    contribution into its representative), so W1 and the bias are sliced
    identically in both arms and the two models differ in exactly one step.
    """
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}

    out = {}
    for f in sorted(glob.glob(f"{PRUNE_DIR}/prune_L*_tau{TAU}.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        keep = np.load(f, allow_pickle=True)["keep"]
        if len(keep) == 4096 or li not in fc1 or li not in fc2:
            continue
        p1, p2 = fc1[li], fc2[li]

        w2 = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w1 = numpy_helper.to_array(inits[p1.weight_initializer]).astype(np.float64)
        # Orient both as (d_model, n_intermediate) so channels index columns.
        if w2.shape[0] > w2.shape[1]:
            w2 = w2.T
        if w1.shape[0] > w1.shape[1]:
            w1 = w1.T
        bname = bias_name_for(model, p1)
        bias = (numpy_helper.to_array(inits[bname]).astype(np.float64)
                if bname else None)

        out[li] = {"keep": keep,
                   "w1": w1[:, keep],
                   "bias": None if bias is None else bias[keep],
                   "w2": w2[:, keep].T,          # raw slice, no compensation
                   "bias_name": bname,
                   "w1_init": p1.weight_initializer,
                   "w2_init": p2.weight_initializer}
        print(f"  L{li:<2d} saqlanadi {len(keep):5d}/4096  (kompensatsiyasiz)",
              flush=True)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_PATH):
        print(f"mavjud: {OUT_PATH}  {os.path.getsize(OUT_PATH)/1024**2:.0f} MiB")
        return

    pm = uncompensated_map()
    print(f"\n{len(pm)} qatlam, bizning tanlov, kompensatsiya O'CHIRILGAN\n")

    from gptq_plus_pruning import build_gptq_model
    build_gptq_model(f"{OUT_DIR}/_tmp_nocomp.onnx", OUT_PATH, pm, "nocomp")
    del pm
    gc.collect()
    print(f"\nsaqlandi: {OUT_PATH}  {os.path.getsize(OUT_PATH)/1024**2:.0f} MiB")
    print("Taqqoslash: bizniki (kompensatsiya bilan) 0.2006, "
          "wanda 0.2202, magnitude 2.7378")


if __name__ == "__main__":
    main()
