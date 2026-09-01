"""Our compensation plus FLAP's: they cover different parts of the error.

Comparing against FLAP surfaced a gap in our own method rather than a
weakness of it. Our compensation projects a removed channel onto its
representative,

    h_j ~= gamma_j * h_p ,     gamma_j = <h_j, h_p> / ||h_p||^2

which captures the part of h_j that co-varies with what stays. By
construction the residual r_j = h_j - gamma_j h_p is orthogonal to h_p -- but
orthogonal does not mean zero-mean, and whatever constant part r_j carries is
simply dropped. FLAP drops the varying part instead and keeps the mean, in
the output bias. The two are complementary, and the combination generalises
both:

    ours        h_j ~= gamma_j h_p
    FLAP        h_j ~= mu_j
    both        h_j ~= gamma_j h_p + mean(r_j)

so in a least-squares sense the hybrid cannot be worse than either.

The correction needs no group bookkeeping. The block output error left by our
pruning, averaged over the calibration set, is just the difference between
what the full and the pruned operator produce at the mean activation:

    correction = W2_full @ mean(h) - W2_pruned @ mean(h)[keep]

Adding that to the output bias removes exactly the constant component of the
residual error, leaving our existing weights and channel choice untouched --
so this is a strict addition to the method, and the comparison against the
unmodified arm is a clean ablation of the bias term alone.
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
from structural_baselines import bias_name_for

OUT_DIR = "models/_hybrid"
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
N_CALIB = 12
FIT_ROWS = 4096
TAUS = (0.99, 0.95)


def build_map(tau):
    """Our existing prune map, plus the bias correction it leaves on the table."""
    model = onnx.load(ENCODER_PATH)
    inits = {i.name: i for i in model.graph.initializer}
    profs = weighted_matmul_profiles(ENCODER_PATH, ENC_DIMS)
    fc1 = {layer_of(p.name): p for p in profs if "/fc1/" in p.name}
    fc2 = {layer_of(p.name): p for p in profs if "/fc2/" in p.name}

    feeds = encoder_feeds(0, N_CALIB)
    out = {}
    t0 = time.time()
    for f in sorted(glob.glob(f"models/_prune/prune_L*_tau{tau}.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        z = np.load(f, allow_pickle=True)
        keep = z["keep"]
        if len(keep) == 4096 or li not in fc1 or li not in fc2:
            continue
        p1, p2 = fc1[li], fc2[li]

        x_by = capture_activations(ENCODER_PATH, [p2.activation_input], feeds,
                                   max_rows=FIT_ROWS)
        x = x_by[p2.activation_input][:FIT_ROWS]
        mean_h = np.mean(x, axis=0)                       # (4096,)

        w2s = numpy_helper.to_array(inits[p2.weight_initializer]).astype(np.float64)
        w2_full = w2s if w2s.shape[1] == x.shape[1] else w2s.T   # (1024, 4096)
        w2_pruned = np.asarray(z["w2"], dtype=np.float64).T       # (1024, keep)

        # What the full operator produces at the mean activation, minus what
        # the compensated+pruned one produces: the constant part our
        # compensation left behind.
        correction = w2_full @ mean_h - w2_pruned @ mean_h[keep]

        obname = out_bias_name(model, p2)
        obias = (numpy_helper.to_array(inits[obname]).astype(np.float64)
                 if obname else None)
        bname = str(z["bias_name"])
        out[li] = {
            "keep": keep, "w1": z["w1"], "w2": z["w2"],
            "bias": z["bias"] if bname != "None" else None,
            "bias_name": bname,
            "w1_init": str(z["w1_init"]), "w2_init": str(z["w2_init"]),
            "out_bias_name": obname,
            "out_bias": None if obias is None else obias + correction,
        }
        print(f"  L{li:<2d} saqlanadi {len(keep):5d}/4096   qoldiq o'rtachasi "
              f"normasi {np.linalg.norm(correction):.4f}   "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del x_by, x, w2_full, w2_pruned
        gc.collect()
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from flap_baseline import apply_and_quantize

    for tau in TAUS:
        path = f"{OUT_DIR}/enc_hybrid_tau{tau}_gptq.onnx"
        if os.path.exists(path):
            print(f"[tau={tau}] mavjud, {os.path.getsize(path)/1024**2:.0f} MiB")
            continue
        print(f"\n[tau={tau}] bizning kompensatsiya + bias tuzatmasi")
        pm = build_map(tau)
        n = apply_and_quantize(pm, path, f"hybrid{tau}")
        print(f"  {n} qatlamda chiqish biasi tuzatildi")
        print(f"  saqlandi: {path}  {os.path.getsize(path)/1024**2:.0f} MiB")
        del pm
        gc.collect()

    print("\nTaqqoslash (254 MiB byudjetida): bizniki 0.2006, FLAP 0.1925, "
          "Wanda mezoni 0.2202.")


if __name__ == "__main__":
    main()
