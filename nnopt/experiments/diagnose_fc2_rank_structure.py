"""Why did fc2's CUR quality (e_loc ~0.37-0.52) come out worse than proj_out's
(e_loc ~0.18) despite fc2 having FAR more measured activation redundancy
(965/4096 nodes mergeable at cos>=0.8 vs 0/1024 for proj_out)?

Two candidate explanations, disentangled here:
  (a) fc2's required reduction is simply harsher (18.3x vs 12.1x, because
      the target is L2=1.25MiB not L3=24MiB) -- a harder target alone
      could explain worse quality independent of matrix structure.
  (b) activation redundancy (which columns of X look alike) does not
      imply the WEIGHT matrix W itself has concentrated singular-value
      energy -- these are different properties. Functional grouping only
      improves WHICH columns CUR picks; it cannot fix an intrinsically
      flat singular spectrum.

Measures the singular-value energy spectrum of fc2's raw W directly, and
reports what rank is needed to capture 80/90/95/99% of its energy --
independent of any calibration data -- to see whether rank=173 (what the
budget search picked, discounted for INT8 headroom) was already an
unfavorable point on the curve.
"""

import numpy as np
import onnx

from nnopt.cur.svd_cur import analyze_spectrum

DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
WEIGHT_INPUT_NAME = "onnx::MatMul_5447"  # fc2 weight, layer 0
REF_PROJ_OUT_WEIGHT = "model.decoder.embed_tokens.weight"


def report(name, w):
    spectrum = analyze_spectrum(w)
    print(f"\n{name}  shape={w.shape}")
    for frac in (0.80, 0.90, 0.95, 0.99):
        r = spectrum.rank_for_energy(frac)
        print(f"  rank needed for {frac*100:5.1f}% energy: {r:5d} / {min(w.shape)} "
              f"({100*r/min(w.shape):5.1f}% of full rank)")
    # energy captured at a FIXED rank, for direct before/after-like comparison
    for r_fixed in (173, 331):
        if r_fixed <= min(w.shape):
            frac = spectrum.energy_cumulative_fraction[r_fixed - 1]
            print(f"  energy captured AT rank={r_fixed}: {frac*100:5.1f}%")


def main():
    model = onnx.load(DECODER_PATH)

    init = next(i for i in model.graph.initializer if i.name == WEIGHT_INPUT_NAME)
    w_fc2 = onnx.numpy_helper.to_array(init).astype(np.float64)
    w_fc2 = w_fc2.T if w_fc2.shape[0] == 4096 else w_fc2  # -> (1024, 4096)
    report("fc2 (decoder.layers.0.fc2)", w_fc2)

    init2 = next(i for i in model.graph.initializer if i.name == REF_PROJ_OUT_WEIGHT)
    w_proj = onnx.numpy_helper.to_array(init2).astype(np.float64)  # (51865, 1024) already m x n
    report("proj_out (reference)", w_proj)


if __name__ == "__main__":
    main()
