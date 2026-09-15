"""Where does the cascade's CUR branch actually TRIGGER?

The cascade's real logic (author, 2026-08-12):

    1. FP32 already fits alpha*L3            -> do nothing
    2. INT8 (mandatory) makes it fit         -> CUR is NOT considered, done
    3. still does not fit after INT8         -> CUR + INT8, and the benefit
                                                must show up as real speed

Every experiment so far measured E_loc at hand-picked compression ratios,
which never tested when case 3 arises at all. This script finds it.

Two corrections to the earlier accounting:

  * The cache budget is consumed by the WHOLE operator footprint, not just
    weights: M_W + M_X + M_Y + M_tmp (graph_profiler already models this).
    For the encoder that dominates -- fc1's output alone is
    1500 x 4096 x 4 B = 24 MiB, larger than all of L3.
  * Quantizing to INT8 shrinks M_W; whether M_X / M_Y also shrink depends
    on whether activations are quantized too, so BOTH variants are
    reported (weight-only vs full INT8).

Output: the list of operators that are still over budget after INT8 --
i.e. the operators where the dissertation's CUR branch is supposed to earn
its place.
"""

import numpy as np

from nnopt.hw.cache_topology import detect_cache_topology
from nnopt.profiler.graph_profiler import profile_onnx_model

ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
ALPHA = 0.7

# Encoder always runs the full 30 s window: 3000 mel frames -> 1500 positions.
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
# Decoder during autoregressive generation: one token at a time, but the
# prefix grows; 16 is a representative mid-utterance length here.
DEC_DIMS = {"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500}


def scaled_footprint(p, weight_bits, act_bits):
    """Operator footprint (bytes) if weights are stored at `weight_bits`
    and activations at `act_bits`. p.M_* are fp32 byte counts."""
    wf = weight_bits / 32.0
    af = act_bits / 32.0
    return p.M_W * wf + (p.M_X + p.M_Y + p.M_tmp) * af


def report(tag, path, dims, budget):
    profs = profile_onnx_model(path, free_dims=dims)
    ops = [p for p in profs if p.weight_initializer is not None]

    rows = []
    for p in ops:
        fp32 = scaled_footprint(p, 32, 32)
        int8_w = scaled_footprint(p, 8, 32)   # weight-only quantization
        int8_full = scaled_footprint(p, 8, 8)  # weights + activations
        rows.append((p, fp32, int8_w, int8_full))

    n = len(rows)
    fits_fp32 = sum(1 for _, a, _, _ in rows if a <= budget)
    fits_w8 = sum(1 for _, _, b, _ in rows if b <= budget)
    fits_full8 = sum(1 for _, _, _, c in rows if c <= budget)

    print(f"\n{'='*84}")
    print(f"{tag}  ({n} vaznli operator, byudjet = alpha*L3 = {budget/1024**2:.1f} MiB)")
    print(f"{'='*84}")
    print(f"  1-holat  FP32 sig'adi            : {fits_fp32:3d}/{n}  -> hech narsa qilinmaydi")
    print(f"  2-holat  INT8(vazn) sig'adi      : {fits_w8:3d}/{n}  -> CUR qaralmaydi")
    print(f"  2b-holat INT8(vazn+faol) sig'adi : {fits_full8:3d}/{n}")
    print(f"  3-holat  INT8 dan keyin SIG'MAYDI: {n-fits_w8:3d}/{n}  <-- CUR shu yerda kerak")

    still_over = [(p, a, b, c) for p, a, b, c in rows if b > budget]
    if still_over:
        print(f"\n  INT8(vazn) dan keyin ham sig'maydigan operatorlar:")
        print(f"  {'operator':46s} {'M_W':>8s} {'M_X':>8s} {'M_Y':>8s} {'int8':>9s} {'ortiqcha':>9s}")
        shown = {}
        for p, a, b, c in still_over:
            kind = p.name.split("/")[-2] if "/" in p.name else p.name
            shown.setdefault(kind, []).append((p, b))
        for kind, items in shown.items():
            p, b = items[0]
            print(f"  {kind + f'  (x{len(items)})':46s} "
                  f"{p.M_W/1024**2:7.1f}M {p.M_X/1024**2:7.1f}M {p.M_Y/1024**2:7.1f}M "
                  f"{b/1024**2:8.1f}M {b/budget:8.2f}x")
    return rows, budget


def main():
    topo = detect_cache_topology()
    g = topo.global_shared_cache()
    budget = ALPHA * g.size_bytes
    print(f"Kafolatlangan umumiy kesh: L{g.level}, {g.size_bytes/1024**2:.0f} MiB, {len(g.core_ids)} yadro")
    print(f"Samarali byudjet (alpha={ALPHA}): {budget/1024**2:.1f} MiB")

    report("ENCODER (1500 pozitsiya)", ENCODER_PATH, ENC_DIMS, budget)
    report("DECODER (16 token)", DECODER_PATH, DEC_DIMS, budget)

    print(f"\n{'='*84}")
    print("XULOSA")
    print(f"{'='*84}")
    print("Agar 3-holat bo'sh bo'lsa -> bu model+mashinada CUR shoxchasi hech qachon")
    print("ishga tushmaydi va uni boshqa rejimda (kattaroq model, kichikroq kesh,")
    print("yoki kattaroq batch) izlash kerak.")


if __name__ == "__main__":
    main()
