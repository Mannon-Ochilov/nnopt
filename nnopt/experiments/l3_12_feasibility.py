"""What would the cascade actually ORDER on a 12 MiB L3, and can it be built?

The sweep (Sec 4.1) showed that at L3 = 12 MiB both the encoder and the
decoder move into case 3, so the cascade stops being satisfied by INT8 and
demands structural reduction on top. Before spending hours building those
models, this asks the question that decides whether they CAN be built: how
much has to come out, and is that much available in the operators our method
is allowed to touch?

The cascade reduces FFN channels; attention is left intact (Sec 3). So the
residual factor after INT8 has to be paid entirely out of the FFN's share of
the layer, and the arithmetic below reports the required FFN removal fraction
rather than the layer-level factor -- the two differ by exactly the attention
mass that cannot be moved.

A required fraction above 1.0 is not a hard experiment; it is a NEGATIVE
result about the derivation's reach, and it is worth knowing before building.
"""

import re

import numpy as np
import onnx

from calib_utils import DECODER_PATH, ENCODER_PATH
from nnopt.profiler.graph_profiler import profile_onnx_model

ALPHA = 0.7
INT8 = 4.0
MIB = 1024.0 ** 2
L3_VALUES = (12.0, 16.0, 24.0)
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}
DEC_DIMS = {"batch_size": 1, "decoder_sequence_length": 16,
            "encoder_sequence_length": 1500}


def layer_index(name):
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def layer_breakdown(path, dims):
    """Bytes per layer, split into what the cascade may and may not touch."""
    profs = [p for p in profile_onnx_model(path, free_dims=dims)
             if p.weight_initializer is not None]
    model = onnx.load(path, load_external_data=False)
    shapes = {i.name: tuple(i.dims) for i in model.graph.initializer}

    ffn, other = {}, {}
    for p in profs:
        li = layer_index(p.name)
        if li < 0 or p.weight_initializer not in shapes:
            continue
        n = int(np.prod(shapes[p.weight_initializer]))
        bucket = ffn if ("/fc1/" in p.name or "/fc2/" in p.name) else other
        bucket[li] = bucket.get(li, 0) + n

    li_max = max(ffn, key=lambda k: ffn.get(k, 0) + other.get(k, 0))
    return ffn.get(li_max, 0) * 4, other.get(li_max, 0) * 4


def main():
    print(f"alpha = {ALPHA}, INT8 = {INT8:.0f}x. Kaskad faqat FFN kanallarini "
          f"qisqartiradi.\n")

    parts = {"ENKODER": layer_breakdown(ENCODER_PATH, ENC_DIMS),
             "DEKODER": layer_breakdown(DECODER_PATH, DEC_DIMS)}
    for tag, (f, o) in parts.items():
        print(f"{tag:9s} eng katta qatlam = {(f+o)/MIB:5.1f} MiB "
              f"(FFN {f/MIB:5.1f} + attn {o/MIB:5.1f})")

    print(f"\n{'L3':>5s}  {'qism':9s} {'talab':>8s} {'INT8 dan keyin':>15s} "
          f"{'FFN dan olish':>15s}  hukm")
    print("-" * 74)
    for l3 in L3_VALUES:
        budget = ALPHA * l3 * MIB
        for tag, (f_b, o_b) in parts.items():
            total = f_b + o_b
            need = total / budget
            if need <= 1.0:
                print(f"{l3:5.0f}  {tag:9s} {'sig`adi':>8s}")
                continue
            resid = need / INT8
            if resid <= 1.0:
                print(f"{l3:5.0f}  {tag:9s} {need:7.2f}x {resid:14.2f}x "
                      f"{'—':>15s}  INT8 yetarli")
                continue
            # After INT8 every byte shrinks 4x, so the layer must lose
            # (1 - 1/resid) of its mass, and all of it from the FFN block.
            must_drop = total * (1.0 - 1.0 / resid)
            frac = must_drop / f_b
            verdict = ("BAJARIB BO'LMAYDI" if frac > 1.0 else
                       "juda agressiv" if frac > 0.6 else "bajarilishi mumkin")
            print(f"{l3:5.0f}  {tag:9s} {need:7.2f}x {resid:14.2f}x "
                  f"{frac*100:14.1f}%  {verdict}")

    f_b, o_b = parts["DEKODER"]
    total = f_b + o_b
    # The largest FFN removal that leaves the FFN non-empty bounds what any
    # channel method can reach; report the factor it would actually deliver.
    best = total / (total - f_b) / 1.0
    print(f"\nDekoder uchun FFN ni BUTUNLAY yo'q qilganda ham qatlam atigi "
          f"{best:.2f}x qisqaradi\n(attn massasi qo'zg'almaydi), ya'ni INT8 "
          f"bilan birga jami {best*INT8:.2f}x — L3 = 12 MiB talab qiladigan "
          f"{total/(ALPHA*12*MIB):.2f}x dan past.")


if __name__ == "__main__":
    main()
