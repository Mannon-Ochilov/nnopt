"""Cache size as the PRIMARY input of the cascade (README Sec 2.2/2.5) --
re-deriving the compression targets that Sec 8.3.6/8.3.7 had picked
arbitrarily (2x / 4x / 8x).

The point the author raised: the cascade's target must COME FROM the cache
budget, not be chosen by hand. This script computes, for the real machine
and the real model, what the cache-anchored requirement actually is at
three granularities -- and shows that the answer depends critically on
whether weights are REUSED, which differs between encoder and decoder.

Granularities:
  per-operator : one weight matrix must fit alpha*L3
  per-layer    : one transformer layer's weights must fit alpha*L3
  whole-model  : all weights resident in alpha*L3

Reuse (the part that decides whether cache-fit means anything):
  encoder, 1500 positions/pass : each weight is used 1500 times per pass
                                 -> weights want to STAY in cache while
                                    positions stream past -> cache-fit is
                                    the right criterion, and FLOP savings
                                    (CUR/SVD) convert into real time.
  decoder, batch=1 autoregr.   : each weight is used ONCE per token, and by
                                 the time layer 0 is revisited for the next
                                 token, 23 other layers have evicted it.
                                 No reuse -> only TOTAL BYTES MOVED matter
                                 -> quantization wins, low-rank does not.

That asymmetry is what the measured numbers already showed (README 8.3.7:
INT8 2.95x speedup ~ its 4x byte reduction; CUR's extra FLOP savings bought
only 2.97x).
"""

import numpy as np
import onnx

from nnopt.hw.cache_topology import detect_cache_topology
from nnopt.profiler.graph_profiler import profile_onnx_model

DECODER_PATH = "models/uzbek_stt_v1_onnx/decoder_model.onnx"
ENCODER_PATH = "models/uzbek_stt_v1_onnx/encoder_model.onnx"
ALPHA = 0.7
DEC_DIMS = {"batch_size": 1, "decoder_sequence_length": 16, "encoder_sequence_length": 1500}
ENC_DIMS = {"batch_size": 1, "encoder_sequence_length": 1500}


def weighted_ops(path, dims):
    profs = profile_onnx_model(path, free_dims=dims)
    return [p for p in profs if p.weight_initializer is not None]


def layer_index(name):
    import re
    m = re.search(r"/layers\.(\d+)/", name)
    return int(m.group(1)) if m else -1


def report(tag, path, dims, cache_eff):
    ops = weighted_ops(path, dims)
    model = onnx.load(path, load_external_data=False)
    dims_by_init = {i.name: tuple(i.dims) for i in model.graph.initializer}

    per_op, by_layer = [], {}
    for p in ops:
        shp = dims_by_init.get(p.weight_initializer)
        if not shp:
            continue
        nparams = int(np.prod(shp))
        per_op.append(nparams)
        by_layer.setdefault(layer_index(p.name), []).append(nparams)

    total = sum(per_op)
    biggest_op = max(per_op)
    layer_sizes = [sum(v) for k, v in by_layer.items() if k >= 0]
    per_layer = max(layer_sizes) if layer_sizes else 0

    print(f"\n=== {tag} ===")
    print(f"  operatorlar: {len(per_op)}, jami {total:,} parametr = {total*4/1024**2:,.0f} MiB (fp32)")
    print(f"  qatlamlar  : {len(layer_sizes)}, eng katta qatlam = {per_layer*4/1024**2:.1f} MiB")
    print(f"  eng katta operator = {biggest_op*4/1024**2:.1f} MiB")
    print(f"  samarali kesh byudjeti (alpha={ALPHA}) = {cache_eff/1024**2:.1f} MiB")
    print()
    print(f"  {'granulyarlik':16s} {'hajm(MiB)':>10s} {'talab qilinadigan siqish':>26s}")
    for label, nparams in (
        ("per-operator", biggest_op),
        ("per-layer", per_layer),
        ("whole-model", total),
    ):
        need = (nparams * 4) / cache_eff
        verdict = "allaqachon sig'adi" if need <= 1.0 else f"{need:.2f}x kerak"
        print(f"  {label:16s} {nparams*4/1024**2:10.1f} {verdict:>26s}")
    return {"total": total, "per_layer": per_layer, "biggest_op": biggest_op}


def main():
    topo = detect_cache_topology()
    g = topo.global_shared_cache()
    cache_eff = ALPHA * g.size_bytes
    print(f"Kafolatlangan umumiy kesh: L{g.level}, {g.size_bytes/1024**2:.0f} MiB, {len(g.core_ids)} yadro")

    dec = report("DECODER", DECODER_PATH, DEC_DIMS, cache_eff)
    enc = report("ENCODER", ENCODER_PATH, ENC_DIMS, cache_eff)

    print("\n" + "=" * 78)
    print("QAYTA ISHLATISH (reuse) TAHLILI -- kesh-sig'imi qachon ma'noga ega")
    print("=" * 78)
    print(f"{'':10s} {'pozitsiya/o''tish':>18s} {'vazn qayta ishlatilishi':>26s} {'cheklovchi':>14s}")
    print(f"{'encoder':10s} {1500:>18d} {'1500x (bitta o''tishda)':>26s} {'hisoblash':>14s}")
    print(f"{'decoder':10s} {'1 (batch=1)':>18s} {'1x (har token uchun)':>26s} {'xotira':>14s}")
    print()
    print("Xulosa: decoderda (batch=1) vazn har token uchun bir marta o'qiladi va")
    print("layer 0 ga qaytguncha qolgan 23 qatlam uni keshdan siqib chiqaradi ->")
    print("kesh-sig'imi qayta ishlatish bermaydi, faqat UMUMIY BAYT hajmi muhim")
    print("-> kvantlash yutadi. Encoderda esa vazn 1500 marta ishlatiladi ->")
    print("kesh-sig'imi haqiqiy mezon va FLOPs tejash (CUR/SVD) real vaqtga aylanadi.")

    need_layer_dec = (dec["per_layer"] * 4) / cache_eff
    print(f"\nDecoder uchun per-layer kesh-bog'langan maqsad: {need_layer_dec:.2f}x")
    print(f"  -> INT8 (4.00x) bu maqsadni qoplaydi; kaskadning 'eng yumshoq yetarli'")
    print(f"     tamoyili aynan shuni tanlaydi (README 8.3.7 bilan mos).")
    need_op_enc = (enc["biggest_op"] * 4) / cache_eff
    print(f"\nEncoder uchun per-operator kesh-bog'langan maqsad: {need_op_enc:.2f}x")


if __name__ == "__main__":
    main()
