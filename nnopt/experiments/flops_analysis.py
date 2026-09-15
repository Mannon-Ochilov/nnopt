"""Where does structural pruning sit on the FLOPs axis?

Three compression axes appear in this work and they do different things to
arithmetic, which is easiest to see stated plainly:

  quantization   leaves the multiply-accumulate COUNT untouched and makes
                 each one cheaper -- 4x fewer bytes, 1x fewer MACs
  low-rank       replaces an (m x n) product with (m x r) + (r x n), so MACs
                 fall to r(m+n) from mn
  structural     removes k channels outright, so MACs fall to m(n-k)

At a matched parameter budget low-rank and structural removal cut the same
number of MACs, because for a matmul the parameter count IS the MAC count per
position. That makes them directly comparable at equal FLOPs, and the
comparison is informative: Sec 4.9 measured 0.1833 for structural removal
against 0.3056 for allocated low-rank.

The criterion comparison (ours / magnitude / Wanda) is deliberately NOT a
FLOPs question: all three remove the same channel count per layer by
construction, so they sit at identical arithmetic and differ only in quality.

MACs are counted from the deployed artifacts rather than from intent: every
weighted matmul in each encoder is read out of the graph and multiplied by
the 1500 positions of one encoder pass. This works uniformly for float and
quantized graphs, and it counts exactly the arithmetic that pruning and
low-rank change. Attention score/context products carry no weights and are
identical across variants, so they are reported separately and excluded from
the comparison.
"""

import json
import os

import onnx
from onnx import numpy_helper

ENC_POSITIONS = 1500
VTUNE_JSON = "experiments/results_vtune_whole_model.json"
WER_JSON = "experiments/results_final_wer_testsplit.json"
OUT_JSON = "experiments/results_flops.json"

VARIANTS = [
    ("FP32", "models/uzbek_stt_v1_onnx/encoder_model.onnx", None),
    ("INT8 per-tensor", "models/_enc_v2/enc_int8.onnx", "INT8"),
    ("GPTQ (qisqartirishsiz)", "models/_gptq/enc_gptq_only.onnx", "GPTQ yolg'iz"),
    ("qisqartirish + GPTQ, tau=0.99", "models/_gptq/enc_gptq_pruned.onnx",
     "qisqartirish + GPTQ"),
    ("qisqartirish + GPTQ, tau=0.97",
     "models/_ratio_sweep/enc_bizniki_tau0.97_gptq.onnx",
     "t97 bizniki + GPTQ (eps=5% tanlovi)"),
    ("qisqartirish + GPTQ, tau=0.95",
     "models/_ratio_sweep/enc_bizniki_tau0.95_gptq.onnx", "t95 bizniki + GPTQ"),
    ("qisqartirish + GPTQ, tau=0.93",
     "models/_ratio_sweep/enc_bizniki_tau0.93_gptq.onnx", "t93 bizniki + GPTQ"),
    ("past-rank, optimal taqsimot", "models/_alloc/enc_greedy.onnx",
     "INT8 + taqsimlangan rank"),
    ("past-rank, bir xil taqsimot", "models/_alloc/enc_uniform.onnx",
     "INT8 + bir xil rank"),
]


def weight_macs(path, positions=ENC_POSITIONS):
    """MACs contributed by weighted matmuls in one forward pass.

    For a weight of shape (a, b) applied at every position, the product costs
    a*b MACs per position regardless of how the tensor is stored, so int8 and
    float graphs are counted identically -- which is the point: quantization
    does not change this number.
    """
    model = onnx.load(path, load_external_data=False)
    total, tensors = 0, 0
    for init in model.graph.initializer:
        name = init.name
        if name.endswith(("_scale", "_zero_point")):
            continue
        dims = list(init.dims)
        if len(dims) != 2 or min(dims) < 8:
            continue
        total += dims[0] * dims[1] * positions
        tensors += 1
    return total, tensors


def load_map(path, key, field):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig") as f:
        rows = json.load(f)
    return {r[key]: r for r in rows if field in r}


def main():
    lat = load_map(VTUNE_JSON, "variant", "ms_per_iter")
    lat_by_path = {r["path"]: r for r in lat.values()}
    wer = {r["variant"]: r for r in json.load(
        open(WER_JSON, encoding="utf-8-sig")) if r.get("n") == 300}

    rows = []
    for label, path, wer_key in VARIANTS:
        if not os.path.exists(path):
            print(f"SKIP {label}: {path} yo'q")
            continue
        macs, n = weight_macs(path)
        r = {"variant": label, "path": path, "macs": macs, "tensors": n,
             "mib": os.path.getsize(path) / 1024 ** 2}
        if wer_key and wer_key in wer:
            r["wer"] = wer[wer_key]["wer"]
        if path in lat_by_path:
            r["ms"] = lat_by_path[path]["ms_per_iter"]
        rows.append(r)

    base = rows[0]
    json.dump(rows, open(OUT_JSON, "w"), indent=2)

    print(f"Enkoderning bitta o'tishi ({ENC_POSITIONS} pozitsiya). "
          f"MAC = vaznli matmullar.\n")
    print(f"{'Variant':34s} {'GMAC':>9s} {'FP32 ga':>8s} {'MiB':>6s} "
          f"{'ms':>8s} {'WER':>7s}")
    print("-" * 82)
    for r in rows:
        red = base["macs"] / r["macs"]
        ms = f"{r['ms']:8.0f}" if "ms" in r else " " * 7 + "-"
        w = f"{r['wer']:7.4f}" if "wer" in r else " " * 6 + "-"
        print(f"{r['variant']:34s} {r['macs']/1e9:9.1f} {red:7.2f}x "
              f"{r['mib']:6.0f} {ms} {w}")

    print("\nFLOP tejash vaqtga qanchalik aylanadi (o'lchangan variantlar):")
    print(f"  {'Variant':34s} {'MAC kamayishi':>14s} {'vaqt kamayishi':>15s} "
          f"{'konversiya':>11s}")
    ref = next((r for r in rows if r["variant"].startswith("GPTQ") and "ms" in r),
               None)
    if ref:
        for r in rows:
            if "ms" not in r or r is ref:
                continue
            mac_gain = ref["macs"] / r["macs"]
            time_gain = ref["ms"] / r["ms"]
            conv = (time_gain - 1) / (mac_gain - 1) if mac_gain > 1.0001 else float("nan")
            print(f"  {r['variant']:34s} {mac_gain:13.3f}x {time_gain:14.3f}x "
                  f"{conv:10.2f}")
        print("\n  Mos yozuvlar: GPTQ (qisqartirishsiz). Konversiya = "
              "(vaqt yutug'i - 1) / (MAC yutug'i - 1);\n  1.0 dan past bo'lsa "
              "arifmetika tejashning bir qismi vaqtga aylanmaydi.")

    lr = [r for r in rows if r["variant"].startswith("past-rank")]
    pr = [r for r in rows if r["variant"].startswith("qisqartirish")]
    if lr and pr:
        print("\nTENG FLOPs da sifat (past-rank va strukturaviy qisqartirish):")
        for r in lr + pr:
            if "wer" not in r:
                continue
            print(f"  {r['variant']:34s} {r['macs']/1e9:8.1f} GMAC   "
                  f"WER {r['wer']:.4f}")


if __name__ == "__main__":
    main()
