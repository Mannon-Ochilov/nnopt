"""The soft-target policy at L3 = 12 MiB: fit if you can, otherwise miss least.

Cache residency was never meant as a gate. The agreed reading is that fitting
inside alpha*L3 is what the cascade AIMS for, and where the model cannot get
there, the objective degrades gracefully into minimising cache misses under
the accuracy budget -- it does not escalate into forcing the fit.

That distinction has teeth at 12 MiB, where the target demands 45% of the
channels of every encoder layer while our criterion finds redundancy only in
the early ones. Under a hard constraint the cascade must take the other 45%
anyway, abandoning its own criterion in the late layers (arm H,
l3_12_cascade.py). Under the soft target it takes what the criterion actually
endorses and stops -- here tau = 0.90, the point past which the grouping is no
longer merging collinear responses in the deep layers.

The two arms differ only in that policy: identical criterion, identical GPTQ
pass, identical decoder. What separates them is whether an unreachable cache
target is allowed to override the accuracy evidence.

A second consequence worth stating: under the hard constraint the objective is
binary, which is why the alpha = 0.7 boundary sat 0.033 away from flipping the
decoder's decision (Sec 4.1). Minimising misses makes the objective
continuous, so a small change in alpha moves the answer a little instead of
inverting it.
"""

import gc
import glob
import os

import numpy as np

TAU = float(os.environ.get("TAU", "0.9"))
SRC_DIR = "models/_prune"
OUT_DIR = "models/_l3_12"
N_CHANNELS = 4096


def load_maps():
    out = {}
    for f in sorted(glob.glob(f"{SRC_DIR}/prune_L*_tau{TAU}.npz")):
        li = int(f.split("_L")[1].split("_")[0])
        z = np.load(f, allow_pickle=True)
        bn = str(z["bias_name"])
        out[li] = {"keep": z["keep"], "w1": z["w1"], "w2": z["w2"],
                   "bias": z["bias"] if bn != "None" else None,
                   "bias_name": bn, "w1_init": str(z["w1_init"]),
                   "w2_init": str(z["w2_init"])}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/enc_soft_tau{TAU}_gptq.onnx"
    if os.path.exists(path):
        print(f"mavjud: {path}  {os.path.getsize(path)/1024**2:.0f} MiB")
        return

    pm = load_maps()
    if not pm:
        raise SystemExit(f"{SRC_DIR} da tau={TAU} xaritalari topilmadi")
    fr = [1 - len(d["keep"]) / N_CHANNELS for _, d in sorted(pm.items())]
    print(f"tau = {TAU}, {len(pm)} qatlam, o'rtacha olib tashlash "
          f"{np.mean(fr)*100:.1f}% (min {min(fr)*100:.1f}, maks {max(fr)*100:.1f})")
    print("Kesh maqsadi har bir qatlamdan 45% so'raydi; mezon buni faqat "
          f"{sum(1 for f in fr if f >= 0.45)} qatlamda beradi — "
          "qolganida yumshoq siyosat to'xtaydi.\n")

    from gptq_plus_pruning import build_gptq_model
    print("[yumshoq arm] GPTQ bilan kvantlanmoqda...", flush=True)
    build_gptq_model(f"{OUT_DIR}/_tmp_soft.onnx", path, pm, f"soft-t{TAU}")
    del pm
    gc.collect()
    print(f"  saqlandi: {path}  {os.path.getsize(path)/1024**2:.0f} MiB")


if __name__ == "__main__":
    main()
