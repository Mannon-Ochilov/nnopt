"""Llama with REAL INT8 kernels, not simulated quantization.

The earlier Llama experiment used fake-quant (weights dequantized back onto
our per-channel grid), which measures quality but says nothing about speed
or memory. That was a limitation of the route taken, not of the method:
ONNX export of a 3B model crosses the 2 GiB protobuf limit and needs
external-data handling.

PyTorch's own dynamic quantization avoids that entirely. It replaces
nn.Linear with real INT8 kernels on CPU, so latency, resident size and
quality are all measurable on the same object. It also lets the
per-tensor/per-channel comparison run on genuine kernels rather than on a
simulated grid.

Because LowRankLinear is itself built from two nn.Linear modules, the same
call quantizes the factorized variant without special handling.

Low-rank factors are loaded from models/_llama_factors/, cached by the
previous run, so the expensive factorization is not repeated.
"""

import gc
import json
import os
import time

import numpy as np
import torch
from torch.ao.quantization import (
    default_dynamic_qconfig,
    per_channel_dynamic_qconfig,
    quantize_dynamic,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("LLAMA_MODEL", "models/open_llama_3b")
FACTOR_CACHE = "models/_llama_factors"
TEXT_CACHE = "models/_calib_cache/uz_text.npz"
SEQ_LEN = 128
N_CALIB_BLOCKS = 160
N_EVAL_BLOCKS = 60
LAT_BATCH, LAT_SEQ = 1, 128
WARMUP, MEASURED = 3, 10
TARGETS = ("gate_proj", "up_proj", "down_proj")
OUT_JSON = "experiments/results_llama_real_quant.json"


class LowRankLinear(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.first = torch.nn.Linear(b.shape[1], b.shape[0], bias=False)
        self.second = torch.nn.Linear(a.shape[1], a.shape[0], bias=False)
        with torch.no_grad():
            self.first.weight.copy_(torch.from_numpy(np.asarray(b)).float())
            self.second.weight.copy_(torch.from_numpy(np.asarray(a)).float())

    def forward(self, x):
        return self.second(self.first(x))


def load_blocks(tok):
    texts = list(np.load(TEXT_CACHE, allow_pickle=True)["texts"])
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False).input_ids)
        ids.append(tok.eos_token_id)
    n_blocks = len(ids) // SEQ_LEN
    blocks = torch.tensor(
        np.asarray(ids[:n_blocks * SEQ_LEN]).reshape(n_blocks, SEQ_LEN), dtype=torch.long)
    need = N_CALIB_BLOCKS + N_EVAL_BLOCKS
    if n_blocks < need:
        raise SystemExit(f"faqat {n_blocks} blok mavjud, {need} kerak")
    return blocks[N_CALIB_BLOCKS:need]


def perplexity(model, blocks, batch=2):
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(blocks), batch):
            ids = blocks[i:i + batch]
            logits = model(input_ids=ids).logits[:, :-1]
            targets = ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
            total_nll += float(nll)
            total_tok += int(targets.numel())
    return float(np.exp(total_nll / max(total_tok, 1)))


def latency_ms(model, n_warmup=WARMUP, n_run=MEASURED):
    ids = torch.randint(0, 3000, (LAT_BATCH, LAT_SEQ), dtype=torch.long)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids=ids)
        times = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            model(input_ids=ids)
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def state_size_mib(model):
    """Resident size of the saved state dict, which reflects INT8 storage."""
    path = "models/_tmp_state.pt"
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024**2
    os.remove(path)
    return size


def apply_lowrank(model):
    n = model.config.num_hidden_layers
    applied = 0
    for li in range(n):
        f = f"{FACTOR_CACHE}/L{li}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        mlp = model.model.layers[li].mlp
        for name in TARGETS:
            setattr(mlp, name, LowRankLinear(z[f"{name}_a"], z[f"{name}_b"]))
        applied += 1
    return applied


def evaluate(tag, model, blocks, rows):
    size = state_size_mib(model)
    ms = latency_ms(model)
    ppl = perplexity(model, blocks)
    rows[tag] = {"ppl": ppl, "ms": ms, "mib": size}
    print(f"  {tag:34s} PPL={ppl:8.3f}  {ms:8.1f} ms  {size:7.0f} MiB", flush=True)
    return rows[tag]


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                             low_cpu_mem_usage=True)
    m.eval()
    return m


def main():
    torch.backends.quantized.engine = "onednn"
    torch.set_num_threads(1)
    print(f"kvantlash backend: {torch.backends.quantized.engine}, 1 oqim")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    blocks = load_blocks(tok)
    print(f"baholash: {len(blocks)} blok x {SEQ_LEN} token (held-out, padding yo'q)\n")

    rows = {}

    print("[1] FP32 baza")
    model = fresh_model()
    evaluate("FP32", model, blocks, rows)
    del model
    gc.collect()

    print("\n[2] REAL INT8 dinamik kvantlash (per-tensor)")
    model = fresh_model()
    qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    del model
    gc.collect()
    evaluate("REAL INT8 per-tensor", qmodel, blocks, rows)
    del qmodel
    gc.collect()

    print("\n[3] REAL INT8 dinamik kvantlash (per-channel)")
    model = fresh_model()
    qmodel = quantize_dynamic(model, {torch.nn.Linear: per_channel_dynamic_qconfig},
                              dtype=torch.qint8)
    del model
    gc.collect()
    evaluate("REAL INT8 per-channel", qmodel, blocks, rows)
    del qmodel
    gc.collect()

    print("\n[4] past-rank + REAL INT8 per-channel")
    model = fresh_model()
    applied = apply_lowrank(model)
    print(f"  {applied} qatlamga keshlangan past-rank faktorlari qo'llandi")
    qmodel = quantize_dynamic(model, {torch.nn.Linear: per_channel_dynamic_qconfig},
                              dtype=torch.qint8)
    del model
    gc.collect()
    evaluate("past-rank + REAL INT8 per-ch.", qmodel, blocks, rows)
    del qmodel
    gc.collect()

    json.dump(rows, open(OUT_JSON, "w"), indent=2)

    base = rows["FP32"]
    print("\n" + "=" * 92)
    print("LLAMA: REAL INT8 YADROLARI (simulyatsiya emas), open_llama_3b")
    print("=" * 92)
    print(f"{'Variant':34s} {'PPL':>9s} {'FP32 ga':>9s} {'ms':>9s} {'tezlanish':>10s} "
          f"{'MiB':>8s} {'siqish':>8s}")
    print("-" * 92)
    for k, v in rows.items():
        print(f"{k:34s} {v['ppl']:9.3f} {v['ppl']/base['ppl']:8.3f}x "
              f"{v['ms']:9.1f} {base['ms']/v['ms']:9.2f}x "
              f"{v['mib']:8.0f} {base['mib']/v['mib']:7.2f}x")
    print("=" * 92)


if __name__ == "__main__":
    main()
