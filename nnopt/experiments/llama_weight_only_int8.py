"""Real WEIGHT-ONLY INT8 kernels for Llama.

Why this run exists. torch.ao.quantization.quantize_dynamic quantizes
activations as well as weights, and on this model that is destructive: with
the feed-forward projections quantized that way, perplexity rose from 230 to
3521, while the same weights quantized to INT8 with activations left in FP32
gave 230.9. The FFN intermediate (SiLU(gate) * up) carries extreme outliers,
so per-tensor dynamic activation quantization collapses it -- the phenomenon
LLM.int8() and SmoothQuant were built to address.

Our method targets weight-only quantization, which is also what practical
INT8 LLM deployments use. PyTorch exposes torch._weight_int8pack_mm, the
weight-only INT8 GEMM behind gpt-fast/torchao, so a genuine measurement is
possible here without adding a dependency: weights are stored as INT8 with
per-output-channel scales, activations stay FP32, and the kernel is real.

Variants: FP32 baseline; weight-only INT8 on the FFN (the cascade's case-3
selection); and low-rank plus weight-only INT8. Quality, latency and resident
size are all measured on the same object.
"""

import gc
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("LLAMA_MODEL", "models/open_llama_3b")
FACTOR_CACHE = "models/_llama_factors"
TEXT_CACHE = "models/_calib_cache/uz_text.npz"
SEQ_LEN = 128
N_CALIB_BLOCKS, N_EVAL_BLOCKS = 160, 60
LAT_BATCH, LAT_SEQ = 1, 128
WARMUP, MEASURED = 2, 6
FFN = ("gate_proj", "up_proj", "down_proj")
OUT_JSON = "experiments/results_llama_weight_only.json"


class WeightOnlyInt8Linear(torch.nn.Module):
    """INT8 weights with per-output-channel FP32 scales; FP32 activations.

    Uses torch._weight_int8pack_mm, which takes int8 weights of shape
    (out, in), fp32 input (B, in) and fp32 per-output-channel scales (out,).
    """

    def __init__(self, weight_fp32: torch.Tensor):
        super().__init__()
        w = weight_fp32.detach().to(torch.float32)
        # symmetric per-output-channel scale
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q.contiguous())
        self.register_buffer("scale", scale.to(torch.float32).contiguous())
        self.out_features, self.in_features = w.shape

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).to(torch.float32).contiguous()
        out = torch._weight_int8pack_mm(flat, self.qweight, self.scale)
        return out.reshape(*shape[:-1], self.out_features)


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
    nb = len(ids) // SEQ_LEN
    blocks = torch.tensor(np.asarray(ids[:nb * SEQ_LEN]).reshape(nb, SEQ_LEN),
                          dtype=torch.long)
    return blocks[N_CALIB_BLOCKS:N_CALIB_BLOCKS + N_EVAL_BLOCKS]


def perplexity(model, blocks, batch=2):
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(blocks), batch):
            ids = blocks[i:i + batch]
            logits = model(input_ids=ids).logits[:, :-1]
            tgt = ids[:, 1:]
            tot_nll += float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum"))
            tot_tok += int(tgt.numel())
    return float(np.exp(tot_nll / max(tot_tok, 1)))


def latency_ms(model):
    ids = torch.randint(0, 3000, (LAT_BATCH, LAT_SEQ), dtype=torch.long)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_ids=ids)
        ts = []
        for _ in range(MEASURED):
            t0 = time.perf_counter()
            model(input_ids=ids)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def state_size_mib(model):
    path = "models/_tmp_wo.pt"
    torch.save(model.state_dict(), path)
    s = os.path.getsize(path) / 1024**2
    os.remove(path)
    return s


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                             low_cpu_mem_usage=True)
    m.eval()
    return m


def quantize_ffn_weight_only(model):
    n = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        for name in FFN:
            mod = getattr(mlp, name)
            if isinstance(mod, torch.nn.Linear):
                setattr(mlp, name, WeightOnlyInt8Linear(mod.weight))
                n += 1
            elif isinstance(mod, LowRankLinear):
                mod.first = WeightOnlyInt8Linear(mod.first.weight)
                mod.second = WeightOnlyInt8Linear(mod.second.weight)
                n += 2
    return n


def apply_lowrank(model):
    n = 0
    for li in range(model.config.num_hidden_layers):
        f = f"{FACTOR_CACHE}/L{li}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        mlp = model.model.layers[li].mlp
        for name in FFN:
            setattr(mlp, name, LowRankLinear(z[f"{name}_a"], z[f"{name}_b"]))
        n += 1
    return n


def evaluate(tag, model, blocks, rows):
    size, ms = state_size_mib(model), latency_ms(model)
    ppl = perplexity(model, blocks)
    rows[tag] = {"ppl": ppl, "ms": ms, "mib": size}
    print(f"  {tag:38s} PPL={ppl:8.3f}  {ms:8.1f} ms  {size:6.0f} MiB", flush=True)


def main():
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    blocks = load_blocks(tok)
    print(f"vazn-only INT8 (torch._weight_int8pack_mm), 1 oqim, "
          f"{len(blocks)} blok baholash\n")

    rows = {}

    print("[1] FP32 baza")
    m = fresh_model()
    evaluate("FP32", m, blocks, rows)
    del m
    gc.collect()

    print("\n[2] FFN da vazn-only INT8 (kaskad tanlovi)")
    m = fresh_model()
    n = quantize_ffn_weight_only(m)
    print(f"  {n} ta operator kvantlandi")
    evaluate("FFN vazn-only INT8", m, blocks, rows)
    del m
    gc.collect()

    print("\n[3] past-rank + FFN vazn-only INT8")
    m = fresh_model()
    applied = apply_lowrank(m)
    n = quantize_ffn_weight_only(m)
    print(f"  {applied} qatlamga past-rank, {n} ta faktor kvantlandi")
    evaluate("past-rank + vazn-only INT8", m, blocks, rows)
    del m
    gc.collect()

    json.dump(rows, open(OUT_JSON, "w"), indent=2)

    base = rows["FP32"]
    print("\n" + "=" * 94)
    print("LLAMA: HAQIQIY VAZN-ONLY INT8 YADROLARI (open_llama_3b)")
    print("=" * 94)
    print(f"{'Variant':32s} {'PPL':>9s} {'FP32 ga':>9s} {'ms':>9s} {'tezlanish':>10s} "
          f"{'MiB':>7s} {'siqish':>8s}")
    print("-" * 94)
    for k, v in rows.items():
        print(f"{k:32s} {v['ppl']:9.3f} {v['ppl']/base['ppl']:8.3f}x "
              f"{v['ms']:9.1f} {base['ms']/v['ms']:9.2f}x "
              f"{v['mib']:7.0f} {base['mib']/v['mib']:7.2f}x")
    print("=" * 94)
    print("\nTaqqoslash uchun, faollashuvlarni ham kvantlaydigan quantize_dynamic:")
    print("  FFN da: PPL 3521.5 (15.3x yomonlashuv) — vazn-only bilan solishtiring.")


if __name__ == "__main__":
    main()
