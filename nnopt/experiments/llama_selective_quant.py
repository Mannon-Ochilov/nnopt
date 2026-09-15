"""Which operators may be quantized? Attribution with real INT8 kernels.

The first real-kernel run quantized EVERY nn.Linear and perplexity rose from
230 to 2401. The simulated run had reported 230.9, but it only touched the
three feed-forward projections -- the operators the cascade actually flags as
case 3. The two runs were therefore not comparable, and the gap is a result
rather than a discrepancy: indiscriminate quantization is destructive here,
while the cascade's selective decision may not be.

This isolates the cause by quantizing progressively larger sets with real
kernels:

    A  FFN only (gate/up/down)      -- what the cascade selects
    B  FFN + attention              -- adds q/k/v/o projections
    C  everything except lm_head    -- isolates the output projection
    D  everything                   -- already measured: PPL 2401

torch.ao.quantization.quantize_dynamic accepts a name-keyed qconfig_spec, so
module sets can be selected exactly without rebuilding the model.
"""

import gc
import json
import os
import time

import numpy as np
import torch
from torch.ao.quantization import per_channel_dynamic_qconfig, quantize_dynamic
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("LLAMA_MODEL", "models/open_llama_3b")
FACTOR_CACHE = "models/_llama_factors"
TEXT_CACHE = "models/_calib_cache/uz_text.npz"
SEQ_LEN = 128
N_CALIB_BLOCKS, N_EVAL_BLOCKS = 160, 60
LAT_BATCH, LAT_SEQ = 1, 128
WARMUP, MEASURED = 2, 6
FFN = ("gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
OUT_JSON = "experiments/results_llama_selective.json"


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
    path = "models/_tmp_sel.pt"
    torch.save(model.state_dict(), path)
    s = os.path.getsize(path) / 1024**2
    os.remove(path)
    return s


def linear_names(model, include_ffn, include_attn, include_head):
    """Fully-qualified names of the Linear modules to quantize."""
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in FFN and include_ffn:
            names.append(name)
        elif leaf in ATTN and include_attn:
            names.append(name)
        elif "lm_head" in name and include_head:
            names.append(name)
        elif include_ffn and leaf in ("first", "second"):
            # LowRankLinear factors sit inside the FFN modules
            parent = ".".join(name.split(".")[:-1]).split(".")[-1]
            if parent in FFN:
                names.append(name)
    return names


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                             low_cpu_mem_usage=True)
    m.eval()
    return m


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


def run_variant(tag, blocks, rows, include_ffn, include_attn, include_head,
                lowrank=False):
    model = fresh_model()
    if lowrank:
        applied = apply_lowrank(model)
        print(f"  ({applied} qatlamga past-rank qo'llandi)", flush=True)
    names = linear_names(model, include_ffn, include_attn, include_head)
    spec = {nm: per_channel_dynamic_qconfig for nm in names}
    qmodel = quantize_dynamic(model, spec, dtype=torch.qint8)
    del model
    gc.collect()
    size, ms = state_size_mib(qmodel), latency_ms(qmodel)
    ppl = perplexity(qmodel, blocks)
    rows[tag] = {"ppl": ppl, "ms": ms, "mib": size, "quantized_linears": len(names)}
    print(f"  {tag:36s} PPL={ppl:9.3f}  {ms:8.1f} ms  {size:6.0f} MiB  "
          f"({len(names)} Linear)", flush=True)
    del qmodel
    gc.collect()


def main():
    torch.backends.quantized.engine = "onednn"
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    blocks = load_blocks(tok)
    print(f"backend onednn, 1 oqim, baholash {len(blocks)} blok\n")

    rows = {}
    # FP32 and all-Linear results are already measured; recorded for the table.
    rows["FP32"] = {"ppl": 230.122, "ms": 8948.6, "mib": 13071, "quantized_linears": 0}
    rows["REAL INT8 barcha Linear"] = {"ppl": 2400.913, "ms": 2441.3, "mib": 3561,
                                       "quantized_linears": -1}

    print("[A] FFN operatorlari (kaskad tanlovi)")
    run_variant("A: FFN only", blocks, rows, True, False, False)

    print("\n[B] FFN + attention")
    run_variant("B: FFN + attention", blocks, rows, True, True, False)

    # In Llama the only Linear families are FFN, attention and lm_head, so
    # "everything except lm_head" is exactly variant B. Quantizing lm_head
    # ALONE is the informative test: it isolates the output projection, the
    # prime suspect for the jump to PPL 2401.
    print("\n[C] faqat lm_head")
    run_variant("C: faqat lm_head", blocks, rows, False, False, True)

    print("\n[D] FFN + past-rank")
    run_variant("D: FFN + past-rank", blocks, rows, True, False, False, lowrank=True)

    json.dump(rows, open(OUT_JSON, "w"), indent=2)

    base = rows["FP32"]
    print("\n" + "=" * 96)
    print("QAYSI OPERATORLARNI KVANTLASH MUMKIN? (real INT8 yadrolari, open_llama_3b)")
    print("=" * 96)
    print(f"{'Variant':36s} {'PPL':>10s} {'FP32 ga':>9s} {'ms':>9s} {'tezlanish':>10s} "
          f"{'MiB':>7s} {'siqish':>8s}")
    print("-" * 96)
    for k, v in rows.items():
        print(f"{k:36s} {v['ppl']:10.3f} {v['ppl']/base['ppl']:8.3f}x "
              f"{v['ms']:9.1f} {base['ms']/v['ms']:9.2f}x "
              f"{v['mib']:7.0f} {base['mib']/v['mib']:7.2f}x")
    print("=" * 96)


if __name__ == "__main__":
    main()
