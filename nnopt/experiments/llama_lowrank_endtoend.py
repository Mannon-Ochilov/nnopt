"""The cascade's low-rank branch, validated end to end for the first time.

Whisper and mBERT never reached case 3 for their FFN operators, so the
branch stayed unexercised. open_llama_3b does reach it: gate_proj, up_proj
and down_proj each sit at 1.57x over alpha*L3 after the mandatory INT8 step
(README Sec 8.3.13). This is therefore the first object where the branch
can actually be tested rather than reasoned about.

Rank comes from the cache budget, not from a hand-picked value. For a
(8640, 3200) operator stored INT8, a rank-r factorization costs
r*(8640+3200) bytes, so fitting alpha*L3 = 16.8 MiB requires

    r <= 16.8 MiB / 11840 bytes ~= 1487        (46% of full rank 3200)

Quality metric: perplexity on held-out Uzbek text -- the causal-LM analogue
of WER, and the same held-out/calibration split discipline used throughout.

Quantization is simulated (fake-quant on our per-channel calibrated grid).
That measures the quality question this experiment asks; wall-clock speed
for Llama would need an INT8 runtime and is out of scope here.
"""

import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnopt.quantizer.per_channel import quantize_weight_per_channel

MODEL_DIR = os.environ.get("LLAMA_MODEL", "models/open_llama_3b")
# Calibration/evaluation come from the 8000-sentence Uzbek text cache, not
# the 120 ASR transcripts. Two reasons, both established earlier in this
# project:
#   * rows/rank >= 10 (README Sec 8.3.10-A). The cache-anchored rank here is
#     1487, so ~15000 token positions are needed; 60 transcripts gave ~5000.
#   * No padding. Sentences are packed into dense SEQ_LEN blocks so every
#     captured position is a real token -- padding positions would otherwise
#     dominate and contaminate the calibration (Fig 2.5).
TEXT_CACHE = "models/_calib_cache/uz_text.npz"
N_CALIB_BLOCKS = 160          # 160 * 128 = 20480 rows -> rows/rank ~ 13.8
N_EVAL_BLOCKS = 60
SEQ_LEN = 128
MAX_ROWS = 20480
Q8 = 127
CACHE_BUDGET_BYTES = 0.7 * 24 * 1024**2
OUT_JSON = "experiments/results_llama_lowrank.json"
FACTOR_CACHE = "models/_llama_factors"
TARGETS = ("gate_proj", "up_proj", "down_proj")


def cache_anchored_rank(m, n):
    """Largest r whose INT8 two-factor form fits alpha*L3."""
    return max(1, min(int(CACHE_BUDGET_BYTES // (m + n)), min(m, n)))


def load_text_blocks(tok, n_calib_blocks, n_eval_blocks, seq_len=SEQ_LEN):
    """Pack sentences into dense token blocks -- no padding anywhere."""
    texts = list(np.load(TEXT_CACHE, allow_pickle=True)["texts"])
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False).input_ids)
        ids.append(tok.eos_token_id)
    total_blocks = len(ids) // seq_len
    blocks = torch.tensor(
        np.asarray(ids[:total_blocks * seq_len]).reshape(total_blocks, seq_len),
        dtype=torch.long)
    need = n_calib_blocks + n_eval_blocks
    if total_blocks < need:
        raise SystemExit(f"faqat {total_blocks} blok mavjud, {need} kerak")
    return blocks[:n_calib_blocks], blocks[n_calib_blocks:need]


def perplexity(model, blocks, batch=4):
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(blocks), batch):
            ids = blocks[i:i + batch]
            logits = model(input_ids=ids).logits[:, :-1]
            targets = ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                reduction="sum")
            total_nll += float(nll)
            total_tok += int(targets.numel())
    return float(np.exp(total_nll / max(total_tok, 1)))


def capture_inputs(model, blocks, layer_indices, batch=4):
    """Input activations of each target Linear, for the given layers.

    Captured one layer-group at a time by the caller: holding 20480 x 8640
    float64 rows for all 26 layers at once would need tens of GB.
    """
    store = {(li, name): [] for li in layer_indices for name in TARGETS}
    handles = []

    def mk(li, name):
        def hook(mod, inputs, output):
            store[(li, name)].append(
                inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]))
        return hook

    for li in layer_indices:
        mlp = model.model.layers[li].mlp
        for name in TARGETS:
            handles.append(getattr(mlp, name).register_forward_hook(mk(li, name)))

    with torch.no_grad():
        for i in range(0, len(blocks), batch):
            model(input_ids=blocks[i:i + batch])
    for h in handles:
        h.remove()

    out = {}
    for key, chunks in store.items():
        x = torch.cat(chunks, dim=0)
        if x.shape[0] > MAX_ROWS:
            x = x[:MAX_ROWS]
        out[key] = x.numpy().astype(np.float64)
    store.clear()
    return out


def activation_aware_factors(w, x_fit, rank, ridge=1e-6):
    """Rank-r A,B minimizing ||X (W - A B)^T||_F  (README Sec 2.5)."""
    n = w.shape[1]
    g = x_fit.T @ x_fit
    scale = float(np.trace(g)) / max(n, 1)
    g = g + np.eye(n) * (ridge * max(scale, 1e-12))
    try:
        L = np.linalg.cholesky(g)
    except np.linalg.LinAlgError:
        ev, evec = np.linalg.eigh(g)
        L = evec * np.sqrt(np.clip(ev, 1e-12, None))
    wl = w @ L
    u, s, vt = np.linalg.svd(wl, full_matrices=False)
    r = max(1, min(rank, len(s)))
    lr = (u[:, :r] * s[:r]) @ vt[:r, :]
    w_approx = np.linalg.solve(L.T, lr.T).T
    uu, ss, vv = np.linalg.svd(w_approx, full_matrices=False)
    rr = min(r, len(ss))
    sq = np.sqrt(ss[:rr])
    return (uu[:, :rr] * sq), (sq[:, None] * vv[:rr, :])


class LowRankLinear(torch.nn.Module):
    def __init__(self, a, b, bias=None):
        super().__init__()
        self.first = torch.nn.Linear(b.shape[1], b.shape[0], bias=False)
        self.second = torch.nn.Linear(a.shape[1], a.shape[0], bias=bias is not None)
        with torch.no_grad():
            self.first.weight.copy_(torch.from_numpy(b).float())
            self.second.weight.copy_(torch.from_numpy(a).float())
            if bias is not None:
                self.second.bias.copy_(bias)

    def forward(self, x):
        return self.second(self.first(x))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("model yuklanmoqda...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    n_layers = model.config.num_hidden_layers

    calib_blocks, eval_blocks = load_text_blocks(tok, N_CALIB_BLOCKS, N_EVAL_BLOCKS)
    print(f"kalibrlash {len(calib_blocks)} blok x {SEQ_LEN} = "
          f"{len(calib_blocks)*SEQ_LEN} token, baholash {len(eval_blocks)} blok "
          f"(kesishmaydi, padding yo'q)")
    print(f"kesh-bog'langan rank {cache_anchored_rank(8640, 3200)} uchun "
          f"qator/rank = {len(calib_blocks)*SEQ_LEN/cache_anchored_rank(8640,3200):.1f}")

    rows = {}
    print("\n[FP32 baza] perplexity hisoblanmoqda...", flush=True)
    ppl_fp32 = perplexity(model, eval_blocks)
    rows["FP32"] = {"ppl": ppl_fp32, "note": "asl model"}
    print(f"  PPL = {ppl_fp32:.3f}")

    # --- snapshot original weights so each variant starts clean ---
    orig = {}
    for li in range(n_layers):
        mlp = model.model.layers[li].mlp
        for name in TARGETS:
            orig[(li, name)] = getattr(mlp, name).weight.detach().clone()

    def restore():
        for li in range(n_layers):
            mlp = model.model.layers[li].mlp
            for name in TARGETS:
                mod = getattr(mlp, name)
                if not isinstance(mod, torch.nn.Linear):
                    w = orig[(li, name)]
                    new = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
                    with torch.no_grad():
                        new.weight.copy_(w)
                    setattr(mlp, name, new)
                else:
                    with torch.no_grad():
                        mod.weight.copy_(orig[(li, name)])

    # --- variant 1: per-channel INT8 only (the mandatory step) ---
    print("\n[INT8 per-channel] qo'llanmoqda...", flush=True)
    for li in range(n_layers):
        mlp = model.model.layers[li].mlp
        for name in TARGETS:
            mod = getattr(mlp, name)
            w = mod.weight.detach().numpy().astype(np.float64)
            wq, _ = quantize_weight_per_channel(w, Q8)
            with torch.no_grad():
                mod.weight.copy_(torch.from_numpy(wq).float())
    ppl_int8 = perplexity(model, eval_blocks)
    rows["INT8 per-channel"] = {"ppl": ppl_int8, "note": "majburiy bosqich"}
    print(f"  PPL = {ppl_int8:.3f}  (FP32 ga nisbatan {ppl_int8/ppl_fp32:.3f}x)")
    restore()
    gc.collect()

    # --- variant 2: cache-anchored low-rank + per-channel INT8 ---
    # Layers are processed in small groups: capturing 20480 x 8640 float64
    # activations for all 26 layers simultaneously would need tens of GB, and
    # each finished layer is cached so a session restart does not discard the
    # whole pass (this has already happened three times).
    print(f"\n[past-rank + INT8] {n_layers} qatlam, guruhlab qayta ishlanmoqda...",
          flush=True)
    os.makedirs(FACTOR_CACHE, exist_ok=True)
    ranks_used, elocs = [], []
    GROUP = 2
    for start in range(0, n_layers, GROUP):
        group = list(range(start, min(start + GROUP, n_layers)))
        pending = [li for li in group
                   if not os.path.exists(f"{FACTOR_CACHE}/L{li}.npz")]
        x_by = capture_inputs(model, calib_blocks, pending) if pending else {}
        for li in group:
            cache_file = f"{FACTOR_CACHE}/L{li}.npz"
            mlp = model.model.layers[li].mlp
            if os.path.exists(cache_file):
                z = np.load(cache_file)
                for name in TARGETS:
                    setattr(mlp, name, LowRankLinear(z[f"{name}_a"], z[f"{name}_b"]))
                    elocs.append(float(z[f"{name}_eloc"]))
                    ranks_used.append(int(z[f"{name}_rank"]))
                continue
            payload = {}
            for name in TARGETS:
                w = orig[(li, name)].numpy().astype(np.float64)
                m, n = w.shape
                r = cache_anchored_rank(m, n)
                x = x_by[(li, name)]
                split = int(x.shape[0] * 0.8)
                x_fit, x_eval = x[:split], x[split:]
                a, b = activation_aware_factors(w, x_fit, r)
                aq, _ = quantize_weight_per_channel(a, Q8)
                bq, _ = quantize_weight_per_channel(b, Q8)
                y_ref = x_eval @ w.T
                e = float(np.linalg.norm(y_ref - (x_eval @ bq.T) @ aq.T) /
                          (np.linalg.norm(y_ref) + 1e-12))
                elocs.append(e)
                ranks_used.append(r)
                payload[f"{name}_a"] = aq.astype(np.float32)
                payload[f"{name}_b"] = bq.astype(np.float32)
                payload[f"{name}_eloc"] = e
                payload[f"{name}_rank"] = r
                setattr(mlp, name, LowRankLinear(aq, bq))
                del w, a, b, aq, bq, x, x_fit, x_eval, y_ref
            np.savez_compressed(cache_file, **payload)
            print(f"    L{li} tayyor  (o'rtacha E_loc {np.mean(elocs):.4f})", flush=True)
        del x_by
        gc.collect()

    ppl_lr = perplexity(model, eval_blocks)
    rows["past-rank + INT8"] = {"ppl": ppl_lr, "rank": int(np.mean(ranks_used)),
                                "eloc_mean": float(np.mean(elocs)),
                                "eloc_max": float(np.max(elocs))}
    print(f"  PPL = {ppl_lr:.3f}  (FP32 ga nisbatan {ppl_lr/ppl_fp32:.3f}x)")

    json.dump(rows, open(OUT_JSON, "w"), indent=2)

    m0, n0 = 8640, 3200
    r0 = cache_anchored_rank(m0, n0)
    print("\n" + "=" * 86)
    print("KASKADNING PAST-RANK SHOXCHASI — birinchi uchdan-uchgacha sinov (open_llama_3b)")
    print("=" * 86)
    print(f"kesh-bog'langan rank: {r0} / {min(m0,n0)} "
          f"({r0/min(m0,n0):.0%} to'liq rankdan)")
    print(f"o'rtacha operator E_loc: {np.mean(elocs):.4f}   maks: {np.max(elocs):.4f}\n")
    print(f"{'Variant':26s} {'PPL':>10s} {'FP32 ga':>10s}")
    print("-" * 86)
    for k, v in rows.items():
        print(f"{k:26s} {v['ppl']:10.3f} {v['ppl']/ppl_fp32:9.3f}x")
    print("=" * 86)


if __name__ == "__main__":
    main()
