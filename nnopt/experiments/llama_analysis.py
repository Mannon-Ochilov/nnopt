"""Does the method reach its low-rank branch on a Llama-class model?

Why this object. On Whisper and mBERT the cascade never entered case 3 for
the FFN: those operators already fit alpha*L3 = 16.8 MiB after the mandatory
INT8 step. Case 3 needs a single operator above ~16.8M parameters, which is
where Llama-class models sit:

    TinyLlama-1.1B   2048 x  5632 = 11.5M -> 11.5 MiB int8  (case 2)
    open_llama_3b    3200 x  8640 = 27.6M -> 27.6 MiB int8  (case 3, 1.65x)
    Llama-2-7B       4096 x 11008 = 45.1M -> 45.1 MiB int8  (case 3, 2.68x)

Architectural note. Llama's FFN is gated:

    h = SiLU(gate_proj(x)) * up_proj(x)
    y = down_proj(h)

so the intermediate width is the OUTPUT of gate_proj and up_proj and the
INPUT of down_proj. Removing one intermediate channel therefore shrinks
THREE matrices from a single decision, versus two in Whisper/BERT.

Activations are captured with PyTorch forward hooks rather than an ONNX
export: a multi-GB export would need external-data handling and buys
nothing for this diagnostic.

Calibration text stays the Uzbek Common Voice transcripts used throughout,
keeping the language domain constant across all three objects.
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calib_utils import load_audio
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.hw.cache_topology import detect_cache_topology

MODEL_ID = os.environ.get("LLAMA_MODEL", "openlm-research/open_llama_3b_v2")
CACHE_DIR = "models/llama_cache"
N_CALIB_TEXT = 200
SEQ_LEN = 128
MAX_ROWS = 3072
ALPHA = 0.7
TAUS = [0.99, 0.95, 0.90]
EPS_THRESHOLD = 0.5
LAYER_STRIDE = 4           # probe every 4th layer to keep the pass tractable
OUT_JSON = "experiments/results_llama.json"


def cache_report(model):
    topo = detect_cache_topology()
    g = topo.global_shared_cache()
    budget = ALPHA * g.size_bytes
    cfg = model.config
    print(f"Model: {MODEL_ID}")
    print(f"  qatlamlar={cfg.num_hidden_layers}  hidden={cfg.hidden_size}  "
          f"FFN={cfg.intermediate_size}")
    print(f"Kesh: L{g.level} = {g.size_bytes/1024**2:.0f} MiB, "
          f"byudjet alpha*L3 = {budget/1024**2:.1f} MiB\n")

    shapes = {
        "q/k/v/o_proj": cfg.hidden_size * cfg.hidden_size,
        "gate/up_proj": cfg.hidden_size * cfg.intermediate_size,
        "down_proj": cfg.intermediate_size * cfg.hidden_size,
        "lm_head": cfg.vocab_size * cfg.hidden_size,
    }
    print(f"{'operator':16s} {'parametr':>12s} {'FP32(MiB)':>10s} {'INT8(MiB)':>10s} "
          f"{'INT8 talab':>11s} {'holat':>8s}")
    rows = {}
    for name, nparams in shapes.items():
        fp32, int8 = nparams * 4 / 1024**2, nparams / 1024**2
        need = int8 * 1024**2 / budget
        case = "3-holat" if need > 1 else ("2-holat" if fp32 * 1024**2 > budget else "1-holat")
        rows[name] = {"params": nparams, "int8_mib": int8, "need": need, "case": case}
        print(f"{name:16s} {nparams:12,d} {fp32:10.1f} {int8:10.1f} {need:10.2f}x {case:>8s}")
    return rows


def capture_intermediate(model, tok, texts, layer_idx_list):
    """Grab down_proj's input (the gated FFN intermediate) per probed layer."""
    store = {li: [] for li in layer_idx_list}
    handles = []

    def make_hook(li):
        def hook(module, inputs, output):
            store[li].append(inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1]))
        return hook

    for li in layer_idx_list:
        handles.append(model.model.layers[li].mlp.down_proj.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        for i in range(0, len(texts), 4):
            chunk = [t for t in texts[i:i + 4] if t and t.strip()]
            if not chunk:
                continue
            enc = tok(chunk, return_tensors="pt", padding="max_length",
                      max_length=SEQ_LEN, truncation=True)
            model(**enc)
            if (i // 4 + 1) % 5 == 0:
                print(f"    {i+4}/{len(texts)} matn", flush=True)
    for hnd in handles:
        hnd.remove()

    rng = np.random.default_rng(0)
    out = {}
    for li, chunks in store.items():
        x = torch.cat(chunks, dim=0).numpy()
        if x.shape[0] > MAX_ROWS:
            x = x[rng.choice(x.shape[0], MAX_ROWS, replace=False)]
        out[li] = x.astype(np.float64)
    return out


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"yuklanmoqda: {MODEL_ID} (bir marta, keshga)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    print("\n" + "=" * 84)
    print("1) APPARAT SAVOLI: kaskad qaysi operatorni muammoli deb topadi?")
    print("=" * 84)
    cache_rows = cache_report(model)

    print("\n" + "=" * 84)
    print("2) MODEL SAVOLI: FFN oraliq kanallarida funksional ortiqchalik bormi?")
    print("=" * 84)
    n_layers = model.config.num_hidden_layers
    probe = list(range(0, n_layers, LAYER_STRIDE))
    print(f"tekshiriladigan qatlamlar: {probe}")

    _, texts = load_audio(0, N_CALIB_TEXT)
    print("faollashuvlar yig'ilmoqda (PyTorch hooks)...", flush=True)
    x_by = capture_intermediate(model, tok, texts, probe)

    results = {}
    print(f"\n{'qatlam':>8s} {'tau':>6s} {'olib tashlanadi':>16s} {'ulush':>8s} {'E_loc':>9s}")
    for li in probe:
        x = x_by[li]
        w = model.model.layers[li].mlp.down_proj.weight.detach().numpy().astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        n = w.shape[1]
        split = int(x.shape[0] * 0.75)
        x_fit, x_eval = x[:split], x[split:]
        y_ref = x_eval @ w.T
        entry = {}
        for tau in TAUS:
            g = greedy_group(x_fit.T, np.linalg.norm(w, axis=0),
                             float(np.linalg.norm(x_fit @ w.T)),
                             tau=tau, eps_threshold=EPS_THRESHOLD)
            w_comp = build_compensated_weight(w, g)
            e = float(np.linalg.norm(y_ref - x_eval @ w_comp.T) / np.linalg.norm(y_ref))
            removed = n - len(g.groups)
            entry[str(tau)] = {"removed": int(removed), "fraction": removed / n, "e_loc": e}
            print(f"{('L'+str(li)):>8s} {tau:6.2f} {removed:16d} {removed/n:7.1%} {e:9.4f}",
                  flush=True)
        results[f"layer_{li}"] = entry

    json.dump({"model": MODEL_ID, "cache": cache_rows, "redundancy": results},
              open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")

    print("\n" + "=" * 84)
    print("XULOSA")
    print("=" * 84)
    print(f"{'tau':>6s} {'o''rtacha ulush':>16s} {'maks ulush':>12s} {'o''rtacha E_loc':>16s}")
    for tau in TAUS:
        fr = [results[k][str(tau)]["fraction"] for k in results]
        el = [results[k][str(tau)]["e_loc"] for k in results]
        print(f"{tau:6.2f} {np.mean(fr):15.1%} {np.max(fr):11.1%} {np.mean(el):16.4f}")
    print("\nTaqqoslash: Whisper encoder tau=0.99 -> o'rtacha 17.1%, maks 58.0%")
    print("            mBERT       tau=0.99 -> o'rtacha  0.1%, maks  0.7%")
    print("\nEslatma: Llama FFN gated (h = SiLU(gate(x)) * up(x)), ya'ni bitta oraliq")
    print("kanalni olib tashlash UCHTA matritsani qisqartiradi (gate, up, down).")


if __name__ == "__main__":
    main()
