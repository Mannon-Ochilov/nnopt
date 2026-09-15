"""Does the method transfer to mBERT?

Two questions, deliberately separated because they have different natures:

  1. HARDWARE question -- what does the cache-anchored cascade decide for
     mBERT on this machine? This depends on the model's size relative to
     alpha*L3 and is answered by arithmetic.

  2. MODEL question -- is there functional redundancy in mBERT's FFN
     intermediate channels? This is independent of any cache and is the
     core claim of the method. Whisper's encoder showed a sharp profile
     (58% removable at L2-L3, ~0% from L15), so the interesting result is
     whether a text encoder behaves the same way or not.

Either answer is useful. If mBERT is also redundant the method transfers
and a second object exists for the Q1 article; if it is not, that is a
structural statement about what kind of model over-parameterizes its FFN.

Calibration/evaluation text is the Uzbek Common Voice transcripts already
cached for the ASR work, so the language domain is held constant.

Quality metric: masked-token prediction accuracy, the task-level analogue
of WER for a masked LM.
"""

import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from transformers import AutoTokenizer

from calib_utils import load_audio
from nnopt.grouping.functional_grouping import build_compensated_weight, greedy_group
from nnopt.hw.cache_topology import detect_cache_topology
from nnopt.profiler.graph_profiler import profile_onnx_model

MBERT_DIR = "models/mbert_onnx"
MBERT_ONNX = f"{MBERT_DIR}/model.onnx"
SEQ_LEN = 128
N_CALIB_TEXT = 400
ALPHA = 0.7
TAUS = [0.99, 0.95, 0.90]
EPS_THRESHOLD = 0.5
MAX_ROWS = 3072
OUT_JSON = "experiments/results_mbert.json"
FREE_DIMS = {"batch": 8, "seq": SEQ_LEN}


def cache_report():
    topo = detect_cache_topology()
    g = topo.global_shared_cache()
    budget = ALPHA * g.size_bytes
    print(f"Kafolatlangan umumiy kesh: L{g.level}, {g.size_bytes/1024**2:.0f} MiB")
    print(f"Samarali byudjet (alpha={ALPHA}): {budget/1024**2:.1f} MiB\n")

    m = onnx.load(MBERT_ONNX, load_external_data=False)
    dims = {i.name: tuple(i.dims) for i in m.graph.initializer}
    profs = profile_onnx_model(MBERT_ONNX, free_dims=FREE_DIMS)
    ops = [p for p in profs if p.weight_initializer]
    sizes = [(p.name, int(np.prod(dims[p.weight_initializer])) * 4) for p in ops]
    total = sum(s for _, s in sizes)
    biggest = max(s for _, s in sizes)

    print(f"{len(ops)} ta vaznli operator, jami {total/1024**2:.0f} MiB (fp32)")
    print(f"{'granulyarlik':16s} {'hajm(MiB)':>10s} {'FP32 talab':>12s} {'INT8 talab':>12s}")
    for label, nbytes in (("per-operator", biggest), ("whole-model", total)):
        print(f"{label:16s} {nbytes/1024**2:10.1f} {max(1, nbytes/budget):11.2f}x "
              f"{max(1, nbytes/4/budget):11.2f}x")
    return budget, ops, dims


def build_text_batches(tok, texts, batch=8):
    batches = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        if len(chunk) < batch:
            break
        enc = tok(chunk, return_tensors="np", padding="max_length",
                  max_length=SEQ_LEN, truncation=True)
        batches.append({
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc["token_type_ids"].astype(np.int64),
        })
    return batches


def capture_ffn_activations(tensor_names, batches, max_rows=MAX_ROWS, seed=0,
                            masked=True):
    """FFN activations as (rows, channels), one entry per tensor.

    `masked` honours the attention mask, which is what build_response_vectors
    requires for padded input and what this function did NOT do originally.
    build_text_batches pads to max_length=128 while the Uzbek transcripts
    average 18.6 tokens, so 85.5% of the positions were [PAD]; padding
    responses are near-constant and inflate apparent collinearity between
    channels. The flag exists only so the unmasked behaviour stays available
    for the comparison that documents the difference.
    """
    from nnopt.calibrator.activation_capture import ActivationCapture, build_response_vectors
    rng = np.random.default_rng(seed)
    cap = ActivationCapture(MBERT_ONNX, tensor_names=list(tensor_names))
    collected = {nm: [] for nm in tensor_names}
    for i, feed in enumerate(batches, 1):
        am = feed["attention_mask"].astype(bool) if masked else None
        for nm, arr in cap.run_batch(feed).items():
            collected[nm].append(build_response_vectors(arr, active_mask=am))
        if i % 5 == 0:
            print(f"    batch {i}/{len(batches)}", flush=True)
    out = {}
    for nm, chunks in collected.items():
        x = np.concatenate(chunks, axis=1).T
        if x.shape[0] > max_rows:
            x = x[rng.choice(x.shape[0], max_rows, replace=False)]
        out[nm] = x.astype(np.float64)
    return out


def main():
    if not os.path.exists(MBERT_ONNX):
        print(f"{MBERT_ONNX} topilmadi — avval mbert_prepare.py ni ishga tushiring")
        return

    print("=" * 78)
    print("1) APPARAT SAVOLI: kaskad mBERT uchun nima qaror qiladi?")
    print("=" * 78)
    budget, ops, dims = cache_report()

    print("\n" + "=" * 78)
    print("2) MODEL SAVOLI: mBERT FFN kanallarida funksional ortiqchalik bormi?")
    print("=" * 78)

    # mBERT FFN: intermediate.dense (768->3072) then output.dense (3072->768).
    # The operator consuming the intermediate is the one whose input width is
    # the FFN width -- find it by shape rather than by name pattern.
    fc2_like = []
    for p in ops:
        shp = dims[p.weight_initializer]
        if len(shp) == 2 and max(shp) >= 3072 and min(shp) <= 768 and shp[0] > shp[1]:
            fc2_like.append(p)
    print(f"FFN chiqish operatorlari topildi: {len(fc2_like)}")
    if not fc2_like:
        print("shakl bo'yicha topilmadi — barcha vaznli operatorlar shakllari:")
        for p in ops[:12]:
            print(f"  {p.name[:60]:60s} {dims[p.weight_initializer]}")
        return

    _, texts = load_audio(0, N_CALIB_TEXT)
    tok = AutoTokenizer.from_pretrained(MBERT_DIR)
    batches = build_text_batches(tok, texts)
    print(f"kalibrlash: {len(batches)} batch x 8 x {SEQ_LEN} token")

    tensors = sorted({p.activation_input for p in fc2_like})
    print("faollashuvlar yig'ilmoqda...", flush=True)
    x_by = capture_ffn_activations(tensors, batches)

    full = onnx.load(MBERT_ONNX)
    inits = {i.name: i for i in full.graph.initializer}

    results = {}
    print(f"\n{'operator':44s} {'tau':>6s} {'olib tashlanadi':>16s} {'ulush':>7s} {'E_loc':>9s}")
    for p in fc2_like:
        x = x_by.get(p.activation_input)
        if x is None:
            continue
        w = numpy_helper.to_array(inits[p.weight_initializer]).astype(np.float64)
        if w.shape[1] != x.shape[1]:
            w = w.T
        if w.shape[1] != x.shape[1]:
            continue
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
            entry[str(tau)] = {"groups": len(g.groups), "removed": int(removed),
                               "fraction": removed / n, "e_loc": e}
            print(f"{p.name[:44]:44s} {tau:6.2f} {removed:16d} {removed/n:6.1%} {e:9.4f}",
                  flush=True)
        results[p.name] = entry

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")

    print("\n" + "=" * 78)
    print("XULOSA: mBERT FFN ortiqchaligi")
    print("=" * 78)
    print(f"{'tau':>6s} {'o''rtacha ulush':>16s} {'o''rtacha E_loc':>16s}")
    for tau in TAUS:
        fr = [results[k][str(tau)]["fraction"] for k in results]
        el = [results[k][str(tau)]["e_loc"] for k in results]
        print(f"{tau:6.2f} {np.mean(fr):15.1%} {np.mean(el):16.4f}")
    print("\nTaqqoslash uchun Whisper encoder (tau=0.99): o'rtacha 17.1%, "
          "eng yuqori qatlamda 58.0%")


if __name__ == "__main__":
    main()
