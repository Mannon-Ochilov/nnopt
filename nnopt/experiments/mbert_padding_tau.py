"""The same padding defect on mBERT, and what a global tau gives there.

mbert_analysis.py tokenizes with padding="max_length" to 128 tokens and then
captures activations with active_mask=None -- the identical mistake found in
the Whisper path. The attention mask is already sitting in the feed dict two
lines above the capture call; it simply was not passed on. The Common Voice
transcripts average 18.6 tokens, so 85.5% of the positions entering h_j are
[PAD] -- more than Whisper's 83%.

Padding responses are near-constant across positions, so two channels can
look collinear because they agree about padding rather than about text. On
Whisper that inflated the removable share by up to 6.45x in a deep layer.
This asks whether mBERT is distorted the same way, and by how much.

The second question is the one the framework needs: under ONE global tau,
what does each layer actually give? Whisper's answer was a sharp depth
profile -- 42-61% in L0-L5, 0-1.3% from L12 on -- which is why a uniform
per-layer removal budget is the wrong instrument. mBERT is a different
object (12 layers, 3072 FFN channels, text rather than speech), so whether
that profile is a property of the method's target or a quirk of the audio
encoder is worth knowing.

Both configurations run at identical taus on identical text, so every
difference is the mask and nothing else.

Usage:  python experiments/mbert_padding_tau.py
"""

import gc
import json

import numpy as np
import onnx
from onnx import numpy_helper
from transformers import AutoTokenizer

from calib_utils import load_audio
from mbert_analysis import (
    MBERT_DIR,
    MBERT_ONNX,
    SEQ_LEN,
    build_text_batches,
)
from nnopt.calibrator.activation_capture import (
    ActivationCapture,
    build_response_vectors,
)
from nnopt.grouping.functional_grouping import greedy_group
from nnopt.profiler.graph_profiler import profile_onnx_model

FREE_DIMS = {"batch": 8, "seq": SEQ_LEN}
TAUS = (0.99, 0.95, 0.90, 0.85, 0.80, 0.75)
EPS_THRESHOLD = 0.5
N_TEXT = 400
MAX_ROWS = 16384
OUT_JSON = "experiments/results_mbert_padding_tau.json"


def capture(tensor_names, batches, masked, seed=0):
    """Activations per tensor as (rows, channels).

    `masked` selects whether the attention mask is honoured. BERT pads on the
    right, so the mask is exactly the (batch, seq) boolean grid that
    build_response_vectors expects -- no reconstruction from lengths needed.
    """
    rng = np.random.default_rng(seed)
    cap = ActivationCapture(MBERT_ONNX, tensor_names=list(tensor_names))
    collected = {nm: [] for nm in tensor_names}
    for i, feed in enumerate(batches, 1):
        mask = feed["attention_mask"].astype(bool) if masked else None
        for nm, arr in cap.run_batch(feed).items():
            collected[nm].append(build_response_vectors(arr, active_mask=mask))
        if i % 10 == 0:
            print(f"    batch {i}/{len(batches)}", flush=True)
    out = {}
    for nm, chunks in collected.items():
        x = np.concatenate(chunks, axis=1).T
        if x.shape[0] > MAX_ROWS:
            x = x[rng.choice(x.shape[0], MAX_ROWS, replace=False)]
        out[nm] = x.astype(np.float64)
    return out


def layer_index(name):
    import re
    m = re.search(r"layer\.(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    tok = AutoTokenizer.from_pretrained(MBERT_DIR)
    _, texts = load_audio(0, N_TEXT)
    batches = build_text_batches(tok, texts)
    real = int(sum(b["attention_mask"].sum() for b in batches))
    total = int(sum(b["attention_mask"].size for b in batches))
    print(f"{len(batches)} batch x 8 x {SEQ_LEN} = {total} pozitsiya, "
          f"real {real} ({real / total * 100:.1f}%)\n", flush=True)

    m = onnx.load(MBERT_ONNX, load_external_data=False)
    dims = {i.name: tuple(i.dims) for i in m.graph.initializer}
    profs = profile_onnx_model(MBERT_ONNX, free_dims=FREE_DIMS)
    ops = [p for p in profs if p.weight_initializer]
    fc2 = [p for p in ops
           if (s := dims[p.weight_initializer]) and len(s) == 2
           and max(s) >= 3072 and min(s) <= 768 and s[0] > s[1]]
    fc2.sort(key=lambda p: layer_index(p.name))
    print(f"{len(fc2)} ta FFN chiqish operatori\n", flush=True)

    full = onnx.load(MBERT_ONNX)
    inits = {i.name: i for i in full.graph.initializer}
    names = sorted({p.activation_input for p in fc2})

    rows = []
    for label, masked in (("niqobsiz", False), ("niqobli", True)):
        print(f"=== {label} ===", flush=True)
        x_by = capture(names, batches, masked)
        for li, p in enumerate(fc2):
            x = x_by[p.activation_input]
            w2s = numpy_helper.to_array(inits[p.weight_initializer]) \
                .astype(np.float64)
            w2 = w2s if w2s.shape[1] == x.shape[1] else w2s.T
            h = x.T
            wn = np.linalg.norm(w2, axis=0)
            y_norm = float(np.linalg.norm(x @ w2.T))
            line = f"  L{li:<2d}"
            for tau in TAUS:
                g = greedy_group(h, wn, y_norm, tau=tau,
                                 eps_threshold=EPS_THRESHOLD)
                merged = h.shape[0] - len(g.groups)
                rows.append({"config": label, "layer": li, "tau": tau,
                             "channels": int(h.shape[0]),
                             "merged": int(merged),
                             "share": merged / h.shape[0],
                             "rows": int(x.shape[0])})
                line += f"  t{tau}: {merged / h.shape[0] * 100:5.2f}%"
            print(line, flush=True)
            del x, w2, h
            gc.collect()
        del x_by
        gc.collect()
        json.dump(rows, open(OUT_JSON, "w"), indent=2)
    report(rows, len(fc2))


def report(rows, n_layers):
    def get(cfg, li, tau):
        r = next((r for r in rows if r["config"] == cfg and r["layer"] == li
                  and r["tau"] == tau), None)
        return r["share"] * 100 if r else None

    print("\n" + "=" * 78)
    print("mBERT — QATLAMLAR BO'YICHA OLIB TASHLASH, % (niqobli)")
    print("=" * 78)
    print(f"{'Qatlam':>7s}" + "".join(f"{'t=' + str(t):>11s}" for t in TAUS))
    for li in range(n_layers):
        print(f"{'L' + str(li):>7s}" + "".join(
            f"{v:10.2f}%" if (v := get('niqobli', li, tau)) is not None
            else " " * 11 for tau in TAUS))
    print("-" * 78)
    for tau in TAUS:
        pass
    line = f"{'GLOBAL':>7s}"
    for tau in TAUS:
        sel = [r for r in rows if r["config"] == "niqobli" and r["tau"] == tau]
        line += (f"{sum(r['merged'] for r in sel) / sum(r['channels'] for r in sel) * 100:10.2f}%"
                 if sel else " " * 11)
    print(line)

    print("\n" + "=" * 78)
    print("PADDING TA'SIRI (niqobsiz -> niqobli, global %)")
    print("=" * 78)
    print(f"{'tau':>6s} {'niqobsiz':>11s} {'niqobli':>11s} {'nisbat':>9s}")
    for tau in TAUS:
        a = [r for r in rows if r["config"] == "niqobsiz" and r["tau"] == tau]
        b = [r for r in rows if r["config"] == "niqobli" and r["tau"] == tau]
        if not a or not b:
            continue
        ga = sum(r["merged"] for r in a) / sum(r["channels"] for r in a) * 100
        gb = sum(r["merged"] for r in b) / sum(r["channels"] for r in b) * 100
        print(f"{tau:6.2f} {ga:10.2f}% {gb:10.2f}% "
              f"{ga / gb if gb else float('nan'):8.2f}x")
    print(f"\nsaqlandi: {OUT_JSON}")


if __name__ == "__main__":
    main()
