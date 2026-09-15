"""Would centring the criterion find redundancy that cosine cannot?

A proposal: replace cos(h_j, h_p) >= tau with a residual ratio

    R_jp     = ||h_j - gamma_jp h_p|| / (||h_j|| + xi)
    R^aff_jp = ||h_j - gamma_jp h_p - c_j|| / (||h_j - mu_j|| + xi)

and threshold R <= eps_R. The two halves are not equally novel, and the
algebra says which is which.

With the optimal gamma, ||h_j - gamma h_p||^2 = ||h_j||^2 (1 - cos^2), so

    R_jp = sqrt(1 - cos^2) = |sin theta|

i.e. the FIRST form is a reparametrisation of the cosine threshold, not a new
criterion -- and it is already available in greedy_group as metric="sin_theta".

The affine form is genuinely different. Fitting gamma AND an intercept is a
regression of h_j on h_p with a constant, whose residual is
||h_j - mu_j||^2 (1 - corr^2), so

    R^aff_jp = sqrt(1 - corr^2)

which thresholds CORRELATION rather than cosine. Cosine on uncentred vectors
is dominated by the shared mean component; correlation removes it. Whether
that changes anything here is an empirical question with one input: how large
the per-channel means actually are. This measures that, and then measures the
correlation distribution directly against the cosine distribution.

The cost side is already known. The per-channel constant c_j has to be stored
and applied, which is the bias vector this architecture does not carry -- the
333 KiB priced in Sec 4.14. So the proposal is worth adopting only if it
finds materially more redundancy, not merely a little.
"""

import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wikitext2_int4 import MODEL_DIR, load_segments

LAYER = int(os.environ.get("LAYER", "8"))
ROWS = 2048
TAUS = (0.99, 0.95, 0.90, 0.70, 0.50)
THREADS = 4


def capture(model, seg, layer_idx):
    store = []
    h = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(
        lambda m, i, o: store.append(i[0].detach().to(torch.float32)
                                     .reshape(-1, i[0].shape[-1])))
    with torch.no_grad():
        model(input_ids=seg)
    h.remove()
    return torch.cat(store, 0).numpy().astype(np.float64)[:ROWS]


def nearest(mat, signed=True):
    """Each channel's best partner, self-pairs excluded.

    `signed=False` takes the magnitude, which is the case the current
    criterion cannot see. greedy_group gates on `cos >= tau`, so a pair with
    cos = -0.95 -- as mergeable as +0.95, since gamma may be negative and
    build_compensated_weight already applies a signed gamma -- is refused.
    The earlier sanity check reported a signed maximum and would therefore
    have shown nothing even if such pairs were everywhere.
    """
    np.fill_diagonal(mat, 0.0 if not signed else -2.0)
    return (np.abs(mat) if not signed else mat).max(axis=1)


def main():
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _, calib = load_segments(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    x = capture(model, calib[0:1], LAYER)
    del model

    h = x.T                                     # (channels, rows)
    n = h.shape[0]
    print(f"L{LAYER}: {n} kanal, {h.shape[1]} qator\n")

    # How much of each channel IS its mean? If this is small the centring
    # cannot change the geometry, whatever the criterion is called.
    mu = h.mean(axis=1)
    norms = np.linalg.norm(h, axis=1)
    centred = h - mu[:, None]
    cnorms = np.linalg.norm(centred, axis=1)
    share = (np.abs(mu) * np.sqrt(h.shape[1])) / (norms + 1e-30)
    print("Kanal o'rtachasining ulushi  ||mu_j|| sqrt(T) / ||h_j||:")
    for q in (50, 90, 99, 100):
        print(f"  {q:5.1f}-protsentil : {np.percentile(share, q):.4f}")
    print(f"  markazlashtirish normani o'rtacha "
          f"{(1 - (cnorms / (norms + 1e-30)).mean())*100:.2f}% kamaytiradi\n")

    hn = (h / (norms[:, None] + 1e-30)).astype(np.float32)
    cn = (centred / (cnorms[:, None] + 1e-30)).astype(np.float32)

    print("kosinus va korrelyatsiya matritsalari hisoblanmoqda...")
    gcos, gcor = hn @ hn.T, cn @ cn.T
    variants = [
        ("kosinus (joriy, ishorali)", nearest(gcos.copy(), signed=True)),
        ("|kosinus| (ishorasiz)", nearest(gcos, signed=False)),
        ("korrelyatsiya (affin, ishorali)", nearest(gcor.copy(), signed=True)),
        ("|korrelyatsiya| (affin, ishorasiz)", nearest(gcor, signed=False)),
    ]

    print(f"\n{'variant':36s}" + "".join(f"{q:>9}%" for q in
                                         (50, 90, 99, 100)))
    for name, best in variants:
        print(f"{name:36s}" + "".join(
            f"{np.percentile(best, q):10.4f}" for q in (50, 90, 99, 100)))

    print(f"\nchegaradan yuqori kanallar ulushi:")
    print(f"{'variant':36s}" + "".join(f"{t:>10.2f}" for t in TAUS))
    for name, best in variants:
        print(f"{name:36s}" + "".join(
            f"{(best >= t).mean()*100:9.2f}%" for t in TAUS))

    print("\nIzoh: '|kosinus|' qatori 'kosinus' dan sezilarli yuqori bo'lsa, "
          "joriy mezon anti-kollinear juftliklarni behuda rad etayotgan "
          "bo'ladi va darvozani |cos| ga o'zgartirish tekin yaxshilanish. "
          "'korrelyatsiya' qatori yuqori bo'lsa, affin variant asoslanadi, "
          "lekin c_j ni saqlash xarajati bilan.")


if __name__ == "__main__":
    main()
