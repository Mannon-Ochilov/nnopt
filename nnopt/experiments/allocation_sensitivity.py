"""Does the cache care WHERE the bytes come out, or only how many?

The framework picks configurations by miss bytes, and its ladder removes the
same fraction from every layer. That uniformity was inherited from the hard
constraint, where it is forced: a per-layer cache target binds on each layer
separately, so every layer has to fit. Under the soft objective nothing forces
it, and whether it is right became an open question the moment the target
stopped being a gate.

The arithmetic settles the main case. With every layer over budget,

    Miss = sum_l [ b_l + (b_l - B)(R - 1) ]
         = R * sum_l b_l  -  (R - 1) * L * B ,

which depends only on the TOTAL. Allocation is then free as far as the cache
is concerned, and should be decided entirely by accuracy -- that is, by tau,
which distributes removal according to where redundancy actually is.

The preference reappears only when a layer can be pushed BELOW the budget,
because it then escapes the (R - 1) multiplier. Whether that is reachable
depends on how much of the layer is attention, and it is exactly the regime
boundary this script locates.

This matters for a design decision, not just for a table: if allocation is
free at the L3 values we care about, the planner's rungs should be indexed by
tau rather than by a uniform keep ratio, and the ladder has to be rewritten.
"""

import numpy as np

from nnopt.cascade.cache_planner import MIB, PartSpec, miss_bytes, Treatment

ALPHA = 0.7
L3_VALUES = (2.0, 6.0, 12.0, 24.0, 32.0, 96.0)
# Whisper-medium encoder: 24 layers of 48 MiB, two thirds of it FFN.
ENC = PartSpec("enkoder", int(48 * MIB), 24, int(32 * MIB), reuse=1500)
INT8_SCALE = 0.25


def layer_bytes(fixed, ffn, keep):
    return (fixed + ffn * keep) * INT8_SCALE


def miss_of(alloc, budget, reuse, fixed, ffn):
    """Miss for an explicit per-layer keep vector."""
    total = 0.0
    for keep in alloc:
        b = layer_bytes(fixed, ffn, keep)
        total += b + max(0.0, b - budget) * (reuse - 1)
    return total


def main():
    fixed = ENC.per_layer_bytes - ENC.prunable_bytes
    ffn = ENC.prunable_bytes
    n = ENC.n_layers
    mean_keep = 0.55                      # the 45% removal L3 = 12 MiB implies

    print(f"Enkoder: {n} qatlam x {ENC.per_layer_bytes/MIB:.0f} MiB "
          f"(FFN {ffn/MIB:.0f}, attn {fixed/MIB:.0f}), INT8 dan keyin "
          f"{ENC.per_layer_bytes*INT8_SCALE/MIB:.1f} MiB/qatlam")
    print(f"Umumiy byudjet bir xil: o'rtacha keep = {mean_keep}\n")

    # Three allocations with IDENTICAL total parameters, differing only in
    # how the removal is spread: flat, tau-like (early layers give more), and
    # an extreme that empties half the layers to push them under budget.
    flat = np.full(n, mean_keep)
    ramp = np.clip(np.linspace(0.15, 0.95, n), 0.05, 1.0)
    ramp = ramp * (mean_keep * n / ramp.sum())
    extreme = np.array([0.10] * (n // 2) + [1.0] * (n - n // 2))
    extreme = extreme * (mean_keep * n / extreme.sum())
    allocs = [("bir xil (nisbat-oilasi)", flat),
              ("nishab (tau-oilasiga o'xshash)", ramp),
              ("ekstremal (yarmi bo'shatiladi)", extreme)]
    for _, a in allocs:
        assert abs(a.mean() - mean_keep) < 1e-9, "byudjet teng bo'lishi shart"

    print(f"{'L3':>5s} {'byudjet':>8s} {'qatlam':>8s}  " +
          "  ".join(f"{lbl[:22]:>22s}" for lbl, _ in allocs) + "   taqsimot muhimmi")
    print("-" * 104)
    for l3 in L3_VALUES:
        budget = ALPHA * l3 * MIB
        misses = [miss_of(a, budget, ENC.reuse, fixed, ffn) for _, a in allocs]
        spread = (max(misses) - min(misses)) / max(misses) * 100
        b_flat = layer_bytes(fixed, ffn, mean_keep)
        verdict = "yo'q (chiziqli)" if spread < 0.5 else f"HA, {spread:.1f}%"
        print(f"{l3:5.0f} {budget/MIB:7.1f}M {b_flat/MIB:7.1f}M  " +
              "  ".join(f"{m/MIB:21.0f}M" for m in misses) + f"   {verdict}")

    # The regime boundary: below which L3 can no layer be pushed under budget,
    # and above which does the flat allocation already fit?
    min_layer = fixed * INT8_SCALE                    # FFN entirely removed
    flat_layer = layer_bytes(fixed, ffn, mean_keep)
    print(f"\nRejim chegaralari (alpha = {ALPHA}):")
    print(f"  FFN butunlay olib tashlansa ham qatlam {min_layer/MIB:.1f} MiB "
          f"dan pastga tushmaydi")
    print(f"  -> L3 < {min_layer/(ALPHA*MIB):5.1f} MiB da hech bir qatlam "
          f"byudjetga sig'maydi: miss faqat UMUMIY hajmga bog'liq,")
    print(f"     taqsimotni butunlay aniqlik hal qilishi kerak (ya'ni tau).")
    print(f"  -> L3 > {flat_layer/(ALPHA*MIB):5.1f} MiB da bir xil taqsimot "
          f"allaqachon sig'adi, qo'shimcha qisqartirish miss bermaydi.")
    print(f"  -> Oraliqda taqsimot muhim: qatlamni byudjetdan pastga "
          f"itarish (R-1) ko'paytuvchidan qutqaradi.")

    # Sanity: the closed form should reproduce the measured numbers exactly in
    # the all-over-budget regime, or the argument above is wrong.
    l3 = 6.0
    budget = ALPHA * l3 * MIB
    closed = ENC.reuse * sum(layer_bytes(fixed, ffn, k) for k in flat) \
        - (ENC.reuse - 1) * n * budget
    direct = miss_of(flat, budget, ENC.reuse, fixed, ffn)
    print(f"\nYopiq shakl tekshiruvi (L3 = {l3:.0f} MiB): "
          f"{closed/MIB:.0f} MiB va {direct/MIB:.0f} MiB — "
          f"{'mos' if abs(closed-direct) < 1 else 'MOS EMAS'}")


if __name__ == "__main__":
    main()
