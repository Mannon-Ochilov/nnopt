"""Figures drawn from the measurements, not described for an image model.

The manuscript carried nine figures as prompts for image generation. For the
four schematics that is a reasonable way to commission a drawing. For the data
figures it is not: an image model asked for "a scatter plot of local versus
global error" will produce plausible points, and those points would be
invented. A paper cannot carry invented data, so every figure that shows
numbers is built here from the result files those numbers came from.

All five data figures are produced. The INT4 panel needed one thing added
first: the perplexities were on disk, but the operator-level gains that
explain them were only ever printed, so int4_bias_and_fix.py now writes them
out. Nothing here is transcribed from the paper's tables -- a second copy of a
number is free to drift from the first.

Style is deliberately plain: one accent colour, everything else greyscale, no
chart junk, and every axis labelled in the units the text uses.
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = "figures"
ACCENT = "#1a5fb4"
GREY = "#606060"
LIGHT = "#b0b0b0"
DPI = 200

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load(name):
    path = f"experiments/results_{name}.json"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return json.load(open(path, encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None


def save(fig, num):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/fig{num}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")
    return path


def layer_of(name):
    m = re.search(r"layers?[./](\d+)", name)
    return int(m.group(1)) if m else -1


def fig4_redundancy_profile():
    """Where redundancy lives, per layer, in three architectures."""
    whisper, llama, mbert = load("ffn_redundancy"), load("llama"), load("mbert")
    if not (whisper and llama and mbert):
        return None

    def whisper_series(tau):
        pts = [(layer_of(k), v["taus"][tau]["fraction"] * 100)
               for k, v in whisper.items() if tau in v.get("taus", {})]
        return zip(*sorted(p for p in pts if p[0] >= 0))

    def llama_series(tau):
        pts = [(int(k.split("_")[1]), v[tau]["fraction"] * 100)
               for k, v in llama.get("redundancy", {}).items() if tau in v]
        return zip(*sorted(pts))

    def mbert_series(tau):
        pts = [(layer_of(k), v[tau]["fraction"] * 100)
               for k, v in mbert.items()
               if isinstance(v, dict) and tau in v]
        return zip(*sorted(p for p in pts if p[0] >= 0))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, tau in zip(axes, ("0.99", "0.95")):
        for series, label, colour, marker in (
                (whisper_series, "Whisper enkoder", ACCENT, "o"),
                (llama_series, "open_llama_3b", GREY, "s"),
                (mbert_series, "mBERT", LIGHT, "^")):
            try:
                x, y = series(tau)
            except ValueError:
                continue
            ax.plot(x, y, marker=marker, ms=3, lw=1.0, color=colour,
                    label=label)
        ax.set_title(f"tau = {tau}", fontsize=8)
        ax.set_xlabel("qatlam indeksi")
        ax.grid(axis="y", lw=0.4, color="#e8e8e8")
    axes[0].set_ylabel("olib tashlanadigan kanallar, %")
    axes[0].legend(fontsize=7, loc="upper right")
    return save(fig, 4)


def fig5_error_absorption():
    """Operator error against network error, on the same operators."""
    d = load("influence")
    if not d:
        return None
    keys = [k for k in d["e_loc"] if k in d["e_glob"]]
    loc = np.array([d["e_loc"][k] for k in keys])
    glob = np.array([d["e_glob"][k] for k in keys])
    keep = (loc > 0) & (glob > 0)
    loc, glob = loc[keep], glob[keep]

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.scatter(loc, glob, s=14, color=ACCENT, alpha=0.75, edgecolors="none")
    lo = min(loc.min(), glob.min()) * 0.7
    hi = max(loc.max(), glob.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], lw=0.8, ls="--", color=LIGHT,
            label="y = x (yutilishsiz)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("operator xatosi $E_{loc}$")
    ax.set_ylabel("tarmoq chiqish xatosi $E_{glob}$")
    ax.grid(lw=0.4, color="#eeeeee")
    span_loc = loc.max() / loc.min()
    span_glob = glob.max() / glob.min()
    ax.set_title(f"$E_{{loc}}$ {span_loc:.0f}x, $E_{{glob}}$ "
                 f"{span_glob:.0f}x oralig'ida", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    return save(fig, 5)


def fig7_whole_model():
    """The cascade's stop point against the uniform policies."""
    rows = load("whole_model_cascade")
    if not rows:
        return None
    base = next((r for r in rows if r["variant"].startswith("A")), None)
    if not base:
        return None

    # The variant strings are long enough to collide and to run off the axis,
    # so each is reduced to the letter the text uses plus a two-word gloss.
    short = {"A": "FP32", "B": "bir xil INT8", "C": "kaskad",
             "D": "bir xil agressiv"}
    pts = []
    for r in rows:
        letter = r["variant"].split(":", 1)[0].strip()
        pts.append((base["mib"] / r["mib"], r["wer"], r.get("wer_lo"),
                    r.get("wer_hi"), short.get(letter, letter)))
    pts.sort()

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    err = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
    ax.errorbar(x, y, yerr=err, fmt="o", ms=4.5, lw=0.8, capsize=2,
                color=GREY, ecolor=LIGHT, zorder=3)
    worst = max(pts, key=lambda p: p[1])
    ax.plot([worst[0]], [worst[1]], "o", ms=7, color="#a51d2d", zorder=4)
    ax.axhline(base["wer"], lw=0.7, ls="--", color=LIGHT, zorder=1)

    # Alternate the label side so neighbouring points cannot overlap, and
    # anchor the rightmost one to its left so it stays inside the axes.
    for i, (cx, cy, _, hi, lab) in enumerate(pts):
        rightmost = cx == max(x)
        ax.annotate(lab, (cx, cy),
                    textcoords="offset points",
                    xytext=(-6 if rightmost else 6, 9 if i % 2 else -14),
                    ha="right" if rightmost else "left",
                    fontsize=7, color=GREY)
    ax.set_xlim(0.6, max(x) + 0.9)
    ax.set_ylim(0.10, max(p[3] for p in pts) + 0.06)
    ax.set_xlabel("siqish koeffitsiyenti (FP32 ga nisbatan)")
    ax.set_ylabel("WER (300 TEST namunasi)")
    ax.grid(lw=0.4, color="#eeeeee")
    return save(fig, 7)


def fig8_operating_points():
    """Every measured encoder configuration, size against quality."""
    lib = load("config_library")
    if not lib:
        return None
    # "blind" covers two very different things -- quantizing everything and
    # picking a pruning ratio -- and the first is a perfectly good
    # configuration while the second destroys the model. Drawing them in one
    # series would put a 0.18 point and a 0.79 point under one label.
    fam_style = {"blind_quant": ("#000000", "P", "ko'r-ko'rona INT8"),
                 "blind": ("#a51d2d", "s", "ko'r-ko'rona pruning"),
                 "tau": (ACCENT, "o", "bizniki (tau)"),
                 "ratio": (GREY, "^", "kesh-majburiy"),
                 "hybrid": ("#2a7f3f", "D", "o'q gibridi"),
                 "reference": (LIGHT, "*", "nazorat (FP32)")}

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    seen = set()
    for r in sorted(lib.values(), key=lambda r: r["enc_mib"]):
        fam = r.get("family", "tau")
        if r.get("key") == "int8":
            fam = "blind_quant"
        colour, marker, label = fam_style.get(fam, (GREY, "o", "?"))
        lo = r["wer"] - r.get("wer_lo", r["wer"])
        hi = r.get("wer_hi", r["wer"]) - r["wer"]
        ax.errorbar([r["enc_mib"]], [r["wer"]], yerr=[[lo], [hi]], fmt=marker,
                    ms=5, lw=0.8, capsize=2, color=colour, ecolor=LIGHT,
                    label=label if label not in seen else None)
        seen.add(label)
    ax.set_xlabel("enkoder hajmi, MiB (logarifmik)")
    ax.set_ylabel("WER (300 TEST namunasi)")
    # Log on both axes: the FP32 control sits at 1172 MiB while everything
    # interesting is packed between 200 and 300, and a linear axis spends
    # three quarters of its width on empty space between them.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([200, 250, 300, 600, 1200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(lw=0.4, color="#eeeeee", which="both")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    return save(fig, 8)


def fig9_int4_disconnect():
    """Why the most accurate operator gave the worst network.

    Left: per-operator output error against WikiText-2 perplexity. The
    calibrated weight-domain scale is the most accurate of the three on every
    operator and the worst by far on the network, which is the disconnect the
    section is about. Right: the reason -- its output gain sits furthest below
    one, and a gain compounds over the 78 feed-forward operators the signal
    passes through.
    """
    ppl = load("wikitext2_int4")
    ops = load("int4_bias_int4")
    if not ppl or not ops:
        return None
    s = ops["summary"]
    # "ours_ls" in the perplexity table is the output-domain rescale, which the
    # operator script calls "fix"; same construction, different script.
    arms = [("RTN", "rtn", "int4 rtn", "#000000", "P"),
            ("bizning (vazn domeni)", "ours", "int4 ours", "#a51d2d", "s"),
            ("bizning + Y-domen", "fix", "int4 ours_ls", ACCENT, "o")]
    have = [a for a in arms if a[2] in ppl["ppl"] and a[1] in s]
    if len(have) < 3:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))

    fp32 = ppl["ppl"]["FP32"]
    for label, okey, pkey, colour, marker in have:
        ax1.plot(s[okey]["err"], ppl["ppl"][pkey], marker, ms=7, color=colour)
        ax1.annotate(label, (s[okey]["err"], ppl["ppl"][pkey]),
                     textcoords="offset points", xytext=(7, -3),
                     fontsize=7, color=GREY)
    ax1.axhline(fp32, lw=0.7, ls="--", color=LIGHT)
    ax1.text(0.118, fp32, "FP32", va="bottom", ha="right", fontsize=7,
             color=GREY)
    ax1.set_xlim(0.070, 0.125)
    ax1.set_xlabel("operator chiqish xatosi (past = aniqroq)")
    ax1.set_ylabel("WikiText-2 perplexity")
    ax1.set_title("aniqroq operator, yomonroq tarmoq", fontsize=8)
    ax1.grid(lw=0.4, color="#eeeeee")

    xs = np.arange(len(have))
    gains = [s[k]["gain"] for _, k, _, _, _ in have]
    comp = [s[k]["gain_pow78"] for _, k, _, _, _ in have]
    colours = [c for _, _, _, c, _ in have]
    # The two bars per group are labelled through the axis caption rather than
    # a legend box: at this figure size any legend lands on top of the bars.
    ax2.bar(xs - 0.2, gains, width=0.38, color=colours, alpha=0.45)
    ax2.bar(xs + 0.2, comp, width=0.38, color=colours)
    ax2.axhline(1.0, lw=0.7, ls="--", color=LIGHT)
    for x, (g, c) in enumerate(zip(gains, comp)):
        ax2.text(x - 0.2, g + 0.03, f"{g:.4f}", ha="center", fontsize=6.5)
        ax2.text(x + 0.2, c + 0.03, f"{c:.2f}", ha="center", fontsize=6.5)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([lbl.replace(" (vazn domeni)", "\n(vazn dom.)")
                         .replace("bizning + ", "bizning\n+ ")
                         for lbl, _, _, _, _ in have], fontsize=7)
    ax2.set_ylim(0, 1.35)
    ax2.set_ylabel("kuchaytirish koeffitsiyenti")
    ax2.set_xlabel("och ustun — bitta operator;  to'q ustun — gain$^{78}$",
                   fontsize=7)
    ax2.set_title("siljish chuqurlik bo'ylab to'planadi", fontsize=8)
    ax2.grid(axis="y", lw=0.4, color="#eeeeee")
    fig.tight_layout()
    return save(fig, 9)


FIGURES = {4: fig4_redundancy_profile, 5: fig5_error_absorption,
           7: fig7_whole_model, 8: fig8_operating_points,
           9: fig9_int4_disconnect}


def main():
    print("O'lchovlardan rasmlar quriladi:")
    made = []
    for num, fn in sorted(FIGURES.items()):
        path = fn()
        if path:
            made.append(num)
        else:
            print(f"  fig{num}: ma'lumot fayli yo'q, o'tkazib yuborildi")
    print(f"\n{len(made)} ta rasm: {made}")
    print("Qolganlari (1, 2, 3, 6) — sxemalar; ular promt sifatida qoladi, "
          "chunki ortida ma'lumot yo'q.")


if __name__ == "__main__":
    main()
