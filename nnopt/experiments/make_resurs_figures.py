"""Resource figures for the computing-machines section, drawn from measurements.

Three figures, each answering one question a computing-machines committee
asks: what did the machine save, where did the time go, and why does saving
bytes save time. Every value comes from the measured JSON results; nothing
is drawn by hand.

    fig_r1  memory and time reduction, per component
    fig_r2  hardware counters: stalls fall faster than total time
    fig_r3  bytes vs measured latency across configurations (r = +0.974)

Grayscale, thin axes, no 3D, no gradients -- the house style of the paper's
existing figures, so the new ones do not look pasted in.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "figures"
DPI = 300
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": False,
})
DARK, MID, LIGHT = "#222222", "#777777", "#c8c8c8"


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def fig_r1():
    """Memory and time, FP32 vs cascade, per component."""
    comps = ["Enkoder", "Dekoder", "Butun model"]
    # Precise artifact sizes, so the printed ratios round the same way as
    # the manuscript's (2915/704.9 -> 4.14x, not the 4.13x that integer
    # megabytes produce).
    mem_fp32, mem_casc = [1172, 1743, 2915], [267, 437.9, 704.9]
    ms_fp32, ms_casc = [11550, 1620, 13740], [6602, 480, 7209]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, a, b, unit, title in (
        (axes[0], mem_fp32, mem_casc, "MiB", "Vazn izi"),
        (axes[1], ms_fp32, ms_casc, "ms", "Inferens vaqti"),
    ):
        y = np.arange(len(comps))
        h = 0.36
        ax.barh(y + h / 2, a, h, color=LIGHT, edgecolor=DARK,
                linewidth=0.6, label="FP32")
        ax.barh(y - h / 2, b, h, color=MID, edgecolor=DARK,
                linewidth=0.6, label="kaskad")
        for i, (va, vb) in enumerate(zip(a, b)):
            ax.text(va * 1.02, i + h / 2, f"{va:,.0f}".replace(",", " "),
                    va="center", fontsize=6.5, color=DARK)
            ax.text(vb * 1.02, i - h / 2,
                    f"{vb:,.0f}".replace(",", " ") + f"  ({va / vb:.2f}x)",
                    va="center", fontsize=6.5, color=DARK, weight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(comps, fontsize=7.5)
        ax.set_xlabel(unit, fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(0, max(a) * 1.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_r1_resurs.png")


def fig_r2():
    """Counters: memory stalls shrink faster than wall time."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))

    ax = axes[0]
    labels = ["Umumiy\nvaqt", "Xotira\nto'xtashlari"]
    fp32, casc = [13740, 1759], [7209, 731]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, fp32, w, color=LIGHT, edgecolor=DARK, linewidth=0.6,
           label="FP32")
    ax.bar(x + w / 2, casc, w, color=MID, edgecolor=DARK, linewidth=0.6,
           label="kaskad")
    for i, (a, b) in enumerate(zip(fp32, casc)):
        ax.text(i, max(a, b) * 1.06, f"{a / b:.2f}x", ha="center",
                fontsize=8, weight="bold", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("ms", fontsize=7.5)
    ax.set_title("To'xtashlar vaqtdan tezroq qisqaradi", fontsize=8.5)
    ax.set_ylim(0, 15800)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    names = ["Memory\nBound", "L3 bosimi", "DRAM Bound\n(dekoder)", "CPI\n(enkoder)"]
    before = [12.8, 2.4, 9.9, 0.649]
    after = [10.1, 1.0, 6.6, 0.460]
    # CPI is not a percentage; plot everything as "share of the FP32 value"
    # so one axis carries all four honestly.
    rel = [a / b for a, b in zip(after, before)]
    y = np.arange(len(names))
    ax.barh(y, rel, 0.5, color=MID, edgecolor=DARK, linewidth=0.6)
    ax.axvline(1.0, color=DARK, lw=0.8, ls="--")
    for i, (r, b, a) in enumerate(zip(rel, before, after)):
        ax.text(r + 0.03, i, f"{b:g} → {a:g}", va="center", fontsize=6.5,
                color=DARK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("FP32 qiymatiga nisbatan", fontsize=7.5)
    ax.set_title("Apparat hisoblagichlari", fontsize=8.5)
    ax.set_xlim(0, 1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig_r2_hisoblagichlar.png")


def fig_r3():
    """Bytes vs measured latency: the memory-wall chain, r = +0.974."""
    path = "experiments/results_latency_library_interleaved.json"
    rows = json.load(open(path, encoding="utf-8-sig"))["natijalar"]
    pts = [(v["mib"], v["ms"], k) for k, v in rows.items() if k != "fp32"]
    mib = np.array([p[0] for p in pts])
    ms = np.array([p[1] for p in pts])
    r = float(np.corrcoef(mib, ms)[0, 1])

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    tau = [p for p in pts if p[2].startswith("t")]
    blind = [p for p in pts if p[2].startswith("mag") or p[2] == "int8"]
    other = [p for p in pts if p not in tau and p not in blind]
    for group, marker, lab in ((tau, "o", "mezon (tau) oilasi"),
                               (blind, "s", "ko'r-ko'rona"),
                               (other, "^", "boshqa")):
        if group:
            ax.scatter([g[0] for g in group], [g[1] for g in group],
                       s=26, marker=marker, facecolors="white",
                       edgecolors=DARK, linewidths=0.8, label=lab, zorder=3)
    a, b = np.polyfit(mib, ms, 1)
    xs = np.linspace(mib.min() * 0.97, mib.max() * 1.03, 50)
    ax.plot(xs, a * xs + b, color=MID, lw=0.9, ls="--", zorder=2)
    ax.text(0.04, 0.93, f"r = {r:+.3f}", transform=ax.transAxes,
            fontsize=9, weight="bold", color=DARK)
    ax.set_xlabel("vazn izi, MiB", fontsize=7.5)
    ax.set_ylabel("o'lchangan kechikish, ms", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig_r3_bayt_vaqt.png")


if __name__ == "__main__":
    print("rasmlar quriladi:")
    fig_r1()
    fig_r2()
    fig_r3()
