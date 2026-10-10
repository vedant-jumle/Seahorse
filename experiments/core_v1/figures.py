"""Paper figures for the core_v1 write-up, built only from saved analysis outputs (no GPU).

Usage:
    python experiments/core_v1/figures.py --run results/core_v1_20261007_1820 --out writeup/figures

Every figure is saved as PDF (vector, for the paper) and PNG (300 dpi, for previews).
Palette: the dataviz reference categorical order (blue, orange, aqua, yellow), validated for CVD
separation on white. Aqua and yellow fall below 3:1 contrast, so every series is also direct-labelled
and/or carries its own marker shape.
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
MODELS = [("qwen35_2b", "Qwen3.5-2B"), ("qwen35_9b", "Qwen3.5-9B")]
DOSES = [0.5, 1, 2, 3]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pick(rs, **kw):
    out = [r for r in rs if all(r.get(k) == v for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def band(ax, x, est, lo, hi, color, marker, label, ls="-"):
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)
    ax.plot(x, est, ls, color=color, lw=1.6, marker=marker, ms=4.5, mec="white", mew=0.8, label=label)


# ---------------------------------------------------------------- C6: a fact is a late word
def fig_lens_facts(run, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axhspan(1, 10, color=GRID, alpha=0.7, lw=0)
    ax.text(0.02, 7.0, "top 10 of ~248k tokens", color=INK2, fontsize=7, va="center")
    for (m, name), c, mk in zip(MODELS, (BLUE, ORANGE), ("o", "s")):
        rs = rows(os.path.join(run, m, "analysis", "lens_facts.csv"))
        d = [fl(r["depth"]) for r in rs]
        rk = [max(fl(r["median_rank"]), 1) for r in rs]
        ax.plot(d, rk, "-", color=c, lw=1.6, marker=mk, ms=3.5, mec="white", mew=0.6, label=name)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks([1, 10, 100, 1e3, 1e4, 1e5], ["1", "10", "100", "1k", "10k", "100k"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("relative depth of the layer the shift is read from")
    ax.set_ylabel("rank of the fact's word (median, log)")
    ax.set_title("A stored fact is the word to say, only at the end", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    save(fig, out, "fig_c6_lens_facts")


# ---------------------------------------------------------------- C3: facts are not used
def fig_facts(run, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
    x = [0] + DOSES
    for ax, (m, name) in zip(axes, MODELS):
        rs = rows(os.path.join(run, m, "analysis", "facts.csv"))
        conds = ["nomem"] + [f"mem@{a:g}" for a in DOSES]
        get = lambda col: ([fl(pick(rs, cond=c)[col]) for c in conds],
                           [fl(pick(rs, cond=c)[col + "_lo"]) for c in conds],
                           [fl(pick(rs, cond=c)[col + "_hi"]) for c in conds])
        series = [("hit_use", BLUE, "o", "mentions the fact"),
                  ("j_incoh_all", ORANGE, "s", "incoherent"),
                  ("j_clean_use", AQUA, "D", "uses the fact cleanly")]
        for col, c, mk, lab in series:
            band(ax, x, *get(col), c, mk, lab)
        ctx = fl(pick(rs, cond="ctx")["j_clean_use"])
        ax.axhline(ctx, color=INK2, lw=1.0, ls="--")
        ax.text(-0.05, ctx + 0.015, "fact in context", color=INK2, fontsize=7, ha="left", va="bottom")
        ax.set_xticks(x, ["none"] + [f"{a:g}" for a in DOSES])
        ax.set_xlabel("memory strength α (none = no memory)")
        ax.set_title(name, loc="left")
        ax.set_ylim(-0.02, 1.08)
    axes[0].set_ylabel("share of answers to 'use' prompts")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Facts come back as mentions and garble, never as clean use", x=0.01, ha="left",
                 fontsize=10, y=1.03)
    save(fig, out, "fig_c3_facts")


# ---------------------------------------------------------------- C2: leanings in a narrow window
def fig_leanings(run, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    x = [0] + DOSES
    for ax, (m, name) in zip(axes, MODELS):
        rs = [r for r in rows(os.path.join(run, m, "analysis", "pref.csv")) if r["group"] == "leanings"]
        conds = ["nomem"] + [f"mem@{a:g}" for a in DOSES]
        get = lambda col: ([fl(pick(rs, cond=c)[col]) for c in conds],
                           [fl(pick(rs, cond=c)[col + "_lo"]) for c in conds],
                           [fl(pick(rs, cond=c)[col + "_hi"]) for c in conds])
        band(ax, x, *get("lex_clean"), BLUE, "o", "word-list lean (non-loop answers)")
        band(ax, x, *get("j_incoh"), ORANGE, "s", "incoherent (judge)")
        band(ax, x, *get("j_lean"), AQUA, "D", "coherent recommendation, judge lean")
        for k, (cond, lab) in enumerate((("rand@2", "rand"), ("swap@2", "swap"), ("placebo@2", "placebo"))):
            v = fl(pick(rs, cond=cond)["j_lean"])
            ax.plot(2.12 + 0.12 * k, v, marker="x", color=INK2, ms=4, mew=1.0)
        ax.annotate("controls (α=2)", xy=(2.24, fl(pick(rs, cond="swap@2")["j_lean"]) - 0.03),
                    xytext=(1.25, -0.3), fontsize=6.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ctx = fl(pick(rs, cond="ctx")["j_lean"])
        ax.axhline(ctx, color=INK2, lw=1.0, ls="--")
        ax.text(-0.05, ctx + 0.02, "experience in context", color=INK2, fontsize=7, va="bottom")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(x, ["none"] + [f"{a:g}" for a in DOSES])
        ax.set_xlabel("memory strength α (none = no memory)")
        ax.set_title(name, loc="left")
        ax.set_ylim(-0.45, 1.05)
    axes[0].set_ylabel("lean (−1…+1)  /  share incoherent")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="x", ls="", color=INK2, ms=4, mew=1.0))
    l.append("controls: random / swapped / placebo memory")
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Leanings transfer only in a narrow window: the words keep rising, coherence collapses",
                 x=0.01, ha="left", fontsize=10, y=1.03)
    save(fig, out, "fig_c2_leanings")


# ---------------------------------------------------------------- C5: what you subtract decides direction
REFS = [("opposite", "subtract the opposite", BLUE),
        ("centroid", "subtract alternatives (centroid)", ORANGE),
        ("without", "subtract nothing (without)", AQUA)]
JUDGES = [("orig", "Qwen3.5-9B (original)", "o"), ("gptoss", "gpt-oss", "s"),
          ("gemma4", "Gemma-4 12B", "^"), ("maj3x", "majority, 3 cross-family", "D")]


def fig_dislikes(run, out):
    claims = json.load(open(os.path.join(run, "panel", "panel_claims.json")))["claims"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    for ax, (m, name) in zip(axes, MODELS):
        ax.axhspan(0, 0.8, color=BLUE, alpha=0.05, lw=0)
        ax.axhspan(-0.8, 0, color=ORANGE, alpha=0.05, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        for i, (ref, lab, c) in enumerate(REFS):
            for j, (jk, jn, mk) in enumerate(JUDGES):
                e = claims[jk][m]["C5"][ref]
                xx = i + (j - 1.5) * 0.15
                ax.errorbar(xx, e["est"], yerr=[[e["est"] - e["lo"]], [e["hi"] - e["est"]]], fmt=mk, color=c,
                            ms=5, mec="white", mew=0.7, elinewidth=1.1, capsize=0)
        ax.set_xticks(range(3), ["opposite", "centroid", "without"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.76, 0.65)
        ax.set_title(name, loc="left")
        ax.text(2.55, 0.58, "right way (avoids what\nthe user dislikes)", color=INK2, fontsize=6.5, ha="right", va="top")
        ax.text(-0.55, -0.72, "wrong way (recommends it)", color=INK2, fontsize=6.5, ha="left", va="bottom")
    axes[0].set_ylabel("change in judge lean vs no memory\n(8 dislikes, 95% CI)")
    handles = [plt.Line2D([], [], marker=mk, ls="", color=INK2, mec="white", ms=5, label=jn) for _, jn, mk in JUDGES]
    fig.supxlabel("what the memory subtracts when it is written", fontsize=8, color=INK2, y=-0.05)
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.18))
    fig.suptitle("Dislikes flip into likes unless the memory subtracts the opposite (every judge agrees)",
                 x=0.01, ha="left", fontsize=10, y=1.03)
    save(fig, out, "fig_c5_dislikes")


def fig_lens_dislikes(run, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    for ax, (m, name) in zip(axes, MODELS):
        rs = [r for r in rows(os.path.join(run, m, "analysis", "lens_prefs.csv")) if r["group"] == "dislikes"]
        ax.axhline(0, color=MUTED, lw=0.8)
        for (ref, lab, c), mk in zip(REFS, ("o", "s", "D")):
            sub = sorted((r for r in rs if r["ref"] == ref), key=lambda r: fl(r["depth"]))
            ax.plot([fl(r["depth"]) for r in sub], [fl(r["lex_gain"]) for r in sub], "-", color=c, lw=1.6,
                    marker=mk, ms=3, mec="white", mew=0.5, label=lab)
        ax.set_xlabel("relative depth")
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("logit-lens gain of the stored shift\n(+ avoids it, − points at it)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Read directly, the shift without the opposite points at the disliked thing",
                 x=0.01, ha="left", fontsize=10, y=1.03)
    save(fig, out, "fig_c5_lens_dislikes")


# ---------------------------------------------------------------- C1: selectivity
def fig_selectivity(run, out):
    claims = json.load(open(os.path.join(run, "analysis", "claims_all.json")))
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    labels = [n for _, n in MODELS]
    gated = [claims[m]["C1"]["evidence"]["unrel_same_mem2"] for m, _ in MODELS]
    forced = [claims[m]["C1"]["evidence"]["unrel_same_gate_on"] for m, _ in MODELS]
    y = range(len(labels))
    for yy, g, f in zip(y, gated, forced):
        ax.plot([f, g], [yy, yy], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(g, yy, "o", color=BLUE, ms=7, mec="white", mew=0.8, zorder=2)
        ax.plot(f, yy, "s", color=ORANGE, ms=6.5, mec="white", mew=0.8, zorder=2)
        ax.text(g, yy + 0.22, f"{g:.0%}", color=INK, fontsize=7, ha="center")
        ax.text(f, yy + 0.22, f"{f:.0%}", color=INK, fontsize=7, ha="center")
    ax.set_yticks(list(y), labels)
    ax.set_ylim(-0.6, len(labels) - 0.2)
    ax.set_xlim(-0.03, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("unrelated answers identical to no memory")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=6, label="with the gate (α=2)"),
                       plt.Line2D([], [], marker="s", ls="", color=ORANGE, mec="white", ms=6, label="gate forced open")],
              loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2)
    ax.set_title("The gate keeps the memory out of unrelated answers", loc="left")
    save(fig, out, "fig_c1_selectivity")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_lens_facts(a.run, a.out)
    fig_facts(a.run, a.out)
    fig_leanings(a.run, a.out)
    fig_dislikes(a.run, a.out)
    fig_lens_dislikes(a.run, a.out)
    fig_selectivity(a.run, a.out)


if __name__ == "__main__":
    main()
