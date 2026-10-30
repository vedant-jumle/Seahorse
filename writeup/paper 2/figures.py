"""Figures for paper 2. The numbers are copied from the report files named beside each block (results/, local and
untracked); means with 95% bootstrap CIs over items (or members) then prompts.

    python "writeup/paper 2/figures.py"        # writes writeup/paper 2/figures/fig{2,3,4,5}.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
MODELS = ["Qwen2.5-1.5B", "Qwen3.5-2B (hybrid)", "Llama-3.2-3B"]
MODELS2 = ["Qwen2.5\n1.5B", "Qwen3.5-2B\n(hybrid)", "Llama-3.2\n3B"]
C = {"ctx": "#4d4d4d", "slice": "#2b6cb0", "text": "#d69e2e", "attn": "#90cdf4", "state": "#68d391",
     "sep": "#c53030", "placebo": "#ed8936", "second": "#2b6cb0", "both": "#9f7aea", "joint": "#4d4d4d",
     "min": "#a0aec0", "minsubj": "#38a169"}


def bars(ax, groups, series, colors, labels, ylabel, ylim=(0, 1.05)):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (s, col, lab) in enumerate(zip(series, colors, labels)):
        m = np.array([v[0] for v in s])
        err = np.array([[v[0] - v[1] for v in s], [v[2] - v[0] for v in s]])
        ax.bar(x + (i - (n - 1) / 2) * w, m, w * 0.92, color=col, label=lab, yerr=err, capsize=1.5,
               error_kw={"elinewidth": 0.6, "capthick": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)


def fig2():
    # clean use, primary items, use prompts. Qwen2.5: dream_bind_add5_20261013_1255/qwen25_15b; 2B:
    # dream_bind_add5_20261013_1255/qwen35_2b (slice = R-min-state); Llama: dream_bind_add7_20261022_1526/llama32_3b.
    ctx = [(0.90, 0.71, 1.00), (1.00, 1.00, 1.00), (1.00, 1.00, 1.00)]
    sl = [(0.87, 0.63, 1.00), (0.97, 0.91, 1.00), (1.00, 1.00, 1.00)]
    tx = [(0.27, 0.06, 0.51), (0.24, 0.06, 0.47), (0.24, 0.03, 0.54)]
    nm = [(0.00, 0.00, 0.00)] * 3
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    bars(ax, MODELS2, [ctx, sl, tx, nm], [C["ctx"], C["slice"], C["text"], "#e2e8f0"],
         ["sentence in context", "2-position slice", "same 2 tokens as text", "no memory"], "clean use")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.savefig(OUT / "fig2.pdf", bbox_inches="tight")


def fig3():
    # Qwen3.5-2B, 7 primary items, teacher-forced: balanced yes/no (vs unrelated foils) and two-choice accuracy
    # (vs unrelated foils). dream_bind_add8_20261025_1812/qwen35_2b_probe/analysis/report.md (part 6 table).
    groups = ["in\ncontext", "attention\nentries", "recurrent\nstate", "both"]
    yn = [(1.00, 1.00, 1.00), (0.50, 0.50, 0.50), (0.93, 0.86, 1.00), (0.93, 0.86, 1.00)]
    tc = [(1.00, 1.00, 1.00), (0.91, 0.80, 0.98), (0.48, 0.36, 0.61), (1.00, 1.00, 1.00)]
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    bars(ax, groups, [yn, tc], [C["state"], C["slice"]],
         ['"Is my dog called X?" (yes/no)', '"X or Y?" (two-choice)'], "accuracy")
    ax.axhline(0.5, color="#a0aec0", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(OUT / "fig3.pdf", bbox_inches="tight")


def fig4():
    # partner confusion, confusable pairs. Left: the 8 addendum-5 pairs; right: the 34 pool pairs.
    # dream_bind_add8_20261025_1812/<model>_collide and <model>_pp_collide (2B: the -jstate conditions).
    conds = ["stored apart", "apart, B after\na sentence", "apart, B stored\nsecond", "apart, both\nstored second",
             "encoded\ntogether", "2 pos.,\ntogether", "2 pos. + subject,\ntogether"]
    cols = [C["sep"], C["placebo"], C["second"], C["both"], C["joint"], C["min"], C["minsubj"]]
    a5 = {  # sep, placebo, tagged, bothtagged, joint, min-joint, minsubj-joint
        "q": [(0.37, 0.18, 0.56), (0.26, 0.09, 0.44), (0.09, 0.01, 0.24), (0.21, 0.07, 0.41), (0.05, 0.00, 0.13),
              (0.20, 0.05, 0.39), (0.03, 0.00, 0.12)],
        "h": [(0.33, 0.15, 0.54), (0.19, 0.06, 0.36), (0.15, 0.03, 0.34), (0.19, 0.09, 0.32), (0.05, 0.00, 0.13),
              (0.06, 0.01, 0.13), (0.03, 0.00, 0.07)],
        "l": [(0.41, 0.23, 0.61), (0.31, 0.13, 0.53), (0.09, 0.00, 0.25), (0.16, 0.01, 0.37), (0.01, 0.00, 0.03),
              (0.17, 0.03, 0.38), (0.01, 0.00, 0.04)]}
    pp = {
        "q": [(0.22, 0.14, 0.32), (0.07, 0.02, 0.14), (0.01, 0.00, 0.04), (0.20, 0.10, 0.31), (0.00, 0.00, 0.00),
              (0.10, 0.04, 0.17), (0.00, 0.00, 0.00)],
        "h": [(0.16, 0.09, 0.25), (0.05, 0.01, 0.11), (0.01, 0.00, 0.03), (0.09, 0.04, 0.17), (0.00, 0.00, 0.00),
              (0.02, 0.00, 0.06), (0.01, 0.00, 0.04)],
        "l": [(0.20, 0.10, 0.32), (0.13, 0.05, 0.24), (0.01, 0.00, 0.03), (0.03, 0.00, 0.07), (0.00, 0.00, 0.00),
              (0.03, 0.00, 0.07), (0.00, 0.00, 0.00)]}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), sharey=True)
    for ax, d, title in zip(axes, [a5, pp], ["8 hand-written pairs", "34 pool pairs"]):
        series = [[d[m][i] for m in "qhl"] for i in range(len(conds))]
        bars(ax, ["Qwen2.5", "Qwen3.5-2B", "Llama-3.2"], series, cols, [c.replace("\n", " ") for c in conds],
             "partner confusion" if ax is axes[0] else "", ylim=(0, 0.65))
        ax.set_title(title, fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.5))
    fig.savefig(OUT / "fig4.pdf", bbox_inches="tight")


def fig5():
    # partner confusion vs the number of stored facts, confusable groups. N = 2: addendum 8 (the 8 pairs; "stored
    # second" = B stored as pair 2, "sentence" = B after the placebo); N = 3, 5: dream_bind_add9_20261030_0107.
    N = [2, 3, 5]
    d = {  # stored apart, k-th fact, neutral sentences, encoded together
        "q": [[0.37, 0.47, 0.54], [0.09, 0.09, 0.22], [0.26, 0.27, 0.37], [0.05, 0.01, 0.03]],
        "h": [[0.33, 0.38, 0.55], [0.15, 0.04, 0.23], [0.19, 0.20, 0.36], [0.05, 0.01, 0.03]],
        "l": [[0.41, 0.41, 0.51], [0.09, 0.09, 0.30], [0.31, 0.34, 0.42], [0.01, 0.00, 0.00]]}
    labs = ["stored apart (all first)", "stored as the k-th fact", "after k−1 neutral sentences", "encoded together"]
    cols = [C["sep"], C["second"], C["placebo"], C["joint"]]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.6), sharey=True)
    for ax, m, title in zip(axes, "qhl", MODELS):
        for s, col, lab in zip(d[m], cols, labs):
            ax.plot(N, s, marker="o", ms=3, lw=1.1, color=col, label=lab)
        ax.set_xticks(N)
        ax.set_xlabel("stored facts N")
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 0.62)
    axes[0].set_ylabel("partner confusion")
    axes[0].legend(frameon=False, fontsize=6.3, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.42))
    fig.savefig(OUT / "fig5.pdf", bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig2, fig3, fig4, fig5):
        f()
