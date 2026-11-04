#!/usr/bin/env python
"""dream_select analysis (PREREG.md): tables with bootstrap CIs, the pre-registered rules, the examples -> report.md.

  python -I experiments/dream_select/analyze.py results/dream_select_<stamp>

The plain-language summary at the top of report.md is filled from the numbers (`summary`); the post-hoc table of
what the selectors keep (punctuation share) is labelled as not pre-registered.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
import gzip  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

_spec = importlib.util.spec_from_file_location("dream_select_select_an", HERE / "selectors_ds.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

N_BOOT, SEED = 2000, 7
MAIN_K = "k4"
ANCHORS = ("nomem", "ctx", "T-full", "R-full", "R-content")
SELS = ("S-dream", "S-dream-split", "S-attn", "S-recon", "S-loo", "last-k", "random", "gold@k")
SHORT_MAX = 60
LOCAL_LABEL = "Qwen2.5-1.5B"     # the local run's meta has no model_label
PUNCT = re.compile(r"^\s*[^\w\s]+\s*$")
CONTRASTS = (("S-dream", "random"), ("S-dream", "S-attn"), ("S-attn", "random"), ("S-dream-split", "random"),
             ("S-dream-split", "S-dream"), ("S-recon", "random"), ("S-loo", "random"), ("S-dream", "S-loo"),
             ("S-dream", "S-recon"), ("last-k", "random"), ("gold@k", "S-dream"))


# ----------------------------------------------------------------------------------------------- stats


def boot(vals, stat=np.mean, n_boot=N_BOOT, seed=SEED):
    """(point, lo, hi, n): percentile bootstrap over items."""
    v = np.asarray([x for x in vals if x is not None and math.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    st = stat(v[idx], axis=1)
    lo, hi = np.quantile(st, [0.025, 0.975])
    return {"mean": float(stat(v)), "lo": float(lo), "hi": float(hi), "n": int(len(v))}


def paired(a, b, items):
    """Per-item a - b over the items present (and finite) in both, bootstrapped."""
    d = [a[i] - b[i] for i in items if a.get(i) is not None and b.get(i) is not None]
    return boot(d)


def f(s, nd=2):
    if s["n"] == 0 or math.isnan(s["mean"]):
        return "n/a"
    return f"{s['mean']:.{nd}f} [{s['lo']:.{nd}f}, {s['hi']:.{nd}f}]"


# ----------------------------------------------------------------------------------------------- rules


def rules(d_rand, d_attn, gap_dream, d_attn_rand):
    """The pre-registered rules (k = 4, answer score, all usable items). Arguments: paired CI dicts of S-dream - random,
    S-dream - S-attn, S-attn - random, and S-dream's mean gap closed (a float). Returns {rule: (holds, reason)}."""
    ok1 = d_rand["lo"] > 0 and d_attn["lo"] > 0 and gap_dream >= 0.5
    r1 = (f"S-dream - random lower bound {d_rand['lo']:.2f}, S-dream - S-attn lower bound {d_attn['lo']:.2f}, "
          f"S-dream mean gap closed {gap_dream:.2f} (need > 0, > 0, >= 0.5)")
    ok2 = d_attn_rand["lo"] > 0
    r2 = f"S-attn - random lower bound {d_attn_rand['lo']:.2f} (need > 0)"
    return {"Dreamed questions select well": (bool(ok1), r1), "Any attention selector works": (bool(ok2), r2)}


# ----------------------------------------------------------------------------------------------- load


def load(out):
    rows = [json.loads(l) for l in gzip.open(out / "rows.jsonl.gz", "rt")]
    meta = json.load(open(out / "meta.json"))
    return rows, meta


def tables(rows, meta):
    """V[measure][(cond, budget)][item] and the gap closed."""
    V = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["cond"], r["budget"])
        for m in ("answer_score", "answer_mean", "refusal", "hit"):
            if r.get(m) is not None:
                V[m][key][r["item"]] = r[m]
    nm, full = V["answer_score"][("nomem", None)], V["answer_score"][("R-full", None)]
    dropped = sorted(i for i in nm if S.gap_closed(0.0, nm[i], full[i]) is None)
    for key, d in list(V["answer_score"].items()):
        for i, x in d.items():
            g = S.gap_closed(x, nm[i], full[i])
            if g is not None:
                V["gap"][key][i] = g
    return V, dropped


# ----------------------------------------------------------------------------------------------- report


def key_of(c, b):
    return (c, None) if c in ANCHORS else (("gold", "own") if c == "gold" else (c, b))


def cond_table(V, items, budget, conds, title):
    L = [f"### {title}", "",
         "| condition | N | answer score log P(answer), nats | per-token mean | refusal P(I+Sorry+As) first | gap closed mean "
         "| gap closed median (N) | hit rate (N verbatim) |", "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        k = key_of(c, budget)
        sub = lambda m: {i: V[m][k][i] for i in items if i in V[m][k]}
        a = sub("answer_score")
        if not a:
            continue
        g = sub("gap")
        h = sub("hit")
        L.append(f"| {c} | {len(a)} | {f(boot(list(a.values())))} | {f(boot(list(sub('answer_mean').values())))} | "
                 f"{f(boot(list(sub('refusal').values())))} | {f(boot(list(g.values())))} | "
                 f"{f(boot(list(g.values()), stat=np.median))} ({len(g)}) | "
                 f"{f(boot(list(h.values()))) if h else ''}{f' ({len(h)})' if h else ''} |")
    return L + [""]


def contrast_table(V, items, budget, title):
    L = [f"### {title}", "", "| contrast (paired, per item) | answer score | gap closed | refusal | N |",
         "|---|---|---|---|---|"]
    for a, b in CONTRASTS:
        ka, kb = key_of(a, budget), key_of(b, budget)
        sa = paired(V["answer_score"][ka], V["answer_score"][kb], items)
        if sa["n"] == 0:
            continue
        L.append(f"| {a} − {b} | {f(sa)} | {f(paired(V['gap'][ka], V['gap'][kb], items))} | "
                 f"{f(paired(V['refusal'][ka], V['refusal'][kb], items))} | {sa['n']} |")
    return L + [""]


def curve_table(V, items, budgets, measure, title, nd=2):
    L = [f"### {title}", "", "| selector | " + " | ".join(budgets) + " |", "|---|" + "---|" * len(budgets)]
    src = "gap" if measure == "gap_median" else measure
    for c in SELS:
        cells = []
        for b in budgets:
            d = {i: V[src][(c, b)][i] for i in items if i in V[src][(c, b)]}
            st = boot(list(d.values()), stat=np.median) if measure == "gap_median" else boot(list(d.values()))
            cells.append(f"{f(st, nd)}" if d else "")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    return L + [""]


def mark(info, sel):
    s = set(sel)
    return "".join(f"**[{t}]**" if p in s else t for p, t in zip(info["content"], info["content_tokens"]))


def punct_share(rows, infos, c, b):
    n = p = 0
    for r in rows:
        if r["cond"] == c and r["budget"] == b:
            tm = dict(zip(infos[r["item"]]["content"], infos[r["item"]]["content_tokens"]))
            for s in r["sel"]:
                n += len(s)
                p += sum(bool(PUNCT.match(tm[x])) for x in s)
    return p / n if n else float("nan")


def summary(V, items, verb, infos, rows, mk, R, d_rand, d_attn, d_ar, label=LOCAL_LABEL):
    """The plain-language summary at the top of report.md, filled from the numbers. The interpretive sentences were
    written for the local Qwen2.5-1.5B run; for another model (PREREG addendum 1) they are left for review."""
    local = label == LOCAL_LABEL
    m = lambda meas, c, b=mk, sub=items: boot([V[meas][key_of(c, b)][i] for i in sub if i in V[meas][key_of(c, b)]])
    g = lambda c, b=mk: m("gap", c, b)["mean"]
    h = lambda c, b=mk: m("hit", c, b, verb)["mean"]
    rf = lambda c, b=mk: m("refusal", c, b)["mean"]
    ids = max(infos[i]["identity_max_abs_diff"] for i in items)
    ok1, ok2 = R["Dreamed questions select well"][0], R["Any attention selector works"][0]
    nc = [infos[i]["n_content"] for i in items]
    base = np.mean([bool(PUNCT.match(t)) for i in items for t in infos[i]["content_tokens"]])
    return [
        "## Summary", "",
        f"**What was compared.** Each memory was one real user turn from LongMemEval's single-session-user items "
        f"({len(items)} usable items, {min(nc)} to {max(nc)} content tokens). It was the turn that holds the answer, given "
        "by the label, so only the choice of positions was tested. The turn's K,V was stored, and k of its positions "
        "(plus the chat head) were put back before the benchmark question. The selectors never saw the question or "
        "the answer:",
        "- **S-dream:** attention from 3 questions the model dreamed up about the turn.",
        "- **S-attn:** attention the position receives inside the turn, with no query.",
        "- **S-recon:** attention while the model repeats the turn.",
        "- **S-loo:** leave-one-out, using the model's own answers to the dreamed questions. Expensive; a reference.",
        "- **Baselines:** random positions, the last k tokens, and gold@k (the answer's own tokens; uses the label).",
        "",
        "The main measure is log P(the gold answer as the first words of the reply). The gap closed puts it on a scale "
        "from no memory (0) to the whole turn (1).", "",
        "**What came out (k = 4).**",
        f"- **The whole slice works, as before.** R-full equals the turn as text (max |Δlogit| {ids:.0e}). It scores "
        f"{m('answer_score', 'R-full')['mean']:.1f} nats, against {m('answer_score', 'nomem')['mean']:.1f} with no "
        f"memory and {m('answer_score', 'ctx')['mean']:.1f} with the turn in context.",
        "- **The automatic selectors against random:**", "",
        "  | k = 4 | gap closed | answer-score gain over random (nats) |", "  |---|---|---|",
        f"  | S-dream | {g('S-dream'):.2f} | {f(d_rand)} |",
        f"  | S-attn | {g('S-attn'):.2f} | {f(d_ar)} |",
        f"  | S-recon | {g('S-recon'):.2f} | |",
        f"  | random | {g('random'):.2f} | |", "",
        f"- **Dreamed questions against plain attention:** S-dream − S-attn is {f(d_attn)}.",
        f"- **Pre-registered verdicts:** \"Dreamed questions select well\" **{'holds' if ok1 else 'does not hold'}** "
        f"(S-dream's gap {g('S-dream'):.2f}; the bar was 0.5). \"Any attention selector works\" "
        f"**{'holds' if ok2 else 'does not hold'}**.",
        "- **The answer's position is what matters.**",
        f"  - gold@k (4 tokens around the answer) closes {g('gold@k'):.2f} of the gap. gold (the answer's tokens plus "
        f"one) closes {g('gold', None):.2f}.",
        f"  - The attention selectors include an answer token in {min(h(c) for c in ('S-dream', 'S-attn', 'S-recon')):.2f} "
        f"to {max(h(c) for c in ('S-dream', 'S-attn', 'S-recon')):.2f} of verbatim items; random does in "
        f"{h('random'):.2f}.",
        f"  - S-loo finds it in {h('S-loo'):.2f} and closes {g('S-loo'):.2f} ({g('S-loo', 'k8'):.2f} at k = 8).",
        f"- **What the attention selectors keep instead (post hoc).** Punctuation-only tokens (sentence ends) make up "
        f"{punct_share(rows, infos, 'S-dream', mk):.2f} of S-dream's picks and {punct_share(rows, infos, 'S-recon', mk):.2f} "
        f"of S-recon's. The base rate is {base:.2f}, and random keeps {punct_share(rows, infos, 'random', mk):.2f}.",
        f"- **Refusal at the first word:** S-dream {rf('S-dream'):.2f}, S-attn {rf('S-attn'):.2f}, S-recon "
        f"{rf('S-recon'):.2f}, S-loo {rf('S-loo'):.2f}, random {rf('random'):.2f}, gold@k {rf('gold@k'):.2f}, R-full "
        f"{rf('R-full'):.2f}." + (" The positions that raise the answer's probability most (gold@k) are not the ones that "
                                "lower refusal most." if local else ""), "",
        ("**What it means.** (Not generated: to be written after reviewing this model's numbers.)" if not local else
         "**What it means.** On natural text, a slice of a few positions picked by attention, with or without dreamed "
        "questions, recovers only a small part of what the whole turn gives. The attention selectors find sentence ends, "
        "not the value. Testing positions by what they do (S-loo) does better, and the answer span itself does much "
        "better. A useful automatic selector would have to target the value span, not raw attention mass. At these turn "
        f"lengths (median {int(np.median(nc))} tokens) storing the whole turn is cheap."), "",
        "**What it can't tell us.**",
        "- **One answer form.** The score is log P of the bare gold string at the start of the reply. Answers phrased as "
        "\"You graduated with …\" are not credited. Gap ratios are noisy, so the medians are reported beside the means.",
        "- **Which turn to store was given.** Retrieval was not tested.",
        ("- **The dreamer is a weak 1.5B model.** Some of its questions are about the help asked for, or are addressed to "
         "the assistant. S-loo, which uses the same dreams, does better, so the dreams are not the whole story." if local else
         f"- **The dreamer is the model itself** ({label}, greedy)."),
        f"- **Scope.** Teacher-forced only, with no generation or judges, on one model and {len(items)} items. The "
        f"non-verbatim split has {len(items) - len(verb)} items.",
    ]


def analyze(out):
    out = Path(out)
    rows, meta = load(out)
    V, dropped = tables(rows, meta)
    infos = meta["items"]
    items = list(infos)
    budgets = meta["budgets"]
    verb = [i for i in items if infos[i]["verbatim"]]
    nonv = [i for i in items if not infos[i]["verbatim"]]
    short = [i for i in items if infos[i]["n_content"] <= SHORT_MAX]
    long_ = [i for i in items if infos[i]["n_content"] > SHORT_MAX]
    mk = MAIN_K if MAIN_K in budgets else budgets[0]
    # --- rules
    A = V["answer_score"]
    d_rand = paired(A[("S-dream", mk)], A[("random", mk)], items)
    d_attn = paired(A[("S-dream", mk)], A[("S-attn", mk)], items)
    d_ar = paired(A[("S-attn", mk)], A[("random", mk)], items)
    g_dream = boot([V["gap"][("S-dream", mk)][i] for i in items if i in V["gap"][("S-dream", mk)]])
    R = rules(d_rand, d_attn, g_dream["mean"], d_ar)
    verdict = {"rules": {k: {"holds": v[0], "reason": v[1]} for k, v in R.items()},
               "S-dream - random": d_rand, "S-dream - S-attn": d_attn, "S-attn - random": d_ar,
               "S-dream gap closed": g_dream, "n_items": len(items), "n_gap_dropped": len(dropped)}
    json.dump(verdict, open(out / "verdict.json", "w"), indent=1)
    # --- report
    label = meta.get("model_label", LOCAL_LABEL)
    L = [f"# dream_select: can the slice's positions be chosen automatically? ({label}, teacher-forced, "
         "LongMemEval)", ""]
    L += summary(V, items, verb, infos, rows, mk, R, d_rand, d_attn, d_ar, label) + [""]
    env = meta["env"]
    nd = sum(1 for i in items if infos[i]["dreams"]["fallback"])
    nr = sum(1 for i in items if infos[i]["dreams"]["retried"])
    ns = sum(1 for i in items if infos[i]["dreams"]["attempts"][0]["strict_ok"])
    L += [f"Run: `{out}`; {len(items)} items ({len(verb)} verbatim); {meta['minutes']:.1f} min; peak GPU "
          f"{meta['peak_gpu_gb_reserved']:.2f} GB reserved ({meta['peak_gpu_gb_allocated']:.2f} GB allocated); "
          f"commit {env['commit']} (dirty: {env['dirty']}); {env['gpu']['name'] if env['gpu'] else 'cpu'}. "
          f"Pre-registration: `experiments/dream_select/PREREG.md`. CIs: 95% percentile bootstrap over items, "
          f"{N_BOOT} replicates, seed {SEED}; contrasts paired on the same items.", ""]
    L += ["## Rule verdicts (pre-registered; k = 4, answer score, all usable items)", ""]
    for k, (ok, why) in R.items():
        L.append(f"- **{k}: {'HOLDS' if ok else 'does not hold'}** ({why}).")
    L += ["", f"Gap closed: {len(dropped)} of {len(items)} items dropped (R-full − nomem < 0.5 nat): "
          f"{', '.join(dropped) or 'none'}.", ""]
    L += ["## Main table, k = 4", ""]
    allc = list(ANCHORS) + list(SELS) + ["gold"]
    L += cond_table(V, items, mk, allc, f"all ({len(items)} items)")
    L += contrast_table(V, items, mk, "Paired contrasts at k = 4 (all items)")
    L += ["## Curves over k", "",
          "k10 = max(2, round(0.10 × n_content)); its median over items is "
          f"{int(np.median([S.budget_k('k10', infos[i]['n_content']) for i in items])) if 'k10' in budgets else 'n/a'}.",
          ""]
    L += curve_table(V, items, budgets, "answer_score", "Answer score (nats)")
    L += curve_table(V, items, budgets, "gap", "Gap closed (mean)")
    L += curve_table(V, items, budgets, "gap_median", "Gap closed (median)")
    L += curve_table(V, items, budgets, "refusal", "Refusal mass at the first token")
    L += curve_table(V, verb, budgets, "hit", f"Hit rate (verbatim items, N = {len(verb)})")
    for b in budgets:
        if b == mk:
            continue
        L += contrast_table(V, items, b, f"Paired contrasts at {b}")
    L += ["## Splits (k = 4)", ""]
    for name, sub in ((f"verbatim ({len(verb)})", verb), (f"non-verbatim ({len(nonv)})", nonv),
                      (f"short turns, ≤ {SHORT_MAX} content tokens ({len(short)})", short),
                      (f"long turns, > {SHORT_MAX} content tokens ({len(long_)})", long_)):
        L += cond_table(V, sub, mk, allc, name)
        L += contrast_table(V, sub, mk, f"Contrasts: {name}")
    # --- dreamer
    L += ["## The dreamer", "",
          f"- First prompt strictly valid (3 lines, all ending in '?'): {ns} of {len(items)}; retried with the stricter "
          f"prompt: {nr}; fell back to fewer than 3 questions: {nd}.",
          f"- S-loo run on {sum(1 for i in items if infos[i]['loo_run'])} items (turns ≤ 200 content tokens; the longest "
          f"has {max(infos[i]['n_content'] for i in items)}).",
          f"- Content tokens per turn: median {int(np.median([infos[i]['n_content'] for i in items]))}, max "
          f"{max(infos[i]['n_content'] for i in items)}; head {infos[items[0]]['n_head']} tokens.", ""]
    # --- post hoc: what kind of token each selector keeps
    punct = lambda t: bool(PUNCT.match(t))
    base = np.mean([punct(t) for i in items for t in infos[i]["content_tokens"]])
    L += ["## What the selectors keep (post hoc, not pre-registered)", "",
          f"Share of the k = 4 picks that are punctuation-only tokens (base rate over all content tokens: {base:.2f}).",
          "", "| selector | punctuation share of picks |", "|---|---|"]
    for c in SELS:
        n = p = 0
        for r in rows:
            if r["cond"] == c and r["budget"] == mk:
                tm = dict(zip(infos[r["item"]]["content"], infos[r["item"]]["content_tokens"]))
                for s in r["sel"]:
                    n += len(s)
                    p += sum(punct(tm[x]) for x in s)
        if n:
            L.append(f"| {c} | {p / n:.2f} |")
    L.append("")
    # --- examples
    L += ["## Kept-token examples (k = 4; the first items in file order)", ""]
    by = {(r["item"], r["cond"], r["budget"]): r for r in rows}
    for i in items[:meta.get("n_examples", 8)]:
        inf = infos[i]
        L += [f"### {i}: Q \"{inf['question']}\" A \"{inf['answer']}\" ({'verbatim' if inf['verbatim'] else 'not verbatim'}"
              f", {inf['n_content']} content tokens)", ""]
        L += ["Dreamed questions: " + " / ".join(f"\"{q}\"" for q in inf["dreams"]["questions"]), ""]
        if inf.get("own_answers"):
            L += ["Own answers (S-loo): " + " / ".join(f"\"{o['answer']}\"" for o in inf["own_answers"]), ""]
        L += ["| selector | answer score | turn with the picks in bold brackets |", "|---|---|---|"]
        for c in ("S-dream", "S-dream-split", "S-attn", "S-recon", "S-loo", "last-k", "gold@k"):
            r = by.get((i, c, mk))
            if r:
                txt = mark(inf, r["sel"][0]).replace("\n", " ").replace("|", "\\|")
                L.append(f"| {c} | {r['answer_score']:.2f} | {txt} |")
        for c in ("nomem", "R-full"):
            r = by.get((i, c, None))
            L.append(f"| {c} | {r['answer_score']:.2f} | |")
        L.append("")
    # --- checks
    L += ["## N and checks", "",
          f"- Items: {len(items)} usable of {meta['data_meta']['n']} single-session-user items "
          f"({meta['data_meta']['n_abstention']} abstention items have no evidence turn and are not run); "
          f"{len(verb)} verbatim.",
          f"- R-full == T-full (the turn as a prior user turn): max |Δlogit| over items and scored positions "
          f"{max(infos[i]['identity_max_abs_diff'] for i in items):.2e} (tolerance 0.01).",
          "- Kept counts (head + exactly k) and content-only picks: asserted for every slice (the run would have "
          "failed).",
          f"- Paris: {meta['paris']['answer']!r}.",
          f"- Per-item seconds: median {np.median([infos[i]['seconds'] for i in items]):.1f}, max "
          f"{max(infos[i]['seconds'] for i in items):.1f}.", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out / 'report.md'}")
    for k, (ok, why) in R.items():
        print(f"{k}: {'HOLDS' if ok else 'does not hold'} ({why})")


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
