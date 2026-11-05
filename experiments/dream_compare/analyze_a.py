#!/usr/bin/env python
"""dream_compare Stage A analysis (PREREG.md, "Stage A: selectors"): teacher-forced and judged tables, the rules
H1-H4 and the paired S-loo vs gold@k test, per model (report_a.md, verdict_a.json) and across models (summary_a.md,
summary_a.json). numpy + the standard library only (no torch): reads <model dir>/tf/items/*.json.gz,
gen/items/*.jsonl.gz and judge/{orig,gptoss,gemma4}.jsonl.

  python -I experiments/dream_compare/run.py --stage a-analyze --model M --out RUN/M
  python -I experiments/dream_compare/run.py --stage a-summary --out RUN
"""

import gzip
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("dream_compare_baselines_an", HERE / "baselines.py")
J = _load("dream_compare_judges_an", HERE / "judges.py")
AG = J.AG                       # dream_select's analyze_gen (ci, paired_ci, gap_ratio, read_panel, REFUSAL_RE, ...)

MODELS = ("qwen25_15b", "qwen25_7b", "llama32_3b", "gemma4_26b_a4b", "granite42_30b")
LABELS = {"qwen25_15b": "Qwen2.5-1.5B", "qwen25_7b": "Qwen2.5-7B", "llama32_3b": "Llama-3.2-3B",
          "gemma4_26b_a4b": "Gemma-4-26B-A4B", "granite42_30b": "Granite-4.2-30B"}
FAMILY = {"gemma4_26b_a4b": "Google"}       # subject models that share a judge's family
JUDGES = J.JUDGES
ANCHORS_TF = ("nomem", "ctx", "T-full", "R-full", "R-content")
ANCHORS_GEN = ("nomem", "ctx", "R-full", "R-content")
TF_MIN_DEN = 0.5            # nats (dream_select)
GEN_MIN_DEN = 0.4           # H3's per-item judged gap denominator
EPS = AG.EPS
MAIN = B.MAIN_K


def read_gz_json(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def read_gz_lines(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_tf(d):
    infos, rows = {}, []
    for f in sorted((Path(d) / "tf" / "items").glob("*.json.gz")):
        r = read_gz_json(f)
        infos[r["info"]["question_id"]] = r["info"]
        rows += r["rows"]
    return infos, rows


def load_gen(d):
    G, I = [], {}
    for f in sorted((Path(d) / "gen" / "items").glob("*.jsonl.gz")):
        for r in read_gz_lines(f):
            if r.get("sec") == "info":
                I[r["item"]] = r
            else:
                G.append(r)
    return G, I


def load_votes(d):
    out = {}
    for j in JUDGES:
        p = Path(d) / "judge" / f"{j}.jsonl"
        if p.exists():
            out[j] = AG.read_panel(p)
    return out


# ----------------------------------------------------------------------------------------------- statistics


def nested(vals):
    """{item: value} -> {item: {0: value}} (None / non-finite dropped)."""
    return {i: {0: float(v)} for i, v in vals.items() if v is not None and math.isfinite(float(v))}


def ci(vals):
    return AG.ci(nested(vals))


def paired(a, b):
    return AG.paired_ci(nested(a), nested(b))


def median_ci(vals, n_boot=AG.N_BOOT, seed=AG.SEED):
    v = np.array([x for x in vals.values() if x is not None and math.isfinite(x)], dtype=float)
    if not len(v):
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = np.median(v[idx], 1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"mean": float(np.median(v)), "lo": float(lo), "hi": float(hi), "n": len(v)}


def holds(c):
    return c.get("n", 0) > 0 and c.get("lo") is not None and math.isfinite(c["lo"]) and c["lo"] > EPS


def direction(c):
    if not c.get("n") or c.get("lo") is None or not math.isfinite(c["lo"]):
        return "n/a"
    if c["lo"] > EPS:
        return "better"
    if c["hi"] < -EPS:
        return "worse"
    return "indistinguishable"


def fmt(s, nd=2):
    return AG.fmt(s, nd)


def num(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def key_of(c, b):
    return c if b is None else f"{c}@{b}"


# ----------------------------------------------------------------------------------------------- tables


def tf_values(infos, rows, items):
    """V[measure][cname][item] over `items`; gap closed per item (R-full - nomem >= 0.5 nats)."""
    V = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["item"] not in items:
            continue
        cn = key_of(r["cond"], r["budget"])
        for m in ("answer_score", "answer_mean", "refusal", "hit", "span_share", "entries"):
            if r.get(m) is not None:
                V[m][cn][r["item"]] = float(r[m])
    for cn, d in list(V["answer_score"].items()):
        for i, x in d.items():
            g = B.gap(x, V["answer_score"]["nomem"].get(i), V["answer_score"]["R-full"].get(i), TF_MIN_DEN)
            if g is not None:
                V["gap"][cn][i] = g
    return V


def gen_values(G, votes, items, family=None):
    """Per answer: each judge's label, the 3-judge majority (tie to the original), the majority without the subject's
    family (Gemma-4 subject only); per condition and item the mean over its answers. -> N[measure][cname][item]."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    miss = 0
    for r in G:
        if r["item"] not in items:
            continue
        v = {j: votes.get(j, {}).get(r["key"]) for j in JUDGES}
        full, excl = J.panel_label(v, subject_family=family)
        miss += full is None
        vals = {"maj": full, "maj_nofam": excl, "orig": v["orig"], "gptoss": v["gptoss"], "gemma4": v["gemma4"],
                "refusal": bool(AG.REFUSAL_RE.search(r["answer"] or "")), "loop": bool(r["loop"]),
                "has_answer": (AG.has_answer(r["answer"], items[r["item"]]["answer"])
                               if items[r["item"]]["verbatim"] else None),
                "greedy_maj": full if r["j"] == 0 else None}
        for m, x in vals.items():
            if x is not None:
                acc[m][r["cname"]][r["item"]].append(float(x))
    N = defaultdict(lambda: defaultdict(dict))
    for m, d in acc.items():
        for c, per in d.items():
            for i, xs in per.items():
                N[m][c][i] = sum(xs) / len(xs)
    return N, miss


def gen_gap(N, m, cname):
    return AG.gap_ratio(nested(N[m].get(cname, {})), nested(N[m].get("nomem", {})), nested(N[m].get("R-full", {})))


# ----------------------------------------------------------------------------------------------- rules


def rules(N, V, m, primary):
    """H1-H4 and the paired S-loo vs gold@k at k8 on measure m (maj or maj_nofam)."""
    k = MAIN
    g = lambda c: N[m].get(f"{c}@{k}", {})
    mean = lambda c: ci(g(c))["mean"] if g(c) else float("nan")
    out = {}
    # H1
    forms = [c for c in B.KVZIP_FORMS if g(c)]
    best = max(forms, key=lambda c: (mean(c), -B.KVZIP_FORMS.index(c))) if forms else None
    d1 = paired(g(best), g("S-recon")) if best else {"n": 0}
    d1g = paired(g("KVzip-g"), g("S-recon")) if g("KVzip-g") else {"n": 0}
    out["H1"] = {"holds": holds(d1), "kvzip_form": best, "ci": d1, "official_kvzip_g_ci": d1g,
                 "means": {c: mean(c) for c in forms + ["S-recon"]}}
    # H2
    per = {}
    for a, b in B.H2_PAIRS:
        d = paired(g(a), g(b)) if (g(a) and g(b)) else {"n": 0}
        per[f"{a} - {b}"] = {"holds": holds(d), "ci": d}
    n2 = sum(v["holds"] for v in per.values())
    out["H2"] = {"holds": n2 >= 2, "n_methods": n2, "per_method": per}
    # H3
    meth = {}
    for c in B.H3_METHODS:
        tf_g = {i: B.gap(V["answer_score"].get(f"{c}@{k}", {}).get(i), V["answer_score"]["nomem"].get(i),
                         V["answer_score"]["R-full"].get(i), TF_MIN_DEN) for i in primary}
        gen_g = {i: B.gap(N[m].get(f"{c}@{k}", {}).get(i), N[m].get("nomem", {}).get(i), N[m].get("R-full", {}).get(i),
                          GEN_MIN_DEN) for i in primary}
        dd = {i: tf_g[i] - gen_g[i] for i in primary if tf_g[i] is not None and gen_g[i] is not None}
        d = ci(dd)
        sc = lambda dct, it: [dct.get(i) for i in it]
        rom = None
        both = [i for i in primary if all(x is not None for x in (
            V["answer_score"].get(f"{c}@{k}", {}).get(i), V["answer_score"]["nomem"].get(i),
            V["answer_score"]["R-full"].get(i), N[m].get(f"{c}@{k}", {}).get(i), N[m].get("nomem", {}).get(i),
            N[m].get("R-full", {}).get(i)))]
        if both:
            mt = lambda dct: float(np.mean(sc(dct, both)))
            den_t = mt(V["answer_score"]["R-full"]) - mt(V["answer_score"]["nomem"])
            den_g = mt(N[m]["R-full"]) - mt(N[m]["nomem"])
            if den_t > 0 and den_g > 0:
                rom = ((mt(V["answer_score"][f"{c}@{k}"]) - mt(V["answer_score"]["nomem"])) / den_t
                       - (mt(N[m][f"{c}@{k}"]) - mt(N[m]["nomem"])) / den_g)
        meth[c] = {"holds": holds(d), "ci": d, "median": median_ci(dd), "ratio_of_means_diff": rom,
                   "n_tf_defined": sum(v is not None for v in tf_g.values()),
                   "n_gen_defined": sum(v is not None for v in gen_g.values())}
    n3 = sum(v["holds"] for v in meth.values())
    out["H3"] = {"passes": n3 >= 3, "n_methods": n3, "per_method": meth}
    # H4
    cand = [c for c in B.NON_ORACLE_BASELINES if g(c)]
    bb = max(cand, key=lambda c: (mean(c), -B.BASELINES.index(c))) if cand else None
    d4 = paired(g("S-loo"), g(bb)) if (bb and g("S-loo")) else {"n": 0}
    out["H4"] = {"best_baseline": bb, "best_mean": mean(bb) if bb else None, "s_loo_mean": mean("S-loo"),
                 "ci": d4, "direction": direction(d4)}
    # S-loo vs gold@k
    sg = {}
    for kk in ("k8", "k4"):
        a, b_ = N[m].get(f"S-loo@{kk}", {}), N[m].get(f"gold@k@{kk}", {})
        d = paired(a, b_) if (a and b_) else {"n": 0}
        sg[kk] = {"ci": d, "direction": direction(d)}
    out["S-loo vs gold@k"] = sg
    return out


def tf_rules(V, primary):
    """Teacher-forced analogues (answer score, k8; information only)."""
    k = MAIN
    a = lambda c: {i: V["answer_score"].get(f"{c}@{k}", {}).get(i) for i in primary}
    a = lambda c, a_=a: {i: x for i, x in a_(c).items() if x is not None}
    out = {}
    forms = [c for c in B.KVZIP_FORMS if a(c)]
    best = max(forms, key=lambda c: ci(a(c))["mean"]) if forms else None
    out["H1_tf"] = {"kvzip_form": best, "ci": paired(a(best), a("S-recon")) if best else {"n": 0}}
    out["H2_tf"] = {f"{x} - {y}": paired(a(x), a(y)) for x, y in B.H2_PAIRS}
    cand = [c for c in B.NON_ORACLE_BASELINES if a(c)]
    bb = max(cand, key=lambda c: ci(a(c))["mean"]) if cand else None
    out["H4_tf"] = {"best_baseline": bb, "ci": paired(a("S-loo"), a(bb)) if bb else {"n": 0}}
    return out


# ----------------------------------------------------------------------------------------------- report


def cond_names(budget):
    return [f"{c}@{budget}" for c in B.SELECTIONS]


def analyze(d, model=None):
    d = Path(d)
    model = model or d.name
    label = LABELS.get(model, model)
    family = FAMILY.get(model)
    infos, rows = load_tf(d)
    assert infos, f"no tf items under {d / 'tf'}"
    primary = {i for i, x in infos.items() if x["primary"]}
    items = {i: x for i, x in infos.items()}
    budgets = sorted({r["budget"] for r in rows if r["budget"] not in (None, "own")}, key=lambda b: B.BUDGETS.index(b))
    V = tf_values(infos, rows, primary)
    G, GI = load_gen(d)
    votes = load_votes(d)
    N, miss = gen_values(G, votes, items, family)
    Np, _ = gen_values(G, votes, {i: items[i] for i in primary}, family)
    keys = [r["key"] for r in G]
    cover = {j: sum(votes.get(j, {}).get(k) is not None for k in keys) for j in JUDGES}
    complete = bool(keys) and all(cover[j] == len(keys) for j in JUDGES)
    gen_b = sorted({r["budget"] for r in G if r["budget"] not in (None, "own")}, key=lambda b: B.BUDGETS.index(b))
    R = rules(Np, V, "maj", primary) if MAIN in gen_b else None
    R_nofam = rules(Np, V, "maj_nofam", primary) if (MAIN in gen_b and family) else None
    TR = tf_rules(V, primary) if MAIN in budgets else None
    # --- secondary sets
    sub = {"single-session-user (64)": {i for i, x in infos.items() if x["ssu"]},
           "preference (19)": {i for i, x in infos.items() if x["category"] == "single-session-preference"},
           "abstention (8)": {i for i, x in infos.items() if x["abstention"]}}
    cats = sorted({x["category"] for x in infos.values() if x["primary"]})
    Ns = {name: gen_values(G, votes, {i: items[i] for i in s}, family)[0] for name, s in sub.items()}
    Nc = {c: gen_values(G, votes, {i: items[i] for i in primary if infos[i]["category"] == c}, family)[0] for c in cats}
    # --- verdict
    gb = gen_b[-1] if gen_b else None
    verdict = {"model": model, "label": label, "n_items": len(infos), "n_primary": len(primary),
               "budgets_tf": budgets, "budgets_gen": gen_b, "n_answers": len(keys), "judges_present": sorted(votes),
               "coverage": cover, "labels_complete": complete, "n_majority_missing": miss, "rules": R,
               "rules_no_same_family": R_nofam, "tf_rules": TR,
               "correct_primary": {c: ci(Np["maj"].get(c, {})) for c in list(ANCHORS_GEN) + sum(
                   [cond_names(b) for b in gen_b], [])},
               "gap_closed_primary": {c: gen_gap(Np, "maj", c) for c in sum([cond_names(b) for b in gen_b], [])},
               "tf_gap_primary": {c: ci(V["gap"].get(c, {})) for b in budgets for c in cond_names(b)}}
    json.dump(verdict, open(d / "verdict_a.json", "w"), indent=1, default=str)
    # --- report
    L = [f"# dream_compare Stage A: selectors on {label} (LongMemEval oracle, one stored user turn)", "",
         "Pre-registration: `experiments/dream_compare/PREREG.md`, \"Stage A: selectors\". CIs: 95% percentile item "
         "bootstrap (2000 replicates, seed 7), paired on the same items. Rules: k = 8, generation, three-judge majority, "
         f"the primary set ({len(primary)} items: not abstention, not single-session-preference).", ""]
    if not complete:
        L += [f"**Labels incomplete.** Judges present: {', '.join(sorted(votes)) or 'none'}; labelled answers per judge: "
              + ", ".join(f"{j} {cover[j]}/{len(keys)}" for j in JUDGES)
              + f"; answers without a majority label: {miss}. Everything judged below uses the votes that exist.", ""]
    L += ["## Rules (k = 8, generation, majority)", ""]
    for tag, RR in (("three-judge majority", R), ("majority without the same-family judge (original + gpt-oss)", R_nofam)):
        if RR is None:
            continue
        h1, h2, h3, h4 = RR["H1"], RR["H2"], RR["H3"], RR["H4"]
        L += [f"**{tag}**", "",
              f"- **H1** (the real KVzip beats S-recon): **{'HOLDS' if h1['holds'] else 'does not hold'}**. Best KVzip "
              f"form {h1['kvzip_form']}: {h1['kvzip_form']} - S-recon {fmt(h1['ci'], 3)} (N {h1['ci'].get('n')}). The "
              f"official KVzip-g - S-recon: {fmt(h1['official_kvzip_g_ci'], 3)}.",
              f"- **H2** (per-head beats shared, >= 2 of 3 methods): **{'HOLDS' if h2['holds'] else 'does not hold'}** "
              f"({h2['n_methods']} of 3): " + "; ".join(f"{k} {fmt(v['ci'], 3)}" for k, v in h2["per_method"].items())
              + ".",
              f"- **H3** (licence gap: gap_tf - gap_gen > 0 for >= 3 of 4 methods): this model "
              f"**{'PASSES' if h3['passes'] else 'does not pass'}** ({h3['n_methods']} of 4): " + "; ".join(
                  f"{k} {fmt(v['ci'], 2)} (median {fmt(v['median'], 2)}; ratio of means {num(v['ratio_of_means_diff'])})"
                  for k, v in h3["per_method"].items()) + ".",
              f"- **H4** (S-loo vs the best non-oracle baseline, no direction): best baseline {h4['best_baseline']} "
              f"(mean {num(h4['best_mean'], 3)}), S-loo {num(h4['s_loo_mean'], 3)}: S-loo - best "
              f"{fmt(h4['ci'], 3)} -> **{h4['direction']}**.",
              "- **S-loo vs gold@k** (paired, verbatim items): " + "; ".join(
                  f"{kk} {fmt(v['ci'], 3)} ({v['direction']}, N {v['ci'].get('n')})"
                  for kk, v in RR["S-loo vs gold@k"].items()) + ".", ""]
    if TR:
        L += ["**Teacher-forced analogues (answer score, k = 8; information only).** H1: "
              f"{TR['H1_tf']['kvzip_form']} - S-recon {fmt(TR['H1_tf']['ci'])}; H2: " + "; ".join(
                  f"{k} {fmt(v)}" for k, v in TR["H2_tf"].items())
              + f"; H4: S-loo - {TR['H4_tf']['best_baseline']} {fmt(TR['H4_tf']['ci'])}.", ""]
    # generation tables
    for b in gen_b:
        L += [f"## Judged correctness, k = {b[1:]} (primary set)", "",
              "| condition | N | correct (majority) | " + ("no same-family | " if family else "")
              + "greedy only | orig | gpt-oss | Gemma-4 | gap closed | refusal | loop | has answer |",
              "|---|---|---|" + ("---|" if family else "") + "---|---|---|---|---|---|---|---|"]
        for c in list(ANCHORS_GEN) + ["gold"] + cond_names(b):
            if c not in Np["maj"] and c not in Np["refusal"]:
                continue
            gp = gen_gap(Np, "maj", c)
            L.append(f"| {c} | {len(Np['refusal'].get(c, {}))} | {fmt(ci(Np['maj'].get(c, {})))} | "
                     + (f"{fmt(ci(Np['maj_nofam'].get(c, {})))} | " if family else "")
                     + f"{fmt(ci(Np['greedy_maj'].get(c, {})))} | {fmt(ci(Np['orig'].get(c, {})))} | "
                     f"{fmt(ci(Np['gptoss'].get(c, {})))} | {fmt(ci(Np['gemma4'].get(c, {})))} | "
                     f"{fmt(gp) if gp.get('mean') is not None else 'n/a'} | {fmt(ci(Np['refusal'].get(c, {})))} | "
                     f"{fmt(ci(Np['loop'].get(c, {})))} | {fmt(ci(Np['has_answer'].get(c, {})))} |")
        L.append("")
    # teacher-forced tables
    for b in budgets:
        L += [f"## Teacher-forced, k = {b[1:]} (primary set)", "",
              "| condition | N | answer score (nats) | per-token mean | gap closed mean | gap closed median | refusal mass | "
              "hit (verbatim) | stored entries |", "|---|---|---|---|---|---|---|---|---|"]
        for c in (list(ANCHORS_TF) if b == budgets[0] else []) + cond_names(b):
            a = V["answer_score"].get(c, {})
            if not a:
                continue
            ent = V["entries"].get(c, {})
            L.append(f"| {c} | {len(a)} | {fmt(ci(a))} | {fmt(ci(V['answer_mean'].get(c, {})))} | "
                     f"{fmt(ci(V['gap'].get(c, {})))} | {fmt(median_ci(V['gap'].get(c, {})))} | "
                     f"{fmt(ci(V['refusal'].get(c, {})))} | {fmt(ci(V['hit'].get(c, {})))} | "
                     f"{'' if not ent else int(np.mean(list(ent.values())))} |")
        L.append("")
    # secondary
    if gb:
        L += [f"## Secondary sets (judged correctness, majority, k = {gb[1:]})", "",
              "| condition | " + " | ".join(list(sub) + cats) + " |", "|---|" + "---|" * (len(sub) + len(cats))]
        for c in list(ANCHORS_GEN) + cond_names(gb):
            cells = [fmt(ci(Ns[s]["maj"].get(c, {}))) for s in sub] + [fmt(ci(Nc[c_]["maj"].get(c, {}))) for c_ in cats]
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L += ["", "Abstention items are judged with LongMemEval's abstention prompt (correct = the reply says it is "
              "not known); preference items with its preference rubric prompt.", ""]
    # costs and checks
    tim = defaultdict(list)
    for x in infos.values():
        for k_, v in (x.get("timing") or {}).items():
            if not (x["reused"] and k_ in ("dreamer", "S-loo")):
                tim[k_].append(v)
    rc = defaultdict(list)
    for x in infos.values():
        for k_, v in (x.get("reuse_check") or {}).items():
            rc[k_].append(v)
    L += ["## Selection cost and checks", "",
          "- Seconds per item (fresh computations only): " + ", ".join(
              f"{k_} {np.mean(v):.2f}" for k_, v in sorted(tim.items())) + ".",
          f"- Identity (asserted <= 0.01): R-full == T-full max {max(x['identity_rfull_tfull'] for x in infos.values()):.1e}; "
          f"per-head keep-all == R-full max {max(x['identity_keep_all'] for x in infos.values()):.1e}; the same picks "
          f"in every head == the splice max {max(x['identity_same_picks_vs_splice'] for x in infos.values()):.1e} "
          f"({len(infos)} items, teacher-forced)."
          + (f" Generation: R-full == T-full max {max(v['identity_rfull_tfull'] for v in GI.values()):.1e}, keep-all "
             f"max {max(v['identity_keep_all'] for v in GI.values()):.1e} ({len(GI)} items)." if GI else ""),
          f"- Stored entries per position E = {next(iter(infos.values()))['entries_per_position']}; every slice holds "
          "k x E entries besides the head (asserted).",
          f"- Items reusing dream_select addendum 1's dreams and scores: {sum(x['reused'] for x in infos.values())}."
          + ("" if not rc else " Recomputed vs saved (information only): " + "; ".join(
              f"{k_} max rel. diff {max(v['rel'] for v in vs if v['rel'] is not None):.1e}, same top-4 picks "
              f"{sum(bool(v['same_top4']) for v in vs)}/{len(vs)}" for k_, vs in rc.items()) + "."),
          f"- S-loo missing (content > 200 tokens): {sum(not x['loo_run'] for x in infos.values())} items; dreamer "
          f"fallbacks: {sum(bool(x['dreams'].get('fallback')) for x in infos.values())}.",
          f"- Judges: labelled answers " + ", ".join(f"{j} {cover[j]}/{len(keys)}" for j in JUDGES) + "; agreement "
          + ", ".join(f"{k_} {v['agree']:.3f}" for k_, v in AG.agreement(votes, keys).items()) + ".", ""]
    (d / "report_a.md").write_text("\n".join(L) + "\n")
    print(f"wrote {d / 'report_a.md'}" + (f": H1 {R['H1']['holds']}, H2 {R['H2']['holds']}, H3 {R['H3']['passes']}, "
                                          f"H4 {R['H4']['direction']}" if R else ""))
    return verdict


def summary(root, models=None):
    root = Path(root)
    models = models or list(MODELS)
    L = ["# dream_compare Stage A: selectors, across models", "",
         "Per model: k = 8, generation, three-judge majority, primary set. H3 holds if >= 3 of the models with results "
         "pass (PREREG).", "",
         "| model | H1 KVzip > S-recon | H2 per-head > shared | H3 licence gap (methods) | H4 S-loo vs best baseline | "
         "S-loo - gold@k (k8) | correct: nomem / R-full / S-loo / best baseline | labels complete |",
         "|---|---|---|---|---|---|---|---|"]
    res = {}
    n1 = n2 = n3 = n_done = 0
    for m in models:
        p = root / m / "verdict_a.json"
        v = json.load(open(p)) if p.exists() else None
        res[m] = v
        if not v or not v.get("rules"):
            L.append(f"| {LABELS.get(m, m)} | missing | | | | | | |")
            continue
        R = v["rules"]
        n_done += 1
        n1 += R["H1"]["holds"]
        n2 += R["H2"]["holds"]
        n3 += R["H3"]["passes"]
        c = v["correct_primary"]
        bb = R["H4"]["best_baseline"]
        cell = " / ".join(f"{c[k]['mean']:.2f}" if (k in c and c[k].get("mean") is not None) else "n/a"
                          for k in ("nomem", "R-full", "S-loo@k8", f"{bb}@k8"))
        L.append(f"| {LABELS.get(m, m)} | {'HOLDS' if R['H1']['holds'] else 'no'} ({R['H1']['kvzip_form']}) | "
                 f"{'HOLDS' if R['H2']['holds'] else 'no'} ({R['H2']['n_methods']}/3) | "
                 f"{'passes' if R['H3']['passes'] else 'no'} ({R['H3']['n_methods']}/4) | {R['H4']['direction']} "
                 f"({bb}) | {fmt(R['S-loo vs gold@k']['k8']['ci'], 3)} | {cell} | {v['labels_complete']} |")
    h3 = "HOLDS" if n3 >= 3 else "does not hold"
    L += ["", f"**H3 across models (pre-registered):** {n3} of {n_done} models with results pass: **{h3}** "
          "(needs >= 3).",
          f"Counts over models with results: H1 holds on {n1} of {n_done}, H2 on {n2} of {n_done}.", ""]
    g = res.get("gemma4_26b_a4b")
    if g and g.get("rules_no_same_family"):
        R = g["rules_no_same_family"]
        L += [f"Gemma-4 without the same-family judge: H1 {R['H1']['holds']}, H2 {R['H2']['holds']}, H3 "
              f"{R['H3']['passes']} ({R['H3']['n_methods']}/4), H4 {R['H4']['direction']}.", ""]
    (root / "summary_a.md").write_text("\n".join(L) + "\n")
    json.dump({"H3": h3, "n_pass_H3": n3, "n_H1": n1, "n_H2": n2, "n_done": n_done,
               "models": {m: (v or {}).get("rules") for m, v in res.items()}},
              open(root / "summary_a.json", "w"), indent=1, default=str)
    print(f"wrote {root / 'summary_a.md'}: H3 {h3} ({n3} of {n_done})")
    return h3
