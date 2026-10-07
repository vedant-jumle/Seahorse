#!/usr/bin/env python
"""Seahorse core_v1 analysis: every table, CSV and claim decision from the saved files (no GPU, no model).

  python experiments/core_v1/analyze.py --run <RUN>/qwen35_2b            # one model -> <RUN>/qwen35_2b/analysis
  python experiments/core_v1/analyze.py --combine <RUN>/qwen35_2b <RUN>/qwen35_9b --out <RUN>/analysis
                                                                         # C7 table + the hand-audit sample

Inputs per model run dir: prep/{items.json, config.json, checks.json, lens.jsonl.gz, facts_analytic.jsonl.gz},
gen/items/*.jsonl.gz (gen / gate / yn rows), gen/unrel/*.jsonl.gz (unrel / ugate / kl rows),
score/{coherence.jsonl.gz, judge.jsonl.gz} (optional: judge-based numbers are nan without them).

Conventions (PREREG.md): primary rows = seed set 1 (1 greedy + 4 samples per prompt); a cell = one (item,
prompt); a group number = the macro mean over items of the mean over the item's prompts; 95% CIs from the
two-level bootstrap (items, then prompts within items), paired across conditions (seahorse.stats).
"""

import argparse
import csv
import gzip
import importlib.util
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from seahorse.bench import label, lexicon_hits
from seahorse.stats import boot_ci, macro_mean, paired_diff, wrong_way

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("seahorse_xlayer_v1_xl", HERE.parent / "xlayer_v1" / "xl.py")
xl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xl)

NAN = float("nan")
PREF_GROUPS = ("leanings", "dislikes", "one_of_many")
STOP = set("not unknown a an the your my their his her something what unclear mentioned given provided known "
           "specified stated available that this it likely probably actually still also just in called named one no "
           "is was be to of for and or but if".split())
CTX_RULE = 0.3       # PREREG: ctx lean - nomem lean < 0.3 -> flagged
LOOP_MATCH = 0.10    # PREREG: matched loop rate for the mechanism comparison
SHORT_WORDS = 40


def nanmean(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else NAN


def read_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt(x, spec="+.3f"):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def ci_str(t, spec="+.3f"):
    p, lo, hi, n = t
    return f"{fmt(p, spec)} [{fmt(lo, spec)}, {fmt(hi, spec)}]"


def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------ facts


def wrong_fn(it):
    """think_v1.wrong_fn: a foil / the counter's value mentioned, or a non-target value after the measure
    phrase ("dog's name is Max")."""
    m = it["measure"]
    tgt = m["target"].strip().lower()
    exp_w = set(re.findall(r"[\w']+", it["experience"].lower()))
    wrong = {f.strip().lower() for f in m["foils"]} | {w for w in re.findall(r"[\w']+", it["counter"].lower())
                                                       if w not in exp_w}
    core = m["prefix"].split()
    core = core[1:] if core[0].lower() in ("your", "you") else core
    rx = r"\s+".join(re.escape(w) for w in core).replace("favourite", "favou?rite").replace("colour", "colou?r")
    rx = re.compile(rx.replace("'", "['’]") + r"\s+[*\"'_`]*([A-Za-z][A-Za-z-]*)", re.I)

    def f(text):
        hits = {w for w in wrong if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text, re.I)}
        hits |= {v.lower() for v in rx.findall(text) if v.lower() not in STOP and v.lower() != tgt}
        return sorted(hits)
    return f


# ------------------------------------------------------------------------- load


class Run:
    def __init__(self, d):
        d = Path(d)
        self.dir, self.name = d, d.name
        self.items = json.load(open(d / "prep" / "items.json"))
        self.prep_cfg = json.load(open(d / "prep" / "config.json"))
        self.cfg = self.prep_cfg["cfg"]
        self.checks = json.load(open(d / "prep" / "checks.json"))
        self.lens = read_gz(d / "prep" / "lens.jsonl.gz")
        self.ftf = read_gz(d / "prep" / "facts_analytic.jsonl.gz")
        rows = []
        for f in sorted((d / "gen" / "items").glob("*.jsonl.gz")) + sorted((d / "gen" / "unrel").glob("*.jsonl.gz")):
            rows += read_gz(f)
        by = defaultdict(list)
        for r in rows:
            by[r["sec"]].append(r)
        self.gen, self.gate, self.yn = by["gen"], by["gate"], by["yn"]
        self.unrel, self.ugate, self.kl = by["unrel"], by["ugate"], by["kl"]
        sc = d / "score"
        self.nll = {r["key"]: r for r in read_gz(sc / "coherence.jsonl.gz")} if (sc / "coherence.jsonl.gz").exists() else {}
        self.judge = {r["key"]: r for r in read_gz(sc / "judge.jsonl.gz")} if (sc / "judge.jsonl.gz").exists() else {}
        # the judge counts only if it parsed >= 80% of its outputs (PREREG); else the lexicon lean is used
        self.judge_parse_rate = nanmean([float(bool(j.get("ok"))) for j in self.judge.values()]) if self.judge else NAN
        self.has_judge = bool(self.judge) and self.judge_parse_rate >= 0.8
        self.ids = {g: [i for i, it in self.items.items() if it["group"] == g] for g in PREF_GROUPS + ("facts",)}
        self.wrong = {i: wrong_fn(it) for i, it in self.items.items() if it["group"] == "facts"}
        for r in self.gen:
            self._metrics(r)
        self.flags = self._ctx_flags()

    def _metrics(self, r):
        it = self.items[r["item"]]
        a = r["answer"]
        r["words"] = len(a.split())
        r["short"] = r["words"] < SHORT_WORDS
        nl = self.nll.get(r["key"])
        r["nll"] = nl["nll"] if nl else NAN
        j = self.judge.get(r["key"])
        jok = bool(j and j.get("ok"))
        r["j_ok"] = jok
        r["j_coh"] = (1.0 if j["coherent"] else 0.0) if jok else NAN
        r["j_self"] = (1.0 if j["self_claim"] else 0.0) if jok else NAN
        r["j_incoh"] = 1.0 - r["j_coh"] if jok else NAN
        if it["group"] == "facts":
            tgt = it["measure"]["target"].strip()
            r["hit"] = xl.has_word(a, tgt)
            r["clean"] = r["hit"] and not r["loop"]
            r["confab"] = bool(self.wrong[r["item"]](a))
            r["self_rx"] = xl.self_attr(a, tgt)
            r["j_use"] = (1.0 if j["uses_fact"] else 0.0) if jok else NAN
            r["j_wrong"] = (1.0 if j["wrong_value"] else 0.0) if jok else NAN
            r["j_clean"] = r["j_use"] * r["j_coh"] if jok else NAN
            r["mention_unclean"] = 1.0 if (r["hit"] and not (jok and j["uses_fact"] and j["coherent"])) else 0.0
        else:
            lab = label(a, it["lexicon"])
            r["lex"] = {"consistent": 1.0, "inconsistent": -1.0, "neutral": 0.0}[lab]
            r["lex_clean"] = NAN if r["loop"] else r["lex"]
            if jok:
                d = {"toward": 1.0, "away": -1.0, "neutral": 0.0}[j["direction"]]
                r["j_dir"] = d
                r["j_lean"] = d * r["j_coh"]  # PRIMARY: a coherent recommendation toward (+1) / away (-1)
            else:
                r["j_dir"] = r["j_lean"] = NAN

    # ------------------------------------------------------------- cells

    def cells(self, metric, cond, items, kinds=None, seedset=1, rows=None):
        """{item: {pi: mean of metric over the cell's rows}} for one condition."""
        d = defaultdict(lambda: defaultdict(list))
        for r in rows if rows is not None else self.gen:
            if r["cond"] != cond or r["item"] not in items or r["seedset"] != seedset:
                continue
            if kinds and r["kind"] not in kinds:
                continue
            v = r.get(metric)
            if isinstance(v, bool):
                v = float(v)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            d[r["item"]][r["pi"]].append(v)
        return {i: {p: sum(v) / len(v) for p, v in ps.items()} for i, ps in d.items()}

    def stat(self, metric, cond, items, kinds=None, seedset=1):
        return boot_ci(self.cells(metric, cond, items, kinds, seedset), self.cfg_boot(), self.seed())

    def gain(self, metric, cond, items, base="nomem", kinds=None):
        """Paired gain cond - base over the (item, prompt) cells (seed set 1), with its bootstrap CI."""
        return boot_ci(self.gain_cells(metric, cond, items, base, kinds), self.cfg_boot(), self.seed())

    def gain_cells(self, metric, cond, items, base="nomem", kinds=None):
        return paired_diff(self.cells(metric, cond, items, kinds), self.cells(metric, base, items, kinds))

    def cfg_boot(self):
        return 2000

    def seed(self):
        return int(self.cfg["seeds"]["bootstrap"])

    def conds(self, mech=False):
        c = ["nomem", "ctx"] + [f"mem@{d:g}" for d in self.cfg["doses"]]
        a = self.cfg["control_alpha"]
        c += [f"rand@{a:g}", f"swap@{a:g}", f"placebo@{a:g}", f"gate_on@{a:g}"]
        if mech:
            c += [f"{r}@{d:g}" for r in ("centroid", "without") for d in self.cfg["mech_doses"]]
        return c

    def _ctx_flags(self):
        """PREREG exclusion rule: a preference item whose ctx lean - nomem lean < 0.3 on this model is flagged
        (reported separately, never dropped silently). Lean = the judge lean (j_lean) when the judge is
        available; the lexicon lean_clean otherwise (and always reported)."""
        out = {}
        for g in PREF_GROUPS:
            for i in self.ids[g]:
                o = {"item": i, "group": g}
                for m in ("j_lean", "lex_clean", "lex"):
                    c = self.cells(m, "ctx", [i])
                    n = self.cells(m, "nomem", [i])
                    o[f"ctx_{m}"], o[f"nomem_{m}"] = macro_mean(c), macro_mean(n)
                    o[f"d_{m}"] = o[f"ctx_{m}"] - o[f"nomem_{m}"]
                key = "d_j_lean" if self.has_judge else "d_lex_clean"
                o["rule_metric"] = key
                o["flagged"] = bool(not (o[key] >= CTX_RULE))  # nan counts as flagged
                out[i] = o
        return out

    def kept(self, group):
        return [i for i in self.ids[group] if not self.flags.get(i, {}).get("flagged")]


# ------------------------------------------------------------------- analyses


def pref_table(R, group, items, mech=False):
    """Per condition: macro means (CI) of the main preference measures and the gains vs nomem."""
    rows = []
    for c in R.conds(mech):
        o = {"model": R.name, "group": group, "n_items": len(items), "cond": c}
        for m in ("j_lean", "j_dir", "lex", "lex_clean", "loop", "j_incoh", "j_self", "nll", "short", "n_tok"):
            p, lo, hi, n = R.stat(m, c, items)
            o[m], o[m + "_lo"], o[m + "_hi"] = p, lo, hi
        if c != "nomem":
            for m in ("j_lean", "lex_clean", "loop", "j_incoh", "nll"):
                p, lo, hi, n = R.gain(m, c, items)
                o["d_" + m], o["d_" + m + "_lo"], o["d_" + m + "_hi"] = p, lo, hi
            ww = wrong_way(R.gain_cells("j_lean", c, items), +1, 0.0)
            o["wrong_items"], o["wrong_prompts"] = ww
        rows.append(o)
    return rows


def fact_table(R, items):
    rows = []
    for c in R.conds():
        o = {"model": R.name, "group": "facts", "n_items": len(items), "cond": c}
        for m, kinds in (("j_clean", ["use"]), ("j_clean", ["related"]), ("clean", ["use"]), ("clean", ["related"]),
                         ("hit", ["use"]), ("hit", ["related"]), ("loop", None), ("j_incoh", None), ("j_self", ["related"]),
                         ("j_wrong", None), ("confab", None), ("self_rx", ["related"]), ("mention_unclean", None),
                         ("nll", None), ("short", None)):
            name = f"{m}_{kinds[0] if kinds else 'all'}"
            p, lo, hi, n = R.stat(m, c, items, kinds)
            o[name], o[name + "_lo"], o[name + "_hi"] = p, lo, hi
        rows.append(o)
    return rows


def yn_table(R, group, items):
    """Balanced yes/no (greedy): accY, accN, bal; margin gains dmY, dmN vs nomem; their mean."""
    by = defaultdict(lambda: defaultdict(list))
    for r in R.yn:
        if r["item"] in items:
            by[r["cond"]][r["item"]].append(r)
    base = {(r["item"], r["prompt"]): r for r in R.yn if r["cond"] == "nomem"}
    out = []
    for c in R.conds(group != "facts" and group != "leanings"):
        if c not in by:
            continue
        per_item_bal, dY, dN, dB = {}, defaultdict(dict), defaultdict(dict), {}
        accY, accN = {}, {}
        for i, rs in by[c].items():
            y = [r for r in rs if r["consistent"] == "Yes"]
            n = [r for r in rs if r["consistent"] == "No"]
            aY = nanmean([float(r["yn"] == "yes") for r in y])
            aN = nanmean([float(r["yn"] == "no") for r in n])
            accY[i], accN[i] = {0: aY}, {0: aN}
            per_item_bal[i] = {0: (aY + aN) / 2}
            for r in y:
                dY[i][r["prompt"]] = r["margin"] - base[(i, r["prompt"])]["margin"]
            for r in n:
                dN[i][r["prompt"]] = r["margin"] - base[(i, r["prompt"])]["margin"]
            dB[i] = {0: (nanmean(dY[i].values()) + nanmean(dN[i].values())) / 2}
        nb = {i: {0: (nanmean([float(r["yn"] == "yes") for r in by["nomem"][i] if r["consistent"] == "Yes"])
                      + nanmean([float(r["yn"] == "no") for r in by["nomem"][i] if r["consistent"] == "No"])) / 2}
              for i in by["nomem"]}
        sd = R.seed()
        o = {"model": R.name, "group": group, "cond": c, "n_items": len(by[c])}
        for k, v in (("accY", accY), ("accN", accN), ("bal", per_item_bal)):
            o[k], o[k + "_lo"], o[k + "_hi"], _ = boot_ci(v, 2000, sd)
        if c != "nomem":
            o["d_bal"], o["d_bal_lo"], o["d_bal_hi"], _ = boot_ci(paired_diff(per_item_bal, nb), 2000, sd)
            for k, v in (("dmY", dY), ("dmN", dN), ("dm_bal", dB)):
                o[k], o[k + "_lo"], o[k + "_hi"], _ = boot_ci(v, 2000, sd)
        out.append(o)
    return out


def selectivity(R, items_all):
    """C1: unrelated answers token-identical to nomem (greedy + samples; shortcut rows are identical by
    construction), gate false-fire, KL(nomem || memory) along the nomem greedy answer, contamination."""
    nomem = {(r["uk"], r["j"]): r["answer"] for r in R.unrel if r["cond"] == "nomem"}
    rows = []
    conds = sorted({r["cond"] for r in R.unrel if r["cond"] != "nomem"})
    for g in PREF_GROUPS + ("facts",):
        items = R.ids[g]
        for c in conds:
            same, cont, kl = defaultdict(dict), defaultdict(dict), defaultdict(dict)
            acc = defaultdict(lambda: defaultdict(list))
            for r in R.unrel:
                if r["cond"] != c or r["item"] not in items:
                    continue
                it = R.items[r["item"]]
                ans = nomem[(r["uk"], r["j"])] if r["shortcut"] else r["answer"]
                if it["group"] == "facts":
                    w = xl.has_word(ans, it["measure"]["target"]) - xl.has_word(nomem[(r["uk"], r["j"])], it["measure"]["target"])
                else:
                    w = float(lexicon_hits(ans, it["lexicon"])["consistent"] > 0) - float(
                        lexicon_hits(nomem[(r["uk"], r["j"])], it["lexicon"])["consistent"] > 0)
                acc[r["item"]][r["uk"]].append((float(r["same"]), w))
            for i, ps in acc.items():
                for uk, vs in ps.items():
                    same[i][uk] = nanmean([v[0] for v in vs])
                    cont[i][uk] = nanmean([v[1] for v in vs])
            for r in R.kl:
                if r["cond"] == c and r["item"] in items:
                    kl[r["item"]][r["uk"]] = r["kl"]
            if not same:
                continue
            sd = R.seed()
            o = {"model": R.name, "group": g, "cond": c, "n_items": len(same)}
            o["unrel_same"], o["unrel_same_lo"], o["unrel_same_hi"], _ = boot_ci(same, 2000, sd)
            o["contam"], o["contam_lo"], o["contam_hi"], _ = boot_ci(cont, 2000, sd)
            o["kl"], o["kl_lo"], o["kl_hi"], _ = boot_ci(kl, 2000, sd)
            rows.append(o)
    fire = []
    for g in PREF_GROUPS + ("facts",):
        d = defaultdict(dict)
        pos = defaultdict(dict)
        for r in R.ugate:
            if r["item"] in R.ids[g]:
                d[r["item"]][r["uk"]] = float(r["open_any"])
                pos[r["item"]][r["uk"]] = nanmean([v["open_frac"] for v in r["layers"].values()])
        p, lo, hi, n = boot_ci(d, 2000, R.seed())
        fire.append({"model": R.name, "group": g, "false_fire_prompts": p, "lo": lo, "hi": hi, "n_items": n,
                     "false_fire_positions": macro_mean(pos)})
    return rows, fire


def mechanism(R):
    """C5: per mechanism group, reference x dose: loop, j_lean, lex_clean and gains vs nomem; and the matched
    loop-rate comparison (per reference the highest dose with loop <= 10%)."""
    rows, matched = [], []
    groups = {"dislikes": R.kept("dislikes"), "one_of_many": R.kept("one_of_many"),
              "leanings(mech)": [i for i in R.kept("leanings") if R.items[i]["mech"]]}
    for g, items in groups.items():
        if not items:
            continue
        for ref in ("opposite", "centroid", "without"):
            best = None
            for d in R.cfg["mech_doses"]:
                c = f"mem@{d:g}" if ref == "opposite" else f"{ref}@{d:g}"
                o = {"model": R.name, "group": g, "ref": ref, "alpha": d, "cond": c, "n_items": len(items)}
                for m in ("loop", "j_lean", "lex_clean", "j_incoh"):
                    o[m] = R.stat(m, c, items)[0]
                for m in ("j_lean", "lex_clean"):
                    p, lo, hi, _ = R.gain(m, c, items)
                    o["d_" + m], o["d_" + m + "_lo"], o["d_" + m + "_hi"] = p, lo, hi
                o["wrong_items"], o["wrong_prompts"] = wrong_way(R.gain_cells("j_lean", c, items), +1)
                rows.append(o)
                if not (o["loop"] > LOOP_MATCH):
                    best = o
            m = dict(best) if best else dict(next(r for r in rows if r["group"] == g and r["ref"] == ref))
            m["matched"] = best is not None
            matched.append(m)
    return rows, matched


def lens_tables(R):
    nb = R.prep_cfg["n_blocks"]
    facts = defaultdict(list)
    prefs = defaultdict(list)
    for r in R.lens:
        if r["group"] == "facts" and r["ref"] == "plain":
            facts[r["layer"]].append(r)
        elif r["group"] != "facts":
            prefs[(r["group"], r["ref"], r["layer"])].append(r["lex_gain"])
    ft = []
    for l in range(nb):
        rs = facts.get(l, [])
        if not rs:
            continue
        ranks = sorted(r["rank_t"] for r in rs)
        med = ranks[len(ranks) // 2] if len(ranks) % 2 else (ranks[len(ranks) // 2 - 1] + ranks[len(ranks) // 2]) / 2
        ft.append({"model": R.name, "layer": l, "ltype": rs[0]["ltype"], "depth": round((l + 1) / nb, 3),
                   "median_rank": med, "top10_share": nanmean([float(r["rank_t"] <= 10) for r in rs]),
                   "mean_z": nanmean([r["z_t"] for r in rs]), "mean_margin": nanmean([r["margin"] for r in rs]),
                   "rel_dose": nanmean([r["rel_dose"] for r in rs])})
    pt = []
    for (g, ref, l), v in sorted(prefs.items()):
        pt.append({"model": R.name, "group": g, "ref": ref, "layer": l, "depth": round((l + 1) / nb, 3),
                   "lex_gain": nanmean(v)})
    return ft, pt


def facts_tf(R):
    """Fact analytic at the measure prefix (prep): Delta log P(target), Delta specificity vs nomem."""
    base = {(r["item"], r["kind"]): r for r in R.ftf if r["cond"] == "nomem"}
    out = []
    for c in sorted({r["cond"] for r in R.ftf}, key=lambda x: R.conds().index(x) if x in R.conds() else 99):
        dl, ds, top = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        for r in R.ftf:
            if r["cond"] == c:
                b = base[(r["item"], r["kind"])]
                dl[r["item"]][r["kind"]] = r["lp_t"] - b["lp_t"]
                ds[r["item"]][r["kind"]] = r["spec"] - b["spec"]
                top[r["item"]][r["kind"]] = float(r["rank"] == 1)
        sd = R.seed()
        o = {"model": R.name, "cond": c}
        for k, v in (("d_lp_t", dl), ("d_spec", ds), ("top1", top)):
            o[k], o[k + "_lo"], o[k + "_hi"], _ = boot_ci(v, 2000, sd)
        out.append(o)
    return out


def seed2(R, group, items):
    """Robustness: the seed-2 samples (4 per prompt) vs the seed-1 samples for the seed-2 conditions."""
    out = []
    for c in R.cfg["seed2"]:
        o = {"model": R.name, "group": group, "cond": c}
        for s in (1, 2):
            rows = [r for r in R.gen if r["j"] > 0]
            for m in (("j_lean", "lex_clean") if group != "facts" else ("j_clean", "clean")):
                o[f"{m}_s{s}"] = macro_mean(R.cells(m, c, items, ["use"] if group == "facts" else None, s, rows))
        out.append(o)
    return out


# ---------------------------------------------------------------------- claims


def excl0(t):
    return not (t[1] <= 0 <= t[2])


def claims(R, T):
    """The PREREG decision rules, evaluated. Returns {claim: {decision, evidence}}."""
    a = R.cfg["control_alpha"]
    C = {}
    # C1 selectivity
    sel = [r for r in T["selectivity"]]
    gated = [r for r in sel if not r["cond"].startswith("gate_on")]
    pooled_same = nanmean([r["unrel_same"] for r in gated])
    go = [r for r in sel if r["cond"].startswith("gate_on")]
    m2 = [r for r in sel if r["cond"] == "mem@2"]
    d = {"unrel_same_gated_mean": pooled_same, "unrel_same_mem2": nanmean([r["unrel_same"] for r in m2]),
         "unrel_same_gate_on": nanmean([r["unrel_same"] for r in go]),
         "kl_mem2": nanmean([r["kl"] for r in m2]), "kl_gate_on": nanmean([r["kl"] for r in go]),
         "false_fire_prompts": nanmean([r["false_fire_prompts"] for r in T["false_fire"]])}
    ok = (all(r["unrel_same"] >= 0.95 for r in gated) and d["unrel_same_gate_on"] < d["unrel_same_mem2"]
          and all(r["unrel_same_hi"] < mr["unrel_same_lo"] for r in go for mr in m2 if mr["group"] == r["group"]))
    C["C1"] = {"holds": ok, "evidence": d,
               "rule": "every gated condition's unrel_same >= 0.95 (per group); gate_on's unrel_same lower than "
                       "mem@2's with non-overlapping 95% CIs in every group"}
    # C2 leanings
    lt = {r["cond"]: r for r in T["pref"] if r["group"] == "leanings"}
    g1, g2 = lt.get("mem@1", {}), lt.get("mem@2", {})
    ctrl = {k: lt.get(f"{k}@{a:g}", {}) for k in ("rand", "swap", "placebo")}
    ok = all(x.get("d_j_lean_lo", NAN) > 0 for x in (g1, g2))
    ctrl_ok = {}
    for k, x in ctrl.items():
        diff = R.gain("j_lean", "mem@2", R.kept("leanings"), base=f"{k}@{a:g}")
        ctrl_ok[k] = {"gain": x.get("d_j_lean"), "mem2_minus_control": diff,
                      "ok": (x.get("d_j_lean", NAN) <= g2.get("d_j_lean", NAN) / 3) and diff[1] > 0}
    C["C2"] = {"holds": ok and all(v["ok"] for v in ctrl_ok.values()),
               "evidence": {"gain_mem1": (g1.get("d_j_lean"), g1.get("d_j_lean_lo"), g1.get("d_j_lean_hi")),
                            "gain_mem2": (g2.get("d_j_lean"), g2.get("d_j_lean_lo"), g2.get("d_j_lean_hi")),
                            "controls": ctrl_ok, "items": R.kept("leanings"),
                            "flagged": [i for i in R.ids["leanings"] if R.flags[i]["flagged"]]},
               "rule": "leanings (ctx-rule kept): judge-lean gain vs nomem > 0 with CI excluding 0 at alpha 1 and 2; "
                       "each control's gain <= 1/3 of mem@2's and the paired CI of (mem@2 - control) excludes 0"}
    # C3 facts
    ft = {r["cond"]: r for r in T["facts"]}
    ctx = ft.get("ctx", {}).get("j_clean_use", NAN)
    per = {}
    ok = True
    for dd in R.cfg["doses"]:
        c = f"mem@{dd:g}"
        x = ft.get(c, {})
        diff = R.gain("j_clean", "ctx", R.ids["facts"], base=c, kinds=["use"])
        far = x.get("j_clean_use", NAN) <= 0.5 * ctx and diff[1] > 0
        per[c] = {"use_judge_clean": x.get("j_clean_use"), "ctx_minus": diff, "far_below": far,
                  "mention_unclean": x.get("mention_unclean_all"), "loop": x.get("loop_all"),
                  "incoherent": x.get("j_incoh_all")}
        ok &= far
    nm = ft.get("nomem", {})
    rise = any(per[f"mem@{dd:g}"]["mention_unclean"] > nm.get("mention_unclean_all", NAN) for dd in R.cfg["doses"]
               if dd >= 2)
    C["C3"] = {"holds": ok and rise, "evidence": {"ctx_use_judge_clean": ctx, "per_dose": per,
                                                  "nomem_mention_unclean": nm.get("mention_unclean_all")},
               "rule": "use_tgt_clean (judge: uses the fact and coherent) <= half of ctx's with the CI of (ctx - mem) "
                       "excluding 0 at every alpha, and at some alpha >= 2 the share of answers that mention the "
                       "target without a clean use exceeds nomem's"}
    # C4 yes/no
    yn = {(r["group"], r["cond"]): r for r in T["yn"]}
    ev, ok = {}, True
    for g in ("facts", "leanings"):
        for c in ("mem@1", "mem@2"):
            r = yn.get((g, c), {})
            inc0 = r.get("d_bal_lo", NAN) <= 0 <= r.get("d_bal_hi", NAN)
            tilt = (r.get("dm_bal_lo", NAN) <= 0 <= r.get("dm_bal_hi", NAN)) or \
                   (abs(r.get("dmY", NAN) + r.get("dmN", NAN)) < 0.25 * (abs(r.get("dmY", NAN)) + abs(r.get("dmN", NAN))))
            ev[f"{g}/{c}"] = {"d_bal": (r.get("d_bal"), r.get("d_bal_lo"), r.get("d_bal_hi")), "dmY": r.get("dmY"),
                              "dmN": r.get("dmN"), "dm_bal": (r.get("dm_bal"), r.get("dm_bal_lo"), r.get("dm_bal_hi")),
                              "gain_ci_includes_0": inc0, "general_tilt": tilt}
            ok &= inc0 and tilt
    C["C4"] = {"holds": ok, "evidence": ev,
               "rule": "facts and leanings at alpha 1 and 2: the balanced yes/no accuracy gain's CI includes 0, and "
                       "the margin shift is a general tilt: the CI of (dmY + dmN)/2 includes 0 or |dmY + dmN| < "
                       "0.25 (|dmY| + |dmN|)"}
    # C5 mechanism (dislikes, matched loop rate)
    mt = {(r["group"], r["ref"]): r for r in T["mech_matched"]}
    o, c_, w = mt.get(("dislikes", "opposite"), {}), mt.get(("dislikes", "centroid"), {}), mt.get(("dislikes", "without"), {})
    ok = o.get("d_j_lean_lo", NAN) > 0 and c_.get("d_j_lean_hi", NAN) < 0 and w.get("d_j_lean_hi", NAN) < 0
    C["C5"] = {"holds": bool(ok), "evidence": {k: {x: v.get(x) for x in ("alpha", "matched", "loop", "d_j_lean",
                                                                        "d_j_lean_lo", "d_j_lean_hi", "wrong_items")}
                                               for k, v in (("opposite", o), ("centroid", c_), ("without", w))},
               "rule": "dislikes, each reference at its highest dose with loop <= 10%: the opposite's judge-lean gain "
                       "vs nomem > 0 (CI excludes 0); centroid's and without's < 0 (CIs exclude 0)"}
    # C6 lens
    lt6 = T["lens_facts"]
    nb = R.prep_cfg["n_blocks"]
    hi = [r for r in lt6 if r["median_rank"] <= 10]
    early = [r for r in lt6 if r["depth"] <= 0.5]
    ok = bool(hi) and all(r["depth"] >= 0.75 for r in hi) and all(r["median_rank"] > 100 for r in early)
    C["C6"] = {"holds": ok, "evidence": {"blocks_median_rank_le10": [(r["layer"], r["depth"]) for r in hi],
                                         "min_median_rank_depth_le_0.5": min((r["median_rank"] for r in early), default=NAN),
                                         "n_blocks": nb},
               "rule": "the blocks where the facts' median logit-lens target rank <= 10 all lie at relative depth >= 0.75 "
                       "(and there is at least one), and every block at depth <= 0.5 has median rank > 100"}
    return C


# ---------------------------------------------------------------------- report


def analyze_run(run_dir, out=None):
    R = Run(run_dir)
    out = Path(out or Path(run_dir) / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    T = {"pref": [], "facts": [], "yn": [], "mech": [], "mech_matched": [], "seed2": []}
    for g in PREF_GROUPS:
        T["pref"] += pref_table(R, g, R.kept(g), mech=g != "leanings")
        T["pref"] += [dict(r, group=g + "(all)") for r in pref_table(R, g, R.ids[g], mech=g != "leanings")
                      if r["cond"] in ("nomem", "ctx", "mem@1", "mem@2")]
        T["yn"] += yn_table(R, g, R.kept(g))
        T["seed2"] += seed2(R, g, R.kept(g))
    T["facts"] = fact_table(R, R.ids["facts"])
    T["yn"] += yn_table(R, "facts", R.ids["facts"])
    T["seed2"] += seed2(R, "facts", R.ids["facts"])
    T["mech"], T["mech_matched"] = mechanism(R)
    T["selectivity"], T["false_fire"] = selectivity(R, list(R.items))
    T["lens_facts"], T["lens_prefs"] = lens_tables(R)
    T["facts_tf"] = facts_tf(R)
    per_item = []
    for i, it in R.items.items():
        mech = it.get("mech", False)
        for c in R.conds(mech):
            o = {"model": R.name, "item": i, "group": it["group"], "cond": c,
                 "flagged": R.flags.get(i, {}).get("flagged")}
            ms = (("j_clean", ["use"]), ("clean", ["use"]), ("hit", ["related"]), ("loop", None), ("j_incoh", None),
                  ("nll", None)) if it["group"] == "facts" else \
                (("j_lean", None), ("j_dir", None), ("lex", None), ("lex_clean", None), ("loop", None), ("j_incoh", None),
                 ("j_self", None), ("nll", None))
            for m, k in ms:
                o[m + ("_" + k[0] if k else "")] = macro_mean(R.cells(m, c, [i], k))
            per_item.append(o)
    CL = claims(R, T)
    for k, v in T.items():
        write_csv(out / f"{k}.csv", v)
    write_csv(out / "per_item.csv", per_item)
    write_csv(out / "flags.csv", list(R.flags.values()))
    json.dump({"model": R.name, "claims": CL, "checks": R.checks, "judge": R.has_judge,
               "flagged": [i for i, f in R.flags.items() if f["flagged"]]},
              open(out / "claims.json", "w"), indent=2, default=str)
    (out / "report.md").write_text(report_md(R, T, CL))
    print(f"analysis of {R.name}: " + ", ".join(f"{k} {'HOLDS' if v['holds'] else 'fails'}" for k, v in CL.items())
          + f" -> {out}")
    return R, T, CL


def report_md(R, T, CL):
    L = []
    w = L.append
    a = R.cfg["control_alpha"]
    w(f"# core_v1 analysis: {R.name} ({R.cfg['model']}, {R.cfg['dtype']}, layers {R.cfg['layers']})")
    env = R.prep_cfg["env"]
    w(f"commit {env.get('commit')} · torch {env.get('torch')} · transformers {env.get('transformers')} · GPU {env.get('gpu')}")
    w(f"judge: {'yes' if R.has_judge else 'NOT USED (missing or parse rate < 0.8)'}; parse rate "
      f"{fmt(R.judge_parse_rate, '.3f')}")
    w("")
    w("## Claims (PREREG.md decision rules)")
    w(md_table(["claim", "holds", "rule"], [[k, "**yes**" if v["holds"] else "no", v["rule"]] for k, v in CL.items()]))
    w("")
    fl = [f for f in R.flags.values() if f["flagged"]]
    w(f"## Items flagged by the ctx rule (ctx lean - nomem lean < {CTX_RULE}; excluded from the primary numbers)")
    w(md_table(["item", "group", "rule metric", "d (rule)", "d j_lean", "d lex_clean"],
               [[f["item"], f["group"], f["rule_metric"], fmt(f[f["rule_metric"]], "+.2f"), fmt(f["d_j_lean"], "+.2f"),
                 fmt(f["d_lex_clean"], "+.2f")] for f in fl]) if fl else "none")
    w("")
    for g in PREF_GROUPS:
        rs = [r for r in T["pref"] if r["group"] == g]
        w(f"## Preferences: {g} ({len(R.kept(g))} items kept)")
        w("j_lean = judge: +1 a coherent recommendation toward the trait, -1 away, 0 otherwise (PRIMARY); "
          "d_* = paired gain vs nomem [95% CI]; lex_clean = lexicon lean over non-loop answers; wrong = share of "
          "items / prompts moved the wrong way (j_lean).")
        w(md_table(["cond", "j_lean", "d_j_lean", "lex_clean", "d_lex_clean", "loop", "incoherent", "self_claim", "nll",
                    "wrong items/prompts"],
                   [[r["cond"], fmt(r["j_lean"]), ci_str((r.get("d_j_lean", NAN), r.get("d_j_lean_lo", NAN),
                                                          r.get("d_j_lean_hi", NAN), 0)),
                     fmt(r["lex_clean"]), ci_str((r.get("d_lex_clean", NAN), r.get("d_lex_clean_lo", NAN),
                                                  r.get("d_lex_clean_hi", NAN), 0)),
                     fmt(r["loop"], ".3f"), fmt(r["j_incoh"], ".3f"), fmt(r["j_self"], ".3f"), fmt(r["nll"], ".2f"),
                     f"{fmt(r.get('wrong_items', NAN), '.2f')}/{fmt(r.get('wrong_prompts', NAN), '.2f')}"] for r in rs]))
        w("")
    w("## Facts (use = the 2 use prompts; related = the direct question)")
    w(md_table(["cond", "use judge-clean", "use clean (regex)", "related judge-clean", "related clean", "mention not clean",
                "loop", "incoherent", "wrong value", "self (related)", "nll"],
               [[r["cond"], ci_str((r["j_clean_use"], r["j_clean_use_lo"], r["j_clean_use_hi"], 0), ".3f"),
                 fmt(r["clean_use"], ".3f"), fmt(r["j_clean_related"], ".3f"), fmt(r["clean_related"], ".3f"),
                 fmt(r["mention_unclean_all"], ".3f"), fmt(r["loop_all"], ".3f"), fmt(r["j_incoh_all"], ".3f"),
                 fmt(r["j_wrong_all"], ".3f"), fmt(r["j_self_related"], ".3f"), fmt(r["nll_all"], ".2f")]
                for r in T["facts"]]))
    w("")
    w("Analytic (measure prefix, teacher-forced): Delta log P(target), Delta specificity vs nomem [95% CI]")
    w(md_table(["cond", "d log P(target)", "d specificity", "top1"],
               [[r["cond"], ci_str((r["d_lp_t"], r["d_lp_t_lo"], r["d_lp_t_hi"], 0), "+.2f"),
                 ci_str((r["d_spec"], r["d_spec_lo"], r["d_spec_hi"], 0), "+.2f"), fmt(r["top1"], ".2f")]
                for r in T["facts_tf"]]))
    w("")
    w("## Balanced yes/no (greedy; margins = log P(correct) - log P(other) at the first answer token)")
    w(md_table(["group", "cond", "bal", "d_bal", "accY", "accN", "dmY", "dmN", "(dmY+dmN)/2"],
               [[r["group"], r["cond"], fmt(r["bal"], ".3f"),
                 ci_str((r.get("d_bal", NAN), r.get("d_bal_lo", NAN), r.get("d_bal_hi", NAN), 0)),
                 fmt(r["accY"], ".2f"), fmt(r["accN"], ".2f"), fmt(r.get("dmY", NAN), "+.2f"), fmt(r.get("dmN", NAN), "+.2f"),
                 ci_str((r.get("dm_bal", NAN), r.get("dm_bal_lo", NAN), r.get("dm_bal_hi", NAN), 0), "+.2f")]
                for r in T["yn"]]))
    w("")
    w(f"## Mechanism: reference x dose (loop-matched choice: the highest dose with loop <= {LOOP_MATCH:.0%})")
    w(md_table(["group", "ref", "alpha", "loop", "j_lean", "d_j_lean", "d_lex_clean", "incoherent", "wrong items"],
               [[r["group"], r["ref"], f"{r['alpha']:g}", fmt(r["loop"], ".3f"), fmt(r["j_lean"]),
                 ci_str((r["d_j_lean"], r["d_j_lean_lo"], r["d_j_lean_hi"], 0)),
                 ci_str((r["d_lex_clean"], r["d_lex_clean_lo"], r["d_lex_clean_hi"], 0)), fmt(r["j_incoh"], ".3f"),
                 fmt(r["wrong_items"], ".2f")] for r in T["mech"]]))
    w("")
    w("Loop-matched:")
    w(md_table(["group", "ref", "alpha", "matched", "loop", "d_j_lean"],
               [[r["group"], r["ref"], f"{r['alpha']:g}", r["matched"], fmt(r["loop"], ".3f"),
                 ci_str((r["d_j_lean"], r["d_j_lean_lo"], r["d_j_lean_hi"], 0))] for r in T["mech_matched"]]))
    w("")
    w("## Selectivity (50 unrelated prompts, greedy + 2 samples)")
    w(md_table(["group", "cond", "unrel_same", "KL", "contam"],
               [[r["group"], r["cond"], ci_str((r["unrel_same"], r["unrel_same_lo"], r["unrel_same_hi"], 0), ".3f"),
                 fmt(r["kl"], ".4f"), fmt(r["contam"], "+.3f")] for r in T["selectivity"]]))
    w("")
    w(md_table(["group", "gate opens anywhere (share of prompts)", "share of open positions"],
               [[r["group"], ci_str((r["false_fire_prompts"], r["lo"], r["hi"], 0), ".3f"),
                 fmt(r["false_fire_positions"], ".4f")] for r in T["false_fire"]]))
    w("")
    w("## Logit lens of the stored fact shift, every block")
    w(md_table(["layer", "type", "depth", "median target rank", "share rank <= 10", "mean z", "mean margin"],
               [[r["layer"], r["ltype"], r["depth"], r["median_rank"], fmt(r["top10_share"], ".2f"),
                 fmt(r["mean_z"], "+.1f"), fmt(r["mean_margin"], "+.2f")] for r in T["lens_facts"]]))
    w("")
    w("Preference shifts' lex_gain (consistent - inconsistent lexicon logits) at the read layers, by reference:")
    rl = set(R.cfg["layers"])
    w(md_table(["group", "ref", "layer", "lex_gain"],
               [[r["group"], r["ref"], r["layer"], fmt(r["lex_gain"], "+.2f")] for r in T["lens_prefs"]
                if r["layer"] in rl]))
    w("")
    w("## Seed robustness (samples only: seed set 1 vs seed set 2)")
    w(md_table(["group", "cond"] + [k for k in T["seed2"][0] if k not in ("model", "group", "cond")] if T["seed2"] else [],
               [[r["group"], r["cond"]] + [fmt(v) for k, v in r.items() if k not in ("model", "group", "cond")]
                for r in T["seed2"]]))
    w("")
    w("## Checks")
    w("```")
    w(json.dumps(R.checks, indent=1, default=str)[:6000])
    w("```")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- combined


def audit_sample(runs, n_total, seed):
    """A stratified hand-audit sample: equal shares per model; within a model, preference vs fact answers
    (60 / 40), spread over condition families. Labels are EMPTY (to fill by hand); audit_key.jsonl holds
    the automatic labels and the condition (kept out of the sample so the audit is blind)."""
    rng = random.Random(seed)
    fam = lambda c: ("base" if c in ("nomem", "ctx") else "mem" if c.startswith("mem@") else
                     "control" if c.split("@")[0] in ("rand", "swap", "placebo", "gate_on") else "mechanism")
    sample, key = [], []
    per_model = n_total // len(runs)
    for R in runs:
        for measure, share in (("preference", 0.6), ("fact", 0.4)):
            pool = [r for r in R.gen if (R.items[r["item"]]["group"] == "facts") == (measure == "fact")]
            strata = defaultdict(list)
            for r in pool:
                strata[fam(r["cond"])].append(r)
            want = round(per_model * share)
            fams = sorted(strata)
            for k, f in enumerate(fams):
                n = want // len(fams) + (1 if k < want % len(fams) else 0)
                for r in rng.sample(strata[f], min(n, len(strata[f]))):
                    it = R.items[r["item"]]
                    aid = f"a{len(sample):04d}"
                    s = {"audit_id": aid, "rubric": measure, "question": r["prompt"], "answer": r["answer"]}
                    if measure == "fact":
                        s.update(fact=it["experience"], target=it["measure"]["target"].strip(),
                                 h_uses_fact=None, h_coherent=None, h_self_claim=None, h_wrong_value=None)
                    else:
                        s.update(trait=it["experience"], h_direction=None, h_coherent=None, h_self_claim=None,
                                 h_lexicon_ok=None)
                    s["h_notes"] = ""
                    sample.append(s)
                    j = R.judge.get(r["key"], {})
                    key.append({"audit_id": aid, "model": R.name, "item": r["item"], "group": it["group"],
                                "cond": r["cond"], "kind": r["kind"], "key": r["key"], "judge": {k: j.get(k) for k in
                                ("direction", "uses_fact", "coherent", "self_claim", "wrong_value", "ok", "raw")},
                                "lexicon": r.get("lex"), "loop": r["loop"], "nll": r.get("nll"), "hit": r.get("hit")})
    order = list(range(len(sample)))
    rng.shuffle(order)
    return [sample[i] for i in order], [key[i] for i in order]


def combine(run_dirs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs, allcl = [], {}
    for d in run_dirs:
        R, T, CL = analyze_run(d)
        runs.append(R)
        allcl[R.name] = CL
    names = [R.name for R in runs]
    L = [f"# core_v1: claims per model (C7 = does each claim hold at every scale?)", "",
         md_table(["claim"] + names + ["all models"],
                  [[c] + [("yes" if allcl[n][c]["holds"] else "no") for n in names]
                   + [("**yes**" if all(allcl[n][c]["holds"] for n in names) else "no")] for c in allcl[names[0]]]), ""]
    for c in allcl[names[0]]:
        L.append(f"## {c}: {allcl[names[0]][c]['rule']}")
        for n in names:
            L.append(f"- {n}: {'holds' if allcl[n][c]['holds'] else 'does not hold'}; "
                     f"`{json.dumps(allcl[n][c]['evidence'], default=str)[:1500]}`")
        L.append("")
    (out / "C7.md").write_text("\n".join(L) + "\n")
    json.dump(allcl, open(out / "claims_all.json", "w"), indent=2, default=str)
    sample, key = audit_sample(runs, 200, int(runs[0].cfg["seeds"]["audit"]))
    with open(out / "audit_sample.jsonl", "w") as f:
        for s in sample:
            f.write(json.dumps(s) + "\n")
    with open(out / "audit_key.jsonl", "w") as f:
        for s in key:
            f.write(json.dumps(s) + "\n")
    print(f"combined: {len(sample)} audit answers; C7 -> {out / 'C7.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="one model's run dir")
    p.add_argument("--combine", nargs="+", default=None, help="several model run dirs")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.combine:
        combine(a.combine, a.out or Path(a.combine[0]).parent / "analysis")
    else:
        analyze_run(a.run, a.out)


if __name__ == "__main__":
    main()
