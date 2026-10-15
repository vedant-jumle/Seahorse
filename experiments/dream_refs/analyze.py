#!/usr/bin/env python
"""dream_refs analysis (phase B; no GPU): tables and the S2 / S4 decisions from the saved files, under the
original judge (decisive, as core_v1) and, when the panel has run, under each panel judge and the majority of
the three (original Qwen3.5-9B + gpt-oss + Gemma-4). S1 and S3 were decided in the lens stage and are copied.

  python experiments/dream_refs/run.py --config ... --stage analyze --out RUN/qwen35_2b [--panel RUN/panel]

Inputs: <out>/prep/{items.json, config.json, checks.json}, <out>/lens/{s1.json, s3.json},
<out>/gen/items/*.jsonl.gz, <out>/gen/unrel/*.jsonl.gz, <out>/score/{coherence,judge}.jsonl.gz,
[<panel>/{gptoss,gemma4}.jsonl]. Outputs: <out>/analysis/{report.md, claims.json, *.csv}.

The statistics are core_v1's (its analyze.py): a cell = one (item, prompt); group number = macro mean over items;
95% CIs from the two-level bootstrap (items, then prompts), 2000 replicates, seed 7; gains paired against nomem on
the same prompt and random streams; the ctx rule flags items whose ctx - nomem j_lean < 0.3 (excluded from the
primary numbers, reported).
"""

import gzip
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cv1a = _load("seahorse_core_v1_analyze", REPO / "experiments" / "core_v1" / "analyze.py")
refs = _load("seahorse_dream_refs_refs", HERE / "refs.py")
NAN = float("nan")
PREF = ("leanings", "dislikes", "one_of_many")
VARIANTS = ("hand:opposite", "gen:opposite", "gen:centroid", "cs:concept_x_sign", "cs:oracle")
PANEL = ("gptoss", "gemma4")       # PREREG: the original judge + gpt-oss + Gemma-4; majority of the three


def read_gz(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


class DRun(cv1a.Run):
    """core_v1's Run on dream_refs' layout (preference items only), optionally with another judge's labels."""

    def __init__(self, d, judge_rows=None, judge_name="orig"):
        d = Path(d)
        self.dir, self.name, self.judge_name = d, d.name, judge_name
        self.items = json.load(open(d / "prep" / "items.json"))
        self.prep_cfg = json.load(open(d / "prep" / "config.json"))
        self.cfg = self.prep_cfg["cfg"]
        self.checks = json.load(open(d / "prep" / "checks.json"))
        self.lens, self.ftf = [], []
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
        if judge_rows is None:
            judge_rows = {r["key"]: r for r in read_gz(sc / "judge.jsonl.gz")} if (sc / "judge.jsonl.gz").exists() else {}
        self.judge = judge_rows
        self.judge_parse_rate = cv1a.nanmean([float(bool(j.get("ok"))) for j in self.judge.values()]) if self.judge else NAN
        self.has_judge = bool(self.judge) and self.judge_parse_rate >= 0.8
        self.ids = {g: [i for i, it in self.items.items() if it["group"] == g] for g in PREF + ("facts",)}
        self.wrong = {}
        for r in self.gen:
            self._metrics(r)
        self.flags = self._ctx_flags()

    def conds(self, mech=False):
        return ["nomem", "ctx"] + [f"{v}@{a:g}" for v in VARIANTS for a in self.cfg["doses"]]


def matched_table(R):
    """Per group and variant: every dose (loop, j_lean, gain [CI], wrong items) and the loop-matched dose (the
    highest alpha in {0.5, 1, 2} with loop <= 10%, as core_v1 C5)."""
    rows, matched = [], []
    for g in PREF:
        kept = R.kept(g)
        if not kept:
            continue
        for v in VARIANTS:
            per, loops = {}, {}
            items = [i for i in kept if v not in excluded_of(R, i)]     # Amendment 2: dreaming-failed items out
            for a in R.cfg["doses"]:
                c = f"{v}@{a:g}"
                o = {"group": g, "variant": v, "alpha": a, "cond": c, "n_items": len(items), "n_group": len(kept)}
                for m in ("loop", "j_lean", "j_incoh", "lex_clean"):
                    o[m] = R.stat(m, c, items)[0]
                for m in ("j_lean", "lex_clean"):
                    p, lo, hi, _ = R.gain(m, c, items)
                    o["d_" + m], o["d_" + m + "_lo"], o["d_" + m + "_hi"] = p, lo, hi
                o["wrong_items"], o["wrong_prompts"] = cv1a.wrong_way(R.gain_cells("j_lean", c, items), +1)
                rows.append(o)
                per[a], loops[a] = o, o["loop"]
            loops = {a: (1.0 if (isinstance(x, float) and math.isnan(x)) else x) for a, x in loops.items()}
            a, ok = refs.loop_matched(loops)
            matched.append(dict(per[a], matched=ok))
    return rows, matched


def excluded_of(R, iid):
    """The variants an item is excluded from because its dreaming failed (PREREG Amendment 2)."""
    return set((R.items[iid].get("dream") or {}).get("excluded_variants") or [])


def testable(row):
    """Amendment 2: a (group, variant) number enters a rule only if >= half of the group's ctx-kept items remain
    after the dreaming-failed exclusions; otherwise the claim is 'not testable (dreaming failed)'."""
    if "n_items" not in row:
        return True
    return row["n_items"] > 0 and row["n_items"] >= 0.5 * row.get("n_group", row["n_items"])


def claims(R, matched, lens_dir):
    mt = {(r["group"], r["variant"]): r for r in matched}
    g = lambda grp, v, k: mt.get((grp, v), {}).get(k, NAN)
    C = {}
    nt = [f"{grp}/{v}" for grp, v in (("dislikes", "gen:opposite"), ("dislikes", "gen:centroid"),
                                      ("dislikes", "cs:concept_x_sign"), ("leanings", "cs:concept_x_sign"),
                                      ("leanings", "gen:opposite")) if (grp, v) in mt and not testable(mt[(grp, v)])]
    C["not_testable"] = nt
    s2 = g("dislikes", "gen:opposite", "d_j_lean_lo") > 0 and g("dislikes", "gen:centroid", "d_j_lean_hi") < 0
    s2 = s2 and not any(x in nt for x in ("dislikes/gen:opposite", "dislikes/gen:centroid"))
    C["S2"] = {"holds": bool(s2), "rule": "dislikes (ctx-rule kept), each variant at its loop-matched alpha: gen:opposite's "
               "j_lean gain vs nomem > 0 (CI excludes 0) and gen:centroid's < 0 (CI excludes 0)",
               "evidence": {v: {k: mt.get(("dislikes", v), {}).get(k) for k in
                                ("alpha", "matched", "loop", "d_j_lean", "d_j_lean_lo", "d_j_lean_hi", "wrong_items")}
                            for v in VARIANTS}}
    dis = g("dislikes", "cs:concept_x_sign", "d_j_lean_lo") > 0
    lean = g("leanings", "cs:concept_x_sign", "d_j_lean") >= g("leanings", "gen:opposite", "d_j_lean_lo")
    dis, lean = dis and "dislikes/cs:concept_x_sign" not in nt, lean and not any(
        x in nt for x in ("leanings/cs:concept_x_sign", "leanings/gen:opposite"))
    C["S4"] = {"holds": bool(dis and lean), "parts": {"dislikes_right_way": bool(dis), "leanings_not_worse": bool(lean)},
               "rule": "cs:concept_x_sign at its loop-matched alpha: dislikes' j_lean gain > 0 (CI excludes 0), AND on the "
                       "leanings its gain >= the lower CI bound of gen:opposite's gain (each at its own loop-matched alpha)",
               "oracle_dislikes_right_way": bool(g("dislikes", "cs:oracle", "d_j_lean_lo") > 0),
               "evidence": {f"{grp}/{v}": {k: mt.get((grp, v), {}).get(k) for k in ("alpha", "matched", "loop", "d_j_lean",
                                                                                   "d_j_lean_lo", "d_j_lean_hi")}
                            for grp in ("dislikes", "leanings") for v in ("cs:concept_x_sign", "cs:oracle", "gen:opposite",
                                                                          "hand:opposite")}}
    if C["S4"]["holds"]:
        C["S4"]["reading"] = "concept x sign works: dislikes right way without a hand-written opposite"
    elif C["S4"]["oracle_dislikes_right_way"]:
        C["S4"]["reading"] = "cs:oracle works but cs:concept_x_sign does not: the sign reader is the weak part"
    else:
        C["S4"]["reading"] = ("neither signed concept moves the dislikes the right way: subtracting a concept does not "
                              "produce avoidance; the dislike channel needs the opposite")
    C["S5"] = {"holds": True, "rule": "without needs no generation (mechanical); reported, not tested"}
    for k, f in (("S1", "s1.json"), ("S3", "s3.json")):
        p = Path(lens_dir) / f
        if p.exists():
            d = json.load(open(p))
            C[k] = {"verdict": d["verdict"], "decided_in": "lens stage (phase A)"}
    s1 = C.get("S1", {}).get("verdict")
    C["decisions"] = {
        "scaffold_removed": bool(C["S2"]["holds"] and s1 in ("holds", "indeterminate")),
        "scaffold_removed_note": "S1 and S2 hold; an indeterminate S1 defers to S2 (PREREG)",
        "sign_readable": C.get("S3", {}).get("verdict") == "holds",
        "concept_x_sign_works": C["S4"]["holds"]}
    return C


def repro_core_v1(R, cv1_gen):
    """Share of nomem / hand:opposite@alpha answers identical to core_v1's (same prompts, same random streams;
    different batch composition, so last-bit float differences are allowed). Reported, not required."""
    p = Path(cv1_gen or "", "items")
    if not cv1_gen or not p.exists():
        return {"skipped": f"no core_v1 gen at {cv1_gen}"}
    mine = {r["key"]: r["ids"] for r in R.gen if r["cond"] == "nomem" or r["cond"].startswith("hand:opposite@")}
    same = defaultdict(lambda: [0, 0])
    for f in sorted(p.glob("*.jsonl.gz")):
        for r in read_gz(f):
            if r.get("sec") != "gen":
                continue
            c = r["cond"]
            mc = c if c == "nomem" else (f"hand:opposite@{c.split('@')[1]}" if c.startswith("mem@") else None)
            if mc is None:
                continue
            k = f"{r['item']}|{r['kind']}|{r['pi']}|{mc}|{r['seedset']}|{r['j']}"
            if k in mine:
                same[mc][0] += mine[k] == r["ids"]
                same[mc][1] += 1
    return {c: {"identical": a / n if n else NAN, "n": n} for c, (a, n) in sorted(same.items())}


def panel_judges(run_dir, panel):
    """{name: {key: judge row}} for each panel judge present and the majority of (original + present panel
    judges); core_v1 analyze_panel's loader and majority rule (a tie goes to the original's label)."""
    if not panel or not Path(panel).exists():
        return {}
    ap = _load("seahorse_core_v1_analyze_panel", REPO / "experiments" / "core_v1" / "analyze_panel.py")
    m = Path(run_dir).name
    orig = {r["key"]: r for r in read_gz(Path(run_dir) / "score" / "judge.jsonl.gz")}
    out, have = {}, []
    for tag in PANEL:
        d, _ = ap.load_panel(panel, tag, models=(m,))
        if d is not None:
            out[tag] = d[m]
            have.append(tag)
    if have:
        maj = {}
        for k, o in orig.items():
            if all(k in out[t] for t in have):
                maj[k] = ap.majority_row([out[t][k] for t in have] + [o], ap.FIELDS["preference"], o)
        out["majority"] = maj
    return out


def f(x, spec="+.3f"):
    return cv1a.fmt(x, spec)


def ci(o, k):
    return cv1a.ci_str((o.get(k, NAN), o.get(k + "_lo", NAN), o.get(k + "_hi", NAN), 0))


def report_md(R, T, CL, others, repro, cv1_ctx):
    L = []
    w = L.append
    w(f"# Experiment 2 (dream_refs), phase B: {R.name} ({R.cfg['model']}, layers {R.cfg['layers']})")
    w(f"judge: original Qwen3.5-9B {'used' if R.has_judge else 'NOT USABLE'} (parse rate {f(R.judge_parse_rate, '.3f')})")
    w("")
    w("## Claims (PREREG.md rules; original judge decisive, panel reported)")
    w(cv1a.md_table(["claim", "verdict", "rule"],
                    [[k, (("**holds**" if v["holds"] else "fails") if "holds" in v else v.get("verdict")),
                      v.get("rule", v.get("decided_in", ""))] for k, v in CL.items() if k not in ("decisions", "not_testable")]))
    w("")
    w(f"S4 reading: {CL['S4']['reading']}")
    w("")
    w("Decisions: " + json.dumps(CL["decisions"]))
    w("")
    if others:
        w("## The same S2 / S4 rules under the other judges")
        w(cv1a.md_table(["judge", "S2", "S4", "S4 dislikes", "S4 leanings", "oracle dislikes right way"],
                        [[n, c["S2"]["holds"], c["S4"]["holds"], c["S4"]["parts"]["dislikes_right_way"],
                          c["S4"]["parts"]["leanings_not_worse"], c["S4"]["oracle_dislikes_right_way"]]
                         for n, c in others.items()]))
        w("")
    fl = [x for x in R.flags.values() if x["flagged"]]
    w(f"## ctx rule: flagged items (excluded from primary numbers): {', '.join(x['item'] for x in fl) or 'none'}")
    w("")
    df = {i: it.get("dream", {}).get("excluded_variants") or [] for i, it in R.items.items()}
    df = {i: v for i, v in df.items() if v}
    w(f"## Dreaming failed (PREREG Amendment 2): {len(df)} items, each excluded from the variants named; "
      "n_items below is the group's ctx-kept count, the variant's own count is lower by its exclusions")
    for v in VARIANTS:
        ex = sorted(i for i, vs in df.items() if v in vs)
        if ex:
            w(f"- {v}: {len(ex)} excluded ({', '.join(ex)})")
    w("")
    w("## Loop-matched comparison (highest alpha with loop <= 10%)")
    w(cv1a.md_table(["group", "variant", "items", "alpha", "matched", "loop", "d_j_lean [95% CI]", "wrong items"],
                    [[r["group"], r["variant"], f"{r['n_items']}/{r['n_group']}", f"{r['alpha']:g}", r["matched"],
                      f(r["loop"], ".3f"), ci(r, "d_j_lean"), f(r["wrong_items"], ".2f")] for r in T["matched"]]))
    if CL.get("not_testable"):
        w(f"Not testable (fewer than half of the group's items remain after dreaming failures; the claims that use them "
          f"do not hold): {', '.join(CL['not_testable'])}")
    w("")
    if cv1_ctx:
        w("Context only (core_v1's own 2B run, a different run with its own nomem; enters no rule): dislikes, "
          "loop-matched: " + "; ".join(f"{r['ref']}@{r['alpha']} d_j_lean {r['d_j_lean']}" for r in cv1_ctx))
        w("")
    w("## Every dose")
    w(cv1a.md_table(["group", "variant", "alpha", "loop", "j_lean", "d_j_lean [95% CI]", "d_lex_clean", "incoherent"],
                    [[r["group"], r["variant"], f"{r['alpha']:g}", f(r["loop"], ".3f"), f(r["j_lean"]), ci(r, "d_j_lean"),
                      ci(r, "d_lex_clean"), f(r["j_incoh"], ".3f")] for r in T["doses"]]))
    w("")
    for g in PREF:
        rs = [r for r in T["pref"] if r["group"] == g]
        if not rs:
            continue
        w(f"## {g}: every condition")
        w(cv1a.md_table(["cond", "j_lean", "d_j_lean", "loop", "incoherent", "self_claim", "nll"],
                        [[r["cond"], f(r["j_lean"]), ci(r, "d_j_lean"), f(r["loop"], ".3f"), f(r["j_incoh"], ".3f"),
                          f(r["j_self"], ".3f"), f(r["nll"], ".2f")] for r in rs]))
        w("")
    w("## Yes/no (alpha = 1 cells)")
    w(cv1a.md_table(["group", "cond", "bal", "d_bal"],
                    [[r["group"], r["cond"], f(r["bal"], ".3f"), cv1a.ci_str((r.get("d_bal", NAN), r.get("d_bal_lo", NAN),
                                                                           r.get("d_bal_hi", NAN), 0))] for r in T["yn"]]))
    w("")
    w("## Selectivity (alpha = 1 cells, 50 unrelated prompts)")
    w(cv1a.md_table(["group", "cond", "unrel_same", "KL"],
                    [[r["group"], r["cond"], cv1a.ci_str((r["unrel_same"], r["unrel_same_lo"], r["unrel_same_hi"], 0), ".3f"),
                      f(r["kl"], ".4f")] for r in T["selectivity"]]))
    w("")
    w("## Reproducibility vs core_v1 (identical answers; batch composition differs, so not required)")
    w("```")
    w(json.dumps(repro, indent=1, default=str))
    w("```")
    return "\n".join(L) + "\n"


def tables(R):
    T = {"pref": []}
    for g in PREF:
        if R.kept(g):
            T["pref"] += cv1a.pref_table(R, g, R.kept(g))
    T["doses"], T["matched"] = matched_table(R)
    return T


def analyze_run(out, cfg=None, panel=None):
    out = Path(out)
    adir = out / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    R = DRun(out)
    T = tables(R)
    T["yn"] = sum((cv1a.yn_table(R, g, R.kept(g)) for g in PREF if R.kept(g)), [])
    T["selectivity"], T["false_fire"] = cv1a.selectivity(R, list(R.items))
    CL = claims(R, T["matched"], out / "lens")
    others = {}
    for name, rows in panel_judges(out, panel).items():
        Rj = DRun(out, judge_rows=rows, judge_name=name)
        Rj.flags = R.flags          # the item set surviving the ctx rule is the original judge's (as core_v1's panel)
        others[name] = claims(Rj, matched_table(Rj)[1], out / "lens")
    repro = repro_core_v1(R, (cfg or R.cfg).get("cv1_gen"))
    cv1_ctx = []
    p = Path((cfg or R.cfg).get("cv1_analysis") or "", "mech_matched.csv")
    if p.exists():
        import csv
        cv1_ctx = [r for r in csv.DictReader(open(p)) if r["group"] == "dislikes"]
    for k in ("pref", "doses", "matched", "yn", "selectivity"):
        cv1a.write_csv(adir / f"{k}.csv", T[k])
    cv1a.write_csv(adir / "flags.csv", list(R.flags.values()))
    json.dump({"claims": CL, "other_judges": others, "repro_core_v1": repro, "judge": R.has_judge,
               "flagged": [i for i, x in R.flags.items() if x["flagged"]]}, open(adir / "claims.json", "w"), indent=2,
              default=str)
    (adir / "report.md").write_text(report_md(R, T, CL, others, repro, cv1_ctx))
    print(f"dream_refs analysis: S2 {'HOLDS' if CL['S2']['holds'] else 'fails'}, S4 {'HOLDS' if CL['S4']['holds'] else 'fails'}"
          f" -> {adir}")
    return CL
