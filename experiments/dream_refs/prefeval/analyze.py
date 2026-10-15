#!/usr/bin/env python
"""PrefEval benchmark analysis (phase B; no GPU). PREREG.md "Addendum: PrefEval" fixes the rules implemented here.

  python experiments/dream_refs/prefeval/run.py --stage analyze --out RUN/qwen35_2b [--panel RUN/panel]

Per answer and judge: PrefEval's violation and helpfulness labels -> preference-following accuracy = helpful and
not violating (bench.pref_labels; an unparsed label leaves the answer's accuracy missing). Judges: the original
(Qwen3.5-9B; decisive), gpt-oss, Gemma-4, and the majority of the three per label (a tie goes to the original).
Acknowledgement / hallucination (PrefEval's error types) and j_lean (core_v1's rubric) come from the original only.

Cell = one preference (one test question): the mean over its 5 seed-set-1 answers (1 greedy + 4 samples). Group
number = mean over preferences; gains are paired per preference against nomem; 95% percentile CIs from 2000
bootstrap resamples of preferences (seed 7). Each variant's items exclude its dreaming failures; a (subset, variant)
with fewer than half of the subset's items left is "not testable".

Decision rules (alpha = 1):
  P1 scaffold-free memory helps on a benchmark: all preferences, gen:opposite - nomem accuracy, CI > 0
  P2 concept x sign fixes dislikes at scale: avoid subset, cs:concept_x_sign - nomem CI > 0 AND
     gen:centroid - nomem CI < 0
"""

import csv
import gzip
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
NAN = float("nan")
VARIANTS = ("gen:opposite", "gen:centroid", "cs:concept_x_sign")
SUBSETS = ("all", "avoid", "mixed", "like", "other", "gate_open")
JUDGES = ("orig", "gptoss", "gemma4", "majority")
DIR = {"toward": 1, "away": -1, "neutral": 0}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("seahorse_prefeval_bench", HERE / "bench.py")


def read_gz(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def boot(nested, n=2000, seed=7):
    from seahorse.stats import boot_ci
    return boot_ci(nested, n, seed)


def isnum(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


# ---------------------------------------------------------------------------------------------- labels


def panel_labels(panel_dir, tag):
    """{rubric: {answer key: True/False/None}} from <panel>/<tag>.jsonl."""
    p = Path(panel_dir or "", f"{tag}.jsonl")
    if not panel_dir or not p.exists():
        return None
    out = defaultdict(dict)
    for line in open(p):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        for k in r["keys"]:
            out[r["rubric"]][k] = r["answer"]
    return out


def judge_labels(pe, panels):
    """{judge: {key: {"violate", "helpful"}}} for the original, each panel judge present, and the majority."""
    J = {"orig": {k: {"violate": v.get("violate"), "helpful": v.get("helpful")} for k, v in pe.items()}}
    for tag, P in panels.items():
        J[tag] = {k: {"violate": P["violate"].get(k), "helpful": P["helpful"].get(k)} for k in pe}
    if panels:
        J["majority"] = {k: {rub: bench.majority([J[j][k][rub] for j in J], J["orig"][k][rub])
                             for rub in ("violate", "helpful")} for k in pe}
    return J


def answer_metrics(lab, pe_row):
    """One answer under one judge: accuracy and error types (PrefEval's formula; missing if a deciding label is
    unparsed). Acknowledgement / hallucination are the original judge's."""
    if lab["violate"] is None or lab["helpful"] is None:
        return {"acc": None, "viol": None, "unhelpful": None}
    L = bench.pref_labels(pe_row.get("acknow"), lab["violate"], lab["helpful"], pe_row.get("hallucinate"))
    return {"acc": float(L["accuracy"]), "viol": float(L["violation"]), "unhelpful": float(L["unhelpful"]),
            "unaware_viol": float(L["unaware_violation"]), "inconsistent": float(L["inconsistent"]),
            "halluc_viol": float(L["hallucination_violation"])}


# ----------------------------------------------------------------------------------------------- tables


class PRun:
    def __init__(self, d, panel=None):
        d = Path(d)
        self.dir = d
        self.items = json.load(open(d / "prep" / "items.json"))
        self.cfg = json.load(open(d / "prep" / "config.json"))["cfg"]
        rows = []
        for f in sorted((d / "gen" / "items").glob("*.jsonl.gz")) + sorted((d / "gen" / "unrel").glob("*.jsonl.gz")):
            rows += read_gz(f)
        by = defaultdict(list)
        for r in rows:
            by[r["sec"]].append(r)
        self.gen, self.gate, self.unrel, self.ugate, self.kl = by["gen"], by["gate"], by["unrel"], by["ugate"], by["kl"]
        sc = d / "score"
        self.nll = {r["key"]: r for r in read_gz(sc / "coherence.jsonl.gz")} if (sc / "coherence.jsonl.gz").exists() else {}
        self.jl = {r["key"]: r for r in read_gz(sc / "judge.jsonl.gz")} if (sc / "judge.jsonl.gz").exists() else {}
        self.pe = {r["key"]: r for r in read_gz(sc / "pe_judge.jsonl.gz")} if (sc / "pe_judge.jsonl.gz").exists() else {}
        panels = {t: p for t in ("gptoss", "gemma4") if (p := panel_labels(panel, t)) is not None}
        self.labels = judge_labels(self.pe, panels)
        self.parse = {j: {rub: (sum(v[rub] is not None for v in L.values()) / len(L) if L else NAN)
                          for rub in ("violate", "helpful")} for j, L in self.labels.items()}
        self.fire = {g["item"]: g["fire"] for g in self.gate}
        self.subset_items = {s: [i for i, it in self.items.items() if s == "all" or it["polarity"] == s] for s in SUBSETS[:5]}
        self.subset_items["gate_open"] = [i for i in self.items if self.fire.get(i, 0) > 0]
        self.metrics = {}
        for r in self.gen:
            m = {"loop": float(r["loop"]), "n_tok": float(r["n_tok"]), "capped": float(r["capped"])}
            n = self.nll.get(r["key"])
            m["nll"] = n["nll"] if n else None
            j = self.jl.get(r["key"])
            m["j_lean"] = (DIR.get(j.get("direction"), 0) if j.get("coherent") else 0) if j and j.get("ok") else None
            m["j_incoh"] = (0.0 if j.get("coherent") else 1.0) if j and j.get("ok") else None
            p = self.pe.get(r["key"], {})
            m["ack"] = None if p.get("acknow") is None else float(p["acknow"])
            m["halluc"] = float(bool(p.get("acknow")) and p.get("hallucinate") is True) if p.get("acknow") is not None else None
            for jn, L in self.labels.items():
                if r["key"] in L:
                    for k, v in answer_metrics(L[r["key"]], p).items():
                        m[f"{jn}:{k}"] = v
            self.metrics[r["key"]] = m

    def cells(self, metric, cond, items):
        d = defaultdict(list)
        it = set(items)
        for r in self.gen:
            if r["cond"] == cond and r["item"] in it and r["seedset"] == 1:
                v = self.metrics[r["key"]].get(metric)
                if isnum(v):
                    d[r["item"]].append(v)
        return {i: {0: sum(v) / len(v)} for i, v in d.items()}

    def stat(self, metric, cond, items):
        return boot(self.cells(metric, cond, items))

    def gain(self, metric, cond, items):
        a, b = self.cells(metric, cond, items), self.cells(metric, "nomem", items)
        return boot({i: {0: a[i][0] - b[i][0]} for i in a if i in b})

    def variant_items(self, v, subset):
        return [i for i in self.subset_items[subset] if v not in self.items[i]["excluded"]]


def gain_table(R, judge="orig"):
    rows = []
    acc = f"{judge}:acc"
    for s in SUBSETS:
        base = R.subset_items[s]
        if not base:
            continue
        for c in ["ctx"] + [f"{v}@{a:g}" for v in VARIANTS for a in R.cfg["doses"]]:
            v = c.split("@")[0]
            its = base if c == "ctx" else R.variant_items(v, s)
            o = {"judge": judge, "subset": s, "cond": c, "n_subset": len(base), "n_items": len(its)}
            for m, name in ((acc, "acc"), (f"{judge}:viol", "viol"), (f"{judge}:unhelpful", "unhelpful")):
                o[name], *_ = R.stat(m, c, its)
                p, lo, hi, n = R.gain(m, c, its)
                o["d_" + name], o["d_" + name + "_lo"], o["d_" + name + "_hi"] = p, lo, hi
                if name == "acc":
                    o["n_paired"] = n
            o["nomem_acc"], *_ = R.stat(acc, "nomem", its)
            if judge == "orig":
                for m in ("j_lean", "ack", "halluc", "loop", "nll", "j_incoh"):
                    o[m], *_ = R.stat(m, c, its)
                p, lo, hi, _ = R.gain("j_lean", c, its)
                o["d_j_lean"], o["d_j_lean_lo"], o["d_j_lean_hi"] = p, lo, hi
            o["testable"] = c == "ctx" or (len(its) > 0 and len(its) >= 0.5 * len(base))
            rows.append(o)
    return rows


def claims(T, alpha=1):
    g = {(r["subset"], r["cond"]): r for r in T}
    a = f"@{alpha:g}"
    get = lambda s, v, k: g.get((s, v + a), {}).get(k, NAN)
    test = lambda s, v: bool(g.get((s, v + a), {}).get("testable"))
    p1 = test("all", "gen:opposite") and get("all", "gen:opposite", "d_acc_lo") > 0
    p2a = test("avoid", "cs:concept_x_sign") and get("avoid", "cs:concept_x_sign", "d_acc_lo") > 0
    p2b = test("avoid", "gen:centroid") and get("avoid", "gen:centroid", "d_acc_hi") < 0
    ev = lambda s, v: {k: get(s, v, k) for k in ("n_items", "n_subset", "d_acc", "d_acc_lo", "d_acc_hi", "testable")}
    return {"P1": {"holds": bool(p1), "rule": "all preferences, alpha 1: gen:opposite - nomem accuracy, 95% CI > 0",
                   "evidence": ev("all", "gen:opposite")},
            "P2": {"holds": bool(p2a and p2b), "parts": {"cs_right_way": bool(p2a), "centroid_wrong_way": bool(p2b)},
                   "rule": "avoid subset, alpha 1: cs:concept_x_sign - nomem CI > 0 AND gen:centroid - nomem CI < 0",
                   "evidence": {v: ev("avoid", v) for v in ("cs:concept_x_sign", "gen:centroid", "gen:opposite")}},
            "reported": {"cs_all": ev("all", "cs:concept_x_sign"), "ctx_all": {k: g.get(("all", "ctx"), {}).get(k)
                                                                             for k in ("d_acc", "d_acc_lo", "d_acc_hi")}}}


def selectivity(R, alpha=1):
    out = []
    for v in VARIANTS:
        c = f"{v}@{alpha:g}"
        same = defaultdict(list)
        for r in R.unrel:
            if r["cond"] == c:
                same[r["item"]].append(float(r["same"]))
        kl = [k["kl"] for k in R.kl if k["cond"] == c]
        p, lo, hi, n = boot({i: {0: sum(x) / len(x)} for i, x in same.items()})
        out.append({"cond": c, "unrel_same": p, "lo": lo, "hi": hi, "n_items": n,
                    "kl": sum(kl) / len(kl) if kl else NAN})
    fire = [g["open_any"] for g in R.ugate]
    return out, {"ugate_open_any": sum(fire) / len(fire) if fire else NAN, "n": len(fire)}


def fmt(x, spec="+.3f"):
    return "—" if not isnum(x) else format(x, spec)


def md_table(h, rows):
    return "\n".join(["| " + " | ".join(map(str, h)) + " |", "|" + "---|" * len(h)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def analyze_run(out, panel=None):
    out = Path(out)
    adir = out / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    R = PRun(out, panel)
    judges = [j for j in JUDGES if j in R.labels]
    T = {j: gain_table(R, j) for j in judges}
    C = {j: claims(T[j], R.cfg["primary_alpha"]) for j in judges}
    decisive = "orig" if min(R.parse["orig"].values()) >= 0.8 else ("majority" if "majority" in C else "orig")
    verdict = {k: C[decisive][k]["holds"] for k in ("P1", "P2")}
    sensitive = {k: ("majority" in C and C["orig"][k]["holds"] != C["majority"][k]["holds"]) for k in ("P1", "P2")}
    sel, ff = selectivity(R, R.cfg["primary_alpha"])
    gate_rate = {s: (sum(R.fire.get(i, 0) > 0 for i in its) / len(its) if its else NAN)
                 for s, its in R.subset_items.items() if s != "gate_open"}
    res = {"decisive_judge": decisive, "verdict": verdict, "judge_sensitive": sensitive, "claims": C,
           "parse_rates": R.parse, "gate_fire_rate_question": gate_rate, "selectivity": sel, "unrel_false_fire": ff,
           "dream_failed": {i: it["excluded"] for i, it in R.items.items() if it["excluded"]}}
    json.dump(res, open(adir / "claims.json", "w"), indent=2, default=str)
    for j in judges:
        write_csv(adir / f"gains_{j}.csv", T[j])
    L = []
    w = L.append
    w("# Experiment 2, PrefEval benchmark, phase B (Qwen3.5-2B, layers 20/21/23)")
    w("")
    w(f"Decisive judge: **{decisive}** (parse rates {json.dumps(R.parse)}). Majority = original + gpt-oss + Gemma-4.")
    w("")
    w(md_table(["claim", "verdict", "judge-sensitive", "rule"],
               [[k, "**holds**" if verdict[k] else "fails", sensitive[k], C[decisive][k]["rule"]] for k in ("P1", "P2")]))
    w("")
    w("Under each judge: " + "; ".join(f"{j}: P1 {C[j]['P1']['holds']}, P2 {C[j]['P2']['holds']} "
                                         f"(cs right way {C[j]['P2']['parts']['cs_right_way']}, centroid wrong way "
                                         f"{C[j]['P2']['parts']['centroid_wrong_way']})" for j in judges))
    w("")
    w(f"Gate open (fire > 0) on the test question: {json.dumps({k: round(v, 3) for k, v in gate_rate.items()})}. "
      f"False fire on the 50 unrelated prompts: {fmt(ff['ugate_open_any'], '.3f')}.")
    w("")
    w(f"Dreaming failed (excluded per variant): {len(res['dream_failed'])} items.")
    w("")
    for j in judges:
        w(f"## Accuracy gains vs nomem, judge {j}")
        w("")
        w(md_table(["subset", "cond", "items", "nomem acc", "acc", "d acc [95% CI]", "d violation", "d unhelpful",
                    "testable"],
                   [[r["subset"], r["cond"], f"{r['n_items']}/{r['n_subset']}", fmt(r["nomem_acc"], ".3f"), fmt(r["acc"], ".3f"),
                     f"{fmt(r['d_acc'])} [{fmt(r['d_acc_lo'])}, {fmt(r['d_acc_hi'])}]", fmt(r["d_viol"]),
                     fmt(r["d_unhelpful"]), r["testable"]] for r in T[j]]))
        w("")
    w("## Original judge: j_lean, acknowledgement, hallucination, loops, NLL")
    w("")
    w(md_table(["subset", "cond", "d j_lean [95% CI]", "ack", "halluc", "loop", "incoherent", "nll"],
               [[r["subset"], r["cond"], f"{fmt(r.get('d_j_lean'))} [{fmt(r.get('d_j_lean_lo'))}, {fmt(r.get('d_j_lean_hi'))}]",
                 fmt(r.get("ack"), ".3f"), fmt(r.get("halluc"), ".3f"), fmt(r.get("loop"), ".3f"),
                 fmt(r.get("j_incoh"), ".3f"), fmt(r.get("nll"), ".2f")] for r in T["orig"]]))
    w("")
    w("## Selectivity (alpha = 1, core_v1's 50 unrelated prompts)")
    w("")
    w(md_table(["cond", "unrel_same [95% CI]", "items", "KL"],
               [[s["cond"], f"{fmt(s['unrel_same'], '.3f')} [{fmt(s['lo'], '.3f')}, {fmt(s['hi'], '.3f')}]", s["n_items"],
                 fmt(s["kl"], ".4f")] for s in sel]))
    (adir / "report.md").write_text("\n".join(L) + "\n")
    print(f"PrefEval analysis: P1 {verdict['P1']}, P2 {verdict['P2']} ({decisive}) -> {adir}")
    return res
