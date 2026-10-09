#!/usr/bin/env python
"""core_v1 multi-judge robustness check: agreement between judges and every pre-registered judge-based
verdict (C2, C3, C5) recomputed under each judge and under majority votes. No GPU; reruns from saved files.

  python experiments/core_v1/analyze_panel.py --run <RUN root> --panel <RUN>/panel --out <RUN>/panel [--n-boot 2000]

Inputs: <RUN>/qwen35_*/{prep,gen,score}/ (the original Qwen3.5-9B judge = "orig"), <RUN>/analysis/claims_all.json,
<PANEL>/<tag>.jsonl from panel_judge.py (one record per unique prompt, expanded here to answer keys).
Outputs: panel_report.md, panel_claims.json, panel_agreement.json.

The claim logic is tools/audit/compare.py (run_c2 / run_c3 / run_c5, the pre-registered rules with its two-level
bootstrap over items then prompts), fed with another judge's labels. The item sets that survive the ctx rule and
the matched doses are the ORIGINAL run's (fixed by the pre-registered run, not re-derived per judge).

Majority votes (per answer and field; an invalid judge abstains; an answer needs a record from every judge
in the vote set):
  maj4   original + the three cross-family judges. A tie goes to the original judge's label if it is among the
         tied labels, else to "neutral" (direction) / False (booleans).
  maj3x  the three cross-family judges only (no original). A three-way direction tie -> "neutral".
The same-family judge (qwen36), if it ran, is reported alone and never votes.
"""

import argparse
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("seahorse_audit_compare", REPO / "tools" / "audit" / "compare.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

MODELS = ("qwen35_2b", "qwen35_9b")
CROSS = ("gptoss", "gemma4", "llama31")      # different families from the original Qwen judge
SAME_FAMILY = ("qwen36",)
LABEL = {"orig": "Qwen3.5-9B (original)", "gptoss": "gpt-oss (OpenAI)", "gemma4": "gemma4:12b (Google)",
         "llama31": "llama3.1 8B (Meta)", "qwen36": "qwen3.6:35b-a3b (same family)", "maj4": "majority (orig + 3)",
         "maj3x": "majority (3 cross-family)"}
FIELDS = {"fact": ["uses_fact", "coherent", "self_claim", "wrong_value"],
          "preference": ["direction", "coherent", "self_claim"]}
CATS = {"direction": ["toward", "away", "neutral"]}
GROUP_NAMES = {"base": "nomem / ctx", "mem": "mem doses", "control": "controls (rand/swap/placebo/gate)",
               "mechanism": "mechanism refs (centroid/without)"}


# ----------------------------------------------------------------------------- statistics


def fleiss_kappa(counts):
    """Fleiss' kappa. counts: [N items][k categories] = number of raters that chose each category; every
    row must sum to the same number of raters n. nan if chance agreement is 1."""
    M = np.asarray(counts, float)
    N, _ = M.shape
    n = M[0].sum()
    assert np.allclose(M.sum(1), n), "every item needs the same number of raters"
    pj = M.sum(0) / (N * n)
    Pi = ((M ** 2).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = Pi.mean(), (pj ** 2).sum()
    return float("nan") if Pe >= 1 - 1e-12 else float((Pbar - Pe) / (1 - Pe))


def fleiss_from_labels(label_lists, cats):
    """label_lists: one list of labels per rater (same length). cats: category list."""
    ix = {c: i for i, c in enumerate(cats)}
    n_items = len(label_lists[0])
    M = np.zeros((n_items, len(cats)))
    for lst in label_lists:
        for i, v in enumerate(lst):
            M[i, ix[v]] += 1
    return fleiss_kappa(M) if n_items else float("nan")


def majority(values, tie_break=None, default=None):
    """Most common value; a tie is resolved to tie_break if it is among the tied values, else `default`."""
    c = Counter(values)
    top = max(c.values())
    tied = [v for v, k in c.items() if k == top]
    if len(tied) == 1:
        return tied[0]
    if tie_break is not None and tie_break in tied:
        return tie_break
    return default


def majority_row(rows, fields, orig=None):
    """Field-wise majority of the valid voter rows. rows: list of judge rows (None = no record / invalid).
    Returns a judge-shaped row {fields..., ok, n_votes} (ok False if no valid voter)."""
    valid = [r for r in rows if r and r.get("ok")]
    if not valid:
        return {"ok": False, "n_votes": 0}
    out = {"ok": True, "n_votes": len(valid)}
    ov = orig if (orig and orig.get("ok")) else None
    for f in fields:
        vals = [r[f] for r in valid]
        out[f] = majority(vals, tie_break=ov[f] if ov else None, default="neutral" if f == "direction" else False)
    return out


# ----------------------------------------------------------------------------- loading


def load_orig(run, model):
    return {r["key"]: r for r in cmp._read_gz(Path(run) / model / "score" / "judge.jsonl.gz")}


def load_panel(panel, tag, models=MODELS):
    """{model: {key: judge row}} from <panel>/<tag>.jsonl (records are per unique prompt; expanded to keys).
    Also returns the number of unique prompts and the invalid ones."""
    out = {m: {} for m in models}
    p = Path(panel) / f"{tag}.jsonl"
    if not p.exists():
        return None, {}
    n = bad = 0
    stats = defaultdict(lambda: [0, 0])
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        n += 1
        bad += not r["ok"]
        stats[r["rubric"]][0] += 1
        stats[r["rubric"]][1] += not r["ok"]
        for mk in r["keys"]:
            m, k = mk.split(":", 1)
            if m in out:
                out[m][k] = r
    return out, {"unique": n, "invalid": bad, "by_rubric": {k: tuple(v) for k, v in stats.items()}}


def gen_meta(run, model):
    """{key: (rubric, cond, group, seedset)} for every judged answer."""
    d = Path(run) / model
    out = {}
    for f in sorted((d / "gen" / "items").glob("*.jsonl.gz")):
        for r in cmp._read_gz(f):
            if r.get("sec") == "gen":
                out[r["key"]] = ("fact" if r["group"] == "facts" else "preference", r["cond"], r["group"], r["seedset"])
    return out


def build_judges(run, panel, metas, models=MODELS):
    """{judge name: {model: {key: row}}} for orig, the panel judges that exist, and the majority votes, plus the
    per-judge validity stats."""
    J = {"orig": {m: load_orig(run, m) for m in models}}
    stats = {}
    for tag in CROSS + SAME_FAMILY:
        d, st = load_panel(panel, tag, models)
        if d is not None:
            J[tag] = d
            stats[tag] = st
    have = [t for t in CROSS if t in J]
    for name, voters, with_orig in (("maj4", have, True), ("maj3x", have, False)):
        if not voters or (name == "maj4" and not have):
            continue
        J[name] = {}
        for m in models:
            meta_keys = set(J["orig"][m])
            for v in voters:
                meta_keys &= set(J[v][m])         # an answer needs a record from every voter
            J[name][m] = {}
            for k in meta_keys:
                rows = [J[v][m][k] for v in voters] + ([J["orig"][m][k]] if with_orig else [])
                fields = FIELDS[metas[m][k][0]] if k in metas[m] else FIELDS["preference"]
                row = majority_row(rows, fields, J["orig"][m][k] if with_orig else None)
                J[name][m][k] = row
    return J, stats


# ----------------------------------------------------------------------------- agreement


def group_of(cond):
    return cmp.cond_family(cond)


def agreement_tables(J, metas, names, models=MODELS):
    """Pairwise agreement/kappa per (rubric, field) over answers where both judges are valid, overall, by
    generating model and by condition group; plus Fleiss' kappa over the answers where all `names` are valid."""
    res = {"pairs": [], "fleiss": []}
    slices = [("all", None, None)] + [(f"model={m}", m, None) for m in models] + \
             [(f"group={g}", None, g) for g in GROUP_NAMES] + [(f"{m} / {g}", m, g) for m in models for g in GROUP_NAMES]
    for rubric, fields in FIELDS.items():
        for field in fields:
            for sname, sm, sg in slices:
                def usable(m, k):
                    if sm and m != sm:
                        return False
                    mt = metas[m].get(k)
                    return bool(mt) and mt[0] == rubric and (not sg or group_of(mt[1]) == sg)
                # pairwise
                for a, b in itertools.combinations(names, 2):
                    A, B = [], []
                    for m in models:
                        ja, jb = J[a][m], J[b][m]
                        for k in ja.keys() & jb.keys():
                            if usable(m, k) and ja[k].get("ok") and jb[k].get("ok"):
                                A.append(ja[k][field]), B.append(jb[k][field])
                    res["pairs"].append({"rubric": rubric, "field": field, "slice": sname, "a": a, "b": b, "n": len(A),
                                         "agree": cmp.agreement(A, B) if A else float("nan"),
                                         "kappa": cmp.cohen_kappa(A, B) if A else float("nan")})
                # Fleiss
                lists = [[] for _ in names]
                for m in models:
                    common = None
                    for n_ in names:
                        s = {k for k, r in J[n_][m].items() if r.get("ok")}
                        common = s if common is None else common & s
                    for k in sorted(common or ()):
                        if usable(m, k):
                            for i, n_ in enumerate(names):
                                lists[i].append(J[n_][m][k][field])
                cats = CATS.get(field, [False, True])
                res["fleiss"].append({"rubric": rubric, "field": field, "slice": sname, "raters": list(names),
                                      "n": len(lists[0]), "fleiss": fleiss_from_labels(lists, cats) if lists[0] else float("nan")})
    return res


def prevalence(J, metas, names, models=MODELS):
    """Per judge and rubric field: share true (or direction shares) among the judge's valid answers on the
    answers all `names` judged, and the invalid rate (over the answers the judge has a record for)."""
    out = {}
    for n_ in names:
        for rubric, fields in FIELDS.items():
            recs = [(m, k, r) for m in models for k, r in J[n_][m].items() if metas[m].get(k, ("",))[0] == rubric]
            valid = [r for _, _, r in recs if r.get("ok")]
            o = {"n": len(recs), "invalid": len(recs) - len(valid),
                 "invalid_rate": (len(recs) - len(valid)) / len(recs) if recs else float("nan")}
            for f in fields:
                if f in CATS:
                    o[f] = {c: sum(1 for r in valid if r[f] == c) / max(len(valid), 1) for c in CATS[f]}
                else:
                    o[f] = sum(1 for r in valid if r[f]) / max(len(valid), 1)
            out[(n_, rubric)] = o
    return out


# ----------------------------------------------------------------------------- claims


def recompute_claims(run_root, model, judge_rows, claims, n_boot, seed):
    """C2, C3, C5 under one judge's labels for one generating model (uncorrected, the pre-registered rules)."""
    run = cmp.load_run(Path(run_root) / model, judge=judge_rows)
    cm = (claims or {}).get(model, {})
    out = {"dropped": run["dropped"]}
    c3 = cm.get("C3", {})
    doses = sorted(float(k.split("@")[1]) for k in c3.get("evidence", {}).get("per_dose", {})) or [0.5, 1.0, 2.0, 3.0]
    r3 = cmp.run_c3(run, doses, None, ("none",), n_boot, seed)["none"]
    out["C3"] = {"holds": r3["holds"], "holds_point": r3["holds_point"], "doses": doses, "far": {f"{d:g}": r3["far"][d] for d in doses},
                 "rise": r3["rise"], "ctx_use": r3["use"].get("ctx"),
                 "use": {f"{d:g}": r3["use"].get(f"use@{d:g}") for d in doses},
                 "ctx_minus": {f"{d:g}": r3["use"].get(f"diff@{d:g}") for d in doses},
                 "mudiff": {f"{d:g}": r3["mu"].get(f"mudiff@{d:g}") for d in doses}}
    c5 = cm.get("C5", {}).get("evidence", {})
    dis = cmp.kept_items(run, "dislikes", cm, "C5")
    al = tuple(float(c5.get(k, {}).get("alpha", 1)) for k in ("opposite", "centroid", "without"))
    r5 = cmp.run_c5(run, dis, al, None, ("none",), n_boot, seed)["none"]
    out["C5"] = {"holds": r5["_holds"], "holds_point": r5["_holds_point"], "n_items": len(dis), "alphas": al,
                 **{k: r5[k] for k in ("opposite", "centroid", "without") if k in r5}}
    lea = cmp.kept_items(run, "leanings", cm, "C2")
    r2 = cmp.run_c2(run, lea, 2.0, None, ("none",), n_boot, seed)["none"]
    out["C2"] = {"holds": r2["_holds"], "holds_point": r2["_holds_point"], "n_items": len(lea),
                 **{k: v for k, v in r2.items() if not k.startswith("_")}}
    out["pre"] = {c: cm.get(c, {}).get("holds") for c in ("C2", "C3", "C5")}
    return out


def all_claims(run_root, J, claims, n_boot, seed, models=MODELS, log=print):
    res = {}
    for name in J:
        res[name] = {}
        for m in models:
            log(f"claims: {name} / {m}")
            res[name][m] = recompute_claims(run_root, m, J[name][m], claims, n_boot, seed)
    return res


# ----------------------------------------------------------------------------- report


def pct(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def f3(x, spec=".3f"):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def ci(d, spec="+.3f"):
    return "n/a" if not d else f"{f3(d['est'], spec)} [{f3(d['lo'], spec)}, {f3(d['hi'], spec)}]"


def yn(b):
    return "yes" if b else "no"


def key_numbers(claim, r):
    if claim == "C2":
        return f"gain@2 {ci(r['gain2'])}; worst control diff lo {f3(min(r[f'diff_{c}@2']['lo'] for c in ('rand', 'swap', 'placebo')), '+.3f')}"
    if claim == "C3":
        d = max(r["use"], key=float)
        return f"ctx use {f3(r['ctx_use']['est'])}; mem@{d} use {f3(r['use'][d]['est'])}, ctx-mem {ci(r['ctx_minus'][d])}"
    return f"opp {ci(r['opposite'])}; cen {ci(r['centroid'])}; wo {ci(r['without'])}"


def claim_summary(claims_res, names):
    """{claim: {model: {judge: holds}}} and the list of judges disagreeing with the original judge."""
    out = {}
    for c in ("C2", "C3", "C5"):
        out[c] = {}
        for m in MODELS:
            h = {n: claims_res[n][m][c]["holds"] for n in names if n in claims_res}
            out[c][m] = {"holds": h, "differs_from_orig": [n for n, v in h.items() if v != h["orig"]],
                         "pre_registered": claims_res["orig"][m]["pre"][c],
                         "orig_reproduces_prereg": h["orig"] == claims_res["orig"][m]["pre"][c]}
    return out


def verdict_text(summary, names):
    lines = []
    cross = [n for n in names if n not in ("orig",) and n not in SAME_FAMILY]
    for c, label in (("C2", "C2 (leanings transfer, specifically)"), ("C3", "C3 (facts do not come out as clean use)"),
                     ("C5", "C5 (what you subtract decides concept vs direction)")):
        diffs = {m: summary[c][m]["differs_from_orig"] for m in MODELS}
        bad = {m: [n for n in d if n in cross] for m, d in diffs.items()}
        same = {m: [n for n in d if n in SAME_FAMILY] for m, d in diffs.items()}
        base = ", ".join(f"{m}: {'holds' if summary[c][m]['holds']['orig'] else 'does not hold'}" for m in MODELS)
        if not any(bad.values()):
            s = f"- **{label}: robust to the choice of judge.** Pre-registered verdict ({base}) is unchanged under every cross-family judge and the majority votes."
        else:
            s = (f"- **{label}: NOT robust.** Verdict differs from the original ({base}) under "
                 + "; ".join(f"{m}: {', '.join(LABEL[n] for n in v)}" for m, v in bad.items() if v) + ".")
        if any(same.values()):
            s += " The same-family judge also differs on " + ", ".join(m for m, v in same.items() if v) + "."
        lines.append(s)
    return "\n".join(lines)


def build_report(J, stats, agree, prev, claims_res, summary, names, meta):
    o = ["# core_v1 multi-judge robustness check\n"]
    o.append("The 34,200 core_v1 answers (17,100 per generating model) were re-labelled with judges from other model families, "
             "using the SAME rubric prompts and parser as the original Qwen3.5-9B judge (only the judge model changes; "
             "temperature 0, fixed seed, JSON-constrained output, reasoning off or low). Every pre-registered C2 / C3 / C5 verdict "
             "was recomputed with the same rules and bootstrap (tools/audit/compare.py) under each judge alone and under majority votes. "
             "The item sets (ctx rule) and matched doses are the original run's.\n")
    o.append("## Verdict\n")
    o.append(verdict_text(summary, names) + "\n")
    o.append("Limitation: agreement between judges shows the verdicts do not depend on WHICH judge labelled them, not that the "
             "labels are correct (the judges share failure modes, e.g. all are LLMs reading the same text). Correctness is what the "
             "human audit (tools/audit) addresses.\n")
    o.append("## Claim x judge: does the claim hold?\n")
    hdr = ["claim", "model", "pre-reg"] + [LABEL[n] for n in names]
    rows = []
    for c in ("C2", "C3", "C5"):
        for m in MODELS:
            rows.append([c, m, yn(summary[c][m]["pre_registered"]) if summary[c][m]["pre_registered"] is not None else "?"]
                        + [f"**{yn(summary[c][m]['holds'][n])}**" + ("" if summary[c][m]["holds"][n] == summary[c][m]["holds"]["orig"] else " (differs)")
                           for n in names])
    o.append(cmp.md_table(hdr, rows) + "\n")
    o.append("`holds` uses the pre-registered CI rules. A judge's invalid outputs are dropped (as in the original analysis); "
             "same-family judge (if present) is for reference and does not vote.\n")
    o.append("## Key numbers (point estimate [95% CI])\n")
    for c in ("C2", "C3", "C5"):
        o.append(f"### {c}\n")
        rows = []
        for m in MODELS:
            for n in names:
                r = claims_res[n][m][c]
                rows.append([m, LABEL[n], yn(r["holds"]), yn(r["holds_point"]), key_numbers(c, r), claims_res[n][m]["dropped"]])
        o.append(cmp.md_table(["model", "judge", "holds (CI rule)", "holds (point)", "key numbers", "answers dropped (invalid)"], rows) + "\n")
    o.append("C2 detail (j_lean gains vs nomem, per judge):\n")
    rows = []
    for m in MODELS:
        for n in names:
            r = claims_res[n][m]["C2"]
            rows.append([m, LABEL[n], ci(r["gain1"]), ci(r["gain2"])] + [ci(r[f"diff_{c}@2"]) for c in ("rand", "swap", "placebo")])
    o.append(cmp.md_table(["model", "judge", "gain mem@1", "gain mem@2", "mem2 - rand", "mem2 - swap", "mem2 - placebo"], rows) + "\n")
    o.append("C3 detail (judge clean use on the use prompts; ctx - mem must have CI above 0 and mem <= 1/2 ctx at every dose):\n")
    rows = []
    for m in MODELS:
        for n in names:
            r = claims_res[n][m]["C3"]
            rows.append([m, LABEL[n], f3(r["ctx_use"]["est"])] + [f"{f3(r['use'][d]['est'])} ({yn(r['far'][d])})" for d in r["use"]] + [yn(r["rise"])])
    ds = list(claims_res["orig"][MODELS[0]]["C3"]["use"])
    o.append(cmp.md_table(["model", "judge", "ctx"] + [f"mem@{d} (far below)" for d in ds] + ["mention_unclean rises"], rows) + "\n")
    o.append("C5 detail (j_lean gain vs nomem at the matched doses; opposite CI > 0, centroid and without CI < 0):\n")
    rows = []
    for m in MODELS:
        for n in names:
            r = claims_res[n][m]["C5"]
            rows.append([m, LABEL[n], ci(r["opposite"]), ci(r["centroid"]), ci(r["without"])])
    o.append(cmp.md_table(["model", "judge", "opposite", "centroid", "without"], rows) + "\n")

    o.append("## Parse validity\n")
    rows = []
    for tag, st in stats.items():
        for rub, (n, bad) in st["by_rubric"].items():
            rows.append([LABEL[tag], rub, n, bad, pct(bad / n if n else float("nan"))])
    o.append("Unique prompts per judge (identical prompts are judged once); invalid = no field set parsed after one retry. "
             "The original judge's invalid rate is in the next table.\n")
    o.append(cmp.md_table(["judge", "rubric", "unique prompts", "invalid", "invalid rate"], rows) + "\n")
    rows = []
    for (n, rub), p in prev.items():
        fl = ", ".join(f"{f}={'/'.join(f'{c} {pct(v)}' for c, v in p[f].items()) if isinstance(p[f], dict) else pct(p[f])}"
                       for f in FIELDS[rub])
        rows.append([LABEL[n], rub, p["n"], pct(p["invalid_rate"]), fl])
    o.append("Label rates per judge (answers the judge has a record for; majority rows are per answer):\n")
    o.append(cmp.md_table(["judge", "rubric", "answers", "invalid", "label rates (valid answers)"], rows) + "\n")

    o.append("## Agreement\n")
    o.append(f"Raters in the pairwise tables and Fleiss' kappa: {', '.join(LABEL[n] for n in agree['names'])}. "
             "Pairwise: answers where both judges are valid. Fleiss: answers where all raters are valid. "
             "Groups: nomem / ctx = no memory or in-context; mem doses = mem@alpha; controls and mechanism references listed separately "
             "(loops and garbled text, where judges may disagree, are concentrated at high doses).\n")
    for rubric, fields in FIELDS.items():
        for field in fields:
            o.append(f"### {rubric} / {field}\n")
            fl = {(x["slice"]): x for x in agree["tables"]["fleiss"] if x["rubric"] == rubric and x["field"] == field}
            rows = []
            for sl in [s for s in dict.fromkeys(x["slice"] for x in agree["tables"]["pairs"])]:
                if sl not in fl:
                    continue
                pairs = [x for x in agree["tables"]["pairs"] if x["rubric"] == rubric and x["field"] == field and x["slice"] == sl]
                kap = [x["kappa"] for x in pairs if not math.isnan(x["kappa"])]
                ag = [x["agree"] for x in pairs if not math.isnan(x["agree"])]
                rows.append([sl, fl[sl]["n"], f3(fl[sl]["fleiss"], ".2f"),
                             f"{pct(float(np.mean(ag)))} ({pct(min(ag))}-{pct(max(ag))})" if ag else "nan",
                             f"{f3(float(np.mean(kap)), '.2f')} ({f3(min(kap), '.2f')} to {f3(max(kap), '.2f')})" if kap else "nan"])
            o.append(cmp.md_table(["slice", "n (all valid)", "Fleiss kappa", "pairwise agreement mean (range)", "Cohen kappa mean (range)"], rows) + "\n")
            pr = [x for x in agree["tables"]["pairs"] if x["rubric"] == rubric and x["field"] == field and x["slice"] == "all"]
            o.append("Pairwise, all answers (agreement / Cohen kappa / n):\n")
            o.append(cmp.md_table(["judge A", "judge B", "agreement", "kappa", "n"],
                                  [[LABEL[x["a"]], LABEL[x["b"]], pct(x["agree"]), f3(x["kappa"], ".2f"), x["n"]] for x in pr]) + "\n")
    o.append("## Method notes\n")
    o.append(meta + "\n")
    return "\n".join(o)


METHOD = """- Judges (Ollama, one A100 80GB): gpt-oss (reasoning low, final content parsed only), gemma4:12b (thinking off), llama3.1 8B (weaker, distinct family); optional qwen3.6:35b-a3b (same family as the original judge; reference only, never votes). temperature 0, seed fixed, `format` = JSON schema of the rubric's fields, num_ctx 4096.
- Prompts: judge_prompts.yaml filled exactly as run.judge_text does (byte-identical, tested). Parsing: the original `core.parse_judge`, after stripping think blocks and code fences. One retry (different seed, unconstrained output); still failing -> invalid, counted above and dropped from that judge's claim recomputation (like the original judge's parse failures).
- Majority votes: see the module docstring of analyze_panel.py (invalid judges abstain; a tie goes to the original judge's label if it is among the tied labels, else neutral / False). maj4 = original + 3 cross-family; maj3x = 3 cross-family.
- Claims: tools/audit/compare.py run_c2 / run_c3 / run_c5 (uncorrected), n_boot as given, seed 7. Pre-registered item sets and matched doses are taken from the original run (claims_all.json, flags.csv)."""


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--claims", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cp = Path(a.claims) if a.claims else Path(a.run) / "analysis" / "claims_all.json"
    claims = json.load(open(cp)) if cp.exists() else None
    if claims is None:
        print(f"note: no claims file at {cp}; pre-registered verdicts will be shown as '?'")
    metas = {m: gen_meta(a.run, m) for m in MODELS}
    J, stats = build_judges(a.run, a.panel, metas)
    print("judges:", {k: sum(len(v) for v in d.values()) for k, d in J.items()})
    names = [n for n in ("orig",) + CROSS + SAME_FAMILY + ("maj4", "maj3x") if n in J]
    raters = [n for n in ("orig",) + CROSS if n in J]
    ag_tables = agreement_tables(J, metas, raters)
    agree = {"names": raters, "tables": ag_tables}
    prev = prevalence(J, metas, [n for n in names])
    cres = all_claims(a.run, J, claims, a.n_boot, a.seed)
    summary = claim_summary(cres, names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "panel_report.md").write_text(build_report(J, stats, agree, prev, cres, summary, names, METHOD), encoding="utf-8")
    json.dump({"judges": names, "summary": summary, "claims": cres, "validity": stats}, open(out / "panel_claims.json", "w"),
              indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    json.dump(ag_tables, open(out / "panel_agreement.json", "w"), indent=1)
    print(f"wrote {out}/panel_report.md, panel_claims.json, panel_agreement.json")


if __name__ == "__main__":
    main()
