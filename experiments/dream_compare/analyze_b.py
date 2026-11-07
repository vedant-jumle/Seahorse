#!/usr/bin/env python
"""dream_compare Stage B analysis (PREREG.md "Stage B": measures and rules B1-B5). Standard library + numpy.

Inputs: a model's gen dir (<bench>_<split>/items/*.json.gz, contexts.jsonl) and judge dir (orig.jsonl, gptoss.jsonl,
gemma4.jsonl: one record per unique prompt, "keys" = model|bench_split|item|cond|rubric index).

Correctness per (item, condition):
  LongMemEval   the panel majority of the per-type yes/no prompts (tie -> the original judge)
  LoCoMo        non-adversarial: the panel majority of Mem0's J (+ the official F1, reported); adversarial: the
                official category-5 rule (the answer says "not mentioned" / "no information available")
  BEAM          per rubric item the median of the judges' scores (two judges: their mean), the item = the mean
  PersonaMem    the option with the highest summed log-probability is the labelled one
For the Gemma-4 subject model the same is also computed without the Gemma-4 judge ("_xfam").

  python -I experiments/dream_compare/analyze_b.py --gen RUN/<model>/gen --judge RUN/<model>/judge --out RUN/<model>
"""

import argparse
import gzip
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


J = _load("dream_compare_judges", HERE / "judges.py")

N_BOOT, SEED = 2000, 7
MARGIN = 0.03                      # B1's equivalence margin
JUDGES = ("orig", "gptoss", "gemma4")
FAMILY = {"gemma4_26b_a4b": "Google"}
CONDS_GEN = ["P0", "P1", "P2-3", "P2-10", "P3-3", "P3-10", "P4loo-3", "P4loo-10", "P4dream-3", "P4dream-10",
             "P5-3", "P5-10", "P5s-3", "P5s-10"]
CONDS_PM = ["P0", "P1", "P2-1", "P2-3", "P3-1", "P3-3", "P4loo-1", "P4loo-3", "P4dream-1", "P4dream-3",
            "P5-1", "P5-3", "P5s-1", "P5s-3"]
KU = {"longmemeval": "knowledge-update", "beam": "knowledge_update"}


def conds_of(bs):
    return CONDS_PM if bs.startswith("personamem") else CONDS_GEN


def ms_of(bs):
    return (1, 3) if bs.startswith("personamem") else (3, 10)


# ----------------------------------------------------------------------------------------------- loading


def read_items(gen_dir):
    out = {}
    for d in sorted(Path(gen_dir).glob("*_*")):
        if (d / "items").is_dir():
            out[d.name] = {}
            for p in sorted((d / "items").glob("*.json.gz")):
                r = json.load(gzip.open(p, "rt"))
                out[d.name][r["item"]] = r
    return out


def read_contexts(gen_dir):
    out = defaultdict(dict)
    for d in sorted(Path(gen_dir).glob("*_*")):
        p = d / "contexts.jsonl"
        if p.exists():
            for l in open(p):
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                out[d.name][r["context"]] = r         # the last run of a context wins
    return out


def read_votes(judge_dir):
    """{key: {judge: label (bool) or score (float) or None}}."""
    votes = defaultdict(dict)
    if not judge_dir:
        return votes
    # the keys of each prompt come from the current tasks.jsonl when it exists: a judge record keeps the keys its
    # prompt had when it was judged, and a resumed judge run (new items after a resubmitted gen job) can add keys to
    # a prompt already judged (an identical prompt from another item); its judgement is reused, not repeated
    tkeys = {}
    tp = Path(judge_dir) / "tasks.jsonl"
    if tp.exists():
        for l in open(tp, encoding="utf-8"):
            try:
                t = json.loads(l)
            except ValueError:
                continue
            tkeys[t["h"]] = t["keys"]
    for j in JUDGES:
        p = Path(judge_dir) / f"{j}.jsonl"
        if not p.exists():
            continue
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            v = (r.get("score") if r.get("kind") == "beam" else r.get("label")) if r.get("ok") else None
            for k in tkeys.get(r.get("h"), r.get("keys", [])):
                votes[k][j] = v
    return votes


def panel_bool(v, subject_family=None):
    return J.panel_label({j: v.get(j) for j in JUDGES}, subject_family)


def panel_score(v, judges=JUDGES):
    s = [v[j] for j in judges if v.get(j) is not None]
    if not s:
        return None
    return float(statistics.median(s))


# ----------------------------------------------------------------------------------------------- correctness


def correctness(items, votes, model, bs):
    """{cond: {item: value}} primary, and the without-same-family variant (None unless a judge shares the family)."""
    fam = FAMILY.get(model)
    prim, xfam = defaultdict(dict), defaultdict(dict)
    for iid, r in items.items():
        for cond, c in r["conds"].items():
            if bs.startswith("personamem"):
                if "correct" in c:
                    prim[cond][iid] = float(c["correct"])
                continue
            base = f"{model}|{bs}|{iid}|{cond}"
            if r["bench"] == "locomo" and r["abstention"]:
                if "text" in c:
                    prim[cond][iid] = float(J.locomo_f1(5, c["text"], "") > 0)
                    xfam[cond][iid] = prim[cond][iid]
                continue
            if r["bench"] == "beam":
                sc, sx = [], []
                for i in range(r.get("n_rubric", 0)):
                    v = votes.get(f"{base}|{i}", {})
                    sc.append(panel_score(v))
                    sx.append(panel_score(v, [j for j in JUDGES if J.JUDGE_FAMILY.get(j) != fam]) if fam else None)
                s = J.beam_item_score(sc)
                if s is not None:
                    prim[cond][iid] = s
                sxv = J.beam_item_score(sx)
                if sxv is not None:
                    xfam[cond][iid] = sxv
                continue
            v = votes.get(f"{base}|0")
            if not v:
                continue
            a, b = panel_bool(v, fam)
            if a is not None:
                prim[cond][iid] = float(a)
            if b is not None:
                xfam[cond][iid] = float(b)
    return prim, xfam


# ----------------------------------------------------------------------------------------------- statistics


def boot_mean(vals, n=N_BOOT, seed=SEED, level=0.95):
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def paired(x, y, level=0.95):
    """Mean of x - y over the items both have, with a bootstrap CI over items."""
    ks = sorted(set(x) & set(y))
    d = [x[k] - y[k] for k in ks]
    m, lo, hi = boot_mean(d, level=level)
    return {"n": len(ks), "diff": m, "lo": lo, "hi": hi}


def tost(x, y):
    """B1: equivalent iff the 90% CI of the paired difference lies inside (-MARGIN, MARGIN) (TOST at 0.05);
    otherwise 'different' if the 95% CI excludes 0, else 'inconclusive'."""
    r90 = paired(x, y, 0.90)
    r95 = paired(x, y, 0.95)
    if r90["n"] == 0:
        return {"verdict": "no data", **r90}
    if r90["lo"] > -MARGIN and r90["hi"] < MARGIN:
        v = "equivalent"
    elif r95["lo"] > 0 or r95["hi"] < 0:
        v = "different (" + ("P3 better" if r95["diff"] > 0 else "P2 better") + ")"
    else:
        v = "inconclusive"
    return {"verdict": v, "n": r90["n"], "diff": r90["diff"], "lo90": r90["lo"], "hi90": r90["hi"],
            "lo95": r95["lo"], "hi95": r95["hi"]}


def pooled_tost(cells):
    """Pooled B1 over cells [(x, y)]: the mean over cells of each cell's mean paired difference; bootstrap resamples
    items within every cell (2000, seed 7)."""
    ds = []
    for x, y in cells:
        ks = sorted(set(x) & set(y))
        if ks:
            ds.append(np.array([x[k] - y[k] for k in ks]))
    if not ds:
        return {"verdict": "no data"}
    rng = np.random.default_rng(SEED)
    bs = np.zeros(N_BOOT)
    for d in ds:
        bs += d[rng.integers(0, d.size, (N_BOOT, d.size))].mean(1)
    bs /= len(ds)
    m = float(np.mean([d.mean() for d in ds]))
    lo90, hi90 = np.quantile(bs, [0.05, 0.95])
    lo95, hi95 = np.quantile(bs, [0.025, 0.975])
    if lo90 > -MARGIN and hi90 < MARGIN:
        v = "equivalent"
    elif lo95 > 0 or hi95 < 0:
        v = "different (" + ("P3 better" if m > 0 else "P2 better") + ")"
    else:
        v = "inconclusive"
    return {"verdict": v, "cells": len(ds), "diff": m, "lo90": float(lo90), "hi90": float(hi90),
            "lo95": float(lo95), "hi95": float(hi95)}


def pareto(points):
    """Names of the non-dominated points {name: (cost, correctness)} (lower cost, higher correctness)."""
    out = []
    for a, (ca, sa) in points.items():
        if ca is None or sa is None:
            continue
        dom = any((cb <= ca and sb >= sa and (cb < ca or sb > sa)) for b, (cb, sb) in points.items()
                  if b != a and cb is not None and sb is not None)
        if not dom:
            out.append(a)
    return sorted(out, key=lambda n: points[n][0])


def cond_storage(cond, st, vonly=False):
    """At-rest bytes of a condition's paradigm (P4loo / P4dream -> P4, P5s -> P5)."""
    return st.get(cond[:2] + ("_vonly" if vonly else ""))


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


# ----------------------------------------------------------------------------------------------- per model


def analyze(gen_dir, judge_dir, out, model):
    out = Path(out)
    items = read_items(gen_dir)
    ctxs = read_contexts(gen_dir)
    votes = read_votes(judge_dir)
    res = {"model": model, "benches": {}, "rules": {}}
    lines = [f"# dream_compare Stage B: {model}", "",
             "Correctness = each benchmark's official scoring (panel majority; PersonaMem: option log-prob). "
             "95% bootstrap CIs over items (2000, seed 7). Storage = mean at-rest bytes per conversation.", ""]
    b1_cells = defaultdict(list)
    for bs, its in items.items():
        conds = conds_of(bs)
        prim, xfam = correctness(its, votes, model, bs)
        bench = bs.split("_")[0]
        tab = {}
        for c in conds:
            m, lo, hi = boot_mean(list(prim.get(c, {}).values()))
            ref = [r["conds"][c]["refusal"] for r in its.values() if c in r["conds"] and "refusal" in r["conds"][c]]
            sref = [x for r in its.values() if c in r["conds"] for x in r["conds"][c].get("sample_refusal", [])]
            abst = [v for k, v in prim.get(c, {}).items() if its[k]["abstention"]]
            ku = [v for k, v in prim.get(c, {}).items() if its[k]["category"] == KU.get(bench)]
            st = [cond_storage(c, r["storage"]) for r in its.values() if c in r["conds"]]
            stv = [cond_storage(c, r["storage"], True) for r in its.values() if c in r["conds"]]
            lat = [r["latency"][c]["first_token_s"] for r in its.values() if r.get("latency") and c in r["latency"]]
            f1 = [r["conds"][c]["f1"] for r in its.values() if c in r["conds"] and "f1" in r["conds"][c]
                  and not r["abstention"]]
            fam, mm = c.split("-")[0], (int(c.split("-")[1]) if "-" in c else None)
            rec_m = [r["recall"].get(str(mm)) for r in its.values() if mm and r["recall"].get(str(mm)) is not None]
            tab[c] = {"n": len(prim.get(c, {})), "correct": m, "lo": lo, "hi": hi,
                      "correct_xfam": mean(list(xfam.get(c, {}).values())) if xfam.get(c) else None,
                      "refusal": mean([float(x) for x in ref]), "refusal_samples": mean([float(x) for x in sref]),
                      "abstention": mean(abst), "n_abstention": len(abst), "knowledge_update": mean(ku),
                      "n_ku": len(ku), "f1": mean(f1), "recall": mean(rec_m),
                      "storage_bytes": mean(st), "storage_bytes_vonly": mean(stv),
                      "prefill_tokens": mean([r["conds"][c]["prefill_tokens"] for r in its.values() if c in r["conds"]]),
                      "restored_entries": mean([r["conds"][c]["restored_entries"] for r in its.values() if c in r["conds"]]),
                      "ttft_s_median": float(np.median(lat)) if lat else None,
                      "n_answered": sum(c in r["conds"] for r in its.values())}
        skipped = Counter(k for r in its.values() for k in r.get("skipped", {}))
        # rules
        rules = {}
        for m in ms_of(bs):
            p2, p3 = prim.get(f"P2-{m}", {}), prim.get(f"P3-{m}", {})
            rules[f"B1_m{m}"] = tost(p3, p2)
            b1_cells[m].append((p3, p2))
            for sel in ("loo", "dream"):
                p4 = prim.get(f"P4{sel}-{m}", {})
                sp3 = mean([r["storage"]["kv_all"] for r in its.values()])
                sp4 = mean([r["storage"]["kv_p4"] for r in its.values()])
                rules[f"B2_{sel}_m{m}"] = dict(paired(p4, p3), storage_saving_factor=(sp3 / sp4) if sp4 else None)
                rules[f"B3_{sel}_m{m}_vs_P5s"] = paired(p4, prim.get(f"P5s-{m}", {}))
                rules[f"B3_{sel}_m{m}_vs_P5"] = paired(p4, prim.get(f"P5-{m}", {}))
            ref = lambda c: {k: float(r["conds"][c]["refusal"]) for k, r in its.items()
                             if c in r["conds"] and "refusal" in r["conds"][c]}
            if not bs.startswith("personamem"):
                rules[f"B5_m{m}"] = {"P3_minus_P2": paired(ref(f"P3-{m}"), ref(f"P2-{m}")),
                                     "P4loo_minus_P3": paired(ref(f"P4loo-{m}"), ref(f"P3-{m}")),
                                     "P4dream_minus_P3": paired(ref(f"P4dream-{m}"), ref(f"P3-{m}")),
                                     "P5_minus_P3": paired(ref(f"P5-{m}"), ref(f"P3-{m}"))}
        pts = {c: (tab[c]["storage_bytes"], tab[c]["correct"]) for c in conds}
        ptl = {c: (tab[c]["prefill_tokens"], tab[c]["correct"]) for c in conds}
        ptt = {c: (tab[c]["ttft_s_median"], tab[c]["correct"]) for c in conds}
        rules["B4"] = {"storage": pareto(pts), "prefill_tokens": pareto(ptl), "ttft": pareto(ptt)}
        cx = ctxs.get(bs, {})
        res["benches"][bs] = {"table": tab, "rules": rules, "n_items": len(its), "skipped": dict(skipped),
                              "contexts": len(cx),
                              "encode_s": sum(r["encode_s"] for r in cx.values()),
                              "select_s": sum(r["select_s"] for r in cx.values()),
                              "gen_s": sum(r["gen_s"] for r in cx.values()),
                              "loo_fallback": None}
        lines += [f"## {bs} ({len(its)} items; P1 skipped {skipped.get('P1', 0)})", "",
                  "| cond | n | correct [95% CI] | refusal | refusal (samples) | abstention | KU | recall@m | "
                  "storage MB/conv | prefill tok | restored | TTFT s |", "|" + "---|" * 12]
        for c in conds:
            t = tab[c]
            f = lambda v, nd=3: "-" if v is None else f"{v:.{nd}f}"
            lines.append(f"| {c} | {t['n']} | {f(t['correct'])} [{f(t['lo'])}, {f(t['hi'])}] | {f(t['refusal'])} | "
                         f"{f(t['refusal_samples'])} | {f(t['abstention'])} | {f(t['knowledge_update'])} | "
                         f"{f(t['recall'])} | {f((t['storage_bytes'] or 0) / 1e6, 2) if t['storage_bytes'] is not None else '-'} | "
                         f"{f(t['prefill_tokens'], 0)} | {f(t['restored_entries'], 0)} | {f(t['ttft_s_median'], 3)} |")
        lines += ["", "Rules:", ""]
        for k, v in rules.items():
            lines.append(f"- {k}: {json.dumps(v)}")
        lines.append("")
    res["rules"]["B1_pooled"] = {f"m{m}": pooled_tost(cells) for m, cells in b1_cells.items()}
    lines += ["## B1 pooled over this model's benchmarks", ""] + \
             [f"- {k}: {json.dumps(v)}" for k, v in res["rules"]["B1_pooled"].items()]
    json.dump(res, open(out / "verdict_b.json", "w"), indent=1)
    open(out / "report_b.md", "w").write("\n".join(lines) + "\n")
    print(f"wrote {out / 'report_b.md'}")
    return res


def summary(root):
    root = Path(root)
    rows, cells = [], defaultdict(list)
    lines = ["# dream_compare Stage B: cross-model summary", ""]
    for d in sorted(root.iterdir()):
        v = d / "verdict_b.json"
        if not v.exists():
            continue
        r = json.load(open(v))
        for bs, b in r["benches"].items():
            for k, x in b["rules"].items():
                if k.startswith("B1_m"):
                    lines.append(f"- {r['model']} {bs} {k}: {x.get('verdict')} diff {x.get('diff')}")
            items = read_items(d / "gen").get(bs, {})
            votes = read_votes(d / "judge")
            prim, _ = correctness(items, votes, r["model"], bs)
            for m in ms_of(bs):
                cells[m].append((prim.get(f"P3-{m}", {}), prim.get(f"P2-{m}", {})))
        rows.append(r["model"])
    pooled = {f"m{m}": pooled_tost(c) for m, c in cells.items()}
    lines += ["", "## B1 pooled over every model x benchmark", ""] + [f"- {k}: {json.dumps(v)}" for k, v in pooled.items()]
    json.dump({"models": rows, "B1_pooled": pooled}, open(root / "summary_b.json", "w"), indent=1)
    open(root / "summary_b.md", "w").write("\n".join(lines) + "\n")
    print(f"wrote {root / 'summary_b.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen")
    ap.add_argument("--judge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    if a.summary:
        summary(Path(a.out))
    else:
        analyze(a.gen, a.judge, Path(a.out), a.model)


if __name__ == "__main__":
    main()
