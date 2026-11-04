#!/usr/bin/env python
"""dream_select generation analysis (PREREG.md addendum 1): judged correctness per condition, the generation rule, the
cross-model summary. Also the pure helpers the generation, judge and panel stages share (no torch at import; unit-tested
in test_dream_select.py).

  python -I experiments/dream_select/run.py --stage analyze-gen --out RUN/<model>     # -> report_gen.md, verdict_gen.json
  python -I experiments/dream_select/run.py --stage summary --out RUN                 # -> RUN/summary.md

The judge criterion is LongMemEval's own correctness check for single-session-user questions (LME_TEMPLATE, verbatim
from the LongMemEval repo, src/evaluation/evaluate_qa.py, get_anscheck_prompt; commit d6dc8b5). Label = LongMemEval's
parse ("yes" in the lower-cased reply); a reply with neither "yes" nor "no" as a word is unparsed (a missing vote).
Primary label = the majority of the original judge (Qwen3.5-9B), gpt-oss and Gemma-4; a tie goes to the original.
"""

import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from seahorse.stats import boot_ci, paired_diff  # noqa: E402

DATA = REPO / "data" / "longmemeval" / "ssu70.json"
GEN_ANCHORS = ("nomem", "ctx", "R-full", "R-content")
GEN_SELECTORS = ("S-dream", "S-attn", "S-recon", "S-loo", "random", "gold@k")
GEN_BUDGETS = ("k4", "k8")
MAIN_K = "k4"
JUDGES = ("orig", "gptoss", "gemma4")
PANEL = ("gptoss", "gemma4")
N_BOOT, SEED = 2000, 7
EPS = 1e-9                  # |bound| < EPS counts as 0 (dream_bind addendum 9)
MIN_DEN = 0.05              # gap closed in correctness: n/a when R-full - nomem < 0.05
GEN_MODELS = ("qwen25_15b", "qwen25_7b", "llama32_3b")

# LongMemEval, src/evaluation/evaluate_qa.py, get_anscheck_prompt(task='single-session-user', abstention=False),
# verbatim (fetched 2026-11-04, repo xiaowu0162/LongMemEval at commit d6dc8b50a2d9ac0c99485ea28fa5755c62414c34; file
# sha256 ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251). Filled with str.format(question, answer,
# response), as there.
LME_TEMPLATE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
                "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
                "correct answer or contains all the intermediate steps to get the correct answer, you should also "
                "answer yes. If the response only contains a subset of the information required by the answer, answer "
                "no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
                "Answer yes or no only.")
LME_SOURCE = {"repo": "xiaowu0162/LongMemEval", "file": "src/evaluation/evaluate_qa.py",
              "commit": "d6dc8b50a2d9ac0c99485ea28fa5755c62414c34",
              "file_sha256": "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
              "function": "get_anscheck_prompt", "task": "single-session-user", "max_tokens": 10, "temperature": 0,
              "label": "'yes' in reply.lower()"}
_YN = re.compile(r"\b(yes|no)\b", re.I)
# dream_bind addendum 7 (B3)'s refusal regex (run.REFUSAL_RE there), copied so that this module does not import torch;
# test_dream_select checks that it is identical
REFUSAL_RE = re.compile(r"\b(don't|do not|can't|cannot|couldn't)\s+(know|recall|remember|have|access|tell)\b|"
                        r"\bI'm sorry\b|\bcould you (tell|share|remind)\b", re.I)


# ----------------------------------------------------------------------------------------------- judge prompt + parse


def lme_prompt(question, answer, response):
    """LongMemEval's single-session-user correctness prompt; an empty response is given as "(empty reply)"."""
    resp = response if (response or "").strip() else "(empty reply)"
    return LME_TEMPLATE.format(question, answer, resp)


def repair(text):
    """Drop <think>/<thinking> blocks and code fences (core_v1 panel_judge.repair)."""
    t = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text or "", flags=re.S | re.I)
    t = re.sub(r"```(?:json)?", "", t)
    return t.strip()


def parse_yesno(text):
    """A judge reply -> {"label": True/False, or None if unparsed; "ok"; "official"}. official = LongMemEval's rule
    ("yes" in the lower-cased reply); ok = "yes" or "no" occurs as a word; label = official if ok else None."""
    t = repair(text)
    ok = bool(_YN.search(t))
    official = "yes" in t.lower()
    return {"label": official if ok else None, "ok": ok, "official": official}


# ----------------------------------------------------------------------------------------------- generation layout


def gen_conditions(sels, verbatim, budgets):
    """One item's generation conditions [(name, cond, budget, selection or None)] (PREREG addendum 1), from its tf
    selections sels = {(cond, budget): [selection per seed]}: the anchors; gold (verbatim items); then per budget
    S-dream, S-attn, S-recon, S-loo, random (seed 0 = the first of the three seeds) and gold@k (verbatim items). A
    selection missing from the tf run (S-loo on a turn over the cap) is skipped."""
    out = [(c, c, None, None) for c in GEN_ANCHORS]
    if verbatim:
        out.append(("gold", "gold", "own", list(sels[("gold", "own")][0])))
    for b in budgets:
        for c in GEN_SELECTORS:
            if c == "gold@k" and not verbatim:
                continue
            if (c, b) in sels:
                out.append((f"{c}@{b}", c, b, list(sels[(c, b)][0])))
    return out


def compare_selections(rows_a, rows_b):
    """Information only (the 1.5B smoke): per (cond, budget), how many items' picks are identical in two tf runs."""
    sa = {(r["item"], r["cond"], r["budget"]): r["sel"] for r in rows_a if "sel" in r}
    sb = {(r["item"], r["cond"], r["budget"]): r["sel"] for r in rows_b if "sel" in r}
    out = defaultdict(lambda: [0, 0])
    for k, v in sa.items():
        if k in sb:
            out[f"{k[1]}@{k[2]}"][0] += v == sb[k]
            out[f"{k[1]}@{k[2]}"][1] += 1
    return {k: {"same": a, "n": n} for k, (a, n) in sorted(out.items())}


# ----------------------------------------------------------------------------------------------- reading


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_gen(model_dir):
    """(gen rows, per-item info rows) of a model dir's gen/items/*.jsonl.gz."""
    G, I = [], {}
    for f in sorted((Path(model_dir) / "gen" / "items").glob("*.jsonl.gz")):
        for r in read_gz(f):
            if r.get("sec") == "info":
                I[r["item"]] = r
            else:
                G.append(r)
    return G, I


def read_panel(path):
    """{answer key: label or None} of a panel file (one record per unique prompt, keys listed; a torn line is
    skipped). The last ok record of a key wins (relink aliases copy labels)."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        lab = rec.get("label") if rec.get("ok") else None
        for k in rec.get("keys", []):
            if lab is not None or k not in out:
                out[k] = lab
    return out


def load_votes(model_dir):
    """{judge: {key: label or None}} for the original judge and the panel judges present."""
    d = Path(model_dir)
    votes = {}
    f = d / "score" / "judge.jsonl.gz"
    if f.exists():
        votes["orig"] = {r["key"]: r.get("label") for r in read_gz(f)}
    for jt in PANEL:
        p = d / "panel" / f"{jt}.jsonl"
        if p.exists():
            votes[jt] = read_panel(p)
    return votes


def majority_labels(keys, votes):
    """{key: majority label} of the judges' votes (missing = None); a tie goes to the original judge
    (dream_bind bind.majority)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dream_select_bind_ag", REPO / "experiments" / "dream_bind" / "bind.py")
    bind = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bind)
    out = {}
    for k in keys:
        v = [votes.get(j, {}).get(k) for j in JUDGES]
        out[k] = bind.majority(v, tie=v[0])
    return out


# ----------------------------------------------------------------------------------------------- statistics


def nested_of(rows, fn):
    """{item: {0: mean of fn(row) over the item's answers}} (None values skipped; an item with none is left out)."""
    acc = defaultdict(list)
    for r in rows:
        v = fn(r)
        if v is not None:
            acc[r["item"]].append(float(v))
    return {i: {0: sum(v) / len(v)} for i, v in acc.items()}


def ci(nested):
    p, lo, hi, n = boot_ci(nested, N_BOOT, SEED)
    return {"mean": p, "lo": lo, "hi": hi, "n": n}


def paired_ci(a, b):
    return ci(paired_diff(a, b))


def gap_ratio(cond, nomem, full, n_boot=N_BOOT, seed=SEED, min_den=MIN_DEN):
    """Gap closed in judged correctness, (mean cond - mean nomem) / (mean R-full - mean nomem), over the items present
    in all three, with a percentile CI from the item bootstrap (the ratio per replicate; replicates with a denominator
    below min_den are dropped). n/a (mean None) when the point denominator < min_den."""
    items = sorted(set(cond) & set(nomem) & set(full))
    if not items:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    c = np.array([cond[i][0] for i in items])
    m = np.array([nomem[i][0] for i in items])
    f = np.array([full[i][0] for i in items])
    den = f.mean() - m.mean()
    if den < min_den:
        return {"mean": None, "lo": None, "hi": None, "n": len(items), "den": float(den)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(n_boot, len(items)))
    dd = f[idx].mean(1) - m[idx].mean(1)
    ok = dd >= min_den
    r = (c[idx].mean(1)[ok] - m[idx].mean(1)[ok]) / dd[ok]
    lo, hi = (np.quantile(r, [0.025, 0.975]) if len(r) else (float("nan"), float("nan")))
    return {"mean": float((c.mean() - m.mean()) / den), "lo": float(lo), "hi": float(hi), "n": len(items),
            "den": float(den)}


def gen_rule(d):
    """The pre-registered generation rule: the paired CI of (S-dream - random) at k = 4 on judged correctness has a
    lower bound > 0 (|bound| < EPS counts as 0). Returns (holds, reason)."""
    lo = d["lo"]
    ok = d["n"] > 0 and lo is not None and math.isfinite(lo) and lo > EPS
    return bool(ok), (f"S-dream - random (judged correctness, k = 4) {d['mean']:.3f} [{lo:.3f}, {d['hi']:.3f}], "
                      f"N = {d['n']} items (need lower bound > 0)")


def agreement(votes, keys):
    """Pairwise share of answers on which two judges give the same (parsed) label."""
    out = {}
    js = [j for j in JUDGES if j in votes]
    for a in range(len(js)):
        for b in range(a + 1, len(js)):
            both = [k for k in keys if votes[js[a]].get(k) is not None and votes[js[b]].get(k) is not None]
            same = sum(votes[js[a]][k] == votes[js[b]][k] for k in both)
            out[f"{js[a]}-{js[b]}"] = {"agree": same / len(both) if both else float("nan"), "n": len(both)}
    return out


def has_answer(text, answer):
    """Judge-free check: the gold answer string occurs in the response (case-insensitive)."""
    return str(answer).strip().lower() in (text or "").lower()


# ----------------------------------------------------------------------------------------------- report


def fmt(s, nd=2):
    if not s or s.get("n", 0) == 0 or s.get("mean") is None or (isinstance(s["mean"], float) and math.isnan(s["mean"])):
        return "n/a"
    return f"{s['mean']:.{nd}f} [{s['lo']:.{nd}f}, {s['hi']:.{nd}f}]"


CONTRASTS = (("S-dream", "random"), ("S-dream", "S-attn"), ("S-attn", "random"), ("S-recon", "random"),
             ("S-loo", "random"), ("S-dream", "S-loo"), ("gold@k", "S-dream"))


def analyze_gen(model_dir, tf_dir=None, label=None):
    """report_gen.md + verdict_gen.json of one model dir (gen/, score/, panel/; tf/ for the teacher-forced verdicts)."""
    d = Path(model_dir)
    tf_dir = Path(tf_dir) if tf_dir else d / "tf"
    G, INFO = read_gen(d)
    assert G, f"no generated answers under {d / 'gen'}"
    gmeta = json.load(open(d / "gen" / "gen_meta.json")) if (d / "gen" / "gen_meta.json").exists() else {}
    label = label or gmeta.get("model_label") or d.name
    data = {x["question_id"]: x for x in json.load(open(DATA))["items"]}
    keys = [r["key"] for r in G]
    votes = load_votes(d)
    maj = majority_labels(keys, votes)
    for r in G:
        r["maj"] = maj[r["key"]]
        for j in JUDGES:
            r[j] = votes.get(j, {}).get(r["key"])
        r["refusal"] = bool(REFUSAL_RE.search(r["answer"] or ""))
        r["has_answer"] = has_answer(r["answer"], data[r["item"]]["answer"]) if data[r["item"]]["verbatim"] else None
    by = defaultdict(list)
    for r in G:
        by[r["cname"]].append(r)
    cnames = list(dict.fromkeys(r["cname"] for r in G))
    N = {c: {m: nested_of(by[c], fn) for m, fn in (
        ("maj", lambda r: r["maj"]), ("orig", lambda r: r["orig"]), ("gptoss", lambda r: r["gptoss"]),
        ("gemma4", lambda r: r["gemma4"]), ("refusal", lambda r: r["refusal"]), ("loop", lambda r: r["loop"]),
        ("has_answer", lambda r: r["has_answer"]), ("greedy_maj", lambda r: r["maj"] if r["j"] == 0 else None))}
         for c in cnames}
    budgets = [b for b in GEN_BUDGETS if f"S-dream@{b}" in N]
    mk = MAIN_K if MAIN_K in budgets else (budgets[0] if budgets else None)
    # --- the rule
    rule_d = paired_ci(N[f"S-dream@{mk}"]["maj"], N[f"random@{mk}"]["maj"]) if mk else {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    holds, why = gen_rule(rule_d)
    cover = {j: sum(votes.get(j, {}).get(k) is not None for k in keys) for j in JUDGES}
    n_missing_maj = sum(maj[k] is None for k in keys)
    gaps = {c: gap_ratio(N[c]["maj"], N["nomem"]["maj"], N["R-full"]["maj"]) for c in cnames}
    tfv = json.load(open(tf_dir / "verdict.json")) if (tf_dir / "verdict.json").exists() else None
    contrasts = {}
    for b in budgets:
        for a_, b_ in CONTRASTS:
            ka, kb = f"{a_}@{b}", f"{b_}@{b}"
            if ka in N and kb in N:
                contrasts[f"{a_} - {b_} @{b}"] = {m: paired_ci(N[ka][m], N[kb][m]) for m in ("maj", "refusal")}
    agree = agreement(votes, keys)
    verdict = {"model": label, "rule": {"name": "S-dream beats random in judged correctness", "holds": holds,
                                        "reason": why, "ci": rule_d},
               "judges_present": [j for j in JUDGES if j in votes], "coverage": cover, "n_answers": len(keys),
               "n_majority_missing": n_missing_maj, "n_items": len(INFO) or len({r['item'] for r in G}),
               "tf_rules": (tfv or {}).get("rules"), "gap_closed": gaps,
               "correct": {c: ci(N[c]["maj"]) for c in cnames}, "refusal": {c: ci(N[c]["refusal"]) for c in cnames},
               "contrasts": contrasts, "agreement": agree, "judge_prompt": LME_SOURCE}
    json.dump(verdict, open(d / "verdict_gen.json", "w"), indent=1, default=str)
    # --- report
    complete = all(j in votes for j in JUDGES) and all(cover[j] == len(keys) for j in JUDGES)
    g = lambda c: gaps.get(c, {}).get("mean")
    gs = lambda c: "n/a" if g(c) is None else f"{g(c):.2f}"
    cm = lambda c: ci(N[c]["maj"])["mean"] if c in N else float("nan")
    L = [f"# dream_select generation: judged correctness on {label} (LongMemEval single-session-user)", "",
         "## Summary", "",
         f"**What was run.** {len(INFO) or len({r['item'] for r in G})} items, {len(keys)} answers (1 greedy + 4 samples "
         "per condition, at most 64 new tokens). Each answer was judged with LongMemEval's own correctness prompt by "
         "three judges (the original Qwen3.5-9B, gpt-oss, Gemma-4). The label used is their majority; a tie goes to the "
         "original judge.", "",
         f"**Pre-registered rule (PREREG addendum 1).** \"S-dream beats random in judged correctness\" "
         f"**{'HOLDS' if holds else 'does not hold'}**: {why}.", ""]
    if not complete:
        L += [f"**Labels incomplete.** Judges present: {', '.join(j for j in JUDGES if j in votes) or 'none'}; labelled "
              f"answers per judge: {', '.join(f'{j} {cover[j]}/{len(keys)}' for j in JUDGES)}; answers with no majority "
              f"label: {n_missing_maj}. The rule above uses the votes that exist.", ""]
    if mk:
        L += [f"**Correct (k = {mk[1:]}).** nomem {cm('nomem'):.2f}, R-full {cm('R-full'):.2f}, ctx {cm('ctx'):.2f}; "
              f"S-dream {cm(f'S-dream@{mk}'):.2f}, S-attn {cm(f'S-attn@{mk}'):.2f}, random {cm(f'random@{mk}'):.2f}, "
              f"S-loo {cm(f'S-loo@{mk}'):.2f}, gold@k {cm(f'gold@k@{mk}'):.2f}. Gap closed (correctness): S-dream "
              f"{gs(f'S-dream@{mk}')}, S-loo {gs(f'S-loo@{mk}')}, gold@k {gs(f'gold@k@{mk}')}.", ""]
    if tfv:
        L += ["**Teacher-forced verdicts (this model's tf run).** " + "; ".join(
            f"{k}: {'HOLDS' if v['holds'] else 'does not hold'} ({v['reason']})" for k, v in tfv["rules"].items()), ""]
    L += ["## Judged correctness per condition", "",
          "Per item: the mean over its answers; CIs: item bootstrap (2000 replicates, seed 7). \"has answer\" = the gold "
          "string occurs in the reply (verbatim items; judge-free). Refusal = dream_bind addendum 7's regex.", "",
          "| condition | N | correct (majority) | greedy only | orig judge | gpt-oss | Gemma-4 | gap closed | refusal | loop | has answer |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cnames:
        n = N[c]
        gp = gaps[c]
        L.append(f"| {c} | {len(n['maj'])} | {fmt(ci(n['maj']))} | {fmt(ci(n['greedy_maj']))} | {fmt(ci(n['orig']))} | "
                 f"{fmt(ci(n['gptoss']))} | {fmt(ci(n['gemma4']))} | {fmt(gp) if gp.get('mean') is not None else 'n/a'} | "
                 f"{fmt(ci(n['refusal']))} | {fmt(ci(n['loop']))} | {fmt(ci(n['has_answer']))} |")
    L += ["", "## Paired contrasts (per item)", "", "| contrast | correct (majority) | refusal | N |", "|---|---|---|---|"]
    for k, v in contrasts.items():
        L.append(f"| {k} | {fmt(v['maj'])} | {fmt(v['refusal'])} | {v['maj']['n']} |")
    L += ["", "## Judges", "",
          f"- Labelled answers: {', '.join(f'{j} {cover[j]} of {len(keys)}' for j in JUDGES)}; no majority label: "
          f"{n_missing_maj}.",
          "- Pairwise agreement (answers both judges labelled): " + ", ".join(
              f"{k} {v['agree']:.3f} (N {v['n']})" for k, v in agree.items()) + ".",
          f"- Prompt: LongMemEval `{LME_SOURCE['file']}` `{LME_SOURCE['function']}` ({LME_SOURCE['task']}), commit "
          f"{LME_SOURCE['commit'][:7]}, verbatim; label = {LME_SOURCE['label']}.", ""]
    # --- examples: the greedy answers of the first 4 items at the main budget
    L += ["## Examples (greedy answers; the first 4 items)", ""]
    first = list(dict.fromkeys(r["item"] for r in G))[:4]
    show = list(GEN_ANCHORS) + ["gold"] + ([f"{c}@{mk}" for c in GEN_SELECTORS] if mk else [])
    for i in first:
        L += [f"### {i}: Q \"{data[i]['question']}\" A \"{data[i]['answer']}\"", "", "| condition | majority | answer |",
              "|---|---|---|"]
        for c in show:
            r = next((x for x in by.get(c, []) if x["item"] == i and x["j"] == 0), None)
            if r:
                txt = (r["answer"] or "").replace("\n", " ").replace("|", "\\|")
                L.append(f"| {c} | {r['maj']} | {txt} |")
        L.append("")
    ids = [v.get("identity_max_abs_diff") for v in INFO.values() if v.get("identity_max_abs_diff") is not None]
    L += ["## Checks", "",
          f"- R-full == T-full at the last prompt position: max |Δlogit| {max(ids):.2e} over {len(ids)} items "
          "(tolerance 0.01; asserted)." if ids else "- R-full == T-full: no info rows.",
          f"- Peak GPU (gen): {gmeta.get('peak_gpu_gb_reserved')} GB reserved; {gmeta.get('minutes', float('nan')):.1f} "
          f"min; commit {gmeta.get('env', {}).get('commit')}; selections from `{gmeta.get('sel_from')}`.", ""]
    (d / "report_gen.md").write_text("\n".join(L) + "\n")
    print(f"wrote {d / 'report_gen.md'}: {'HOLDS' if holds else 'does not hold'} ({why})")
    return verdict


def replication_verdict(held, n_done, n_models, need=2):
    """'replicates' if the rule held on >= need models; 'does not replicate' if it cannot reach need even if every
    missing model held; else 'incomplete'."""
    if held >= need:
        return "replicates"
    if held + (n_models - n_done) < need:
        return "does not replicate"
    return "incomplete"


def summary(root, labels, models=GEN_MODELS):
    """RUN/summary.md: each model's teacher-forced and generation verdicts, and the replication verdict ("replicates"
    if the generation rule holds on at least 2 of the 3 models)."""
    root = Path(root)
    L = ["# dream_select on DelftBlue: replication + generation (PREREG addendum 1)", "",
         "| model | tf: dreamed questions select well | tf: any attention selector works | gen: S-dream beats random "
         "(judged) | S-dream - random, k = 4 | correct: nomem / R-full / S-dream / random / S-loo / gold@k (k = 4) |",
         "|---|---|---|---|---|---|"]
    held, n_done = 0, 0
    rows = {}
    for m in models:
        d = root / m
        vg = json.load(open(d / "verdict_gen.json")) if (d / "verdict_gen.json").exists() else None
        tfp = d / "tf" / "verdict.json"
        vt = json.load(open(tfp)) if tfp.exists() else None
        rows[m] = {"gen": vg, "tf": vt}
        tfr = (vt or {}).get("rules", {})
        t1 = tfr.get("Dreamed questions select well", {}).get("holds")
        t2 = tfr.get("Any attention selector works", {}).get("holds")
        yn = lambda x: "n/a" if x is None else ("HOLDS" if x else "does not hold")
        if vg:
            n_done += 1
            held += bool(vg["rule"]["holds"])
            c = vg["correct"]
            cell = " / ".join(f"{c[k]['mean']:.2f}" if k in c else "n/a" for k in
                              ("nomem", "R-full", "S-dream@k4", "random@k4", "S-loo@k4", "gold@k@k4"))
            L.append(f"| {labels.get(m, m)} | {yn(t1)} | {yn(t2)} | {yn(vg['rule']['holds'])} | {fmt(vg['rule']['ci'], 3)} "
                     f"| {cell} |")
        else:
            L.append(f"| {labels.get(m, m)} | {yn(t1)} | {yn(t2)} | missing | | |")
    verdict = replication_verdict(held, n_done, len(models))
    L +=["", f"**Replication across models (pre-registered):** the generation rule holds on {held} of {n_done} models "
          f"with results ({len(models)} planned): **{verdict}** (needs at least 2 of 3).", "",
          "Teacher-forced verdicts for Qwen2.5-1.5B are the local run's (`results/dream_select_20261104_0032`, copied to "
          "`qwen25_15b/tf`). Per-model detail: `<model>/report_gen.md` and `<model>/tf/report.md`.", ""]
    for m in models:
        vg = rows[m]["gen"]
        if vg:
            L.append(f"- {labels.get(m, m)}: labels per judge {vg['coverage']} of {vg['n_answers']} answers; no majority "
                     f"label {vg['n_majority_missing']}.")
    (root / "summary.md").write_text("\n".join(L) + "\n")
    json.dump({"verdict": verdict, "held": held, "n_done": n_done, "models": {m: (rows[m]["gen"] or {}).get("rule")
                                                                              for m in models}},
              open(root / "summary.json", "w"), indent=1, default=str)
    print(f"wrote {root / 'summary.md'}: {verdict} ({held} of {n_done})")
    return verdict


if __name__ == "__main__":
    analyze_gen(Path(sys.argv[1]))
