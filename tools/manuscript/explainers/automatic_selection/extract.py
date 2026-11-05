#!/usr/bin/env python3
"""Data for the Automatic Slice Selection page, rebuilt from the dream_select runs.

    python3 -I extract.py <results dir> <out.json>

Runs (under the results dir):
  dream_select_20261104_0032            teacher-forced, Qwen2.5-1.5B (local); also the four kept-token examples
  dream_select_cluster_20261104_0129    <model>/tf (teacher-forced, Qwen2.5-7B and Llama-3.2-3B) and
                                        <model>/verdict_gen.json (generation judged by the three-judge panel)

Output schema: {"models": {mk: {...}}, "examples": {item: {...}}}
  gap        "sel|k": [mean, lo, hi, N] gap closed, (cond - nomem) / (R-full - nomem), items with
             R-full - nomem < 0.5 nats dropped (as selectors_ds.gap_closed); 95% percentile bootstrap over items
             (random.Random(7), 2000 replicates, items in sorted order)
  tf_refusal "sel|k": mean first-token refusal mass P(I + Sorry + As);  tf_refusal_anchor: the same for anchors
  punct      "sel|k": share of picked positions whose token is punctuation only (PUNCT below; all three random
             seeds);  punct_base: the punctuation share of a turn's content tokens, mean over items
  hit        "sel|k": mean hit rate over the verbatim items (picks include the answer's position)
  gen_correct, gen_refusal  cond: [mean, lo, hi, N] from verdict_gen.json (judged correctness, refusal)
  rule       the pre-registered generation rule (verdict_gen.json);  tf_rules: the teacher-forced rules
Standard library only.
"""

import gzip
import json
import random
import re
import sys
from pathlib import Path

LOCAL = "dream_select_20261104_0032"
CLUSTER = "dream_select_cluster_20261104_0129"
MODELS = [("qwen25_15b", "Qwen2.5-1.5B", LOCAL), ("qwen25_7b", "Qwen2.5-7B", CLUSTER + "/qwen25_7b/tf"),
          ("llama32_3b", "Llama-3.2-3B", CLUSTER + "/llama32_3b/tf")]
SELS = ["S-dream", "S-attn", "S-recon", "S-loo", "random", "gold@k"]
BUDGETS = ["k2", "k4", "k8"]
ANCHORS = ["nomem", "R-full", "ctx"]
GEN_CONDS = ["nomem", "ctx", "R-full", "R-content", "gold"] + [f"{s}@{k}" for k in ("k4", "k8") for s in SELS]
EXAMPLES = ["c5e8278d", "6ade9755", "e47becba", "1e043500"]
EX_SELS = ["S-dream", "S-attn", "S-loo", "random"]
# Sentence punctuation, quotes and dashes only; symbols that sit inside numbers or words ($, %, /) are not counted.
# (analyze.py's post-hoc table counts every non-word token, so its shares are slightly higher.)
PUNCT = re.compile(r"^\s*[.,!?;:'\"()\[\]\-–—…]+\s*$")
MIN_DEN = 0.5                                   # as selectors_ds.gap_closed
N_BOOT, SEED = 2000, 7


def jload(p):
    return json.loads(Path(p).read_text())


def gz_rows(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def boot(vals):
    """[mean, lo, hi, n]: percentile bootstrap of the mean."""
    n = len(vals)
    rng = random.Random(SEED)
    st = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return [sum(vals) / n, st[int(0.025 * N_BOOT)], st[int(0.975 * N_BOOT) - 1], n]


def ci4(d):
    return [d["mean"], d["lo"], d["hi"], d["n"]]


def tf_model(run):
    rows = gz_rows(run / "rows.jsonl.gz")
    meta = jload(run / "meta.json")
    verdict = jload(run / "verdict.json")
    infos = meta["items"]
    by = {}
    for r in rows:
        by[(r["cond"], r["budget"], r["item"])] = r
    nm = {i: by[("nomem", None, i)]["answer_score"] for i in infos}
    full = {i: by[("R-full", None, i)]["answer_score"] for i in infos}

    def gap(cond, budget):
        vals = []
        for i in sorted(infos):
            r = by.get((cond, budget, i))
            den = full[i] - nm[i]
            if r is None or den < MIN_DEN:
                continue
            vals.append((r["answer_score"] - nm[i]) / den)
        return boot(vals)

    def mean_of(cond, budget, field, items):
        v = [by[(cond, budget, i)][field] for i in items if (cond, budget, i) in by]
        return round(sum(v) / len(v), 3)

    def punct(cond, budget):
        n = p = 0
        for r in rows:
            if r["cond"] == cond and r["budget"] == budget:
                tm = dict(zip(infos[r["item"]]["content"], infos[r["item"]]["content_tokens"]))
                for s in r["sel"]:
                    n += len(s)
                    p += sum(bool(PUNCT.match(tm[x])) for x in s)
        return round(p / n, 3)

    items = list(infos)
    verb = [i for i in items if infos[i]["verbatim"]]
    keys = [(s, b) for s in SELS for b in BUDGETS]
    out = {"gap": {f"{s}|{b}": gap(s, b) for s, b in keys}}
    out["gap"]["gold|own"] = gap("gold", "own")
    out["tf_refusal"] = {f"{s}|{b}": mean_of(s, b, "refusal", items) for s, b in keys}
    out["tf_refusal_anchor"] = {a: mean_of(a, None, "refusal", items) for a in ANCHORS}
    out["punct"] = {f"{s}|{b}": punct(s, b) for s, b in keys}
    per_item = [sum(bool(PUNCT.match(t)) for t in infos[i]["content_tokens"]) / len(infos[i]["content_tokens"])
                for i in items]
    out["punct_base"] = round(sum(per_item) / len(per_item), 3)
    out["hit"] = {f"{s}|{b}": mean_of(s, b, "hit", verb) for s, b in keys}
    out["tf_rules"] = verdict["rules"]
    out["n_items"] = verdict["n_items"]
    return out, rows, infos


def examples(rows, infos):
    by = {(r["cond"], r["budget"], r["item"]): r for r in rows}
    ex = {}
    for i in EXAMPLES:
        it = infos[i]
        picks = {s: by[(s, "k4", i)]["sel"][0] for s in EX_SELS}
        scores = {"nomem": by[("nomem", None, i)], "R-full": by[("R-full", None, i)]}
        scores.update({s: by[(s, "k4", i)] for s in EX_SELS})
        ex[i] = {"q": it["question"], "a": it["answer"], "content": it["content"], "tokens": it["content_tokens"],
                 "gold": it["gold_span"], "dreams": it["dreams"]["questions"], "picks": picks,
                 "scores": {k: round(r["answer_score"], 2) for k, r in scores.items()}}
    return ex


def extract(results):
    results = Path(results)
    models, ex = {}, None
    for mk, name, run in MODELS:
        tf, rows, infos = tf_model(results / run)
        vg = jload(results / CLUSTER / mk / "verdict_gen.json")
        m = {"name": name, "gap": tf["gap"], "tf_refusal": tf["tf_refusal"],
             "tf_refusal_anchor": tf["tf_refusal_anchor"], "punct": tf["punct"], "punct_base": tf["punct_base"],
             "hit": tf["hit"],
             "gen_correct": {c: ci4(vg["correct"][c]) for c in GEN_CONDS},
             "gen_refusal": {c: ci4(vg["refusal"][c]) for c in GEN_CONDS},
             "rule": vg["rule"], "tf_rules": tf["tf_rules"], "n_items": tf["n_items"]}
        models[mk] = m
        if run == LOCAL:
            ex = examples(rows, infos)
    return {"models": models, "examples": ex}


def placeholders(results):
    """Template placeholder -> JSON object, for build_explainers.py."""
    return {"__DATA__": extract(results)}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    Path(sys.argv[2]).write_text(json.dumps(extract(sys.argv[1]), ensure_ascii=False, separators=(",", ":")))
