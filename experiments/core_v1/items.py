"""core_v1 items: load experiments/core_v1/items.yaml into bench_v1-style LOADED items, and validate them.

Pure Python (yaml + seahorse.bench); no torch. Unit-tested in tests/test_core_v1.py.

Every loaded item is a bench_v1 loaded item (seahorse.bench.data.expand_item) plus:
  group     leanings | dislikes | one_of_many | facts
  swap      the partner whose stored shift the `swap` control injects
  centroid  8 same-frame alternatives (preference items)
  mech      True for the items that get the mechanism references (dislikes, one_of_many, 4 leanings)
  prompts   [(kind, text)]: preferences related + 5 ambiguous; facts related + 2 use
  related, paraphrase, use (facts)
"""

import copy
from pathlib import Path

import yaml

from seahorse.bench import load_bench, validate_bench
from seahorse.bench.data import _rp, expand_item
from seahorse.bench.validate import _duplicates

GROUPS = ("leanings", "dislikes", "one_of_many", "facts")
PREF_GROUPS = ("leanings", "dislikes", "one_of_many")
N_AMBIGUOUS = 5
N_CENTROID = 8
MIN_LEXICON = 10


def probe_text(it, distance):
    return next(p["text"] for p in it["probes"] if p["distance"] == distance)


def _relation(rel):
    rel = {({True: "yes", False: "no"}[k] if isinstance(k, bool) else str(k).lower()): v for k, v in rel.items()}
    return [_rp(q, "Yes", "inference") for q in rel["yes"]] + [_rp(q, "No", "inference") for q in rel["no"]]


def negate_item(base, iid, neg):
    """An aversion derived from a liking: base's follow-ups and prompts, contrasts a/b swapped, lexicon sides
    swapped (consistent = away from the disliked thing), its own experience / counter / relation probes."""
    it = copy.deepcopy(base)
    it.update(id=iid, experience=neg["experience"], counter=neg["counter"], source=f"core_v1 negation of {base['id']}",
              negated_from=base["id"])
    it["lexicon"] = {"consistent": list(base["lexicon"]["inconsistent"]),
                     "inconsistent": list(base["lexicon"]["consistent"])}
    it["contrasts"] = [{"prefix": c["prefix"], "a": c["b"], "b": c["a"]} for c in base["contrasts"]]
    it["measure"] = dict(it["contrasts"][0])
    it["relation_probes"] = _relation(neg["relation"])
    return it


def load_items(path, bench_dir):
    """items.yaml -> {"items": [loaded items, group order], "by_id", "groups": {group: [ids]}, "placebo",
    "mechanism": [ids], "unrelated": [50 prompts]}."""
    cfg = yaml.safe_load(open(path))
    bench = load_bench(bench_dir)
    bench_by = {it["id"]: it for it in bench["scenarios"]}
    entries = [(g, e) for g in GROUPS for e in cfg[g]]
    base = {}  # every non-negated item (bench or new), for negations to refer to
    for g, e in entries:
        if "negate" in e:
            continue
        if "new" in e:
            base[e["id"]] = expand_item({**e["new"], "id": e["id"]}, "disposition", source="core_v1")
        else:
            assert e["id"] in bench_by, f"{e['id']}: not a bench_v1 item"
            base[e["id"]] = copy.deepcopy(bench_by[e["id"]])
    for b in bench["scenarios"]:  # bench items a negation may refer to without using them itself
        base.setdefault(b["id"], copy.deepcopy(b))
    mech = set(cfg.get("mechanism_leanings", []))
    items = []
    for g, e in entries:
        it = negate_item(base[e["negate"]["of"]], e["id"], e["negate"]) if "negate" in e else copy.deepcopy(base[e["id"]])
        it.update(group=g, swap=e["swap"])
        it["related"] = probe_text(it, "related")
        it["paraphrase"] = probe_text(it, "paraphrase")
        if g == "facts":
            assert it["type"] == "fact", it["id"]
            it["use"] = list(e["use"])
            it["prompts"] = [("related", it["related"])] + [("use", u) for u in it["use"]]
            it["mech"] = False
        else:
            assert it["type"] == "disposition", it["id"]
            it["centroid"] = list(e["centroid"])
            it["prompts"] = [("related", it["related"])] + [("ambiguous", t) for t in it["ambiguous"][:N_AMBIGUOUS]]
            it["mech"] = g in ("dislikes", "one_of_many") or it["id"] in mech
        items.append(it)
    by = {it["id"]: it for it in items}
    unrelated = list(bench["unrelated_probes"]) + list(cfg.get("unrelated_extra", []))
    return {"items": items, "by_id": by, "groups": {g: [it["id"] for it in items if it["group"] == g] for g in GROUPS},
            "placebo": cfg["placebo"], "mechanism": sorted(mech), "unrelated": unrelated}


def _has_word(text, w):
    import re
    return bool(re.search(rf"(?<![\w-]){re.escape(w.strip())}(?![\w-])", text, re.I))


def validate_items(L, tok=None):
    """bench_v1's validator on every item (schema, Yes/No balance, duplicates against the unrelated set,
    tokenisation when `tok` is given), cross-item duplicates among the non-derived items, and core_v1's
    own rules (centroids, lexicon size, swap permutations, use prompts, the unrelated set)."""
    err, warn = [], []
    shared = {"unrelated_probes": L["unrelated"], "placebos": [L["placebo"]]}
    for it in L["items"]:
        r = validate_bench({"scenarios": [it], **shared}, tok)
        err += [f"[{it['id']}] {e}" for e in r["errors"]]
        warn += [f"[{it['id']}] {w}" for w in r["warnings"]]
    _duplicates([it for it in L["items"] if "negated_from" not in it], [], err)
    for it in L["items"]:
        i = it["id"]
        if len(it["followups"]) != 3:
            err.append(f"{i}: {len(it['followups'])} follow-ups (want 3)")
        if it["group"] == "facts":
            if len(it["use"]) != 2 or any(_has_word(u, it["measure"]["target"]) for u in it["use"]):
                err.append(f"{i}: need 2 use prompts without the target")
            ans = sorted(r["a"] for r in it["relation_probes"])
            if ans.count("Yes") != ans.count("No"):
                err.append(f"{i}: unbalanced verification probes")
        else:
            c = it["centroid"]
            if len(c) != N_CENTROID or len(set(c)) != N_CENTROID or it["experience"] in c:
                err.append(f"{i}: centroid needs {N_CENTROID} distinct alternatives other than the experience")
            ans = [r["a"] for r in it["relation_probes"]]
            if ans.count("Yes") != 3 or ans.count("No") != 3:
                err.append(f"{i}: relation probes {ans.count('Yes')} Yes / {ans.count('No')} No (want 3 / 3)")
            if len(it["ambiguous"]) < N_AMBIGUOUS:
                err.append(f"{i}: < {N_AMBIGUOUS} ambiguous prompts")
            for side in ("consistent", "inconsistent"):
                n = len(it["lexicon"][side])
                if n < MIN_LEXICON:
                    (err if it.get("source") == "core_v1" else warn).append(f"{i}: lexicon.{side} has {n} < {MIN_LEXICON}")
        if it["experience"] == L["placebo"]:
            err.append(f"{i}: experience equals the placebo")
    for g, ids in L["groups"].items():
        partners = [L["by_id"][i]["swap"] for i in ids]
        for i, p in zip(ids, partners):
            if p not in ids or p == i:
                err.append(f"{i}: swap partner {p!r} must be another item of group {g}")
        if sorted(partners) != sorted(ids):
            err.append(f"group {g}: swap partners are not a permutation of the group")
    for m in L["mechanism"]:
        if m not in L["groups"]["leanings"]:
            err.append(f"mechanism_leanings: {m} is not a leaning")
    if len(L["unrelated"]) != 50 or len(set(u.lower() for u in L["unrelated"])) != 50:
        err.append(f"unrelated: {len(L['unrelated'])} prompts / {len(set(L['unrelated']))} distinct (want 50)")
    counts = {g: len(ids) for g, ids in L["groups"].items()}
    counts["mechanism"] = sum(it["mech"] for it in L["items"])
    return {"errors": err, "warnings": warn, "counts": counts}


if __name__ == "__main__":
    import argparse
    import sys
    HERE = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="validate experiments/core_v1/items.yaml")
    p.add_argument("--items", default=str(HERE / "items.yaml"))
    p.add_argument("--bench", default=str(HERE.parent.parent / "data" / "bench_v1"))
    p.add_argument("--model", default=None, help="tokenizer for the tokenisation checks (default: none)")
    a = p.parse_args()
    tok = None
    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
    L = load_items(a.items, a.bench)
    res = validate_items(L, tok)
    print(f"counts: {res['counts']}; errors {len(res['errors'])}; warnings {len(res['warnings'])}")
    for e in res["errors"]:
        print("  ERROR", e)
    for w in res["warnings"]:
        print("  warn ", w)
    sys.exit(1 if res["errors"] else 0)
