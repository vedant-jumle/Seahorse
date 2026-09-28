"""Load bench_v1 (data/bench_v1/*.yaml) into v0_1-compatible scenario dicts.

The YAML is written in a compact authoring format; `load_bench` expands it into the loaded
format documented at the top of data/bench_v1/shared.yaml (a superset of the v0_1 schema).
"""

import random
import re
from pathlib import Path

import yaml

ANSWER_SUFFIX = "Answer yes or no."
FILES = ("shared.yaml", "dispositions.yaml", "facts.yaml", "emotional.yaml", "fact_pool.yaml")
_AUTHOR_ONLY = ("relation", "verify", "neutral", "emotional")


def _rp(question, consistent, kind):
    return {"text": f"{question} {ANSWER_SUFFIX}", "a": consistent,
            "b": "No" if consistent == "Yes" else "Yes",
            "consistent": consistent, "kind": kind, "question": question}


def expand_item(raw, typ, source="hand"):
    """Authoring-format item -> loaded item (v0_1 keys plus the bench_v1 extras)."""
    it = {k: v for k, v in raw.items() if k not in _AUTHOR_ONLY}
    it.update(type=typ, source=source)
    fu, pr = list(raw["followups"]), raw["probes"]
    it["followups"] = fu
    it["probes"] = [{"distance": "exact", "text": pr.get("exact", fu[0])},
                    {"distance": "paraphrase", "text": pr["paraphrase"]},
                    {"distance": "related", "text": pr["related"]}]
    if typ == "disposition":
        it["contrasts"] = [{"prefix": p, "a": a, "b": b} for p, a, b in raw["contrasts"]]
        it["measure"] = dict(it["contrasts"][0])
        # YAML 1.1 reads bare yes:/no: keys as booleans
        rel = {({True: "yes", False: "no"}[k] if isinstance(k, bool) else str(k).lower()): v
               for k, v in raw["relation"].items()}
        it["relation_probes"] = ([_rp(q, "Yes", "inference") for q in rel["yes"]]
                                 + [_rp(q, "No", "inference") for q in rel["no"]])
    elif typ == "fact":
        m = raw["measure"]
        it["measure"] = {"prefix": m["prefix"], "target": m["target"], "foils": list(m["foils"])}
        rps = []
        for i, tpl in enumerate(raw["verify"]):
            rps.append(_rp(tpl.replace("{v}", m["target"].strip()), "Yes", "verification"))
            rps.append(_rp(tpl.replace("{v}", m["foils"][i % len(m["foils"])].strip()), "No",
                           "verification"))
        it["relation_probes"] = rps
    else:
        raise ValueError(f"{raw.get('id')}: unknown type {typ!r}")
    return it


def expand_pair(raw):
    """Emotional pair -> [<id>_neu, <id>_emo], identical except experience and counter."""
    out = []
    for phrasing in ("neutral", "emotional"):
        exp, counter = raw[phrasing]
        base = {k: v for k, v in raw.items() if k != "type"}
        base.update(id=f"{raw['id']}_{phrasing[:3]}", experience=exp, counter=counter,
                    category="emotional", pair=raw["id"], phrasing=phrasing, neuromod=True)
        out.append(expand_item(base, raw["type"]))
    return out


def load_bench(path):
    """Load a bench directory. Returns a dict with v0_1's top-level keys plus extras:
    {"scenarios": [...], "unrelated_probes": [...], "placebos": [...], "pool": raw pool}.
    `scenarios` holds the dispositions, then the facts, then the emotional pairs."""
    path = Path(path)
    raw = {}
    for name in FILES:
        f = path / name
        if f.exists():
            raw.update(yaml.safe_load(f.read_text()) or {})
    items = [expand_item(r, "disposition") for r in raw.get("dispositions", [])]
    items += [expand_item(r, "fact") for r in raw.get("facts", [])]
    for r in raw.get("emotional_pairs", []):
        items += expand_pair(r)
    return {"scenarios": items, "unrelated_probes": raw.get("unrelated_probes", []),
            "placebos": raw.get("placebos", []),
            "pool": {k: raw[k] for k in ("names", "kinds", "slots", "generic_followups") if k in raw}}


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def pool_items(pool, seed=0):
    """One templated fact per pool slot (value, counter and 3 foils drawn with `seed`)."""
    rng = random.Random(seed)
    items = []
    for i, sl in enumerate(pool["slots"]):
        tpl = {**pool["kinds"].get(sl.get("kind"), {}), **sl}
        values = sl.get("values") or pool["names"][sl["names"]]
        v, counter, *foils = rng.sample(values, 5)
        fill = lambda s, x=v: s.replace("{slot}", sl["slot"]).replace("{v}", x)
        raw = {
            "id": f"pool_{_slug(sl['slot'])}", "category": "pool", "slot": sl["slot"],
            "group": sl.get("group", sl["slot"]),
            "experience": fill(tpl["experience"]), "counter": fill(tpl["experience"], counter),
            "followups": [fill(f) for f in tpl["followups"]] + [pool["generic_followups"][i]],
            "probes": {"paraphrase": fill(tpl["paraphrase"]), "related": fill(tpl["related"])},
            "measure": {"prefix": fill(tpl["prefix"]), "target": f" {v}",
                        "foils": [f" {x}" for x in foils]},
            "verify": [t.replace("{slot}", sl["slot"]) for t in tpl["verify"]],
        }
        items.append(expand_item(raw, "fact", source="pool"))
    return items


def sample_pool(pool, n=None, seed=0, exclude_slots=()):
    """n templated facts with distinct slots and at most one per group (e.g. one partner),
    skipping `exclude_slots` (e.g. the slots of the hand-written facts in the same memory).
    Values are fixed by `seed`; the choice and order of slots is shuffled with it too."""
    items = [it for it in pool_items(pool, seed) if it["slot"] not in set(exclude_slots)]
    random.Random(seed + 1).shuffle(items)
    out, groups = [], set()
    for it in items:
        if it["group"] in groups:
            continue
        groups.add(it["group"])
        out.append(it)
    if n is not None and n > len(out):
        raise ValueError(f"asked for {n} pool facts, only {len(out)} distinct slots available")
    return out if n is None else out[:n]


def pool_size(pool):
    """(number of slots, number of distinct (slot, value) facts)."""
    n = sum(len(sl.get("values") or pool["names"][sl["names"]]) for sl in pool["slots"])
    return len(pool["slots"]), n
