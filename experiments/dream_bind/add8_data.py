#!/usr/bin/env python
"""dream_bind PREREG addendum 8: build the new data files deterministically (run once, locally, before any job; the
files are the pre-registered record and are checked by tests/test_dream_bind.py and at run time).

  conda run -n torch env PYTHONPATH=src python experiments/dream_bind/add8_data.py [--check]

Writes into data/bench_v1/:
  dummy_pairs.yaml   the dummy "pair 1" sentences of part 1 (slots and names disjoint from every test item)
  pool_lures.yaml    part 2a: per non-primary fact, its counter value and the drawn form / binding lures (bind.FORM_RULE,
                     bind.BINDING_RULE)
  pool_split.yaml    part 2b: the two fixed halves of the 52 non-primary facts (bind.pool_split, seed 8, by kind)
  pool_pairs.yaml    part 2c: 34 two-fact pairs from the pool slots (17 confusable, 17 different-subject; seed 13)
comply_pairs.yaml (part 3) is hand-written; this script only validates it.
--check: rebuild in memory and fail if a written file differs (used by the tests when the tokenizers are cached).
"""

import argparse
import importlib.util
import random
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO / "data" / "bench_v1"
TOKENIZERS = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3.5-2B", "meta-llama/Llama-3.2-3B-Instruct")
SPLIT_SEED, PAIR_SEED = 8, 13

# part 1: dummy "pair 1" sentences "My {slot}'s name is {name}." Slots outside fact_pool.yaml's slots and every collide
# member's subject; names outside every name / value list, foil, counter, lure and collide value (checked below).
DUMMIES = [{"slot": "gardener", "name": "Brian"}, {"slot": "mechanic", "name": "Keith"},
           {"slot": "postman", "name": "Derek"}, {"slot": "locksmith", "name": "Trevor"},
           {"slot": "tailor", "name": "Colin"}, {"slot": "barber", "name": "Graham"},
           {"slot": "florist", "name": "Janet"}, {"slot": "butcher", "name": "Gary"}]

# part 2c: confusable pool pairs
CONF_PERSON = ["sister", "brother", "cousin", "niece", "nephew", "son", "daughter"]          # "older / younger {slot}"
CONF_PET = ["dog", "cat", "rabbit", "parrot", "horse", "hamster", "goldfish", "tortoise", "guinea pig", "ferret"]  # first / second
N_DIFF_PP, N_DIFF_PET = 12, 5          # different-subject pairs: person + person, person + pet


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load("dream_bind_run_add8data", HERE / "run.py")
bind = run.bind


def tokenizers():
    from transformers import AutoTokenizer
    return [AutoTokenizer.from_pretrained(n, local_files_only=True) for n in TOKENIZERS]


def one_token_fn(toks):
    return lambda v: all(len(t(" " + v.strip(), add_special_tokens=False).input_ids) == 1 for t in toks)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def test_names(cfg):
    """Every name / value used by a test item anywhere in dream_bind (lower case): the pool lists, the hand-written facts'
    targets, foils and counters, the lures, the collide values."""
    pool = yaml.safe_load(open(DATA / "fact_pool.yaml"))
    names = {w for l in pool["names"].values() for w in l} | {v for s in pool["slots"] for v in s.get("values", [])}
    for f in yaml.safe_load(open(DATA / "facts.yaml"))["facts"]:
        names |= {f["measure"]["target"].strip(), *[x.strip() for x in f["measure"]["foils"]],
                  bind.counter_value(f["experience"], f["counter"])}
    names |= {v.strip() for L in cfg["lures"].values() for k in L.values() for v in k}
    for p in cfg["collide"]["pairs"]:
        for k in ("a", "b"):
            if isinstance(p[k], dict):
                names.add(p[k]["value"].strip())
    names |= {"plumber"}                       # the job counter override
    return {n.lower() for n in names}


def pool_facts(cfg):
    """The 52 non-primary facts (run.pool_gen_items order) with slot, kind, target, counter value, foils and the slot's
    value list."""
    from seahorse.bench import load_bench
    from seahorse.bench.data import sample_pool
    pool = load_bench(REPO / cfg["bench"])["pool"]
    slots = {s["slot"]: s for s in pool["slots"]}
    raw = run._facts_raw(cfg)
    hand = {it["id"]: it for it in run.load_items(cfg)}
    sp = {p["id"]: p for p in sample_pool(pool, None, seed=cfg["gkey"]["pool_seed"],
                                          exclude_slots={raw[i]["slot"] for i in cfg["primary"] + cfg["secondary"]})}
    out = []
    for f in run.pool_gen_items(cfg):
        if f["id"] in hand:
            slot, exp, counter = raw[f["id"]]["slot"], hand[f["id"]]["experience"], hand[f["id"]]["counter"]
        else:
            p = sp[f["id"]]
            slot, exp, counter = p["slot"], p["experience"], p["counter"]
        sl = slots[slot]
        out.append({"id": f["id"], "slot": slot, "kind": f["kind"], "target": f["target"].strip(),
                    "counter": bind.counter_value(exp, counter), "foils": [x.strip() for x in f["foils"]],
                    "values": list(sl.get("values") or pool["names"][sl["names"]]),
                    "list": sl["names"] if f["kind"] in ("person", "pet") else "attribute"})
    return out


def build_lures(cfg, one):
    F = pool_facts(cfg)
    vals = [(f["list"], f["target"]) for f in F]          # the binding draw runs within a name list (female / male / pet)
    out = []
    for i, f in enumerate(F):
        ex = [f["counter"], *f["foils"]]
        form = bind.form_lures(f["target"], f["values"], ex, one)
        bnd = bind.binding_lures(i, vals, ex + form, one)
        # attribute slots whose list has only the 5 drawn values (target, counter, 3 foils) get fewer form lures (recorded)
        assert f["kind"] == "attribute" or (len(form) == 2 and len(bnd) == 2), (f["id"], form, bnd)
        out.append({"id": f["id"], "kind": f["kind"], "slot": f["slot"], "target": f["target"], "counter": f["counter"],
                    "foils": f["foils"], "form": form, "binding": bnd})
    return out


def build_split(cfg):
    F = run.pool_gen_items(cfg)
    A, B = bind.pool_split([(f["id"], f["kind"]) for f in F], SPLIT_SEED)
    kind = {f["id"]: f["kind"] for f in F}
    return {"seed": SPLIT_SEED, "A": A, "B": B,
            "counts": {h: {k: sum(kind[i] == k for i in ids) for k in ("person", "pet", "attribute")}
                       for h, ids in (("A", A), ("B", B))}}


def _member(mid, slot, kind, v):
    use = ("Write a short birthday message to my {slot}." if kind == "person" else
           "Write a short, funny poem about my {slot}.").replace("{slot}", slot)
    return {"id": mid, "experience": f"My {slot}'s name is {v}.", "value": f" {v}", "question": f"What's my {slot}'s name?",
            "prefix": f"Your {slot}'s name is", "use": [use],
            "verify": [f"Is my {slot}'s name {{v}}?", f"Is {{v}} the name of my {slot}?"]}


def build_pairs(cfg, one, banned):
    pool = yaml.safe_load(open(DATA / "fact_pool.yaml"))
    slots = {s["slot"]: s for s in pool["slots"]}
    rng = random.Random(PAIR_SEED)
    used = set(banned)

    def draw(slot, n=2):
        vals = [v for v in pool["names"][slots[slot]["names"]] if v.lower() not in used and one(v)]
        got = rng.sample(vals, n)
        used.update(v.lower() for v in got)
        return got
    pairs = []
    for s in CONF_PERSON:
        a, b = draw(s)
        pairs.append({"id": f"ppc_{slug(s)}", "kind": "confusable",
                      "a": _member(f"ppc_older_{slug(s)}", f"older {s}", "person", a),
                      "b": _member(f"ppc_younger_{slug(s)}", f"younger {s}", "person", b)})
    for s in CONF_PET:
        a, b = draw(s)
        pairs.append({"id": f"ppc_{slug(s)}", "kind": "confusable",
                      "a": _member(f"ppc_first_{slug(s)}", f"first {s}", "pet", a),
                      "b": _member(f"ppc_second_{slug(s)}", f"second {s}", "pet", b)})
    persons = [s["slot"] for s in pool["slots"] if s.get("kind") == "person" and s["slot"] != "husband"]   # one partner
    pets = [s["slot"] for s in pool["slots"] if s.get("kind") == "pet"]
    random.Random(PAIR_SEED + 4).shuffle(persons)
    random.Random(PAIR_SEED + 5).shuffle(pets)
    duo = [(persons[2 * i], persons[2 * i + 1]) for i in range(N_DIFF_PP)]
    duo += [(persons[2 * N_DIFF_PP + i], pets[i]) for i in range(N_DIFF_PET)]
    for sa, sb in duo:
        a, = draw(sa, 1)
        b, = draw(sb, 1)
        pairs.append({"id": f"ppd_{slug(sa)}_{slug(sb)}", "kind": "different",
                      "a": _member(f"ppd_{slug(sa)}", sa, slots[sa]["kind"], a),
                      "b": _member(f"ppd_{slug(sb)}", sb, slots[sb]["kind"], b)})
    return pairs


def check_dummies(cfg, one):
    pool = yaml.safe_load(open(DATA / "fact_pool.yaml"))
    pslots = {s["slot"] for s in pool["slots"]}
    names = test_names(cfg)
    for d in DUMMIES:
        assert d["slot"] not in pslots and d["name"].lower() not in names, d
        assert one(d["name"]), d
    assert len({d["name"] for d in DUMMIES}) == len({d["slot"] for d in DUMMIES}) == len(DUMMIES)


HEADERS = {
    "dummy_pairs.yaml": "# dream_bind PREREG addendum 8, part 1: the dummy 'pair 1' sentences \"My {slot}'s name is {name}.\" stored\n"
                        "# before a member's sentence in the same user turn (its positions are dropped at reinstatement). Member\n"
                        "# role r (A = 0, B = 1) of the k-th pair (the collide set's pair order) gets dummies[(2k + r) % 8]\n"
                        "# (bind.dummy_for). Slots: none of fact_pool.yaml's slots; names: in no name / value list, foil, counter,\n"
                        "# lure or collide value; one token (leading space) on the Qwen2.5, Qwen3.5 and Llama-3.2 tokenizers.\n"
                        "# Written by experiments/dream_bind/add8_data.py.\n",
    "pool_lures.yaml": "# dream_bind PREREG addendum 8, part 2a: lures for the 52 non-primary facts (run.pool_gen_items order).\n"
                       "# counter = the counter sentence's value (bind.counter_value; the job override 'plumber').\n"
                       "# " + bind.FORM_RULE + "\n# " + bind.BINDING_RULE + "\n"
                       "# Attribute slots with no unused one-token value get fewer (or no) form lures. Associate lures are not drawn for\n"
                       "# the pool. Written by experiments/dream_bind/add8_data.py.\n",
    "pool_split.yaml": "# dream_bind PREREG addendum 8, part 2b: the two fixed halves of the 52 non-primary facts (bind.pool_split:\n"
                       "# per kind, random.Random(seed + kind index).shuffle, the first ceil(n / 2) to half A). The told and comply\n"
                       "# directions and their (layer set, alpha) are built and picked on half A only; the refusal arms run on B.\n"
                       "# Written by experiments/dream_bind/add8_data.py.\n",
    "pool_pairs.yaml": "# dream_bind PREREG addendum 8, part 2c: two-fact pairs from the pool slots. Confusable: 'older / younger\n"
                       "# {slot}' (person slots) and 'first / second {pet}'; different-subject: two person slots (12) or a person\n"
                       "# and a pet (5), slots from a seeded shuffle (seed 17 / 18), one partner slot (no husband). Values drawn\n"
                       "# with random.Random(13) from the slot's name list, distinct across the whole file, never a primary /\n"
                       "# addendum-5 collide value or a dummy name, one token (leading space) on all three tokenizers. Templates:\n"
                       "# the pool's person / pet kinds (experience, question, prefix, verify, ONE use prompt).\n"
                       "# Written by experiments/dream_bind/add8_data.py.\n"}


def dump(name, obj):
    return HEADERS[name] + yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=140)


def build_all(cfg):
    toks = tokenizers()
    one = one_token_fn(toks)
    check_dummies(cfg, one)
    banned = {n.lower() for n in ["Pepper", "Jasper", "Petra", "Hugo", "Esther", "Omar", "Gordon", "Maria", "Rocky", "Kevin",
                                  "Daniel", "Rosa"]} | {d["name"].lower() for d in DUMMIES}
    return {"dummy_pairs.yaml": dump("dummy_pairs.yaml", {"dummies": DUMMIES}),
            "pool_lures.yaml": dump("pool_lures.yaml", {"facts": build_lures(cfg, one)}),
            "pool_split.yaml": dump("pool_split.yaml", build_split(cfg)),
            "pool_pairs.yaml": dump("pool_pairs.yaml", {"pairs": build_pairs(cfg, one, banned)})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    files = build_all(cfg)
    bad = []
    for name, text in files.items():
        p = DATA / name
        if a.check:
            if not p.exists() or p.read_text() != text:
                bad.append(name)
        else:
            p.write_text(text)
            print(f"wrote {p}")
    if bad:
        sys.exit(f"differs from the generator: {bad}")


if __name__ == "__main__":
    main()
