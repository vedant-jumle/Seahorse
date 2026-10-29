#!/usr/bin/env python
"""dream_bind PREREG addendum 9: build data/bench_v1/groups9.yaml deterministically (run once, locally, before any job;
the file is the pre-registered record and is checked by tests/test_dream_bind.py and at run time).

  conda run -n torch python experiments/dream_bind/add9_data.py [--check]

No torch and no model: only yaml, random and the tokenizers' tokenizer.json files from the local Hugging Face cache
(read with the `tokenizers` library), for the one-token check on the Qwen2.5, Qwen3.5 and Llama-3.2 tokenizers.

groups9.yaml: groups of N = 3 or N = 5 facts "My {subject}'s name is {v}." stored together (addendum 9):
  n3 confusable   7 person groups "eldest / middle / youngest {slot}" + 5 pet groups "first / second / third {pet}"
  n3 different    6 groups of 3 distinct slots (persons and pets mixed; seeded shuffle)
  n5 confusable   8 pet groups "first / second / third / fourth / fifth {pet}"
  n5 different    4 groups of 5 distinct slots (the next slots of the same shuffle)
Values: drawn with random.Random(VALUE_SEED) from this file's own name lists (FEMALE9 / MALE9 / PET9, by the slot's name
list in fact_pool.yaml), distinct across the file. Every list entry is one token (leading space) on all three tokenizers
and is in NO earlier dream_bind name list or use: not a fact_pool name or value, not a facts.yaml target, foil or counter
word, not a lure (config.yaml lures, pool_lures.yaml), not a collide or pool-pair value, not a dummy name. (The fact_pool
lists cannot be used: after those exclusions they keep 1 female, 2 male and 2 pet names.)
--check: rebuild in memory and fail if the written file differs.
"""

import argparse
import random
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO / "data" / "bench_v1"
OUT = DATA / "groups9.yaml"
TOKENIZERS = ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3.5-2B", "meta-llama/Llama-3.2-3B-Instruct")
VALUE_SEED, SLOT_SEED = 9, 10

# the confusable groups' ordinal subjects, member 1 first
PERSON_ORD = ["eldest", "middle", "youngest"]
PET_ORD = ["first", "second", "third", "fourth", "fifth"]
CONF_PERSON = ["sister", "brother", "cousin", "niece", "nephew", "son", "daughter"]   # n3: "eldest / middle / youngest {slot}"
CONF_PET = ["dog", "cat", "rabbit", "parrot", "horse", "hamster", "goldfish", "tortoise"]
N3_PETS, N5_PETS = CONF_PET[:5], CONF_PET[:8]
N3_DIFF, N5_DIFF = 6, 4

# new name lists (addendum 9): one token (leading space) on all three tokenizers, in no earlier list (checked below).
# Hand-picked: no two forms of one name (Tom / Thomas), no homophones (Lewis / Louis), and for pets no very common word
# a pet poem would use for itself (rocket, bolt, buttons, moose, ...).
FEMALE9 = ["Lisa", "Linda", "Susan", "Karen", "Nancy", "Betty", "Sandra", "Donna", "Carol", "Amy", "Anne", "Jane", "Lucy",
           "Sophie", "Amanda", "Jessica", "Ellen", "Kelly", "Megan", "Holly", "Heather", "Diane", "Joan", "Julie", "Claire",
           "Barbara", "Monica", "Tina", "Gina", "Natalie", "Heidi", "Elsa", "Lara", "Isabel", "Charlotte", "Caroline",
           "Louise", "Dana", "Tara", "Sally", "Alison", "Peggy", "Rita", "Carla", "Sonia", "Gloria", "Teresa", "Andrea"]
MALE9 = ["Sam", "Tom", "Jim", "Joe", "Tim", "Dan", "Ian", "Alan", "Eric", "Steve", "Mike", "Jeff", "Brad", "Greg", "Chris",
         "Matt", "Nick", "Josh", "Ryan", "Kyle", "Sean", "Luke", "Andy", "Tony", "Jake", "Adrian", "Arthur", "Albert",
         "Edward", "Charles", "Robert", "Patrick", "Scott", "Larry", "Harry", "Ronald", "Kenneth", "Donald", "Tyler",
         "Aaron", "Nathan", "Ethan", "Jeremy", "Gerald", "Austin", "Dylan", "Bryan", "Logan", "Wayne", "Russell", "Johnny",
         "Carlos", "Pablo", "Miguel", "Boris", "Igor", "Neil", "Craig", "Stuart", "Malcolm", "Owen", "Connor", "Nigel",
         "Rupert", "Spencer", "Lloyd", "Lewis", "Wesley", "Warren", "Stanley", "Leonard", "Howard"]
PET9 = ["Clover", "Willow", "Atlas", "Nova", "Bingo", "Buster", "Bruno", "Diesel", "Domino", "Baxter", "Murphy", "Bailey",
        "Riley", "Cooper", "Tucker", "Bentley", "Harley", "Lola", "Sasha", "Pip", "Bowie", "Elvis", "Dexter", "Teddy",
        "Benny", "Bonnie", "Fritz", "Gus", "Hank", "Mickey", "Pluto", "Spike", "Taco", "Tango", "Merlin", "Mozart",
        "Darwin", "Newton", "Tesla", "Plato", "Caesar", "Nero", "Hercules", "Achilles", "Jupiter", "Saturn", "Orion",
        "Phoenix", "Raven", "Ember", "Bach", "Bert", "Basil", "Polly", "Archie", "Tyson", "Hermes", "Zelda", "Mario",
        "Luigi", "Yoshi", "Kirby", "Lennon", "Romeo", "Juliet", "Watson", "Holmes", "Kobe", "Cher", "Madonna", "Oprah"]
LISTS = {"female": FEMALE9, "male": MALE9, "pet": PET9}


def _hf_file(repo, name):
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(repo, name)
    assert isinstance(p, str), f"{repo}/{name} is not in the local Hugging Face cache"
    return p


def tokenizers():
    from tokenizers import Tokenizer
    return [Tokenizer.from_file(_hf_file(r, "tokenizer.json")) for r in TOKENIZERS]


def one_token_fn(toks):
    return lambda v: all(len(t.encode(" " + v.strip(), add_special_tokens=False).ids) == 1 for t in toks)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def used_names(cfg):
    """Every name / value / word used by an earlier dream_bind test item (lower case): the fact_pool lists and slot values,
    facts.yaml targets, foils and every word of the counter sentences, config.yaml lures and collide values, the pool facts'
    targets, counters, foils and lures (pool_lures.yaml), the pool-pair values and the dummy names."""
    pool = yaml.safe_load(open(DATA / "fact_pool.yaml"))
    names = {w for l in pool["names"].values() for w in l} | {v for s in pool["slots"] for v in s.get("values", [])}
    for f in yaml.safe_load(open(DATA / "facts.yaml"))["facts"]:
        names |= {f["measure"]["target"].strip(), *[x.strip() for x in f["measure"]["foils"]]}
        names |= set(re.findall(r"[A-Za-z][A-Za-z'-]*", f["counter"]))
    names |= {v.strip() for L in cfg["lures"].values() for k in L.values() for v in k}
    for p in cfg["collide"]["pairs"]:
        for k in ("a", "b"):
            if isinstance(p[k], dict):
                names.add(p[k]["value"].strip())
    for f in yaml.safe_load(open(DATA / "pool_lures.yaml"))["facts"]:
        names |= {f["target"], f["counter"], *f["foils"], *f["form"], *f["binding"]}
    for p in yaml.safe_load(open(DATA / "pool_pairs.yaml"))["pairs"]:
        names |= {p["a"]["value"].strip(), p["b"]["value"].strip()}
    names |= {d["name"] for d in yaml.safe_load(open(DATA / "dummy_pairs.yaml"))["dummies"]}
    return {n.lower() for n in names}


def _member(mid, subject, kind, v):
    use = (f"Write a short birthday message to my {subject}." if kind == "person" else
           f"Write a short, funny poem about my {subject}.")
    return {"id": mid, "subject": subject, "slot_kind": kind, "experience": f"My {subject}'s name is {v}.", "value": f" {v}",
            "question": f"What's my {subject}'s name?", "prefix": f"Your {subject}'s name is", "use": [use]}


def build_groups(cfg, one):
    pool = yaml.safe_load(open(DATA / "fact_pool.yaml"))
    slots = {s["slot"]: s for s in pool["slots"]}
    used = used_names(cfg)
    for name, L in LISTS.items():
        assert len(set(L)) == len(L), f"{name}: duplicate"
        bad = [v for v in L if v.lower() in used or not one(v)]
        assert not bad, f"{name}: used before or not one token: {bad}"
    allv = [v.lower() for L in LISTS.values() for v in L]
    assert len(set(allv)) == len(allv), "the three lists overlap"
    rng = random.Random(VALUE_SEED)
    taken = set()

    def draw(slot, n):
        L = [v for v in LISTS[slots[slot]["names"]] if v not in taken]
        got = rng.sample(L, n)
        taken.update(got)
        return got

    groups = []

    def conf(n, slot, ords, kind):
        vals = draw(slot, n)
        gid = f"g{n}c_{slug(slot)}"
        groups.append({"id": gid, "n": n, "kind": "confusable",
                       "members": [_member(f"{gid}_{k + 1}_{o}", f"{o} {slot}", kind, v)
                                   for k, (o, v) in enumerate(zip(ords[:n], vals))]})
    for s in CONF_PERSON:
        conf(3, s, PERSON_ORD, "person")
    for s in N3_PETS:
        conf(3, s, PET_ORD, "pet")
    for s in N5_PETS:
        conf(5, s, PET_ORD, "pet")
    # different-subject groups: the person slots (no husband: one partner) and the pet slots, in fact_pool order, shuffled
    # with random.Random(SLOT_SEED); the first 18 make the six N = 3 groups, the next 20 the four N = 5 groups
    mixed = [s["slot"] for s in pool["slots"] if s.get("kind") in ("person", "pet") and s["slot"] != "husband"]
    random.Random(SLOT_SEED).shuffle(mixed)
    pos = 0
    for n, cnt in ((3, N3_DIFF), (5, N5_DIFF)):
        for g in range(cnt):
            ss = mixed[pos:pos + n]
            pos += n
            gid = f"g{n}d_{g + 1}"
            mem = []
            for k, s in enumerate(ss):
                v, = draw(s, 1)
                mem.append(_member(f"{gid}_{k + 1}_{slug(s)}", s, slots[s]["kind"], v))
            groups.append({"id": gid, "n": n, "kind": "different", "members": mem})
    order = {(3, "confusable"): 0, (3, "different"): 1, (5, "confusable"): 2, (5, "different"): 3}
    groups.sort(key=lambda g: order[(g["n"], g["kind"])])       # stable: the build order within each block
    check_groups(groups, cfg)
    return groups


def check_groups(groups, cfg):
    """Distinct values and ids; no value is a word of any sentence or prompt of the addendum (members' templates, the
    dummies, the placebos); counts as pre-registered."""
    vals = [m["value"].strip() for g in groups for m in g["members"]]
    ids = [m["id"] for g in groups for m in g["members"]]
    assert len(set(v.lower() for v in vals)) == len(vals) and len(set(ids)) == len(ids)
    assert len(vals) == 114, len(vals)
    texts = [t for g in groups for m in g["members"] for t in (m["experience"].replace(m["value"].strip(), ""), m["question"],
                                                               m["prefix"], *m["use"])]
    texts += cfg["add9"]["placebos"]
    texts += [f"My {d['slot']}'s name is {d['name']}." for d in yaml.safe_load(open(DATA / "dummy_pairs.yaml"))["dummies"]]
    words = {w.lower() for t in texts for w in re.findall(r"[A-Za-z]+", t)}
    assert not {v.lower() for v in vals} & words, {v for v in vals if v.lower() in words}
    for g in groups:
        assert len(g["members"]) == g["n"]
        subj = [m["subject"] for m in g["members"]]
        assert len(set(subj)) == len(subj)
    cnt = lambda n, k: sum(g["n"] == n and g["kind"] == k for g in groups)
    assert (cnt(3, "confusable"), cnt(3, "different"), cnt(5, "confusable"), cnt(5, "different")) == (12, 6, 8, 4)
    diff = [m["subject"] for g in groups if g["kind"] == "different" for m in g["members"]]
    assert len(set(diff)) == len(diff) == 38


HEADER = ("# dream_bind PREREG addendum 9: groups of N = 3 / N = 5 facts stored together (does storage order fix the slice\n"
          "# collision for more than two facts?). Confusable: 'eldest / middle / youngest {slot}' (7 person slots) and\n"
          "# 'first / second / third {pet}' (5 pet slots) at N = 3; 'first .. fifth {pet}' (8 pet slots) at N = 5. Different-\n"
          "# subject: distinct slots from a seeded shuffle (random.Random(10)) of the fact_pool person (no husband) and pet\n"
          "# slots, 6 groups of 3 then 4 of 5. Values drawn with random.Random(9) from add9_data.py's own lists (FEMALE9 /\n"
          "# MALE9 / PET9, by the slot's fact_pool name list), distinct across the file, in no earlier dream_bind list (no\n"
          "# fact_pool name, target, foil, counter, lure, collide / pool-pair value or dummy name), one token (leading space)\n"
          "# on the Qwen2.5, Qwen3.5 and Llama-3.2 tokenizers. Member k (1-based, the list order) is stored as the k-th fact\n"
          "# in RN-sep-ordered. Templates: the pool's person / pet kinds, ONE use prompt.\n"
          "# Written by experiments/dream_bind/add9_data.py.\n")


def build_text(cfg):
    one = one_token_fn(tokenizers())
    return HEADER + yaml.safe_dump({"groups": build_groups(cfg, one)}, sort_keys=False, allow_unicode=True, width=140)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    text = build_text(cfg)
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"differs from the generator: {OUT}")
        print(f"ok: {OUT}")
    else:
        OUT.write_text(text)
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
