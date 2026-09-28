"""CPU validation of a bench directory (tokenizer only, no model).

    python -m seahorse.bench.validate data/bench_v1 [--model Qwen/Qwen2.5-1.5B-Instruct]

Checks schema completeness, Yes/No balance, duplicate prompts, target/foil/contrast
tokenisation, lexicon tokenisation, and the pool; prints counts per category.
Exits non-zero on errors (warnings are reported only).
"""

import argparse
import sys
from collections import Counter

from .data import load_bench, pool_items, pool_size

DISTANCES = ["exact", "paraphrase", "related"]


def _ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def _schema(it, err):
    i = it.get("id", "?")
    need = ["id", "type", "category", "experience", "counter", "followups", "probes", "measure",
            "relation_probes"]
    need += ["contrasts", "ambiguous", "lexicon"] if it["type"] == "disposition" else ["slot"]
    missing = [k for k in need if not it.get(k)]
    if missing:
        err.append(f"{i}: missing {missing}")
        return
    if it["experience"] == it["counter"]:
        err.append(f"{i}: counter equals experience")
    if not 3 <= len(it["followups"]) <= 4:
        err.append(f"{i}: {len(it['followups'])} followups (want 3-4, last generic)")
    if [p["distance"] for p in it["probes"]] != DISTANCES:
        err.append(f"{i}: probes must be exact/paraphrase/related")
    elif it["probes"][0]["text"] != it["followups"][0]:
        err.append(f"{i}: exact probe != followups[0]")
    for rp in it["relation_probes"]:
        if {rp["a"], rp["b"]} != {"Yes", "No"}:
            err.append(f"{i}: relation probe answers must be Yes/No: {rp['text']}")
    kind = "inference" if it["type"] == "disposition" else "verification"
    ans = Counter(rp["a"] for rp in it["relation_probes"] if rp["kind"] == kind)
    need_each = 2 if it["type"] == "disposition" else 1
    if ans["Yes"] != ans["No"] or ans["Yes"] < need_each:
        err.append(f"{i}: unbalanced {kind} probes {dict(ans)} (need equal Yes/No, >= {need_each} each)")
    if it["type"] == "disposition":
        if len(it["contrasts"]) < 3:
            err.append(f"{i}: {len(it['contrasts'])} contrasts (want >= 3)")
        for c in it["contrasts"]:
            if not (c["prefix"] and c["a"] and c["b"]) or c["a"] == c["b"]:
                err.append(f"{i}: bad contrast {c}")
        if it["measure"] != it["contrasts"][0]:
            err.append(f"{i}: measure != contrasts[0]")
        if len(it["ambiguous"]) < 5:
            err.append(f"{i}: {len(it['ambiguous'])} ambiguous prompts (want 5)")
        lex = it["lexicon"]
        for side in ("consistent", "inconsistent"):
            if len(lex.get(side, [])) < 5:
                err.append(f"{i}: lexicon.{side} has < 5 words")
        both = {w.lower() for w in lex["consistent"]} & {w.lower() for w in lex["inconsistent"]}
        if both:
            err.append(f"{i}: lexicon words on both sides: {sorted(both)}")
    else:
        m = it["measure"]
        if not m.get("prefix") or not m.get("target") or len(m.get("foils", [])) < 3:
            err.append(f"{i}: fact measure needs prefix, target and >= 3 foils")
        elif m["target"] in m["foils"] or len(set(m["foils"])) != len(m["foils"]):
            err.append(f"{i}: target among foils or duplicate foils")
    if it.get("pair") and it.get("phrasing") not in ("neutral", "emotional"):
        err.append(f"{i}: emotional item needs phrasing neutral|emotional")


def _texts(it):
    yield from (("followup", t) for t in it["followups"])
    yield from ((p["distance"], p["text"]) for p in it["probes"][1:])
    yield from (("relation", rp["question"]) for rp in it["relation_probes"])
    yield from (("ambiguous", t) for t in it.get("ambiguous", []))


def _duplicates(items, shared, err):
    seen = {}
    for role, text in shared:
        key = text.strip().lower()
        if key in seen:
            err.append(f"duplicate prompt ({role}): {text!r}")
        seen[key] = ("shared", role, None)
    for it in items:
        group = it.get("pair", it["id"])
        for role, text in _texts(it):
            key = text.strip().lower()
            if key in seen:
                g, r, other = seen[key]
                if g == group and r == role and other != it["id"]:
                    continue  # shared by design within an emotional pair
                err.append(f"duplicate prompt: {text!r} ({it['id']}/{role} vs {other or g}/{r})")
                continue
            seen[key] = (group, role, it["id"])


def _tokens(it, tok, err, warn):
    i = it["id"]
    if it["type"] == "fact":
        m = it["measure"]
        pre = _ids(tok, m["prefix"])
        firsts = {}
        for x in [m["target"]] + m["foils"]:
            if not x.startswith(" ") or x.startswith("  "):
                err.append(f"{i}: {x!r} must start with exactly one space")
                continue
            ids = _ids(tok, x)
            if _ids(tok, m["prefix"] + x) != pre + ids:
                err.append(f"{i}: {x!r} does not tokenise consistently after the prefix")
            firsts[x] = ids[0]
        t = m["target"]
        clash = [f for f in m["foils"] if f in firsts and firsts.get(t) == firsts[f]]
        if clash:
            err.append(f"{i}: target {t!r} shares its first token with foils {clash}")
        tid = _ids(tok, t)
        if len(tid) > 1 and len(tok.decode(tid[:1]).strip()) < 3:
            warn.append(f"{i}: target {t!r} starts with a short, non-distinctive token "
                        f"{tok.decode(tid[:1])!r}")
    else:
        for c in it["contrasts"]:
            for x in (c["a"], c["b"]):
                if not x.startswith(" "):
                    err.append(f"{i}: contrast continuation {x!r} must start with a space")
                elif _ids(tok, c["prefix"] + x) != _ids(tok, c["prefix"]) + _ids(tok, x):
                    warn.append(f"{i}: contrast {x!r} tokenises differently after {c['prefix']!r}")
        for side in ("consistent", "inconsistent"):
            for w in it["lexicon"][side]:
                ids = _ids(tok, " " + w)
                if not ids or tok.decode(ids) != " " + w:
                    err.append(f"{i}: lexicon word {w!r} does not round-trip through the tokenizer")
                elif len(ids) > 5:
                    warn.append(f"{i}: lexicon word {w!r} is {len(ids)} tokens")


def _pool_lists(pool, tok, err, warn):
    lists = dict(pool["names"])
    lists.update({sl["slot"]: sl["values"] for sl in pool["slots"] if "values" in sl})
    for name, values in lists.items():
        if len(set(values)) != len(values):
            err.append(f"pool list {name}: duplicate values")
        if len(values) < 5:
            err.append(f"pool list {name}: < 5 values (need target + counter + 3 foils)")
        firsts = Counter()
        for v in values:
            ids = _ids(tok, " " + v) if tok else [v]
            firsts[ids[0]] += 1
            if tok and len(ids) > 1 and name in pool["names"]:
                warn.append(f"pool name {v!r} ({name}) is {len(ids)} tokens")
            elif tok and len(ids) > 1 and len(tok.decode(ids[:1]).strip()) < 3:
                warn.append(f"pool value {v!r} ({name}) starts with a short token {tok.decode(ids[:1])!r}")
        dup = [tok.decode([t]) if tok else t for t, c in firsts.items() if c > 1]
        if dup:
            err.append(f"pool list {name}: values share first tokens {dup}")
    if len(pool["generic_followups"]) < len(pool["slots"]):
        err.append("pool: fewer generic_followups than slots")


def validate_bench(bench, tok=None, pool_seed=0):
    """Validate a loaded bench (see load_bench). With `tok`, also run the tokenisation checks.
    Returns {"errors": [...], "warnings": [...], "counts": {...}}."""
    err, warn = [], []
    items = bench["scenarios"]
    ids = Counter(it["id"] for it in items)
    err += [f"duplicate id {k}" for k, c in ids.items() if c > 1]
    for it in items:
        _schema(it, err)
    pairs = Counter(it["pair"] for it in items if it.get("pair"))
    err += [f"pair {p} has {c} items (want 2)" for p, c in pairs.items() if c != 2]
    if len(bench["unrelated_probes"]) < 25:
        err.append(f"only {len(bench['unrelated_probes'])} unrelated probes (want >= 25)")
    shared = [("unrelated", t) for t in bench["unrelated_probes"]]
    shared += [("placebo", t) for t in bench.get("placebos", [])]
    _duplicates(items, shared, err)

    pool = bench.get("pool") or {}
    pitems = pool_items(pool, pool_seed) if pool.get("slots") else []
    for it in pitems:
        _schema(it, err)
    _duplicates(pitems, shared, err)
    if pool.get("slots"):
        _pool_lists(pool, tok, err, warn)

    if tok is not None:
        for ans in ("Yes", "No"):
            if len(_ids(tok, ans)) != 1:
                err.append(f"answer {ans!r} is not a single token")
        for it in items + pitems:
            _tokens(it, tok, err, warn)

    rel = Counter((it["type"], rp["kind"], rp["a"]) for it in items for rp in it["relation_probes"])
    n_slots, n_facts = pool_size(pool) if pool.get("slots") else (0, 0)
    counts = {
        "items": len(items),
        "by_type": dict(Counter(it["type"] for it in items)),
        "by_category": dict(Counter(f"{it['type']}/{it['category']}" for it in items)),
        "emotional_pairs": len(pairs),
        "relation_probes": {f"{t}/{k}/{a}": c for (t, k, a), c in sorted(rel.items())},
        "contrasts": sum(len(it.get("contrasts", [])) for it in items),
        "ambiguous_prompts": sum(len(it.get("ambiguous", [])) for it in items),
        "unrelated_probes": len(bench["unrelated_probes"]),
        "placebos": len(bench.get("placebos", [])),
        "pool_slots": n_slots,
        "pool_facts": n_facts,
        "tokenizer_checks": tok is not None,
    }
    return {"errors": err, "warnings": warn, "counts": counts}


def format_report(res):
    c = res["counts"]
    lines = [f"items: {c['items']}  {c['by_type']}  emotional pairs: {c['emotional_pairs']}"]
    lines += [f"  {k}: {v}" for k, v in sorted(c["by_category"].items())]
    lines.append("relation probes (type/kind/consistent answer):")
    lines += [f"  {k}: {v}" for k, v in c["relation_probes"].items()]
    lines.append(f"contrasts: {c['contrasts']}  ambiguous prompts: {c['ambiguous_prompts']}  "
                 f"unrelated probes: {c['unrelated_probes']}  placebos: {c['placebos']}")
    lines.append(f"fact pool: {c['pool_slots']} slots, {c['pool_facts']} distinct facts")
    lines.append(f"tokenizer checks: {'on' if c['tokenizer_checks'] else 'OFF'}")
    lines.append(f"errors: {len(res['errors'])}  warnings: {len(res['warnings'])}")
    lines += [f"  ERROR {e}" for e in res["errors"]]
    lines += [f"  warn  {w}" for w in res["warnings"]]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default="data/bench_v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--no-tokenizer", action="store_true")
    args = p.parse_args(argv)
    tok = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
    res = validate_bench(load_bench(args.path), tok)
    print(format_report(res))
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
