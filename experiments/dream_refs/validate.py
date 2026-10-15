#!/usr/bin/env python
"""dream_refs validation (CPU, no model): turn the dreamer's raw outputs into the references prep uses, by the
deterministic rules of refs.py (PREREG.md "The dreaming pass"), and write the human spot-check file.

  python experiments/dream_refs/validate.py --run RUN [--config experiments/dream_refs/config_dream.yaml]

Inputs:  RUN/dream/dream.json (dream.py), core_v1's items (hand-written references: used here ONLY for the
         null_swap references, the topic exclusion list and the side-by-side spot check; never by the dreamer)
Outputs: RUN/validate/refs.json        the references prep uses (generated, null_swap, null_frame, valence)
         RUN/validate/validation.json  every validator flag, per item, and the summary counts
         RUN/spotcheck.md              per item: experience, hand vs generated opposite, hand vs generated
                                       alternatives, paraphrases, nulls, flags (for the user to read)
Dreaming failures (PREREG Amendment 2, 2026-10-15): an item whose opposite fails a hard check (or is empty) or
that has fewer than refs.MIN_VALID_ALTERNATIVES valid alternatives is recorded as "dreaming failed" for that
reference and excluded, with a count, from the variants that need it (refs.NEEDS); the run continues. Failures
are a reported outcome. Exit code 1 (the chain stops) only on fewer than min_topics_fatal valence topics, or on
a dream that misses items.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


refs = _load("seahorse_dream_refs_refs", HERE / "refs.py")


def load_items(cfg):
    itm = _load("seahorse_core_v1_items", REPO / "experiments" / "core_v1" / "items.py")
    L = itm.load_items(str(REPO / cfg["items"]), str(REPO / cfg["bench"]))
    return {it["id"]: it for it in L["items"] if it["group"] in cfg["groups"]}


def validate_item(d, keep):
    """One item's raw dream -> its references and flags."""
    e = d["experience"]
    fr, tried = refs.select_frame(d["frame_raw"]["greedy"], d["frame_raw"]["samples"], e)
    used = d.get("frame_used")
    consistent = (fr is None and used is None) or (fr is not None and used is not None and fr["frame"] == used["frame"]
                                                   and fr["value"] == used["value"])
    opp = refs.select_opposite(d["opposite_raw"]["greedy"], d["opposite_raw"]["samples"], e, fr)
    alts, ainfo = refs.select_alternatives(d["alternatives_raw"]["samples"], e, fr, opp["text"], keep["alternatives"])
    paras, pinfo = refs.select_paraphrases(d["paraphrases_raw"]["samples"], e, fr["key_words"] if fr else [],
                                           keep["paraphrases"])
    nv = d.get("null_values_raw")
    nulls, ninfo = refs.select_null_values(nv["greedy"] if nv else None, nv["samples"] if nv else [], e, fr,
                                           [a["filler"] for a in alts], keep["null_values"])
    flags = []
    if fr is None:
        flags.append("NO VALID FRAME (alternatives not frame-checked; no null_frame: S1 indeterminate)")
    elif fr["source"] != "greedy":
        flags.append(f"frame from {fr['source']} (greedy invalid: {tried[0]['problems']})")
    if not consistent:
        flags.append("frame used by dream.py != frame re-derived here")
    if not opp["ok"]:
        flags.append("opposite FAILS a hard check: " + ", ".join(k for k, v in opp["hard"].items() if not v))
    elif opp["source"] != "greedy":
        flags.append(f"opposite from {opp['source']} (greedy failed a hard check)")
    for k, v in opp["soft"].items():
        if (k == "neg_kept" and v) or (k != "neg_kept" and v is False):
            flags.append(f"opposite soft flag: {k}={v}")
    if ainfo["n_kept"] < keep["alternatives"]:
        flags.append(f"only {ainfo['n_kept']} alternatives")
    if ainfo["n_frame_miss"]:
        flags.append(f"{ainfo['n_frame_miss']} alternatives do not fit the frame")
    if pinfo["n_kept"] < keep["paraphrases"]:
        flags.append(f"only {pinfo['n_kept']} paraphrases")
    if any(p["keeps_key_words"] is False for p in paras):
        flags.append("a paraphrase drops a key word")
    if fr is not None and ninfo["n_kept"] < keep["null_values"]:
        flags.append(f"only {ninfo['n_kept']} null_frame values")
    if fr is not None and fr.get("dropped"):
        flags.append(f"frame drops the clause \"{fr['dropped']}\" (prompts v3)")
    failed = refs.dream_failures(opp, ainfo, keep.get("min_valid_alternatives", refs.MIN_VALID_ALTERNATIVES))
    for ref, why in failed.items():
        if why:
            flags.append(f"DREAMING FAILED ({ref}): {why}")
    # the sentences prep writes before the follow-up end in punctuation, like every hand-written reference
    opp["text"] = refs.as_sentence(opp["text"])
    for a in alts + paras:
        a["text"] = refs.as_sentence(a["text"])
    nulls = [refs.as_sentence(s) for s in nulls]
    out = {"id": d["id"], "group": d["group"], "experience": e, "frame": fr, "frame_tried": tried,
           "opposite": opp, "alternatives": alts, "alternatives_info": ainfo, "paraphrases": paras,
           "paraphrases_info": pinfo, "null_frame": {"sentences": nulls, "info": ninfo,
                                                     "opposite": nulls[0] if nulls else None},
           "failed": failed, "excluded_variants": sorted(refs.excluded_variants(failed)), "flags": flags}
    return out


def overlap(gen, hand):
    """Lexical overlap (report only): Jaccard of content words for the opposite; for the alternatives, how many
    generated values share a content word with some hand-written alternative's distinctive words."""
    hand_words = [refs.content_words(h) for h in hand]
    common = set.intersection(*hand_words) if hand_words else set()
    hits = 0
    for g in gen:
        gw = refs.content_words(g) - common
        hits += any(gw & (hw - common) for hw in hand_words)
    return hits


def validate(dream, items, cfg, partial=False):
    """items: all 29 preference items (partners and the topic exclusion list need all of them). partial: a
    tiny / mock dream of a subset (no error for the items it skipped)."""
    keep = cfg["keep"]
    out, errors = {}, []
    for iid, it in items.items():
        if iid not in dream["items"]:
            if not partial:
                errors.append(f"{iid}: not dreamed")
            continue
        v = validate_item(dream["items"][iid], keep)
        v["null_swap"] = refs.null_swap_refs(items, iid)
        v["overlap"] = {"opposite_jaccard": refs.jaccard(v["opposite"]["text"], it["counter"]),
                        "opposite_exact": refs.ckey(v["opposite"]["text"]) == refs.ckey(it["counter"]),
                        "alternatives_shared": overlap([a["text"] for a in v["alternatives"]], it["centroid"])}
        out[iid] = v
    kw = [k for v in out.values() if v["frame"] for k in v["frame"]["key_words"]]
    words, phrases = refs.item_value_words([it["experience"] for it in items.values()], kw,
                                           [it["lexicon"] for it in items.values()])
    topics, tinfo = refs.select_topics(dream["topics_raw"]["samples"], words, phrases, keep["topics"])
    if len(topics) < cfg["min_topics_fatal"]:
        errors.append(f"only {len(topics)} valence topics (< {cfg['min_topics_fatal']})")
    pairs = refs.valence_pairs(topics, cfg["valence"]["frames"])
    val = {"topics": topics, "info": tinfo, "frames": cfg["valence"]["frames"], "followup": cfg["valence"]["followup"],
           "pairs": pairs, "deviation": None if len(topics) >= keep["topics"] else f"{len(topics)} topics, not {keep['topics']}"}
    return out, val, errors


def summary(out, val):
    n = len(out)
    return {"n_items": n,
            "frame_valid": sum(v["frame"] is not None for v in out.values()),
            "frame_from_greedy": sum(bool(v["frame"]) and v["frame"]["source"] == "greedy" for v in out.values()),
            "opposite_ok": sum(v["opposite"]["ok"] for v in out.values()),
            "opposite_from_greedy": sum(v["opposite"]["source"] == "greedy" for v in out.values()),
            "opposite_same_frame": sum(bool(v["opposite"]["soft"].get("same_frame")) for v in out.values()),
            "opposite_keeps_topic": sum(bool(v["opposite"]["soft"].get("keeps_topic")) for v in out.values()),
            "opposite_equals_hand": sum(v["overlap"]["opposite_exact"] for v in out.values()),
            "alternatives_full_8": sum(v["alternatives_info"]["n_kept"] >= 8 for v in out.values()),
            "alternatives_frame_miss_items": sum(v["alternatives_info"]["n_frame_miss"] > 0 for v in out.values()),
            "paraphrases_full_4": sum(v["paraphrases_info"]["n_kept"] >= 4 for v in out.values()),
            "null_frame_available": sum(bool(v["null_frame"]["sentences"]) for v in out.values()),
            "items_with_flags": sum(bool(v["flags"]) for v in out.values()),
            "frame_drops_clause": sum(bool(v["frame"] and v["frame"].get("dropped")) for v in out.values()),
            "dream_failed_opposite": sum(bool(v["failed"]["opposite"]) for v in out.values()),
            "dream_failed_centroid": sum(bool(v["failed"]["centroid"]) for v in out.values()),
            "dream_failed_any": sum(any(v["failed"].values()) for v in out.values()),
            "valence_topics": len(val["topics"]), "valence_pairs": len(val["pairs"])}


def spotcheck_md(out, val, items, dream, summ):
    L = []
    w = L.append
    w("# Spot check: the dreamed references (experiment 2, dream_refs)")
    w("")
    w(f"Dreamer: {dream.get('model')} (bf16, thinking off){' — MOCK OUTPUTS (plumbing test only)' if dream.get('mock') else ''}. "
      "The dreamer saw only each experience sentence; the hand-written columns are shown here for comparison only.")
    w("")
    w("**What to check** (one evening's work, 02-scaffold.md §4): is each generated opposite really the opposite "
      "preference, in the same frame, without a bolted-on negation? Are the 8 alternatives the same kind of "
      "thing as the original, in the same frame, and none of them the original or the opposite? "
      "Validator flags are listed per item; they are reported, never used to drop an item.")
    w("")
    w("## Summary")
    w("")
    for k, v in summ.items():
        w(f"- {k}: {v}")
    w("")
    for g in ("leanings", "dislikes", "one_of_many"):
        w(f"## {g}")
        w("")
        for iid, v in out.items():
            if v["group"] != g:
                continue
            it = items[iid]
            fr = v["frame"]
            w(f"### {iid}")
            w("")
            w(f"**Experience:** {v['experience']}")
            w("")
            if any(v["failed"].values()):
                w("**DREAMING FAILED:** " + "; ".join(f"{r}: {why}" for r, why in v["failed"].items() if why)
                  + f" -> excluded from {', '.join(v['excluded_variants'])}")
                w("")
            w(f"**Frame:** `{fr['frame']}` · value `{fr['value']}` · key words {fr['key_words']} · category "
              f"\"{fr.get('category', '')}\"" + (f" · dropped clause \"{fr['dropped']}\"" if fr.get("dropped") else "")
              + f" ({fr['source']})" if fr else "**Frame:** none valid")
            w("")
            o = v["opposite"]
            w("| | hand-written | generated |")
            w("|---|---|---|")
            w(f"| opposite | {it['counter']} | {o['text']} ({o['source']}{'' if o['ok'] else ', FAILS hard check'}) |")
            alts = v["alternatives"]
            for k in range(max(len(alts), len(it["centroid"]))):
                h = it["centroid"][k] if k < len(it["centroid"]) else ""
                a = alts[k] if k < len(alts) else None
                g_ = (f"{a['text']} ({a['count']}/{len(dream['items'][iid]['alternatives_raw']['samples'])}"
                      f"{', not in frame' if a.get('frame_miss') else ''})") if a else ""
                w(f"| alternative {k + 1} | {h} | {g_} |")
            w("")
            w("Paraphrases (generated): " + " · ".join(f"\"{p['text']}\"" for p in v["paraphrases"]))
            w("")
            ns = v["null_swap"]
            nf = v["null_frame"]["sentences"]
            w(f"Nulls: **null_swap** = partner `{ns['partner']}`'s hand opposite \"{ns['opposite']}\" and its 8 hand "
              f"alternatives; **null_frame** = " + (" · ".join(f"\"{s}\"" for s in nf) if nf else "none"))
            w("")
            ov = v["overlap"]
            w(f"Overlap with hand (report only): opposite Jaccard {ov['opposite_jaccard']:.2f}"
              f"{' (identical)' if ov['opposite_exact'] else ''}; generated alternatives sharing a value word with a hand "
              f"alternative: {ov['alternatives_shared']}/{len(alts)}")
            w("")
            w("Flags: " + ("; ".join(v["flags"]) if v["flags"] else "none"))
            w("")
    w("## Valence topics")
    w("")
    w(f"{len(val['topics'])} topics (round-robin over areas; excluded if they touch an item's value or lexicon): "
      + ", ".join(f"{t['thing']} ({t['area']})" for t in val["topics"]))
    w("")
    w(f"Excluded examples: {val['info']['excluded_examples']}")
    w("")
    w(f"Frames: {val['frames']}; follow-up: \"{val['followup']}\". Example pair: "
      + (f"\"{val['pairs'][0]['like']}\" / \"{val['pairs'][0]['hate']}\"" if val["pairs"] else "none"))
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run root (holds dream/)")
    p.add_argument("--config", default=str(HERE / "config_dream.yaml"))
    a = p.parse_args(argv)
    run = Path(a.run)
    cfg = yaml.safe_load(open(a.config))
    dream = json.load(open(run / "dream" / "dream.json"))
    items = load_items(cfg)
    out, val, errors = validate(dream, items, cfg, partial=bool(dream.get("tiny") or dream.get("mock")))
    summ = summary(out, val)
    vd = run / "validate"
    vd.mkdir(parents=True, exist_ok=True)
    dsha = hashlib.sha256((run / "dream" / "dream.json").read_bytes()).hexdigest()[:16]
    json.dump({"mock": dream.get("mock", False), "tiny": dream.get("tiny", False), "dream_sha": dsha,
               "dreamer": dream.get("model"), "items": out, "valence": val}, open(vd / "refs.json", "w"), indent=1)
    json.dump({"summary": summ, "errors": errors, "valence_info": val["info"], "valence_deviation": val["deviation"],
               "flags": {i: v["flags"] for i, v in out.items()},
               "dream_failed": {i: {"failed": v["failed"], "excluded_variants": v["excluded_variants"]}
                                for i, v in out.items() if any(v["failed"].values())},
               "opposite_checks": {i: {"hard": v["opposite"]["hard"], "soft": v["opposite"]["soft"]} for i, v in out.items()},
               "alternatives_info": {i: v["alternatives_info"] for i, v in out.items()},
               "overlap": {i: v["overlap"] for i, v in out.items()}},
              open(vd / "validation.json", "w"), indent=1)
    (run / "spotcheck.md").write_text(spotcheck_md(out, val, items, dream, summ))
    print(json.dumps({"summary": summ, "errors": errors}, indent=1))
    if errors:
        sys.exit("VALIDATION FAILED: " + "; ".join(errors))


if __name__ == "__main__":
    main()
