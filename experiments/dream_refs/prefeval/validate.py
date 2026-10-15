#!/usr/bin/env python
"""PrefEval validation (CPU, no model): the dreamer's raw outputs -> the references and follow-ups prep uses, by
dream_refs' deterministic rules (refs.py, unchanged) plus the follow-up checks (bench.select_followups), and the
spot-check file.

  python experiments/dream_refs/prefeval/validate.py --run RUN

Dreaming failed (PREREG Amendment 2 and the PrefEval addendum): an item whose opposite fails (empty or a hard
check), whose alternatives have fewer than min_valid_alternatives valid ones, or whose follow-ups have fewer than
followups.min valid ones is excluded, with a count, from the variants that need the failed part (follow-ups:
every memory variant, since there is nothing to write or key). The run continues; only a dream that misses items
stops it. The test question is used here ONLY to check the follow-ups against it, and is shown in the spot check.

Outputs: RUN/validate/refs.json, RUN/validate/validation.json, RUN/spotcheck.md
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


refs = _load("seahorse_dream_refs_refs", EXP / "refs.py")
bench = _load("seahorse_prefeval_bench", HERE / "bench.py")
VARIANTS = ("gen:opposite", "gen:centroid", "cs:concept_x_sign")
NEEDS = {"opposite": ("gen:opposite",), "centroid": ("gen:centroid", "cs:concept_x_sign"),
         "followups": VARIANTS}


def validate_item(d, s, cfg, keep_alts=8):
    e = d["experience"]
    fr, tried = refs.select_frame(d["frame_raw"]["greedy"], d["frame_raw"]["samples"], e)
    used = d.get("frame_used")
    consistent = (fr is None and used is None) or (fr is not None and used is not None and fr["frame"] == used["frame"]
                                                   and fr["value"] == used["value"])
    opp = refs.select_opposite(d["opposite_raw"]["greedy"], d["opposite_raw"]["samples"], e, fr)
    alts, ainfo = refs.select_alternatives(d["alternatives_raw"]["samples"], e, fr, opp["text"], keep_alts)
    kw = fr["key_words"] if fr else sorted(refs.content_words(e))
    fus, finfo = bench.select_followups(d["followups_raw"]["samples"], s["question"], kw, refs, cfg["followups"]["keep"])
    failed = refs.dream_failures(opp, ainfo, cfg["min_valid_alternatives"])
    failed["followups"] = (None if finfo["n_kept"] >= cfg["followups"]["min"]
                           else f"{finfo['n_kept']} valid follow-ups (< {cfg['followups']['min']})")
    excluded = sorted({v for r, why in failed.items() if why for v in NEEDS[r]})
    flags = []
    if fr is None:
        flags.append("NO VALID FRAME")
    elif fr["source"] != "greedy":
        flags.append(f"frame from {fr['source']}")
    if fr is not None and fr.get("dropped"):
        flags.append(f"frame drops the clause \"{fr['dropped']}\"")
    if not consistent:
        flags.append("frame used by dream.py != frame re-derived here")
    if opp["ok"] and opp["source"] != "greedy":
        flags.append(f"opposite from {opp['source']}")
    for k, v in opp["soft"].items():
        if (k == "neg_kept" and v) or (k != "neg_kept" and v is False):
            flags.append(f"opposite soft flag: {k}={v}")
    if 0 < ainfo["n_kept"] < keep_alts:
        flags.append(f"only {ainfo['n_kept']} alternatives")
    if ainfo["n_frame_miss"]:
        flags.append(f"{ainfo['n_frame_miss']} alternatives do not fit the frame")
    for r, why in failed.items():
        if why:
            flags.append(f"DREAMING FAILED ({r}): {why}")
    opp["text"] = refs.as_sentence(opp["text"])
    for a in alts:
        a["text"] = refs.as_sentence(a["text"])
    return {"id": d["id"], "row": s["row"], "topic": s["topic"], "polarity": s["polarity"], "experience": e,
            "question": s["question"], "frame": fr, "frame_tried": tried, "opposite": opp, "alternatives": alts,
            "alternatives_info": ainfo, "followups": fus, "followups_info": finfo, "failed": failed,
            "excluded_variants": excluded, "flags": flags}


def validate(dream, sample, cfg, partial=False):
    out, errors = {}, []
    for s in sample["items"]:
        if s["id"] not in dream["items"]:
            if not partial:
                errors.append(f"{s['id']}: not dreamed")
            continue
        out[s["id"]] = validate_item(dream["items"][s["id"]], s, cfg)
    return out, errors


def summary(out):
    n = len(out)
    by_pol = Counter(v["polarity"] for v in out.values())
    fail = lambda r: [i for i, v in out.items() if v["failed"][r]]
    return {"n_items": n, "polarity": dict(by_pol),
            "frame_valid": sum(v["frame"] is not None for v in out.values()),
            "frame_drops_clause": sum(bool(v["frame"] and v["frame"].get("dropped")) for v in out.values()),
            "opposite_ok": sum(v["opposite"]["ok"] for v in out.values()),
            "alternatives_full_8": sum(v["alternatives_info"]["n_kept"] >= 8 for v in out.values()),
            "followups_3": sum(v["followups_info"]["n_kept"] >= 3 for v in out.values()),
            "dream_failed_opposite": len(fail("opposite")), "dream_failed_centroid": len(fail("centroid")),
            "dream_failed_followups": len(fail("followups")),
            "dream_failed_any": sum(any(v["failed"].values()) for v in out.values()),
            "excluded_per_variant": {var: sum(var in v["excluded_variants"] for v in out.values()) for var in VARIANTS},
            "excluded_per_variant_avoid": {var: sum(var in v["excluded_variants"] for v in out.values()
                                                    if v["polarity"] == "avoid") for var in VARIANTS},
            "followup_fail_reasons": dict(sum((Counter(v["followups_info"]["fail_reasons"]) for v in out.values()),
                                              Counter()))}


def spotcheck_md(out, dream, summ):
    L = []
    w = L.append
    w("# Spot check: the dreamed references and follow-ups for the PrefEval sample (experiment 2 addendum)")
    w("")
    w(f"Dreamer: {dream.get('model')} (bf16, thinking off){' — MOCK OUTPUTS' if dream.get('mock') else ''}, prompts v3 "
      "(final) + the follow-up prompt. The dreamer saw only the preference sentence; the test question is shown here "
      "for the check only (it was used only to reject follow-ups too close to it).")
    w("")
    w("**What to check:** is the opposite the opposite preference? Are the alternatives other values of the same kind "
      "(not near-synonyms, not the same thing in another amount)? Are the follow-ups neutral, on the same area of life, "
      "and clearly not the test question?")
    w("")
    w("## Summary")
    w("")
    for k, v in summ.items():
        w(f"- {k}: {v}")
    w("")
    for pol in ("avoid", "mixed", "like", "other"):
        ids = [i for i, v in out.items() if v["polarity"] == pol]
        if not ids:
            continue
        w(f"## {pol} ({len(ids)} items)")
        w("")
        for i in ids:
            v = out[i]
            fr = v["frame"]
            w(f"### {i} ({v['topic']})")
            w("")
            w(f"**Preference:** {v['experience']}")
            w("")
            w(f"**Test question (not seen by the dreamer):** {v['question']}")
            w("")
            if any(v["failed"].values()):
                w("**DREAMING FAILED:** " + "; ".join(f"{r}: {why}" for r, why in v["failed"].items() if why)
                  + f" -> excluded from {', '.join(v['excluded_variants'])}")
                w("")
            w(f"**Frame:** `{fr['frame']}` · value `{fr['value']}` · key words {fr['key_words']} · category "
              f"\"{fr.get('category', '')}\"" + (f" · dropped \"{fr['dropped']}\"" if fr.get("dropped") else "")
              if fr else "**Frame:** none valid")
            w("")
            o = v["opposite"]
            w(f"- opposite: {o['text']}{'' if o['ok'] else ' (FAILS a hard check)'}")
            w("- alternatives: " + " · ".join(f"{a['text']}{' (not in frame)' if a.get('frame_miss') else ''}"
                                               for a in v["alternatives"]) if v["alternatives"] else "- alternatives: none")
            w("- follow-ups: " + " · ".join(f"\"{f['text']}\" (J {f['jaccard_q']:.2f})" for f in v["followups"])
              if v["followups"] else "- follow-ups: none")
            w(f"- flags: {'; '.join(v['flags']) or 'none'}")
            w("")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    a = p.parse_args(argv)
    run = Path(a.run)
    cfg = yaml.safe_load(open(a.config))
    dream = json.load(open(run / "dream" / "dream.json"))
    sample = json.load(open(REPO / cfg["sample"]["file"]))
    out, errors = validate(dream, sample, cfg, partial=bool(dream.get("tiny") or dream.get("mock") or dream.get("subset")))
    summ = summary(out)
    vd = run / "validate"
    vd.mkdir(parents=True, exist_ok=True)
    dsha = hashlib.sha256((run / "dream" / "dream.json").read_bytes()).hexdigest()[:16]
    json.dump({"mock": dream.get("mock", False), "tiny": dream.get("tiny", False), "dream_sha": dsha,
               "dreamer": dream.get("model"), "items": out}, open(vd / "refs.json", "w"), indent=1)
    json.dump({"summary": summ, "errors": errors,
               "dream_failed": {i: {"failed": v["failed"], "excluded_variants": v["excluded_variants"]}
                                for i, v in out.items() if any(v["failed"].values())},
               "flags": {i: v["flags"] for i, v in out.items()}}, open(vd / "validation.json", "w"), indent=1)
    (run / "spotcheck.md").write_text(spotcheck_md(out, dream, summ))
    print(json.dumps({"summary": summ, "errors": errors}, indent=1))
    if errors:
        sys.exit("VALIDATION FAILED: " + "; ".join(errors))


if __name__ == "__main__":
    main()
