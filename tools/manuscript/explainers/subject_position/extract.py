#!/usr/bin/env python3
"""Data for the Subject-Position Test page, rebuilt from the dream_subj run.

    python3 -I extract.py <results dir> <out.json> [<stats.json>]

Reads results/dream_subj_20261022_1755/{meta.json,rows.jsonl.gz}. Only the 45 person and pet facts
(the 10 attribute facts are reported separately in the run's report.md and not drawn on the page).

Output schema:
  conds   display order of the conditions
  facts   {id: {kind, tier, exp, q, subj, val,
                c: {cond: {a, lp, your, val, ref, asx, yn, gain}}}}
            a     answer_score of the direct question (tf row, pkind "direct")
            lp    lp_target of that row
            your  p_first["Your"];  val  p_first["value"];  ref  p_first["I"] + p_first["Sorry"];
            asx   p_first["As"]
            yn    balanced first-token yes/no over the fact's "yn" rows
                  (mean of acc on "Yes"-consistent probes and acc on "No"-consistent probes;
                   "yes" when margin = log P(Yes) - log P(No) > 0)
            gain  lp - lp under nomem (both rounded to 3 decimals first)
  attn    {cond: {last_subj, last_value, qsubj_subj, qsubj_value, n}} for R+subj and R-subj2:
            per-layer mean over facts of the direct row's attn field
stats() reads verdict.json of the same run (the run's own bootstrap: 95% percentile over facts, 2000 replicates,
seed 7) for the dot plots, the tables and the tiles: {answer, gain, yn: {cond: [mean, lo, hi]}, refuse: {cond: mean},
contrasts: [[label, mean, lo, hi, yes/no mean]], closed: share of the R-min to R-full gap closed by R+subj}.
Standard library only.
"""

import gzip
import json
import sys
from pathlib import Path

RUN = "dream_subj_20261022_1755"
ORDER = ["nomem", "T-min", "T-subj3", "R-min", "R+fill", "R+rel", "R-subj2", "R+subj", "R-full", "ctx"]
ATTN_CONDS = ["R+subj", "R-subj2"]
KINDS = ("person", "pet")


def extract(results):
    run = Path(results) / RUN
    meta = json.loads((run / "meta.json").read_text())
    with gzip.open(run / "rows.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert sorted(meta["conds"]) == sorted(ORDER), meta["conds"]

    ids = [fid for fid, m in meta["facts"].items() if m["kind"] in KINDS]
    direct, yn = {}, {}
    for r in rows:
        if r["fact"] not in ids:
            continue
        if r["sec"] == "tf" and r.get("pkind") == "direct":
            key = (r["fact"], r["cond"])
            assert key not in direct, key
            direct[key] = r
        elif r["sec"] == "yn":
            yn.setdefault((r["fact"], r["cond"]), []).append(r)

    def balanced(rs):
        acc = {"Yes": [], "No": []}
        for r in rs:
            said_yes = r["margin"] > 0
            acc[r["consistent"]].append(1.0 if said_yes == (r["consistent"] == "Yes") else 0.0)
        return 0.5 * (sum(acc["Yes"]) / len(acc["Yes"]) + sum(acc["No"]) / len(acc["No"]))

    facts = {}
    for fid in ids:
        m = meta["facts"][fid]
        base = round(direct[(fid, "nomem")]["lp_target"], 3)
        c = {}
        for cond in meta["conds"]:
            r = direct[(fid, cond)]
            p = r["p_first"]
            c[cond] = {
                "a": round(r["answer_score"], 3),
                "lp": round(r["lp_target"], 3),
                "your": round(p["Your"], 4),
                "val": round(p["value"], 4),
                "ref": round(p["I"] + p["Sorry"], 4),
                "asx": round(p["As"], 4),
                "yn": round(balanced(yn[(fid, cond)]), 3),
                "gain": round(round(r["lp_target"], 3) - base, 3),
            }
        facts[fid] = {"kind": m["kind"], "tier": m["tier"], "c": c, "exp": m["experience"], "q": m["question"],
                      "subj": m["tok"]["subj"].strip(), "val": m["target"].strip()}

    attn = {}
    for cond in ATTN_CONDS:
        rs = [direct[(fid, cond)]["attn"] for fid in ids]
        out = {}
        for q in ("last", "qsubj"):
            for k in ("subj", "value"):
                layers = list(zip(*(a[q][k] for a in rs)))
                out[f"{q}_{k}"] = [round(sum(v) / len(v), 4) for v in layers]
        out["n"] = len(rs)
        attn[cond] = out
    return {"conds": ORDER, "facts": facts, "attn": attn}


# Contrasts shown in the rule table: (key in verdict.json, label on the page).
CONTRASTS = [("R+subj - R-min", "R+subj − R-min"), ("R+subj - R+fill", "R+subj − R+fill"),
             ("R+subj - R+rel", "R+subj − R+rel"), ("R+fill - R-min", "R+fill − R-min"),
             ("R+rel - R-min", "R+rel − R-min"), ("R-subj2 - R-min", "R-subj2 − R-min (both 2 positions)"),
             ("R-full - R+subj", "R-full − R+subj"), ("R+subj - T-subj3", "R+subj − T-subj3 (slice vs text)")]


def stats(results):
    v = json.loads((Path(results) / RUN / "verdict.json").read_text())
    a, c = v["splits"]["all"], v["contrasts"]["all"]
    r2 = lambda x: round(x, 2)
    tri = lambda d: [r2(d["mean"]), r2(d["lo"]), r2(d["hi"])]
    out = {m: {cond: tri(a[cond][m]) for cond in ORDER} for m in ("answer", "gain", "yn")}
    out["refuse"] = {cond: r2(a[cond]["p_refuse"]["mean"]) for cond in ORDER}
    out["contrasts"] = [[label] + tri(c[key]["answer"]) + [r2(c[key]["yn"]["mean"])] for key, label in CONTRASTS]
    ans = lambda cond: a[cond]["answer"]["mean"]
    out["closed"] = r2((ans("R+subj") - ans("R-min")) / (ans("R-full") - ans("R-min")))
    return out


def placeholders(results):
    """Template placeholder -> JSON object, for build_explainers.py."""
    return {"__DATA__": extract(results), "__STATS__": stats(results)}


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    data = extract(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    if len(sys.argv) == 4:
        Path(sys.argv[3]).write_text(json.dumps(stats(sys.argv[1]), ensure_ascii=False, separators=(",", ":")))
