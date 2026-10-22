"""dream_subj analysis: tables with item-bootstrap CIs (2000 reps, seed 7), paired contrasts, the pre-registered rule
(PREREG.md), attention to the restored subject entry, example facts. Writes <run>/report.md and <run>/verdict.json."""

import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "src"))
import subj  # noqa: E402
from seahorse.stats import boot_ci  # noqa: E402

N_BOOT, SEED = 2000, 7
SPLITS = {"all": ("primary", "pool"), "primary": ("primary",), "pool": ("pool",), "attribute": ("attribute",)}
CONTRASTS = [("R+subj", "R-min"), ("R+subj", "R+fill"), ("R+subj", "R+rel"), ("R+fill", "R-min"), ("R+rel", "R-min"),
             ("R-subj2", "R-min"), ("R-full", "R-min"), ("R-full", "R+subj"), ("ctx", "R-full"), ("T-subj3", "T-min"),
             ("R+subj", "T-subj3")]


def ci(nested):
    p, lo, hi, n = boot_ci(nested, n_boot=N_BOOT, seed=SEED)
    return {"mean": p, "lo": lo, "hi": hi, "n": n}


def f2(s, d=2):
    if s is None or s["n"] == 0 or math.isnan(s["mean"]):
        return "n/a"
    return f"{s['mean']:.{d}f} [{s['lo']:.{d}f}, {s['hi']:.{d}f}]"


def load(run):
    with gzip.open(run / "rows.jsonl.gz", "rt") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    meta = json.load(open(run / "meta.json"))
    return rows, meta


def measures(rows):
    """{measure: {cond: {fact: {prompt: value}}}} and {fact: tier}."""
    M = defaultdict(lambda: defaultdict(dict))
    tier = {}
    nm = {(r["fact"], r["pi"]): r["lp_target"] for r in rows if r["sec"] == "tf" and r["cond"] == "nomem"}
    for r in rows:
        tier[r["fact"]] = r["tier"]
        f, c = r["fact"], r["cond"]
        if r["sec"] == "tf":
            g = r["lp_target"] - nm[(f, r["pi"])]
            if r["pkind"] == "direct":
                M["answer"][c][f] = {0: r["answer_score"]}
                M["gain"][c][f] = {0: g}
                M["p_your"][c][f] = {0: r["p_first"]["Your"]}
                M["p_refuse"][c][f] = {0: r["p_first"]["I"] + r["p_first"]["Sorry"]}
                M["p_value"][c][f] = {0: r["p_first"]["value"]}
                M["lp_target"][c][f] = {0: r["lp_target"]}
            else:
                M["gain_use"][c].setdefault(f, {})[r["pi"]] = g
    yn = defaultdict(list)
    for r in rows:
        if r["sec"] == "yn":
            yn[(r["cond"], r["fact"])].append((r["consistent"], r["margin"]))
    for (c, f), ms in yn.items():
        M["yn"][c][f] = {0: subj.balanced_acc(ms)}
        M["yn_margin"][c][f] = {0: float(np.mean([m if a == "Yes" else -m for a, m in ms]))}
    return M, tier


def sel(d, facts):
    return {f: v for f, v in d.items() if f in facts}


def paired(a, b):
    """{fact: {prompt: a - b}} over the cells present in both."""
    out = {}
    for f, pa in a.items():
        if f in b:
            dd = {p: pa[p] - b[f][p] for p in pa if p in b[f]}
            if dd:
                out[f] = dd
    return out


def direct_minus_use(M, x, y, facts):
    """Per fact: (x - y target gain on the direct question) - (x - y target gain on the use prompts, mean)."""
    d = paired(sel(M["gain"][x], facts), sel(M["gain"][y], facts))
    u = paired(sel(M["gain_use"][x], facts), sel(M["gain_use"][y], facts))
    return {f: {0: d[f][0] - float(np.mean(list(u[f].values())))} for f in d if f in u}


def analyze(run):
    run = Path(run)
    rows, meta = load(run)
    M, tier = measures(rows)
    conds = meta["conds"]
    facts_of = {s: {f for f, t in tier.items() if t in ts} for s, ts in SPLITS.items()}
    V = {"splits": {}}
    L = ["# dream_subj: does a restored slice need the subject's position? (Qwen2.5-1.5B, teacher-forced)", "",
         "## Summary", "", "_(written after reading the tables; see below)_", "",
         f"Run: `{run}`; {len(meta['facts'])} facts; {meta['minutes']:.1f} min; peak GPU {meta['peak_gpu_gb']:.2f} GB; "
         f"commit {meta['env'].get('commit')} (dirty: {meta['env'].get('dirty')}). Pre-registration: "
         "`experiments/dream_subj/PREREG.md`. CIs: 95% percentile bootstrap over facts (use prompts: facts then prompts), "
         "2000 replicates, seed 7; contrasts are paired on the same prompts.", ""]
    # ---------------------------------------------------------------- the rule
    allf = facts_of["all"]
    A = lambda x, y, m="answer", fs=allf: ci(paired(sel(M[m][x], fs), sel(M[m][y], fs)))
    rule_in = {"d_min": A("R+subj", "R-min"), "d_fill": A("R+subj", "R+fill"), "d_rel": A("R+subj", "R+rel"),
               "f_min": A("R+fill", "R-min"), "r_min": A("R+rel", "R-min")}
    label, why = subj.verdict(**rule_in)
    V["rule"] = {"label": label, "reasons": why, **{k: v for k, v in rule_in.items()}}
    L += ["## Rule verdict (pre-registered; direct question, answer score, all person/pet facts)", "",
          f"**{label}**: {'; '.join(why)}", "",
          "| contrast (answer score, nats) | paired mean [95% CI] | N |", "|---|---|---|"]
    names = {"d_min": "R+subj − R-min", "d_fill": "R+subj − R+fill", "d_rel": "R+subj − R+rel",
             "f_min": "R+fill − R-min", "r_min": "R+rel − R-min"}
    for k, v in rule_in.items():
        L.append(f"| {names[k]} | {f2(v)} | {v['n']} |")
    L.append("")
    # ---------------------------------------------------------------- per-condition tables
    for s in ("all", "primary", "pool", "attribute"):
        fs = facts_of[s]
        if not fs:
            continue
        L += [f"## Per condition: {s} ({len(fs)} facts)", "",
              "| condition | N | answer score log P(prefix) | P('Your') first | P(value) first | P('I'+'Sorry') first | "
              "target gain, direct (nats) | target gain, use (forced prefix) | TF yes/no balanced |",
              "|---|---|---|---|---|---|---|---|---|"]
        V["splits"][s] = {}
        for c in conds:
            st = {m: ci(sel(M[m][c], fs)) for m in ("answer", "p_your", "p_value", "p_refuse", "gain", "gain_use", "yn")}
            V["splits"][s][c] = st
            if st["answer"]["n"] == 0:
                continue
            L.append(f"| {c} | {st['answer']['n']} | {f2(st['answer'])} | {f2(st['p_your'])} | {f2(st['p_value'])} | "
                     f"{f2(st['p_refuse'])} | {f2(st['gain'])} | {f2(st['gain_use'])} | {f2(st['yn'])} |")
        L.append("")
    # ---------------------------------------------------------------- contrasts
    L += ["## Paired contrasts", "", "Each cell: paired mean [95% CI] (N facts).", ""]
    V["contrasts"] = {}
    for s in ("all", "primary", "pool", "attribute"):
        fs = facts_of[s]
        if not fs:
            continue
        L += [f"### {s}", "", "| contrast | answer score | target gain, direct | target gain, use | TF yes/no |",
              "|---|---|---|---|---|"]
        V["contrasts"][s] = {}
        for x, y in CONTRASTS:
            st = {m: ci(paired(sel(M[m][x], fs), sel(M[m][y], fs))) for m in ("answer", "gain", "gain_use", "yn")}
            V["contrasts"][s][f"{x} - {y}"] = st
            if st["answer"]["n"] == 0:
                continue
            L.append(f"| {x} − {y} | {f2(st['answer'])} ({st['answer']['n']}) | {f2(st['gain'])} | "
                     f"{f2(st['gain_use'])} | {f2(st['yn'])} |")
        L.append("")
    # ---------------------------------------------------------------- direct vs use
    L += ["## Direct question vs use prompts: (R+subj − R-min) target gain, direct minus use", "",
          "Two-step lookup predicts a positive value (the subject entry helps more on a direct question). The use "
          "prompts' target gain is measured with the direct question's measure prefix forced after the use prompt "
          "(dream_bind has no use-prompt prefixes); descriptive.", "", "| split | (direct − use) [95% CI] | N |",
          "|---|---|---|"]
    V["direct_vs_use"] = {}
    for s in ("all", "primary", "pool"):
        st = ci(direct_minus_use(M, "R+subj", "R-min", facts_of[s]))
        V["direct_vs_use"][s] = st
        L.append(f"| {s} | {f2(st)} | {st['n']} |")
    L.append("")
    # ---------------------------------------------------------------- attention
    L += attention_section(rows, allf)
    # ---------------------------------------------------------------- N, skips, checks
    fm = meta["facts"]
    L += ["## N, skipped and excluded facts, checks", ""]
    for s in ("all", "primary", "pool", "attribute"):
        cnt = {c: sum(1 for f in facts_of[s] if c in M["answer"] and f in M["answer"][c]) for c in conds}
        L.append(f"- **{s}:** " + ", ".join(f"{c} {n}" for c, n in cnt.items()))
    sk = {f: v["skipped"] for f, v in fm.items() if v["skipped"]}
    L.append(f"- Skipped conditions (no 'is' before the value / no relation token): "
             f"{json.dumps(sk) if sk else 'none'}")
    L.append(f"- Excluded facts (no subject alias in the experience sentence): "
             f"{', '.join(e['fact'] for e in meta['excluded']) or 'none'}")
    L.append(f"- R-full == the sentence as a prior turn: max |Δlogit| over facts "
             f"{max(v['identity_max_abs_diff'] for v in fm.values()):.2e} (tolerance 0.01)")
    L.append("- Kept-entry counts, R-min == bind.splice_keep, subject < value, subject == ea_positions' entity: "
             "asserted per fact (the run would have failed).")
    multi = [f for f, v in fm.items() if v["n_value_tokens"] > 1]
    L.append(f"- Multi-token values (first sub-token + next kept): {', '.join(multi) or 'none'}")
    sub = {f: v["tok"]["subj"] for f, v in fm.items() if not v["tok"]["subj"].startswith(" ")}
    L.append(f"- Subjects whose last token is a word piece: {json.dumps(sub)}")
    L.append("")
    # ---------------------------------------------------------------- examples
    L += examples_section(rows, meta, M)
    (run / "report.md").write_text("\n".join(L) + "\n")
    json.dump(V, open(run / "verdict.json", "w"), indent=1)
    print(f"report: {run / 'report.md'}\nverdict: {label}")
    return V


def attention_section(rows, facts):
    out = ["## Attention to the restored subject entry (descriptive; eager attention, mean over heads, then facts)", ""]
    for c in ("R+subj", "R-subj2"):
        R = [r for r in rows if r["sec"] == "tf" and r["cond"] == c and r["pkind"] == "direct" and r["fact"] in facts
             and "attn" in r]
        if not R:
            continue
        for site, sname in (("last", "last prompt token"), ("qsubj", "the question's subject token")):
            Rs = [r for r in R if site in r["attn"]]
            if not Rs:
                continue
            ents = [e for e in ("subj", "value", "next", "head", "own") if e in Rs[0]["attn"][site]]
            arr = {e: np.array([r["attn"][site][e] for r in Rs]) for e in ents}   # [facts, layers]
            nl = arr["subj"].shape[1]
            out += [f"### {c}, query = {sname} ({len(Rs)} facts, direct question)", "",
                    "| layer | " + " | ".join(ents) + " | subj − value [95% CI over facts] |",
                    "|---|" + "---|" * (len(ents) + 1)]
            for l in range(nl):
                d = {f"f{i}": {0: float(arr['subj'][i, l] - arr['value'][i, l])} for i in range(len(Rs))}
                s = ci(d)
                out.append(f"| {l} | " + " | ".join(f"{arr[e][:, l].mean():.3f}" for e in ents) + f" | {f2(s, 3)} |")
            out.append("")
    return out


def examples_section(rows, meta, M):
    out = ["## Example facts", ""]
    greedy = {(r["fact"], r["cond"]): r["answer"] for r in rows if r["sec"] == "greedy"}
    top = {(r["fact"], r["cond"]): r["top5_first"] for r in rows if r["sec"] == "tf" and r["pkind"] == "direct"}
    for f in meta["examples"]:
        if f not in meta["facts"]:
            continue
        fm = meta["facts"][f]
        out += [f"### {f}: \"{fm['experience']}\" Q: \"{fm['question']}\"", "",
                f"Kept tokens: subj {fm['tok']['subj']!r}, rel {fm['tok']['rel']!r}, fill {fm['tok']['fill']!r}, "
                f"value {fm['tok']['value']!r}, next {fm['tok']['next']!r}; T-subj3 text {fm['t_subj3']!r}", "",
                "| condition | answer score | target gain | TF yes/no | top first tokens | greedy answer (40 tokens, read only) |",
                "|---|---|---|---|---|---|"]
        for c in meta["conds"]:
            if f not in M["answer"].get(c, {}):
                continue
            tp = ", ".join(f"{t!r} {p:.2f}" for t, p in top.get((f, c), [])[:3])
            g = greedy.get((f, c), "").replace("\n", " ").replace("|", "/")
            out.append(f"| {c} | {M['answer'][c][f][0]:.2f} | {M['gain'][c][f][0]:.2f} | {M['yn'][c][f][0]:.2f} | "
                       f"{tp} | {g} |")
        out.append("")
    return out


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
