#!/usr/bin/env python3
"""Build the Seahorse manuscript page (vedantjumle.com/seahorse/) from the result files.

Every chart and every quoted number in the generated figures is computed here from the run
outputs in results/, so the page can be rebuilt whenever a run changes. The prose lives in
page.html next to this file; {{NAME}} placeholders are filled with the figures below.

    python tools/manuscript/build.py --results results \
        --out ../personal-website/seahorse/index.html

Runs used: think_v1_20261003_0004 (Qwen3.5-2B, layers / thinking phase) and
ref_v1_20261005_2347 (Qwen3.5-2B, five shift references). Only the standard library.
"""

import argparse
import csv
import gzip
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
THINK = "think_v1_20261003_0004"
REF = "ref_v1_20261005_2347/main"


# ------------------------------------------------------------------ loading


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def texts(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def truthy(v):
    return v is True or v == "True"


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------------ data


def ref_summary(res):
    """(item, cond) -> summary row of ref_v1."""
    rows = list(csv.DictReader(open(res / REF / "summary.csv")))
    return {(r["item"], r["cond"]): r for r in rows}


def unrelated_identical(summ):
    """Unrelated greedy answers identical to no memory, over every memory condition."""
    same = total = 0
    for (item, cond), r in summ.items():
        if cond in ("nomem", "ctx") or not r["unrel_n"]:
            continue
        n = int(num(r["unrel_n"]))
        same += round(num(r["unrel_same"]) * n)
        total += n
    return same, total


TARGET = {"dog_name": "pepper", "sister_name": "petra", "job": "pharmacist", "favourite_colour": "teal"}
DENY = re.compile(r"(don't|do not|not|no) (know|have|access)|i am an ai|i'm an ai")


def fact_recount(res):
    """think_v1 stage 3, facts, isolated memories, the direct question ("What's my dog's name?").
    Splits answers that name the fact into clean / named-while-refusing / loop."""
    base = res / THINK / "s3_fact"
    T = {(t["mem"], t["cond"], t["item"], t["kind"], t["prompt"], str(t["row"])): t
         for t in texts(base / "texts.jsonl.gz")}
    out = defaultdict(lambda: defaultdict(int))
    for x in jsonl(base / "results.jsonl"):
        if x["mem"] != "iso" or x["kind"] != "related":
            continue
        t = T[(x["mem"], x["cond"], x["item"], x["kind"], x["prompt"], str(x["row"]))]
        ans = (t["answer"] or "").lower()
        c = out[x["cond"]]
        c["n"] += 1
        c["think"] += truthy(x["think_mem"])
        if not truthy(x["ans_target"]):
            continue
        loop = len(re.findall(TARGET[x["item"]], ans)) >= 4 or num(x["rep_ans"]) >= 0.3
        if loop:
            c["loop"] += 1
        elif DENY.search(ans):
            c["hedged"] += 1
        else:
            c["clean"] += 1
    return out


MEAT = re.compile(r"\b(chicken|beef|pork|steak|salmon|shrimp|bacon|turkey|fish|lamb|sausage|meat|tuna|ham)\b", re.I)


def vegetarian_rates(res):
    """think_v1 stage 3, the vegetarian memory: answers that fit the preference, answers mentioning meat."""
    base = res / THINK / "s3_disp"
    T = {(t["mem"], t["cond"], t["item"], t["kind"], t["prompt"], str(t["row"])): t
         for t in texts(base / "texts.jsonl.gz")}
    out = defaultdict(lambda: defaultdict(int))
    for x in jsonl(base / "results.jsonl"):
        if x["mem"] != "iso" or x["item"] != "vegetarian" or x["kind"] not in ("related", "ambiguous"):
            continue
        t = T[(x["mem"], x["cond"], x["item"], x["kind"], x["prompt"], str(x["row"]))]
        c = out[x["cond"]]
        c["n"] += 1
        c["fit"] += x["ans_label"] == "consistent"
        c["meat"] += bool(MEAT.search(t["answer"] or ""))
    return out


def layer_sweep(res):
    rows = list(csv.DictReader(open(res / THINK / "s1" / "summary.csv")))
    rows = [r for r in rows if r["key"] == "pooled_w256" and r["mode"] == "isolated"]
    rows.sort(key=lambda r: int(r["layer"]))
    pts = [{"layer": int(r["layer"]), "ltype": r["ltype"], "spec": num(r["spec_nx"]),
            "acc": num(r["acc_bal"]), "si": num(r["si"])} for r in rows]
    r0 = rows[0]
    ceil = {"spec": num(r0["spec_ceil_nx"]),
            "acc": (num(r0["acc_ceil_Yes"]) + num(r0["acc_ceil_No"])) / 2,
            "base": (num(r0["acc_base_Yes"]) + num(r0["acc_base_No"])) / 2}
    return pts, ceil


def fact_balanced(res):
    """Per stage-3 condition (isolated memories, fact items): balanced yes/no accuracy, and the share of
    thinking traces cut off at the token cap."""
    rows = [r for r in csv.DictReader(open(res / THINK / "s3_fact" / "summary.csv")) if r["mem"] == "iso"]
    return {r["cond"]: num(r["f_acc_bal"]) for r in rows}, {r["cond"]: num(r["forced"]) for r in rows}


PREFIX_TARGET = {"dog_name": "pepper", "sister_name": "petra", "job": "pharmacist", "favourite_colour": "teal"}


def prefix_counts(res):
    """think_v1 stage 2: short continuations after a fact prompt. Share of samples naming the fact once
    or twice (clean) vs three or more times (a loop), for the 3-layer memory at split and full strength."""
    out = {}
    for t in texts(res / THINK / "s2" / "texts.jsonl.gz"):
        if t["section"] != "prefix" or t["scenario"] not in PREFIX_TARGET:
            continue
        samples = json.loads(t["samples"]) if isinstance(t["samples"], str) else t["samples"]
        c = out.setdefault(t["condition"], {"n": 0, "clean": 0, "loop": 0})
        for s in samples:
            k = len(re.findall(PREFIX_TARGET[t["scenario"]], s.lower()))
            c["n"] += 1
            c["clean"] += k in (1, 2)
            c["loop"] += k >= 3
    return out


def collapse(s):
    """Shorten degenerate repetition so a loop stays readable: 'jazz jazz jazz ...' -> 'jazz [x143]'."""
    def one(m):
        w = m.group(1)
        n = len(re.findall(r"\b" + re.escape(w) + r"\b", m.group(0), re.I))
        return f"{w} [×{n}] "
    s = re.sub(r"\b([\w'’-]+)\b(?:[\s.,!*🇳-🇿]+\1\b){3,}[\s.,!*]*", one, s, flags=re.I)

    def two(m):
        p = m.group(1)
        n = m.group(0).lower().count(p.lower())
        return f"{p} [×{n}] "
    return re.sub(r"\b([\w'’-]+ [\w'’-]+)\b(?:\s+\1\b){3,}\s*", two, s, flags=re.I)


VIEW_ITEMS = ["vegetarian", "no_alcohol", "loves_jazz", "hates_jazz", "norway"]
VIEW_TITLES = {"vegetarian": "vegetarian", "no_alcohol": "doesn't drink", "loves_jazz": "loves jazz",
               "hates_jazz": "can't stand jazz", "norway": "lives in Norway"}
VIEW_CONDS = ["nomem", "ctx"] + [f"{r}@{d}" for r in ("opposite", "without", "hum", "disclosure", "centroid")
                                 for d in (1, 2)]


def viewer_data(res, summ):
    base = res / REF
    cfg = json.load(open(base / "config.json"))
    items = {x["id"]: x for x in cfg["items"]}
    T = {(t["item"], t["cond"], t["kind"], t["prompt"], str(t["row"])): t["answer"]
         for t in texts(base / "texts.jsonl.gz")}
    rec = {}
    for x in jsonl(base / "results.jsonl"):
        if x.get("section") == "gen" and x.get("kind") == "related":
            rec[(x["item"], x["cond"], str(x["row"]))] = x
    out = {}
    for it in VIEW_ITEMS:
        x = items[it]
        disc = x["disclosure"]
        disc = json.loads(disc) if isinstance(disc, str) else disc
        cells = {}
        for cond in VIEW_CONDS:
            s = summ[(it, cond)]
            answers = []
            for row in range(4):
                r = rec.get((it, cond, str(row)))
                a = T.get((it, cond, "related", x["related"], str(row)))
                if r is None or a is None:
                    continue
                a = collapse(re.sub(r"\s+", " ", a)).strip()
                if len(a) > 760:
                    a = a[:760].rsplit(" ", 1)[0] + " …"
                answers.append({"text": a, "label": r["label"], "loop": truthy(r["loop"]),
                                "greedy": truthy(r["greedy"])})
            lc = num(s["lean_clean"])
            cells[cond] = {"lean_clean": None if math.isnan(lc) else round(lc, 2),
                           "lean": round(num(s["lean"]), 2), "loop": round(num(s["loop"]), 2),
                           "n": int(num(s["n"])), "answers": answers}
        out[it] = {"title": VIEW_TITLES[it], "tag": x["tag"], "experience": x["experience"],
                   "counter": x["counter"], "centroid": x["centroid"],
                   "disclosure": [d["experience"] for d in disc], "prompt": x["related"], "cells": cells}
    return out


# ------------------------------------------------------------------ svg helpers


def esc(s):
    return html.escape(str(s), quote=True)


def fmt_signed(v):
    return f"{v:+.2f}".replace("-", "−")


def pct(v):
    return f"{round(100 * v)}%"


def svg_open(w, h, label, cls="chart"):
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{esc(label)}">')


def table(headers, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="data"><summary>Data table</summary><div class="table-wrap">'
            f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div></details>')


# ------------------------------------------------------------------ figures


REF_ROWS = [("nomem", "no memory"), ("ctx", "in context"), ("opposite@1", "opposite"),
            ("without@1", "without"), ("hum@1", "hum"), ("disclosure@1", "disclosure"),
            ("centroid@1", "centroid")]


def fig_negation(summ):
    """Lean (loops removed) under each reference, three memories side by side."""
    panels = [("loves_jazz", "“I’m a huge jazz fan.”", "+ = toward jazz"),
              ("hates_jazz", "“I can’t stand jazz music.”", "+ = away from jazz"),
              ("no_alcohol", "“I don’t drink alcohol.”", "+ = away from alcohol")]
    W, H, lab, x0, x1, top, rh = 340, 262, 80, 90, 232, 58, 27
    xv = 284  # value column (right-aligned)
    X = lambda v: x0 + (v + 1) / 2 * (x1 - x0)
    parts, data = [], []
    for item, quote, sub in panels:
        s = [svg_open(W, H, f"Lean of answers under each reference for the memory {quote}")]
        s.append(f'<text class="c-title" x="0" y="16">{esc(quote)}</text>')
        s.append(f'<text class="c-sub" x="0" y="33">{esc(sub)}</text>')
        s.append(f'<text class="c-sub" x="{xv}" y="33" text-anchor="end">lean</text>')
        s.append(f'<text class="c-sub" x="{W}" y="33" text-anchor="end">loops</text>')
        yb = top + len(REF_ROWS) * rh - 8
        for v in (-1, -0.5, 0.5, 1):
            s.append(f'<line class="c-grid" x1="{X(v):.1f}" y1="{top - 10}" x2="{X(v):.1f}" y2="{yb}"/>')
        s.append(f'<line class="c-axis" x1="{X(0):.1f}" y1="{top - 10}" x2="{X(0):.1f}" y2="{yb}"/>')
        for v, t in ((-1, "−1"), (0, "0"), (1, "+1")):
            s.append(f'<text class="c-tick" x="{X(v):.1f}" y="{yb + 16}" text-anchor="middle">{t}</text>')
        s.append(f'<line class="c-sep" x1="0" y1="{top + 2 * rh - 13}" x2="{W}" y2="{top + 2 * rh - 13}"/>')
        for i, (cond, name) in enumerate(REF_ROWS):
            r = summ[(item, cond)]
            v, loop, n = num(r["lean_clean"]), num(r["loop"]), int(num(r["n"]))
            y = top + i * rh
            s.append(f'<text class="c-label" x="{lab}" y="{y + 4}" text-anchor="end">{esc(name)}</text>')
            s.append(f'<text class="c-tick" x="{W}" y="{y + 4}" text-anchor="end">{pct(loop)}</text>')
            data.append([quote, name, "all loops" if math.isnan(v) else fmt_signed(v), pct(loop), n])
            if math.isnan(v):
                s.append(f'<text class="c-tick" x="{xv}" y="{y + 4}" text-anchor="end">n/a</text>')
                continue
            cls = "pos" if v > 0.1 else "neg" if v < -0.1 else "neu"
            tip = f"{name}: lean {fmt_signed(v)} over non-loop answers · {pct(loop)} of {n} answers loop"
            s.append(f'<line class="c-stem {cls}" x1="{X(0):.1f}" y1="{y}" x2="{X(v):.1f}" y2="{y}"/>')
            s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}">'
                     f'<rect x="{min(X(0), X(v)) - 8:.1f}" y="{y - 11}" width="{abs(X(v) - X(0)) + 16:.1f}" '
                     f'height="22" fill="transparent"/>'
                     f'<circle class="c-dot {cls}" cx="{X(v):.1f}" cy="{y}" r="5.5"/></g>')
            s.append(f'<text class="c-val" x="{xv}" y="{y + 4}" text-anchor="end">{fmt_signed(v)}</text>')
        s.append("</svg>")
        parts.append("".join(s))
    return ('<div class="panels">' + "".join(f'<div class="panel">{p}</div>' for p in parts) + "</div>"
            + table(["memory", "reference", "lean (loops removed)", "loops", "answers"], data))


FACT_ROWS = [("nomem", "no memory", "0 / 0"), ("answer_only", "while answering", "0 / 2"),
             ("both", "thinking + answering", "2 / 2"), ("think4", "amplified, thinking only", "4 / 0"),
             ("think6", "amplified, thinking only", "6 / 0"), ("think4_ans1", "amplified + weak answer", "4 / 1"),
             ("think6_ans1", "amplified + weak answer", "6 / 1"), ("ctx", "in context", "")]


def fig_facts(rc):
    W, H, lab, x0, x1, top, rh, bh = 700, 300, 226, 236, 610, 34, 30, 14
    X = lambda f: x0 + f * (x1 - x0)
    s = [svg_open(W, H, "Answers naming the stored fact, split into clean, refusing and looping answers")]
    s.append(f'<text class="c-sub" x="{lab}" y="14" text-anchor="end">memory strength (thinking / answer)</text>')
    s.append(f'<text class="c-sub" x="{W}" y="14" text-anchor="end">named in answer</text>')
    yb = top + len(FACT_ROWS) * rh - 6
    for f in (0, 0.25, 0.5, 0.75, 1):
        s.append(f'<line class="{"c-axis" if f == 0 else "c-grid"}" x1="{X(f):.1f}" y1="{top - 12}" '
                 f'x2="{X(f):.1f}" y2="{yb}"/>')
        s.append(f'<text class="c-tick" x="{X(f):.1f}" y="{yb + 16}" text-anchor="middle">{pct(f)}</text>')
    data = []
    for i, (cond, name, dose) in enumerate(FACT_ROWS):
        c = rc[cond]
        n = c["n"]
        y = top + i * rh
        lbl = f"{name} · {dose}" if dose else name
        s.append(f'<text class="c-label" x="{lab}" y="{y + 4}" text-anchor="end">{esc(lbl)}</text>')
        x = X(0)
        for key, cls, word in (("clean", "s1", "clean"), ("hedged", "s3", "named while refusing"),
                               ("loop", "s2", "inside a loop")):
            w = c[key] / n * (x1 - x0)
            if w <= 0:
                continue
            tip = f"{lbl}: {c[key]} of {n} answers name the fact — {word}"
            s.append(f'<rect class="seg {cls}" tabindex="0" data-tip="{esc(tip)}" x="{x:.1f}" y="{y - bh / 2}" '
                     f'width="{max(w - 2, 1):.1f}" height="{bh}" rx="2"/>')
            x += w
        named = c["clean"] + c["hedged"] + c["loop"]
        s.append(f'<text class="c-tick" x="{W}" y="{y + 4}" text-anchor="end">{named} / {n}</text>')
        tf = c["think"] / n
        if tf > 0:
            tip = f"{lbl}: the thinking trace mentions the fact in {c['think']} of {n}"
            s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}"><circle cx="{X(tf):.1f}" cy="{y}" '
                     f'r="11" fill="transparent"/><circle class="c-ring" cx="{X(tf):.1f}" cy="{y}" r="5"/></g>')
        data.append([lbl, n, c["clean"], c["hedged"], c["loop"], c["think"]])
    s.append("</svg>")
    legend = ('<div class="legend"><span><i class="sw s1"></i>clean answer</span>'
              '<span><i class="sw s3"></i>named while refusing</span>'
              '<span><i class="sw s2"></i>inside a loop</span>'
              '<span><i class="ring"></i>fact in the thinking trace</span></div>')
    return (legend + "".join(s) + table(
        ["condition", "answers", "clean", "named while refusing", "inside a loop", "thinking mentions it"], data))


def fig_layers(pts, ceil):
    def panel(key, title, ymin, ymax, yticks, fmt, ceil_v, ceil_lbl, extra=None):
        W, H, l, r, t, b = 440, 236, 40, 14, 40, 34
        X = lambda L: l + (L - 4) / 19 * (W - l - r)
        Y = lambda v: H - b - (v - ymin) / (ymax - ymin) * (H - t - b)
        s = [svg_open(W, H, title)]
        s.append(f'<text class="c-title" x="0" y="16">{esc(title)}</text>')
        for v in yticks:
            s.append(f'<line class="{"c-axis" if v == 0 else "c-grid"}" x1="{l}" y1="{Y(v):.1f}" '
                     f'x2="{W - r}" y2="{Y(v):.1f}"/>')
            s.append(f'<text class="c-tick" x="{l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{fmt(v)}</text>')
        for L in (4, 8, 12, 16, 20, 23):
            s.append(f'<text class="c-tick" x="{X(L):.1f}" y="{H - b + 16}" text-anchor="middle">{L}</text>')
        s.append(f'<text class="c-sub" x="{W - r}" y="{H - 2}" text-anchor="end">layer (of 24)</text>')
        s.append(f'<line class="c-mark" x1="{X(16):.1f}" y1="{t - 6}" x2="{X(16):.1f}" y2="{H - b}"/>')
        s.append(f'<text class="c-sub" x="{X(16) - 4:.1f}" y="{t - 10}" text-anchor="end">≈ ⅔ depth</text>')
        s.append(f'<line class="c-ref" x1="{l}" y1="{Y(ceil_v):.1f}" x2="{W - r}" y2="{Y(ceil_v):.1f}"/>')
        s.append(f'<text class="c-sub" x="{l + 4}" y="{Y(ceil_v) - 5:.1f}">{esc(ceil_lbl)}</text>')
        if extra:
            s.append(extra(X, Y, l, W - r))
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(p['layer']):.1f},{Y(p[key]):.1f}" for i, p in enumerate(pts))
        s.append(f'<path class="c-line s1" d="{d}"/>')
        for p in pts:
            used = p["layer"] in (20, 21, 23)
            tip = f"layer {p['layer']} ({p['ltype']} attention): {title.lower()} {fmt(p[key])}"
            s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}"><circle cx="{X(p["layer"]):.1f}" '
                     f'cy="{Y(p[key]):.1f}" r="10" fill="transparent"/><circle class="c-dot s1{" big" if used else ""}" '
                     f'cx="{X(p["layer"]):.1f}" cy="{Y(p[key]):.1f}" r="{5.5 if used else 4}"/></g>')
        s.append("</svg>")
        return "".join(s)

    chance = lambda X, Y, a, b: (f'<line class="c-ref" x1="{a}" y1="{Y(0.5):.1f}" x2="{b}" y2="{Y(0.5):.1f}"/>'
                                 f'<text class="c-sub" x="{a + 4}" y="{Y(0.5) + 15:.1f}">chance</text>')
    p1 = panel("spec", "Fact specificity (nats)", -1, 14, (0, 4, 8, 12),
               lambda v: f"{v:.0f}" if v == int(v) else f"{v:.1f}", ceil["spec"], f"in context {ceil['spec']:.1f}")
    p2 = panel("acc", "Balanced yes/no accuracy", 0, 1, (0, 0.25, 0.5, 0.75, 1), lambda v: f"{v:.2f}",
               ceil["acc"], f"in context {ceil['acc']:.2f}", chance)
    data = [[p["layer"], p["ltype"], f"{p['spec']:.2f}", f"{p['acc']:.2f}", f"{p['si']:.1f}"] for p in pts]
    return ('<div class="panels two">' + f'<div class="panel">{p1}</div><div class="panel">{p2}</div></div>'
            + table(["layer", "attention", "fact specificity (nats)", "balanced yes/no accuracy",
                     "selectivity index"], data))


def fig_vegetarian(vr):
    rows = [("nomem", "no memory"), ("answer_only", "memory"), ("ctx", "in context")]

    def panel(key, title):
        W, H, lab, x0, x1, top, rh = 330, 132, 84, 92, 290, 44, 28
        X = lambda f: x0 + f * (x1 - x0)
        s = [svg_open(W, H, title), f'<text class="c-title" x="0" y="16">{esc(title)}</text>']
        s.append(f'<line class="c-axis" x1="{x0}" y1="{top - 12}" x2="{x0}" y2="{top + 3 * rh - 12}"/>')
        for i, (cond, name) in enumerate(rows):
            c = vr[cond]
            f = c[key] / c["n"]
            y = top + i * rh
            s.append(f'<text class="c-label" x="{lab}" y="{y + 4}" text-anchor="end">{name}</text>')
            tip = f"{name}: {c[key]} of {c['n']} answers"
            s.append(f'<rect class="seg s1" tabindex="0" data-tip="{esc(tip)}" x="{x0}" y="{y - 7}" '
                     f'width="{max(X(f) - x0, 1):.1f}" height="14" rx="2"/>')
            s.append(f'<text class="c-val" x="{X(f) + 8:.1f}" y="{y + 4}">{pct(f)}</text>')
        s.append("</svg>")
        return "".join(s)

    data = [[name, vr[c]["n"], vr[c]["fit"], vr[c]["meat"]] for c, name in rows]
    return ('<div class="panels two">'
            f'<div class="panel">{panel("fit", "Answers that fit “vegetarian”")}</div>'
            f'<div class="panel">{panel("meat", "Answers that mention meat or fish")}</div></div>'
            + table(["condition", "answers", "fit the preference", "mention meat or fish"], data))


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE.parents[1] / "results")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = a.results

    summ = ref_summary(res)
    same, total = unrelated_identical(summ)
    rc = fact_recount(res)
    vr = vegetarian_rates(res)
    pts, ceil = layer_sweep(res)
    view = viewer_data(res, summ)
    bal, forced = fact_balanced(res)
    pc = prefix_counts(res)
    split, full = pc["L23+L20+L21 split|iso"], pc["L23+L20+L21 full|iso"]
    mem_bal = [v for k, v in bal.items() if k not in ("nomem", "ctx")]

    late = [p["spec"] for p in pts if p["layer"] >= 19]
    early = [p["spec"] for p in pts if p["layer"] <= 16]
    accs = [p["acc"] for p in pts]
    fill = {
        "FIG_NEGATION": fig_negation(summ),
        "FIG_FACTS": fig_facts(rc),
        "FIG_LAYERS": fig_layers(pts, ceil),
        "FIG_VEGETARIAN": fig_vegetarian(vr),
        "N_UNREL_SAME": str(same), "N_UNREL_TOTAL": str(total),
        "VEG_NOMEM": pct(vr["nomem"]["fit"] / vr["nomem"]["n"]),
        "VEG_MEM": pct(vr["answer_only"]["fit"] / vr["answer_only"]["n"]),
        "VEG_CTX": pct(vr["ctx"]["fit"] / vr["ctx"]["n"]),
        "MEAT_NOMEM": pct(vr["nomem"]["meat"] / vr["nomem"]["n"]),
        "MEAT_MEM": pct(vr["answer_only"]["meat"] / vr["answer_only"]["n"]),
        "MEAT_CTX": pct(vr["ctx"]["meat"] / vr["ctx"]["n"]),
        "FACT_MEM_NAMED": str(rc["answer_only"]["clean"] + rc["answer_only"]["hedged"] + rc["answer_only"]["loop"]),
        "FACT_MEM_CLEAN": str(rc["answer_only"]["clean"]),
        "FACT_MEM_LOOP": str(rc["answer_only"]["loop"]),
        "FACT_CTX_CLEAN": str(rc["ctx"]["clean"]),
        "FACT_N": str(rc["answer_only"]["n"]),
        "SPEC_LATE": f"{min(late):.1f}–{max(late):.1f}", "SPEC_EARLY": f"{max(early):.1f}",
        "SPEC_CEIL": f"{ceil['spec']:.1f}",
        "ACC_MIN": f"{min(accs):.2f}", "ACC_MAX": f"{max(accs):.2f}", "ACC_CEIL": f"{ceil['acc']:.2f}",
        "BAL_MIN": f"{min(mem_bal):.2f}", "BAL_MAX": f"{max(mem_bal):.2f}", "BAL_CTX": f"{bal['ctx']:.2f}",
        "PFX_FULL_CLEAN": pct(full["clean"] / full["n"]), "PFX_FULL_LOOP": pct(full["loop"] / full["n"]),
        "PFX_SPLIT_CLEAN": pct(split["clean"] / split["n"]), "PFX_SPLIT_LOOP": pct(split["loop"] / split["n"]),
        "THINK_FORCED_NOMEM": pct(forced["nomem"]),
        "VIEWER_DATA": json.dumps(view, ensure_ascii=False).replace("</", "<\\/"),
    }
    page = (HERE / "page.html").read_text()
    missing = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", page)) - set(fill))
    if missing:
        raise SystemExit(f"page.html uses unknown placeholders: {missing}")
    for k, v in fill.items():
        page = page.replace("{{" + k + "}}", v)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page)
    print(f"wrote {a.out} ({len(page) / 1024:.0f} KB)")
    for k in ("N_UNREL_SAME", "N_UNREL_TOTAL", "VEG_NOMEM", "VEG_MEM", "VEG_CTX", "MEAT_NOMEM", "MEAT_MEM",
              "MEAT_CTX", "FACT_MEM_NAMED", "FACT_MEM_CLEAN", "FACT_MEM_LOOP", "FACT_CTX_CLEAN", "FACT_N",
              "SPEC_LATE", "SPEC_EARLY", "SPEC_CEIL", "ACC_MIN", "ACC_MAX", "ACC_CEIL", "BAL_MIN", "BAL_MAX",
              "BAL_CTX", "PFX_FULL_CLEAN", "PFX_FULL_LOOP", "PFX_SPLIT_CLEAN", "PFX_SPLIT_LOOP",
              "THINK_FORCED_NOMEM"):
        print(f"  {k} = {fill[k]}")


if __name__ == "__main__":
    main()
