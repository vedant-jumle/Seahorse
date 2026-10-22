#!/usr/bin/env python3
"""Build the Seahorse project page (vedantjumle.com/seahorse/) from the result files.

Every chart and every number tied to a chart or a claim is computed here from the run outputs in
results/, so the page can be rebuilt whenever a run changes. The prose lives in page.html next to this
file; {{NAME}} placeholders are filled with the figures and numbers below. A few numbers exist only in a
run's report.md (the generation reports of dream_bind, the G-key retrieval report, the S3 table of
dream_refs phase A); those are parsed from the report's tables, never retyped (see `md_tables`).

    python3 tools/manuscript/build.py --results results \
        --out ../personal-website/seahorse/index.html

Runs used (all local, under results/):
  core_v1_20261007_1820           paper 1: the pushed vector on Qwen3.5-2B and 9B (claims C1-C6, judge panel)
  dream_bind_stage0_20261011_2321 where the answer reads a fact (activation patching, both models)
  dream_bind_gen_20261012_0038    stored slices on Qwen2.5-1.5B (R-full, R-min, R-orphan, the gate)
  dream_bind_state2b_20261012_1759 stored slices on the hybrid Qwen3.5-2B (with and without the recurrent state)
  dream_bind_add5_20261013_1255   the same tokens as text (T-min), and two-fact collisions
  dream_bind_add6_20261014_1608   binding-ID repair, retrieval by key (59 facts), the recurrent-state split
  dream_refs_20261015_1545        model-written references on 29 items
  dream_refs_pe_20261015_1603     model-written references on PrefEval (200 items)
Only the standard library.
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
CORE = "core_v1_20261007_1820"
STAGE0 = "dream_bind_stage0_20261011_2321"
GEN25 = "dream_bind_gen_20261012_0038/qwen25_15b"
STATE2B = "dream_bind_state2b_20261012_1759/qwen35_2b"
ADD5 = "dream_bind_add5_20261013_1255"
ADD6 = "dream_bind_add6_20261014_1608"
REFS = "dream_refs_20261015_1545"
PE = "dream_refs_pe_20261015_1603"
RUNS = [CORE, STAGE0, GEN25.split("/")[0], STATE2B.split("/")[0], ADD5, ADD6, REFS, PE]
NAN = float("nan")

# Qwen3.5-2B architecture (model config): 18 Gated DeltaNet blocks, 16 heads, a 128 x 128 state per head.
DELTANET_STATE = (18, 16, 128, 128)


# ------------------------------------------------------------------ loading


def jload(path):
    with open(path) as f:
        return json.load(f)


def csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def gz_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


# ------------------------------------------------------------------ report.md parsing
# Used only where a number exists in a report and in no json/csv. Tables and bullet lists are read as
# written by the run's own analysis script.

CI_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*\[\s*([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\s*\]")
NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def md_section(text, heading):
    i = text.find(heading)
    if i < 0:
        raise SystemExit(f"report section not found: {heading!r}")
    j = text.find("\n## ", i + len(heading))
    return text[i:j if j >= 0 else len(text)]


def md_tables(sec):
    """Every markdown table in a section, as a list of row dicts keyed by the header."""
    blocks, cur = [], []
    for line in sec.splitlines():
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    out = []
    for b in blocks:
        head = [c.strip() for c in b[0].strip().strip("|").split("|")]
        out.append([dict(zip(head, (c.strip() for c in ln.strip().strip("|").split("|")))) for ln in b[2:]])
    return out


def md_bullets(sec):
    """'- name: value …' lines (a name may itself contain a colon, e.g. 'R-full:ungated')."""
    out = {}
    for line in sec.splitlines():
        m = re.match(r"- (.+?): ([+-]?\d.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def ci(s):
    """'0.86 [0.63, 1.00]' -> (0.86, 0.63, 1.00); a bare number -> (v, nan, nan)."""
    m = CI_RE.search(s)
    if m:
        return tuple(float(x) for x in m.groups())
    return (float(NUM_RE.search(s).group()), NAN, NAN)


def first_num(s):
    return float(NUM_RE.search(s).group())


def gen_report(path):
    """A dream_bind generation report: use-prompt table, direct-question bullets, unrelated prompts."""
    text = Path(path).read_text()
    sec = md_section(text, "## Use prompts (primary)")
    tab = {r["condition"]: r for r in md_tables(sec)[0]}
    out = {"clean": {c: ci(r["clean (majority)"]) for c, r in tab.items()},
           "self": {c: ci(r["self-claim"]) for c, r in tab.items()},
           "confab": {c: ci(r["confab (wrong value)"]) for c, r in tab.items()},
           "bal": {c: ci(r["bal yes/no acc"]) for c, r in tab.items()},
           "direct": {c: ci(v) for c, v in md_bullets(sec).items()}}
    if "## Unrelated prompts" in text:
        out["unrel"] = {c: first_num(v) for c, v in md_bullets(md_section(text, "## Unrelated prompts")).items()}
    return out


# ------------------------------------------------------------------ data: paper 1 (core_v1)


def core(res):
    claims = jload(res / CORE / "analysis" / "claims_all.json")
    pref, facts, yn = {}, {}, {}
    for m in ("qwen35_2b", "qwen35_9b"):
        pref[m] = {(r["group"], r["cond"]): r for r in csv_rows(res / CORE / m / "analysis" / "pref.csv")}
        facts[m] = {r["cond"]: r for r in csv_rows(res / CORE / m / "analysis" / "facts.csv")}
        yn[m] = {(r["group"], r["cond"]): r for r in csv_rows(res / CORE / m / "analysis" / "yn.csv")}
    panel = jload(res / CORE / "panel" / "panel_claims.json")
    return claims, pref, facts, yn, panel


# ------------------------------------------------------------------ data: stage 0


def stage0(res):
    """Share of the Pepper-vs-Rocky preference moved by patching one block at one position (primary items)."""
    out = {}
    for m in ("qwen25_15b", "qwen35_2b"):
        acc = defaultdict(list)
        for r in gz_rows(res / STAGE0 / m / "patch" / "results.jsonl.gz"):
            if r["kind"] == "block" and r["tier"] == "primary" and r["site"] in ("subj_span", "attr"):
                acc[(r["site"], r["start"])].append(r["share"])
        n_layers = max(L for _, L in acc) + 1
        out[m] = {s: [sum(acc[(s, L)]) / len(acc[(s, L)]) for L in range(n_layers)] for s in ("subj_span", "attr")}
        out[m]["n_items"] = len(acc[("attr", 0)])
        out[m]["verdict"] = jload(res / STAGE0 / m / "verdict.json")
    return out


# ------------------------------------------------------------------ data: viewer of real answers (2B)

VIEW_ITEMS = ["dog_name", "cat_name", "sister_name", "brother_name", "grandmother_name", "best_friend_name",
              "boss_name"]
# (key, source run, condition in that run, button label)
VIEW_CONDS = [
    ("nomem", STATE2B, "nomem", "no memory"),
    ("push1", CORE + "/qwen35_2b", "mem@1", "push, α = 1"),
    ("push2", CORE + "/qwen35_2b", "mem@2", "push, α = 2"),
    ("tmin", ADD5 + "/qwen35_2b", "T-min", "same tokens as text"),
    ("kv6", STATE2B, "R-min-kv6", "slice, attention only"),
    ("state", STATE2B, "R-state-only", "recurrent state only"),
    ("minstate", STATE2B, "R-min-state", "slice + state"),
    ("orphan", STATE2B, "R-orphan-state", "stored after another chat"),
    ("swap", STATE2B, "R-swap-min-state", "another fact’s slice"),
    ("ctx", STATE2B, "ctx", "in context"),
]


def collapse(s):
    """Shorten degenerate repetition so a loop stays readable: 'Pepper Pepper Pepper ...' -> 'Pepper [×40]'."""
    def one(m):
        w = m.group(1)
        n = len(re.findall(r"\b" + re.escape(w) + r"\b", m.group(0), re.I))
        return f"{w} [×{n}] "
    s = re.sub(r"\b([\w'’-]+)\b(?:[\s.,!*]+\1\b){3,}[\s.,!*]*", one, s, flags=re.I)

    def two(m):
        p = m.group(1)
        n = m.group(0).lower().count(p.lower())
        return f"{p} [×{n}] "
    return re.sub(r"\b([\w'’-]+ [\w'’-]+)\b(?:\s+\1\b){3,}\s*", two, s, flags=re.I)


def clean_text(a, cap=700):
    a = collapse(re.sub(r"\s+", " ", (a or "").replace("**", ""))).strip()
    return a if len(a) <= cap else a[:cap].rsplit(" ", 1)[0] + " …"


def viewer_data(res):
    items = jload(res / STATE2B / "prep" / "items.json")
    rows = {}
    for key, run, cond, _ in VIEW_CONDS:
        for it in VIEW_ITEMS:
            for r in gz_rows(res / run / "gen" / "items" / f"{it}.jsonl.gz"):
                if r.get("cond") != cond:
                    continue
                if r.get("sec") == "yn":
                    rows.setdefault((key, it, "yn"), []).append(r)
                elif str(r.get("j")) == "0" and str(r.get("seedset")) == "1":  # the greedy answer
                    k = (key, it, r.get("kind"))  # the first prompt of each kind ("related" = direct question)
                    if k not in rows or int(r["pi"]) < int(rows[k]["pi"]):
                        rows[k] = r
    out = {"items": {}, "conds": [[k, label] for k, _, _, label in VIEW_CONDS]}
    for it in VIEW_ITEMS:
        x = items[it]
        cells = {}
        for key, *_ in VIEW_CONDS:
            d, u = rows.get((key, it, "related")), rows.get((key, it, "use"))
            if d is None:
                continue
            yn = rows.get((key, it, "yn"), [])
            pick = [next((r for r in yn if r["consistent"] == c), None) for c in ("Yes", "No")]
            cells[key] = {
                "direct": {"q": d["prompt"], "a": clean_text(d["answer"])},
                "use": {"q": u["prompt"], "a": clean_text(u["answer"])} if u else None,
                "yn": [{"q": r["prompt"].replace(" Answer yes or no.", ""), "a": (r["answer"] or "").strip()[:40],
                        "right": r["consistent"]} for r in pick if r],
            }
        out["items"][it] = {"slot": x["slot"].replace("_", " "), "experience": x["experience"],
                            "target": x["measure"]["target"].strip(),
                            "swap": items[x["swap"]]["slot"].replace("_", " "), "cells": cells}
    return out


# ------------------------------------------------------------------ formatting


def esc(s):
    return html.escape(str(s), quote=True)


def f2(v):
    return f"{v:.2f}".replace("-", "−")


def s2(v):
    return f"{v:+.2f}".replace("-", "−")


def pct(v):
    return f"{round(100 * v)}%"


def cis(t, signed=False):
    """(est, lo, hi) -> '+0.20 [0.04, 0.38]'."""
    f = s2 if signed else f2
    return f"{f(t[0])} [{f2(t[1])}, {f2(t[2])}]"


def rng(vals, f=f2, sep="–"):
    lo, hi = min(vals), max(vals)
    if f is pct:  # '19–37%'
        a, b = round(100 * lo), round(100 * hi)
        return f"{a}%" if a == b else f"{a}{sep}{b}%"
    return f(lo) if f(lo) == f(hi) else f"{f(lo)}{sep}{f(hi)}"


def triple(d, k):
    return (num(d[k]), num(d[k + "_lo"]), num(d[k + "_hi"]))


def svg_open(w, h, label, cls="chart"):
    return (f'<svg class="{cls}" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{esc(label)}">')


def table(headers, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="data"><summary>Data table</summary><div class="table-wrap">'
            f'<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div></details>')


def legend(entries):
    return ('<div class="legend">' + "".join(f'<span><i class="sw {c}"></i>{esc(t)}</span>' for c, t in entries)
            + "</div>")


def hit(tip, inner):
    return f'<g class="hit" tabindex="0" data-tip="{esc(tip)}">{inner}</g>'


# ------------------------------------------------------------------ chart helpers


def line_panel(title, sub, xs, xlabels, series, ymin, ymax, yticks, yfmt, W=440, H=250, extra=None,
               xtitle=None, end_labels=True):
    """series: list of (cls, name, [(v, lo, hi) or v], tipfmt). Categorical or numeric x positions."""
    l, r, t, b = 46, 40, 46, 38
    xmin, xmax = min(xs), max(xs)
    X = lambda x: l + (x - xmin) / (xmax - xmin) * (W - l - r)
    Y = lambda v: H - b - (max(min(v, ymax), ymin) - ymin) / (ymax - ymin) * (H - t - b)
    s = [svg_open(W, H, title), f'<text class="c-title" x="0" y="16">{esc(title)}</text>']
    if sub:
        s.append(f'<text class="c-sub" x="0" y="33">{esc(sub)}</text>')
    for v in yticks:
        s.append(f'<line class="{"c-axis" if v == 0 else "c-grid"}" x1="{l}" y1="{Y(v):.1f}" x2="{W - r}" '
                 f'y2="{Y(v):.1f}"/>')
        s.append(f'<text class="c-tick" x="{l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{yfmt(v)}</text>')
    for x, lab in zip(xs, xlabels):
        if lab is not None:
            s.append(f'<text class="c-tick" x="{X(x):.1f}" y="{H - b + 16}" text-anchor="middle">{esc(lab)}</text>')
    if xtitle:
        s.append(f'<text class="c-sub" x="{W - r}" y="{H - 3}" text-anchor="end">{esc(xtitle)}</text>')
    if extra:
        s.append(extra(X, Y, l, W - r, t, H - b))
    nser = len(series)
    ends = []
    for k, (cls, name, vals, tipf) in enumerate(series):
        off = (k - (nser - 1) / 2) * 5 if any(isinstance(v, tuple) for v in vals) else 0
        pts = [(X(x) + off, v if isinstance(v, tuple) else (v, NAN, NAN)) for x, v in zip(xs, vals)]
        d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{Y(v[0]):.1f}" for i, (px, v) in enumerate(pts))
        s.append(f'<path class="c-line {cls}" d="{d}"/>')
        for (px, v), x, lab in zip(pts, xs, xlabels):
            if not math.isnan(v[1]):
                s.append(f'<line class="c-ci {cls}" x1="{px:.1f}" y1="{Y(v[1]):.1f}" x2="{px:.1f}" '
                         f'y2="{Y(v[2]):.1f}"/>')
            dense = len(xs) >= 12  # many points: small dots without the surface ring
            s.append(hit(tipf(x, lab, v), f'<circle cx="{px:.1f}" cy="{Y(v[0]):.1f}" r="{6 if dense else 10}" '
                         f'fill="transparent"/><circle class="c-dot {cls}{" sm" if dense else ""}" cx="{px:.1f}" '
                         f'cy="{Y(v[0]):.1f}" r="{2.6 if dense else 4.5}"/>'))
        ends.append([Y(pts[-1][1][0]) + 4, pts[-1][0] + 8, name])
    # direct labels at the line ends, nudged apart so they never overlap
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 13:
            ends[i][0] = ends[i - 1][0] + 13
    for y, x, name in ends if end_labels else []:
        s.append(f'<text class="c-lab" x="{x:.1f}" y="{y:.1f}">{esc(name)}</text>')
    s.append("</svg>")
    return "".join(s)


def dot_panel(title, sub, rows, xmin, xmax, ticks, W=360, lab=128, right=None):
    """Dot + 95% CI per row on a signed axis. rows: dicts name, v (est, lo, hi), tip, note, sep (line above)."""
    x0, x1, xv, top, rh = lab + 10, W - 118, W - 66, 58, 30
    X = lambda v: x0 + (max(min(v, xmax), xmin) - xmin) / (xmax - xmin) * (x1 - x0)
    H = top + len(rows) * rh + 14
    s = [svg_open(W, H, title), f'<text class="c-title" x="0" y="16">{esc(title)}</text>']
    if sub:
        s.append(f'<text class="c-sub" x="0" y="33">{esc(sub)}</text>')
    s.append(f'<text class="c-sub" x="{xv}" y="{top - 16}" text-anchor="end">gain</text>')
    if right:
        s.append(f'<text class="c-sub" x="{W}" y="{top - 16}" text-anchor="end">{esc(right)}</text>')
    yb = top + len(rows) * rh - 12
    for v in ticks:
        s.append(f'<line class="{"c-axis" if v == 0 else "c-grid"}" x1="{X(v):.1f}" y1="{top - 12}" '
                 f'x2="{X(v):.1f}" y2="{yb}"/>')
        s.append(f'<text class="c-tick" x="{X(v):.1f}" y="{yb + 15}" text-anchor="middle">'
                 f'{("+" if v > 0 else "") + f"{v:g}".replace("-", "−")}</text>')
    for i, row in enumerate(rows):
        y = top + i * rh
        if row.get("sep"):
            s.append(f'<line class="c-sep" x1="0" y1="{y - rh / 2:.1f}" x2="{W}" y2="{y - rh / 2:.1f}"/>')
        s.append(f'<text class="c-label" x="{lab}" y="{y + 4}" text-anchor="end">{esc(row["name"])}</text>')
        v, lo, hi = row["v"]
        cls = row.get("cls") or ("pos" if lo > 0 else "neg" if hi < 0 else "neu")
        s.append(f'<line class="c-ci {cls}" x1="{X(lo):.1f}" y1="{y}" x2="{X(hi):.1f}" y2="{y}"/>')
        s.append(hit(row["tip"], f'<rect x="{X(lo) - 6:.1f}" y="{y - 11}" width="{X(hi) - X(lo) + 12:.1f}" '
                     f'height="22" fill="transparent"/><circle class="c-dot {cls}" cx="{X(v):.1f}" cy="{y}" r="5.5"/>'))
        s.append(f'<text class="c-val" x="{xv}" y="{y + 4}" text-anchor="end">{s2(v)}</text>')
        if row.get("note") is not None:
            s.append(f'<text class="c-tick" x="{W}" y="{y + 4}" text-anchor="end">{esc(row["note"])}</text>')
    s.append("</svg>")
    return "".join(s)


def bar_panel(title, sub, rows, xmax, ticks, fmt, W=360, lab=150, ref=None):
    """Horizontal bars with a 95% CI whisker. rows: name, v (est, lo, hi), cls, tip."""
    x0, x1, top, rh, bh = lab + 10, W - 44, 56, 28, 12
    X = lambda v: x0 + max(min(v, xmax), 0) / xmax * (x1 - x0)
    H = top + len(rows) * rh + 12
    s = [svg_open(W, H, title), f'<text class="c-title" x="0" y="16">{esc(title)}</text>']
    if sub:
        s.append(f'<text class="c-sub" x="0" y="33">{esc(sub)}</text>')
    yb = top + len(rows) * rh - 12
    for v in ticks:
        s.append(f'<line class="{"c-axis" if v == 0 else "c-grid"}" x1="{X(v):.1f}" y1="{top - 14}" '
                 f'x2="{X(v):.1f}" y2="{yb}"/>')
        s.append(f'<text class="c-tick" x="{X(v):.1f}" y="{yb + 15}" text-anchor="middle">{fmt(v)}</text>')
    if ref:
        rv, rl = ref
        s.append(f'<line class="c-ref dash" x1="{X(rv):.1f}" y1="{top - 14}" x2="{X(rv):.1f}" y2="{yb}"/>'
                 f'<text class="c-sub" x="{X(rv) + 4:.1f}" y="{top - 18}">{esc(rl)}</text>')
    for i, row in enumerate(rows):
        y = top + i * rh
        s.append(f'<text class="c-label" x="{lab}" y="{y + 4}" text-anchor="end">{esc(row["name"])}</text>')
        v, lo, hi = row["v"]
        w = max(X(v) - x0, 1.5)
        s.append(f'<rect class="seg {row.get("cls", "s1")}" tabindex="0" data-tip="{esc(row["tip"])}" x="{x0}" '
                 f'y="{y - bh / 2}" width="{w:.1f}" height="{bh}" rx="2"/>')
        if not math.isnan(lo) and hi > lo:
            s.append(f'<line class="c-ci ink" x1="{X(lo):.1f}" y1="{y}" x2="{X(hi):.1f}" y2="{y}"/>')
        s.append(f'<text class="c-val" x="{W}" y="{y + 4}" text-anchor="end">{fmt(v)}</text>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ figures


def fig_dose(pref):
    """Paper 1, leanings: judged lean gain and judged incoherence against the dose, both sizes."""
    xs, labs, conds = [0, 1, 2, 3, 4], ["none", "0.5", "1", "2", "3"], ["mem@0.5", "mem@1", "mem@2", "mem@3"]
    ser_gain, ser_inc, data = [], [], []
    for m, cls, name in (("qwen35_2b", "s1", "2B"), ("qwen35_9b", "s2", "9B")):
        P = pref[m]
        gains = [(0.0, NAN, NAN)] + [triple(P[("leanings", c)], "d_j_lean") for c in conds]
        incs = [num(P[("leanings", "nomem")]["j_incoh"])] + [num(P[("leanings", c)]["j_incoh"]) for c in conds]
        ser_gain.append((cls, name, gains,
                         lambda x, lab, v, name=name: f"Qwen3.5-{name}, α {lab}: judged lean gain {s2(v[0])}"
                         + ("" if math.isnan(v[1]) else f" [{f2(v[1])}, {f2(v[2])}]")))
        ser_inc.append((cls, name, incs,
                        lambda x, lab, v, name=name: f"Qwen3.5-{name}, α {lab}: {pct(v[0])} of answers incoherent"))
        for lab, c, g, inc in zip(labs, ["nomem"] + conds, gains, incs):
            data.append([f"Qwen3.5-{name}", lab, "0 (reference)" if c == "nomem" else cis(g, True), pct(inc),
                         f2(num(P[("leanings", c)]["d_lex_clean"])) if c != "nomem" else "—"])
    mark = lambda X, Y, a, b, t, yb: (f'<line class="c-mark" x1="{X(2):.1f}" y1="{t - 6}" x2="{X(2):.1f}" '
                                       f'y2="{yb}"/>')
    p1 = line_panel("Judged lean gain vs no memory", "−1..+1 scale; whiskers: 95% CI", xs, labs, ser_gain,
                    -0.25, 0.5, (-0.2, 0, 0.2, 0.4), lambda v: s2(v) if v else "0", extra=mark,
                    xtitle="α per layer")
    p2 = line_panel("Answers judged incoherent", "share of all answers", xs, labs, ser_inc, 0, 1,
                    (0, 0.25, 0.5, 0.75, 1), pct, extra=mark, xtitle="α per layer")
    return (legend([("s1", "Qwen3.5-2B"), ("s2", "Qwen3.5-9B")])
            + '<div class="panels two">' + f'<div class="panel">{p1}</div><div class="panel">{p2}</div></div>'
            + table(["model", "α per layer", "judged lean gain [95% CI]", "incoherent", "word-count lean gain"],
                    data))


def fig_c5(claims, pref):
    """Paper 1, C5: dislikes under each reference, loop-matched dose, both sizes."""
    panels, data = [], []
    for m, name in (("qwen35_2b", "Qwen3.5-2B"), ("qwen35_9b", "Qwen3.5-9B")):
        ev = claims[m]["C5"]["evidence"]
        n = int(num(pref[m][("dislikes", "ctx")]["n_items"]))
        ctx = triple(pref[m][("dislikes", "ctx")], "d_j_lean")
        rows = [{"name": "in context", "v": ctx, "cls": "neu", "note": "",
                 "tip": f"{name}, the dislike written in the prompt: gain {cis(ctx, True)}"}]
        data.append([name, "in context", "", cis(ctx, True), ""])
        for ref, lab in (("opposite", "minus the opposite"), ("without", "minus “without”"),
                         ("centroid", "minus the alternatives")):
            e = ev[ref]
            v = (e["d_j_lean"], e["d_j_lean_lo"], e["d_j_lean_hi"])
            rows.append({"name": lab, "v": v, "sep": ref == "opposite", "note": pct(e["wrong_items"]),
                         "tip": f"{name}, {lab} (α {e['alpha']:g}): gain {cis(v, True)}; "
                                f"{pct(e['wrong_items'])} of {n} dislikes pushed the wrong way"})
            data.append([name, lab, f"{e['alpha']:g}", cis(v, True), pct(e["wrong_items"])])
        panels.append(dot_panel(name, f"{n} dislikes · + = away from the disliked thing", rows, -1, 1.25,
                                (-1, 0, 1), W=400, lab=128, right="wrong way"))
    return ('<div class="panels two">' + "".join(f'<div class="panel">{p}</div>' for p in panels) + "</div>"
            + table(["model", "reference subtracted", "α per layer", "judged lean gain [95% CI]",
                     "dislikes pushed the wrong way"], data))


def fig_stage0(s0):
    panels, data = [], []
    for m, name in (("qwen25_15b", "Qwen2.5-1.5B"), ("qwen35_2b", "Qwen3.5-2B")):
        c = s0[m]
        L = len(c["attr"])
        xs = list(range(L))
        labs = [str(x) if x % 4 == 0 or x == L - 1 else None for x in xs]
        ser = [("s1", "the name", c["attr"],
                lambda x, lab, v, name=name: f"{name}, layer {x}: patching the name’s position moves {f2(v[0])}"),
               ("s2", "the subject", c["subj_span"],
                lambda x, lab, v, name=name: f"{name}, layer {x}: patching the question’s subject moves {f2(v[0])}")]
        panels.append(line_panel(name, f"share of the Pepper-vs-Rocky preference moved · {c['n_items']} facts",
                                 xs, labs, ser, -0.05, 1.05, (0, 0.25, 0.5, 0.75, 1), f2,
                                 xtitle=f"layer (of {L})", end_labels=False))
        for x in xs:
            data.append([name, x, f2(c["attr"][x]), f2(c["subj_span"][x])])
    return (legend([("s1", "patch at the name’s own position (“Pepper”)"),
                    ("s2", "patch at the question’s subject (“my dog”)")])
            + '<div class="panels two">' + "".join(f'<div class="panel">{p}</div>' for p in panels) + "</div>"
            + table(["model", "layer", "name position", "question subject"], data))


def fig_clean_use(D):
    """Clean use on the use prompts, both models, every way of carrying the fact."""
    g25, s2b, t25, t2b, push = D["gen25"], D["state2b"], D["add5_25"], D["add5_2b"], D["push2b"]
    rows = [
        ("no memory", g25["clean"]["nomem"], s2b["nomem"]),
        ("pushed vector (paper 1, α = 1)", None, push),
        ("the same 2 tokens as text", t25["T-min"], t2b["T-min"]),
        ("2-position slice, attention only", g25["clean"]["R-min"], s2b["R-min-kv6"]),
        ("2-position slice + recurrent state", "n/a", s2b["R-min-state"]),
        ("recurrent state only", "n/a", s2b["R-state-only"]),
        ("whole turn (+ state on the 2B)", g25["clean"]["R-full"], s2b["R-full-state"]),
        ("stored after an unrelated chat", g25["clean"]["R-orphan"], s2b["R-orphan-state"]),
        ("the sentence in context", g25["clean"]["ctx"], s2b["ctx"]),
    ]
    W, lab, x0, x1, top, rh, bh = 700, 246, 258, 640, 34, 40, 12
    X = lambda v: x0 + v * (x1 - x0)
    H = top + len(rows) * rh + 14
    s = [svg_open(W, H, "Clean use of a stored fact, by how it is carried, on two models")]
    s.append(f'<text class="c-sub" x="{lab}" y="14" text-anchor="end">how the fact is carried</text>')
    s.append(f'<text class="c-sub" x="{x0}" y="14">share of answers that use the fact correctly and coherently</text>')
    yb = top + len(rows) * rh - 12
    for v in (0, 0.25, 0.5, 0.75, 1):
        s.append(f'<line class="{"c-axis" if v == 0 else "c-grid"}" x1="{X(v):.1f}" y1="{top - 12}" '
                 f'x2="{X(v):.1f}" y2="{yb}"/>')
        s.append(f'<text class="c-tick" x="{X(v):.1f}" y="{yb + 15}" text-anchor="middle">{f2(v)}</text>')
    data = []
    for i, (name, a, b) in enumerate(rows):
        y = top + i * rh
        if i:
            s.append(f'<line class="c-sep" x1="0" y1="{y - rh / 2:.1f}" x2="{W}" y2="{y - rh / 2:.1f}"/>')
        s.append(f'<text class="c-label" x="{lab}" y="{y + 4}" text-anchor="end">{esc(name)}</text>')
        cells = []
        for v, cls, mname, dy in ((a, "s1", "Qwen2.5-1.5B", -8), (b, "s2", "Qwen3.5-2B", 8)):
            yy = y + dy
            if not isinstance(v, tuple):
                txt = "not run" if v is None else "n/a: no recurrent state"
                s.append(f'<text class="c-tick" x="{x0 + 6}" y="{yy + 4}">{mname}: {txt}</text>')
                cells.append(txt)
                continue
            w = max(X(v[0]) - x0, 1.5)
            tip = f"{mname}, {name}: clean use {cis(v)}"
            s.append(f'<rect class="seg {cls}" tabindex="0" data-tip="{esc(tip)}" x="{x0}" y="{yy - bh / 2}" '
                     f'width="{w:.1f}" height="{bh}" rx="2"/>')
            if v[2] > v[1]:
                s.append(f'<line class="c-ci ink" x1="{X(v[1]):.1f}" y1="{yy}" x2="{X(v[2]):.1f}" y2="{yy}"/>')
            s.append(f'<text class="c-val" x="{W}" y="{yy + 4}" text-anchor="end">{f2(v[0])}</text>')
            cells.append(cis(v))
        data.append([name] + cells)
    s.append("</svg>")
    return (legend([("s1", "Qwen2.5-1.5B (all attention)"), ("s2", "Qwen3.5-2B (hybrid)")]) + "".join(s)
            + table(["how the fact is carried", "Qwen2.5-1.5B clean use [95% CI]",
                     "Qwen3.5-2B clean use [95% CI]"], data))


def fig_hybrid(dec):
    rows = [("ctx", "sentence in context"), ("R-min-state-rec", "2 positions + recurrent state"),
            ("R-min-kv6", "2 positions, attention only"), ("R-state-only-rec", "recurrent state only"),
            ("R-state-only-conv", "conv buffer only")]
    t = dec["table"]
    tri = lambda c, k: (t[c][k]["mean"], t[c][k]["lo"], t[c][k]["hi"])
    va = [{"name": n, "v": tri(c, "dlp"), "cls": "s1",
           "tip": f"{n}: log-probability of the right name up {cis(tri(c, 'dlp'))} nats vs no memory"} for c, n in rows]
    yn = [{"name": n, "v": tri(c, "yn_bal_acc"), "cls": "s2",
           "tip": f"{n}: balanced yes/no {cis(tri(c, 'yn_bal_acc'))}"} for c, n in rows]
    p1 = bar_panel("The value: the right name’s log-prob gain", "nats vs no memory, teacher-forced", va, 12,
                   (0, 4, 8, 12), lambda v: f"{v:.1f}" if v % 1 else f"{v:.0f}", W=380, lab=168)
    p2 = bar_panel("Verification: balanced yes/no", "0.50 = chance", yn, 1, (0, 0.5, 1), f2, W=380, lab=168,
                   ref=(0.5, "chance"))
    data = [[n, cis(tri(c, "dlp")), cis(tri(c, "yn_bal_acc"))] for c, n in rows]
    return ('<div class="panels two">' + f'<div class="panel">{p1}</div><div class="panel">{p2}</div></div>'
            + table(["what is restored", "gain in log P(right name), nats [95% CI]", "balanced yes/no [95% CI]"],
                    data))


COLL_ROWS = {"qwen25_15b": [("ctx2", "both in context"), ("R2-joint", "encoded together"),
                            ("R2-sep-overlap", "stored separately"), ("R2-sep-shift", "separately, shifted")],
             "qwen35_2b": [("ctx2", "both in context"), ("R2-joint-state", "encoded together"),
                           ("R2-sep-overlap-jstate", "stored separately"),
                           ("R2-sep-shift-jstate", "separately, shifted")]}


def fig_collide(coll):
    panels, data = [], []
    for m, name in (("qwen25_15b", "Qwen2.5-1.5B"), ("qwen35_2b", "Qwen3.5-2B")):
        T = coll[m]
        W, lab, x0, x1, rh, bh = 380, 124, 134, 336, 24, 13
        X = lambda f: x0 + f * (x1 - x0)
        y = 52
        s = [svg_open(W, 330, f"{name}: two stored facts used together"),
             f'<text class="c-title" x="0" y="16">{name}</text>',
             f'<text class="c-sub" x="0" y="33">share of answers</text>',
             f'<text class="c-sub" x="{W}" y="33" text-anchor="end">swap</text>']
        for grp, gname in (("confusable", "confusable pairs (older / younger sister)"),
                           ("different", "different subjects (dog / cat)")):
            s.append(f'<text class="c-lane" x="0" y="{y}">{esc(gname)}</text>')
            y += 22
            for c, rname in COLL_ROWS[m]:
                r = T[f"{grp}/{c}"]
                cor, conf = r["correct (majority)"]["mean"], r["partner confusion"]["mean"]
                rest = max(0.0, 1 - cor - conf)
                s.append(f'<text class="c-label" x="{lab}" y="{y + 4}" text-anchor="end">{esc(rname)}</text>')
                x = X(0)
                for v, cls, word in ((cor, "s1", "the right name"), (conf, "s2", "the partner’s name"),
                                     (rest, "neu", "neither")):
                    w = v * (x1 - x0)
                    if w > 0.5:
                        tip = f"{name}, {gname}, {rname}: {word} in {pct(v)}"
                        s.append(f'<rect class="seg {cls}" tabindex="0" data-tip="{esc(tip)}" x="{x:.1f}" '
                                 f'y="{y - bh / 2}" width="{max(w - 2, 1):.1f}" height="{bh}" rx="2"/>')
                    x += w
                s.append(f'<text class="c-val" x="{W}" y="{y + 4}" text-anchor="end">{pct(conf)}</text>')
                data.append([name, gname, rname, cis((cor, r["correct (majority)"]["lo"], r["correct (majority)"]["hi"])),
                             cis((conf, r["partner confusion"]["lo"], r["partner confusion"]["hi"]))])
                y += rh
            y += 14
        s[0] = svg_open(W, y - 4, f"{name}: two stored facts used together")
        s.append("</svg>")
        panels.append("".join(s))
    return (legend([("s1", "the right name"), ("s2", "the partner’s name (a swap)"), ("neu", "neither")])
            + '<div class="panels two">' + "".join(f'<div class="panel">{p}</div>' for p in panels) + "</div>"
            + table(["model", "pairs", "how the two facts were stored", "correct [95% CI]",
                     "partner’s name [95% CI]"], data))


REF_ROWS = [("hand:opposite", "hand-written opposite"), ("gen:opposite", "generated opposite"),
            ("gen:centroid", "generated alternatives"), ("cs:concept_x_sign", "concept × read sign"),
            ("cs:oracle", "concept × true sign")]


def fig_refs(matched, ctxpref):
    panels, data = [], []
    for grp, gname in (("dislikes", "Dislikes"), ("leanings", "Leanings")):
        n = int(num(matched[(grp, "gen:opposite")]["n_group"]))
        ctx = triple(ctxpref[(grp, "ctx")], "d_j_lean")
        rows = [{"name": "in context", "v": ctx, "cls": "neu", "note": "",
                 "tip": f"{gname}, the experience in the prompt: gain {cis(ctx, True)}"}]
        data.append([gname, "in context", "", cis(ctx, True), ""])
        for k, lab in REF_ROWS:
            r = matched[(grp, k)]
            v = triple(r, "d_j_lean")
            a = num(r["alpha"])
            rows.append({"name": lab, "v": v, "sep": k == "hand:opposite", "note": f"α {a:g}",
                         "tip": f"{gname}, {lab} at α {a:g}: gain {cis(v, True)}; "
                                f"{pct(num(r['wrong_items']))} of items pushed the wrong way"})
            data.append([gname, lab, f"{a:g}", cis(v, True), pct(num(r["wrong_items"]))])
        panels.append(dot_panel(gname, f"{n} items · Qwen3.5-2B · loop-matched dose", rows, -0.75, 1,
                                (-0.5, 0, 0.5, 1), W=420, lab=150, right="dose"))
    return ('<div class="panels two">' + "".join(f'<div class="panel">{p}</div>' for p in panels) + "</div>"
            + table(["group", "memory written from", "α per layer", "judged lean gain [95% CI]",
                     "items pushed the wrong way"], data))


def tbl_panel(panel):
    """Which pre-registered claims hold under which judge (paper 1)."""
    judges = [("orig", "Qwen3.5-9B (original)"), ("gptoss", "gpt-oss-20B"), ("gemma4", "Gemma-4-12B"),
              ("maj4", "majority: original + 3"), ("maj3x", "majority: 3 cross-family")]
    names = {"C2": "C2 leanings transfer", "C3": "C3 facts are not cleanly used",
             "C5": "C5 the reference decides direction"}
    th = "".join(f"<th>{esc(n)}</th>" for _, n in judges)
    trs = []
    for c in ("C2", "C3", "C5"):
        for m, mn in (("qwen35_2b", "2B"), ("qwen35_9b", "9B")):
            h = panel["summary"][c][m]["holds"]
            tds = "".join(f'<td class="{"ok" if h[j] else "no"}">{"holds" if h[j] else "fails"}</td>' for j, _ in judges)
            trs.append(f"<tr><td>{esc(names[c])} · {mn}</td>{tds}</tr>")
    return (f'<div class="table-wrap"><table class="tbl small"><thead><tr><th>claim · model</th>{th}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def tbl_prefeval(pe):
    judges = [("orig", "Qwen3.5-9B (original)"), ("gptoss", "gpt-oss-20B"), ("gemma4", "Gemma-4-12B"),
              ("majority", "majority")]
    trs = []
    for j, jn in judges:
        R = pe[j]
        g1, g2, cx = R[("all", "gen:opposite@1")], R[("all", "gen:opposite@2")], R[("all", "ctx")]
        trs.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in (
            jn, f2(num(g1["nomem_acc"])), cis(triple(g1, "d_acc"), True), cis(triple(g2, "d_acc"), True),
            cis(triple(cx, "d_acc"), True))) + "</tr>")
    return ('<div class="table-wrap"><table class="tbl small"><thead><tr><th>judge</th><th>no-memory accuracy</th>'
            '<th>generated opposite, α = 1</th><th>α = 2</th><th>experience in context</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE.parents[1] / "results")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = a.results

    # paper 1
    claims, pref, facts, yn, panel = core(res)
    c2, c9 = claims["qwen35_2b"], claims["qwen35_9b"]
    lean = lambda m, c: triple(pref[m][("leanings", c)], "d_j_lean")
    inc = lambda m, c: num(pref[m][("leanings", c)]["j_incoh"])
    push = {m: max(v["use_judge_clean"] for v in claims[m]["C3"]["evidence"]["per_dose"].values())
            for m in claims}
    mention2 = [claims[m]["C3"]["evidence"]["per_dose"][d]["mention_unclean"] for m in claims
                for d in ("mem@2", "mem@3")]
    dbal = {(m, k): claims[m]["C4"]["evidence"][k]["d_bal"] for m in claims for k in ("facts/mem@1", "facts/mem@2")}
    c5 = {m: claims[m]["C5"]["evidence"] for m in claims}
    cen_panel = [panel["claims"][j][m]["C5"]["centroid"]["est"] for j in ("gptoss", "gemma4") for m in claims]
    c6 = {m: claims[m]["C6"]["evidence"] for m in claims}
    push2b = triple(facts["qwen35_2b"]["mem@1"], "j_clean_use")

    # stage 0
    s0 = stage0(res)
    subj_max = max(abs(v) for m in s0 for v in s0[m]["subj_span"])
    a25, a2b = s0["qwen25_15b"]["attr"], s0["qwen35_2b"]["attr"]
    last_full = lambda a: max(i for i, v in enumerate(a) if v >= 0.995)  # still moves (rounded) 1.00
    first_drop = lambda a: min(i for i, v in enumerate(a) if v <= 0.1)
    ko = s0["qwen25_15b"]["verdict"]["ko_attr_best"]

    # slices
    g25 = gen_report(res / GEN25 / "analysis" / "report.md")
    st2b_v = jload(res / STATE2B / "analysis" / "verdict.json")
    st2b = {k.split("/", 1)[1]: (v["mean"], v["lo"], v["hi"]) for k, v in st2b_v["clean_use"].items()
            if k.startswith("primary/")}
    st2b_r = gen_report(res / STATE2B / "analysis" / "report.md")
    a5 = {m: jload(res / ADD5 / m / "analysis" / "verdict.json") for m in ("qwen25_15b", "qwen35_2b")}
    a5u = {m: {k.split("/", 1)[1]: (v["mean"], v["lo"], v["hi"]) for k, v in a5[m]["clean_use"].items()
               if k.startswith("primary/")} for m in a5}
    a5r = {m: gen_report(res / ADD5 / m / "analysis" / "report.md") for m in a5}
    rep = lambda d: (d["mean"], d["lo"], d["hi"])
    tmin25 = rep(a5["qwen25_15b"]["verdict"]["reported"]["T-min - R-min (addendum 5: expected well below 0)"])
    tmin2b = rep(a5["qwen35_2b"]["verdict"]["reported"]["T-min - R-min-state (addendum 5: expected well below 0)"])
    st_minus_kv = rep(st2b_v["verdict"]["reported"]["R-min-state - R-min-kv6 (expected > 0)"])
    dec = jload(res / ADD6 / "qwen35_2b" / "analysis" / "verdict.json")["verdict"]["decomp"]
    dt = dec["table"]
    state_n = math.prod(DELTANET_STATE)

    # collisions, repair, retrieval
    coll = {m: jload(res / ADD5 / f"{m}_collide" / "analysis" / "verdict.json")["tables"]
            for m in ("qwen25_15b", "qwen35_2b")}
    sep_conf = [coll[m][f"{g}/{c}"]["partner confusion"]["mean"] for m in coll for g in ("confusable", "different")
                for c, _ in COLL_ROWS[m][2:]]
    joint_corr = [coll[m][f"{g}/{COLL_ROWS[m][1][0]}"]["correct (majority)"]["mean"] for m in coll
                  for g in ("confusable", "different")]
    ctx2_corr = [coll[m][f"{g}/ctx2"]["correct (majority)"]["mean"] for m in coll for g in ("confusable", "different")]
    idr = jload(res / ADD6 / "qwen25_15b_idrepair" / "analysis" / "verdict.json")
    gk_text = (res / ADD6 / "qwen25_15b_gkey" / "analysis" / "report.md").read_text()
    gk_ret = md_tables(md_section(gk_text, "## Retrieval"))[0]
    gk_tf = {r["condition"]: r for r in md_tables(md_section(gk_text, "## Teacher-forced downstream"))[0]}
    gk_gen = {(r["tier / prompts"], r["condition"]): r for r in md_tables(md_section(gk_text, "## Generation at N"))[0]}
    nmax = max(int(r["N"]) for r in gk_ret)
    k1 = [r for r in gk_ret if r["key"] == "K1"]
    k1max = [r for r in k1 if int(r["N"]) == nmax]
    k2max = [r for r in gk_ret if r["key"] == "K2" and int(r["N"]) == nmax]
    wrong_vals = [ci(gk_tf[f"gkey@{nmax}"]["wrong value (of injected wrong slices)"])[0],
                  ci(gk_tf["swap"]["wrong value (of injected wrong slices)"])[0]]

    # experiment 2
    matched = {(r["group"], r["variant"]): r for r in csv_rows(res / REFS / "qwen35_2b" / "analysis" / "matched.csv")}
    refpref = {(r["group"], r["cond"]): r for r in csv_rows(res / REFS / "qwen35_2b" / "analysis" / "pref.csv")}
    rA = (res / REFS / "report_A.md").read_text()
    s3 = [float(r["accuracy"]) for r in md_tables(md_section(rA, "## S3"))[0]]
    pe = {j: {(r["subset"], r["cond"]): r for r in csv_rows(res / PE / "qwen35_2b" / "analysis" / f"gains_{j}.csv")}
          for j in ("orig", "gptoss", "gemma4", "majority")}
    pe1 = [num(pe[j][("all", "gen:opposite@1")]["d_acc"]) for j in ("orig", "gptoss", "gemma4")]
    pectx = [num(pe[j][("all", "ctx")]["d_acc"]) for j in pe]
    m_ = lambda g, k: triple(matched[(g, k)], "d_j_lean")

    D = {"gen25": g25, "state2b": st2b, "add5_25": a5u["qwen25_15b"], "add5_2b": a5u["qwen35_2b"], "push2b": push2b}
    view = viewer_data(res)
    fill = {
        "FIG_DOSE": fig_dose(pref),
        "FIG_C5": fig_c5(claims, pref),
        "FIG_STAGE0": fig_stage0(s0),
        "FIG_CLEAN_USE": fig_clean_use(D),
        "FIG_HYBRID": fig_hybrid(dec),
        "FIG_COLLIDE": fig_collide(coll),
        "FIG_REFS": fig_refs(matched, refpref),
        "TBL_PANEL": tbl_panel(panel),
        "TBL_PREFEVAL": tbl_prefeval(pe),
        "VIEWER_DATA": json.dumps(view, ensure_ascii=False).replace("</", "<\\/"),
        # paper 1
        "C1_SAME_2B": pct(c2["C1"]["evidence"]["unrel_same_gated_mean"]),
        "C1_SAME_9B": pct(c9["C1"]["evidence"]["unrel_same_gated_mean"]),
        "C1_OFF_2B": pct(c2["C1"]["evidence"]["unrel_same_gate_on"]),
        "C1_OFF_9B": pct(c9["C1"]["evidence"]["unrel_same_gate_on"]),
        "N_LEAN": str(int(num(pref["qwen35_2b"][("leanings", "mem@1")]["n_items"]))),
        "N_DIS": str(int(num(pref["qwen35_2b"][("dislikes(all)", "mem@1")]["n_items"]))),
        "N_FACTS": str(int(num(facts["qwen35_2b"]["mem@1"]["n_items"]))),
        "LEAN1_2B": cis(lean("qwen35_2b", "mem@1"), True), "LEAN1_9B": cis(lean("qwen35_9b", "mem@1"), True),
        "LEAN2_2B": cis(lean("qwen35_2b", "mem@2"), True), "LEAN2_9B": cis(lean("qwen35_9b", "mem@2"), True),
        "LEAN1_2B_V": s2(lean("qwen35_2b", "mem@1")[0]), "LEAN1_9B_V": s2(lean("qwen35_9b", "mem@1")[0]),
        "INC0_2B": pct(inc("qwen35_2b", "nomem")), "INC0_9B": pct(inc("qwen35_9b", "nomem")),
        "INC1_2B": pct(inc("qwen35_2b", "mem@1")), "INC1_9B": pct(inc("qwen35_9b", "mem@1")),
        "INC2_2B": pct(inc("qwen35_2b", "mem@2")), "INC2_9B": pct(inc("qwen35_9b", "mem@2")),
        "LEX1_2B": s2(num(pref["qwen35_2b"][("leanings", "mem@1")]["d_lex_clean"])),
        "LEX2_2B": s2(num(pref["qwen35_2b"][("leanings", "mem@2")]["d_lex_clean"])),
        "CTXLEAN_2B": s2(lean("qwen35_2b", "ctx")[0]), "CTXLEAN_9B": s2(lean("qwen35_9b", "ctx")[0]),
        "PUSH_MAX_2B": f2(push["qwen35_2b"]), "PUSH_MAX_9B": f2(push["qwen35_9b"]),
        "PUSH_MAX": f2(max(push.values())),
        "FACT_CTX": rng([c2["C3"]["evidence"]["ctx_use_judge_clean"], c9["C3"]["evidence"]["ctx_use_judge_clean"]]),
        "MENTION2": rng(mention2, pct),
        "BAL0": rng([num(yn[m][("facts", "nomem")]["bal"]) for m in yn]),
        "BAL1": rng([num(yn[m][("facts", "mem@1")]["bal"]) for m in yn]),
        "BALCTX": rng([num(yn[m][("facts", "ctx")]["bal"]) for m in yn]),
        "DBAL2_2B": cis(dbal[("qwen35_2b", "facts/mem@2")], True),
        "OPP_2B": s2(c5["qwen35_2b"]["opposite"]["d_j_lean"]), "OPP_9B": s2(c5["qwen35_9b"]["opposite"]["d_j_lean"]),
        "CEN_2B": s2(c5["qwen35_2b"]["centroid"]["d_j_lean"]), "CEN_9B": s2(c5["qwen35_9b"]["centroid"]["d_j_lean"]),
        "WO_2B": s2(c5["qwen35_2b"]["without"]["d_j_lean"]), "WO_9B": s2(c5["qwen35_9b"]["without"]["d_j_lean"]),
        "CEN_PANEL": rng(cen_panel, s2, " to "),
        "C6_DEPTH_2B": f"{min(d for _, d in c6['qwen35_2b']['blocks_median_rank_le10']):.3g}",
        "C6_DEPTH_9B": f"{min(d for _, d in c6['qwen35_9b']['blocks_median_rank_le10']):.3g}",
        "C6_RANK_HALF": rng([c6[m]["min_median_rank_depth_le_0.5"] for m in c6], lambda v: f"{round(v, -3):,.0f}"),
        # stage 0
        "S0_SUBJ_MAX": f2(subj_max),
        "S0_25_FULL_TO": str(last_full(a25)), "S0_2B_FULL_TO": str(last_full(a2b)),
        "S0_2B_DROP_L": str(first_drop(a2b)), "S0_2B_DROP_V": f2(a2b[first_drop(a2b)]),
        "S0_25_DROP_L": str(first_drop(a25)), "S0_25_LAST": str(len(a25) - 1),
        "S0_KO_ATTR": cis((ko["mean"], ko["lo"], ko["hi"])),
        "SUBJ_CLEAN": f2(g25["clean"]["subj1"][0]),
        # slices
        "R25_MIN": f2(g25["clean"]["R-min"][0]), "R25_MIN_CI": cis(g25["clean"]["R-min"]),
        "R25_FULL": f2(g25["clean"]["R-full"][0]), "R25_ORPHAN": f2(g25["clean"]["R-orphan"][0]),
        "CTX25": f2(g25["clean"]["ctx"][0]),
        "TMIN25": f2(a5u["qwen25_15b"]["T-min"][0]), "TMIN2B": f2(a5u["qwen35_2b"]["T-min"][0]),
        "TMIN25_DIRECT": f2(a5r["qwen25_15b"]["direct"]["T-min"][0]),
        "TMIN_DIFF_25": cis(tmin25, True), "TMIN_DIFF_2B": cis(tmin2b, True),
        "TFULL_RFULL": cis(rep(a5["qwen25_15b"]["verdict"]["reported"]["T-full - R-full (= by construction)"]), True),
        "R25_MIN_DIRECT": cis(g25["direct"]["R-min"]), "R25_FULL_DIRECT": f2(g25["direct"]["R-full"][0]),
        "UNREL_GATED": f"{g25['unrel']['R-full']:.3f}", "UNREL_UNGATED": f"{g25['unrel']['R-full:ungated']:.3f}",
        "SELF_RFULL": f2(g25["self"]["R-full"][0]), "SELF_CTX": f2(g25["self"]["ctx"][0]),
        "SWAP_CONFAB": pct(g25["confab"]["R-swap"][0]),
        "R2B_MINSTATE": f2(st2b["R-min-state"][0]), "R2B_MINSTATE_CI": cis(st2b["R-min-state"]),
        "R2B_KV6": f2(st2b["R-min-kv6"][0]), "R2B_STATEONLY": f2(st2b["R-state-only"][0]),
        "R2B_ORPHAN": f2(st2b["R-orphan-state"][0]), "CTX2B": f2(st2b["ctx"][0]),
        "R2B_KV6_DIRECT": f2(st2b_r["direct"]["R-min-kv6"][0]),
        "R2B_MINSTATE_DIRECT": f2(st2b_r["direct"]["R-min-state"][0]),
        "R2B_STATEONLY_DIRECT": f2(st2b_r["direct"]["R-state-only"][0]),
        "ST_MINUS_KV": cis(st_minus_kv, True),
        "TF_KV6_NATS": f"{dt['R-min-kv6']['dlp']['mean']:.1f}", "TF_CTX_NATS": f"{dt['ctx']['dlp']['mean']:.1f}",
        "TF_STATE_NATS": f"{dt['R-state-only-rec']['dlp']['mean']:.1f}",
        "TF_KV6_YN": f2(dt["R-min-kv6"]["yn_bal_acc"]["mean"]),
        "TF_STATE_YN": f2(dt["R-state-only-rec"]["yn_bal_acc"]["mean"]),
        "TF_CONV_YN": f2(dt["R-state-only-conv"]["yn_bal_acc"]["mean"]),
        "TF_MINREC_YN": f2(dt["R-min-state-rec"]["yn_bal_acc"]["mean"]),
        "N_PRIMARY": str(st2b_v["clean_use"]["primary/ctx"]["n"]),
        "STATE_NUMBERS": f"{state_n / 1e6:.1f} million", "STATE_MB": f"{state_n * 4 / 1e6:.0f} MB",
        # collisions and retrieval
        "COLL_RANGE": rng(sep_conf, pct), "JOINT_CORR": rng(joint_corr), "CTX2_CORR": rng(ctx2_corr),
        "ID_VALID": cis((idr["id_valid"]["mean"], idr["id_valid"]["lo"], idr["id_valid"]["hi"])),
        "ID_REPAIR": cis(rep(idr["rules"]["repair_vs_sep"]), True),
        "GK_N": str(nmax),
        "GK_K1": rng([ci(r["correct"])[0] for r in k1max]), "GK_K2": rng([ci(r["correct"])[0] for r in k2max]),
        "GK_TOP2": rng([ci(r["top-2 correct"])[0] for r in k1max]),
        "GK_FF": rng([first_num(r["false fire (unrelated)"]) for r in k1], pct),
        "GK_WRONG": rng(wrong_vals),
        "GK_ORACLE_NATS": f"{ci(gk_tf['oracle']['Δlog P(target)'])[0]:.2f}",
        "GK_CTX_NATS": f"{ci(gk_tf['ctx']['Δlog P(target)'])[0]:.2f}",
        "GK_SWAP_NATS": s2(ci(gk_tf["swap"]["Δlog P(target)"])[0]),
        "GK_USE_ORACLE": f2(ci(gk_gen[("primary / use", "oracle")]["clean use"])[0]),
        "GK_USE_KEY": f2(ci(gk_gen[("primary / use", "gkey")]["clean use"])[0]),
        "GK_USE_CONFAB": f2(ci(gk_gen[("primary / use", "gkey")]["confab (wrong value)"])[0]),
        # experiment 2
        "GEN_OPP_DIS": s2(m_("dislikes", "gen:opposite")[0]), "HAND_OPP_DIS": s2(m_("dislikes", "hand:opposite")[0]),
        "GEN_CEN_DIS": s2(m_("dislikes", "gen:centroid")[0]), "GEN_CEN_DIS_CI": cis(m_("dislikes", "gen:centroid"), True),
        "GEN_CEN_WRONG": pct(num(matched[("dislikes", "gen:centroid")]["wrong_items"])),
        "CS_ORACLE_LEAN": s2(m_("leanings", "cs:oracle")[0]), "CS_ORACLE_DIS": s2(m_("dislikes", "cs:oracle")[0]),
        "CS_AUTO_LEAN": cis(m_("leanings", "cs:concept_x_sign"), True),
        "GEN_OPP_LEAN": s2(m_("leanings", "gen:opposite")[0]), "HAND_OPP_LEAN": s2(m_("leanings", "hand:opposite")[0]),
        "S3_ACC": rng(s3), "PE1": rng(pe1, s2, " to "), "PECTX": rng(pectx, s2, " to "),
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
    for k, v in fill.items():
        if not (k.startswith("FIG_") or k.startswith("TBL_") or k == "VIEWER_DATA"):
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
