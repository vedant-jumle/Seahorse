"""Pull every number plotted or quoted on the "KVzip vs Slices" explainer page from the result files, and write them
into the page's data block (between the DATA-BEGIN / DATA-END markers in index.html) and into data.json beside it.

Run from anywhere, in the env that has transformers (the worked example decodes Llama-3.2-3B-Instruct token strings):

    conda run -n torch python docs/explainers/kvzip-vs-slices/extract.py

Sources (all under Seahorse/):
  results/dream_select_cluster_20261104_0129/<model>/{tf/report.md, report_gen.md}      automatic selection
  results/dream_compare_a_20261105_1822/<model>/report_a.md, summary_a.md               selector comparison
  results/dream_compare_a_20261105_1822/llama32_3b/{tf,gen}/items/<id>                  worked example (turn, answers)
  results/dream_heads_20261111/<model>/{report.md, items/<id>.json.gz}                  head analysis
  experiments/dream_heads/RESULTS.md (section 2 table)                                  head analysis, exploratory
  results/dream_compare_b_20261105_1915/<model>/report_b.md                             memory-system comparison
"""

import glob
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SH = HERE.parents[2]                                   # Seahorse/
R = SH / "results"
DS = R / "dream_select_cluster_20261104_0129"
SA = R / "dream_compare_a_20261105_1822"
SB = R / "dream_compare_b_20261105_1915"
DH = R / "dream_heads_20261111"
EXAMPLE = ("llama32_3b", "4fd1909e")                  # single-session-user, KVzip-g judged correct, shared selectors not
NAMES = {"qwen25_15b": "Qwen2.5-1.5B", "qwen25_7b": "Qwen2.5-7B", "llama32_3b": "Llama-3.2-3B",
         "granite42_30b": "Granite-4.2-30B"}
NUM = r"(-?\d+\.\d+|-?\d+)"


def ci(s):
    """'0.63 [0.54, 0.71]' -> [0.63, 0.54, 0.71]; '-0.75 [-0.80, -0.69] (n 66)' too."""
    m = re.match(rf"\s*\+?{NUM}\s*\[\+?{NUM},\s*\+?{NUM}\]", s)
    return [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else None


def table(md, header_start, first_col=None):
    """Rows of the first markdown table after the line that starts with header_start: {first cell: [cells]}."""
    lines = md.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith(header_start))
    while not lines[i].startswith("|"):
        i += 1
    rows = {}
    for line in lines[i + 2:]:
        if not line.startswith("|"):
            break
        line = re.sub(r"(?<=[\w-])\|(?=k\d+ )", "\x00", line)      # "KVzip-g|k8" is one cell in the dream_heads reports
        cells = [c.strip().replace("\x00", "|") for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells
    return rows


# ------------------------------------------------------------------------------------------- automatic selection


def dream_select():
    out = {}
    for m in ["qwen25_15b", "qwen25_7b", "llama32_3b"]:
        tf = (DS / m / "tf" / "report.md").read_text()
        gen = (DS / m / "report_gen.md").read_text()
        # tf main table (all items): | condition | n | answer score | per-token | refusal | gap closed mean | median | hit |
        gold_tf = next(line for line in tf.splitlines() if line.startswith("| gold |"))
        c = [x.strip() for x in gold_tf.strip("|").split("|")]
        g = table(gen, "| condition | N | correct (majority)")
        out[m] = {"name": NAMES[m],
                  "gold_tf_gap": ci(c[5]),                   # teacher-forced gap closed, mean [CI]
                  "gold_judged": ci(g["gold"][2]),
                  "gold_judged_gap": ci(g["gold"][7]),                # judged gap closed, same 0-1 scale
                  "gold_refusal": ci(g["gold"][8]),
                  "whole_judged": ci(g["R-full"][2]),
                  "whole_refusal": ci(g["R-full"][8]),
                  "nomem_judged": ci(g["nomem"][2]),
                  "nomem_refusal": ci(g["nomem"][8])}
    return out


# ------------------------------------------------------------------------------------------- selector comparison

STAGE_A_SELS = ["nomem", "R-full", "KVzip-g@k8", "KVzip-ph@k8", "KVzip-sh@k8", "SnapKV-oracle-ph@k8", "EA-ph@k8",
                "SnapKV-dream-ph@k8", "S-loo@k8", "S-dream@k8", "S-recon@k8", "S-attn@k8", "H2O-ph@k8", "random@k8"]


def stage_a():
    out = {}
    for m in ["qwen25_15b", "qwen25_7b", "llama32_3b", "granite42_30b"]:
        md = (SA / m / "report_a.md").read_text()
        t = table(md, "## Judged correctness, k = 8")
        d = {"name": NAMES[m], "judged": {}, "refusal": {}}
        for s in STAGE_A_SELS:
            d["judged"][s] = ci(t[s][2])
            d["refusal"][s] = ci(t[s][8])
        h1 = re.search(r"\*\*H1\*\*.*?\*\*(HOLDS|does not hold)\*\*.*?Best KVzip form (\S+?):", md)
        h2 = re.search(r"\*\*H2\*\*.*?\*\*(HOLDS|does not hold)\*\* \((\d) of 3\): (.*)", md)
        d["H1"] = {"holds": h1.group(1) == "HOLDS", "best_form": h1.group(2)}
        d["H2"] = {"holds": h2.group(1) == "HOLDS", "n": int(h2.group(2)),
                   "pairs": {p: ci(v) for p, v in re.findall(r"(\S+ - \S+) (\S+ \[[^\]]+\])", h2.group(3))}}
        h4 = re.search(r"\*\*H4\*\*.*?best baseline (\S+) \(mean", md)
        d["best_baseline"] = h4.group(1)
        cost = re.search(r"Seconds per item \(fresh computations only\): (.*)\.", md).group(1)
        d["seconds"] = {k: float(v) for k, v in re.findall(r"([\w-]+) ([\d.]+)", cost)}
        d["E"] = int(re.search(r"Stored entries per position E = (\d+)", md).group(1))
        out[m] = d
    return out


# ------------------------------------------------------------------------------------------- head analysis


def dream_heads():
    res = (SH / "experiments" / "dream_heads" / "RESULTS.md").read_text()
    sec2 = res[res.index("## 2. Exploratory"):]
    t = table(sec2, "| | 1.5B | 7B | Llama |")
    cols = {"qwen25_15b": 1, "qwen25_7b": 2, "llama32_3b": 3}

    def first(s):
        return float(re.search(NUM, s.replace("**", "").replace("+", "")).group(1))

    out = {}
    for m, j in cols.items():
        md = (DH / m / "report.md").read_text()
        ha = re.search(r"H-A \(d_comply tracks refusal\): \*\*(\w+)\*\*, mean within-item Spearman (.*?) \(n", md)
        hb = re.search(r"H-B \(8 heads carry >= half\): \*\*(\w+)\*\*, pooled share (.*?);", md)
        jac = float(re.search(r"Jaccard between the folds' sets: (\d+\.\d+)", md).group(1))
        pc = table(md, "## Per condition")
        keep = table(md, "## What each selector keeps")
        hat = table(md, "## H-A: does the projection")
        causal = table(md, "## Exploratory causal check")
        conds = ["nomem", "R-full", "KVzip-g|k8", "KVzip-ph|k8", "KVzip-sh|k8", "S-loo|k8", "S-dream|k8", "random|k8"]
        def row(label):
            return next(v for k, v in t.items() if k.startswith(label))[j]
        wc = re.match(rf"{NUM} / {NUM} \((.*)\)", row("mass-weighted coverage: KVzip-g / best other").replace("*", ""))
        wc_g, wc_best = (wc.group(1), wc.group(2)), wc.group(3)
        out[m] = {
            "name": NAMES[m],
            "HA": {"holds": ha.group(1) == "holds", "rho_refusal": ci(ha.group(2)),
                   "rho_judged": ci(hat["d_comply (primary)"][2])},
            "HB": {"holds": hb.group(1) == "holds", "share": ci(hb.group(2)), "jaccard": jac},
            "lic_hide_refusal": ci(causal["R-full-lic - R-full"][1]),
            "ctrl_hide_refusal": ci(causal["R-full-ctrl - R-full"][1]),
            "lic_budget_vs_even": float(keep["KVzip-g"][3]),
            "kvzipg_layer_share": [float(x) for x in keep["KVzip-g"][1].split(" / ")],
            "cond": {c: {"refusal": float(pc[c][2]), "judged": float(pc[c][4]),
                         "lic_cov": float(pc[c][7]), "all_cov": float(pc[c][8])} for c in conds},
            "budget_vs_mass": first(row("KVzip-g budget vs R-full last-token memory attention")),
            "budget_vs_layer": first(row("KVzip-g budget vs layer index")),
            "wcov_kvzipg": float(wc_g[0]), "wcov_best": float(wc_g[1]), "wcov_best_name": wc_best,
            "wcov_vs_judged": ci(row("mass-weighted coverage vs judged").replace("**", "")),
            "wcov_vs_refusal": first(row("... vs refusal")),
        }
    return out


# ------------------------------------------------------------------------------------------- memory-system comparison


def stage_b():
    out = {}
    for m in ["qwen25_15b", "qwen25_7b", "llama32_3b", "granite42_30b"]:
        md = (SB / m / "report_b.md").read_text()
        pooled = md[md.index("## B1 pooled"):]
        d = {"name": NAMES[m]}
        for k in ["m3", "m10"]:
            j = json.loads(re.search(rf"- {k}: (\{{.*\}})", pooled).group(1))
            d[k] = {"diff": j["diff"], "lo95": j["lo95"], "hi95": j["hi95"], "lo90": j["lo90"], "hi90": j["hi90"],
                    "cells": j["cells"], "verdict": j["verdict"]}
        lme = md[md.index("## longmemeval_oracle"):]
        t = table(lme, "| cond | n | correct")
        d["lme_oracle_m3"] = {c: {"correct": ci(t[c][2]), "refusal": float(t[c][3]), "storage_mb": float(t[c][8]),
                                  "prefill": int(t[c][9]), "ttft": float(t[c][11])}
                              for c in ["P0", "P2-3", "P3-3", "P4dream-3"]}
        out[m] = d
    return out


# ------------------------------------------------------------------------------------------- worked example


def example():
    from transformers import AutoTokenizer
    sys.path.insert(0, str(SH / "experiments" / "dream_heads"))
    sys.path.insert(0, str(SH))
    import analyze as AN                                                             # noqa: E402

    m, q = EXAMPLE
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    a = json.load(gzip.open(SA / m / "tf" / "items" / f"{q}.json.gz"))
    h = json.load(gzip.open(DH / m / "items" / f"{q}.json.gz"))
    info, C = h["info"], h["conds"]
    turn = a["info"]["turn"]
    full = tok.apply_chat_template([{"role": "user", "content": turn}], tokenize=True)
    seg = full[info["n_head"]:info["n_head"] + info["n_seg"]]
    assert len(seg) == info["n_seg"] and len(info["content"]) == len(tok(turn, add_special_tokens=False)["input_ids"])
    toks = [tok.decode([t]) for t in seg]
    heads = json.load(open(DH / m / "heads.json"))
    L, nkv = heads["n_layers"], heads["n_kv"]
    sel = C["KVzip-g|k8"]["sel"]
    bud = [[len(sel[str(l)][k]) for k in range(nkv)] for l in range(L)]
    per_tok = {c: 0 for c in info["content"]}
    for l in range(L):
        for k in range(nkv):
            for c in sel[str(l)][k]:
                per_tok[c] += 1
    E = L * nkv
    assert sum(map(sum, bud)) == 8 * E

    recs = {}
    for f in sorted(glob.glob(str(DH / m / "items" / "*.json.gz"))):
        r = json.load(gzip.open(f))
        recs[r["info"]["question_id"]] = r
    J = AN.judged(str(SA / m), recs)
    judged = {c: J.get(c, {}).get(q) for c in ["R-full", "KVzip-g@k8", "KVzip-sh@k8", "S-loo@k8", "S-dream@k8",
                                               "KVzip-ph@k8"]}
    greedy = {}
    for line in gzip.open(SA / m / "gen" / "items" / f"{q}.jsonl.gz"):
        r = json.loads(line)
        if r.get("greedy") and r.get("cname") in ("R-full", "KVzip-g@k8", "KVzip-sh@k8", "S-loo@k8", "nomem"):
            greedy[r["cname"]] = r["answer"]
    return {"model": NAMES[m], "id": q, "question": info["question"], "answer": info["answer"],
            "n_layers": L, "n_kv": nkv, "E": E,
            "tokens": [toks[c] for c in info["content"]], "content": info["content"],
            "classes": info["classes"], "gold": info["gold_span"],
            "kvzipg_budget": bud, "kvzipg_per_token": [per_tok[c] for c in info["content"]],
            "shared": {s: C[f"{s}|k8"]["sel"] for s in ["KVzip-sh", "S-loo", "S-dream"]},
            "judged": judged, "greedy": greedy,
            "refusal": {c: C[c]["refusal"] for c in ["nomem", "R-full", "KVzip-g|k8", "KVzip-sh|k8", "S-loo|k8"]}}


def main():
    data = {"dream_select": dream_select(), "stage_a": stage_a(), "dream_heads": dream_heads(),
            "stage_b": stage_b(), "example": example()}
    txt = json.dumps(data, separators=(",", ":"))
    (HERE / "data.json").write_text(json.dumps(data, indent=1) + "\n")
    page = HERE / "index.html"
    html = page.read_text()
    new = re.sub(r"(<!-- DATA-BEGIN -->\n<script id=\"data\" type=\"application/json\">).*?(</script>\n<!-- DATA-END -->)",
                 lambda mm: mm.group(1) + txt + mm.group(2), html, flags=re.S)
    assert new != html or txt in html, "data markers not found in index.html"
    page.write_text(new)
    print(f"wrote data.json and index.html's data block ({len(txt):,} bytes)")


if __name__ == "__main__":
    main()
