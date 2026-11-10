"""Extract every number plotted in the 2026-11-10 catch-up page from the run outputs.

Run from the Seahorse repo root:
    python3 docs/catchup/2026-11-10/extract.py
Writes docs/catchup/2026-11-10/data.json. Pure standard library; reads only results/.
No number on the page is typed by hand: the page embeds data.json verbatim.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results"
OUT = Path(__file__).resolve().parent / "data.json"

NUM = re.compile(r"(-?\d+\.\d+|-?\d+)\s*\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\]")


def md_table(path, heading, nth=0):
    """Rows of the first markdown table after the line that equals `heading` (nth match)."""
    lines = Path(path).read_text().splitlines()
    hits = [i for i, l in enumerate(lines) if l.strip() == heading]
    i = hits[nth] + 1
    while not lines[i].startswith("|"):
        i += 1
    head = [c.strip() for c in lines[i].strip("|").split("|")]
    rows = {}
    i += 2
    while i < len(lines) and lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip("|").split("|")]
        rows[cells[0]] = dict(zip(head, cells))
        i += 1
    return rows


def ci(cell):
    m = NUM.search(cell or "")
    if not m:
        return None
    return {"mean": float(m.group(1)), "lo": float(m.group(2)), "hi": float(m.group(3))}


def f(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


data = {}

# ---- dream_subj (local, Qwen2.5-1.5B, teacher-forced, 45 person/pet facts) ----
p = R / "dream_subj_20261022_1755/report.md"
t = md_table(p, "## Per condition: all (45 facts)")
data["subj"] = {
    c: {
        "answer": ci(t[c]["answer score log P(prefix)"]),
        "refusal": ci(t[c]["P('I'+'Sorry') first"]),
        "yesno": ci(t[c]["TF yes/no balanced"]),
        "target": ci(t[c]["target gain, direct (nats)"]),
    }
    for c in ["nomem", "T-min", "T-subj3", "R-min", "R+fill", "R+rel", "R-subj2", "R+subj", "R-full", "ctx"]
}
tc = md_table(p, "### all")
data["subj_contrasts"] = {k: ci(v["answer score"]) for k, v in tc.items()}

# ---- dream_select: teacher-forced gap closed vs judged correctness (k = 4 and 8) ----
DS = R / "dream_select_cluster_20261104_0129"
SEL_MODELS = [("qwen25_15b", "Qwen2.5-1.5B"), ("qwen25_7b", "Qwen2.5-7B"), ("llama32_3b", "Llama-3.2-3B")]
data["select"] = {}
for key, label in SEL_MODELS:
    tf = DS / key / "tf/report.md"
    main = md_table(tf, "### all (64 items)")
    gap = md_table(tf, "### Gap closed (mean)")
    ref = md_table(tf, "### Refusal mass at the first token")
    g = json.loads((DS / key / "verdict_gen.json").read_text())
    rows = {}
    for cond in ["S-dream", "S-attn", "S-recon", "S-loo", "random", "gold@k"]:
        for k in ("k4", "k8"):
            gk = f"{cond}@{k}"
            rows[gk] = {
                "tf_gap": ci(gap[cond][k]),
                "tf_refusal": ci(ref[cond][k]),
                "correct": g["correct"].get(gk),
                "gen_gap": g["gap_closed"].get(gk),
                "refusal": g["refusal"].get(gk),
            }
    for cond in ["gold", "R-full", "nomem", "ctx"]:
        rows[cond] = {
            "tf_gap": ci(main[cond]["gap closed mean"]),
            "tf_refusal": ci(main[cond]["refusal P(I+Sorry+As) first"]),
            "correct": g["correct"].get(cond),
            "gen_gap": g["gap_closed"].get(cond),
            "refusal": g["refusal"].get(cond),
        }
    data["select"][key] = {"label": label, "rows": rows, "rule": g["rule"]}
sj = json.loads((DS / "summary.json").read_text())
data["select_summary"] = {"verdict": sj["verdict"], "held": sj["held"], "n_done": sj["n_done"]}
# local 1.5B punctuation share (post hoc)
pt = md_table(R / "dream_select_20261104_0032/report.md", "## What the selectors keep (post hoc, not pre-registered)")
data["select_punct"] = {k: f(v["punctuation share of picks"]) for k, v in pt.items()}

# ---- dream_compare Stage A ----
A = R / "dream_compare_a_20261105_1822"
A_MODELS = [("qwen25_15b", "Qwen2.5-1.5B"), ("qwen25_7b", "Qwen2.5-7B"), ("llama32_3b", "Llama-3.2-3B"), ("granite42_30b", "Granite-4.2-30B")]
A_CONDS = ["S-dream", "S-attn", "S-recon", "S-loo", "random", "gold@k", "KVzip-g", "KVzip-ph", "KVzip-sh",
           "SnapKV-dream-ph", "SnapKV-dream-sh", "SnapKV-oracle-ph", "H2O-ph", "H2O-sh", "VATP-ph", "EA-ph",
           "StreamingLLM", "S-dream-VATP"]
data["stageA"] = {}
for key, label in A_MODELS:
    v = json.loads((A / key / "verdict_a.json").read_text())
    rep = A / key / "report_a.md"
    j8 = md_table(rep, "## Judged correctness, k = 8 (primary set)")
    rows = {}
    for c in A_CONDS:
        ck = f"{c}@k8"
        rows[c] = {
            "correct": v["correct_primary"].get(ck),
            "gen_gap": v["gap_closed_primary"].get(ck),
            "tf_gap": v["tf_gap_primary"].get(ck),
            "refusal": ci(j8[ck]["refusal"]),
        }
    for c in ["nomem", "ctx", "R-full", "R-content", "gold"]:
        rows[c] = {"correct": v["correct_primary"].get(c), "refusal": ci(j8[c]["refusal"])}
    data["stageA"][key] = {"label": label, "rows": rows, "rules": v["rules"], "n_primary": v["n_primary"]}
sa = json.loads((A / "summary_a.json").read_text())
data["stageA_summary"] = {k: sa[k] for k in ("H3", "n_pass_H3", "n_H1", "n_H2", "n_done")}

# ---- dream_compare Stage B (partial: models with an analysed report) ----
B = R / "dream_compare_b_20261105_1915"
B_MODELS = [("qwen25_15b", "Qwen2.5-1.5B"), ("qwen25_7b", "Qwen2.5-7B"), ("llama32_3b", "Llama-3.2-3B")]
data["stageB"] = {}
for key, label in B_MODELS:
    vp = B / key / "verdict_b.json"
    if not vp.exists():
        continue
    v = json.loads(vp.read_text())
    benches = {}
    for bn, b in v["benches"].items():
        benches[bn] = {
            "n_items": b["n_items"],
            "table": {c: {k: r.get(k) for k in ("n", "correct", "lo", "hi", "refusal", "storage_bytes",
                                                 "prefill_tokens", "restored_entries", "ttft_s_median", "recall",
                                                 "abstention", "knowledge_update")}
                      for c, r in b["table"].items()},
            "rules": b["rules"],
        }
    data["stageB"][key] = {"label": label, "benches": benches, "pooled": v["rules"].get("B1_pooled")}

OUT.write_text(json.dumps(data, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

# ---- build the page: embed data.json into template.html -> catchup.html ----
TPL = OUT.parent / "template.html"
if TPL.exists():
    html = TPL.read_text().replace("/*DATA*/{}", json.dumps(data, separators=(",", ":")))
    (OUT.parent / "catchup.html").write_text(html)
    print(f"wrote {OUT.parent / 'catchup.html'}")
