#!/usr/bin/env python
"""Seahorse diag_order: is it write ORDER that decides who survives in a combined memory?

A measurement pass. The method is unchanged (the v0.1 pipeline with gate=entropy,
position=all, layers 23/26, alpha=2, baselines without + contrastive); only the set and
order of writes vary. Each condition builds a fresh memory:

  iso                  one memory per scenario: the reference for retention
  a_comb_orig          combined, original order veg > peanut > norway > dog > sister > job.
                       Must reproduce v0.1 (checked against its results.jsonl).
  b_comb_rev           combined, reversed order job > sister > dog > norway > peanut > veg
  c_veg_<x>            vegetarian written first, then one interferer x in
                       {peanut_allergy (food), dog_name (non-food fact), norway (non-food
                       disposition)}
  d_comb_notail        combined, original order, but the writes skip the template tail
                       (end-of-turn + assistant header, the same 5 tokens at the end of every
                       follow-up). The kept tokens keep exactly the gate values they had in (a).
  d_comb_notail_renorm as (d), but the entropy gate is renormalised over the kept tokens
  iso_notail, c_veg_<x>_notail   references for the no-tail variants

Plus, without extra forward passes:
  (e) forgetting curves: after each scenario's writes, the recall fidelity of every
      scenario at its own stored write keys (mean cos(M k_t, delta_t) and mean
      |M k_t - delta_t| / |delta_t| over its write tokens)
  (f) key overlap: mean cosine between the write keys of each pair of scenarios, per layer,
      split into template-tail and content tokens

Outputs: results.jsonl, summary.csv, forgetting.csv, key_overlap.csv, report.txt, config.json
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import transformers
import yaml

from seahorse import metrics as mx
from seahorse.memory import FastWeightMemory
from seahorse.residual import load_model
from seahorse.sessions import chat_ids, common_suffix_len

HERE = Path(__file__).resolve().parent
V01_DIR = HERE.parent / "v0_1"
ORIG_ORDER = ["vegetarian", "peanut_allergy", "norway", "dog_name", "sister_name", "job"]
VEG = "vegetarian"
INTERFERERS = ["peanut_allergy", "dog_name", "norway"]
# |isolated effect| below these is treated as ~0: retention is not computed (flagged instead)
EPS = {"gap": 0.05, "dtarget": 0.5, "spec": 0.5, "drel": 0.3}
SHORT = {"vegetarian": "veg", "peanut_allergy": "peanut", "norway": "norway",
         "dog_name": "dog", "sister_name": "sister", "job": "job"}


def _load_v01():
    """The v0.1 pipeline (collect_writes, select, build_memory, reference, evaluate, ...)."""
    spec = importlib.util.spec_from_file_location("seahorse_v0_1_run", V01_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v01 = _load_v01()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[23, 26])
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--baselines", nargs="+", default=["without", "contrastive"],
                   choices=["without", "contrastive"])
    p.add_argument("--scenarios", default=str(V01_DIR / "scenarios.yaml"))
    p.add_argument("--generic", default=str(v01.GENERIC))
    p.add_argument("--cont-len", type=int, default=20)
    p.add_argument("--v01-results",
                   default=f"/scratch/{os.environ.get('USER', 'unknown')}/seahorse_runs/v0_1_415909/results.jsonl",
                   help="v0.1 results.jsonl that condition (a) and iso must reproduce")
    p.add_argument("--repro-tol", type=float, default=1e-2,
                   help="max abs difference (nats) allowed against v0.1")
    p.add_argument("--tiny", action="store_true",
                   help="smoke test: tiny random Qwen2 with --model's tokenizer; no reproduction check")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(args):
    if not args.tiny:
        return load_model(args.model, device=args.device)
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=max(args.layers) + 2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    model = Qwen2ForCausalLM(cfg).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ------------------------------------------------------------------ conditions


def conditions(ids):
    """(name, family, members in write order, write variant)."""
    out = [("iso", "iso", [s], "all") for s in ids]
    out += [("iso_notail", "iso", [s], "notail") for s in ids]
    out.append(("a_comb_orig", "comb", list(ids), "all"))
    out.append(("b_comb_rev", "comb", list(ids)[::-1], "all"))
    for x in INTERFERERS:
        out.append((f"c_veg_{SHORT[x]}", "pair", [VEG, x], "all"))
        out.append((f"c_veg_{SHORT[x]}_notail", "pair", [VEG, x], "notail"))
    out.append(("d_comb_notail", "comb", list(ids), "notail"))
    out.append(("d_comb_notail_renorm", "comb", list(ids), "notail_renorm"))
    return out


def select(chunk, layer, variant, baseline, tail_len):
    """v0.1's `all`/entropy selection, optionally without the template tail.

    Returns (delta, h_key, gate, is_tail) for one follow-up."""
    delta, h_key, g = v01.select(chunk, layer, "all", baseline, "entropy", tail_len, 4)
    n = delta.shape[0]
    assert n > tail_len, "follow-up has no content tokens before the template tail"
    is_tail = torch.zeros(n, dtype=torch.bool)
    is_tail[n - tail_len:] = True
    if variant == "all":
        return delta, h_key, g, is_tail
    keep = n - tail_len
    delta, h_key, g = delta[:keep], h_key[:keep], g[:keep]
    if variant == "notail_renorm":
        ent = chunk["entropy"][:keep]
        g = ent / ent.max().clamp_min(1e-8)
    elif variant != "notail":
        raise ValueError(variant)
    return delta, h_key, g, is_tail[:keep]


def fidelity(mem, chunks):
    """Recall of stored (delta, key) pairs: mean cos(M k, delta), mean |M k - delta|/|delta|."""
    cos, rel, tail = [], [], []
    for delta, h_key, _, is_tail in chunks:
        rec = mem.predict(mem.key(h_key))
        cos.append(F.cosine_similarity(rec, delta, dim=-1).cpu())
        rel.append(((rec - delta).norm(dim=-1) / delta.norm(dim=-1).clamp_min(1e-8)).cpu())
        tail.append(is_tail)
    cos, rel, tail = torch.cat(cos), torch.cat(rel), torch.cat(tail)
    out = {"n_tokens": len(cos), "cos": cos.mean().item(), "relerr": rel.mean().item()}
    for name, m in (("content", ~tail), ("tail", tail)):
        out[f"cos_{name}"] = cos[m].mean().item() if m.any() else float("nan")
        out[f"relerr_{name}"] = rel[m].mean().item() if m.any() else float("nan")
    return out


def build(mu_l, stored, members, device):
    """Sequential delta-rule writes, scenario by scenario (same calls as v0.1's
    build_memory). Records the forgetting curve after each scenario's writes."""
    mem = FastWeightMemory(mu_l.shape[0], mu_l, device=device)
    curve = []
    for i, s in enumerate(members):
        for delta, h_key, g, _ in stored[s]:
            mem.write(delta, h_key, gate=g, eta=1.0)
        for j, t in enumerate(members):
            curve.append({"step": i, "after": s, "scenario": t, "written": j <= i, **fidelity(mem, stored[t])})
    return mem, curve


# --------------------------------------------------------------------- summary


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def scenario_effects(rs, scen_type):
    probes = [r for r in rs if r["kind"] == "probe"]
    rels = [r for r in rs if r["kind"] == "relation"]
    kb, km = sum(r["kl_base"] for r in probes), sum(r["kl_mem"] for r in probes)
    out = {"gap": 1 - km / kb if kb > 0 else float("nan"),
           "dtarget": mean(r["tgt_mem"] - r["tgt_base"] for r in probes),
           "dtarget_ceil": mean(r["tgt_ceil"] - r["tgt_base"] for r in probes),
           "dfoil": float("nan"), "spec": float("nan"), "drel": float("nan"), "drel_ceil": float("nan")}
    if scen_type == "fact":
        out["dfoil"] = mean(mean(m - b for m, b in zip(r["foil_mem"], r["foil_base"])) for r in probes)
        out["spec"] = out["dtarget"] - out["dfoil"]
    if rels:
        out["drel"] = mean(r["rel_mem"] - r["rel_base"] for r in rels)
        out["drel_ceil"] = mean(r["rel_ceil"] - r["rel_base"] for r in rels)
    return out


def retention(x, ref, eps):
    """(ratio, flag). flag is '' when computed, else why not."""
    if ref is None or math.isnan(ref) or math.isnan(x):
        return float("nan"), "na"
    if abs(ref) < eps:
        return float("nan"), "iso~0"
    if ref < 0:
        return float("nan"), "iso<0"
    return x / ref, ""


def summarize(rows, leak, types):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["baseline"], r["layer"], r["scenario"])].append(r)
    eff = {k: scenario_effects(rs, types[k[3]]) for k, rs in groups.items()}
    out = []
    for (cond, base, layer, sid), e in sorted(eff.items()):
        rs = groups[(cond, base, layer, sid)]
        row = {"condition": cond, "baseline": base, "layer": layer, "scenario": sid, "type": types[sid],
               "write_index": rs[0]["write_index"], "n_members": rs[0]["n_members"], **e,
               "leak": mean(leak[(cond, base, layer, rs[0]["memory"])])}
        for ref_name in ("iso", "iso_notail"):
            ref = eff.get((ref_name, base, layer, sid))
            suffix = "" if ref_name == "iso" else "_vs_iso_notail"
            for m in ("gap", "dtarget", "spec", "drel"):
                val, flag = retention(e[m], ref[m] if ref else None, EPS[m])
                row[f"ret_{m}{suffix}"] = val
                row[f"flag_{m}{suffix}"] = flag
        out.append(row)
    return out


# ------------------------------------------------------------ reproduction check


def repro_check(path, rows, leak, alpha, tol):
    """Compare iso and a_comb_orig against the v0.1 run (same config there)."""
    if not Path(path).exists():
        return {"status": "SKIPPED", "reason": f"{path} not found"}
    ref = {}
    for line in open(path):
        r = json.loads(line)
        if r["gate"] == "entropy" and r["position"] == "all" and r["alpha"] == alpha:
            ref[(r["mode"], r["baseline"], r["layer"], r["memory"], r["kind"], r["probe"])] = r
    fields = ["kl_base", "kl_mem", "tgt_base", "tgt_ceil", "tgt_mem", "foil_base", "foil_ceil",
              "foil_mem", "rel_base", "rel_ceil", "rel_mem"]
    n, missing, maxdiff, worst = 0, 0, 0.0, None
    for r in rows:
        if r["condition"] not in ("iso", "a_comb_orig"):
            continue
        mode = "isolated" if r["condition"] == "iso" else "combined"
        memory = r["scenario"] if mode == "isolated" else "combined"
        key = (mode, r["baseline"], r["layer"], memory, r["kind"], r["probe"])
        if key not in ref:
            missing += 1
            continue
        n += 1
        for f in fields:
            if f not in r:
                continue
            a, b = r[f], ref[key][f]
            pairs = zip(a, b) if isinstance(a, list) else [(a, b)]
            for x, y in pairs:
                d = abs(x - y)
                if d > maxdiff:
                    maxdiff, worst = d, (key, f, x, y)
    # leakage: v0.1 summary.csv holds the mean per (mode, baseline, layer, alpha)
    leak_note = ""
    summ = Path(path).with_name("summary.csv")
    if summ.exists():
        ref_leak = {}
        for r in csv.DictReader(open(summ)):
            if r["gate"] == "entropy" and r["position"] == "all" and float(r["alpha"]) == alpha:
                ref_leak[(r["mode"], r["baseline"], int(r["layer"]))] = float(r["leak"])
        ld = []
        for (mode, cond) in (("isolated", "iso"), ("combined", "a_comb_orig")):
            by = defaultdict(list)
            for (c, base, layer, mem), vals in leak.items():
                if c == cond:
                    by[(base, layer)] += vals
            for (base, layer), vals in by.items():
                if (mode, base, layer) in ref_leak:
                    ld.append(abs(mean(vals) - ref_leak[(mode, base, layer)]))
        if ld:
            leak_note = f"max |leak - v0.1 leak| = {max(ld):.2e} over {len(ld)} groups"
            maxdiff = max(maxdiff, max(ld))
    status = "PASS" if (n > 0 and missing == 0 and maxdiff <= tol) else "FAIL"
    return {"status": status, "rows_compared": n, "rows_missing": missing, "max_abs_diff": maxdiff,
            "tol": tol, "worst": str(worst), "leak": leak_note, "reference": str(path)}


# --------------------------------------------------------------------- analysis


def key_overlap(mu, writes, layers, ids, tail_len):
    """Mean cosine between write keys of scenario pairs; tail vs content tokens.

    Classes: content (content x content), tail_same (tail token at the same offset in
    both follow-ups), tail_all (any tail x tail). For a == b, identical tokens are excluded."""
    out = []
    for layer in layers:
        K, TL, OFF = {}, {}, {}
        for s in ids:
            ks, tl, off = [], [], []
            for ch in writes[s]:
                h_wo = ch["h"][layer][1]
                n = h_wo.shape[0]
                ks.append(F.normalize(h_wo - mu[layer], dim=-1, eps=1e-6))
                t = torch.zeros(n, dtype=torch.bool)
                t[n - tail_len:] = True
                o = torch.full((n,), -1)
                o[n - tail_len:] = torch.arange(tail_len)
                tl.append(t)
                off.append(o)
            K[s], TL[s], OFF[s] = torch.cat(ks).cpu(), torch.cat(tl), torch.cat(off)
        for i, a in enumerate(ids):
            for b in ids[i:]:
                C = K[a] @ K[b].T
                not_self = ~torch.eye(len(K[a]), dtype=torch.bool) if a == b else torch.ones_like(C, dtype=torch.bool)
                masks = {
                    "content": (~TL[a])[:, None] & (~TL[b])[None, :],
                    "tail_same": TL[a][:, None] & TL[b][None, :] & (OFF[a][:, None] == OFF[b][None, :]),
                    "tail_all": TL[a][:, None] & TL[b][None, :],
                }
                for cls, m in masks.items():
                    m = m & not_self
                    out.append({"layer": layer, "a": a, "b": b, "cls": cls,
                                "mean_cos": C[m].mean().item() if m.any() else float("nan"),
                                "n_pairs": int(m.sum())})
    return out


def gate_stats(writes, ids, tail_len):
    out = []
    for s in ids:
        for ci, ch in enumerate(writes[s]):
            ent = ch["entropy"].cpu()
            g = ent / ent.max().clamp_min(1e-8)
            out.append({"scenario": s, "followup_idx": ci, "n": len(ent),
                        "argmax_in_tail": bool(int(ent.argmax()) >= len(ent) - tail_len),
                        "gate_tail_mean": g[-tail_len:].mean().item(),
                        "gate_content_mean": g[:-tail_len].mean().item(),
                        "gate_share_tail": (g[-tail_len:].sum() / g.sum()).item()})
    return out


# ----------------------------------------------------------------------- report


def fmt(x, spec="+.3f"):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return format(x, spec)


def table(header, rows):
    rows = [[str(c) for c in r] for r in rows]
    header = [str(h) for h in header]
    w = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def ret_cell(row, m, suffix=""):
    flag = row[f"flag_{m}{suffix}"]
    return flag if flag else fmt(row[f"ret_{m}{suffix}"], "+.2f")


def report(summary, curves, overlap, gates, repro, args, ids, types):
    S = {(r["condition"], r["baseline"], r["layer"], r["scenario"]): r for r in summary}
    L = []
    w = L.append
    w("Seahorse diag_order report")
    w("=" * 26)
    w(f"write: gate=entropy position=all; alpha={args.alpha}; layers={args.layers}; baselines={args.baselines}")
    w(f"original order: {' > '.join(SHORT[s] for s in ids)}")
    w(f"retention = condition effect / isolated effect. 'iso~0' = |isolated effect| < "
      f"{EPS} (not divided); 'iso<0' = isolated effect negative (not divided).")
    w("gap = 1 - sum KL_mem / sum KL_base over the scenario's 3 probes; dtarget = mean d logP(target) "
      "(facts) or contrast (dispositions); spec = dtarget - dfoils; drel = mean d(relation score).\n")
    w(f"REPRODUCTION CHECK (iso + a_comb_orig vs v0.1): {repro['status']}")
    for k, v in repro.items():
        if k != "status":
            w(f"  {k}: {v}")
    w("")

    comb_conds = ["a_comb_orig", "b_comb_rev", "d_comb_notail", "d_comb_notail_renorm"]
    pos = {"a_comb_orig": ids, "b_comb_rev": ids[::-1], "d_comb_notail": ids, "d_comb_notail_renorm": ids}
    for base in args.baselines:
        for layer in args.layers:
            w(f"--- baseline={base} layer={layer} " + "-" * 40)
            for m, label in (("gap", "gap closed"), ("dtarget", "d target"), ("spec", "specificity (facts)"),
                             ("drel", "d relation (dispositions)")):
                hdr = ["scenario", "iso", "iso_notail"] + comb_conds
                raw, ret = [], []
                for s in ids:
                    if m == "spec" and types[s] != "fact":
                        continue
                    if m == "drel" and types[s] != "disposition":
                        continue
                    cells = [fmt(S.get((c, base, layer, s), {}).get(m)) for c in ["iso", "iso_notail"] + comb_conds]
                    raw.append([SHORT[s]] + cells)
                    rc = ["", ""] + [ret_cell(S[(c, base, layer, s)], m) + f" (#{pos[c].index(s) + 1})" for c in comb_conds]
                    ret.append([SHORT[s]] + rc)
                w(f"{label}: raw")
                w(table(hdr, raw))
                w(f"{label}: retention vs iso (#k = write position)")
                w(table(hdr, ret))
                if m == "gap":
                    rr = [[SHORT[s], ret_cell(S[(c, base, layer, s)], m, "_vs_iso_notail")] for s in ids
                          for c in ["d_comb_notail"]]
                    w("gap retention of d_comb_notail vs iso_notail: " + ", ".join(f"{a}={b}" for a, b in rr))
                w("")
            # (c) pairs
            w("(c) vegetarian written first, then one interferer")
            hdr = ["condition", "veg gap", "ret", "veg dtarget", "ret", "veg drel", "ret", "interferer gap", "ret"]
            rows = []
            for c in ["iso"] + [f"c_veg_{SHORT[x]}{v}" for x in INTERFERERS for v in ("", "_notail")]:
                r = S[(c, base, layer, VEG)]
                if c == "iso":
                    rows.append([c, fmt(r["gap"]), "", fmt(r["dtarget"]), "", fmt(r["drel"]), "", "", ""])
                    continue
                x = next(x for x in INTERFERERS if f"_{SHORT[x]}" in c)
                rx = S[(c, base, layer, x)]
                suffix = "_vs_iso_notail" if c.endswith("_notail") else ""
                rows.append([c, fmt(r["gap"]), ret_cell(r, "gap", suffix), fmt(r["dtarget"]),
                             ret_cell(r, "dtarget", suffix), fmt(r["drel"]), ret_cell(r, "drel", suffix),
                             f"{SHORT[x]} {fmt(rx['gap'])}", ret_cell(rx, "gap", suffix)])
            w(table(hdr, rows))
            w("(notail rows: retention vs iso_notail)\n")
            # leakage
            leaks = defaultdict(list)
            for r in summary:
                if r["baseline"] == base and r["layer"] == layer:
                    leaks[r["condition"]].append(r["leak"])
            w("leakage (mean KL on unrelated probes; iso = mean over the 6 memories): " +
              ", ".join(f"{c}={mean(v):.4f}" for c, v in leaks.items()))
            w("")

    w("=" * 60)
    w("(e) FORGETTING: recall fidelity at each scenario's own stored write keys")
    w("rows = after scenario i's writes; cols = scenario j (written so far only).")
    w("cos = mean cos(M k_t, delta_t); relerr = mean |M k_t - delta_t|/|delta_t|; 1.0 relerr = nothing recalled.")
    w("Reference 'iso' row = fidelity of each isolated memory right after its own writes.\n")
    C = defaultdict(dict)
    for r in curves:
        C[(r["condition"], r["baseline"], r["layer"])][(r["step"], r["scenario"])] = r
    for base in args.baselines:
        for layer in args.layers:
            for cond in comb_conds:
                cur = C[(cond, base, layer)]
                order = pos[cond]
                for metric in ("cos", "relerr", "cos_content", "cos_tail"):
                    if metric == "cos_tail" and "notail" in cond:
                        continue
                    hdr = [f"{cond} {base} L{layer} {metric}"] + [SHORT[s] for s in ids]
                    rows = []
                    for i, s in enumerate(order):
                        rows.append([f"after {SHORT[s]}"] + [
                            fmt(cur[(i, t)][metric], ".3f") if cur[(i, t)]["written"] else "." for t in ids])
                    iso_cond = "iso_notail" if "notail" in cond else "iso"
                    isorow = []
                    for t in ids:
                        rr = [r for r in curves if r["condition"] == iso_cond and r["baseline"] == base
                              and r["layer"] == layer and r["scenario"] == t]
                        isorow.append(fmt(rr[0][metric], ".3f") if rr else "-")
                    rows.append([f"({iso_cond})"] + isorow)
                    w(table(hdr, rows))
                    w("")

    w("=" * 60)
    w("(f) KEY OVERLAP: mean cosine between write keys of scenario pairs (keys = normalize(h_without - mu))")
    w("content = follow-up tokens; tail_same = the same template-tail token in both; diagonal excludes a token with itself.\n")
    O = {(r["layer"], r["a"], r["b"], r["cls"]): r["mean_cos"] for r in overlap}
    for layer in args.layers:
        for cls in ("content", "tail_same", "tail_all"):
            hdr = [f"L{layer} {cls}"] + [SHORT[s] for s in ids]
            rows = []
            for i, a in enumerate(ids):
                rows.append([SHORT[a]] + [fmt(O.get((layer, a, b, cls), O.get((layer, b, a, cls))), ".3f")
                                          for b in ids])
            w(table(hdr, rows))
            w("")
    w("gate (entropy) statistics per follow-up (same for all layers):")
    w(table(["scenario", "fu", "n", "argmax in tail", "mean g tail", "mean g content", "tail share of sum g"],
            [[SHORT[g["scenario"]], g["followup_idx"], g["n"], g["argmax_in_tail"], fmt(g["gate_tail_mean"], ".2f"),
              fmt(g["gate_content_mean"], ".2f"), fmt(g["gate_share_tail"], ".2f")] for g in gates]))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------- main


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.scenarios))
    scenarios, unrelated = cfg["scenarios"], cfg["unrelated_probes"]
    ids = [s["id"] for s in scenarios]
    assert ids == ORIG_ORDER, f"unexpected scenario order {ids}"
    by_id = {s["id"]: s for s in scenarios}
    types = {s["id"]: s["type"] for s in scenarios}
    generic = [l.strip() for l in open(args.generic) if l.strip()]
    bargs = SimpleNamespace(device=args.device, topk=4, eta=1.0, cont_len=args.cont_len)

    log(f"loading {args.model} on {args.device}{' (tiny random model)' if args.tiny else ''}")
    model, tok = load(args)
    v01.tok_global = tok
    tail_len = common_suffix_len(chat_ids(tok, "alpha"), chat_ids(tok, "beta"))

    rows, curves = [], []
    leak = defaultdict(list)
    with torch.inference_mode():
        log(f"mu over {len(generic)} generic prompts, layers {args.layers}")
        mu, skip = v01.compute_mu(model, tok, generic, args.layers, args.device)
        log("session 1: collecting writes (with / without / counter)")
        writes = {s["id"]: v01.collect_writes(model, tok, s, args.layers, args.device) for s in scenarios}
        log("references (no memory)")
        refs = {s["id"]: v01.reference(model, tok, s, bargs) for s in scenarios}
        unrel = []
        for text in unrelated:
            u_ids = chat_ids(tok, text)
            cont = mx.greedy(model, tok, u_ids, args.cont_len, args.device)
            unrel.append({"ids": u_ids, "cont": cont,
                          "lp_base": mx.cont_logprobs(model, u_ids, cont, args.device)})

        conds = conditions(ids)
        for base in args.baselines:
            for layer in args.layers:
                stored = {v: {s: [select(ch, layer, v, base, tail_len) for ch in writes[s]] for s in ids}
                          for v in ("all", "notail", "notail_renorm")}
                for name, family, members, variant in conds:
                    mem, curve = build(mu[layer], stored[variant], members, args.device)
                    if variant == "all":  # identical to the v0.1 builder
                        ref_mem, _, _ = v01.build_memory(mu[layer], writes, [by_id[s] for s in members], layer,
                                                         "all", base, "entropy", bargs, tail_len)
                        assert torch.equal(ref_mem.M, mem.M), f"{name}: memory differs from v0.1 build_memory"
                    memory = members[0] if family == "iso" else name
                    curves += [{"condition": name, "baseline": base, "layer": layer, **c} for c in curve]
                    for wi, s in enumerate(members):
                        tag = {"condition": name, "baseline": base, "layer": layer, "alpha": args.alpha,
                               "memory": memory, "members": members, "n_members": len(members),
                               "write_index": wi, "variant": variant}
                        rows += [{**tag, **r} for r in
                                 v01.evaluate(model, by_id[s], refs[s], mem, layer, args.alpha, args.device)]
                    leak[(name, base, layer, memory)] += v01.eval_leakage(model, unrel, mem, layer, args.alpha,
                                                                         args.device)
                log(f"done baseline={base} layer={layer} ({len(conds)} memories)")

        overlap = key_overlap(mu, writes, args.layers, ids, tail_len)
        gates = gate_stats(writes, ids, tail_len)

    summary = summarize(rows, leak, types)
    if args.tiny:
        repro = {"status": "SKIPPED", "reason": "--tiny"}
    else:
        repro = repro_check(args.v01_results, rows, leak, args.alpha, args.repro_tol)
    log(f"reproduction check: {repro}")

    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for name, data in (("summary.csv", summary), ("forgetting.csv", curves), ("key_overlap.csv", overlap)):
        with open(out_dir / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            wr.writeheader()
            wr.writerows(data)
    (out_dir / "report.txt").write_text(report(summary, curves, overlap, gates, repro, args, ids, types))
    json.dump({**vars(args), "mu_skip_tokens": skip, "tail_len": tail_len, "eps": EPS,
               "conditions": [{"name": n, "members": m, "variant": v} for n, _, m, v in conditions(ids)],
               "reproduction": repro, "gate_stats": gates,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              open(out_dir / "config.json", "w"), indent=2)
    log(f"wrote {len(rows)} rows, {len(curves)} forgetting rows to {out_dir}")
    assert repro["status"] != "FAIL", f"condition (a) / iso do not reproduce v0.1: {repro}"


if __name__ == "__main__":
    main()
