#!/usr/bin/env python
"""Seahorse v0.1: follow-up to v0 (job 408770).

Adds four probes of the mechanism on top of the v0 two-session experiment:
  1. specificity   - fact foils: does the memory raise *this* name or any name?
  2. relations     - yes/no relation probes, plus a contrastive write baseline
                     (delta = h(experience) - h(counter-experience)) that cancels the
                     shared concept and keeps the relation/identity
  3. key granularity - write positions: all tokens / boundary tokens / top-k entropy
                     tokens / one pooled write per follow-up
  4. behaviour     - greedy samples at the best v0 settings for every variant
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
import yaml

from seahorse import metrics as mx
from seahorse.memory import FastWeightMemory
from seahorse.residual import capture, inject, load_model
from seahorse.sessions import (
    ceiling_ids, chat_ids, common_prefix_len, common_suffix_len, text_ids,
)

HERE = Path(__file__).parent
GENERIC = HERE.parent / "v0" / "generic_prompts.txt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[20, 23, 26])
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--positions", nargs="+", default=["all", "boundary", "topk", "pooled"],
                   choices=["all", "boundary", "topk", "pooled"])
    p.add_argument("--baselines", nargs="+", default=["without", "contrastive"],
                   choices=["without", "contrastive"])
    p.add_argument("--modes", nargs="+", default=["isolated", "combined"],
                   choices=["isolated", "combined"])
    p.add_argument("--gates", nargs="+", default=["entropy"], choices=["none", "entropy"])
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--scenarios", default=str(HERE / "scenarios.yaml"))
    p.add_argument("--generic", default=str(GENERIC))
    p.add_argument("--cont-len", type=int, default=20)
    p.add_argument("--sample-len", type=int, default=40)
    p.add_argument("--sample-alpha", type=float, default=2.0)
    p.add_argument("--sample-layer-disposition", type=int, default=23)
    p.add_argument("--sample-layer-fact", type=int, default=26)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_mu(model, tok, prompts, layers, device):
    seqs = [chat_ids(tok, p) for p in prompts]
    skip = min(common_prefix_len(seqs[0], s) for s in seqs[1:])
    sums, count = {l: 0.0 for l in layers}, 0
    for ids in seqs:
        with capture(model, layers) as store:
            model(ids[None].to(device))
        for l in layers:
            sums[l] = sums[l] + store[l][skip:].sum(0)
        count += len(ids) - skip
    return {l: sums[l] / count for l in layers}, skip


# -------------------------------------------------------------------------- write


def collect_writes(model, tok, scen, layers, device):
    """Per follow-up: residuals on the shared suffix for the with / without /
    counter runs, and next-token entropy from the without run."""
    chunks = []
    for fu in scen["followups"]:
        w = chat_ids(tok, f"{scen['experience']} {fu}")
        wo = chat_ids(tok, fu)
        c = chat_ids(tok, f"{scen['counter']} {fu}")
        n = min(common_suffix_len(w, wo), common_suffix_len(w, c))
        assert n > 0
        with capture(model, layers) as s_w:
            model(w[None].to(device))
        with capture(model, layers) as s_wo:
            out = model(wo[None].to(device))
        with capture(model, layers) as s_c:
            model(c[None].to(device))
        chunks.append({
            "followup": fu,
            "n": n,
            "h": {l: (s_w[l][-n:], s_wo[l][-n:], s_c[l][-n:]) for l in layers},
            "entropy": mx.entropy(out.logits[0, -n:]),
        })
    return chunks


def select(chunk, layer, position, baseline, gate, tail_len, topk):
    """Return (delta [T,d], h_key [T,d], gate [T] or None) for one follow-up."""
    h_with, h_without, h_counter = chunk["h"][layer]
    delta = h_with - (h_without if baseline == "without" else h_counter)
    h_key = h_without  # future sessions look like the without run
    ent = chunk["entropy"]
    g = ent / ent.max().clamp_min(1e-8) if gate == "entropy" else torch.ones_like(ent)
    n = delta.shape[0]
    if position == "all":
        return delta, h_key, g
    if position == "boundary":
        idx = torch.arange(n - min(tail_len, n), n, device=delta.device)
        return delta[idx], h_key[idx], g[idx]
    if position == "topk":
        idx = torch.topk(ent, min(topk, n)).indices.sort().values
        return delta[idx], h_key[idx], None
    if position == "pooled":
        w = (g / g.sum())[:, None]
        return (w * delta).sum(0, keepdim=True), (w * h_key).sum(0, keepdim=True), None
    raise ValueError(position)


def build_memory(mu_l, writes, members, layer, position, baseline, gate, args, tail_len):
    mem = FastWeightMemory(mu_l.shape[0], mu_l, device=args.device)
    stats, last = [], None
    for s in members:
        for ci, ch in enumerate(writes[s["id"]]):
            delta, h_key, g = select(ch, layer, position, baseline, gate, tail_len, args.topk)
            err, dn = mem.write(delta, h_key, gate=g, eta=args.eta)
            stats.append({"scenario": s["id"], "followup_idx": ci, "tokens": int(delta.shape[0]),
                          "mean_rel_error": (err / dn.clamp_min(1e-8)).mean().item()})
            last = (delta[-1], h_key[-1], g is None)
    return mem, stats, last


def check_recall(mem, last, eta):
    """Exact recall of the most recent write when it was ungated and eta=1."""
    delta, h_key, ungated = last
    if not ungated or eta != 1.0:
        return
    rec = mem.predict(mem.key(h_key))
    rel = ((rec - delta).norm() / delta.norm().clamp_min(1e-8)).item()
    assert rel < 1e-3, f"recall sanity check failed: rel error {rel:.2e}"


# ------------------------------------------------------------------------ measure


def lp(model, prompt_ids, text, device):
    return mx.seq_logprob(model, prompt_ids, text_ids(tok_global, text), device)


def target_scores(model, scen, prompt_ids, device):
    """Main target score, plus foil scores for facts."""
    meas = scen["measure"]
    ids = torch.cat([prompt_ids, text_ids(tok_global, meas["prefix"])])
    if scen["type"] == "fact":
        return lp(model, ids, meas["target"], device), [lp(model, ids, f, device) for f in meas["foils"]]
    return lp(model, ids, meas["a"], device) - lp(model, ids, meas["b"], device), []


def relation_score(model, prompt_ids, rp, device):
    return lp(model, prompt_ids, rp["a"], device) - lp(model, prompt_ids, rp["b"], device)


def reference(model, tok, scen, args):
    refs = []
    for pr in scen["probes"]:
        base = chat_ids(tok, pr["text"])
        ceil = ceiling_ids(tok, scen["experience"], pr["text"])
        cont = mx.greedy(model, tok, ceil, args.cont_len, args.device)
        lp_ceil = mx.cont_logprobs(model, ceil, cont, args.device)
        lp_base = mx.cont_logprobs(model, base, cont, args.device)
        tb, fb = target_scores(model, scen, base, args.device)
        tc, fc = target_scores(model, scen, ceil, args.device)
        refs.append({"kind": "probe", "probe": pr["text"], "distance": pr["distance"],
                     "base_ids": base, "cont": cont, "lp_ceil": lp_ceil,
                     "kl_base": mx.kl(lp_ceil, lp_base),
                     "tgt_base": tb, "tgt_ceil": tc, "foil_base": fb, "foil_ceil": fc})
    for rp in scen.get("relation_probes", []):
        base = chat_ids(tok, rp["text"])
        ceil = ceiling_ids(tok, scen["experience"], rp["text"])
        refs.append({"kind": "relation", "probe": rp["text"], "rp": rp, "base_ids": base,
                     "rel_base": relation_score(model, base, rp, args.device),
                     "rel_ceil": relation_score(model, ceil, rp, args.device)})
    return refs


def evaluate(model, scen, refs, mem, layer, alpha, device):
    rows = []
    with inject(model, layer, mem, alpha):
        for r in refs:
            row = {"scenario": scen["id"], "type": scen["type"], "kind": r["kind"], "probe": r["probe"]}
            if r["kind"] == "probe":
                lp_mem = mx.cont_logprobs(model, r["base_ids"], r["cont"], device)
                tm, fm = target_scores(model, scen, r["base_ids"], device)
                row.update(distance=r["distance"], kl_base=r["kl_base"], kl_mem=mx.kl(r["lp_ceil"], lp_mem),
                           tgt_base=r["tgt_base"], tgt_ceil=r["tgt_ceil"], tgt_mem=tm,
                           foil_base=r["foil_base"], foil_ceil=r["foil_ceil"], foil_mem=fm)
            else:
                row.update(distance="relation", rel_base=r["rel_base"], rel_ceil=r["rel_ceil"],
                           rel_mem=relation_score(model, r["base_ids"], r["rp"], device))
            rows.append(row)
    return rows


def eval_leakage(model, unrel, mem, layer, alpha, device):
    with inject(model, layer, mem, alpha):
        return [mx.kl(u["lp_base"], mx.cont_logprobs(model, u["ids"], u["cont"], device)) for u in unrel]


# ------------------------------------------------------------------------ summary


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows, leak):
    groups = defaultdict(list)
    for r in rows:
        key = (r["mode"], r["gate"], r["position"], r["baseline"], r["layer"], r["alpha"], r["type"], r["distance"])
        groups[key].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        mode, gate, pos, base, layer, alpha, typ, dist = key
        row = {"mode": mode, "gate": gate, "position": pos, "baseline": base, "layer": layer,
               "alpha": alpha, "type": typ, "distance": dist, "n": len(rs)}
        if dist == "relation":
            d_mem = mean(r["rel_mem"] - r["rel_base"] for r in rs)
            d_ceil = mean(r["rel_ceil"] - r["rel_base"] for r in rs)
            row.update(gap_closed="", dtarget_mem=d_mem, dtarget_ceil=d_ceil,
                       target_frac=d_mem / d_ceil if abs(d_ceil) > 1e-6 else float("nan"),
                       dfoil_mem="", specificity="")
        else:
            kl_b, kl_m = sum(r["kl_base"] for r in rs), sum(r["kl_mem"] for r in rs)
            d_mem = mean(r["tgt_mem"] - r["tgt_base"] for r in rs)
            d_ceil = mean(r["tgt_ceil"] - r["tgt_base"] for r in rs)
            if typ == "fact":
                d_foil = mean(mean(m - b for m, b in zip(r["foil_mem"], r["foil_base"])) for r in rs)
                spec = d_mem - d_foil
            else:
                d_foil, spec = "", ""
            row.update(gap_closed=1 - kl_m / kl_b if kl_b > 0 else float("nan"),
                       dtarget_mem=d_mem, dtarget_ceil=d_ceil,
                       target_frac=d_mem / d_ceil if abs(d_ceil) > 1e-6 else float("nan"),
                       dfoil_mem=d_foil, specificity=spec)
        row["leak"] = mean(leak[(mode, gate, pos, base, layer, alpha)])
        out.append(row)
    return out


def write_samples(model, tok, scenarios, mu, writes, args, tail_len, path):
    """Greedy samples at the best v0 settings, for each write position x baseline."""
    variants = [(p, b) for p in ("all", "boundary", "pooled") for b in ("without", "contrastive")]
    gen = lambda ids: tok.decode(mx.greedy(model, tok, ids, args.sample_len, args.device),
                                 skip_special_tokens=True).replace("\n", " ")
    with open(path, "w") as f:
        f.write(f"# isolated memory, gate=entropy, alpha={args.sample_alpha}; "
                f"layer {args.sample_layer_disposition} (dispositions) / {args.sample_layer_fact} (facts)\n\n")
        for scen in scenarios:
            layer = args.sample_layer_disposition if scen["type"] == "disposition" else args.sample_layer_fact
            probes = [next(p["text"] for p in scen["probes"] if p["distance"] == "related")]
            probes += [rp["text"] for rp in scen.get("relation_probes", [])[:1]]
            mems = {}
            for pos, base in variants:
                mems[(pos, base)], _, _ = build_memory(mu[layer], writes, [scen], layer, pos, base,
                                                       "entropy", args, tail_len)
            f.write(f"## {scen['id']} ({scen['type']}, layer {layer})\nexperience: {scen['experience']}\n\n")
            for text in probes:
                base_ids = chat_ids(tok, text)
                f.write(f"probe: {text}\n[baseline]           {gen(base_ids)}\n")
                f.write(f"[ceiling]            {gen(ceiling_ids(tok, scen['experience'], text))}\n")
                for (pos, base), mem in mems.items():
                    with inject(model, layer, mem, args.sample_alpha):
                        f.write(f"[{pos}/{base}]".ljust(21) + f" {gen(base_ids)}\n")
                f.write("\n")


# --------------------------------------------------------------------------- main

tok_global = None


def main():
    global tok_global
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.scenarios))
    scenarios, unrelated = cfg["scenarios"], cfg["unrelated_probes"]
    generic = [l.strip() for l in open(args.generic) if l.strip()]
    layers = sorted(set(args.layers) | {args.sample_layer_disposition, args.sample_layer_fact})

    log(f"loading {args.model} on {args.device}")
    model, tok = load_model(args.model, device=args.device)
    tok_global = tok
    # template tail after the user content (end of turn + assistant header)
    tail_len = common_suffix_len(chat_ids(tok, "alpha"), chat_ids(tok, "beta"))

    with torch.inference_mode():
        log(f"mu over {len(generic)} generic prompts")
        mu, skip = compute_mu(model, tok, generic, layers, args.device)
        log("session 1: collecting writes (with / without / counter)")
        writes = {s["id"]: collect_writes(model, tok, s, layers, args.device) for s in scenarios}
        log("references (no memory)")
        refs = {s["id"]: reference(model, tok, s, args) for s in scenarios}
        unrel = []
        for text in unrelated:
            ids = chat_ids(tok, text)
            cont = mx.greedy(model, tok, ids, args.cont_len, args.device)
            unrel.append({"ids": ids, "cont": cont, "lp_base": mx.cont_logprobs(model, ids, cont, args.device)})

        rows, write_rows = [], []
        leak = defaultdict(list)
        for mode in args.modes:
            groups = [(s["id"], [s]) for s in scenarios] if mode == "isolated" else [("combined", scenarios)]
            for gate in args.gates:
                for pos in args.positions:
                    for base in args.baselines:
                        for layer in args.layers:
                            for gid, members in groups:
                                mem, stats, last = build_memory(mu[layer], writes, members, layer, pos,
                                                                base, gate, args, tail_len)
                                check_recall(mem, last, args.eta)
                                write_rows += [{"mode": mode, "gate": gate, "position": pos, "baseline": base,
                                                "layer": layer, "memory": gid, **st} for st in stats]
                                for alpha in args.alphas:
                                    tag = {"mode": mode, "gate": gate, "position": pos, "baseline": base,
                                           "layer": layer, "alpha": alpha, "memory": gid}
                                    for s in members:
                                        rows += [{**tag, **r} for r in
                                                 evaluate(model, s, refs[s["id"]], mem, layer, alpha, args.device)]
                                    leak[(mode, gate, pos, base, layer, alpha)] += eval_leakage(
                                        model, unrel, mem, layer, alpha, args.device)
                        log(f"done mode={mode} gate={gate} position={pos} baseline={base}")

        log("samples")
        write_samples(model, tok, scenarios, mu, writes, args, tail_len, out_dir / "samples.txt")

    summary = summarize(rows, leak)
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "write_stats.jsonl", "w") as f:
        for r in write_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    json.dump({**vars(args), "mu_skip_tokens": skip, "tail_len": tail_len,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              open(out_dir / "config.json", "w"), indent=2)
    log(f"wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
