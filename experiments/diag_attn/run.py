#!/usr/bin/env python
"""Seahorse diag_attn: does the model's own attention pick good memory writes?

Idea: "you remember what you attended to". Session 1 now produces an assistant
response (greedy, --resp-len tokens, repetition_penalty 1.0, from the WITH prompt).
A second forward pass over prompt + response with eager attention gives the
attention each input token j receives from the response. Deltas and keys still come
from the v0.1 with / without / counter runs WITHOUT the response; the response's
attention only decides WHICH shared-suffix tokens are written.

Attention received by input token j. Queries are the R positions that generate the
response (prompt[-1] .. response[-2]). Scores are averaged over those queries, over
heads, and over the layers of a set: early 4-9, mid 10-19, late 20-27, L23, L26.
  (a) raw             mean_{i,h} A_h[i, j]
  (b) value-weighted  mean_{i,h} A_h[i, j] * ||W_O^h v_j^{g(h)}||, i.e. the norm of
                      head h's output contribution through o_proj. g(h) is the GQA
                      key/value group of head h.
  (c) position-corrected (b):
      - The template head is dropped. It is the system prompt + '<|im_start|>user\\n',
        including the sink; its length is the common prefix of the generic prompts,
        as for mu.
      - Content tokens (between head and template tail) are normalised by the
        prompt's mean content score, so uniform = 1: s_j = b_j / mean_content(b).
      - c_j = s_j - B(class_j). B is the mean s of the same position class over
        --n-generic generic prompts, each with its own greedy response.
      - Position classes: the first content token, the last content token, and
        --nbins equal bins of relative position (j - start) / (L - 1) for the
        tokens in between.
      - Template-tail tokens are excluded from (c) proper. The 'extended' (c)
        scores them the same way against a per-offset tail baseline, so they can
        compete when the tail is allowed into the candidate set (P2 tail=in).

P1 attention signal (forward passes only):
   - region mass fractions
   - key-word ranks in the experience span
   - top-5 suffix tokens by (c)
   - overlap of attention top-k with v0.1's entropy top-k
   - a turn-level control that swaps the experience for an irrelevant sentence
P2 deterministic write selection among shared-suffix tokens:
   - variants: attn_topk (k by (c), ungated), entropy_topk (v0.1 topk),
     all / pooled (v0.1, entropy gate), random_k (--seeds seeds, ungated)
   - crossed with: template tail in / out, isolated / combined (yaml order),
     layers x baselines at one alpha
   - evaluation = v0.1: gap closed, d target, foils / specificity, relation
     probes, leakage
   - relation probes with the tail IN are contaminated by the shared template
     tail; the tail-OUT numbers are the valid comparison
P3 probabilistic selection, tail out:
   - token j is written with p_j = sigmoid((z_j - theta) / T)
   - z = (c) z-scored within the follow-up's candidates; theta is set so that
     sum_j p_j = k
   - T in --temps, --seeds seeds each, gate 1

Selection uses the (c) scores of layer set --attn-src (default late, 20-27) for
both memory layers.

Outputs (--out):
  p1_attention.csv, p1_tokens.jsonl
  p2_results.jsonl, p2_conditions.csv, p2_summary.csv
  p3_results.jsonl, p3_conditions.csv, p3_summary.csv
  report.txt, config.json

  python experiments/diag_attn/run.py --out runs/diag_attn
  python experiments/diag_attn/run.py --tiny --out /tmp/x      # CPU smoke test
  EXP=diag_attn sbatch --time=03:00:00 slurm/run.slurm
  EXP=diag_attn EXTRA_ARGS="--parts p3" sbatch slurm/run.slurm   # split if needed
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
import yaml

from seahorse import metrics as mx
from seahorse.memory import FastWeightMemory
from seahorse.residual import decoder_layers, load_model
from seahorse.sessions import chat_ids, common_suffix_len

HERE = Path(__file__).parent
V01_DIR = HERE.parent / "v0_1"


def _load_v01():
    """The v0.1 pipeline (compute_mu, collect_writes, reference, evaluate, ...)."""
    spec = importlib.util.spec_from_file_location("seahorse_v0_1_run", V01_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v01 = _load_v01()

KEYWORDS = {
    "vegetarian": ["vegetarian"],
    "peanut_allergy": ["peanut", "allergy"],
    "norway": ["Norway"],
    "dog_name": ["Biscuit"],
    "sister_name": ["Ines"],
    "job": ["deep-sea", "welder"],
}
IRRELEVANT = "The weather was mild today."
LAYER_SETS = {"early": list(range(4, 10)), "mid": list(range(10, 20)),
              "late": list(range(20, 28)), "L23": [23], "L26": [26]}
SHORT = {"vegetarian": "veg", "peanut_allergy": "peanut", "norway": "norway",
         "dog_name": "dog", "sister_name": "sister", "job": "job"}
HIGHER = {"gap": True, "gap_rel": True, "d_disp": True, "d_fact": True, "spec": True,
          "drel": True, "leak": False}
METRICS = list(HIGHER)
P2_METHODS = ["attn_topk", "entropy_topk", "random_k", "all", "pooled"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[23, 26])
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--baselines", nargs="+", default=["without", "contrastive"],
                   choices=["without", "contrastive"])
    p.add_argument("--p3-baselines", nargs="+", default=["without", "contrastive"],
                   choices=["without", "contrastive"])
    p.add_argument("--modes", nargs="+", default=["isolated", "combined"],
                   choices=["isolated", "combined"])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--temps", type=float, nargs="+", default=[0.5, 2.0])
    p.add_argument("--resp-len", type=int, default=40)
    p.add_argument("--n-generic", type=int, default=30)
    p.add_argument("--nbins", type=int, default=8)
    p.add_argument("--attn-src", default="late", choices=list(LAYER_SETS))
    p.add_argument("--parts", nargs="+", default=["p2", "p3"], choices=["p2", "p3"],
                   help="P1 always runs (it is cheap and P2/P3 need its scores)")
    p.add_argument("--scenarios", default=str(V01_DIR / "scenarios.yaml"))
    p.add_argument("--generic", default=str(v01.GENERIC))
    p.add_argument("--cont-len", type=int, default=20)
    p.add_argument("--v01-results",
                   default=f"/scratch/{os.environ.get('USER', 'unknown')}/seahorse_runs/v0_1_415909/results.jsonl",
                   help="v0.1 results.jsonl; tail-in all/pooled/entropy_topk must reproduce it")
    p.add_argument("--repro-tol", type=float, default=1e-2)
    p.add_argument("--tiny", action="store_true",
                   help="smoke test: tiny random Qwen2 (28 layers) with --model's tokenizer, 3 scenarios")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def load(args):
    if not args.tiny:
        return load_model(args.model, device=args.device)
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=28, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    model = Qwen2ForCausalLM(cfg).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ------------------------------------------------------------------- attention


def set_attn(model, impl):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
    else:
        model.config._attn_implementation = impl


def attn_profile(model, prompt, resp, device):
    """Attention received by each prompt token from the response-generating queries.

    Returns {set: {"raw": [Tp], "vw": [Tp], "to_prompt": float}} (CPU tensors), where
    to_prompt is the share of those queries' attention that lands on the prompt."""
    ids = torch.cat([prompt.cpu(), resp.cpu()])[None].to(device)
    Tp, R = len(prompt), len(resp)
    cfg = model.config
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    blocks = decoder_layers(model)
    need = sorted({l for ls in LAYER_SETS.values() for l in ls if l < len(blocks)})
    vs, handles = {}, []
    for l in need:
        handles.append(blocks[l].self_attn.v_proj.register_forward_hook(
            lambda m, a, o, l=l: vs.__setitem__(l, o[0].detach())))
    orig = model.config._attn_implementation or "sdpa"
    try:
        set_attn(model, "eager")
        att = model(ids, output_attentions=True).attentions
    finally:
        set_attn(model, orig)
        for h in handles:
            h.remove()
    assert att is not None and att[need[0]] is not None, "no attention weights returned"
    per = {}
    for l in need:
        A = att[l][0, :, Tp - 1:Tp + R - 1].float()            # [H, R, T]
        to_prompt = A[..., :Tp].sum(-1).mean().item()
        A = A[..., :Tp]
        # head h reads kv group h // (H / KV) (repeat_kv order)
        v = vs[l][:Tp].float().view(Tp, KV, hd).repeat_interleave(H // KV, dim=1)  # [Tp, H, hd]
        Wo = blocks[l].self_attn.o_proj.weight.float().view(-1, H, hd)             # [d, H, hd]
        N = torch.einsum("dhe,the->htd", Wo, v).norm(dim=-1)                        # [H, Tp]
        per[l] = (A.mean((0, 1)), (A * N[:, None, :]).mean((0, 1)), to_prompt)
    out = {}
    for name, ls in LAYER_SETS.items():
        ls = [l for l in ls if l in per]
        if ls:
            out[name] = {"raw": torch.stack([per[l][0] for l in ls]).mean(0).cpu(),
                         "vw": torch.stack([per[l][1] for l in ls]).mean(0).cpu(),
                         "to_prompt": sum(per[l][2] for l in ls) / len(ls)}
    return out


def pos_class(j, lo, hi, nbins):
    if j == lo:
        return "first"
    if j == hi - 1:
        return "last"
    return min(int((j - lo) / (hi - 1 - lo) * nbins), nbins - 1)


def norm_scores(vw, lo, hi):
    return vw / vw[lo:hi].mean().clamp_min(1e-12)


def fit_baseline(gprofs, skip, tail_len, nbins):
    """Positional baseline of normalised (b) scores from generic prompts, per layer set."""
    base = {}
    for name in gprofs[0][1]:
        acc = defaultdict(list)
        for Tp, prof in gprofs:
            lo, hi = skip, Tp - tail_len
            s = norm_scores(prof[name]["vw"], lo, hi)
            for j in range(lo, hi):
                acc[pos_class(j, lo, hi, nbins)].append(s[j].item())
            for t in range(tail_len):
                acc[("tail", t)].append(s[hi + t].item())
        mid = [x for key, xs in acc.items() if isinstance(key, int) for x in xs]
        fb = mean(mid) if mid else 1.0
        base[name] = {"first": mean(acc["first"]), "last": mean(acc["last"]),
                      "bins": [mean(acc[b]) if acc[b] else fb for b in range(nbins)],
                      "bin_counts": [len(acc[b]) for b in range(nbins)],
                      "tail": [mean(acc[("tail", t)]) for t in range(tail_len)]}
    return base


def correct(vw, Tp, skip, tail_len, B, nbins):
    """Extended variant (c): NaN on the template head; tail scored vs the tail baseline."""
    lo, hi = skip, Tp - tail_len
    s = norm_scores(vw, lo, hi)
    c = torch.full((Tp,), float("nan"))
    for j in range(lo, hi):
        k = pos_class(j, lo, hi, nbins)
        c[j] = s[j] - (B["bins"][k] if isinstance(k, int) else B[k])
    for t in range(tail_len):
        c[hi + t] = s[hi + t] - B["tail"][t]
    return c


def ranks_desc(x):
    order = torch.argsort(x, descending=True)
    r = torch.empty_like(order)
    r[order] = torch.arange(1, len(x) + 1)
    return r


def keyword_positions(tok, ids, lo, hi, words):
    pieces = [tok.decode([int(t)]) for t in ids[lo:hi]]
    text, spans = "", []
    for p in pieces:
        spans.append((len(text), len(text) + len(p)))
        text += p
    out = {}
    for w in words:
        i = text.lower().find(w.lower())
        if i >= 0:
            out[w] = [lo + k for k, (a, b) in enumerate(spans) if a < i + len(w) and b > i]
    return out


def analyse(tok, ids, prof, skip, n, tail_len, B, nbins, words):
    Tp = len(ids)
    regions = {"head": (0, skip), "exp": (skip, Tp - n), "fu": (Tp - n, Tp - tail_len),
               "tail": (Tp - tail_len, Tp)}
    kwpos = keyword_positions(tok, ids, skip, Tp - n, words) if words else {}
    rec = {}
    for name, p in prof.items():
        c = correct(p["vw"], Tp, skip, tail_len, B[name], nbins)
        d = {"to_prompt": p["to_prompt"], "c": c}
        for var in ("raw", "vw"):
            x = p[var]
            tot = x.sum().item()
            for rg, (a, b) in regions.items():
                d[f"{var}_{rg}"] = x[a:b].sum().item() / tot
        for rg in ("exp", "fu", "tail"):
            a, b = regions[rg]
            d[f"c_{rg}"] = c[a:b].mean().item()
        e = c[slice(*regions["exp"])].clamp_min(0).sum().item()
        f = c[slice(*regions["fu"])].clamp_min(0).sum().item()
        d["c_exp_posshare"] = e / (e + f) if e + f > 0 else float("nan")
        a, b = regions["exp"]
        for var, x in (("raw", p["raw"]), ("vw", p["vw"]), ("c", c)):
            rk = ranks_desc(x[a:b])
            wr = {w: int(min(rk[j - a] for j in js)) for w, js in kwpos.items() if js}
            d[f"kw_{var}"] = min(wr.values()) if wr else None
            d[f"kwords_{var}"] = wr
        rec[name] = d
    kw_ntok = len({j for js in kwpos.values() for j in js})
    return regions, rec, kw_ntok


def overlap_and_top(tok, ids, n, c, ent, tail_len, k):
    """Attention top-k (extended c) vs v0.1 entropy top-k on the shared suffix."""
    Tp, m = len(ids), n - tail_len
    toks = [tok.decode([int(t)]) for t in ids[Tp - n:]]
    out = {}
    for tail, size in (("in", n), ("out", m)):
        kk = min(k, size)
        a = set(torch.topk(c[:size], kk).indices.tolist())
        e = set(torch.topk(ent[:size], kk).indices.tolist())
        out[f"jac_{tail}"] = len(a & e) / len(a | e)
        out[f"hits_{tail}"] = len(a & e)
        out[f"hits_rand_{tail}"] = kk * kk / size
        out[f"attn_sel_{tail}"] = [toks[i] for i in sorted(a)]
        out[f"ent_sel_{tail}"] = [toks[i] for i in sorted(e)]
        if tail == "in":
            out["attn_tailshare"] = sum(i >= m for i in a) / kk
            out["ent_tailshare"] = sum(i >= m for i in e) / kk
    out["top5"] = [(toks[i], round(c[i].item(), 3), i >= m)
                   for i in torch.argsort(c, descending=True)[:5].tolist()]
    return out


def p1_run(model, tok, scenarios, writes, generic, skip, tail_len, args):
    log(f"P1: positional baseline from {args.n_generic} generic prompts")
    gprofs = []
    for text in generic[: args.n_generic]:
        ids = chat_ids(tok, text)
        resp = mx.greedy(model, tok, ids, args.resp_len, args.device)
        gprofs.append((len(ids), attn_profile(model, ids, resp, args.device)))
    B = fit_baseline(gprofs, skip, tail_len, args.nbins)
    log("P1: with / irrelevant prompts")
    recs, sel_scores = [], {}
    for s in scenarios:
        for ci, fu in enumerate(s["followups"]):
            ch = writes[s["id"]][ci]
            for kind, exp in (("with", s["experience"]), ("irrelevant", IRRELEVANT)):
                ids = chat_ids(tok, f"{exp} {fu}")
                n = ch["n"] if kind == "with" else common_suffix_len(ids, chat_ids(tok, fu))
                resp = mx.greedy(model, tok, ids, args.resp_len, args.device)
                prof = attn_profile(model, ids, resp, args.device)
                words = KEYWORDS.get(s["id"], []) if kind == "with" else []
                regions, rec, kw_ntok = analyse(tok, ids, prof, skip, n, tail_len, B, args.nbins, words)
                r = {"kind": kind, "scenario": s["id"], "fu_idx": ci, "followup": fu, "n": n,
                     "exp_len": regions["exp"][1] - regions["exp"][0], "kw_ntok": kw_ntok,
                     "regions": regions, "rec": rec, "prof": prof, "ids": ids,
                     "response": tok.decode(resp, skip_special_tokens=True)}
                if kind == "with":
                    c = rec[args.attn_src]["c"][len(ids) - n:].clone()
                    assert c.shape[0] == ch["entropy"].shape[0] and not torch.isnan(c).any()
                    sel_scores[(s["id"], ci)] = c
                    r.update(overlap_and_top(tok, ids, n, c, ch["entropy"].float().cpu(), tail_len, args.k))
                recs.append(r)
    return B, recs, sel_scores


# -------------------------------------------------------------------- selection


def bern_probs(z, k, temp):
    if len(z) <= k:
        return torch.ones_like(z)
    lo, hi = z.min().item() - 50 * temp, z.max().item() + 50 * temp
    for _ in range(100):
        mid = (lo + hi) / 2
        if torch.sigmoid((z - mid) / temp).sum().item() > k:
            lo = mid
        else:
            hi = mid
    return torch.sigmoid((z - (lo + hi) / 2) / temp)


def make_selection(method, tail, seed, temp, scenarios, writes, sel_scores, tail_len, k):
    """(scenario, follow-up) -> sorted suffix indices; None for methods v0.1 selects itself."""
    if method in ("entropy_topk", "all", "pooled"):
        return None
    out = {}
    for si, s in enumerate(scenarios):
        for ci, ch in enumerate(writes[s["id"]]):
            n = ch["n"]
            m = n if tail == "in" else n - tail_len
            sc = sel_scores[(s["id"], ci)][:m]
            gen = None
            if seed is not None:
                t_code = int(round((temp or 0) * 100))
                gen = torch.Generator().manual_seed(
                    100003 * seed + 1009 * si + 31 * ci + (7 if tail == "out" else 0) + 13 * t_code)
            if method == "attn_topk":
                idx = torch.topk(sc, min(k, m)).indices
            elif method == "random_k":
                idx = torch.randperm(m, generator=gen)[: min(k, m)]
            elif method == "bern":
                z = (sc - sc.mean()) / sc.std(unbiased=False).clamp_min(1e-8)
                p = bern_probs(z, k, temp)
                idx = torch.nonzero(torch.rand(m, generator=gen) < p).flatten()
            else:
                raise ValueError(method)
            out[(s["id"], ci)] = idx.sort().values
    return out


def pick(ch, layer, baseline, method, tail, idx, tail_len, k):
    """(delta, h_key, gate or None) for one follow-up. Tail-in all / pooled / entropy_topk
    are exactly v0.1's all / pooled / topk."""
    h_with, h_without, h_counter = ch["h"][layer]
    delta = h_with - (h_without if baseline == "without" else h_counter)
    h_key = h_without
    ent = ch["entropy"]
    n = delta.shape[0]
    m = n if tail == "in" else n - tail_len
    if method in ("all", "pooled"):
        g = ent / ent.max().clamp_min(1e-8)  # v0.1 entropy gate over the full suffix
        if method == "all":
            return delta[:m], h_key[:m], g[:m]
        w = (g[:m] / g[:m].sum())[:, None]
        return (w * delta[:m]).sum(0, keepdim=True), (w * h_key[:m]).sum(0, keepdim=True), None
    if method == "entropy_topk":
        idx = torch.topk(ent[:m], min(k, m)).indices.sort().values
    idx = idx.to(delta.device)
    return delta[idx], h_key[idx], None


def build(mu_l, writes, members, layer, baseline, method, tail, sel, tail_len, args):
    mem = FastWeightMemory(mu_l.shape[0], mu_l, device=args.device)
    last, ntok, nfu = None, 0, 0
    for s in members:
        for ci, ch in enumerate(writes[s["id"]]):
            idx = sel[(s["id"], ci)] if sel is not None else None
            delta, h_key, g = pick(ch, layer, baseline, method, tail, idx, tail_len, args.k)
            nfu += 1
            if delta.shape[0] == 0:
                continue
            mem.write(delta, h_key, gate=g, eta=1.0)
            ntok += delta.shape[0]
            last = (delta[-1], h_key[-1], g is None)
    if last is not None:
        v01.check_recall(mem, last, 1.0)  # exact recall of the last write when ungated
    return mem, ntok / nfu


def cond_metrics(rs, leak_vals):
    by = defaultdict(list)
    for r in rs:
        by[r["scenario"]].append(r)
    gap, gap_rel, d_disp, d_fact, spec, drel = [], [], [], [], [], []
    for xs in by.values():
        pr = [x for x in xs if x["kind"] == "probe"]
        kb, km = sum(x["kl_base"] for x in pr), sum(x["kl_mem"] for x in pr)
        gap.append(1 - km / kb if kb > 0 else float("nan"))
        gap_rel += [1 - x["kl_mem"] / x["kl_base"] for x in pr
                    if x["distance"] == "related" and x["kl_base"] > 0]
        d = mean(x["tgt_mem"] - x["tgt_base"] for x in pr)
        if xs[0]["type"] == "fact":
            d_fact.append(d)
            spec.append(d - mean(mean(m - b for m, b in zip(x["foil_mem"], x["foil_base"])) for x in pr))
        else:
            d_disp.append(d)
            rl = [x for x in xs if x["kind"] == "relation"]
            if rl:
                drel.append(mean(x["rel_mem"] - x["rel_base"] for x in rl))
    return {"gap": mean(gap), "gap_rel": mean(gap_rel), "d_disp": mean(d_disp),
            "d_fact": mean(d_fact), "spec": mean(spec), "drel": mean(drel), "leak": mean(leak_vals)}


def run_phase(phase, specs, baselines, ctx):
    model, args, scenarios, writes, mu, refs, unrel, sel_scores, tail_len = (
        ctx[k] for k in ("model", "args", "scenarios", "writes", "mu", "refs", "unrel", "sel_scores", "tail_len"))
    rows, conds = [], []
    for method, tail, seed, temp in specs:
        sel = make_selection(method, tail, seed, temp, scenarios, writes, sel_scores, tail_len, args.k)
        for mode in args.modes:
            groups = ([(s["id"], [s]) for s in scenarios] if mode == "isolated"
                      else [("combined", scenarios)])
            for baseline in baselines:
                for layer in args.layers:
                    tag = {"phase": phase, "method": method, "tail": tail, "seed": seed, "temp": temp,
                           "mode": mode, "baseline": baseline, "layer": layer, "alpha": args.alpha}
                    crow, leak, ntoks = [], [], []
                    for gid, members in groups:
                        mem, ntok = build(mu[layer], writes, members, layer, baseline, method, tail,
                                          sel, tail_len, args)
                        ntoks.append(ntok)
                        for s in members:
                            crow += [{**tag, "memory": gid, **r} for r in
                                     v01.evaluate(model, s, refs[s["id"]], mem, layer, args.alpha, args.device)]
                        leak += v01.eval_leakage(model, unrel, mem, layer, args.alpha, args.device)
                    rows += crow
                    conds.append({**tag, "ntok": mean(ntoks), **cond_metrics(crow, leak)})
        log(f"{phase}: done method={method} tail={tail} seed={seed} temp={temp}")
    return rows, conds


def aggregate(conds):
    groups = defaultdict(list)
    keys = ("method", "tail", "temp", "mode", "baseline", "layer")
    for c in conds:
        groups[tuple(c[k] for k in keys)].append(c)
    out = {}
    for key, cs in groups.items():
        row = dict(zip(keys, key))
        row["n_seeds"] = len(cs)
        for m in ["ntok"] + METRICS:
            xs = [c[m] for c in cs if not math.isnan(c[m])]
            row[f"{m}_mean"] = mean(xs)
            row[f"{m}_sd"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
            row[f"{m}_min"] = min(xs) if xs else float("nan")
            row[f"{m}_max"] = max(xs) if xs else float("nan")
        out[key] = row
    return out


def repro_check(path, rows, alpha, tol):
    """Tail-in all / pooled / entropy_topk vs the v0.1 run (position all / pooled / topk)."""
    if not Path(path).exists():
        return {"status": "SKIPPED", "reason": f"{path} not found"}
    pmap = {"all": "all", "pooled": "pooled", "entropy_topk": "topk"}
    ref = {}
    for line in open(path):
        r = json.loads(line)
        if r["gate"] == "entropy" and r["alpha"] == alpha and r["position"] in pmap.values():
            ref[(r["position"], r["mode"], r["baseline"], r["layer"], r["memory"], r["kind"], r["probe"])] = r
    fields = ["kl_mem", "tgt_mem", "foil_mem", "rel_mem", "kl_base", "tgt_base", "rel_base"]
    n, missing, maxdiff, worst = 0, 0, 0.0, None
    for r in rows:
        if r["tail"] != "in" or r["method"] not in pmap:
            continue
        key = (pmap[r["method"]], r["mode"], r["baseline"], r["layer"], r["memory"], r["kind"], r["probe"])
        if key not in ref:
            missing += 1
            continue
        n += 1
        for f in fields:
            if f not in r:
                continue
            a, b = r[f], ref[key][f]
            for x, y in (zip(a, b) if isinstance(a, list) else [(a, b)]):
                if abs(x - y) > maxdiff:
                    maxdiff, worst = abs(x - y), (key, f, x, y)
    status = "PASS" if (n > 0 and missing == 0 and maxdiff <= tol) else "FAIL"
    return {"status": status, "rows_compared": n, "rows_missing": missing, "max_abs_diff": maxdiff,
            "tol": tol, "worst": str(worst), "reference": str(path)}


# ---------------------------------------------------------------------- output


def fnum(x, sign=True, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fms(a, m):
    if a is None:
        return "-"
    return fnum(a[f"{m}_mean"]) + (f"±{a[f'{m}_sd']:.3f}" if a["n_seeds"] > 1 else "")


def table(headers, rows):
    cols = list(zip(*([headers] + rows)))
    w = [max(len(str(x)) for x in c) for c in cols]
    fmt = lambda r: "  ".join(str(x).rjust(wi) for x, wi in zip(r, w))
    return "\n".join([fmt(headers), "  ".join("-" * wi for wi in w)] + [fmt(r) for r in rows])


def rl(t):
    return [None if math.isnan(x) else round(x, 5) for x in t.tolist()]


def write_p1(out_dir, recs, tok):
    fields = (["kind", "scenario", "fu_idx", "set", "n_suffix", "exp_len", "to_prompt"]
              + [f"{v}_{rg}" for v in ("raw", "vw") for rg in ("head", "exp", "fu", "tail")]
              + ["c_exp", "c_fu", "c_tail", "c_exp_posshare", "kw_raw", "kw_vw", "kw_c"])
    with open(out_dir / "p1_attention.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            for name, d in r["rec"].items():
                w.writerow({"kind": r["kind"], "scenario": r["scenario"], "fu_idx": r["fu_idx"], "set": name,
                            "n_suffix": r["n"], "exp_len": r["exp_len"], **d})
    skipk = {"rec", "prof", "ids"}
    with open(out_dir / "p1_tokens.jsonl", "w") as f:
        for r in recs:
            j = {k: v for k, v in r.items() if k not in skipk}
            j["tokens"] = [tok.decode([int(t)]) for t in r["ids"]]
            j["scores"] = {name: {"raw": rl(r["prof"][name]["raw"]), "vw": rl(r["prof"][name]["vw"]),
                                  "c": rl(d["c"]), "kwords_c": d["kwords_c"], "kwords_vw": d["kwords_vw"],
                                  "kwords_raw": d["kwords_raw"]}
                           for name, d in r["rec"].items()}
            f.write(json.dumps(j) + "\n")


def p1_report(recs, scenarios, args):
    L = ["", "=" * 78, "P1: is the attention signal meaningful?", "=" * 78]
    withs = [r for r in recs if r["kind"] == "with"]
    irr = {(r["scenario"], r["fu_idx"]): r for r in recs if r["kind"] == "irrelevant"}
    sets = list(withs[0]["rec"])
    src = args.attn_src
    L.append(f"{len(withs)} with-prompts (+ {len(irr)} irrelevant controls); response = greedy {args.resp_len} "
             f"tokens; selection source = '{src}'.")
    L.append("a = raw, b = value-weighted (x ||W_O^h v_j||), c = b normalised to the content mean and minus "
             "the positional baseline (excess; 0 = typical).")
    L.append("\nResponse->prompt attention mass fractions (mean over with-prompts). c: mean excess per token; "
             "c:tail* = extended c; c:exp+ = share of positive excess on exp vs fu.")
    hdr = (["set"] + [f"{v}:{rg}" for v in ("a", "b") for rg in ("head", "exp", "fu", "tail")]
           + ["c:exp", "c:fu", "c:tail*", "c:exp+", "a:to_prompt"])
    rows = []
    for name in sets:
        g = lambda k: mean(r["rec"][name][k] for r in withs)
        rows.append([name] + [fnum(g(f"{v}_{rg}"), False) for v in ("raw", "vw")
                              for rg in ("head", "exp", "fu", "tail")]
                    + [fnum(g("c_exp")), fnum(g("c_fu")), fnum(g("c_tail")), fnum(g("c_exp_posshare"), False),
                       fnum(g("to_prompt"), False)])
    L.append(table(hdr, rows))

    L.append("\nKey-word rank in the experience span (1 = most attended; best key-word token; median over "
             "the 3 follow-ups). chance = (L+1)/(m+1) for m key-word tokens among L.")
    hdr = ["scenario", "words", "L", "chance", f"a:{src}", f"b:{src}"] + [f"c:{n}" for n in sets]
    rows, top3 = [], defaultdict(int)
    for s in scenarios:
        rs = [r for r in withs if r["scenario"] == s["id"]]
        med = lambda var, name: (statistics.median([r["rec"][name][f"kw_{var}"] for r in rs
                                                    if r["rec"][name][f"kw_{var}"] is not None])
                                 if any(r["rec"][name][f"kw_{var}"] is not None for r in rs) else None)
        Lx, m = rs[0]["exp_len"], rs[0]["kw_ntok"]
        cells = [med("raw", src), med("vw", src)] + [med("c", n) for n in sets]
        for name in sets:
            v = med("c", name)
            if v is not None and v <= 3:
                top3[name] += 1
        rows.append([SHORT.get(s["id"], s["id"]), "/".join(KEYWORDS.get(s["id"], [])), Lx,
                     f"{(Lx + 1) / (m + 1):.1f}" if m else "-"] + ["-" if v is None else f"{v:g}" for v in cells])
    L.append(table(hdr, rows))
    L.append("scenarios with median rank <= 3 under c: " + ", ".join(f"{n}={top3[n]}/{len(scenarios)}" for n in sets))

    L.append(f"\nTop-5 shared-suffix tokens by extended c ({src}); * = template tail. ent4 = v0.1 entropy top-4 "
             "(full suffix).")
    for r in withs:
        top = "  ".join(f"{t!r}{'*' if tl else ''} {v:+.2f}" for t, v, tl in r["top5"])
        L.append(f"{SHORT.get(r['scenario'], r['scenario'])}#{r['fu_idx'] + 1}: {top}")
        L.append(f"        ent4: {r['ent_sel_in']}   attn4(out): {r['attn_sel_out']}")

    L.append(f"\nOverlap of attention top-{args.k} (extended c, {src}) with entropy top-{args.k} "
             "(mean over follow-ups)")
    hdr = ["candidates", "jaccard", "hits", "random hits"]
    rows = [[f"tail {t}", fnum(mean(r[f"jac_{t}"] for r in withs), False),
             fnum(mean(r[f"hits_{t}"] for r in withs), False, 2),
             fnum(mean(r[f"hits_rand_{t}"] for r in withs), False, 2)] for t in ("in", "out")]
    L.append(table(hdr, rows))
    at_tail = mean(r["attn_tailshare"] for r in withs)
    en_tail = mean(r["ent_tailshare"] for r in withs)
    L.append(f"share of template-tail tokens in top-{args.k} (tail in): attention {at_tail:.2f}, entropy {en_tail:.2f}")

    L.append("\nTurn-level control: attention on the experience span, relevant vs irrelevant "
             f"('{IRRELEVANT}'), same follow-ups. a/b = mass fraction, c = mean excess per token; "
             "wins = pairs with relevant > irrelevant.")
    hdr = ["set", "a:rel", "a:irr", "b:rel", "b:irr", "c:rel", "c:irr", "wins a", "wins b", "wins c"]
    rows, wins_c_src, diff_src = [], None, None
    pairs = [(r, irr[(r["scenario"], r["fu_idx"])]) for r in withs]
    for name in sets:
        row = [name]
        for key in ("raw_exp", "vw_exp", "c_exp"):
            row += [fnum(mean(a["rec"][name][key] for a, _ in pairs), key == "c_exp"),
                    fnum(mean(b["rec"][name][key] for _, b in pairs), key == "c_exp")]
        w = [sum(a["rec"][name][k] > b["rec"][name][k] for a, b in pairs) for k in ("raw_exp", "vw_exp", "c_exp")]
        row += [f"{x}/{len(pairs)}" for x in w]
        rows.append(row)
        if name == src:
            wins_c_src = w[2]
            diff_src = mean(a["rec"][name]["c_exp"] - b["rec"][name]["c_exp"] for a, b in pairs)
    L.append(table(hdr, rows))
    L.append(f"by follow-up ({src}, c):")
    for ci in range(3):
        ps = [(a, b) for a, b in pairs if a["fu_idx"] == ci]
        if ps:
            L.append(f"  follow-up #{ci + 1}{' (off-topic)' if ci == 2 else ''}: rel {mean(a['rec'][src]['c_exp'] for a, _ in ps):+.3f}"
                     f"  irr {mean(b['rec'][src]['c_exp'] for _, b in ps):+.3f}"
                     f"  wins {sum(a['rec'][src]['c_exp'] > b['rec'][src]['c_exp'] for a, b in ps)}/{len(ps)}")

    L.append("\nP1 GO / NO-GO")
    ns = len(scenarios)
    L.append(f"  [1] content words in top 3 of the experience span after correction (c, {src}): "
             f"{top3[src]}/{ns} scenarios -> {'GO' if top3[src] > ns / 2 else 'NO-GO'}")
    tmpl = mean(r["rec"][src]["vw_head"] + r["rec"][src]["vw_tail"] for r in withs)
    L.append(f"  [2] template no longer dominates: b template (head+tail) mass {tmpl:.2f}; after correction, "
             f"tail share of attention top-{args.k} (tail allowed) = {at_tail:.2f} (entropy top-{args.k}: {en_tail:.2f})"
             f" -> {'GO' if at_tail < 0.5 else 'NO-GO'}")
    L.append(f"  [3] relevant experiences draw more response attention (c, {src}): wins {wins_c_src}/{len(pairs)}, "
             f"mean diff {diff_src:+.3f} -> {'GO' if wins_c_src >= 0.75 * len(pairs) and diff_src > 0 else 'NO-GO'}")
    return L


def p2_report(agg, repro, args):
    L = ["", "=" * 78, "P2: attention-based selection vs baselines (deterministic)", "=" * 78]
    L.append(f"alpha={args.alpha}, k={args.k}, random_k over {args.seeds} seeds (mean±sd). gap = mean over "
             "scenarios of 1 - sum KL_mem/sum KL_base; gap_rel = related-distance probes only; d_disp / d_fact = "
             "d target; spec = d_fact - d foils; drel = d relation score (disp); leak = KL on unrelated probes.")
    L.append("Tail IN relation probes (drel) are contaminated by the shared template tail; tail OUT is the valid "
             "comparison.")
    L.append(f"reproduction of v0.1 (tail-in all/pooled/entropy_topk): {repro}")
    hdr = ["tail", "mode", "method", "ntok"] + METRICS
    for b in args.baselines:
        for l in args.layers:
            L.append(f"\n--- baseline={b} layer={l} ---")
            rows = []
            for tail in ("out", "in"):
                for mode in args.modes:
                    for method in P2_METHODS:
                        a = agg.get((method, tail, None, mode, b, l))
                        if a:
                            rows.append([tail, mode, method, f"{a['ntok_mean']:.1f}"] + [fms(a, m) for m in METRICS])
            L.append(table(hdr, rows))
    L.append("\nP2 GO / NO-GO (counts over mode x baseline x layer cells)")
    verdict = {}
    for tail in ("out", "in"):
        cells = [(mode, b, l) for mode in args.modes for b in args.baselines for l in args.layers]
        res = {}
        for m in ("gap", "gap_rel", "d_disp", "spec", "drel", "leak"):
            c1 = c2 = c3 = N = 0
            for mode, b, l in cells:
                at, rk, en = (agg.get((x, tail, None, mode, b, l)) for x in ("attn_topk", "random_k", "entropy_topk"))
                if not (at and rk and en) or any(math.isnan(v) for v in (at[f"{m}_mean"], rk[f"{m}_mean"], en[f"{m}_mean"])):
                    continue
                N += 1
                x, hi = at[f"{m}_mean"], HIGHER[m]
                c1 += (x > rk[f"{m}_mean"]) if hi else (x < rk[f"{m}_mean"])
                c2 += (x > rk[f"{m}_max"]) if hi else (x < rk[f"{m}_min"])
                c3 += (x >= en[f"{m}_mean"]) if hi else (x <= en[f"{m}_mean"])
            res[m] = (c1, c2, c3, N)
            L.append(f"  tail={tail} {m:8s} attn beats random mean {c1}/{N}; outside random seed range {c2}/{N}; "
                     f">= entropy_topk {c3}/{N}")
        N = max(v[3] for v in res.values()) or 1
        beats = max(res[m][1] for m in ("gap", "gap_rel", "spec")) >= 0.75 * N
        matches = max(res[m][2] for m in ("gap", "gap_rel", "spec", "leak")) >= 0.5 * N
        verdict[tail] = beats and matches
        L.append(f"  tail={tail} heuristic: beats random outside spread on a recall metric in >=75% of cells: "
                 f"{beats}; matches/beats entropy_topk on recall or leak in >=50%: {matches} -> "
                 f"{'GO' if verdict[tail] else 'NO-GO'}")
    return L


def p3_report(agg, args):
    L = ["", "=" * 78, "P3: probabilistic (Bernoulli) selection, template tail excluded", "=" * 78]
    L.append(f"p_j = sigmoid((z_j - theta)/T), sum p = {args.k}; {args.seeds} seeds per T; mean±sd over seeds. "
             "attn_topk / random_k = tail-out P2 rows.")
    hdr = ["method", "ntok"] + METRICS
    spread = defaultdict(list)
    for b in args.p3_baselines:
        for l in args.layers:
            for mode in args.modes:
                L.append(f"\n--- baseline={b} layer={l} mode={mode} ---")
                at = agg.get(("attn_topk", "out", None, mode, b, l))
                rk = agg.get(("random_k", "out", None, mode, b, l))
                rows = [[name, f"{a['ntok_mean']:.1f}"] + [fms(a, m) for m in METRICS]
                        for name, a in (("attn_topk", at), ("random_k", rk)) if a]
                for T in args.temps:
                    bt = agg.get(("bern", "out", T, mode, b, l))
                    if bt:
                        rows.append([f"bern T={T:g}", f"{bt['ntok_mean']:.1f}"] + [fms(bt, m) for m in METRICS])
                        for m in METRICS:
                            spread[(m, f"sd bern T={T:g}")].append(bt[f"{m}_sd"])
                            if rk:
                                spread[(m, f"|bern T={T:g} - random|")].append(abs(bt[f"{m}_mean"] - rk[f"{m}_mean"]))
                            if at:
                                spread[(m, f"|bern T={T:g} - attn_topk|")].append(abs(bt[f"{m}_mean"] - at[f"{m}_mean"]))
                if at and rk:
                    for m in METRICS:
                        spread[(m, "sd random")].append(rk[f"{m}_sd"])
                        spread[(m, "|attn_topk - random|")].append(abs(at[f"{m}_mean"] - rk[f"{m}_mean"]))
                L.append(table(hdr, rows))
    L.append("\nSeed spread vs method gaps (median over baseline x layer x mode cells)")
    cols = sorted({c for _, c in spread})
    rows = [[m] + [fnum(statistics.median([x for x in spread[(m, c)] if not math.isnan(x)]), False)
                   if any(not math.isnan(x) for x in spread.get((m, c), [])) else "-" for c in cols]
            for m in METRICS]
    L.append(table(["metric"] + cols, rows))
    return L


def write_phase(out_dir, name, rows, conds, agg):
    with open(out_dir / f"{name}_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for fname, data in ((f"{name}_conditions.csv", conds), (f"{name}_summary.csv", list(agg.values()))):
        with open(out_dir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)


# ------------------------------------------------------------------------- main


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.scenarios))
    scenarios, unrelated = cfg["scenarios"], cfg["unrelated_probes"]
    generic = [l.strip() for l in open(args.generic) if l.strip()]
    if args.tiny:
        scenarios = [s for s in scenarios if s["id"] in ("vegetarian", "peanut_allergy", "dog_name")]
        unrelated = unrelated[:2]
        args.seeds, args.n_generic, args.resp_len = 2, 5, 8

    log(f"loading {args.model} on {args.device}{' (tiny random model)' if args.tiny else ''}")
    model, tok = load(args)
    v01.tok_global = tok
    tail_len = common_suffix_len(chat_ids(tok, "alpha"), chat_ids(tok, "beta"))
    config = {**vars(args), "tail_len": tail_len, "irrelevant": IRRELEVANT, "keywords": KEYWORDS,
              "layer_sets": LAYER_SETS, "torch": torch.__version__, "transformers": transformers.__version__,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    report = [f"Seahorse diag_attn report ({args.model}{', TINY' if args.tiny else ''})"]

    def save():
        (out_dir / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        json.dump({**config, "runtime_s": round(time.time() - t0)}, open(out_dir / "config.json", "w"),
                  indent=2, default=str)

    with torch.inference_mode():
        log(f"mu over {len(generic)} generic prompts")
        mu, skip = v01.compute_mu(model, tok, generic, args.layers, args.device)
        config["mu_skip_tokens"] = skip
        log("session 1: collecting writes (with / without / counter)")
        writes = {s["id"]: v01.collect_writes(model, tok, s, args.layers, args.device) for s in scenarios}
        B, recs, sel_scores = p1_run(model, tok, scenarios, writes, generic, skip, tail_len, args)
        config["positional_baseline"] = B
        write_p1(out_dir, recs, tok)
        report += p1_report(recs, scenarios, args)
        save()
        log("P1 written")
        if not set(args.parts) & {"p2", "p3"}:
            return
        log("references (no memory)")
        refs = {s["id"]: v01.reference(model, tok, s, args) for s in scenarios}
        unrel = []
        for text in unrelated:
            ids = chat_ids(tok, text)
            cont = mx.greedy(model, tok, ids, args.cont_len, args.device)
            unrel.append({"ids": ids, "cont": cont, "lp_base": mx.cont_logprobs(model, ids, cont, args.device)})
        ceil = {"d_disp": [], "d_fact": [], "drel": []}
        for s in scenarios:
            for r in refs[s["id"]]:
                if r["kind"] == "probe":
                    ceil["d_fact" if s["type"] == "fact" else "d_disp"].append(r["tgt_ceil"] - r["tgt_base"])
                else:
                    ceil["drel"].append(r["rel_ceil"] - r["rel_base"])
        report.append("\nceiling (experience in context): " + ", ".join(f"{k} {mean(v):+.3f}" for k, v in ceil.items()))
        ctx = {"model": model, "args": args, "scenarios": scenarios, "writes": writes, "mu": mu, "refs": refs,
               "unrel": unrel, "sel_scores": sel_scores, "tail_len": tail_len}
        agg = {}
        if "p2" in args.parts:
            specs = []
            for tail in ("in", "out"):
                specs += [(m, tail, None, None) for m in ("attn_topk", "entropy_topk", "all", "pooled")]
                specs += [("random_k", tail, sd, None) for sd in range(args.seeds)]
            rows, conds = run_phase("p2", specs, args.baselines, ctx)
            agg = aggregate(conds)
            repro = (repro_check(args.v01_results, rows, args.alpha, args.repro_tol) if not args.tiny
                     else {"status": "SKIPPED", "reason": "tiny"})
            config["repro_check"] = repro
            log(f"reproduction check: {repro}")
            write_phase(out_dir, "p2", rows, conds, agg)
            report += p2_report(agg, repro["status"] + ("" if repro["status"] != "PASS" else
                                f" (max abs diff {repro['max_abs_diff']:.1e} over {repro['rows_compared']} rows)"), args)
            save()
        if "p3" in args.parts:
            specs = [("bern", "out", sd, T) for T in args.temps for sd in range(args.seeds)]
            if "p2" not in args.parts:  # stand-alone P3 job: add its deterministic comparisons
                specs = ([("attn_topk", "out", None, None)]
                         + [("random_k", "out", sd, None) for sd in range(args.seeds)] + specs)
            rows3, conds3 = run_phase("p3", specs, args.p3_baselines, ctx)
            agg3 = aggregate(conds3)
            write_phase(out_dir, "p3", rows3, conds3, agg3)
            report += p3_report({**agg, **agg3}, args)
            save()
    log(f"done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
