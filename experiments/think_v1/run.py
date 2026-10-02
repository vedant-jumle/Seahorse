#!/usr/bin/env python
"""Seahorse think_v1: the current memory design on Qwen3.5-2B, a thinking model.

Design (docs/Seahorse - How It Works.md, unchanged): one entropy-weighted pooled write per
follow-up (template tail excluded; dispositions contrastive, facts without); key pooled_w256 =
running mean over user-text positions of PCA-256-whitened (h - mu) (spread floor 0.01 x mean
eigenvalue), normalised; the write key comes from the WITHOUT run; read hard: max-cos match to
the stored keys > the q0.95 match on generic prompts + greedy continuations (per memory), no read
at the template head, h <- h + alpha * gate * M q; delta rule for one memory, RLS (lambda 0.1)
for the combined memory. Secondary key pooled_wLW: the same with full Ledoit-Wolf whitening.

Model: Qwen/Qwen3.5-2B (Qwen3_5ForConditionalGeneration, text path only), fp32. Hook point: the
output of text decoder block l (model.model.language_model.layers[l]); 24 blocks, full attention
at 3, 7, 11, 15, 19, 23 and Gated DeltaNet (linear attention) elsewhere. Thinking is the chat
template's enable_thinking: off -> the prompt ends "<think>\\n\\n</think>\\n\\n" (the default);
on -> it ends "<think>\\n" and the model thinks until it emits </think>.

Items (data/bench_v1): dispositions vegetarian, norway, loves_hiking, loves_jazz; facts dog_name,
sister_name, job, favourite_colour; the first 8 shared unrelated probes.

--stage 1  layer sweep, thinking off, no generation. Every layer in --layers (one capture pass;
           mu, whitening and thresholds per layer), keys w256 + wLW, one memory per item
           (delta), injection at that layer only, alpha 2. Analytic selectivity (recall
           fraction by probe kind, SI, template share, active fraction; isolated + combined
           RLS) and teacher-forced metrics: fact dlogP(target) / foils / specificity,
           disposition contrasts (all 3), relation/verification dscore and accuracy with Yes-
           and No-consistent probes kept separate, KL gap to the ceiling, leakage KL on the
           unrelated probes. Picks the best 3 layers (rule in report.txt / choice.json).
--stage 2  single vs multi-layer, thinking off: the best 1 / 2 / 3 layers of stage 1, alpha split
           (alpha/n per layer) or full (alpha at every layer); isolated (delta) and combined
           (RLS). Stage-1 metrics (wLW: isolated only) plus samples_v2-style generation: 20-
           sample disposition rates on the ambiguous prompts (T 0.7, bench lexicon labels), 50-
           sample fact prefix completions (T 1.0), greedy related + unrelated probes. Picks the
           injection set for stage 3 (choice.json).
--stage 3  thinking on, the stage-2 set, isolated memories (+ one combined-RLS run of the best
           condition). Generated positions before </think> are read with alpha_think, </think>
           and everything after with alpha_answer; prompt positions use alpha_answer. Thinking
           is capped at --think-cap tokens (then </think> is forced), the answer at --answer-cap.
           Per prompt x condition: 1 greedy + --samples3 samples with the model card's thinking
           settings (T 1.0, top-p 0.95, top-k 20, presence penalty 1.5 on generated tokens), the
           same random numbers for every condition.

Outputs per stage (--out): report.txt, summary.csv, results.jsonl, config.json; choice.json
(stages 1-2); samples.txt and texts.jsonl.gz (stages 2-3).
"""

import argparse
import contextlib
import gzip
import importlib.util
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers

from seahorse import metrics as mx
from seahorse.bench import label, lexicon_hits, load_bench
from seahorse.residual import inject, load_model, text_model
from seahorse.sessions import ceiling_ids, chat_ids, common_suffix_len, text_ids

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s2 = _load("seahorse_samples_v2_run", HERE.parent / "samples_v2" / "run.py")
dk = s2.dk  # one diag_keys instance (patched below), shared with samples_v2
v01 = dk.v01
log, fmt, table, mean = dk.log, dk.fmt, dk.table, dk.mean

W256, WLW = "pooled_w256", "pooled_wLW"
KEYS = (W256, WLW)
DK_KEYS = {W256: (("pca", 256), True), WLW: (("lw", None), True)}
DIST = ("exact", "paraphrase", "related")
SI_EPS = 0.01
COND3 = [("nomem", 0.0, 0.0), ("answer_only", 0.0, 2.0), ("both", 2.0, 2.0), ("think4", 4.0, 0.0),
         ("think6", 6.0, 0.0), ("think4_ans1", 4.0, 1.0), ("think6_ans1", 6.0, 1.0)]
STOP = set("not unknown a an the your my their his her something what unclear mentioned given provided known "
           "specified stated available that this it likely probably actually still also just in called named one no "
           "is was be to of for and or but if".split())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--bench", default=str(REPO / "data" / "bench_v1"))
    p.add_argument("--disp-items", nargs="+", default=["vegetarian", "norway", "loves_hiking", "loves_jazz"])
    p.add_argument("--fact-items", nargs="+", default=["dog_name", "sister_name", "job", "favourite_colour"])
    p.add_argument("--n-unrelated", type=int, default=8, help="shared unrelated probes (leakage KL)")
    p.add_argument("--unrel-gen", type=int, nargs="+", default=[0, 2, 3, 5],
                   help="indices of the unrelated probes used in generation (stages 2-3)")
    p.add_argument("--layers", type=int, nargs="+", default=list(range(4, 24)), help="stage 1 sweep")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--thr-q", type=float, default=0.95)
    p.add_argument("--white-eps", type=float, default=0.01)
    p.add_argument("--rls-lambda", type=float, default=0.1)
    p.add_argument("--leak-mult", type=float, default=2.0, help="stage 1/2 leakage cap (see choice rules)")
    p.add_argument("--generic", default=str(v01.GENERIC))
    p.add_argument("--cont-len", type=int, default=20)
    p.add_argument("--gen-cont-len", type=int, default=20)
    p.add_argument("--s1", default=None, help="stage 1 output dir (stage 2 input)")
    p.add_argument("--s2", default=None, help="stage 2 output dir (stage 3 input)")
    # stage 2 generation (samples_v2 settings)
    p.add_argument("--n-ambiguous", type=int, default=5)
    p.add_argument("--disp-samples", type=int, default=20)
    p.add_argument("--disp-temp", type=float, default=0.7)
    p.add_argument("--disp-len", type=int, default=60)
    p.add_argument("--prefix-samples", type=int, default=50)
    p.add_argument("--prefix-temp", type=float, default=1.0)
    p.add_argument("--prefix-len", type=int, default=12)
    p.add_argument("--greedy-len", type=int, default=60)
    # stage 3
    p.add_argument("--think-cap", type=int, default=384)
    p.add_argument("--answer-cap", type=int, default=150)
    p.add_argument("--samples3", type=int, default=10)
    p.add_argument("--temp3", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence", type=float, default=1.5)
    p.add_argument("--amb3", type=int, default=2, help="ambiguous prompts per disposition in stage 3")
    p.add_argument("--eval-items", nargs="+", default=None,
                   help="stage 3: items whose prompts are generated (default all; e.g. one job per item group). "
                        "The memories (and the combined RLS memory) always hold every item.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-tol", type=float, default=0.05)
    p.add_argument("--tiny", default=None, choices=["qwen2", "qwen3_5"],
                   help="smoke test: tiny random model (qwen2 = old architecture; qwen3_5 = the Qwen3.5 "
                        "classes, needs transformers with qwen3_5) with --model's tokenizer")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.tiny:
        a.disp_items, a.fact_items, a.n_unrelated, a.unrel_gen = a.disp_items[:2], a.fact_items[:2], 4, [0, 1]
        if a.layers == list(range(4, 24)):
            a.layers = [3, 4, 7, 23]
        a.n_ambiguous, a.disp_samples, a.disp_len, a.prefix_samples, a.prefix_len, a.greedy_len = 2, 3, 6, 4, 4, 6
        a.think_cap, a.answer_cap, a.samples3, a.amb3, a.cont_len, a.gen_cont_len = 10, 6, 2, 1, 6, 6
    return a


# ------------------------------------------------------------------------ model


def load_any(args):
    if not args.tiny:
        return load_model(args.model, device=args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    torch.manual_seed(0)
    if args.tiny == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=24,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048)
        model = Qwen2ForCausalLM(cfg)
    else:
        from transformers import AutoConfig, AutoModelForImageTextToText
        cfg = AutoConfig.from_pretrained(args.model)
        tc, vc = cfg.text_config, cfg.vision_config
        tc.hidden_size, tc.intermediate_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 64, 128, 2, 1, 64
        tc.linear_num_key_heads = tc.linear_num_value_heads = 2
        tc.linear_key_head_dim = tc.linear_value_head_dim = 32
        tc.rope_parameters = {**tc.rope_parameters, "mrope_section": [3, 3, 2]}
        vc.depth, vc.hidden_size, vc.intermediate_size, vc.num_heads, vc.out_hidden_size = 1, 32, 64, 2, 64
        model = AutoModelForImageTextToText.from_config(cfg)
    model = model.float().to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def chat(tok, text, think):
    """One user turn + the assistant header, thinking on/off (Qwen3.5: enable_thinking)."""
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=think)
    if think and not s.rstrip().endswith("<think>"):
        s += "<think>\n"  # templates that leave the opening tag to the model
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids[0]


@contextlib.contextmanager
def injecting(model, readers):
    """readers: [(layer, reader, alpha)], all active at once."""
    with contextlib.ExitStack() as st:
        for layer, rd, alpha in readers:
            st.enter_context(inject(model, layer, rd, alpha))
        yield


# ------------------------------------------------------------------------ setup


def setup(args, out):
    bench = load_bench(args.bench)
    by = {it["id"]: it for it in bench["scenarios"]}
    items = [by[i] for i in args.disp_items + args.fact_items]
    unrel = bench["unrelated_probes"][:args.n_unrelated]
    path = out / "items.json"  # JSON is YAML: diag_keys.setup reads it unchanged
    json.dump({"scenarios": items, "unrelated_probes": unrel}, open(path, "w"), indent=1)
    generic = args.generic
    if args.tiny:
        generic = out / "generic_tiny.txt"
        generic.write_text("\n".join([l for l in open(args.generic) if l.strip()][:16]))
    layers = args.layers_used
    dargs = SimpleNamespace(model=args.model, layers=layers, analytic_layers=layers, scenarios=str(path),
                            generic=str(generic), cont_len=args.cont_len, gen_cont_len=args.gen_cont_len,
                            white_eps=args.white_eps, tiny=bool(args.tiny), device=args.device)
    dk.load = lambda _a: load_any(args)
    dk.KEY_SPECS = dict(DK_KEYS)
    C = dk.setup(dargs)
    tok, model, dev = C.tok, C.model, args.device
    C.args, C.dev, C.chk, C.thr_log = args, dev, defaultdict(float), {}
    C.base_of = {s: "without" if C.by_id[s]["type"] == "fact" else "contrastive" for s in C.ids}
    C.body, C.lm_head = text_model(model), model.lm_head
    C.vocab = C.lm_head.out_features
    tcfg = getattr(model.config, "text_config", model.config)
    lt = getattr(tcfg, "layer_types", None) or ["full_attention"] * tcfg.num_hidden_layers
    C.ltype = {l: "full" if lt[l] == "full_attention" else "linear" for l in range(len(lt))}
    eos = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            eos.add(i)
    ge = model.generation_config.eos_token_id
    eos |= set(ge if isinstance(ge, (list, tuple)) else [ge])
    C.eos = torch.tensor(sorted(e for e in eos if e is not None), device=dev)
    C.think_end = tok.convert_tokens_to_ids("</think>")
    assert len(text_ids(tok, "</think>")) == 1 and C.think_end not in (None, tok.unk_token_id), "</think> not one token"
    on_a = chat(tok, "alpha", True)
    C.tail_len_on = common_suffix_len(on_a, chat(tok, "beta", True))
    C.catfn_on = dk.make_catfn(C.head, on_a[-C.tail_len_on:], dev)
    C.stash_on = dk.Stash(model, C.catfn_on)
    C.phase = SimpleNamespace(a_prompt=0.0, alpha=None)
    off_a = chat_ids(tok, "alpha")
    assert torch.equal(off_a, chat(tok, "alpha", False)), "default template != enable_thinking=False"
    C.template = {"head": tok.decode(C.head), "tail_off": tok.decode(off_a[-C.tail_len:]),
                  "tail_on": tok.decode(on_a[-C.tail_len_on:]), "eos": C.eos.tolist(), "think_end": C.think_end}
    log(f"template {json.dumps(C.template)}")
    C.unrel_gen = [C.unrelated[i] for i in args.unrel_gen]
    C.xref = {}
    for s in C.ids:
        scen = C.by_id[s]
        if scen["type"] == "disposition":
            for p in scen["probes"]:
                C.xref[(s, p["text"])] = {"base": contrasts(C, scen, chat_ids(tok, p["text"])),
                                          "ceil": contrasts(C, scen, ceiling_ids(tok, scen["experience"], p["text"]))}
    C.wrong = {s: wrong_fn(C.by_id[s]) for s in C.ids if C.by_id[s]["type"] == "fact"}
    return C


def wrong_fn(scen):
    """text -> sorted wrong values: a foil / the counter's value mentioned, or a value asserted after the
    measure prefix ("dog's name is X", "work as a X") that is not the target."""
    m = scen["measure"]
    tgt = m["target"].strip().lower()
    exp_w = set(re.findall(r"[\w']+", scen["experience"].lower()))
    wrong = {f.strip().lower() for f in m["foils"]} | {w for w in re.findall(r"[\w']+", scen["counter"].lower())
                                                       if w not in exp_w}
    core = m["prefix"].split()
    core = core[1:] if core[0].lower() in ("your", "you") else core
    rx = r"\s+".join(re.escape(w) for w in core).replace("favourite", "favou?rite").replace("colour", "colou?r")
    rx = re.compile(rx.replace("'", "['’]") + r"\s+[*\"'_`]*([A-Za-z][A-Za-z-]*)", re.I)

    def f(text):
        hits = {w for w in wrong if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text, re.I)}
        hits |= {v.lower() for v in rx.findall(text) if v.lower() not in STOP and v.lower() != tgt}
        return sorted(hits)
    return f


# --------------------------------------------------------------------- memories


def memories(C, l, key, combined):
    """gid -> (mem, thr, sd): one delta memory per item (+ the combined RLS memory), per-type baseline."""
    A, ks = C.args, C.KS[l][key]
    st = {b: dk.make_stored(C, l, b, "pooled", ks, False) for b in ("without", "contrastive")}
    mix = {s: st[C.base_of[s]][s] for s in C.ids}
    Kg = dk.gen_keys(C, l, ks)
    out = {s: dk.build(mix, [s], "delta", A.rls_lambda, A.device, C.chk) for s in C.ids}
    if combined:
        out["combined"] = dk.build(mix, C.ids, "rls", A.rls_lambda, A.device, C.chk)
    for gid, mem in list(out.items()):
        thr, sd = dk.calib(mem, Kg, A.thr_q)
        out[gid] = (mem, thr, sd)
        C.thr_log[f"L{l}/{key}/{gid}"] = thr
    return out


def tf_readers(C, layers, M, key, gid, alpha, scale, stash=None):
    """Teacher-forced (full-sequence) readers: [(layer, diag_keys.Reader, alpha * scale)]."""
    return [(l, dk.Reader(M[(l, key)][gid][0], C.KS[l][key], "hard", M[(l, key)][gid][1], M[(l, key)][gid][2], 0,
                          stash or C.stash), alpha * scale) for l in layers]


# ------------------------------------------------------------- teacher-forced


def contrasts(C, scen, prompt):
    out = []
    for c in scen["contrasts"]:
        ids = torch.cat([prompt, text_ids(C.tok, c["prefix"])])
        out.append(v01.lp(C.model, ids, c["a"], C.dev) - v01.lp(C.model, ids, c["b"], C.dev))
    return out


def tf_rows(C, s, readers, tag):
    """v0_1 metrics for item s with `readers` active, + the leakage row (unrelated probes)."""
    scen, model, dev = C.by_id[s], C.model, C.dev
    rows = []
    with injecting(model, readers):
        for r in C.refs[s]:
            row = {**tag, "scenario": s, "type": scen["type"], "kind": r["kind"], "probe": r["probe"]}
            if r["kind"] == "probe":
                lp_mem = mx.cont_logprobs(model, r["base_ids"], r["cont"], dev)
                row.update(distance=r["distance"], kl_base=r["kl_base"], kl_mem=mx.kl(r["lp_ceil"], lp_mem))
                if scen["type"] == "fact":
                    tm, fm = v01.target_scores(model, scen, r["base_ids"], dev)
                    row.update(tgt_base=r["tgt_base"], tgt_ceil=r["tgt_ceil"], tgt_mem=tm, foil_base=r["foil_base"],
                               foil_ceil=r["foil_ceil"], foil_mem=fm)
                else:
                    x = C.xref[(s, r["probe"])]
                    row.update(con_base=x["base"], con_ceil=x["ceil"], con_mem=contrasts(C, scen, r["base_ids"]))
            else:
                row.update(distance="relation", consistent=r["rp"]["consistent"], rel_base=r["rel_base"],
                           rel_ceil=r["rel_ceil"], rel_mem=v01.relation_score(model, r["base_ids"], r["rp"], dev))
            rows.append(row)
        kls = [mx.kl(u["lp_base"], mx.cont_logprobs(model, u["ids"], u["cont"], dev)) for u in C.unrel]
    rows.append({**tag, "scenario": s, "type": scen["type"], "kind": "leak", "kl": mean(kls), "kl_max": max(kls)})
    return rows


def effects(rows):
    pr = [r for r in rows if r["kind"] == "probe"]
    kb, km = sum(r["kl_base"] for r in pr), sum(r["kl_mem"] for r in pr)
    out = {"gap": 1 - km / kb if kb > 0 else float("nan")}
    F_ = [r for r in pr if r["type"] == "fact"]
    D_ = [r for r in pr if r["type"] != "fact"]
    for tag, dists in (("nx", ("paraphrase", "related")), ("rel", ("related",)), ("ex", ("exact",))):
        f = [r for r in F_ if r["distance"] in dists]
        for w in ("mem", "ceil"):
            dt = mean(r[f"tgt_{w}"] - r["tgt_base"] for r in f)
            dfo = mean(mean(m - b for m, b in zip(r[f"foil_{w}"], r["foil_base"])) for r in f)
            sfx = "" if w == "mem" else "_ceil"
            out[f"dt{sfx}_{tag}"], out[f"dfoil{sfx}_{tag}"], out[f"spec{sfx}_{tag}"] = dt, dfo, dt - dfo
            d = [r for r in D_ if r["distance"] in dists]
            out[f"con{sfx}_{tag}"] = mean(m - b for r in d for m, b in zip(r[f"con_{w}"], r["con_base"]))
    rel = [r for r in rows if r["kind"] == "relation"]
    for c in ("Yes", "No"):
        g = [r for r in rel if r["consistent"] == c]
        out[f"drel_{c}"] = mean(r["rel_mem"] - r["rel_base"] for r in g)
        out[f"drel_ceil_{c}"] = mean(r["rel_ceil"] - r["rel_base"] for r in g)
        for w in ("mem", "base", "ceil"):
            out[f"acc_{w}_{c}"] = mean(float(r[f"rel_{w}"] > 0) for r in g)
    out["drel_bal"] = mean([out["drel_Yes"], out["drel_No"]])
    out["acc_bal"] = mean([out["acc_mem_Yes"], out["acc_mem_No"]])
    out["leak"] = mean(r["kl"] for r in rows if r["kind"] == "leak")
    return out


def typed_effects(rows):
    e = effects(rows)
    for typ, p in (("fact", "f_"), ("disposition", "d_")):
        sub = effects([r for r in rows if r["type"] == typ])
        e.update({p + k: sub[k] for k in sub if k.startswith(("drel", "acc", "leak"))})
    return e


# --------------------------------------------------------------------- analytic


def analytic(C, l, key, M):
    """Recall fraction |g M k| / mean|delta| per probe kind (hard read), SI = rf_related / (rf_unrelated
    + SI_EPS), active fraction, template share of steer energy on unrelated probes, final match."""
    P = C.P
    cats, idx = P["cats"], P["idx"]
    n = len(P["meta"])
    nh = cats != 0
    unrel_pos = torch.tensor([m["sid"] is None for m in P["meta"]], device=idx.device)[idx]
    Kr = dk.read_keys(C, l, C.KS[l][key])
    rows = []
    for mode, gids in (("isolated", C.ids), ("combined", ["combined"])):
        acc = defaultdict(list)
        E = torch.zeros(4, device=idx.device)
        for gid in gids:
            mem, thr, sd = M[gid]
            rn = (Kr @ mem.M.T).norm(dim=-1)
            match = (Kr @ mem.Kst.T).max(-1).values
            g = dk.gate_fn("hard", match, cats, thr, sd)
            v = g * rn
            rf, act = dk.seg_mean(v, idx, n), dk.seg_mean(g, idx, n, nh)
            mfin = dk.seg_mean(match, idx, n, cats == 2)  # tail positions: the key of all user text
            for i, m in enumerate(P["meta"]):
                s, k = m["sid"], m["kind"]
                if s is None:
                    cls, dn = "unrelated", mem.dn_mean()
                elif mode == "combined":
                    cls, dn = k, mem.dn_mean(s)
                elif s == gid:
                    cls, dn = k, mem.dn_mean()
                elif k in DIST:
                    cls, dn = "other", mem.dn_mean()
                else:
                    continue
                acc[("rf", cls)].append(rf[i] / dn)
                acc[("act", cls)].append(act[i])
                acc[("mfin", cls)].append(mfin[i] - thr)
            E += torch.stack([((v ** 2) * (unrel_pos & (cats == c))).sum() for c in range(4)])
            acc["thr"].append(thr)
        row = {"layer": l, "ltype": C.ltype[l], "key": key, "mode": mode, "thr": mean(acc["thr"])}
        for k in DIST + ("relation", "other", "unrelated"):
            row[f"rf_{k}"] = mean(acc[("rf", k)])
            row[f"act_{k}"] = mean(acc[("act", k)])
            row[f"margin_{k}"] = mean(acc[("mfin", k)])
        row["si"] = row["rf_related"] / (row["rf_unrelated"] + SI_EPS)
        tot = E.sum().item()
        row["tmpl_share"] = (E[0] + E[2]).item() / tot if tot > 0 else float("nan")
        rows.append(row)
    return rows


# ------------------------------------------------------------------- generation


class PhaseReader:
    """Read hook for KV-cached generation of B rows of one prompt (pooled keys). Prefill = the whole
    prompt: diag_keys' hard read with the prompt's categories, at C.phase.a_prompt * scale. Later calls
    see generated positions only; the pooled key no longer changes, so the last prompt position's gated
    recall is reused, at C.phase.alpha[row] * scale (per-row phase: think / answer)."""

    def __init__(self, C, mem, ks, thr, sd, catfn, layer, scale):
        assert ks.pooled
        self.C, self.mem, self.ks, self.thr, self.sd, self.catfn = C, mem, ks, thr, sd, catfn
        self.layer, self.scale = layer, scale
        self.last = None

    def begin(self, ids):
        self.cats, self.prefilled, self.v = self.catfn(ids), False, None

    def read(self, h, _alpha):
        ph = self.C.phase
        if not self.prefilled:
            assert h.shape[1] == len(self.cats), "prefill: call begin(prompt) first"
            k = self.ks.keys(h[0], self.cats == 1)
            match = (k @ self.mem.Kst.T).max(-1).values
            g = dk.gate_fn("hard", match, self.cats, self.thr, self.sd)
            r = k @ self.mem.M.T
            self.v, self.prefilled = g[-1] * r[-1], True
            self.last = {"layer": self.layer, "match": match[-1].item(), "thr": self.thr, "gate": g[-1].item()}
            return h + (self.scale * ph.a_prompt) * (g[:, None] * r)[None]
        return h + (self.scale * ph.alpha).to(h.dtype).view(-1, 1, 1) * self.v


def plain_sampler(T):
    def f(logits, u, seen):  # samples_v2: plain softmax, inverse CDF
        cdf = torch.softmax(logits.double() / T, -1).cumsum(-1)
        return torch.searchsorted(cdf, u * cdf[:, -1:]).squeeze(1).clamp_max(cdf.shape[1] - 1)
    f.presence = False
    return f


def qwen_sampler(T, top_k, top_p, presence):
    def f(logits, u, seen):  # presence penalty (generated tokens) -> top-k -> temperature -> top-p
        l = logits - presence * seen.float() if presence else logits
        v, i = l.topk(top_k, -1)
        p = torch.softmax(v.double() / T, -1)
        p = p * ((p.cumsum(-1) - p) < top_p)
        c = p.cumsum(-1)
        j = torch.searchsorted(c, u * c[:, -1:]).clamp_max(top_k - 1)
        return i.gather(1, j).squeeze(1)
    f.presence = presence > 0
    return f


def generate(C, prompt, B, readers, *, a_think, a_answer, think, max_new, sampler, seed, greedy0,
             think_cap=10 ** 9, answer_cap=None, keep_logits=False):
    """B continuations of `prompt` (row 0 greedy if greedy0). think: rows start inside the thinking block;
    the phase flips to answer at </think>; </think> is forced after think_cap thinking tokens."""
    dev = C.dev
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    answer_cap = answer_cap or max_new
    x = prompt[None].to(dev).expand(B, -1).contiguous()
    in_ans = torch.full((B,), not think, dtype=torch.bool, device=dev)
    n_think = torch.zeros(B, dtype=torch.long, device=dev)
    n_ans = torch.zeros(B, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    forced = torch.zeros(B, dtype=torch.bool, device=dev)
    seen = torch.zeros(B, C.vocab, dtype=torch.bool, device=dev) if sampler.presence else None
    ar = torch.arange(B, device=dev)
    readers = readers or []
    C.phase.a_prompt = a_answer
    for r in readers:
        r.begin(prompt)
    steps, logs, first, past = [], [], None, None
    with injecting(C.model, [(r.layer, r, 1.0) for r in readers]):
        for t in range(max_new):
            C.phase.alpha = torch.where(in_ans, torch.tensor(a_answer, device=dev), torch.tensor(a_think, device=dev))
            o = C.body(input_ids=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = C.lm_head(o.last_hidden_state[:, -1]).float()
            if t == 0:
                first = torch.log_softmax(logits[0], -1)
            if keep_logits:
                logs.append(logits[0])
            u = torch.rand(B, 1, generator=gen, device=dev, dtype=torch.float64)
            nxt = sampler(logits, u, seen)
            if greedy0:
                nxt[0] = logits[0].argmax()
            active = ~done
            if think:
                force = active & ~in_ans & (n_think >= think_cap)
                nxt = torch.where(force, torch.tensor(C.think_end, device=dev), nxt)
                forced |= force
            nxt = torch.where(done, C.eos[0], nxt)
            steps.append(nxt)
            if seen is not None:
                seen[ar, nxt] |= active
            is_end = active & ~in_ans & (nxt == C.think_end)
            n_think += (active & ~in_ans & ~is_end).long()
            n_ans += (active & in_ans).long()
            in_ans = in_ans | is_end
            done = done | (active & torch.isin(nxt, C.eos)) | (active & in_ans & (n_ans >= answer_cap))
            if bool(done.all()):
                break
            x = nxt[:, None]
    toks = torch.stack(steps, 1).cpu()
    eos = set(C.eos.tolist())
    rows = []
    for b, row in enumerate(toks.tolist()):
        row = row[:next((i for i, t in enumerate(row) if t in eos), len(row))]
        if think:
            e = row.index(C.think_end) if C.think_end in row else None
            th, an = (row, []) if e is None else (row[:e], row[e + 1:])
        else:
            th, an, e = [], row, None
        rows.append({"think_ids": th, "ans_ids": an, "closed": e is not None, "forced": bool(forced[b]),
                     "n_think": len(th), "think": C.tok.decode(th, skip_special_tokens=True).strip(),
                     "answer": C.tok.decode(an, skip_special_tokens=True).strip()})
    return SimpleNamespace(rows=rows, first=first, raw=toks[0], logits=torch.stack(logs) if keep_logits else None,
                           gate=[r.last for r in readers] or None)


def check_cached(C, layers, scale, get, think):
    """KV-cached PhaseReader generation must match diag_keys.Reader on the full sequence, on the first
    fact's exact probe (its memory fires there). get(layer, item) -> (mem, ks, thr, sd); uniform alpha."""
    A = C.args
    s = next(x for x in C.ids if C.by_id[x]["type"] == "fact")
    text = next(p["text"] for p in C.by_id[s]["probes"] if p["distance"] == "exact")
    spec = [(l, *get(l, s), scale) for l in layers]
    prompt = chat(C.tok, text, think)
    catfn, stash = (C.catfn_on, C.stash_on) if think else (C.catfn, C.stash)
    prs = [PhaseReader(C, m, ks, thr, sd, catfn, l, sc) for l, m, ks, thr, sd, sc in spec]
    g = generate(C, prompt, 1, prs, a_think=A.alpha, a_answer=A.alpha, think=think, max_new=8,
                 sampler=plain_sampler(1.0), seed=0, greedy0=True, keep_logits=True)
    full = torch.cat([prompt, g.raw[:len(g.logits) - 1]])[None].to(C.dev)
    rds = [(l, dk.Reader(m, ks, "hard", thr, sd, 0, stash), A.alpha * sc) for l, m, ks, thr, sd, sc in spec]
    with injecting(C.model, rds):
        lr = C.model(full).logits[0, len(prompt) - 1:].float()
    lb = C.model(full).logits[0, len(prompt) - 1:].float()
    res = {"think": think, "layers": [x[0] for x in spec], "steps": len(g.logits),
           "max_abs_diff": (lr - g.logits).abs().max().item(), "memory_effect_max": (lr - lb).abs().max().item(),
           "gate": g.gate}
    log(f"cached-reader check: {json.dumps(res)}")
    assert res["max_abs_diff"] < A.check_tol, f"cached reader != diag_keys.Reader: {res}"
    return res


def seed_of(tag, text, base):
    return s2.seed_of(tag, text, base)


def rep_rate(ids, n=4):
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def has_word(text, w):
    return bool(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", text, re.I))


def parse_yn(text):
    m = re.search(r"\b(yes|no)\b", text, re.I)
    return m.group(1).lower() if m else None


# ------------------------------------------------------------------------ ranks


def ranks(vals, higher=True):
    """1 = best; ties (and nan, the worst) share their mean rank, so an all-nan metric ranks nothing."""
    k = [(1, 0.0) if math.isnan(v) else (0, -v if higher else v) for v in vals]
    order = sorted(range(len(vals)), key=lambda i: k[i])
    out, i = [0.0] * len(vals), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and k[order[j + 1]] == k[order[i]]:
            j += 1
        for t in range(i, j + 1):
            out[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return out


def nanmedian(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


# ====================================================================== stage 1


S1_RULE = ("best 3 layers (key pooled_w256, isolated, alpha {alpha}): eligible = leakage KL <= {mult} x the "
           "median over layers (if fewer than 3 are eligible, the 3 lowest-leakage layers); score = rank(fact "
           "specificity, paraphrase+related probes) + rank(disposition contrast, paraphrase+related) + rank(analytic "
           "SI); lowest score wins, ties -> higher fact specificity.")


def stage1(C, out):
    A = C.args
    rows, sel, t0 = [], [], time.time()
    for i, l in enumerate(C.AL):
        for key in KEYS:
            M = memories(C, l, key, combined=True)
            sel += analytic(C, l, key, M)
            Ml = {(l, key): M}
            for s in C.ids:
                tag = {"layer": l, "ltype": C.ltype[l], "key": key, "cfg": f"L{l}", "mode": "isolated", "memory": s,
                       "thr": M[s][1]}
                rows += tf_rows(C, s, tf_readers(C, [l], Ml, key, s, A.alpha, 1.0), tag)
        el = time.time() - t0
        log(f"stage 1: L{l} ({C.ltype[l]}) done | {el / 60:.1f} min, eta {el / (i + 1) * (len(C.AL) - i - 1) / 60:.1f} min")
    SEL = {(r["layer"], r["key"], r["mode"]): r for r in sel}
    by = defaultdict(list)
    for r in rows:
        by[(r["layer"], r["key"])].append(r)
    summ = []
    for (l, key), rs in sorted(by.items()):
        a = SEL[(l, key, "isolated")]
        summ.append({**{k: a[k] for k in a if k != "mode"}, "mode": "isolated", **typed_effects(rs)})
    # choice
    cand = [r for r in summ if r["key"] == W256]
    med = nanmedian([r["leak"] for r in cand])
    cap = max(A.leak_mult * med, 1e-4) if not math.isnan(med) else float("inf")
    elig = [r for r in cand if r["leak"] <= cap]
    if len(elig) < 3:
        elig = sorted(cand, key=lambda r: r["leak"])[:3]
    rk = [ranks([r[m] for r in elig]) for m in ("spec_nx", "con_nx", "si")]
    for j, r in enumerate(elig):
        r["rank_sum"] = sum(x[j] for x in rk)
    order = sorted(elig, key=lambda r: (r["rank_sum"], -r["spec_nx"] if not math.isnan(r["spec_nx"]) else 0))
    choice = {"best": [r["layer"] for r in order[:3]], "rule": S1_RULE.format(alpha=A.alpha, mult=A.leak_mult),
              "leak_median": med, "leak_cap": cap,
              "ranking": [{k: r[k] for k in ("layer", "ltype", "rank_sum", "spec_nx", "con_nx", "si", "leak")}
                          for r in order]}
    log(f"stage 1 choice: {choice['best']}")
    return rows, summ, sel, choice


def report1(C, summ, sel, choice, meta):
    A = C.args
    L = []
    w = L.append
    f3 = lambda x: fmt(x, "+.3f")
    w(f"Seahorse think_v1 STAGE 1 (layer sweep, thinking off): {A.model}{' TINY ' + A.tiny if A.tiny else ''}")
    w(f"items: dispositions {A.disp_items}; facts {A.fact_items}; {len(C.unrel)} unrelated probes. alpha {A.alpha}, "
      f"read hard (thr = q{A.thr_q} generic match, per memory), pooled writes, delta per item; injection at ONE layer.")
    w(f"template: {json.dumps(C.template)}")
    w("rf_* = recall fraction |g M k| / mean|delta| (all positions); SI = rf_related / (rf_unrelated + 0.01); act = "
      "share of non-head positions where the gate is open; tmpl = steer energy on template positions / all (unrelated "
      "probes); margin = final match - thr. Fact: dt = dlogP(target), spec = dt - mean dlogP(foils); nx = paraphrase + "
      "related probes, rel = related only. con = disposition contrast dlog[P(a)/P(b)] (3 contrasts). drelY/N = "
      "d[logP(consistent) - logP(other)] on Yes- / No-consistent relation probes (f_ facts = verification, d_ "
      "dispositions = inference); gap = KL closure to the ceiling; leak = mean KL on unrelated probes.")
    ref = summ[0]
    w(f"ceilings (experience in context): fact spec nx {f3(ref['spec_ceil_nx'])}, rel {f3(ref['spec_ceil_rel'])}; "
      f"disposition con nx {f3(ref['con_ceil_nx'])}; drel ceil Yes {f3(ref['drel_ceil_Yes'])} No "
      f"{f3(ref['drel_ceil_No'])}; relation accuracy base Y/N {fmt(ref['acc_base_Yes'], '.2f')}/"
      f"{fmt(ref['acc_base_No'], '.2f')}, ceil {fmt(ref['acc_ceil_Yes'], '.2f')}/{fmt(ref['acc_ceil_No'], '.2f')}")
    w(f"checks: {json.dumps(meta['checks'])}")
    w("")
    best = set(choice["best"])
    for key in KEYS:
        w("=" * 100)
        w(f"KEY {key} (isolated, injection at the layer only)")
        hdr = ["L", "type", "thr", "SI", "rf_rel", "rf_par", "rf_unr", "rf_oth", "act_rel", "act_unr", "tmpl", "dt_nx",
               "spec_nx", "spec_rel", "con_nx", "f_drelY", "f_drelN", "d_drelY", "d_drelN", "gap", "leak"]
        rr = []
        for r in [x for x in summ if x["key"] == key]:
            mark = "*" if key == W256 and r["layer"] in best else ""
            rr.append([f"{r['layer']}{mark}", r["ltype"], fmt(r["thr"], ".3f"), fmt(r["si"], ".1f")]
                      + [fmt(r[k], ".3f") for k in ("rf_related", "rf_paraphrase", "rf_unrelated", "rf_other",
                                                    "act_related", "act_unrelated", "tmpl_share")]
                      + [f3(r[k]) for k in ("dt_nx", "spec_nx", "spec_rel", "con_nx", "f_drel_Yes", "f_drel_No",
                                            "d_drel_Yes", "d_drel_No", "gap")] + [fmt(r["leak"], ".4f")])
        w(table(hdr, rr))
        w("")
        w(f"relation accuracy with memory (score > 0), {key}: f_ = verification (facts), d_ = inference (dispositions)")
        w(table(["L", "type", "f_accY", "f_accN", "d_accY", "d_accN", "bal"],
                [[r["layer"], r["ltype"]] + [fmt(r[k], ".2f") for k in ("f_acc_mem_Yes", "f_acc_mem_No", "d_acc_mem_Yes",
                                                                        "d_acc_mem_No", "acc_bal")]
                 for r in summ if r["key"] == key]))
        w("")
    w("=" * 100)
    w("FULL-ATTENTION vs LINEAR-ATTENTION (Gated DeltaNet) layers: means over the layers of each type")
    rr = []
    for key in KEYS:
        for lt in ("full", "linear"):
            sub = [r for r in summ if r["key"] == key and r["ltype"] == lt]
            if sub:
                rr.append([f"{key} {lt}", len(sub)] + [fmt(mean(r[k] for r in sub), f) for k, f in (
                    ("si", ".1f"), ("rf_related", ".3f"), ("rf_unrelated", ".3f"), ("spec_nx", "+.3f"),
                    ("spec_rel", "+.3f"), ("con_nx", "+.3f"), ("drel_bal", "+.3f"), ("gap", "+.3f"), ("leak", ".4f"))]
                          + [max(sub, key=lambda r: r["spec_nx"] if not math.isnan(r["spec_nx"]) else -1e9)["layer"]])
    w(table(["key type", "n", "SI", "rf_rel", "rf_unr", "spec_nx", "spec_rel", "con_nx", "drel_bal", "gap", "leak",
             "best spec L"], rr))
    w("")
    w("COMBINED memory (all items, RLS), analytic only")
    for key in KEYS:
        w(table([f"{key} L", "type", "thr", "SI", "rf_rel", "rf_par", "rf_rel'n", "rf_unr", "act_rel", "act_unr"],
                [[r["layer"], r["ltype"], fmt(r["thr"], ".3f"), fmt(r["si"], ".1f")]
                 + [fmt(r[k], ".3f") for k in ("rf_related", "rf_paraphrase", "rf_relation", "rf_unrelated",
                                               "act_related", "act_unrelated")]
                 for r in sel if r["key"] == key and r["mode"] == "combined"]))
        w("")
    w("=" * 100)
    w("CHOICE: " + choice["rule"])
    w(f"leakage median {fmt(choice['leak_median'], '.4f')}, cap {fmt(choice['leak_cap'], '.4f')}")
    w(table(["L", "type", "rank_sum", "spec_nx", "con_nx", "SI", "leak"],
            [[r["layer"], r["ltype"], r["rank_sum"], f3(r["spec_nx"]), f3(r["con_nx"]), fmt(r["si"], ".1f"),
              fmt(r["leak"], ".4f")] for r in choice["ranking"]]))
    w(f"BEST 3 LAYERS (in rank order): {choice['best']}")
    return "\n".join(L) + "\n"


# ====================================================================== stage 2


S2_RULE = ("injection set for stage 3 (key pooled_w256, isolated): eligible = leakage KL <= {mult} x the single-"
           "layer config's (+0.005) and unrelated greedy identical-to-nomem rate >= the single-layer config's - 0.25; "
           "score = rank(fact spec nx) + rank(disposition contrast nx) + rank(balanced relation drel) + rank(prefix "
           "samples naming the target) + rank(disposition lean, consistent - inconsistent rate); lowest wins, ties -> "
           "fewer layers, then split.")


def configs(best):
    out = []
    for n in (1, 2, 3):
        ls = best[:n]
        for am in (("full",) if n == 1 else ("split", "full")):
            out.append({"name": "+".join(f"L{l}" for l in ls) + ("" if n == 1 else f" {am}"), "layers": ls,
                        "alpha_mode": am, "scale": 1.0 if am == "full" else 1.0 / n})
    return out


def stage2_conditions(C, cfgs, M):
    A = C.args
    conds = {}
    for s in C.ids:
        cs = [SimpleNamespace(cid="nomem", readers=None, tf=[], cache="nomem", ctx=False),
              SimpleNamespace(cid="ctx", readers=None, tf=[], cache=f"ctx/{s}", ctx=True)]
        for cfg in cfgs:
            for mode in ("iso", "comb"):
                gid = s if mode == "iso" else "combined"
                prs = [PhaseReader(C, M[(l, W256)][gid][0], C.KS[l][W256], *M[(l, W256)][gid][1:], C.catfn, l,
                                   cfg["scale"]) for l in cfg["layers"]]
                cs.append(SimpleNamespace(cid=f"{cfg['name']}|{mode}", readers=prs, cfg=cfg, mode=mode, ctx=False,
                                          tf=tf_readers(C, cfg["layers"], M, W256, gid, A.alpha, cfg["scale"]),
                                          cache=f"{cfg['name']}|{mode}" + (f"/{s}" if mode == "iso" else "")))
        conds[s] = cs
    return conds


def prompt2(C, c, s, text):
    return ceiling_ids(C.tok, C.by_id[s]["experience"], text) if c.ctx else chat_ids(C.tok, text)


def run_gen2(C, conds, texts):
    A = C.args
    recs, cache, t0 = [], {}, time.time()
    kw = dict(a_think=A.alpha, a_answer=A.alpha, think=False)
    for s in C.ids:
        scen = C.by_id[s]
        rel = next(p["text"] for p in scen["probes"] if p["distance"] == "related")
        for kind, text in [("related", rel)] + [("unrelated", u) for u in C.unrel_gen]:
            grp = []
            for c in conds[s]:
                if kind == "unrelated" and c.ctx:
                    continue
                key = (c.cache, text)
                if key not in cache:
                    g = generate(C, prompt2(C, c, s, text), 1, c.readers, max_new=A.greedy_len, greedy0=True,
                                 sampler=plain_sampler(1.0), seed=seed_of("greedy", text, A.seed), **kw)
                    cache[key] = {"tokens": g.rows[0]["ans_ids"], "text": g.rows[0]["answer"], "gate": g.gate}
                grp.append({"section": "greedy", "scenario": s, "kind": kind, "prompt": text, "condition": c.cid,
                            **cache[key]})
            nm = next(r for r in grp if r["condition"] == "nomem")["tokens"]
            for r in grp:
                r["identical_to_nomem"] = r["tokens"] == nm
            recs += grp
        if scen["type"] == "fact":
            m = scen["measure"]
            pre, tgt = text_ids(C.tok, m["prefix"]), text_ids(C.tok, m["target"])
            names = [m["target"]] + list(m["foils"])
            firsts = [int(text_ids(C.tok, x)[0]) for x in names]
            for p in scen["probes"]:
                for c in conds[s]:
                    prompt = torch.cat([prompt2(C, c, s, p["text"]), pre])
                    g = generate(C, prompt, 1 + A.prefix_samples, c.readers, max_new=A.prefix_len, greedy0=True,
                                 sampler=plain_sampler(A.prefix_temp), seed=seed_of("prefix", p["text"], A.seed), **kw)
                    smp = g.rows[1:]
                    counts = {n: {"full": sum(s2.lenient_starts(r["answer"], n) for r in smp),
                                  "first_tok": sum(1 for r in smp if r["ans_ids"][:1] == [ft]),
                                  "p_first": g.first[ft].exp().item()} for n, ft in zip(names, firsts)}
                    with injecting(C.model, c.tf):
                        lpt = mx.seq_logprob(C.model, prompt, tgt, C.dev)
                    recs.append({"section": "prefix", "scenario": s, "kind": p["distance"], "prompt": p["text"],
                                 "condition": c.cid, "gate": g.gate, "greedy": g.rows[0]["answer"], "n": len(smp),
                                 "target": m["target"], "foils": list(m["foils"]), "counts": counts, "logp_target": lpt})
                    texts.append({"stage": 2, "section": "prefix", "scenario": s, "prompt": p["text"],
                                  "condition": c.cid, "samples": [r["answer"] for r in g.rows]})
        else:
            for text in scen["ambiguous"][:A.n_ambiguous]:
                for c in conds[s]:
                    g = generate(C, prompt2(C, c, s, text), A.disp_samples, c.readers, max_new=A.disp_len,
                                 greedy0=False, sampler=plain_sampler(A.disp_temp), seed=seed_of("disp", text, A.seed),
                                 **kw)
                    labs = [label(r["answer"], scen["lexicon"]) for r in g.rows]
                    recs.append({"section": "disposition", "scenario": s, "kind": "ambiguous", "prompt": text,
                                 "condition": c.cid, "gate": g.gate, "n": len(labs),
                                 **{k: labs.count(k) for k in ("consistent", "inconsistent", "neutral")},
                                 "examples": [r["answer"] for r in g.rows[:2]]})
                    texts.append({"stage": 2, "section": "disposition", "scenario": s, "prompt": text,
                                  "condition": c.cid, "samples": [r["answer"] for r in g.rows], "labels": labs})
        log(f"stage 2 generation: {s} done ({(time.time() - t0) / 60:.1f} min)")
    return recs


def gen_metrics2(recs):
    agg = defaultdict(lambda: defaultdict(float))
    for r in recs:
        a = agg[r["condition"]]
        if r["section"] == "prefix":
            a["pref_n"] += r["n"]
            a["pref_target"] += r["counts"][r["target"]]["full"]
            a["pref_first"] += r["counts"][r["target"]]["first_tok"]
            a["pref_foils"] += sum(r["counts"][f]["full"] for f in r["foils"])
        elif r["section"] == "disposition":
            for k in ("n", "consistent", "inconsistent", "neutral"):
                a[f"disp_{k}"] += r[k]
        elif r["kind"] == "unrelated" and r["condition"] not in ("nomem", "ctx"):
            a["unrel_n"] += 1
            a["unrel_same"] += r["identical_to_nomem"]
    out = {}
    for cid, a in agg.items():
        dn = a["disp_n"] or float("nan")
        out[cid] = {"prefix_target_rate": a["pref_target"] / a["pref_n"] if a["pref_n"] else float("nan"),
                    "prefix_first_rate": a["pref_first"] / a["pref_n"] if a["pref_n"] else float("nan"),
                    "prefix_foil_rate": a["pref_foils"] / a["pref_n"] if a["pref_n"] else float("nan"),
                    "disp_cons": a["disp_consistent"] / dn, "disp_inc": a["disp_inconsistent"] / dn,
                    "disp_lean": (a["disp_consistent"] - a["disp_inconsistent"]) / dn,
                    "unrel_identical": a["unrel_same"] / a["unrel_n"] if a["unrel_n"] else float("nan")}
    return out


def stage2(C, out, best):
    A = C.args
    cfgs = configs(best)
    M = {(l, key): memories(C, l, key, combined=(key == W256)) for l in best for key in KEYS}
    big = cfgs[-1]
    check = check_cached(C, big["layers"], big["scale"],
                         lambda l, s: (M[(l, W256)][s][0], C.KS[l][W256], *M[(l, W256)][s][1:]), think=False)
    rows, t0 = [], time.time()
    for cfg in cfgs:
        for key, modes in ((W256, ("iso", "comb")), (WLW, ("iso",))):
            for mode in modes:
                groups = [(s, [s]) for s in C.ids] if mode == "iso" else [("combined", C.ids)]
                for gid, members in groups:
                    rd = tf_readers(C, cfg["layers"], M, key, gid, A.alpha, cfg["scale"])
                    tag = {"cfg": cfg["name"], "layers": cfg["layers"], "alpha_mode": cfg["alpha_mode"], "key": key,
                           "mode": mode, "memory": gid}
                    for s in members:
                        rows += tf_rows(C, s, rd, tag)
        log(f"stage 2 teacher-forced: {cfg['name']} done ({(time.time() - t0) / 60:.1f} min)")
    texts = []
    conds = stage2_conditions(C, cfgs, M)
    recs = run_gen2(C, conds, texts)
    G = gen_metrics2(recs)
    by = defaultdict(list)
    for r in rows:
        by[(r["cfg"], r["key"], r["mode"])].append(r)
    summ = []
    for cfg in cfgs:
        for key, mode in ((W256, "iso"), (W256, "comb"), (WLW, "iso")):
            e = typed_effects(by[(cfg["name"], key, mode)])
            g = G.get(f"{cfg['name']}|{mode}", {}) if key == W256 else {}
            summ.append({"cfg": cfg["name"], "layers": "+".join(map(str, cfg["layers"])),
                         "ltypes": "+".join(C.ltype[l] for l in cfg["layers"]), "alpha_mode": cfg["alpha_mode"],
                         "key": key, "mode": mode, **e, **g})
    for cid in ("nomem", "ctx"):
        summ.append({"cfg": cid, "key": "-", "mode": "-", **G.get(cid, {})})
    # choice
    cand = [r for r in summ if r["key"] == W256 and r["mode"] == "iso"]
    ref = cand[0]
    elig = [r for r in cand if r["leak"] <= A.leak_mult * ref["leak"] + 0.005
            and not (r["unrel_identical"] < ref["unrel_identical"] - 0.25)] or cand
    metrics = ("spec_nx", "con_nx", "drel_bal", "prefix_target_rate", "disp_lean")
    rk = [ranks([r[m] for r in elig]) for m in metrics]
    for j, r in enumerate(elig):
        r["rank_sum"] = sum(x[j] for x in rk)
    order = sorted(elig, key=lambda r: (r["rank_sum"], r["layers"].count("+"), r["alpha_mode"] != "split"))
    win = next(c for c in cfgs if c["name"] == order[0]["cfg"])
    choice = {"chosen": win, "rule": S2_RULE.format(mult=A.leak_mult), "from_stage1": best,
              "ranking": [{k: r.get(k) for k in ("cfg", "rank_sum", "leak", "unrel_identical") + metrics} for r in order],
              "ineligible": [r["cfg"] for r in cand if r not in elig]}
    log(f"stage 2 choice: {win}")
    return rows, summ, recs, texts, choice, check, conds


def report2(C, summ, recs, choice, check, meta):
    A = C.args
    L = []
    w = L.append
    f3 = lambda x: fmt(x, "+.3f")
    w(f"Seahorse think_v1 STAGE 2 (single vs multi-layer, thinking off): {A.model}{' TINY ' + A.tiny if A.tiny else ''}")
    w(f"layers from stage 1 (rank order): {choice['from_stage1']}; alpha {A.alpha}: split = alpha/n at each layer, "
      f"full = alpha at each layer. iso = one memory per item (delta), comb = all items in one memory (RLS "
      f"{A.rls_lambda}). Keys at a later layer see the earlier injection; thresholds are calibrated without injection.")
    w("metrics as in stage 1; prefix = samples (T {0:g}, {1} per probe) whose completion starts with the target; "
      "disp = {2} samples x {3} ambiguous prompts at T {4:g}, bench lexicon labels; unrel_same = unrelated greedy "
      "texts identical to no memory.".format(A.prefix_temp, A.prefix_samples, A.disp_samples, A.n_ambiguous, A.disp_temp))
    w(f"cached-reader check: {json.dumps(check)}")
    w(f"checks: {json.dumps(meta['checks'])}")
    w("")
    hdr = ["cfg", "key", "mode", "dt_nx", "spec_nx", "spec_rel", "con_nx", "f_drelY", "f_drelN", "d_drelY", "d_drelN",
           "acc_bal", "gap", "leak", "pref_tgt", "pref_foil", "disp_cons", "disp_inc", "lean", "unrel_same"]
    rr = []
    for r in summ:
        rr.append([r["cfg"], r["key"].replace("pooled_", ""), r["mode"]]
                  + [f3(r.get(k, float("nan"))) for k in ("dt_nx", "spec_nx", "spec_rel", "con_nx", "f_drel_Yes",
                                                          "f_drel_No", "d_drel_Yes", "d_drel_No")]
                  + [fmt(r.get("acc_bal", float("nan")), ".2f"), f3(r.get("gap", float("nan"))),
                     fmt(r.get("leak", float("nan")), ".4f")]
                  + [fmt(r.get(k, float("nan")), ".3f") for k in ("prefix_target_rate", "prefix_foil_rate", "disp_cons",
                                                                  "disp_inc", "disp_lean", "unrel_identical")])
    w(table(hdr, rr))
    w("")
    w("=" * 100)
    w("CHOICE: " + choice["rule"])
    if choice["ineligible"]:
        w(f"ineligible: {choice['ineligible']}")
    w(table(["cfg", "rank_sum", "leak", "unrel_same", "spec_nx", "con_nx", "drel_bal", "pref_tgt", "lean"],
            [[r["cfg"], r["rank_sum"], fmt(r["leak"], ".4f"), fmt(r["unrel_identical"], ".2f")]
             + [fmt(r[k], "+.3f") for k in ("spec_nx", "con_nx", "drel_bal", "prefix_target_rate", "disp_lean")]
             for r in choice["ranking"]]))
    w(f"CHOSEN for stage 3: {choice['chosen']}")
    return "\n".join(L) + "\n"


def samples2(C, conds, recs, path):
    A = C.args
    L = []
    w = L.append
    w(f"# Seahorse think_v1 stage 2 samples (thinking off): {A.model}{' TINY' if A.tiny else ''}")
    w("# (g=gate m=match/thr) at the last prompt position, first injected layer; (=nomem): identical greedy text")
    for s in C.ids:
        scen = C.by_id[s]
        w("")
        w("=" * 110)
        w(f"## {s} ({scen['type']}, baseline {C.base_of[s]}): {scen['experience']}")
        for r in [x for x in recs if x["scenario"] == s and x["section"] == "greedy" and x["kind"] == "related"]:
            gs = f"(g={r['gate'][0]['gate']:.0f} m={r['gate'][0]['match']:.3f}/{r['gate'][0]['thr']:.3f}) " if r["gate"] else ""
            same = "(=nomem) " if r["condition"] != "nomem" and r["identical_to_nomem"] else ""
            w(f"[{r['condition']}]".ljust(28) + f" {gs}{same}{s2.one_line(r['text'])}")
        un = defaultdict(list)
        for r in recs:
            if r["scenario"] == s and r["section"] == "greedy" and r["kind"] == "unrelated" and r["condition"] != "nomem":
                un[r["condition"]].append("same" if r["identical_to_nomem"] else "DIFF")
        w("unrelated greedy vs nomem: " + "; ".join(f"{k}: {' '.join(v)}" for k, v in un.items()))
        for r in [x for x in recs if x["scenario"] == s and x["section"] == "prefix"]:
            tg = r["counts"][r["target"]]
            foils = ", ".join(f"{f.strip()} {r['counts'][f]['full']}" for f in r["foils"])
            w(f"prefix [{r['kind']}] [{r['condition']}]".ljust(44) + f" {r['target'].strip()} {tg['full']}/{r['n']} | "
              f"{foils} | logP {r['logp_target']:.2f} | greedy: {s2.one_line(r['greedy'])}")
        for r in [x for x in recs if x["scenario"] == s and x["section"] == "disposition"]:
            w(f"disp [{r['condition']}] {r['prompt']}".ljust(80) + f" cons {r['consistent']} inc {r['inconsistent']} "
              f"of {r['n']} | e.g. {s2.one_line(r['examples'][0])[:160]}")
    Path(path).write_text("\n".join(L) + "\n")


# ====================================================================== stage 3


S3_RULE = ("best condition for the combined run (isolated results, memory conditions only): score = rank(fact "
           "answers naming the target, related probe) + rank(disposition lean in answers, related + ambiguous) + "
           "rank(balanced yes/no accuracy) + rank(-fact confabulation, related probe, thinking or answer) + "
           "rank(-thinking repeated-4-gram rate); lowest wins, ties -> lower alpha_think + alpha_answer.")


def prompts3(C, s):
    A, scen = C.args, C.by_id[s]
    out = [("related", next(p["text"] for p in scen["probes"] if p["distance"] == "related"), None)]
    out += [(f"rel_{rp['consistent'].lower()}", rp["text"], rp["consistent"]) for rp in scen["relation_probes"]]
    if scen["type"] == "disposition":
        out += [("ambiguous", t, None) for t in scen["ambiguous"][:A.amb3]]
    return out + [("unrelated", u, None) for u in C.unrel_gen]


def row_metrics(C, s, kind, cons, r):
    scen = C.by_id[s]
    th, an = r["think"], r["answer"]
    m = {"n_think": r["n_think"], "closed": r["closed"], "forced": r["forced"], "n_ans": len(r["ans_ids"]),
         "rep_think": rep_rate(r["think_ids"]), "rep_ans": rep_rate(r["ans_ids"])}
    if scen["type"] == "fact":
        tgt = scen["measure"]["target"].strip()
        wt, wa = C.wrong[s](th), C.wrong[s](an)
        m.update(think_mem=has_word(th, tgt), ans_target=has_word(an, tgt), confab_think=bool(wt), confab_ans=bool(wa),
                 wrong=sorted(set(wt) | set(wa)))
        m["mem_word"] = has_word(th + " " + an, tgt)
    else:
        lex = scen["lexicon"]
        ht = lexicon_hits(th, lex)
        m.update(think_mem=ht["consistent"] > 0, think_inc=ht["inconsistent"] > 0, ans_label=label(an, lex))
        m["mem_word"] = lexicon_hits(th + " " + an, lex)["consistent"] > 0
    if cons:
        yn = parse_yn(an)
        m.update(yn=yn, yn_correct=(yn == cons.lower()))
    return m


def stage3_readers(C, cfg, M, gid):
    return [PhaseReader(C, M[l][gid][0], C.KS[l][W256], *M[l][gid][1:], C.catfn_on, l, cfg["scale"])
            for l in cfg["layers"]]


def run3(C, cfg, M, mem, cond_list, rows, texts, shared):
    """Generate every prompt x condition for memory `mem` (iso | comb); appends per-row records."""
    A = C.args
    smp = qwen_sampler(A.temp3, A.top_k, A.top_p, A.presence)
    t0, n_done = time.time(), 0
    for s in C.eval_ids:
        scen = C.by_id[s]
        for kind, text, cons in prompts3(C, s):
            seed = seed_of("s3", text, A.seed)
            for cid, a_t, a_a in cond_list:
                if kind == "unrelated" and cid == "ctx":
                    continue
                gid = s if mem == "iso" else "combined"
                if cid == "nomem":  # shared by every memory (and, for unrelated prompts, every item)
                    ck = ("nomem", text) if kind == "unrelated" else ("nomem", s, text)
                elif mem == "comb" and kind == "unrelated":  # one combined memory for all items
                    ck = ("comb", cid, text)
                else:
                    ck = (mem, cid, s, text)
                if ck not in shared:
                    rd = None if cid in ("nomem", "ctx") else stage3_readers(C, cfg, M, gid)
                    q = f"{scen['experience']} {text}" if cid == "ctx" else text
                    g = generate(C, chat(C.tok, q, True), 1 + A.samples3, rd, a_think=a_t or 0.0,
                                 a_answer=a_a or 0.0, think=True, max_new=A.think_cap + 1 + A.answer_cap,
                                 sampler=smp, seed=seed, greedy0=True, think_cap=A.think_cap, answer_cap=A.answer_cap)
                    shared[ck] = g
                    n_done += 1
                g = shared[ck]
                for b, r in enumerate(g.rows):
                    rec = {"mem": mem, "cond": cid, "a_think": a_t, "a_answer": a_a, "item": s, "type": scen["type"],
                           "kind": kind, "consistent": cons, "prompt": text, "row": b, "greedy": b == 0,
                           "gate": g.gate[0] if g.gate else None, "think_ids": r["think_ids"], "ans_ids": r["ans_ids"],
                           **row_metrics(C, s, kind, cons, r)}
                    if b == 0:
                        rec.update(think_text=r["think"], answer_text=r["answer"])
                    rows.append(rec)
                    texts.append({"stage": 3, "mem": mem, "cond": cid, "item": s, "kind": kind, "prompt": text,
                                  "row": b, "think": r["think"], "answer": r["answer"]})
        el = time.time() - t0
        log(f"stage 3 [{mem}]: {s} done | {n_done} generations, {el / 60:.1f} min")


def agg3(rows):
    """Per (mem, cond): the headline metrics (see report)."""
    G = defaultdict(list)
    for r in rows:
        G[(r["mem"], r["cond"])].append(r)
    nomem_greedy = {(r["prompt"]): r["think_ids"] + [-1] + r["ans_ids"] for r in rows
                    if r["cond"] == "nomem" and r["greedy"] and r["kind"] == "unrelated"}
    nomem_word = {}
    for r in rows:
        if r["cond"] == "nomem" and r["kind"] == "unrelated":
            nomem_word.setdefault(r["item"], []).append(r["mem_word"])
    out = {}
    for (mem, cid), rs in G.items():
        F_ = [r for r in rs if r["type"] == "fact"]
        D_ = [r for r in rs if r["type"] == "disposition"]
        fr = [r for r in F_ if r["kind"] == "related"]
        da = [r for r in D_ if r["kind"] in ("related", "ambiguous")]
        o = {"n_rows": len(rs),
             "f_think_target": mean(float(r["think_mem"]) for r in fr), "f_ans_target": mean(float(r["ans_target"]) for r in fr),
             "f_confab_think": mean(float(r["confab_think"]) for r in fr),
             "f_confab_ans": mean(float(r["confab_ans"]) for r in fr),
             "f_confab": mean(float(r["confab_think"] or r["confab_ans"]) for r in fr),
             "d_think_lex": mean(float(r["think_mem"]) for r in da),
             "d_cons": mean(float(r["ans_label"] == "consistent") for r in da),
             "d_inc": mean(float(r["ans_label"] == "inconsistent") for r in da)}
        o["d_lean"] = o["d_cons"] - o["d_inc"]
        for typ, sub in (("f", F_), ("d", D_), ("all", rs)):
            for c in ("yes", "no"):
                g = [r for r in sub if r["kind"] == f"rel_{c}"]
                o[f"{typ}_acc_{c}"] = mean(float(r["yn_correct"]) for r in g)
                o[f"{typ}_parsed_{c}"] = mean(float(r["yn"] is not None) for r in g)
            o[f"{typ}_acc_bal"] = mean([o[f"{typ}_acc_yes"], o[f"{typ}_acc_no"]])
        nu = [r for r in rs if r["kind"] != "unrelated"]
        o.update(rep_think=mean(r["rep_think"] for r in nu), rep_ans=mean(r["rep_ans"] for r in nu),
                 think_len=mean(r["n_think"] for r in nu), forced=mean(float(r["forced"]) for r in nu),
                 closed=mean(float(r["closed"]) for r in nu))
        un = [r for r in rs if r["kind"] == "unrelated"]
        ug = [r for r in un if r["greedy"]]
        o["unrel_identical"] = mean(float(r["think_ids"] + [-1] + r["ans_ids"] == nomem_greedy.get(r["prompt"])) for r in ug)
        o["unrel_mem_word"] = mean(float(r["mem_word"]) for r in un)
        o["unrel_mem_word_nomem"] = mean(float(x) for r in un for x in [mean(nomem_word.get(r["item"], [float("nan")]))])
        o["unrel_contam"] = o["unrel_mem_word"] - o["unrel_mem_word_nomem"]
        out[(mem, cid)] = o
    return out


def stage3(C, out, cfg):
    A = C.args
    C.eval_ids = [s for s in C.ids if A.eval_items is None or s in A.eval_items]
    assert C.eval_ids, f"--eval-items {A.eval_items} not among {C.ids}"
    M = {l: memories(C, l, W256, combined=True) for l in cfg["layers"]}
    check = check_cached(C, cfg["layers"], cfg["scale"],
                         lambda l, s: (M[l][s][0], C.KS[l][W256], *M[l][s][1:]), think=True)
    rows, texts, shared = [], [], {}
    conds = COND3 + [("ctx", None, None)]
    run3(C, cfg, M, "iso", conds, rows, texts, shared)
    S = agg3(rows)
    mem_conds = [c for c in COND3 if c[0] != "nomem"]
    vals = {m: [S[("iso", c[0])][m] for c in mem_conds] for m in ("f_ans_target", "d_lean", "all_acc_bal", "f_confab",
                                                                   "rep_think")}
    rk = [ranks(vals["f_ans_target"]), ranks(vals["d_lean"]), ranks(vals["all_acc_bal"]),
          ranks(vals["f_confab"], higher=False), ranks(vals["rep_think"], higher=False)]
    score = [sum(x[j] for x in rk) for j in range(len(mem_conds))]
    best_i = min(range(len(mem_conds)), key=lambda j: (score[j], mem_conds[j][1] + mem_conds[j][2]))
    best = mem_conds[best_i]
    log(f"stage 3: best isolated condition {best}; combined-RLS run")
    run3(C, cfg, M, "comb", [("nomem", 0.0, 0.0), best], rows, texts, shared)
    S = agg3(rows)
    choice = {"best_condition": best, "rule": S3_RULE,
              "ranking": [{"cond": c[0], "score": score[j], **{m: vals[m][j] for m in vals}} for j, c in enumerate(mem_conds)]}
    return rows, texts, S, choice, check


def report3(C, cfg, S, choice, check, meta):
    A = C.args
    L = []
    w = L.append
    w(f"Seahorse think_v1 STAGE 3 (strength during thinking vs answer, thinking ON): {A.model}"
      f"{' TINY ' + A.tiny if A.tiny else ''}")
    w(f"injection set (stage 2 choice): {json.dumps(cfg)}; key pooled_w256, read hard; isolated = delta per item; comb "
      f"= all items in one RLS memory. Prompt positions use alpha_answer; generated tokens before </think> alpha_think, "
      f"</think> and after alpha_answer (x the per-layer scale). Prompts of: {C.eval_ids}.")
    w(f"generation: thinking cap {A.think_cap} tokens (then </think> is forced), answer cap {A.answer_cap}; per prompt x "
      f"condition 1 greedy + {A.samples3} samples (T {A.temp3:g}, top-p {A.top_p:g}, top-k {A.top_k}, presence penalty "
      f"{A.presence:g} on generated tokens), the same random numbers for every condition; thresholds calibrated with "
      f"thinking off (pooled keys at generated positions do not depend on the generated text).")
    w("metrics over all rows (greedy + samples). facts (related probe): think_tgt = thinking mentions the target; "
      "ans_tgt = answer names the exact target; confab = a foil / the counter value / a non-target value asserted after "
      "the measure phrase (\"dog's name is X\") in thinking (t) or answer (a). dispositions (related + ambiguous): "
      "think_lex = thinking has a consistent lexicon word; cons/inc = answer label (bench lexicon), lean = cons - inc. "
      "yes/no: first yes|no in the answer; accY/accN = correct on Yes- / No-consistent probes (unparsed = wrong), bal = "
      "their mean. rep = repeated 4-gram rate (non-unrelated prompts); len = thinking tokens; forced = thinking hit the "
      "cap. unrelated: same = greedy thinking + answer identical to no memory; contam = rows with a memory word minus "
      "the no-memory rate.")
    w(f"cached-reader check: {json.dumps(check)}")
    w(f"checks: {json.dumps(meta['checks'])}")
    w("")
    hdr = ["mem cond", "a_t", "a_a", "f think_tgt", "f ans_tgt", "f confab t", "f confab a", "f accY", "f accN",
           "d think_lex", "d cons", "d inc", "d lean", "d accY", "d accN", "bal", "rep_t", "rep_a", "len", "forced",
           "unrel same", "contam"]
    av = {c: (t, a) for c, t, a in COND3 + [("ctx", None, None)]}
    rr = []
    for (mem, cid), o in sorted(S.items(), key=lambda kv: (kv[0][0] != "iso", [c for c, _, _ in COND3 + [("ctx", 0, 0)]].index(kv[0][1]))):
        t, a = av[cid]
        rr.append([f"{mem} {cid}", fmt(t, "g") if t is not None else "-", fmt(a, "g") if a is not None else "-"]
                  + [fmt(o[k], ".2f") for k in ("f_think_target", "f_ans_target", "f_confab_think", "f_confab_ans",
                                                "f_acc_yes", "f_acc_no", "d_think_lex", "d_cons", "d_inc")]
                  + [fmt(o["d_lean"], "+.2f")] + [fmt(o[k], ".2f") for k in ("d_acc_yes", "d_acc_no", "all_acc_bal",
                                                                            "rep_think", "rep_ans")]
                  + [fmt(o["think_len"], ".0f"), fmt(o["forced"], ".2f"), fmt(o["unrel_identical"], ".2f"),
                     fmt(o["unrel_contam"], "+.2f")])
    w(table(hdr, rr))
    w("")
    w("CHOICE (combined run): " + choice["rule"])
    w(table(["cond", "score", "f_ans_tgt", "d_lean", "bal", "f_confab", "rep_t"],
            [[r["cond"], r["score"]] + [fmt(r[k], ".3f") for k in ("f_ans_target", "d_lean", "all_acc_bal", "f_confab",
                                                                   "rep_think")] for r in choice["ranking"]]))
    w(f"best isolated condition -> combined run: {choice['best_condition']}")
    return "\n".join(L) + "\n"


def samples3(C, rows, path):
    A = C.args
    L = []
    w = L.append
    w(f"# Seahorse think_v1 stage 3 thinking traces: {A.model}{' TINY' if A.tiny else ''}; per item x condition: the "
      f"related probe (greedy + sample 1) and one more prompt (greedy; ambiguous for dispositions, the first "
      f"verification probe for facts). g = gate/match/thr at the last prompt position (first injected layer).")
    for s in C.eval_ids:
        scen = C.by_id[s]
        w("")
        w("=" * 110)
        w(f"## {s} ({scen['type']}): {scen['experience']}")
        third = "ambiguous" if scen["type"] == "disposition" else "rel_yes"
        keys = sorted({(r["mem"], r["cond"]) for r in rows if r["item"] == s}, key=lambda k: (k[0] != "iso", k[1]))
        for mem, cid in keys:
            rs = [r for r in rows if r["item"] == s and r["mem"] == mem and r["cond"] == cid]
            show = [r for r in rs if r["kind"] == "related" and r["row"] in (0, 1)]
            show += [next((r for r in rs if r["kind"] == third and r["row"] == 0), None)]
            w(f"--- [{mem} {cid}] ---")
            for r in [x for x in show if x is not None]:
                txt = next((t for t in C._texts if t["mem"] == mem and t["cond"] == cid and t["item"] == s
                            and t["prompt"] == r["prompt"] and t["row"] == r["row"]), None)
                g = r["gate"]
                gs = f" g={g['gate']:.0f} m={g['match']:.3f}/{g['thr']:.3f}" if g else ""
                w(f"[{r['kind']} {'greedy' if r['greedy'] else 'sample ' + str(r['row'])}{gs}] {r['prompt']}")
                w(f"  THINK ({r['n_think']} tok, {'forced' if r['forced'] else 'closed' if r['closed'] else 'open'}): "
                  f"{s2.one_line(txt['think']) if txt else ''}")
                w(f"  ANSWER: {s2.one_line(txt['answer']) if txt else ''}")
        nm = {r["prompt"]: r["think_ids"] + [-1] + r["ans_ids"] for r in rows
              if r["cond"] == "nomem" and r["greedy"] and r["kind"] == "unrelated"}
        same = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["item"] == s and r["kind"] == "unrelated" and r["greedy"] and r["cond"] not in ("nomem", "ctx"):
                same[(r["mem"], r["cond"])][0] += int(r["think_ids"] + [-1] + r["ans_ids"] == nm.get(r["prompt"]))
                same[(r["mem"], r["cond"])][1] += 1
        w("unrelated greedy (thinking + answer) identical to nomem: "
          + "; ".join(f"{m} {c}: {a}/{b}" for (m, c), (a, b) in same.items()))
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------- I/O


def write_jsonl(path, rows, drop=()):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k not in drop}) + "\n")


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prev = None
    if args.stage == 1:
        args.layers_used = args.layers
    elif args.stage == 2:
        assert args.s1, "--s1 (stage 1 output dir) is required"
        prev = json.load(open(Path(args.s1) / "choice.json"))
        args.layers_used = sorted(prev["best"])
    else:
        assert args.s2, "--s2 (stage 2 output dir) is required"
        prev = json.load(open(Path(args.s2) / "choice.json"))
        args.layers_used = sorted(prev["chosen"]["layers"])
    log(f"stage {args.stage}: layers {args.layers_used}; out {out}")
    with torch.inference_mode():
        C = setup(args, out)
        log(f"setup done in {(time.time() - t0) / 60:.1f} min; layer types "
            f"{ {l: C.ltype[l] for l in args.layers_used} }")
        meta = {"args": vars(args), "ids": C.ids, "baseline_of": C.base_of, "template": C.template,
                "layer_types": C.ltype, "white_info": C.white_info, "mu_skip_tokens": C.skip, "tail_len": C.tail_len,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "linear_attention_kernels": _kernels()}
        if args.stage == 1:
            rows, summ, sel, choice = stage1(C, out)
            meta["checks"] = dict(C.chk)
            write_jsonl(out / "results.jsonl", rows)
            dk.write_csv(out / "summary.csv", summ)
            dk.write_csv(out / "selectivity.csv", sel)
            json.dump(choice, open(out / "choice.json", "w"), indent=2)
            (out / "report.txt").write_text(report1(C, summ, sel, choice, meta))
        elif args.stage == 2:
            rows, summ, recs, texts, choice, check, conds = stage2(C, out, prev["best"])
            meta["checks"] = dict(C.chk)
            choice["stage1_dir"] = args.s1
            write_jsonl(out / "results.jsonl", rows + recs)
            dk.write_csv(out / "summary.csv", summ)
            json.dump(choice, open(out / "choice.json", "w"), indent=2)
            with gzip.open(out / "texts.jsonl.gz", "wt") as f:
                for t in texts:
                    f.write(json.dumps(t) + "\n")
            (out / "report.txt").write_text(report2(C, summ, recs, choice, check, meta))
            samples2(C, conds, recs, out / "samples.txt")
            meta["reader_check"] = check
            s1cfg = json.load(open(Path(args.s1) / "config.json"))
            meta["thr_vs_stage1"] = _thr_diff(C.thr_log, s1cfg.get("thresholds", {}))
        else:
            cfg = prev["chosen"]
            rows, texts, S, choice, check = stage3(C, out, cfg)
            meta["checks"] = dict(C.chk)
            C._texts = texts
            choice["stage2_dir"] = args.s2
            write_jsonl(out / "results.jsonl", rows, drop=("think_ids", "ans_ids"))
            dk.write_csv(out / "summary.csv", [{"mem": m, "cond": c, **o} for (m, c), o in S.items()])
            json.dump(choice, open(out / "choice.json", "w"), indent=2)
            with gzip.open(out / "texts.jsonl.gz", "wt") as f:
                for t in texts:
                    f.write(json.dumps(t) + "\n")
            (out / "report.txt").write_text(report3(C, cfg, S, choice, check, meta))
            samples3(C, rows, out / "samples.txt")
            meta["reader_check"] = check
            s2cfg = json.load(open(Path(args.s2) / "config.json"))
            meta["thr_vs_stage2"] = _thr_diff(C.thr_log, s2cfg.get("thresholds", {}))
    meta.update(thresholds=C.thr_log, minutes=(time.time() - t0) / 60)
    json.dump(meta, open(out / "config.json", "w"), indent=2, default=str)
    if "thr_vs_stage1" in meta or "thr_vs_stage2" in meta:
        log(f"thresholds vs previous stage: {meta.get('thr_vs_stage1') or meta.get('thr_vs_stage2')}")
    log(f"stage {args.stage} done in {(time.time() - t0) / 60:.1f} min -> {out}")


def _thr_diff(now, before):
    common = [k for k in now if k in before]
    return {"n": len(common), "max_abs_diff": max((abs(now[k] - before[k]) for k in common), default=float("nan"))}


def _kernels():
    try:
        from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
        return {"flash_linear_attention": is_flash_linear_attention_available(),
                "causal_conv1d": is_causal_conv1d_available()}
    except ImportError:
        return None


if __name__ == "__main__":
    main()
