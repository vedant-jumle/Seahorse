#!/usr/bin/env python
"""Seahorse ref_v1: what should a preference shift be measured AGAINST?

A preference is stored as a shift of the residual stream: the state with the experience minus a
REFERENCE state, pooled over each follow-up moment. think_v1 used the opposite ("counter") as the
reference: it worked for vegetarian (veg vs meat) but did nothing for jazz (the counter "I can't stand
jazz music" also says jazz, so jazz cancels) and flooded Norway (the counter is one arbitrary country).
Hypothesis: "opposite" only suits two-ended attributes; one-of-many attributes need another reference.

References (per follow-up moment, the same entropy-weighted pooling over the follow-up's words as
diag_keys.unit_writes, template tail excluded; experiments/ref_v1/shifts.py):
  opposite    with - counter                                   (the current method)
  without     with - the follow-up alone                       (plain)
  hum         with - mu, the generic-prompt mean used for whitening   (control: ~ the key, re-injects topic)
  disclosure  with - mean over K=24 runs of the same follow-up preceded by OTHER bench_v1 experiences
              (facts + dispositions; the item, its category and overlapping topics excluded; seeded)
  centroid    with - mean over 6-8 hand-written same-category alternatives (config.yaml)
Every shift is rescaled to the norm of the `without` shift of the same moment and layer (directions are
compared; raw norms are logged).

Fixed: Qwen/Qwen3.5-2B fp32, thinking OFF (enable_thinking=False, the default template); injection at
the outputs of layers 23, 20, 21, each layer with its own keys and M; key pooled_w256 (calibrated as
think_v1 / diag_keys.setup: mu, PCA-256 whitening on generic prompts); read hard (match > the q0.95
match on generic prompts + greedy continuations, per memory; never on the template head);
h <- h + alpha * gate * M q; isolated memories (one per item, 3 moments, delta rule).

Conditions per item: nomem, ctx (experience in the prompt: the ceiling), 5 references x alpha per
layer in --doses (1, 2). Generation (KV-cached think_v1 PhaseReader): the related + ambiguous prompts,
1 greedy + --samples samples (T 1, top-p 0.95, top-k 20, presence penalty 1.5), answer cap 200, the same
random numbers in every condition; the unrelated prompts greedy only.

Metrics: cons / inc / lean (bench lexicon labels), loop (repeated-4-gram rate >= 0.3), lean_clean (lean
over non-loop answers), unrel_same, contam; logit lens of every stored shift (final norm + lm_head): top
tokens and lex_gain; cosines between references and with the key.

Outputs (--out): report.txt, summary.csv, results.jsonl, texts.jsonl.gz, samples.txt, vocab.txt,
config.json. --tiny: 2 items, 1 dose, few rows. --random-model: a tiny random model (CPU checks).
"""

import argparse
import gzip
import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
import yaml

from seahorse import metrics as mx
from seahorse.bench import label, lexicon_hits, load_bench
from seahorse.bench.data import pool_items
from seahorse.residual import capture, text_model
from seahorse.sessions import ceiling_ids, chat_ids, common_suffix_len

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tv1 = _load("seahorse_think_v1_run", HERE.parent / "think_v1" / "run.py")
sh = _load("seahorse_ref_v1_shifts", HERE / "shifts.py")
dk, v01, s2 = tv1.dk, tv1.v01, tv1.s2
log, fmt, table = dk.log, dk.fmt, dk.table
nanmean = sh.nanmean

W256 = "pooled_w256"
REFS = ("opposite", "without", "hum", "disclosure", "centroid")
TAGS = ("two_ended", "one_of_many", "negated")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--bench", default=str(REPO / "data" / "bench_v1"))
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--items", nargs="+", default=None, help="subset of the config's item ids (default all)")
    p.add_argument("--layers", type=int, nargs="+", default=[23, 20, 21])
    p.add_argument("--doses", type=float, nargs="+", default=[1.0, 2.0], help="alpha per layer")
    p.add_argument("--refs", nargs="+", default=list(REFS), choices=REFS, help="references that are generated")
    p.add_argument("--thr-q", type=float, default=0.95)
    p.add_argument("--white-eps", type=float, default=0.01)
    p.add_argument("--rls-lambda", type=float, default=0.1, help="(unused by the delta rule; passed through)")
    p.add_argument("--generic", default=str(v01.GENERIC))
    p.add_argument("--cont-len", type=int, default=20, help="diag_keys.setup: unrelated-probe continuations")
    p.add_argument("--gen-cont-len", type=int, default=20, help="greedy continuation of generic prompts (calibration)")
    p.add_argument("--n-ambiguous", type=int, default=5)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--answer-cap", type=int, default=200)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence", type=float, default=1.5)
    p.add_argument("--unrel-gen", type=int, nargs="+", default=[0, 2, 3, 5],
                   help="indices of bench_v1 unrelated_probes generated greedily (selectivity)")
    p.add_argument("--loop-n", type=int, default=4)
    p.add_argument("--loop-thr", type=float, default=0.3)
    p.add_argument("--topn", type=int, default=20, help="logit-lens top tokens shown")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-tol", type=float, default=0.05)
    p.add_argument("--tiny", action="store_true", help="smoke run: 2 items, 1 dose, few rows")
    p.add_argument("--random-model", default=None, choices=["qwen2", "qwen3_5"],
                   help="tiny random model with --model's tokenizer (CPU checks; think_v1's --tiny)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.tiny:
        a.items = a.items or ["vegetarian", "loves_jazz"]
        a.doses = a.doses[-1:]
        a.n_ambiguous, a.samples, a.answer_cap, a.unrel_gen = 1, 2, 24, a.unrel_gen[:1]
    return a


# ------------------------------------------------------------------------ items


def related_of(scen):
    return next(p["text"] for p in scen["probes"] if p["distance"] == "related")


def load_items(A, bench):
    cfg = yaml.safe_load(open(A.config))
    by = {it["id"]: it for it in bench["scenarios"]}
    items = []
    for c in cfg["items"]:
        if A.items and c["id"] not in A.items:
            continue
        if "new" in c:
            n = c["new"]
            it = {k: n[k] for k in ("category", "experience", "counter", "related", "lexicon")}
            it.update(followups=list(n["followups"]), ambiguous=list(n["ambiguous"]), source="ref_v1 config")
        elif "negate" in c:
            b = by[c["negate"]["of"]]
            it = {"category": b["category"], "experience": c["negate"]["experience"], "counter": c["negate"]["counter"],
                  "followups": list(b["followups"]), "related": related_of(b), "ambiguous": list(b["ambiguous"]),
                  "lexicon": {"consistent": b["lexicon"]["inconsistent"], "inconsistent": b["lexicon"]["consistent"]},
                  "source": f"bench_v1 {b['id']}, negated (lexicon sides swapped)", "exclude_ids": [b["id"]]}
        else:
            b = by[c["id"]]
            assert b["type"] == "disposition", c["id"]
            it = {k: b[k] for k in ("category", "experience", "counter", "lexicon")}
            it.update(followups=list(b["followups"]), related=related_of(b), ambiguous=list(b["ambiguous"]),
                      source="bench_v1")
        it.update(id=c["id"], tag=c["tag"], topics=list(c.get("topics", [])), centroid=list(c["centroid"]))
        assert it["tag"] in TAGS, (it["id"], it["tag"])
        assert 6 <= len(it["centroid"]) <= 8 and it["experience"] not in it["centroid"], it["id"]
        assert len(it["followups"]) == 3 and len(it["ambiguous"]) >= A.n_ambiguous, it["id"]
        items.append(it)
    missing = set(A.items or []) - {it["id"] for it in items}
    assert not missing, f"--items not in {A.config}: {missing}"
    return items, cfg


def disclosure_candidates(bench, dcfg):
    """bench_v1 dispositions + facts (one per slot: the hand-written ones, then the pool's other slots)."""
    topics = dcfg["candidate_topics"]
    out, hand_slots = [], set()
    for it in bench["scenarios"]:
        if it["category"] == "emotional":
            continue
        if it["type"] == "disposition":
            out.append({"id": it["id"], "kind": "disposition", "category": it["category"],
                        "topics": topics.get(it["id"], []), "experience": it["experience"]})
        else:
            hand_slots.add(it["slot"])
            out.append({"id": it["id"], "kind": "fact", "category": it["category"],
                        "topics": topics.get(it["slot"], []), "experience": it["experience"]})
    for it in pool_items(bench["pool"], seed=dcfg["seed"]):
        if it["slot"] not in hand_slots:
            out.append({"id": it["id"], "kind": "fact", "category": "pool", "topics": topics.get(it["slot"], []),
                        "experience": it["experience"]})
    return out


# ------------------------------------------------------------------------ setup


def setup(A, out):
    bench = load_bench(A.bench)
    items, cfg = load_items(A, bench)
    unrel = [bench["unrelated_probes"][i] for i in A.unrel_gen]
    path = out / "dk_input.json"  # diag_keys.setup: model, mu, whitening, calibration (no scenarios needed)
    json.dump({"scenarios": [], "unrelated_probes": unrel}, open(path, "w"), indent=1)
    generic = A.generic
    if A.random_model:
        generic = out / "generic_tiny.txt"
        generic.write_text("\n".join([l for l in open(A.generic) if l.strip()][:16]))
    dargs = SimpleNamespace(model=A.model, layers=A.layers, analytic_layers=A.layers, scenarios=str(path),
                            generic=str(generic), cont_len=A.cont_len, gen_cont_len=A.gen_cont_len,
                            white_eps=A.white_eps, tiny=bool(A.random_model), device=A.device)
    dk.load = lambda _a: tv1.load_any(SimpleNamespace(tiny=A.random_model, model=A.model, device=A.device))
    dk.KEY_SPECS = {W256: (("pca", 256), True)}
    C = dk.setup(dargs)
    tok, model, dev = C.tok, C.model, A.device
    C.args, C.dev, C.chk = A, dev, defaultdict(float)
    C.items, C.layers = items, list(A.layers)
    C.body, C.lm_head = text_model(model), model.lm_head
    C.vocab = C.lm_head.out_features
    eos = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            eos.add(i)
    ge = model.generation_config.eos_token_id
    eos |= set(ge if isinstance(ge, (list, tuple)) else [ge])
    C.eos = torch.tensor(sorted(e for e in eos if e is not None), device=dev)
    te = tok.convert_tokens_to_ids("</think>")
    C.think_end = te if isinstance(te, int) and te != tok.unk_token_id else -1  # unused: thinking is off
    C.phase = SimpleNamespace(a_prompt=0.0, alpha=None)
    off = chat_ids(tok, "alpha")
    assert torch.equal(off, tv1.chat(tok, "alpha", False)), "default template != enable_thinking=False"
    C.template = {"head": tok.decode(C.head), "tail": tok.decode(off[-C.tail_len:]), "eos": C.eos.tolist()}
    log(f"template {json.dumps(C.template)}")
    C.unrel_gen = unrel
    C.cands = disclosure_candidates(bench, cfg["disclosure"])
    dc = cfg["disclosure"]
    for it in items:
        it["disclosure"] = sh.pick_disclosure(C.cands, it, dc["k"], dc["k_disp_max"], dc["seed"])
    C.dcfg = {k: v for k, v in dc.items() if k != "candidate_topics"}
    C.Kg = {l: dk.gen_keys(C, l, C.KS[l][W256]) for l in C.layers}
    C.encode = lambda t: tok(t, add_special_tokens=False).input_ids
    return C


# ------------------------------------------------------------------------ writes


def capture_run(C, ids, logits=False):
    with capture(C.model, C.layers) as st:
        out = C.model(ids[None].to(C.dev))
    return {l: st[l] for l in C.layers}, (out.logits[0].float() if logits else None)


def build_moments(C, it, check=False):
    """Per follow-up: the raw shift for every reference and layer, the write key, the topic vector."""
    tok = C.tok
    moments = []
    for j, fu in enumerate(it["followups"]):
        s_with, s_wo, s_c = chat_ids(tok, f"{it['experience']} {fu}"), chat_ids(tok, fu), chat_ids(tok, f"{it['counter']} {fu}")
        s_d = [chat_ids(tok, f"{e['experience']} {fu}") for e in it["disclosure"]]
        s_a = [chat_ids(tok, f"{a} {fu}") for a in it["centroid"]]
        n_wo = common_suffix_len(s_with, s_wo)
        n = min([n_wo, common_suffix_len(s_with, s_c)] + [common_suffix_len(s_with, s) for s in s_d + s_a])
        C.chk["moments"] += 1
        C.chk["n_below_without_suffix"] += int(n != n_wo)
        st_wo, lg_wo = capture_run(C, s_wo, logits=True)
        w = sh.moment_weights(mx.entropy(lg_wo[-n:]), C.tail_len)
        pooled = lambda st: {l: sh.pool(st[l][-n:], w) for l in C.layers}
        p_with, p_wo, p_c = pooled(capture_run(C, s_with)[0]), pooled(st_wo), pooled(capture_run(C, s_c)[0])
        p_d = [pooled(capture_run(C, s)[0]) for s in s_d]
        p_a = [pooled(capture_run(C, s)[0]) for s in s_a]
        cats = C.catfn(s_wo)
        m = {"j": j, "followup": fu, "n": n, "n_without": n_wo, "n_content": int(w.shape[0]), "raw": {}, "key": {},
             "topic": {}}
        for l in C.layers:
            ks = C.KS[l][W256]
            m["key"][l] = ks.keys(st_wo[l], cats == 1)[-1:]  # the pooled key over the follow-up's user text
            m["topic"][l] = (st_wo[l][cats == 1] - C.mu[l]).mean(0)
            m["raw"][l] = {"opposite": sh.reference_shift(p_with[l], p_c[l]),
                           "without": sh.reference_shift(p_with[l], p_wo[l]),
                           "hum": sh.reference_shift(p_with[l], C.mu[l]),
                           "disclosure": sh.reference_shift(p_with[l], [p[l] for p in p_d]),
                           "centroid": sh.reference_shift(p_with[l], [p[l] for p in p_a])}
        if check:  # opposite / without must equal think_v1's write (diag_keys.unit_writes, pooled)
            ch = v01.collect_writes(C.model, tok, {"experience": it["experience"], "counter": it["counter"],
                                                   "followups": [fu]}, C.layers, C.dev)[0]
            full = {**st_wo, "cats": cats}
            for l in C.layers:
                for base, ref in (("contrastive", "opposite"), ("without", "without")):
                    d, k, _ = dk.unit_writes(ch, full, l, base, "pooled", C.KS[l][W256], False, C.tail_len)
                    rel = ((d[0] - m["raw"][l][ref]).norm() / d[0].norm().clamp_min(1e-12)).item()
                    krel = (k - m["key"][l]).norm().item()
                    C.chk["repro_n"] += 1
                    C.chk["repro_shift_max_rel"] = max(C.chk["repro_shift_max_rel"], rel)
                    C.chk["repro_key_max_abs"] = max(C.chk["repro_key_max_abs"], krel)
                    if ch["n"] == n:
                        assert rel < 1e-3 and krel < 1e-3, f"{it['id']} L{l} {ref}: != diag_keys.unit_writes ({rel}, {krel})"
        moments.append(m)
    return moments


def memories(C, it, moments):
    """(ref, layer) -> (Mem, thr, sd): delta rule over the 3 norm-matched moments; per-layer thresholds."""
    A, out = C.args, {}
    for l in C.layers:
        for r in REFS:
            chunks = [(sh.match_norm(m["raw"][l][r], m["raw"][l]["without"].norm())[None], m["key"][l], None)
                      for m in moments]
            mem = dk.build({it["id"]: chunks}, [it["id"]], "delta", A.rls_lambda, A.device, C.chk)
            thr, sd = dk.calib(mem, C.Kg[l], A.thr_q)
            out[(r, l)] = (mem, thr, sd)
        C.thr_log[f"{it['id']}/L{l}"] = out[(REFS[0], l)][1]
    return out


# ------------------------------------------------------------- shift geometry


def logit_lens(C, s):
    return C.lm_head(C.body.norm(s[None, None].float()))[0, 0].float()


def tokstr(C, i):
    return C.tok.decode([i]).replace(" ", "▁").replace("\n", "\\n")


def geometry(C, it, moments):
    """Raw norms, cosines between references / with the key / with the topic, logit lens (lex_gain, top)."""
    A = C.args
    cons_ids, inc_ids = sh.lexicon_token_ids(C.encode, it["lexicon"])
    rows, lens = [], {}
    for l in C.layers:
        W = C.KS[l][W256].W
        for r in REFS:
            lls = []
            for m in moments:
                s = m["raw"][l][r]
                ll = logit_lens(C, s)
                lls.append(ll)
                rows.append({"section": "shift", "item": it["id"], "tag": it["tag"], "layer": l, "ref": r,
                             "moment": m["j"], "raw_norm": s.norm().item(),
                             "norm_ratio": s.norm().item() / m["raw"][l]["without"].norm().clamp_min(1e-12).item(),
                             "cos_key": sh.cos(s @ W, m["key"][l][0]), "cos_topic": sh.cos(s, m["topic"][l]),
                             "lex_gain": sh.lex_gain(ll, cons_ids, inc_ids),
                             **{f"cos_{r2}": sh.cos(s, m["raw"][l][r2]) for r2 in REFS}})
            mean_ll = torch.stack(lls).mean(0)
            top, bot = mean_ll.topk(A.topn).indices.tolist(), (-mean_ll).topk(10).indices.tolist()
            lens[(l, r)] = {"top": [(tokstr(C, i), round(mean_ll[i].item(), 2)) for i in top],
                            "bottom": [(tokstr(C, i), round(mean_ll[i].item(), 2)) for i in bot],
                            "lex_gain": sh.lex_gain(mean_ll, cons_ids, inc_ids)}
    lex = {"consistent": [tokstr(C, i) for i in cons_ids], "inconsistent": [tokstr(C, i) for i in inc_ids]}
    return rows, lens, lex


# ------------------------------------------------------------------- generation


def conditions(A):
    cs = [SimpleNamespace(cid="nomem", ref=None, dose=0.0, ctx=False),
          SimpleNamespace(cid="ctx", ref=None, dose=0.0, ctx=True)]
    cs += [SimpleNamespace(cid=f"{r}@{d:g}", ref=r, dose=d, ctx=False) for r in A.refs for d in A.doses]
    return cs


def readers(C, M, c):
    if c.ref is None:
        return None
    return [tv1.PhaseReader(C, M[(c.ref, l)][0], C.KS[l][W256], M[(c.ref, l)][1], M[(c.ref, l)][2], C.catfn, l, 1.0)
            for l in C.layers]


def gen(C, prompt, B, rd, dose, seed):
    A = C.args
    smp = tv1.qwen_sampler(A.temp, A.top_k, A.top_p, A.presence)
    return tv1.generate(C, prompt, B, rd, a_think=0.0, a_answer=dose, think=False, max_new=A.answer_cap,
                        sampler=smp, seed=seed, greedy0=True, answer_cap=A.answer_cap)


def check_cached(C, it, M, ref, dose):
    """KV-cached PhaseReader generation == diag_keys.Reader on the full sequence (thinking off)."""
    prompt = chat_ids(C.tok, it["related"])
    spec = [(l, *M[(ref, l)]) for l in C.layers]
    prs = [tv1.PhaseReader(C, m, C.KS[l][W256], thr, sd, C.catfn, l, 1.0) for l, m, thr, sd in spec]
    g = tv1.generate(C, prompt, 1, prs, a_think=0.0, a_answer=dose, think=False, max_new=8,
                     sampler=tv1.plain_sampler(1.0), seed=0, greedy0=True, keep_logits=True)
    full = torch.cat([prompt, g.raw[:len(g.logits) - 1]])[None].to(C.dev)
    rds = [(l, dk.Reader(m, C.KS[l][W256], "hard", thr, sd, 0, C.stash), dose) for l, m, thr, sd in spec]
    with tv1.injecting(C.model, rds):
        lr = C.model(full).logits[0, len(prompt) - 1:].float()
    lb = C.model(full).logits[0, len(prompt) - 1:].float()
    res = {"item": it["id"], "ref": ref, "dose": dose, "layers": C.layers, "steps": len(g.logits),
           "max_abs_diff": (lr - g.logits).abs().max().item(), "memory_effect_max": (lr - lb).abs().max().item(),
           "gate": g.gate}
    log(f"cached-reader check: {json.dumps(res)}")
    assert res["max_abs_diff"] < C.args.check_tol, f"cached reader != diag_keys.Reader: {res}"
    return res


def gate_mean(g):
    return nanmean([x["gate"] for x in g.gate]) if g.gate else float("nan")


def run_item(C, it, M, rows, texts, nomem_unrel):
    A = C.args
    lex = it["lexicon"]
    prompts = [("related", it["related"])] + [("ambiguous", t) for t in it["ambiguous"][:A.n_ambiguous]]
    nomem_cache = C.nomem_cache
    for kind, text in prompts:
        seed = tv1.seed_of("ref_v1", text, A.seed)
        for c in conditions(A):
            if c.cid == "nomem" and text in nomem_cache:
                g = nomem_cache[text]
            else:
                prompt = ceiling_ids(C.tok, it["experience"], text) if c.ctx else chat_ids(C.tok, text)
                g = gen(C, prompt, 1 + A.samples, readers(C, M, c), c.dose, seed)
                C.n_gen += 1
                if c.cid == "nomem":
                    nomem_cache[text] = g
            for b, r in enumerate(g.rows):
                rr = sh.rep_rate(r["ans_ids"], A.loop_n)
                rows.append({"section": "gen", "item": it["id"], "tag": it["tag"], "cond": c.cid, "ref": c.ref,
                             "dose": c.dose, "kind": kind, "prompt": text, "row": b, "greedy": b == 0,
                             "gate": g.gate, "fire": gate_mean(g), "n_tok": len(r["ans_ids"]), "rep": rr,
                             "loop": rr >= A.loop_thr, "label": label(r["answer"], lex),
                             **{f"hits_{k[:4]}": v for k, v in lexicon_hits(r["answer"], lex).items()}})
                texts.append({"item": it["id"], "cond": c.cid, "kind": kind, "prompt": text, "row": b,
                              "answer": r["answer"]})
    for text in C.unrel_gen:
        if text not in nomem_unrel:
            nomem_unrel[text] = gen(C, chat_ids(C.tok, text), 1, None, 0.0, 0)
            C.n_gen += 1
        nm = nomem_unrel[text].rows[0]
        for c in conditions(A):
            if c.ctx:
                continue
            g = nomem_unrel[text] if c.ref is None else gen(C, chat_ids(C.tok, text), 1, readers(C, M, c), c.dose, 0)
            C.n_gen += c.ref is not None
            r = g.rows[0]
            rows.append({"section": "unrel", "item": it["id"], "tag": it["tag"], "cond": c.cid, "ref": c.ref,
                         "dose": c.dose, "kind": "unrelated", "prompt": text, "row": 0, "greedy": True, "gate": g.gate,
                         "fire": gate_mean(g), "n_tok": len(r["ans_ids"]), "same": r["ans_ids"] == nm["ans_ids"],
                         "mem_word": lexicon_hits(r["answer"], lex)["consistent"] > 0,
                         "nomem_word": lexicon_hits(nm["answer"], lex)["consistent"] > 0})
            texts.append({"item": it["id"], "cond": c.cid, "kind": "unrelated", "prompt": text, "row": 0,
                          "answer": r["answer"]})


# ---------------------------------------------------------------------- metrics


def summarize(C, rows):
    A = C.args
    by = defaultdict(list)
    for r in rows:
        if r["section"] in ("gen", "unrel"):
            by[(r["item"], r["cond"])].append(r)
    tag_of = {it["id"]: it["tag"] for it in C.items}
    summ = []
    for it in C.items:
        for c in conditions(A):
            rs = by[(it["id"], c.cid)]
            g = [r for r in rs if r["section"] == "gen"]
            u = [r for r in rs if r["section"] == "unrel"]
            s = sh.lean_stats([r["label"] for r in g], [r["loop"] for r in g])
            o = {"item": it["id"], "tag": tag_of[it["id"]], "cond": c.cid, "ref": c.ref or "-", "dose": c.dose, **s,
                 "len": nanmean([r["n_tok"] for r in g]), "fire": nanmean([r["fire"] for r in g if r["greedy"]]),
                 "unrel_n": len(u)}
            if c.ref is not None and u:
                o["unrel_same"] = nanmean([float(r["same"]) for r in u])
                o["contam"] = nanmean([float(r["mem_word"]) for r in u]) - nanmean([float(r["nomem_word"]) for r in u])
            else:
                o["unrel_same"] = o["contam"] = float("nan")
            summ.append(o)
    base = {(o["item"]): o for o in summ if o["cond"] == "nomem"}
    for o in summ:
        o["d_lean"] = o["lean"] - base[o["item"]]["lean"]
        o["d_lean_clean"] = o["lean_clean"] - base[o["item"]]["lean_clean"]
    agg = []
    for tag in TAGS + ("all",):
        its = [it["id"] for it in C.items if tag == "all" or it["tag"] == tag]
        if not its:
            continue
        for c in conditions(A):
            sub = [o for o in summ if o["item"] in its and o["cond"] == c.cid]
            a = {"item": f"<{tag}>", "tag": tag, "cond": c.cid, "ref": c.ref or "-", "dose": c.dose, "n_items": len(sub)}
            for k in ("cons", "inc", "lean", "d_lean", "loop", "lean_clean", "d_lean_clean", "len", "fire", "unrel_same",
                      "contam"):
                a[k] = nanmean([o[k] for o in sub])
            agg.append(a)
    return summ, agg


def shift_tables(C, srows):
    """(item, layer, ref) -> means over the moments of the shift rows."""
    acc = defaultdict(list)
    for r in srows:
        acc[(r["item"], r["layer"], r["ref"])].append(r)
    keys = ["raw_norm", "norm_ratio", "cos_key", "cos_topic", "lex_gain"] + [f"cos_{r}" for r in REFS]
    return {k: {m: nanmean([r[m] for r in v]) for m in keys} for k, v in acc.items()}


# ----------------------------------------------------------------------- report


EXPLAIN = """WHAT EACH NUMBER MEANS
- cons / inc: share of answers (greedy + samples, related + ambiguous prompts) that the bench lexicon labels
  consistent / inconsistent with the stored preference (more consistent-side words than inconsistent-side
  words = consistent; for the negated item the sides are swapped, so "consistent" = away from jazz).
- lean = cons - inc, from -1 to +1. d_lean = lean minus the same item's no-memory lean.
- loop: share of answers that repeat themselves (repeated-4-gram rate >= {thr}; e.g. "Norway Norway ...").
- lean_clean: lean over the NON-loop answers only. Loops often contain lexicon words, so they inflate lean;
  lean_clean is the honest number. d_lean_clean = minus the no-memory lean_clean.
- len: mean answer length in tokens (cap {cap}). fire: share of prompts where the memory's gate was open at
  the last prompt position (mean over the injected layers). Stored keys and thresholds are the same for every
  reference, but the read keys at L21/L23 see the injection at L20, so fire can differ a little.
- unrel_same: share of unrelated prompts whose greedy answer is token-for-token identical to no memory
  (1 = no leakage). contam: unrelated greedy answers containing a consistent lexicon word, minus no memory.
- norm ratio: |with - reference| / |with - without| BEFORE rescaling (every stored shift is then rescaled to
  the `without` norm of the same moment and layer, so only directions differ between references).
- cos(a, b): cosine between two references' shifts for the same moment and layer (mean over the 3 moments).
- cos_key: cosine between the shift pushed through the key's whitening (256 numbers) and the moment's
  stored key. High = the shift points along the very topic the memory is filed under (it re-injects the
  topic rather than the preference). cos_topic: the same in raw state space, against the mean
  (state - hum) over the follow-up's user words.
- lex_gain (logit lens): the shift alone is passed through the model's final norm and output layer; the
  mean logit of the consistent lexicon words' first subword minus that of the inconsistent words. Positive
  = read out directly, the shift favours the preference's own words. No generation needed.
"""


def report(C, summ, agg, ST, lens_all, checks):
    A = C.args
    L = []
    w = L.append
    f2, f3 = (lambda x: fmt(x, "+.2f")), (lambda x: fmt(x, ".2f"))
    w(f"Seahorse ref_v1: which reference should a preference shift subtract? {A.model}"
      f"{' TINY' if A.tiny else ''}{' RANDOM ' + A.random_model if A.random_model else ''}")
    w(f"layers {C.layers} (each its own keys, M, threshold); key pooled_w256, read hard (q{A.thr_q}); isolated "
      f"delta memory per item (3 moments); thinking OFF; alpha per layer {A.doses}; references {list(REFS)} "
      f"(generated: {A.refs}); every shift rescaled to the `without` norm of its moment and layer.")
    w(f"generation: related + {A.n_ambiguous} ambiguous prompts x (1 greedy + {A.samples} samples: T {A.temp:g}, "
      f"top-p {A.top_p:g}, top-k {A.top_k}, presence {A.presence:g}), answer cap {A.answer_cap}, the same random "
      f"numbers in every condition; {len(C.unrel_gen)} unrelated prompts greedy. disclosure: K={C.dcfg['k']} "
      f"(<= {C.dcfg['k_disp_max']} dispositions, rest facts, seed {C.dcfg['seed']}).")
    w("items: " + "; ".join(f"{it['id']} [{it['tag']}]" for it in C.items))
    w(f"checks: {json.dumps(checks)}")
    w("")
    w(EXPLAIN.format(thr=A.loop_thr, cap=A.answer_cap))
    hdr = ["cond", "cons", "inc", "lean", "d_lean", "loop", "lean_clean", "d_clean", "len", "fire", "unrel_same",
           "contam"]

    def rowfor(o):
        return [o["cond"], f3(o["cons"]), f3(o["inc"]), f2(o["lean"]), f2(o["d_lean"]), f3(o["loop"]),
                f2(o["lean_clean"]), f2(o["d_lean_clean"]), fmt(o["len"], ".0f"), f3(o["fire"]), f3(o["unrel_same"]),
                f2(o["contam"])]
    w("=" * 110)
    w("1. BY ITEM TAG (macro means over the items of the tag)")
    for tag in TAGS + ("all",):
        sub = [o for o in agg if o["tag"] == tag]
        if sub:
            w(f"[{tag}] items: {[it['id'] for it in C.items if tag == 'all' or it['tag'] == tag]}")
            w(table(hdr, [rowfor(o) for o in sub]))
            w("")
    conds = [c.cid for c in conditions(A)]
    S = {(o["item"], o["cond"]): o for o in summ}
    for title, key, f in (("2. lean_clean PER ITEM (rows) x CONDITION", "lean_clean", f2),
                          ("3. loop PER ITEM", "loop", f3), ("4. lean (loops included) PER ITEM", "lean", f2),
                          ("5. unrel_same PER ITEM (memory conditions)", "unrel_same", f3)):
        w("=" * 110)
        w(title)
        w(table(["item", "tag"] + conds, [[it["id"], it["tag"]] + [f(S[(it["id"], c)][key]) for c in conds]
                                          for it in C.items]))
        w("")
    w("=" * 110)
    w("6. SHIFT GEOMETRY (before rescaling; means over the 3 moments)")
    for l in C.layers:
        w(f"L{l}: norm ratio |with - ref| / |with - without|")
        w(table(["item", "tag"] + list(REFS), [[it["id"], it["tag"]] + [fmt(ST[(it["id"], l, r)]["norm_ratio"], ".2f")
                                                                       for r in REFS] for it in C.items]))
        w("")
    l0 = C.layers[0]
    pairs = [(a, b) for i, a in enumerate(REFS) for b in REFS[i + 1:]]
    w(f"L{l0}: cosine between references' shifts")
    w(table(["item"] + [f"{a[:4]}~{b[:4]}" for a, b in pairs],
            [[it["id"]] + [f2(ST[(it["id"], l0, a)][f"cos_{b}"]) for a, b in pairs] for it in C.items]
            + [[f"<{t}>"] + [f2(nanmean([ST[(it["id"], l0, a)][f"cos_{b}"] for it in C.items if it["tag"] == t]))
                             for a, b in pairs] for t in TAGS if any(it["tag"] == t for it in C.items)]))
    w("")
    for l in C.layers:
        w(f"L{l}: cos_key (whitened shift vs the stored key) | cos_topic (raw shift vs the follow-up's topic)")
        w(table(["item"] + [f"key {r}" for r in REFS] + [f"topic {r}" for r in REFS],
                [[it["id"]] + [f2(ST[(it["id"], l, r)]["cos_key"]) for r in REFS]
                 + [f2(ST[(it["id"], l, r)]["cos_topic"]) for r in REFS] for it in C.items]
                + [[f"<{t}>"] + [f2(nanmean([ST[(it["id"], l, r)][m] for it in C.items if it["tag"] == t]))
                                 for m in ("cos_key", "cos_topic") for r in REFS]
                   for t in TAGS if any(it["tag"] == t for it in C.items)]))
        w("")
    w("=" * 110)
    w("7. LOGIT LENS lex_gain (consistent - inconsistent lexicon logits of the shift read out directly; mean over "
      "moments; top tokens in vocab.txt)")
    for l in C.layers:
        w(f"L{l}")
        w(table(["item", "tag"] + list(REFS),
                [[it["id"], it["tag"]] + [f2(ST[(it["id"], l, r)]["lex_gain"]) for r in REFS] for it in C.items]
                + [[f"<{t}>", ""] + [f2(nanmean([ST[(it["id"], l, r)]["lex_gain"] for it in C.items if it["tag"] == t]))
                                     for r in REFS] for t in TAGS if any(it["tag"] == t for it in C.items)]))
        w("")
    return "\n".join(L) + "\n"


def vocab_txt(C, lens_all, lexs, path):
    A = C.args
    L = []
    w = L.append
    w(f"# Seahorse ref_v1 logit lens: each stored shift (before rescaling; the final norm makes the scale irrelevant) "
      f"-> final norm -> lm_head; mean over the item's 3 moments. top {A.topn} boosted and 10 most suppressed "
      f"tokens (▁ = leading space). lex_gain = mean logit of the consistent lexicon tokens minus the "
      f"inconsistent ones (first subword of ' ' + word; tokens on both sides dropped).")
    for l in C.layers:
        w("")
        w("#" * 110)
        w(f"# LAYER {l}")
        for it in C.items:
            w("")
            w(f"## {it['id']} [{it['tag']}]: {it['experience']} (counter: {it['counter']})")
            if l == C.layers[0]:
                w(f"   lexicon tokens + {' '.join(lexs[it['id']]['consistent'])}")
                w(f"   lexicon tokens - {' '.join(lexs[it['id']]['inconsistent'])}")
            for r in REFS:
                z = lens_all[it["id"]][(l, r)]
                w(f"  {r:<10} lex_gain {z['lex_gain']:+.2f} | top: " + " ".join(f"{t}({v:+.1f})" for t, v in z["top"]))
                w(f"  {'':<10} suppressed: " + " ".join(f"{t}({v:+.1f})" for t, v in z["bottom"]))
    Path(path).write_text("\n".join(L) + "\n")


def samples_txt(C, rows, texts, path):
    A = C.args
    T = {(t["item"], t["cond"], t["prompt"], t["row"]): t["answer"] for t in texts}
    L = []
    w = L.append
    w(f"# Seahorse ref_v1 samples: per item x condition, the related prompt (greedy + sample 1) and the first "
      f"ambiguous prompt (greedy). [g=gate m=match/thr] at the last prompt position, first injected layer "
      f"(L{C.layers[0]}); LOOP = repeated-4-gram rate >= {A.loop_thr}.")
    G = {(r["item"], r["cond"], r["prompt"], r["row"]): r for r in rows if r["section"] == "gen"}
    for it in C.items:
        w("")
        w("=" * 110)
        w(f"## {it['id']} [{it['tag']}]: {it['experience']}  (counter: {it['counter']})")
        w(f"   disclosure ({len(it['disclosure'])}): " + " | ".join(e["experience"] for e in it["disclosure"]))
        w(f"   centroid: " + " | ".join(it["centroid"]))
        shows = [(it["related"], 0), (it["related"], 1)] + [(it["ambiguous"][0], 0)]
        for c in conditions(A):
            w(f"--- [{c.cid}] ---")
            for text, row in shows:
                r = G.get((it["id"], c.cid, text, row))
                if r is None:
                    continue
                gs = ""
                if r["gate"]:
                    g0 = r["gate"][0]
                    gs = f" [g={g0['gate']:.0f} m={g0['match']:.3f}/{g0['thr']:.3f}]"
                lp = " LOOP" if r["loop"] else ""
                w(f"[{'greedy' if row == 0 else 'sample ' + str(row)}{gs} {r['label']}{lp}] {text}")
                w(f"  {s2.one_line(T[(it['id'], c.cid, text, row)])}")
        un = defaultdict(list)
        for r in rows:
            if r["section"] == "unrel" and r["item"] == it["id"] and r["ref"] is not None:
                un[r["cond"]].append("same" if r["same"] else "DIFF")
        w("unrelated greedy vs nomem: " + "; ".join(f"{k}: {' '.join(v)}" for k, v in un.items()))
    Path(path).write_text("\n".join(L) + "\n")


# -------------------------------------------------------------------------- main


def main():
    A = parse_args()
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with torch.inference_mode():
        C = setup(A, out)
        C.thr_log, C.nomem_cache, C.n_gen = {}, {}, 0
        log(f"setup done in {(time.time() - t0) / 60:.1f} min; items {[it['id'] for it in C.items]}")
        n_prompts = sum(1 + min(A.n_ambiguous, len(it["ambiguous"])) for it in C.items)
        plan = n_prompts * len(conditions(A)) + len(C.items) * len(C.unrel_gen) * len(A.refs) * len(A.doses)
        log(f"planned generations: ~{plan} ({n_prompts} prompts x {len(conditions(A))} conditions + unrelated)")
        rows, texts, srows, lens_all, lexs, nomem_unrel, check = [], [], [], {}, {}, {}, None
        part = open(out / "results.partial.jsonl", "w")
        t1 = time.time()
        for i, it in enumerate(C.items):
            moments = build_moments(C, it, check=(i == 0))
            M = memories(C, it, moments)
            sr, lens, lex = geometry(C, it, moments)
            srows += sr
            lens_all[it["id"]], lexs[it["id"]] = lens, lex
            it["moments"] = [{k: m[k] for k in ("j", "followup", "n", "n_without", "n_content")} for m in moments]
            if check is None:
                check = check_cached(C, it, M, "opposite", max(A.doses))
            n0 = len(rows)
            run_item(C, it, M, rows, texts, nomem_unrel)
            for r in sr + rows[n0:]:
                part.write(json.dumps(r) + "\n")
            part.flush()
            el = time.time() - t1
            log(f"{it['id']} done | {C.n_gen} generations, {el / 60:.1f} min, eta {el / (i + 1) * (len(C.items) - i - 1) / 60:.1f} min")
        part.close()
    summ, agg = summarize(C, rows)
    ST = shift_tables(C, srows)
    checks = {**dict(C.chk), "reader": {k: check[k] for k in ("max_abs_diff", "memory_effect_max", "steps")}}
    meta = {"args": vars(A), "items": [{k: v for k, v in it.items()} for it in C.items], "template": C.template,
            "disclosure": C.dcfg, "white_info": C.white_info, "mu_skip_tokens": C.skip, "tail_len": C.tail_len,
            "thresholds": C.thr_log, "checks": checks, "reader_check": check, "n_generations": C.n_gen,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "linear_attention_kernels": tv1._kernels()}
    tv1.write_jsonl(out / "results.jsonl", srows + rows)
    dk.write_csv(out / "summary.csv", summ + agg)
    with gzip.open(out / "texts.jsonl.gz", "wt") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    (out / "report.txt").write_text(report(C, summ, agg, ST, lens_all, checks))
    vocab_txt(C, lens_all, lexs, out / "vocab.txt")
    samples_txt(C, rows, texts, out / "samples.txt")
    (out / "results.partial.jsonl").unlink()
    meta["minutes"] = (time.time() - t0) / 60
    json.dump(meta, open(out / "config.json", "w"), indent=2, default=str)
    log(f"ref_v1 done in {meta['minutes']:.1f} min ({C.n_gen} generations) -> {out}")


if __name__ == "__main__":
    main()
