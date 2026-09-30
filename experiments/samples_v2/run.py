#!/usr/bin/env python
"""Seahorse samples_v2: real text outputs of the current best memory design, for the report.

Current design (diag_keys): write unit pooled (one entropy-weighted write per follow-up, template
tail excluded); key pooled_w256 (running mean over user-text positions of PCA-256-whitened (h - mu),
normalised; the write key is the mean over the follow-up's user text in the WITHOUT run); read hard
(template head masked; recall only where max-cos(read key, stored write keys) > the q0.95 threshold
calibrated on generic prompts + greedy continuations, per memory) or soft_t0.5; alpha 2, layer 26;
delta rule for isolated memories, RLS (lambda 0.1) for the combined 6-scenario memory.

Conditions (per scenario; baselines: facts without, dispositions contrastive)
  nomem    no memory
  ctx      in context: experience + prompt in the same user turn (the ceiling)
  old      old design: centred key, read everywhere, same pooled unit (tail excluded), delta, isolated
  hard     current design, isolated: pooled_w256 + hard
  soft     current design, isolated: pooled_w256 + soft_t0.5
  comb     current design, combined: all 6 scenarios in ONE memory, RLS, pooled_w256 + hard; each
           scenario written with its own type's baseline
  hard_wo  current design, isolated, hard, baseline WITHOUT (vegetarian, peanut_allergy: does concept
           priming, e.g. peanut butter, survive the new key?)

Outputs per scenario
  1. greedy generations (--greedy-len tokens, repetition_penalty 1.0): the 3 probes, the relation
     probes and 4 shared unrelated probes; each flagged identical / not to no memory
  2. prefix completions (facts): the assistant answer is forced to start with measure.prefix; greedy
     --prefix-len tokens; --prefix-samples samples at T=--prefix-temp: counts of samples naming the
     target (first target token; full target, lenient text match) and each foil; P(target's first
     token) and log P(full target)
  3. disposition rates: 4 ambiguous prompts x --disp-samples samples at T=--disp-temp, keyword counts
  4. leakage: are the unrelated probes' greedy texts identical to no memory?

Generation is an own KV-cached loop (no HF generate defaults: no top-k / top-p, no repetition
penalty). Sampling: inverse CDF with one uniform per row and step from a generator seeded by the
prompt only, so every condition sees the same random numbers (common random numbers). The pooled
read key does not change after the user text, so the read at generated positions reuses the last
prompt position's; per-token keys (old) are computed per position. Both are checked against
diag_keys.Reader on full sequences at start-up.

Outputs: samples.txt, samples.jsonl, counts.csv, config.json
"""

import argparse
import importlib.util
import json
import re
import time
import zlib
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers import DynamicCache

from seahorse.residual import inject
from seahorse.sessions import ceiling_ids, chat_ids, text_ids

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dk = _load("seahorse_diag_keys_run", HERE.parent / "diag_keys" / "run.py")

# Ambiguous prompts + keyword rules for the disposition rates (regexes, case-insensitive).
DISP = {
    "vegetarian": {
        "prompts": ["Suggest something for dinner tonight.",
                    "What should I cook for a family barbecue?",
                    "Give me a quick lunch idea.",
                    "What's a good high-protein meal?"],
        "kw": {"meat": r"\b(chicken|beef|pork|steaks?|bacon|ham|turkey|lamb|sausages?|meatballs?|meatloaf|ribs|brisket|"
                       r"pepperoni|salami|prosciutto|fish|salmon|tuna|cod|shrimps?|prawns?|anchov\w*|duck|veal|venison)\b",
               "veg": r"\b(vegetarian|vegan|veggie|tofu|tempeh|seitan|lentils?|chickpeas?|beans?|plant-based|meatless|"
                      r"meat-free|paneer|halloumi|falafel|mushrooms?|quinoa)\b",
               "explicit": r"\b(vegetarian|vegan|plant-based|meatless|meat-free)\b"},
        "rule": "consistent = no meat/fish word; inconsistent = any meat/fish word",
        "consistent": lambda h: not h["meat"],
        "inconsistent": lambda h: h["meat"],
    },
    "peanut_allergy": {
        "prompts": ["Suggest a quick snack I can take to work.",
                    "What's a good sauce for stir-fried noodles?",
                    "What should I put in a protein smoothie?",
                    "What cookies should I bake for a bake sale?"],
        "kw": {"peanut": r"\bpeanuts?\b|\bsatay\b|\bPB ?& ?J\b",
               "allergy": r"allerg\w*|\bavoid\w*|\bnut-free\b|\bpeanut-free\b"},
        "rule": "inconsistent = mentions peanut/satay without allergy/avoid; consistent = the rest",
        "consistent": lambda h: not (h["peanut"] and not h["allergy"]),
        "inconsistent": lambda h: h["peanut"] and not h["allergy"],
    },
    "norway": {
        "prompts": ["How much does a good pair of running shoes cost?",
                    "What's a reasonable monthly budget for groceries?",
                    "How much should I expect to pay for a new laptop?",
                    "How much does a cup of coffee usually cost at a cafe?"],
        "kw": {"nok": r"\bkroner\b|\bkrone\b|\bNOK\b|\bnorw\w*|\boslo\b|\bkr\b",
               "usd": r"\$|\bdollars?\b|\bUSD\b"},
        "rule": "consistent = kroner/NOK/Norway and no dollars; inconsistent = dollars/USD/$ and no kroner/Norway",
        "consistent": lambda h: h["nok"] and not h["usd"],
        "inconsistent": lambda h: h["usd"] and not h["nok"],
    },
}

SHORT = {"nomem": "no memory", "ctx": "in context", "old": "old: centred+everywhere, iso",
         "hard": "current: w256+hard, iso", "soft": "current: w256+soft{t}, iso",
         "comb": "current: w256+hard, combined-6 RLS", "hard_wo": "current: w256+hard, iso, base=without"}
LONG = {"nomem": "no memory",
        "ctx": "in context: the experience in the same user turn (ceiling)",
        "old": "old design: key centred, read everywhere, pooled unit (tail excluded), delta, isolated, baseline {b}",
        "hard": "current design: key pooled_w256, read hard (q{q}), pooled unit, delta, isolated, baseline {b}",
        "soft": "current design: key pooled_w256, read soft_t{t} (q{q}), pooled unit, delta, isolated, baseline {b}",
        "comb": "current design, combined: all scenarios in ONE memory, RLS lambda {lam}, key pooled_w256, read hard "
                "(q{q}); facts written with baseline without, dispositions contrastive",
        "hard_wo": "current design: key pooled_w256, read hard (q{q}), delta, isolated, baseline WITHOUT "
                   "(concept-priming check)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--layer", type=int, default=26)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--thr-q", type=float, default=0.95)
    p.add_argument("--soft-t", type=float, default=0.5)
    p.add_argument("--rls-lambda", type=float, default=0.1)
    p.add_argument("--white-eps", type=float, default=0.01)
    p.add_argument("--greedy-len", type=int, default=60)
    p.add_argument("--prefix-len", type=int, default=12)
    p.add_argument("--prefix-samples", type=int, default=100)
    p.add_argument("--prefix-temp", type=float, default=1.0)
    p.add_argument("--disp-len", type=int, default=60)
    p.add_argument("--disp-samples", type=int, default=20)
    p.add_argument("--disp-temp", type=float, default=0.7)
    p.add_argument("--n-show", type=int, default=2, help="full sampled texts per prompt x condition in samples.txt")
    p.add_argument("--unrelated-idx", type=int, nargs="+", default=[0, 2, 3, 5],
                   help="indices into scenarios.yaml unrelated_probes")
    p.add_argument("--priming", nargs="+", default=["vegetarian", "peanut_allergy"],
                   help="dispositions that also get the current design with baseline without")
    p.add_argument("--scenarios", default=str(HERE.parent / "v0_1" / "scenarios.yaml"))
    p.add_argument("--generic", default=str(dk.v01.GENERIC))
    p.add_argument("--cont-len", type=int, default=20, help="diag_keys setup: probe continuations")
    p.add_argument("--gen-cont-len", type=int, default=20, help="greedy continuation of generic prompts (calibration)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-tol", type=float, default=0.05, help="max |logit diff| cached reader vs diag_keys.Reader")
    p.add_argument("--tiny", action="store_true", help="smoke test: tiny random Qwen2 with --model's tokenizer")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


log = dk.log


def seed_of(tag, text, base):
    return (base * 1000003 + zlib.crc32(f"{tag}|{text}".encode())) % (2 ** 31 - 1)


# ----------------------------------------------------------------------- reader


class GenReader:
    """Read hook for KV-cached generation over a batch of identical prompts; begin(prompt) first.

    Prefill (the whole prompt): diag_keys.Reader's read, categories from the prompt ids. Later calls
    see only generated (continuation) positions: pooled keys -> the last prompt position's key (the
    user-text mean no longer changes), so its gated recall is reused; per-token keys -> per position.
    """

    def __init__(self, mem, ks, read, thr, sd, catfn):
        self.mem, self.ks, self.read_name, self.thr, self.sd, self.catfn = mem, ks, read, thr, sd, catfn
        self.cats, self.prefilled, self.v, self.last = None, False, None, None

    def begin(self, ids):
        self.cats, self.prefilled, self.v, self.last = self.catfn(ids), False, None, None

    def _recall(self, k, cats):
        match = (k @ self.mem.Kst.T).max(-1).values
        return k @ self.mem.M.T, match, dk.gate_fn(self.read_name, match, cats, self.thr, self.sd)

    def read(self, h, alpha):
        B, T, d = h.shape
        if not self.prefilled:
            assert T == len(self.cats), "prefill: call begin(prompt) first"
            r, match, g = self._recall(self.ks.keys(h[0], self.cats == 1), self.cats)
            self.prefilled = True
            self.v = g[-1] * r[-1]
            self.last = {"match": match[-1].item(), "gate": g[-1].item()}
            return h + alpha * (g[:, None] * r)[None]  # rows are identical prompts
        if self.ks.pooled:
            return h + alpha * self.v
        cats = torch.full((B * T,), 3, dtype=torch.long, device=h.device)
        r, _, g = self._recall(self.ks.keys(h.reshape(-1, d)), cats)
        return h + alpha * (g[:, None] * r).reshape(B, T, d)


def generate(C, prompt, n_new, n, temp, reader, seed, keep_logits=False):
    """n continuations of `prompt` (temp 0 = greedy). Rows are cut at the first EOS."""
    model, dev = C.model, C.dev
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    x = prompt[None].to(dev).expand(n, -1).contiguous()
    past = DynamicCache()
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    steps, logs, first = [], [], None
    if reader is not None:
        reader.begin(prompt)
    with inject(model, C.layer, reader, C.args.alpha) if reader is not None else nullcontext():
        for t in range(n_new):
            hs = model.model(input_ids=x, past_key_values=past, use_cache=True).last_hidden_state[:, -1]
            logits = model.lm_head(hs).float()
            if t == 0:
                first = torch.log_softmax(logits[0], -1)
            if keep_logits:
                logs.append(logits[0])
            if temp == 0:
                nxt = logits.argmax(-1)
            else:
                cdf = torch.softmax(logits.double() / temp, -1).cumsum(-1)
                u = torch.rand(n, 1, generator=gen, device=dev, dtype=torch.float64) * cdf[:, -1:]
                nxt = torch.searchsorted(cdf, u).squeeze(1).clamp_max(cdf.shape[1] - 1)
            nxt = torch.where(done, C.eos[0], nxt)
            steps.append(nxt)
            done = done | torch.isin(nxt, C.eos)
            if bool(done.all()):
                break
            x = nxt[:, None]
    toks = torch.stack(steps, 1).cpu()
    eos = set(C.eos.tolist())
    rows = []
    for r in toks.tolist():
        rows.append(r[:next((i for i, t in enumerate(r) if t in eos), len(r))])
    return SimpleNamespace(rows=rows, texts=[C.tok.decode(r, skip_special_tokens=True) for r in rows],
                           first=first, raw=toks[0], logits=torch.stack(logs) if keep_logits else None,
                           gate=dict(reader.last) if reader is not None else None)


def score(C, prompt, cont, reader):
    """log P(cont | prompt) under the condition (teacher-forced)."""
    ids = torch.cat([prompt, cont])
    if reader is not None:
        reader.begin(ids)
    with inject(C.model, C.layer, reader, C.args.alpha) if reader is not None else nullcontext():
        hs = C.model.model(input_ids=ids[None].to(C.dev), use_cache=False).last_hidden_state[0, len(prompt) - 1:-1]
    lp = torch.log_softmax(C.model.lm_head(hs).float(), -1)
    return lp.gather(1, cont.to(C.dev)[:, None]).sum().item()


# ------------------------------------------------------------------- conditions


def conditions(C):
    args, L = C.args, C.layer
    KS = C.KS[L]
    soft = f"soft_t{args.soft_t:g}"
    Kg = {k: dk.gen_keys(C, L, KS[k]) for k in ("centred", "pooled_w256")}
    st = {(k, b): dk.make_stored(C, L, b, "pooled", KS[k], False)
          for k in ("centred", "pooled_w256") for b in ("without", "contrastive")}
    chk = defaultdict(float)
    mems, built = {}, {}

    def memory(name, key, members, rule, stored, base):
        if name not in built:
            mem = dk.build(stored, members, rule, args.rls_lambda, args.device, chk)
            thr, sd = dk.calib(mem, Kg[key], args.thr_q)
            built[name] = (mem, thr, sd)
            mems[name] = {"key": key, "members": list(members), "rule": rule, "baseline": base, "thr": thr,
                          "sd_generic_match": sd, "n_writes": int(mem.Kst.shape[0])}
        return built[name]

    def reader(name, key, read, *a):
        mem, thr, sd = memory(name, key, *a)
        return GenReader(mem, KS[key], read, thr, sd, C.catfn)

    fmt = {"q": f"{args.thr_q:g}", "t": f"{args.soft_t:g}", "lam": f"{args.rls_lambda:g}"}
    mix = {s: st[("pooled_w256", C.base_of[s])][s] for s in C.ids}
    comb = reader("comb", "pooled_w256", "hard", C.ids, "rls", mix, "per type")
    conds = {}
    for s in C.ids:
        b = C.base_of[s]
        spec = [("nomem", None, "nomem"), ("ctx", None, f"ctx/{s}"),
                ("old", reader(f"centred/{s}/{b}", "centred", "everywhere", [s], "delta", st[("centred", b)], b), f"old/{s}"),
                ("hard", reader(f"w256/{s}/{b}", "pooled_w256", "hard", [s], "delta", st[("pooled_w256", b)], b), f"hard/{s}"),
                ("soft", reader(f"w256/{s}/{b}", "pooled_w256", soft, [s], "delta", st[("pooled_w256", b)], b), f"soft/{s}"),
                ("comb", comb, "comb")]
        if s in args.priming and b != "without":
            spec.append(("hard_wo", reader(f"w256/{s}/without", "pooled_w256", "hard", [s], "delta",
                                           st[("pooled_w256", "without")], "without"), f"hard_wo/{s}"))
        conds[s] = [SimpleNamespace(cid=c, reader=r, cache=ck, short=SHORT[c].format(**fmt),
                                    label=LONG[c].format(b=b, **fmt)) for c, r, ck in spec]
    return conds, mems, dict(chk)


def prompt_for(C, cond, s, text):
    return ceiling_ids(C.tok, C.by_id[s]["experience"], text) if cond.cid == "ctx" else chat_ids(C.tok, text)


def check_readers(C, conds):
    """GenReader (KV cache, reused last-prompt read) must match diag_keys.Reader on the full sequence."""
    s = next(x for x in C.ids if C.by_id[x]["type"] == "fact")
    scen = C.by_id[s]
    rel = next(p["text"] for p in scen["probes"] if p["distance"] == "related")
    prompts = {"probe": chat_ids(C.tok, rel),
               "prefix": torch.cat([chat_ids(C.tok, rel), text_ids(C.tok, scen["measure"]["prefix"])])}
    out = []
    for cond in conds[s]:
        R = cond.reader
        if R is None:
            continue
        for pname, prompt in prompts.items():
            g = generate(C, prompt, 8, 1, 0.0, R, 0, keep_logits=True)
            full = torch.cat([prompt, g.raw[:-1]])[None].to(C.dev)
            with inject(C.model, C.layer, dk.Reader(R.mem, R.ks, R.read_name, R.thr, R.sd, 0, C.stash), C.args.alpha):
                lr = C.model(full).logits[0, len(prompt) - 1:].float()
            lb = C.model(full).logits[0, len(prompt) - 1:].float()
            diff = (lr - g.logits).abs().max().item()
            out.append({"scenario": s, "condition": cond.cid, "prompt": pname, "steps": len(g.raw),
                        "max_abs_diff": diff, "memory_effect_max": (lr - lb).abs().max().item(), **(g.gate or {})})
            log(f"check {cond.cid}/{pname}: max|diff| {diff:.2e}, memory effect {out[-1]['memory_effect_max']:.3f}, "
                f"gate {g.gate}")
            assert diff < C.args.check_tol, f"cached reader != diag_keys.Reader: {out[-1]}"
    return out


# ----------------------------------------------------------------------- passes


def run_greedy(C, conds):
    recs, cache, t0 = [], {}, time.time()
    for s in C.ids:
        scen = C.by_id[s]
        items = [(p["distance"], p["text"]) for p in scen["probes"]]
        items += [("relation", rp["text"]) for rp in scen.get("relation_probes", [])]
        items += [("unrelated", u) for u in C.unrel_texts]
        for kind, text in items:
            grp = []
            for cond in conds[s]:
                key = (cond.cache, text)
                if key not in cache:
                    g = generate(C, prompt_for(C, cond, s, text), C.args.greedy_len, 1, 0.0, cond.reader,
                                 seed_of("greedy", text, C.args.seed))
                    cache[key] = {"tokens": g.rows[0], "text": g.texts[0], "gate": g.gate}
                grp.append({"section": "greedy", "scenario": s, "kind": kind, "prompt": text, "condition": cond.cid,
                            "label": cond.label, **cache[key]})
            nm = next(r for r in grp if r["condition"] == "nomem")
            for r in grp:
                r["identical_to_nomem"] = r["tokens"] == nm["tokens"]
            recs += grp
        log(f"greedy: {s} done ({(time.time() - t0) / 60:.1f} min)")
    return recs


def lenient_starts(text, target):
    norm = lambda x: re.sub(r"^[\s*\"'_`]+", "", x).lower()
    return norm(text).startswith(norm(target))


def run_prefix(C, conds):
    args, recs, t0 = C.args, [], time.time()
    for s in [x for x in C.ids if C.by_id[x]["type"] == "fact"]:
        scen = C.by_id[s]
        m = scen["measure"]
        pre, tgt = text_ids(C.tok, m["prefix"]), text_ids(C.tok, m["target"])
        names = [m["target"]] + list(m["foils"])
        firsts = [int(text_ids(C.tok, x)[0]) for x in names]
        for p in scen["probes"]:
            seed = seed_of("prefix", p["text"], args.seed)
            for cond in conds[s]:
                prompt = torch.cat([prompt_for(C, cond, s, p["text"]), pre])
                g = generate(C, prompt, args.prefix_len, 1, 0.0, cond.reader, seed)
                smp = generate(C, prompt, args.prefix_len, args.prefix_samples, args.prefix_temp, cond.reader, seed)
                counts = {}
                for name, ft in zip(names, firsts):
                    counts[name] = {"first_tok": sum(1 for r in smp.rows if r and r[0] == ft),
                                    "full": sum(lenient_starts(t, name) for t in smp.texts),
                                    "p_first": g.first[ft].exp().item(),
                                    "first_tok_str": C.tok.decode([ft])}
                recs.append({"section": "prefix", "scenario": s, "kind": p["distance"], "prompt": p["text"],
                             "prefix": m["prefix"], "condition": cond.cid, "label": cond.label, "gate": g.gate,
                             "greedy": g.texts[0], "n": len(smp.rows), "temp": args.prefix_temp,
                             "target": m["target"], "foils": list(m["foils"]), "counts": counts,
                             "logp_target": score(C, prompt, tgt, cond.reader), "samples": smp.texts})
        log(f"prefix: {s} done ({(time.time() - t0) / 60:.1f} min)")
    return recs


def classify(s, text):
    d = DISP[s]
    h = {k: bool(re.search(rx, text, re.I)) for k, rx in d["kw"].items()}
    return {"hits": h, "consistent": bool(d["consistent"](h)), "inconsistent": bool(d["inconsistent"](h))}


def run_disp(C, conds):
    args, recs, t0 = C.args, [], time.time()
    for s in [x for x in C.ids if x in DISP]:
        for text in DISP[s]["prompts"]:
            seed = seed_of("disp", text, args.seed)
            for cond in conds[s]:
                g = generate(C, prompt_for(C, cond, s, text), args.disp_len, args.disp_samples, args.disp_temp,
                             cond.reader, seed)
                cls = [classify(s, t) for t in g.texts]
                recs.append({"section": "disposition", "scenario": s, "prompt": text, "condition": cond.cid,
                             "label": cond.label, "gate": g.gate, "n": len(cls), "temp": args.disp_temp,
                             "consistent": sum(c["consistent"] for c in cls),
                             "inconsistent": sum(c["inconsistent"] for c in cls),
                             "kw": {k: sum(c["hits"][k] for c in cls) for k in DISP[s]["kw"]},
                             "samples": g.texts, "flags": cls})
        log(f"disposition: {s} done ({(time.time() - t0) / 60:.1f} min)")
    return recs


# ---------------------------------------------------------------------- outputs


def gate_cols(g):
    return {"gate": g["gate"], "match": g["match"]} if g else {}


def count_rows(C, prefix, disp, greedy):
    rows = []
    for r in prefix:
        row = {"section": "prefix", "scenario": r["scenario"], "kind": r["kind"], "prompt": r["prompt"],
               "condition": r["condition"], "n": r["n"], "temp": r["temp"], "target": r["target"]}
        c = r["counts"][r["target"]]
        row.update(target_full=c["full"], target_first_tok=c["first_tok"], p_target_first=c["p_first"],
                   logp_target=r["logp_target"])
        for i, f in enumerate(r["foils"], 1):
            row.update({f"foil{i}": f, f"foil{i}_full": r["counts"][f]["full"],
                        f"foil{i}_first_tok": r["counts"][f]["first_tok"], f"p_foil{i}_first": r["counts"][f]["p_first"]})
        rows.append({**row, **gate_cols(r["gate"])})
    for r in disp:
        rows.append({"section": "disposition", "scenario": r["scenario"], "kind": "ambiguous", "prompt": r["prompt"],
                     "condition": r["condition"], "n": r["n"], "temp": r["temp"], "consistent": r["consistent"],
                     "inconsistent": r["inconsistent"], **{f"kw_{k}": v for k, v in r["kw"].items()},
                     **gate_cols(r["gate"])})
    for r in greedy:
        if r["condition"] != "nomem":
            rows.append({"section": "greedy", "scenario": r["scenario"], "kind": r["kind"], "prompt": r["prompt"],
                         "condition": r["condition"], "identical_to_nomem": int(r["identical_to_nomem"]),
                         **gate_cols(r["gate"])})
    return rows


def one_line(t):
    return " ".join(t.split())


def write_txt(path, C, conds, mems, greedy, prefix, disp):
    args = C.args
    W = 40
    L = []
    w = L.append
    w(f"# Seahorse samples_v2: model {args.model}{' (TINY RANDOM)' if args.tiny else ''}; layer {args.layer}, "
      f"alpha {args.alpha:g}; entropy-gated pooled writes (template tail excluded)")
    w("# conditions (baselines: facts without, dispositions contrastive):")
    for cid in ("nomem", "ctx", "old", "hard", "soft", "comb", "hard_wo"):
        c = next((c for cs in conds.values() for c in cs if c.cid == cid), None)
        if c:
            w(f"#   [{c.short}] {LONG[cid].format(b='<type>', q=f'{args.thr_q:g}', t=f'{args.soft_t:g}', lam=f'{args.rls_lambda:g}')}")
    w(f"# greedy: {args.greedy_len} new tokens, repetition_penalty 1.0. Sampling: plain softmax at the stated T (no "
      f"top-k/top-p), same random numbers for every condition of a prompt.")
    w("# (g=gate m=match/thr): the read gate at the last prompt position (= at every generated token); "
      "(=nomem): greedy text identical to no memory.")
    for s in C.ids:
        scen = C.by_id[s]
        cs = conds[s]
        by = defaultdict(dict)
        for r in greedy:
            if r["scenario"] == s:
                by[(r["kind"], r["prompt"])][r["condition"]] = r
        w("")
        w("=" * 110)
        w(f"## {s} ({scen['type']}; memory baseline {C.base_of[s]})")
        w(f"experience: {scen['experience']}")
        info = [f"{n}: thr {m['thr']:.3f}" for n, m in mems.items() if s in m["members"]]
        w("memories: " + "; ".join(info))
        w("")
        w(f"--- 1. greedy ({args.greedy_len} tokens) ---")
        for (kind, text), rs in by.items():
            w(f"probe [{kind}]: {text}")
            for c in cs:
                r = rs[c.cid]
                gs = f"(g={r['gate']['gate']:.2f} m={r['gate']['match']:.3f}/{c.reader.thr:.3f}) " \
                    if r["gate"] and c.reader.read_name != "everywhere" else ""
                same = "(=nomem) " if c.cid != "nomem" and r["identical_to_nomem"] else ""
                w(f"[{c.short}]".ljust(W) + f" {gs}{same}{one_line(r['text'])}")
            w("")
        pr = [r for r in prefix if r["scenario"] == s]
        if pr:
            r0 = pr[0]
            ft = r0["counts"][r0["target"]]["first_tok_str"]
            w(f"--- 2. prefix completions: answer forced to start with \"{r0['prefix']}\"; greedy {args.prefix_len} "
              f"tokens; {r0['n']} samples at T={r0['temp']:g}, {args.prefix_len} tokens ---")
            w(f"counts = samples whose continuation starts with the name (case, leading space and markup ignored); "
              f"first = samples whose first token is the name's first token ({ft!r} for the target). "
              f"P = model probability of the target's first token; logP = log P(full target).")
            for p in dict.fromkeys((r["kind"], r["prompt"]) for r in pr):
                w(f"probe [{p[0]}]: {p[1]}")
                for c in cs:
                    r = next(x for x in pr if (x["kind"], x["prompt"]) == p and x["condition"] == c.cid)
                    ct = r["counts"]
                    tg = ct[r["target"]]
                    foils = ", ".join(f"{f.strip()} {ct[f]['full']}" for f in r["foils"])
                    gs = f"(g={r['gate']['gate']:.2f}) " if r["gate"] and c.reader.read_name != "everywhere" else ""
                    w(f"[{c.short}]".ljust(W) + f" {gs}greedy: \"{r0['prefix']}{one_line(r['greedy'])}\"")
                    w(" " * (W + 1) + f"samples: {r['target'].strip()} {tg['full']}/{r['n']} (first {tg['first_tok']}), "
                      f"foils {foils} | P({ft!r})={tg['p_first']:.4f} logP(target)={r['logp_target']:.2f}")
                w("")
        dr = [r for r in disp if r["scenario"] == s]
        if dr:
            w(f"--- 3. disposition rates: {dr[0]['n']} samples at T={dr[0]['temp']:g}, {args.disp_len} tokens, per "
              f"prompt x condition ---")
            w(f"rule: {DISP[s]['rule']}. keywords: " + "; ".join(f"{k}=/{v}/" for k, v in DISP[s]["kw"].items()))
            tot = defaultdict(lambda: [0, 0, 0])
            for text in dict.fromkeys(r["prompt"] for r in dr):
                w(f"prompt: {text}")
                for c in cs:
                    r = next(x for x in dr if x["prompt"] == text and x["condition"] == c.cid)
                    kw = ", ".join(f"{k} {v}" for k, v in r["kw"].items())
                    gs = f"(g={r['gate']['gate']:.2f}) " if r["gate"] and c.reader.read_name != "everywhere" else ""
                    w(f"[{c.short}]".ljust(W) + f" {gs}consistent {r['consistent']}/{r['n']}, inconsistent "
                      f"{r['inconsistent']}/{r['n']} | {kw}")
                    for t in r["samples"][:args.n_show]:
                        w(" " * 6 + f"e.g. {one_line(t)}")
                    tt = tot[c.cid]
                    tt[0] += r["consistent"]
                    tt[1] += r["inconsistent"]
                    tt[2] += r["n"]
                w("")
            w("totals over prompts: " + "; ".join(f"[{c.short}] cons {tot[c.cid][0]}/{tot[c.cid][2]} inc "
                                                  f"{tot[c.cid][1]}/{tot[c.cid][2]}" for c in cs))
            w("")
        w("--- 4. leakage: unrelated probes, greedy text identical to no memory? ---")
        un = [k for k in by if k[0] == "unrelated"]
        for i, (_, text) in enumerate(un, 1):
            w(f"  u{i}: {text}")
        for c in cs:
            if c.cid in ("nomem", "ctx"):
                continue
            flags = [by[k][c.cid]["identical_to_nomem"] for k in un]
            w(f"[{c.short}]".ljust(W) + " " + " ".join(f"u{i}:{'same' if f else 'DIFF'}" for i, f in enumerate(flags, 1))
              + f"  ({sum(flags)}/{len(flags)} identical)")

    w("")
    w("=" * 110)
    w("SUMMARY")
    w("facts, prefix completions: samples naming the target (summed over the 3 probes)")
    for s in [x for x in C.ids if C.by_id[x]["type"] == "fact"]:
        parts = []
        for c in conds[s]:
            rs = [r for r in prefix if r["scenario"] == s and r["condition"] == c.cid]
            parts.append(f"[{c.short}] {sum(r['counts'][r['target']]['full'] for r in rs)}/{sum(r['n'] for r in rs)}")
        w(f"  {s}: " + "; ".join(parts))
    w("dispositions: consistent / inconsistent samples (summed over the prompts)")
    for s in [x for x in C.ids if x in DISP]:
        parts = []
        for c in conds[s]:
            rs = [r for r in disp if r["scenario"] == s and r["condition"] == c.cid]
            parts.append(f"[{c.short}] {sum(r['consistent'] for r in rs)}/{sum(r['inconsistent'] for r in rs)} "
                         f"of {sum(r['n'] for r in rs)}")
        w(f"  {s}: " + "; ".join(parts))
    w("leakage: unrelated greedy texts identical to no memory")
    agg = defaultdict(lambda: [0, 0])
    for r in greedy:
        if r["kind"] == "unrelated" and r["condition"] not in ("nomem", "ctx"):
            agg[r["condition"]][0] += r["identical_to_nomem"]
            agg[r["condition"]][1] += 1
    w("  " + "; ".join(f"{cid}: {a}/{b}" for cid, (a, b) in agg.items()))
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------- main


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with torch.inference_mode():
        dargs = SimpleNamespace(model=args.model, layers=[args.layer], analytic_layers=[args.layer],
                                scenarios=args.scenarios, generic=args.generic, cont_len=args.cont_len,
                                gen_cont_len=args.gen_cont_len, white_eps=args.white_eps, tiny=args.tiny,
                                device=args.device)
        C = dk.setup(dargs)
        C.args, C.layer, C.dev = args, args.layer, args.device
        C.base_of = {s: "without" if C.by_id[s]["type"] == "fact" else "contrastive" for s in C.ids}
        C.unrel_texts = [C.unrelated[i] for i in args.unrelated_idx]
        ge = C.model.generation_config.eos_token_id
        eos = {C.tok.eos_token_id} | set(ge if isinstance(ge, (list, tuple)) else [ge])
        C.eos = torch.tensor(sorted(e for e in eos if e is not None), device=args.device)
        t_setup = time.time() - t0
        log(f"setup done in {t_setup / 60:.1f} min; eos {C.eos.tolist()}")

        conds, mems, chk = conditions(C)
        for n, m in mems.items():
            log(f"memory {n}: thr {m['thr']:.4f} sd {m['sd_generic_match']:.4f} writes {m['n_writes']}")
        check = check_readers(C, conds)

        n_gen = sum(len(conds[s]) * (3 + len(C.by_id[s].get("relation_probes", [])) + len(C.unrel_texts)) for s in C.ids)
        log(f"plan: <= {n_gen} greedy x {args.greedy_len} tok; prefix {args.prefix_samples} x {args.prefix_len} tok; "
            f"dispositions {args.disp_samples} x {args.disp_len} tok")
        times = {}
        t1 = time.time()
        greedy = run_greedy(C, conds)
        times["greedy_min"] = (time.time() - t1) / 60
        t1 = time.time()
        prefix = run_prefix(C, conds)
        times["prefix_min"] = (time.time() - t1) / 60
        t1 = time.time()
        disp = run_disp(C, conds)
        times["disp_min"] = (time.time() - t1) / 60

    with open(out / "samples.jsonl", "w") as f:
        for r in check:
            f.write(json.dumps({"section": "check", **r}) + "\n")
        for r in greedy + prefix + disp:
            f.write(json.dumps(r) + "\n")
    dk.write_csv(out / "counts.csv", count_rows(C, prefix, disp, greedy))
    write_txt(out / "samples.txt", C, conds, mems, greedy, prefix, disp)
    times.update(setup_min=t_setup / 60, total_min=(time.time() - t0) / 60)
    cfg = {"args": vars(args), "ids": C.ids, "baseline_of": C.base_of, "unrelated_probes": C.unrel_texts,
           "conditions": {s: [{"cid": c.cid, "short": c.short, "label": c.label} for c in cs] for s, cs in conds.items()},
           "memories": mems, "build_checks": chk, "reader_check": check,
           "disposition_rules": {s: {"prompts": d["prompts"], "kw": d["kw"], "rule": d["rule"]} for s, d in DISP.items()},
           "mu_skip_tokens": C.skip, "tail_len": C.tail_len, "white_info": C.white_info, "eos": C.eos.tolist(),
           "seeding": "seed = crc32(section|prompt) mixed with --seed; the same for every condition of a prompt",
           "times": times, "torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    log(f"wrote {out} in {times['total_min']:.1f} min ({json.dumps({k: round(v, 1) for k, v in times.items()})})")


if __name__ == "__main__":
    main()
