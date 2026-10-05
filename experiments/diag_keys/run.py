#!/usr/bin/env python
"""Seahorse diag_keys: can memory recall be made SELECTIVE?

Known (docs/history/experiment-log.md §4.5): recall is strong but unselective (~0.4 of an imprint
comes back on unrelated prompts, two-thirds of it on the chat template); centred keys of
different scenarios are as similar as keys within one scenario (~0.2 cosine); in combined
mode the last write wins and same-topic memories cross-talk at read time.

Factors (the method is otherwise v0.1: entropy gate, alpha 2, baselines without/contrastive):
  key (the same function at write and read)
    centred      normalize(h - mu), per token (the current key)
    white64/256  normalize(Lambda_r^-1/2 U_r^T (h - mu)): PCA-truncated whitening with the
                 covariance of generic-prompt residuals (template head excluded, as for mu);
                 eigenvalues floored by + --white-eps x mean eigenvalue
    whiteLW      full whitening, Ledoit-Wolf-shrunk covariance (analytic pass only)
    pooled_c     turn-level key: normalize(running mean of (h - mu) over the user text so
                 far). Write: the follow-up's whole user text (unit pooled) or the running
                 mean at each written token (unit all). Read at t: user-text positions <= t;
                 tail/continuation use all user text; template head: no user text -> zero
                 key -> no recall (so `everywhere` == `mask_head` and is skipped)
    pooled_w256  the same over white256 features
  read
    everywhere | mask_head (no read at template-head positions) | hard (mask_head + recall
    only where match > thr) | soft_t<x> (mask_head + recall x sigmoid((match-thr)/T)) |
    stoch_t<x> (mask_head + Bernoulli recall with the soft probability, --seeds seeds).
    match = max cos(read key, the memory's stored write keys). thr = --thr-q quantile of
    match over generic prompts + their greedy continuations (non-head positions), per
    memory; T = x * sd(generic match).
  write rule (combined mode): delta (current) | rls (recursive least squares with key
    covariance + ridge --rls-lambda; equals the batch ridge solution, so order-free up to
    numerics), each in the original and the reversed write order
  write unit: pooled (one entropy-weighted write per follow-up; main) | all (per token)
  mode: isolated | combined. Writes exclude the 5-token template tail unless --write-tail.

Passes
  analytic   (no forward passes with memory; h at the injection layer does not depend on the
             memory) full factorial on --analytic-layers: recall fraction |g M k|/mean|delta|
             by probe kind, selectivity index (related / unrelated), steer-energy share on
             template positions, active fraction -> selectivity.csv
  token vectors, key geometry -> token_vectors.csv, key_geometry.csv
  sanity     centred + everywhere + tail-in + all + delta, built with the library, must
             reproduce v0.1 (--v01-results); the same memory built here must be identical
  behaviour  (a) key x read x {isolated, combined delta orig} for this part's write units;
             (b) the best --n-best key/read combos (analytic SI, recall kept) + the current
             one: combined x {delta, rls} x {orig, rev}. v0.1 metrics + leakage KL.
  --part 1: unit pooled, --part 2: unit all, --part all: both. --merge DIR... rebuilds
  summary.csv/report.txt from several part outputs.

Outputs: results.jsonl, summary.csv, selectivity.csv, token_vectors.csv, key_geometry.csv,
report.txt, config.json
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
from seahorse.residual import capture, inject, load_model
from seahorse.sessions import chat_ids, common_suffix_len

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v01 = _load("seahorse_v0_1_run", HERE.parent / "v0_1" / "run.py")
dord = _load("seahorse_diag_order_run", HERE.parent / "diag_order" / "run.py")  # repro_check, table, fmt

KEY_SPECS = {  # name: (whitening, pooled)
    "centred": (None, False),
    "white64": (("pca", 64), False),
    "white256": (("pca", 256), False),
    "whiteLW": (("lw", None), False),
    "pooled_c": (None, True),
    "pooled_w256": (("pca", 256), True),
}
DIST = ("exact", "paraphrase", "related")
KINDS = DIST + ("relation", "other", "unrelated")
CK = ("unit", "key", "read", "tail", "rule", "order", "mode", "baseline", "layer")
MEMCONDS = [("isolated", "delta", "-"), ("combined", "delta", "orig"), ("combined", "delta", "rev"),
            ("combined", "rls", "orig"), ("combined", "rls", "rev")]
REF = ("centred", "everywhere")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[23, 26], help="behaviour layers")
    p.add_argument("--analytic-layers", type=int, nargs="+", default=[17, 23, 26])
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--baselines", nargs="+", default=["without", "contrastive"], choices=["without", "contrastive"])
    p.add_argument("--keys", nargs="+", default=["centred", "white64", "white256", "pooled_c", "pooled_w256"],
                   choices=list(KEY_SPECS), help="behaviour keys")
    p.add_argument("--analytic-keys", nargs="+", default=list(KEY_SPECS), choices=list(KEY_SPECS))
    p.add_argument("--part", default="all", choices=["all", "1", "2"],
                   help="1 = write unit pooled (+ sanity), 2 = write unit all (+ sanity), all = both")
    p.add_argument("--write-tail", action="store_true", help="also write the template tail (default: excluded)")
    p.add_argument("--thr-q", type=float, default=0.95)
    p.add_argument("--soft-t", type=float, nargs="+", default=[0.5, 1.0], help="T in units of sd(generic match)")
    p.add_argument("--stoch-t", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rls-lambda", type=float, default=0.1)
    p.add_argument("--white-eps", type=float, default=0.01)
    p.add_argument("--n-best", type=int, default=2, help="best key/read combos (besides the current) for (b)")
    p.add_argument("--recall-keep", type=float, default=0.75,
                   help="a combo 'keeps recall' if its related recall (and gap) >= this x the current one's")
    p.add_argument("--no-sanity", action="store_true")
    p.add_argument("--scenarios", default=str(HERE.parent / "v0_1" / "scenarios.yaml"))
    p.add_argument("--generic", default=str(v01.GENERIC))
    p.add_argument("--cont-len", type=int, default=20)
    p.add_argument("--gen-cont-len", type=int, default=20, help="greedy continuation of generic prompts (calibration)")
    p.add_argument("--v01-results",
                   default=f"/scratch/{os.environ.get('USER', 'unknown')}/seahorse_runs/v0_1_415909/results.jsonl")
    p.add_argument("--repro-tol", type=float, default=1e-2)
    p.add_argument("--tiny", action="store_true", help="smoke test: tiny random Qwen2 with --model's tokenizer")
    p.add_argument("--merge", nargs="+", default=None, help="part output dirs to merge into --out (no model)")
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
                      num_hidden_layers=max(args.layers + args.analytic_layers) + 2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    model = Qwen2ForCausalLM(cfg).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def read_names(args):
    return ["everywhere", "mask_head", "hard"] + [f"soft_t{t:g}" for t in args.soft_t] + [f"stoch_t{args.stoch_t:g}"]


def read_spec(name):
    """(mask_head, gate mode, t)."""
    if name == "everywhere":
        return False, None, None
    if name == "mask_head":
        return True, None, None
    if name == "hard":
        return True, "hard", None
    for g in ("soft", "stoch"):
        if name.startswith(g + "_t"):
            return True, g, float(name[len(g) + 2:])
    raise ValueError(name)


def reads_for(key, names):
    return [r for r in names if not (KEY_SPECS[key][1] and r == "everywhere")]


# ------------------------------------------------------------------------- keys


class KeySpace:
    """k(h) = normalize(f(h)), f(h) = (h - mu) [@ W]; pooled: running mean of f over user text."""

    def __init__(self, name, mu, W):
        self.name, self.mu, self.W = name, mu, W
        self.pooled = KEY_SPECS[name][1]

    def keys(self, h, user=None):
        f = h - self.mu
        if self.W is not None:
            f = f @ self.W
        if self.pooled:
            u = user.to(f.dtype)[:, None]
            f = (f * u).cumsum(0) / u.cumsum(0).clamp_min(1.0)  # zero before the first user token
        return F.normalize(f, dim=-1, eps=1e-6)


def whitening(X, spec, eps):
    """X [N, d] centred residuals -> (W [d, r] float32, info)."""
    X = X.double()
    N, d = X.shape
    S = X.T @ X / N
    ev, U = torch.linalg.eigh(S)
    ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    m = ev.mean()
    info = {"n_tokens": N, "d": d, "top1_var_share": (ev[0] / ev.sum()).item()}
    if spec[0] == "pca":
        r = min(spec[1], d)
        W = U[:, :r] / (ev[:r] + eps * m).sqrt()
        info.update(kind="pca", r=r, eps=eps, var_captured=(ev[:r].sum() / ev.sum()).item())
    else:  # Ledoit-Wolf shrinkage towards m I
        d2 = ((S - m * torch.eye(d, dtype=S.dtype, device=S.device)) ** 2).sum()
        xn2 = (X * X).sum(1)
        b2 = (xn2 ** 2 - 2 * ((X @ S) * X).sum(1) + (S * S).sum()).sum() / N ** 2
        rho = (torch.minimum(b2, d2) / d2).item()
        W = U / ((1 - rho) * ev + rho * m).sqrt()
        info.update(kind="ledoit_wolf", r=d, rho=rho)
    return W.float(), info


# ------------------------------------------------------------------ positions


def make_catfn(head, tail, device):
    """ids -> [T] categories: 0 template head, 1 user text, 2 template tail, 3 continuation."""
    head_l, tail_l = head.tolist(), tail.tolist()
    nh, nt, end = len(head_l), len(tail_l), tail_l[0]
    cache = {}

    def f(ids):
        key = tuple(ids.tolist())
        if key not in cache:
            L = list(key)
            assert L[:nh] == head_l, "template head mismatch"
            e = L.index(end, nh)
            assert L[e:e + nt] == tail_l, "template tail mismatch"
            c = torch.full((len(L),), 3, dtype=torch.long)
            c[:nh], c[nh:e], c[e:e + nt] = 0, 1, 2
            cache[key] = c.to(device)
        return cache[key]
    return f


class Stash:
    """Remembers the input ids of the current forward (the read hook needs their categories)."""

    def __init__(self, model, catfn):
        self.ids, self.catfn = None, catfn
        model.register_forward_pre_hook(self.pre, with_kwargs=True)

    def pre(self, module, args, kwargs):
        ids = kwargs.get("input_ids")
        self.ids = (ids if ids is not None else args[0])[0]

    def cats(self):
        return self.catfn(self.ids)


# ----------------------------------------------------------------------- memory


class Mem:
    def __init__(self, d, r, device):
        self.M = torch.zeros(d, r, device=device)
        self.K, self.dn, self.own = [], [], []

    def add(self, sid, delta, keys):
        self.K.append(keys)
        self.dn.append(delta.norm(dim=-1))
        self.own += [sid] * keys.shape[0]

    def finalize(self):
        self.Kst, self.dnorm = torch.cat(self.K), torch.cat(self.dn)
        self._dn = {None: self.dnorm.mean().item()}
        for s in set(self.own):
            self._dn[s] = self.dnorm[torch.tensor([o == s for o in self.own], device=self.dnorm.device)].mean().item()
        return self

    def dn_mean(self, sid=None):
        return self._dn[sid]


def unit_writes(chunk, full, layer, base, unit, ks, tail_in, tail_len):
    """One follow-up -> (delta [T,d], keys [T,r], gate [T] or None) for write unit `unit`."""
    delta, h_key, g = v01.select(chunk, layer, "all", base, "entropy", tail_len, 4)
    n = delta.shape[0]
    assert n > tail_len, "follow-up has no content tokens before the template tail"
    if ks.pooled:
        kseq = ks.keys(full[layer], full["cats"] == 1)
        ktok, kpool = kseq[-n:], kseq[-1:]  # last position (tail) = mean over all user text
    else:
        ktok = ks.keys(h_key)
    keep = n if tail_in else n - tail_len
    delta, h_key, g, ktok = delta[:keep], h_key[:keep], g[:keep], ktok[:keep]
    if unit == "all":
        return delta, ktok, g
    w = (g / g.sum())[:, None]
    kp = kpool if ks.pooled else ks.keys((w * h_key).sum(0, keepdim=True))
    return (w * delta).sum(0, keepdim=True), kp, None


def make_stored(C, layer, base, unit, ks, tail_in):
    return {s: [unit_writes(ch, fu, layer, base, unit, ks, tail_in, C.tail_len)
                for ch, fu in zip(C.writes[s], C.full_wo[s])] for s in C.ids}


def build(stored, members, rule, lam, device, chk):
    d = stored[members[0]][0][0].shape[1]
    r = stored[members[0]][0][1].shape[1]
    mem = Mem(d, r, device)
    chunks = [(s, *c) for s in members for c in stored[s]]
    if rule == "delta":  # the same operations as FastWeightMemory.write (eta = 1)
        for s, delta, keys, g in chunks:
            for t in range(keys.shape[0]):
                k = keys[t]
                e = delta[t] - mem.M @ k
                gt = 1.0 if g is None else float(g[t])
                mem.M += (1.0 * gt) * torch.outer(e, k)
            mem.add(s, delta, keys)
        s, delta, keys, g = chunks[-1]
        if g is None:  # exact recall of the last, ungated write
            rel = ((mem.M @ keys[-1] - delta[-1]).norm() / delta[-1].norm().clamp_min(1e-8)).item()
            assert rel < 1e-3, f"recall sanity check failed: rel error {rel:.2e}"
            chk["exact_recall_n"] += 1
            chk["exact_recall_max"] = max(chk["exact_recall_max"], rel)
    elif rule == "rls":  # M = (sum g delta k^T)(lam I + sum g k k^T)^-1, recursively (Sherman-Morrison)
        P = torch.eye(r, dtype=torch.float64, device=device) / lam
        M = torch.zeros(d, r, dtype=torch.float64, device=device)
        A = torch.zeros(d, r, dtype=torch.float64, device=device)
        B = lam * torch.eye(r, dtype=torch.float64, device=device)
        for s, delta, keys, g in chunks:
            for t in range(keys.shape[0]):
                gt = 1.0 if g is None else float(g[t])
                if gt <= 0:
                    continue
                k, dl = keys[t].double(), delta[t].double()
                Pk = P @ k
                gain = Pk / (1.0 / gt + k @ Pk)
                M += torch.outer(dl - M @ k, gain)
                P -= torch.outer(gain, Pk)
                A += gt * torch.outer(dl, k)
                B += gt * torch.outer(k, k)
            mem.add(s, delta, keys)
        Mb = torch.linalg.solve(B, A.T).T
        rel = ((M - Mb).norm() / Mb.norm().clamp_min(1e-12)).item()
        assert rel < 1e-3, f"RLS != batch ridge solution: rel {rel:.2e}"
        chk["rls_n"] += 1
        chk["rls_batch_max_rel"] = max(chk["rls_batch_max_rel"], rel)
        mem.M = M.float()
    else:
        raise ValueError(rule)
    return mem.finalize()


def calib(mem, Kg, q):
    m = (Kg @ mem.Kst.T).max(-1).values
    return torch.quantile(m, q).item(), m.std().item()


def gate_fn(read, match, cats, thr, sd, gen=None):
    mask_head, mode, t = read_spec(read)
    g = torch.ones_like(match)
    if mask_head:
        g = g * (cats != 0)
    if mode == "hard":
        g = g * (match > thr)
    elif mode in ("soft", "stoch"):
        p = torch.sigmoid((match - thr) / max(t * sd, 1e-8))
        if mode == "soft" or gen is None:  # stoch without a generator = its expected value
            g = g * p
        else:
            g = g * (torch.rand(p.shape, generator=gen, device=p.device) < p)
    return g


class Reader:
    """Passed as `memory` to seahorse.residual.inject: read(h, alpha) = h + alpha g(t) M k(t)."""

    def __init__(self, mem, ks, read, thr, sd, seed, stash):
        self.mem, self.ks, self.read_name, self.thr, self.sd, self.stash = mem, ks, read, thr, sd, stash
        self.gen = None
        if read_spec(read)[1] == "stoch":
            self.gen = torch.Generator(device=mem.M.device)
            self.gen.manual_seed(1000 + seed)

    def recall(self, h, cats):
        k = self.ks.keys(h, cats == 1)
        r = k @ self.mem.M.T
        match = (k @ self.mem.Kst.T).max(-1).values
        return r, gate_fn(self.read_name, match, cats, self.thr, self.sd, self.gen)

    def read(self, h, alpha):
        assert h.shape[0] == 1 and h.shape[1] == len(self.stash.ids), "read hook: ids/sequence mismatch"
        r, g = self.recall(h[0], self.stash.cats())
        return h + alpha * (g[:, None] * r)[None]


# ------------------------------------------------------------------------ setup


def setup(args):
    cfg = yaml.safe_load(open(args.scenarios))
    C = SimpleNamespace(args=args, scenarios=cfg["scenarios"], unrelated=cfg["unrelated_probes"])
    C.ids = [s["id"] for s in C.scenarios]
    C.by_id = {s["id"]: s for s in C.scenarios}
    C.AL = sorted(set(args.layers) | set(args.analytic_layers))
    C.bargs = SimpleNamespace(device=args.device, topk=4, eta=1.0, cont_len=args.cont_len)
    generic = [l.strip() for l in open(args.generic) if l.strip()]
    dev = args.device

    log(f"loading {args.model} on {dev}{' (tiny random model)' if args.tiny else ''}")
    C.model, C.tok = load(args)
    model, tok = C.model, C.tok
    v01.tok_global = tok
    C.tail_len = common_suffix_len(chat_ids(tok, "alpha"), chat_ids(tok, "beta"))

    log(f"mu over {len(generic)} generic prompts, layers {C.AL}")
    C.mu, C.skip = v01.compute_mu(model, tok, generic, C.AL, dev)
    C.head = chat_ids(tok, generic[0])[:C.skip]
    C.tail = chat_ids(tok, "alpha")[-C.tail_len:]
    C.catfn = make_catfn(C.head, C.tail, dev)
    C.stash = Stash(model, C.catfn)

    def cap(ids):
        with capture(model, C.AL) as st:
            model(ids[None].to(dev))
        return {**{l: st[l] for l in C.AL}, "cats": C.catfn(ids)}

    log("session 1: writes (with / without / counter) + full without-run residuals")
    C.writes = {s["id"]: v01.collect_writes(model, tok, s, C.AL, dev) for s in C.scenarios}
    C.full_wo = {s["id"]: [cap(chat_ids(tok, fu)) for fu in s["followups"]] for s in C.scenarios}
    for s in C.ids:
        for ch, fu in zip(C.writes[s], C.full_wo[s]):
            for l in C.AL:
                assert torch.allclose(ch["h"][l][1], fu[l][-ch["n"]:], atol=1e-4, rtol=1e-4)

    log("references (no memory)")
    C.refs = {s["id"]: v01.reference(model, tok, s, C.bargs) for s in C.scenarios}
    C.unrel = []
    for text in C.unrelated:
        u = chat_ids(tok, text)
        cont = mx.greedy(model, tok, u, args.cont_len, dev)
        C.unrel.append({"ids": u, "cont": cont, "lp_base": mx.cont_logprobs(model, u, cont, dev)})

    log("analytic inputs: probes + continuations, relation probes, unrelated probes")
    inputs = []
    for s in C.ids:
        for r in C.refs[s]:
            if r["kind"] == "probe":
                inputs.append((s, r["distance"], torch.cat([r["base_ids"], r["cont"]])))
            else:
                inputs.append((s, "relation", r["base_ids"]))
    inputs += [(None, "unrelated", torch.cat([u["ids"], u["cont"]])) for u in C.unrel]
    caps = [cap(x[2]) for x in inputs]
    C.P = {"meta": [{"sid": s, "kind": k, "len": len(x)} for s, k, x in inputs],
           "H": {l: torch.cat([c[l] for c in caps]) for l in C.AL},
           "cats": torch.cat([c["cats"] for c in caps]),
           "idx": torch.cat([torch.full((len(x),), i, dtype=torch.long) for i, (_, _, x) in enumerate(inputs)]).to(dev),
           "caps": caps}

    log(f"generic prompts + greedy continuations ({args.gen_cont_len} tokens): covariance + calibration")
    C.gen = []
    for text in generic:
        g_ids = chat_ids(tok, text)
        cont = mx.greedy(model, tok, g_ids, args.gen_cont_len, dev)
        c = cap(torch.cat([g_ids, cont]))
        c["plen"] = len(g_ids)
        C.gen.append(c)
    C.white_info, C.KS = {}, {}
    for l in C.AL:
        X = torch.cat([c[l][C.skip:c["plen"]] for c in C.gen]) - C.mu[l]
        Ws = {}
        for name, (spec, _) in KEY_SPECS.items():
            if spec is not None and spec not in Ws:
                Ws[spec] = whitening(X, spec, args.white_eps)
                C.white_info[f"L{l} {spec[0]}{spec[1] or ''}"] = Ws[spec][1]
        C.KS[l] = {name: KeySpace(name, C.mu[l], Ws[spec][0] if spec else None) for name, (spec, _) in KEY_SPECS.items()}
    C.fwd = {s: sum(1 + (1 + len(r["foil_base"]) if C.by_id[s]["type"] == "fact" else 2) if r["kind"] == "probe" else 2
                    for r in C.refs[s]) for s in C.ids}
    return C


def read_keys(C, layer, ks):
    """Read keys at every analytic-input position (per input, for the running mean)."""
    return torch.cat([ks.keys(c[layer], c["cats"] == 1) for c in C.P["caps"]])


def gen_keys(C, layer, ks):
    """Read keys at the non-head positions of the generic calibration sequences."""
    return torch.cat([ks.keys(c[layer], c["cats"] == 1)[c["cats"] != 0] for c in C.gen])


# --------------------------------------------------------------------- analytic


def seg_mean(v, idx, n, mask=None):
    w = torch.ones_like(v) if mask is None else mask.to(v.dtype)
    s = torch.zeros(n, device=v.device, dtype=v.dtype).index_add_(0, idx, v * w)
    c = torch.zeros(n, device=v.device, dtype=v.dtype).index_add_(0, idx, w)
    return (s / c.clamp_min(1)).cpu().tolist()


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def ratio(a, b):
    return a / b if b and not math.isnan(b) and abs(b) > 1e-12 and not math.isnan(a) else float("nan")


def analytic(C, chk):
    args, P = C.args, C.P
    cats, idx = P["cats"], P["idx"]
    n_inp = len(P["meta"])
    nh, us = cats != 0, cats == 1
    unrel_pos = torch.tensor([P["meta"][i]["sid"] is None for i in range(n_inp)], device=idx.device)[idx]
    reads = [r for r in read_names(args) if read_spec(r)[1] != "stoch"]
    rows, thr_cache = [], {}
    for layer in C.AL:
        for key in args.analytic_keys:
            ks = C.KS[layer][key]
            Kr, Kg = read_keys(C, layer, ks), gen_keys(C, layer, ks)
            for base in args.baselines:
                for unit in ("pooled", "all"):
                    stored = make_stored(C, layer, base, unit, ks, args.write_tail)
                    for mode, rule, order in MEMCONDS:
                        if mode == "isolated":
                            mems = {s: build(stored, [s], "delta", args.rls_lambda, args.device, chk) for s in C.ids}
                        else:
                            members = C.ids if order == "orig" else C.ids[::-1]
                            mems = {"combined": build(stored, members, rule, args.rls_lambda, args.device, chk)}
                        pre = {}
                        for gid, mem in mems.items():
                            thr, sd = calib(mem, Kg, args.thr_q)
                            thr_cache[(layer, key, base, unit, mode, rule, order, gid)] = (thr, sd)
                            pre[gid] = ((Kr @ mem.M.T).norm(dim=-1), (Kr @ mem.Kst.T).max(-1).values, thr, sd, mem)
                        for read in reads_for(key, reads):
                            acc = defaultdict(list)
                            E = torch.zeros(4, device=idx.device)
                            for gid, (rn, match, thr, sd, mem) in pre.items():
                                g = gate_fn(read, match, cats, thr, sd)
                                v = g * rn
                                rf, rfnh, rfu = seg_mean(v, idx, n_inp), seg_mean(v, idx, n_inp, nh), seg_mean(v, idx, n_inp, us)
                                act = seg_mean(g, idx, n_inp, nh)
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
                                    acc[("rfnh", cls)].append(rfnh[i] / dn)
                                    acc[("rfu", cls)].append(rfu[i] / dn)
                                    acc[("act", cls)].append(act[i])
                                E += torch.stack([((v ** 2) * (unrel_pos & (cats == c))).sum() for c in range(4)])
                                acc["thr"].append(thr)
                                acc["sd"].append(sd)
                            E = E.cpu().tolist()
                            row = {"layer": layer, "baseline": base, "unit": unit, "key": key, "mode": mode,
                                   "rule": rule, "order": order, "read": read,
                                   "thr": mean(acc["thr"]), "sd": mean(acc["sd"])}
                            for stat in ("rf", "rfnh", "rfu"):
                                for kind in KINDS:
                                    row[f"{stat}_{kind}"] = mean(acc[(stat, kind)])
                            row["act_related"], row["act_unrelated"] = mean(acc[("act", "related")]), mean(acc[("act", "unrelated")])
                            row["si"] = ratio(row["rf_related"], row["rf_unrelated"])
                            row["si_nohead"] = ratio(row["rfnh_related"], row["rfnh_unrelated"])
                            row["si_user"] = ratio(row["rfu_related"], row["rfu_unrelated"])
                            row["si_other"] = ratio(row["rf_related"], row["rf_other"])
                            tot = sum(E)
                            row["tmpl_share_unrel"] = ratio(E[0] + E[2], tot)
                            row["head_share_unrel"], row["tail_share_unrel"] = ratio(E[0], tot), ratio(E[2], tot)
                            rows.append(row)
            log(f"analytic: layer {layer} key {key} done")
    return rows, thr_cache


def vec_stats(Y, S, Fu):
    """Cosine structure of vectors Y [N, d] with scenario labels S and follow-up labels Fu."""
    out = {}
    same_fu, same_sc = Fu[:, None] == Fu[None, :], S[:, None] == S[None, :]
    eye = torch.eye(len(Y), dtype=torch.bool, device=Y.device)
    masks = {"fu": same_fu & ~eye, "scen": same_sc & ~same_fu, "across": ~same_sc}
    for pre, Z in (("cos", Y), ("ccos", Y - Y.mean(0))):
        Zn = F.normalize(Z, dim=-1, eps=1e-8)
        Cm = Zn @ Zn.T
        for name, m in masks.items():
            out[f"{pre}_{name}"] = Cm[m].mean().item() if m.any() else float("nan")
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    fu_mean, sc_mean = torch.zeros_like(Y), torch.zeros_like(Y)
    for lab, tgt in ((Fu, fu_mean), (S, sc_mean)):
        for u in lab.unique():
            m = lab == u
            tgt[m] = Y[m].mean(0)
    out["r2_fu"] = (1 - ((Y - fu_mean) ** 2).sum() / ss_tot).item()
    out["r2_scen"] = (1 - ((Y - sc_mean) ** 2).sum() / ss_tot).item()
    out["energy_fu_mean"] = ((fu_mean ** 2).sum() / (Y ** 2).sum()).item()
    out["cos_to_fu_mean"] = F.cosine_similarity(Y, fu_mean, dim=-1).mean().item()
    out["mean_norm"] = Y.norm(dim=-1).mean().item()
    return out


def token_vectors(C):
    rows = []
    for layer in C.AL:
        for base in C.args.baselines:
            X, S, Fu, T = [], [], [], []
            for si, s in enumerate(C.ids):
                for ci, ch in enumerate(C.writes[s]):
                    delta, _, _ = v01.select(ch, layer, "all", base, "entropy", C.tail_len, 4)
                    n = delta.shape[0]
                    X.append(delta)
                    S += [si] * n
                    Fu += [si * 100 + ci] * n
                    T += [t >= n - C.tail_len for t in range(n)]
            X = torch.cat(X)
            dev = X.device
            S, Fu, T = torch.tensor(S, device=dev), torch.tensor(Fu, device=dev), torch.tensor(T, device=dev)
            for name, m in (("content", ~T), ("tail", T), ("all", torch.ones_like(T))):
                rows.append({"layer": layer, "baseline": base, "tokens": name, "n": int(m.sum()),
                             **vec_stats(X[m], S[m], Fu[m])})
    return rows


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return ((d > 0).float().mean() + 0.5 * (d == 0).float().mean()).item()


def key_geometry(C):
    args, P = C.args, C.P
    cats, idx = P["cats"], P["idx"]
    rows = []
    for layer in C.AL:
        for key in args.analytic_keys:
            ks = C.KS[layer][key]
            Kr, Kg = read_keys(C, layer, ks), gen_keys(C, layer, ks)
            tok = make_stored(C, layer, "without", "all", ks, False)
            pool = make_stored(C, layer, "without", "pooled", ks, False)
            Kt = {s: torch.cat([c[1] for c in tok[s]]) for s in C.ids}
            Kp = {s: torch.cat([c[1] for c in pool[s]]) for s in C.ids}
            dev = Kr.device
            S = torch.tensor([si for si, s in enumerate(C.ids) for c in tok[s] for _ in range(c[1].shape[0])], device=dev)
            Fu = torch.tensor([si * 100 + ci for si, s in enumerate(C.ids) for ci, c in enumerate(tok[s])
                               for _ in range(c[1].shape[0])], device=dev)
            row = {"layer": layer, "key": key}
            st = vec_stats(torch.cat([Kt[s] for s in C.ids]), S, Fu)
            row.update(tok_within_fu=st["cos_fu"], tok_within_scen=st["cos_scen"], tok_between=st["cos_across"])
            Sp = torch.tensor([si for si, s in enumerate(C.ids) for _ in range(Kp[s].shape[0])], device=dev)
            stp = vec_stats(torch.cat([Kp[s] for s in C.ids]), Sp, torch.arange(len(Sp), device=dev))
            row.update(pool_within_scen=stp["cos_scen"], pool_between=stp["cos_across"])
            row["tok_sep"] = row["tok_within_scen"] - row["tok_between"]
            row["pool_sep"] = row["pool_within_scen"] - row["pool_between"]
            user = cats == 1
            for kind_name, Ks in (("tok", Kt), ("pool", Kp)):
                correct = defaultdict(list)
                au, ao, p95, rel_med = [], [], [], []
                match = {s: (Kr @ Ks[s].T).max(-1).values for s in C.ids}
                for i, m in enumerate(P["meta"]):
                    if m["sid"] is None or m["kind"] not in DIST:
                        continue
                    pos = (idx == i) & user
                    scores = [match[s][pos].mean().item() for s in C.ids]
                    correct[m["kind"]].append(float(C.ids[max(range(len(scores)), key=scores.__getitem__)] == m["sid"]))
                own = lambda s: torch.tensor([mm["sid"] == s and mm["kind"] in DIST for mm in P["meta"]], device=dev)[idx]
                unrel = torch.tensor([mm["sid"] is None for mm in P["meta"]], device=dev)[idx]
                for s in C.ids:
                    o = own(s)
                    other = torch.tensor([mm["sid"] not in (None, s) and mm["kind"] in DIST for mm in P["meta"]], device=dev)[idx]
                    au.append(auc(match[s][o & user], match[s][unrel & user]))
                    ao.append(auc(match[s][o & user], match[s][other & user]))
                    gm = (Kg @ Ks[s].T).max(-1).values
                    p95.append(torch.quantile(gm, args.thr_q).item())
                    rel_med.append(match[s][o & user].median().item())
                for k in DIST:
                    row[f"{kind_name}_retr_{k}"] = mean(correct[k])
                row[f"{kind_name}_retr_all"] = mean(sum((correct[k] for k in DIST), []))
                row[f"{kind_name}_auc_unrel"], row[f"{kind_name}_auc_other"] = mean(au), mean(ao)
                row[f"{kind_name}_generic_p95"], row[f"{kind_name}_own_median"] = mean(p95), mean(rel_med)
            rows.append(row)
    return rows


def choose_combos(sel, C):
    """Per unit: the current combo + the --n-best (key, read) by analytic SI (all positions),
    averaged over baselines x behaviour layers x {isolated, combined delta orig}, among combos
    whose non-head related recall >= --recall-keep x the current combo's. Stochastic reads
    are left out (their expected value is the soft read's)."""
    args = C.args
    out, table = {}, {}
    for unit in ("pooled", "all"):
        agg = defaultdict(lambda: defaultdict(list))
        for r in sel:
            if (r["unit"] == unit and r["layer"] in args.layers and r["key"] in args.keys
                    and r["rule"] == "delta" and r["order"] in ("-", "orig")):
                agg[(r["key"], r["read"])]["si"].append(r["si"])
                agg[(r["key"], r["read"])]["rec"].append(r["rfnh_related"])
        ref_rec = mean(agg[REF]["rec"])
        cands = []
        for combo, a in agg.items():
            si, rec = mean(a["si"]), mean(a["rec"])
            ok = rec >= args.recall_keep * ref_rec
            cands.append((combo, si, rec / ref_rec if ref_rec else float("nan"), ok))
        cands.sort(key=lambda x: -x[1] if not math.isnan(x[1]) else 0)
        best = [c for c, _, _, ok in cands if ok and c != REF][:args.n_best]
        out[unit] = [REF] + best
        table[unit] = cands
    return out, table


# ------------------------------------------------------------------ behaviour


def evaluate_mem(C, reader, members, layer, tag):
    rows = []
    for wi, s in enumerate(members):
        rows += [{**tag, "write_index": wi, **r} for r in
                 v01.evaluate(C.model, C.by_id[s], C.refs[s], reader, layer, C.args.alpha, C.args.device)]
    rows.append({**tag, "kind": "leak", "kl": v01.eval_leakage(C.model, C.unrel, reader, layer, C.args.alpha,
                                                                C.args.device)})
    return rows


def sanity(C, chk):
    """centred + everywhere + tail-in + all + delta, built by the library: must reproduce v0.1."""
    args = C.args
    rows, rc_rows, rc_leak = [], [], defaultdict(list)
    for base in args.baselines:
        for layer in args.layers:
            ks = C.KS[layer]["centred"]
            stored = make_stored(C, layer, base, "all", ks, True)
            for mode, groups in (("isolated", [(s, [s]) for s in C.ids]), ("combined", [("combined", C.ids)])):
                for gid, members in groups:
                    lib, _, last = v01.build_memory(C.mu[layer], C.writes, [C.by_id[s] for s in members], layer, "all",
                                                    base, "entropy", C.bargs, C.tail_len)
                    v01.check_recall(lib, last, 1.0)
                    mine = build(stored, members, "delta", args.rls_lambda, args.device, chk)
                    assert torch.equal(mine.M, lib.M), f"{gid}: memory differs from the library build"
                    chk["lib_equal_n"] += 1
                    if gid == C.ids[0]:  # the read path here == the library's inject
                        r0 = C.refs[gid][0]
                        ids = torch.cat([r0["base_ids"], r0["cont"]])[None].to(args.device)
                        with inject(C.model, layer, lib, args.alpha):
                            la = C.model(ids).logits
                        with inject(C.model, layer, Reader(mine, ks, "everywhere", 0.0, 1.0, 0, C.stash), args.alpha):
                            lb = C.model(ids).logits
                        chk["reader_vs_inject_max"] = max(chk["reader_vs_inject_max"], (la - lb).abs().max().item())
                    tag = {"unit": "all", "key": "centred", "read": "everywhere", "tail": "in", "rule": "delta",
                           "order": "-" if mode == "isolated" else "orig", "mode": mode, "baseline": base,
                           "layer": layer, "alpha": args.alpha, "memory": gid, "seed": 0, "part": "sanity"}
                    rs = evaluate_mem(C, lib, members, layer, tag)
                    rows += rs
                    cond = "iso" if mode == "isolated" else "a_comb_orig"
                    for r in rs:
                        if r["kind"] == "leak":
                            rc_leak[(cond, base, layer, gid)] += r["kl"]
                        else:
                            rc_rows.append({**r, "condition": cond})
            log(f"sanity: baseline={base} layer={layer} done")
    assert chk["reader_vs_inject_max"] < 1e-3, f"reader != library inject: {chk['reader_vs_inject_max']}"
    if args.tiny:
        repro = {"status": "SKIPPED", "reason": "--tiny"}
    else:
        repro = dord.repro_check(args.v01_results, rc_rows, rc_leak, args.alpha, args.repro_tol)
    log(f"reproduction check: {repro}")
    return rows, repro


def behaviour(C, units, combos, thr_cache, chk):
    args = C.args
    names = read_names(args)
    n_scen_fwd = sum(C.fwd.values())
    per_iso, per_comb = n_scen_fwd + len(C.ids) * len(C.unrel), n_scen_fwd + len(C.unrel)
    plan = 0
    for unit in units:
        for key in args.keys:
            for read in reads_for(key, names):
                ns = args.seeds if read_spec(read)[1] == "stoch" else 1
                extra = 3 if (key, read) in combos[unit] else 0
                plan += ns * (per_iso + per_comb * (1 + extra))
    plan *= len(args.baselines) * len(args.layers)
    log(f"behaviour: ~{plan} forward passes planned (units {units})")
    rows, done_fwd, t0 = [], 0, time.time()
    for unit in units:
        for base in args.baselines:
            for layer in args.layers:
                for key in args.keys:
                    ks = C.KS[layer][key]
                    stored = make_stored(C, layer, base, unit, ks, args.write_tail)
                    mems = {("isolated", "delta", "-", s): build(stored, [s], "delta", args.rls_lambda, args.device, chk)
                            for s in C.ids}
                    need = [("delta", "orig")]
                    if any((key, r) in combos[unit] for r in names):
                        need += [("delta", "rev"), ("rls", "orig"), ("rls", "rev")]
                    for rule, order in need:
                        members = C.ids if order == "orig" else C.ids[::-1]
                        mems[("combined", rule, order, "combined")] = build(stored, members, rule, args.rls_lambda,
                                                                            args.device, chk)
                    Kg = gen_keys(C, layer, ks)
                    thr = {}
                    for mk, mem in mems.items():
                        thr[mk] = calib(mem, Kg, args.thr_q)
                        ref = thr_cache.get((layer, key, base, unit) + mk)
                        if ref is not None:
                            assert abs(ref[0] - thr[mk][0]) < 1e-5, "threshold differs from the analytic pass"
                    for read in reads_for(key, names):
                        use = [mk for mk in mems if mk[0] == "isolated" or mk[1:3] == ("delta", "orig")
                               or (key, read) in combos[unit]]
                        seeds = range(args.seeds) if read_spec(read)[1] == "stoch" else [0]
                        for seed in seeds:
                            for mk in use:
                                mode, rule, order, gid = mk
                                members = [gid] if mode == "isolated" else (C.ids if order == "orig" else C.ids[::-1])
                                reader = Reader(mems[mk], ks, read, thr[mk][0], thr[mk][1], seed, C.stash)
                                tag = {"unit": unit, "key": key, "read": read, "tail": "in" if args.write_tail else "out",
                                       "rule": rule, "order": order, "mode": mode, "baseline": base, "layer": layer,
                                       "alpha": args.alpha, "memory": gid, "seed": seed, "thr": thr[mk][0],
                                       "T": (read_spec(read)[2] or 0) * thr[mk][1], "part": args.part}
                                rows += evaluate_mem(C, reader, members, layer, tag)
                                done_fwd += (per_iso // len(C.ids)) if mode == "isolated" else per_comb
                    el = time.time() - t0
                    log(f"behaviour: unit={unit} base={base} L{layer} key={key} | {done_fwd}/{plan} fwd, "
                        f"{el / 60:.1f} min, eta {el / max(done_fwd, 1) * (plan - done_fwd) / 60:.1f} min")
    return rows


# ---------------------------------------------------------------------- summary


def ck(r):
    return tuple(r[k] for k in CK)


def effects(rs):
    pr = [r for r in rs if r["kind"] == "probe"]
    rel = [r for r in rs if r["kind"] == "relation"]

    def gap(sub):
        kb, km = sum(r["kl_base"] for r in sub), sum(r["kl_mem"] for r in sub)
        return 1 - km / kb if kb > 0 else float("nan")
    facts = [r for r in pr if r["type"] == "fact"]
    disp = [r for r in pr if r["type"] != "fact"]
    out = {"n": len(pr), "gap": gap(pr)}
    for d in DIST:
        out[f"gap_{d}"] = gap([r for r in pr if r["distance"] == d])
    out["dt_fact"] = mean(r["tgt_mem"] - r["tgt_base"] for r in facts)
    out["dt_fact_ceil"] = mean(r["tgt_ceil"] - r["tgt_base"] for r in facts)
    out["dfoil"] = mean(mean(m - b for m, b in zip(r["foil_mem"], r["foil_base"])) for r in facts)
    out["spec"] = out["dt_fact"] - out["dfoil"] if facts else float("nan")
    out["contrast"] = mean(r["tgt_mem"] - r["tgt_base"] for r in disp)
    out["contrast_ceil"] = mean(r["tgt_ceil"] - r["tgt_base"] for r in disp)
    out["drel_FLAGGED"] = mean(r["rel_mem"] - r["rel_base"] for r in rel)
    out["drel_ceil"] = mean(r["rel_ceil"] - r["rel_base"] for r in rel)
    return out


def index(rows):
    by, leak = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["kind"] == "leak":
            leak[ck(r)] += r["kl"]
        else:
            by[ck(r)].append(r)
    return by, leak


def summarize(rows):
    by, leak = index(rows)
    out = []
    for key in sorted(by, key=lambda k: tuple(str(x) for x in k)):
        rs = by[key]
        lk = mean(leak[key])
        seeds = len({r["seed"] for r in rs})
        groups = [("all", "all", rs)]
        for typ in sorted({r["type"] for r in rs}):
            for dist in DIST + ("relation",):
                sub = [r for r in rs if r["type"] == typ and r.get("distance") == dist]
                if sub:
                    groups.append((typ, dist, sub))
        for typ, dist, sub in groups:
            e = effects(sub)
            out.append({**dict(zip(CK, key)), "type": typ, "distance": dist, "n_seeds": seeds, **e, "leak": lk,
                        "gap_leak": ratio(e["gap"], lk)})
    return out


# ----------------------------------------------------------------------- report


fmt, table = dord.fmt, dord.table


def f3(x):
    return fmt(x, ".3f")


def report(rows, sel, tv, kg, meta):
    args = SimpleNamespace(**meta["args"])
    by, leak = index(rows)
    S = {}
    for key, rs in by.items():
        S[key] = {**effects(rs), "leak": mean(leak[key])}
        S[key]["gap_leak"] = ratio(S[key]["gap"], S[key]["leak"])
    SEL = {(r["unit"], r["key"], r["read"], r["mode"], r["rule"], r["order"], r["baseline"], r["layer"]): r for r in sel}
    tail = "in" if args.write_tail else "out"
    names = read_names(args)
    L = []
    w = L.append
    w("Seahorse diag_keys report")
    w("=" * 25)
    w(f"model {args.model}; alpha={args.alpha}; gate entropy; behaviour layers {args.layers}; analytic layers "
      f"{args.analytic_layers}; baselines {args.baselines}; writes: template tail {tail}")
    w(f"thr = q{args.thr_q} of max-cos match on generic prompts + greedy continuations (non-head), per memory; "
      f"soft T = {args.soft_t} x sd(generic match); stoch T = {args.stoch_t} x sd, {args.seeds} seeds; "
      f"rls lambda = {args.rls_lambda}; white eps = {args.white_eps} x mean eigenvalue")
    w("rf = recall fraction |g M k| / mean|delta| (mean over positions; rfnh = non-head positions); "
      "SI = rf(related) / rf(unrelated), all positions; tmpl = share of steer energy on template head+tail, unrelated probes.")
    w("gap = 1 - sum KL_mem / sum KL_base (probes vs the ceiling's continuation); leak = mean KL on unrelated probes; "
      "g/l = gap / leak.")
    w("RELATION PROBES ARE FLAWED: all six have 'No' as the consistent answer, so any shift towards 'No' scores as "
      "a gain (drel*). Reported, not interpreted.\n")
    w("CHECKS")
    for part, m in meta["parts"].items():
        w(f"  [{part}] reproduction of v0.1 (centred/everywhere/tail-in/all/delta, iso + combined): "
          f"{m['repro'].get('status')}  {json.dumps({k: v for k, v in m['repro'].items() if k != 'status'})}")
        w(f"  [{part}] {json.dumps(m['checks'])}")
    w("  whitening: " + "; ".join(f"{k}: " + ", ".join(f"{a}={fmt(b, '.3g') if isinstance(b, float) else b}"
                                                       for a, b in v.items()) for k, v in meta["white_info"].items()))
    w("")

    w("=" * 70)
    w("1. TOKEN VECTORS: is the stored change one vector per experience?")
    w("cos_* = mean cosine between delta vectors: fu = different tokens of one follow-up; scen = different follow-ups of "
      "one scenario; across = different scenarios. ccos = the same after removing the global mean delta. "
      "r2_fu / r2_scen = share of delta variance explained by the per-follow-up / per-scenario mean; "
      "E_fu = share of delta energy in the per-follow-up means.")
    hdr = ["layer base tokens", "n", "cos fu", "cos scen", "cos across", "ccos fu", "ccos scen", "ccos across",
           "r2_fu", "r2_scen", "E_fu", "cos(d,fu mean)", "|d|"]
    w(table(hdr, [[f"L{r['layer']} {r['baseline']} {r['tokens']}", r["n"]] +
                  [f3(r[k]) for k in ("cos_fu", "cos_scen", "cos_across", "ccos_fu", "ccos_scen", "ccos_across",
                                      "r2_fu", "r2_scen", "energy_fu_mean", "cos_to_fu_mean")] + [fmt(r["mean_norm"], ".3g")]
                  for r in tv]))
    w("")

    w("=" * 70)
    w("2. KEY GEOMETRY (write keys, tail excluded; keys do not depend on the baseline)")
    w("tok = per-token write keys, pool = one pooled key per follow-up. within = same scenario (tok: different follow-ups), "
      "between = different scenarios; sep = within - between. retr = top-1 retrieval of the right scenario from a probe's "
      "user-text read keys (mean max-match; chance 0.17). auc = P(match on own probes' user text > match on unrelated / "
      "other scenarios' probes). p95 = generic threshold; own = median match on own probes.")
    hdr = ["layer key", "tok w-fu", "tok within", "tok between", "tok sep", "pool within", "pool between", "pool sep",
           "retr tok", "retr pool", "auc unrel tok", "auc other tok", "auc unrel pool", "auc other pool", "p95 tok", "own tok"]
    w(table(hdr, [[f"L{r['layer']} {r['key']}"] + [f3(r[k]) for k in (
        "tok_within_fu", "tok_within_scen", "tok_between", "tok_sep", "pool_within_scen", "pool_between", "pool_sep",
        "tok_retr_all", "pool_retr_all", "tok_auc_unrel", "tok_auc_other", "pool_auc_unrel", "pool_auc_other",
        "tok_generic_p95", "tok_own_median")] for r in kg]))
    w("")

    w("=" * 70)
    w("3. SELECTIVITY (analytic, no forward passes; baseline without). iso = isolated; comb = combined, delta, original order.")
    for unit in ("pooled", "all"):
        for mode, rule, order in (("isolated", "delta", "-"), ("combined", "delta", "orig")):
            hdr = [f"unit={unit} {mode}"]
            for l in args.analytic_layers:
                hdr += [f"L{l} rfR", "rfU", "rfO" if mode == "isolated" else "rfX", "SI", "SInh", "tmpl", "actU"]
            rr = []
            for key in args.analytic_keys:
                for read in reads_for(key, [n for n in names if read_spec(n)[1] != "stoch"]):
                    cells = [f"{key}/{read}"]
                    for l in args.analytic_layers:
                        r = SEL.get((unit, key, read, mode, rule, order, "without", l))
                        if r is None:
                            cells += ["-"] * 7
                            continue
                        cells += [f3(r["rf_related"]), f3(r["rf_unrelated"]),
                                  f3(r["rf_other"] if mode == "isolated" else r["rf_exact"]), fmt(r["si"], ".2f"),
                                  fmt(r["si_nohead"], ".2f"), fmt(r["tmpl_share_unrel"], ".2f"), f3(r["act_unrelated"])]
                    rr.append(cells)
            w(table(hdr, rr))
            w("(rfR related, rfU unrelated, rfO other scenarios' probes (isolated), rfX exact (combined); actU = mean gate on "
              "non-head unrelated positions)\n")
    if meta.get("combo_table"):
        w("Combo choice for (b): analytic SI averaged over baselines x behaviour layers x {iso, comb}; rec = non-head related "
          f"recall / the current combo's (must be >= {args.recall_keep}).")
        for unit, cands in meta["combo_table"].items():
            w(f"  unit={unit}: chosen {meta['combos'][unit]}")
            w("    " + "; ".join(f"{k}/{r} SI={fmt(si, '.2f')} rec={fmt(rec, '.2f')}{'' if ok else ' (x)'}"
                                 for (k, r), si, rec, ok in cands[:12]))
        w("")

    w("=" * 70)
    w("4. BEHAVIOUR (a): key x read, delta rule. gapR = gap on related probes; SI from the analytic pass.")
    units = sorted({k[0] for k in S if k[3] == tail}, key=lambda u: u != "pooled")
    for unit in units:
        for base in args.baselines:
            for l in args.layers:
                hdr = [f"unit={unit} {base} L{l}"]
                for m in ("iso", "comb"):
                    hdr += [f"{m} gap", "gapR", "leak", "g/l", "SI"]
                rr = []
                for key in args.keys:
                    for read in reads_for(key, names):
                        cells = [f"{key}/{read}"]
                        for mode, order in (("isolated", "-"), ("combined", "orig")):
                            e = S.get((unit, key, read, tail, "delta", order, mode, base, l))
                            sread = read if read_spec(read)[1] != "stoch" else f"soft_t{args.stoch_t:g}"
                            sr = SEL.get((unit, key, sread, mode, "delta", order, base, l))
                            if e is None:
                                cells += ["-"] * 5
                                continue
                            cells += [f3(e["gap"]), f3(e["gap_related"]), fmt(e["leak"], ".4f"), fmt(e["gap_leak"], ".1f"),
                                      fmt(sr["si"], ".2f") if sr else "-"]
                        rr.append(cells)
                if any(any(c != "-" for c in r[1:]) for r in rr):
                    w(table(hdr, rr))
                    w("")
    w("Targets (unit pooled if present): dT = fact d logP(target); spec = dT - d foils; con = disposition contrast; "
      "drel* = relation (flagged)")
    unit = units[0] if units else "pooled"
    for base in args.baselines:
        for l in args.layers:
            hdr = [f"unit={unit} {base} L{l}"]
            for m in ("iso", "comb"):
                hdr += [f"{m} dT", "spec", "con", "drel*"]
            rr = []
            for key in args.keys:
                for read in reads_for(key, names):
                    cells = [f"{key}/{read}"]
                    for mode, order in (("isolated", "-"), ("combined", "orig")):
                        e = S.get((unit, key, read, tail, "delta", order, mode, base, l))
                        cells += ["-"] * 4 if e is None else [fmt(e["dt_fact"], "+.2f"), fmt(e["spec"], "+.2f"),
                                                             fmt(e["contrast"], "+.2f"), fmt(e["drel_FLAGGED"], "+.2f")]
                    rr.append(cells)
            if any(any(c != "-" for c in r[1:]) for r in rr):
                w(table(hdr, rr))
                w("")
    sk = [k for k in S if k[3] == "in"]
    if sk:
        w("Sanity condition (centred/everywhere/tail-in/all/delta; = v0.1): " + "; ".join(
            f"{k[6]} {k[7]} L{k[8]}: gap={f3(S[k]['gap'])} leak={fmt(S[k]['leak'], '.4f')}" for k in sorted(sk, key=str)))
        w("")

    w("=" * 70)
    w("5. WRITE RULE x ORDER (b), combined mode. Per-scenario gap; OD = mean_s |gap(orig) - gap(rev)| "
      "(order dependence); ret = gap / isolated gap (flagged when |iso gap| < 0.05).")
    combos = meta.get("combos", {})
    od_all, ret_all = defaultdict(list), defaultdict(list)
    per_scen = defaultdict(lambda: defaultdict(list))
    for unit in units:
        for combo in combos.get(unit, []):
            key, read = map(str, combo)
            for base in args.baselines:
                for l in args.layers:
                    cells = {}
                    for rule in ("delta", "rls"):
                        for order in ("orig", "rev"):
                            rs = by.get((unit, key, read, tail, rule, order, "combined", base, l))
                            if rs:
                                cells[(rule, order)] = {s: effects([r for r in rs if r["scenario"] == s])["gap"]
                                                       for s in meta["ids"]}
                    iso = by.get((unit, key, read, tail, "delta", "-", "isolated", base, l))
                    if len(cells) < 4 or not iso:
                        continue
                    iso_g = {s: effects([r for r in iso if r["scenario"] == s])["gap"] for s in meta["ids"]}
                    for rule in ("delta", "rls"):
                        od = mean(abs(cells[(rule, "orig")][s] - cells[(rule, "rev")][s]) for s in meta["ids"])
                        od_all[(unit, key, read, rule)].append(od)
                        for order in ("orig", "rev"):
                            ret_all[(unit, key, read, rule, order)] += [
                                cells[(rule, order)][s] / iso_g[s] for s in meta["ids"] if iso_g[s] >= 0.05]
                    per_scen[(unit, key, read)][(base, l)] = (iso_g, cells)
    if od_all:
        hdr = ["unit key/read", "OD delta", "OD rls", "rls/delta", "ret delta orig", "ret delta rev", "ret rls orig",
               "ret rls rev"]
        rr = []
        for (unit, key, read) in sorted({k[:3] for k in od_all}, key=str):
            a, b = mean(od_all[(unit, key, read, "delta")]), mean(od_all[(unit, key, read, "rls")])
            rr.append([f"{unit} {key}/{read}", f3(a), f3(b), fmt(ratio(b, a), ".2f")] +
                      [fmt(mean(ret_all[(unit, key, read, ru, o)]), "+.2f") for ru in ("delta", "rls") for o in ("orig", "rev")])
        w("OD and mean gap retention (scenarios with iso gap >= 0.05) averaged over baselines x layers:")
        w(table(hdr, rr))
        w("")
        for (unit, key, read), d in sorted(per_scen.items(), key=str):
            for (base, l), (iso_g, cells) in sorted(d.items(), key=str):
                hdr = [f"{unit} {key}/{read} {base} L{l}", "iso", "delta orig", "delta rev", "rls orig", "rls rev"]
                rr = []
                n = len(meta["ids"])
                for i, s in enumerate(meta["ids"]):
                    row = [f"{s} (#{i + 1}/#{n - i})", f3(iso_g[s])]
                    for rule in ("delta", "rls"):
                        for order in ("orig", "rev"):
                            g = cells[(rule, order)][s]
                            rt = "iso~0" if abs(iso_g[s]) < 0.05 else fmt(g / iso_g[s], "+.2f")
                            row.append(f"{f3(g)} ({rt})")
                    rr.append(row)
                w(table(hdr, rr))
                w("")
    else:
        w("(no complete write-rule x order cells in these outputs)\n")

    w("=" * 70)
    w("6. VERDICT")
    tvc = [r for r in tv if r["tokens"] == "content" and r["baseline"] == "without" and r["layer"] in args.layers]
    if tvc:
        cf, cx, r2 = mean(r["cos_fu"] for r in tvc), mean(r["cos_across"] for r in tvc), mean(r["r2_fu"] for r in tvc)
        ccf, ccx = mean(r["ccos_fu"] for r in tvc), mean(r["ccos_across"] for r in tvc)
        verdict = "SUPPORTED" if (cf >= 0.6 and r2 >= 0.5) else ("PARTLY" if cf >= 0.4 else "REFUTED")
        w(f"- One vector per experience ({verdict}): content-token deltas (without, L{args.layers}) have cos {cf:.2f} "
          f"within a follow-up vs {cx:.2f} across scenarios ({ccf:.2f} vs {ccx:.2f} after removing the global mean); "
          f"the per-follow-up mean explains {r2:.2f} of the variance.")
    for key in args.analytic_keys:
        ks_ = [r for r in kg if r["key"] == key and r["layer"] in args.layers]
        if ks_:
            w(f"- keys {key}: within {mean(r['tok_within_scen'] for r in ks_):.3f} vs between "
              f"{mean(r['tok_between'] for r in ks_):.3f} (tok), pooled {mean(r['pool_within_scen'] for r in ks_):.3f} vs "
              f"{mean(r['pool_between'] for r in ks_):.3f}; retrieval tok {mean(r['tok_retr_all'] for r in ks_):.2f} "
              f"pool {mean(r['pool_retr_all'] for r in ks_):.2f}; auc(unrel) tok {mean(r['tok_auc_unrel'] for r in ks_):.2f}")
    for unit in units:
        for mode, order in (("isolated", "-"), ("combined", "orig")):
            ref = [S.get((unit, *REF, tail, "delta", order, mode, b, l)) for b in args.baselines for l in args.layers]
            ref_si = mean(SEL[(unit, *REF, mode, "delta", order, b, l)]["si"] for b in args.baselines for l in args.layers
                          if (unit, *REF, mode, "delta", order, b, l) in SEL)
            ref_gr = mean(e["gap_related"] for e in ref if e)
            ref_rec = mean(SEL[(unit, *REF, mode, "delta", order, b, l)]["rfnh_related"] for b in args.baselines
                           for l in args.layers if (unit, *REF, mode, "delta", order, b, l) in SEL)
            cands = []
            for key in args.keys:
                for read in reads_for(key, names):
                    es = [S.get((unit, key, read, tail, "delta", order, mode, b, l)) for b in args.baselines for l in args.layers]
                    if not any(es):
                        continue
                    sread = read if read_spec(read)[1] != "stoch" else f"soft_t{args.stoch_t:g}"
                    srs = [SEL.get((unit, key, sread, mode, "delta", order, b, l)) for b in args.baselines for l in args.layers]
                    si = mean(s["si"] for s in srs if s)
                    rec = mean(s["rfnh_related"] for s in srs if s)
                    gr = mean(e["gap_related"] for e in es if e)
                    lk = mean(e["leak"] for e in es if e)
                    gl = mean(e["gap_leak"] for e in es if e)
                    ok = rec >= args.recall_keep * ref_rec and gr >= args.recall_keep * ref_gr
                    cands.append((si, key, read, rec, gr, lk, gl, ok))
            cands.sort(key=lambda c: -c[0] if not math.isnan(c[0]) else 0)
            good = [c for c in cands if c[7]]
            w(f"- unit={unit} {mode}: current centred/everywhere SI={ref_si:.2f}, related gap={ref_gr:.3f}. "
              f"Best SI keeping recall (rfnh and gapR >= {args.recall_keep} x current):")
            for si, key, read, rec, gr, lk, gl, _ in good[:3]:
                w(f"    {key}/{read}: SI={si:.2f} rfnh_rel={rec:.3f} gapR={gr:.3f} leak={lk:.4f} g/l={gl:.1f}")
            if not good:
                w("    none keeps recall")
            top = cands[0] if cands else None
            if top and not top[7]:
                w(f"    (highest SI overall: {top[1]}/{top[2]} SI={top[0]:.2f}, but it loses recall: rfnh_rel={top[3]:.3f}, "
                  f"gapR={top[4]:.3f})")
            bgl = max(cands, key=lambda c: c[6] if not math.isnan(c[6]) else -1e9) if cands else None
            if bgl:
                w(f"    best gap/leak: {bgl[1]}/{bgl[2]} g/l={bgl[6]:.1f} (gapR={bgl[4]:.3f}, leak={bgl[5]:.4f})")
    if od_all:
        d_ = mean(v for k, vs in od_all.items() if k[3] == "delta" for v in vs)
        r_ = mean(v for k, vs in od_all.items() if k[3] == "rls" for v in vs)
        v = "YES" if r_ <= 0.5 * d_ else ("PARTLY" if r_ < 0.8 * d_ else "NO")
        w(f"- Does RLS remove the order dependence? {v}: mean OD (|gap orig - gap rev| per scenario) delta {d_:.3f} "
          f"vs rls {r_:.3f} (ratio {ratio(r_, d_):.2f}). RLS equals the batch ridge solution (checked), so OD ~ 0 is "
          f"by construction; whether it keeps the memories is the retention: delta orig/rev "
          f"{mean(v for k, vs in ret_all.items() if k[3] == 'delta' and k[4] == 'orig' for v in vs):+.2f}/"
          f"{mean(v for k, vs in ret_all.items() if k[3] == 'delta' and k[4] == 'rev' for v in vs):+.2f}, rls "
          f"{mean(v for k, vs in ret_all.items() if k[3] == 'rls' for v in vs):+.2f} (1 = as isolated).")
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------------- I/O


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def read_csv(path):
    def conv(v):
        for t in (int, float):
            try:
                return t(v)
            except ValueError:
                pass
        return v
    return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(open(path))]


def finish(out_dir, rows, sel, tv, kg, meta):
    write_csv(out_dir / "summary.csv", summarize(rows))
    (out_dir / "report.txt").write_text(report(rows, sel, tv, kg, meta))


def merge(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, seen, meta = [], set(), None
    for d in map(Path, args.merge):
        m = json.load(open(d / "config.json"))
        if meta is None:
            meta = m
            sel, tv, kg = (read_csv(d / n) for n in ("selectivity.csv", "token_vectors.csv", "key_geometry.csv"))
        else:
            meta["parts"].update(m["parts"])
            for u, c in m.get("combos", {}).items():
                meta.setdefault("combos", {}).setdefault(u, c)
        for line in open(d / "results.jsonl"):
            r = json.loads(line)
            key = ck(r) + (r["seed"], r["memory"], r["kind"], r.get("scenario"), r.get("probe"))
            if key not in seen:
                seen.add(key)
                rows.append(r)
    meta["combos"] = {u: [tuple(c) for c in cs] for u, cs in meta.get("combos", {}).items()}
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for n, data in (("selectivity.csv", sel), ("token_vectors.csv", tv), ("key_geometry.csv", kg)):
        write_csv(out_dir / n, data)
    json.dump({**meta, "merged_from": args.merge}, open(out_dir / "config.json", "w"), indent=2)
    finish(out_dir, rows, sel, tv, kg, meta)
    log(f"merged {len(rows)} rows from {args.merge} into {out_dir}")


def main():
    args = parse_args()
    if args.merge:
        return merge(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    units = {"1": ["pooled"], "2": ["all"], "all": ["pooled", "all"]}[args.part]
    chk = defaultdict(float)
    chk.update(exact_recall_n=0, exact_recall_max=0.0, rls_n=0, rls_batch_max_rel=0.0, lib_equal_n=0,
               reader_vs_inject_max=0.0)
    t0 = time.time()
    with torch.inference_mode():
        C = setup(args)
        log("token vectors + key geometry")
        tv = token_vectors(C)
        kg = key_geometry(C)
        log("analytic selectivity (full factorial)")
        sel, thr_cache = analytic(C, chk)
        combos, combo_table = choose_combos(sel, C)
        log(f"(b) combos: {combos}")
        rows, repro = [], {"status": "SKIPPED", "reason": "--no-sanity"}
        if not args.no_sanity:
            t1 = time.time()
            rows, repro = sanity(C, chk)
            log(f"sanity took {(time.time() - t1) / 60:.1f} min")
        rows += behaviour(C, units, combos, thr_cache, chk)

    part = f"part{args.part}"
    meta = {"args": vars(args), "ids": C.ids, "units": units, "mu_skip_tokens": C.skip, "tail_len": C.tail_len,
            "white_info": C.white_info, "combos": {u: [list(c) for c in cs] for u, cs in combos.items()},
            "combo_table": {u: [[list(c[0]), c[1], c[2], c[3]] for c in cs] for u, cs in combo_table.items()},
            "parts": {part: {"repro": repro, "checks": dict(chk), "minutes": (time.time() - t0) / 60}},
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for n, data in (("selectivity.csv", sel), ("token_vectors.csv", tv), ("key_geometry.csv", kg)):
        write_csv(out_dir / n, data)
    json.dump(meta, open(out_dir / "config.json", "w"), indent=2)
    meta["combos"] = combos
    finish(out_dir, rows, sel, tv, kg, meta)
    log(f"wrote {len(rows)} rows to {out_dir} in {(time.time() - t0) / 60:.1f} min")
    assert repro["status"] != "FAIL", f"sanity condition does not reproduce v0.1: {repro}"


if __name__ == "__main__":
    main()
