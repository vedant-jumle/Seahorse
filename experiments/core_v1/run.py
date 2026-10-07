#!/usr/bin/env python
"""Seahorse core_v1: one consolidated, reproducible run of the core claims on one model.

Design (identical for every model; PREREG.md states the claims and decision rules):
  model      text path only, thinking OFF everywhere (enable_thinking=False is forced on the tokenizer: the
             9B template defaults to thinking ON); fp32
  layers     3 read/inject layers (block outputs): 2B [20, 21, 23]; other models by the relative-depth rule
             (core.map_layers, asserted against the config)
  write      one entropy-weighted pooled moment per follow-up (3), chat-template tail excluded; isolated
             memory per item, delta rule. Facts: plain shift (with - without). Preferences: opposite
             (with - counter), the design reference. Also stored: without (= plain), centroid (with - mean
             of 8 same-frame alternatives), placebo (the placebo experience's plain shift); each reference
             over its own shared suffix with the `with` run.
  key/read   pooled_w256 (PCA-256 whitening on the generic prompts, per model and layer); hard threshold at
             the q0.95 match on generic prompts + continuations, per item and layer; never on the template
             head; TWO-PASS gating (a memory-free pass 1 gives keys, gates and recalls; pass 2 injects), so
             the injection never changes its own gate; h <- h + alpha * gate * M q at each of the 3 layers
  doses      mem@{0.5,1,2,3} on the raw shift (no rescaling); controls at alpha 2 (core.py: rand, swap,
             placebo, gate_on); mechanism items: centroid / without @ {0.5,1,2} (opposite = mem)

Stages (each a separate job; outputs under --out = <run>/<model name>/<stage>):
  smoke  the whole pipeline on a tiny subset with the real model + numerics checks (no-memory greedy answers
         finite and coherent) and, with --judge-smoke, the judge
  prep   calibration (mu, whitening, thresholds), writes, memories, the logit lens of every stored shift at
         EVERY block, the fact analytic (log P target / specificity at the measure prefix per condition), the
         checks (exact recall, write repro vs diag_keys.unit_writes, cached == full-sequence two-pass reader,
         two-pass == one-pass for one layer, core fields == xlayer_v1 fields, batched == unbatched
         generation, and for the 2B a repro of xlayer_v1's fact shifts and thresholds) -> state.pt
  gen    generation (reads prep/state.pt): per item, the main prompts (1 greedy + 4 samples, + 4 seed-set-2
         samples for nomem, ctx, mem@1, mem@2; every condition of a prompt in one batch with the same
         random streams), the balanced yes/no probes (greedy, first-token margins), and the 50 unrelated
         prompts (greedy + 2 samples; gated conditions are not regenerated when the item's gate is shut at
         every position; gate_on always is) + KL(nomem || memory) along the nomem greedy answer
  score  coherence (mean token NLL of every answer under the no-memory model) and the LLM judge
         (judge_prompts.yaml; Qwen3.5-9B bf16, thinking off)
analyze.py turns the saved files into tables without a GPU.
"""

import argparse
import copy
import gc
import gzip
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
import zlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
import yaml

from seahorse import metrics as mx
from seahorse.residual import capture, load_model, text_model
from seahorse.sessions import ceiling_ids, chat_ids, common_suffix_len, text_ids

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


core = _load("seahorse_core_v1_core", HERE / "core.py")
itm = _load("seahorse_core_v1_items", HERE / "items.py")
xr = _load("seahorse_xlayer_v1_run", HERE.parent / "xlayer_v1" / "run.py")
tv1, sh, xl, dk, v01 = xr.tv1, xr.sh, xr.xl, xr.dk, xr.v01
log = dk.log

W256 = "pooled_w256"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
TAG = "core_v1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True, choices=["smoke", "prep", "gen", "score"])
    p.add_argument("--out", required=True, help="the model's run dir; each stage writes <out>/<stage>")
    p.add_argument("--items", nargs="+", default=None, help="gen: only these items (default all)")
    p.add_argument("--sections", nargs="+", default=["main", "yn", "unrel"], choices=["main", "yn", "unrel"])
    p.add_argument("--no-judge", action="store_true", help="score: coherence only")
    p.add_argument("--judge-smoke", action="store_true", help="smoke: also load and run the judge")
    p.add_argument("--random-model", default=None, choices=["qwen2"],
                   help="tiny random model with the config model's tokenizer (CPU dry runs)")
    p.add_argument("--tiny", action="store_true", help="tiny settings (implied by smoke)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def load_cfg(path):
    cfg = yaml.safe_load(open(path))
    for k in ("items", "bench", "generic"):
        cfg[k] = str((REPO / cfg[k]).resolve()) if not os.path.isabs(cfg[k]) else cfg[k]
    return cfg


def tiny_cfg(cfg):
    c = copy.deepcopy(cfg)
    c.update(samples=2, answer_cap=24, yn_cap=4, unrel_samples=1, max_batch=48)
    c["n_unrel"], c["n_prompts"], c["tiny"] = 3, 2, True
    return c


# ------------------------------------------------------------------------- model


def thinking_off(tok):
    """Force enable_thinking=False in every chat-template call (sessions.chat_ids, diag_keys, v0_1, ...)."""
    orig = tok.apply_chat_template

    def act(conversation, *a, **kw):
        kw.setdefault("enable_thinking", False)
        return orig(conversation, *a, **kw)
    tok.apply_chat_template = act
    s = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    assert s.endswith("<think>\n\n</think>\n\n"), f"thinking not off: {s[-40:]!r}"
    return tok


def load(cfg, A, judge=False):
    name = cfg["judge"]["model"] if judge else cfg["model"]
    dtype = DTYPES[cfg["judge"]["dtype"] if judge else cfg["dtype"]]
    t0 = time.time()
    if A.random_model:
        model, tok = tv1.load_any(SimpleNamespace(tiny=A.random_model, model=cfg["model"], device=A.device))
    else:
        model, tok = load_model(name, device=A.device, dtype=dtype)
    log(f"loaded {name} ({'random ' + A.random_model if A.random_model else dtype}) in {time.time() - t0:.0f}s")
    return model, thinking_off(tok)


def release():
    """Return freed GPU memory (call after the last reference to a model went out of scope)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def context(A, cfg, model, tok):
    C = SimpleNamespace(A=A, cfg=cfg, model=model, tok=tok, dev=A.device, chk=defaultdict(float))
    C.body, C.lm_head = text_model(model), model.lm_head
    C.vocab = C.lm_head.out_features
    C.d = C.body.norm.weight.shape[0]
    C.n_blocks = len(C.body.layers)
    C.ltype = core.layer_types(model.config)
    C.layers = list(cfg["layers"])
    want = core.map_layers(cfg["reference_layers"], 24, C.n_blocks)
    assert C.layers == want, f"config layers {C.layers} != relative-depth rule {want}"
    eos = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            eos.add(i)
    ge = model.generation_config.eos_token_id
    eos |= set(ge if isinstance(ge, (list, tuple)) else [ge])
    C.eos_list = sorted(e for e in eos if e is not None)
    C.eos = torch.tensor(C.eos_list, device=C.dev)
    C.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    C.think_end, C.phase = -1, SimpleNamespace(a_prompt=0.0, alpha=None)  # tv1.generate (checks only)
    C.yes, C.no = text_ids(tok, "Yes").tolist(), text_ids(tok, "No").tolist()
    assert len(C.yes) == 1 and len(C.no) == 1, "Yes / No must be single tokens"
    C.yes, C.no = C.yes[0], C.no[0]
    s = cfg["sampler"]
    C.sampler = core.qwen_sampler(s["temp"], s["top_k"], s["top_p"], s["presence"])
    C.ratio = lambda W, R: 1.0  # xlayer_v1 checks: read layer = injection layer
    C.args = SimpleNamespace(check_tol=cfg["check_tol"])  # xlayer_v1 checks read C.args.check_tol
    return C


def env_meta(C):
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
        except Exception:
            return None
    gpu = None
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        gpu = {"name": pr.name, "memory_gb": round(pr.total_memory / 2 ** 30, 1)}
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "torch": torch.__version__, "transformers": transformers.__version__, "gpu": gpu,
            "cuda": torch.version.cuda, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "linear_attention_kernels": tv1._kernels(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename}


def write_jsonl_gz(path, rows):
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------- calibration


def calibrate(C, L, out):
    """diag_keys.setup at the read layers (mu, PCA-256 whitening, generic prompts + continuations), the
    generic keys for the thresholds, and the median residual norm of every block (logit-lens doses)."""
    cfg, A = C.cfg, C.A
    path = out / "dk_input.json"
    json.dump({"scenarios": [], "unrelated_probes": L["unrelated"]}, open(path, "w"), indent=1)
    generic = cfg["generic"]
    if cfg.get("tiny"):
        generic = out / "generic_tiny.txt"
        generic.write_text("\n".join([l for l in open(cfg["generic"]) if l.strip()][:16]))
    k = cfg["key"]
    dargs = SimpleNamespace(model=cfg["model"], layers=C.layers, analytic_layers=C.layers, scenarios=str(path),
                            generic=str(generic), cont_len=k["cont_len"], gen_cont_len=k["gen_cont_len"],
                            white_eps=k["white_eps"], tiny=bool(A.random_model), device=C.dev)
    orig_load = dk.load
    dk.load = lambda _a: (C.model, C.tok)
    dk.KEY_SPECS = {W256: (("pca", 256), True)}
    try:
        D = dk.setup(dargs)
    finally:
        dk.load = orig_load  # no module-level reference to the model survives the stage
    C.D = D
    C.mu, C.KS, C.tail_len, C.head, C.catfn, C.stash = D.mu, D.KS, D.tail_len, D.head, D.catfn, D.stash
    C.Kg = {l: dk.gen_keys(D, l, D.KS[l][W256]) for l in C.layers}
    C.med = {l: xl.median_norm([c[l][c["cats"] != 0] for c in D.gen]) for l in C.layers}
    gp = [l.strip() for l in open(generic) if l.strip()][:24]
    allL = list(range(C.n_blocks))
    norms = defaultdict(list)
    for t in gp:
        ids = chat_ids(C.tok, t)
        st = pass1(C, ids, allL)
        cats = C.catfn(ids)
        for l in allL:
            norms[l].append(st[l][cats != 0])
    C.med_all = {l: xl.median_norm(norms[l]) for l in allL}
    C.template = {"head": C.tok.decode(C.head), "tail": C.tok.decode(chat_ids(C.tok, "a")[-C.tail_len:]),
                  "eos": C.eos_list}
    log(f"template {json.dumps(C.template)}; median |h| at read layers "
        + ", ".join(f"L{l} ({C.ltype[l]}) {C.med[l]:.1f}" for l in C.layers))


def restore(C, state):
    """gen / score: the calibration saved by prep (no recomputation)."""
    C.tail_len, C.head = state["tail_len"], state["head"]
    tail = chat_ids(C.tok, "a")[-C.tail_len:]
    assert torch.equal(chat_ids(C.tok, "b")[-C.tail_len:], tail)
    C.catfn = dk.make_catfn(C.head, tail, C.dev)
    dk.KEY_SPECS = {W256: (("pca", 256), True)}
    C.mu = {l: state["mu"][l].to(C.dev) for l in C.layers}
    C.KS = {l: {W256: dk.KeySpace(W256, C.mu[l], state["W"][l].to(C.dev))} for l in C.layers}
    C.thr, C.sd = state["thr"], state["sd"]


def pass1(C, ids, layers):
    """Memory-free forward: residuals [T, d] at `layers`."""
    with capture(C.model, layers) as st:
        C.body(input_ids=ids[None].to(C.dev))
    return {l: st[l] for l in layers}


# ------------------------------------------------------------------------ writes


def run_states(C, ids, layers, logits=False):
    with capture(C.model, layers) as st:
        o = C.model(ids[None].to(C.dev))
    return {l: st[l] for l in layers}, (o.logits[0].float() if logits else None)


def build_moments(C, it, placebo, check=False):
    """Per follow-up: shifts at EVERY block for each reference and the write key at the read layers.
    plain = with - without (facts' design; preferences' `without`), opposite = with - counter (preferences'
    design), centroid = with - mean over the 8 alternatives, placebo = (placebo + follow-up) - without.
    Each reference uses the shared suffix of `with` and its own runs (n <= n_without), entropy-weighted
    over the without run (ref_v1.shifts), template tail excluded."""
    tok, allL = C.tok, list(range(C.n_blocks))
    pref = it["group"] != "facts"
    moments = []
    for j, fu in enumerate(it["followups"]):
        s_wo, s_with = chat_ids(tok, fu), chat_ids(tok, f"{it['experience']} {fu}")
        refs = {"placebo": [chat_ids(tok, f"{placebo} {fu}")]}
        if pref:
            refs["opposite"] = [chat_ids(tok, f"{it['counter']} {fu}")]
            refs["centroid"] = [chat_ids(tok, f"{a} {fu}") for a in it["centroid"]]
        n_wo = common_suffix_len(s_with, s_wo)
        st_wo, lg_wo = run_states(C, s_wo, allL, logits=True)
        st_w, _ = run_states(C, s_with, allL)

        def pooled(st, n):
            w = sh.moment_weights(mx.entropy(lg_wo[-n:]), C.tail_len)
            return {l: sh.pool(st[l][-n:], w) for l in allL}
        shifts = {}
        pw, pwo = pooled(st_w, n_wo), pooled(st_wo, n_wo)
        shifts["plain"] = {l: sh.reference_shift(pw[l], pwo[l]) for l in allL}
        ns = {"plain": n_wo}
        for r, runs in refs.items():
            n = min([n_wo] + [common_suffix_len(s_with, s) for s in runs])
            ns[r] = n
            C.chk["moments_" + r] += 1
            C.chk["n_below_without_" + r] += int(n != n_wo)
            pw_r = pooled(st_w, n) if n != n_wo else pw
            if r == "placebo":
                base = pooled(st_wo, n) if n != n_wo else pwo
                pr = pooled(run_states(C, runs[0], allL)[0], n)
                shifts[r] = {l: pr[l] - base[l] for l in allL}
            else:
                pr = [pooled(run_states(C, s, allL)[0], n) for s in runs]
                shifts[r] = {l: sh.reference_shift(pw_r[l], [p[l] for p in pr]) for l in allL}
        if pref:
            shifts["without"] = shifts["plain"]
        cats = C.catfn(s_wo)
        m = {"j": j, "followup": fu, "n": ns, "shifts": shifts,
             "key": {l: C.KS[l][W256].keys(st_wo[l], cats == 1)[-1:] for l in C.layers}}
        for r, s in shifts.items():
            for l in C.layers:
                assert bool(torch.isfinite(s[l]).all()), f"{it['id']} {r} L{l}: non-finite shift"
        if check:  # the plain write must equal diag_keys.unit_writes (think_v1 / xlayer_v1's fact write)
            ch = v01.collect_writes(C.model, tok, {"experience": it["experience"], "counter": it["counter"],
                                                   "followups": [fu]}, C.layers, C.dev)[0]
            full = {**{l: st_wo[l] for l in C.layers}, "cats": cats}
            for l in C.layers:
                d_, k_, _ = dk.unit_writes(ch, full, l, "without", "pooled", C.KS[l][W256], False, C.tail_len)
                rel = ((d_[0] - shifts["plain"][l]).norm() / d_[0].norm().clamp_min(1e-12)).item()
                kab = (k_ - m["key"][l]).norm().item()
                C.chk["repro_write_n"] += 1
                C.chk["repro_write_shift_max_rel"] = max(C.chk["repro_write_shift_max_rel"], rel)
                C.chk["repro_write_key_max_abs"] = max(C.chk["repro_write_key_max_abs"], kab)
                if ch["n"] == n_wo:
                    assert rel < 1e-3 and kab < 1e-3, f"{it['id']} L{l}: != diag_keys.unit_writes ({rel}, {kab})"
        moments.append(m)
    return moments


def design_ref(it):
    return "plain" if it["group"] == "facts" else "opposite"


def item_memories(C, it, moments_of, by_id):
    """{layer: {name: diag_keys Mem}}: design, placebo, swap (the partner's design shifts under this item's
    keys) and, for preferences, centroid and without. Delta rule over the 3 moments; all share the keys."""
    M = {}
    own = moments_of[it["id"]]
    partner = moments_of[it["swap"]]
    pd = design_ref(by_id[it["swap"]])
    lam = C.cfg["key"]["rls_lambda"]
    for l in C.layers:
        keys = torch.cat([m["key"][l] for m in own]).to(C.dev)
        src = {"design": [m["shifts"][design_ref(it)][l] for m in own],
               "placebo": [m["shifts"]["placebo"][l] for m in own],
               "swap": [m["shifts"][pd][l] for m in partner]}
        if it["group"] != "facts":
            src["centroid"] = [m["shifts"]["centroid"][l] for m in own]
            src["without"] = [m["shifts"]["without"][l] for m in own]
        M[l] = {}
        for name, sl in src.items():
            S = torch.stack([s.to(C.dev).float() for s in sl])
            M[l][name] = dk.build({it["id"]: core.memory_chunks(S, keys)}, [it["id"]], "delta", lam, C.dev, C.chk)
            assert torch.equal(M[l][name].Kst, M[l]["design"].Kst)
            assert bool(torch.isfinite(M[l][name].M).all()), f"{it['id']} L{l} {name}: non-finite M"
    return M


# ---------------------------------------------------------------- two-pass read


def read_prompt(C, it, M, ids, st=None):
    cats = C.catfn(ids)
    st = st if st is not None else pass1(C, ids, C.layers)
    R = {}
    for l in C.layers:
        k = C.KS[l][W256].keys(st[l], cats == 1)
        match, g = core.read_layer(k, M[l]["design"].Kst, C.thr[it["id"]][l], cats)
        R[l] = SimpleNamespace(k=k, match=match, g=g, recall={})
    return SimpleNamespace(R=R, cats=cats, T=len(ids))


def fields_for(C, it, M, P, c):
    if c.kind in ("nomem", "ctx"):
        return None
    F = {}
    for l in C.layers:
        Rl = P.R[l]
        if c.mem not in Rl.recall:
            Rl.recall[c.mem] = Rl.k @ M[l][c.mem].M.T
        F[l] = core.cond_field(c.kind, c.alpha, Rl.g, P.cats, Rl.recall[c.mem],
                               C.rand_dir[it["id"]][l] if c.kind == "rand" else None)
    return F


def gate_info(C, P):
    nh = P.cats != 0
    out = {}
    for l in C.layers:
        Rl = P.R[l]
        out[str(l)] = {"match_last": round(Rl.match[-1].item(), 4), "thr": round(C.thr_now[l], 4),
                       "g_last": Rl.g[-1].item(), "open_frac": round(Rl.g[nh].mean().item(), 4),
                       "match_max": round(Rl.match[nh].max().item(), 4)}
    return {"layers": out, "fire": sum(P.R[l].g[-1].item() for l in C.layers) / len(C.layers),
            "open_any": any(bool(P.R[l].g.any()) for l in C.layers)}


# ============================================================================ prep


def lens_logits(C, X):
    """[n, d] -> logit lens [n, V] (final norm + output layer)."""
    return C.lm_head(C.body.norm(X.float().to(C.dev)[:, None]))[:, 0].float()


def tokstr(C, i):
    return C.tok.decode([i]).replace(" ", "▁").replace("\n", "\\n")


def lens_rows(C, it, moments, topn=10):
    """Logit lens of the stored shift (mean over the 3 moments) at every block: facts target rank / z /
    margin vs the foils; preferences lex_gain per reference."""
    allL = list(range(C.n_blocks))
    rows = []
    enc = lambda s: C.tok(s, add_special_tokens=False).input_ids
    if it["group"] == "facts":
        meas = it["measure"]
        t0, f0 = enc(meas["target"])[0], [enc(f)[0] for f in meas["foils"]]
        refs = ["plain"]
    else:
        cons, inc = sh.lexicon_token_ids(enc, it["lexicon"])
        refs = ["opposite", "without", "centroid", "placebo"]
    for r in refs:
        X = torch.stack([m["shifts"][r][l] for l in allL for m in moments])  # [L * 3, d], layer-major
        ll = lens_logits(C, X).view(len(allL), len(moments), -1).mean(1)
        for li, l in enumerate(allL):
            v = ll[li]
            nrm = sum(m["shifts"][r][l].norm().item() for m in moments) / len(moments)
            o = {"sec": "lens", "item": it["id"], "group": it["group"], "ref": r, "layer": l, "ltype": C.ltype[l],
                 "depth": (l + 1) / C.n_blocks, "shift_norm": nrm, "rel_dose": nrm / C.med_all[l],
                 "top": [(tokstr(C, i), round(v[i].item(), 2)) for i in v.topk(topn).indices.tolist()]}
            if it["group"] == "facts":
                o.update(rank_t=int((v > v[t0]).sum().item()) + 1, z_t=((v[t0] - v.mean()) / v.std()).item(),
                         margin=(v[t0] - v[f0].mean()).item())
            else:
                o["lex_gain"] = sh.lex_gain(v, cons, inc)
            rows.append(o)
    return rows


def facts_analytic(C, it, M, conds):
    """Teacher-forced at the measure prefix ("Your dog's name is" -> " Pepper" vs the foils), related and
    paraphrase probes, per condition (two-pass fields; ctx = the fact in the prompt)."""
    tok, meas = C.tok, it["measure"]
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    cands = [enc(meas["target"])] + [enc(f) for f in meas["foils"]]
    pre = text_ids(tok, meas["prefix"])
    rows = []
    P, ids, ceil = {}, {}, {}
    for kind in ("related", "paraphrase"):
        ids[kind] = torch.cat([chat_ids(tok, it[kind]), pre])
        ceil[kind] = torch.cat([ceiling_ids(tok, it["experience"], it[kind]), pre])
        P[kind] = read_prompt(C, it, M, ids[kind])
    for c in conds:
        kinds = ("related", "paraphrase")
        if c.kind == "ctx":
            mm = xr.eval_measure(C, [ceil[k] for k in kinds], cands, [None, None])
        else:
            mm = xr.eval_measure(C, [ids[k] for k in kinds], cands, [fields_for(C, it, M, P[k], c) for k in kinds])
        for k, m in zip(kinds, mm):
            rows.append({"sec": "fact_tf", "item": it["id"], "cond": c.cid, "ckind": c.kind, "alpha": c.alpha,
                         "kind": k, **m, "fire": None if c.kind in ("nomem", "ctx") else
                         sum(P[k].R[l].g[-1].item() for l in C.layers) / len(C.layers)})
    return rows


def checks_two_pass(C, it, M):
    """xlayer_v1's checks on this design: cached KV generation == the full-sequence two-pass reader, and
    two-pass == the one-pass diag_keys.Reader for one layer; core_v1's fields == xlayer_v1's."""
    Mx = {l: SimpleNamespace(mem=M[l]["design"], thr=C.thr[it["id"]][l], sd=C.sd[it["id"]][l]) for l in C.layers}
    specs = [(l, l, 2.0) for l in C.layers]
    res = {"cached": xr.check_cached(C, it, Mx, specs, "mem@2"),
           "onepass_one_layer": xr.check_onepass(C, it, Mx, [(C.layers[-1], C.layers[-1], 2.0)],
                                                 f"L{C.layers[-1]}@2", strict=True)}
    ids = chat_ids(C.tok, it["related"])
    st = pass1(C, ids, C.layers)
    Fx, _ = xr.fields_for(C, xr.read_units(C, Mx, ids, st, C.layers), specs)
    P = read_prompt(C, it, M, ids, st)
    F = fields_for(C, it, M, P, core.Cond("mem@2", "mem", "design", 2.0))
    res["fields_vs_xlayer_max_abs"] = max((F[l] - Fx[l]).abs().max().item() for l in C.layers)
    assert res["fields_vs_xlayer_max_abs"] < 1e-4, res["fields_vs_xlayer_max_abs"]
    rnd = fields_for(C, it, M, P, core.Cond("rand@2", "rand", "design", 2.0))
    res["rand_norm_max_rel"] = max(((rnd[l].norm(dim=-1) - F[l].norm(dim=-1)).abs()
                                    / F[l].norm(dim=-1).clamp_min(1e-6)).max().item() for l in C.layers)
    assert res["rand_norm_max_rel"] < 1e-3
    return res


def check_batching(C, it, M, conds, cap):
    """Every condition of a prompt in one batch == each condition generated alone (core.generate_batch),
    and one condition alone == think_v1.generate with xlayer_v1's XInject (the old generator)."""
    cfg = C.cfg
    text = it["related"]
    ids = chat_ids(C.tok, text)
    P = read_prompt(C, it, M, ids)
    S = cfg["samples"]
    streams = {1: (core.seed_of(TAG, text, cfg["seeds"]["base"]), S + 1)}
    cs = [c for c in conds if c.kind != "ctx"]
    specs = [(c, 1, j) for c in cs for j in range(S + 1)]
    Fc = {c.cid: fields_for(C, it, M, P, c) for c in cs}
    fields = core.stack_fields([Fc[c.cid] for c, _, _ in specs], C.layers, P.T, C.d, C.dev)
    rows = [(j == 0, None if j == 0 else (1, j)) for _, _, j in specs]
    batched = core.generate_batch(C.model, C.body, C.lm_head, ids, rows, max_new=cap, eos=C.eos_list, vocab=C.vocab,
                                  sampler=C.sampler, streams=streams, fields=fields)
    same, n = 0, 0
    first_diff = []
    for c in cs:
        one = core.generate_batch(C.model, C.body, C.lm_head, ids, rows[:S + 1], max_new=cap, eos=C.eos_list,
                                  vocab=C.vocab, sampler=C.sampler, streams=streams,
                                  fields=core.stack_fields([Fc[c.cid]] * (S + 1), C.layers, P.T, C.d, C.dev))
        for j in range(S + 1):
            b = batched[[k for k, (cc, _, jj) in enumerate(specs) if cc.cid == c.cid and jj == j][0]]
            n += 1
            same += b["ids"] == one[j]["ids"]
            if b["ids"] != one[j]["ids"]:
                k = next((i for i, (x, y) in enumerate(zip(b["ids"], one[j]["ids"])) if x != y),
                         min(len(b["ids"]), len(one[j]["ids"])))
                first_diff.append({"cond": c.cid, "j": j, "first_diff_token": k})
    # one condition: generate_batch == think_v1.generate + XInject (identical streams: B = 1 + S, row 0 greedy)
    c = next(x for x in cs if x.cid == "mem@2")
    F = Fc[c.cid]
    info = [{"W": l, "gate": P.R[l].g[-1].item()} for l in C.layers]
    old = tv1.generate(C, ids, S + 1, [xl.XInject(l, F[l], info[i]) for i, l in enumerate(C.layers)], a_think=0.0,
                       a_answer=1.0, think=False, max_new=cap, sampler=tv1.qwen_sampler(
                           cfg["sampler"]["temp"], cfg["sampler"]["top_k"], cfg["sampler"]["top_p"],
                           cfg["sampler"]["presence"]), seed=streams[1][0], greedy0=True, answer_cap=cap)
    new = core.generate_batch(C.model, C.body, C.lm_head, ids, rows[:S + 1], max_new=cap, eos=C.eos_list,
                              vocab=C.vocab, sampler=C.sampler, streams=streams,
                              fields=core.stack_fields([F] * (S + 1), C.layers, P.T, C.d, C.dev))
    old_same = sum(o["ans_ids"] == nw["ids"] for o, nw in zip(old.rows, new))
    res = {"item": it["id"], "prompt": text, "cap": cap, "rows": n, "batched_eq_unbatched": same / n,
           "first_diffs": first_diff[:10], "eq_think_v1_generate": old_same / (S + 1)}
    log(f"batching check: {json.dumps(res)}")
    return res


def repro_xlayer(C, lens_by_item, path):
    """2B: the fact shifts (mean norm, logit-lens z and margin at R in {12, 16, 20, 23}) and the thresholds
    at the read layers must reproduce xlayer_v1 Stage A (same model, same write, same calibration)."""
    if not path or not Path(path, "results.jsonl").exists():
        return {"skipped": f"no xlayer_v1 Stage A at {path}"}
    ref = [json.loads(l) for l in open(Path(path, "results.jsonl"))]
    refl = {(r["item"], r["R"]): r for r in ref if r.get("section") == "lens"}
    thr_ref = json.load(open(Path(path, "config.json")))["thresholds"]
    res = {"n_lens": 0, "n_thr": 0, "shift_norm_max_rel": 0.0, "z_max_abs": 0.0, "margin_max_abs": 0.0,
           "thr_max_abs": 0.0}
    for (iid, R), r in refl.items():
        mine = next((x for x in lens_by_item.get(iid, []) if x["ref"] == "plain" and x["layer"] == R), None)
        if mine is None:
            continue
        res["n_lens"] += 1
        res["shift_norm_max_rel"] = max(res["shift_norm_max_rel"], abs(mine["shift_norm"] - r["shift_norm"]) / r["shift_norm"])
        res["z_max_abs"] = max(res["z_max_abs"], abs(mine["z_t"] - r["z_t"]))
        res["margin_max_abs"] = max(res["margin_max_abs"], abs(mine["margin"] - r["margin"]))
    for key, v in thr_ref.items():
        iid, l = key.split("/L")
        if iid in C.thr and int(l) in C.thr[iid]:
            res["n_thr"] += 1
            res["thr_max_abs"] = max(res["thr_max_abs"], abs(C.thr[iid][int(l)] - v))
    log(f"repro vs xlayer_v1: {json.dumps(res)}")
    assert res["n_lens"] > 0 and res["n_thr"] > 0, "repro: nothing to compare"
    assert res["shift_norm_max_rel"] < 1e-3 and res["thr_max_abs"] < 1e-3, f"repro vs xlayer_v1 failed: {res}"
    return res


def prep(C, L, out):
    """Calibration, writes, memories, logit lens, fact analytic and checks -> out/state.pt etc."""
    cfg = C.cfg
    out.mkdir(parents=True, exist_ok=True)
    json.dump({it["id"]: it for it in L["items"]}, open(out / "items.json", "w"), indent=1)
    t0 = time.time()
    calibrate(C, L, out)
    items, by_id = L["items"], L["by_id"]
    need = {it["id"] for it in items} | {it["swap"] for it in items}
    moments_of, lens_all = {}, {}
    first_fact = next((it["id"] for it in items if it["group"] == "facts"), None)
    for iid in [i for i in L["all_ids"] if i in need]:
        it = by_id[iid]
        moments_of[iid] = build_moments(C, it, L["placebo"], check=(iid == first_fact))
        lens_all[iid] = lens_rows(C, it, moments_of[iid])
    log(f"writes + lens done ({len(moments_of)} items) in {(time.time() - t0) / 60:.1f} min")
    C.thr, C.sd = defaultdict(dict), defaultdict(dict)
    C.rand_dir = {}
    Ms = {}
    for it in items:
        M = item_memories(C, it, moments_of, by_id)
        for l in C.layers:
            C.thr[it["id"]][l], C.sd[it["id"]][l] = dk.calib(M[l]["design"], C.Kg[l], cfg["key"]["thr_q"])
        C.rand_dir[it["id"]] = {l: core.rand_direction(C.d, cfg["seeds"]["rand"] + 1000 * l + zlib.crc32(
            it["id"].encode()) % 100000).to(C.dev) for l in C.layers}
        Ms[it["id"]] = M
    C.thr, C.sd = dict(C.thr), dict(C.sd)
    analytic = []
    for it in items:
        if it["group"] == "facts":
            analytic += facts_analytic(C, it, Ms[it["id"]], core.conditions(cfg, False))
    checks = {}
    if first_fact:
        it = by_id[first_fact]
        checks["two_pass"] = checks_two_pass(C, it, Ms[first_fact])
        checks["batching_fact"] = check_batching(C, it, Ms[first_fact], core.conditions(cfg, False),
                                                 min(40, cfg["answer_cap"]))
    pm = next((it for it in items if it["mech"]), None)
    if pm is not None:
        checks["batching_mech"] = check_batching(C, pm, Ms[pm["id"]], core.conditions(cfg, True),
                                                 min(40, cfg["answer_cap"]))
    if cfg.get("repro_xlayer") and not C.A.random_model and not cfg.get("tiny"):
        checks["repro_xlayer"] = repro_xlayer(C, lens_all, cfg["repro_xlayer"])
    checks["counters"] = dict(C.chk)
    state = {"layers": C.layers, "tail_len": C.tail_len, "head": C.head.cpu(),
             "mu": {l: C.mu[l].cpu() for l in C.layers}, "W": {l: C.KS[l][W256].W.cpu() for l in C.layers},
             "thr": C.thr, "sd": C.sd, "med": C.med, "med_all": C.med_all,
             "moments": {iid: [{"j": m["j"], "n": m["n"], "key": {l: m["key"][l].cpu() for l in C.layers},
                                "shifts": {r: {l: s[l].cpu() for l in C.layers} for r, s in m["shifts"].items()}}
                               for m in ms] for iid, ms in moments_of.items()},
             "rand_dir": {iid: {l: v.cpu() for l, v in d.items()} for iid, d in C.rand_dir.items()}}
    torch.save(state, out / "state.pt")
    write_jsonl_gz(out / "lens.jsonl.gz", [r for iid in lens_all for r in lens_all[iid]])
    write_jsonl_gz(out / "facts_analytic.jsonl.gz", analytic)
    json.dump(checks, open(out / "checks.json", "w"), indent=2, default=str)
    meta = {"stage": "prep", "cfg": cfg, "env": env_meta(C), "template": C.template, "layers": C.layers,
            "layer_types": {l: C.ltype[l] for l in range(C.n_blocks)}, "n_blocks": C.n_blocks,
            "median_norm_read": C.med, "median_norm_all": C.med_all, "white_info": C.D.white_info,
            "mu_skip_tokens": C.D.skip, "tail_len": C.tail_len, "thresholds": {f"{i}/L{l}": v for i, d in C.thr.items()
                                                                                for l, v in d.items()},
            "items_sha": file_sha(cfg["items"]), "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(out / "config.json", "w"), indent=2, default=str)
    log(f"prep done in {meta['minutes']:.1f} min -> {out}")
    return checks


# ============================================================================= gen


def load_state(C, prep_dir, L):
    state = torch.load(Path(prep_dir) / "state.pt", map_location="cpu", weights_only=False)
    assert state["layers"] == C.layers, (state["layers"], C.layers)
    restore(C, state)
    by_id = L["by_id"]
    moments_of = {iid: [{"j": m["j"], "n": m["n"], "key": m["key"], "shifts": m["shifts"]} for m in ms]
                  for iid, ms in state["moments"].items()}
    C.rand_dir = {iid: {l: v.to(C.dev) for l, v in d.items()} for iid, d in state["rand_dir"].items()}
    Ms = {it["id"]: item_memories(C, it, moments_of, by_id) for it in L["items"]}
    return state, Ms


def gen_main(C, it, M, conds, prompts):
    """The item's main prompts: all non-ctx conditions of a prompt in one batch, ctx in another."""
    cfg, tok = C.cfg, C.tok
    S, cap, thr = cfg["samples"], cfg["answer_cap"], cfg["loop"]
    rows, gates = [], []
    s1, s2 = 1, 2
    for pi, (kind, text) in enumerate(prompts):
        streams = {s1: (core.seed_of(TAG, text, cfg["seeds"]["base"]), S + 1),
                   s2: (core.seed_of(TAG, text, cfg["seeds"]["second"]), S + 1)}
        ids = chat_ids(tok, text)
        P = read_prompt(C, it, M, ids)
        C.thr_now = {l: C.thr[it["id"]][l] for l in C.layers}
        gi = gate_info(C, P)
        gates.append({"sec": "gate", "item": it["id"], "group": it["group"], "kind": kind, "pi": pi, "prompt": text, **gi})
        specs = core.row_specs([c for c in conds if c.kind != "ctx"], S, s1, s2)
        Fc = {c.cid: fields_for(C, it, M, P, c) for c in conds if c.kind != "ctx"}
        ctx = [c for c in conds if c.kind == "ctx"]
        jobs = [(ids, specs[i:i + cfg["max_batch"]], True) for i in range(0, len(specs), cfg["max_batch"])]
        if ctx:
            jobs.append((ceiling_ids(tok, it["experience"], text), core.row_specs(ctx, S, s1, s2), False))
        for pids, chunk, use_f in jobs:
            fields = core.stack_fields([Fc[c.cid] for c, _, _ in chunk], C.layers, len(pids), C.d, C.dev) if use_f else {}
            outs = core.generate_batch(C.model, C.body, C.lm_head, pids, [(j == 0, None if j == 0 else (s, j))
                                                                         for _, s, j in chunk],
                                       max_new=cap, eos=C.eos_list, vocab=C.vocab, sampler=C.sampler,
                                       streams=streams, fields=fields)
            C.n_rows += len(chunk)
            C.n_batches += 1
            for (c, s, j), o in zip(chunk, outs):
                rep = core.rep_rate(o["ids"], thr["n"])
                rows.append({"sec": "gen", "key": f"{it['id']}|{kind}|{pi}|{c.cid}|{s}|{j}", "item": it["id"],
                             "group": it["group"], "kind": kind, "pi": pi, "prompt": text, "cond": c.cid,
                             "ckind": c.kind, "mem": c.mem, "alpha": c.alpha, "mech": c.mech, "seedset": s, "j": j,
                             "ids": o["ids"], "answer": tok.decode(o["ids"], skip_special_tokens=True).strip(),
                             "n_tok": o["n_tok"], "capped": o["capped"], "rep": round(rep, 4),
                             "loop": rep >= thr["thr"], "fire": None if c.kind in ("nomem", "ctx") else gi["fire"],
                             "open_any": None if c.kind in ("nomem", "ctx") else gi["open_any"]})
    return rows, gates


def gen_yn(C, it, M, conds):
    """Balanced yes/no probes, greedy (yn_cap tokens), every condition of a probe in one batch; the
    first-token margin log P(consistent answer) - log P(other) is read from the same pass."""
    cfg, tok = C.cfg, C.tok
    rows = []
    for rp in it["relation_probes"]:
        text = rp["text"]
        ids = chat_ids(tok, text)
        P = read_prompt(C, it, M, ids)
        cs = [c for c in conds if c.kind != "ctx"]
        fields = core.stack_fields([fields_for(C, it, M, P, c) for c in cs], C.layers, len(ids), C.d, C.dev)
        outs = core.generate_batch(C.model, C.body, C.lm_head, ids, [(True, None)] * len(cs), max_new=cfg["yn_cap"],
                                   eos=C.eos_list, vocab=C.vocab, fields=fields, first_ids=[C.yes, C.no])
        cx = [c for c in conds if c.kind == "ctx"]
        if cx:
            outs += core.generate_batch(C.model, C.body, C.lm_head, ceiling_ids(tok, it["experience"], text),
                                        [(True, None)], max_new=cfg["yn_cap"], eos=C.eos_list, vocab=C.vocab,
                                        first_ids=[C.yes, C.no])
            cs = cs + cx
        fire = sum(P.R[l].g[-1].item() for l in C.layers) / len(C.layers)
        for c, o in zip(cs, outs):
            ans = tok.decode(o["ids"], skip_special_tokens=True).strip()
            lpy, lpn = o["first"]
            m = lpy - lpn if rp["a"] == "Yes" else lpn - lpy
            rows.append({"sec": "yn", "item": it["id"], "group": it["group"], "prompt": text, "consistent": rp["a"],
                         "cond": c.cid, "ckind": c.kind, "alpha": c.alpha, "mech": c.mech, "answer": ans,
                         "yn": xl.parse_yn(ans), "margin": m, "lp_yes": lpy, "lp_no": lpn,
                         "fire": None if c.kind in ("nomem", "ctx") else fire})
    return rows


def kl_rows(C, text, ids, ans_ids, todo):
    """KL(nomem || condition) per answer position along the nomem greedy answer (teacher forcing), mean
    over positions. todo: [(item, cond, field {l: [T_prompt, d]})]."""
    if not ans_ids or not todo:
        return []
    seq = torch.cat([ids, torch.tensor(ans_ids, dtype=torch.long)])
    T, T0, n = len(seq), len(ids), len(ans_ids)
    out = []
    bs = C.cfg["kl_batch"]
    for i in range(0, len(todo), bs - 1):
        chunk = todo[i:i + bs - 1]
        per = [None] + [{l: xl.extend_field(f[l], T) for l in C.layers} for _, _, f in chunk]
        fields = core.stack_fields(per, C.layers, T, C.d, C.dev)
        X = seq[None].expand(len(per), -1).to(C.dev)
        with core.injecting(C.model, [core.BatchField(l, F) for l, F in fields.items()]):
            h = C.body(input_ids=X).last_hidden_state[:, T0 - 1:T - 1]
        lp = torch.log_softmax(C.lm_head(h).float(), -1)
        if not bool(torch.isfinite(lp).all()):
            raise FloatingPointError("non-finite log-probs in KL")
        base = lp[0]
        for b, (iid, cid, _) in enumerate(chunk, start=1):
            out.append({"sec": "kl", "item": iid, "cond": cid, "prompt": text, "n_pos": n,
                        "kl": (base.exp() * (base - lp[b])).sum(-1).mean().item()})
    return out


def gen_unrelated(C, L, Ms, conds_of, uk, text):
    """One unrelated prompt for every item: nomem, gate_on for every item, and the gated conditions of the
    items whose gate opens somewhere (else their answers ARE nomem's: the injection is exactly zero).
    Every batch carries its own nomem rows; `same` compares with the nomem row of the same batch and
    stream."""
    cfg, tok = C.cfg, C.tok
    S, cap = cfg["unrel_samples"], cfg["answer_cap"]
    ids = chat_ids(tok, text)
    st = pass1(C, ids, C.layers)
    streams = {1: (core.seed_of(TAG + "/unrel", text, cfg["seeds"]["base"]), S + 1)}
    rows, gates, todo_gen, todo_kl = [], [], [], []
    for it in L["items"]:
        M = Ms[it["id"]]
        P = read_prompt(C, it, M, ids, st)
        C.thr_now = {l: C.thr[it["id"]][l] for l in C.layers}
        gi = gate_info(C, P)
        gates.append({"sec": "ugate", "item": it["id"], "group": it["group"], "prompt": text, "uk": uk, **gi})
        for c in conds_of[it["id"]]:
            if c.kind in ("nomem", "ctx"):
                continue
            if core.gated(c) and not gi["open_any"]:
                for j in range(S + 1):
                    rows.append({"sec": "unrel", "item": it["id"], "group": it["group"], "cond": c.cid, "ckind": c.kind,
                                 "prompt": text, "uk": uk, "j": j, "shortcut": True, "same": True})
                todo_kl.append((it["id"], c.cid, None))
                continue
            F = fields_for(C, it, M, P, c)
            todo_gen += [(it, c, j, F) for j in range(S + 1)]
            todo_kl.append((it["id"], c.cid, F))
    nm_rows = [(True, None)] + [(False, (1, j)) for j in range(1, S + 1)]
    per_batch = cfg["max_batch"] - len(nm_rows)
    nomem_ref = None
    for i in range(0, max(len(todo_gen), 1), per_batch):
        chunk = todo_gen[i:i + per_batch]
        fields = core.stack_fields([None] * len(nm_rows) + [F for _, _, _, F in chunk], C.layers, len(ids), C.d, C.dev)
        outs = core.generate_batch(C.model, C.body, C.lm_head, ids,
                                   nm_rows + [(j == 0, None if j == 0 else (1, j)) for _, _, j, _ in chunk],
                                   max_new=cap, eos=C.eos_list, vocab=C.vocab, sampler=C.sampler, streams=streams,
                                   fields=fields)
        C.n_rows += len(nm_rows) + len(chunk)
        C.n_batches += 1
        nm = outs[:len(nm_rows)]
        if nomem_ref is None:
            nomem_ref = nm
            for j, o in enumerate(nm):
                rows.append({"sec": "unrel", "item": None, "group": None, "cond": "nomem", "ckind": "nomem",
                             "prompt": text, "uk": uk, "j": j, "shortcut": False, "same": True, "ids": o["ids"],
                             "answer": tok.decode(o["ids"], skip_special_tokens=True).strip(), "n_tok": o["n_tok"],
                             "capped": o["capped"]})
        C.chk["unrel_batch_nomem_eq_first"] += sum(a["ids"] == b["ids"] for a, b in zip(nm, nomem_ref))
        C.chk["unrel_batch_nomem_n"] += len(nm)
        for (it, c, j, _), o in zip(chunk, outs[len(nm_rows):]):
            rows.append({"sec": "unrel", "item": it["id"], "group": it["group"], "cond": c.cid, "ckind": c.kind,
                         "prompt": text, "uk": uk, "j": j, "shortcut": False, "same": o["ids"] == nm[j]["ids"],
                         "same_first": o["ids"] == nomem_ref[j]["ids"], "ids": o["ids"],
                         "answer": tok.decode(o["ids"], skip_special_tokens=True).strip(), "n_tok": o["n_tok"],
                         "capped": o["capped"]})
    kl = [{"sec": "kl", "item": iid, "cond": cid, "prompt": text, "uk": uk, "kl": 0.0, "n_pos": len(nomem_ref[0]["ids"]),
           "shortcut": True} for iid, cid, F in todo_kl if F is None]
    kl += [dict(r, uk=uk, shortcut=False) for r in kl_rows(C, text, ids, nomem_ref[0]["ids"],
                                                           [x for x in todo_kl if x[2] is not None])]
    return rows, gates, kl


def plan_counts(cfg, L):
    """Rows and decode steps (upper bound: every row runs to the cap), for the time estimate."""
    S = cfg["samples"]
    main_rows = main_batches = yn_rows = 0
    for it in L["items"]:
        cs = core.conditions(cfg, it["mech"])
        r = len(core.row_specs([c for c in cs if c.kind != "ctx"], S))
        rc = len(core.row_specs([c for c in cs if c.kind == "ctx"], S))
        n = len(it["prompts"])
        main_rows += n * (r + rc)
        main_batches += n * (math.ceil(r / cfg["max_batch"]) + 1)
        yn_rows += len(it["relation_probes"]) * len(cs)
    u = len(L["unrelated"]) * (cfg["unrel_samples"] + 1) * (1 + len(L["items"]))
    return {"main_rows": main_rows, "main_batches": main_batches, "main_decode_steps": main_batches * cfg["answer_cap"],
            "yn_rows": yn_rows, "unrel_rows_min": u, "unrel_batches_min": len(L["unrelated"]) * math.ceil(
                u / len(L["unrelated"]) / (cfg["max_batch"] - cfg["unrel_samples"] - 1))}


def gen(C, L, prep_dir, out):
    cfg = C.cfg
    out.mkdir(parents=True, exist_ok=True)
    (out / "items").mkdir(exist_ok=True)
    (out / "unrel").mkdir(exist_ok=True)
    t0 = time.time()
    state, Ms = load_state(C, prep_dir, L)
    log(f"gen: state loaded; plan {json.dumps(plan_counts(cfg, L))}")
    C.n_rows = C.n_batches = 0
    conds_of = {it["id"]: core.conditions(cfg, it["mech"]) for it in L["items"]}
    sel = [it for it in L["items"] if not C.A.items or it["id"] in C.A.items]
    if "main" in C.A.sections or "yn" in C.A.sections:
        for i, it in enumerate(sel):
            f = out / "items" / f"{it['id']}.jsonl.gz"
            if f.exists():
                log(f"gen: {it['id']} already done, skipped")
                continue
            prompts = it["prompts"][:cfg.get("n_prompts", len(it["prompts"]))]
            rows, gates = gen_main(C, it, Ms[it["id"]], conds_of[it["id"]], prompts) if "main" in C.A.sections else ([], [])
            yn = gen_yn(C, it, Ms[it["id"]], conds_of[it["id"]]) if "yn" in C.A.sections else []
            write_jsonl_gz(f, rows + gates + yn)
            el = time.time() - t0
            log(f"gen: {it['id']} ({it['group']}) done: {len(rows)} rows | {C.n_rows} rows, {C.n_batches} batches, "
                f"{el / 60:.1f} min, eta {el / (i + 1) * (len(sel) - i - 1) / 60:.1f} min")
    if "unrel" in C.A.sections:
        unrel = L["unrelated"][:cfg.get("n_unrel", len(L["unrelated"]))]
        t1 = time.time()
        for uk, text in enumerate(unrel):
            f = out / "unrel" / f"u{uk:02d}.jsonl.gz"
            if f.exists():
                continue
            rows, gates, kl = gen_unrelated(C, L, Ms, conds_of, uk, text)
            write_jsonl_gz(f, rows + gates + kl)
            el = time.time() - t1
            n_open = sum(g["open_any"] for g in gates)
            log(f"gen: unrelated {uk} done ({n_open}/{len(gates)} items' gates open) | {el / 60:.1f} min, "
                f"eta {el / (uk + 1) * (len(unrel) - uk - 1) / 60:.1f} min")
    meta = {"stage": "gen", "cfg": cfg, "env": env_meta(C), "args": vars(C.A), "plan": plan_counts(cfg, L),
            "n_rows": C.n_rows, "n_batches": C.n_batches, "counters": dict(C.chk), "minutes": (time.time() - t0) / 60,
            "items_sha": file_sha(cfg["items"])}
    json.dump(meta, open(out / f"config{'_' + '_'.join(C.A.items) if C.A.items else ''}.json", "w"), indent=2,
              default=str)
    log(f"gen done in {meta['minutes']:.1f} min ({C.n_rows} rows) -> {out}")


# =========================================================================== score


def gen_rows(gen_dir):
    rows = []
    for f in sorted(Path(gen_dir, "items").glob("*.jsonl.gz")) + sorted(Path(gen_dir, "unrel").glob("*.jsonl.gz")):
        rows += read_jsonl_gz(f)
    return rows


def coherence(C, L, rows, out):
    """Mean token NLL of every generated answer under the no-memory model (same model and precision),
    given the plain prompt; ctx rows also given the ctx prompt (nll_ctx). Identical (prompt, answer) pairs
    are scored once. Right-padded batches (causal: padding never changes earlier positions)."""
    tok = C.tok
    todo = {}
    for r in rows:
        if r["sec"] in ("gen", "unrel") and r.get("ids") is not None:
            todo.setdefault(("plain", r["prompt"], tuple(r["ids"])), None)
            if r["sec"] == "gen" and r["ckind"] == "ctx":
                todo.setdefault(("ctx:" + L["by_id"][r["item"]]["experience"], r["prompt"], tuple(r["ids"])), None)
    seqs = []
    for key in todo:
        mode, text, ans = key
        p = chat_ids(tok, text) if mode == "plain" else ceiling_ids(tok, mode[4:], text)
        seqs.append((key, p, list(ans)))
    seqs.sort(key=lambda s: len(s[1]) + len(s[2]))
    budget = C.cfg["nll_tokens"]
    res = {}
    i = 0
    t0 = time.time()
    while i < len(seqs):
        T = len(seqs[i][1]) + len(seqs[i][2])
        j = i + 1
        while j < len(seqs) and (j - i + 1) * (len(seqs[j][1]) + len(seqs[j][2])) <= budget:
            j += 1
        batch = seqs[i:j]
        Tm = max(len(p) + len(a) for _, p, a in batch)
        X = torch.full((len(batch), Tm), C.pad, dtype=torch.long)
        for b, (_, p, a) in enumerate(batch):
            X[b, :len(p) + len(a)] = torch.cat([p, torch.tensor(a, dtype=torch.long)]) if a else p
        h = C.body(input_ids=X.to(C.dev)).last_hidden_state
        for b, (key, p, a) in enumerate(batch):
            if not a:
                res[key] = (float("nan"), 0)
                continue
            lg = C.lm_head(h[b, len(p) - 1:len(p) + len(a) - 1]).float()
            lp = torch.log_softmax(lg, -1)
            nll = -lp.gather(1, torch.tensor(a, device=C.dev)[:, None]).mean().item()
            if not math.isfinite(nll):
                raise FloatingPointError(f"non-finite NLL for {key[1]!r}")
            res[key] = (nll, len(a))
        i = j
    log(f"coherence: {len(seqs)} unique answers scored in {(time.time() - t0) / 60:.1f} min")
    outrows = []
    for r in rows:
        if r["sec"] in ("gen", "unrel") and r.get("ids") is not None:
            o = {"sec": r["sec"], "key": r.get("key") or f"unrel|{r['uk']}|{r['item']}|{r['cond']}|{r['j']}",
                 "nll": res[("plain", r["prompt"], tuple(r["ids"]))][0], "n_tok": len(r["ids"])}
            if r["sec"] == "gen" and r["ckind"] == "ctx":
                o["nll_ctx"] = res[("ctx:" + L["by_id"][r["item"]]["experience"], r["prompt"], tuple(r["ids"]))][0]
            outrows.append(o)
    write_jsonl_gz(out / "coherence.jsonl.gz", outrows)
    return len(seqs)


def judge_text(J, L, r):
    it = L["by_id"][r["item"]]
    ans = r["answer"] if r["answer"].strip() else "(empty reply)"
    if it["group"] == "facts":
        return core.render(J["fact"]["template"], fact=it["experience"], target=it["measure"]["target"].strip(),
                           question=r["prompt"], answer=ans)
    return core.render(J["preference"]["template"], trait=it["experience"], question=r["prompt"], answer=ans)


def judge(C, L, rows, out, limit=None):
    """Qwen3.5-9B bf16 (C is the judge context), thinking off, greedy. Prompts are batched by exact token
    length (no padding at all), identical prompts judged once."""
    J = yaml.safe_load(open(HERE / "judge_prompts.yaml"))
    tok = C.tok
    todo = {}
    for r in rows:
        if r["sec"] == "gen":
            todo.setdefault(judge_text(J, L, r), []).append(r)
    texts = list(todo)[:limit] if limit else list(todo)
    enc = {t: chat_ids(tok, t) for t in texts}
    by_len = defaultdict(list)
    for t in texts:
        by_len[len(enc[t])].append(t)
    bs, cap = C.cfg["judge"]["batch"], C.cfg["judge"]["max_new"]
    raw = {}
    t0 = time.time()
    for n, ts in sorted(by_len.items()):
        for i in range(0, len(ts), bs):
            chunk = ts[i:i + bs]
            P = torch.stack([enc[t] for t in chunk])
            outs = core.generate_batch(C.model, C.body, C.lm_head, P, [(True, None)] * len(chunk), max_new=cap,
                                       eos=C.eos_list, vocab=C.vocab)
            for t, o in zip(chunk, outs):
                raw[t] = tok.decode(o["ids"], skip_special_tokens=True).strip()
    log(f"judge: {len(texts)} unique prompts in {(time.time() - t0) / 60:.1f} min")
    outrows, n_ok = [], 0
    for t in texts:
        for r in todo[t]:
            schema = J["fact" if r["group"] == "facts" else "preference"]["schema"]
            p = core.parse_judge(raw[t], schema)
            n_ok += p["ok"]
            outrows.append({"key": r["key"], "raw": raw[t], **p})
    write_jsonl_gz(out / "judge.jsonl.gz", outrows)
    return {"unique": len(texts), "rows": len(outrows), "parsed_ok": n_ok / max(len(outrows), 1),
            "judge_prompts_sha": file_sha(HERE / "judge_prompts.yaml")}


def _score_coherence(cfg, A, L, rows, out):
    model, tok = load(cfg, A)
    C = context(A, cfg, model, tok)
    return coherence(C, L, rows, out), env_meta(C)


def _score_judge(cfg, A, L, rows, out, limit):
    jm, jt = load(cfg, A, judge=True)
    Cj = SimpleNamespace(A=A, cfg=cfg, model=jm, tok=jt, dev=A.device, body=text_model(jm), lm_head=jm.lm_head)
    Cj.vocab = Cj.lm_head.out_features
    eos = {jt.eos_token_id} | {jt.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    Cj.eos_list = sorted(e for e in eos if isinstance(e, int) and e != jt.unk_token_id)
    return judge(Cj, L, rows, out, limit)


def score(cfg, A, L, gen_dir, out, do_judge=True, limit=None):
    """Coherence with the run's model, then (after it is released) the judge."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = gen_rows(gen_dir)
    n_unique, env = _score_coherence(cfg, A, L, rows, out)
    release()
    meta = {"stage": "score", "cfg": cfg, "env": env, "coherence_unique": n_unique}
    if do_judge:
        meta["judge"] = _score_judge(cfg, A, L, rows, out, limit)
        meta["judge"]["model"] = cfg["judge"]
        release()
    meta["minutes"] = (time.time() - t0) / 60
    json.dump(meta, open(out / "config.json", "w"), indent=2, default=str)
    log(f"score done in {meta['minutes']:.1f} min -> {out}")
    return meta


# =========================================================================== smoke


def nomem_sanity(C, prompts, cap=80):
    """No-memory greedy answers must be finite (generate_batch raises otherwise) and coherent: >= 10 tokens,
    not a loop, mostly letters."""
    res = []
    for t in prompts:
        o = core.generate_batch(C.model, C.body, C.lm_head, chat_ids(C.tok, t), [(True, None)], max_new=cap,
                                eos=C.eos_list, vocab=C.vocab)[0]
        ans = C.tok.decode(o["ids"], skip_special_tokens=True).strip()
        letters = sum(ch.isalpha() or ch.isspace() for ch in ans) / max(len(ans), 1)
        ok = o["n_tok"] >= 10 and core.rep_rate(o["ids"]) < 0.3 and letters > 0.7
        res.append({"prompt": t, "answer": ans, "n_tok": o["n_tok"], "rep": core.rep_rate(o["ids"]),
                    "letters": round(letters, 3), "ok": ok})
    return res


def smoke(A, cfg, L, out):
    """The whole pipeline on a tiny subset with the real model, plus the numerics checks."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tc = tiny_cfg(cfg)
    first = [L["groups"][g][0] for g in itm.GROUPS if L["groups"][g]]
    Ls = subset(L, first)
    report = {"items": first}
    _smoke_prep_gen(A, tc, L, Ls, out, report)
    release()
    report["score"] = score(tc, A, Ls, out / "gen", out / "score", do_judge=A.judge_smoke, limit=24)
    rows = gen_rows(out / "gen")
    report["gen_rows"] = sum(r["sec"] == "gen" for r in rows)
    report["unrel_rows"] = sum(r["sec"] == "unrel" for r in rows)
    report["samples"] = [{"cond": r["cond"], "prompt": r["prompt"], "answer": r["answer"][:200]}
                         for r in rows if r["sec"] == "gen" and r["j"] == 0 and r["kind"] == "related"][:12]
    if A.judge_smoke:
        jr = read_jsonl_gz(out / "score" / "judge.jsonl.gz")
        report["judge_examples"] = [{"key": r["key"], "raw": r["raw"], "ok": r["ok"]} for r in jr[:8]]
        report["judge_ok"] = report["score"]["judge"]["parsed_ok"] >= 0.9
    report["minutes"] = (time.time() - t0) / 60
    json.dump(report, open(out / "smoke_report.json", "w"), indent=2, default=str)
    failed = []
    if not report["nomem_ok"] and not A.random_model:
        failed.append("no-memory greedy answers are not coherent")
    if A.judge_smoke and not report.get("judge_ok") and not A.random_model:
        failed.append("judge outputs do not parse")
    bt = report["prep_checks"].get("batching_fact", {})
    log(f"smoke: nomem_ok {report['nomem_ok']}, batched==unbatched {bt.get('batched_eq_unbatched')}, "
        f"== think_v1.generate {bt.get('eq_think_v1_generate')}, {report['minutes']:.1f} min -> {out}")
    if failed:
        raise SystemExit("SMOKE FAILED: " + "; ".join(failed))


def _smoke_prep_gen(A, tc, L, Ls, out, report):
    model, tok = load(tc, A)
    C = context(A, tc, model, tok)
    report["env"] = env_meta(C)
    report["nomem_sanity"] = nomem_sanity(C, L["unrelated"][:3] + [L["by_id"][Ls["items"][0]["id"]]["related"],
                                                                   L["by_id"][Ls["items"][-1]["id"]]["related"]])
    report["nomem_ok"] = all(r["ok"] for r in report["nomem_sanity"])
    report["prep_checks"] = prep(C, Ls, out / "prep")
    C.A.items, C.A.sections = None, ["main", "yn", "unrel"]
    gen(C, Ls, out / "prep", out / "gen")
    if torch.cuda.is_available():
        report["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)


def subset(L, ids):
    """L restricted to `ids` for generation; their swap partners stay available for the writes."""
    keep = [it for it in L["items"] if it["id"] in ids]
    return {**L, "items": keep, "groups": {g: [i for i in v if i in ids] for g, v in L["groups"].items()}}


# ============================================================================ main


def main(argv=None):
    A = parse_args(argv)
    cfg = load_cfg(A.config)
    if A.tiny:
        cfg = tiny_cfg(cfg)
    out = Path(A.out)
    L = itm.load_items(cfg["items"], cfg["bench"])
    L["all_ids"] = [it["id"] for it in L["items"]]
    res = itm.validate_items(L)
    assert not res["errors"], res["errors"]
    if A.tiny and A.stage != "smoke":
        L = subset(L, [L["groups"][g][0] for g in itm.GROUPS])
    with torch.inference_mode():
        if A.stage == "smoke":
            smoke(A, cfg, L, out / "smoke")
            return
        if A.stage == "score":
            score(cfg, A, L, out / "gen", out / "score", do_judge=not A.no_judge)
            return
        model, tok = load(cfg, A)
        C = context(A, cfg, model, tok)
        if A.stage == "prep":
            pdir = out / "prep"
            pdir.mkdir(parents=True, exist_ok=True)
            res = itm.validate_items(L, tok)  # with the tokenizer checks
            json.dump({"counts": res["counts"], "errors": res["errors"], "warnings": res["warnings"]},
                      open(pdir / "validation.json", "w"), indent=2)
            assert not res["errors"], res["errors"]
            log(f"plan: {json.dumps(plan_counts(cfg, L))}")
            prep(C, L, out / "prep")
        elif A.stage == "gen":
            gen(C, L, out / "prep", out / "gen")


if __name__ == "__main__":
    main()
