#!/usr/bin/env python
"""Seahorse xlayer_v1: cross-layer injection for FACTS (read late, inject earlier).

So far a memory was read (keys, shifts) and injected at the SAME late layers (L20, L21, L23 of
Qwen3.5-2B's 24 blocks). Facts come out as words or floods ("Petra Petra Petra") and never act as
premises (balanced yes/no stays at chance). Hypothesis (cf. Lindsey 2026, injected concepts act as
thoughts about two-thirds of the way through a model): read the shift where its meaning is most complete
(late, layer R) but inject it at a middle layer W, so the remaining blocks (attention included) can
PROCESS the memory instead of blurting it out.

Fixed (ref_v1 / think_v1 for facts): Qwen/Qwen3.5-2B fp32, thinking OFF. One isolated memory per fact
(delta rule, 3 moments = the 3 follow-ups): plain shift (with - without), entropy-weighted pooled over
the follow-up, template tail excluded; key pooled_w256 of the without run; threshold = the q0.95 match
on generic prompts + greedy continuations at the read layer; never on the template head. Every read
layer R has its own M, built from layer-R shifts and keys.

Cross-layer read (two passes; experiments/xlayer_v1/xl.py). Pass 1 runs the model WITHOUT memory and
computes the layer-R keys, gates g and recalls M k at every position. Pass 2 adds, at the output of
block W,
    h_W <- h_W + alpha * ratio(W, R) * g_R * M_R k_R,   ratio = median |h_W| / median |h_R|
(medians over the non-head positions of the generic calibration prompts + continuations; 1 if W = R).
The gate never sees the injection; positions after the user text reuse the last prompt position's field,
so KV-cached generation equals the full-sequence two-pass reader (checked), and W = R equals the old
one-pass diag_keys.Reader (checked).

--stage A  analytic grid, no generation: R in --grid-R x W in --grid-W x alpha in --grid-alpha. Per
           fact: log P / rank / specificity of the target vs the foils at the measure prefix (related +
           paraphrase probes); balanced yes/no margins (Yes- and No-correct verification probes kept
           apart) and their sign accuracy; KL to no memory on unrelated prompts with the real gate (leak)
           and with the cell's vector forced on (dmg: the damage proxy); logit lens of the stored shift at
           R and the direct-path target boost of the rescaled shift added to the layer-W state.
           choice.json: the --n-picks best extra cells (xl.PICK_RULE).
--stage B  generation (reads --stageA/choice.json): nomem, ctx (fact in the prompt: the ceiling), the 2x2
           core (R, W) in --core-layers^2 x --core-alpha, the current design (R = W = --cur-layers at
           --cur-alpha, think_v1's "L23+L20+L21 full"), the multi-layer cross (--xml-R -> --xml-W, paired
           in order) x --xml-alpha, and the Stage-A picks. Per fact: the related question + 2 "use"
           prompts (config.yaml) x (1 greedy + --samples samples: T 1, top-p 0.95, top-k 20, presence
           1.5; answer cap --answer-cap; the same random numbers in every condition); the balanced yes/no
           probes greedy (--yn-cap tokens); --unrel-gen unrelated prompts greedy.
--stage all  A then B in one process (--out/stageA, --out/stageB). With --tiny: the smoke run.

Outputs (--out): report.txt, summary.csv, results.jsonl, config.json; Stage A also grid.csv, lens.txt,
choice.json; Stage B also texts.jsonl.gz, samples.txt.
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
from seahorse.bench import load_bench
from seahorse.residual import capture, text_model
from seahorse.sessions import ceiling_ids, chat_ids, common_suffix_len, text_ids

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tv1 = _load("seahorse_think_v1_run", HERE.parent / "think_v1" / "run.py")
sh = _load("seahorse_ref_v1_shifts", HERE.parent / "ref_v1" / "shifts.py")
xl = _load("seahorse_xlayer_v1_xl", HERE / "xl.py")
dk, v01, s2 = tv1.dk, tv1.v01, tv1.s2
log, fmt, table = dk.log, dk.fmt, dk.table
nanmean = xl.nanmean

W256 = "pooled_w256"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["A", "B", "all"])
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--bench", default=str(REPO / "data" / "bench_v1"))
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--items", nargs="+", default=None, help="subset of the config's fact ids (default all)")
    p.add_argument("--grid-R", type=int, nargs="+", default=[12, 16, 20, 23], help="Stage A read layers")
    p.add_argument("--grid-W", type=int, nargs="+", default=[8, 12, 16, 20, 23], help="Stage A injection layers")
    p.add_argument("--grid-alpha", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--core-layers", type=int, nargs="+", default=[16, 23], help="Stage B core: (R, W) in this^2")
    p.add_argument("--core-alpha", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--cur-layers", type=int, nargs="+", default=[20, 21, 23], help="current design: R = W")
    p.add_argument("--cur-alpha", type=float, nargs="+", default=[2.0])
    p.add_argument("--xml-R", type=int, nargs="+", default=[20, 21, 23], help="multi-layer cross: read layers")
    p.add_argument("--xml-W", type=int, nargs="+", default=[14, 15, 16], help="... injected at (paired in order)")
    p.add_argument("--xml-alpha", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--n-picks", type=int, default=2, help="extra Stage-A cells generated in Stage B")
    p.add_argument("--stageA", default=None, help="Stage A output dir (Stage B reads its choice.json)")
    p.add_argument("--n-unrelated", type=int, default=8, help="unrelated probes for the Stage-A leak / dmg KL")
    p.add_argument("--unrel-gen", type=int, nargs="+", default=[0, 2, 3, 5],
                   help="indices of bench_v1 unrelated_probes generated greedily in Stage B (selectivity)")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--answer-cap", type=int, default=150)
    p.add_argument("--yn-cap", type=int, default=10)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence", type=float, default=1.5)
    p.add_argument("--thr-q", type=float, default=0.95)
    p.add_argument("--white-eps", type=float, default=0.01)
    p.add_argument("--rls-lambda", type=float, default=0.1, help="(unused by the delta rule; passed through)")
    p.add_argument("--generic", default=str(v01.GENERIC))
    p.add_argument("--cont-len", type=int, default=20, help="greedy continuation of the unrelated probes (KL)")
    p.add_argument("--gen-cont-len", type=int, default=20, help="greedy continuation of generic prompts (calibration)")
    p.add_argument("--loop-n", type=int, default=4)
    p.add_argument("--loop-thr", type=float, default=0.3)
    p.add_argument("--topn", type=int, default=12, help="logit-lens top tokens in lens.txt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-tol", type=float, default=0.05)
    p.add_argument("--tiny", action="store_true",
                   help="smoke run: 2 facts, a 2x3x1 grid, one Stage-B condition per family, few tokens")
    p.add_argument("--random-model", default=None, choices=["qwen2", "qwen3_5"],
                   help="tiny random model with --model's tokenizer (CPU checks)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    a.core_pairs = [(R, W) for R in a.core_layers for W in a.core_layers]
    if a.tiny:
        a.items = a.items or ["dog_name", "sister_name"]
        a.grid_R, a.grid_W, a.grid_alpha = [16, 23], [8, 16, 23], [2.0]
        a.core_pairs, a.core_alpha, a.cur_alpha, a.xml_alpha, a.n_picks = [(23, 16)], [2.0], [2.0], [2.0], 1
        a.n_unrelated, a.unrel_gen, a.samples, a.answer_cap, a.yn_cap = 2, [0], 2, 16, 6
    return a


# ------------------------------------------------------------------------ items


def probe_text(b, dist):
    return next(p["text"] for p in b["probes"] if p["distance"] == dist)


def load_items(A, bench):
    cfg = yaml.safe_load(open(A.config))
    by = {it["id"]: it for it in bench["scenarios"]}
    items = []
    for c in cfg["items"]:
        if A.items and c["id"] not in A.items:
            continue
        b = by[c["id"]]
        assert b["type"] == "fact", c["id"]
        it = {k: b[k] for k in ("id", "category", "slot", "experience", "counter", "followups", "measure",
                                "relation_probes")}
        it.update(related=probe_text(b, "related"), paraphrase=probe_text(b, "paraphrase"), use=list(c["use"]))
        tgt = b["measure"]["target"].strip()
        assert len(it["use"]) == 2 and not any(xl.has_word(u, tgt) for u in it["use"]), c["id"]
        assert len(it["followups"]) == 3, c["id"]
        assert sorted(r["consistent"] for r in it["relation_probes"]) == ["No", "No", "Yes", "Yes"], c["id"]
        items.append(it)
    missing = set(A.items or []) - {it["id"] for it in items}
    assert not missing, f"--items not in {A.config}: {missing}"
    return items


# ----------------------------------------------------------------- conditions


def b_conditions(A, picks):
    """Stage-B conditions: cid, family, specs = [(R, W, alpha)] (one read/inject pair each)."""
    C_ = SimpleNamespace
    cs = [C_(cid="nomem", fam="base", specs=[], ctx=False), C_(cid="ctx", fam="base", specs=[], ctx=True)]
    for R, W in A.core_pairs:
        for a in A.core_alpha:
            cs.append(C_(cid=f"R{R}>W{W}@{a:g}", fam="core", specs=[(R, W, a)], ctx=False))
    cl = ",".join(map(str, A.cur_layers))
    for a in A.cur_alpha:
        cs.append(C_(cid=f"cur[{cl}]@{a:g}", fam="current", specs=[(l, l, a) for l in A.cur_layers], ctx=False))
    assert len(A.xml_R) == len(A.xml_W)
    xr, xw = ",".join(map(str, A.xml_R)), ",".join(map(str, A.xml_W))
    for a in A.xml_alpha:
        cs.append(C_(cid=f"xml[{xr}>{xw}]@{a:g}", fam="multi-cross", specs=[(r, w, a) for r, w in zip(A.xml_R, A.xml_W)],
                     ctx=False))
    for R, W, a in picks:
        cs.append(C_(cid=f"pick:R{R}>W{W}@{a:g}", fam="stageA-pick", specs=[(R, W, a)], ctx=False))
    return cs


def read_write_layers(A, stage, picks=()):
    Rs, Ws = set(), set()
    if stage in ("A", "all"):
        Rs |= set(A.grid_R)
        Ws |= set(A.grid_W)
    if stage in ("B", "all"):  # with "all" the picks come from the grid
        for c in b_conditions(A, picks):
            for R, W, _ in c.specs:
                Rs.add(R)
                Ws.add(W)
    return sorted(Rs), sorted(Ws)


# ------------------------------------------------------------------------ setup


def setup(A, out, layers):
    bench = load_bench(A.bench)
    items = load_items(A, bench)
    unrel = bench["unrelated_probes"][:A.n_unrelated]
    path = out / "dk_input.json"  # diag_keys.setup: model, mu, whitening, calibration (no scenarios needed)
    json.dump({"scenarios": [], "unrelated_probes": unrel}, open(path, "w"), indent=1)
    generic = A.generic
    if A.random_model:
        generic = out / "generic_tiny.txt"
        generic.write_text("\n".join([l for l in open(A.generic) if l.strip()][:16]))
    dargs = SimpleNamespace(model=A.model, layers=layers, analytic_layers=layers, scenarios=str(path),
                            generic=str(generic), cont_len=A.cont_len, gen_cont_len=A.gen_cont_len,
                            white_eps=A.white_eps, tiny=bool(A.random_model), device=A.device)
    dk.load = lambda _a: tv1.load_any(SimpleNamespace(tiny=A.random_model, model=A.model, device=A.device))
    dk.KEY_SPECS = {W256: (("pca", 256), True)}
    C = dk.setup(dargs)
    tok, model, dev = C.tok, C.model, A.device
    C.args, C.dev, C.chk, C.thr_log = A, dev, defaultdict(float), {}
    C.items, C.layers = items, sorted(layers)
    C.body, C.lm_head = text_model(model), model.lm_head
    C.vocab = C.lm_head.out_features
    C.d = C.mu[C.layers[0]].shape[0]
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
    C.think_end = -1  # thinking is off
    C.phase = SimpleNamespace(a_prompt=0.0, alpha=None)  # tv1.generate sets it; XInject ignores it
    C.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    off = chat_ids(tok, "alpha")
    assert torch.equal(off, tv1.chat(tok, "alpha", False)), "default template != enable_thinking=False"
    C.template = {"head": tok.decode(C.head), "tail": tok.decode(off[-C.tail_len:]), "eos": C.eos.tolist()}
    log(f"template {json.dumps(C.template)}")
    C.unrel_gen = [bench["unrelated_probes"][i] for i in A.unrel_gen]
    C.Kg = {l: dk.gen_keys(C, l, C.KS[l][W256]) for l in C.layers}
    C.med = {l: xl.median_norm([c[l][c["cats"] != 0] for c in C.gen]) for l in C.layers}
    C.ratio = lambda W, R: xl.dose_ratio(C.med, W, R)
    C.wrong = {it["id"]: tv1.wrong_fn(it) for it in items}
    log("median residual norm (generic prompts + continuations, non-head positions): "
        + ", ".join(f"L{l} ({C.ltype[l]}) {C.med[l]:.1f}" for l in C.layers))
    return C


def norm_table(C, Rs, Ws):
    """W rows x R cols: the dose ratio median|h_W| / median|h_R|."""
    return table(["W \\ R"] + [f"R{R} ({C.ltype[R]})" for R in Rs],
                 [[f"W{W} ({C.ltype[W]})"] + [f"{C.ratio(W, R):.3f}" for R in Rs] for W in Ws])


# ------------------------------------------------------------- writes + memories


def pass1(C, ids, layers, logits=False):
    """Memory-free forward: residuals [T, d] at `layers` (+ logits). Never call inside an injecting block."""
    with capture(C.model, layers) as st:
        out = C.model(ids[None].to(C.dev))
    return {l: st[l] for l in layers}, (out.logits[0].float() if logits else None)


def build_moments(C, it, Rs, check=False):
    """Per follow-up: the plain shift (with - without, entropy-weighted pooled, tail excluded) and the write
    key (pooled_w256 over the without run's user text) at every read layer R."""
    tok = C.tok
    moments = []
    for j, fu in enumerate(it["followups"]):
        s_with, s_wo = chat_ids(tok, f"{it['experience']} {fu}"), chat_ids(tok, fu)
        n = common_suffix_len(s_with, s_wo)
        st_wo, lg_wo = pass1(C, s_wo, Rs, logits=True)
        st_w, _ = pass1(C, s_with, Rs)
        w = sh.moment_weights(mx.entropy(lg_wo[-n:]), C.tail_len)
        cats = C.catfn(s_wo)
        m = {"j": j, "followup": fu, "n": n, "n_content": int(w.shape[0]), "shift": {}, "key": {}}
        for R in Rs:
            m["shift"][R] = sh.reference_shift(sh.pool(st_w[R][-n:], w), sh.pool(st_wo[R][-n:], w))
            m["key"][R] = C.KS[R][W256].keys(st_wo[R], cats == 1)[-1:]
        if check:  # must equal think_v1's fact write (diag_keys.unit_writes, baseline without, pooled)
            ch = v01.collect_writes(C.model, tok, {"experience": it["experience"], "counter": it["counter"],
                                                   "followups": [fu]}, Rs, C.dev)[0]
            full = {**st_wo, "cats": cats}
            for R in Rs:
                d, k, _ = dk.unit_writes(ch, full, R, "without", "pooled", C.KS[R][W256], False, C.tail_len)
                rel = ((d[0] - m["shift"][R]).norm() / d[0].norm().clamp_min(1e-12)).item()
                kabs = (k - m["key"][R]).norm().item()
                C.chk["repro_n"] += 1
                C.chk["repro_shift_max_rel"] = max(C.chk["repro_shift_max_rel"], rel)
                C.chk["repro_key_max_abs"] = max(C.chk["repro_key_max_abs"], kabs)
                if ch["n"] == n:
                    assert rel < 1e-3 and kabs < 1e-3, f"{it['id']} L{R}: != diag_keys.unit_writes ({rel}, {kabs})"
        moments.append(m)
    return moments


def item_memories(C, it, moments, Rs):
    """R -> (mem, thr, sd): delta rule over the 3 moments of layer R; threshold at layer R."""
    A, M = C.args, {}
    for R in Rs:
        chunks = [(m["shift"][R][None], m["key"][R], None) for m in moments]
        mem = dk.build({it["id"]: chunks}, [it["id"]], "delta", A.rls_lambda, A.device, C.chk)
        thr, sd = dk.calib(mem, C.Kg[R], A.thr_q)
        M[R] = SimpleNamespace(mem=mem, thr=thr, sd=sd)
        C.thr_log[f"{it['id']}/L{R}"] = thr
    return M


# ---------------------------------------------------------- two-pass reading


def read_units(C, M, ids, st, Rs):
    """Pass-1 states st (memory-free) -> per R: unit field g * M k [T, d], gate g, match, ungated last recall."""
    cats = C.catfn(ids)
    out = {}
    for R in Rs:
        k = C.KS[R][W256].keys(st[R], cats == 1)
        f, g, match = xl.cross_field(k, M[R].mem.Kst, M[R].mem.M, cats, M[R].thr, 1.0)
        out[R] = SimpleNamespace(f=f, g=g, match=match, r_last=k[-1] @ M[R].mem.M.T, thr=M[R].thr)
    return out


def fields_for(C, U, specs):
    """specs [(R, W, alpha)] -> ({W: field [T, d]}, per-spec gate info at the last position)."""
    F, info = {}, []
    for R, W, a in specs:
        s = a * C.ratio(W, R)
        xl.add_fields(F, W, s * U[R].f)
        info.append({"R": R, "W": W, "alpha": a, "scale": s, "match": U[R].match[-1].item(), "thr": U[R].thr,
                     "gate": U[R].g[-1].item(), "open_any": bool(U[R].g.any())})
    return F, info


def plan(C, M, ids, specs):
    Rs = sorted({R for R, _, _ in specs})
    st, _ = pass1(C, ids, Rs)
    return fields_for(C, read_units(C, M, ids, st, Rs), specs)


def forward_rows(C, rows):
    """rows: [(ids [T_i], {W: field [T0_i <= T_i, d]} or None)] -> logits [B, Tmax, V]. Right-padded (causal:
    padding never changes earlier positions); each row's field is extended past its prompt with its last row."""
    B, T = len(rows), max(len(ids) for ids, _ in rows)
    X = torch.full((B, T), C.pad, dtype=torch.long)
    Ws = sorted({W for _, F in rows if F for W in F})
    Fb = {W: torch.zeros(B, T, C.d, device=C.dev) for W in Ws}
    for b, (ids, F) in enumerate(rows):
        X[b, :len(ids)] = ids
        for W, f in (F or {}).items():
            Fb[W][b, :len(ids)] = xl.extend_field(f, len(ids))
    with tv1.injecting(C.model, [(W, xl.RowField(Fb[W]), 1.0) for W in Ws]):
        return C.model(X.to(C.dev)).logits.float()


def gen(C, prompt, B, max_new, seed, sampler, F=None, info=None, keep_logits=False):
    """tv1.generate (KV cache; row 0 greedy) with pass-2 hooks xl.XInject at every injection layer of F."""
    readers = [xl.XInject(W, F[W], next(i for i in info if i["W"] == W)) for W in sorted(F)] if F else None
    g = tv1.generate(C, prompt, B, readers, a_think=0.0, a_answer=1.0, think=False, max_new=max_new,
                     sampler=sampler, seed=seed, greedy0=True, answer_cap=max_new, keep_logits=keep_logits)
    g.info = info
    g.fire = nanmean([i["gate"] for i in info]) if info else float("nan")
    return g


# ---------------------------------------------------------------------- checks


def check_cached(C, it, M, specs, name):
    """KV-cached two-pass generation == the full-sequence two-pass reader (pass 1 on the whole sequence)."""
    prompt = chat_ids(C.tok, it["related"])
    F, info = plan(C, M, prompt, specs)
    g = gen(C, prompt, 1, 8, 0, tv1.plain_sampler(1.0), F, info, keep_logits=True)
    full = torch.cat([prompt, g.raw[:len(g.logits) - 1]])
    Ff, _ = plan(C, M, full, specs)
    lr = forward_rows(C, [(full, Ff)])[0, len(prompt) - 1:]
    lb = forward_rows(C, [(full, None)])[0, len(prompt) - 1:]
    res = {"check": "cached == full-sequence", "cond": name, "item": it["id"], "specs": specs, "steps": len(g.logits),
           "max_abs_diff": (lr - g.logits).abs().max().item(), "memory_effect_max": (lr - lb).abs().max().item(),
           "gate": info}
    log(f"cached-reader check: {json.dumps(res)}")
    assert res["max_abs_diff"] < C.args.check_tol, f"cached reader != full-sequence reader: {res}"
    return res


def check_onepass(C, it, M, specs, name, strict):
    """Two-pass with W = R vs the old one-pass diag_keys.Reader (keys read inside the hook). Equal for one
    layer (strict); for several layers the one-pass keys at later layers see the earlier injections."""
    assert all(R == W for R, W, _ in specs)
    ids = torch.cat([chat_ids(C.tok, it["related"]), text_ids(C.tok, it["measure"]["prefix"])])
    F, info = plan(C, M, ids, specs)
    lt = forward_rows(C, [(ids, F)])[0]
    rds = [(R, dk.Reader(M[R].mem, C.KS[R][W256], "hard", M[R].thr, M[R].sd, 0, C.stash), a) for R, _, a in specs]
    with tv1.injecting(C.model, rds):
        lo = C.model(ids[None].to(C.dev)).logits[0].float()
    lb = forward_rows(C, [(ids, None)])[0]
    res = {"check": "two-pass == one-pass (W = R)", "cond": name, "item": it["id"], "specs": specs, "strict": strict,
           "max_abs_diff": (lt - lo).abs().max().item(), "memory_effect_max": (lt - lb).abs().max().item(), "gate": info}
    log(f"one-pass check: {json.dumps(res)}")
    if strict:
        assert res["max_abs_diff"] < C.args.check_tol, f"two-pass (W = R) != diag_keys.Reader: {res}"
    return res


# ====================================================================== Stage A


def lens(C, x):
    """Logit lens: x [d] through the final norm + output layer."""
    return C.lm_head(C.body.norm(x.float().to(C.dev)[None, None]))[0, 0].float()


def lp_cand(logits_row, T0, cand):
    lp = torch.log_softmax(logits_row[T0 - 1:T0 - 1 + len(cand)], -1)
    return float(sum(lp[j, c] for j, c in enumerate(cand)))


def eval_measure(C, prompts, cands, Fs):
    """prompts [ids] (chat + measure prefix); cands [target, *foils] token lists; Fs per prompt {W: field}|None."""
    rows = [(torch.cat([ids, torch.tensor(c, dtype=torch.long)]), F) for ids, F in zip(prompts, Fs) for c in cands]
    lg = forward_rows(C, rows)
    t0, f0 = cands[0][0], [c[0] for c in cands[1:]]
    out, i = [], 0
    for ids in prompts:
        T0 = len(ids)
        lps = [lp_cand(lg[i + j], T0, c) for j, c in enumerate(cands)]
        first = lg[i, T0 - 1]
        lf = sum(lps[1:]) / len(lps[1:])
        out.append({"lp_t": lps[0], "lp_f": lf, "spec": lps[0] - lf, "rank": int((first > first[t0]).sum().item()) + 1,
                    "m1": (first[t0] - first[f0].mean()).item()})
        i += len(cands)
    return out


def eval_relation(C, prompts, rps, Fs):
    """Yes/no margin log P(consistent answer) - log P(other) at the assistant's first token, per probe."""
    rows, ab = [], []
    for ids, rp, F in zip(prompts, rps, Fs):
        a, b = text_ids(C.tok, rp["a"]), text_ids(C.tok, rp["b"])
        ab.append((a.tolist(), b.tolist()))
        rows += [(torch.cat([ids, a]), F), (torch.cat([ids, b]), F)]
    lg = forward_rows(C, rows)
    return [lp_cand(lg[2 * j], len(ids), a) - lp_cand(lg[2 * j + 1], len(ids), b)
            for j, (ids, (a, b)) in enumerate(zip(prompts, ab))]


def eval_kl(C, UN, Fs, base):
    """Mean-over-continuation KL(no memory || memory) per unrelated prompt."""
    lg = forward_rows(C, [(un.ids, F) for un, F in zip(UN, Fs)])
    return [mx.kl(b0, torch.log_softmax(lg[b, un.plen - 1:len(un.ids) - 1], -1)) for b, (un, b0) in
            enumerate(zip(UN, base))]


def relation_cols(RP, rr, base=None):
    o = {"margins": [round(x, 4) for x in rr]}
    for side in ("Yes", "No"):
        idx = [j for j, x in enumerate(RP) if x.rp["consistent"] == side]
        o[f"acc{side[0]}"] = nanmean([float(rr[j] > 0) for j in idx])
        o[f"m{side[0]}"] = nanmean([rr[j] for j in idx])
        if base is not None:
            o[f"dm{side[0]}"] = nanmean([rr[j] - base[j] for j in idx])
    o["bal_acc"] = (o["accY"] + o["accN"]) / 2
    if base is not None:
        o["drel_bal"] = (o["dmY"] + o["dmN"]) / 2
    return o


def measure_cols(mm):
    o = {}
    for k, m in zip(("rel", "par"), mm):
        o.update({f"{k}_lp_t": m["lp_t"], f"{k}_spec": m["spec"], f"{k}_rank": m["rank"], f"{k}_m1": m["m1"]})
    o.update(lp_t=nanmean([m["lp_t"] for m in mm]), spec=nanmean([m["spec"] for m in mm]),
             top1=nanmean([float(m["rank"] == 1) for m in mm]))
    return o


def tokstr(C, i):
    return C.tok.decode([i]).replace(" ", "▁").replace("\n", "\\n")


def stageA_item(C, it, M, moments, Rs, Ws, alphas, UN, UN_base):
    tok, iid = C.tok, it["id"]
    meas = it["measure"]
    enc = lambda s: C.tok(s, add_special_tokens=False).input_ids
    cands = [enc(meas["target"])] + [enc(f) for f in meas["foils"]]
    t0, f0 = cands[0][0], [c[0] for c in cands[1:]]
    margin = lambda l: (l[t0] - l[f0].mean()).item()
    pre = text_ids(tok, meas["prefix"])
    rows = []
    MP = []
    for kind in ("related", "paraphrase"):
        ids = torch.cat([chat_ids(tok, it[kind]), pre])
        st, _ = pass1(C, ids, sorted(set(Rs) | set(Ws)))
        hW = {W: st[W][-1] for W in Ws}
        MP.append(SimpleNamespace(kind=kind, ids=ids, U=read_units(C, M, ids, st, Rs), hW=hW,
                                  l0={W: margin(lens(C, hW[W])) for W in Ws},
                                  ceil=torch.cat([ceiling_ids(tok, it["experience"], it[kind]), pre])))
    RP = []
    for rp in it["relation_probes"]:
        ids = chat_ids(tok, rp["text"])
        st, _ = pass1(C, ids, Rs)
        RP.append(SimpleNamespace(rp=rp, ids=ids, U=read_units(C, M, ids, st, Rs),
                                  ceil=ceiling_ids(tok, it["experience"], rp["text"])))
    UNi = [read_units(C, M, un.ids, un.st, Rs) for un in UN]
    # references: no memory, and the fact in the prompt (ceiling)
    base_m = eval_measure(C, [mp.ids for mp in MP], cands, [None] * len(MP))
    base_r = eval_relation(C, [x.ids for x in RP], [x.rp for x in RP], [None] * len(RP))
    ceil_m = eval_measure(C, [mp.ceil for mp in MP], cands, [None] * len(MP))
    ceil_r = eval_relation(C, [x.ceil for x in RP], [x.rp for x in RP], [None] * len(RP))
    for cond, mm, rr in (("nomem", base_m, base_r), ("ctx", ceil_m, ceil_r)):
        rows.append({"section": "ref", "item": iid, "cond": cond, **measure_cols(mm), **relation_cols(RP, rr)})
    # fire: the gate at the last prompt position, per read layer
    for kind, text in (("related", it["related"]), ("paraphrase", it["paraphrase"]), ("use1", it["use"][0]),
                       ("use2", it["use"][1])):
        ids = chat_ids(tok, text)
        st, _ = pass1(C, ids, Rs)
        U = read_units(C, M, ids, st, Rs)
        for R in Rs:
            rows.append({"section": "fire", "item": iid, "R": R, "kind": kind, "prompt": text,
                         "match": U[R].match[-1].item(), "thr": U[R].thr, "gate": U[R].g[-1].item()})
    for x in RP:
        for R in Rs:
            rows.append({"section": "fire", "item": iid, "R": R, "kind": f"yn_{x.rp['consistent']}", "prompt": x.rp["text"],
                         "match": x.U[R].match[-1].item(), "thr": x.U[R].thr, "gate": x.U[R].g[-1].item()})
    for un, U in zip(UN, UNi):
        for R in Rs:
            nh = un.cats != 0
            rows.append({"section": "fire", "item": iid, "R": R, "kind": "unrelated", "prompt": un.text,
                         "match": U[R].match[nh].max().item(), "thr": U[R].thr, "gate": U[R].g[nh].mean().item()})
    # logit lens of the stored shift (mean over the 3 moments)
    for R in Rs:
        ll = torch.stack([lens(C, m["shift"][R]) for m in moments]).mean(0)
        top = ll.topk(C.args.topn).indices.tolist()
        rows.append({"section": "lens", "item": iid, "R": R, "rank_t": int((ll > ll[t0]).sum().item()) + 1,
                     "z_t": ((ll[t0] - ll.mean()) / ll.std()).item(), "margin": margin(ll),
                     "shift_norm": nanmean([m["shift"][R].norm().item() for m in moments]),
                     "rel_dose": nanmean([m["shift"][R].norm().item() for m in moments]) / C.med[R],
                     "recall_rel": MP[0].U[R].r_last.norm().item() / C.med[R],
                     "top": [(tokstr(C, i), round(ll[i].item(), 2)) for i in top]})
    # the grid
    for R in Rs:
        for W in Ws:
            for a in alphas:
                s = a * C.ratio(W, R)
                mm = eval_measure(C, [mp.ids for mp in MP], cands, [{W: s * mp.U[R].f} for mp in MP])
                rr = eval_relation(C, [x.ids for x in RP], [x.rp for x in RP], [{W: s * x.U[R].f} for x in RP])
                const = s * MP[0].U[R].r_last
                dmg = eval_kl(C, UN, [{W: (un.cats != 0)[:, None].to(const.dtype) * const} for un in UN], UN_base)
                opn = [j for j, U in enumerate(UNi) if bool(U[R].g.any())]
                leak = [0.0] * len(UN)  # gate shut everywhere -> the injection is exactly zero
                if opn:
                    for j, v in zip(opn, eval_kl(C, [UN[j] for j in opn], [{W: s * UNi[j][R].f} for j in opn],
                                                 [UN_base[j] for j in opn])):
                        leak[j] = v
                o = {"section": "cell", "item": iid, "R": R, "W": W, "alpha": a, "ratio": C.ratio(W, R), "scale": s,
                     "fire_rel": MP[0].U[R].g[-1].item(), "fire_par": MP[1].U[R].g[-1].item()}
                for mp, m1, m0 in zip(MP, mm, base_m):
                    k = mp.kind[:3]
                    o.update({f"{k}_lp_t": m1["lp_t"], f"{k}_dlp_t": m1["lp_t"] - m0["lp_t"],
                              f"{k}_dspec": m1["spec"] - m0["spec"], f"{k}_rank": m1["rank"],
                              f"{k}_dm1": m1["m1"] - m0["m1"],
                              f"{k}_ddirect": margin(lens(C, mp.hW[W] + s * mp.U[R].f[-1])) - mp.l0[W]})
                for key in ("dlp_t", "dspec", "dm1", "ddirect"):
                    o[key] = nanmean([o[f"rel_{key}"], o[f"par_{key}"]])
                o["top1"] = nanmean([float(o["rel_rank"] == 1), float(o["par_rank"] == 1)])
                o.update(relation_cols(RP, rr, base_r))
                o.update(dmg=nanmean(dmg), dmg_max=max(dmg), leak=nanmean(leak), leak_open=len(opn) / len(UN))
                rows.append(o)
    return rows


CELL_KEYS = ("dlp_t", "dspec", "rel_dspec", "par_dspec", "top1", "dm1", "ddirect", "dmY", "dmN", "drel_bal", "accY",
             "accN", "bal_acc", "dmg", "leak", "leak_open", "fire_rel", "fire_par")
REF_KEYS = ("lp_t", "spec", "top1", "mY", "mN", "accY", "accN", "bal_acc")


def summarizeA(C, rows, Rs, Ws, alphas):
    by = defaultdict(list)
    for r in rows:
        if r["section"] == "cell":
            by[(r["R"], r["W"], r["alpha"])].append(r)
    cells = []
    for R in Rs:
        for W in Ws:
            for a in alphas:
                rs = by[(R, W, a)]
                c = {"section": "cell", "R": R, "W": W, "alpha": a, "ltype_R": C.ltype[R], "ltype_W": C.ltype[W],
                     "ratio": C.ratio(W, R), "n_items": len(rs)}
                c.update({k: nanmean([r[k] for r in rs]) for k in CELL_KEYS})
                cells.append(c)
    refs = []
    for cond in ("nomem", "ctx"):
        rs = [r for r in rows if r["section"] == "ref" and r["cond"] == cond]
        refs.append({"section": "ref", "cond": cond, "n_items": len(rs), **{k: nanmean([r[k] for r in rs]) for k in REF_KEYS}})
    return cells, refs


EXPLAIN_A = """WHAT EACH NUMBER MEANS (Stage A: teacher-forced scores, no generation)
- cell (R, W, alpha): the memory is READ at the output of block R (keys, gate and M from layer-R states of a
  memory-free first pass) and ADDED at the output of block W, scaled by alpha x ratio, ratio = median residual
  norm at W / at R on generic prompts (1 when W = R). The gate depends only on R, never on W or alpha.
- dlp_t: change in log P(target) at the measure prefix ("What's my dog's name?" -> assistant: "Your dog's name
  is" -> " Pepper"), vs no memory; mean over the related and paraphrase probes.
- dspec: change in specificity = log P(target) - mean log P(foils) (" Max", " Buddy", " Charlie"): does the
  memory raise THIS value rather than any value of its kind. rel_dspec / par_dspec: related / paraphrase only.
- top1: share of (fact, probe) where the target is the single most likely next token (the nomem / ctx rows
  give the baseline and the in-context ceiling).
- dmY / dmN: change in the yes/no margin log P(consistent answer) - log P(other answer) at the assistant's first
  token, on the verification probes whose correct answer is Yes ("Is my dog called Pepper?") / No ("Is my dog
  called Max?"), vs no memory. drel_bal = their mean: a memory that just pushes "Yes" (or "No") everywhere gains
  on one side and loses on the other and nets ~0; only a premise-like memory raises both.
- accY / accN: share of those probes answered correctly by the sign of the margin (> 0); bal_acc = their mean
  (0.5 = chance, 1 = perfect).
- dmg: KL(no memory || memory) per token over {n} unrelated prompts + their no-memory greedy continuations, with
  the cell's vector FORCED on at every non-head position (the fact's recall at the related prompt x alpha x
  ratio). The damage proxy: how much this vector, at this layer and dose, disturbs ordinary text if it fires.
- leak: the same KL with the REAL gate (selectivity; exactly 0 when the gate stays shut); leak_open = share of
  unrelated prompts where the gate opened anywhere.
- dm1 / ddirect: change in the first-token margin logit(target) - mean logit(foils) at the measure prefix. dm1:
  through the whole model. ddirect: the "direct path" only: the rescaled vector added to the layer-W state at the
  last position and read out at once through the final norm + output layer (logit lens), as if no later block
  did anything. ddirect >> dm1: later blocks damp the memory; dm1 >> ddirect: they amplify / process it.
  (Rescaling a vector does not change its own logit lens, so the rescaled shift at W is judged this way.)
- logit lens of the stored shift (per R): the mean stored shift alone through the final norm + output layer:
  rank of the target among all ~248k tokens (1 = top), its z-score over the vocabulary, target - mean foils
  logit. rel_dose = |shift| / median residual norm at R; recall = |M q| at the related prompt / the same median.
- fire: share of prompts whose gate is open at the last prompt position (per read layer R).
"""


def pivot(C, cells, key, Rs, Ws, alphas, spec):
    S = {(c["R"], c["W"], c["alpha"]): c for c in cells}
    return table(["R", "alpha"] + [f"W{W}" for W in Ws],
                 [[f"{R}", f"{a:g}"] + [fmt(S[(R, W, a)][key], spec) for W in Ws] for R in Rs for a in alphas])


def reportA(C, rows, cells, refs, choice, checks, Rs, Ws, alphas):
    A = C.args
    L = []
    w = L.append
    f2, f3 = (lambda x: fmt(x, "+.2f")), (lambda x: fmt(x, ".2f"))
    w(f"Seahorse xlayer_v1 STAGE A (analytic grid, no generation): {A.model}"
      f"{' TINY' if A.tiny else ''}{' RANDOM ' + A.random_model if A.random_model else ''}")
    w(f"read R {Rs} x inject W {Ws} x alpha {alphas}; key pooled_w256, read hard (q{A.thr_q} at R), two-pass gates, "
      f"isolated delta memory per fact (3 moments, plain with - without shift); thinking OFF.")
    w("facts: " + "; ".join(f"{it['id']} ({it['measure']['target'].strip()})" for it in C.items))
    w(f"layer types: " + ", ".join(f"L{l} {C.ltype[l]}" for l in sorted(set(Rs) | set(Ws))))
    w(f"checks: {json.dumps(checks)}")
    w("")
    w(EXPLAIN_A.format(n=len(C.unrel)))
    w("=" * 110)
    w("0. DOSE: median residual norm per layer (generic prompts + continuations, non-head positions) and the ratio "
      "median|h_W| / median|h_R| that rescales a layer-R shift injected at W")
    w(table(["layer", "type", "median |h|"], [[l, C.ltype[l], f"{C.med[l]:.1f}"] for l in C.layers]))
    w("")
    w(norm_table(C, Rs, Ws))
    w("")
    w("=" * 110)
    w("1. REFERENCES (macro means over facts): nomem = no memory, ctx = the fact in the prompt (ceiling)")
    w(table(["cond"] + list(REF_KEYS), [[r["cond"]] + [f2(r[k]) if k in ("lp_t", "spec", "mY", "mN") else f3(r[k])
                                                       for k in REF_KEYS] for r in refs]))
    w("")
    w("=" * 110)
    w("2. FIRE: share of facts whose gate is open at the last prompt position, by read layer and prompt kind "
      "(unrelated: mean share of open non-head positions)")
    kinds = ["related", "paraphrase", "use1", "use2", "yn_Yes", "yn_No", "unrelated"]
    F = defaultdict(list)
    for r in rows:
        if r["section"] == "fire":
            F[(r["R"], r["kind"])].append(r["gate"])
    w(table(["R"] + kinds, [[R] + [f3(nanmean(F[(R, k)])) for k in kinds] for R in Rs]))
    w("")
    w("use prompts per fact: gate (match / thr) at the last prompt position, per R")
    FI = {(r["item"], r["R"], r["kind"]): r for r in rows if r["section"] == "fire"}
    w(table(["fact", "prompt"] + [f"R{R}" for R in Rs],
            [[it["id"], k] + [f"{FI[(it['id'], R, k)]['gate']:.0f} ({FI[(it['id'], R, k)]['match']:.2f}/"
                              f"{FI[(it['id'], R, k)]['thr']:.2f})" for R in Rs]
             for it in C.items for k in ("related", "use1", "use2")]))
    w("")
    w("=" * 110)
    w("3. LOGIT LENS of the stored shift alone (mean over moments), per fact x R: target rank / z / target - foils; "
      "rel_dose = |shift| / median|h_R|")
    LZ = {(r["item"], r["R"]): r for r in rows if r["section"] == "lens"}
    w(table(["fact"] + [f"R{R} rank/z/marg/dose" for R in Rs],
            [[it["id"]] + [f"{LZ[(it['id'], R)]['rank_t']}/{LZ[(it['id'], R)]['z_t']:.1f}/"
                           f"{LZ[(it['id'], R)]['margin']:+.1f}/{LZ[(it['id'], R)]['rel_dose']:.2f}" for R in Rs]
             for it in C.items]))
    w("")
    w(table(["R", "median rank", "mean z", "mean margin", "rel_dose", "recall/med"],
            [[R, tv1.nanmedian([float(LZ[(it['id'], R)]['rank_t']) for it in C.items]),
              f2(nanmean([LZ[(it['id'], R)]['z_t'] for it in C.items])),
              f2(nanmean([LZ[(it['id'], R)]['margin'] for it in C.items])),
              f3(nanmean([LZ[(it['id'], R)]['rel_dose'] for it in C.items])),
              f3(nanmean([LZ[(it['id'], R)]['recall_rel'] for it in C.items]))] for R in Rs]))
    w("")
    for j, (key, title, spec) in enumerate((
            ("dspec", "target specificity gain dspec", "+.2f"), ("top1", "top1 (target is the top token)", ".2f"),
            ("drel_bal", "balanced yes/no margin gain drel_bal", "+.2f"),
            ("bal_acc", "balanced yes/no accuracy bal_acc (sign of the margin)", ".2f"),
            ("dmY", "yes/no margin gain on Yes-correct probes dmY", "+.2f"),
            ("dmN", "yes/no margin gain on No-correct probes dmN", "+.2f"),
            ("dmg", "damage dmg (KL, vector forced on, unrelated prompts)", ".3f"),
            ("ddirect", "direct-path first-token boost ddirect", "+.2f"),
            ("dm1", "full-model first-token boost dm1", "+.2f"))):
        w("=" * 110)
        w(f"4{'abcdefghi'[j]}. {title}: rows R x alpha, columns W (macro mean over facts)")
        w(pivot(C, cells, key, Rs, Ws, alphas, spec))
        w("")
    w("=" * 110)
    w("5. FULL GRID (macro means over facts)")
    hdr = ["R", "W", "alpha", "ratio", "dlp_t", "dspec", "rel", "par", "top1", "dmY", "dmN", "drel_bal", "accY", "accN",
           "bal_acc", "dmg", "leak", "dm1", "ddirect", "fire_rel"]
    w(table(hdr, [[c["R"], c["W"], f"{c['alpha']:g}", f"{c['ratio']:.3f}", f2(c["dlp_t"]), f2(c["dspec"]),
                   f2(c["rel_dspec"]), f2(c["par_dspec"]), f3(c["top1"]), f2(c["dmY"]), f2(c["dmN"]), f2(c["drel_bal"]),
                   f3(c["accY"]), f3(c["accN"]), f3(c["bal_acc"]), fmt(c["dmg"], ".4f"), fmt(c["leak"], ".4f"),
                   f2(c["dm1"]), f2(c["ddirect"]), f3(c["fire_rel"])] for c in cells]))
    w("")
    w("=" * 110)
    w("6. CHOICE for Stage B: " + choice["rule"])
    w(table(["#", "R", "W", "alpha", "score", "dspec", "drel_bal", "dmg", "bal_acc", "top1"],
            [[j + 1, c["R"], c["W"], f"{c['alpha']:g}", c["score"], f2(c["dspec"]), f2(c["drel_bal"]),
              fmt(c["dmg"], ".4f"), f3(c["bal_acc"]), f3(c["top1"])] for j, c in enumerate(choice["ranking"][:20])]))
    w(f"PICKS: {[(p['R'], p['W'], p['alpha']) for p in choice['picks']]}")
    return "\n".join(L) + "\n"


def lens_txt(C, rows, Rs, path):
    L = [f"# Seahorse xlayer_v1 logit lens: the mean stored shift (3 moments) of each fact at each read layer R -> "
         f"final norm -> lm_head; top {C.args.topn} tokens (▁ = leading space)."]
    for r in rows:
        if r["section"] == "lens":
            L.append(f"{r['item']:<18} R{r['R']:<3} target rank {r['rank_t']:<7} z {r['z_t']:+.1f} | "
                     + " ".join(f"{t}({v:+.1f})" for t, v in r["top"]))
    Path(path).write_text("\n".join(L) + "\n")


def run_stage_A(C, out, meta):
    A = C.args
    Rs, Ws, alphas = sorted(set(A.grid_R)), sorted(set(A.grid_W)), list(A.grid_alpha)
    t0 = time.time()
    log(f"STAGE A: R {Rs} x W {Ws} x alpha {alphas} x {len(C.items)} facts -> {out}")
    log("dose ratios median|h_W| / median|h_R|:\n" + norm_table(C, Rs, Ws))
    UN = []
    for text, u in zip(C.unrelated, C.unrel):  # diag_keys.setup: the unrelated probes + greedy continuations
        ids = torch.cat([u["ids"], u["cont"]])
        st, _ = pass1(C, ids, Rs)
        UN.append(SimpleNamespace(text=text, ids=ids, plen=len(u["ids"]), cats=C.catfn(ids), st=st))
    lg0 = forward_rows(C, [(un.ids, None) for un in UN])
    UN_base = [torch.log_softmax(lg0[b, un.plen - 1:len(un.ids) - 1], -1) for b, un in enumerate(UN)]
    rows, checks = [], {}
    part = open(out / "results.partial.jsonl", "w")
    for i, it in enumerate(C.items):
        moments = build_moments(C, it, Rs, check=(i == 0))
        M = item_memories(C, it, moments, Rs)
        if i == 0:
            Rt = Rs[-1]
            Wt = 16 if (16 in Ws and 16 < Rt) else Ws[0]
            checks["onepass"] = check_onepass(C, it, M, [(Rt, Rt, 2.0)], f"R{Rt}>W{Rt}@2", strict=True)
            checks["cached"] = check_cached(C, it, M, [(Rt, Wt, 2.0)], f"R{Rt}>W{Wt}@2")
        rs = stageA_item(C, it, M, moments, Rs, Ws, alphas, UN, UN_base)
        rows += rs
        for r in rs:
            part.write(json.dumps(r) + "\n")
        part.flush()
        el = time.time() - t0
        log(f"stage A: {it['id']} done | {el / 60:.1f} min, eta {el / (i + 1) * (len(C.items) - i - 1) / 60:.1f} min")
    part.close()
    cells, refs = summarizeA(C, rows, Rs, Ws, alphas)
    fixed = {(R, W, a) for R, W in A.core_pairs for a in A.core_alpha}
    picks, ranking = xl.pick_cells(cells, fixed, A.n_picks)
    keep = ("R", "W", "alpha", "score", "dspec", "drel_bal", "dmg", "bal_acc", "top1", "dlp_t", "accY", "accN")
    choice = {"picks": [{k: p[k] for k in keep} for p in picks],
              "rule": xl.PICK_RULE.format(fixed=sorted(fixed), n=A.n_picks),
              "ranking": [{k: c[k] for k in keep} for c in ranking[:40]], "fixed": sorted(fixed),
              "median_norm": C.med}
    log(f"stage A picks: {[(p['R'], p['W'], p['alpha']) for p in picks]}")
    small = {k: checks[k] for k in checks}
    cks = {**dict(C.chk), **{k: {x: v[x] for x in ("max_abs_diff", "memory_effect_max")} for k, v in small.items()}}
    tv1.write_jsonl(out / "results.jsonl", rows)
    dk.write_csv(out / "summary.csv", cells + refs)
    dk.write_csv(out / "grid.csv", [{k: v for k, v in r.items() if k != "margins"} for r in rows if r["section"] == "cell"])
    json.dump(choice, open(out / "choice.json", "w"), indent=2)
    (out / "report.txt").write_text(reportA(C, rows, cells, refs, choice, cks, Rs, Ws, alphas))
    lens_txt(C, rows, Rs, out / "lens.txt")
    (out / "results.partial.jsonl").unlink()
    m = {**meta, "stage": "A", "checks": cks, "check_details": checks, "thresholds": dict(C.thr_log),
         "ratios": {f"W{W}/R{R}": C.ratio(W, R) for R in Rs for W in Ws}, "minutes": (time.time() - t0) / 60}
    json.dump(m, open(out / "config.json", "w"), indent=2, default=str)
    log(f"stage A done in {m['minutes']:.1f} min -> {out}")
    return [(p["R"], p["W"], p["alpha"]) for p in picks]


# ====================================================================== Stage B


def stageB_item(C, it, M, conds, rows, texts, nomem_unrel):
    A, tok, iid = C.args, C.tok, it["id"]
    tgt = it["measure"]["target"].strip()
    smp = tv1.qwen_sampler(A.temp, A.top_k, A.top_p, A.presence)
    greedy = tv1.plain_sampler(1.0)
    prompts = [("related", 0, it["related"])] + [("use", j, u) for j, u in enumerate(it["use"])]
    for kind, pi, text in prompts:
        seed = tv1.seed_of("xlayer_v1", text, A.seed)
        for c in conds:
            prompt = ceiling_ids(tok, it["experience"], text) if c.ctx else chat_ids(tok, text)
            F, info = plan(C, M, prompt, c.specs) if c.specs else (None, None)
            g = gen(C, prompt, 1 + A.samples, A.answer_cap, seed, smp, F, info)
            C.n_gen += 1
            for b, r in enumerate(g.rows):
                hit, loop, clean = xl.clean_hit(r["answer"], r["ans_ids"], tgt, A.loop_n, A.loop_thr)
                wrong = C.wrong[iid](r["answer"])
                rows.append({"section": "gen", "item": iid, "cond": c.cid, "fam": c.fam, "kind": kind, "pi": pi,
                             "prompt": text, "row": b, "greedy": b == 0, "fire": g.fire, "gate": info if b == 0 else None,
                             "n_tok": len(r["ans_ids"]), "rep": xl.rep_rate(r["ans_ids"], A.loop_n), "loop": loop,
                             "tgt": hit, "clean": clean, "wrong": wrong, "confab": bool(wrong),
                             "self_attr": xl.self_attr(r["answer"], tgt)})
                texts.append({"item": iid, "cond": c.cid, "kind": kind, "prompt": text, "row": b, "answer": r["answer"]})
    for rp in it["relation_probes"]:
        for c in conds:
            prompt = ceiling_ids(tok, it["experience"], rp["text"]) if c.ctx else chat_ids(tok, rp["text"])
            F, info = plan(C, M, prompt, c.specs) if c.specs else (None, None)
            g = gen(C, prompt, 1, A.yn_cap, 0, greedy, F, info)
            C.n_gen += 1
            ans = g.rows[0]["answer"]
            yn = xl.parse_yn(ans)
            rows.append({"section": "yn", "item": iid, "cond": c.cid, "fam": c.fam, "prompt": rp["text"],
                         "consistent": rp["consistent"], "answer": ans, "yn": yn, "correct": yn == rp["consistent"].lower(),
                         "fire": g.fire, "gate": info})
            texts.append({"item": iid, "cond": c.cid, "kind": f"yn_{rp['consistent']}", "prompt": rp["text"], "row": 0,
                          "answer": ans})
    for text in C.unrel_gen:
        prompt = chat_ids(tok, text)
        if text not in nomem_unrel:
            nomem_unrel[text] = gen(C, prompt, 1, A.answer_cap, 0, greedy)
            C.n_gen += 1
        nm = nomem_unrel[text].rows[0]
        for c in conds:
            if not c.specs:
                continue
            F, info = plan(C, M, prompt, c.specs)
            opened = any(i["open_any"] for i in info)
            if opened:
                g = gen(C, prompt, 1, A.answer_cap, 0, greedy, F, info)
                C.n_gen += 1
                r = g.rows[0]
            else:  # every gate shut at every position: the injection is exactly zero, the answer IS nomem's
                r = nm
                C.n_shortcut += 1
            rows.append({"section": "unrel", "item": iid, "cond": c.cid, "fam": c.fam, "prompt": text, "open_any": opened,
                         "shortcut": not opened, "same": r["ans_ids"] == nm["ans_ids"], "n_tok": len(r["ans_ids"]),
                         "tgt": xl.has_word(r["answer"], tgt), "nomem_tgt": xl.has_word(nm["answer"], tgt)})
            texts.append({"item": iid, "cond": c.cid, "kind": "unrelated", "prompt": text, "row": 0, "answer": r["answer"]})


B_KEYS = ("fire", "ans_tgt", "loop_rel", "ans_tgt_clean", "use_tgt", "loop_use", "use_tgt_clean", "confab_rel",
          "confab_use", "self_rel", "self_use", "len", "accY", "accN", "bal", "yn_parsed", "unrel_same", "contam",
          "unrel_open")


def summarizeB(C, rows, conds):
    by = defaultdict(list)
    for r in rows:
        by[(r["item"], r["cond"])].append(r)
    nan = float("nan")
    summ = []
    for it in C.items:
        for c in conds:
            rs = by[(it["id"], c.cid)]
            rel = [r for r in rs if r["section"] == "gen" and r["kind"] == "related"]
            use = [r for r in rs if r["section"] == "gen" and r["kind"] == "use"]
            yn = [r for r in rs if r["section"] == "yn"]
            un = [r for r in rs if r["section"] == "unrel"]
            f = lambda xs, k: nanmean([float(x[k]) for x in xs])
            o = {"item": it["id"], "cond": c.cid, "fam": c.fam,
                 "fire": nanmean([r["fire"] for r in rel + use if r["greedy"]]),
                 "ans_tgt": f(rel, "tgt"), "loop_rel": f(rel, "loop"), "ans_tgt_clean": f(rel, "clean"),
                 "use_tgt": f(use, "tgt"), "loop_use": f(use, "loop"), "use_tgt_clean": f(use, "clean"),
                 "confab_rel": f(rel, "confab"), "confab_use": f(use, "confab"),
                 "self_rel": f(rel, "self_attr"), "self_use": f(use, "self_attr"), "len": f(rel + use, "n_tok"),
                 **xl.balanced_acc([(r["consistent"], r["yn"]) for r in yn]),
                 "yn_parsed": nanmean([float(r["yn"] is not None) for r in yn]),
                 "unrel_same": f(un, "same") if un else nan,
                 "contam": f(un, "tgt") - f(un, "nomem_tgt") if un else nan,
                 "unrel_open": f(un, "open_any") if un else nan, "n_rel": len(rel), "n_use": len(use)}
            summ.append(o)
    agg = []
    for c in conds:
        sub = [o for o in summ if o["cond"] == c.cid]
        a = {"item": "<all>", "cond": c.cid, "fam": c.fam, "n_items": len(sub)}
        a.update({k: nanmean([o[k] for o in sub]) for k in B_KEYS})
        agg.append(a)
    return summ, agg


EXPLAIN_B = """WHAT EACH NUMBER MEANS (Stage B: generated answers, thinking off)
Per fact: the related question ("What's my dog's name?") and 2 "use" prompts that need the fact without asking
for it ("Draft a quick note to the vet about my dog's checkup tomorrow."), each 1 greedy + {S} samples; the 4
balanced yes/no probes greedy ({yn} tokens); {U} unrelated prompts greedy. Conditions are named
R<read>>W<inject>@alpha (alpha before the norm rescaling; see the legend).
- ans_tgt: share of answers to the related question that contain the exact target word ("Pepper"). loop_rel:
  share of them that loop (repeated-4-gram rate >= {thr}). ans_tgt_clean: the target is there AND the answer is
  not a loop. A flood ("Petra Petra Petra ...") counts in ans_tgt but not in ans_tgt_clean.
- use_tgt / loop_use / use_tgt_clean: the same on the 2 use prompts. use_tgt_clean is the key "used it" measure:
  the fact shows up, unasked, inside a normal answer.
- bal: balanced yes/no accuracy from the first yes/no in the greedy answer (unparsed = wrong); accY / accN on the
  probes whose correct answer is Yes / No. 0.5 = chance (an always-"No" model scores 0.5).
- confab_rel / confab_use: share of answers asserting a wrong value: a foil, the counter value ("Rocky"), or a
  non-target value after the measure phrase ("dog's name is Max"). Vague guesses are not caught.
- self_rel / self_use (ROUGH, regex): the target inside a first-person claim ("my dog Pepper", "I am a
  pharmacist", "Pepper is my dog"). On the related question: the assistant claims the user's fact as its own.
  On use prompts a draft in the user's voice ("my dog Pepper" in a note to the vet) is CORRECT and also counts.
- len: mean answer length in tokens (cap {cap}; related + use). fire: share of related + use prompts whose gate
  (mean over the condition's read layers) is open at the last prompt position.
- unrel_same: share of unrelated greedy answers token-for-token identical to no memory (1 = no leak). When every
  gate stays shut at every prompt position the injection is exactly zero, so the answer IS the no-memory answer
  and is not regenerated (unrel_open = share where a gate opened somewhere). contam: unrelated answers containing
  the target minus the no-memory share.
"""


def reportB(C, summ, agg, conds, picks, checks, prev):
    A = C.args
    L = []
    w = L.append
    f2, f3 = (lambda x: fmt(x, "+.2f")), (lambda x: fmt(x, ".2f"))
    w(f"Seahorse xlayer_v1 STAGE B (generation, thinking off): {A.model}"
      f"{' TINY' if A.tiny else ''}{' RANDOM ' + A.random_model if A.random_model else ''}")
    w(f"key pooled_w256, read hard (q{A.thr_q} at the read layer), two-pass gates, isolated delta memory per fact; "
      f"related + 2 use prompts x (1 greedy + {A.samples} samples: T {A.temp:g}, top-p {A.top_p:g}, top-k {A.top_k}, "
      f"presence {A.presence:g}), answer cap {A.answer_cap}, the same random numbers in every condition; yes/no greedy "
      f"({A.yn_cap} tokens); {len(C.unrel_gen)} unrelated prompts greedy.")
    w("facts: " + "; ".join(f"{it['id']} ({it['measure']['target'].strip()})" for it in C.items))
    w(f"generations: {C.n_gen} (+ {C.n_shortcut} unrelated answers taken from nomem because every gate stayed shut)")
    w(f"checks: {json.dumps(checks)}")
    if prev:
        w(f"Stage A picks: {picks}; rule: {prev.get('rule', '')}")
    w("")
    w(EXPLAIN_B.format(S=A.samples, yn=A.yn_cap, U=len(C.unrel_gen), thr=A.loop_thr, cap=A.answer_cap))
    w("=" * 110)
    w("CONDITIONS: read layer R -> injection layer W (block types), alpha, ratio = median|h_W| / median|h_R|, "
      "effective scale = alpha x ratio")
    w(table(["cond", "family", "R -> W (types)", "alpha", "ratio", "scale"],
            [[c.cid, c.fam, "; ".join(f"{R}->{W} ({C.ltype[R][0]}->{C.ltype[W][0]})" for R, W, _ in c.specs) or "-",
              ",".join(f"{a:g}" for _, _, a in c.specs) or "-",
              ",".join(f"{C.ratio(W, R):.3f}" for R, W, _ in c.specs) or "-",
              ",".join(f"{a * C.ratio(W, R):.3f}" for R, W, a in c.specs) or "-"] for c in conds]))
    w("")
    w("=" * 110)
    w("1. MACRO MEANS OVER FACTS")
    hdr = ["cond", "fire", "ans_tgt", "loop_rel", "ans_clean", "use_tgt", "loop_use", "use_clean", "bal", "accY", "accN",
           "confab_r", "confab_u", "self_r", "self_u", "len", "unrel_same", "contam"]
    w(table(hdr, [[o["cond"], f3(o["fire"]), f3(o["ans_tgt"]), f3(o["loop_rel"]), f3(o["ans_tgt_clean"]),
                   f3(o["use_tgt"]), f3(o["loop_use"]), f3(o["use_tgt_clean"]), f3(o["bal"]), f3(o["accY"]),
                   f3(o["accN"]), f3(o["confab_rel"]), f3(o["confab_use"]), f3(o["self_rel"]), f3(o["self_use"]),
                   fmt(o["len"], ".0f"), f3(o["unrel_same"]), f2(o["contam"])] for o in agg]))
    w("")
    S = {(o["item"], o["cond"]): o for o in summ}
    cids = [c.cid for c in conds]
    for title, key in (("2. use_tgt_clean PER FACT (rows) x CONDITION", "use_tgt_clean"),
                       ("3. ans_tgt_clean PER FACT", "ans_tgt_clean"), ("4. ans_tgt (loops included) PER FACT", "ans_tgt"),
                       ("5. loop rate (related + use) PER FACT", None), ("6. bal PER FACT", "bal"),
                       ("7. confab_use PER FACT", "confab_use"), ("8. self_rel PER FACT", "self_rel")):
        w("=" * 110)
        w(title)
        get = (lambda o: nanmean([o["loop_rel"], o["loop_use"]])) if key is None else (lambda o, k=key: o[k])
        w(table(["fact"] + cids, [[it["id"]] + [f3(get(S[(it["id"], c)])) for c in cids] for it in C.items]))
        w("")
    return "\n".join(L) + "\n"


def samples_txtB(C, conds, rows, texts, path):
    A = C.args
    T = {(t["item"], t["cond"], t["prompt"], t["row"]): t["answer"] for t in texts}
    G = {(r["item"], r["cond"], r["prompt"], r["row"]): r for r in rows if r["section"] == "gen"}
    L = [f"# Seahorse xlayer_v1 Stage B samples: per fact x condition, the related question and the 2 use prompts "
         f"(greedy + sample 1), then the yes/no greedy answers. [g=gate m=match/thr] at the last prompt position "
         f"(per read layer); TGT = exact target, LOOP = repeated-4-gram rate >= {A.loop_thr}, CONFAB = wrong value, "
         f"SELF = rough first-person claim."]
    w = L.append
    for it in C.items:
        w("")
        w("=" * 110)
        w(f"## {it['id']}: {it['experience']}  (target {it['measure']['target'].strip()}; counter: {it['counter']})")
        for c in conds:
            w(f"--- [{c.cid}] ({c.fam}) ---")
            for kind, text in [("related", it["related"])] + [("use", u) for u in it["use"]]:
                for row in (0, 1):
                    r = G.get((it["id"], c.cid, text, row))
                    if r is None:
                        continue
                    gs = ""
                    if r["gate"]:
                        gs = " [" + " ".join(f"g{i['R']}>{i['W']}={i['gate']:.0f} m={i['match']:.2f}/{i['thr']:.2f}"
                                             for i in r["gate"]) + "]"
                    fl = "".join(f" {n}" for n, v in (("TGT", r["tgt"]), ("LOOP", r["loop"]), ("CONFAB", r["confab"]),
                                                      ("SELF", r["self_attr"])) if v)
                    w(f"[{kind} {'greedy' if row == 0 else 'sample 1'}{gs}{fl}] {text}")
                    w(f"  {s2.one_line(T[(it['id'], c.cid, text, row)])}")
            yn = [r for r in rows if r["section"] == "yn" and r["item"] == it["id"] and r["cond"] == c.cid]
            w("yes/no: " + " | ".join(f"{r['prompt'].replace(' Answer yes or no.', '')} ({r['consistent']}) -> "
                                      f"{s2.one_line(r['answer'])[:40]}" for r in yn))
        un = defaultdict(list)
        for r in rows:
            if r["section"] == "unrel" and r["item"] == it["id"]:
                un[r["cond"]].append("same" if r["same"] else "DIFF")
        w("unrelated greedy vs nomem: " + "; ".join(f"{k}: {' '.join(v)}" for k, v in un.items()))
    Path(path).write_text("\n".join(L) + "\n")


def run_stage_B(C, out, picks, prev, meta):
    A = C.args
    conds = b_conditions(A, picks)
    if A.tiny:  # smoke: one condition per family
        seen, keep = set(), []
        for c in conds:
            if c.fam == "base" or c.fam not in seen:
                keep.append(c)
                seen.add(c.fam)
        conds = keep
    Rs = sorted({R for c in conds for R, _, _ in c.specs})
    t0 = time.time()
    C.n_gen, C.n_shortcut = 0, 0
    per_item = 3 * len(conds) + len(conds) * 4 + len(C.unrel_gen) * (len(conds) - 2)
    log(f"STAGE B: {len(conds)} conditions {[c.cid for c in conds]} x {len(C.items)} facts; read layers {Rs}; "
        f"<= {per_item * len(C.items)} generations (3 x {1 + A.samples} rows, cap {A.answer_cap}) -> {out}")
    rows, texts, nomem_unrel, checks = [], [], {}, {}
    part = open(out / "results.partial.jsonl", "w")
    for i, it in enumerate(C.items):
        moments = build_moments(C, it, Rs)
        M = item_memories(C, it, moments, Rs)
        if i == 0:
            xm = [c for c in conds if c.fam == "multi-cross"]
            cr = ([c for c in conds if c.fam == "core" and c.specs[0][0] != c.specs[0][1]]
                  or [c for c in conds if c.fam == "stageA-pick" and c.specs[0][0] != c.specs[0][1]])
            cu = [c for c in conds if c.fam == "current"]
            if xm:
                checks["cached_xml"] = check_cached(C, it, M, xm[-1].specs, xm[-1].cid)
            if cr:
                checks["cached_cross"] = check_cached(C, it, M, cr[-1].specs, cr[-1].cid)
            Rt = max(Rs)
            checks["onepass_same"] = check_onepass(C, it, M, [(Rt, Rt, 2.0)], f"R{Rt}>W{Rt}@2", strict=True)
            if cu:
                checks["onepass_cur"] = check_onepass(C, it, M, cu[-1].specs, cu[-1].cid, strict=False)
        n0 = len(rows)
        stageB_item(C, it, M, conds, rows, texts, nomem_unrel)
        for r in rows[n0:]:
            part.write(json.dumps(r) + "\n")
        part.flush()
        el = time.time() - t0
        log(f"stage B: {it['id']} done | {C.n_gen} generations ({C.n_shortcut} unrelated shortcuts), {el / 60:.1f} min, "
            f"eta {el / (i + 1) * (len(C.items) - i - 1) / 60:.1f} min")
    part.close()
    summ, agg = summarizeB(C, rows, conds)
    cks = {**dict(C.chk), **{k: {x: v[x] for x in ("max_abs_diff", "memory_effect_max")} for k, v in checks.items()}}
    tv1.write_jsonl(out / "results.jsonl", rows)
    dk.write_csv(out / "summary.csv", summ + agg)
    with gzip.open(out / "texts.jsonl.gz", "wt") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    (out / "report.txt").write_text(reportB(C, summ, agg, conds, picks, cks, prev))
    samples_txtB(C, conds, rows, texts, out / "samples.txt")
    (out / "results.partial.jsonl").unlink()
    m = {**meta, "stage": "B", "conditions": [{"cid": c.cid, "fam": c.fam, "specs": c.specs, "ctx": c.ctx} for c in conds],
         "picks": picks, "stageA": A.stageA, "checks": cks, "check_details": checks, "thresholds": dict(C.thr_log),
         "ratios": {c.cid: [C.ratio(W, R) for R, W, _ in c.specs] for c in conds},
         "n_generations": C.n_gen, "n_unrelated_shortcuts": C.n_shortcut, "minutes": (time.time() - t0) / 60}
    if A.stageA:
        prevcfg = json.load(open(Path(A.stageA) / "config.json"))
        common = [k for k in C.thr_log if k in prevcfg.get("thresholds", {})]
        m["thr_vs_stageA"] = {"n": len(common), "max_abs_diff": max((abs(C.thr_log[k] - prevcfg["thresholds"][k])
                                                                      for k in common), default=float("nan"))}
        log(f"thresholds vs stage A: {m['thr_vs_stageA']}")
    json.dump(m, open(out / "config.json", "w"), indent=2, default=str)
    log(f"stage B done in {m['minutes']:.1f} min ({C.n_gen} generations) -> {out}")


# -------------------------------------------------------------------------- main


def main():
    A = parse_args()
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    picks, prev = [], None
    if A.stage == "B":
        assert A.stageA, "--stageA (the Stage A output dir) is required for --stage B"
        prev = json.load(open(Path(A.stageA) / "choice.json"))
        picks = [(p["R"], p["W"], p["alpha"]) for p in prev["picks"]][:A.n_picks]
    Rl, Wl = read_write_layers(A, A.stage, picks)
    layers = sorted(set(Rl) | set(Wl))
    log(f"xlayer_v1 stage {A.stage}: read layers {Rl}, injection layers {Wl}; out {out}")
    with torch.inference_mode():
        C = setup(A, out, layers)
        log(f"setup done in {(time.time() - t0) / 60:.1f} min; facts {[it['id'] for it in C.items]}; layer types "
            f"{ {l: C.ltype[l] for l in layers} }")
        meta = {"args": vars(A), "items": [{k: it[k] for k in ("id", "experience", "counter", "related", "paraphrase",
                                                                 "use", "measure")} for it in C.items],
                "template": C.template, "layer_types": C.ltype, "median_norm": C.med, "white_info": C.white_info,
                "mu_skip_tokens": C.skip, "tail_len": C.tail_len, "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "linear_attention_kernels": tv1._kernels()}
        if A.stage in ("A", "all"):
            outA = out / "stageA" if A.stage == "all" else out
            outA.mkdir(parents=True, exist_ok=True)
            picks = run_stage_A(C, outA, meta)
            if A.stage == "all":
                prev = json.load(open(outA / "choice.json"))
        if A.stage in ("B", "all"):
            outB = out / "stageB" if A.stage == "all" else out
            outB.mkdir(parents=True, exist_ok=True)
            run_stage_B(C, outB, picks, prev, meta)
    log(f"xlayer_v1 stage {A.stage} done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
