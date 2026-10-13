#!/usr/bin/env python
"""dream_bind stage 0 (docs/dreaming/01-binding.md section 2; PREREG.md): does an in-context fact live at
the question's subject, or only at the attribute's own position?

Prompts (both models; Qwen3.5 with enable_thinking=False and the template tail asserted):
  ctx source   one user turn "My dog's name is Pepper. What's my dog's name?", the assistant header, then
               the teacher-forced measure prefix "Your dog's name is"; logits at the last prefix token
  ctx target   identical with the counter value (Rocky); asserted: same length, one differing token
  nomem        the question alone + the same prefix (knockout shares)
  subject      the alias span inside the QUESTION (never the experience's own mention); primary site =
               its last token ("dog", "friend"), also the whole span

Stages (--stage, several allowed; each writes <out>/<stage>/{results.jsonl.gz, checks.json, config.json};
<out>/report.md and <out>/verdict.json are rebuilt from whatever stages exist after every run):
  patch     (both models) residual output of block l at the site, source -> target, every single block and
            every band of 5; sites subj_last, subj_span, attr. Measure logit(target) - logit(counter);
            share = (patched - target run) / (source run - target run)
  knockout  (Qwen2.5 only, eager attention) in the ctx source run, block attention from the answer
            position to the subject span / the attribute, windows of 5 layers; share of ctx's gain in
            log P(target) over nomem. Plus the subject-alone cell: in the target run, attention to the
            attribute blocked from every query after the experience sentence at all layers, and the
            subject patched from the source (with the blocked-no-patch sanity row)
  report    rebuild report.md / verdict.json only (no model)

Checks (the job fails if one is violated): layouts as asserted; patching every position at a block
reproduces the source logits; patching nothing reproduces the target logits; an empty knockout equals the
unmodified run; a knockout of everything but itself at every layer equals the last token run alone (the
mask edit is effective); all logits finite; real models answer "What is the capital of France?" with Paris.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import subprocess
import time
import zlib
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
import yaml

from seahorse.residual import capture, decoder_layers, text_model
from seahorse.sessions import chat_ids, text_ids

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
_TF_MAJOR = int(transformers.__version__.split(".")[0])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bind = _load("dream_bind_bind", HERE / "bind.py")
itm = _load("dream_bind_core_v1_items", HERE.parent / "core_v1" / "items.py")
cv1 = _load("dream_bind_core_v1_core", HERE.parent / "core_v1" / "core.py")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", nargs="+", required=True, choices=["patch", "knockout", "prefix", "analytic", "gen", "collide", "idrepair", "gkey", "judge", "analyze", "report"])
    p.add_argument("--model", required=True, choices=["qwen35_2b", "qwen25_15b"])
    p.add_argument("--out", required=True, help="the model's run dir, e.g. <run>/qwen25_15b")
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--tiny", action="store_true", help="2 items (smoke)")
    p.add_argument("--random-model", default=None, choices=["qwen2", "qwen3_5"],
                   help="tiny random model with the config model's tokenizer (CPU dry runs): qwen2 (eager) or qwen3_5 "
                        "(the hybrid DeltaNet + full-attention classes; needs transformers >= 5)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--gen-set", default=None, help="gen: use the model's gen_conds_<SET> condition set (e.g. state)")
    p.add_argument("--max-reserved-gb", type=float, default=None,
                   help="fit check: fail if the peak reserved GPU memory exceeds this (smoke jobs before a long run)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- items


def load_items(cfg, tiny=False):
    L = itm.load_items(REPO / cfg["items"], REPO / cfg["bench"])
    res = itm.validate_items(L)
    assert not res["errors"], res["errors"]
    ids = cfg["primary"] + cfg["secondary"]
    if tiny:
        ids = ["dog_name", "best_friend_name"]
    out = []
    for iid in ids:
        it = L["by_id"][iid]
        sub = cfg["subjects"][iid]
        out.append({"id": iid, "tier": "primary" if iid in cfg["primary"] else "secondary",
                    "experience": it["experience"], "counter": cfg.get("counter_override", {}).get(iid, it["counter"]),
                    "counter_overridden": iid in cfg.get("counter_override", {}), "question": it["related"],
                    "prefix": it["measure"]["prefix"], "target": it["measure"]["target"],
                    "foils": list(it["measure"]["foils"]), "swap": it["swap"],
                    "relation_probes": [{"text": r["text"], "a": r["a"]} for r in it["relation_probes"]],
                    "harvest_followups": list(sub.get("harvest_followups", [])), "h2": sub.get("h2"),
                    "use": list(it["use"]),
                    "aliases": sub["subject"], "no_site": bool(sub.get("no_site", False))})
    return out


# --------------------------------------------------------------------------- model


def thinking_off(tok):
    """Force enable_thinking=False in every chat-template call and assert the template tail (core_v1)."""
    orig = tok.apply_chat_template

    def act(conversation, *a, **kw):
        kw.setdefault("enable_thinking", False)
        return orig(conversation, *a, **kw)
    tok.apply_chat_template = act
    s = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    assert s.endswith("<think>\n\n</think>\n\n"), f"thinking not off: {s[-40:]!r}"
    return tok


def random_qwen2(tok, n_layers, seed, device):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    c = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                    attn_implementation="eager")
    return Qwen2ForCausalLM(c).float().to(device).eval()


def random_qwen35(name, seed, device):
    """A tiny random Qwen3.5 text model (8 blocks: 3 Gated DeltaNet + 1 full attention, twice) from the real config."""
    from transformers import AutoConfig, AutoModelForCausalLM
    tc = AutoConfig.from_pretrained(name).text_config
    tc.hidden_size, tc.intermediate_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 64, 128, 2, 1, 64
    tc.linear_num_key_heads = tc.linear_num_value_heads = 2
    tc.linear_key_head_dim = tc.linear_value_head_dim = 32
    tc.num_hidden_layers, tc.layer_types = 8, list(tc.layer_types[:8])
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(tc).float().to(device).eval()


def load_text_only(name, device, dtype):
    """Qwen3.5 (a *ForConditionalGeneration checkpoint): the text decoder + tied lm_head only, no vision
    tower. Built from config.text_config on the target device, then every parameter is copied from the
    checkpoint (model.language_model.* -> model.*; model.visual.* and mtp.* skipped), cast on the CPU so the
    GPU never holds a second copy. Asserts that every parameter was loaded exactly once."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    tc = AutoConfig.from_pretrained(name).text_config
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(tc)
    model = model.to(dtype).eval()
    snap = Path(snapshot_download(name, allow_patterns=["*.safetensors", "*.json"]))
    params = dict(model.named_parameters())
    done, skipped = set(), set()
    with torch.no_grad():
        for f in sorted(snap.glob("*.safetensors")):
            with safe_open(str(f), "pt", device="cpu") as s:
                for k in s.keys():
                    if k.startswith("model.language_model."):
                        n = "model." + k[len("model.language_model."):]
                    elif k == "lm_head.weight":
                        n = k
                    else:
                        skipped.add(".".join(k.split(".")[:2]))
                        continue
                    assert n in params, f"checkpoint key {k} -> {n} not in the text model"
                    assert n not in done, f"{n} loaded twice"
                    t = s.get_tensor(k)
                    assert tuple(t.shape) == tuple(params[n].shape), (n, t.shape, params[n].shape)
                    params[n].copy_(t.to(params[n].dtype))
                    done.add(n)
    missing = sorted(set(params) - done)
    assert not missing, f"text-only load: parameters not in the checkpoint: {missing[:5]}"
    tied = model.lm_head.weight.data_ptr() == text_model(model).embed_tokens.weight.data_ptr()
    assert tied or "lm_head.weight" in done, "lm_head neither tied nor loaded"
    return model, {"n_params_loaded": len(done), "skipped_prefixes": sorted(skipped), "lm_head_tied": tied,
                   "snapshot": str(snap)}


def load(A, cfg, mc):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mc["model"])
    info = {}
    dtype = DTYPES[A.dtype]
    if A.random_model == "qwen3_5":
        model = random_qwen35(mc["model"], cfg["seeds"]["random_model"], A.device)
    elif A.random_model:
        model = random_qwen2(tok, 8, cfg["seeds"]["random_model"], A.device)
    elif mc["text_only"]:
        model, info = load_text_only(mc["model"], A.device, dtype)
    else:
        kw = {"dtype": dtype} if _TF_MAJOR >= 5 else {"torch_dtype": dtype}
        if mc["eager"]:
            kw["attn_implementation"] = "eager"
        try:
            import accelerate  # noqa: F401  (device_map: weights straight onto the device)
            kw.update(device_map=A.device, low_cpu_mem_usage=True)
        except ImportError:
            pass
        model = AutoModelForCausalLM.from_pretrained(mc["model"], **kw).to(A.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if mc["thinking_template"]:
        thinking_off(tok)
    impl = getattr(model.config, "_attn_implementation", None)
    if mc["eager"] or A.random_model == "qwen2":
        assert impl == "eager", f"eager attention required, got {impl}"
    log(f"loaded {mc['model']} ({'random ' + A.random_model if A.random_model else A.dtype}, attn {impl}) "
        f"in {time.time() - t0:.0f}s {json.dumps(info)}")
    info["attn_implementation"] = impl
    info["load_seconds"] = round(time.time() - t0, 1)
    return model, tok, info


def context(A, cfg, mc, model, tok):
    C = SimpleNamespace(A=A, cfg=cfg, mc=mc, model=model, tok=tok, dev=A.device)
    C.body, C.lm_head = text_model(model), model.lm_head
    C.blocks = decoder_layers(model)
    C.n = len(C.blocks)
    if not A.random_model:
        assert C.n == mc["n_blocks"], (C.n, mc["n_blocks"])
    C.d = C.lm_head.in_features
    C.band = cfg["band"]
    C.mb = cfg["max_batch"]
    C.tol = cfg["check_tol"][A.dtype]
    return C


def env_meta():
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
        except Exception:
            return None
    gpu = None
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        gpu = {"name": pr.name, "memory_gb": round(pr.total_memory / 2 ** 30, 2)}
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "torch": torch.__version__, "transformers": transformers.__version__, "gpu": gpu,
            "cuda": torch.version.cuda, "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def write_jsonl_gz(path, rows):
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------- prompts


def build(C, it):
    """Token layouts of one item (asserted)."""
    tok = C.tok
    pre = text_ids(tok, it["prefix"])
    u_src, u_tgt = f"{it['experience']} {it['question']}", f"{it['counter']} {it['question']}"
    src = torch.cat([chat_ids(tok, u_src), pre])
    tgt = torch.cat([chat_ids(tok, u_tgt), pre])
    nomem = torch.cat([chat_ids(tok, it["question"]), pre])
    a = bind.attribute_position(src, tgt)
    t_ids = text_ids(tok, it["target"]).tolist()
    assert int(src[a]) == t_ids[0], f"{it['id']}: the attribute token {tok.decode([int(src[a])])!r} is not the " \
                                    f"measure target's first token {tok.decode(t_ids[:1])!r}"
    c_tok = int(tgt[a])
    assert c_tok != t_ids[0]
    ls = bind.question_layout(tok, src, u_src, it["experience"], it["aliases"])
    lt = bind.question_layout(tok, tgt, u_tgt, it["counter"], it["aliases"])
    assert ls == lt, f"{it['id']}: source / target question layouts differ {ls} {lt}"
    assert a < ls["q0"], f"{it['id']}: attribute {a} not before the question ({ls['q0']})"
    assert len(nomem) < len(src)
    T = len(src)
    dec = lambda i: tok.decode([int(i)])
    # exploratory (A): the subject mention inside the teacher-forced answer prefix ("Your dog's name is")
    ps = bind.subject_sites(tok, src, u_src, it["aliases"], prefix=it["prefix"])["prefix_spans"]
    psubj = ps[0] if ps else []
    assert not psubj or (min(psubj) > max(ls["subj"], default=-1) and min(psubj) > a)
    return SimpleNamespace(src=src, tgt=tgt, nomem=nomem, T=T, a=a, t_tok=t_ids[0], c_tok=c_tok, q0=ls["q0"],
                           subj=ls["subj"], alias=ls["alias"], psubj=psubj,
                           info={"T": T, "attr_pos": a, "attr_src": dec(src[a]), "attr_tgt": dec(c_tok), "q0": ls["q0"],
                                 "q0_tok": dec(src[ls["q0"]]), "alias": ls["alias"], "subj": ls["subj"],
                                 "subj_toks": [dec(src[i]) for i in ls["subj"]], "target_multi_token": len(t_ids) > 1,
                                 "prefix_subj": psubj, "prefix_subj_toks": [dec(src[i]) for i in psubj],
                                 "last_tok": dec(src[-1]), "T_nomem": len(nomem)})


def sites(P):
    s = {}
    if P.subj:
        s["subj_last"] = [P.subj[-1]]
        s["subj_span"] = list(P.subj)
    s["attr"] = [P.a]
    return s


# --------------------------------------------------------------------------- forward


def last_logits(C, X, patch=None, src=None, block=None, add=None):
    """[B, V] fp32 logits at the last position, with optional per-row patching / blocking / adding plans."""
    with ExitStack() as es:
        if patch:
            es.enter_context(bind.patching(C.blocks, patch, src))
        if block is not None:
            es.enter_context(bind.blocking(C.blocks, block))
        if add:
            es.enter_context(bind.adding(C.blocks, add))
        h = C.body(input_ids=X.to(C.dev), use_cache=False).last_hidden_state[:, -1]
    lg = C.lm_head(h).float()
    if not bool(torch.isfinite(lg).all()):
        raise FloatingPointError("non-finite logits")
    return lg


def source_run(C, ids):
    """Batch-1 run: residual states at every block + last-position logits."""
    with capture(C.model, list(range(C.n))) as st:
        lg = last_logits(C, ids[None])
    return {l: st[l] for l in range(C.n)}, lg[0]


def run_rows(C, ids, rows, src=None, reduce=None):
    """rows: [{patch: [(layers, positions)], block: [(layers, queries, keys)], add: [(layers, positions,
    {layer: vec})], hook_all: bool}] on copies of `ids`, in batches of C.mb. Returns the last-position
    logits of every row [len(rows), V], or reduce(logits) per batch, concatenated (keeps memory small)."""
    out = []
    for i in range(0, len(rows), C.mb):
        ch = rows[i:i + C.mb]
        pp, bp, ap = defaultdict(list), None, defaultdict(list)
        for r, row in enumerate(ch):
            for layers, pos in row.get("patch", []):
                for l in layers:
                    pp[l].append((r, pos))
            for layers, pos, vecs in row.get("add", []):
                for l in layers:
                    ap[l].append((r, pos, vecs[l]))
            if row.get("block") or row.get("hook_all"):
                bp = bp if bp is not None else {l: [] for l in range(C.n)}
                for layers, qs, ks in row.get("block", []):
                    for l in layers:
                        bp[l].append((r, qs, ks))
        X = ids[None].expand(len(ch), -1)
        lg = last_logits(C, X, patch=dict(pp), src=src, block=bp, add=dict(ap))
        out.append(reduce(lg) if reduce else lg)
    return torch.cat(out) if out else torch.empty(0)


def measure(lg, t, c):
    """lg [..., V] -> (logit(t) - logit(c), log P(t))."""
    m = (lg[..., t] - lg[..., c])
    lp = torch.log_softmax(lg, -1)[..., t]
    return m, lp


# --------------------------------------------------------------------------- checks


def coherence_check(C):
    """Real models: greedy answer to a plain question names Paris (the load is not garbage)."""
    ids = chat_ids(C.tok, "What is the capital of France? Answer in one word.")
    seq = ids.tolist()
    for _ in range(8):
        lg = last_logits(C, torch.tensor(seq)[None])
        seq.append(int(lg[0].argmax()))
    ans = C.tok.decode(seq[len(ids):], skip_special_tokens=True)
    return {"answer": ans, "ok": "paris" in ans.lower()}


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# --------------------------------------------------------------------------- stages


def stage_patch(C, items, out):
    rows_out, checks, layouts = [], {"identity": [], "tol": C.tol}, {}
    cells = bind.cells(C.n, C.band)
    probe_blocks = sorted({0, C.n // 2, C.n - 1})
    for it in items:
        t0 = time.time()
        P = build(C, it)
        layouts[it["id"]] = P.info
        src_st, lg_src = source_run(C, P.src)
        lg_tgt = last_logits(C, P.tgt[None])[0]
        (m_src, _), (m_tgt, _) = measure(lg_src, P.t_tok, P.c_tok), measure(lg_tgt, P.t_tok, P.c_tok)
        m_src, m_tgt = m_src.item(), m_tgt.item()
        D = m_src - m_tgt
        flagged = D < bind.MIN_FLIP
        rows, meta = [], []
        for site, pos in sites(P).items():
            for kind, start, layers in cells:
                rows.append({"patch": [(layers, pos)]})
                meta.append({"site": site, "kind": kind, "start": start, "layers": layers, "positions": pos})
        n_main = len(rows)
        rows.append({"keep": True})                                             # nothing patched
        rows += [{"patch": [([l], list(range(P.T)))], "keep": True} for l in probe_blocks]  # everything at l
        lg = run_rows(C, P.tgt, rows, src_st)
        m, _ = measure(lg, P.t_tok, P.c_tok)
        d_none = maxdiff(lg[n_main], lg_tgt)
        d_all = [maxdiff(lg[n_main + 1 + j], lg_src) for j in range(len(probe_blocks))]
        checks["identity"].append({"item": it["id"], "patch_nothing_vs_target": d_none,
                                   "patch_all_vs_source": dict(zip(map(str, probe_blocks), d_all))})
        assert d_none <= C.tol, f"{it['id']}: patching nothing != target run ({d_none})"
        assert max(d_all) <= C.tol, f"{it['id']}: patching every position != source run ({d_all})"
        base = {"stage": "patch", "model": C.A.model, "item": it["id"], "tier": it["tier"], "no_site": it["no_site"],
                "m_src": m_src, "m_tgt": m_tgt, "D": D, "flagged": flagged}
        for mm, mt in zip(m[:n_main].tolist(), meta):
            rows_out.append({**base, **mt, "cond": "patch", "m": mm, "share": bind.share(mm, m_tgt, m_src)})
        log(f"patch {it['id']}: m_src {m_src:.2f} m_tgt {m_tgt:.2f} D {D:.2f}{' FLAGGED' if flagged else ''}; "
            f"subject {P.info['subj_toks']} ({P.alias}); {len(rows)} rows, {time.time() - t0:.1f}s; "
            f"identity {d_none:.1e} / {max(d_all):.1e}")
    return rows_out, checks, layouts


def stage_knockout(C, items, out):
    assert C.mc["eager"] or C.A.random_model, "knockout runs on Qwen2.5 only (eager attention)"
    rows_out, checks, layouts = [], {"identity": [], "tol": C.tol}, {}
    wins = bind.windows(C.n, C.band)
    cells = bind.cells(C.n, C.band)
    allL = list(range(C.n))
    for k, it in enumerate(items):
        t0 = time.time()
        P = build(C, it)
        layouts[it["id"]] = P.info
        src_st, lg_src = source_run(C, P.src)
        lg_tgt = last_logits(C, P.tgt[None])[0]
        lg_nm = last_logits(C, P.nomem[None])[0]
        (m_src, lp_ctx), (m_tgt, _) = measure(lg_src, P.t_tok, P.c_tok), measure(lg_tgt, P.t_tok, P.c_tok)
        _, lp_nm = measure(lg_nm, P.t_tok, P.c_tok)
        m_src, m_tgt, lp_ctx, lp_nm = m_src.item(), m_tgt.item(), lp_ctx.item(), lp_nm.item()
        G, D = lp_ctx - lp_nm, m_src - m_tgt
        ans = P.T - 1
        # --- knockout in the source run
        rows, meta = [], []
        targets = {"attr": [P.a]}
        if P.subj:
            targets["subj_span"] = list(P.subj)
        for tname, ks in targets.items():
            for s, layers in wins:
                rows.append({"block": [(layers, [ans], ks)]})
                meta.append({"site": tname, "kind": "window", "start": s, "layers": layers, "positions": ks})
        n_main = len(rows)
        rows.append({"hook_all": True, "keep": True})                         # empty knockout
        if k == 0:                                                            # the mask edit is effective
            rows.append({"block": [(allL, [ans], list(range(ans)))], "keep": True})
        lg = run_rows(C, P.src, rows)
        _, lp = measure(lg, P.t_tok, P.c_tok)
        d_empty = maxdiff(lg[n_main], lg_src)
        chk = {"item": it["id"], "empty_knockout_vs_unmodified": d_empty}
        assert d_empty <= C.tol, f"{it['id']}: empty knockout != unmodified run ({d_empty})"
        if k == 0:
            alone = last_logits(C, P.src[-1:][None])[0]
            chk["self_only_vs_token_alone"] = maxdiff(lg[n_main + 1], alone)
            assert chk["self_only_vs_token_alone"] <= C.tol, f"knockout mask edit not effective {chk}"
        base = {"model": C.A.model, "item": it["id"], "tier": it["tier"], "no_site": it["no_site"],
                "m_src": m_src, "m_tgt": m_tgt, "D": D, "lp_ctx": lp_ctx, "lp_nomem": lp_nm, "G": G,
                "flagged": G < bind.MIN_FLIP, "flagged_D": D < bind.MIN_FLIP}
        for v, mt in zip(lp[:n_main].tolist(), meta):
            rows_out.append({**base, "stage": "knockout", **mt, "cond": "knockout", "lp": v,
                             "share": bind.share(lp_ctx - v, 0.0, G)})
        # --- subject alone: target run, attribute blocked from every query after the experience sentence
        blk = (allL, list(range(P.q0, P.T)), [P.a])
        rows, meta = [{"block": [blk], "keep": True}], [{"site": "none", "kind": "sanity", "start": None,
                                                         "layers": [], "positions": []}]
        if P.subj:
            for site, pos in (("subj_last", [P.subj[-1]]), ("subj_span", list(P.subj))):
                for kind, start, layers in cells:
                    rows.append({"block": [blk], "patch": [(layers, pos)]})
                    meta.append({"site": site, "kind": kind, "start": start, "layers": layers, "positions": pos})
        lg = run_rows(C, P.tgt, rows, src_st)
        m, _ = measure(lg, P.t_tok, P.c_tok)
        m_blk = m[0].item()
        chk["suff_blocked_no_patch_share"] = bind.share(m_blk, m_tgt, m_src)
        chk["suff_blocked_lp_counter_drop"] = (torch.log_softmax(lg_tgt, -1)[P.c_tok]
                                               - torch.log_softmax(lg[0], -1)[P.c_tok]).item()
        checks["identity"].append(chk)
        sb = {**base, "flagged": D < bind.MIN_FLIP, "m_blocked": m_blk}
        for mm, mt in zip(m.tolist(), meta):
            rows_out.append({**sb, "stage": "suff", **mt, "cond": "suff", "m": mm,
                             "share": bind.share(mm, m_tgt, m_src), "share_net": (mm - m_blk) / D if D else float("nan")})
        log(f"knockout {it['id']}: lp_ctx {lp_ctx:.2f} lp_nomem {lp_nm:.2f} G {G:.2f}; D {D:.2f}; "
            f"blocked-no-patch share {chk['suff_blocked_no_patch_share']:.2f}; {time.time() - t0:.1f}s")
    return rows_out, checks, layouts


# ------------------------------------------------- (A) EXPLORATORY: the prefix subject mention


def stage_prefix(C, items, out):
    """EXPLORATORY, NOT PRE-REGISTERED. The teacher-forced answer prefix "Your dog's name is" has its own
    subject mention; stage 0 only tested the question's. Same measures as stage 0 at site subj_prefix (last
    token and span): patching source -> target (single blocks and bands, both models) and, on Qwen2.5, the
    knockout answer -> subj_prefix span (windows of 5 layers). Says whether that token relays the value; it
    does not change the stage-0 verdict."""
    rows_out, checks, layouts = [], {"identity": [], "tol": C.tol, "exploratory": True}, {}
    cells, wins = bind.cells(C.n, C.band), bind.windows(C.n, C.band)
    do_ko = C.mc["eager"] or C.A.random_model
    for it in items:
        t0 = time.time()
        P = build(C, it)
        layouts[it["id"]] = P.info
        if not P.psubj:
            log(f"prefix {it['id']}: no subject alias in the prefix {it['prefix']!r}; skipped")
            continue
        src_st, lg_src = source_run(C, P.src)
        lg_tgt = last_logits(C, P.tgt[None])[0]
        (m_src, lp_ctx), (m_tgt, _) = measure(lg_src, P.t_tok, P.c_tok), measure(lg_tgt, P.t_tok, P.c_tok)
        m_src, m_tgt, lp_ctx = m_src.item(), m_tgt.item(), lp_ctx.item()
        D = m_src - m_tgt
        rows, meta = [], []
        for site, pos in (("subj_prefix_last", [P.psubj[-1]]), ("subj_prefix_span", list(P.psubj))):
            for kind, start, layers in cells:
                rows.append({"patch": [(layers, pos)]})
                meta.append({"site": site, "kind": kind, "start": start, "layers": layers, "positions": pos})
        n_main = len(rows)
        rows.append({})
        tc = [P.t_tok, P.c_tok]
        lgm = run_rows(C, P.tgt, rows, src_st, reduce=lambda lg: lg[:, tc])
        m = (lgm[:, 0] - lgm[:, 1]).tolist()
        chk = {"item": it["id"], "patch_nothing_vs_target_margin": abs(m[n_main] - m_tgt)}
        assert chk["patch_nothing_vs_target_margin"] <= C.tol, chk
        base = {"model": C.A.model, "item": it["id"], "tier": it["tier"], "no_site": it["no_site"], "m_src": m_src,
                "m_tgt": m_tgt, "D": D, "flagged": D < bind.MIN_FLIP, "exploratory": True}
        for mm, mt in zip(m[:n_main], meta):
            rows_out.append({**base, "stage": "prefix_patch", **mt, "m": mm, "share": bind.share(mm, m_tgt, m_src)})
        if do_ko:
            _, lp_nm = measure(last_logits(C, P.nomem[None])[0], P.t_tok, P.c_tok)
            G = lp_ctx - lp_nm.item()
            krows = [{"block": [(layers, [P.T - 1], list(P.psubj))]} for _, layers in wins] + [{"hook_all": True}]
            lp = run_rows(C, P.src, krows, reduce=lambda lg: torch.log_softmax(lg, -1)[:, P.t_tok]).tolist()
            chk["empty_knockout_vs_unmodified_lp"] = abs(lp[-1] - lp_ctx)
            assert chk["empty_knockout_vs_unmodified_lp"] <= C.tol, chk
            for (s, layers), v in zip(wins, lp[:-1]):
                rows_out.append({**base, "stage": "prefix_ko", "site": "subj_prefix_span", "kind": "window", "start": s,
                                 "layers": layers, "positions": list(P.psubj), "lp": v, "lp_ctx": lp_ctx,
                                 "lp_nomem": lp_nm.item(), "G": G, "flagged": G < bind.MIN_FLIP,
                                 "share": bind.share(lp_ctx - v, 0.0, G)})
        checks["identity"].append(chk)
        log(f"prefix {it['id']}: prefix subject {P.info['prefix_subj_toks']} at {P.psubj}; {len(rows)} patch rows; "
            f"{time.time() - t0:.1f}s")
    return rows_out, checks, layouts


# ------------------------------------------------- (B) harvest + analytic pre-pass (pre-registered)


def analytic_sets(C, A_cfg, tiny=False):
    ref = A_cfg["ref_blocks"]
    singles = bind.map_sets([[l] for l in A_cfg["singles"]], ref, C.n, cv1.map_layers)
    bands = bind.map_sets([list(range(a, b + 1)) for a, b in A_cfg["bands"]], ref, C.n, cv1.map_layers)
    alphas = list(A_cfg["alphas"])
    if tiny:
        singles, bands, alphas = singles[:1], bands[:1], [1.0]
    write_range = (min(l for s in singles for l in s), max(l for s in singles for l in s))
    return singles + bands, alphas, write_range


def user_mask(T, toks, dev):
    u = torch.zeros(T, dtype=torch.bool, device=dev)
    u[toks] = True
    return u


def calibrate_generic(C, A_cfg, key_layers, n_generic):
    """Generic prompts (no memory): median residual norm of every block over user-text tokens (harvest norm
    check) and, at the key layers, mu + PCA-256 whitening and the generic read keys (G-key thresholds)."""
    prompts = [l.strip() for l in open(REPO / A_cfg["generic"]) if l.strip()][:n_generic]
    norms, X, per = defaultdict(list), defaultdict(list), []
    for t in prompts:
        ids = chat_ids(C.tok, t)
        ut = bind.subject_sites(C.tok, ids, t, [])["user_tokens"]
        st, _ = source_run(C, ids)
        for l in range(C.n):
            norms[l].append(st[l][ut].float().norm(dim=-1).cpu())
        for l in key_layers:
            X[l].append(st[l][ut].float())
        per.append({"user": user_mask(len(ids), ut, C.dev), "first": ut[0], "H": {l: st[l] for l in key_layers}})
    med = {l: torch.cat(norms[l]).median().item() for l in range(C.n)}
    w = A_cfg["white"]
    mu, W, gk = {}, {}, {}
    for l in key_layers:
        Xl = torch.cat(X[l])
        mu[l] = Xl.mean(0)
        W[l] = bind.whitening(Xl - mu[l], w["r"], w["eps"]).to(C.dev)
        gk[l] = torch.cat([bind.pooled_keys(p["H"][l], p["user"], mu[l], W[l])[p["first"]:] for p in per])
    return SimpleNamespace(med=med, mu=mu, W=W, gen_keys=gk, n=len(prompts))


def harvest(C, it, key_layers):
    """H1 (subject delta from the harvest follow-ups, last subject token, every block), H2 (minimal replay,
    second mention), H3 (H2's raw `with` state, rescaled to H2's delta norm per block). Returns
    ({variant: [n_blocks, d]}, info) with the without-run subject-span states for G-key."""
    tok, exp, allL = C.tok, it["experience"], range(C.n)
    info = {"checks": [], "H1_each": [], "key_states": []}
    for fu in it["harvest_followups"]:
        uw = f"{exp} {fu}"
        ids_w, ids_o = chat_ids(tok, uw), chat_ids(tok, fu)
        sw = bind.subject_sites(tok, ids_w, uw, it["aliases"], user_from=len(exp) + 1)
        so = bind.subject_sites(tok, ids_o, fu, it["aliases"])
        ck = {"variant": "H1", "text": fu, "subject_found": bool(sw["main"] and so["main"])}
        info["checks"].append(ck)
        if not ck["subject_found"]:
            assert it["tier"] != "primary", f"{it['id']}: no subject in harvest follow-up {fu!r}"
            continue
        pw, po = sw["main"][0], so["main"][0]
        ck.update(pos_with=pw, pos_without=po, token=tok.decode([int(ids_w[pw])]), same_token=int(ids_w[pw]) == int(ids_o[po]))
        assert ck["same_token"], f"{it['id']}: subject token differs with / without {ck}"
        st_w, _ = source_run(C, ids_w)
        st_o, _ = source_run(C, ids_o)
        info["H1_each"].append(torch.stack([st_w[l][pw] - st_o[l][po] for l in allL]).float())
        info["key_states"].append({l: st_o[l][so["spans"][0]] for l in key_layers})
    V = {}
    if info["H1_each"]:
        V["H1"] = torch.stack(info["H1_each"]).mean(0)
    if it.get("h2"):
        wo, mention = it["h2"]["without"], it["h2"]["mention"]
        uw, uo = f"{exp} {mention}", f"{wo} {mention}"
        ids_w, ids_o = chat_ids(tok, uw), chat_ids(tok, uo)
        sw = bind.subject_sites(tok, ids_w, uw, it["aliases"], user_from=len(exp) + 1)
        so = bind.subject_sites(tok, ids_o, uo, it["aliases"], user_from=len(wo) + 1)
        ck = {"variant": "H2", "text": uw, "without": uo, "subject_found": bool(sw["main"] and so["main"])}
        info["checks"].append(ck)
        assert ck["subject_found"] or it["tier"] != "primary", f"{it['id']}: no second mention in {uw!r}"
        if ck["subject_found"]:
            pw, po = sw["main"][0], so["main"][0]
            ck.update(pos_with=pw, pos_without=po, token=tok.decode([int(ids_w[pw])]),
                      same_token=int(ids_w[pw]) == int(ids_o[po]))
            assert ck["same_token"], f"{it['id']}: H2 subject token differs {ck}"
            st_w, _ = source_run(C, ids_w)
            st_o, _ = source_run(C, ids_o)
            V["H2"] = torch.stack([st_w[l][pw] - st_o[l][po] for l in allL]).float()
            raw = torch.stack([st_w[l][pw] for l in allL]).float()
            V["H3"] = bind.norm_match(raw, V["H2"])
            info["H3_raw_norm"] = raw.norm(dim=-1).tolist()
    return V, info


def harvest_checks(it, V, info, med, factor):
    """Finite, and |delta| <= factor x the median residual norm at every block (H1 per follow-up and mean,
    H2). Fails the job on violation."""
    out = []
    medv = torch.tensor([med[l] for l in range(len(med))])
    named = [(f"H1[{j}]", d) for j, d in enumerate(info["H1_each"])] + [(k, V[k]) for k in ("H1", "H2") if k in V]
    for name, d in named:
        ratio = (d.norm(dim=-1).cpu() / medv)
        r = {"item": it["id"], "vec": name, "finite": bool(torch.isfinite(d).all()), "max_ratio": ratio.max().item(),
             "argmax_block": int(ratio.argmax())}
        out.append(r)
        assert r["finite"], f"{it['id']} {name}: non-finite delta"
        assert r["max_ratio"] <= factor, f"{it['id']} {name}: |delta| / median = {r['max_ratio']:.1f} > {factor}"
    if "H3" in V:
        assert bool(torch.isfinite(V["H3"]).all()), f"{it['id']} H3: non-finite"
    return out


def lens_ranks(C, X, t):
    """Logit lens of X [n_blocks, d] (final norm + output layer) -> target rank per block (1 = top)."""
    lg = C.lm_head(C.body.norm(X.to(device=C.dev, dtype=C.lm_head.weight.dtype)[:, None]))[:, 0].float()
    return ((lg > lg[:, t:t + 1]).sum(1) + 1).tolist()


def rand_dirs(C, iid, seed):
    g = torch.Generator().manual_seed(seed + zlib.crc32(iid.encode()) % 100000)
    u = torch.randn(C.n, C.d, generator=g)
    return (u / u.norm(dim=-1, keepdim=True)).to(C.dev)


def analytic_item(C, it, V, Vp, U, lsets, alphas):
    """Teacher-forced pre-pass for one item: the measure prompt (related question + measure prefix: log P
    and specificity of the target) and the balanced yes/no probes (first-token log P(Yes), log P(No)), per
    condition: nomem, ctx (experience in the prompt), and for every (variant, layer set, alpha) the subject
    write at site main / plus1 / gen (measure prompt only; gen = main on the probes) and the controls
    wrongpos / rand (matched norm, per block) / swap (the partner item's same-variant vector) at main."""
    tok = C.tok
    t = int(text_ids(tok, it["target"])[0])
    foils = [int(text_ids(tok, f)[0]) for f in it["foils"]]
    yes, no = int(text_ids(tok, "Yes")[0]), int(text_ids(tok, "No")[0])
    pre = text_ids(tok, it["prefix"])
    q, exp = it["question"], it["experience"]
    prompts = [("m", None, torch.cat([chat_ids(tok, q), pre]), q, it["prefix"], torch.cat([chat_ids(tok, f"{exp} {q}"), pre]))]
    prompts += [(f"y{j}", rp["a"], chat_ids(tok, rp["text"]), rp["text"], None, chat_ids(tok, f"{exp} {rp['text']}"))
                for j, rp in enumerate(it["relation_probes"])]
    red_m = lambda lg: torch.stack([torch.log_softmax(lg, -1)[:, t], torch.log_softmax(lg, -1)[:, foils].mean(-1)], -1)
    red_y = lambda lg: torch.log_softmax(lg, -1)[:, [yes, no]]
    rows_out, sites_info = [], {}
    base = {"item": it["id"], "tier": it["tier"]}
    for pk, cons, ids, utext, prefix, ctx_ids in prompts:
        S = bind.subject_sites(tok, ids, utext, it["aliases"], prefix=prefix)
        sites_info[pk] = {k: S[k] for k in ("main", "plus1", "gen", "wrongpos")}
        red = red_m if pk == "m" else red_y
        rows, conds = [{}], [(None, None, None, "nomem")]
        if S["main"]:
            for var, vec_all in V.items():
                for ls in lsets:
                    for a in alphas:
                        vec = {l: a * vec_all[l] for l in ls}
                        sites = [("main", S["main"]), ("plus1", S["plus1"])] + ([("gen", S["gen"])] if prefix else [])
                        sites += [("wrongpos", S["wrongpos"])]
                        for sname, pos in sites:
                            rows.append({"add": [(ls, pos, vec)]})
                            conds.append((var, ls, a, sname))
                        nrm = {l: a * U[l] * vec_all[l].norm() for l in ls}
                        rows.append({"add": [(ls, S["main"], nrm)]})
                        conds.append((var, ls, a, "rand"))
                        if var in Vp:
                            rows.append({"add": [(ls, S["main"], {l: a * Vp[var][l] for l in ls})]})
                            conds.append((var, ls, a, "swap"))
        vals = run_rows(C, ids, rows, reduce=red).tolist()
        ctx = red(last_logits(C, ctx_ids[None]))[0].tolist()
        for (var, ls, a, site), v in [(c, v) for c, v in zip(conds, vals)] + [((None, None, None, "ctx"), ctx)]:
            rows_out.append({**base, "pk": pk, "consistent": cons, "variant": var,
                             "lset": bind.lset_label(ls) if ls else None, "alpha": a, "site": site,
                             "v0": v[0], "v1": v[1], "no_site": not S["main"]})
    return rows_out, sites_info


def gkey_item(C, it, info, cal, key_layers, thr_q, unrel):
    """G-key (reported): stored keys = whitened subject-span keys of the H1 without runs; the read key is the
    pooled running key (core_v1's pooled_w256); threshold = the q0.95 generic match. Does the gate open at the
    question's subject, and how often on unrelated prompts?"""
    if not info["key_states"]:
        return None
    tok = C.tok
    ids = torch.cat([chat_ids(tok, it["question"]), text_ids(tok, it["prefix"])])
    S = bind.subject_sites(tok, ids, it["question"], it["aliases"], prefix=it["prefix"])
    if not S["main"]:
        return None
    st, _ = source_run(C, ids)
    u = user_mask(len(ids), S["user_tokens"], C.dev)
    out = {}
    for l in key_layers:
        K = torch.stack([bind.subject_key(ks[l], cal.mu[l], cal.W[l]) for ks in info["key_states"]])
        thr = bind.gate_threshold(cal.gen_keys[l], K, thr_q)
        rk = bind.pooled_keys(st[l], u, cal.mu[l], cal.W[l])
        mt = (rk @ K.T).max(-1).values
        p = S["main"][0]
        ut = S["user_tokens"]
        fire = [bool(((ur["keys"][l] @ K.T).max(-1).values > thr).any()) for ur in unrel]
        out[str(l)] = {"thr": thr, "match_subject": mt[p].item(), "open_subject": bool(mt[p] > thr),
                       "open_frac_user": (mt[ut] > thr).float().mean().item(),
                       "unrel_fire_rate": sum(fire) / max(len(fire), 1)}
    return out


def stage_analytic(C, items, out, all_items):
    """Harvest (H1 / H2 / H3) + the analytic pre-pass (pre-registered; spec sections 4-6, PREREG.md)."""
    A_cfg = C.cfg["analytic"]
    lsets, alphas, wr = analytic_sets(C, A_cfg, C.A.tiny)
    key_layers = sorted({s[0] for s in lsets if len(s) == 1})
    n_gen = 16 if C.A.tiny else A_cfg["n_generic"]
    t0 = time.time()
    cal = calibrate_generic(C, A_cfg, key_layers, n_gen)
    log(f"analytic: layer sets {[bind.lset_label(s) for s in lsets]}, alphas {alphas}, key layers {key_layers}, "
        f"write range {wr}; generic calibration over {cal.n} prompts in {time.time() - t0:.0f}s")
    L = itm.load_items(REPO / C.cfg["items"], REPO / C.cfg["bench"])
    unrel_txt = L["unrelated"][:5 if C.A.tiny else len(L["unrelated"])]
    unrel = []
    for t in unrel_txt:
        ids = chat_ids(C.tok, t)
        ut = bind.subject_sites(C.tok, ids, t, [])["user_tokens"]
        st, _ = source_run(C, ids)
        u = user_mask(len(ids), ut, C.dev)
        unrel.append({"keys": {l: bind.pooled_keys(st[l], u, cal.mu[l], cal.W[l])[ut[0]:] for l in key_layers}})
    by_id = {it["id"]: it for it in all_items}
    need = {it["id"] for it in items} | {it["swap"] for it in items}
    V, infos, hchecks, lens = {}, {}, [], []
    for iid in [i for i in by_id if i in need]:
        it = by_id[iid]
        V[iid], infos[iid] = harvest(C, it, key_layers)
        hchecks += harvest_checks(it, V[iid], infos[iid], cal.med, A_cfg["norm_check"])
        t = int(text_ids(C.tok, it["target"])[0])
        for var, X in V[iid].items():
            rk = lens_ranks(C, X, t)
            lens.append({"item": iid, "tier": it["tier"], "variant": var, "ranks": rk,
                         "min_rank_write_range": min(rk[wr[0]:wr[1] + 1]),
                         "flag": min(rk[wr[0]:wr[1] + 1]) <= A_cfg["lens_topk"]})
        for k in V[iid]:
            V[iid][k] = V[iid][k].to(C.dev)
    log(f"harvest done ({len(V)} items) in {(time.time() - t0) / 60:.1f} min")
    rows_out, sites, gkey = [], {}, {}
    for k, it in enumerate(items):
        t1 = time.time()
        U = rand_dirs(C, it["id"], A_cfg["rand_seed"])
        r, sites[it["id"]] = analytic_item(C, it, V[it["id"]], V.get(it["swap"], {}), U, lsets, alphas)
        rows_out += r
        gkey[it["id"]] = gkey_item(C, it, infos[it["id"]], cal, key_layers, A_cfg["white"]["thr_q"], unrel)
        el = time.time() - t0
        log(f"analytic {it['id']}: {len(r)} rows in {time.time() - t1:.0f}s; variants {list(V[it['id']])}; "
            f"elapsed {el / 60:.1f} min, eta {(time.time() - t1) * (len(items) - k - 1) / 60:.1f} min")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"V": {i: {k: v.cpu() for k, v in d.items()} for i, d in V.items()}, "med": cal.med,
                "lsets": lsets, "alphas": alphas, "key_layers": key_layers, "write_range": wr}, out / "state.pt")
    write_jsonl_gz(out / "lens.jsonl.gz", lens)
    json.dump({"harvest": {i: {"checks": infos[i]["checks"], "H3_raw_norm": infos[i].get("H3_raw_norm")} for i in infos},
               "norm_checks": hchecks, "sites": sites, "gkey": gkey, "median_norm": cal.med,
               "write_range": wr, "lsets": [bind.lset_label(s) for s in lsets], "alphas": alphas},
              open(out / "harvest.json", "w"), indent=1, default=str)
    checks = {"harvest_norm_max_ratio": max((h["max_ratio"] for h in hchecks), default=float("nan")),
              "harvest_all_finite": all(h["finite"] for h in hchecks),
              "subject_found": {i: [c["subject_found"] for c in infos[i]["checks"]] for i in infos},
              "lens_flags": [f"{x['item']}:{x['variant']}" for x in lens if x["flag"]]}
    return rows_out, checks, {}


# --------------------------------------------------------------------------- report


def _cell_label(r):
    return f"{r['kind']} {r['start']}" if r["kind"] != "sanity" else "sanity"


def curves_from(rows, stage, site, tier, n_boot, seed, field="share", exclude_flagged=True):
    per = {}
    for r in rows:
        if r["stage"] != stage or r["site"] != site or r["tier"] != tier:
            continue
        if exclude_flagged and r.get("flagged"):
            continue
        per.setdefault(_cell_label(r), {})[r["item"]] = r[field]
    return bind.curve(per, n_boot, seed)


def fmt(s):
    if not s or s.get("n", 0) == 0 or math.isnan(s["mean"]):
        return "–"
    return f"{s['mean']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"


def table(title, cvs, labels):
    names = list(cvs)
    lines = [f"**{title}**", "", "| cell | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for lab in labels:
        lines.append(f"| {lab} | " + " | ".join(fmt(cvs[n].get(lab)) for n in names) + " |")
    return lines + [""]


def report(out, cfg, model_key):
    out = Path(out)
    rows, checks, confs = [], {}, {}
    for st in ("patch", "knockout"):
        f = out / st / "results.jsonl.gz"
        if f.exists():
            rows += read_jsonl_gz(f)
            checks[st] = json.load(open(out / st / "checks.json"))
            confs[st] = json.load(open(out / st / "config.json"))
    if not rows:
        return None
    nb, seed = cfg["n_boot"], cfg["seeds"]["bootstrap"]
    have_ko = any(r["stage"] == "knockout" for r in rows)
    is_qwen25 = model_key == "qwen25_15b"
    L = [f"# dream_bind stage 0: {model_key}", ""]
    c0 = next(iter(confs.values()))
    L += [f"Model {c0['model']} ({c0['dtype']}), commit {c0['env']['commit']}, GPU {c0['env']['gpu']}, "
          f"transformers {c0['env']['transformers']}. Stages: {', '.join(confs)}. Shares are means over items with "
          f"95% bootstrap CIs over items ({nb} replicates, seed {seed}); flagged items excluded.", ""]
    # items
    items = {}
    for r in rows:
        d = items.setdefault(r["item"], {"tier": r["tier"]})
        if r["stage"] == "patch":
            d.update(D=r["D"], m_src=r["m_src"], m_tgt=r["m_tgt"], flagged_D=r["flagged"])
        if r["stage"] == "knockout":
            d.update(G=r["G"], lp_ctx=r["lp_ctx"], lp_nomem=r["lp_nomem"], flagged_G=r["flagged"])
    lay = {}
    for c in confs.values():
        lay.update(c.get("layouts", {}))
    L += ["## Items", "", "| item | tier | subject tokens (question) | attr | D = m_src − m_tgt | G = ctx gain (nats) | flagged |",
          "|---|---|---|---|---|---|---|"]
    for iid, d in items.items():
        ly = lay.get(iid, {})
        fl = [k for k in ("flagged_D", "flagged_G") if d.get(k)]
        L.append(f"| {iid} | {d['tier']} | {ly.get('subj_toks') or 'none'} | {ly.get('attr_src')}→{ly.get('attr_tgt')} | "
                 f"{d.get('D', float('nan')):.2f} | {d.get('G', float('nan')):.2f} | {', '.join(fl) or ''} |")
    L.append("")
    nflag = sum(1 for d in items.values() if d["tier"] == "primary" and (d.get("flagged_D") or d.get("flagged_G")))
    # curves
    cv = {}
    for tier in ("primary", "secondary"):
        for site in ("subj_last", "subj_span", "attr"):
            cv[(tier, "patch", site)] = curves_from(rows, "patch", site, tier, nb, seed)
            cv[(tier, "knockout", site)] = curves_from(rows, "knockout", site, tier, nb, seed)
            cv[(tier, "suff", site)] = curves_from(rows, "suff", site, tier, nb, seed)
            cv[(tier, "suff_net", site)] = curves_from(rows, "suff", site, tier, nb, seed, field="share_net")
        cv[(tier, "suff", "none")] = curves_from(rows, "suff", "none", tier, nb, seed)
    # verdict
    P = lambda st, site: cv[("primary", st, site)]
    v = bind.verdict(P("patch", "subj_last"), P("patch", "attr"),
                     ko_subj=P("knockout", "subj_span") if have_ko else None,
                     ko_attr=P("knockout", "attr") if have_ko else None,
                     suff=P("suff", "subj_last") if have_ko else None,
                     suff_net=P("suff_net", "subj_last") if have_ko else None,
                     suff_sanity=P("suff", "none").get("sanity") if have_ko else None, n_flagged=nflag)
    if is_qwen25 and not have_ko:
        v["label"] = f"provisional: {v['label']} (patch only; knockout not run)"
    v["n_flagged_primary"] = nflag
    L += ["## Verdict (pre-registered, 7 primary items)", "", f"**{v['label']}**", ""]
    L += [f"- {x}" for x in v["reasons"]]
    if "suff_best" in v:
        L.append(f"- subject alone ({v['suff_readout']}): best {v['suff_best'].get('cell')} {fmt(v['suff_best'])}")
    L += [f"- rule checks: {json.dumps(v.get('checks'))}", f"- flagged primary items: {nflag}",
          "- Stage 0 alone can only flag *pointer-suspected*: the `subj` grid is skipped only if the analytic "
          "pre-pass also shows no gain (01-binding.md section 2).", ""]
    # sanity / checks
    L += ["## Checks", ""]
    for st, ch in checks.items():
        idt = ch["identity"]
        keys = sorted({k for x in idt for k in x if k != "item"})
        for k in keys:
            vals = []
            for x in idt:
                if k in x:
                    vals += list(x[k].values()) if isinstance(x[k], dict) else [x[k]]
            if k.startswith("suff_"):
                L.append(f"- {st}: {k}: per item " + ", ".join(f"{x['item']} {x[k]:.2f}" for x in idt if k in x))
            else:
                L.append(f"- {st}: {k}: max {max(vals):.2e} (tolerance {ch['tol']})")
        if "coherence" in ch:
            L.append(f"- {st}: coherence: {ch['coherence']}")
        if "peak_gpu_gb" in confs[st]:
            L.append(f"- {st}: peak GPU memory {confs[st]['peak_gpu_gb']} GB allocated, "
                     f"{confs[st].get('peak_gpu_reserved_gb')} GB reserved; peak host RSS "
                     f"{confs[st].get('peak_host_rss_gb')} GB")
    if have_ko:
        s = cv[("primary", "suff", "none")].get("sanity")
        L.append(f"- subject-alone sanity (attribute blocked, no patch; should be ≈ 0, |·| ≤ {bind.SANITY_ZERO}): "
                 f"{fmt(s)}")
    L.append("")
    # tables
    n = max(r["start"] for r in rows if r["kind"] == "block") + 1
    blocks = [f"block {l}" for l in range(n)]
    bands = [f"band {s}" for s in range(n - cfg["band"] + 1)]
    for tier in ("primary", "secondary"):
        L += [f"## Patching curves ({tier})", ""]
        pc = {s: cv[(tier, "patch", s)] for s in ("subj_last", "subj_span", "attr")}
        L += table("single blocks: flipped share of logit(target) − logit(counter)", pc, blocks)
        L += table("bands of 5 blocks (start)", pc, bands)
        if have_ko:
            kc = {s: cv[(tier, "knockout", s)] for s in ("subj_span", "attr")}
            L += [f"## Knockout ({tier}): share of ctx's log P(target) gain lost, answer → site blocked", ""]
            L += table("windows of 5 layers (start)", kc, [f"window {s}" for s in range(n - cfg["band"] + 1)])
            sc = {"subj_last raw": cv[(tier, "suff", "subj_last")], "subj_last net": cv[(tier, "suff_net", "subj_last")],
                  "subj_span raw": cv[(tier, "suff", "subj_span")], "subj_span net": cv[(tier, "suff_net", "subj_span")]}
            L += [f"## Subject alone ({tier}): attribute blocked after the experience, subject patched", ""]
            L += table("blocks and bands (net = minus the blocked-no-patch baseline)", sc, blocks + bands)
    # per item best
    L += ["## Per item (best cell)", "", "| item | tier | subj_last patch | attr patch | ko subj | ko attr | subj alone net |",
          "|---|---|---|---|---|---|---|"]
    for iid, d in items.items():
        def bst(stage, site, field="share"):
            vals = [(r[field], _cell_label(r)) for r in rows if r["item"] == iid and r["stage"] == stage
                    and r["site"] == site and not math.isnan(r[field])]
            return f"{max(vals)[0]:.2f} ({max(vals)[1]})" if vals else "–"
        L.append(f"| {iid} | {d['tier']} | {bst('patch', 'subj_last')} | {bst('patch', 'attr')} | "
                 f"{bst('knockout', 'subj_span')} | {bst('knockout', 'attr')} | {bst('suff', 'subj_last', 'share_net')} |")
    L.append("")
    (out / "report.md").write_text("\n".join(L))
    json.dump(v, open(out / "verdict.json", "w"), indent=2, default=str)
    log(f"report -> {out / 'report.md'}: {v['label']}")
    return v


def report_prefix(out, cfg, model_key):
    """(A) EXPLORATORY report -> <out>/prefix/report.md."""
    d = Path(out) / "prefix"
    if not (d / "results.jsonl.gz").exists():
        return None
    rows = read_jsonl_gz(d / "results.jsonl.gz")
    conf = json.load(open(d / "config.json"))
    checks = json.load(open(d / "checks.json"))
    nb, seed = cfg["n_boot"], cfg["seeds"]["bootstrap"]
    L = [f"# dream_bind (A), EXPLORATORY: the subject mention inside the answer prefix ({model_key})", "",
         "**Exploratory, not pre-registered.** Stage 0 patched and knocked out the subject in the *question*; the "
         "teacher-forced answer prefix (\"Your dog's name is\") has its own subject mention (\"Your dog\"). Same "
         "measures as stage 0 at that site. It does not change the stage-0 verdict; it says whether that token "
         "relays the value.", "",
         f"Model {conf['model']} ({conf['dtype']}), commit {conf['env']['commit']}, GPU {conf['env']['gpu']}. Means over "
         f"items, 95% bootstrap CIs over items ({nb}, seed {seed}).", ""]
    lay = conf.get("layouts", {})
    L += ["| item | tier | prefix subject tokens | positions |", "|---|---|---|---|"]
    for iid, ly in lay.items():
        L.append(f"| {iid} | {'primary' if iid in cfg['primary'] else 'secondary'} | {ly.get('prefix_subj_toks') or 'none'} | "
                 f"{ly.get('prefix_subj')} |")
    L.append("")
    have_ko = any(r["stage"] == "prefix_ko" for r in rows)
    n = max(r["start"] for r in rows if r["kind"] == "block") + 1
    blocks = [f"block {l}" for l in range(n)]
    bands = [f"band {s}" for s in range(n - cfg["band"] + 1)]
    summ = {}
    for tier in ("primary", "secondary"):
        pc = {s: curves_from(rows, "prefix_patch", s, tier, nb, seed) for s in ("subj_prefix_last", "subj_prefix_span")}
        kc = {"subj_prefix_span": curves_from(rows, "prefix_ko", "subj_prefix_span", tier, nb, seed)} if have_ko else {}
        summ[tier] = {k: dict(zip(("cell", "stats"), bind.best(v))) for k, v in {**{f"patch {s}": c for s, c in pc.items()},
                                                                             **{f"knockout {s}": c for s, c in kc.items()}}.items()}
        L += [f"## Summary ({tier}): best cell", ""]
        for k, v in summ[tier].items():
            L.append(f"- {k}: {v['cell']} {fmt(v['stats'])}")
        L.append("")
        L += table(f"patching ({tier}): flipped share, single blocks", pc, blocks)
        L += table(f"patching ({tier}): bands of 5", pc, bands)
        if have_ko:
            L += table(f"knockout ({tier}): answer -> prefix subject, share of ctx's gain lost",
                       kc, [f"window {s}" for s in range(n - cfg["band"] + 1)])
    L += ["## Checks", ""] + [f"- {json.dumps(c)}" for c in checks["identity"]] + [""]
    (d / "report.md").write_text("\n".join(L))
    json.dump({"exploratory": True, "summary": summ}, open(d / "summary.json", "w"), indent=2, default=str)
    log(f"prefix report -> {d / 'report.md'}")
    return summ


def push_reference(cfg, model_key):
    """core_v1's push@1 (mem@1) on the same session-2 measure prompt and yes/no probes, read from its saved run:
    {item: {dlp, dspec, dbal, dacc}} or None (with the reason)."""
    for p in cfg["analytic"].get("push_ref", {}).get(model_key, []):
        base = Path(p) if os.path.isabs(p) else REPO / p
        fa = base / "prep" / "facts_analytic.jsonl.gz"
        if not fa.exists():
            continue
        tf = [r for r in read_jsonl_gz(fa) if r["kind"] == "related" and r["cond"] in ("nomem", "mem@1")]
        out = {}
        for iid in {r["item"] for r in tf}:
            g = {r["cond"]: r for r in tf if r["item"] == iid}
            if "nomem" not in g or "mem@1" not in g:
                continue
            o = {"dlp": g["mem@1"]["lp_t"] - g["nomem"]["lp_t"], "dspec": g["mem@1"]["spec"] - g["nomem"]["spec"]}
            f = base / "gen" / "items" / f"{iid}.jsonl.gz"
            if f.exists():
                yn = [r for r in read_jsonl_gz(f) if r.get("sec") == "yn" and r["cond"] in ("nomem", "mem@1")]
                o.update(_yn_gain({(r["prompt"], r["cond"]): r["margin"] for r in yn},
                                  {r["prompt"]: r["consistent"] for r in yn}, "mem@1", "nomem"))
            out[iid] = o
        return out, str(base)
    return None, "core_v1 push@1 outputs not found (" + ", ".join(cfg["analytic"].get("push_ref", {}).get(model_key, [])) + ")"


def _yn_gain(margin, cons, cond, ref):
    """margin {(probe, cond): consistent-signed margin}, cons {probe: Yes|No} -> dbal = (dmY + dmN) / 2 and the
    balanced-accuracy gain (accuracy = margin > 0)."""
    dm, acc_c, acc_r = {"Yes": [], "No": []}, {"Yes": [], "No": []}, {"Yes": [], "No": []}
    for p, side in cons.items():
        if (p, cond) in margin and (p, ref) in margin:
            dm[side].append(margin[(p, cond)] - margin[(p, ref)])
            acc_c[side].append(float(margin[(p, cond)] > 0))
            acc_r[side].append(float(margin[(p, ref)] > 0))
    mean = lambda x: sum(x) / len(x) if x else float("nan")
    return {"dbal": (mean(dm["Yes"]) + mean(dm["No"])) / 2, "dmY": mean(dm["Yes"]), "dmN": mean(dm["No"]),
            "dacc": (mean(acc_c["Yes"]) + mean(acc_c["No"])) / 2 - (mean(acc_r["Yes"]) + mean(acc_r["No"])) / 2,
            "bal_acc": (mean(acc_c["Yes"]) + mean(acc_c["No"])) / 2}


def analytic_values(rows):
    """rows -> {cond key (variant, lset, alpha, site): {item: {dlp, dspec, dbal, dmY, dmN, dacc, bal_acc}}}; cond
    'ctx' = (None, None, None, 'ctx'). On the yes/no probes 'gen' = 'main' (no assistant prefix)."""
    idx = defaultdict(dict)        # (item, cond) -> {pk: row}
    cons = defaultdict(dict)
    for r in rows:
        key = (r["variant"], r["lset"], r["alpha"], r["site"])
        idx[(r["item"], key)][r["pk"]] = r
        if r["pk"] != "m":
            cons[r["item"]][r["pk"]] = r["consistent"]
            if r["site"] == "main":
                idx[(r["item"], (r["variant"], r["lset"], r["alpha"], "gen"))][r["pk"]] = r
    nom = {it: idx[(it, (None, None, None, "nomem"))] for it in cons}
    out = defaultdict(dict)
    for (it, key), pr in idx.items():
        if key[3] == "nomem" or it not in nom:
            continue
        b = nom[it]
        o = {"dlp": float("nan"), "dspec": float("nan")}
        if "m" in pr and "m" in b:
            o["dlp"] = pr["m"]["v0"] - b["m"]["v0"]
            o["dspec"] = (pr["m"]["v0"] - pr["m"]["v1"]) - (b["m"]["v0"] - b["m"]["v1"])
        sgn = lambda row, pk: (row["v0"] - row["v1"]) if cons[it][pk] == "Yes" else (row["v1"] - row["v0"])
        margin = {}
        for pk in cons[it]:
            if pk in pr and pk in b:
                margin[(pk, "c")], margin[(pk, "n")] = sgn(pr[pk], pk), sgn(b[pk], pk)
        o.update(_yn_gain(margin, {pk: cons[it][pk] for pk in cons[it] if (pk, "c") in margin}, "c", "n"))
        out[key][it] = o
    return out


def report_analytic(out, cfg, model_key):
    """(B) the pre-registered harvest + analytic pre-pass report -> <out>/analytic/{report.md, verdict.json}."""
    d = Path(out) / "analytic"
    if not (d / "results.jsonl.gz").exists():
        return None
    rows = read_jsonl_gz(d / "results.jsonl.gz")
    conf = json.load(open(d / "config.json"))
    H = json.load(open(d / "harvest.json"))
    lens = read_jsonl_gz(d / "lens.jsonl.gz")
    nb, seed = cfg["n_boot"], cfg["seeds"]["bootstrap"]
    vals = analytic_values(rows)
    tier_of = {r["item"]: r["tier"] for r in rows}
    variants = sorted({r["variant"] for r in rows if r["variant"]})
    lsets, alphas = H["lsets"], H["alphas"]
    metrics = ("dlp", "dspec", "dbal", "dacc")

    def stats(key, tier, m):
        per = {i: v[m] for i, v in vals.get(key, {}).items() if tier_of.get(i) == tier}
        return bind.ci(per, nb, seed)

    def pstats(key_a, per_b, tier, m="dlp"):
        a = {i: v[m] for i, v in vals.get(key_a, {}).items() if tier_of.get(i) == tier}
        diff = {i: a[i] - per_b[i] for i in a if i in per_b and not math.isnan(a[i]) and not math.isnan(per_b[i])}
        return bind.ci(diff, nb, seed)

    push, push_src = push_reference(cfg, model_key)
    L = [f"# dream_bind (B): harvest + analytic pre-pass ({model_key})", "",
         f"Pre-registered (PREREG.md, 'Later stages'). Teacher-forced, no generation. Model {conf['model']} "
         f"({conf['dtype']}), commit {conf['env']['commit']}, GPU {conf['env']['gpu']}. Session-2 prompt = the related "
         f"question + the measure prefix; yes/no = the balanced verify probes (first-token margins). Values are gains "
         f"vs nomem on the same prompt; means over items with 95% bootstrap CIs over items ({nb}, seed {seed}). "
         f"Layer sets {lsets}, alphas {alphas}.", ""]
    # ---- the pre-registered read-out
    cells = []
    for v in variants:
        for ls in lsets:
            for a in alphas:
                key = (v, ls, a, "main")
                s = {m: stats(key, "primary", m) for m in metrics}
                cells.append({"variant": v, "lset": ls, "alpha": a, **{m: s[m]["mean"] for m in metrics}, "_s": s})
    ok = [c for c in cells if not math.isnan(c["dlp"])]
    best = max(ok, key=lambda c: c["dlp"]) if ok else None
    ng = bind.no_gain(best["_s"]["dlp"] if best else None)
    pick = bind.analytic_pick([{k: c[k] for k in ("variant", "lset", "alpha", "dspec", "dbal", "dlp")} for c in ok])
    ctx = {m: stats((None, None, None, "ctx"), "primary", m) for m in metrics}
    L += ["## Verdict (pre-registered): does the subject write show a gain?", ""]
    if best:
        L += [f"**{'NO GAIN' if ng else 'GAIN'}**: the best cell (highest mean Δlog P(target), site main = G-match) is "
              f"**{best['variant']} @ {best['lset']} α={best['alpha']}**: Δlog P(target) {fmt(best['_s']['dlp'])}, "
              f"Δspec {fmt(best['_s']['dspec'])}, Δbal margin {fmt(best['_s']['dbal'])}, Δbal acc {fmt(best['_s']['dacc'])}.",
              "", f"- 'No gain' = that CI includes 0 or lies below it (PREREG). Ceiling (ctx): Δlog P {fmt(ctx['dlp'])}, "
              f"Δspec {fmt(ctx['dspec'])}, Δbal margin {fmt(ctx['dbal'])}, Δbal acc {fmt(ctx['dacc'])}.",
              "- Combined with stage 0 (pointer-suspected on this model): the `subj` grid is skipped only if this says "
              "NO GAIN.", ""]
    else:
        L += ["**undetermined** (no primary cell with a value)", ""]
    # ---- best per variant + controls
    L += ["## Best cell per variant, with the sites and controls (primary)", "",
          "Δ = gain vs nomem. 'main − ctrl' = paired difference in Δlog P(target) over items.", ""]
    focus = []
    for v in variants:
        bv = [c for c in ok if c["variant"] == v]
        if bv:
            focus.append(("best " + v, max(bv, key=lambda c: c["dlp"])))
    focus += [(f"pick {j + 1}", c) for j, c in enumerate(pick["picks"])]
    per_push = {i: o["dlp"] for i, o in (push or {}).items() if tier_of.get(i) == "primary"}
    for name, c in focus:
        key_of = lambda site: (c["variant"], c["lset"], c["alpha"], site)
        main_dlp = {i: v["dlp"] for i, v in vals.get(key_of("main"), {}).items() if tier_of.get(i) == "primary"}
        L += [f"**{name}: {c['variant']} @ {c['lset']} α={c['alpha']}**", "",
              "| condition | Δlog P(target) | Δspec | Δbal margin | Δbal acc | main − ctrl (Δlog P) |", "|---|---|---|---|---|---|"]
        for site in ("main", "plus1", "gen", "wrongpos", "rand", "swap"):
            s = {m: stats(key_of(site), "primary", m) for m in metrics}
            pd = "" if site == "main" else fmt(bind.ci({i: main_dlp[i] - v["dlp"] for i, v in vals.get(key_of(site), {}).items()
                                                        if i in main_dlp and not math.isnan(v["dlp"])}, nb, seed))
            L.append(f"| {site} | {fmt(s['dlp'])} | {fmt(s['dspec'])} | {fmt(s['dbal'])} | {fmt(s['dacc'])} | {pd} |")
        if push:
            s = {m: bind.ci({i: o.get(m, float('nan')) for i, o in push.items() if tier_of.get(i) == "primary"}, nb, seed)
                 for m in metrics}
            pd = fmt(bind.ci({i: main_dlp[i] - per_push[i] for i in main_dlp if i in per_push}, nb, seed))
            L.append(f"| push@1 (core_v1) | {fmt(s['dlp'])} | {fmt(s['dspec'])} | {fmt(s['dbal'])} | {fmt(s['dacc'])} | {pd} |")
        L.append(f"| ctx (ceiling) | {fmt(ctx['dlp'])} | {fmt(ctx['dspec'])} | {fmt(ctx['dbal'])} | {fmt(ctx['dacc'])} | |")
        L.append("")
    L.append(f"push@1: {'from ' + push_src if push else 'SKIPPED: ' + push_src}")
    L.append("")
    # ---- the analytic pick
    L += ["## Analytic pick (for the record; generation is not run)", "", pick["rule"], "",
          "| # | variant | layer set | α | Δspec | Δbal margin | Δlog P | score |", "|---|---|---|---|---|---|---|---|"]
    for j, c in enumerate(pick["picks"]):
        L.append(f"| {j + 1} | {c['variant']} | {c['lset']} | {c['alpha']} | {c['dspec']:.3f} | {c['dbal']:.3f} | "
                 f"{c['dlp']:.3f} | {c['score']:.1f} |")
    L.append("")
    # ---- full grid (main site)
    L += ["## Every cell, site main (primary)", "", "| variant | layer set | α | Δlog P(target) | Δspec | Δbal margin |",
          "|---|---|---|---|---|---|"]
    for c in cells:
        L.append(f"| {c['variant']} | {c['lset']} | {c['alpha']} | {fmt(c['_s']['dlp'])} | {fmt(c['_s']['dspec'])} | "
                 f"{fmt(c['_s']['dbal'])} |")
    L.append("")
    # ---- logit lens
    wr = H["write_range"]
    L += ["## Logit lens of the stored vectors", "",
          f"Target rank per block (1 = top) through the final norm + output layer. **Flag** = rank ≤ "
          f"{cfg['analytic']['lens_topk']} at an early-middle block (the write range, blocks {wr[0]}–{wr[1]}): a "
          "late-layer 'say the name' vector harvested again.", "",
          "| item | variant | min rank in write range | flag | ranks (every block) |", "|---|---|---|---|---|"]
    for x in lens:
        L.append(f"| {x['item']} | {x['variant']} | {x['min_rank_write_range']} | {'FLAG' if x['flag'] else ''} | "
                 f"{' '.join(str(r) for r in x['ranks'])} |")
    L.append("")
    # ---- G-key
    L += ["## G-key (reported): would the subject key alone open the gate?", "",
          "Stored key = whitened subject-span key of the H1 without runs; read key = core_v1's pooled running key; "
          "threshold = q0.95 generic match. open@subject = the gate is open at the question's subject token.", "",
          "| key layer | open@subject (primary) | mean match − thr | unrelated fire rate |", "|---|---|---|---|"]
    gk = {i: g for i, g in H["gkey"].items() if g and tier_of.get(i) == "primary"}
    for l in sorted({int(l) for g in gk.values() for l in g}):
        xs = [g[str(l)] for g in gk.values()]
        L.append(f"| {l} | {sum(x['open_subject'] for x in xs)}/{len(xs)} | "
                 f"{sum(x['match_subject'] - x['thr'] for x in xs) / len(xs):.3f} | "
                 f"{sum(x['unrel_fire_rate'] for x in xs) / len(xs):.2f} |")
    L.append("")
    # ---- secondary
    L += ["## Secondary items (reported separately; no verdict)", "",
          "| variant | best cell | Δlog P(target) | Δspec | Δbal margin | items with a site |", "|---|---|---|---|---|---|"]
    for v in variants:
        sc = []
        for ls in lsets:
            for a in alphas:
                key = (v, ls, a, "main")
                s = {m: stats(key, "secondary", m) for m in metrics}
                if not math.isnan(s["dlp"]["mean"]):
                    sc.append((s["dlp"]["mean"], ls, a, s))
        if sc:
            _, ls, a, s = max(sc, key=lambda x: x[0])
            L.append(f"| {v} | {ls} α={a} | {fmt(s['dlp'])} | {fmt(s['dspec'])} | {fmt(s['dbal'])} | {s['dlp']['n']} |")
    L.append("")
    if best:
        L += ["## Per item at the best cell", "", "| item | tier | main | wrongpos | rand | swap | ctx |", "|---|---|---|---|---|---|---|"]
        for i in sorted(tier_of, key=lambda i: (tier_of[i] != "primary", i)):
            g = lambda site: vals.get((best["variant"], best["lset"], best["alpha"], site), {}).get(i, {}).get("dlp", float("nan"))
            cx = vals.get((None, None, None, "ctx"), {}).get(i, {}).get("dlp", float("nan"))
            L.append(f"| {i} | {tier_of[i]} | {g('main'):.2f} | {g('wrongpos'):.2f} | {g('rand'):.2f} | {g('swap'):.2f} | {cx:.2f} |")
        L.append("")
    # ---- checks
    L += ["## Checks", "",
          f"- harvest: every delta finite; max |delta| / median residual norm = "
          f"{max((h['max_ratio'] for h in H['norm_checks']), default=float('nan')):.2f} (limit {cfg['analytic']['norm_check']})",
          "- subject found: " + "; ".join(f"{i} {[c['subject_found'] for c in h['checks']]}" for i, h in H["harvest"].items()),
          f"- peak GPU {conf.get('peak_gpu_gb')} GB allocated / {conf.get('peak_gpu_reserved_gb')} GB reserved; "
          f"host RSS {conf.get('peak_host_rss_gb')} GB; {conf.get('minutes')} min", ""]
    (d / "report.md").write_text("\n".join(L))
    v = {"no_gain": ng, "best": {k: best[k] for k in ("variant", "lset", "alpha")} if best else None,
         "best_dlp": best["_s"]["dlp"] if best else None, "ctx": ctx, "picks": pick["picks"],
         "push_ref": push_src, "lens_flags": [f"{x['item']}:{x['variant']}" for x in lens if x["flag"]]}
    json.dump(v, open(d / "verdict.json", "w"), indent=2, default=str)
    log(f"analytic report -> {d / 'report.md'}: {'NO GAIN' if ng else 'GAIN'}")
    return v


# =============================================================== arm R + generation (Qwen2.5; PREREG addendum)

TAG = "dream_bind"
RESID = ("subj1", "subj2", "ctrl:wrongpos", "ctrl:rand", "ctrl:swap", "ctrl:placebo")
RCONDS = ("R-full", "R-min", "R-orphan", "R-swap")
PRIMARY_CONDS = ("nomem", "ctx") + RCONDS + RESID
SECONDARY_CONDS = ("nomem", "ctx", "R-full", "subj1")
PAIRS = (("dog_name", "cat_name"), ("sister_name", "brother_name"), ("best_friend_name", "boss_name"))


def render_ids(tok, msgs):
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s, tok(s, add_special_tokens=False).input_ids


def turn_pieces(tok, exp, exchange=None):
    """Token pieces of the conversation [head][unrelated exchange U][experience user turn Seg][question turn]:
    head = everything before the first user turn (the system block), Seg = '<|im_start|>user\\n{exp}<|im_end|>\\n'.
    Asserts that head + U + Seg + question turn tokenises exactly as the whole rendered conversation, and that
    head + question turn is the plain (nomem) prompt."""
    q = "What is two plus two?"
    s_nm, ids_nm = render_ids(tok, [{"role": "user", "content": q}])
    msgs = ([{"role": "user", "content": exchange["user"]}, {"role": "assistant", "content": exchange["assistant"]}]
            if exchange else []) + [{"role": "user", "content": exp}, {"role": "user", "content": q}]
    s_all, ids_all = render_ids(tok, msgs)
    k = s_nm.index("<|im_start|>user")
    head, qt = s_nm[:k], s_nm[k:]
    assert s_all.startswith(head) and s_all.endswith(qt)
    mid = s_all[len(head):len(s_all) - len(qt)]
    seg = f"<|im_start|>user\n{exp}<|im_end|>\n"
    assert mid.endswith(seg), repr(mid[-80:])
    u = mid[:len(mid) - len(seg)]
    enc = lambda s: tok(s, add_special_tokens=False).input_ids if s else []
    H, U, Seg, Qt = enc(head), enc(u), enc(seg), enc(qt)
    assert H + U + Seg + Qt == list(ids_all), "turn pieces do not tokenise like the whole conversation"
    assert H + Qt == list(ids_nm), "head + question turn != the plain prompt"
    return SimpleNamespace(H=H, U=U, Seg=Seg)


def qt_ids(C, text):
    """The question turn of a plain prompt (the plain chat ids minus the head)."""
    ids = chat_ids(C.tok, text)
    assert ids[:len(C.head)].tolist() == C.head, "the prompt does not start with the template head"
    return ids[len(C.head):]


def run_cache(C, ids):
    """The cache object of a batch-1 run of `ids` (a fresh, empty cache of the right layer types if empty)."""
    if not ids:
        from transformers import DynamicCache
        return DynamicCache(config=C.model.config)
    return C.body(input_ids=torch.tensor(ids)[None].to(C.dev), use_cache=True).past_key_values


def is_hybrid(C):
    return "linear_attention" in (getattr(C.model.config, "layer_types", None) or [])


def harvest_R_hybrid(C, it, exchange):
    """Qwen3.5 (6 full-attention blocks, 18 Gated DeltaNet blocks): cache objects (PREREG addendum 3).
      R-full-state  the whole session cache after [head][experience turn]: K,V at the full blocks AND the DeltaNet
                    conv + recurrent states (prompt caching for hybrids; == the sentence as a prior user turn)
      R-full-kv6    full blocks: K,V of head + experience turn at their stored positions; DeltaNet blocks: the
                    state after the head only (the segment is invisible to them; they have no positions)
      R-min-kv6     full blocks: head + the attribute and the next token only; DeltaNet as R-full-kv6
      R-orphan-kv6  full blocks: head + experience turn harvested after the unrelated exchange (dropped);
                    DeltaNet as R-full-kv6; the question continues after the harvest positions
    Addendum 4 (what the DeltaNet state contributes):
      R-min-state   full blocks: head + the attribute and the next token (as R-min-kv6) PLUS the DeltaNet states
                    after the experience turn (a fixed-size state that summarises the whole sentence)
      R-state-only  full blocks: head only (no segment entries); DeltaNet states after the experience turn; the
                    question starts at n_head + n_seg, the position it has after the segment (as R-full-state)
      R-orphan-state R-min's two entries harvested after the unrelated exchange + experience, and the DeltaNet
                    state after exchange + experience (contaminated by the exchange); the question continues after
                    the harvest positions"""
    P, Po = turn_pieces(C.tok, it["experience"]), turn_pieces(C.tok, it["experience"], exchange)
    assert P.H == Po.H and P.Seg == Po.Seg
    t = int(text_ids(C.tok, it["target"])[0])
    a = [i for i, x in enumerate(P.Seg) if x == t]
    assert len(a) == 1, f"{it['id']}: attribute token found {len(a)} times in the experience turn"
    full, head, orph = run_cache(C, P.H + P.Seg), run_cache(C, P.H), run_cache(C, Po.H + Po.U + Po.Seg)
    nh, ns, nu = len(P.H), len(P.Seg), len(Po.U)
    k_full, s_full = bind.splice_keep("R-full", nh, ns, a[0])
    k_min, _ = bind.splice_keep("R-min", nh, ns, a[0])
    k_orph, s_orph = bind.splice_keep("R-orphan", nh, ns, a[0], nu)
    k_omin, _ = bind.splice_keep("R-orphan-min", nh, ns, a[0], nu)
    k_so, s_so = bind.splice_keep("R-state-only", nh, ns, a[0])
    out = {"R-full-state": (full, s_full), "R-full-kv6": (bind.kv6_cache(full, head, k_full), s_full),
           "R-min-kv6": (bind.kv6_cache(full, head, k_min), s_full),
           "R-orphan-kv6": (bind.kv6_cache(orph, head, k_orph), s_orph),
           "R-min-state": (bind.mix_cache(full, k_min, full), s_full),
           "R-state-only": (bind.mix_cache(full, k_so, full), s_so),
           "R-orphan-state": (bind.mix_cache(orph, k_omin, orph), s_orph)}
    # addendum 6b: the DeltaNet recurrent state vs the short-convolution buffer
    out["R-state-only-rec"] = (bind.split_linear(bind.mix_cache(full, k_so, full), head, "full", "head"), s_so)
    out["R-state-only-conv"] = (bind.split_linear(bind.mix_cache(full, k_so, full), head, "head", "full"), s_so)
    out["R-min-state-rec"] = (bind.split_linear(bind.mix_cache(full, k_min, full), head, "full", "head"), s_full)
    dec = lambda i: C.tok.decode([i])
    info = {"n_head": nh, "n_seg": ns, "n_unrel": nu, "attr_in_seg": a[0], "min_tokens": [dec(P.Seg[a[0]]), dec(P.Seg[a[0] + 1])],
            "starts": {v: st for v, (_, st) in out.items()},
            "layers": {v: bind.cache_summary(c) for v, (c, _) in out.items()},
            "head_layers": bind.cache_summary(head),
            "lin_equal_full": {v: bind.linear_states_equal(c, full) for v, (c, _) in out.items()},
            "lin_equal_orphan": {v: bind.linear_states_equal(c, orph) for v, (c, _) in out.items()},
            "restored": {v: _restored(c, full) for v, (c, _) in out.items()}}
    return out, info


def _restored(c, full):
    """Which DeltaNet parts of a reinstatement cache equal the after-experience state: {rec, conv} -> bool."""
    rec = conv = True
    for L, F in zip(c.layers, full.layers):
        if bind.is_linear_layer(L):
            for d, fd, key in ((L.recurrent_states, F.recurrent_states, "rec"), (L.conv_states, F.conv_states, "conv")):
                for j in fd:
                    same = d.get(j) is not None and fd[j] is not None and torch.equal(d[j], fd[j])
                    if key == "rec":
                        rec = rec and same
                    else:
                        conv = conv and same
    return {"rec": rec, "conv": conv}


def harvest_R(C, it, exchange):
    """The item's reinstatement caches: {variant: ([(K, V)] per layer, question start position)} for R-full,
    R-min and R-orphan (bind.splice_keep), harvested from the frozen model."""
    P, Po = turn_pieces(C.tok, it["experience"]), turn_pieces(C.tok, it["experience"], exchange)
    assert P.H == Po.H and P.Seg == Po.Seg
    t = int(text_ids(C.tok, it["target"])[0])
    a = [i for i, x in enumerate(P.Seg) if x == t]
    assert len(a) == 1, f"{it['id']}: attribute token found {len(a)} times in the experience turn"
    if is_hybrid(C):
        return harvest_R_hybrid(C, it, exchange)
    run = lambda ids: bind.kv_list(C.body(input_ids=torch.tensor(ids)[None].to(C.dev), use_cache=True).past_key_values)
    kv, kvo = run(P.H + P.Seg), run(Po.H + Po.U + Po.Seg)
    out = {}
    for var, src, nu in (("R-full", kv, 0), ("R-min", kv, 0), ("R-orphan", kvo, len(Po.U))):
        keep, start = bind.splice_keep(var, len(P.H), len(P.Seg), a[0], nu)
        ki = torch.tensor(keep, device=C.dev)
        out[var] = ([(k[:, :, ki].clone(), v[:, :, ki].clone()) for k, v in src], start)
    dec = lambda i: C.tok.decode([i])
    info = {"n_head": len(P.H), "n_seg": len(P.Seg), "n_unrel": len(Po.U), "attr_in_seg": a[0],
            "min_tokens": [dec(P.Seg[a[0]]), dec(P.Seg[a[0] + 1])],
            "starts": {v: s for v, (_, s) in out.items()}, "kept": {v: k[0][0].shape[2] for v, (k, _) in out.items()}}
    return out, info


def gen_batch(C, ids, rows, *, prefix=None, add=None, max_new, streams=None, first_ids=None):
    """KV-cached generation of B rows (core_v1.core.generate_batch's sampling and random-stream layout), with an
    optional reinstatement prefix (prefix = ([(K, V)] per layer, start position): the cache starts with the stored
    entries and the prompt's positions start at `start`) and an optional prefill injection plan (add = {layer:
    [(row, positions, vec)]}, applied to the prompt only; later tokens see it through the cache)."""
    from contextlib import nullcontext
    dev = C.dev
    B, T = len(rows), len(ids)
    start = prefix[1] if prefix else 0
    if prefix is None:
        cache = None if is_hybrid(C) else bind.make_cache([], B)
    elif isinstance(prefix[0], list):
        cache = bind.make_cache(prefix[0], B)
    else:
        cache = bind.expand_cache(prefix[0], B)
    x = ids[None].expand(B, -1).to(dev).contiguous()
    streams = streams or {}
    gens = {}
    for s, (seed, n_draw) in streams.items():
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed))
        gens[s] = (g, n_draw)
    greedy = torch.tensor([bool(r[0]) for r in rows], device=dev)
    by_stream = {}
    for b, (gr, st) in enumerate(rows):
        if not gr:
            assert st is not None and st[0] in gens and 1 <= st[1] < gens[st[0]][1], f"row {b}: bad stream {st}"
            by_stream.setdefault(st[0], ([], []))
            by_stream[st[0]][0].append(b)
            by_stream[st[0]][1].append(st[1])
    by_stream = {s: (torch.tensor(bs, device=dev), torch.tensor(js, device=dev)) for s, (bs, js) in by_stream.items()}
    eos_t = torch.tensor(C.eos_list, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    seen = torch.zeros(B, C.vocab, dtype=torch.bool, device=dev) if C.sampler.presence else None
    ar = torch.arange(B, device=dev)
    steps, first = [], None
    for t in range(max_new):
        pos = (torch.arange(start, start + T) if t == 0 else torch.tensor([start + T + t - 1]))[None].expand(B, -1).to(dev)
        with (bind.adding(C.blocks, add) if (t == 0 and add) else nullcontext()):
            o = C.body(input_ids=x, past_key_values=cache, position_ids=pos, use_cache=True)
        cache = o.past_key_values
        logits = C.lm_head(o.last_hidden_state[:, -1]).float()
        if not bool(torch.isfinite(logits[~done]).all()):
            raise FloatingPointError(f"non-finite logits at step {t}")
        if t == 0 and first_ids is not None:
            first = torch.log_softmax(logits, -1)[:, first_ids].cpu()
        u = torch.zeros(B, 1, dtype=torch.float64, device=dev)
        draws = {s: torch.rand(n, 1, generator=g, device=dev, dtype=torch.float64) for s, (g, n) in gens.items()}
        for s, (bs, js) in by_stream.items():
            u[bs] = draws[s][js]
        nxt = logits.argmax(-1)
        if not bool(greedy.all()):
            nxt = torch.where(greedy, nxt, C.sampler(logits, u, seen))
        active = ~done
        nxt = torch.where(done, eos_t[0], nxt)
        steps.append(nxt)
        if seen is not None:
            seen[ar, nxt] |= active
        done = done | (active & torch.isin(nxt, eos_t))
        if bool(done.all()):
            break
        x = nxt[:, None]
    toks = torch.stack(steps, 1).cpu().tolist()
    eset = set(C.eos_list)
    out = []
    for b, row in enumerate(toks):
        k = next((i for i, tk in enumerate(row) if tk in eset), None)
        ids_ = row if k is None else row[:k]
        out.append({"ids": ids_, "n_tok": len(ids_), "capped": k is None,
                    "first": None if first is None else first[b].tolist()})
    return out


def gen_context(C, cfg):
    """Generation settings on the stage-0 context: eos ids, sampler, head tokens."""
    g = cfg["gen"]
    eos = {C.tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = C.tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != C.tok.unk_token_id:
            eos.add(i)
    ge = getattr(C.model.generation_config, "eos_token_id", None)
    eos |= set(ge if isinstance(ge, (list, tuple)) else [ge])
    C.eos_list = sorted(e for e in eos if e is not None)
    C.vocab = C.lm_head.out_features
    s = g["sampler"]
    C.sampler = cv1.qwen_sampler(s["temp"], s["top_k"], s["top_p"], s["presence"])
    C.head = turn_pieces(C.tok, "x").H
    gs = getattr(C.A, "gen_set", None)
    gc = (C.mc.get(f"gen_conds_{gs}") if gs else C.mc.get("gen_conds")) or {}
    assert not gs or gc, f"no gen_conds_{gs} for this model"
    C.gc = SimpleNamespace(primary=list(gc.get("primary", PRIMARY_CONDS)), secondary=list(gc.get("secondary", SECONDARY_CONDS)),
                           seed2=list(gc.get("seed2", g["seed2"])), leak=gc.get("leak", "R-full"),
                           swap_from=gc.get("swap_from", "R-full"), check=gc.get("check", "R-full"),
                           swap_name=gc.get("swap_name"),
                           b5_ref=gc.get("b5_ref"))
    C.yes, C.no = int(text_ids(C.tok, "Yes")[0]), int(text_ids(C.tok, "No")[0])


def item_memory(C, cfg, it, V, Vp, Vplac, U, R, Rp):
    """Per-condition specs of one item: residual conditions -> (site key, {layer: vec}); R conditions ->
    (cache, start). The swap condition is the partner's `swap_from` reinstatement under this item's gate."""
    swap_name = C.gc.swap_name or ("R-swap" + C.gc.swap_from[len("R-full"):])
    rr = {**R, swap_name: Rp[C.gc.swap_from] if Rp else None}
    if V is None:
        return SimpleNamespace(add={}, R={k: v for k, v in rr.items() if v})
    cells = {c["name"]: {**c, "layers": [l for l in c["layers"] if l < C.n]} for c in cfg["gen"]["subj_cells"]}
    if not C.A.random_model:
        assert all(len(c["layers"]) == len(c0["layers"]) for c, c0 in zip(cells.values(), cfg["gen"]["subj_cells"]))
    c1 = cells["subj1"]
    vec = lambda VV, c: {l: c["alpha"] * VV[c["variant"]][l] for l in c["layers"]} if VV and c["variant"] in VV else None
    add = {"subj1": ("main", vec(V, c1)), "subj2": ("main", vec(V, cells["subj2"])),
           "ctrl:wrongpos": ("wrongpos", vec(V, c1)),
           "ctrl:rand": ("main", {l: c1["alpha"] * U[l] * V[c1["variant"]][l].norm() for l in c1["layers"]}
                         if c1["variant"] in V else None),
           "ctrl:swap": ("main", vec(Vp, c1)), "ctrl:placebo": ("main", vec(Vplac, c1))}
    return SimpleNamespace(add={k: v for k, v in add.items() if v[1] is not None}, R={k: v for k, v in rr.items() if v})


def row_layout(conds, seed2, S):
    """[(cond, seedset, j)]: 1 greedy + S samples (seed set 1), + S samples (seed set 2) for seed2 conditions."""
    out = []
    for c in conds:
        out += [(c, 1, j) for j in range(S + 1)]
        if c in seed2:
            out += [(c, 2, j) for j in range(1, S + 1)]
    return out


def run_prompt(C, cfg, it, M, conds, text, *, S, cap, streams, first_ids=None):
    """Every condition of one prompt (same random streams). Returns {cond: [(seedset, j, out | None)]} where
    None = gate closed (the answer is nomem's by construction) and the gate info."""
    tok, g = C.tok, cfg["gen"]
    ids = chat_ids(tok, text)
    site = bind.subject_sites(tok, ids, text, it["aliases"])
    gate = bool(site["main"])
    lay = row_layout(conds, C.gc.seed2, S)
    res = defaultdict(list)
    rows_of = lambda specs: [(j == 0, None if j == 0 else (s, j)) for _, s, j in specs]
    # batch A: nomem + residual conditions (same prompt ids, per-row injection plans)
    A = [x for x in lay if x[0] == "nomem" or (x[0] in RESID and x[0] in M.add and gate)]
    for i in range(0, len(A), g["max_batch"]):
        ch = A[i:i + g["max_batch"]]
        plan = defaultdict(list)
        for r, (c, _, _) in enumerate(ch):
            if c != "nomem":
                key, vecs = M.add[c]
                for l, v in vecs.items():
                    plan[l].append((r, site[key], v))
        outs = gen_batch(C, ids, rows_of(ch), add=dict(plan), max_new=cap, streams=streams, first_ids=first_ids)
        for (c, s, j), o in zip(ch, outs):
            res[c].append((s, j, o))
    for c in conds:
        specs = [x for x in lay if x[0] == c]
        if c == "ctx":
            outs = gen_batch(C, chat_ids(tok, f"{it['experience']} {text}"), rows_of(specs), max_new=cap, streams=streams,
                             first_ids=first_ids)
            res[c] += [(s, j, o) for (_, s, j), o in zip(specs, outs)]
        elif c.startswith("T-"):
            if gate:
                prior = it["experience"] if c == "T-full" else f"{it['target'].strip()}."
                _, tids = render_ids(tok, [{"role": "user", "content": prior}, {"role": "user", "content": text}])
                outs = gen_batch(C, torch.tensor(tids), rows_of(specs), max_new=cap, streams=streams, first_ids=first_ids)
                res[c] += [(s, j, o) for (_, s, j), o in zip(specs, outs)]
            else:
                res[c] += [(s, j, None) for _, s, j in specs]
        elif c.startswith("R-") and c in M.R:
            if gate:
                outs = gen_batch(C, qt_ids(C, text), rows_of(specs), prefix=M.R[c], max_new=cap, streams=streams,
                                 first_ids=first_ids)
                res[c] += [(s, j, o) for (_, s, j), o in zip(specs, outs)]
            else:
                res[c] += [(s, j, None) for _, s, j in specs]
        elif c in RESID and (not gate) and c in M.add:
            res[c] += [(s, j, None) for _, s, j in specs]
    return res, {"gate": gate, "sites": site["main"], "wrongpos": site["wrongpos"]}


def decomp_analytic(C, items, R, conds):
    """Addendum 6b analytic (teacher-forced, primary items): Delta log P(target) at the direct question's measure
    prefix vs nomem, and the balanced verify probes' first-token margins log P(Yes) - log P(No), consistent-signed,
    with accuracy = margin > 0."""
    rows = []
    for it in items:
        if it["tier"] != "primary":
            continue
        t = int(text_ids(C.tok, it["target"])[0])
        pre = text_ids(C.tok, it["prefix"])
        q = it["question"]

        def ids_for(c, text, with_prefix):
            p_ = pre if with_prefix else torch.tensor([], dtype=torch.long)
            if c == "nomem":
                return torch.cat([chat_ids(C.tok, text), p_]), None
            if c == "ctx":
                return torch.cat([chat_ids(C.tok, f"{it['experience']} {text}"), p_]), None
            return torch.cat([qt_ids(C, text), p_]), R[it["id"]][c]
        lp_nm = None
        for c in conds:
            ids, pf = ids_for(c, q, True)
            lp = torch.log_softmax(tf_last(C, ids, pf), -1)[t].item()
            lp_nm = lp if c == "nomem" else lp_nm
            ms = []
            for rp in it["relation_probes"]:
                ids, pf = ids_for(c, rp["text"], False)
                lg = torch.log_softmax(tf_last(C, ids, pf), -1)
                d_ = (lg[C.yes] - lg[C.no]).item()
                ms.append((rp["a"], d_ if rp["a"] == "Yes" else -d_))
            acc = {side: [float(m > 0) for a, m in ms if a == side] for side in ("Yes", "No")}
            rows.append({"item": it["id"], "cond": c, "lp_target": lp, "dlp": lp - lp_nm, "yn_margins": ms,
                         "yn_margin": sum(m for _, m in ms) / len(ms),
                         "yn_bal_acc": (sum(acc["Yes"]) / len(acc["Yes"]) + sum(acc["No"]) / len(acc["No"])) / 2})
    return rows


def stage_gen(C, items, out, all_items):
    """Arm R + subj + controls generation on Qwen2.5 (PREREG addendum): main prompts (direct question + 2 use
    prompts), balanced yes/no probes, 50 unrelated prompts (+ the ungated R-full leak baseline), then coherence."""
    cfg, g = C.cfg, C.cfg["gen"]
    gen_context(C, cfg)
    t0 = time.time()
    A_cfg = cfg["analytic"]
    by_id = {it["id"]: it for it in all_items}
    sel = {it["id"] for it in items}
    need = sel | {by_id[i]["swap"] for i in sel}
    V, Vplac, R, Rinfo = {}, {}, {}, {}
    for iid in [i for i in by_id if i in need]:
        it = by_id[iid]
        if any(c in RESID for c in C.gc.primary):
            V[iid], _ = harvest(C, it, [])
            Vplac[iid], _ = harvest(C, {**it, "experience": g["placebo"]}, [])
            for d in (V[iid], Vplac[iid]):
                for k in d:
                    d[k] = d[k].to(C.dev)
        R[iid], Rinfo[iid] = harvest_R(C, it, g["orphan_exchange"])
    # checks: R-full == the sentence as a prior user turn; R-min keeps 2 entries + head
    checks = {"splice": [], "R_info": Rinfo}
    (out).mkdir(parents=True, exist_ok=True)
    json.dump({**vars(C.gc), "gen_set": getattr(C.A, "gen_set", None)}, open(out / "gen_conds.json", "w"), indent=1)
    for it in items:
        kvs, start = R[it["id"]][C.gc.check]
        qt = qt_ids(C, it["question"])
        cache = bind.make_cache(kvs, 1) if isinstance(kvs, list) else bind.expand_cache(kvs, 1)
        pos = torch.arange(start, start + len(qt))[None].to(C.dev)
        a = C.lm_head(C.body(input_ids=qt[None].to(C.dev), past_key_values=cache, position_ids=pos,
                             use_cache=True).last_hidden_state[:, -1]).float()
        _, ids_prior = render_ids(C.tok, [{"role": "user", "content": it["experience"]},
                                          {"role": "user", "content": it["question"]}])
        b = C.lm_head(C.body(input_ids=torch.tensor(ids_prior)[None].to(C.dev)).last_hidden_state[:, -1]).float()
        d = maxdiff(a, b)
        checks["splice"].append({"item": it["id"], "max_abs_logit_diff": d, "same_argmax": int(a.argmax()) == int(b.argmax())})
        assert d <= g["splice_tol"], f"{it['id']}: {C.gc.check} != prior-turn text ({d})"
        ri = Rinfo[it["id"]]
        if "kept" in ri:
            assert ri["kept"]["R-min"] == ri["n_head"] + 2
        else:   # hybrid: attention blocks hold head + segment / head + 2; DeltaNet blocks hold the head's state only
            nh, ns = ri["n_head"], ri["n_seg"]
            for v, want in (("R-full-state", nh + ns), ("R-full-kv6", nh + ns), ("R-min-kv6", nh + 2), ("R-orphan-kv6", nh + ns),
                            ("R-min-state", nh + 2), ("R-state-only", nh), ("R-orphan-state", nh + 2)):
                assert all(n == want for k, n in ri["layers"][v] if k == "attn"), (v, ri["layers"][v])
            assert ri["lin_equal_full"]["R-min-state"] and ri["lin_equal_full"]["R-state-only"], ri["lin_equal_full"]
            rs = ri["restored"]
            assert rs["R-state-only"] == {"rec": True, "conv": True} == rs["R-min-state"], rs
            assert rs["R-state-only-rec"] == {"rec": True, "conv": False} == rs["R-min-state-rec"], rs
            assert rs["R-state-only-conv"] == {"rec": False, "conv": True}, rs
            for v, want in (("R-state-only-rec", nh), ("R-state-only-conv", nh), ("R-min-state-rec", nh + 2)):
                assert all(n == want for k, n in ri["layers"][v] if k == "attn"), (v, ri["layers"][v])
            assert ri["lin_equal_orphan"]["R-orphan-state"] and not ri["lin_equal_full"]["R-orphan-state"]
            assert all(f for k, f in ri["layers"]["R-full-state"] if k == "lin")
            for v in ("R-full-kv6", "R-min-kv6", "R-orphan-kv6"):
                assert [x for x in ri["layers"][v] if x[0] == "lin"] == [x for x in ri["head_layers"] if x[0] == "lin"]
    log(f"gen: harvest + splice checks done in {(time.time() - t0) / 60:.1f} min; max splice diff "
        f"{max(c['max_abs_logit_diff'] for c in checks['splice']):.2e}")
    U = {iid: rand_dirs(C, iid, A_cfg["rand_seed"]) for iid in sel}
    Ms = {iid: item_memory(C, cfg, by_id[iid], V.get(iid), V.get(by_id[iid]["swap"]), Vplac.get(iid), U[iid], R[iid],
                           R.get(by_id[iid]["swap"])) for iid in sel}
    (out / "items").mkdir(parents=True, exist_ok=True)
    (out / "unrel").mkdir(parents=True, exist_ok=True)
    an_conds = (C.mc.get(f"gen_conds_{getattr(C.A, 'gen_set', None)}") or {}).get("analytic")
    if an_conds:
        an = decomp_analytic(C, items, R, an_conds)
        write_jsonl_gz(out / "analytic_tf.jsonl.gz", an)
        log(f"gen: teacher-forced analytic ({len(an)} rows) in {(time.time() - t0) / 60:.1f} min")
    S, cap = g["samples"], g["answer_cap"]
    seeds = g["seeds"]
    n_rows = 0
    for k, it in enumerate(items):
        f = out / "items" / f"{it['id']}.jsonl.gz"
        if f.exists():
            log(f"gen: {it['id']} already done, skipped")
            continue
        t1 = time.time()
        conds = C.gc.primary if it["tier"] == "primary" else C.gc.secondary
        if not conds:
            continue
        M = Ms[it["id"]]
        rows = []
        prompts = [("related", it["question"])] + [("use", u) for u in it["use"]]
        prompts = prompts[:2] if C.A.tiny else prompts
        for pi, (kind, text) in enumerate(prompts):
            streams = {1: (cv1.seed_of(TAG, text, seeds["base"]), S + 1), 2: (cv1.seed_of(TAG, text, seeds["second"]), S + 1)}
            res, gi = run_prompt(C, cfg, it, M, conds, text, S=S, cap=cap, streams=streams)
            nm = {(s, j): o for s, j, o in res["nomem"]}
            for c, lst in res.items():
                for s, j, o in lst:
                    shortcut = o is None
                    o = nm.get((s, j)) if shortcut else o
                    if o is None:   # seed-set-2 rows of a non-seed2 condition never exist; nothing to copy
                        continue
                    rep = cv1.rep_rate(o["ids"], g["loop"]["n"])
                    rows.append({"sec": "gen", "key": f"{it['id']}|{kind}|{pi}|{c}|{s}|{j}", "item": it["id"],
                                 "group": "facts", "tier": it["tier"], "kind": kind, "pi": pi, "prompt": text, "cond": c,
                                 "seedset": s, "j": j, "ids": o["ids"],
                                 "answer": C.tok.decode(o["ids"], skip_special_tokens=True).strip(),
                                 "n_tok": o["n_tok"], "capped": o["capped"], "rep": round(rep, 4),
                                 "loop": rep >= g["loop"]["thr"], "gate": gi["gate"], "shortcut": shortcut,
                                 "no_site": not gi["gate"]})
        for rp in it["relation_probes"]:
            text = rp["text"]
            res, gi = run_prompt(C, cfg, it, M, conds, text, S=0, cap=g["yn_cap"], streams={},
                                 first_ids=[C.yes, C.no])
            nm = res["nomem"][0][2]
            for c, lst in res.items():
                s, j, o = lst[0]
                o = nm if o is None else o
                ans = C.tok.decode(o["ids"], skip_special_tokens=True).strip()
                lpy, lpn = o["first"]
                rows.append({"sec": "yn", "item": it["id"], "group": "facts", "tier": it["tier"], "prompt": text,
                             "consistent": rp["a"], "cond": c, "answer": ans, "yn": xr_parse_yn(ans),
                             "margin": (lpy - lpn) if rp["a"] == "Yes" else (lpn - lpy), "lp_yes": lpy, "lp_no": lpn,
                             "gate": gi["gate"], "shortcut": lst[0][2] is None})
        write_jsonl_gz(f, rows)
        n_rows += len(rows)
        el = time.time() - t0
        log(f"gen: {it['id']} ({it['tier']}) {len(rows)} rows in {(time.time() - t1) / 60:.1f} min | elapsed "
            f"{el / 60:.1f} min, eta {(time.time() - t1) * (len(items) - k - 1) / 60:.1f} min")
    # unrelated prompts: nomem; each item's gated conditions (closed gate = nomem by construction); the
    # ungated R-full leak baseline for every primary item
    L = itm.load_items(REPO / cfg["items"], REPO / cfg["bench"])
    unrel = L["unrelated"][:3 if C.A.tiny else len(L["unrelated"])]
    Us = g["unrel_samples"]
    t2 = time.time()
    for uk, text in enumerate(unrel):
        f = out / "unrel" / f"u{uk:02d}.jsonl.gz"
        if f.exists():
            continue
        streams = {1: (cv1.seed_of(TAG + "/unrel", text, seeds["base"]), Us + 1)}
        rws = [(j == 0, None if j == 0 else (1, j)) for j in range(Us + 1)]
        nm = gen_batch(C, chat_ids(C.tok, text), rws, max_new=cap, streams=streams)
        rows = [{"sec": "unrel", "uk": uk, "prompt": text, "item": None, "cond": "nomem", "j": j, "same": True,
                 "shortcut": False, "ids": o["ids"], "answer": C.tok.decode(o["ids"], skip_special_tokens=True).strip()}
                for j, o in enumerate(nm)]
        for it in items:
            conds = [c for c in (C.gc.primary if it["tier"] == "primary" else C.gc.secondary) if c not in ("nomem", "ctx")]
            if not conds:
                continue
            gate = bind.gate_open(text, it["aliases"])
            if gate:
                res, _ = run_prompt(C, cfg, it, Ms[it["id"]], ["nomem"] + conds, text, S=Us, cap=cap, streams=streams)
            for c in conds:
                for j in range(Us + 1):
                    o = None
                    if gate:
                        o = next((oo for s, jj, oo in res.get(c, []) if s == 1 and jj == j), None)
                    rows.append({"sec": "unrel", "uk": uk, "prompt": text, "item": it["id"], "cond": c, "j": j,
                                 "gate": gate, "shortcut": o is None, "same": o is None or o["ids"] == nm[j]["ids"],
                                 "ids": None if o is None else o["ids"],
                                 "answer": None if o is None else C.tok.decode(o["ids"], skip_special_tokens=True).strip()})
            if it["tier"] == "primary" and C.gc.leak:
                lk = gen_batch(C, qt_ids(C, text), rws, prefix=Ms[it["id"]].R[C.gc.leak], max_new=cap, streams=streams)
                for j, o in enumerate(lk):
                    rows.append({"sec": "unrel", "uk": uk, "prompt": text, "item": it["id"], "cond": f"{C.gc.leak}:ungated",
                                 "j": j, "gate": True, "shortcut": False, "same": o["ids"] == nm[j]["ids"], "ids": o["ids"],
                                 "answer": C.tok.decode(o["ids"], skip_special_tokens=True).strip()})
        write_jsonl_gz(f, rows)
        if uk % 10 == 0:
            el = time.time() - t2
            log(f"gen: unrelated {uk + 1}/{len(unrel)} | {el / 60:.1f} min, eta {el / (uk + 1) * (len(unrel) - uk - 1) / 60:.1f} min")
    # the judge's / panel's item file (core_v1 layout) and coherence
    prep = out.parent / "prep"
    prep.mkdir(parents=True, exist_ok=True)
    json.dump({i: L["by_id"][i] for i in by_id}, open(prep / "items.json", "w"), indent=1)
    nll = coherence_nll(C, out)
    checks["coherence_unique"] = nll
    checks["minutes"] = (time.time() - t0) / 60
    return [], checks, {}


def xr_parse_yn(text):
    m = re.search(r"\b(yes|no)\b", text, re.I)
    return m.group(1).lower() if m else None


def coherence_nll(C, gen_dir):
    """Mean token NLL of every generated answer under the no-memory model, given the plain prompt (core_v1's
    coherence); identical (prompt, answer) pairs scored once -> <run>/score/coherence.jsonl.gz."""
    rows = []
    for f in sorted((gen_dir / "items").glob("*.jsonl.gz")) + sorted((gen_dir / "unrel").glob("*.jsonl.gz")):
        rows += [r for r in read_jsonl_gz(f) if r["sec"] in ("gen", "unrel") and r.get("ids") is not None]
    todo = {}
    for r in rows:
        todo.setdefault((r["prompt"], tuple(r["ids"])), None)
    seqs = sorted([(k, chat_ids(C.tok, k[0]), list(k[1])) for k in todo], key=lambda s: len(s[1]) + len(s[2]))
    pad = C.tok.pad_token_id if C.tok.pad_token_id is not None else C.eos_list[0]
    budget = C.mc.get("nll_tokens", C.cfg["gen"]["nll_tokens"])   # per-model override (2B on the 10 GB slice)
    res, i = {}, 0
    while i < len(seqs):
        j = i + 1
        while j < len(seqs) and (j - i + 1) * (len(seqs[j][1]) + len(seqs[j][2])) <= budget:
            j += 1
        batch = seqs[i:j]
        Tm = max(len(p) + len(a) for _, p, a in batch)
        X = torch.full((len(batch), Tm), pad, dtype=torch.long)
        for b, (_, p, a) in enumerate(batch):
            X[b, :len(p) + len(a)] = torch.cat([p, torch.tensor(a, dtype=torch.long)]) if a else p
        h = C.body(input_ids=X.to(C.dev)).last_hidden_state
        for b, (key, p, a) in enumerate(batch):
            if not a:
                res[key] = float("nan")
                continue
            lp = torch.log_softmax(C.lm_head(h[b, len(p) - 1:len(p) + len(a) - 1]).float(), -1)
            res[key] = -lp.gather(1, torch.tensor(a, device=C.dev)[:, None]).mean().item()
        i = j
    sc = gen_dir.parent / "score"
    sc.mkdir(parents=True, exist_ok=True)
    outrows = [{"sec": r["sec"], "key": r.get("key") or f"unrel|{r['uk']}|{r['item']}|{r['cond']}|{r['j']}",
                "nll": res[(r["prompt"], tuple(r["ids"]))], "n_tok": len(r["ids"])} for r in rows]
    write_jsonl_gz(sc / "coherence.jsonl.gz", outrows)
    log(f"coherence: {len(seqs)} unique answers")
    return len(seqs)


def gen_plan(cfg, items, n_unrel=50, mc=None, gen_set=None):
    """Generation counts (for the budget)."""
    g = cfg["gen"]
    gc = (mc or {}).get(f"gen_conds_{gen_set}" if gen_set else "gen_conds") or {}
    P, Sd, s2 = gc.get("primary", PRIMARY_CONDS), gc.get("secondary", SECONDARY_CONDS), gc.get("seed2", g["seed2"])
    S, Us = g["samples"], g["unrel_samples"]
    per = lambda conds: len(row_layout(conds, s2, S))
    main = sum(3 * per(P if it["tier"] == "primary" else Sd) for it in items)
    yn = sum(len(it["relation_probes"]) * len(P if it["tier"] == "primary" else Sd) for it in items)
    n_prim = sum(it["tier"] == "primary" for it in items)
    unrel = n_unrel * (Us + 1) * (1 + n_prim)
    return {"main_answers_max": main, "yn_answers": yn, "unrel_answers_min": unrel, "total_long_max": main + unrel}


# =============================================================== two-fact collision (PREREG addendum 5 (2))


def collide_dir(out):
    """The collision run's model dir, a sibling of the model's run dir (<run>/<model>_collide), so that the judge
    and the panel read it like any model dir."""
    return out.parent / f"{out.name}_collide"


def rope_shift(C, K, delta):
    """Stored keys K [B, kv_heads, S, head_dim] (already rotated at their positions) moved by `delta` positions:
    RoPE rotations compose additively, so applying the model's own rotary embedding at position `delta` gives the
    keys at position + delta (partial rotary and Qwen3.5's text mrope included)."""
    import sys as _sys
    S = K.shape[2]
    pos = torch.full((1, S), int(delta), device=K.device, dtype=torch.long)
    if is_hybrid(C):
        pos = pos[None].expand(3, 1, S)
    cos, sin = C.body.rotary_emb(K, pos)
    layer = next(b for b in C.blocks if hasattr(b, "self_attn"))
    mod = _sys.modules[type(layer.self_attn).__module__]
    return mod.apply_rotary_pos_emb(K, K, cos, sin)[1]


def _attn_kv(src, l):
    if isinstance(src, list):
        return src[l]
    L = src.layers[l]
    return L.keys, L.values


def compose_prefix(C, parts, srcs, lin=None):
    """A reinstatement prefix from several harvests: per attention layer, the concatenation of each part's entries
    (source cache, indices, RoPE shift). Non-hybrid: a list of (K, V) per layer. Hybrid: a cache object whose
    attention layers hold the composition and whose DeltaNet layers hold `lin`'s state."""
    import copy
    if not is_hybrid(C):
        kv = {k: bind.kv_list(v) if not isinstance(v, list) else v for k, v in srcs.items()}
        out = []
        for l in range(len(kv[parts[0][0]])):
            Ks, Vs = [], []
            for name, idx, shift in parts:
                K, V = kv[name][l]
                ii = torch.as_tensor(idx, device=K.device, dtype=torch.long)
                K, V = K[:, :, ii], V[:, :, ii]
                Ks.append(rope_shift(C, K, shift) if shift else K)
                Vs.append(V)
            out.append((torch.cat(Ks, 2).contiguous(), torch.cat(Vs, 2).contiguous()))
        return out
    c = copy.deepcopy(srcs["J"])
    for l, L in enumerate(c.layers):
        if bind.is_linear_layer(L):
            c.layers[l] = copy.deepcopy(lin.layers[l])
        else:
            Ks, Vs = [], []
            for name, idx, shift in parts:
                K, V = _attn_kv(srcs[name], l)
                ii = torch.as_tensor(idx, device=K.device, dtype=torch.long)
                K, V = K[:, :, ii], V[:, :, ii]
                Ks.append(rope_shift(C, K, shift) if shift else K)
                Vs.append(V)
            L.keys, L.values = torch.cat(Ks, 2).contiguous(), torch.cat(Vs, 2).contiguous()
    return c


def collide_pairs(cfg):
    """[(pair id, kind, member A, member B)]; a member = {id, experience, value, question, prefix, use, verify,
    pair, kind, role, partner, partner_value}."""
    import copy
    facts = {f["id"]: f for f in yaml.safe_load(open(REPO / cfg["bench"] / "facts.yaml"))["facts"]}
    items = {it["id"]: it for it in load_items(cfg)}
    out = []
    for p in cfg["collide"]["pairs"]:
        if p["kind"] == "different":
            mem = [{"id": items[p[k]]["id"], "experience": items[p[k]]["experience"], "value": items[p[k]]["target"],
                    "question": items[p[k]]["question"], "prefix": items[p[k]]["prefix"], "use": list(items[p[k]]["use"]),
                    "verify": list(facts[p[k]]["verify"])} for k in ("a", "b")]
        else:
            mem = [copy.deepcopy(p["a"]), copy.deepcopy(p["b"])]
        A, B = mem
        A.update(pair=p["id"], kind=p["kind"], role="A", partner=B["id"], partner_value=B["value"])
        B.update(pair=p["id"], kind=p["kind"], role="B", partner=A["id"], partner_value=A["value"])
        out.append((p["id"], p["kind"], A, B))
    return out


def harvest_pair(C, A, B):
    """The R2 reinstatement prefixes of one pair: {condition: (prefix, question start)} + info."""
    tok = C.tok
    PA, PB = turn_pieces(tok, A["experience"]), turn_pieces(tok, B["experience"])
    assert PA.H == PB.H
    nh, nA, nB = len(PA.H), len(PA.Seg), len(PB.Seg)
    att = []
    for P, M in ((PA, A), (PB, B)):
        t = text_ids(tok, M["value"]).tolist()
        assert len(t) == 1, f"{M['id']}: value {M['value']!r} is not a single token"
        hits = [i for i, x in enumerate(P.Seg) if x == t[0]]
        assert len(hits) == 1, f"{M['id']}: value found {len(hits)} times in the experience turn"
        att.append(hits[0])
    plan = bind.pair_plan(nh, nA, nB, att[0], att[1])
    srcs = {"A": run_cache(C, PA.H + PA.Seg), "B": run_cache(C, PB.H + PB.Seg), "J": run_cache(C, PA.H + PA.Seg + PB.Seg)}
    out = {}
    if is_hybrid(C):
        lin = {"J": srcs["J"], "H": run_cache(C, PA.H)}
        for name, (base, ls) in bind.hybrid_collide_names(plan).items():
            parts, start = plan[base]
            out[name] = (compose_prefix(C, parts, srcs, lin[ls]), start)
    else:
        for name, (parts, start) in plan.items():
            out[name] = (compose_prefix(C, parts, srcs), start)
    info = {"n_head": nh, "n_A": nA, "n_B": nB, "attr_A": att[0], "attr_B": att[1], "starts": {k: s for k, (_, s) in out.items()}}
    return out, info


def tf_last(C, ids, prefix=None):
    """Last-position logits of `ids` (teacher-forced), optionally after a reinstatement prefix."""
    start = prefix[1] if prefix else 0
    if prefix is None:
        cache = None if is_hybrid(C) else bind.make_cache([], 1)
    elif isinstance(prefix[0], list):
        cache = bind.make_cache(prefix[0], 1)
    else:
        cache = bind.expand_cache(prefix[0], 1)
    pos = torch.arange(start, start + len(ids))[None].to(C.dev)
    h = C.body(input_ids=ids[None].to(C.dev), past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
    return C.lm_head(h[:, -1]).float()[0]


def stage_collide(C, items, out):
    """The two-fact collision test (addendum 5 (2)): forced gates (oracle retrieval). Per member: the direct question
    and 2 use prompts (1 greedy + 4 samples, seed set 1), the balanced yes/no probes (Yes = own value, No = the
    partner's), and the teacher-forced readout at the measure prefix (log P(own), logit(own) - logit(partner))."""
    cfg = C.cfg
    gen_context(C, cfg)
    t0 = time.time()
    cd = collide_dir(out.parent if out.name == "collide" else out)
    gdir = cd / "gen"
    (gdir / "items").mkdir(parents=True, exist_ok=True)
    conds = list(C.mc["collide_conds"])
    pairs = collide_pairs(cfg)
    if C.A.tiny:
        pairs = [p for p in pairs if p[0] in ("dog_cat", "sisters")]
    S = cfg["collide"]["samples"]
    g = cfg["gen"]
    cap, seeds = g["answer_cap"], g["seeds"]
    checks, items_json = {"t2_equals_joint": [], "pairs": {}}, {}
    joint_name = "R2-joint-state" if is_hybrid(C) else "R2-joint"
    for pid, kind, A, B in pairs:
        t1 = time.time()
        R, info = harvest_pair(C, A, B)
        checks["pairs"][pid] = info
        # check: R2-joint == both sentences as separate prior user turns (T2), for both questions
        for M in (A, B):
            qt = qt_ids(C, M["question"])
            a = tf_last(C, qt, R[joint_name])
            _, t2 = render_ids(C.tok, [{"role": "user", "content": A["experience"]}, {"role": "user", "content": B["experience"]},
                                       {"role": "user", "content": M["question"]}])
            b = tf_last(C, torch.tensor(t2))
            d = maxdiff(a, b)
            checks["t2_equals_joint"].append({"member": M["id"], "max_abs_logit_diff": d})
            assert d <= g["splice_tol"], f"{M['id']}: {joint_name} != T2 text ({d})"
        for M in (A, B):
            f = gdir / "items" / f"{M['id']}.jsonl.gz"
            items_json[M["id"]] = {"id": M["id"], "experience": M["experience"], "group": "facts",
                                   "measure": {"target": M["value"], "prefix": M["prefix"]}, "pair": pid, "kind": kind,
                                   "role": M["role"], "partner": M["partner"], "partner_value": M["partner_value"],
                                   "question": M["question"], "use": M["use"]}
            if f.exists():
                continue
            rows = []
            own, partner = int(text_ids(C.tok, M["value"])[0]), int(text_ids(C.tok, M["partner_value"])[0])
            ctx2 = f"{A['experience']} {B['experience']}"
            prompts = [("related", M["question"])] + [("use", u) for u in M["use"]]
            prompts = prompts[:2] if C.A.tiny else prompts
            for pi, (pk, text) in enumerate(prompts):
                streams = {1: (cv1.seed_of(TAG + "/collide", text, seeds["base"]), S + 1)}
                rws = [(j == 0, None if j == 0 else (1, j)) for j in range(S + 1)]
                for c in conds:
                    if c == "nomem":
                        outs = gen_batch(C, chat_ids(C.tok, text), rws, max_new=cap, streams=streams)
                    elif c == "ctx2":
                        outs = gen_batch(C, chat_ids(C.tok, f"{ctx2} {text}"), rws, max_new=cap, streams=streams)
                    else:
                        outs = gen_batch(C, qt_ids(C, text), rws, prefix=R[c], max_new=cap, streams=streams)
                    for j, o in enumerate(outs):
                        rep = cv1.rep_rate(o["ids"], g["loop"]["n"])
                        rows.append({"sec": "gen", "key": f"{M['id']}|{pk}|{pi}|{c}|1|{j}", "item": M["id"], "group": "facts",
                                     "tier": kind, "pair": pid, "role": M["role"], "kind": pk, "pi": pi, "prompt": text,
                                     "cond": c, "seedset": 1, "j": j, "ids": o["ids"],
                                     "answer": C.tok.decode(o["ids"], skip_special_tokens=True).strip(), "n_tok": o["n_tok"],
                                     "capped": o["capped"], "rep": round(rep, 4), "loop": rep >= g["loop"]["thr"],
                                     "gate": True, "shortcut": False, "no_site": False})
            # balanced yes/no: Yes = own value, No = the partner's value as the foil
            probes = [(tpl.format(v=M["value"].strip()) + " Answer yes or no.", "Yes") for tpl in M["verify"]] + \
                     [(tpl.format(v=M["partner_value"].strip()) + " Answer yes or no.", "No") for tpl in M["verify"]]
            for text, cons in probes:
                for c in conds:
                    if c == "nomem":
                        o = gen_batch(C, chat_ids(C.tok, text), [(True, None)], max_new=g["yn_cap"], first_ids=[C.yes, C.no])[0]
                    elif c == "ctx2":
                        o = gen_batch(C, chat_ids(C.tok, f"{ctx2} {text}"), [(True, None)], max_new=g["yn_cap"],
                                      first_ids=[C.yes, C.no])[0]
                    else:
                        o = gen_batch(C, qt_ids(C, text), [(True, None)], prefix=R[c], max_new=g["yn_cap"],
                                      first_ids=[C.yes, C.no])[0]
                    ans = C.tok.decode(o["ids"], skip_special_tokens=True).strip()
                    lpy, lpn = o["first"]
                    rows.append({"sec": "yn", "item": M["id"], "group": "facts", "tier": kind, "pair": pid, "prompt": text,
                                 "consistent": cons, "cond": c, "answer": ans, "yn": xr_parse_yn(ans),
                                 "margin": (lpy - lpn) if cons == "Yes" else (lpn - lpy), "lp_yes": lpy, "lp_no": lpn})
            # teacher-forced readout at the measure prefix
            pre = text_ids(C.tok, M["prefix"])
            for c in conds:
                if c == "nomem":
                    lg = tf_last(C, torch.cat([chat_ids(C.tok, M["question"]), pre]))
                elif c == "ctx2":
                    lg = tf_last(C, torch.cat([chat_ids(C.tok, f"{ctx2} {M['question']}"), pre]))
                else:
                    lg = tf_last(C, torch.cat([qt_ids(C, M["question"]), pre]), R[c])
                lp = torch.log_softmax(lg, -1)
                rows.append({"sec": "tf", "item": M["id"], "tier": kind, "pair": pid, "cond": c, "lp_own": lp[own].item(),
                             "lp_partner": lp[partner].item(), "margin": (lg[own] - lg[partner]).item()})
            write_jsonl_gz(f, rows)
        log(f"collide: {pid} ({kind}) done in {(time.time() - t1) / 60:.1f} min; elapsed {(time.time() - t0) / 60:.1f} min")
    (cd / "prep").mkdir(parents=True, exist_ok=True)
    json.dump(items_json, open(cd / "prep" / "items.json", "w"), indent=1)
    json.dump({"conds": conds, "is_hybrid": is_hybrid(C)}, open(gdir / "collide_conds.json", "w"))
    checks["coherence_unique"] = coherence_nll(C, gdir)
    checks["minutes"] = (time.time() - t0) / 60
    return [], checks, {}


def analyze_collide(cfg, cd, model_key):
    """Addendum 5 (2) tables and rules from the saved files: per pair kind (confusable / different) and condition,
    correct-target clean use (judges' majority), partner confusion (the partner's value in the answer, the member's
    own value not), the judges' wrong_value, balanced yes/no with the partner as foil, the teacher-forced margin."""
    from seahorse.stats import boot_ci, paired_diff
    nb, seed = cfg["n_boot"], cfg["seeds"]["bootstrap"]
    items = json.load(open(cd / "prep" / "items.json"))
    G, YN, _, _, target, n_panel = load_labeled(cfg, cd, f"{model_key}_collide")
    TF = []
    for f in sorted((cd / "gen" / "items").glob("*.jsonl.gz")):
        TF += [r for r in read_jsonl_gz(f) if r["sec"] == "tf"]
    hyb = json.load(open(cd / "gen" / "collide_conds.json"))["is_hybrid"]
    conds = list(dict.fromkeys(r["cond"] for r in G))
    pv = {i: v["partner_value"].strip() for i, v in items.items()}
    for r in G:
        r["confusion"] = _has_word(r["answer"], pv[r["item"]]) and not _has_word(r["answer"], target[r["item"]])

    def nested(kind, cond, fn, kinds=("related", "use")):
        acc = defaultdict(list)
        for r in G:
            if items[r["item"]]["kind"] == kind and r["cond"] == cond and r["kind"] in kinds:
                v = fn(r)
                if v is not None:
                    acc[(r["item"], r["pi"])].append(float(v))
        d = defaultdict(dict)
        for (i, p), vs in acc.items():
            d[i][p] = sum(vs) / len(vs)
        return dict(d)

    def ci(n):
        p, lo, hi, k = boot_ci(n, nb, seed)
        return {"mean": p, "lo": lo, "hi": hi, "n": k}

    def yn(kind, cond):
        per = defaultdict(list)
        for r in YN:
            if items[r["item"]]["kind"] == kind and r["cond"] == cond:
                per[r["item"]].append((r["consistent"], r["yn"]))
        return {i: {0: bind.balanced_yn(v)} for i, v in per.items()}

    def tfm(kind, cond):
        return {r["item"]: {0: r["margin"]} for r in TF if r["tier"] == kind and r["cond"] == cond}

    M = {"correct (majority)": lambda r: r["clean_maj"], "correct (orig judge)": lambda r: r["clean_orig"],
         "partner confusion": lambda r: r["confusion"], "wrong value (judges)": lambda r: r["maj"]["wrong_value"],
         "loop": lambda r: r["loop"]}
    L = [f"# dream_bind: two-fact collision ({model_key}; PREREG addendum 5 (2), exploratory)", "",
         f"Forced gates (oracle retrieval). Members x (direct question + 2 use prompts), seed set 1; clean use = majority "
         f"of the original judge + gpt-oss + Gemma-4 (panel labels for {n_panel}/{len(G)} answers). Bootstrap over members "
         f"then prompts ({nb}, seed {seed}).", ""]
    T = {}
    for kind in ("confusable", "different"):
        L += [f"## {kind} pairs", "", "| condition | " + " | ".join(M) + " | yes/no (partner foil) | TF margin own − partner |",
              "|---|" + "---|" * (len(M) + 2)]
        for c in conds:
            v = {m: ci(nested(kind, c, fn)) for m, fn in M.items()}
            v["yn"], v["tf"] = ci(yn(kind, c)), ci(tfm(kind, c))
            T[(kind, c)] = v
            L.append(f"| {c} | " + " | ".join(fmt(v[m]) for m in M) + f" | {fmt(v['yn'])} | {fmt(v['tf'])} |")
        L.append("")
    sfx = "-jstate" if hyb else ""
    joint = "R2-joint-state" if hyb else "R2-joint"
    mjoint = "R2-min-joint-jstate" if hyb else "R2-min-joint"
    rules, lines = {}, []
    for full_or_min, jn, sep_base in (("full", joint, "R2-sep"), ("min", mjoint, "R2-min-sep")):
        for scheme in ("overlap", "shift"):
            sep = f"{sep_base}-{scheme}{sfx}"
            if sep not in conds or jn not in conds:
                continue
            conf = lambda c: nested("confusable", c, M["partner confusion"])
            cor = lambda kind, c: nested(kind, c, M["correct (majority)"])
            d_coll = ci(paired_diff(conf(sep), conf(jn)))
            d_cont = ci(paired_diff(cor("different", sep), cor("different", "ctx2")))
            d_reen = ci(paired_diff(cor("confusable", jn), cor("confusable", sep)))
            d_reen_diff = ci(paired_diff(cor("different", jn), cor("different", sep)))
            key = f"{full_or_min}/{scheme}"
            na = lambda d, ok: None if math.isnan(d["mean"]) else bool(ok)
            rules[key] = {
                "IDs collide (confusion sep - joint, CI > 0)": (d_coll, na(d_coll, d_coll["lo"] > 0)),
                "content suffices (different: sep - ctx2 CI includes 0, |point| < 0.1)":
                    (d_cont, na(d_cont, (d_cont["lo"] <= 0 <= d_cont["hi"]) and abs(d_cont["mean"]) < 0.1)),
                "re-encoding helps (confusable: joint - sep correct, CI > 0)": (d_reen, na(d_reen, d_reen["lo"] > 0)),
                "re-encoding (different pairs, reported)": (d_reen_diff, None)}
            lines += [f"**{sep} vs {jn}**", ""]
            lines += [f"- {k}: {fmt(v)}{'' if ok is None else (' → holds' if ok else ' → does not hold')}"
                      for k, (v, ok) in rules[key].items()]
            lines.append("")
    L[4:4] = ["## Rules (addendum 5, exploratory)", "", *lines]
    out = cd / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L))
    json.dump({"rules": {k: {kk: {"ci": v, "holds": ok} for kk, (v, ok) in r.items()} for k, r in rules.items()},
               "tables": {f"{k}/{c}": v for (k, c), v in T.items()}, "n_panel_labels": n_panel, "n_answers": len(G)},
              open(out / "verdict.json", "w"), indent=2, default=str)
    log(f"collide analyze -> {out / 'report.md'}")
    return rules


def collide_plan(cfg, mc, n_prompts=3, n_probes=4):
    n_members = 2 * len(cfg["collide"]["pairs"])
    nc = len(mc["collide_conds"])
    return {"members": n_members, "long_answers": n_members * n_prompts * nc * (cfg["collide"]["samples"] + 1),
            "yn_answers": n_members * n_probes * nc}


# =============================================================== addendum 6 (A): binding-ID repair (Qwen2.5)


def sibling(out, tag):
    return out.parent / f"{out.name}_{tag}"


def block_inputs(C, ids):
    """Block-input residuals [n_blocks, T, d] (block 0's input = the embeddings) and the K,V list of a batch-1 run."""
    with capture(C.model, list(range(C.n))) as st:
        o = C.body(input_ids=ids[None].to(C.dev), use_cache=True)
    emb = C.body.embed_tokens(ids[None].to(C.dev))[0]
    return torch.stack([emb] + [st[l] for l in range(C.n - 1)]), bind.kv_list(o.past_key_values)


def kv_from_resid(C, l, H, pos):
    """Block l's K, V [1, kv_heads, T, head_dim] recomputed from block-input residuals H [T, d] at absolute
    positions `pos` (frozen input norm, W_k, W_v and RoPE)."""
    import sys as _sys
    layer = C.blocks[l]
    att = layer.self_attn
    cfgm = C.model.config
    hd = getattr(att, "head_dim", None) or cfgm.hidden_size // cfgm.num_attention_heads
    x = layer.input_layernorm(H[None])
    T = H.shape[0]
    k = att.k_proj(x).view(1, T, -1, hd).transpose(1, 2)
    v = att.v_proj(x).view(1, T, -1, hd).transpose(1, 2)
    p = torch.as_tensor(pos, device=H.device, dtype=torch.long)[None]
    cos, sin = C.body.rotary_emb(v, p)
    k = _sys.modules[type(att).__module__].apply_rotary_pos_emb(k, k, cos, sin)[1]
    return k, v


def edit_kv(C, kv, H, edits, layers):
    """A copy of the K,V list `kv` (batch-1 run, cache index = position) where, at every layer in `layers`, the
    positions of each edit get K,V recomputed from H[l][positions] + offsets. edits: [(positions, {l: [len, d]})]."""
    out = [(K.clone(), V.clone()) for K, V in kv]
    for l in layers:
        for pos, vecs in edits:
            if l not in vecs:
                continue
            k, v = kv_from_resid(C, l, H[l][pos] + vecs[l].to(H.dtype), pos)
            ii = torch.as_tensor(pos, device=k.device, dtype=torch.long)
            out[l][0][:, :, ii] = k
            out[l][1][:, :, ii] = v
    return out


def _turns(C, *sentences):
    """[head][user: s1][user: s2]... token ids (each turn as turn_pieces tokenises it)."""
    ids = list(C.head)
    for s in sentences:
        ids += turn_pieces(C.tok, s).Seg
    return torch.tensor(ids)


def tf_two(C, ids, prefix, t_own, t_partner):
    """Teacher-forced (log P(own), log P(partner), margin logit(own) - logit(partner)) at the last position."""
    lg = tf_last(C, ids, prefix)
    lp = torch.log_softmax(lg, -1)
    return lp[t_own].item(), lp[t_partner].item(), (lg[t_own] - lg[t_partner]).item()


def estimate_ids(C, pool, n, seed, keep_pca=True):
    """Feng & Steinhardt's mean-difference offsets per block (block inputs): Delta_E [L, 2, d] (entity token, next)
    and Delta_A [L, 2, d] (value token, full stop) = mean(pair 2) - mean(pair 1) over n random two-pair contexts;
    plus Dai's order-PC projections."""
    L = C.n
    sums = {k: torch.zeros(L, 2, 2, C.d, device=C.dev) for k in ("entity", "attr")}   # [L, pair, slot, d]
    pcs = {k: [] for k in ("entity", "attr")}
    for s1, v1, s2, v2 in bind.id_templates(pool, n, seed):
        e1, e2 = f"My {s1}'s name is {v1}.", f"My {s2}'s name is {v2}."
        ids = _turns(C, e1, e2)
        H, _ = block_inputs(C, ids)
        for pi, (sent, val) in enumerate(((e1, v1), (e2, v2))):
            P = bind.ea_positions(C.tok, ids, sent, val)
            for k in ("entity", "attr"):
                sums[k][:, pi] += H[:, P[k]].float()
                if keep_pca:
                    pcs[k].append((pi, H[:, P[k][0]].float().cpu()))
    D = {k: (sums[k][:, 1] - sums[k][:, 0]) / n for k in sums}
    dai, rinfo = {}, {}
    for k in ("entity", "attr"):
        order = torch.tensor([pi for pi, _ in pcs[k]])
        X = torch.stack([h for _, h in pcs[k]])            # [2n, L, d]
        proj = torch.zeros_like(D[k])
        rinfo[k] = []
        for l in range(L):
            u, r = bind.order_pc(X[:, l], order)
            u = u.to(C.dev)
            proj[l] = (D[k][l] @ u)[:, None] * u[None]
            rinfo[k].append(round(r, 3))
        dai[k] = proj
    return D, dai, rinfo


def rand_like(D, seed):
    """Random directions with D's norm per (layer, slot)."""
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(D.shape, generator=g).to(D.device)
    return R / R.norm(dim=-1, keepdim=True) * D.norm(dim=-1, keepdim=True)


LAYER_SETS = {"all": None, "early": (0, 9), "mid": (10, 18), "late": (19, 27)}


def layer_set(name, n):
    a = LAYER_SETS[name]
    return list(range(n)) if a is None else list(range(a[0], min(a[1], n - 1) + 1))


def validate_ids(C, pool, n, seed, D, dai, sets):
    """Feng's mean interventions on held-out joint contexts: pair 1 + Delta, pair 2 - Delta at the attribute /
    entity / both positions, or a random direction of equal norm at the attribute positions; per layer set and
    direction (feng / dai). Both entities queried teacher-forced; flip = the own-vs-partner margin changes sign."""
    rows = []
    R = {k: rand_like(D[k], 1000 + i) for i, k in enumerate(D)}
    for ci, (s1, v1, s2, v2) in enumerate(bind.id_templates(pool, n, seed)):
        e1, e2 = f"My {s1}'s name is {v1}.", f"My {s2}'s name is {v2}."
        ids = _turns(C, e1, e2)
        H, kv = block_inputs(C, ids)
        P1, P2 = bind.ea_positions(C.tok, ids, e1, v1), bind.ea_positions(C.tok, ids, e2, v2)
        t1, t2 = int(text_ids(C.tok, f" {v1}")[0]), int(text_ids(C.tok, f" {v2}")[0])
        qs = [(f"What's my {s}'s name?", f"Your {s}'s name is", to, tp) for s, to, tp in ((s1, t1, t2), (s2, t2, t1))]
        start = len(ids)
        base = [tf_two(C, torch.cat([qt_ids(C, q), text_ids(C.tok, pre)]), (kv, start), to, tp) for q, pre, to, tp in qs]
        for direc, DD in (("feng", D), ("dai", dai)):
            for sname in sets:
                layers = layer_set(sname, C.n)
                for iv in ("swap-attr", "swap-ent", "swap-both", "rand-attr"):
                    if iv == "rand-attr" and direc == "dai":
                        continue
                    keys = {"swap-attr": ["attr"], "swap-ent": ["entity"], "swap-both": ["attr", "entity"],
                            "rand-attr": ["attr"]}[iv]
                    src = R if iv == "rand-attr" else DD
                    edits = []
                    for k in keys:
                        edits += [(P1[k], {l: src[k][l] for l in layers}), (P2[k], {l: -src[k][l] for l in layers})]
                    kv2 = edit_kv(C, kv, H, edits, layers)
                    for qi, (q, pre, to, tp) in enumerate(qs):
                        lo, lp_, m = tf_two(C, torch.cat([qt_ids(C, q), text_ids(C.tok, pre)]), (kv2, start), to, tp)
                        rows.append({"ctx": ci, "q": qi, "dir": direc, "set": sname, "iv": iv, "margin0": base[qi][2],
                                     "margin": m, "flip": (m > 0) != (base[qi][2] > 0)})
    return rows


def stage_idrepair(C, items, out):
    """Addendum 6 (A): estimate the binding-ID offsets, validate them, test the repair on the addendum-5 pairs
    (teacher-forced), then generate for the best repair cell. Output in the sibling dir <model>_idrepair."""
    from seahorse.bench import load_bench
    cfg = C.cfg
    gen_context(C, cfg)
    assert not is_hybrid(C), "the ID repair runs on Qwen2.5"
    a6 = cfg["idrepair"]
    tiny = C.A.tiny
    cd = sibling(out.parent if out.name == "idrepair" else out, "idrepair")
    gdir = cd / "gen"
    (gdir / "items").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pool = load_bench(REPO / cfg["bench"])["pool"]
    n_est, n_val = (20, 6) if tiny else (a6["n_estimate"], a6["n_validate"])
    sets = list(LAYER_SETS) if not tiny else ["all", "mid"]
    D, dai, rinfo = estimate_ids(C, pool, n_est, a6["seed_estimate"])
    log(f"idrepair: offsets from {n_est} contexts in {(time.time() - t0) / 60:.1f} min; |Delta_A| per layer "
        f"{[round(x, 1) for x in D['attr'].norm(dim=-1).mean(-1).tolist()]}")
    val = validate_ids(C, pool, n_val, a6["seed_validate"], D, dai, sets)
    write_jsonl_gz(gdir / "validation.jsonl.gz", val)
    fr = lambda s, iv, d="feng": sum(r["flip"] for r in val if r["set"] == s and r["iv"] == iv and r["dir"] == d) / \
        max(1, sum(1 for r in val if r["set"] == s and r["iv"] == iv and r["dir"] == d))
    flips = {s: fr(s, "swap-attr") for s in sets}
    pick_set = max(sets, key=lambda s: (flips[s], s == "all"))
    layers = layer_set(pick_set, C.n)
    log(f"idrepair: validation done ({len(val)} rows) in {(time.time() - t0) / 60:.1f} min; swap-attr flip rates "
        f"{flips}; rand {fr('all', 'rand-attr')}; picked layer set '{pick_set}'")
    torch.save({"D": {k: v.cpu() for k, v in D.items()}, "dai": {k: v.cpu() for k, v in dai.items()}, "pc_r": rinfo,
                "pick_set": pick_set, "flips": flips}, gdir / "ids.pt")
    # --- repair, teacher-forced, on every addendum-5 pair
    pairs = collide_pairs(cfg)
    if tiny:
        pairs = [p for p in pairs if p[0] in ("dog_cat", "sisters")]
    R_ = {k: rand_like(D[k], 2000 + i) for i, k in enumerate(D)}
    variants = {"ID": D, "ID-dai": dai, "rand": R_, "neg": {k: -v for k, v in D.items()}}
    arows, prefixes, items_json = [], {}, {}
    for pid, kind, A, B in pairs:
        PA, PB = turn_pieces(C.tok, A["experience"]), turn_pieces(C.tok, B["experience"])
        ids_a, ids_b = torch.tensor(PA.H + PA.Seg), torch.tensor(PB.H + PB.Seg)
        HA, kvA = block_inputs(C, ids_a)
        HB, kvB = block_inputs(C, ids_b)
        kvJ = bind.kv_list(C.body(input_ids=_turns(C, A["experience"], B["experience"])[None].to(C.dev),
                                  use_cache=True).past_key_values)
        Pa, Pb = bind.ea_positions(C.tok, ids_a, A["experience"], A["value"]), bind.ea_positions(C.tok, ids_b, B["experience"], B["value"])
        nh, nA, nB = len(PA.H), len(PA.Seg), len(PB.Seg)
        plan = bind.pair_plan(nh, nA, nB, Pa["attr"][0] - nh, Pb["attr"][0] - nh)
        Bv = {"base": kvB}
        for vn, DD in variants.items():
            Bv[vn] = edit_kv(C, kvB, HB, [(Pb["entity"], {l: DD["entity"][l] for l in layers}),
                                          (Pb["attr"], {l: DD["attr"][l] for l in layers})], layers)
        Bv["wrongtok"] = edit_kv(C, kvB, HB, [(Pb["wrong"], {l: D["entity"][l] for l in layers})], layers)
        conds = {"R2-joint": compose_prefix(C, plan["R2-joint"][0], {"J": kvJ}), "R2-min-joint":
                 compose_prefix(C, plan["R2-min-joint"][0], {"J": kvJ})}
        starts = {"R2-joint": plan["R2-joint"][1], "R2-min-joint": plan["R2-min-joint"][1]}
        for size, base in (("full", "R2-sep"), ("min", "R2-min-sep")):
            for scheme in ("overlap", "shift"):
                parts, st = plan[f"{base}-{scheme}"]
                for vn in ("base", "ID", "ID-dai", "rand", "neg", "wrongtok"):
                    if size == "min" and vn == "wrongtok":
                        continue   # the slice holds only the attribute and full-stop entries
                    name = f"{base}-{scheme}" + ("" if vn == "base" else f"+{vn}")
                    conds[name] = compose_prefix(C, parts, {"A": kvA, "B": Bv[vn], "J": kvJ})
                    starts[name] = st
        prefixes[pid] = {c: (conds[c], starts[c]) for c in conds}
        for M in (A, B):
            own, partner = int(text_ids(C.tok, M["value"])[0]), int(text_ids(C.tok, M["partner_value"])[0])
            q = torch.cat([qt_ids(C, M["question"]), text_ids(C.tok, M["prefix"])])
            probes = [(tpl.format(v=M["value"].strip()) + " Answer yes or no.", "Yes") for tpl in M["verify"]] + \
                     [(tpl.format(v=M["partner_value"].strip()) + " Answer yes or no.", "No") for tpl in M["verify"]]
            for c, pre in prefixes[pid].items():
                lo, lp_, m = tf_two(C, q, pre, own, partner)
                ym = []
                for text, cons in probes:
                    lg = torch.log_softmax(tf_last(C, qt_ids(C, text), pre), -1)
                    d_ = (lg[C.yes] - lg[C.no]).item()
                    ym.append(d_ if cons == "Yes" else -d_)
                arows.append({"item": M["id"], "pair": pid, "kind": kind, "cond": c, "lp_own": lo, "lp_partner": lp_,
                              "margin": m, "tf_conf": lp_ > lo, "yn_margin": sum(ym) / len(ym),
                              "yn_correct": sum(x > 0 for x in ym) / len(ym)})
            items_json[M["id"]] = {"id": M["id"], "experience": M["experience"], "group": "facts",
                                   "measure": {"target": M["value"], "prefix": M["prefix"]}, "pair": pid, "kind": kind,
                                   "role": M["role"], "partner": M["partner"], "partner_value": M["partner_value"]}
    write_jsonl_gz(gdir / "analytic.jsonl.gz", arows)
    # --- the best repair cell (pre-registered): largest mean reduction in TF confusion vs R2-sep, ties -> margin gain
    def mean(c, f):
        v = [r[f] for r in arows if r["cond"] == c]
        return sum(float(x) for x in v) / len(v) if v else float("nan")
    cells = []
    for base in ("R2-sep", "R2-min-sep"):
        for scheme in ("overlap", "shift"):
            for d in ("ID", "ID-dai"):
                b, x = f"{base}-{scheme}", f"{base}-{scheme}+{d}"
                cells.append({"base": b, "cell": x, "dir": d, "d_conf": mean(b, "tf_conf") - mean(x, "tf_conf"),
                              "d_margin": mean(x, "margin") - mean(b, "margin")})
    best = max(cells, key=lambda c: (c["d_conf"], c["d_margin"]))
    pick = {"layer_set": pick_set, "layers": layers, "cells": cells, "best": best, "flips_swap_attr": flips}
    json.dump(pick, open(gdir / "pick.json", "w"), indent=1)
    log(f"idrepair: analytic done in {(time.time() - t0) / 60:.1f} min; best cell {best}")
    # --- generation for the best cell
    b = best["base"]
    joint = "R2-min-joint" if b.startswith("R2-min") else "R2-joint"
    gconds = [joint, b, best["cell"], f"{b}+rand", f"{b}+neg"] + ([f"{b}+wrongtok"] if not b.startswith("R2-min") else [])
    if best["dir"] == "ID-dai":
        gconds.append(f"{b}+ID")
    S, g = cfg["collide"]["samples"], cfg["gen"]
    cap, seeds = g["answer_cap"], g["seeds"]
    for pid, kind, A, B in pairs:
        for M in (A, B):
            f = gdir / "items" / f"{M['id']}.jsonl.gz"
            if f.exists():
                continue
            rows = []
            prompts = [("related", M["question"])] + [("use", u) for u in M["use"]]
            prompts = prompts[:2] if tiny else prompts
            for pi, (pk, text) in enumerate(prompts):
                streams = {1: (cv1.seed_of(TAG + "/idrepair", text, seeds["base"]), S + 1)}
                rws = [(j == 0, None if j == 0 else (1, j)) for j in range(S + 1)]
                for c in gconds:
                    outs = gen_batch(C, qt_ids(C, text), rws, prefix=prefixes[pid][c], max_new=cap, streams=streams)
                    for j, o in enumerate(outs):
                        rep = cv1.rep_rate(o["ids"], g["loop"]["n"])
                        rows.append({"sec": "gen", "key": f"{M['id']}|{pk}|{pi}|{c}|1|{j}", "item": M["id"], "group": "facts",
                                     "tier": kind, "pair": pid, "kind": pk, "pi": pi, "prompt": text, "cond": c, "seedset": 1,
                                     "j": j, "ids": o["ids"], "answer": C.tok.decode(o["ids"], skip_special_tokens=True).strip(),
                                     "n_tok": o["n_tok"], "capped": o["capped"], "rep": round(rep, 4),
                                     "loop": rep >= g["loop"]["thr"], "gate": True, "shortcut": False, "no_site": False})
            probes = [(tpl.format(v=M["value"].strip()) + " Answer yes or no.", "Yes") for tpl in M["verify"]] + \
                     [(tpl.format(v=M["partner_value"].strip()) + " Answer yes or no.", "No") for tpl in M["verify"]]
            for text, cons in probes:
                for c in gconds:
                    o = gen_batch(C, qt_ids(C, text), [(True, None)], prefix=prefixes[pid][c], max_new=g["yn_cap"],
                                  first_ids=[C.yes, C.no])[0]
                    ans = C.tok.decode(o["ids"], skip_special_tokens=True).strip()
                    lpy, lpn = o["first"]
                    rows.append({"sec": "yn", "item": M["id"], "group": "facts", "tier": kind, "pair": pid, "prompt": text,
                                 "consistent": cons, "cond": c, "answer": ans, "yn": xr_parse_yn(ans),
                                 "margin": (lpy - lpn) if cons == "Yes" else (lpn - lpy)})
            write_jsonl_gz(f, rows)
        log(f"idrepair gen: {pid} done; elapsed {(time.time() - t0) / 60:.1f} min")
    (cd / "prep").mkdir(parents=True, exist_ok=True)
    json.dump(items_json, open(cd / "prep" / "items.json", "w"), indent=1)
    json.dump({"conds": gconds, "best": best}, open(gdir / "gen_conds.json", "w"), indent=1)
    nll = coherence_nll(C, gdir)
    return [], {"pick": pick, "coherence_unique": nll, "minutes": (time.time() - t0) / 60}, {}


# =============================================================== addendum 6 (B): G-key retrieval + capacity


def gkey_facts(cfg):
    """The fact order for the stores: 7 primary, 5 secondary (hand-written), then the pool facts with distinct slots
    not used by the hand-written facts (sample_pool order, seed 0). Unified dicts."""
    from seahorse.bench import load_bench
    from seahorse.bench.data import sample_pool
    hand = load_items(cfg)
    by_hand = {it["id"]: it for it in hand}
    hand_slots = {"dog", "cat", "sister", "brother", "grandmother", "best friend", "boss", "job", "hometown",
                  "university", "first language", "favourite colour"}
    pool = sample_pool(load_bench(REPO / cfg["bench"])["pool"], None, seed=cfg["gkey"]["pool_seed"], exclude_slots=hand_slots)
    out = []
    for it in hand:
        out.append({"id": it["id"], "tier": it["tier"], "experience": it["experience"], "target": it["target"],
                    "question": it["question"], "prefix": it["prefix"], "use": list(it["use"]), "aliases": it["aliases"],
                    "harvest_followups": list(it["harvest_followups"]), "swap": it["swap"],
                    "h2": None})
    for p in pool:
        slot = p["slot"]
        out.append({"id": p["id"], "tier": "pool", "experience": p["experience"], "target": p["measure"]["target"],
                    "question": itm.probe_text(p, "related"), "prefix": p["measure"]["prefix"], "use": [],
                    "aliases": [f"my {slot}", f"your {slot}", slot], "harvest_followups": list(p["followups"][:2]),
                    "swap": None, "h2": None})
    ids = [f["id"] for f in out]
    for i, f in enumerate(out):     # swap partner for the R-swap reference: hand-written as before, pool cyclic
        if f["swap"] not in ids:
            f["swap"] = ids[(i + 1) % len(ids)]
    # favourite_colour has no harvest follow-ups in the subject config: use its bench follow-ups for the key
    return out


def fact_keys(C, f, cal, layers):
    """Per key layer: K1 keys (whitened subject-span key of each follow-up's without run; the whole-user-text key
    when the follow-up names no alias) and K2 keys (the whole-user-text pooled key at the last user token)."""
    K1, K2, fallback = {l: [] for l in layers}, {l: [] for l in layers}, 0
    for fu in f["harvest_followups"]:
        ids = chat_ids(C.tok, fu)
        S = bind.subject_sites(C.tok, ids, fu, f["aliases"])
        st, _ = source_run(C, ids)
        u = user_mask(len(ids), S["user_tokens"], C.dev)
        last = S["user_tokens"][-1]
        if not S["spans"]:
            fallback += 1
        for l in layers:
            k2 = bind.pooled_keys(st[l], u, cal.mu[l], cal.W[l])[last]
            K2[l].append(k2)
            K1[l].append(bind.subject_key(st[l][S["spans"][0]], cal.mu[l], cal.W[l]) if S["spans"] else k2)
    return {l: torch.stack(K1[l]) for l in layers}, {l: torch.stack(K2[l]) for l in layers}, fallback


def query_key(C, text, cal, layers):
    ids = chat_ids(C.tok, text)
    ut = bind.subject_sites(C.tok, ids, text, [])["user_tokens"]
    st, _ = source_run(C, ids)
    u = user_mask(len(ids), ut, C.dev)
    return {l: bind.pooled_keys(st[l], u, cal.mu[l], cal.W[l])[ut[-1]] for l in layers}


def retrieve(qk, store_keys, thr):
    """qk [r]; store_keys [(fact id, [m, r])] -> (top-1 id or None if below thr, top-2 ids, top-1 match)."""
    sc = [(fid, (K @ qk).max().item()) for fid, K in store_keys]
    sc.sort(key=lambda x: -x[1])
    top1 = sc[0][0] if sc and sc[0][1] > thr else None
    return top1, [x[0] for x in sc[:2]], sc[0][1] if sc else float("nan")


def stage_gkey(C, items, out):
    """Addendum 6 (B): R-min slices + subject keys for N facts; top-1 retrieval above the q0.95 generic threshold;
    accuracy / wrong-slice / false-fire curves; teacher-forced downstream; a judged generation subset at N max."""
    cfg = C.cfg
    gen_context(C, cfg)
    a6 = cfg["gkey"]
    tiny = C.A.tiny
    cd = sibling(out.parent if out.name == "gkey" else out, "gkey")
    gdir = cd / "gen"
    (gdir / "items").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    facts = gkey_facts(cfg)
    by_id = {f["id"]: f for f in facts}
    for f in facts:   # items without subject-config follow-ups (favourite_colour): its bench follow-ups
        if not f["harvest_followups"]:
            L_ = itm.load_items(REPO / cfg["items"], REPO / cfg["bench"])["by_id"]
            f["harvest_followups"] = list(L_[f["id"]]["followups"][:2]) if f["id"] in L_ else []
    Ns = [5, 10] if tiny else list(a6["N"])
    Ns = [min(n, len(facts)) for n in Ns]
    facts = facts[:max(Ns)]
    layers = sorted({min(5, C.n - 1), min(14, C.n - 1)}) if tiny else list(a6["key_layers"])
    lsel = a6["primary_layer"] if a6["primary_layer"] in layers else layers[-1]
    A_cfg = dict(cfg["analytic"])
    cal = calibrate_generic(C, A_cfg, layers, 16 if tiny else A_cfg["n_generic"])
    generic = [l.strip() for l in open(REPO / A_cfg["generic"]) if l.strip()][:16 if tiny else A_cfg["n_generic"]]
    gk = [query_key(C, t, cal, layers) for t in generic]
    L_all = itm.load_items(REPO / cfg["items"], REPO / cfg["bench"])
    unrel = L_all["unrelated"][:5 if tiny else 50]
    uk = [query_key(C, t, cal, layers) for t in unrel]
    slices, keys, fallbacks = {}, {}, {}
    for f in facts:
        R, _ = harvest_R(C, {**f, "target": f["target"]}, cfg["gen"]["orphan_exchange"])
        slices[f["id"]] = R["R-min"]
        K1, K2, fb = fact_keys(C, f, cal, layers)
        keys[f["id"]] = {"K1": K1, "K2": K2}
        fallbacks[f["id"]] = fb
    log(f"gkey: {len(facts)} facts harvested in {(time.time() - t0) / 60:.1f} min; K1 fallbacks "
        f"{sum(fallbacks.values())} follow-ups")
    # queries: every fact's direct question; the hand-written facts' use prompts too
    queries = [(f["id"], "related", f["question"]) for f in facts] + [(f["id"], "use", u) for f in facts for u in f["use"]]
    qkeys = {(fid, t): query_key(C, t, cal, layers) for fid, _, t in queries}
    ret = []
    picks = {}
    for N in Ns:
        store = facts[:N]
        sids = {f["id"] for f in store}
        for var in ("K1", "K2"):
            for l in layers:
                sk = [(f["id"], keys[f["id"]][var][l]) for f in store]
                best = lambda q: max((K @ q).max().item() for _, K in sk)
                thr = torch.quantile(torch.tensor([best(g[l]) for g in gk]), a6["thr_q"]).item()
                ff = sum(best(u[l]) > thr for u in uk) / len(uk)
                for fid, kind, t in queries:
                    if fid not in sids:
                        continue
                    top1, top2, m = retrieve(qkeys[(fid, t)][l], sk, thr)
                    ret.append({"N": N, "var": var, "layer": l, "item": fid, "tier": by_id[fid]["tier"], "kind": kind,
                                "prompt": t, "thr": thr, "top1": top1, "top2": top2, "match": m,
                                "correct": top1 == fid, "wrong": top1 is not None and top1 != fid, "miss": top1 is None,
                                "top2_correct": fid in top2, "false_fire_unrel": ff})
                    if var == "K1" and l == lsel:
                        picks[(N, fid, t)] = top1
    write_jsonl_gz(gdir / "retrieval.jsonl.gz", ret)
    log(f"gkey: retrieval done in {(time.time() - t0) / 60:.1f} min")
    # teacher-forced downstream on the direct question: nomem, ctx, oracle, swap, retrieved (per N)
    tf = []
    for f in facts:
        t_own = int(text_ids(C.tok, f["target"])[0])
        pre = text_ids(C.tok, f["prefix"])
        qn = torch.cat([chat_ids(C.tok, f["question"]), pre])
        qx = torch.cat([chat_ids(C.tok, f"{f['experience']} {f['question']}"), pre])
        qq = torch.cat([qt_ids(C, f["question"]), pre])
        base = {"nomem": torch.log_softmax(tf_last(C, qn), -1), "ctx": torch.log_softmax(tf_last(C, qx), -1),
                "oracle": torch.log_softmax(tf_last(C, qq, slices[f["id"]]), -1)}
        sw = f["swap"] if f["swap"] in slices else None
        if sw:
            base["swap"] = torch.log_softmax(tf_last(C, qq, slices[sw]), -1)
        rows = {c: (lp, f["id"] if c in ("oracle", "ctx") else (sw if c == "swap" else None)) for c, lp in base.items()}
        for N in Ns:
            if f["id"] not in {x["id"] for x in facts[:N]}:
                continue
            top1 = picks.get((N, f["id"], f["question"]))
            lp = base["nomem"] if top1 is None else torch.log_softmax(tf_last(C, qq, slices[top1]), -1)
            rows[f"gkey@{N}"] = (lp, top1)
        for c, (lp, inj) in rows.items():
            wrong_v = int(text_ids(C.tok, by_id[inj]["target"])[0]) if inj and inj != f["id"] else None
            tf.append({"item": f["id"], "tier": f["tier"], "cond": c, "injected": inj, "lp_target": lp[t_own].item(),
                       "dlp": (lp[t_own] - base["nomem"][t_own]).item(),
                       "lp_wrong": None if wrong_v is None else lp[wrong_v].item(),
                       "wrong_value": None if wrong_v is None else bool(lp[wrong_v] > lp[t_own])})
    write_jsonl_gz(gdir / "tf.jsonl.gz", tf)
    # generation subset at the largest N: the primary facts (question + use prompts) + the first pool facts
    Nm = max(Ns)
    sub = [f for f in facts if f["tier"] == "primary"][:2 if tiny else 7] + \
          [f for f in facts if f["tier"] == "pool"][:2 if tiny else a6["n_pool_gen"]]
    S, g = cfg["collide"]["samples"], cfg["gen"]
    cap, seeds = g["answer_cap"], g["seeds"]
    items_json = {}
    for f in sub:
        items_json[f["id"]] = {"id": f["id"], "experience": f["experience"], "group": "facts", "tier": f["tier"],
                               "measure": {"target": f["target"], "prefix": f["prefix"]}}
        fp = gdir / "items" / f"{f['id']}.jsonl.gz"
        if fp.exists():
            continue
        rows = []
        prompts = [("related", f["question"])] + [("use", u) for u in f["use"]]
        for pi, (pk, text) in enumerate(prompts):
            streams = {1: (cv1.seed_of(TAG + "/gkey", text, seeds["base"]), S + 1)}
            rws = [(j == 0, None if j == 0 else (1, j)) for j in range(S + 1)]
            q = qkeys.get((f["id"], text)) or query_key(C, text, cal, layers)
            sk = [(x["id"], keys[x["id"]]["K1"][lsel]) for x in facts[:Nm]]
            thr = next(r["thr"] for r in ret if r["N"] == Nm and r["var"] == "K1" and r["layer"] == lsel)
            top1, _, _ = retrieve(q[lsel], sk, thr)
            for c in ("nomem", "ctx", "oracle", "gkey"):
                if c == "nomem" or (c == "gkey" and top1 is None):
                    outs = gen_batch(C, chat_ids(C.tok, text), rws, max_new=cap, streams=streams)
                elif c == "ctx":
                    outs = gen_batch(C, chat_ids(C.tok, f"{f['experience']} {text}"), rws, max_new=cap, streams=streams)
                else:
                    sl = slices[f["id"]] if c == "oracle" else slices[top1]
                    outs = gen_batch(C, qt_ids(C, text), rws, prefix=sl, max_new=cap, streams=streams)
                for j, o in enumerate(outs):
                    rep = cv1.rep_rate(o["ids"], g["loop"]["n"])
                    rows.append({"sec": "gen", "key": f"{f['id']}|{pk}|{pi}|{c}|1|{j}", "item": f["id"], "group": "facts",
                                 "tier": f["tier"], "kind": pk, "pi": pi, "prompt": text, "cond": c, "seedset": 1, "j": j,
                                 "retrieved": top1 if c == "gkey" else None, "ids": o["ids"],
                                 "answer": C.tok.decode(o["ids"], skip_special_tokens=True).strip(), "n_tok": o["n_tok"],
                                 "capped": o["capped"], "rep": round(rep, 4), "loop": rep >= g["loop"]["thr"],
                                 "gate": True, "shortcut": False, "no_site": False})
        write_jsonl_gz(fp, rows)
    (cd / "prep").mkdir(parents=True, exist_ok=True)
    json.dump(items_json, open(cd / "prep" / "items.json", "w"), indent=1)
    json.dump({"N": Ns, "layers": layers, "fallbacks": fallbacks, "n_facts": len(facts),
               "facts": [{"id": f["id"], "tier": f["tier"]} for f in facts]}, open(gdir / "gkey_info.json", "w"), indent=1)
    nll = coherence_nll(C, gdir)
    return [], {"coherence_unique": nll, "minutes": (time.time() - t0) / 60, "K1_fallbacks": sum(fallbacks.values())}, {}


def _ci(nested, nb, seed):
    from seahorse.stats import boot_ci
    p, lo, hi, k = boot_ci(nested, nb, seed)
    return {"mean": p, "lo": lo, "hi": hi, "n": k}


def analyze_idrepair(cfg, cd, model_key):
    """Addendum 6 (A) report: the ID offsets' validation, the teacher-forced repair table, and the generated
    best-cell comparison with the pre-registered rules."""
    from seahorse.stats import paired_diff
    nb, seed = cfg["n_boot"], cfg["seeds"]["bootstrap"]
    g = cd / "gen"
    val = read_jsonl_gz(g / "validation.jsonl.gz")
    ana = read_jsonl_gz(g / "analytic.jsonl.gz")
    pick = json.load(open(g / "pick.json"))
    L = [f"# dream_bind (A): binding-ID repair ({model_key}; PREREG addendum 6, exploratory)", "",
         f"Offsets: Feng & Steinhardt mean differences (pair 2 − pair 1) of block-input residuals at the entity (+ next) and "
         f"attribute (+ full stop) positions; Dai = projection on the order-correlated PC. Picked layer set: "
         f"**{pick['layer_set']}** (largest swap-attr flip rate on the held-out contexts).", "",
         "## Validation (held-out joint contexts): flip rate of the own-vs-partner margin", "",
         "| direction | layer set | swap-attr | swap-ent | swap-both | rand-attr | swap-attr − rand [CI] |", "|---|---|---|---|---|---|---|"]
    keyf = lambda r: (r["dir"], r["set"], r["iv"])
    by = defaultdict(lambda: defaultdict(dict))
    for r in val:
        by[keyf(r)][r["ctx"]][r["q"]] = float(r["flip"])
    rule_a = None
    for d in ("feng", "dai"):
        for sname in dict.fromkeys(r["set"] for r in val):
            cell = lambda iv: _ci(by[(d, sname, iv)], nb, seed) if by[(d, sname, iv)] else None
            diff = _ci(paired_diff(by[(d, sname, "swap-attr")], by[("feng", sname, "rand-attr")]), nb, seed)
            if d == "feng" and sname == pick["layer_set"]:
                rule_a = diff
            L.append(f"| {d} | {sname} | {fmt(cell('swap-attr'))} | {fmt(cell('swap-ent'))} | {fmt(cell('swap-both'))} | "
                     f"{fmt(cell('rand-attr'))} | {fmt(diff)} |")
    L += ["", f"**ID direction valid** (swap-attr − random, picked set, CI > 0): "
          f"{'holds' if rule_a and rule_a['lo'] > 0 else 'does not hold'} {fmt(rule_a)}", ""]
    # analytic repair table
    L += ["## Repair, teacher-forced (16 addendum-5 members)", "",
          "| condition | TF confusion | margin own − partner | log P(own) | yes/no correct (partner foil) |", "|---|---|---|---|---|"]
    for c in dict.fromkeys(r["cond"] for r in ana):
        f = lambda fld: _ci({r["item"]: {0: float(r[fld])} for r in ana if r["cond"] == c}, nb, seed)
        L.append(f"| {c} | {fmt(f('tf_conf'))} | {fmt(f('margin'))} | {fmt(f('lp_own'))} | {fmt(f('yn_correct'))} |")
    L += ["", f"Best cell (pre-registered pick): **{pick['best']['cell']}** (Δ TF confusion {pick['best']['d_conf']:.3f}, "
          f"Δ margin {pick['best']['d_margin']:.3f})", ""]
    # generation for the best cell
    rules = {}
    if (g / "items").exists() and any((g / "items").glob("*.jsonl.gz")):
        items = json.load(open(cd / "prep" / "items.json"))
        G, YN, _, _, target, n_panel = load_labeled(cfg, cd, f"{model_key}_idrepair")
        pv = {i: v["partner_value"].strip() for i, v in items.items()}
        for r in G:
            r["confusion"] = _has_word(r["answer"], pv[r["item"]]) and not _has_word(r["answer"], target[r["item"]])
        conds = list(dict.fromkeys(r["cond"] for r in G))

        def nested(c, fn):
            acc = defaultdict(list)
            for r in G:
                if r["cond"] == c and fn(r) is not None:
                    acc[(r["item"], r["pi"])].append(float(fn(r)))
            d = defaultdict(dict)
            for (i, p), vs in acc.items():
                d[i][p] = sum(vs) / len(vs)
            return dict(d)
        conf, cor = lambda c: nested(c, lambda r: r["confusion"]), lambda c: nested(c, lambda r: r["clean_maj"])
        L += [f"## Generation (best cell; judges' majority, panel labels for {n_panel}/{len(G)} answers)", "",
              "| condition | partner confusion | correct clean use | wrong value (judges) | yes/no correct |", "|---|---|---|---|---|"]
        for c in conds:
            yn = defaultdict(list)
            for r in YN:
                if r["cond"] == c:
                    yn[r["item"]].append((r["consistent"], r["yn"]))
            L.append(f"| {c} | {fmt(_ci(conf(c), nb, seed))} | {fmt(_ci(cor(c), nb, seed))} | "
                     f"{fmt(_ci(nested(c, lambda r: r['maj']['wrong_value']), nb, seed))} | "
                     f"{fmt(_ci({i: {0: bind.balanced_yn(v)} for i, v in yn.items()}, nb, seed))} |")
        best = pick["best"]
        x, b = best["cell"], best["base"]
        d0 = _ci(paired_diff(conf(x), conf(b)), nb, seed)
        ctrl = {c: _ci(paired_diff(conf(x), conf(c)), nb, seed) for c in conds if c.startswith(b + "+") and c != x
                and c.split("+")[-1] in ("rand", "neg", "wrongtok")}
        works = d0["hi"] < 0 and all(v["hi"] < 0 for v in ctrl.values()) if not math.isnan(d0["mean"]) else None
        rules = {"repair_vs_sep": d0, "repair_vs_controls": ctrl, "repair_works": works,
                 "joint_vs_sep": _ci(paired_diff(conf([c for c in conds if "joint" in c][0]), conf(b)), nb, seed)}
        L += ["", f"**Repair works** (confusion {x} − {b} CI < 0 AND below every control): "
              f"{'n/a' if works is None else ('holds' if works else 'does not hold')}",
              f"- {x} − {b}: {fmt(d0)}", *[f"- {x} − {c}: {fmt(v)}" for c, v in ctrl.items()],
              f"- joint − {b}: {fmt(rules['joint_vs_sep'])}", ""]
    out = cd / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L))
    json.dump({"id_valid": rule_a, "pick": pick, "rules": rules}, open(out / "verdict.json", "w"), indent=2, default=str)
    log(f"idrepair analyze -> {out / 'report.md'}")


def analyze_gkey(cfg, cd, model_key):
    """Addendum 6 (B) report: retrieval curves over N, key variant and layer; teacher-forced downstream; the judged
    generation subset at N max. Curves only (no pass/fail)."""
    nb, seed = cfg["n_boot"], cfg["seeds"]["bootstrap"]
    g = cd / "gen"
    ret = read_jsonl_gz(g / "retrieval.jsonl.gz")
    tf = read_jsonl_gz(g / "tf.jsonl.gz")
    info = json.load(open(g / "gkey_info.json"))
    L = [f"# dream_bind (B): G-key retrieval and capacity ({model_key}; PREREG addendum 6, exploratory)", "",
         f"{info['n_facts']} facts; stores N = {info['N']}; key layers {info['layers']}; K1 follow-ups without an alias "
         f"(whole-text key used): {sum(info['fallbacks'].values())}. Query key at the question's last user token; threshold "
         f"q{cfg['gkey']['thr_q']} of the generic prompts' best match.", "",
         "## Retrieval (direct questions; use prompts of the hand-written facts in brackets)", "",
         "| N | key | layer | correct | wrong slice | miss | top-2 correct | false fire (unrelated) | [use prompts: correct] |",
         "|---|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in ret:
        groups[(r["N"], r["var"], r["layer"], r["kind"])].append(r)
    for (N, var, l, kind), rs in sorted(groups.items()):
        if kind != "related":
            continue
        m = lambda f: _ci({r["item"]: {0: float(r[f])} for r in rs}, nb, seed)
        us = groups.get((N, var, l, "use"), [])
        u = _ci({r["item"]: {r["prompt"]: float(r["correct"])} for r in us}, nb, seed) if us else None
        L.append(f"| {N} | {var} | {l} | {fmt(m('correct'))} | {fmt(m('wrong'))} | {fmt(m('miss'))} | "
                 f"{fmt(m('top2_correct'))} | {rs[0]['false_fire_unrel']:.2f} | {fmt(u)} |")
    L += ["", "## Teacher-forced downstream (direct question): Δlog P(target) vs nomem; wrong-value rate when a wrong "
          "slice is injected", "", "| condition | Δlog P(target) | wrong value (of injected wrong slices) | n wrong injections |",
          "|---|---|---|---|"]
    for c in dict.fromkeys(r["cond"] for r in tf):
        rs = [r for r in tf if r["cond"] == c]
        wv = [r for r in rs if r["wrong_value"] is not None]
        L.append(f"| {c} | {fmt(_ci({r['item']: {0: r['dlp']} for r in rs}, nb, seed))} | "
                 f"{fmt(_ci({r['item']: {0: float(r['wrong_value'])} for r in wv}, nb, seed)) if wv else '–'} | {len(wv)} |")
    L.append("")
    if (g / "items").exists() and any((g / "items").glob("*.jsonl.gz")):
        G, _, _, _, target, n_panel = load_labeled(cfg, cd, f"{model_key}_gkey")
        items = json.load(open(cd / "prep" / "items.json"))
        L += [f"## Generation at N = {max(info['N'])} (judges' majority; panel labels for {n_panel}/{len(G)} answers)", "",
              "| tier / prompts | condition | clean use | confab (wrong value) | target mentioned |", "|---|---|---|---|---|"]
        for tier, kinds in (("primary", ("use",)), ("primary", ("related",)), ("pool", ("related",))):
            for c in ("nomem", "ctx", "oracle", "gkey"):
                rs = [r for r in G if items[r["item"]].get("tier") == tier and r["kind"] in kinds and r["cond"] == c]
                if not rs:
                    continue
                def nest(fn):
                    acc = defaultdict(list)
                    for r in rs:
                        if fn(r) is not None:
                            acc[(r["item"], r["pi"])].append(float(fn(r)))
                    d = defaultdict(dict)
                    for (i, p), vs in acc.items():
                        d[i][p] = sum(vs) / len(vs)
                    return dict(d)
                L.append(f"| {tier} / {','.join(kinds)} | {c} | {fmt(_ci(nest(lambda r: r['clean_maj']), nb, seed))} | "
                         f"{fmt(_ci(nest(lambda r: r['maj']['wrong_value']), nb, seed))} | "
                         f"{fmt(_ci(nest(lambda r: _has_word(r['answer'], target[r['item']])), nb, seed))} |")
        L.append("")
    out = cd / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L))
    log(f"gkey analyze -> {out / 'report.md'}")


# --------------------------------------------------------------------------- judge (original) + analyze


def stage_judge(A, cfg, run_dir):
    """The original judge (Qwen3.5-9B bf16, core_v1's judge_prompts.yaml, prompt filling and parser), through
    core_v1's own score code -> <run>/score/judge.jsonl.gz."""
    cvr = _load("dream_bind_core_v1_run", HERE.parent / "core_v1" / "run.py")
    rows = []
    for f in sorted((run_dir / "gen" / "items").glob("*.jsonl.gz")):
        rows += [r for r in read_jsonl_gz(f) if r["sec"] == "gen"]
    L = {"by_id": json.load(open(run_dir / "prep" / "items.json"))}
    cfg_j = {"judge": cfg["judge"], "model": "Qwen/Qwen3.5-2B"}   # 'model' only names the random-model tokenizer
    Aj = SimpleNamespace(random_model="qwen2" if A.random_model else None, device=A.device)
    out = run_dir / "score"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = cvr._score_judge(cfg_j, Aj, L, rows, out, 24 if A.tiny else None)
    res.update(minutes=(time.time() - t0) / 60, env=env_meta())
    if torch.cuda.is_available():
        res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    res["peak_host_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)
    json.dump(res, open(out / "judge_config.json", "w"), indent=2, default=str)
    log(f"judge: {json.dumps({k: v for k, v in res.items() if k != 'env'})}")


def _has_word(text, w):
    return bool(re.search(rf"(?<![\w-]){re.escape(w.strip())}(?![\w-])", text or "", re.I))


def load_labeled(cfg, run_dir, model_key):
    """A run's answers with their labels: (G gen rows, YN yes/no rows, UN unrelated rows, tier, target, n_panel).
    Clean use = the judges' majority (original + gpt-oss + Gemma-4; a tie goes to the original's label)."""
    items = json.load(open(run_dir / "prep" / "items.json"))
    tier = {i: ("primary" if i in cfg["primary"] else "secondary") for i in items}
    target = {i: items[i]["measure"]["target"].strip() for i in items}
    G, YN = [], []
    for f in sorted((run_dir / "gen" / "items").glob("*.jsonl.gz")):
        for r in read_jsonl_gz(f):
            if r["sec"] in ("gen", "yn"):
                (G if r["sec"] == "gen" else YN).append(r)
    UN = []
    for f in sorted((run_dir / "gen" / "unrel").glob("*.jsonl.gz")):
        UN += read_jsonl_gz(f)
    sc = run_dir / "score"
    nll = {r["key"]: r["nll"] for r in read_jsonl_gz(sc / "coherence.jsonl.gz")} if (sc / "coherence.jsonl.gz").exists() else {}
    orig = {r["key"]: r for r in read_jsonl_gz(sc / "judge.jsonl.gz")} if (sc / "judge.jsonl.gz").exists() else {}
    panel = {}
    pdir = run_dir.parent / "panel"
    for jt in ("gptoss", "gemma4"):
        p = pdir / f"{jt}.jsonl"
        if p.exists():
            for line in open(p):
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                for k in rec["keys"]:
                    if k.startswith(model_key + ":"):
                        panel.setdefault(k[len(model_key) + 1:], {})[jt] = rec if rec.get("ok") else None
    fields = ("uses_fact", "coherent", "self_claim", "wrong_value")
    for r in G:
        o = orig.get(r["key"])
        o = o if o and o.get("ok") else None
        lab = {}
        for fld in fields:
            votes = [o.get(fld) if o else None] + [(panel.get(r["key"], {}).get(jt) or {}).get(fld) for jt in ("gptoss", "gemma4")]
            lab[fld] = bind.majority(votes, tie=o.get(fld) if o else None)
        r["maj"] = lab
        r["clean_maj"] = None if lab["uses_fact"] is None or lab["coherent"] is None else bool(lab["uses_fact"] and lab["coherent"])
        r["clean_orig"] = None if not o else bool(o["uses_fact"] and o["coherent"])
        r["regex_clean"] = _has_word(r["answer"], target[r["item"]]) and not r["loop"]
        r["mention"] = _has_word(r["answer"], target[r["item"]])
        r["mention_unclean"] = r["mention"] and not bool(r["clean_maj"])
        r["nll"] = nll.get(r["key"])
    n_panel = sum(1 for r in G if r["key"] in panel)
    return G, YN, UN, tier, target, n_panel


def stage_analyze(cfg, run_dir, model_key="qwen25_15b"):
    """Every table and the pre-registered verdicts from the saved files (no GPU). Clean use = the judges'
    majority (original + gpt-oss + Gemma-4; a tie goes to the original's label) of uses_fact AND coherent; the
    original judge alone is reported beside it."""
    from seahorse.stats import boot_ci, paired_diff
    nb, seed = cfg["n_boot"], cfg["seeds"]["bootstrap"]
    G, YN, UN, tier, target, n_panel = load_labeled(cfg, run_dir, model_key)

    def nested(rows, fn, ss=1):
        d = defaultdict(dict)
        acc = defaultdict(list)
        for r in rows:
            if r["seedset"] != ss:
                continue
            v = fn(r)
            if v is None:
                continue
            acc[(r["item"], r["pi"])].append(float(v))
        for (i, p), vs in acc.items():
            d[i][p] = sum(vs) / len(vs)
        return dict(d)

    def sel(cond, tr, kinds=("use",)):
        return [r for r in G if r["cond"] == cond and tier[r["item"]] == tr and r["kind"] in kinds]

    def ci(n):
        p, lo, hi, k = boot_ci(n, nb, seed)
        return {"mean": p, "lo": lo, "hi": hi, "n": k}

    conds = list(dict.fromkeys(r["cond"] for r in G))
    measures = {"clean (majority)": lambda r: r["clean_maj"], "clean (orig judge)": lambda r: r["clean_orig"],
                "regex clean": lambda r: r["regex_clean"], "mention, unclean": lambda r: r["mention_unclean"],
                "confab (wrong value)": lambda r: r["maj"]["wrong_value"], "self-claim": lambda r: r["maj"]["self_claim"],
                "loop": lambda r: r["loop"], "incoherent": lambda r: None if r["maj"]["coherent"] is None else not r["maj"]["coherent"],
                "NLL": lambda r: r["nll"]}

    def yn_bal(cond, tr):
        per = defaultdict(list)
        for r in YN:
            if r["cond"] == cond and tier[r["item"]] == tr:
                per[r["item"]].append((r["consistent"], r["yn"]))
        return {i: {0: bind.balanced_yn(v)} for i, v in per.items()}

    T = {}
    L = [f"# dream_bind: arm R + subj generation on {model_key}", "",
         f"Pre-registered (PREREG.md addendum 'Arm R and the Qwen2.5 generation run'). Clean use = the majority of the "
         f"original judge + gpt-oss + Gemma-4 (panel labels for {n_panel}/{len(G)} answers; a tie goes to the original's "
         f"label). Means over items then prompts, two-level bootstrap ({nb}, seed {seed}); seed set 1.", ""]
    for tr in ("primary", "secondary"):
        L += [f"## Use prompts ({tr})", "", "| condition | " + " | ".join(measures) + " | bal yes/no acc | Δ bal acc vs nomem |",
              "|---|" + "---|" * (len(measures) + 2)]
        bal_nm = yn_bal("nomem", tr)
        for c in conds:
            rows = sel(c, tr)
            if not rows:
                continue
            vals = {m: ci(nested(rows, fn)) for m, fn in measures.items()}
            T[(tr, c)] = vals
            b = yn_bal(c, tr)
            vals["bal"] = ci(b)
            vals["bal_gain"] = ci(paired_diff(b, bal_nm))
            L.append(f"| {c} | " + " | ".join(fmt(vals[m]) for m in measures) + f" | {fmt(vals['bal'])} | {fmt(vals['bal_gain'])} |")
        L.append("")
        # direct question: regex clean / clean
        L += [f"**Direct question ({tr})**: clean (majority) / regex clean", ""]
        for c in conds:
            rows = sel(c, tr, ("related",))
            if rows:
                L.append(f"- {c}: {fmt(ci(nested(rows, measures['clean (majority)'])))} / {fmt(ci(nested(rows, measures['regex clean'])))}")
        L.append("")
    # paired comparisons (primary, clean majority, use prompts)
    cm = lambda c: nested(sel(c, "primary"), measures["clean (majority)"])
    pd = lambda a, b: ci(paired_diff(cm(a), cm(b)))
    L += ["## Paired differences in clean use (primary, use prompts)", "", "| comparison | difference [CI] |", "|---|---|"]
    has_subj = "subj1" in conds
    comps = [("subj1", "nomem"), ("subj1", "ctrl:wrongpos"), ("subj1", "ctrl:rand"), ("subj1", "ctrl:swap"),
             ("subj1", "ctrl:placebo"), ("ctrl:wrongpos", "nomem"), ("R-full", "subj1"), ("R-full", "nomem"),
             ("R-min", "nomem"), ("R-min", "R-full"), ("R-min", "subj1"), ("R-orphan", "nomem"), ("R-orphan", "R-full"),
             ("R-orphan", "subj1"), ("R-swap", "nomem"), ("ctx", "R-full"), ("subj2", "nomem")] if has_subj else [
             ("R-full-state", "ctx"), ("R-full-state", "nomem"), ("R-full-kv6", "nomem"), ("R-full-kv6", "R-full-state"),
             ("R-min-kv6", "R-full-kv6"), ("R-min-kv6", "nomem"), ("R-orphan-kv6", "R-full-kv6"), ("R-orphan-kv6", "nomem"),
             ("R-swap-kv6", "nomem"), ("ctx", "R-full-kv6"),
             ("R-min-state", "R-full-state"), ("R-min-state", "R-min-kv6"), ("R-min-state", "nomem"),
             ("R-state-only", "nomem"), ("R-state-only", "R-min-kv6"), ("R-state-only", "R-min-state"),
             ("R-orphan-state", "R-min-state"), ("R-orphan-state", "nomem"), ("R-swap-min-state", "nomem"),
             ("T-min", "nomem"), ("T-min", "R-min"), ("T-min", "R-min-state"), ("R-min", "nomem"),
             ("T-full", "R-full"), ("T-full", "R-full-state"), ("T-full", "ctx")]
    comps = [(x, y) for x, y in comps if x in conds and y in conds]
    PD = {}
    for a, b in comps:
        PD[(a, b)] = pd(a, b)
        L.append(f"| {a} − {b} | {fmt(PD[(a, b)])} |")
    L.append("")
    # yes/no balanced accuracy gains (primary)
    bal = lambda c: yn_bal(c, "primary")
    ref1, ref2 = ("R-full", "subj1") if has_subj else (("R-min-kv6", "R-full-state") if "R-min-state" in conds
                                                         else ("R-full-kv6", "R-full-state"))
    L += ["## Balanced yes/no (primary)", "",
          f"| condition | balanced accuracy | gain vs nomem | − {ref1} | − {ref2} | − ctx |", "|---|---|---|---|---|---|"]
    for c in conds:
        b = bal(c)
        if b:
            L.append(f"| {c} | {fmt(ci(b))} | {fmt(ci(paired_diff(b, bal('nomem'))))} | {fmt(ci(paired_diff(b, bal(ref1))))} | "
                     f"{fmt(ci(paired_diff(b, bal(ref2))))} | {fmt(ci(paired_diff(b, bal('ctx'))))} |")
    L.append("")
    # selectivity
    us = defaultdict(list)
    for r in UN:
        if r["cond"] != "nomem":
            us[r["cond"]].append(bool(r["same"]))
    unrel_same = {c: sum(v) / len(v) for c, v in us.items()}
    L += ["## Unrelated prompts: share token-identical to nomem", ""]
    L += [f"- {c}: {v:.3f} ({len(us[c])} answers{'; ungated leak baseline' if c.endswith(':ungated') else ''})"
          for c, v in sorted(unrel_same.items())]
    L.append("")
    # cross-binding: the partner's target in this item's answers
    leak = {}
    for c in conds:
        d = defaultdict(dict)
        for a, b in PAIRS:
            for x, y in ((a, b), (b, a)):   # on y's prompts, x's target appearing
                acc = defaultdict(list)
                for r in G:
                    if r["item"] == y and r["cond"] == c and r["seedset"] == 1:
                        acc[r["pi"]].append(float(_has_word(r["answer"], target[x])))
                for p, vs in acc.items():
                    d[f"{x}->{y}"][p] = sum(vs) / len(vs)
        leak[c] = dict(d)
    L += ["## Cross-binding leak (the partner's target in the answer; 6 directed pairs, main prompts)", ""]
    for c in conds:
        if leak[c]:
            L.append(f"- {c}: {fmt(ci(leak[c]))}; − nomem {fmt(ci(paired_diff(leak[c], leak['nomem'])))}")
    L.append("")
    # seed set 2
    L += ["## Seed set 2 (robustness): clean use (majority), primary use prompts", ""]
    for c in [c for c in conds if any(r["seedset"] == 2 and r["cond"] == c for r in G)]:
        L.append(f"- {c}: {fmt(ci(nested(sel(c, 'primary'), measures['clean (majority)'], ss=2)))}")
    L.append("")
    # direct question ("knows it knows"): clean use per condition, paired contrasts, and the greedy answers
    cq = lambda c: nested(sel(c, "primary", ("related",)), measures["clean (majority)"])
    L += ["## Direct question (primary): does the model know that it knows?", "",
          "Clean use (majority) on the direct question per condition, paired contrasts, and every item's greedy answer.", ""]
    for a_, b_ in [x for x in (("T-min", "nomem"), ("T-min", "R-min"), ("T-min", "R-min-state"),
                               ("R-min-state", "R-min-kv6"), ("R-min-state", "R-full-state"), ("R-state-only", "nomem"),
                               ("R-state-only", "R-min-kv6"), ("R-orphan-state", "R-min-state"), ("R-full-kv6", "R-full-state"))
                   if x[0] in conds and x[1] in conds]:
        L.append(f"- {a_} − {b_}: {fmt(ci(paired_diff(cq(a_), cq(b_))))}")
    L += ["", "| item | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for i in sorted({r["item"] for r in G if tier[r["item"]] == "primary"}):
        cells_ = []
        for c in conds:
            r = next((r for r in G if r["item"] == i and r["cond"] == c and r["kind"] == "related" and r["seedset"] == 1
                      and r["j"] == 0), None)
            cells_.append("" if r is None else r["answer"][:90].replace("|", "/").replace("\n", " "))
        L.append(f"| {i} | " + " | ".join(cells_) + " |")
    L.append("")
    # rules
    if not has_subj:
        return analyze_hybrid(cfg, run_dir, model_key, L, T, PD, unrel_same, leak, cm, ci, paired_diff, n_panel, len(G))
    s = {"subj": T[("primary", "subj1")]["clean (majority)"], "subj_m_wrong": PD[("subj1", "ctrl:wrongpos")],
         "bal_gain_subj": T[("primary", "subj1")]["bal_gain"], "bal_subj": T[("primary", "subj1")]["bal"],
         "wrong_gain": PD[("ctrl:wrongpos", "nomem")], "leak_gain": ci(paired_diff(leak["subj1"], leak["nomem"])),
         "unrel_same": {c: v for c, v in unrel_same.items() if c in RESID + RCONDS},
         "reinst_m_subj": PD[("R-full", "subj1")], "analytic_gain": bool(cfg["gen"].get("analytic_gain_qwen25", True)),
         "subj_all": [T[("primary", c)]["clean (majority)"] for c in ("subj1", "subj2") if ("primary", c) in T]}
    V = bind.gen_rules(s)
    L[4:4] = ["## Verdicts (pre-registered)", "", f"**{'; '.join(V['labels'])}**", "",
              *[f"- {k}: {'holds' if v else 'fails'}" for k, v in V["B"].items()],
              f"- B1 reference on Qwen2.5: core_v1's push ceiling {bind.PUSH_CEILING} (no push@1 run on this model).",
              "- B5 is decided with R-full as `reinst`; R-min / R-orphan are reported, not decisive.", ""]
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L))
    json.dump({"verdict": V, "inputs": s, "n_panel_labels": n_panel, "n_answers": len(G)}, open(out / "verdict.json", "w"),
              indent=2, default=str)
    log(f"analyze -> {out / 'report.md'}: {V['labels']}")
    return V


def analyze_hybrid(cfg, run_dir, model_key, L, T, PD, unrel_same, leak, cm, ci, paired_diff, n_panel, n_answers):
    """Qwen3.5-2B (PREREG addendum 3): no subj arm (pre-registered skip), so B1-B4 do not apply; B6 for the gated R
    conditions; the predictions are reported: R-full-state ~ ctx, kv6 weaker than Qwen2.5's R (B5), kv6 vs state,
    R-min-kv6 vs R-full-kv6."""
    gcf = run_dir / "gen" / "gen_conds.json"
    gc = json.load(open(gcf)) if gcf.exists() else (cfg["models"][model_key].get("gen_conds") or {})
    gated = [c for c in unrel_same if c.startswith(("R-", "T-")) and not c.endswith(":ungated")]
    B6 = all(unrel_same[c] >= 0.95 for c in gated) if gated else None
    rep_ = {"R-full-state - ctx (expected ~0)": PD.get(("R-full-state", "ctx")),
            "R-full-kv6 - R-full-state": PD.get(("R-full-kv6", "R-full-state")),
            "R-min-kv6 - R-full-kv6": PD.get(("R-min-kv6", "R-full-kv6")),
            "R-min-state - R-full-state (expected: CI includes 0, point > -0.1)": PD.get(("R-min-state", "R-full-state")),
            "R-min-state - R-min-kv6 (expected > 0)": PD.get(("R-min-state", "R-min-kv6")),
            "R-state-only - nomem (open)": PD.get(("R-state-only", "nomem")),
            "R-state-only - R-min-kv6 (open)": PD.get(("R-state-only", "R-min-kv6")),
            "R-orphan-state - R-min-state (reported)": PD.get(("R-orphan-state", "R-min-state")),
            "T-min - nomem (addendum 5: expected ~0)": PD.get(("T-min", "nomem")),
            "T-min - R-min (addendum 5: expected well below 0)": PD.get(("T-min", "R-min")),
            "T-min - R-min-state (addendum 5: expected well below 0)": PD.get(("T-min", "R-min-state")),
            "T-full - R-full (= by construction)": PD.get(("T-full", "R-full")),
            "T-full - R-full-state (= by construction)": PD.get(("T-full", "R-full-state"))}
    if ("R-min-state", "R-full-state") in PD and not math.isnan(PD[("R-min-state", "R-full-state")]["mean"]):
        d = PD[("R-min-state", "R-full-state")]
        rep_["compact hybrid memory (CI of R-min-state - R-full-state includes 0 AND point > -0.1)"] = {
            "holds": bool(d["lo"] <= 0 <= d["hi"] and d["mean"] > -0.1)}
    b5 = None
    ref = gc.get("b5_ref")
    if ref and Path(ref, "gen", "items").exists():
        try:
            G2, _, _, tier2, _, _ = load_labeled(cfg, Path(ref), "qwen25_15b")
            d2 = defaultdict(lambda: defaultdict(list))
            for r in G2:
                if r["cond"] == "R-full" and r["seedset"] == 1 and r["kind"] == "use" and tier2[r["item"]] == "primary" \
                        and r["clean_maj"] is not None:
                    d2[r["item"]][r["pi"]].append(float(r["clean_maj"]))
            q25 = {i: {p: sum(v) / len(v) for p, v in ps.items()} for i, ps in d2.items()}
            b5 = {"Qwen2.5 R-full": ci(q25), "2B R-full-kv6 - Qwen2.5 R-full (expected < 0)": ci(paired_diff(cm("R-full-kv6"), q25)),
                  "2B R-full-state - Qwen2.5 R-full": ci(paired_diff(cm("R-full-state"), q25))}
        except Exception as e:   # the reference run is missing or unlabeled: reported, not fatal
            b5 = {"error": f"{type(e).__name__}: {e}"}
    decomp = None
    af = run_dir / "gen" / "analytic_tf.jsonl.gz"
    if af.exists():
        from seahorse.stats import boot_ci
        an = read_jsonl_gz(af)
        cq = lambda c, f: (lambda p: {"mean": p[0], "lo": p[1], "hi": p[2], "n": p[3]})(
            boot_ci({r["item"]: {0: r[f]} for r in an if r["cond"] == c}, cfg["n_boot"], cfg["seeds"]["bootstrap"]))
        acs = list(dict.fromkeys(r["cond"] for r in an))
        tab = {c: {f: cq(c, f) for f in ("yn_bal_acc", "yn_margin", "dlp")} for c in acs}
        rec, both = tab.get("R-state-only-rec"), tab.get("R-state-only")
        holds = None
        if rec and both and not math.isnan(rec["yn_bal_acc"]["mean"]):
            holds = bool(rec["yn_bal_acc"]["lo"] > 0.5 and abs(rec["yn_bal_acc"]["mean"] - both["yn_bal_acc"]["mean"]) <= 0.1)
        conv = tab.get("R-state-only-conv")
        conv_only = bool(conv and holds is False and conv["yn_bal_acc"]["lo"] > 0.5)
        decomp = {"table": tab, "recognition_in_recurrent_state": holds, "conv_only": conv_only}
        L += ["", "## Addendum 6b: what in the restored DeltaNet state carries recognition (teacher-forced, primary items)", "",
              "R-state-only and R-min-state restore BOTH the conv buffer and the recurrent state; -rec = recurrent only "
              "(conv fresh); -conv = conv buffer only (recurrent fresh).", "",
              "| condition | yes/no balanced accuracy (TF) | yes/no margin | Δlog P(target) |", "|---|---|---|---|"]
        for c in acs:
            L.append(f"| {c} | {fmt(tab[c]['yn_bal_acc'])} | {fmt(tab[c]['yn_margin'])} | {fmt(tab[c]['dlp'])} |")
        L += ["", f"**Recognition lives in the recurrent state** (R-state-only-rec CI > 0.5 AND within 0.1 of "
              f"R-state-only): {'n/a' if holds is None else ('holds' if holds else 'does not hold')}"
              + ("; only the conv buffer carries it: a short verbatim-buffer effect, not familiarity" if conv_only else ""), ""]
    is2b = model_key.startswith("qwen35")
    V = {"labels": ["reported (no subj arm on the 2B: pre-registered skip; B1-B4 not applicable)" if is2b else
                    "reported (no subj arm in this run; B1-B4 not applicable)"],
         "B": {"B6": B6}, "reported": rep_, "B5_vs_qwen25": b5, "decomp": decomp}
    add4 = gc.get("gen_set") in ("state", "text", "decomp")
    L[4:4] = [f"## Verdicts (PREREG addendum {({'state': '4', 'text': '5 (1)', 'decomp': '6b (C)'}).get(gc.get('gen_set'), '3')}"
              f"{', exploratory: all reported, not decisive' if add4 else ''})", "",
              ("**No subj arm on Qwen3.5-2B** (pre-registered skip: stage 0 pointer-suspected AND the analytic pre-pass "
               "said NO GAIN), so B1-B4 do not apply. B5 on the 2B is reported, not decisive." if is2b else
               "**No subj arm in this run** (addendum 5's text controls), so B1-B4 do not apply."), "",
              f"- B6 (gated R / T conditions, unrel_same >= 0.95): {'holds' if B6 else ('fails' if B6 is not None else 'n/a')}",
              *[f"- {k}: {fmt(v) if 'mean' in v else v}" for k, v in rep_.items() if v],
              *([f"- B5: {k}: {fmt(v) if isinstance(v, dict) and 'mean' in v else v}" for k, v in b5.items()] if b5 else
                ([] if not ref else ["- B5: Qwen2.5 reference run not found (reported as missing)"])), ""]
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L))
    json.dump({"verdict": V, "n_panel_labels": n_panel, "n_answers": n_answers,
               "clean_use": {f"{tr}/{c}": v["clean (majority)"] for (tr, c), v in T.items()}},
              open(out / "verdict.json", "w"), indent=2, default=str)
    log(f"analyze -> {out / 'report.md'}: {V['labels']}")
    return V


# --------------------------------------------------------------------------- main


def main(argv=None):
    A = parse_args(argv)
    cfg = yaml.safe_load(open(A.config))
    mc = cfg["models"][A.model]
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in A.stage if s != "report"]
    for s in stages:
        assert s in mc["stages"] or A.random_model, f"stage {s} is not run on {A.model} (config stages {mc['stages']})"
    if "judge" in stages:          # no experiment model: the original judge on the saved answers
        for d_ in (out, collide_dir(out), sibling(out, "idrepair"), sibling(out, "gkey")):
            if (d_ / "gen" / "items").exists() and any((d_ / "gen" / "items").glob("*.jsonl.gz")):
                stage_judge(A, cfg, d_)
    if "analyze" in stages:
        if (out / "gen" / "items").exists():
            stage_analyze(cfg, out, A.model)
        if (collide_dir(out) / "gen" / "items").exists():
            analyze_collide(cfg, collide_dir(out), A.model)
        if (sibling(out, "idrepair") / "gen" / "pick.json").exists():
            analyze_idrepair(cfg, sibling(out, "idrepair"), A.model)
        if (sibling(out, "gkey") / "gen" / "retrieval.jsonl.gz").exists():
            analyze_gkey(cfg, sibling(out, "gkey"), A.model)
    stages = [s for s in stages if s not in ("judge", "analyze")]
    if not stages and ("judge" in A.stage or "analyze" in A.stage):
        return
    if stages:
        items = load_items(cfg, A.tiny)
        all_items = load_items(cfg)
        fns = {"patch": stage_patch, "knockout": stage_knockout, "prefix": stage_prefix,
               "analytic": lambda C_, its, d_: stage_analytic(C_, its, d_, all_items),
               "gen": lambda C_, its, d_: stage_gen(C_, its, d_, all_items),
               "collide": stage_collide, "idrepair": stage_idrepair, "gkey": stage_gkey}
        if "gen" in stages:
            log(f"gen plan: {json.dumps(gen_plan(cfg, items, mc=mc, gen_set=A.gen_set))}")
        if "collide" in stages:
            log(f"collide plan: {json.dumps(collide_plan(cfg, mc))}")
        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model, tok, info = load(A, cfg, mc)
            C = context(A, cfg, mc, model, tok)
            coh = None
            if not A.random_model:
                coh = coherence_check(C)
                log(f"coherence: {coh}")
                assert coh["ok"], f"the model does not answer coherently: {coh}"
            for st in stages:
                t0 = time.time()
                rows, checks, layouts = fns[st](C, items, out / st)
                checks["coherence"] = coh
                d = {"collide": collide_dir(out) / "gen", "idrepair": sibling(out, "idrepair") / "gen",
                     "gkey": sibling(out, "gkey") / "gen"}.get(st, out / st)
                d.mkdir(parents=True, exist_ok=True)
                write_jsonl_gz(d / "results.jsonl.gz", rows)
                json.dump(checks, open(d / "checks.json", "w"), indent=2)
                meta = {"stage": st, "model_key": A.model, "model": mc["model"], "dtype": A.dtype,
                        "random_model": A.random_model, "tiny": A.tiny, "args": vars(A), "cfg": cfg, "env": env_meta(),
                        "load": info, "n_blocks": C.n, "items": [it["id"] for it in items], "layouts": layouts,
                        "items_sha": {"config": file_sha(A.config), "facts": file_sha(REPO / cfg["bench"] / "facts.yaml"),
                                      "core_v1_items": file_sha(REPO / cfg["items"])},
                        "minutes": round((time.time() - t0) / 60, 2)}
                if torch.cuda.is_available():
                    meta["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                    meta["peak_gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
                meta["peak_host_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)
                json.dump(meta, open(d / "config.json", "w"), indent=2, default=str)
                log(f"{st} done: {len(rows)} rows in {meta['minutes']} min -> {d}"
                    + (f"; peak GPU {meta['peak_gpu_gb']} GB" if "peak_gpu_gb" in meta else ""))
    report(out, cfg, A.model)
    report_prefix(out, cfg, A.model)
    report_analytic(out, cfg, A.model)
    if A.max_reserved_gb is not None and torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved() / 2 ** 30
        log(f"fit check: peak reserved {peak:.3f} GB (limit {A.max_reserved_gb} GB), allocated "
            f"{torch.cuda.max_memory_allocated() / 2 ** 30:.3f} GB")
        if peak > A.max_reserved_gb:
            raise SystemExit(f"FIT CHECK FAILED: peak reserved {peak:.2f} GB > {A.max_reserved_gb} GB")


if __name__ == "__main__":
    main()
