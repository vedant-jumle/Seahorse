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
import resource
import subprocess
import time
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", nargs="+", required=True, choices=["patch", "knockout", "report"])
    p.add_argument("--model", required=True, choices=["qwen35_2b", "qwen25_15b"])
    p.add_argument("--out", required=True, help="the model's run dir, e.g. <run>/qwen25_15b")
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--tiny", action="store_true", help="2 items (smoke)")
    p.add_argument("--random-model", default=None, choices=["qwen2"],
                   help="tiny random Qwen2 (eager) with the config model's tokenizer (CPU dry runs)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
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
    if A.random_model:
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
    if mc["eager"] or A.random_model:
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
    return SimpleNamespace(src=src, tgt=tgt, nomem=nomem, T=T, a=a, t_tok=t_ids[0], c_tok=c_tok, q0=ls["q0"],
                           subj=ls["subj"], alias=ls["alias"],
                           info={"T": T, "attr_pos": a, "attr_src": dec(src[a]), "attr_tgt": dec(c_tok), "q0": ls["q0"],
                                 "q0_tok": dec(src[ls["q0"]]), "alias": ls["alias"], "subj": ls["subj"],
                                 "subj_toks": [dec(src[i]) for i in ls["subj"]], "target_multi_token": len(t_ids) > 1,
                                 "last_tok": dec(src[-1]), "T_nomem": len(nomem)})


def sites(P):
    s = {}
    if P.subj:
        s["subj_last"] = [P.subj[-1]]
        s["subj_span"] = list(P.subj)
    s["attr"] = [P.a]
    return s


# --------------------------------------------------------------------------- forward


def last_logits(C, X, patch=None, src=None, block=None):
    """[B, V] fp32 logits at the last position, with optional per-row patching / blocking plans."""
    with ExitStack() as es:
        if patch:
            es.enter_context(bind.patching(C.blocks, patch, src))
        if block is not None:
            es.enter_context(bind.blocking(C.blocks, block))
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


def run_rows(C, ids, rows, src=None):
    """rows: [{patch: [(layers, positions)], block: [(layers, queries, keys)], hook_all: bool, keep: bool}]
    on copies of `ids`, in batches. Returns per row (logits[t], logits[c] filled by the caller's measure)
    the full last-position logits for rows with keep=True, else None, plus the measure inputs."""
    out = []
    for i in range(0, len(rows), C.mb):
        ch = rows[i:i + C.mb]
        pp, bp = defaultdict(list), None
        for r, row in enumerate(ch):
            for layers, pos in row.get("patch", []):
                for l in layers:
                    pp[l].append((r, pos))
            if row.get("block") or row.get("hook_all"):
                bp = bp if bp is not None else {l: [] for l in range(C.n)}
                for layers, qs, ks in row.get("block", []):
                    for l in layers:
                        bp[l].append((r, qs, ks))
        X = ids[None].expand(len(ch), -1)
        lg = last_logits(C, X, patch=dict(pp), src=src, block=bp)
        out.append(lg)
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
    if stages:
        items = load_items(cfg, A.tiny)
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
                rows, checks, layouts = (stage_patch if st == "patch" else stage_knockout)(C, items, out / st)
                checks["coherence"] = coh
                d = out / st
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


if __name__ == "__main__":
    main()
