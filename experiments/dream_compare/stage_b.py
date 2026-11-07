#!/usr/bin/env python
"""dream_compare Stage B runner: memory paradigms end to end (PREREG.md "Stage B"). Stages (--stage):

  b-run        (GPU) per benchmark split (--bench longmemeval:oracle,locomo:locomo10,...): per context, the memory is
               rendered, encoded once per session (session-level prefill, SDPA) into a CPU store; per item BM25
               retrieval (m = 3, 10 turns; PersonaMem 1, 3 sessions); P4's picks (S-loo, S-dream; query-free, per
               turn, cached); then every condition (P0 ... P5s) answered in ragged batches: 1 greedy answer per
               item (PersonaMem: option log-probs), + 4 samples per generative condition on the 100-item subset;
               latency (B = 1) on its first 20 items; storage accounting. Identity checks first (fail the run):
               P3 with every turn of a session restored == the session as text; ragged rows == gen_batch per row.
               Resumable: items already in <out>/<bench>_<split>/items/ are skipped. --max-hours stops cleanly.
  b-smoke      (GPU) b-run on --limit 1 item per benchmark (the subset's first), + the identity checks, peak memory
  b-tasks      (CPU) judge prompts of a model's greedy answers -> <out>/tasks.jsonl (LongMemEval per-type prompts,
               LoCoMo J on the non-adversarial items, BEAM rubric per item); --from = the model's gen dir
  b-judge-local (GPU) the original judge (Qwen3.5-9B bf16, thinking off, greedy) on <out>/tasks.jsonl, batched with
               left padding (checked against run.py's unpadded path on the first prompts) -> <out>/orig.jsonl
  b-analyze    (CPU) analyze_b.analyze: per-model report_b.md / verdict_b.json (--from gen dir, --judge-dir)
  b-summary    (CPU) analyze_b.summary over a run root (--out = the root)
  other stages are run.py's (judge-local, judge-summary, ...)

  python -I experiments/dream_compare/stage_b.py --stage b-smoke --model qwen25_15b --out RUN/qwen25_15b/smoke
  python -I experiments/dream_compare/stage_b.py --stage b-run --model qwen25_15b --out RUN/qwen25_15b/gen \
         --bench longmemeval:oracle,longmemeval:s120
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


J = _load("dream_compare_judges", HERE / "judges.py")
PC = _load("dream_compare_prep_common", HERE / "benchmarks" / "prep_common.py")

TF_TOL = 0.01                # dense models; identity_checks uses models.tf_tol(C)
BENCHES = {"longmemeval": ("oracle", "s120"), "locomo": ("locomo10",), "personamem": ("32k",), "beam": ("100K",)}
SUBSET_SPLITS = {("longmemeval", "oracle"), ("locomo", "locomo10"), ("beam", "100K")}
LAT_SPLITS = {("longmemeval", "oracle"), ("locomo", "locomo10"), ("beam", "100K"), ("personamem", "32k")}
B_MAX = 16
JUDGE_MAX_NEW = {"lme": 10, "locomo_j": 200, "beam": 512}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def P():
    return _load("dream_compare_paradigms", HERE / "paradigms.py")


def safe(s):
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in s)


def subset_ids(b, n=None, seed=None):
    """The fixed sample subset of a split: random.Random(7).sample(sorted item ids, 100), sorted."""
    Pm = P()
    n, seed = n or Pm.SUBSET_N, seed if seed is not None else Pm.SUBSET_SEED
    ids = sorted(it.item_id for it in b.items)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def latency_ids(b, name, split):
    """The latency items: the first 20 of the subset (PersonaMem, no subset: the first 20 sorted ids)."""
    Pm = P()
    if (name, split) in SUBSET_SPLITS:
        return subset_ids(b)[:Pm.LAT_N]
    if (name, split) in LAT_SPLITS:
        return sorted(it.item_id for it in b.items)[:Pm.LAT_N]
    return []


def seed_of(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------------------------- the model


def model_setup(A):
    Pm = P()
    M = Pm.M
    if A.random_model:
        from transformers import AutoTokenizer
        C = M.tiny_context(A.model, AutoTokenizer.from_pretrained(M.SPECS[A.model]["hf"]), attn="sdpa")
    else:
        C = M.setup(A.model, A.device, attn="sdpa", gb_per_gpu=A.gb_per_gpu)
    log(f"{C.label}: layers {len(C.layout)}, sliding {C.sliding}, softcap {C.softcap}; {C.load_info}")
    return C


def p1_cap_tokens(C, A):
    """P1 runs where head + visible history + question <= this: the model's context (minus the answer budget) and
    the plain cache fitting --p1-kv-gb."""
    Pm = P()
    ctx = int(C.spec["max_context"]) - 512
    mem = int(A.p1_kv_gb * 2 ** 30 / Pm.bytes_per_position(C))
    return min(ctx, mem)


# ----------------------------------------------------------------------------------------------- identity checks


def identity_checks(C, store, mem, Q):
    """(1) P3 with the leading turns (<= 2,048 tokens) of the first segment restored at their own positions == them as text
    (teacher-forced logits at the question rows, max |dlogit| <= 0.01, and the same greedy tokens); (2) ragged
    rows == the adapter's gen_batch per row (greedy tokens), for P0 / P2-like / P3-like / P4-like rows."""
    import torch
    Pm = P()
    M = Pm.M
    res = {}
    seg0 = mem.segments[0]
    seg, used = [], 0                   # the segment's leading turns, <= 2,048 tokens (one-pass text reference)
    for i in seg0:
        if seg and used + mem.turns[i].n > 2048:
            break
        seg.append(i)
        used += mem.turns[i].n
    res["turns"] = len(seg)
    with torch.no_grad():
        Pf = Pm.p3_prefix(C, store, seg)
        ids_t, _, qs = Pm.layout_text(mem, seg)
        assert Pf.start == qs
        tq = torch.tensor(Q)
        h, _ = M.forward(C, tq, Pf)
        ht, _ = M.forward(C, torch.tensor(ids_t + Q))
        res["p3_all_vs_text"] = float((M.logits(C, h) - M.logits(C, ht[-len(Q):])).abs().max())
        a = Pm.gen1(C, tq, prefix=Pf, max_new=16)["ids"]
        b = Pm.gen1(C, ids_t + Q, max_new=16)["ids"]
        res["p3_all_vs_text_tokens_equal"] = a == b
        # ragged vs per-row
        first = seg[:min(2, len(seg))]
        P3 = Pm.p3_prefix(C, store, first)
        keep = Pm.p4_keep(mem, first, {i: list(range(0, mem.turns[i].n, 2)) for i in first})
        P4 = M.index_prefix(P3, keep)
        ids2, _, _ = Pm.layout_text(mem, first)
        rows = [dict(prefix=None, ids=list(mem.H) + Q), dict(prefix=None, ids=ids2 + Q), dict(prefix=P3, ids=Q),
                dict(prefix=P4, ids=Q)]
        rg = Pm.ragged_gen(C, rows, 12)
        ref = [Pm.gen1(C, r["ids"], prefix=r["prefix"], max_new=12)["ids"] for r in rows]
        res["ragged_vs_gen_batch_equal"] = [x["ids"] == y for x, y in zip(rg, ref)]
    res["ok"] = (res["p3_all_vs_text"] <= M.tf_tol(C) and res["p3_all_vs_text_tokens_equal"]
                 and all(res["ragged_vs_gen_batch_equal"]))
    return res


# ----------------------------------------------------------------------------------------------- per-context run


class SelCache:
    """P4 picks per turn (key: role + rendered content), appended to <dir>/selections.jsonl (resumable)."""

    def __init__(self, path):
        self.path = Path(path)
        self.d = {}
        if self.path.exists():
            for l in open(self.path, encoding="utf-8"):
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                self.d[r["key"]] = r

    def get_many(self, C, mem, idx, loo):
        """Records for the memory turns `idx` (computing the missing ones in batches); turns in `loo` need S-loo
        (a cached record without it is recomputed)."""
        Pm = P()
        keys = {i: Pm.turn_key(mem.turns[i]) for i in idx}
        miss = [i for i in idx if keys[i] not in self.d or (i in loo and self.d[keys[i]].get("loo") is None)]
        if miss:
            recs = Pm.select_turns(C, mem.H, [mem.turns[i] for i in miss], loo={j for j, i in enumerate(miss) if i in loo},
                                   log=log)
            with open(self.path, "a", encoding="utf-8") as f:
                for j, i in enumerate(miss):
                    r = dict(recs[j], key=keys[i], turn_id=mem.turns[i].turn_id)
                    self.d[keys[i]] = r
                    f.write(json.dumps(r) + "\n")
        return {i: self.d[keys[i]] for i in idx}, len(miss)


def row_len(r):
    return (r["S"], r["T"])


def batches(rows, C, A, max_new):
    """Rows sorted by length, cut into batches of <= B_MAX rows under the KV and mask budgets."""
    Pm = P()
    bpp = Pm.bytes_per_position(C)
    rows = sorted(rows, key=lambda r: (r["group"], r["S"] + r["T"], r["T"]))
    out, cur = [], []
    def fits(cur):
        S = max(r["S"] for r in cur)
        T = max(r["T"] for r in cur)
        kv = len(cur) * (S + T + max_new) * bpp
        ch = min(T, Pm.PREFILL_CHUNK)
        mask = len(cur) * ch * (S + T) * 4 * (3 if C.sliding else 2)
        attn = len(cur) * C.spec["n_q_heads"] * ch * (S + T) * 4      # SDPA's math path (fp32 with a mask on V100)
        return (kv <= A.batch_kv_gb * 2 ** 30 and mask <= A.mask_gb * 2 ** 30 and attn <= A.attn_gb * 2 ** 30)
    for r in rows:
        if cur and (len(cur) >= B_MAX or r["group"] != cur[0]["group"] or not fits(cur + [r])):
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def run_context(C, A, b, ctx_id, items, outdir, selc, subset, lat, p1cap, first_context, loo_items):
    import torch
    Pm = P()
    M = Pm.M
    name = b.name
    context = b.contexts[ctx_id]
    t0 = time.time()
    mem = Pm.build_memory(C, name, context)
    store = Pm.Store(C, mem, raw_gpu_gb=A.raw_gpu_gb)
    log(f"{ctx_id}: {len(mem.turns)} turns, {mem.N} tokens, {len(mem.segments)} segments, store "
        f"{store.nbytes() / 2 ** 30:.2f} GB, encoded in {store.encode_seconds:.0f}s; {len(items)} items")
    max_new = Pm.GEN_MAX_NEW[name]
    mc = name == "personamem"
    ms = Pm.m_values(name)
    info = {}
    for it in items:
        qtext = Pm.question_text(name, it, context)
        Q = Pm.question_ids(C, qtext)
        vis = Pm.visible_turns(mem, b, it)
        ret = {m: Pm.retrieve(b, it, mem, m) for m in ms}
        info[it.item_id] = SimpleNamespace(it=it, qtext=qtext, Q=Q, vis=vis, ret=ret)
    ident = None
    if first_context:
        ident = identity_checks(C, store, mem, info[items[0].item_id].Q)
        log(f"identity: {json.dumps(ident)}")
        assert ident["ok"], f"identity checks failed: {ident}"
    # P4 picks (query-free, per turn) for every retrieved turn
    need = sorted({i for x in info.values() for m in ms for i in x.ret[m]})
    loo_need = {i for x in info.values() if x.it.item_id in loo_items for m in ms for i in x.ret[m]}
    ts = time.time()
    picks, n_new = selc.get_many(C, mem, need, loo_need)
    sel_s = time.time() - ts
    if need:
        log(f"{ctx_id}: P4 picks for {len(need)} turns ({n_new} new) in {sel_s:.0f}s")
    # P1: the longest visible history that fits, prefilled once, just before the first P1 batch, and freed after the
    # last one (P1 rows form their own group of batches; 2026-11-07: Llama 1219646 ran out of memory in a P5 batch
    # while the shared history was still on the GPU)
    p1_need = [x for x in info.values() if mem.nh + sum(mem.turns[i].n for i in x.vis) + len(x.Q) <= p1cap]
    P1full, p1_s, ids_full = None, 0.0, None
    if p1_need:
        nvis = max(len(x.vis) for x in p1_need)
        ids_full = list(mem.H) + [y for i in range(nvis) for y in mem.turns[i].piece]
    # rows
    rows = []
    for x in info.values():
        it = x.it
        n_hist = sum(mem.turns[i].n for i in x.vis)
        opts = [M.DB().text_ids(C.tok, o).tolist() for o in it.options] if mc else [None]
        samples = (not mc) and it.item_id in subset
        for cond in Pm.conds_for(name):
            fam, var, m = Pm.parse_cond(cond)
            spec = dict(item=it.item_id, cond=cond, fam=fam, var=var, m=m)
            if fam == "P0":
                spec.update(S=0, T=mem.nh + len(x.Q), group=0)
            elif fam == "P1":
                if mem.nh + n_hist + len(x.Q) > p1cap:
                    x.__dict__.setdefault("skipped", {})["P1"] = "exceeds the P1 cap"
                    continue
                spec.update(S=mem.nh + n_hist, T=len(x.Q), group=1, n_vis=len(x.vis))
            elif fam == "P2":
                spec.update(S=0, T=mem.nh + sum(mem.turns[i].n for i in x.ret[m]) + len(x.Q), group=0)
            elif fam == "P3":
                spec.update(S=mem.nh + sum(mem.turns[i].n for i in x.ret[m]), T=len(x.Q), group=0)
            elif fam == "P4":
                if var == "loo" and it.item_id not in loo_items:
                    x.__dict__.setdefault("skipped", {})[cond] = "not in the S-loo subset"
                    continue
                spec.update(S=mem.nh + sum(len(picks[i][var]) for i in x.ret[m]), T=len(x.Q), group=0)
            elif fam == "P5":
                if var == "s":
                    budget = sum(len(picks[i]["dream"]) for i in x.ret[m])
                else:
                    budget = sum(mem.turns[i].n for i in x.ret[m])
                nb = Pm.MA.n_blocks_for(budget)
                spec.update(S=mem.nh + Pm.MA.BLOCK * nb, T=len(x.Q), group=2, n_blocks=nb, budget=budget)
            for oi, o in enumerate(opts):
                reps = [(True, None)] + ([(False, j) for j in range(Pm.N_SAMPLES)] if samples else [])
                for greedy, j in reps:
                    r = dict(spec, greedy=greedy, sample=j, opt=oi if mc else None)
                    if mc:
                        r["T"] += len(o)
                    rows.append(r)
    outs = {}
    tb = time.time()
    bl = batches(rows, C, A, 0 if mc else max_new)
    last_p1 = max((bk for bk, bt in enumerate(bl) if bt[0]["fam"] == "P1"), default=None)
    for bk, bt in enumerate(bl):
        tbk = time.time()
        if bt[0]["fam"] == "P1" and P1full is None:
            tp = time.time()
            P1full = Pm.prefill_text(C, ids_full)
            p1_s = time.time() - tp
        mat = [materialize(C, r, info[r["item"]], mem, store, picks, P1full, opts_of(C, info[r["item"]].it) if mc else None,
                           seed_of(r["item"], r["cond"], r["sample"])) for r in bt]
        if mc:
            sc = Pm.ragged_score(C, mat)
            for r, (s, n) in zip(bt, sc):
                outs[(r["item"], r["cond"], r["opt"], r["sample"])] = {"lp": s, "n": n}
        else:
            gg = Pm.ragged_gen(C, mat, max_new)
            for r, g in zip(bt, gg):
                outs[(r["item"], r["cond"], None, r["sample"])] = dict(g, text=C.tok.decode(g["ids"], skip_special_tokens=True).strip())
        del mat
        if bk == last_p1:
            P1full = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if first_context or bk % 50 == 0:
            log(f"  batch {bk + 1}/{len(bl)}: {len(bt)} rows ({bt[0]['cond']}...), S {max(r['S'] for r in bt)}, "
                f"T {max(r['T'] for r in bt)}: {time.time() - tbk:.1f}s; peak {M.peak_gb()}")
    gen_s = time.time() - tb
    # latency (B = 1): the shared P1 history is freed first (P1's latency prefills its own)
    P1full = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lat_res = {}
    for x in info.values():
        if x.it.item_id not in lat:
            continue
        lat_res[x.it.item_id] = latency(C, x, mem, store, picks, p1cap)
    # write
    for x in info.values():
        it = x.it
        rec = {"item": it.item_id, "bench": name, "split": b.split, "context": ctx_id, "category": it.category,
               "category_code": it.category_code, "abstention": it.abstention, "question": it.question,
               "question_text": x.qtext, "answers": it.answers, "n_rubric": len(it.rubric or []),
               "rubric": list(it.rubric or []),
               "retrieved": {str(m): [mem.turns[i].turn_id for i in x.ret[m]] for m in ms},
               "recall": {str(m): Pm.recall_of(b, it, mem, x.ret[m]) for m in ms},
               "n_hist_tokens": sum(mem.turns[i].n for i in x.vis), "n_question_tokens": len(x.Q), "nh": mem.nh,
               "skipped": getattr(x, "skipped", {}), "storage": Pm.storage(C, mem, upto=len(x.vis)),
               "conds": {}, "latency": lat_res.get(it.item_id)}
        for cond in Pm.conds_for(name):
            if cond in getattr(x, "skipped", {}):
                continue
            fam, var, m = Pm.parse_cond(cond)
            c = {"prefill_tokens": prefill_tokens(fam, mem, x, m, picks, var),
                 "restored_entries": restored(fam, mem, x, m, picks, var)}
            if mc:
                lps = [outs.get((it.item_id, cond, oi, None)) for oi in range(len(it.options))]
                if any(v is None for v in lps):
                    continue
                s = [v["lp"] for v in lps]
                n = [v["n"] for v in lps]
                ps, pm = J.mc_pick(s, n)
                c.update(option_logprob=s, option_ntok=n, pick_sum=ps, pick_mean=pm,
                         correct=ps == it.correct_option, correct_mean=pm == it.correct_option)
            else:
                g = outs.get((it.item_id, cond, None, None))
                if g is None:
                    continue
                c.update(text=g["text"], n_tok=g["n_tok"], capped=g["capped"], refusal=Pm.is_refusal(g["text"]))
                if name == "locomo":
                    c["f1"] = (J.locomo_f1(int(it.category_code), g["text"], it.answers[0]) if it.answers
                               else J.locomo_f1(5, g["text"], ""))
                if fam == "P5" and g.get("p5_blocks"):
                    pb = g["p5_blocks"]
                    c["p5_blocks_layer0"] = pb.get(0)
                    c["p5_blocks_last"] = pb.get(max(pb))
                sm = [outs.get((it.item_id, cond, None, j)) for j in range(Pm.N_SAMPLES)]
                if all(v is not None for v in sm) and it.item_id in subset:
                    c["samples"] = [v["text"] for v in sm]
                    c["sample_refusal"] = [Pm.is_refusal(v["text"]) for v in sm]
            rec["conds"][cond] = c
        with gzip.open(outdir / "items" / f"{safe(it.item_id)}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(rec, f)
    st = Pm.storage(C, mem)
    crec = {"context": ctx_id, "n_items": len(items), "n_turns": len(mem.turns), "n_tokens": mem.N,
            "n_segments": len(mem.segments), "encode_s": store.encode_seconds, "encoded_tokens": store.encoded_tokens,
            "select_s": sel_s, "n_select_new": n_new, "p1_prefill_s": p1_s, "gen_s": gen_s, "rows": len(rows),
            "total_s": time.time() - t0, "storage": st, "store_gb": store.nbytes() / 2 ** 30,
            "peak_gb": M.peak_gb(), "identity": ident}
    with open(outdir / "contexts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(crec) + "\n")
    del store, P1full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"{ctx_id}: done in {crec['total_s']:.0f}s (select {sel_s:.0f}, gen {gen_s:.0f}, {len(rows)} rows)")
    return crec


def opts_of(C, it):
    M = P().M
    return [M.DB().text_ids(C.tok, o).tolist() for o in it.options]


def materialize(C, r, x, mem, store, picks, P1full, opts, seed):
    """The ragged row of a spec: its prefix (built now, on the GPU), prompt ids, sampling seed, P5 spec."""
    Pm = P()
    M = Pm.M
    fam, var, m = r["fam"], r["var"], r["m"]
    Q = list(x.Q)
    prefix, ids, p5 = None, None, None
    if fam == "P0":
        ids = list(mem.H) + Q
    elif fam == "P1":
        n = mem.nh + sum(mem.turns[i].n for i in x.vis)
        prefix = Pm.prefix_head(P1full, n)
        ids = Q
    elif fam == "P2":
        ids = Pm.layout_text(mem, x.ret[m])[0] + Q
    elif fam == "P3":
        prefix, ids = Pm.p3_prefix(C, store, x.ret[m]), Q
    elif fam == "P4":
        P3 = Pm.p3_prefix(C, store, x.ret[m])
        prefix = M.index_prefix(P3, Pm.p4_keep(mem, x.ret[m], {i: picks[i][var] for i in x.ret[m]}))
        ids = Q
    elif fam == "P5":
        prefix = p5_placeholder(C, store, r["n_blocks"])
        cand = None
        if len(x.vis) < len(mem.turns):             # PersonaMem: blocks inside the visible turns only
            g_end = mem.turns[len(x.vis) - 1].g0 + mem.turns[len(x.vis) - 1].n
            cand = __import__("torch").tensor([g0 + n <= g_end for g0, n in store.blocks])
        p5 = dict(store=store, n_blocks=r["n_blocks"], nh=mem.nh, cand=cand)
        ids = Q
    q_len = len(Q)
    row = dict(prefix=prefix, ids=list(ids), greedy=r["greedy"], seed=seed, p5=p5, q_len=q_len if fam == "P5" else len(ids))
    if opts is not None:
        o = opts[r["opt"]]
        row["ids"] = row["ids"] + list(o)
        row["n_score"] = len(o)
        if fam == "P5":
            row["q_len"] = len(Q)
    return row


def p5_placeholder(C, store, n_blocks):
    """P5's prefix before retrieval: the stored head (positions 0..nh-1) + 16 x n_blocks empty slots at positions
    nh..; the question starts after them. The MemArt hook fills the slots per layer."""
    import torch
    Pm = P()
    M = Pm.M
    mem = store.mem
    nh = mem.nh
    W = Pm.MA.BLOCK * n_blocks
    kv = []
    for l, L in enumerate(C.layout):
        dev = Pm.layer_device(C, l)
        K = torch.zeros(1, L.n_kv, nh + W, L.head_dim, device=dev)
        V = torch.zeros(1, L.n_kv, nh + W, L.head_dim, device=dev)
        K[0, :, :nh] = store.K[l][:, :nh].to(dev)
        V[0, :, :nh] = store.V[l][:, :nh].to(dev)
        kv.append((K, V))
    return M.Prefix(kv, torch.arange(nh + W), nh + W)


def prefill_tokens(fam, mem, x, m, picks, var):
    """Tokens computed at question time."""
    nq = len(x.Q)
    if fam == "P0":
        return mem.nh + nq
    if fam == "P1":
        return mem.nh + sum(mem.turns[i].n for i in x.vis) + nq
    if fam == "P2":
        return mem.nh + sum(mem.turns[i].n for i in x.ret[m]) + nq
    return nq


def restored(fam, mem, x, m, picks, var):
    """Stored positions restored into the cache at question time (head included)."""
    if fam in ("P0", "P1", "P2"):
        return 0
    if fam == "P3":
        return mem.nh + sum(mem.turns[i].n for i in x.ret[m])
    if fam == "P4":
        return mem.nh + sum(len(picks[i][var]) for i in x.ret[m])
    MA = P().MA
    budget = (sum(len(picks[i]["dream"]) for i in x.ret[m]) if var == "s" else sum(mem.turns[i].n for i in x.ret[m]))
    return mem.nh + MA.BLOCK * MA.n_blocks_for(budget)


def latency(C, x, mem, store, picks, p1cap):
    """Per condition, alone (B = 1, greedy, LAT_DECODE new tokens, synchronised): the question-time work up to the
    first token (P1: the whole history's prefill; P2: the text; P3 / P4: compose from the CPU store + the question;
    P5: block scoring inside the prefill) and the decode time."""
    import torch
    Pm = P()
    M = Pm.M
    name = x.it.bench
    out = {}
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)
    for cond in Pm.conds_for(name):
        fam, var, m = Pm.parse_cond(cond)
        sync()
        t0 = time.perf_counter()
        if fam == "P1":
            n_hist = sum(mem.turns[i].n for i in x.vis)
            if mem.nh + n_hist + len(x.Q) > p1cap:
                continue
            Pf = Pm.prefill_text(C, list(mem.H) + [y for i in x.vis for y in mem.turns[i].piece])
            row = dict(prefix=Pf, ids=list(x.Q))
        else:
            if fam == "P4" and any(picks[i].get(var) is None for i in x.ret[m]):
                continue
            r = dict(fam=fam, var=var, m=m, greedy=True, n_blocks=None, opt=None)
            if fam == "P5":
                budget = (sum(len(picks[i]["dream"]) for i in x.ret[m]) if var == "s"
                          else sum(mem.turns[i].n for i in x.ret[m]))
                r["n_blocks"] = Pm.MA.n_blocks_for(budget)
            row = materialize(C, r, x, mem, store, picks, None, None, 0)
        build = time.perf_counter() - t0
        tm = {}
        Pm.ragged_gen(C, [row], Pm.LAT_DECODE, timing=tm)
        out[cond] = {"build_s": build, "first_token_s": build + tm["first_token_s"],
                     "decode_s_per_token": tm["decode_s"] / max(tm["decode_steps"], 1),
                     "p5_select_s": tm.get("p5_select_s")}
        del row
    return out


def run_bench(C, A, name, split, root, t_start):
    Pm = P()
    b = PC.load_split(name, split)
    outdir = Path(root) / f"{name}_{split}"
    (outdir / "items").mkdir(parents=True, exist_ok=True)
    subset = set(subset_ids(b)) if (name, split) in SUBSET_SPLITS else set()
    lat = set(latency_ids(b, name, split))
    items = list(b.items)
    if A.limit:
        order = sorted(subset) if subset else sorted(it.item_id for it in items)
        keep = set(order[:A.limit])
        items = [it for it in items if it.item_id in keep]
    if A.contexts:
        ctxs = list(dict.fromkeys(it.context_id for it in items))[:A.contexts]
        items = [it for it in items if it.context_id in set(ctxs)]
    by = defaultdict(list)
    for it in items:
        by[it.context_id].append(it)
    done = {p.name[:-len(".json.gz")] for p in (outdir / "items").glob("*.json.gz")}
    selc = SelCache(outdir / "selections.jsonl")
    p1cap = p1_cap_tokens(C, A)
    json.dump({"bench": name, "split": split, "model": A.model, "n_items": len(items), "n_contexts": len(by),
               "subset": sorted(subset), "latency_items": sorted(lat), "p1_cap_tokens": p1cap,
               "args": {k: v for k, v in vars(A).items() if isinstance(v, (str, int, float, bool, type(None)))},
               "env": Pm.M.DB().env_meta(), "started": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(outdir / f"run_{int(time.time())}.json", "w"), indent=1)
    first = not (outdir / "identity.json").exists()
    n_ctx = 0
    for cid, its in by.items():
        todo = [it for it in its if safe(it.item_id) not in done]
        if not todo:
            continue
        if A.max_hours and (time.time() - t_start) / 3600 > A.max_hours:
            log(f"--max-hours {A.max_hours} reached: stopping before {cid} (resubmit to continue)")
            return False
        # run every item of the context (shared memory; already-done items are rewritten identically)
        crec = run_context(C, A, b, cid, its, outdir, selc, subset, lat, p1cap, first, subset)
        if first:
            json.dump(crec["identity"], open(outdir / "identity.json", "w"), indent=1)
            first = False
        n_ctx += 1
    log(f"{name}/{split}: {n_ctx} contexts run; items written {len(list((outdir / 'items').glob('*.json.gz')))}")
    return True


def bench_list(A):
    out = []
    for bs in [x for x in A.bench.replace("+", ",").split(",") if x]:
        name, split = (bs.split(":") + [None])[:2]
        out.append((name, split or BENCHES[name][0]))
    return out


def all_done(A):
    """True when every item of every requested split already has its record (a continuation job then exits before
    loading the model). Without --limit / --contexts only."""
    if A.limit or A.contexts:
        return False
    for name, split in bench_list(A):
        b = PC.load_split(name, split)
        d = Path(A.out) / f"{name}_{split}" / "items"
        done = {p.name[:-len(".json.gz")] for p in d.glob("*.json.gz")} if d.is_dir() else set()
        if any(safe(it.item_id) not in done for it in b.items):
            return False
    return True


def stage_run(A):
    t0 = time.time()
    Pm = P()
    Pm.ATTN_GB = A.attn_gb
    Pm.BATCH_KV_GB, Pm.MASK_GB, Pm.MLP_GB = A.batch_kv_gb, A.mask_gb, A.mlp_gb
    if all_done(A):
        log(f"b-run: every item of {A.bench} is already written in {A.out}: nothing to do")
        return
    C = model_setup(A)
    complete = True
    for name, split in bench_list(A):
        complete &= run_bench(C, A, name, split, A.out, t0)
    log(f"b-run {'complete' if complete else 'STOPPED (time)'}: peak {Pm.M.peak_gb()}, {(time.time() - t0) / 60:.1f} min")


def stage_smoke(A):
    """b-run on one item per benchmark (the subset's first) into <out>, with the identity checks."""
    import torch
    A.limit = A.limit or 1
    A.bench = A.bench or "longmemeval:oracle+longmemeval:s120+locomo:locomo10+personamem:32k+beam:100K"
    t0 = time.time()
    stage_run(A)
    Pm = P()
    res = {"model": A.model, "minutes": (time.time() - t0) / 60, "peak_gb": Pm.M.peak_gb(), "benches": {}}
    for d in sorted(Path(A.out).glob("*_*")):
        if not (d / "items").is_dir():
            continue
        recs = [json.load(gzip.open(p, "rt")) for p in sorted((d / "items").glob("*.json.gz"))]
        ctx = [json.loads(l) for l in open(d / "contexts.jsonl")] if (d / "contexts.jsonl").exists() else []
        ident = json.load(open(d / "identity.json")) if (d / "identity.json").exists() else None
        res["benches"][d.name] = {"items": len(recs), "identity": ident, "contexts": ctx,
                                  "answers": {r["item"]: {c: v.get("text", v.get("pick_sum")) for c, v in r["conds"].items()}
                                              for r in recs}}
    res["ok"] = all(v["identity"] and v["identity"]["ok"] and v["items"] >= 1 for v in res["benches"].values())
    json.dump(res, open(Path(A.out) / "smoke_b.json", "w"), indent=1, default=str)
    log(f"Stage B smoke {'PASSED' if res['ok'] else 'FAILED'}: {res['minutes']:.1f} min, peak {res['peak_gb']}")
    assert res["ok"], "Stage B smoke failed (see smoke_b.json)"


# ----------------------------------------------------------------------------------------------- judge tasks


def read_items(gen_dir):
    out = {}
    for d in sorted(Path(gen_dir).glob("*_*")):
        if (d / "items").is_dir():
            out[d.name] = [json.load(gzip.open(p, "rt")) for p in sorted((d / "items").glob("*.json.gz"))]
    return out


def tasks_for_item(rec, model):
    """Judge prompts of one item record's greedy answers (keys model|bench_split|item|cond|rubric index)."""
    out = []
    bs = f"{rec['bench']}_{rec['split']}"
    for cond, c in rec["conds"].items():
        if "text" not in c:
            continue
        resp = c["text"]
        base = f"{model}|{bs}|{rec['item']}|{cond}"
        if rec["bench"] == "longmemeval":
            out.append(("lme", J.lme_prompt(rec["category"], rec["question"], rec["answers"][0], resp, rec["abstention"]),
                        f"{base}|0"))
        elif rec["bench"] == "locomo" and not rec["abstention"]:
            out.append(("locomo_j", J.j_prompt(rec["question"], rec["answers"][0], resp), f"{base}|0"))
        elif rec["bench"] == "beam":
            for i, rb in enumerate(rec.get("rubric") or []):
                out.append(("beam", J.beam_prompt(rb, resp), f"{base}|{i}"))
    return out


def stage_tasks(A):
    """<out>/tasks.jsonl from the gen dir (--from): every greedy answer's judge prompts, merged by prompt hash."""
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    by = {}
    n_items = Counter()
    rubrics = {}
    for bs, recs in read_items(A.from_dir).items():
        if bs.startswith("beam"):
            b = PC.load_split("beam", bs.split("_", 1)[1])
            rubrics = {it.item_id: it.rubric for it in b.items}
        for rec in recs:
            if rec["bench"] == "beam":
                rec["rubric"] = rubrics.get(rec["item"], [])
            n_items[bs] += 1
            for kind, prompt, key in tasks_for_item(rec, A.model):
                h = J.phash(prompt)
                if h in by:
                    by[h]["keys"].append(key)
                else:
                    by[h] = {"h": h, "kind": kind, "prompt": prompt, "keys": [key], "bench": rec["bench"]}
    tasks = sorted(by.values(), key=lambda t: t["h"])
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    json.dump({"items": dict(n_items), "tasks": len(tasks), "by_kind": dict(Counter(t["kind"] for t in tasks)),
               "keys": sum(len(t["keys"]) for t in tasks)}, open(out / "tasks_meta.json", "w"), indent=1)
    log(f"b-tasks: {len(tasks)} unique prompts {dict(Counter(t['kind'] for t in tasks))} from {dict(n_items)}")


def stage_judge_local(A):
    """The original judge (dream_select's / run.py's set-up: Qwen3.5-9B bf16 via core_v1's loader, the prompt
    rendered by dream_bind's chat_ids, greedy, max new tokens per kind), batched with LEFT padding through HF
    generate (run.py batches only equal-length prompts, too slow for 10^4 long prompts). Checked first: the first
    CHECK_N prompts batched == run.py's unpadded generate_batch (reported; > 1 mismatch -> unpadded for all).
    Resumable (appends to <out>/orig.jsonl)."""
    import torch
    import yaml
    out = Path(A.out)
    tasks = [json.loads(l) for l in open(out / "tasks.jsonl", encoding="utf-8") if l.strip()]
    if A.limit:
        tasks = tasks[:A.limit]
    done = set()
    op = out / "orig.jsonl"
    if op.exists():
        for l in open(op, encoding="utf-8"):
            try:
                done.add(json.loads(l)["h"])
            except (ValueError, KeyError):
                pass
    todo = [t for t in tasks if t["h"] not in done]
    log(f"orig judge: {len(tasks)} tasks, {len(done)} done, {len(todo)} to do")
    if not todo:
        return
    D = _load("dream_compare_ds_run_jb", REPO / "experiments" / "dream_select" / "run.py")
    cvr = _load("dream_compare_core_v1_run_b", REPO / "experiments" / "core_v1" / "run.py")
    cfg = yaml.safe_load(open(D.DB_CFG))
    Aj = SimpleNamespace(random_model="qwen2" if A.random_model else None, device=A.device)
    t0 = time.time()
    jm, jt = cvr.load({"judge": cfg["judge"], "model": "Qwen/Qwen3.5-2B"}, Aj, judge=True)
    body, lm_head = D.DB.text_model(jm), jm.lm_head
    eos = {jt.eos_token_id} | {jt.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    eos_list = sorted(e for e in eos if isinstance(e, int) and e != jt.unk_token_id)
    pad = jt.pad_token_id if jt.pad_token_id is not None else eos_list[0]
    dev = next(jm.parameters()).device
    enc = {t["h"]: D.DB.chat_ids(jt, t["prompt"]).tolist() for t in todo}

    def unpadded(t):
        o = cvr.core.generate_batch(jm, body, lm_head, torch.tensor(enc[t["h"]]), [(True, None)],
                                    max_new=JUDGE_MAX_NEW[t["kind"]], eos=eos_list, vocab=lm_head.out_features)[0]
        return o["ids"], o["capped"]

    def padded(ts):
        L = max(len(enc[t["h"]]) for t in ts)
        x = torch.full((len(ts), L), pad, dtype=torch.long)
        am = torch.zeros(len(ts), L, dtype=torch.long)
        for i, t in enumerate(ts):
            e = enc[t["h"]]
            x[i, L - len(e):] = torch.tensor(e)
            am[i, L - len(e):] = 1
        mx = max(JUDGE_MAX_NEW[t["kind"]] for t in ts)
        with torch.no_grad():
            g = jm.generate(input_ids=x.to(dev), attention_mask=am.to(dev), max_new_tokens=mx, do_sample=False,
                            eos_token_id=eos_list, pad_token_id=pad, temperature=None, top_p=None, top_k=None)
        res = []
        for i, t in enumerate(ts):
            row = g[i, L:].tolist()[:JUDGE_MAX_NEW[t["kind"]]]
            k = next((j for j, tk in enumerate(row) if tk in eos_list), None)
            res.append((row if k is None else row[:k], k is None))
        return res

    check = todo[:A.check_n]
    use_pad = True
    with torch.no_grad():
        ref = [unpadded(t) for t in check]
        got = padded(check) if check else []
    mism = sum(a[0] != b[0] for a, b in zip(ref, got))
    log(f"orig judge: batched (left-padded) vs unpadded on {len(check)} prompts: {mism} token mismatches")
    if mism > 1:
        use_pad = False
        log("orig judge: falling back to unpadded generation for every prompt")
    order = sorted(todo, key=lambda t: (t["kind"], len(enc[t["h"]])))
    n = 0
    with open(op, "a", encoding="utf-8") as f:
        i = 0
        while i < len(order):
            kind = order[i]["kind"]
            bs = A.judge_batch if kind == "lme" else max(1, A.judge_batch // (2 if kind == "locomo_j" else 4))
            chunk = [t for t in order[i:i + bs] if t["kind"] == kind] or order[i:i + 1]
            i += len(chunk)
            outs = padded(chunk) if use_pad else [unpadded(t) for t in chunk]
            for t, (ids, capped) in zip(chunk, outs):
                raw = jt.decode(ids, skip_special_tokens=True).strip()
                f.write(json.dumps({"h": t["h"], "judge": "orig", "kind": kind, "keys": t["keys"], "raw": raw[:600],
                                    **J.PARSERS[kind](raw), "capped": capped}) + "\n")
                n += 1
            f.flush()
            if n % 2000 < len(chunk):
                log(f"orig judge: {n}/{len(order)} in {(time.time() - t0) / 60:.1f} min")
    json.dump({"n": n, "minutes": (time.time() - t0) / 60, "padded": use_pad, "check_mismatch": mism,
               "check_n": len(check)}, open(out / "orig.jsonl.stats.json", "w"))
    log(f"orig judge: {n} prompts in {(time.time() - t0) / 60:.1f} min")


def stage_analyze(A):
    AB = _load("dream_compare_analyze_b", HERE / "analyze_b.py")
    AB.analyze(A.from_dir, A.judge_dir, Path(A.out), model=A.model)


def stage_summary(A):
    AB = _load("dream_compare_analyze_b", HERE / "analyze_b.py")
    AB.summary(Path(A.out))


STAGES = {"b-run": stage_run, "b-smoke": stage_smoke, "b-tasks": stage_tasks, "b-judge-local": stage_judge_local,
          "b-analyze": stage_analyze, "b-summary": stage_summary}


def main(argv=None):
    import torch
    argv = list(sys.argv[1:] if argv is None else argv)
    st = argv[argv.index("--stage") + 1] if "--stage" in argv else None
    if st not in STAGES:                          # run.py's stages (judge-summary, ...)
        run = _load("dream_compare_run_b", HERE / "run.py")
        return run.main(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--model", default="qwen25_15b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bench", default=None, help="name[:split] list, separated by '+' or ','")
    ap.add_argument("--limit", type=int, default=None, help="items per split (the subset's first)")
    ap.add_argument("--contexts", type=int, default=None, help="at most this many contexts per split")
    ap.add_argument("--random-model", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="alias: --limit 1")
    ap.add_argument("--gb-per-gpu", type=int, default=66)
    ap.add_argument("--p1-kv-gb", type=float, default=16.0)
    ap.add_argument("--batch-kv-gb", type=float, default=6.0)
    ap.add_argument("--mask-gb", type=float, default=2.0)
    ap.add_argument("--raw-gpu-gb", type=float, default=4.0)
    ap.add_argument("--attn-gb", type=float, default=4.0, help="batch budget for one layer's attention matrix")
    ap.add_argument("--mlp-gb", type=float, default=4.0, help="one prefill chunk's MLP activations (sets its width)")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--from", dest="from_dir", default=None)
    ap.add_argument("--judge-dir", default=None)
    ap.add_argument("--judge-batch", type=int, default=32)
    ap.add_argument("--check-n", type=int, default=8)
    A = ap.parse_args(argv)
    if A.tiny and not A.limit:
        A.limit = 1
    STAGES[A.stage](A)


if __name__ == "__main__":
    main()
