#!/usr/bin/env python
"""dream_compare Stage A: selectors (PREREG.md, "Stage A: selectors"). The torch side: harvest, selection passes,
teacher-forced measures, generation, judge tasks. Scorers and selectors: baselines.py (numpy); analysis: analyze_a.py.
Everything model-specific goes through the Stage 0 adapter (models.py: Prefix, index_prefix, harvest_kv, forward,
gen_batch with position masks on Gemma; headmask.py for the per-head conditions), so it runs on 2-GPU models.

Stages (run.py dispatches every "--stage a-*" here; jobs.py runs several in one process):
  a-tf          per item (resumable; --part i --nparts n splits the items): harvest (one eager pass over [head][turn];
                its attention gives S-attn, H2O, VATP; its attention inputs give Expected Attention), the dreamer (or
                dream_select addendum 1's saved dreams, --reuse), the selection passes (S-recon, KVzip, the dreamed
                questions: S-dream + SnapKV-dream + S-dream-VATP, S-loo, the oracle question: SnapKV-oracle), the 18
                selections x 4 budgets, the checks (kept entries = k x E; R-full == T-full; per-head keep-all == R-full;
                the same picks in every head == the splice) and the teacher-forced measures -> <out>/tf/items/<qid>.json.gz
  a-gen         per item: 1 greedy + 4 samples for the anchors and every selection at k4 / k8 -> <out>/gen/items/
  a-judge-tasks the LongMemEval prompts of every answer (its question type's template; abstention items: the abstention
                template) -> <out>/judge/tasks.jsonl (atomic; every judge job writes it, identically)
  a-analyze     analyze_a.analyze(<out>) -> report_a.md, verdict_a.json (CPU)
  a-summary     analyze_a.summary(<run root>) -> summary_a.md (CPU)

  python -I experiments/dream_compare/run.py --model qwen25_15b --stage a-tf --out RUN/qwen25_15b --tiny \
         --reuse /path/dream_select_cluster_20261104_0129
"""

import argparse
import gzip
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("dream_compare_models_a", HERE / "models.py")
HM = _load("dream_compare_headmask_a", HERE / "headmask.py")
B = _load("dream_compare_baselines_a", HERE / "baselines.py")
J = _load("dream_compare_judges_a", HERE / "judges.py")
PC = _load("dream_compare_prep_common_a", HERE / "benchmarks" / "prep_common.py")
LME = _load("dream_compare_lme_a", HERE / "benchmarks" / "longmemeval.py")
DSR = M.DS()                 # dream_select's run.py (its dreamer prompt, selectors_ds, constants)
DB = M.DB()
S = DSR.S                    # dream_select's selectors_ds (topk, content_positions, gold, ...)

TF_TOL = 0.01                # dense models; the asserts use models.tf_tol(C) (Gemma 4: PREREG "Stage 0 results")
N_ITEMS = 133
LOO_MAX_CONTENT = DSR.LOO_MAX_CONTENT          # 200
LOO_BATCH = DSR.LOO_BATCH                      # 16
OWN_MAX_NEW = DSR.OWN_MAX_NEW                  # 24
DREAM_MAX_NEW = DSR.DREAM_MAX_NEW              # 96
RANDOM_SEEDS = (0, 1, 2)
ANCHORS_TF = ("nomem", "ctx", "T-full", "R-full", "R-content")
ANCHORS_GEN = ("nomem", "ctx", "R-full", "R-content")
GEN_TAG = "dream_select"     # dream_select addendum 1's random streams (the same draws in every condition)
GEN_SAMPLES = 4
GEN_MAX_NEW = 64
REUSE_SCORES = ("S-attn", "S-dream", "S-recon", "S-loo")
TF_MAJOR = int(__import__("transformers").__version__.split(".")[0])
NEED_OUTPUT_ATTN = TF_MAJOR < 5      # transformers 4.x eager returns the weights only with output_attentions


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------------------------- items


def load_items(tiny=False):
    """The 133 LongMemEval oracle items whose labelled evidence holds exactly one user turn (file order), with the
    stored turn (that user turn's text), the primary-set flag (not abstention, not single-session-preference) and the
    single-session-user flag (dream_select's 64; turn, question and answer asserted equal to ssu70.json). tiny: the
    first 2 single-session-user items (reused) and the first 2 other primary items (fresh)."""
    b = PC.load_split("longmemeval", "oracle")
    ids = LME.single_turn_ids(b, user_only=True)
    by = {it.item_id: it for it in b.items}
    ssu = {x["question_id"]: x for x in json.load(open(DSR.DATA))["items"] if x["usable"]}
    out = []
    for n, i in enumerate(ids):
        it = by[i]
        turns = {t.turn_id: t for t in b.contexts[it.context_id].turns()}
        ev = [e for e in it.evidence_turns if turns[e].role == "user"]
        assert len(ev) == 1, (i, ev)
        turn = turns[ev[0]].text
        ans = str(it.answers[0])
        verb = (not it.abstention) and bool(ans.strip()) and ans.strip().lower() in turn.lower()
        d = {"question_id": i, "question": it.question, "answer": ans, "category": it.category,
             "abstention": bool(it.abstention), "stored_turn": turn, "evidence_turn": ev[0], "verbatim": verb,
             "primary": (not it.abstention) and it.category != "single-session-preference", "ssu": i in ssu,
             "index": n}
        if i in ssu:
            s = ssu[i]
            assert (turn, it.question, ans, verb) == (s["stored_turn"], s["question"], s["answer"], s["verbatim"]), i
        out.append(d)
    assert len(out) == N_ITEMS and sum(x["ssu"] for x in out) == 64, (len(out), sum(x["ssu"] for x in out))
    if tiny:
        out = [x for x in out if x["ssu"]][:2] + [x for x in out if x["primary"] and not x["ssu"]][:2]
    return out


def part_of(items, part, nparts):
    return [x for x in items if x["index"] % nparts == part] if nparts > 1 else items


def load_reuse(root, model):
    """dream_select addendum 1's tf meta of this model ({qid: item info}) or {}."""
    if not root:
        return {}
    p = Path(root) / model / "tf" / "meta.json"
    if not p.exists():
        return {}
    m = json.load(open(p))
    assert m.get("model", "qwen25_15b") == model, (p, m.get("model"))
    return m["items"]


# ----------------------------------------------------------------------------------------------- attention taps


class Tap:
    """Forward hooks on every attention module: its softmax attention w [B, Hq, Tq, Tk] is handed to fn(layer, w)
    while the pass runs (transformers 5 eager returns the weights without output_attentions, so nothing is kept unless
    fn keeps it); inputs=True also keeps each attention module's input hidden states [T, d] (Expected Attention)."""

    def __init__(self, C, fn=None, inputs=False):
        self.C, self.fn, self.want_inputs = C, fn, inputs
        self.inputs, self.seen, self.handles = {}, set(), []

    def __enter__(self):
        for l, blk in enumerate(self.C.blocks):
            a = blk.self_attn
            if self.fn is not None:
                def post(mod, args, out, l=l):
                    w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                    assert w is not None, f"layer {l}: no attention weights (eager attention needed)"
                    self.seen.add(l)
                    self.fn(l, w.detach())
                self.handles.append(a.register_forward_hook(post))
            if self.want_inputs:
                def pre(mod, args, kwargs, l=l):
                    hs = kwargs.get("hidden_states", args[0] if args else None)
                    self.inputs[l] = hs[0].detach()
                self.handles.append(a.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def tapped_forward(C, ids, prefix, fn, inputs=False):
    """M.forward with a Tap; asserts every layer reported its attention."""
    with Tap(C, fn, inputs) as tap:
        h, _ = M.forward(C, ids, prefix, attn=NEED_OUTPUT_ATTN)
    assert tap.seen == set(range(len(C.blocks))), "not every layer reported its attention"
    return h, tap


def _np(w):
    return w.detach().float().cpu().numpy()


def n_kv_list(C):
    return [l.n_kv for l in C.layout]


# ----------------------------------------------------------------------------------------------- the store


def harvest(C, turn, stats=True):
    """The stored memory of one user turn (built from the turn alone): pieces, content positions, the Prefix of
    [head][turn] and, with stats, the harvest pass's statistics: A (head- and layer-mean attention [T, T], S-attn),
    h2o {layer: [n_kv, T]} (accumulated attention), the value norms and the attention inputs (Expected Attention)."""
    P = DB.turn_pieces(C.tok, turn)
    assert P.H == C.head, "head differs from the template head"
    nh, ns = len(P.H), len(P.Seg)
    content = S.content_positions(C.tok, P.Seg, turn)
    assert content[0] > 0 and content[-1] < ns - 1, "content touches the turn's template tokens"
    ids = torch.tensor(P.H + P.Seg, dtype=torch.long)
    st = SimpleNamespace(turn=turn, H=P.H, Seg=P.Seg, nh=nh, ns=ns, n=nh + ns, content=content, dstats=None)
    if not stats:
        st.P, _, _ = M.harvest_kv(C, ids)
        return st
    T = nh + ns
    acc = {"A": np.zeros((T, T)), "h2o": {}}

    def fn(l, w):
        wn = _np(w[0])
        acc["A"] += wn.mean(0)
        acc["h2o"][l] = B.h2o_layer(wn, C.layout[l].n_kv)
    with Tap(C, fn, inputs=True) as tap:
        st.P, _, _ = M.harvest_kv(C, ids, attn=NEED_OUTPUT_ATTN)
    assert tap.seen == set(range(len(C.blocks)))
    st.A = acc["A"] / len(C.blocks)
    st.h2o = acc["h2o"]
    st.inputs = tap.inputs
    st.vn1 = {l: B.value_norms(_np(V[0]), 1) for l, (K, V) in enumerate(st.P.kv)}
    st.vn2 = {l: B.value_norms(_np(V[0]), 2) for l, (K, V) in enumerate(st.P.kv)}
    return st


def splice(store, sel):
    """The shared slice: head + the selected content positions at their stored positions (dream_select's prefix_of,
    through the adapter: multi-device safe)."""
    return M.index_prefix(store.P, S.keep_indices(store.nh, sel))


def keep_of(C, store, picks):
    """HeadMask keep for per-head picks {layer: [[content positions] per KV head]}: the head + each head's picks."""
    pk = {int(l): [[store.nh + int(c) for c in x] for x in hs] for l, hs in picks.items()}
    return HM.keep_per_head(C, store.n, pk, always=range(store.nh))


def picks_from_json(p):
    return {int(l): [list(map(int, x)) for x in hs] for l, hs in p.items()}


# ----------------------------------------------------------------------------------------------- selectors
# Every selector takes (C, store) or (C, store, dreams); the store is built from the turn alone. The one exception is
# sel_snapkv_oracle (ORACLE: the real benchmark question, a ceiling that cheats).


def sel_attn(C, store):
    """S-attn (dream_select): mean attention received from later content tokens in the harvest pass."""
    return S.attn_received(store.A, [store.nh + c for c in store.content])


def dream(C, store):
    """dream_select's dreamer (its prompt, few-shot examples, greedy, strict retry), through the adapter."""
    out = {"attempts": []}
    for p in (DSR.DREAM_PROMPT, DSR.DREAM_PROMPT_STRICT):
        ids = torch.tensor(DB.render_ids(C.tok, DSR.dream_messages(p, store.turn))[1], dtype=torch.long)
        g = gen(C, ids, [(True, None)], max_new=DREAM_MAX_NEW)[0]
        text = C.tok.decode(g["ids"], skip_special_tokens=True)
        qs, strict, lines = S.parse_dreams(text, DSR.N_DREAM)
        out["attempts"].append({"raw": text, "questions": qs, "strict_ok": strict, "lines": lines})
        if len(qs) == DSR.N_DREAM:
            break
    best = max(out["attempts"], key=lambda a: len(a["questions"]))
    qs = best["questions"]
    out["fallback"] = len(qs) < DSR.N_DREAM
    if not qs:
        qs = best["lines"][:DSR.N_DREAM]
    assert qs, "the dreamer produced no usable line"
    out["questions"] = qs
    out["retried"] = len(out["attempts"]) > 1
    return out


def _question_pass(C, store, q):
    """A question turn asked after the whole stored turn: per layer the SnapKV vote [n_kv, n_stored] and S-dream-VATP's
    row mean [n_kv, n_stored], plus S-dream's head- and layer-mean attention of the question's content rows and last
    prompt row to the content positions."""
    ids = DB.qt_ids(C, q)
    qc = S.content_positions(C.tok, ids, q)
    rows = qc + [len(ids) - 1]
    cols = [store.nh + c for c in store.content]
    acc = {"M": np.zeros((len(ids), store.n + len(ids))), "snap": {}, "rm": {}}

    def fn(l, w):
        wn = _np(w[0])
        n_kv = C.layout[l].n_kv
        acc["M"] += wn.mean(0)
        acc["snap"][l] = B.snapkv_layer(wn, n_kv, store.n)
        acc["rm"][l] = B.rows_mean_layer(wn, rows, n_kv, store.n)
    tapped_forward(C, ids, store.P, fn)
    Mm = acc["M"] / len(C.blocks)
    return {"q": q, "rows": rows, "sdream": S.query_mean(Mm, rows, cols), "snap": acc["snap"], "rm": acc["rm"]}


def dream_stats(C, store, dreams):
    """The dreamed questions' passes (one per question; computed once per store)."""
    if store.dstats is None:
        store.dstats = [_question_pass(C, store, q) for q in dreams["questions"]]
    return store.dstats


def sel_dream(C, store, dreams):
    """S-dream (dream_select): per dreamed question, the head- and layer-mean attention from its content tokens and
    the last prompt token to each content position; summed over the questions."""
    return S.dream_combine([d["sdream"] for d in dream_stats(C, store, dreams)])


def sel_snapkv_dream(C, store, dreams):
    """SnapKV-dream: SnapKV's pooled votes with each dreamed question as the window, summed over the questions."""
    ds = dream_stats(C, store, dreams)
    return {l: sum(d["snap"][l] for d in ds) for l in ds[0]["snap"]}


def sel_sdream_vatp(C, store, dreams):
    """S-dream-VATP (exploratory): per dreamed question, per (layer, KV head) the mean attention of S-dream's rows x
    the value's l1 norm, averaged over all (layer, KV head); summed over the questions. -> [n_stored]."""
    ds = dream_stats(C, store, dreams)
    tot = np.zeros(store.n)
    for d in ds:
        tot += B.shared_score({l: d["rm"][l] * store.vn1[l] for l in d["rm"]})
    return tot


def sel_snapkv_oracle(C, store, oracle_question):
    """ORACLE (flagged): SnapKV with the real benchmark question as the window (a ceiling that cheats)."""
    return _question_pass(C, store, oracle_question)["snap"]


def sel_recon(C, store):
    """S-recon (dream_select): after the whole stored turn, 'Please repeat exactly what I just told you.' + the
    assistant header + the turn's text teacher-forced; mean attention each content position receives from the
    reconstruction rows (heads and layers averaged)."""
    q = DB.qt_ids(C, DSR.RECON_PROMPT)
    rec = DB.text_ids(C.tok, store.turn)
    ids = torch.cat([q, rec])
    acc = {"M": np.zeros((len(ids), store.n + len(ids)))}

    def fn(l, w):
        acc["M"] += _np(w[0]).mean(0)
    tapped_forward(C, ids, store.P, fn)
    Mm = acc["M"] / len(C.blocks)
    return S.query_mean(Mm, list(range(len(q), len(q) + len(rec))), [store.nh + c for c in store.content])


def sel_kvzip(C, store):
    """KVzip (reimplemented _get_score): after the whole stored turn, a new user turn 'Repeat the previous context
    exactly.' + the assistant header + the turn's text teacher-forced; per stored entry the max attention over its GQA
    group and every input row. -> {layer: [n_kv, n_stored]}."""
    q = DB.qt_ids(C, B.KVZIP_PROMPT)
    rec = DB.text_ids(C.tok, store.turn)
    out = {}

    def fn(l, w):
        out[l] = B.kvzip_layer(_np(w[0]), C.layout[l].n_kv, store.n)
    tapped_forward(C, torch.cat([q, rec]), store.P, fn)
    return out


def sel_h2o(C, store):
    """H2O: the harvest pass's accumulated attention per (layer, KV head)."""
    return store.h2o


def sel_vatp(C, store):
    """VATP: H2O's score x the value's l1 norm."""
    return {l: B.vatp_layer(store.h2o[l], store.vn1[l]) for l in store.h2o}


def avg_rope(C, l, start, n=B.EA_N_FUTURE, device=None):
    """Expected Attention's averaged RoPE matrix of layer l over positions start .. start + n - 1, from the layer's own
    rotary embedding (partial rotation included): RoPE is linear in (cos, sin), so the mean matrix = the map with the
    mean cos / sin; rows = RoPE(e_i) (row-vector convention q' = q @ A). -> [d, d] numpy."""
    d = C.layout[l].head_dim
    dev = device or C.blocks[l].self_attn.q_proj.weight.device
    x = torch.zeros(1, 1, n, d, device=dev)
    pos = torch.arange(start, start + n, device=dev)[None]
    cos, sin = M._rotary_call(C, x, pos, l)
    cb = cos.float().mean(1, keepdim=True).expand(1, d, d)
    sb = sin.float().mean(1, keepdim=True).expand(1, d, d)
    E = torch.eye(d, device=dev)[None, None]
    return _np(M._apply_rope(C, l, E, cb.to(dev), sb.to(dev))[0, 0])


def prerope_queries(C, l, h):
    """Pre-RoPE queries of layer l from its attention input h [T, d_model]: q_proj (+ q_norm where the model has one,
    Gemma 4) -> [Hq, T, d]."""
    a = C.blocks[l].self_attn
    d = C.layout[l].head_dim
    q = a.q_proj(h.to(a.q_proj.weight.device)).view(h.shape[0], -1, d)
    qn = getattr(a, "q_norm", None)
    if qn is not None:
        q = qn(q)
    return q.transpose(0, 1)


def attn_scaling(C, l):
    a = C.blocks[l].self_attn
    s = getattr(a, "scaling", None)
    return float(s) if s is not None else C.layout[l].head_dim ** -0.5


def sel_ea(C, store):
    """Expected Attention (KVPress): query statistics of the harvest pass (first EA_N_SINK rows dropped), the RoPE
    rotation averaged over the next 512 positions, the second-order expected score with the layer's own scaling,
    softmax over the turn's keys, group mean, x the value's l2 norm. -> {layer: [n_kv, n_stored]}."""
    out, rope = {}, {}
    cols = range(store.nh, store.n)
    for l in range(len(C.blocks)):
        q = prerope_queries(C, l, store.inputs[l][B.EA_N_SINK:])
        mu, cov = B.query_stats(_np(q))
        kind = C.layout[l].kind
        if kind not in rope:
            rope[kind] = avg_rope(C, l, store.n)
        K, V = store.P.kv[l]
        out[l] = B.ea_layer(mu, cov, rope[kind], _np(K[0]), _np(V[0]), attn_scaling(C, l), cols)
    return out


def _seq_lp(C, h, Lq, own):
    """[B] summed log P(own | prompt) rows of h [B, T, d] (multi-device safe: the ids go to the logits' device)."""
    rows = torch.arange(Lq - 1, Lq - 1 + len(own), device=h.device)
    lp = torch.log_softmax(C.lm_head(h[:, rows]).float(), -1)
    return lp.gather(-1, own.to(lp.device)[None, :, None].expand(h.shape[0], -1, 1))[..., 0].sum(-1)


def drop_forward(C, P, drops, ids):
    """One batched pass of ids after the whole stored Prefix P minus one slot per row (`drops`): -> hidden [B, T, d].
    Sliding models get per-row position masks."""
    T = P.n
    keep = torch.tensor([[i for i in range(T) if i != j] for j in drops], dtype=torch.long)      # [B, T-1]
    kv = []
    for K, V in P.kv:
        ii = keep.to(K.device)
        kv.append((K[0][:, ii].permute(1, 0, 2, 3).contiguous(), V[0][:, ii].permute(1, 0, 2, 3).contiguous()))
    Bn = len(drops)
    cache = M.make_cache(kv, Bn)
    q_pos = torch.arange(P.start, P.start + len(ids))
    kw = {}
    if M.needs_pos_masks(C):
        ms = [M.pos_masks(C, torch.cat([P.pos[keep[b]], q_pos]), q_pos) for b in range(Bn)]
        kw["attention_mask"] = {k: torch.cat([m[k] for m in ms], 0) for k in ms[0]}
    dev = M.input_device(C)
    o = C.body(input_ids=ids[None].expand(Bn, -1).to(dev), past_key_values=cache,
               position_ids=q_pos[None].expand(Bn, -1).to(dev), use_cache=True, **kw)
    return o.last_hidden_state


def sel_loo(C, store, dreams):
    """S-loo (dream_select): per dreamed question, the model's own greedy short answer with the whole slice; a content
    position's score = the drop in log P(own answer | dreamed question) when it alone is removed, summed over the
    questions. -> (scores, own answers)."""
    total = np.zeros(len(store.content))
    owns = []
    for q in dreams["questions"]:
        qi = DB.qt_ids(C, q)
        g = gen(C, qi, [(True, None)], prefix=store.P, max_new=OWN_MAX_NEW)[0]
        own = torch.tensor(g["ids"], dtype=torch.long)
        owns.append({"q": q, "answer": C.tok.decode(g["ids"], skip_special_tokens=True), "n_tok": len(g["ids"])})
        if len(own) == 0:
            continue
        ids = torch.cat([qi, own])
        Lq = len(qi)
        h, _ = M.forward(C, ids, store.P)
        base = float(_seq_lp(C, h[None], Lq, own)[0])
        drops = []
        for b in range(0, len(store.content), LOO_BATCH):
            js = [store.nh + c for c in store.content[b:b + LOO_BATCH]]
            hb = drop_forward(C, store.P, js, ids)
            drops += (base - _seq_lp(C, hb, Lq, own)).tolist()
            del hb
        total += np.array(drops)
    return total.tolist(), owns


SELECTOR_FNS = {"S-attn": sel_attn, "S-dream": sel_dream, "S-recon": sel_recon, "S-loo": sel_loo, "dreamer": dream,
                "KVzip": sel_kvzip, "SnapKV-dream": sel_snapkv_dream, "S-dream-VATP": sel_sdream_vatp,
                "H2O": sel_h2o, "VATP": sel_vatp, "EA": sel_ea, "SnapKV-oracle": sel_snapkv_oracle}
ORACLE_FNS = {"SnapKV-oracle"}


# ----------------------------------------------------------------------------------------------- generation helper


def gen(C, ids, rows, *, prefix=None, max_new, streams=None, keep=None, n_stored=None):
    """M.gen_batch (multi-device models take the adapter's masked loop, which keeps the logits on the input device;
    it equals dream_bind's loop token for token), optionally under a per-head mask."""
    fm = bool(getattr(C, "multi_device", False))
    with (HM.HeadMask(C, keep, n_stored) if keep is not None else nullcontext()):
        return M.gen_batch(C, ids, rows, prefix=prefix, max_new=max_new, streams=streams, force_masks=fm)


# ----------------------------------------------------------------------------------------------- measures


def prompt_ids(C, store, cond, question):
    tok = C.tok
    if cond == "nomem":
        return DB.chat_ids(tok, question)
    if cond == "ctx":
        return DB.chat_ids(tok, f"{store.turn} {question}")
    if cond == "T-full":
        _, q = DB.render_ids(tok, [{"role": "user", "content": store.turn}, {"role": "user", "content": question}])
        return torch.tensor(q, dtype=torch.long)
    return DB.qt_ids(C, question)


def measure(C, store, cond, question, ans, prefix=None, keep=None):
    """dream_select's measure through the adapter: log P(gold answer | question) (sum, per-token mean) with the
    answer starting the reply, the first-token refusal mass, the top-5 first tokens; + the scored rows' logits."""
    q = prompt_ids(C, store, cond, question)
    if cond not in ("nomem", "ctx", "T-full"):
        assert prefix is not None
    ids = torch.cat([q, ans])
    with (HM.HeadMask(C, keep, store.n) if keep is not None else nullcontext()):
        h, _ = M.forward(C, ids, prefix)
    Lq = len(q)
    lg = M.logits(C, h[Lq - 1:len(ids) - 1])
    lp = torch.log_softmax(lg, -1)
    s, m = S.answer_logprob(lp, 1, ans)
    first = lp[0]
    p = {k: math.exp(float(first[v])) for k, v in C.ref_ids.items()}
    top = first.topk(5)
    return {"answer_score": s, "answer_mean": m, "refusal": p["I"] + p["Sorry"] + p["As"], "p_first": p,
            "top5_first": [[C.tok.decode([int(i)]), math.exp(float(v))] for v, i in zip(top.values, top.indices)]}, lg


# ----------------------------------------------------------------------------------------------- stage a-tf


def selections(C, store, it, sc, ph, sdv, budgets, span):
    """{(condition, budget): selection}: shared = [content positions] (random: one per seed); per-head = picks."""
    content, nh = store.content, store.nh
    nkv = n_kv_list(C)
    sels = {}
    for b in budgets:
        k = B.budget_k(b, len(content))
        E = B.budget_entries(k, nkv)
        for name in ("S-dream", "S-attn", "S-recon", "S-loo"):
            if sc.get(name) is not None:
                sels[(name, b)] = S.topk(sc[name], content, k)
        sels[("random", b)] = [S.random_k(content, k, (s, it["index"], k)) for s in RANDOM_SEEDS]
        if span:
            sels[("gold@k", b)] = S.gold_at_k(span, content, k)
        sels[("KVzip-g", b)] = B.global_topk(ph["KVzip"], content, nh, E)
        sels[("KVzip-ph", b)] = B.per_head_topk(ph["KVzip"], content, nh, k)
        sels[("KVzip-sh", b)] = B.shared_topk(ph["KVzip"], content, nh, k)
        sels[("SnapKV-dream-ph", b)] = B.per_head_topk(ph["SnapKV-dream"], content, nh, k)
        sels[("SnapKV-dream-sh", b)] = B.shared_topk(ph["SnapKV-dream"], content, nh, k)
        sels[("SnapKV-oracle-ph", b)] = B.per_head_topk(ph["SnapKV-oracle"], content, nh, k)
        sels[("H2O-ph", b)] = B.per_head_recent_heavy(ph["H2O"], content, nh, k)
        sels[("H2O-sh", b)] = B.shared_recent_heavy(ph["H2O"], content, nh, k)
        sels[("VATP-ph", b)] = B.per_head_recent_heavy(ph["VATP"], content, nh, k)
        sels[("EA-ph", b)] = B.per_head_topk(ph["EA"], content, nh, k)
        sels[("StreamingLLM", b)] = B.streaming(content, k)
        sels[("S-dream-VATP", b)] = B.topk_content(sdv, content, nh, k)
    if span:
        sels[("gold", "own")] = S.gold_plus_next(span, content)
    # checks: shared slices keep the head + exactly k content entries; per-head picks hold exactly k x E entries
    for (c, b), sel in sels.items():
        if c in B.PER_HEAD:
            k = B.budget_k(b, len(content))
            B.check_picks(sel, content, B.budget_entries(k, nkv), nkv)
        else:
            k = len(S.gold_plus_next(span, content)) if c == "gold" else B.budget_k(b, len(content))
            for s in (sel if c == "random" else [sel]):
                S.check_kept(S.keep_indices(nh, s), nh, k, content)
    return sels


def tf_item(C, it, budgets, reuse):
    tok = C.tok
    t0 = time.time()
    turn = it["stored_turn"]
    store = harvest(C, turn)
    content, n_c = store.content, len(store.content)
    tim = {"harvest": time.time() - t0}
    if reuse:
        assert (store.nh, store.ns, store.content) == (reuse["n_head"], reuse["n_seg"], reuse["content"]), \
            f"{it['question_id']}: head / turn / content positions differ from dream_select's"
        assert reuse["turn"] == turn
    # --- selection: nothing below sees the question or the answer, except the oracle (flagged) and the gold reference
    t = time.time()
    dr = reuse["dreams"] if reuse else dream(C, store)
    tim["dreamer"] = 0.0 if reuse else time.time() - t
    fresh = {}
    for name, fn in (("S-attn", lambda: sel_attn(C, store)), ("S-dream", lambda: sel_dream(C, store, dr)),
                     ("S-recon", lambda: sel_recon(C, store))):
        t = time.time()
        fresh[name] = fn()
        tim[name] = time.time() - t
    loo_ok = n_c <= LOO_MAX_CONTENT
    if reuse:
        owns = reuse["own_answers"]
        sc = {k: reuse["scores"].get(k) for k in REUSE_SCORES}
        tim["S-loo"] = 0.0
    else:
        t = time.time()
        sc_loo, owns = sel_loo(C, store, dr) if loo_ok else (None, None)
        tim["S-loo"] = time.time() - t
        sc = dict(fresh, **{"S-loo": sc_loo})
    ph = {}
    for name, fn in (("KVzip", lambda: sel_kvzip(C, store)), ("SnapKV-dream", lambda: sel_snapkv_dream(C, store, dr)),
                     ("H2O", lambda: sel_h2o(C, store)), ("VATP", lambda: sel_vatp(C, store)),
                     ("EA", lambda: sel_ea(C, store))):
        t = time.time()
        ph[name] = fn()
        tim[name] = time.time() - t
    t = time.time()
    sdv = sel_sdream_vatp(C, store, dr)
    tim["S-dream-VATP"] = time.time() - t
    store.inputs = None
    t = time.time()
    ph["SnapKV-oracle"] = sel_snapkv_oracle(C, store, it["question"])           # ORACLE
    tim["SnapKV-oracle"] = time.time() - t
    # --- gold reference (uses the label)
    span = S.gold_span(tok, store.Seg, content, it["answer"]) if it["verbatim"] else None
    if it["verbatim"]:
        assert span, f"{it['question_id']}: verbatim item but no gold span"
    sels = selections(C, store, it, sc, ph, sdv, budgets, span)
    # --- reuse check (information only): the recomputed S-attn / S-dream / S-recon vs dream_select's saved scores
    rcheck = None
    if reuse:
        rcheck = {}
        for name in ("S-attn", "S-dream", "S-recon"):
            a, b_ = np.array(fresh[name], dtype=float), np.array(reuse["scores"][name], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b_)
            rcheck[name] = {"max_abs": float(np.abs(a[ok] - b_[ok]).max()) if ok.any() else None,
                            "rel": float(np.abs(a[ok] - b_[ok]).max() / max(np.abs(b_[ok]).max(), 1e-12)) if ok.any() else None,
                            "same_top4": S.topk(fresh[name], content, B.budget_k("k4", n_c)) ==
                            S.topk(reuse["scores"][name], content, B.budget_k("k4", n_c))}
    # --- measures
    ans = DB.text_ids(tok, it["answer"])
    res = {}
    for c in ANCHORS_TF:
        pf = store.P if c == "R-full" else (splice(store, content) if c == "R-content" else None)
        res[c] = measure(C, store, c, it["question"], ans, pf)
    d_id = DB.maxdiff(res["R-full"][1], res["T-full"][1])
    assert d_id <= M.tf_tol(C), f"{it['question_id']}: R-full TF != T-full ({d_id})"
    m_all, lg_all = measure(C, store, "slice", it["question"], ans, store.P, HM.keep_all(C, store.n))
    d_keep = DB.maxdiff(lg_all, res["R-full"][1])
    assert d_keep <= M.tf_tol(C), f"{it['question_id']}: per-head keep-all != R-full ({d_keep})"
    ref_sel = sels[("StreamingLLM", budgets[0])]
    same = {l: [list(ref_sel)] * C.layout[l].n_kv for l in range(len(C.blocks))}
    _, lg_m = measure(C, store, "slice", it["question"], ans, store.P, keep_of(C, store, same))
    _, lg_s = measure(C, store, "slice", it["question"], ans, splice(store, ref_sel))
    d_same = DB.maxdiff(lg_m, lg_s)
    assert d_same <= M.tf_tol(C), f"{it['question_id']}: the same picks in every head != the splice ({d_same})"
    rows = [{"item": it["question_id"], "cond": c, "budget": None, "k": None, "entries": None, **res[c][0]}
            for c in ANCHORS_TF]
    nkv = n_kv_list(C)
    for (c, b), sel in sels.items():
        if c in B.PER_HEAD:
            ms = [measure(C, store, "slice", it["question"], ans, store.P, keep_of(C, store, sel))[0]]
            ent = B.picks_entries(sel)
            hit = (any(set(x) & set(span) for hs in sel.values() for x in hs) if span else None)
            extra = {"span_share": B.span_share(sel, span)}
        else:
            L = sel if c == "random" else [sel]
            ms = [measure(C, store, "slice", it["question"], ans, splice(store, s))[0] for s in L]
            ent = len(L[0]) * B.entries_per_position(nkv)
            hit = float(np.mean([S.hit(s, span) for s in L])) if span else None
            extra = {"per_seed_answer_score": [m["answer_score"] for m in ms]} if c == "random" else {}
        k = None if b == "own" else B.budget_k(b, n_c)
        r = {"item": it["question_id"], "cond": c, "budget": b, "k": k, "entries": ent,
             "answer_score": float(np.mean([m["answer_score"] for m in ms])),
             "answer_mean": float(np.mean([m["answer_mean"] for m in ms])),
             "refusal": float(np.mean([m["refusal"] for m in ms])), "hit": hit, **extra}
        if len(ms) == 1:
            r["top5_first"] = ms[0]["top5_first"]
        rows.append(r)
    js = lambda v: v if not isinstance(v, np.ndarray) else v.tolist()
    shared_scores = {k: js(v) for k, v in sc.items()}
    shared_scores["S-dream-VATP"] = [float(sdv[store.nh + c]) for c in content]
    info = {"question_id": it["question_id"], "question": it["question"], "answer": it["answer"],
            "category": it["category"], "abstention": it["abstention"], "primary": it["primary"], "ssu": it["ssu"],
            "verbatim": it["verbatim"], "turn": turn, "n_head": store.nh, "n_seg": store.ns, "n_content": n_c,
            "content": content, "gold_span": span, "dreams": dr, "reused": bool(reuse), "own_answers": owns,
            "loo_run": sc.get("S-loo") is not None, "scores": shared_scores, "reuse_check": rcheck,
            "identity_rfull_tfull": d_id, "identity_keep_all": d_keep, "identity_same_picks_vs_splice": d_same,
            "n_answer_tokens": len(ans), "entries_per_position": B.entries_per_position(nkv),
            "seconds": time.time() - t0, "timing": tim}
    sel_json = {f"{c}|{b}": sel for (c, b), sel in sels.items()}
    del store
    return {"info": info, "rows": rows, "sels": sel_json}


def write_json_gz(path, obj):
    tmp = Path(str(path) + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    tmp.replace(path)


def read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def setup(A):
    if A.random_model:
        from transformers import AutoTokenizer
        C = M.tiny_context(A.model, AutoTokenizer.from_pretrained(M.SPECS[A.model]["hf"]), attn="eager")
    else:
        C = M.setup(A.model, A.device, attn="eager", gb_per_gpu=A.gb_per_gpu)
    C.multi_device = len((C.load_info or {}).get("device_map_devices") or []) > 1
    log(f"{C.label}: layers {len(C.layout)}, E = {B.entries_per_position(n_kv_list(C))} entries per position, "
        f"sliding {C.sliding}, multi-device {C.multi_device}; {C.load_info}")
    return C


def fit_check(A, what):
    pk = M.peak_gb()
    log(f"{what}: peak GPU {pk}")
    if A.max_reserved_gb is not None and pk:
        worst = max(v[1] for v in pk.values())
        assert worst <= A.max_reserved_gb, f"fit check: peak reserved {worst} GB > {A.max_reserved_gb} GB ({what})"


def stage_tf(A):
    out = Path(A.out) / "tf"
    (out / "items").mkdir(parents=True, exist_ok=True)
    items = part_of(load_items(A.tiny), A.part, A.nparts)
    reuse = load_reuse(A.reuse, A.model)
    C = setup(A)
    budgets = ["k4"] if A.tiny else list(B.BUDGETS)
    t0 = time.time()
    n_new = 0
    for i, it in enumerate(items):
        f = out / "items" / f"{it['question_id']}.json.gz"
        if f.exists():
            log(f"tf: {it['question_id']} already done, skipped")
            continue
        ru = reuse.get(it["question_id"]) if it["ssu"] else None
        with torch.no_grad():
            r = tf_item(C, it, budgets, ru)
        write_json_gz(f, r)
        n_new += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inf = r["info"]
        el = (time.time() - t0) / 60
        log(f"tf {i + 1}/{len(items)} {it['question_id']} ({it['category']}{', reused' if ru else ''}): n_content "
            f"{inf['n_content']}, dreams {inf['dreams']['questions']}, identity {inf['identity_rfull_tfull']:.1e} / "
            f"{inf['identity_keep_all']:.1e} / {inf['identity_same_picks_vs_splice']:.1e} | {inf['seconds']:.1f}s, "
            f"{el:.1f} min, eta {el / (i + 1) * (len(items) - i - 1):.1f} min")
    meta = {"model": A.model, "label": C.label, "hf": C.spec["hf"], "budgets": budgets, "tiny": A.tiny,
            "part": A.part, "nparts": A.nparts, "n_items_part": len(items), "new": n_new, "reuse": A.reuse,
            "n_reuse_items": len(reuse), "minutes": (time.time() - t0) / 60, "load": C.load_info, "paris": C.paris,
            "env": DB.env_meta(), "peak_gb": M.peak_gb(), "random_model": bool(A.random_model),
            "entries_per_position": B.entries_per_position(n_kv_list(C)), "layout_n_kv": n_kv_list(C),
            "prompts": {"kvzip": B.KVZIP_PROMPT, "recon": DSR.RECON_PROMPT, "dream": DSR.DREAM_PROMPT}}
    json.dump(meta, open(out / f"meta_part{A.part}.json", "w"), indent=1, default=str)
    log(f"tf done: {n_new} new items of {len(items)} in {meta['minutes']:.1f} min")
    fit_check(A, "tf")


# ----------------------------------------------------------------------------------------------- stage a-gen


def gen_conditions(it, sels, budgets):
    """[(name, cond, budget, selection)] of one item: the anchors, gold (verbatim), then per budget every selection."""
    out = [(c, c, None, None) for c in ANCHORS_GEN]
    if "gold|own" in sels:
        out.append(("gold", "gold", "own", sels["gold|own"]))
    for b in budgets:
        for c in B.SELECTIONS:
            key = f"{c}|{b}"
            if key in sels:
                s = sels[key]
                out.append((f"{c}@{b}", c, b, s[0] if c == "random" else s))
    return out


def gen_item(C, it, rec, budgets):
    tok = C.tok
    t0 = time.time()
    info, sels = rec["info"], rec["sels"]
    q = it["question"]
    store = harvest(C, it["stored_turn"], stats=False)
    assert (store.nh, store.ns, store.content) == (info["n_head"], info["n_seg"], info["content"]), \
        f"{it['question_id']}: head / turn / content positions differ from the tf run's"
    qt = DB.qt_ids(C, q)
    # identity at the last prompt row: R-full == T-full; per-head keep-all == R-full
    h, _ = M.forward(C, qt, store.P)
    h2, _ = M.forward(C, prompt_ids(C, store, "T-full", q))
    d_id = DB.maxdiff(M.logits(C, h[-1:]), M.logits(C, h2[-1:]))
    assert d_id <= M.tf_tol(C), f"{it['question_id']}: R-full != T-full at the last prompt row ({d_id})"
    with HM.HeadMask(C, HM.keep_all(C, store.n), store.n):
        h3, _ = M.forward(C, qt, store.P)
    d_keep = DB.maxdiff(M.logits(C, h3[-1:]), M.logits(C, h[-1:]))
    assert d_keep <= M.tf_tol(C), f"{it['question_id']}: keep-all != R-full ({d_keep})"
    conds = gen_conditions(it, sels, budgets)
    streams = {1: (DB.cv1.seed_of(GEN_TAG, q, 0), GEN_SAMPLES + 1)}
    lay = [(True, None)] + [(False, (1, j)) for j in range(1, GEN_SAMPLES + 1)]
    loop = C.cfg["gen"]["loop"]
    nkv = n_kv_list(C)
    rows = []
    for name, c, b, sel in conds:
        keep = None
        if c in ANCHORS_GEN:
            ids = prompt_ids(C, store, c, q)
            pf = {"nomem": None, "ctx": None, "R-full": store.P, "R-content": splice(store, store.content)}[c]
        elif c in B.PER_HEAD:
            picks = picks_from_json(sel)
            B.check_picks(picks, store.content, B.budget_entries(B.budget_k(b, len(store.content)), nkv), nkv)
            ids, pf, keep = qt, store.P, keep_of(C, store, picks)
        else:
            k = len(sel) if b == "own" else B.budget_k(b, len(store.content))
            S.check_kept(S.keep_indices(store.nh, sel), store.nh, k, store.content)
            ids, pf = qt, splice(store, sel)
        outs = gen(C, ids, lay, prefix=pf, max_new=GEN_MAX_NEW, streams=streams, keep=keep, n_stored=store.n)
        for j, o in enumerate(outs):
            rep = DB.cv1.rep_rate(o["ids"], loop["n"])
            rows.append({"sec": "gen", "key": f"{it['question_id']}|{name}|{j}", "item": it["question_id"],
                         "cname": name, "cond": c, "budget": b, "j": j, "greedy": j == 0, "ids": o["ids"],
                         "answer": tok.decode(o["ids"], skip_special_tokens=True).strip(), "n_tok": o["n_tok"],
                         "capped": o["capped"], "rep": round(rep, 4), "loop": rep >= loop["thr"]})
    inf = {"sec": "info", "item": it["question_id"], "identity_rfull_tfull": d_id, "identity_keep_all": d_keep,
           "n_conds": len(conds), "conds": [x[0] for x in conds], "seconds": time.time() - t0}
    del store
    return rows, inf


def stage_gen(A):
    d = Path(A.out)
    gdir = d / "gen"
    (gdir / "items").mkdir(parents=True, exist_ok=True)
    items = part_of(load_items(A.tiny), A.part, A.nparts)
    C = setup(A)
    t0 = time.time()
    n_new = 0
    budgets = None
    for i, it in enumerate(items):
        f = gdir / "items" / f"{it['question_id']}.jsonl.gz"
        if f.exists():
            log(f"gen: {it['question_id']} already done, skipped")
            continue
        tfp = d / "tf" / "items" / f"{it['question_id']}.json.gz"
        assert tfp.exists(), f"no tf record for {it['question_id']} ({tfp})"
        rec = read_json_gz(tfp)
        tf_budgets = sorted({r["budget"] for r in rec["rows"] if r["budget"] not in (None, "own")})
        budgets = [b for b in B.GEN_BUDGETS if b in tf_budgets]
        with torch.no_grad():
            rows, inf = gen_item(C, it, rec, budgets)
        DB.write_jsonl_gz(f, rows + [inf])
        n_new += len(rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        el = (time.time() - t0) / 60
        log(f"gen {i + 1}/{len(items)} {it['question_id']}: {inf['n_conds']} conditions, {len(rows)} answers | "
            f"{inf['seconds']:.1f}s, {el:.1f} min, eta {el / (i + 1) * (len(items) - i - 1):.1f} min")
    meta = {"model": A.model, "label": C.label, "budgets": budgets, "n_items_part": len(items), "answers_new": n_new,
            "part": A.part, "nparts": A.nparts, "tiny": A.tiny, "samples": GEN_SAMPLES, "max_new": GEN_MAX_NEW,
            "seed_tag": GEN_TAG, "sampler": C.cfg["gen"]["sampler"], "minutes": (time.time() - t0) / 60,
            "env": DB.env_meta(), "peak_gb": M.peak_gb(), "random_model": bool(A.random_model)}
    json.dump(meta, open(gdir / f"gen_meta_part{A.part}.json", "w"), indent=1, default=str)
    log(f"gen done: {n_new} new answers in {meta['minutes']:.1f} min")
    fit_check(A, "gen")


# ----------------------------------------------------------------------------------------------- stage a-judge-tasks


def read_gen(d):
    G, I = [], {}
    for f in sorted((Path(d) / "gen" / "items").glob("*.jsonl.gz")):
        for r in DB.read_jsonl_gz(f):
            (I.__setitem__(r["item"], r) if r.get("sec") == "info" else G.append(r))
    return G, I


def judge_tasks(d, items=None):
    """Unique LongMemEval prompts of a model dir's answers: [{"h", "kind": "lme", "prompt", "keys", "item"}] by hash."""
    items = items or {x["question_id"]: x for x in load_items()}
    G, _ = read_gen(d)
    by = {}
    for r in G:
        it = items[r["item"]]
        p = J.lme_prompt(it["category"], it["question"], it["answer"], r["answer"], it["abstention"])
        h = J.phash(p)
        by.setdefault(h, {"h": h, "kind": "lme", "prompt": p, "keys": [], "item": r["item"], "bench": "longmemeval"})
        by[h]["keys"].append(r["key"])
    return sorted(by.values(), key=lambda t: t["h"])


def stage_judge_tasks(A):
    d = Path(A.out)
    jd = d / "judge"
    jd.mkdir(parents=True, exist_ok=True)
    tasks = judge_tasks(d)
    tmp = jd / f"tasks.jsonl.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    tmp.replace(jd / "tasks.jsonl")
    log(f"judge tasks: {len(tasks)} unique prompts for {sum(len(t['keys']) for t in tasks)} answers -> "
        f"{jd / 'tasks.jsonl'}")


# ----------------------------------------------------------------------------------------------- main


STAGES = ("a-tf", "a-gen", "a-judge-tasks", "a-analyze", "a-summary")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--model", default="qwen25_15b", choices=sorted(M.SPECS))
    ap.add_argument("--out", required=True, help="the model dir (a-summary: the run root)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help="4 items (2 reused + 2 fresh), k4 only (smoke)")
    ap.add_argument("--random-model", action="store_true", help="a tiny random model with the real tokenizer (CPU)")
    ap.add_argument("--reuse", default=None, help="dream_select addendum 1's run root (<root>/<model>/tf/meta.json)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--gb-per-gpu", type=int, default=66)
    ap.add_argument("--max-reserved-gb", type=float, default=None)
    ap.add_argument("--models", default=None, help="a-summary: comma-separated model keys (default all five)")
    A = ap.parse_args(argv)
    if A.stage == "a-tf":
        stage_tf(A)
    elif A.stage == "a-gen":
        stage_gen(A)
    elif A.stage == "a-judge-tasks":
        stage_judge_tasks(A)
    elif A.stage == "a-analyze":
        _load("dream_compare_analyze_a", HERE / "analyze_a.py").analyze(Path(A.out), A.model)
    elif A.stage == "a-summary":
        AA = _load("dream_compare_analyze_a", HERE / "analyze_a.py")
        AA.summary(Path(A.out), A.models.split(",") if A.models else None)


if __name__ == "__main__":
    main()
