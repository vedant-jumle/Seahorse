#!/usr/bin/env python
"""dream_select (PREREG.md): can the positions of a stored K,V slice be chosen automatically, on natural text?

Qwen2.5-1.5B-Instruct, fp32, eager attention, local (RTX 4060). Items: the 64 usable LongMemEval single-session-user
items (data/longmemeval/ssu70.json, prep_data.py). The memory is the labelled evidence user turn; this tests positions
only. Teacher-forced measures only (the dreamer and S-loo's own answers are greedy generations used for SELECTION, never
scored). Reuses dream_bind's model loading, chat-turn pieces and splice code by import (nothing there is modified).

Per item:
  harvest   one pass over [head][evidence user turn] (eager attention): K,V at every layer, the content positions
            (the turn's text tokens, not the template / role tokens) and S-attn's attention statistics
  selectors S-attn, S-dream (+ S-dream-split, exploratory), S-recon, S-loo, last-k, random (3 seeds); the gold
            references (gold, gold@k) use the label. A selector sees the stored turn and its own dreamed questions
            only: every selector function takes (C, store[, dreams]) and the store is built from the turn alone
  measure   for each slice (head + the k picks at their stored positions; the question turn starts at n_head + n_seg)
            and each anchor (nomem, ctx, T-full, R-full, R-content): log P(gold answer | benchmark question) with the
            answer starting the assistant reply, and the first-token refusal mass P(I) + P(Sorry) + P(As)

  python -I experiments/dream_select/run.py --out results/dream_select_<stamp> [--tiny]
  python -I experiments/dream_select/run.py --out <same dir> --analyze-only

Checks that fail the run: R-full TF == the turn as a prior user turn (T-full; logits within 0.01); every slice keeps the
head + exactly k entries; every selected index is a content position; Paris.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# select.py would shadow the stdlib `select` module: never keep this folder on sys.path (python -I does not add it)
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(REPO / "src"))

import argparse  # noqa: E402
import gzip  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("dream_select_select", HERE / "select.py")
DB = _load("dream_select_db_run", REPO / "experiments" / "dream_bind" / "run.py")   # dream_bind's run.py (import only)
bind = DB.bind
DB_CFG = REPO / "experiments" / "dream_bind" / "config.yaml"
DATA = REPO / "data" / "longmemeval" / "ssu70.json"
MODEL_KEY = "qwen25_15b"
TF_TOL = 0.01
N_DREAM = 3
DREAM_MAX_NEW = 96
OWN_MAX_NEW = 24
LOO_MAX_CONTENT = 200
LOO_BATCH = 16
RANDOM_SEEDS = (0, 1, 2)
SPLIT_EARLY, SPLIT_LATE = range(0, 14), range(20, 28)
N_EXAMPLES = 8          # the first 8 usable items in file order (pre-registered)

# The dreamer prompt: an instruction plus two made-up worked examples as earlier chat exchanges (few-shot). The
# examples are invented (not from LongMemEval). Fixed after format checks of the dreamed text on the first 10 turns
# (PREREG.md, "Dreamer"); no score was used to choose it.
DREAM_PROMPT = (
    "A user told an AI assistant:\n\n\"{turn}\"\n\nWrite 3 short memory-quiz questions that the USER could ask the "
    "assistant later about facts from the user's own life in this message. The user is speaking, so each question "
    "uses \"I\" or \"my\" (like \"What is my dog's name?\"), never \"you\" or \"your\". One question per line, nothing "
    "else.")
DREAM_STRICT_SUFFIX = ("\n\nOutput exactly 3 lines. Each line must be one short question ending with a question mark. "
                       "No numbering, no introduction, no other text.")
DREAM_EXAMPLES = (
    ("I just adopted a beagle named Max, and I'm moving to Denver next month for my new nursing job. Any tips for "
     "flying with a dog?",
     "What is my dog's name?\nWhere am I moving next month?\nWhat is my new job?"),
    ("Can you suggest a good recipe for a vegetarian lasagna? By the way, I started a pottery class at the community "
     "center last week and made my first bowl.",
     "What class did I start last week?\nWhere do I take my pottery class?\nWhat did I make in my first pottery class?"),
)
DREAM_PROMPT_STRICT = DREAM_PROMPT + DREAM_STRICT_SUFFIX
RECON_PROMPT = "Please repeat exactly what I just told you."
SELECTORS = ("S-attn", "S-dream", "S-dream-split", "S-recon", "S-loo", "last-k", "random")
ANCHORS = ("nomem", "ctx", "T-full", "R-full", "R-content")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------------------------- forward helpers


def fwd(C, ids, prefix=None, attn=False):
    """Teacher-forced pass after an optional stored prefix ([(K, V)] per layer, start position): (hidden [T, d],
    attentions or None)."""
    start = prefix[1] if prefix else 0
    cache = bind.make_cache(prefix[0] if prefix else [], 1)
    pos = torch.arange(start, start + len(ids))[None].to(C.dev)
    o = C.body(input_ids=ids[None].to(C.dev), past_key_values=cache, position_ids=pos, use_cache=True,
               output_attentions=attn)
    return o.last_hidden_state[0], (o.attentions if attn else None)


def head_mean(att):
    """Eager attentions (tuple over layers of [1, heads, Tq, Tk]) -> [layers, Tq, Tk] numpy, mean over heads."""
    return torch.stack([a[0].float().mean(0) for a in att]).cpu().numpy()


def logprobs_rows(C, h, rows):
    lg = C.lm_head(h[rows]).float()
    if not bool(torch.isfinite(lg).all()):
        raise FloatingPointError("non-finite logits")
    return lg, torch.log_softmax(lg, -1)


# ----------------------------------------------------------------------------------------------- the store


def harvest(C, turn):
    """The stored memory of one user turn (built from the turn alone: no question, no answer): token pieces, content
    positions (indices into the turn's tokens), K,V at every layer, and the harvest pass's head-and-layer-mean
    attention (S-attn's input)."""
    P = DB.turn_pieces(C.tok, turn)
    assert P.H == C.head, "head differs from the template head"
    nh, ns = len(P.H), len(P.Seg)
    content = S.content_positions(C.tok, P.Seg, turn)
    assert content[0] > 0 and content[-1] < ns - 1, "content touches the turn's template tokens"
    o = C.body(input_ids=torch.tensor(P.H + P.Seg)[None].to(C.dev), use_cache=True, output_attentions=True)
    kv = [(K.clone(), V.clone()) for K, V in bind.kv_list(o.past_key_values)]
    A = head_mean(o.attentions).mean(0)                     # [T, T], mean over heads then layers
    del o
    return SimpleNamespace(turn=turn, H=P.H, Seg=P.Seg, nh=nh, ns=ns, content=content, kv=kv, A=A)


def prefix_of(store, sel=None, keep=None):
    """([(K, V)] per layer, question start) of a slice: the head + the selected turn positions (sel = indices into the
    turn); sel None = the whole turn (R-full). `keep` overrides with explicit harvest-sequence indices."""
    if keep is None:
        keep = list(range(store.nh + store.ns)) if sel is None else S.keep_indices(store.nh, sel)
    ki = torch.tensor(keep, device=store.kv[0][0].device)
    return [(K[:, :, ki], V[:, :, ki]) for K, V in store.kv], store.nh + store.ns


# ----------------------------------------------------------------------------------------------- selectors
# Every selector takes (C, store) or (C, store, dreams) and returns one score per content position (higher = keep).


def sel_attn(C, store):
    """S-attn: mean attention received from later content tokens in the harvest pass (heads and layers averaged)."""
    return S.attn_received(store.A, [store.nh + c for c in store.content])


def dream_messages(prompt, turn):
    """The dreamer's chat: the two worked examples as earlier exchanges, then the instruction on this turn."""
    msgs = []
    for t, a in DREAM_EXAMPLES:
        msgs += [{"role": "user", "content": prompt.format(turn=t)}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": prompt.format(turn=turn)}]


def dream(C, store):
    """The dreamer (the same model, greedy, no memory): 3 short questions about the turn's personal facts. Retry once
    with the stricter prompt; on failure fall back to what was parsed (recorded)."""
    out = {"attempts": []}
    for p in (DREAM_PROMPT, DREAM_PROMPT_STRICT):
        ids = torch.tensor(DB.render_ids(C.tok, dream_messages(p, store.turn))[1], dtype=torch.long)
        g = DB.gen_batch(C, ids, [(True, None)], max_new=DREAM_MAX_NEW)[0]
        text = C.tok.decode(g["ids"], skip_special_tokens=True)
        qs, strict, lines = S.parse_dreams(text, N_DREAM)
        out["attempts"].append({"raw": text, "questions": qs, "strict_ok": strict, "lines": lines})
        if len(qs) == N_DREAM:
            break
    best = max(out["attempts"], key=lambda a: len(a["questions"]))
    qs = best["questions"]
    out["fallback"] = len(qs) < N_DREAM
    if not qs:                                  # nothing ends in '?': use the non-empty lines (recorded)
        qs = best["lines"][:N_DREAM]
    assert qs, "the dreamer produced no usable line"
    out["questions"] = qs
    out["retried"] = len(out["attempts"]) > 1
    return out


def _dq_attention(C, store, q):
    """[layers, Tq, P+Tq] head-mean attention of a dreamed question asked after the whole stored turn, the question's
    own content rows and the last prompt row."""
    ids = DB.qt_ids(C, q)
    _, att = fwd(C, ids, prefix_of(store), attn=True)
    M = head_mean(att)
    qc = S.content_positions(C.tok, ids, q)
    return M, qc, len(ids) - 1


def sel_dream(C, store, dreams):
    """S-dream (primary): per dreamed question, the mean attention from its question tokens and the last prompt token
    to each stored content position (heads and layers averaged); summed over the questions. Also S-dream-split
    (exploratory): 0.5 x question tokens in layers 0-13 + 0.5 x the last prompt token in layers 20-27."""
    cols = [store.nh + c for c in store.content]
    prim, split = [], []
    for q in dreams["questions"]:
        M, qc, last = _dq_attention(C, store, q)
        prim.append(S.query_mean(M.mean(0), qc + [last], cols))
        e = S.query_mean(M[list(SPLIT_EARLY)].mean(0), qc, cols)
        l_ = S.query_mean(M[list(SPLIT_LATE)].mean(0), [last], cols)
        split.append([0.5 * a + 0.5 * b for a, b in zip(e, l_)])
    return S.dream_combine(prim), S.dream_combine(split)


def sel_recon(C, store):
    """S-recon (KVzip-like): after the whole stored turn, 'Please repeat exactly what I just told you.' and the
    assistant header, then the turn's own text teacher-forced as the reply; mean attention each stored content
    position receives from the reconstruction tokens (heads and layers averaged)."""
    q = DB.qt_ids(C, RECON_PROMPT)
    rec = DB.text_ids(C.tok, store.turn)
    _, att = fwd(C, torch.cat([q, rec]), prefix_of(store), attn=True)
    M = head_mean(att).mean(0)
    rows = list(range(len(q), len(q) + len(rec)))
    return S.query_mean(M, rows, [store.nh + c for c in store.content])


def _batched_drop_cache(store, drops):
    """A DynamicCache of B rows: the whole stored turn minus one harvest-sequence index per row."""
    from transformers import DynamicCache
    T = store.nh + store.ns
    idx = torch.tensor([[i for i in range(T) if i != j] for j in drops], device=store.kv[0][0].device)   # [B, T-1]
    cache = DynamicCache()
    for l, (K, V) in enumerate(store.kv):
        cache.update(K[0][:, idx].permute(1, 0, 2, 3).contiguous(), V[0][:, idx].permute(1, 0, 2, 3).contiguous(), l)
    return cache


def _seq_lp(C, h, Lq, own):
    rows = torch.arange(Lq - 1, Lq - 1 + len(own), device=h.device)
    lg = C.lm_head(h[:, rows]).float()
    lp = torch.log_softmax(lg, -1)
    return lp.gather(-1, own.to(h.device)[None, :, None].expand(h.shape[0], -1, 1))[..., 0].sum(-1)


def sel_loo(C, store, dreams):
    """S-loo (leave-one-out reference): per dreamed question, the model's own greedy short answer with the whole slice;
    each content position's score = the drop in log P(own answer | dreamed question) when that position alone is
    removed from the whole slice, summed over the questions. Returns (scores, own answers)."""
    pf = prefix_of(store)
    start = pf[1]
    total = np.zeros(len(store.content))
    owns = []
    for q in dreams["questions"]:
        qi = DB.qt_ids(C, q)
        g = DB.gen_batch(C, qi, [(True, None)], prefix=pf, max_new=OWN_MAX_NEW)[0]
        own = torch.tensor(g["ids"], dtype=torch.long)
        owns.append({"q": q, "answer": C.tok.decode(g["ids"], skip_special_tokens=True), "n_tok": len(g["ids"])})
        if len(own) == 0:
            continue
        ids = torch.cat([qi, own])
        Lq = len(qi)
        h, _ = fwd(C, ids, pf)
        base = float(_seq_lp(C, h[None], Lq, own)[0])
        drops = []
        for b in range(0, len(store.content), LOO_BATCH):
            js = [store.nh + c for c in store.content[b:b + LOO_BATCH]]
            cache = _batched_drop_cache(store, js)
            pos = torch.arange(start, start + len(ids))[None].expand(len(js), -1).to(C.dev)
            o = C.body(input_ids=ids[None].expand(len(js), -1).to(C.dev), past_key_values=cache, position_ids=pos,
                       use_cache=True)
            drops += (base - _seq_lp(C, o.last_hidden_state, Lq, own)).tolist()
            del o, cache
        total += np.array(drops)
    return total.tolist(), owns


SELECTOR_FNS = {"S-attn": sel_attn, "S-dream": sel_dream, "S-recon": sel_recon, "S-loo": sel_loo, "dreamer": dream,
                "harvest": harvest}


# ----------------------------------------------------------------------------------------------- measures


def measure(C, store, cond, question, ans, prefix=None):
    """log P(gold answer | question) (sum, per-token mean) with the answer starting the reply, the first-token refusal
    mass and the scored rows' raw logits (for the identity check)."""
    tok = C.tok
    if cond == "nomem":
        q = DB.chat_ids(tok, question)
    elif cond == "ctx":
        q = DB.chat_ids(tok, f"{store.turn} {question}")
    elif cond == "T-full":
        _, q = DB.render_ids(tok, [{"role": "user", "content": store.turn}, {"role": "user", "content": question}])
        q = torch.tensor(q, dtype=torch.long)
    else:
        q = DB.qt_ids(C, question)
        assert prefix is not None
    ids = torch.cat([q, ans])
    h, _ = fwd(C, ids, prefix)
    Lq = len(q)
    rows = torch.arange(Lq - 1, len(ids) - 1, device=h.device)
    lg, lp = logprobs_rows(C, h, rows)
    s, m = S.answer_logprob(lp, 1, ans)
    first = lp[0]
    p = {k: math.exp(float(first[v])) for k, v in C.ref_ids.items()}
    top = first.topk(5)
    return {"answer_score": s, "answer_mean": m, "refusal": p["I"] + p["Sorry"] + p["As"], "p_first": p,
            "p_ans_first": math.exp(float(first[int(ans[0])])),
            "top5_first": [[tok.decode([int(i)]), math.exp(float(v))] for v, i in zip(top.values, top.indices)]}, lg


def run_item(C, it, budgets, idx):
    tok = C.tok
    t0 = time.time()
    turn = it["stored_turn"]
    store = harvest(C, turn)
    content = store.content
    n_c = len(content)
    # --- selection (no question / answer below until the measures)
    tim = {}
    t = time.time(); sc_attn = sel_attn(C, store); tim["S-attn"] = time.time() - t
    t = time.time(); dr = dream(C, store); tim["dreamer"] = time.time() - t
    t = time.time(); sc_dream, sc_split = sel_dream(C, store, dr); tim["S-dream"] = time.time() - t
    t = time.time(); sc_recon = sel_recon(C, store); tim["S-recon"] = time.time() - t
    loo_ok = n_c <= LOO_MAX_CONTENT
    t = time.time()
    sc_loo, owns = sel_loo(C, store, dr) if loo_ok else (None, None)
    tim["S-loo"] = time.time() - t
    scores = {"S-attn": sc_attn, "S-dream": sc_dream, "S-dream-split": sc_split, "S-recon": sc_recon, "S-loo": sc_loo}
    sels = {}           # (cond, budget) -> list of selections (random: one per seed)
    for b in budgets:
        k = S.budget_k(b, n_c)
        for name, sc in scores.items():
            if sc is not None:
                sels[(name, b)] = [S.topk(sc, content, k)]
        sels[("last-k", b)] = [S.last_k(content, k)]
        sels[("random", b)] = [S.random_k(content, k, (s, idx, k)) for s in RANDOM_SEEDS]
    # --- gold references (use the label)
    span = S.gold_span(tok, store.Seg, content, it["answer"]) if it["verbatim"] else None
    if it["verbatim"]:
        assert span, f"{it['question_id']}: verbatim item but no gold span"
    if span:
        sels[("gold", "own")] = [S.gold_plus_next(span, content)]
        for b in budgets:
            sels[("gold@k", b)] = [S.gold_at_k(span, content, S.budget_k(b, n_c))]
    # --- checks: kept counts and content-only picks
    for (c, b), L in sels.items():
        for sel in L:
            k = len(S.gold_plus_next(span, content)) if c == "gold" else S.budget_k(b, n_c)
            S.check_kept(S.keep_indices(store.nh, sel), store.nh, k, content)
    # --- measures
    ans = DB.text_ids(tok, it["answer"])
    rows = []
    res = {}
    for c in ANCHORS:
        pf = None
        if c == "R-full":
            pf = prefix_of(store)
        elif c == "R-content":
            pf = prefix_of(store, sel=content)
        res[c] = measure(C, store, c, it["question"], ans, pf)
    d_id = DB.maxdiff(res["R-full"][1], res["T-full"][1])
    assert d_id <= TF_TOL, f"{it['question_id']}: R-full TF != T-full ({d_id})"
    for c in ANCHORS:
        rows.append({"item": it["question_id"], "cond": c, "budget": None, "k": None, **res[c][0]})
    for (c, b), L in sels.items():
        ms = [measure(C, store, "slice", it["question"], ans, prefix_of(store, sel=sel))[0] for sel in L]
        r = {"item": it["question_id"], "cond": c, "budget": b, "k": len(L[0]), "sel": L,
             "answer_score": float(np.mean([m["answer_score"] for m in ms])),
             "answer_mean": float(np.mean([m["answer_mean"] for m in ms])),
             "refusal": float(np.mean([m["refusal"] for m in ms])),
             "hit": (float(np.mean([S.hit(s, span) for s in L])) if span else None)}
        if len(L) == 1:
            r["top5_first"] = ms[0]["top5_first"]
        else:
            r["per_seed_answer_score"] = [m["answer_score"] for m in ms]
        rows.append(r)
    dec = lambda i: tok.decode([int(store.Seg[i])])
    info = {"question_id": it["question_id"], "question": it["question"], "answer": it["answer"],
            "verbatim": it["verbatim"], "turn": turn, "n_head": store.nh, "n_seg": store.ns, "n_content": n_c,
            "content": content, "content_tokens": [dec(i) for i in content], "gold_span": span,
            "dreams": dr, "own_answers": owns, "loo_run": loo_ok, "scores": scores, "identity_max_abs_diff": d_id,
            "n_answer_tokens": len(ans), "seconds": time.time() - t0, "timing": tim}
    del store
    return rows, info


# ----------------------------------------------------------------------------------------------- main


def load_items(tiny=False):
    d = json.load(open(DATA))
    items = [x for x in d["items"] if x["usable"]]
    return (items[:3] if tiny else items), d["meta"]


def run(A):
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(DB_CFG))
    mc = cfg["models"][MODEL_KEY]
    dA = SimpleNamespace(random_model=None, dtype="fp32", device=A.device, tiny=A.tiny, gen_set=None, model=MODEL_KEY)
    model, tok, linfo = DB.load(dA, cfg, mc)
    C = DB.context(dA, cfg, mc, model, tok)
    DB.gen_context(C, cfg)
    C.ref_ids = {w: int(DB.text_ids(tok, w)[0]) for w in ("I", "Sorry", "As")}
    coh = DB.coherence_check(C)
    assert coh["ok"], f"Paris check failed: {coh}"
    log(f"Paris check: {coh['answer']!r}")
    items, dmeta = load_items(A.tiny)
    budgets = ["k4"] if A.tiny else list(S.BUDGETS)
    rows, infos = [], {}
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for i, it in enumerate(items):
        with torch.no_grad():
            rs, info = run_item(C, it, budgets, i)
        rows += rs
        infos[it["question_id"]] = info
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"{i + 1}/{len(items)} {it['question_id']}: n_content {info['n_content']} dreams "
            f"{'fallback ' if info['dreams']['fallback'] else ''}{info['dreams']['questions']} identity "
            f"{info['identity_max_abs_diff']:.1e} | {info['seconds']:.1f}s, {(time.time() - t0) / 60:.1f} min")
    with gzip.open(out / "rows.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {"items": infos, "budgets": budgets, "tiny": A.tiny, "data_meta": dmeta, "minutes": (time.time() - t0) / 60,
            "load": linfo, "paris": coh, "env": DB.env_meta(), "n_examples": N_EXAMPLES,
            "prompts": {"dream": DREAM_PROMPT, "dream_strict": DREAM_PROMPT_STRICT, "examples": DREAM_EXAMPLES, "recon": RECON_PROMPT},
            "peak_gpu_gb_allocated": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
            "peak_gpu_gb_reserved": torch.cuda.max_memory_reserved() / 2 ** 30 if torch.cuda.is_available() else None}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    log(f"done: {len(rows)} rows, {len(infos)} items in {meta['minutes']:.1f} min; peak GPU "
        f"{meta['peak_gpu_gb_reserved']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--tiny", action="store_true", help="3 items, k = 4 only (smoke)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--analyze-only", action="store_true")
    A = p.parse_args(argv)
    if not A.analyze_only:
        run(A)
    AN = _load("dream_select_analyze", HERE / "analyze.py")
    AN.analyze(Path(A.out))


if __name__ == "__main__":
    main()
