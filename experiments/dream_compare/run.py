#!/usr/bin/env python
"""dream_compare runner. Stage 0 (infrastructure; PREREG.md "Stage 0") has these stages; Stages A and B add theirs.

  smoke        (one model; GPU) load (fp32; Gemma: text decoder across 2 GPUs) + Paris, then
               identity     one short turn: the full stored turn restored == the turn as text (teacher-forced logits
                            and greedy tokens); per-head keep-all == the full cache; the same positions in every head ==
                            the physical splice; rope_shift == recompute in every layer (per layer type: partial RoPE
                            incl.); Gemma: the sliding window by RoPE position at the real window (1024) on a ~1,150-token
                            sequence (nothing evicted; contiguous == the model's own pass; sparse prefix == the masked
                            reference)
               benchmarks   one item per adapter (LongMemEval oracle, LoCoMo, PersonaMem 32k, BEAM 100K): its memory
                            (the evidence turn and the turn before it, session-level prefill with the chat head) is
                            harvested, restored (R-full, asserted == the text; R-evidence: head + the evidence turn's
                            slots; R-shift: the evidence turn harvested alone and moved to its place by compose_prefix +
                            rope_shift) and the gold answer is scored teacher-forced (PersonaMem: every option); BM25
                            recall@3 / @10 of the item; one greedy answer (R-full) for the judge smoke
               -> <out>/smoke.json, <out>/judge_tasks.jsonl; peak GPU memory per device
  judge-local  the original judge (Qwen3.5-9B bf16, core_v1's loader, thinking off, greedy) on a tasks file: LongMemEval
               (<= 10 new tokens), LoCoMo J (<= 200) and BEAM rubric (<= 512) prompts -> <out>/orig.jsonl
  judge-tasks  (CPU) the judge smoke's prompts: the model smokes' answers + a sanity set with known labels (the gold
               answer as the response = positive; "I'm sorry, I don't remember that." = negative) for every new
               prompt kind -> <out>/tasks.jsonl
  judge-summary (CPU) parse rates per judge and prompt kind, accuracy on the sanity set -> <out>/judge_smoke.json

  python -I experiments/dream_compare/run.py --stage smoke --model qwen25_15b --out RUN/smoke/qwen25_15b [--random-model]
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "benchmarks"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


J = _load("dream_compare_judges", HERE / "judges.py")
R = _load("dream_compare_retrieval", HERE / "retrieval.py")
PC = _load("dream_compare_prep_common", HERE / "benchmarks" / "prep_common.py")

BENCH_SPLITS = {"longmemeval": "oracle", "locomo": "locomo10", "personamem": "32k", "beam": "100K"}
SMOKE_PICK = {"longmemeval": "single-session-user", "locomo": "single-hop", "personamem": "recall_user_shared_facts",
              "beam": "information_extraction"}
MEM_MAX_TOKENS = 1200
TF_TOL = 0.01                 # dream_select's identity tolerance (max |logit| difference, fp32)
ROPE_RTOL = 1e-4              # rope_shift vs recompute: max |diff| / max |K|
GEN_MAX_NEW = 48
SANITY_N = 8
NEG = "I'm sorry, I don't remember that."


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------------------------- identity (short turn)


def smoke_identity(C, M, HM):
    import torch
    DB = M.DB()
    turn = "My dog's name is Pepper and I adopted him in Denver last spring, right after I started my nursing job."
    q = "What is my dog's name?"
    P = DB.turn_pieces(C.tok, turn)
    qt = DB.qt_ids(C, q)
    nh = len(P.H)
    res = {}
    with torch.no_grad():
        store, _, _ = M.harvest_kv(C, P.H + P.Seg)
        _, tids = DB.render_ids(C.tok, [{"role": "user", "content": turn}, {"role": "user", "content": q}])
        assert tids == P.H + P.Seg + qt.tolist()
        h, _ = M.forward(C, qt, store)
        ht, _ = M.forward(C, torch.tensor(tids))
        res["full_vs_text"] = DB.maxdiff(M.logits(C, h), M.logits(C, ht[-len(qt):]))
        a = M.gen_batch(C, qt, [(True, None)], prefix=store, max_new=12)[0]["ids"]
        b = M.gen_batch(C, torch.tensor(tids), [(True, None)], max_new=12)[0]["ids"]
        res["full_vs_text_tokens_equal"] = a == b
        with HM.HeadMask(C, HM.keep_all(C, store.n), store.n):
            h1, _ = M.forward(C, qt, store)
        res["headmask_keep_all_vs_full"] = DB.maxdiff(M.logits(C, h1), M.logits(C, h))
        slots = list(range(nh)) + [nh + 3, nh + 4, nh + 9, nh + 10, nh + 11]
        hp, _ = M.forward(C, qt, M.index_prefix(store, slots))
        with HM.HeadMask(C, HM.keep_shared(C, store.n, slots), store.n):
            hm, _ = M.forward(C, qt, store)
            gm = M.gen_batch(C, qt, [(True, None)], prefix=store, max_new=12)[0]["ids"]
        gp = M.gen_batch(C, qt, [(True, None)], prefix=M.index_prefix(store, slots), max_new=12)[0]["ids"]
        res["headmask_shared_vs_splice"] = DB.maxdiff(M.logits(C, hm), M.logits(C, hp))
        res["headmask_shared_vs_splice_tokens_equal"] = gm == gp
        # rope_shift == recompute, every layer, per layer type
        ids = DB.chat_ids(C.tok, "My dog's name is Pepper.").tolist()
        d = 37
        A0, _, _ = M.harvest_kv(C, ids)
        A1, _, _ = M.harvest_kv(C, ids, M.Prefix([], torch.zeros(0, dtype=torch.long), d))
        per = defaultdict(float)
        for l, ((K0, _), (K1, _)) in enumerate(zip(A0.kv, A1.kv)):
            s = M.rope_shift(C, K0, d, l)
            per[C.layout[l].kind] = max(per[C.layout[l].kind], DB.maxdiff(s, K1) / max(float(K1.abs().max()), 1e-6))
        res["rope_shift_rel_by_type"] = dict(per)
    res["ok"] = (res["full_vs_text"] <= TF_TOL and res["full_vs_text_tokens_equal"]
                 and res["headmask_keep_all_vs_full"] <= TF_TOL and res["headmask_shared_vs_splice"] <= TF_TOL
                 and res["headmask_shared_vs_splice_tokens_equal"]
                 and all(v <= ROPE_RTOL for v in res["rope_shift_rel_by_type"].values()))
    return res


def smoke_sliding(C, M):
    """Gemma at its real window: a sequence of window + ~130 tokens."""
    import torch
    DB = M.DB()
    W = C.window
    text = " ".join(f"Note {i}: the {['red', 'blue', 'green'][i % 3]} box holds item {i * 7 % 101}." for i in range(160))
    ids = DB.chat_ids(C.tok, text)
    assert len(ids) > W + 100, len(ids)
    ids = ids[:W + 130]
    n = len(ids)
    res = {"n_tokens": n, "window": W}
    with torch.no_grad():
        st, _, _ = M.harvest_kv(C, ids[:n - 20])
        res["kept_entries_min"] = min(int(K.shape[2]) for K, _ in st.kv)
        h, _ = M.forward(C, ids[n - 20:], st)
        own = C.body(input_ids=ids[None].to(M.input_device(C))).last_hidden_state[0, n - 20:]
        res["contiguous_vs_own"] = DB.maxdiff(M.logits(C, h), M.logits(C, own))
        # sparse: keep the head (first 3) and the last 200 slots; the far ones are > W away from the question
        keep = list(range(3)) + list(range(n - 220, n - 20))
        sp = M.index_prefix(st, keep, start=n - 20)
        hs, _ = M.forward(C, ids[n - 20:], sp)
        allp = torch.arange(n)
        Rm = {k: v.clone() for k, v in M.pos_masks(C, allp, allp).items()}
        drop = torch.ones(n, dtype=torch.bool)
        drop[torch.tensor(keep)] = False
        drop[n - 20:] = False
        for k in Rm:
            Rm[k][0, 0, n - 20:, drop.to(Rm[k].device)] = torch.finfo(torch.float32).min
        ref = C.body(input_ids=ids[None].to(M.input_device(C)), attention_mask=Rm).last_hidden_state[0, n - 20:]
        res["sparse_vs_masked_reference"] = DB.maxdiff(M.logits(C, hs), M.logits(C, ref))
        slot = C.body(input_ids=ids[None, n - 20:].to(M.input_device(C)), past_key_values=M.make_cache(sp.kv),
                      position_ids=torch.arange(n - 20, n)[None].to(M.input_device(C)), use_cache=True).last_hidden_state[0]
        res["sparse_slot_masks_vs_reference (expected large)"] = DB.maxdiff(M.logits(C, slot), M.logits(C, ref))
    res["ok"] = (res["kept_entries_min"] == n - 20 and res["contiguous_vs_own"] <= TF_TOL
                 and res["sparse_vs_masked_reference"] <= TF_TOL)
    return res


# ----------------------------------------------------------------------------------------------- benchmark items


def load_bench(name):
    return PC.load_split(name, BENCH_SPLITS[name])


def pick_item(b, tok, category):
    """The first item (file order) of `category` with answers (or options), labelled evidence, and a memory (the
    evidence turn + the turn before it in its session, system turns skipped) of <= MEM_MAX_TOKENS tokens."""
    for it in b.items:
        if it.category != category or it.abstention or not (it.answers or it.options):
            continue
        mem = memory_of(b, it)
        if mem is None:
            continue
        n = sum(len(tok(m["content"], add_special_tokens=False).input_ids) for m in mem[0])
        if n <= MEM_MAX_TOKENS:
            return it, mem
    raise AssertionError(f"{b.name}: no smoke item in {category}")


def memory_of(b, it):
    """([{role, content}], index of the evidence message) for an item: its first evidence turn and the turn before it
    in the same session (PersonaMem: the last user turn of its evidence session); None if there is none."""
    c = b.context(it)
    for s in c.sessions:
        turns = [t for t in s.turns if t.role != "system"]
        ids = [t.turn_id for t in turns]
        if it.evidence_turns:
            hit = next((i for i, t in enumerate(ids) if t in it.evidence_turns), None)
        elif s.session_id in it.evidence_sessions:
            hit = max((i for i, t in enumerate(turns) if t.role == "user"), default=None)
        else:
            hit = None
        if hit is None:
            continue
        lo = max(0, hit - 1)
        msgs = [{"role": t.role, "content": t.text.strip()} for t in turns[lo:hit + 1]]
        return msgs, hit - lo, [t.turn_id for t in turns[lo:hit + 1]]
    return None


def smoke_bench(C, M, name):
    import torch
    DB = M.DB()
    b = load_bench(name)
    it, (msgs, ev_i, mem_ids) = pick_item(b, C.tok, SMOKE_PICK[name])
    H, T, Q = M.session_pieces(C.tok, msgs, it.question)
    out = {"item": it.item_id, "category": it.category, "memory_turns": mem_ids, "n_head": len(H),
           "n_turn_tokens": [len(x) for x in T], "n_question_tokens": len(Q)}
    with torch.no_grad():
        mem = H + [t for x in T for t in x]
        store, _, _ = M.harvest_kv(C, mem)
        Qt = torch.tensor(Q)
        text_ids = torch.tensor(mem + Q)
        # identity at benchmark scale: R-full == the memory as text, question + answer rows
        ans = it.answers[0] if it.answers else it.options[it.correct_option]
        a_ids = DB.text_ids(C.tok, ans)
        hR, _ = M.forward(C, torch.cat([Qt, a_ids]), store)
        hT, _ = M.forward(C, torch.cat([text_ids, a_ids]))
        out["rfull_vs_text"] = DB.maxdiff(M.logits(C, hR), M.logits(C, hT[len(mem):]))
        # slices: head + the evidence turn's slots; the evidence turn harvested alone, shifted into place
        off = len(H) + sum(len(x) for x in T[:ev_i])
        ev_slots = list(range(len(H))) + list(range(off, off + len(T[ev_i])))
        r_ev = M.index_prefix(store, ev_slots)
        alone, _, _ = M.harvest_kv(C, H + T[ev_i])
        r_sh = M.compose_prefix(C, [("A", list(range(len(H))), 0),
                                    ("A", list(range(len(H), len(H) + len(T[ev_i]))), off - len(H))],
                                {"A": alone}, start=store.start)
        conds = {"nomem": (DB.chat_ids(C.tok, it.question), None), "text": (text_ids, None), "R-full": (Qt, store),
                 "R-evidence": (Qt, r_ev), "R-shift": (Qt, r_sh)}
        if it.options:
            sc = {}
            for c, (q, pf) in conds.items():
                lps, nts = [], []
                for o in it.options:
                    oi = DB.text_ids(C.tok, o)
                    s, _ = M.answer_score(C, q, oi, pf)
                    lps.append(s)
                    nts.append(len(oi))
                p_sum, p_mean = J.mc_pick(lps, nts)
                sc[c] = {"option_logprob": lps, "pick_sum": p_sum, "pick_mean": p_mean,
                         "correct_sum": p_sum == it.correct_option, "correct_mean": p_mean == it.correct_option}
            out["scores"] = sc
        else:
            out["scores"] = {c: dict(zip(("answer_score", "answer_mean"), M.answer_score(C, q, a_ids, pf)))
                             for c, (q, pf) in conds.items()}
        g = M.gen_batch(C, Qt, [(True, None)], prefix=store, max_new=GEN_MAX_NEW)[0]
        out["answer_R_full"] = C.tok.decode(g["ids"], skip_special_tokens=True).strip()
        gt = M.gen_batch(C, text_ids, [(True, None)], max_new=GEN_MAX_NEW)[0]
        out["answer_text_tokens_equal"] = g["ids"] == gt["ids"]
    out["recall"] = {m: R.item_recall(b, it, R.retrieve(b, it, m)) for m in (3, 10)}
    out["ok"] = out["rfull_vs_text"] <= TF_TOL and out["answer_text_tokens_equal"]
    tasks = judge_tasks_for(b, it, out["answer_R_full"], source=f"{C.key}|{name}")
    return out, tasks


# ----------------------------------------------------------------------------------------------- judge tasks


def judge_tasks_for(b, it, response, source, expect=None):
    """The judge prompts of one (item, response): LongMemEval (its type's template), LoCoMo J, BEAM (one per rubric
    item); PersonaMem none."""
    out = []
    if b.name == "longmemeval":
        out.append(("lme", J.lme_prompt(it.category, it.question, it.answers[0], response, it.abstention)))
    elif b.name == "locomo":
        out.append(("locomo_j", J.j_prompt(it.question, it.answers[0] if it.answers else "Not mentioned", response)))
    elif b.name == "beam":
        for r in it.rubric:
            out.append(("beam", J.beam_prompt(r, response)))
    return [{"h": J.phash(p), "kind": k, "prompt": p, "keys": [f"{source}|{it.item_id}|{i}"], "expect": expect,
             "item": it.item_id, "bench": b.name} for i, (k, p) in enumerate(out)]


def sanity_tasks():
    """Known-label prompts per new kind: SANITY_N items each (the first of their file with an answer), the gold answer
    as the response (expect positive) and NEG (expect negative). LongMemEval: types other than single-session-user
    first (dream_select's smoke covered that template)."""
    out = []
    for name, pick in (("longmemeval", lambda it: not it.abstention and it.category != "single-session-user"),
                       ("locomo", lambda it: not it.abstention and it.category != "adversarial"),
                       ("beam", lambda it: it.answers and it.rubric and not it.abstention)):
        b = load_bench(name)
        seen = Counter()
        chosen = []
        for it in b.items:          # spread over categories
            if pick(it) and seen[it.category] < max(1, SANITY_N // 4):
                chosen.append(it)
                seen[it.category] += 1
            if len(chosen) >= SANITY_N:
                break
        for it in chosen:
            out += judge_tasks_for(b, it, it.answers[0], "sanity_pos", expect=True)
            out += judge_tasks_for(b, it, NEG, "sanity_neg", expect=False)
    return out


def merge_tasks(lists):
    by = {}
    for t in (x for L in lists for x in L):
        if t["h"] in by:
            by[t["h"]]["keys"] += t["keys"]
        else:
            by[t["h"]] = dict(t)
    return sorted(by.values(), key=lambda t: t["h"])


# ----------------------------------------------------------------------------------------------- stages


def stage_smoke(A):
    import torch
    M = _load("dream_compare_models", HERE / "models.py")
    HM = _load("dream_compare_headmask", HERE / "headmask.py")
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if A.random_model:
        from transformers import AutoTokenizer
        C = M.tiny_context(A.model, AutoTokenizer.from_pretrained(M.SPECS[A.model]["hf"]), attn=A.attn)
    else:
        C = M.setup(A.model, A.device, attn=A.attn, gb_per_gpu=A.gb_per_gpu)
    log(f"{C.label}: loaded in {time.time() - t0:.0f}s; layers {len(C.layout)}, sliding {C.sliding}, "
        f"softcap {C.softcap}; paris {C.paris}; {C.load_info}")
    res = {"model": A.model, "hf": C.spec["hf"], "random_model": bool(A.random_model), "load": C.load_info,
           "paris": C.paris, "load_peak_gb": M.peak_gb(), "env": M.DB().env_meta()}
    res["identity"] = smoke_identity(C, M, HM)
    log(f"identity: {json.dumps(res['identity'])}")
    if C.sliding:
        res["sliding"] = smoke_sliding(C, M) if not A.random_model else {"ok": True, "skipped": "random model"}
        log(f"sliding: {json.dumps(res['sliding'])}")
    tasks = []
    res["benchmarks"] = {}
    for name in BENCH_SPLITS:
        r, t = smoke_bench(C, M, name)
        res["benchmarks"][name] = r
        tasks += t
        log(f"{name}: {json.dumps({k: v for k, v in r.items() if k != 'scores'})}")
        log(f"{name} scores: {json.dumps(r['scores'])}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    res["peak_gb"] = M.peak_gb()
    res["minutes"] = (time.time() - t0) / 60
    res["ok"] = res["identity"]["ok"] and res.get("sliding", {}).get("ok", True) and \
        all(r["ok"] for r in res["benchmarks"].values())
    json.dump(res, open(out / "smoke.json", "w"), indent=1, default=str)
    with open(out / "judge_tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    log(f"smoke {'PASSED' if res['ok'] else 'FAILED'}: peak GPU {res['peak_gb']}, {res['minutes']:.1f} min")
    assert res["ok"], "smoke checks failed (see smoke.json)"


def stage_judge_tasks(A):
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    lists = [sanity_tasks()]
    for p in sorted(Path(A.tasks_from).glob("*/judge_tasks.jsonl")) if A.tasks_from else []:
        lists.append([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])
    tasks = merge_tasks(lists)
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    log(f"judge tasks: {len(tasks)} unique prompts ({dict(Counter(t['kind'] for t in tasks))}) -> {out / 'tasks.jsonl'}")


JUDGE_MAX_NEW = {"lme": 10, "locomo_j": 200, "beam": 512}


def stage_judge_local(A):
    """The original judge (dream_select stage_judge's set-up) on every task, max new tokens per kind."""
    import torch
    import yaml
    out = Path(A.out)
    tasks = [json.loads(l) for l in open(out / "tasks.jsonl", encoding="utf-8") if l.strip()]
    if A.limit:
        tasks = tasks[:A.limit]
    D = _load("dream_compare_ds_run_j", REPO / "experiments" / "dream_select" / "run.py")
    cvr = _load("dream_compare_core_v1_run", REPO / "experiments" / "core_v1" / "run.py")
    cfg = yaml.safe_load(open(D.DB_CFG))
    Aj = SimpleNamespace(random_model="qwen2" if A.random_model else None, device=A.device)
    t0 = time.time()
    jm, jt = cvr.load({"judge": cfg["judge"], "model": "Qwen/Qwen3.5-2B"}, Aj, judge=True)
    body, lm_head = D.DB.text_model(jm), jm.lm_head
    eos = {jt.eos_token_id} | {jt.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    eos_list = sorted(e for e in eos if isinstance(e, int) and e != jt.unk_token_id)
    enc = {t["h"]: D.DB.chat_ids(jt, t["prompt"]) for t in tasks}
    by = defaultdict(list)
    for t in tasks:
        by[(t["kind"], len(enc[t["h"]]))].append(t)
    bs = cfg["judge"]["batch"]
    recs = []
    with torch.no_grad():
        for (kind, n), ts in sorted(by.items()):
            for i in range(0, len(ts), bs):
                chunk = ts[i:i + bs]
                P = torch.stack([enc[t["h"]] for t in chunk])
                outs = cvr.core.generate_batch(jm, body, lm_head, P, [(True, None)] * len(chunk),
                                               max_new=JUDGE_MAX_NEW[kind], eos=eos_list, vocab=lm_head.out_features)
                for t, o in zip(chunk, outs):
                    raw = jt.decode(o["ids"], skip_special_tokens=True).strip()
                    recs.append({"h": t["h"], "judge": "orig", "kind": kind, "keys": t["keys"], "raw": raw[:600],
                                 **J.PARSERS[kind](raw), "capped": o["capped"]})
    with open(out / "orig.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    log(f"original judge: {len(recs)} prompts in {(time.time() - t0) / 60:.1f} min; parsed "
        f"{sum(r['ok'] for r in recs) / max(len(recs), 1):.3f}")


def _label(rec):
    if not rec.get("ok"):
        return None
    if rec["kind"] == "beam":
        return rec["score"]
    return rec["label"]


def stage_judge_summary(A):
    out = Path(A.out)
    tasks = {t["h"]: t for t in (json.loads(l) for l in open(out / "tasks.jsonl", encoding="utf-8") if l.strip())}
    res = {}
    for jf in ("orig", "gptoss", "gemma4"):
        p = out / f"{jf}.jsonl"
        if not p.exists():
            res[jf] = None
            continue
        recs = {}
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            recs[r["h"]] = r
        by = defaultdict(lambda: {"n": 0, "parsed": 0, "sanity_n": 0, "sanity_right": 0})
        for h, r in recs.items():
            t = tasks.get(h)
            if t is None:
                continue
            k = by[t["kind"]]
            k["n"] += 1
            k["parsed"] += bool(r.get("ok"))
            lab = _label(r)
            if t.get("expect") is not None and lab is not None:
                k["sanity_n"] += 1
                pos = (lab >= 0.5) if t["kind"] == "beam" else bool(lab)
                k["sanity_right"] += pos == t["expect"]
        res[jf] = {kind: dict(v, parse_rate=v["parsed"] / max(v["n"], 1),
                              sanity_accuracy=(v["sanity_right"] / v["sanity_n"]) if v["sanity_n"] else None)
                   for kind, v in sorted(by.items())}
    res["n_tasks"] = dict(Counter(t["kind"] for t in tasks.values()))
    json.dump(res, open(out / "judge_smoke.json", "w"), indent=1)
    log("judge smoke: " + json.dumps(res))


def main(argv=None):
    import torch
    av = list(sys.argv[1:] if argv is None else argv)
    if "--stage" in av and av.index("--stage") + 1 < len(av) and av[av.index("--stage") + 1].startswith("a-"):
        return _load("dream_compare_stage_a_run", HERE / "stage_a.py").main(av)      # Stage A (stage_a.py)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["smoke", "judge-tasks", "judge-local", "judge-summary"])
    ap.add_argument("--model", default="qwen25_15b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--random-model", action="store_true", help="tiny random model with the real tokenizer (CPU)")
    ap.add_argument("--gb-per-gpu", type=int, default=66, help="max_memory per GPU for multi-GPU loading")
    ap.add_argument("--tasks-from", default=None, help="judge-tasks: the smoke run root (holds <model>/judge_tasks.jsonl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true")
    A = ap.parse_args(argv)
    {"smoke": stage_smoke, "judge-tasks": stage_judge_tasks, "judge-local": stage_judge_local,
     "judge-summary": stage_judge_summary}[A.stage](A)


if __name__ == "__main__":
    main()
