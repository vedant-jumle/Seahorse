#!/usr/bin/env python
"""dream_subj (PREREG.md): does a restored K,V slice need the subject's own position to answer a direct question?

Qwen2.5-1.5B-Instruct, fp32, eager attention, local (RTX 4060). Teacher-forced only (plus a short greedy answer for the
5 pre-named example facts, read, never scored). Reuses dream_bind's model loading, chat-turn pieces and splice code by
import (experiments/dream_bind/run.py and bind.py are not modified).

Per fact: one pass over [head][experience user turn] gives K,V at every layer. Each condition keeps the head plus a set
of experience-turn entries at their stored RoPE positions (subj.keep_sets); the question turn starts at n_head + n_seg.

  python experiments/dream_subj/run.py --out results/dream_subj_<stamp> [--tiny]
  python experiments/dream_subj/run.py --out <same dir> --analyze-only

Checks that fail the run: R-full TF == the sentence as a prior user turn (logits within 0.01); R-min's kept indices ==
bind.splice_keep('R-min'); subject index < value index for every included fact; every condition keeps the specified
number of entries; the subject token == dream_bind's ea_positions entity token (person / pet facts); Paris.
"""

import argparse
import gzip
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import subj  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DB = _load("dream_subj_db_run", REPO / "experiments" / "dream_bind" / "run.py")   # dream_bind's run.py (import only)
bind = DB.bind
DB_CFG = REPO / "experiments" / "dream_bind" / "config.yaml"
MODEL_KEY = "qwen25_15b"
EXAMPLES = ["dog_name", "sister_name", "best_friend_name", "pool_hairdresser", "pool_mother_in_law"]
ATTN_CONDS = ("R+subj", "R-subj2")
TF_TOL = 0.01
N_BOOT, SEED = 2000, 7


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------------------------- facts


def load_facts(cfg, tiny=False):
    """The 7 primary facts (dream_bind load_items, tier 'primary'), then the pool facts exactly as addendum 7 C builds
    them (dream_bind pool_gen_items: the 5 secondary items + 47 pool facts, sample_pool seed 0, hand-written slots
    excluded, aliases 'my {slot}' / 'your {slot}' / '{slot}'). Tiers: primary / pool (person or pet) / attribute."""
    out = []
    for it in DB.load_items(cfg):
        if it["tier"] == "primary":
            out.append({**it, "tier": "primary", "kind": "pet" if it["id"] in ("dog_name", "cat_name") else "person"})
    for it in DB.pool_gen_items(cfg):
        out.append({**it, "tier": "pool" if it["kind"] in ("person", "pet") else "attribute"})
    if tiny:
        out = [f for f in out if f["id"] in ("dog_name", "pool_hairdresser")]
    return out


# ----------------------------------------------------------------------------------------------- forward


def ids_for(C, f, c, text, pre, pos, prefixes):
    """(ids, reinstatement prefix or None) of one teacher-forced prompt under condition c; `pre` = the measure-prefix
    ids appended at the end (or an empty tensor)."""
    tok = C.tok
    if c == "nomem":
        return torch.cat([DB.chat_ids(tok, text), pre]), None
    if c == "ctx":
        return torch.cat([DB.chat_ids(tok, f"{f['experience']} {text}"), pre]), None
    if c.startswith("T-"):
        prior = {"T-full": f["experience"], "T-min": f"{f['target'].strip()}.", "T-subj3": subj.text_subj3(pos)}[c]
        _, ids = DB.render_ids(tok, [{"role": "user", "content": prior}, {"role": "user", "content": text}])
        return torch.cat([torch.tensor(ids, dtype=torch.long), pre]), None
    return torch.cat([DB.qt_ids(C, text), pre]), prefixes[c]


def forward(C, ids, prefix, attn=False):
    """Teacher-forced pass after an optional stored prefix ([(K, V)] per layer, start position). Returns (log-probs
    [T, V], attentions or None)."""
    start = prefix[1] if prefix else 0
    cache = bind.make_cache(prefix[0] if prefix else [], 1)
    pos = torch.arange(start, start + len(ids))[None].to(C.dev)
    o = C.body(input_ids=ids[None].to(C.dev), past_key_values=cache, position_ids=pos, use_cache=True,
               output_attentions=attn)
    lp = torch.log_softmax(C.lm_head(o.last_hidden_state[0]).float(), -1)
    if not bool(torch.isfinite(lp).all()):
        raise FloatingPointError("non-finite log-probs")
    return lp, (o.attentions if attn else None)


def harvest(C, f):
    """One pass over [head][experience turn]: the kept slices of every condition, the positions and the checks."""
    P = DB.turn_pieces(C.tok, f["experience"])
    assert P.H == C.head, "head differs from the template head"
    nh, ns = len(P.H), len(P.Seg)
    t = int(DB.text_ids(C.tok, f["target"])[0])
    pos = subj.slice_positions(C.tok, P.Seg, f["experience"], f["target"], f["aliases"], value_tok=t)
    assert pos["subj"] < pos["value"], f"{f['id']}: subject {pos['subj']} not before the value {pos['value']}"
    if f["tier"] != "primary" and f["kind"] not in ("person", "pet"):
        pos["rel"] = None            # attribute facts: no relation token (PREREG: R+rel skipped, recorded)
    keeps = subj.keep_sets(nh, ns, pos)
    subj.check_counts(keeps, nh, ns)
    k_min, s_min = bind.splice_keep("R-min", nh, ns, pos["value"])
    assert keeps["R-min"] == (k_min, s_min), f"{f['id']}: R-min != bind.splice_keep {keeps['R-min']} {k_min}"
    assert keeps["R-full"] == bind.splice_keep("R-full", nh, ns, pos["value"]), f"{f['id']}: R-full != splice_keep"
    ea = None
    if f["kind"] in ("person", "pet"):
        ids = P.H + P.Seg
        ea = bind.ea_positions(C.tok, ids, f["experience"], f["target"])
        assert ea["entity"][0] == nh + pos["subj"], f"{f['id']}: subject {nh + pos['subj']} != ea_positions {ea}"
        assert ea["attr"][0] == nh + pos["value"], f"{f['id']}: value != ea_positions {ea}"
    kv = bind.kv_list(C.body(input_ids=torch.tensor(P.H + P.Seg)[None].to(C.dev), use_cache=True).past_key_values)
    prefixes = {}
    for c, (k, start) in keeps.items():
        ki = torch.tensor(k, device=C.dev)
        prefixes[c] = ([(K[:, :, ki].clone(), V[:, :, ki].clone()) for K, V in kv], start)
    skipped = [c for c in subj.R_CONDS if c not in keeps]
    info = {"n_head": nh, "n_seg": ns, "pos": {k: pos[k] for k in ("subj", "value", "next", "fill", "rel")},
            "tok": pos["text"], "alias": pos["alias"], "subj_span": pos["subj_span"], "skipped": skipped,
            "kept": {c: len(k) - nh for c, (k, _) in keeps.items()}, "ea": ea, "t_subj3": subj.text_subj3(pos),
            "n_value_tokens": len(DB.text_ids(C.tok, f["target"]))}
    return pos, keeps, prefixes, info


def run_fact(C, f, conds, greedy=False):
    tok = C.tok
    pos, keeps, prefixes, info = harvest(C, f)
    nh = info["n_head"]
    t = int(DB.text_ids(tok, f["target"])[0])
    pre = DB.text_ids(tok, f["prefix"])
    empty = torch.tensor([], dtype=torch.long)
    have = [c for c in conds if not (c in subj.R_CONDS and c not in prefixes)]
    rows = []
    # check: R-full == the sentence as a prior user turn (direct question + measure prefix)
    d_id = DB.maxdiff(DB.tf_last(C, *ids_for(C, f, "R-full", f["question"], pre, pos, prefixes)),
                      DB.tf_last(C, *ids_for(C, f, "T-full", f["question"], pre, pos, prefixes)))   # last-position logits
    assert d_id <= TF_TOL, f"{f['id']}: R-full TF != the sentence as a prior turn ({d_id})"
    info["identity_max_abs_diff"] = d_id
    tid = {"Your": int(pre[0]), "I": int(DB.text_ids(tok, "I")[0]), "Sorry": int(DB.text_ids(tok, "Sorry")[0]),
           "As": int(DB.text_ids(tok, "As")[0]), "value": int(DB.text_ids(tok, f["target"].strip())[0])}
    prompts = [("direct", 0, f["question"])] + [("use", i + 1, u) for i, u in enumerate(f["use"])]
    for kind, pi, text in prompts:
        q_site = bind.subject_sites(tok, DB.chat_ids(tok, text), text, f["aliases"])["main"]
        for c in have:
            ids, pf = ids_for(C, f, c, text, pre, pos, prefixes)
            attn = c in ATTN_CONDS and pf is not None
            lp, att = forward(C, ids, pf, attn=attn)
            Lq = len(ids) - len(pre)
            first = lp[Lq - 1]
            top = first.topk(5)
            r = {"sec": "tf", "fact": f["id"], "tier": f["tier"], "kind": f["kind"], "pkind": kind, "pi": pi, "cond": c,
                 "answer_score": subj.prefix_score(lp, Lq, pre), "lp_target": float(lp[-1, t]),
                 "p_first": {k: math.exp(float(first[v])) for k, v in tid.items()},
                 "top5_first": [[tok.decode([int(i)]), math.exp(float(v))] for v, i in zip(top.values, top.indices)]}
            if attn:
                # the question turn's own indices (the kept cache holds nh + k entries before them)
                qs = [q - nh for q in q_site] if q_site else []
                P_ = prefixes[c][0][0][0].shape[2]           # kept entries (head + the slice)
                ent = {"subj": nh, "value": nh + 1} | ({"next": nh + 2} if c == "R+subj" else {})
                A = {}
                for site, q in (("last", Lq - 1), ("qsubj", qs[0] if qs else None)):
                    if q is None:
                        continue
                    A[site] = {e: [float(L[0, :, q, k].mean()) for L in att] for e, k in ent.items()}
                    A[site]["head"] = [float(L[0, :, q, :nh].sum(-1).mean()) for L in att]
                    A[site]["own"] = [float(L[0, :, q, P_:].sum(-1).mean()) for L in att]
                r["attn"] = A
                r["attn_kept"] = P_
            rows.append(r)
    # TF yes/no on the bench's balanced verify probes
    for rp in f["relation_probes"]:
        for c in have:
            lp, _ = forward(C, *ids_for(C, f, c, rp["text"], empty, pos, prefixes))
            m = float(lp[-1, C.yes] - lp[-1, C.no])
            rows.append({"sec": "yn", "fact": f["id"], "tier": f["tier"], "kind": f["kind"], "cond": c,
                         "probe": rp["text"], "consistent": rp["a"], "margin": m})
    if greedy:
        for c in have:
            ids, pf = ids_for(C, f, c, f["question"], empty, pos, prefixes)
            o = DB.gen_batch(C, ids, [(True, None)], prefix=pf, max_new=40)[0]
            rows.append({"sec": "greedy", "fact": f["id"], "tier": f["tier"], "kind": f["kind"], "cond": c,
                         "prompt": f["question"], "answer": tok.decode(o["ids"], skip_special_tokens=True)})
    return rows, info


# ----------------------------------------------------------------------------------------------- main


def run(A):
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(DB_CFG))
    mc = cfg["models"][MODEL_KEY]
    dA = SimpleNamespace(random_model=None, dtype="fp32", device=A.device, tiny=A.tiny, gen_set=None, model=MODEL_KEY)
    model, tok, linfo = DB.load(dA, cfg, mc)
    C = DB.context(dA, cfg, mc, model, tok)
    DB.gen_context(C, cfg)
    coh = DB.coherence_check(C)
    assert coh["ok"], f"Paris check failed: {coh}"
    log(f"Paris check: {coh['answer']!r}")
    facts = load_facts(cfg, A.tiny)
    conds = list(subj.CONDS)
    rows, infos, excluded = [], {}, []
    t0 = time.time()
    for i, f in enumerate(facts):
        try:
            with torch.no_grad():
                rs, info = run_fact(C, f, conds, greedy=f["id"] in EXAMPLES or A.tiny)
        except AssertionError as e:
            if f["tier"] == "attribute" and "no subject alias" in str(e):
                excluded.append({"fact": f["id"], "tier": f["tier"], "reason": str(e)})
                log(f"excluded {f['id']}: {e}")
                continue
            raise
        rows += rs
        infos[f["id"]] = {**info, "tier": f["tier"], "kind": f["kind"], "experience": f["experience"],
                          "question": f["question"], "use": f["use"], "target": f["target"]}
        log(f"{i + 1}/{len(facts)} {f['id']} ({f['tier']}): subj {info['tok']['subj']!r} skipped {info['skipped']} "
            f"identity {info['identity_max_abs_diff']:.1e} | {(time.time() - t0) / 60:.1f} min")
    with gzip.open(out / "rows.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {"facts": infos, "excluded": excluded, "conds": conds, "examples": EXAMPLES, "tiny": A.tiny,
            "minutes": (time.time() - t0) / 60, "load": linfo, "paris": coh, "env": DB.env_meta(),
            "peak_gpu_gb": torch.cuda.max_memory_reserved() / 2 ** 30 if torch.cuda.is_available() else None}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    log(f"done: {len(rows)} rows, {len(infos)} facts in {meta['minutes']:.1f} min; peak GPU {meta['peak_gpu_gb']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--tiny", action="store_true", help="2 facts (smoke)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--analyze-only", action="store_true")
    A = p.parse_args(argv)
    if not A.analyze_only:
        run(A)
    import analyze
    analyze.analyze(Path(A.out))


if __name__ == "__main__":
    main()
