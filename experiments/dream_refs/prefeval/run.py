#!/usr/bin/env python
"""PrefEval benchmark for experiment 2 (dream_refs) on Qwen3.5-2B. PREREG.md "Addendum: PrefEval" fixes the
sample, conditions, measures and decision rules.

  python experiments/dream_refs/prefeval/run.py --stage prep --out RUN/qwen35_2b --valence V   # phase A
  ... --stage gen [--items ...] [--sections main | unrel]                                        # phase B
  ... --stage score                                                                              # coherence + judge
  ... --stage analyze [--panel RUN/panel]
  ... --tiny --random-model qwen2                                                                # CPU smoke

V = the dream_refs (29-item, prompts v3) run's qwen35_2b/prep/state.pt: its general valence direction v_l is
the sign reader here (no PrefEval item enters it).

Stages
  prep   core_v1's calibration recomputed (checked against core_v1's saved mu / W / thresholds); per item with
         valid follow-ups, core_v1's pooled write (cv1.build_moments, the same code path) of
           without        with - without (mechanical)
           gen:opposite   with - the dreamed opposite
           gen:centroid   with - mean of the dreamed alternatives
           cs:concept_x_sign  s_l * gen:centroid, s_l = sign(<without_l, v_l>)
         under the pooled_w256 gist keys of the dreamed follow-ups (never the test question), the q0.95
         threshold per item and layer; then, with no generation, the gate on each test question (the gate open
         rate), the gate on core_v1's 50 unrelated prompts (false fire), the key cosine between the follow-ups and
         the question, the valence sign vs the lexical polarity, and the logit lens of each shift -> report_A.md
  gen    nomem, ctx (the preference in the prompt), and gen:opposite / gen:centroid / cs:concept_x_sign at
         alpha 0.5 / 1 / 2 through the item's own gate; core_v1's generator, sampler and random streams (1 greedy
         + 4 samples per question), cap 200; the 50 unrelated prompts for nomem and the alpha = 1 cells
  score  coherence (NLL under the no-memory 2B) and the original judge (Qwen3.5-9B bf16): core_v1's preference
         rubric (j_lean) and PrefEval's four rubrics (acknowledgement, violation, helpfulness, hallucination)
  analyze  analyze.py
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


drun = _load("seahorse_dream_refs_run", EXP / "run.py")
cv1, core, dk, sh, log = drun.cv1, drun.core, drun.dk, drun.sh, drun.log
refs = drun.refs
bench = _load("seahorse_prefeval_bench", HERE / "bench.py")
W256 = cv1.W256
from seahorse.sessions import chat_ids  # noqa: E402

# condition prefix -> stored memory name (= shift name)
VARIANTS = {"gen:opposite": "gen_opp", "gen:centroid": "gen_cent", "cs:concept_x_sign": "cs_sign"}
KEYS_MEM = "without"   # the plain memory: never a condition; it holds the item's keys for the gate ("design")
LENS = {"without": "without", "gen_opp": "gen:opposite", "gen_cent": "gen:centroid", "cs_sign": "cs:concept_x_sign"}
PE_SYSTEM = ("You are a helpful assistant in evaluating an AI assistant's reponse. You should be fair and strict and "
             "follow the user's instruction")   # PrefEval's evaluator system prompt, verbatim (typo included)
PE_FILES = {"acknow": "check_acknowledge.txt", "violate": "check_violation.txt", "helpful": "check_helpful.txt",
            "hallucinate": "check_hallucination.txt"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--stage", required=True, nargs="+", choices=["prep", "gen", "score", "analyze"])
    p.add_argument("--out", required=True, help="RUN/qwen35_2b")
    p.add_argument("--refs", default=None, help="validate.py's refs.json (default <out>/../validate/refs.json)")
    p.add_argument("--valence", default=None, help="the dream_refs v3 run's qwen35_2b/prep/state.pt (v_l)")
    p.add_argument("--items", nargs="+", default=None)
    p.add_argument("--shard", default=None, help="gen: 'k/n' = every n-th item starting at k (0-based)")
    p.add_argument("--sections", nargs="+", default=["main", "unrel"], choices=["main", "unrel"])
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--panel", default=None)
    p.add_argument("--random-model", default=None, choices=["qwen2"])
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def load_cfg(path):
    cfg = yaml.safe_load(open(path))
    for k in ("items", "bench", "generic", "data"):
        cfg[k] = str((REPO / cfg[k]).resolve()) if not os.path.isabs(cfg[k]) else cfg[k]
    return cfg


def tiny_cfg(cfg):
    c = copy.deepcopy(cfg)
    c.update(samples=2, answer_cap=24, unrel_samples=1, max_batch=48, tiny=True, n_unrel=3)
    c["pe_judge"] = dict(c["pe_judge"], max_new=48)
    return c


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_items(cfg, R):
    """PrefEval items (from refs.json) in core_v1's item shape, plus core_v1's unrelated prompts and placebo."""
    cv = cv1.itm.load_items(cfg["items"], cfg["bench"])
    items = []
    for iid, r in R["items"].items():
        items.append({"id": iid, "group": "pref", "polarity": r["polarity"], "topic": r["topic"], "row": r["row"],
                      "experience": r["experience"], "question": r["question"], "prompts": [("question", r["question"])],
                      "followups": [f["text"] for f in r["followups"]], "excluded": r["excluded_variants"],
                      "failed": r["failed"]})
    return {"items": items, "by_id": {it["id"]: it for it in items}, "unrelated": cv["unrelated"],
            "placebo": cv["placebo"], "cv1_by_id": cv["by_id"]}


def has_memory(it):
    return not it["failed"].get("followups")


# ============================================================================================== prep


def calib_check(C, cfg, L):
    """The recomputed calibration must reproduce core_v1's saved one (same model, prompts, code): mu and W, and
    the q0.95 thresholds of three core_v1 items recomputed from their saved keys with the new generic keys."""
    p = Path(cfg.get("cv1_prep") or "", "state.pt")
    if cfg.get("tiny") or C.A.random_model or not p.exists():
        return {"skipped": f"no core_v1 state at {p}" if not (cfg.get("tiny") or C.A.random_model) else "tiny"}
    st = torch.load(p, map_location="cpu", weights_only=False)
    res = {"mu_max_rel": max(drun.rel(C.mu[l].cpu(), st["mu"][l]) for l in C.layers),
           "W_max_rel": max(drun.rel(C.KS[l][W256].W.cpu(), st["W"][l]) for l in C.layers), "thr_max_abs": 0.0}
    for iid in list(st["moments"])[:3]:
        for l in C.layers:
            K = torch.cat([m["key"][l] for m in st["moments"][iid]]).to(C.dev).float()
            thr, _ = dk.calib(SimpleNamespace(Kst=K), C.Kg[l], cfg["key"]["thr_q"])
            res["thr_max_abs"] = max(res["thr_max_abs"], abs(thr - st["thr"][iid][l]))
    res["ok"] = res["mu_max_rel"] < 1e-3 and res["W_max_rel"] < 1e-3 and res["thr_max_abs"] < 1e-3
    log(f"calibration vs core_v1: {json.dumps(res)}")
    assert res["ok"], f"recomputed calibration does not reproduce core_v1's: {res}"
    return res


def item_moments(C, it, r, placebo):
    """core_v1's build_moments on a pseudo-item; a failed reference is replaced by a stand-in whose shift is
    discarded (as dream_refs prep). Returns moments with shifts {without, gen_opp?, gen_cent?} at the read layers."""
    ok_o, ok_c = not it["failed"]["opposite"], not it["failed"]["centroid"]
    alts = [a["text"] for a in r["alternatives"]]
    if ok_o or ok_c:
        ps = {"id": it["id"], "group": "leanings", "experience": it["experience"], "followups": it["followups"],
              "counter": r["opposite"]["text"] if ok_o else alts[0],
              "centroid": alts if ok_c else [r["opposite"]["text"]]}
    else:
        ps = {"id": it["id"], "group": "facts", "experience": it["experience"], "followups": it["followups"],
              "counter": None, "centroid": None}
    ms = drun.moments_at(C, ps, placebo)
    out = []
    for m in ms:
        S = {"without": m["shifts"]["plain"]}
        if ok_o:
            S["gen_opp"] = m["shifts"]["opposite"]
        if ok_c:
            S["gen_cent"] = m["shifts"]["centroid"]
        out.append({"j": m["j"], "n": m["n"], "key": m["key"], "shifts": S})
    return out


def memories(C, it, moments):
    """{layer: {name: Mem}} for every stored shift; "design" = the plain memory (keys for the gate only)."""
    lam = C.cfg["key"]["rls_lambda"]
    M = {}
    for l in C.layers:
        keys = torch.cat([m["key"][l] for m in moments]).to(C.dev)
        M[l] = {}
        for name in moments[0]["shifts"]:
            S = torch.stack([m["shifts"][name][l].to(C.dev).float() for m in moments])
            M[l][name] = dk.build({it["id"]: core.memory_chunks(S, keys)}, [it["id"]], "delta", lam, C.dev, C.chk)
        M[l]["design"] = M[l][KEYS_MEM]
    return M


def dummy_memory(C):
    """For an item whose follow-ups failed (no memory): nomem / ctx only; the gate is closed by an infinite
    threshold and reported as such."""
    r = C.KS[C.layers[0]][W256].W.shape[1]
    return {l: {"design": SimpleNamespace(Kst=torch.zeros(1, r, device=C.dev))} for l in C.layers}


def question_gate(C, it, M, text):
    ids = chat_ids(C.tok, text)
    st = cv1.pass1(C, ids, C.layers)
    P = cv1.read_prompt(C, it, M, ids, st)
    C.thr_now = {l: C.thr[it["id"]][l] for l in C.layers}
    gi = cv1.gate_info(C, P)
    return gi, P, st, ids


def lens_top(C, moments, topn=8):
    out = {}
    for name, label in LENS.items():
        if name not in moments[0]["shifts"]:
            continue
        for l in C.layers:
            X = torch.stack([m["shifts"][name][l] for m in moments])
            ll = cv1.lens_logits(C, X).mean(0)
            out[f"{label}@L{l}"] = [cv1.tokstr(C, i) for i in ll.topk(topn).indices.tolist()]
    return out


def prep(C, L, R, V, out):
    cfg = C.cfg
    pdir = out / "prep"
    pdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cv1.calibrate(C, L, pdir)
    checks = {"calibration_vs_core_v1": calib_check(C, cfg, L), "valence_state": V["_path"], "valence_sha": V["_sha"]}
    v = {l: V["valence"][l].float() for l in C.layers}
    Mo, signs, Ms, gates, lens, cosq = {}, {}, {}, {}, {}, {}
    C.thr, C.sd = {}, {}
    plain_rel = 0.0
    for k, it in enumerate(L["items"]):
        iid = it["id"]
        if not has_memory(it):
            C.thr[iid] = {l: float("inf") for l in C.layers}
            C.sd[iid] = {l: float("nan") for l in C.layers}
            Ms[iid] = dummy_memory(C)
            gates[iid], *_ = question_gate(C, it, Ms[iid], it["question"])
            continue
        ms = item_moments(C, it, R["items"][iid], L["placebo"])
        plain = {l: refs.mean_shift(ms, "without", l) for l in C.layers}
        sv = refs.read_signs(plain, v)
        signs[iid] = {"polarity": it["polarity"], "valence": sv}
        if "gen_cent" in ms[0]["shifts"]:
            refs.signed_shifts(ms, "gen_cent", "cs_sign", {l: sv[l]["sign"] for l in C.layers})
        Mo[iid] = ms
        M = memories(C, it, ms)
        C.thr[iid], C.sd[iid] = {}, {}
        for l in C.layers:
            C.thr[iid][l], C.sd[iid][l] = dk.calib(M[l]["design"], C.Kg[l], cfg["key"]["thr_q"])
        Ms[iid] = M
        gi, P, st, ids = question_gate(C, it, M, it["question"])
        gates[iid] = gi
        cats = C.catfn(ids)
        cosq[iid] = {}
        for l in C.layers:
            qk = C.KS[l][W256].keys(st[l], cats == 1)[-1]
            cosq[iid][l] = [float(m["key"][l].to(C.dev)[0] @ qk) for m in ms]
        lens[iid] = lens_top(C, ms)
        if (k + 1) % 20 == 0:
            log(f"prep: {k + 1}/{len(L['items'])} items ({(time.time() - t0) / 60:.1f} min)")
    mem_items = [it for it in L["items"] if has_memory(it)]
    unrel = L["unrelated"][:cfg.get("n_unrel", len(L["unrelated"]))]
    ufire = []
    for uk, text in enumerate(unrel):
        ids = chat_ids(C.tok, text)
        st = cv1.pass1(C, ids, C.layers)
        for it in mem_items:
            P = cv1.read_prompt(C, it, Ms[it["id"]], ids, st)
            C.thr_now = {l: C.thr[it["id"]][l] for l in C.layers}
            gi = cv1.gate_info(C, P)
            ufire.append({"uk": uk, "item": it["id"], "open_any": gi["open_any"], "fire": gi["fire"]})
    state = {"layers": C.layers, "tail_len": C.tail_len, "head": C.head.cpu(),
             "mu": {l: C.mu[l].cpu() for l in C.layers}, "W": {l: C.KS[l][W256].W.cpu() for l in C.layers},
             "thr": C.thr, "sd": C.sd, "med": C.med, "med_all": C.med_all, "valence": v, "valence_sha": V["_sha"],
             "moments": {iid: [{"j": m["j"], "n": m["n"], "key": {l: m["key"][l].cpu() for l in C.layers},
                                "shifts": {r: {l: s[l].cpu() for l in C.layers} for r, s in m["shifts"].items()}}
                               for m in ms] for iid, ms in Mo.items()},
             "signs": signs, "items": [it["id"] for it in L["items"]], "refs_sha": R["_sha"]}
    torch.save(state, pdir / "state.pt")
    json.dump({it["id"]: {k: it[k] for k in ("id", "group", "polarity", "topic", "row", "experience", "question",
                                             "prompts", "followups", "excluded", "failed")} for it in L["items"]},
              open(pdir / "items.json", "w"), indent=1)
    checks["counters"] = dict(C.chk)
    json.dump(checks, open(pdir / "checks.json", "w"), indent=2, default=str)
    diag = {"gates": gates, "unrel_fire": ufire, "cos_followup_question": cosq, "lens": lens,
            "signs": {i: {"polarity": s["polarity"], **{str(l): s["valence"][l] for l in C.layers}} for i, s in signs.items()}}
    json.dump(diag, open(pdir / "diagnostics.json", "w"), indent=1, default=str)
    meta = {"stage": "prep", "cfg": cfg, "env": cv1.env_meta(C), "layers": C.layers, "refs": R["_path"],
            "refs_sha": R["_sha"], "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(pdir / "config.json", "w"), indent=2, default=str)
    md = report_a(cfg, L, R, diag, checks, C.layers, out.parent)
    (out.parent / "report_A.md").write_text(md)
    (pdir / "report_A.md").write_text(md)
    log(f"prep done in {meta['minutes']:.1f} min -> {pdir}; report -> {out.parent / 'report_A.md'}")
    return state


def _rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _f(x, spec=".3f"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def report_a(cfg, L, R, D, checks, layers, run):
    md_table = drun.md_table
    w = []
    a = w.append
    items = L["items"]
    mem = [it for it in items if has_memory(it)]
    pol = ("avoid", "mixed", "like", "other")
    a("# Experiment 2, PrefEval benchmark, phase A: references, follow-ups, gate")
    a("")
    a(f"Model {cfg['model']} (fp32), layers {layers}. Dreamer Qwen3.5-9B (bf16, thinking off), dream_refs prompts v3 "
      "(final) + the follow-up prompt. Sample: PREREG addendum (stratified, topic x polarity, seed "
      f"{cfg['sample']['seed']}). Spot check: `spotcheck.md`. **Nothing here selects a generation cell**; the phase B "
      "grid is fixed in the addendum.")
    a("")
    V = json.load(open(run / "validate" / "validation.json")) if (run / "validate" / "validation.json").exists() else {}
    a("## Dreaming outcome (validate.py)")
    a("")
    a(md_table(["measure", "value"], [[k, v] for k, v in V.get("summary", {}).items()]))
    a("")
    df = V.get("dream_failed") or {}
    a(f"**Dreaming failed:** {len(df)} of {len(items)} items (each excluded from the variants that need the failed part; "
      "follow-ups failed = no memory at all).")
    a("")
    a(md_table(["polarity", "items", "opposite failed", "alternatives failed", "follow-ups failed", "usable for gen:opposite",
                "usable for gen:centroid / cs"],
               [[p, sum(it["polarity"] == p for it in items),
                 sum(bool(it["failed"]["opposite"]) for it in items if it["polarity"] == p),
                 sum(bool(it["failed"]["centroid"]) for it in items if it["polarity"] == p),
                 sum(bool(it["failed"]["followups"]) for it in items if it["polarity"] == p),
                 sum("gen:opposite" not in it["excluded"] for it in items if it["polarity"] == p),
                 sum("gen:centroid" not in it["excluded"] for it in items if it["polarity"] == p)] for p in pol]))
    a("")
    G = D["gates"]
    a("## The gate on the PrefEval test questions (failure 7: a memory that never opens does nothing)")
    a("")
    a("Gate = core_v1's: the question's pooled_w256 key matched against the item's follow-up keys, open where the match "
      "exceeds the item's q0.95 threshold on generic prompts. `fire` = the gate at the last prompt position (what the "
      "answer receives), mean over the 3 layers; `open any` = open at some position and layer. Items with memory only.")
    a("")
    rows = []
    for p in ("all",) + pol:
        its = [it for it in mem if p == "all" or it["polarity"] == p]
        rows.append([p, len(its), _f(_rate(G[it["id"]]["open_any"] for it in its)),
                     _f(_rate(G[it["id"]]["fire"] > 0 for it in its)), _f(_rate(G[it["id"]]["fire"] for it in its))]
                    + [_f(_rate(G[it["id"]]["layers"][str(l)]["g_last"] > 0 for it in its)) for l in layers])
    a(md_table(["subset", "items", "open any", "fire > 0", "mean fire"] + [f"open at last pos L{l}" for l in layers], rows))
    a("")
    U = D["unrel_fire"]
    a(f"False fire on core_v1's 50 unrelated prompts: open any {_f(_rate(x['open_any'] for x in U))}, fire > 0 "
      f"{_f(_rate(x['fire'] > 0 for x in U))} over {len(U)} (prompt, item) pairs.")
    a("")
    cq = D["cos_followup_question"]
    if cq:
        a("Key cosine between each follow-up and the test question (pooled_w256; the semantic side of the follow-up check, "
          "reported only), mean / max over items of the per-item max: " + "; ".join(
              f"L{l} {_f(_rate(max(cq[i][str(l)] if str(l) in cq[i] else cq[i][l]) for i in cq))} / "
              f"{_f(max(max(cq[i][str(l)] if str(l) in cq[i] else cq[i][l]) for i in cq))}" for l in layers))
        a("")
    S = D["signs"]
    a("## The valence sign (S3's reader, v_l from the 29-item run) vs the lexical polarity")
    a("")
    a("Expected: avoid -> -, like -> +. Descriptive; cs:concept_x_sign uses this sign per layer.")
    a("")
    a(md_table(["polarity", "items"] + [f"share read - at L{l}" for l in layers],
               [[p, sum(s["polarity"] == p for s in S.values())]
                + [_f(_rate(s[str(l)]["sign"] < 0 for s in S.values() if s["polarity"] == p)) for l in layers] for p in pol]))
    a("")
    a("## Logit lens (top tokens of the mean shift at L23), first 12 items with memory")
    a("")
    for it in mem[:12]:
        lz = D["lens"].get(it["id"], {})
        a(f"- **{it['id']}** ({it['polarity']}) \"{it['experience']}\": " + " | ".join(
            f"{k.split('@')[0]}: {' '.join(v)}" for k, v in lz.items() if k.endswith("@L23")))
    a("")
    a("## Checks")
    a("")
    a("```")
    a(json.dumps({k: v for k, v in checks.items() if k != "counters"}, indent=1, default=str)[:3000])
    a("```")
    return "\n".join(w) + "\n"


# =============================================================================================== gen


def conditions(cfg):
    cs = [core.Cond("nomem", "nomem"), core.Cond("ctx", "ctx")]
    for v, mem in VARIANTS.items():
        for a in cfg["doses"]:
            cs.append(core.Cond(f"{v}@{a:g}", "ref", mem, float(a), mech=True))
    return cs


def primary(cfg, conds):
    return [c for c in conds if c.kind in ("nomem", "ctx") or c.alpha == cfg["primary_alpha"]]


def item_conds(conds, it):
    return [c for c in conds if c.kind in ("nomem", "ctx") or (has_memory(it) and next(
        v for v, m in VARIANTS.items() if m == c.mem) not in it["excluded"])]


def load_state(C, out, L):
    state = torch.load(out / "prep" / "state.pt", map_location="cpu", weights_only=False)
    assert state["layers"] == C.layers
    cv1.restore(C, state)
    Ms = {}
    for it in L["items"]:
        Ms[it["id"]] = memories(C, it, state["moments"][it["id"]]) if has_memory(it) else dummy_memory(C)
    return state, Ms


def gen(C, L, out):
    cfg = C.cfg
    gdir = out / "gen"
    (gdir / "items").mkdir(parents=True, exist_ok=True)
    (gdir / "unrel").mkdir(exist_ok=True)
    t0 = time.time()
    state, Ms = load_state(C, out, L)
    conds = conditions(cfg)
    prim = primary(cfg, conds)
    C.n_rows = C.n_batches = 0
    C.rand_dir = {}
    sel = [it for it in L["items"] if not C.A.items or it["id"] in C.A.items]
    if C.A.shard:
        k, n = map(int, C.A.shard.split("/"))
        sel = sel[k::n]
    log(f"gen: {len(conds)} conditions, {len(sel)} items, sections {C.A.sections}")
    if "main" in C.A.sections:
        for i, it in enumerate(sel):
            f = gdir / "items" / f"{it['id']}.jsonl.gz"
            if f.exists():
                continue
            rows, gates = cv1.gen_main(C, it, Ms[it["id"]], item_conds(conds, it), it["prompts"])
            for g in gates:
                g["has_memory"] = has_memory(it)
            cv1.write_jsonl_gz(f, rows + gates)
            el = time.time() - t0
            log(f"gen: {it['id']} done: {len(rows)} rows | {C.n_rows} rows, {el / 60:.1f} min, "
                f"eta {el / (i + 1) * (len(sel) - i - 1) / 60:.1f} min")
    if "unrel" in C.A.sections:
        Lm = {**L, "items": [it for it in L["items"] if has_memory(it)]}
        conds_of = {it["id"]: [c for c in item_conds(prim, it) if c.kind not in ("nomem", "ctx")] for it in Lm["items"]}
        for uk, text in enumerate(L["unrelated"][:cfg.get("n_unrel", len(L["unrelated"]))]):
            f = gdir / "unrel" / f"u{uk:02d}.jsonl.gz"
            if f.exists():
                continue
            rows, gates, kl = cv1.gen_unrelated(C, Lm, Ms, conds_of, uk, text)
            cv1.write_jsonl_gz(f, rows + gates + kl)
            log(f"gen: unrelated {uk} done ({sum(g['open_any'] for g in gates)}/{len(gates)} gates open)")
    tag = "_".join(C.A.items) if C.A.items else (C.A.shard or "all")
    import zlib
    meta = {"stage": "gen", "cfg": cfg, "env": cv1.env_meta(C), "args": vars(C.A), "conditions": [c.cid for c in conds],
            "n_rows": C.n_rows, "n_batches": C.n_batches, "counters": dict(C.chk), "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(gdir / f"config_{'_'.join(C.A.sections)}_{zlib.crc32(tag.encode()):08x}.json", "w"), indent=2,
              default=str)
    log(f"gen done in {meta['minutes']:.1f} min ({C.n_rows} rows) -> {gdir}")


# ============================================================================================= score


def pe_prompts():
    return {k: (HERE / "upstream" / f).read_text() for k, f in PE_FILES.items()}


def pe_fill(T, rubric, preference, question, answer, restatement=None):
    """PrefEval's filling (llm_based_evaluation_errortypes.py): plain replacement of the named placeholders."""
    t = T[rubric]
    if rubric == "acknow":
        return t.replace("{end_generation}", answer).replace("{question}", question)
    if rubric in ("violate", "helpful"):
        return t.replace("{preference}", preference).replace("{question}", question).replace("{end_generation}", answer)
    return t.replace("{preference}", preference).replace("{assistant_restatement}", restatement or "")


def sys_chat_ids(tok, system, user):
    text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0]


def judge_texts(Cj, texts, max_new, bs, system=PE_SYSTEM):
    """Greedy outputs of the judge for unique texts, batched by exact token length (no padding), as cv1.judge."""
    enc = {t: sys_chat_ids(Cj.tok, system, t) for t in texts}
    by_len = defaultdict(list)
    for t in texts:
        by_len[len(enc[t])].append(t)
    raw = {}
    for n, ts in sorted(by_len.items()):
        for i in range(0, len(ts), bs):
            chunk = ts[i:i + bs]
            outs = core.generate_batch(Cj.model, Cj.body, Cj.lm_head, torch.stack([enc[t] for t in chunk]),
                                       [(True, None)] * len(chunk), max_new=max_new, eos=Cj.eos_list, vocab=Cj.vocab)
            for t, o in zip(chunk, outs):
                raw[t] = Cj.tok.decode(o["ids"], skip_special_tokens=True).strip()
    return raw


def pe_judge(Cj, L, rows, out, limit=None):
    """PrefEval's four rubrics with the original judge: acknowledgement, violation and helpfulness for every
    answer, then hallucination for the answers whose acknowledgement is yes (filled with the extracted
    restatement, as PrefEval)."""
    T = pe_prompts()
    cfg = Cj.cfg["pe_judge"]
    gen = [r for r in rows if r["sec"] == "gen"][:limit] if limit else [r for r in rows if r["sec"] == "gen"]
    task = {}
    for r in gen:
        it = L["by_id"][r["item"]]
        ans = r["answer"] if r["answer"].strip() else "(empty reply)"
        for rub in ("acknow", "violate", "helpful"):
            task[(r["key"], rub)] = pe_fill(T, rub, it["experience"], r["prompt"], ans)
    t0 = time.time()
    raw = judge_texts(Cj, sorted(set(task.values())), cfg["max_new"], cfg["batch"])
    res = defaultdict(dict)
    for (k, rub), t in task.items():
        if rub == "acknow":
            a, pref = bench.parse_acknowledge(raw[t])
            res[k].update(acknow=a, extract_pref=pref, raw_acknow=raw[t][:400])
        else:
            res[k].update({rub: bench.parse_answer_tag(raw[t]), f"raw_{rub}": raw[t][:400]})
    hall = {k: pe_fill(T, "hallucinate", L["by_id"][k.split("|")[0]]["experience"], None, None, v["extract_pref"])
            for k, v in res.items() if v["acknow"] is True}
    raw_h = judge_texts(Cj, sorted(set(hall.values())), cfg["max_new"], cfg["batch"]) if hall else {}
    for k, t in hall.items():
        res[k].update(hallucinate=bench.parse_answer_tag(raw_h[t]), raw_hallucinate=raw_h[t][:400])
    outrows = [{"key": k, **v} for k, v in res.items()]
    cv1.write_jsonl_gz(out / "pe_judge.jsonl.gz", outrows)
    n = max(len(outrows), 1)
    st = {"unique": len(raw) + len(raw_h), "rows": len(outrows), "minutes": (time.time() - t0) / 60,
          "parsed": {rub: sum(r.get(rub) is not None for r in outrows) / n for rub in ("acknow", "violate", "helpful")},
          "prompts_sha": {k: sha(HERE / "upstream" / f) for k, f in PE_FILES.items()}}
    log(f"PrefEval judge: {json.dumps(st)}")
    return st


def fake_pe_judge(out, rows):
    import zlib
    jr = []
    for r in rows:
        if r["sec"] != "gen":
            continue
        h = zlib.crc32(r["key"].encode())
        jr.append({"key": r["key"], "acknow": h % 3 == 0, "extract_pref": "FAKE", "violate": r["cond"] != "ctx" and h % 2 == 0,
                   "helpful": h % 5 != 0, "hallucinate": False if h % 3 == 0 else None})
    cv1.write_jsonl_gz(out / "pe_judge.jsonl.gz", jr)


def score(cfg, A, L, out):
    sdir = out / "score"
    sdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = cv1.gen_rows(out / "gen")
    tiny = bool(cfg.get("tiny") or A.random_model)
    n_unique, env = cv1._score_coherence(cfg, A, L, rows, sdir)
    cv1.release()
    meta = {"stage": "score", "cfg": cfg, "env": env, "coherence_unique": n_unique}
    if not A.no_judge:
        jm, jt = cv1.load(cfg, A, judge=True)
        Cj = SimpleNamespace(A=A, cfg=cfg, model=jm, tok=jt, dev=A.device, body=cv1.text_model(jm), lm_head=jm.lm_head)
        Cj.vocab = Cj.lm_head.out_features
        eos = {jt.eos_token_id} | {jt.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        Cj.eos_list = sorted(e for e in eos if isinstance(e, int) and e != jt.unk_token_id)
        lim = 16 if tiny else None
        meta["judge"] = cv1.judge(Cj, L, rows, sdir, lim)
        meta["pe_judge"] = pe_judge(Cj, L, rows, sdir, lim)
        del Cj, jm
        cv1.release()
    if tiny:
        for f in ("judge.jsonl.gz", "pe_judge.jsonl.gz"):
            if (sdir / f).exists():
                os.replace(sdir / f, sdir / f.replace(".jsonl.gz", "_plumbing.jsonl.gz"))
        drun.fake_judge(sdir, rows)
        fake_pe_judge(sdir, rows)
    meta["minutes"] = (time.time() - t0) / 60
    json.dump(meta, open(sdir / "config.json", "w"), indent=2, default=str)
    log(f"score done in {meta['minutes']:.1f} min -> {sdir}")


# ============================================================================================== main


def load_refs(A, out):
    p = Path(A.refs) if A.refs else out.parent / "validate" / "refs.json"
    R = json.load(open(p))
    if R.get("mock") and not (A.tiny or A.random_model):
        raise SystemExit(f"{p} holds MOCK references; only allowed with --tiny / --random-model")
    R["_path"], R["_sha"] = str(p), sha(p)
    return R


def load_valence(A, C):
    if A.valence:
        V = torch.load(A.valence, map_location="cpu", weights_only=False)
        V["_path"], V["_sha"] = A.valence, sha(A.valence)
        return V
    if not (A.tiny or A.random_model):
        raise SystemExit("--valence (the dream_refs v3 run's qwen35_2b/prep/state.pt) is required")
    g = torch.Generator().manual_seed(0)   # tiny: a random unit direction stands in
    d = C.d
    return {"valence": {l: torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0) for l in C.layers},
            "_path": "random (tiny)", "_sha": "random"}


def main(argv=None):
    A = parse_args(argv)
    cfg = load_cfg(A.config)
    if A.tiny:
        cfg = tiny_cfg(cfg)
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = [s for s in ("prep", "gen", "score", "analyze") if s in A.stage]
    if any(s in stages for s in ("prep", "gen", "score")):
        R = load_refs(A, out)
        L = load_items(cfg, R)
    with torch.inference_mode():
        if any(s in stages for s in ("prep", "gen")):
            model, tok = cv1.load(cfg, A)
            C = cv1.context(A, cfg, model, tok)
            if "prep" in stages:
                prep(C, L, R, load_valence(A, C), out)
            if "gen" in stages:
                gen(C, L, out)
            del C, model
            cv1.release()
        if "score" in stages:
            score(cfg, A, L, out)
    if "analyze" in stages:
        an = _load("seahorse_prefeval_analyze", HERE / "analyze.py")
        an.analyze_run(out, A.panel)


if __name__ == "__main__":
    main()
