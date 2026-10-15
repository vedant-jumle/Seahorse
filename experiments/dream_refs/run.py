#!/usr/bin/env python
"""Seahorse experiment 2 (dream_refs): model-generated references and concept x sign, on Qwen3.5-2B.
docs/dreaming/02-scaffold.md is the spec; PREREG.md fixes the claims S1-S5, the grid and the rules.

  python experiments/dream_refs/run.py --config experiments/dream_refs/config_qwen35_2b.yaml \
         --stage prep lens --out RUN/qwen35_2b            # phase A (after dream.py and validate.py)
  ... --stage gen [--items a b ...] [--sections main yn | unrel]          # phase B
  ... --stage score                                                       # coherence + the original judge
  ... --stage analyze [--panel RUN/panel]                                 # tables, S2 / S4 (+ S1 / S3 copied)
  ... --tiny --random-model qwen2                                         # CPU smoke on a tiny random model

Stages
  prep   reuses core_v1's 2B prep/state.pt (calibration, keys, thresholds, and the hand:opposite, hand:centroid,
         without and placebo shifts at the read layers 20/21/23; nothing of it is recomputed) and adds, pooled
         exactly as core_v1 (run.build_moments, the same code path):
           gen:opposite   with - generated opposite
           gen:centroid   with - mean of the 8 generated alternatives
           gen:para       mean over the 4 paraphrases of (with_para - without)          (lens / cosine only)
           null_swap_*    with - the partner item's hand opposite / mean of its 8 hand alternatives (S1 null)
           null_frame_*   with - the frame filled with unrelated values (1 / mean of 8)       (S1 null)
           v_l            the valence direction: mean over 120 like/hate pairs (40 topics x 3 frames, one
                          neutral follow-up) of (like - hate), normalised, per read layer
           signs          per item and layer: sign(<plain, v_l>) (valence reader) and sign(<plain, gen:opposite>)
           cs:concept_x_sign  s_l * gen:centroid (s_l = the valence reader's sign at layer l)
           cs:oracle          the true sign * gen:centroid
         Check (stops the job): one item per group recomputed with core_v1's code reproduces its stored shifts
         and keys (relative < 1e-3).
  lens   logit-lens lex_gain of every variant at the read layers; S1 cosines; S3 accuracies and margins; the
         valence direction's logit lens -> RUN/report_A.md (S1 and S3 by the pre-registered rules)
  gen    the FIXED grid (PREREG): nomem, ctx, and at alpha 0.5 / 1 / 2: hand:opposite, gen:opposite,
         gen:centroid, cs:concept_x_sign, cs:oracle; core_v1's prompts, random streams (seed tag "core_v1"),
         batching, two-pass gated reader and generator. Yes/no and the 50 unrelated prompts: nomem, ctx and the
         alpha = 1 cells only.
  score  core_v1's coherence (NLL) and judge (Qwen3.5-9B bf16, core_v1's rubric)
  analyze analyze.py (the S2 / S4 rules; S1 / S3 are decided in lens)
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import time
import zlib
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cv1 = _load("seahorse_core_v1_run", REPO / "experiments" / "core_v1" / "run.py")
core, itm, sh, dk, log = cv1.core, cv1.itm, cv1.sh, cv1.dk, cv1.log
refs = _load("seahorse_dream_refs_refs", HERE / "refs.py")
W256 = cv1.W256
PREF = ("leanings", "dislikes", "one_of_many")

# generated condition (cid prefix) -> stored memory name -> shift name in prep/state.pt
VARIANTS = {"hand:opposite": "design", "gen:opposite": "gen_opp", "gen:centroid": "gen_cent",
            "cs:concept_x_sign": "cs_sign", "cs:oracle": "cs_oracle"}
MEM_SHIFT = {"design": "hand_opp", "gen_opp": "gen_opp", "gen_cent": "gen_cent", "cs_sign": "cs_sign",
             "cs_oracle": "cs_oracle"}
# the logit-lens variants (shift names) and their report labels
LENS = {"hand_opp": "hand:opposite", "hand_cent": "hand:centroid", "without": "without", "gen_opp": "gen:opposite",
        "gen_cent": "gen:centroid", "gen_para": "gen:para", "cs_sign": "cs:concept_x_sign", "cs_oracle": "cs:oracle",
        "null_swap_opp": "null_swap:opposite", "null_swap_cent": "null_swap:centroid",
        "null_frame_opp": "null_frame:opposite", "null_frame_cent": "null_frame:centroid"}
S1_REFS = {"opposite": ("gen_opp", "hand_opp", "null_swap_opp", "null_frame_opp"),
           "centroid": ("gen_cent", "hand_cent", "null_swap_cent", "null_frame_cent")}
CV1_NAMES = {"opposite": "hand_opp", "centroid": "hand_cent", "plain": "without", "placebo": "placebo"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True, nargs="+", choices=["prep", "lens", "gen", "score", "analyze"])
    p.add_argument("--out", required=True, help="the model's run dir (RUN/qwen35_2b); stages write <out>/<stage>")
    p.add_argument("--refs", default=None, help="validate.py's refs.json (default <out>/../validate/refs.json)")
    p.add_argument("--items", nargs="+", default=None, help="gen: only these items (default all)")
    p.add_argument("--sections", nargs="+", default=["main", "yn", "unrel"], choices=["main", "yn", "unrel"])
    p.add_argument("--no-judge", action="store_true", help="score: coherence only")
    p.add_argument("--panel", default=None, help="analyze: the panel dir (gptoss.jsonl, gemma4.jsonl)")
    p.add_argument("--random-model", default=None, choices=["qwen2"])
    p.add_argument("--tiny", action="store_true", help="3 items (one per group), tiny settings")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def load_cfg(path):
    cfg = yaml.safe_load(open(path))
    for k in ("items", "bench", "generic"):
        cfg[k] = str((REPO / cfg[k]).resolve()) if not os.path.isabs(cfg[k]) else cfg[k]
    return cfg


def tiny_cfg(cfg):
    c = cv1.tiny_cfg(cfg)
    c.setdefault("control_alpha", 2)      # core_v1's own prep (tiny: it builds the stand-in state.pt) needs these
    c.setdefault("mech_doses", list(cfg["doses"]))
    return c


def conditions(cfg):
    """The fixed grid (PREREG): nomem, ctx, and every variant at every dose (kind "ref": h += alpha g M q
    through the item's own gate). Seed set 2 for cfg["seed2"]."""
    s2 = set(cfg["seed2"])
    cs = [core.Cond("nomem", "nomem", seed2="nomem" in s2), core.Cond("ctx", "ctx", seed2="ctx" in s2)]
    for v, mem in VARIANTS.items():
        for a in cfg["doses"]:
            cid = f"{v}@{a:g}"
            cs.append(core.Cond(cid, "ref", mem, float(a), seed2=cid in s2, mech=True))
    return cs


def primary(cfg, conds):
    """nomem, ctx and the alpha = primary_alpha cells (yes/no and unrelated prompts)."""
    return [c for c in conds if c.kind in ("nomem", "ctx") or c.alpha == cfg["primary_alpha"]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def pref_only(L):
    keep = [it for it in L["items"] if it["group"] in PREF]
    return {**L, "items": keep, "groups": {g: v for g, v in L["groups"].items() if g in PREF}}


# ============================================================================================== prep


def pseudo(it, experience=None, counter=None, centroid=None, group=None, followups=None, iid=None):
    """An item-shaped dict for core_v1's build_moments: `opposite` = with - counter, `centroid` = with - mean
    of `centroid`, `plain` = with - without (group "facts": only plain and placebo are computed)."""
    return {"id": iid or it["id"], "group": group or "leanings", "experience": experience or it["experience"],
            "counter": counter, "centroid": centroid, "followups": followups or it["followups"]}


def moments_at(C, it, placebo):
    """core_v1 build_moments (the same code that wrote core_v1's shifts), cut to the read layers, on CPU."""
    ms = cv1.build_moments(C, it, placebo)
    return [{"j": m["j"], "n": m["n"], "key": {l: m["key"][l].float().cpu() for l in C.layers},
             "shifts": {r: {l: s[l].float().cpu() for l in C.layers} for r, s in m["shifts"].items()}} for m in ms]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def cv1_state(C, L, out):
    """core_v1's 2B prep state (reused, not recomputed). With --tiny / --random-model, core_v1's own prep is
    run on the tiny subset first to make a stand-in with the same layout."""
    if C.cfg.get("tiny") or C.A.random_model:
        d = out / "cv1_prep_tiny"
        if not (d / "state.pt").exists():
            cv1.prep(C, L, d)
        path = d / "state.pt"
    else:
        path = Path(C.cfg["cv1_prep"]) / "state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["layers"] == C.layers, (state["layers"], C.layers)
    for it in L["items"]:
        assert it["id"] in state["moments"], f"{it['id']}: not in core_v1's state"
        assert len(state["moments"][it["id"]]) == len(it["followups"])
        for r in ("plain", "opposite", "centroid", "placebo"):
            assert r in state["moments"][it["id"]][0]["shifts"], f"{it['id']}: no {r} shift in core_v1's state"
    return state, str(path)


def hand_recompute(C, L, state0, ids):
    """One item per group: core_v1's build_moments run again with the same calibration must reproduce the
    stored plain / opposite / centroid / placebo shifts and keys (relative < 1e-3; as core_v1 vs xlayer_v1)."""
    res = {"items": ids, "max_rel": {}, "key_max_abs": 0.0}
    for iid in ids:
        ms = moments_at(C, L["by_id"][iid], L["placebo"])
        for j, m in enumerate(ms):
            st = state0["moments"][iid][j]
            for r in ("plain", "opposite", "centroid", "placebo"):
                for l in C.layers:
                    res["max_rel"][r] = max(res["max_rel"].get(r, 0.0), rel(m["shifts"][r][l], st["shifts"][r][l]))
            for l in C.layers:
                res["key_max_abs"] = max(res["key_max_abs"], (m["key"][l] - st["key"][l].float()).abs().max().item())
    res["ok"] = all(v < 1e-3 for v in res["max_rel"].values()) and res["key_max_abs"] < 1e-3
    log(f"hand recompute vs core_v1 state: {json.dumps(res)}")
    assert res["ok"], f"recomputed hand shifts do not reproduce core_v1's state.pt: {res}"
    return res


def valence(C, V, placebo):
    """v_l = normalise(mean over the like/hate pairs of (pooled like - pooled hate)): each sentence a user turn
    followed by the neutral follow-up, pooled over the follow-up exactly as core_v1 (shared suffix, entropy
    weights of the follow-up-only run, template tail excluded). Also: per-frame directions and their cosines."""
    diffs = {l: [] for l in C.layers}
    frames = []
    for p in V["pairs"]:
        ps = {"id": f"valence:{p['topic']}:{p['frame']}", "group": "leanings", "experience": p["like"],
              "counter": p["hate"], "centroid": [p["hate"]], "followups": [V["followup"]]}
        m = moments_at(C, ps, placebo)[0]
        for l in C.layers:
            diffs[l].append(m["shifts"]["opposite"][l])
        frames.append(p["frame"])
    v, info = {}, {"n_pairs": len(frames), "n_topics": len({p["topic"] for p in V["pairs"]}), "per_layer": {},
                   "followup": V["followup"]}
    for l in C.layers:
        D = torch.stack(diffs[l])
        mean = D.mean(0)
        v[l] = mean / mean.norm()
        cos_each = torch.nn.functional.cosine_similarity(D, v[l][None], dim=-1)
        fdir = {f: D[[i for i, x in enumerate(frames) if x == f]].mean(0) for f in sorted(set(frames))}
        fc = {f"{a}-{b}": refs.cosine(fdir[a], fdir[b]) for a in fdir for b in fdir if a < b}
        info["per_layer"][l] = {"mean_pair_norm": D.norm(dim=-1).mean().item(), "mean_norm": mean.norm().item(),
                                "pair_cos_with_v_mean": cos_each.mean().item(),
                                "pair_cos_with_v_min": cos_each.min().item(),
                                "share_pairs_positive": (cos_each > 0).float().mean().item(), "frame_dir_cos": fc}
    return v, info


def prep(C, L, R, out):
    cfg = C.cfg
    pdir = out / "prep"
    pdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state0, s0_path = cv1_state(C, L, out)
    cv1.restore(C, state0)
    checks = {"cv1_state": s0_path, "cv1_state_sha": sha(s0_path)}
    first = [L["groups"][g][0] for g in PREF if L["groups"].get(g)]
    checks["hand_recompute"] = hand_recompute(C, L, state0, first)
    Mo, plain_rel = {}, 0.0
    for k, it in enumerate(L["items"]):
        iid = it["id"]
        r = R["items"][iid]
        own = [{"j": m["j"], "n": m["n"], "key": {l: m["key"][l].float() for l in C.layers},
                "shifts": {CV1_NAMES[name]: {l: m["shifts"][name][l].float() for l in C.layers}
                           for name in CV1_NAMES}} for m in state0["moments"][iid]]
        # PREREG Amendment 2: a reference the dreaming failed to produce is not stored (the item is excluded from the
        # variants that need it). core_v1's build_moments needs both references, so a failed one is replaced by a
        # stand-in whose shift is discarded; with both failed only the plain shift is computed (its check stays).
        fail = r.get("failed") or {"opposite": None, "centroid": None}
        alts = [a["text"] for a in r["alternatives"]]
        ok_o, ok_c = not fail["opposite"], not fail["centroid"]
        if ok_o or ok_c:
            g = moments_at(C, pseudo(it, counter=r["opposite"]["text"] if ok_o else alts[0],
                                     centroid=alts if ok_c else [r["opposite"]["text"]]), L["placebo"])
        else:
            g = moments_at(C, pseudo(it, group="facts"), L["placebo"])
        ns = moments_at(C, pseudo(it, counter=r["null_swap"]["opposite"], centroid=r["null_swap"]["alternatives"]),
                        L["placebo"])
        nf_s = r["null_frame"]["sentences"]
        nf = moments_at(C, pseudo(it, counter=nf_s[0], centroid=nf_s), L["placebo"]) if nf_s else None
        paras = [moments_at(C, pseudo(it, experience=p["text"], group="facts"), L["placebo"]) for p in r["paraphrases"]]
        for j, m in enumerate(own):
            S = m["shifts"]
            if ok_o:
                S["gen_opp"] = g[j]["shifts"]["opposite"]
            if ok_c:
                S["gen_cent"] = g[j]["shifts"]["centroid"]
            S["null_swap_opp"], S["null_swap_cent"] = ns[j]["shifts"]["opposite"], ns[j]["shifts"]["centroid"]
            if nf:
                S["null_frame_opp"], S["null_frame_cent"] = nf[j]["shifts"]["opposite"], nf[j]["shifts"]["centroid"]
            if paras:
                S["gen_para"] = {l: sum(p[j]["shifts"]["plain"][l] for p in paras) / len(paras) for l in C.layers}
            for l in C.layers:  # the same with/without runs as core_v1's: the plain shift must come out identical
                plain_rel = max(plain_rel, rel(g[j]["shifts"]["plain"][l], S["without"][l]))
        Mo[iid] = own
        log(f"prep: {iid} shifts done ({k + 1}/{len(L['items'])}, {(time.time() - t0) / 60:.1f} min)")
    checks["plain_recompute_max_rel"] = plain_rel
    assert plain_rel < 1e-3, f"plain shift recomputed with the generated references differs: {plain_rel}"
    v, vinfo = valence(C, R["valence"], L["placebo"])
    log(f"valence: {json.dumps(vinfo)}")
    signs = {}
    for it in L["items"]:
        own = Mo[it["id"]]
        plain = {l: refs.mean_shift(own, "without", l) for l in C.layers}
        gopp = {l: refs.mean_shift(own, "gen_opp", l) for l in C.layers} if "gen_opp" in own[0]["shifts"] else None
        true = refs.TRUE_SIGN[it["group"]]
        sv = refs.read_signs(plain, v)
        so = refs.read_signs(plain, gopp) if "gen_opp" in own[0]["shifts"] else None
        signs[it["id"]] = {"group": it["group"], "true": true, "valence": sv, "opp": so}
        if "gen_cent" in own[0]["shifts"]:
            refs.signed_shifts(own, "gen_cent", "cs_sign", {l: sv[l]["sign"] for l in C.layers})
            refs.signed_shifts(own, "gen_cent", "cs_oracle", {l: true for l in C.layers})
    state = {k: state0[k] for k in ("layers", "tail_len", "head", "mu", "W", "thr", "sd", "med", "med_all")}
    state.update(moments=Mo, valence=v, valence_info=vinfo, signs=signs, items=[it["id"] for it in L["items"]],
                 refs_sha=R.get("_sha"), cv1_state_sha=checks["cv1_state_sha"])
    torch.save(state, pdir / "state.pt")
    items_out = {}
    for it in L["items"]:
        d = {k: it[k] for k in ("id", "group", "experience", "counter", "centroid", "lexicon", "followups", "related",
                                "paraphrase", "ambiguous", "prompts", "swap", "mech")}
        d["dream"] = {k: R["items"][it["id"]].get(k) for k in ("frame", "opposite", "alternatives", "paraphrases",
                                                               "null_frame", "null_swap", "flags", "failed",
                                                               "excluded_variants")}
        items_out[it["id"]] = d
    json.dump(items_out, open(pdir / "items.json", "w"), indent=1)
    checks["counters"] = dict(C.chk)
    json.dump(checks, open(pdir / "checks.json", "w"), indent=2, default=str)
    meta = {"stage": "prep", "cfg": cfg, "env": cv1.env_meta(C), "layers": C.layers, "n_blocks": C.n_blocks,
            "layer_types": {l: C.ltype[l] for l in range(C.n_blocks)}, "refs": R.get("_path"), "refs_sha": R.get("_sha"),
            "dreamer": R.get("dreamer"), "mock_refs": R.get("mock"), "items_sha": sha(cfg["items"]),
            "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(pdir / "config.json", "w"), indent=2, default=str)
    log(f"prep done in {meta['minutes']:.1f} min -> {pdir}")
    return state


# ============================================================================================== lens


def lens_rows(C, it, moments, topn=10):
    enc = lambda s: C.tok(s, add_special_tokens=False).input_ids
    cons, inc = sh.lexicon_token_ids(enc, it["lexicon"])
    rows = []
    for name, label in LENS.items():
        if name not in moments[0]["shifts"]:
            continue
        for l in C.layers:
            X = torch.stack([m["shifts"][name][l] for m in moments])
            ll = cv1.lens_logits(C, X).mean(0)
            rows.append({"item": it["id"], "group": it["group"], "variant": label, "shift": name, "layer": l,
                         "lex_gain": sh.lex_gain(ll, cons, inc),
                         "shift_norm": sum(m["shifts"][name][l].norm().item() for m in moments) / len(moments),
                         "top": [(cv1.tokstr(C, i), round(ll[i].item(), 2)) for i in ll.topk(topn).indices.tolist()]})
    return rows


def valence_lens(C, state, topn=20):
    out = {}
    for l in C.layers:
        x = (state["valence"][l] * state["med"][l])[None]
        ll = cv1.lens_logits(C, x)[0]
        out[l] = {"top": [(cv1.tokstr(C, i), round(ll[i].item(), 2)) for i in ll.topk(topn).indices.tolist()],
                  "bottom": [(cv1.tokstr(C, i), round(ll[i].item(), 2)) for i in (-ll).topk(topn).indices.tolist()]}
    return out


def s1_tables(state, L, layers):
    """Per reference type, item and layer: cos(gen, hand), cos(null_swap, hand), cos(null_frame, hand) of the
    moment-mean shifts, the item status and the rule (refs.s1_item / s1_rule). Also cos(gen:para, without)."""
    rows, verdicts, rules = [], {}, {}
    for ref, (gn, hn, sn, fn) in S1_REFS.items():
        status = {l: {} for l in layers}
        for it in L["items"]:
            ms = state["moments"][it["id"]]
            for l in layers:
                H = refs.mean_shift(ms, hn, l)
                if gn not in ms[0]["shifts"]:   # dreaming failed for this reference (Amendment 2): not counted
                    status[l][it["id"]] = "failed"
                    rows.append({"ref": ref, "item": it["id"], "group": it["group"], "layer": l, "cos_gen": None,
                                 "cos_null_swap": None, "cos_null_frame": None, "status": "failed"})
                    continue
                cg = refs.cosine(refs.mean_shift(ms, gn, l), H)
                cs_ = refs.cosine(refs.mean_shift(ms, sn, l), H)
                cf = refs.cosine(refs.mean_shift(ms, fn, l), H) if fn in ms[0]["shifts"] else None
                st = refs.s1_item(cg, cs_, cf)
                status[l][it["id"]] = st
                rows.append({"ref": ref, "item": it["id"], "group": it["group"], "layer": l, "cos_gen": cg,
                             "cos_null_swap": cs_, "cos_null_frame": cf, "status": st})
        rules[ref] = refs.s1_rule(status)
        verdicts[ref] = rules[ref]["verdict"]
    para = []
    for it in L["items"]:
        ms = state["moments"][it["id"]]
        if "gen_para" in ms[0]["shifts"]:
            for l in layers:
                para.append({"item": it["id"], "layer": l, "cos_para_without": refs.cosine(
                    refs.mean_shift(ms, "gen_para", l), refs.mean_shift(ms, "without", l))})
    return {"rows": rows, "rules": rules, "verdict": refs.s1_combine(verdicts), "para": para}


def s3_tables(state, L, layers, n_boot=2000, seed=7):
    from seahorse.stats import boot_ci
    boot = lambda nested: boot_ci(nested, n_boot, seed)
    S = state["signs"]
    signed = [it["id"] for it in L["items"] if it["group"] in refs.SIGNED_GROUPS]
    allp = [it["id"] for it in L["items"]]
    true = {i: S[i]["true"] for i in allp}
    out = {}
    for reader in ("valence", "opp"):
        have = [i for i in allp if S[i][reader] is not None]   # the opp reader needs a gen:opposite (Amendment 2)
        sg = [i for i in signed if i in have]
        pred = {l: {i: S[i][reader][l]["sign"] for i in have} for l in layers}
        marg = {l: {i: S[i][reader][l]["cos"] for i in have} for l in layers}
        out[reader] = {"signed": {l: refs.s3_layer(pred[l], marg[l], true, sg, boot) for l in layers},
                       "all29": {l: refs.s3_layer(pred[l], marg[l], true, have, boot) for l in layers},
                       "per_item": {i: {l: {"pred": pred[l][i], "cos": marg[l][i]} for l in layers} for i in have}}
        out[reader]["rule"] = refs.s3_rule(out[reader]["signed"])
    out["verdict"] = "holds" if out["valence"]["rule"]["holds"] else "fails"
    out["true"] = true
    return out


def lens_stage(C, L, out):
    cfg = C.cfg
    ldir = out / "lens"
    ldir.mkdir(parents=True, exist_ok=True)
    state = torch.load(out / "prep" / "state.pt", map_location="cpu", weights_only=False)
    rows = []
    for it in L["items"]:
        rows += lens_rows(C, it, state["moments"][it["id"]])
    cv1.write_jsonl_gz(ldir / "lens.jsonl.gz", rows)
    chk = {}
    cv1_lens = Path(cfg.get("cv1_prep") or "", "lens.jsonl.gz")
    if cfg.get("cv1_prep") and cv1_lens.exists() and not (cfg.get("tiny") or C.A.random_model):
        ref = {(r["item"], r["ref"], r["layer"]): r["lex_gain"] for r in cv1.read_jsonl_gz(cv1_lens) if "lex_gain" in r}
        back = {"hand_opp": "opposite", "hand_cent": "centroid", "without": "without"}
        diffs = [abs(r["lex_gain"] - ref[(r["item"], back[r["shift"]], r["layer"])]) for r in rows
                 if r["shift"] in back and (r["item"], back[r["shift"]], r["layer"]) in ref]
        chk["hand_lex_gain_vs_core_v1"] = {"n": len(diffs), "max_abs_diff": max(diffs) if diffs else None}
    vl = valence_lens(C, state)
    s1 = s1_tables(state, L, C.layers)
    s3 = s3_tables(state, L, C.layers, seed=cfg["seeds"]["bootstrap"])
    json.dump(s1, open(ldir / "s1.json", "w"), indent=1, default=str)
    json.dump(s3, open(ldir / "s3.json", "w"), indent=1, default=str)
    json.dump({"valence_lens": vl, "valence_info": state["valence_info"], "checks": chk}, open(ldir / "valence.json", "w"),
              indent=1, default=str)
    R = json.load(open(out.parent / "validate" / "validation.json")) if (out.parent / "validate" / "validation.json").exists() else {}
    prep_checks = json.load(open(out / "prep" / "checks.json"))
    md = report_a(cfg, L, state, rows, s1, s3, vl, chk, prep_checks, R, C.layers)
    (out.parent / "report_A.md").write_text(md)
    (ldir / "report_A.md").write_text(md)
    log(f"lens done: S1 {s1['verdict']}, S3 {s3['verdict']} -> {out.parent / 'report_A.md'}")


def _f(x, spec="+.2f"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def report_a(cfg, L, state, lens, s1, s3, vl, chk, prep_checks, V, layers):
    """Phase A report: S5, S1, S3, the valence direction and the logit lens. States explicitly that nothing
    here selects a generation cell."""
    w = []
    a = w.append
    a("# Experiment 2 (dream_refs), phase A: references, S1 and S3")
    a("")
    a(f"Model {cfg['model']} ({cfg['dtype']}), layers {layers}. Dreamer: Qwen3.5-9B (bf16, thinking off). "
      f"References: `validate/refs.json`. Spot check: `spotcheck.md`.")
    a("")
    a("**Nothing in this report selects a generation cell.** The phase B grid is fixed in PREREG.md (nomem, ctx, and "
      "hand:opposite / gen:opposite / gen:centroid / cs:concept_x_sign / cs:oracle at alpha 0.5, 1, 2). S1 and S3 are "
      "decided here by their pre-registered rules and reported; the logit lens is descriptive.")
    a("")
    if V:
        a("## Validation summary (validate.py)")
        a("")
        a(md_table(["measure", "value"], [[k, v] for k, v in V.get("summary", {}).items()]))
        a("")
        if V.get("valence_deviation"):
            a(f"Deviation: {V['valence_deviation']}.")
            a("")
        df = V.get("dream_failed") or {}
        a(f"**Dreaming failed** (PREREG Amendment 2; a reported outcome, the item is excluded from the variants that "
          f"need the failed reference): {len(df)} of {V.get('summary', {}).get('n_items', '?')} items.")
        a("")
        if df:
            a(md_table(["item", "opposite", "centroid", "excluded from"],
                       [[i, x["failed"]["opposite"] or "ok", x["failed"]["centroid"] or "ok",
                         ", ".join(x["excluded_variants"])] for i, x in df.items()]))
            a("")
    a("## S5: `without` needs no generation")
    a("")
    a("`without` is the same follow-up with the experience sentence left out: mechanical, no human and no model writes "
      "anything. It is reused from core_v1's state (the `plain` shift). **S5 holds by construction**, as it already did.")
    a("")
    a("## S1: generated references reproduce the hand-written shifts, beyond the nulls")
    a("")
    a("Rule (PREREG): per item and layer, pass iff cos(gen, hand) >= 0.8 and it exceeds BOTH nulls' cosine with hand by "
      ">= 0.2; an item is *indeterminate* at a layer if a null reaches 0.8 itself (or a null is missing). A reference "
      "type holds if, at every read layer, >= 80% of the determinate items pass (a layer with < 15 determinate items "
      "is indeterminate). S1 holds iff both the opposite and the centroid hold; indeterminate -> decided by S2.")
    a("")
    for ref, rule in s1["rules"].items():
        a(f"**{ref}: {rule['verdict'].upper()}**")
        a("")
        a(md_table(["layer", "items", "determinate", "pass", "share pass", "indeterminate", "dreaming failed",
                    "layer verdict"],
                   [[l, p["n_items"], p["n_determinate"], p["n_pass"], _f(p["share_pass"], ".2f"), p["n_indeterminate"],
                     p.get("n_dream_failed", 0), p["verdict"]] for l, p in rule["per_layer"].items()]))
        a("")
    a(f"**S1 overall: {s1['verdict'].upper()}**")
    a("")
    a("Per item (cosine with the hand-written shift of: gen / null_swap / null_frame; status, DF = dreaming failed, "
      "excluded: PREREG Amendment 2), opposite then centroid:")
    a("")
    by = {}
    for r in s1["rows"]:
        by.setdefault((r["ref"], r["item"]), {})[r["layer"]] = r
    hdr = ["item", "group"] + [f"L{l} gen / swap / frame" for l in layers] + ["status " + "/".join(f"L{l}" for l in layers)]
    for ref in S1_REFS:
        a(f"*{ref}*")
        a("")
        rows_ = []
        for it in L["items"]:
            d = by[(ref, it["id"])]
            rows_.append([it["id"], it["group"]] + [f"{_f(d[l]['cos_gen'])} / {_f(d[l]['cos_null_swap'])} / "
                                                    f"{_f(d[l]['cos_null_frame'])}" for l in layers]
                         + ["/".join("DF" if d[l]["status"] == "failed" else d[l]["status"][:4] for l in layers)])
        a(md_table(hdr, rows_))
        a("")
    if s1["para"]:
        pm = {l: sum(r["cos_para_without"] for r in s1["para"] if r["layer"] == l) /
              max(1, sum(r["layer"] == l for r in s1["para"])) for l in layers}
        a("gen:para (robustness only): mean cos(gen:para, without) per layer: "
          + ", ".join(f"L{l} {_f(v)}" for l, v in pm.items()))
        a("")
    a("## S3: a general valence direction reads the sign of each preference")
    a("")
    vi = state["valence_info"]
    a(f"v_l = mean over {vi['n_pairs']} like/hate pairs ({vi['n_topics']} dreamer-listed topics x 3 frames, follow-up "
      f"\"{vi.get('followup')}\") of (like - hate), normalised. Sign of an "
      "item = sign(<plain shift, v_l>), plain = with - without (core_v1's), mean over the 3 moments. True sign: leanings "
      "and one-of-many +, dislikes -. Rule: S3 holds if at some layer the accuracy over the 23 signed items "
      "(15 leanings + 8 dislikes) is >= 0.9 AND the bootstrap CI of the mean signed margin (true sign x cosine) is above 0.")
    a("")
    for reader, title in (("valence", "valence reader (decides S3)"), ("opp", "generated-opposite reader, sign(<plain, gen:opposite>) (reported)")):
        a(f"*{title}*")
        a("")
        a(md_table(["layer", "n", "accuracy", "mean signed margin [95% CI]", "wrong items", "holds",
                    "all 29: accuracy"],
                   [[l, r["n"], _f(r["accuracy"], ".3f"), f"{_f(r['margin'], '+.3f')} [{_f(r['margin_lo'], '+.3f')}, "
                     f"{_f(r['margin_hi'], '+.3f')}]", ", ".join(r["wrong"]) or "none", r["holds"],
                     _f(s3[reader]["all29"][l]["accuracy"], ".3f")] for l, r in s3[reader]["signed"].items()]))
        a("")
    a(f"**S3: {s3['verdict'].upper()}** (layers holding: {s3['valence']['rule']['layers_holding'] or 'none'}). "
      "The generated-opposite reader shares the `with` term with the plain shift, so it is expected to say + for "
      "almost every item; it is reported because the spec asks for it, not as a competitor.")
    a("")
    a("Per item, valence reader (cosine of the plain shift with v_l; sign):")
    a("")
    a(md_table(["item", "group", "true"] + [f"L{l}" for l in layers],
               [[i, state["signs"][i]["group"], f"{s3['true'][i]:+d}"]
                + [f"{_f(s3['valence']['per_item'][i][l]['cos'], '+.3f')} ({s3['valence']['per_item'][i][l]['pred']:+d})"
                   for l in layers] for i in s3["valence"]["per_item"]]))
    a("")
    a("Valence direction health per layer (pairs agreeing with v; cosines between the three frames' directions):")
    a("")
    a(md_table(["layer", "mean pair cos with v", "min", "share > 0", "frame-direction cosines"],
               [[l, _f(p["pair_cos_with_v_mean"], ".2f"), _f(p["pair_cos_with_v_min"], ".2f"),
                 _f(p["share_pairs_positive"], ".2f"), ", ".join(f"{k} {_f(v)}" for k, v in p["frame_dir_cos"].items())]
                for l, p in vi["per_layer"].items()]))
    a("")
    a("Logit lens of v_l (scaled to the layer's median residual norm). Expected: generic positive vs negative words, "
      "no topic.")
    a("")
    for l, d in vl.items():
        a(f"- L{l} top: " + ", ".join(f"{t} {v}" for t, v in d["top"]))
        a(f"- L{l} bottom: " + ", ".join(f"{t} {v}" for t, v in d["bottom"]))
    a("")
    a("## Logit lens: lex_gain per variant (consistent - inconsistent lexicon logits; mean over items)")
    a("")
    groups = [g for g in PREF if any(it["group"] == g for it in L["items"])]
    vars_ = [v for v in LENS.values() if any(r["variant"] == v for r in lens)]
    hdr = ["variant"] + [f"{g} L{l}" for g in groups for l in layers]
    tab = []
    for v in vars_:
        row = [v]
        for g in groups:
            for l in layers:
                xs = [r["lex_gain"] for r in lens if r["variant"] == v and r["group"] == g and r["layer"] == l
                      and not math.isnan(r["lex_gain"])]
                row.append(_f(sum(xs) / len(xs) if xs else None))
        tab.append(row)
    a(md_table(hdr, tab))
    a("")
    a("Expected from core_v1 / ref_v1: on dislikes, hand:centroid is wrong-signed (negative) and hand:opposite positive; "
      "the question here is whether gen:* and cs:* follow their hand counterparts. Descriptive only.")
    a("")
    a("## Checks")
    a("")
    a("```")
    a(json.dumps({"prep": {k: v for k, v in prep_checks.items() if k != "counters"}, "lens": chk}, indent=1, default=str)[:4000])
    a("```")
    return "\n".join(w) + "\n"


# =============================================================================================== gen


def memories(C, it, moments):
    """{layer: {memory name: diag_keys Mem}}: delta rule over the item's moments under core_v1's keys (every
    memory of an item shares them, so the gate is the same for every variant)."""
    lam = C.cfg["key"]["rls_lambda"]
    M = {}
    for l in C.layers:
        keys = torch.cat([m["key"][l] for m in moments]).to(C.dev)
        M[l] = {}
        for mem, name in MEM_SHIFT.items():
            if name not in moments[0]["shifts"]:     # dreaming failed (Amendment 2): no memory, no condition
                continue
            S = torch.stack([m["shifts"][name][l].to(C.dev).float() for m in moments])
            M[l][mem] = dk.build({it["id"]: core.memory_chunks(S, keys)}, [it["id"]], "delta", lam, C.dev, C.chk)
            assert torch.equal(M[l][mem].Kst, M[l]["design"].Kst)
    return M


def item_conds(conds, M):
    """The item's conditions: those whose memory exists (a variant whose reference the dreaming failed to
    produce is excluded for that item; PREREG Amendment 2)."""
    have = set(next(iter(M.values())))
    return [c for c in conds if c.kind in ("nomem", "ctx") or c.mem in have]


def load_prep(C, out, L):
    state = torch.load(out / "prep" / "state.pt", map_location="cpu", weights_only=False)
    assert state["layers"] == C.layers
    cv1.restore(C, state)
    return state, {it["id"]: memories(C, it, state["moments"][it["id"]]) for it in L["items"]}


def gen(C, L, out):
    cfg = C.cfg
    gdir = out / "gen"
    (gdir / "items").mkdir(parents=True, exist_ok=True)
    (gdir / "unrel").mkdir(exist_ok=True)
    t0 = time.time()
    state, Ms = load_prep(C, out, L)
    conds = conditions(cfg)
    prim = primary(cfg, conds)
    C.n_rows = C.n_batches = 0
    C.rand_dir = {}
    sel = [it for it in L["items"] if not C.A.items or it["id"] in C.A.items]
    log(f"gen: {len(conds)} conditions, {len(sel)} items, sections {C.A.sections}")
    if "main" in C.A.sections or "yn" in C.A.sections:
        for i, it in enumerate(sel):
            f = gdir / "items" / f"{it['id']}.jsonl.gz"
            if f.exists():
                log(f"gen: {it['id']} already done, skipped")
                continue
            prompts = it["prompts"][:cfg.get("n_prompts", len(it["prompts"]))]
            ci, pi_ = item_conds(conds, Ms[it["id"]]), item_conds(prim, Ms[it["id"]])
            rows, gates = cv1.gen_main(C, it, Ms[it["id"]], ci, prompts) if "main" in C.A.sections else ([], [])
            yn = cv1.gen_yn(C, it, Ms[it["id"]], pi_) if "yn" in C.A.sections else []
            cv1.write_jsonl_gz(f, rows + gates + yn)
            el = time.time() - t0
            log(f"gen: {it['id']} done: {len(rows)} rows | {C.n_rows} rows, {el / 60:.1f} min, "
                f"eta {el / (i + 1) * (len(sel) - i - 1) / 60:.1f} min")
    if "unrel" in C.A.sections:
        conds_of = {it["id"]: item_conds(prim, Ms[it["id"]]) for it in L["items"]}
        unrel = L["unrelated"][:cfg.get("n_unrel", len(L["unrelated"]))]
        for uk, text in enumerate(unrel):
            f = gdir / "unrel" / f"u{uk:02d}.jsonl.gz"
            if f.exists():
                continue
            rows, gates, kl = cv1.gen_unrelated(C, L, Ms, conds_of, uk, text)
            cv1.write_jsonl_gz(f, rows + gates + kl)
            log(f"gen: unrelated {uk} done ({sum(g['open_any'] for g in gates)}/{len(gates)} gates open)")
    tag = "_".join(C.A.items) if C.A.items else "all"
    meta = {"stage": "gen", "cfg": cfg, "env": cv1.env_meta(C), "args": vars(C.A), "conditions": [c.cid for c in conds],
            "primary": [c.cid for c in prim], "n_rows": C.n_rows, "n_batches": C.n_batches, "counters": dict(C.chk),
            "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(gdir / f"config_{'_'.join(C.A.sections)}_{zlib.crc32(tag.encode()):08x}.json", "w"), indent=2,
              default=str)
    log(f"gen done in {meta['minutes']:.1f} min ({C.n_rows} rows) -> {gdir}")


# ============================================================================================= score


def fake_judge(out, rows):
    """--tiny only: deterministic stand-in judge labels (hash of the key) so analyze runs end to end."""
    jr = []
    for r in rows:
        if r["sec"] != "gen":
            continue
        h = zlib.crc32(r["key"].encode())
        ctx = r["cond"] == "ctx"  # ctx leans toward, so the tiny items survive the ctx rule and every table fills
        jr.append({"key": r["key"], "raw": "FAKE", "direction": "toward" if ctx else ["toward", "away", "neutral"][h % 3],
                   "coherent": ctx or bool(h % 4), "self_claim": h % 7 == 0, "ok": True})
    cv1.write_jsonl_gz(out / "judge.jsonl.gz", jr)


def score(cfg, A, L, out):
    tiny = bool(cfg.get("tiny") or A.random_model)
    meta = cv1.score(cfg, A, L, out / "gen", out / "score", do_judge=not A.no_judge, limit=16 if tiny else None)
    if tiny:
        if (out / "score" / "judge.jsonl.gz").exists():
            os.replace(out / "score" / "judge.jsonl.gz", out / "score" / "judge_plumbing.jsonl.gz")
        fake_judge(out / "score", cv1.gen_rows(out / "gen"))
    return meta


# ============================================================================================== main


def load_refs(A, out):
    p = Path(A.refs) if A.refs else out.parent / "validate" / "refs.json"
    R = json.load(open(p))
    if R.get("mock") and not (A.tiny or A.random_model):
        raise SystemExit(f"{p} holds MOCK references; they are only allowed with --tiny / --random-model")
    R["_path"], R["_sha"] = str(p), sha(p)
    return R


def main(argv=None):
    A = parse_args(argv)
    cfg = load_cfg(A.config)
    if A.tiny:
        cfg = tiny_cfg(cfg)
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    L = itm.load_items(cfg["items"], cfg["bench"])
    L["all_ids"] = [it["id"] for it in L["items"]]
    res = itm.validate_items(L)
    assert not res["errors"], res["errors"]
    L = pref_only(L)
    if A.tiny:
        L = cv1.subset(L, [L["groups"][g][0] for g in PREF])
    stages = [s for s in ("prep", "lens", "gen", "score", "analyze") if s in A.stage]
    with torch.inference_mode():
        if any(s in stages for s in ("prep", "lens", "gen")):
            model, tok = cv1.load(cfg, A)
            C = cv1.context(A, cfg, model, tok)
            if "prep" in stages:
                prep(C, L, load_refs(A, out), out)
            if "lens" in stages:
                lens_stage(C, L, out)
            if "gen" in stages:
                gen(C, L, out)
            del C, model
            cv1.release()
        if "score" in stages:
            score(cfg, A, L, out)
    if "analyze" in stages:
        an = _load("seahorse_dream_refs_analyze", HERE / "analyze.py")
        an.analyze_run(out, cfg, A.panel)


if __name__ == "__main__":
    main()
