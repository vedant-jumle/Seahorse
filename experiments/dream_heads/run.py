#!/usr/bin/env python
"""dream_heads (PREREG.md): why KVzip-g's sparse slice keeps the licence. Teacher-forced only; reuses dream_compare
Stage A's store, selections (saved), per-head masks and judged labels.

Stages (dream_compare's jobs.py runs them through DC_RUNNER=../dream_heads/run):
  h-run      per model, resumable: d_comply (addendum 8's recipe, behaviour-checked pairs, no memory) -> comply.pt;
             anchors (nomem, R-full; the identity checks) per item -> anchors/<qid>.npz; folds, readout layer L*,
             licence / control heads and d_told per fold -> heads.json, dtold.pt; every condition (k8, k4 selections,
             the causal check) per item -> items/<qid>.json.gz
  h-analyze  analyze.analyze(<model dir>, <Stage A model dir>) -> report.md, verdict.json (CPU)
  h-summary  analyze.summary(<run root>) -> summary.md

  python experiments/dream_compare/jobs.py ... with SPECS="h-run|RUN/qwen25_15b|--stage-a SA/qwen25_15b --tiny"
"""

import argparse
import gzip
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DC = REPO / "experiments" / "dream_compare"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SA = _load("dream_heads_stage_a", DC / "stage_a.py")
AN = _load("dream_heads_analyze_r", HERE / "analyze.py")
M, HM, DB, S, B = SA.M, SA.HM, SA.DB, SA.S, SA.B
from seahorse.residual import _hidden  # noqa: E402

CATS = ("single-session-user", "knowledge-update")
N_ITEMS = 66
SELS = AN.SELS
BUDGETS = AN.BUDGETS
COMPLY_PAIRS = "data/bench_v1/comply_pairs.yaml"
COMPLY_CAP = 60
STAGE_A_TOL = 0.01


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------------------------- items


def load_items(tiny=False):
    """Stage A's primary items minus temporal-reasoning and single-session-assistant (66), with their fold."""
    its = [x for x in SA.load_items() if x["primary"] and x["category"] in CATS]
    cnt = Counter(x["category"] for x in its)
    assert len(its) == N_ITEMS and cnt == {"single-session-user": 64, "knowledge-update": 2}, cnt
    F = AN.folds([x["question_id"] for x in its])
    for x in its:
        x["fold"] = F[x["question_id"]]
    if tiny:
        its = [x for x in its if x["fold"] == "A"][:2] + [x for x in its if x["fold"] == "B"][:2]
    return its


def stage_a_record(stage_a_dir, qid):
    p = Path(stage_a_dir) / "tf" / "items" / f"{qid}.json.gz"
    assert p.exists(), f"no Stage A record {p}"
    return SA.read_json_gz(p)


# ----------------------------------------------------------------------------------------------- the probe


class Probe:
    """Hooks for one teacher-forced pass, reading row `row` (the last prompt token): the block outputs (residual)
    res[l] [d], the o_proj inputs z[l] [Hq * hd], the attention module outputs out[l] [d] and (att=True) the softmax
    attention weights att[l] [Hq, Tk] (eager attention)."""

    def __init__(self, C, row, att=True):
        self.C, self.row, self.want_att = C, row, att
        self.res, self.z, self.out, self.att, self.handles = {}, {}, {}, {}, []

    def __enter__(self):
        r = self.row
        for l, blk in enumerate(self.C.blocks):
            def post_block(m, a, o, l=l):
                self.res[l] = _hidden(o)[0, r].detach().float()

            def pre_o(m, args, l=l):
                self.z[l] = args[0][0, r].detach().float()

            def post_attn(m, a, o, l=l):
                self.out[l] = o[0][0, r].detach().float()
                if self.want_att:
                    w = o[1] if isinstance(o, tuple) and len(o) > 1 else None
                    assert w is not None, f"layer {l}: no attention weights (eager attention needed)"
                    self.att[l] = w[0, :, r].detach().float()
            self.handles += [blk.register_forward_hook(post_block),
                             blk.self_attn.o_proj.register_forward_pre_hook(pre_o),
                             blk.self_attn.register_forward_hook(post_attn)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

    def stack_res(self):
        return torch.stack([self.res[l].cpu() for l in range(len(self.C.blocks))])


@torch.no_grad()
def probe(C, store, cond, question, ans, prefix=None, keep=None):
    """Stage A's measure (answer score, first-token refusal mass) with the probe at the last prompt token."""
    q = SA.prompt_ids(C, store, cond, question)
    ids = torch.cat([q, ans])
    r = len(q) - 1
    with Probe(C, r) as pr, (HM.HeadMask(C, keep, store.n) if keep is not None else nullcontext()):
        h, _ = M.forward(C, ids, prefix, attn=SA.NEED_OUTPUT_ATTN)
    lg = M.logits(C, h[r:len(ids) - 1])
    lp = torch.log_softmax(lg, -1)
    s, m = S.answer_logprob(lp, 1, ans)
    p = {k: math.exp(float(lp[0][v])) for k, v in C.ref_ids.items()}
    return {"answer_score": s, "answer_mean": m, "refusal": p["I"] + p["Sorry"] + p["As"], "p_first": p}, lg, pr


# ----------------------------------------------------------------------------------------------- heads, attribution


def head_info(C):
    n_q = int(C.model.config.num_attention_heads)
    n_kv = C.layout[0].n_kv
    assert all(l.n_kv == n_kv for l in C.layout), "one KV head count per model expected (Gemma is not run)"
    return SimpleNamespace(L=len(C.blocks), n_q=n_q, n_kv=n_kv, group=n_q // n_kv, hd=C.layout[0].head_dim)


def unit(D):
    D = torch.as_tensor(D).float()
    return D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def attr_maps(C, Dhat):
    """Per layer l: U_l = W_O_l^T @ Dhat^T  [Hq * hd, n_dirs] (head h's write projected on each unit direction)."""
    out = []
    for l, blk in enumerate(C.blocks):
        W = blk.self_attn.o_proj.weight
        out.append((W.float().T @ Dhat.to(W.device).T).contiguous())
    return out


@torch.no_grad()
def attributions(pr, U, hi):
    """[L, Hq, n_dirs]: head (l, h)'s direct write z_{l,h} . U_l[h] onto each direction."""
    A = []
    for l in range(hi.L):
        z = pr.z[l].to(U[l].device).view(hi.n_q, hi.hd)
        A.append(torch.einsum("hd,hdk->hk", z, U[l].view(hi.n_q, hi.hd, -1)).cpu())
    return torch.stack(A).numpy()


def o_bias(blk):
    b = getattr(blk.self_attn.o_proj, "bias", None)
    return None if b is None else b.detach().float()


# ----------------------------------------------------------------------------------------------- d_comply


@torch.no_grad()
def comply_direction(C, out, tiny=False):
    """dream_bind addendum 8 (stage_directions8, part 3): greedy answers without memory; a pair behaves if the personal
    question is refused and the general one answered (COMPLY_REFUSAL_RE); d_comply = mean(general) - mean(personal) of
    the block outputs at the last prompt token over the behaving pairs (fewer than 2: all pairs, recorded)."""
    import yaml
    f = Path(out) / "comply.pt"
    if f.exists():
        return torch.load(f)
    a8 = (C.cfg or {}).get("add8", {}) if isinstance(C.cfg, dict) else {}
    path = REPO / a8.get("comply_pairs", COMPLY_PAIRS)
    cap = int(a8.get("comply_cap", COMPLY_CAP))
    P = yaml.safe_load(open(path))["pairs"][:4 if tiny else None]
    rows, keep = [], []
    for p in P:
        ans = {}
        for side in ("personal", "general"):
            o = M.gen_batch(C, DB.chat_ids(C.tok, p[side]), [(True, None)], max_new=cap)[0]
            ans[side] = C.tok.decode(o["ids"], skip_special_tokens=True).strip()
        refused = bool(DB.COMPLY_REFUSAL_RE.search(ans["personal"]))
        answered = not DB.COMPLY_REFUSAL_RE.search(ans["general"])
        rows.append({"pair": p["id"], "answer_personal": ans["personal"], "answer_general": ans["general"],
                     "refused": refused, "answered": answered, "behaves": refused and answered})
        if refused and answered:
            keep.append(p)
    fallback = len(keep) < 2
    use = P if fallback else keep

    def last(text):
        ids = DB.chat_ids(C.tok, text)
        with Probe(C, len(ids) - 1, att=False) as pr:
            M.forward(C, ids)
        return pr.stack_res()
    Cg = torch.stack([last(p["general"]) for p in use])
    Cp = torch.stack([last(p["personal"]) for p in use])
    d, sep = DB.diff_means(Cg, Cp)
    D = {"d": d, "sep": sep, "n_pairs": len(P), "n_kept": len(keep), "fallback": fallback,
         "kept": [p["id"] for p in keep], "dropped": [p["id"] for p in P if p not in keep], "rows": rows}
    torch.save(D, f)
    json.dump({k: v for k, v in D.items() if k != "d"}, open(Path(out) / "comply.json", "w"), indent=1)
    log(f"d_comply: {len(keep)} of {len(P)} pairs behave{' (fallback: all used)' if fallback else ''}")
    return D


# ----------------------------------------------------------------------------------------------- anchors


def check_store(store, info, qid):
    assert (store.nh, store.ns, list(store.content)) == (info["n_head"], info["n_seg"], info["content"]), \
        f"{qid}: head / turn / content positions differ from Stage A's"


def stage_a_rows(rec):
    return {(r["cond"], r["budget"]): r for r in rec["rows"]}


def stage_a_score(rows, cond, budget):
    r = rows.get((cond, budget))
    if r is None:
        return None
    if cond == "random":
        return r["per_seed_answer_score"][0]
    return r["answer_score"]


@torch.no_grad()
def anchor_item(C, it, sa, Dhat, U, hi, tol):
    store = SA.harvest(C, it["stored_turn"], stats=False)
    check_store(store, sa["info"], it["question_id"])
    ans = DB.text_ids(C.tok, it["answer"])
    q = it["question"]
    mn, lg_n, pn = probe(C, store, "nomem", q, ans)
    mf, lg_f, pf = probe(C, store, "R-full", q, ans, store.P)
    _, lg_t = SA.measure(C, store, "T-full", q, ans)
    d_id = DB.maxdiff(lg_f, lg_t)
    assert d_id <= tol, f"{it['question_id']}: R-full TF != T-full ({d_id})"
    _, lg_k, _ = probe(C, store, "R-full", q, ans, store.P, HM.keep_all(C, store.n))
    d_keep = DB.maxdiff(lg_k, lg_f)
    assert d_keep <= tol, f"{it['question_id']}: per-head keep-all != R-full ({d_keep})"
    rows = stage_a_rows(sa)
    dsa = {}
    for c, m in (("nomem", mn), ("R-full", mf)):
        ref = stage_a_score(rows, c, None)
        dsa[c] = abs(m["answer_score"] - ref)
        assert dsa[c] <= STAGE_A_TOL, f"{it['question_id']} {c}: answer score {m['answer_score']} != Stage A {ref}"
    res_n, res_f = pn.stack_res(), pf.stack_res()
    out = {"res_nomem": res_n.numpy(), "res_full": res_f.numpy(),
           "proj_nomem": (res_n * Dhat).sum(-1).numpy(), "proj_full": (res_f * Dhat).sum(-1).numpy(),
           "A_nomem": attributions(pn, U, hi), "A_full": attributions(pf, U, hi),
           "meta": json.dumps({"identity_rfull_tfull": d_id, "identity_keep_all": d_keep, "stage_a_diff": dsa,
                               "refusal": {"nomem": mn["refusal"], "R-full": mf["refusal"]}})}
    return out


# ----------------------------------------------------------------------------------------------- selection of heads


@torch.no_grad()
def select(C, items, adir, Dc, hi, out):
    """Folds -> per evaluation fold f (chosen on the other fold): L*, licence heads, control heads, d_told."""
    A = {it["question_id"]: np.load(adir / f"{it['question_id']}.npz") for it in items}
    cands = AN.candidates(hi.L)
    H = {"model": C.key, "n_layers": hi.L, "n_q": hi.n_q, "n_kv": hi.n_kv, "group": hi.group, "hd": hi.hd,
         "candidates": cands, "folds": {}, "items": [it["question_id"] for it in items]}
    pf = {q: a["proj_full"] for q, a in A.items()}
    pn = {q: a["proj_nomem"] for q, a in A.items()}
    H["separation"] = AN.separation(np.stack(list(pf.values())), np.stack(list(pn.values()))).tolist()
    dtold = {}
    for f in ("A", "B"):
        train = [it["question_id"] for it in items if it["fold"] == AN.other(f)]
        ls, sep = AN.readout_layer(np.stack([pf[q] for q in train]), np.stack([pn[q] for q in train]), cands)
        dA = np.mean([A[q]["A_full"][:, :, ls] - A[q]["A_nomem"][:, :, ls] for q in train], 0)
        lic = AN.licence_heads(dA, ls)
        ctrl, ov = AN.control_heads(lic, hi.n_q, hi.group)
        dtold[f] = torch.tensor(np.mean([A[q]["res_full"] - A[q]["res_nomem"] for q in train], 0))
        H["folds"][f] = {"lstar": ls, "chosen_on": AN.other(f), "n_train": len(train), "sep_train": sep.tolist(),
                         "licence": lic, "control": ctrl, "control_overlap": ov,
                         "dA_train_licence": [float(dA[l, h]) for l, h in lic],
                         "cos_told_comply": torch.nn.functional.cosine_similarity(dtold[f], Dc, dim=-1).tolist()}
        log(f"fold {f} (chosen on {AN.other(f)}): L* {ls}; licence {lic}; control {ctrl} (overlap {ov})")
    metas = [json.loads(str(a["meta"])) for a in A.values()]
    H["checks_summary"] = {"identity_rfull_tfull": max(m["identity_rfull_tfull"] for m in metas),
                           "identity_keep_all": max(m["identity_keep_all"] for m in metas),
                           "stage_a_diff_anchors": max(max(m["stage_a_diff"].values()) for m in metas)}
    torch.save(dtold, out / "dtold.pt")
    return H


# ----------------------------------------------------------------------------------------------- all conditions


def keep_from_mask(C, store, K):
    """per-head keep (head always) from a content kept mask K [L, n_kv, n_c]."""
    picks = {l: [[store.content[j] for j in np.nonzero(K[l, g])[0]] for g in range(K.shape[1])]
             for l in range(K.shape[0])}
    return SA.keep_of(C, store, picks)


@torch.no_grad()
def cond_item(C, it, sa, Dhat, U, hi, H, dtold, tol):
    qid, q, f = it["question_id"], it["question"], it["fold"]
    Fh = H["folds"][f]
    ls = Fh["lstar"]
    lic, ctrl = [tuple(x) for x in Fh["licence"]], [tuple(x) for x in Fh["control"]]
    That = unit(dtold[f])
    store = SA.harvest(C, it["stored_turn"], stats=False)
    info = sa["info"]
    check_store(store, info, qid)
    ans = DB.text_ids(C.tok, it["answer"])
    content, nh, n = store.content, store.nh, store.n
    sels = sa["sels"]
    rows = stage_a_rows(sa)
    L, nkv, g = hi.L, hi.n_kv, hi.group
    pieces = [C.tok.decode([int(t)]) for t in store.Seg]
    classes = AN.token_classes(pieces, content, q, info.get("gold_span"))
    # R-full first: its attention to the content positions is the coverage reference
    plan = [("nomem", "nomem", None, None, None, np.zeros((L, nkv, len(content)), bool)),
            ("R-full", "R-full", None, store.P, None, np.ones((L, nkv, len(content)), bool))]
    for b in BUDGETS:
        for s in SELS:
            key = f"{s}|{b}"
            if key not in sels:
                continue
            sel = sels[key][0] if s == "random" else sels[key]
            K = AN.kept_mask(sel, content, L, nkv)
            if s in AN.PER_HEAD:
                picks = SA.picks_from_json(sel)
                B.check_picks(picks, content, B.budget_entries(B.budget_k(b, len(content)), [nkv] * L), [nkv] * L)
                plan.append((key, s, b, store.P, SA.keep_of(C, store, picks), K))
            else:
                S.check_kept(S.keep_indices(nh, sel), nh, B.budget_k(b, len(content)), content)
                plan.append((key, s, b, SA.splice(store, sel), None, K))
    # causal (k8): masks on whole KV groups
    gl, gc = AN.groups_of(lic, g), AN.groups_of(ctrl, g)
    sh = sels["KVzip-sh|k8"]
    Ksh = AN.kept_mask(sh, content, L, nkv)
    for name, gs in (("lic", gl), ("ctrl", gc)):
        K = np.ones((L, nkv, len(content)), bool)
        keep = HM.keep_all(C, n)
        for l, groups in gs.items():
            for gg in groups:
                K[l, gg] = False
                keep[l][gg, nh:n] = False
        plan.append((f"R-full-{name}", "R-full", None, store.P, keep, K))
        K2 = Ksh.copy()
        keep2 = HM.keep_shared(C, n, list(range(nh)) + [nh + c for c in sh])
        for l, groups in gs.items():
            for gg in groups:
                K2[l, gg] = True
                keep2[l][gg, :] = True
        plan.append((f"KVzip-sh+{name}", "KVzip-sh", "k8", store.P, keep2, K2))
    conds, a_c, lg_ref = {}, None, {}
    for name, c, b, prefix, keep, K in plan:
        pc = c if c in ("nomem",) else ("R-full" if prefix is not None and name.startswith("R-full") else "slice")
        m, lg, pr = probe(C, store, pc, q, ans, prefix, keep)
        res = pr.stack_res()
        A = attributions(pr, U, hi)[:, :, ls]                                          # [L, Hq] on d_comply @ L*
        if prefix is None:
            mass = np.zeros((L, hi.n_q))
        else:
            mem = torch.nonzero(prefix.pos >= nh).flatten()
            mass = np.stack([pr.att[l][:, mem.to(pr.att[l].device)].sum(-1).cpu().numpy() for l in range(L)])
        if name == "R-full":
            cols = torch.tensor([nh + x for x in content])
            a_c = np.stack([pr.att[l][:, cols.to(pr.att[l].device)].cpu().numpy() for l in range(L)])
            lg_ref["R-full"] = lg
            A_full_all = attributions(pr, U, hi)
        if name == "nomem":
            A_nomem_all = attributions(pr, U, hi)
        if name == "KVzip-sh|k8":
            lg_ref["KVzip-sh"] = lg
        cov = AN.coverage(a_c, K, g) if a_c is not None else np.zeros((L, hi.n_q))
        ref = stage_a_score(rows, c, b) if name in ("nomem", "R-full") or "|" in name else None
        dsa = abs(m["answer_score"] - ref) if ref is not None else None
        if dsa is not None:
            assert dsa <= STAGE_A_TOL, f"{qid} {name}: answer score {m['answer_score']} != Stage A {ref} ({dsa})"
        rec = {"answer_score": m["answer_score"], "answer_mean": m["answer_mean"], "refusal": m["refusal"],
               "p_first": m["p_first"], "proj_comply": (res * Dhat).sum(-1).tolist(),
               "proj_told": (res * That).sum(-1).tolist(), "attr_lstar": np.round(A, 6).tolist(),
               "mass": np.round(mass, 6).tolist(), "coverage": np.round(cov, 6).tolist(), "stage_a_diff": dsa}
        if "|" in name:
            rec["sel"] = sels[name][0] if c == "random" else sels[name]
        conds[name] = rec
        del pr
    # check: the KVzip-sh mask form == the splice
    keep_sh = HM.keep_shared(C, n, list(range(nh)) + [nh + c for c in sh])
    _, lg_m, _ = probe(C, store, "slice", q, ans, store.P, keep_sh)
    d_sh = DB.maxdiff(lg_m, lg_ref["KVzip-sh"])
    assert d_sh <= tol, f"{qid}: the KVzip-sh mask form != the splice ({d_sh})"
    dA = (A_full_all - A_nomem_all)[: ls + 1, :, ls]
    info_out = {"question_id": qid, "question": q, "answer": it["answer"], "category": it["category"],
                "verbatim": it["verbatim"], "fold": f, "lstar": ls, "n_head": nh, "n_seg": store.ns,
                "content": list(content), "gold_span": info.get("gold_span"), "classes": classes,
                "check_sh_mask_vs_splice": d_sh,
                "max_stage_a_diff": max(v["stage_a_diff"] for v in conds.values() if v["stage_a_diff"] is not None)}
    return {"info": info_out, "conds": conds, "anchors": {"dA_lstar": np.round(dA, 6).tolist()}}


# ----------------------------------------------------------------------------------------------- stages


def setup(A):
    if A.random_model:
        from transformers import AutoTokenizer
        C = M.tiny_context(A.model, AutoTokenizer.from_pretrained(M.SPECS[A.model]["hf"]), attn="eager")
        C.multi_device = False
        return C
    return SA.setup(SimpleNamespace(random_model=False, model=A.model, device=A.device, gb_per_gpu=A.gb_per_gpu))


def stage_run(A, C=None, items=None, tol=None):
    out = Path(A.out)
    (out / "anchors").mkdir(parents=True, exist_ok=True)
    (out / "items").mkdir(parents=True, exist_ok=True)
    items = items if items is not None else load_items(A.tiny)
    C = C or setup(A)
    tol = tol if tol is not None else M.tf_tol(C)
    hi = head_info(C)
    t0 = time.time()
    with torch.no_grad():
        D = comply_direction(C, out, A.tiny)
        Dc = D["d"].float()
        Dhat = unit(Dc)
        U = attr_maps(C, Dhat)
        for i, it in enumerate(items):
            f = out / "anchors" / f"{it['question_id']}.npz"
            if f.exists():
                continue
            r = anchor_item(C, it, stage_a_record(A.stage_a, it["question_id"]), Dhat, U, hi, tol)
            tmp = out / "anchors" / f"{it['question_id']}.tmp.npz"
            np.savez_compressed(tmp, **r)
            tmp.replace(f)
            log(f"anchors {i + 1}/{len(items)} {it['question_id']}: {json.loads(r['meta'])}")
        hp = out / "heads.json"
        if hp.exists():
            H = json.load(open(hp))
            assert H["items"] == [it["question_id"] for it in items], "heads.json was built on other items"
            dtold = torch.load(out / "dtold.pt")
        else:
            H = select(C, items, out / "anchors", Dc, hi, out)
            H["comply"] = {k: D[k] for k in ("n_pairs", "n_kept", "fallback", "kept", "dropped")}
            H["comply_sep"] = D["sep"]
            json.dump(H, open(hp, "w"), indent=1)
            dtold = torch.load(out / "dtold.pt")
        for i, it in enumerate(items):
            f = out / "items" / f"{it['question_id']}.json.gz"
            if f.exists():
                continue
            t1 = time.time()
            r = cond_item(C, it, stage_a_record(A.stage_a, it["question_id"]), Dhat, U, hi, H, dtold, tol)
            SA.write_json_gz(f, r)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            el = (time.time() - t0) / 60
            log(f"items {i + 1}/{len(items)} {it['question_id']} (fold {it['fold']}): {len(r['conds'])} conditions, "
                f"max Stage A diff {r['info']['max_stage_a_diff']:.1e}, sh-mask {r['info']['check_sh_mask_vs_splice']:.1e}"
                f" | {time.time() - t1:.1f}s, {el:.1f} min")
    H = json.load(open(out / "heads.json"))
    recs = AN.load_items(out)
    H["checks_summary"]["stage_a_diff_all"] = max(r["info"]["max_stage_a_diff"] for r in recs.values())
    H["checks_summary"]["sh_mask_vs_splice"] = max(r["info"]["check_sh_mask_vs_splice"] for r in recs.values())
    H["peak_gb"] = M.peak_gb()
    H["minutes"] = (time.time() - t0) / 60
    H["env"] = DB.env_meta()
    json.dump(H, open(out / "heads.json", "w"), indent=1, default=str)
    log(f"h-run done: {len(recs)} items; checks {H['checks_summary']}; peak GPU {H['peak_gb']}")
    if A.max_reserved_gb is not None and H["peak_gb"]:
        worst = max(v[1] for v in H["peak_gb"].values())
        assert worst <= A.max_reserved_gb, f"fit check: peak reserved {worst} GB > {A.max_reserved_gb} GB"
    return H


STAGES = ("h-run", "h-analyze", "h-summary")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--model", default="qwen25_15b", choices=sorted(M.SPECS))
    ap.add_argument("--out", required=True, help="the model dir (h-summary: the run root)")
    ap.add_argument("--stage-a", default=None, help="Stage A's model dir (tf/items, gen/items, judge/)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help="4 items (2 per fold), 4 comply pairs (smoke)")
    ap.add_argument("--random-model", action="store_true")
    ap.add_argument("--gb-per-gpu", type=int, default=66)
    ap.add_argument("--max-reserved-gb", type=float, default=None)
    ap.add_argument("--models", default=None)
    A = ap.parse_args(argv)
    if A.stage == "h-run":
        assert A.stage_a, "--stage-a is required"
        stage_run(A)
    elif A.stage == "h-analyze":
        assert A.stage_a, "--stage-a is required"
        V = AN.analyze(A.out, A.stage_a, A.model)
        log(f"h-analyze: verdict {V['verdict']}")
    elif A.stage == "h-summary":
        log(f"h-summary: {AN.summary(A.out, A.models.split(',') if A.models else None)}")


if __name__ == "__main__":
    main()
