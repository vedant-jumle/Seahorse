#!/usr/bin/env python
"""dream_heads: the numpy side (PREREG.md). Folds, the readout layer, licence / control heads, coverage, token-class
shares, the statistics, the rules H-A / H-B / H-C, the per-model report (report.md, verdict.json) and the cross-model
summary (summary.md, summary.json). No torch: reads <model dir>/items/*.json.gz, <model dir>/heads.json and Stage A's
judged labels (dream_compare analyze_a's loaders) from the Stage A model dir.
"""

import gzip
import importlib.util
import json
import math
import random
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DC = REPO / "experiments" / "dream_compare"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


N_BOOT, SEED = 2000, 7
N_LIC = 8
FOLD_SEED = 7
CTRL_SEED = 0
HA_THR = -0.4
HB_THR = 0.5
COV_MIN = 1e-6
MODELS = ("qwen25_15b", "qwen25_7b", "llama32_3b")
LABELS = {"qwen25_15b": "Qwen2.5-1.5B", "qwen25_7b": "Qwen2.5-7B", "llama32_3b": "Llama-3.2-3B"}
SELS = ("KVzip-g", "KVzip-ph", "KVzip-sh", "S-loo", "S-dream", "random", "gold@k")
PER_HEAD = frozenset({"KVzip-g", "KVzip-ph"})
SHARED_CMP = ("S-loo", "S-dream", "KVzip-sh")
BUDGETS = ("k8", "k4")
CAUSAL = ("R-full-lic", "R-full-ctrl", "KVzip-sh+lic", "KVzip-sh+ctrl")
STOP = frozenset("the and for are but not you all any can had her was one our out day get has him his how man new now "
                 "old see two way who boy did its let put say she too use what with this that have from they will "
                 "would there their been when which were your about into than them then some could other more these "
                 "also just like only over such very much many most where after before while because does doing my "
                 "me i a an to of in on at is it be do so if or as by up we he".split())


# ----------------------------------------------------------------------------------------------- folds, heads


def folds(ids):
    """{id: "A" | "B"}: the ids sorted, random.Random(7).shuffle, the first half (ceil) = A."""
    s = sorted(ids)
    random.Random(FOLD_SEED).shuffle(s)
    h = (len(s) + 1) // 2
    return {i: ("A" if j < h else "B") for j, i in enumerate(s)}


def other(f):
    return "B" if f == "A" else "A"


def candidates(n_layers):
    """Readout-layer candidates: floor(0.2 n) <= l < floor(0.8 n) (PREREG)."""
    return list(range(int(math.floor(0.2 * n_layers)), int(math.floor(0.8 * n_layers))))


def separation(p_full, p_nomem):
    """Per layer: paired standardised difference mean(full - nomem) / sd(full - nomem); p_* [items, L]."""
    d = np.asarray(p_full, float) - np.asarray(p_nomem, float)
    sd = d.std(0, ddof=1) if d.shape[0] > 1 else np.ones(d.shape[1])
    return d.mean(0) / np.maximum(sd, 1e-12)


def readout_layer(p_full, p_nomem, cands):
    sep = separation(p_full, p_nomem)
    return int(max(cands, key=lambda l: sep[l])), sep


def licence_heads(dA, lstar, n=N_LIC):
    """Top n (layer, query head) with layer <= lstar by the mean attribution difference dA [L, Hq] (fold mean)."""
    m = np.asarray(dA, float)[: lstar + 1]
    flat = [(float(m[l, h]), l, h) for l in range(m.shape[0]) for h in range(m.shape[1])]
    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(l, h) for _, l, h in flat[:n]]


def control_heads(lic, n_q, group, seed=CTRL_SEED):
    """Per licence head, a random query head of the same layer that is not a licence head and whose KV group holds no
    licence head of that layer; else any non-licence head of that layer (counted in 'overlap')."""
    rng = np.random.default_rng(seed)
    lic_set = set(map(tuple, lic))
    out, overlap = [], 0
    for l, h in lic:
        lic_groups = {hh // group for (ll, hh) in lic_set if ll == l}
        taken = {hh for (ll, hh) in out if ll == l}
        c = [x for x in range(n_q) if (l, x) not in lic_set and x // group not in lic_groups and x not in taken]
        if not c:
            overlap += 1
            c = ([x for x in range(n_q) if (l, x) not in lic_set and x not in taken]
                 or [x for x in range(n_q) if (l, x) not in lic_set] or list(range(n_q)))   # last: tiny test models
        out.append((l, int(rng.choice(c))))
    return out, overlap


def groups_of(heads, group):
    """{layer: sorted KV groups} of (layer, query head) pairs."""
    g = {}
    for l, h in heads:
        g.setdefault(int(l), set()).add(int(h) // group)
    return {l: sorted(v) for l, v in g.items()}


# ----------------------------------------------------------------------------------------------- coverage, classes


def kept_mask(sel, content, n_layers, n_kv):
    """bool [L, n_kv, n_content]: which content entries a selection keeps per (layer, KV head). sel: shared = [content
    positions]; per-head = {layer: [[content positions] per KV head]} (JSON keys may be strings)."""
    idx = {c: j for j, c in enumerate(content)}
    K = np.zeros((n_layers, n_kv, len(content)), bool)
    if isinstance(sel, dict):
        for l, hs in sel.items():
            for g, xs in enumerate(hs):
                for c in xs:
                    K[int(l), g, idx[int(c)]] = True
    else:
        for c in sel:
            K[:, :, idx[int(c)]] = True
    return K


def coverage(a, K, group):
    """a [L, Hq, n_c] (R-full attention from the last prompt token to the content positions), K [L, n_kv, n_c] ->
    [L, Hq]: the share of each query head's content attention on entries its KV group keeps (NaN if < COV_MIN)."""
    a = np.asarray(a, float)
    Kq = np.repeat(K, group, axis=1)
    tot = a.sum(-1)
    kept = (a * Kq).sum(-1)
    return np.where(tot >= COV_MIN, kept / np.maximum(tot, 1e-30), np.nan)


def head_mean(M, heads):
    v = [M[l][h] for l, h in heads]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def all_mean(M):
    M = np.asarray(M, float)
    return float(np.nanmean(M)) if np.isfinite(M).any() else float("nan")


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def token_classes(pieces, content, question, gold_span=None):
    """Class per content position: gold (in the gold span) > qword (inside a turn word of >= 3 letters, not a stop word,
    that also occurs in the question) > punct (no letter or digit) > other. pieces: the decoded token strings of the
    turn's segment."""
    qw = {w for w in _words(question) if len(w) >= 3 and w not in STOP}
    qw |= {w[:-1] for w in qw if w.endswith("s")}
    words, cur = [], []
    for j, c in enumerate(content):
        p = pieces[c]
        if not any(ch.isalnum() for ch in p):
            if cur:
                words.append(cur)
                cur = []
            words.append([j])
            continue
        if cur and p[:1].isspace():
            words.append(cur)
            cur = []
        cur.append(j)
    if cur:
        words.append(cur)
    cls = ["other"] * len(content)
    for w in words:
        text = "".join(pieces[content[j]] for j in w).strip().lower()
        tw = "".join(ch for ch in text if ch.isalnum())
        if not any(ch.isalnum() for ch in text):
            for j in w:
                cls[j] = "punct"
        elif len(tw) >= 3 and tw not in STOP and (tw in qw or (tw.endswith("s") and tw[:-1] in qw)):
            for j in w:
                cls[j] = "qword"
    gs = set(gold_span or [])
    for j, c in enumerate(content):
        if c in gs:
            cls[j] = "gold"
    return cls


CLASSES = ("gold", "qword", "punct", "other")


def class_shares(K, cls):
    """Share of kept entries (all layers and KV heads) per class; K [L, n_kv, n_c]."""
    w = K.sum((0, 1)).astype(float)
    tot = w.sum()
    return {c: float(sum(w[j] for j in range(len(cls)) if cls[j] == c) / tot) if tot else float("nan")
            for c in CLASSES}


# ----------------------------------------------------------------------------------------------- statistics


def ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def boot_mean(vals, n_boot=N_BOOT, seed=SEED):
    """Mean of the finite per-item values with a percentile item-bootstrap CI."""
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(v))}


def share_fold(dA_items, lic):
    """share = sum over lic of m(h) / sum over all heads of max(m(h), 0); dA_items [n, L', Hq] (layers <= L*)."""
    m = np.asarray(dA_items, float).mean(0)
    den = np.clip(m, 0, None).sum()
    num = sum(m[l, h] for l, h in lic)
    return float(num / den) if den > 0 else float("nan")


def hb_stat(dA, lic, n_boot=N_BOOT, seed=SEED):
    """H-B: dA {fold: [n_f, L*_f + 1, Hq]} (each fold's items at its own L*), lic {fold: heads chosen on the other fold}.
    Pooled share = mean of the two folds' shares; items resampled within each fold."""
    sh = {f: share_fold(dA[f], lic[f]) for f in dA}
    pooled = float(np.mean(list(sh.values())))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        s = []
        for f in sorted(dA):
            X = np.asarray(dA[f])
            s.append(share_fold(X[rng.integers(0, len(X), len(X))], lic[f]))
        reps.append(np.mean(s))
    reps = np.array([r for r in reps if np.isfinite(r)])
    lo, hi = np.quantile(reps, [0.025, 0.975]) if len(reps) else (float("nan"), float("nan"))
    return {"pooled": pooled, "lo": float(lo), "hi": float(hi), "folds": sh,
            "n": {f: int(len(dA[f])) for f in dA}}


def holds_lo(c, thr=0.0):
    return bool(c and c.get("n") and c.get("lo") is not None and np.isfinite(c["lo"]) and c["lo"] > thr)


# ----------------------------------------------------------------------------------------------- loading


def read_gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_items(d):
    return {r["info"]["question_id"]: r for r in (read_gz(f) for f in sorted((Path(d) / "items").glob("*.json.gz")))}


def judged(stage_a_dir, recs):
    """Stage A's per item x condition judged correctness (three-judge majority, mean over the 5 answers):
    {cname: {item: value}} with cname as in Stage A ("nomem", "R-full", "<sel>@<budget>")."""
    AA = _load("dream_heads_analyze_a", DC / "analyze_a.py")
    G, _ = AA.load_gen(stage_a_dir)
    if not G:
        return {}
    votes = AA.load_votes(stage_a_dir)
    items = {q: {"answer": r["info"]["answer"], "verbatim": r["info"]["verbatim"]} for q, r in recs.items()}
    N, _ = AA.gen_values(G, votes, items)
    return {c: dict(v) for c, v in N["maj"].items()}


def cname(c):
    """our condition key -> Stage A's generation cname."""
    if "|" in c:
        s, b = c.split("|")
        return f"{s}@{b}"
    return c


# ----------------------------------------------------------------------------------------------- analysis


def fmt(c, nd=2):
    if not c or c.get("mean") is None:
        return "n/a"
    return f"{c['mean']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}] (n {c['n']})"


def num(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def k8_conds(rec):
    out = ["nomem", "R-full"]
    out += [f"{s}|k8" for s in SELS if f"{s}|k8" in rec["conds"]]
    return out


def analyze(d, stage_a_dir, model=None):
    d = Path(d)
    H = json.load(open(d / "heads.json"))
    recs = load_items(d)
    assert recs, f"no items in {d}"
    model = model or H.get("model")
    J = judged(stage_a_dir, recs)
    fold = {q: r["info"]["fold"] for q, r in recs.items()}
    Lstar = {f: H["folds"][f]["lstar"] for f in H["folds"]}
    lic = {f: [tuple(x) for x in H["folds"][f]["licence"]] for f in H["folds"]}
    ctrl = {f: [tuple(x) for x in H["folds"][f]["control"]] for f in H["folds"]}
    get = lambda q, c, m: recs[q]["conds"].get(c, {}).get(m)
    jv = lambda q, c: J.get(cname(c), {}).get(q)

    # ---- per item x condition summary values
    def lic_cov(q, c):
        cv = get(q, c, "coverage")
        return head_mean(cv, lic[fold[q]]) if cv is not None else float("nan")

    def all_cov(q, c):
        cv = get(q, c, "coverage")
        return all_mean(cv) if cv is not None else float("nan")

    def proj(q, c, key="proj_comply"):
        p = get(q, c, key)
        return float(p[Lstar[fold[q]]]) if p is not None else float("nan")

    def lic_mass(q, c):
        m = get(q, c, "mass")
        return head_mean(m, lic[fold[q]]) if m is not None else float("nan")

    def lic_attr(q, c):
        a = get(q, c, "attr_lstar")
        return float(sum(a[l][h] for l, h in lic[fold[q]])) if a is not None else float("nan")

    conds_all = ["nomem", "R-full"] + [f"{s}|{b}" for b in BUDGETS for s in SELS] + list(CAUSAL)
    table = {}
    for c in conds_all:
        qs = [q for q in recs if c in recs[q]["conds"]]
        if not qs:
            continue
        mean = lambda f: float(np.nanmean([f(q) for q in qs])) if qs else float("nan")
        table[c] = {"n": len(qs), "refusal": mean(lambda q: get(q, c, "refusal")),
                    "answer_score": mean(lambda q: get(q, c, "answer_score")),
                    "judged": (float(np.mean([jv(q, c) for q in qs if jv(q, c) is not None]))
                               if any(jv(q, c) is not None for q in qs) else float("nan")),
                    "proj": mean(lambda q: proj(q, c)), "proj_told": mean(lambda q: proj(q, c, "proj_told")),
                    "lic_cov": mean(lambda q: lic_cov(q, c)), "all_cov": mean(lambda q: all_cov(q, c)),
                    "mass_all": mean(lambda q: all_mean(get(q, c, "mass"))), "mass_lic": mean(lambda q: lic_mass(q, c)),
                    "lic_attr": mean(lambda q: lic_attr(q, c))}

    # ---- H-A
    def ha(measure_fn, key="proj_comply"):
        rho = []
        for q in recs:
            cs = k8_conds(recs[q])
            rho.append(spearman([proj(q, c, key) for c in cs], [measure_fn(q, c) for c in cs]))
        return boot_mean(rho)
    ref = lambda q, c: get(q, c, "refusal")
    jud = lambda q, c: (jv(q, c) if jv(q, c) is not None else float("nan"))
    HA = {"comply_refusal": ha(ref), "comply_judged": ha(jud), "told_refusal": ha(ref, "proj_told"),
          "told_judged": ha(jud, "proj_told")}
    c0 = HA["comply_refusal"]
    ha_holds = bool(c0["n"] and c0["mean"] <= HA_THR and c0["hi"] < 0)

    # ---- H-B
    dA = {}
    for f in ("A", "B"):
        X = [np.asarray(recs[q]["anchors"]["dA_lstar"], float) for q in recs if fold[q] == f]
        if X:
            dA[f] = np.stack(X)
    HB = hb_stat(dA, lic) if len(dA) == 2 else {"pooled": None, "lo": None, "hi": None, "folds": {}, "n": {}}
    hb_holds = bool(HB["lo"] is not None and np.isfinite(HB["lo"]) and HB["lo"] > HB_THR)

    # ---- H-C (i)
    HCi = {}
    for s in SHARED_CMP:
        HCi[s] = boot_mean([lic_cov(q, "KVzip-g|k8") - lic_cov(q, f"{s}|k8") for q in recs
                            if "KVzip-g|k8" in recs[q]["conds"] and f"{s}|k8" in recs[q]["conds"]])
    hci_holds = all(holds_lo(HCi[s]) for s in SHARED_CMP)
    # also all-head coverage (descriptive)
    HCi_all = {s: boot_mean([all_cov(q, "KVzip-g|k8") - all_cov(q, f"{s}|k8") for q in recs
                             if f"{s}|k8" in recs[q]["conds"]]) for s in SHARED_CMP}

    # ---- H-C (ii)
    dref, dcor, rl_ref, ra_ref, rl_cor, ra_cor = [], [], [], [], [], []
    for q in recs:
        cs = [f"{s}|k8" for s in SELS if f"{s}|k8" in recs[q]["conds"]]
        lc, ac = [lic_cov(q, c) for c in cs], [all_cov(q, c) for c in cs]
        rr, jj = [ref(q, c) for c in cs], [jud(q, c) for c in cs]
        a, b = spearman(lc, rr), spearman(ac, rr)
        rl_ref.append(a), ra_ref.append(b)
        dref.append(b - a if np.isfinite(a) and np.isfinite(b) else float("nan"))
        a, b = spearman(lc, jj), spearman(ac, jj)
        rl_cor.append(a), ra_cor.append(b)
        dcor.append(a - b if np.isfinite(a) and np.isfinite(b) else float("nan"))
    HCii = {"d_ref": boot_mean(dref), "d_cor": boot_mean(dcor), "rho_lic_ref": boot_mean(rl_ref),
            "rho_all_ref": boot_mean(ra_ref), "rho_lic_cor": boot_mean(rl_cor), "rho_all_cor": boot_mean(ra_cor)}
    hcii_holds = holds_lo(HCii["d_ref"]) and holds_lo(HCii["d_cor"])

    # ---- H-C (iii) / descriptive heads
    jac = (len(set(lic["A"]) & set(lic["B"])) / len(set(lic["A"]) | set(lic["B"]))) if lic.get("A") and lic.get("B") else None
    # ---- causal
    CA = {}
    pairs = [("R-full-lic", "R-full"), ("R-full-ctrl", "R-full"), ("KVzip-sh+lic", "KVzip-sh|k8"),
             ("KVzip-sh+ctrl", "KVzip-sh|k8"), ("R-full-lic", "R-full-ctrl"), ("KVzip-sh+lic", "KVzip-sh+ctrl")]
    for a, b in pairs:
        qs = [q for q in recs if a in recs[q]["conds"] and b in recs[q]["conds"]]
        CA[f"{a} - {b}"] = {m: boot_mean([(proj(q, a) - proj(q, b)) if m == "proj" else (get(q, a, m) - get(q, b, m))
                                          for q in qs]) for m in ("refusal", "answer_score", "proj")}

    # ---- descriptive: budgets per layer, token classes
    desc = {}
    nL, nkv = H["n_layers"], H["n_kv"]
    for s in SELS:
        key = f"{s}|k8"
        thirds, cv, cls_sh, lic_share = [], [], {c: [] for c in CLASSES}, []
        base = {c: [] for c in CLASSES}
        for q, r in recs.items():
            if key not in r["conds"]:
                continue
            K = kept_mask(r["conds"][key]["sel"], r["info"]["content"], nL, nkv)
            per = K.sum(-1).astype(float)                    # [L, n_kv]
            tot = per.sum()
            t = [per[: nL // 3].sum() / tot, per[nL // 3: 2 * nL // 3].sum() / tot, per[2 * nL // 3:].sum() / tot]
            thirds.append(t)
            cv.append(float(per.std() / per.mean()) if per.mean() > 0 else float("nan"))
            gl = groups_of(lic[fold[q]], H["group"])
            n_g = sum(len(v) for v in gl.values())
            lic_share.append(float(sum(per[l, g] for l, gs in gl.items() for g in gs) / tot / (n_g / (nL * nkv))))
            sh = class_shares(K, r["info"]["classes"])
            for c in CLASSES:
                cls_sh[c].append(sh[c])
                base[c].append(np.mean([x == c for x in r["info"]["classes"]]))
        if thirds:
            desc[s] = {"thirds": np.mean(thirds, 0).tolist(), "cv": float(np.nanmean(cv)),
                       "lic_group_budget_ratio": float(np.mean(lic_share)),
                       "classes": {c: float(np.nanmean(v)) for c, v in cls_sh.items()},
                       "base": {c: float(np.mean(v)) for c, v in base.items()}}

    # ---- per-layer curves
    curve = {}
    for c in ("nomem", "R-full", "KVzip-g|k8", "KVzip-sh|k8", "S-loo|k8", "S-dream|k8"):
        P = [recs[q]["conds"][c]["proj_comply"] for q in recs if c in recs[q]["conds"]]
        if P:
            curve[c] = np.mean(np.asarray(P, float), 0).tolist()

    verdict = ("no single gate" if not ha_holds else
               "head-gated licence" if (hb_holds and hci_holds and hcii_holds) else "in between")
    V = {"model": model, "n_items": len(recs), "folds": {f: sum(1 for q in fold if fold[q] == f) for f in ("A", "B")},
         "lstar": Lstar, "licence": {f: lic[f] for f in lic}, "control": {f: ctrl[f] for f in ctrl},
         "control_overlap": {f: H["folds"][f]["control_overlap"] for f in H["folds"]}, "licence_jaccard": jac,
         "comply": H["comply"], "H-A": HA, "H-A holds": ha_holds, "H-B": HB, "H-B holds": hb_holds,
         "H-C(i)": HCi, "H-C(i) all-head": HCi_all, "H-C(i) holds": hci_holds, "H-C(ii)": HCii,
         "H-C(ii) holds": hcii_holds, "verdict": verdict, "descriptive_only": not ha_holds, "table": table,
         "causal": CA, "budgets": desc, "curve": curve, "separation": H.get("separation"),
         "checks": H.get("checks_summary"), "judged_available": bool(J)}
    json.dump(V, open(d / "verdict.json", "w"), indent=1, default=float)
    (d / "report.md").write_text(report(V, H))
    return V


def report(V, H):
    lab = LABELS.get(V["model"], V["model"])
    T = V["table"]
    L = [f"# dream_heads: {lab}", "",
         f"*Teacher-forced; judged correctness = Stage A's three-judge majority (reused). {V['n_items']} items "
         f"(fold A {V['folds']['A']}, fold B {V['folds']['B']}). PREREG: `experiments/dream_heads/PREREG.md`.*", "",
         f"## Verdict: **{V['verdict']}**", "",
         f"- H-A (d_comply tracks refusal): **{'holds' if V['H-A holds'] else 'fails'}**, mean within-item Spearman "
         f"{fmt(V['H-A']['comply_refusal'])} (rule: <= {HA_THR}, CI upper < 0)",
         f"- H-B (8 heads carry >= half): **{'holds' if V['H-B holds'] else 'fails'}**, pooled share "
         f"{num(V['H-B']['pooled'])} [{num(V['H-B']['lo'])}, {num(V['H-B']['hi'])}]; folds "
         + ", ".join(f"{f} {num(v)}" for f, v in V['H-B']['folds'].items()),
         f"- H-C(i) (KVzip-g covers licence heads more): **{'holds' if V['H-C(i) holds'] else 'fails'}**",
         f"- H-C(ii) (licence coverage beats all-head coverage): **{'holds' if V['H-C(ii) holds'] else 'fails'}**",
         ]
    if V["descriptive_only"]:
        L += ["", "H-A failed, so H-B and H-C are descriptive only (PREREG)."]
    c = V["comply"]
    L += ["", "## Setup facts", "",
          f"- d_comply: {c['n_kept']} of {c['n_pairs']} pairs behave (personal refused, general answered)"
          f"{' (fallback: all pairs used)' if c['fallback'] else ''}.",
          f"- Readout layer L* (chosen on the other fold): fold A items {V['lstar']['A']}, fold B items {V['lstar']['B']}.",
          f"- Licence heads (layer.head) used on fold A: {', '.join(f'{l}.{h}' for l, h in V['licence']['A'])}; on fold B: "
          f"{', '.join(f'{l}.{h}' for l, h in V['licence']['B'])}. Jaccard between the folds' sets: {num(V['licence_jaccard'])}.",
          f"- Control heads (layer-matched): fold A {', '.join(f'{l}.{h}' for l, h in V['control']['A'])}; fold B "
          f"{', '.join(f'{l}.{h}' for l, h in V['control']['B'])}; fell back to a licence head's KV group: "
          f"{V['control_overlap']}.",
          f"- Checks (max over items): {json.dumps(V['checks'])}", "",
          "## Per condition (means over items)", "",
          "| condition | n | refusal | answer score | judged (Stage A) | proj d_comply @L* | proj d_told @L* | "
          "licence coverage | all-head coverage | memory mass (all) | memory mass (licence) | licence attribution |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in T.items():
        L.append(f"| {k} | {r['n']} | {num(r['refusal'])} | {num(r['answer_score'])} | {num(r['judged'])} | "
                 f"{num(r['proj'])} | {num(r['proj_told'])} | {num(r['lic_cov'])} | {num(r['all_cov'])} | "
                 f"{num(r['mass_all'], 3)} | {num(r['mass_lic'], 3)} | {num(r['lic_attr'])} |")
    HA = V["H-A"]
    L += ["", "## H-A: does the projection track the decision? (within-item Spearman over nomem, R-full, the k8 "
          "selections)", "", "| direction | vs refusal mass | vs judged correctness |", "|---|---|---|",
          f"| d_comply (primary) | {fmt(HA['comply_refusal'])} | {fmt(HA['comply_judged'])} |",
          f"| d_told (cross-fitted) | {fmt(HA['told_refusal'])} | {fmt(HA['told_judged'])} |", "",
          "## H-C(i): licence-head coverage, KVzip-g minus each shared selector (k8, paired)", "",
          "| vs | licence heads | all heads (descriptive) |", "|---|---|---|"]
    for s in SHARED_CMP:
        L.append(f"| {s} | {fmt(V['H-C(i)'][s])} | {fmt(V['H-C(i) all-head'][s])} |")
    h2 = V["H-C(ii)"]
    L += ["", "## H-C(ii): which coverage tracks the outcome? (within-item Spearman over the k8 selections)", "",
          "| | licence coverage | all-head coverage | difference (positive = licence better) |", "|---|---|---|---|",
          f"| vs refusal | {fmt(h2['rho_lic_ref'])} | {fmt(h2['rho_all_ref'])} | {fmt(h2['d_ref'])} |",
          f"| vs judged | {fmt(h2['rho_lic_cor'])} | {fmt(h2['rho_all_cor'])} | {fmt(h2['d_cor'])} |", "",
          "## Exploratory causal check (k8, paired over items)", "",
          "| contrast | refusal | answer score | proj d_comply @L* |", "|---|---|---|---|"]
    for k, r in V["causal"].items():
        L.append(f"| {k} | {fmt(r['refusal'])} | {fmt(r['answer_score'])} | {fmt(r['proj'])} |")
    L += ["", "## What each selector keeps (k8)", "",
          "| selector | budget share early / mid / late layers | CV over (layer, KV head) | licence-group budget vs uniform "
          "| gold | question words | punctuation | other |", "|---|---|---|---|---|---|---|---|"]
    for s, r in V["budgets"].items():
        cl = " | ".join(f"{num(r['classes'][c])} ({num(r['base'][c])})" for c in CLASSES)
        L.append(f"| {s} | {' / '.join(num(x) for x in r['thirds'])} | {num(r['cv'])} | "
                 f"{num(r['lic_group_budget_ratio'])} | {cl} |")
    L += ["", "Class columns: share of kept entries (in brackets the share among content tokens). Licence-group budget: "
          "the kept entries in the licence heads' KV groups relative to an even spread (1 = even).", "",
          "## d_comply projection per layer (means)", "",
          "| layer | separation (all items) | " + " | ".join(V["curve"]) + " |",
          "|---|---|" + "---|" * len(V["curve"])]
    sep = V.get("separation") or []
    n = len(next(iter(V["curve"].values()))) if V["curve"] else 0
    for l in range(n):
        L.append(f"| {l} | {num(sep[l]) if l < len(sep) else 'n/a'} | "
                 + " | ".join(num(V['curve'][c][l]) for c in V["curve"]) + " |")
    L += ["", "## In plain words", ""] + plain(V) + ["", "## What it can't tell us", "",
          "- Teacher-forced only: the refusal is the first token's probability; correctness is Stage A's judged label.",
          "- Direct attribution counts what a head writes straight onto the direction at L*, not what later layers do "
          "with it.",
          "- Coverage uses R-full's attention as the reference, not each slice's own attention.",
          "- 'Question words' is a proxy for the subject span (LongMemEval does not mark it).",
          "- The causal masks act on whole KV groups (several query heads), and the check is exploratory.",
          "- 66 items in 2 folds of 33; correlations within an item rest on 6-9 conditions."]
    return "\n".join(L) + "\n"


def plain(V):
    T = V["table"]
    g, sh = T.get("KVzip-g|k8", {}), T.get("KVzip-sh|k8", {})
    out = [f"- With the whole turn restored the first-token refusal is {num(T.get('R-full', {}).get('refusal'))}, "
           f"with nothing {num(T.get('nomem', {}).get('refusal'))}; KVzip-g at 8 positions' worth "
           f"{num(g.get('refusal'))}, KVzip-sh {num(sh.get('refusal'))}.",
           f"- Along the comply direction at layer L*, conditions that refuse sit lower within an item: mean Spearman "
           f"{num(V['H-A']['comply_refusal']['mean'])} ({'H-A holds' if V['H-A holds'] else 'H-A fails'}).",
           f"- The 8 heads picked on the other half of the items carry {num(V['H-B']['pooled'])} of the heads' "
           f"positive push from 'no memory' to 'whole turn' ({'H-B holds' if V['H-B holds'] else 'H-B fails'}).",
           f"- Those heads see {num(g.get('lic_cov'))} of their whole-turn attention under KVzip-g, "
           f"{num(sh.get('lic_cov'))} under KVzip-sh, {num(T.get('S-loo|k8', {}).get('lic_cov'))} under S-loo, "
           f"{num(T.get('S-dream|k8', {}).get('lic_cov'))} under S-dream.",
           f"- Verdict: **{V['verdict']}**."]
    return out


def summary(root, models=None):
    root = Path(root)
    rows, Vs = [], {}
    for m in models or MODELS:
        p = root / m / "verdict.json"
        if p.exists():
            Vs[m] = json.load(open(p))
    L = ["# dream_heads: summary", "", f"Models with results: {', '.join(LABELS.get(m, m) for m in Vs) or 'none'}.", "",
         "| model | H-A (rho, refusal) | H-B (pooled share) | H-C(i) | H-C(ii) | verdict | KVzip-g / KVzip-sh / S-loo "
         "licence coverage |", "|---|---|---|---|---|---|---|"]
    for m, V in Vs.items():
        T = V["table"]
        L.append(f"| {LABELS.get(m, m)} | {num(V['H-A']['comply_refusal']['mean'])} ({'holds' if V['H-A holds'] else 'fails'}) "
                 f"| {num(V['H-B']['pooled'])} ({'holds' if V['H-B holds'] else 'fails'}) | "
                 f"{'holds' if V['H-C(i) holds'] else 'fails'} | {'holds' if V['H-C(ii) holds'] else 'fails'} | "
                 f"{V['verdict']} | {num(T.get('KVzip-g|k8', {}).get('lic_cov'))} / "
                 f"{num(T.get('KVzip-sh|k8', {}).get('lic_cov'))} / {num(T.get('S-loo|k8', {}).get('lic_cov'))} |")
    n_hg = sum(V["verdict"] == "head-gated licence" for V in Vs.values())
    n_haf = sum(not V["H-A holds"] for V in Vs.values())
    cross = ("head-gated licence" if n_hg >= 2 else "no single gate" if n_haf >= 2 else "in between")
    L += ["", f"**Cross-model verdict: {cross}** (head-gated on {n_hg} of {len(Vs)}; H-A fails on {n_haf} of {len(Vs)}; "
          "rule: >= 2 of 3).", ""]
    (root / "summary.md").write_text("\n".join(L) + "\n")
    json.dump({"models": list(Vs), "verdicts": {m: V["verdict"] for m, V in Vs.items()}, "cross": cross},
              open(root / "summary.json", "w"), indent=1)
    return cross
