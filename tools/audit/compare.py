#!/usr/bin/env python
"""Compare the human audit labels with the judge's labels (core_v1) and check whether the pre-registered
judge-based claims survive the judge's measured error rates.

  python tools/audit/compare.py --labelled audit_labelled.jsonl --key audit_key.jsonl \
      [--claims results/core_v1_20261007_1820/analysis/claims_all.json] [--run results/core_v1_20261007_1820] \
      [--out audit_report.md] [--n-boot 2000] [--seed 7]

Standard library + numpy only. The human never saw audit_key.jsonl; condition / model come from it only
(joined on audit_id, never from the labelled file).

Sections of the report: (1) agreement per rubric and field, (2) breakdown by model and condition,
(3) robustness of C2 / C3 / C5 to the judge's measured error rates, (4) a verdict per claim.
"""

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "results" / "core_v1_20261007_1820"
FIELDS = {"fact": ["uses_fact", "coherent", "self_claim", "wrong_value"],
          "preference": ["direction", "coherent", "self_claim"]}
CAT = {"direction": ["toward", "away", "neutral"]}
BASE_CONDS = ("nomem", "ctx")
LEX_NAME = {1.0: "toward", 0.0: "neutral", -1.0: "away"}


# ----------------------------------------------------------------------------- value handling


def norm_bool(v):
    """True / False / 'unsure' / None (missing) from whatever the labelling page or a person wrote."""
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    if s in ("unsure", "?", "u", "unclear"):
        return "unsure"
    return None


def norm_cat(v, allowed):
    """A category string from `allowed`, 'unsure', or None (missing / unrecognised)."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in allowed:
        return s
    return "unsure" if s in ("unsure", "?", "u", "unclear") else None


def norm_field(rubric, field, v):
    return norm_cat(v, CAT[field]) if field in CAT else norm_bool(v)


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}: line {ln} is not valid JSON ({e})")
    return out


def join_rows(labelled, key):
    """Join on audit_id. Returns (joined, missing_in_key, extra_in_key). Each joined row is
    {'h': labelled row, 'k': key row}. Order follows the labelled file. Duplicate ids are an error."""
    kmap = {}
    for r in key:
        if r["audit_id"] in kmap:
            raise ValueError(f"duplicate audit_id {r['audit_id']} in key")
        kmap[r["audit_id"]] = r
    seen, joined, missing = set(), [], []
    for r in labelled:
        a = r["audit_id"]
        if a in seen:
            raise ValueError(f"duplicate audit_id {a} in labelled file")
        seen.add(a)
        if a in kmap:
            joined.append({"h": r, "k": kmap[a]})
        else:
            missing.append(a)
    extra = [a for a in kmap if a not in seen]
    return joined, missing, extra


def cond_family(c):
    if c in BASE_CONDS:
        return "base"
    head = c.split("@")[0]
    if head == "mem":
        return "mem"
    if head in ("rand", "swap", "placebo", "gate_on"):
        return "control"
    return "mechanism"


def judge_value(k, field):
    """The judge's value for a field, or None when the judge did not parse (ok false) or has no value."""
    j = k.get("judge") or {}
    if not j.get("ok"):
        return None
    v = j.get(field)
    return v


def field_pairs(joined, rubric, field):
    """Pairs (human, judge, joined_row) for one field. Missing human values and unparsed judge rows are
    dropped (and counted); 'unsure' human values are dropped from the pairs and counted separately.
    Returns pairs, counts{n_rows, n_labelled, n_unsure, n_missing, n_judge_unparsed}."""
    pairs, c = [], defaultdict(int)
    for r in joined:
        if r["h"].get("rubric") != rubric:
            continue
        c["n_rows"] += 1
        h = norm_field(rubric, field, r["h"].get("h_" + field))
        if h is None:
            c["n_missing"] += 1
            continue
        if h == "unsure":
            c["n_unsure"] += 1
            continue
        j = judge_value(r["k"], field)
        if j is None:
            c["n_judge_unparsed"] += 1
            continue
        if field not in CAT:
            j = bool(j)
        pairs.append((h, j, r))
    c["n_labelled"] = len(pairs)
    return pairs, dict(c)


# ----------------------------------------------------------------------------- statistics


def agreement(a, b):
    """Share of equal pairs (nan for no pairs)."""
    if len(a) == 0:
        return float("nan")
    return float(np.mean([x == y for x, y in zip(a, b)]))


def _codes(a, b):
    labs = sorted({str(x) for x in a} | {str(x) for x in b})
    m = {l: i for i, l in enumerate(labs)}
    return np.array([m[str(x)] for x in a]), np.array([m[str(x)] for x in b]), labs


def _kappa_codes(ca, cb, k):
    n = len(ca)
    if n == 0:
        return float("nan")
    po = float(np.mean(ca == cb))
    pa = np.bincount(ca, minlength=k) / n
    pb = np.bincount(cb, minlength=k) / n
    pe = float(pa @ pb)
    if pe >= 1.0 - 1e-12:
        return float("nan")  # both raters constant: kappa is undefined
    return (po - pe) / (1.0 - pe)


def cohen_kappa(a, b):
    """Cohen's kappa for two label sequences (any hashable labels). nan if undefined."""
    ca, cb, labs = _codes(a, b)
    return _kappa_codes(ca, cb, len(labs))


def bootstrap_kappa(a, b, n_boot=2000, seed=7):
    """Percentile 95% CI of kappa from resampling pairs; replicates with undefined kappa are skipped."""
    if len(a) < 2:
        return float("nan"), float("nan")
    ca, cb, labs = _codes(a, b)
    rng = np.random.default_rng(seed)
    n, ks = len(ca), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        k = _kappa_codes(ca[i], cb[i], len(labs))
        if not math.isnan(k):
            ks.append(k)
    if len(ks) < max(20, n_boot // 20):
        return float("nan"), float("nan")
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


def confusion(h, j, labels):
    """Matrix M[i][k] = #pairs with human == labels[i] and judge == labels[k]."""
    ix = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for x, y in zip(h, j):
        M[ix[x], ix[y]] += 1
    return M


def rogan_gladen(p_obs, sens, spec):
    """Rogan-Gladen corrected prevalence: (p_obs + spec - 1) / (sens + spec - 1), clipped to [0, 1].
    nan when the test is no better than chance (sens + spec - 1 <= 1e-9) or an input is nan."""
    d = sens + spec - 1.0
    if any(isinstance(x, float) and math.isnan(x) for x in (p_obs, sens, spec)) or d <= 1e-9:
        return float("nan")
    return float(min(1.0, max(0.0, (p_obs + spec - 1.0) / d)))


def misclass_correct(p_obs, M):
    """Multi-class version. M[t][o] = P(judge = o | human = t) (rows sum to 1). p_obs = p_true @ M, so
    solve for p_true, clip negatives to 0 and renormalise. nan vector if M is (near) singular."""
    p_obs = np.asarray(p_obs, float)
    M = np.asarray(M, float)
    if np.linalg.cond(M) > 1e6:
        return np.full_like(p_obs, np.nan)
    p = np.linalg.solve(M.T, p_obs)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, np.nan)


def rates_from_pairs(h, j):
    """(sens, spec) of a binary judge vs binary human truth; nan where a class is absent."""
    h, j = np.asarray(h, bool), np.asarray(j, bool)
    sens = float(j[h].mean()) if h.any() else float("nan")
    spec = float((~j[~h]).mean()) if (~h).any() else float("nan")
    return sens, spec


def conf_rows(pairs_h, pairs_j, k):
    """Row-normalised misclassification matrix from class codes; a class with no human example gets an
    identity row (assumed correct) - documented in the report."""
    C = np.zeros((k, k))
    for t, o in zip(pairs_h, pairs_j):
        C[t, o] += 1
    M = np.eye(k)
    for t in range(k):
        if C[t].sum() > 0:
            M[t] = C[t] / C[t].sum()
    return M


# ----------------------------------------------------------------------------- derived clean-use / lean labels


def tri_and(*vals):
    """Three-valued AND over True / False / unsure|None (-> None)."""
    if any(v is False for v in vals):
        return False
    if any(v is None or v == "unsure" for v in vals):
        return None
    return True


def human_fact_vals(h):
    """(clean-use, hit-and-clean) of a labelled fact row; None where undetermined by 'unsure'/missing."""
    return tri_and(norm_bool(h.get("h_uses_fact")), norm_bool(h.get("h_coherent")))


def judge_fact_clean(k):
    j = k.get("judge") or {}
    if not j.get("ok"):
        return None
    return bool(j.get("uses_fact")) and bool(j.get("coherent"))


def lean_class(direction, coherent):
    """Class index of j_lean: 0 = +1 (coherent, toward), 1 = 0 (neutral or incoherent), 2 = -1 (coherent, away).
    None if undetermined by unsure / missing."""
    if coherent is False:
        return 1
    if coherent is True:
        return {"toward": 0, "neutral": 1, "away": 2}.get(direction) if direction not in (None, "unsure") else None
    return 1 if direction == "neutral" else None


def human_lean_class(h):
    return lean_class(norm_cat(h.get("h_direction"), CAT["direction"]), norm_bool(h.get("h_coherent")))


def judge_lean_class(k):
    j = k.get("judge") or {}
    if not j.get("ok") or j.get("direction") is None:
        return None
    return lean_class(j["direction"], bool(j.get("coherent")))


# ----------------------------------------------------------------------------- audit-derived correction objects


class FactAudit:
    """Sens/spec of the judge's 'clean use' (uses_fact AND coherent) and of 'hit AND clean use' against the human,
    from the fact rows of the audit. Two variants: pooled (all conditions) and stratified (base = nomem/ctx
    versus every other condition). A stratum without positives/negatives falls back to the pooled rate."""

    def __init__(self, joined):
        self.rows = []
        for r in joined:
            if r["h"].get("rubric") != "fact":
                continue
            hc = human_fact_vals(r["h"])
            jc = judge_fact_clean(r["k"])
            if hc is None or jc is None:
                continue
            hit = bool(r["k"].get("hit"))
            self.rows.append({"base": r["k"].get("cond") in BASE_CONDS, "hc": hc, "jc": jc,
                              "hhc": hit and hc, "jhc": hit and jc})

    def params(self, idx=None):
        """{variant: {indicator(1,2): {stratum: (sens, spec)}}} from rows idx (default all)."""
        rows = self.rows if idx is None else [self.rows[i] for i in idx]
        out = {}
        for ind, (hk, jk) in {1: ("hc", "jc"), 2: ("hhc", "jhc")}.items():
            pooled = rates_from_pairs([r[hk] for r in rows], [r[jk] for r in rows]) if rows else (float("nan"),) * 2
            strata = {}
            for name, sel in (("base", True), ("other", False)):
                rs = [r for r in rows if r["base"] == sel]
                s = rates_from_pairs([r[hk] for r in rs], [r[jk] for r in rs]) if rs else (float("nan"),) * 2
                strata[name] = (s[0] if not math.isnan(s[0]) else pooled[0], s[1] if not math.isnan(s[1]) else pooled[1])
            out.setdefault("pooled", {})[ind] = {"base": pooled, "other": pooled}
            out.setdefault("strat", {})[ind] = strata
        return out

    def full(self):
        return self.params()

    def sample(self, rng):
        n = len(self.rows)
        return self.params(rng.integers(0, n, n)) if n else self.params()


class FactCorr:
    """Applies the Rogan-Gladen correction to a fact feature vector [clean, hit&clean, hit] of a condition."""

    def __init__(self, params, variant):
        self.p, self.variant = params, variant

    def __call__(self, vec, cond):
        st = "base" if cond in BASE_CONDS else "other"
        v = self.p[self.variant]
        out = np.array(vec, float)
        out[0] = rogan_gladen(out[0], *v[1][st])
        out[1] = rogan_gladen(out[1], *v[2][st])
        return out


class PrefAudit:
    """3x3 misclassification matrix M[t][o] of the judge's lean class (toward&coherent / zero / away&coherent)
    against the human's, from all preference rows."""

    def __init__(self, joined):
        self.pairs = []
        for r in joined:
            if r["h"].get("rubric") != "preference":
                continue
            h, j = human_lean_class(r["h"]), judge_lean_class(r["k"])
            if h is not None and j is not None:
                self.pairs.append((h, j))

    def matrix(self, idx=None):
        ps = self.pairs if idx is None else [self.pairs[i] for i in idx]
        return conf_rows([p[0] for p in ps], [p[1] for p in ps], 3)

    def full(self):
        return self.matrix()

    def sample(self, rng):
        n = len(self.pairs)
        return self.matrix(rng.integers(0, n, n)) if n else self.matrix()


class PrefCorr:
    def __init__(self, M):
        self.M = M

    def __call__(self, vec, cond):
        return misclass_correct(vec, self.M)


class NoCorr:
    def __call__(self, vec, cond):
        return np.asarray(vec, float)


# ----------------------------------------------------------------------------- run data (score outputs)


def has_word(text, w):
    """Whole-word, case-insensitive match (as experiments/xlayer_v1/xl.py has_word)."""
    return bool(re.search(rf"(?<![\w-]){re.escape(w.strip())}(?![\w-])", text, re.I))


def _read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(run_dir):
    """Per-row features from one model's saved generations + judge outputs (seed set 1, 'gen' rows).
    Facts: [clean, hit&clean, hit] with clean = judge uses_fact AND coherent. Preference: one-hot of the
    judge's lean class. Rows whose judge did not parse are dropped (and counted)."""
    run_dir = Path(run_dir)
    items = json.load(open(run_dir / "prep" / "items.json"))
    judge = {r["key"]: r for r in _read_gz(run_dir / "score" / "judge.jsonl.gz")}
    rows, dropped = [], 0
    for f in sorted((run_dir / "gen" / "items").glob("*.jsonl.gz")):
        for r in _read_gz(f):
            if r.get("sec") != "gen" or r.get("seedset") != 1:
                continue
            it = items[r["item"]]
            j = judge.get(r["key"])
            if not j or not j.get("ok"):
                dropped += 1
                continue
            if it["group"] == "facts":
                clean = bool(j["uses_fact"]) and bool(j["coherent"])
                hit = has_word(r["answer"], it["measure"]["target"])
                feat = np.array([clean, hit and clean, hit], float)
            else:
                feat = np.zeros(3)
                feat[lean_class(j["direction"], bool(j["coherent"]))] = 1.0
            rows.append({"item": r["item"], "group": it["group"], "kind": r["kind"], "pi": r["pi"],
                         "cond": r["cond"], "feat": feat})
    return {"rows": rows, "dropped": dropped, "items": items, "dir": run_dir}


def build_arrays(rows, conds, items, kinds=None):
    """For each item: ndarray [C, P, K] of per-prompt mean features (mean over the prompt's rows), on the
    prompts present in every condition. Items without any common prompt are dropped."""
    cell = defaultdict(list)
    for r in rows:
        if r["cond"] in conds and r["item"] in items and (not kinds or r["kind"] in kinds):
            cell[(r["item"], r["cond"], r["pi"])].append(r["feat"])
    arrs = []
    for it in items:
        pis = None
        for c in conds:
            s = {pi for (i, cc, pi) in cell if i == it and cc == c}
            pis = s if pis is None else pis & s
        if not pis:
            continue
        pis = sorted(pis)
        arrs.append(np.array([[np.mean(cell[(it, c, pi)], axis=0) for pi in pis] for c in conds]))
    return arrs


def boot_stat(arrs, conds, stat_fn, audit=None, corr_factory=None, n_boot=2000, seed=7):
    """Point estimate + two-level percentile bootstrap (items, then prompts within items; the audit rows are
    resampled too when `audit` is given). stat_fn(means [C,K], corr) -> {name: float}.
    corr_factory(sample) -> correction callable: sample is the audit's params / matrix (full audit for the
    point estimate, a bootstrap resample otherwise)."""
    rng = np.random.default_rng(seed)
    n = len(arrs)
    if n == 0:
        return {}

    def means(item_idx, prompt_rng):
        acc = 0.0
        for i in item_idx:
            a = arrs[i]
            if prompt_rng is not None:
                a = a[:, prompt_rng.integers(0, a.shape[1], a.shape[1]), :]
            acc = acc + a.mean(axis=1)
        return acc / len(item_idx)

    def make(sample):
        return corr_factory(sample) if corr_factory else NoCorr()

    full = audit.full() if audit is not None else None
    point = stat_fn(means(range(n), None), make(full))
    reps = defaultdict(list)
    for _ in range(n_boot):
        s = audit.sample(rng) if audit is not None else None
        d = stat_fn(means(rng.integers(0, n, n), rng), make(s))
        for k, v in d.items():
            reps[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array(reps[k], float)
        good = arr[~np.isnan(arr)]
        lo, hi = (np.percentile(good, 2.5), np.percentile(good, 97.5)) if len(good) >= max(20, n_boot // 20) else (float("nan"),) * 2
        out[k] = {"est": float(v), "lo": float(lo), "hi": float(hi), "n_valid": int(len(good))}
    return out


# ----------------------------------------------------------------------------- claim robustness


def lean(vec):
    return vec[0] - vec[2]


def run_c3(run, doses, audit, variants, n_boot, seed):
    """C3 (facts): use_judge_clean(mem@a) <= 1/2 ctx's with the CI of (ctx - mem@a) above 0 at every dose, and
    at some a >= 2 mention_unclean(mem@a) > nomem's. Returns {variant: {...}} for variants in
    'none' / 'pooled' / 'strat'."""
    facts = [i for i, it in run["items"].items() if it["group"] == "facts"]
    out = {}
    for var in variants:
        factory = None if var == "none" else (lambda s, v=var: FactCorr(s, v))
        aud = None if var == "none" else audit
        use_conds = ["ctx"] + [f"mem@{d:g}" for d in doses]
        arrs = build_arrays(run["rows"], use_conds, facts, kinds=["use"])

        def f_use(m, corr, use_conds=use_conds):
            v = [corr(m[i], c) for i, c in enumerate(use_conds)]
            d = {"ctx": v[0][0]}
            for i, dd in enumerate(doses, 1):
                d[f"use@{dd:g}"] = v[i][0]
                d[f"diff@{dd:g}"] = v[0][0] - v[i][0]
            return d
        use = boot_stat(arrs, use_conds, f_use, aud, factory, n_boot, seed)
        mu_conds = ["nomem"] + [f"mem@{d:g}" for d in doses]
        arrs2 = build_arrays(run["rows"], mu_conds, facts)

        def f_mu(m, corr, mu_conds=mu_conds):
            v = [corr(m[i], c) for i, c in enumerate(mu_conds)]
            d = {"nomem": v[0][2] - v[0][1]}
            for i, dd in enumerate(doses, 1):
                d[f"mu@{dd:g}"] = v[i][2] - v[i][1]
                d[f"mudiff@{dd:g}"] = (v[i][2] - v[i][1]) - (v[0][2] - v[0][1])
            return d
        mu = boot_stat(arrs2, mu_conds, f_mu, aud, factory, n_boot, seed)
        far = {}
        for dd in doses:
            u, df = use.get(f"use@{dd:g}"), use.get(f"diff@{dd:g}")
            far[dd] = bool(u and df and u["est"] <= 0.5 * use["ctx"]["est"] and df["lo"] > 0)
        rise = any(mu[f"mudiff@{dd:g}"]["est"] > 0 for dd in doses if dd >= 2 and f"mudiff@{dd:g}" in mu)
        far_pt = all(use[f"use@{dd:g}"]["est"] <= 0.5 * use["ctx"]["est"] for dd in doses) if use else False
        out[var] = {"use": use, "mu": mu, "far": far, "rise": bool(rise), "holds": bool(all(far.values()) and rise),
                    "holds_point": bool(far_pt and rise)}
    return out


def run_pref(run, group_items, conds, stat_fn, audit, variants, n_boot, seed):
    out = {}
    for var in variants:
        factory = None if var == "none" else (lambda s: PrefCorr(s))
        aud = None if var == "none" else audit
        arrs = build_arrays(run["rows"], conds, group_items)
        out[var] = boot_stat(arrs, conds, stat_fn, aud, factory, n_boot, seed)
    return out


def run_c5(run, dislike_items, alphas, audit, variants, n_boot, seed):
    """C5 (dislikes): at each reference's matched dose, the opposite's judge-lean gain vs nomem has a CI above 0,
    the centroid's and without's have CIs below 0."""
    ao, ac, aw = alphas
    conds = ["nomem", f"mem@{ao:g}", f"centroid@{ac:g}", f"without@{aw:g}"]

    def f(m, corr):
        v = [lean(corr(m[i], c)) for i, c in enumerate(conds)]
        return {"opposite": v[1] - v[0], "centroid": v[2] - v[0], "without": v[3] - v[0]}
    res = run_pref(run, dislike_items, conds, f, audit, variants, n_boot, seed)
    for var, r in res.items():
        r["_holds"] = bool(r and r["opposite"]["lo"] > 0 and r["centroid"]["hi"] < 0 and r["without"]["hi"] < 0)
        r["_holds_point"] = bool(r and r["opposite"]["est"] > 0 and r["centroid"]["est"] < 0 and r["without"]["est"] < 0)
    return res


def run_c2(run, leaning_items, a_ctrl, audit, variants, n_boot, seed):
    """C2 (leanings): gain of mem@1 and mem@2 vs nomem above 0 (CI), each control's gain <= 1/3 of mem@2's and the
    CI of (mem@2 - control) above 0."""
    ctrls = [f"{k}@{a_ctrl:g}" for k in ("rand", "swap", "placebo")]
    conds = ["nomem", "mem@1", "mem@2"] + ctrls

    def f(m, corr):
        v = [lean(corr(m[i], c)) for i, c in enumerate(conds)]
        d = {"gain1": v[1] - v[0], "gain2": v[2] - v[0]}
        for k, c in enumerate(ctrls):
            d[f"gain_{c}"] = v[3 + k] - v[0]
            d[f"diff_{c}"] = v[2] - v[3 + k]
        return d
    res = run_pref(run, leaning_items, conds, f, audit, variants, n_boot, seed)
    for var, r in res.items():
        if not r:
            r["_holds"] = r["_holds_point"] = False
            continue
        ok = r["gain1"]["lo"] > 0 and r["gain2"]["lo"] > 0
        for c in ctrls:
            ok = ok and r[f"gain_{c}"]["est"] <= r["gain2"]["est"] / 3 and r[f"diff_{c}"]["lo"] > 0
        r["_holds"] = bool(ok)
        okp = r["gain1"]["est"] > 0 and r["gain2"]["est"] > 0
        for c in ctrls:
            okp = okp and r[f"gain_{c}"]["est"] <= r["gain2"]["est"] / 3 and r[f"diff_{c}"]["est"] > 0
        r["_holds_point"] = bool(okp)
    return res


def kept_items(run, group, claims_model, claim_key):
    """Item ids of a preference group that passed the ctx rule: from claims_all.json (C2 'items') if available,
    else from the model's analysis/flags.csv."""
    if claim_key == "C2" and claims_model:
        its = claims_model.get("C2", {}).get("evidence", {}).get("items")
        if its:
            return list(its)
    flags = run["dir"] / "analysis" / "flags.csv"
    ids = [i for i, it in run["items"].items() if it["group"] == group]
    if flags.exists():
        import csv
        with open(flags) as f:
            bad = {r["item"] for r in csv.DictReader(f) if r["flagged"].strip().lower() == "true"}
        ids = [i for i in ids if i not in bad]
    return ids


# ----------------------------------------------------------------------------- report helpers


def f3(x, spec=".3f"):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def pct(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def ci(d, spec="+.3f"):
    return f"{f3(d['est'], spec)} [{f3(d['lo'], spec)}, {f3(d['hi'], spec)}]"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pos_rate(vals, field):
    """Share positive (bool field) or the class shares (direction)."""
    if not vals:
        return "nan"
    if field in CAT:
        return ", ".join(f"{c} {pct(sum(1 for v in vals if v == c) / len(vals))}" for c in CAT[field])
    return pct(sum(1 for v in vals if v) / len(vals))


def field_section(joined, n_boot, seed):
    """Per rubric and field: the agreement table + confusion matrices. Returns (markdown, summary dict)."""
    lines, summ = [], {}
    for rubric, fields in FIELDS.items():
        rows = [r for r in joined if r["h"].get("rubric") == rubric]
        if not rows:
            continue
        lines.append(f"### {rubric} rubric ({len(rows)} rows)\n")
        tab, confs = [], []
        for f in fields:
            pairs, c = field_pairs(joined, rubric, f)
            h = [p[0] for p in pairs]
            j = [p[1] for p in pairs]
            k = cohen_kappa(h, j) if pairs else float("nan")
            lo, hi = bootstrap_kappa(h, j, n_boot, seed) if pairs else (float("nan"),) * 2
            ag = agreement(h, j)
            summ[(rubric, f)] = {"n": len(pairs), "agree": ag, "kappa": k, "k_lo": lo, "k_hi": hi, **c}
            tab.append([f, len(pairs), c.get("n_unsure", 0), c.get("n_missing", 0), c.get("n_judge_unparsed", 0),
                        pct(ag), f"{f3(k)} [{f3(lo)}, {f3(hi)}]", pos_rate(j, f) + " (judge)", pos_rate(h, f) + " (human)"])
            labels = CAT[f] if f in CAT else [False, True]
            M = confusion(h, j, labels) if pairs else np.zeros((len(labels), len(labels)), int)
            name = lambda v: str(v).lower() if isinstance(v, bool) else str(v)
            confs.append(f"**{f}** (rows = human, columns = judge)\n\n" + md_table(
                [""] + [name(l) for l in labels], [[name(l)] + [int(x) for x in M[i]] for i, l in enumerate(labels)]))
        lines.append(md_table(["field", "n used", "unsure", "missing", "judge unparsed", "agreement", "Cohen kappa [95% CI]",
                               "judge rate", "human rate"], tab))
        lines.append("\n" + "\n\n".join(confs) + "\n")
        if rubric == "preference":
            lex = []
            for r in rows:
                lv = r["k"].get("lexicon")
                h = norm_cat(r["h"].get("h_direction"), CAT["direction"])
                if lv is None or h in (None, "unsure"):
                    continue
                lex.append((h, LEX_NAME.get(float(lv), "neutral")))
            if lex:
                hh, ll = [x[0] for x in lex], [x[1] for x in lex]
                M = confusion(hh, ll, CAT["direction"])
                lines.append(f"**Lexicon label vs human direction** (the lexicon check h_lexicon_ok is derived here, "
                             f"not asked): n = {len(lex)}, agreement {pct(agreement(hh, ll))}, kappa {f3(cohen_kappa(hh, ll))}\n\n"
                             + md_table(["human \\ lexicon"] + CAT["direction"],
                                        [[c] + [int(x) for x in M[i]] for i, c in enumerate(CAT["direction"])]) + "\n")
                summ[("preference", "lexicon_vs_direction")] = {"n": len(lex), "agree": agreement(hh, ll), "kappa": cohen_kappa(hh, ll)}
    return "\n".join(lines), summ


def breakdown_section(joined):
    """Agreement and kappa by model, by condition family, and by exact condition (all from the KEY)."""
    lines = []
    for rubric, fields in FIELDS.items():
        if not any(r["h"].get("rubric") == rubric for r in joined):
            continue
        lines.append(f"### {rubric} rubric\n")
        for title, kf in (("model", lambda r: r["k"].get("model", "?")),
                          ("condition family", lambda r: cond_family(r["k"].get("cond", "?"))),
                          ("model x family", lambda r: f"{r['k'].get('model', '?')} / {cond_family(r['k'].get('cond', '?'))}")):
            groups = sorted({kf(r) for r in joined if r["h"].get("rubric") == rubric})
            tab = []
            for g in groups:
                sub = [r for r in joined if r["h"].get("rubric") == rubric and kf(r) == g]
                row = [g, len(sub)]
                for f in fields:
                    pairs, _ = field_pairs(sub, rubric, f)
                    h, j = [p[0] for p in pairs], [p[1] for p in pairs]
                    row.append(f"{pct(agreement(h, j))} (n={len(pairs)}, k={f3(cohen_kappa(h, j) if pairs else float('nan'), '.2f')})")
                tab.append(row)
            lines.append(f"By {title}:\n\n" + md_table([title, "rows"] + [f"{f}: agree (n, kappa)" for f in fields], tab) + "\n")
        conds = sorted({r["k"].get("cond", "?") for r in joined if r["h"].get("rubric") == rubric})
        tab = []
        for c in conds:
            sub = [r for r in joined if r["h"].get("rubric") == rubric and r["k"].get("cond") == c]
            row = [c, len(sub)]
            for f in fields:
                pairs, _ = field_pairs(sub, rubric, f)
                row.append(f"{pct(agreement([p[0] for p in pairs], [p[1] for p in pairs]))} ({len(pairs)})")
            tab.append(row)
        lines.append("By exact condition (agreement, n in brackets; small n, indicative only):\n\n"
                     + md_table(["condition", "rows"] + fields, tab) + "\n")
    return "\n".join(lines)


def audit_rates_text(fa, pa):
    lines = []
    if fa.rows:
        p = fa.params()
        for var in ("pooled", "strat"):
            for ind, nm in ((1, "clean use (uses_fact AND coherent)"), (2, "hit AND clean use")):
                for st in ("base", "other"):
                    if var == "pooled" and st == "other":
                        continue
                    s, sp = p[var][ind][st]
                    lines.append([var if var == "strat" else "pooled", nm, st if var == "strat" else "all", f3(s, ".2f"), f3(sp, ".2f")])
    return lines


def robustness_section(joined, claims, run_dir, n_boot, seed):
    """Corrected claims C2 / C3 / C5 per model. Returns (markdown, verdicts)."""
    lines, verdicts = [], {}
    fa, pa = FactAudit(joined), PrefAudit(joined)
    lines.append("### Judge error rates measured on the audit\n")
    lines.append(f"Fact rows usable for the clean-use rates: {len(fa.rows)}. Preference rows usable for the lean-class matrix: {len(pa.pairs)}.\n")
    tab = audit_rates_text(fa, pa)
    if tab:
        lines.append(md_table(["variant", "indicator", "stratum", "sensitivity", "specificity"], tab) + "\n")
        lines.append("Stratified = 'base' (nomem, ctx) versus 'other' (every memory, control and mechanism condition); a stratum "
                     "without positives or negatives falls back to the pooled rate.\n")
    if len(pa.pairs):
        M = pa.matrix()
        lines.append("Lean-class matrix P(judge class | human class), classes = [toward & coherent (+1), neutral or incoherent (0), "
                     "away & coherent (-1)]; a human class with no example gets an identity row:\n")
        lines.append(md_table(["human \\ judge", "+1", "0", "-1"], [[n] + [f3(x, ".2f") for x in M[i]] for i, n in enumerate(["+1", "0", "-1"])]) + "\n")
        lines.append(f"(n per human class: +1 = {sum(1 for h, _ in pa.pairs if h == 0)}, 0 = {sum(1 for h, _ in pa.pairs if h == 1)}, "
                     f"-1 = {sum(1 for h, _ in pa.pairs if h == 2)})\n")
    run_dir = Path(run_dir)
    models = sorted(p.name for p in run_dir.glob("qwen35_*") if (p / "score" / "judge.jsonl.gz").exists()) if run_dir.exists() else []
    if not models:
        lines.append(f"**Score outputs not found under {run_dir}; the per-condition correction was skipped.** Only the agreement "
                     "tables above are available.\n")
        return "\n".join(lines), verdicts
    for mname in models:
        cm = (claims or {}).get(mname, {})
        run = load_run(run_dir / mname)
        lines.append(f"## {mname}\n")
        lines.append(f"Judge rows dropped for parse failure: {run['dropped']}.\n")
        # ---------------------------------------------------------------- C3
        c3 = cm.get("C3", {})
        doses = sorted(float(k.split("@")[1]) for k in c3.get("evidence", {}).get("per_dose", {})) or [0.5, 1.0, 2.0, 3.0]
        res = run_c3(run, doses, fa, ("none", "pooled", "strat"), n_boot, seed)
        lines.append("### C3 (facts do not come out as clean use)\n")
        lines.append("`use_judge_clean` on the use prompts and `mention_unclean` (target mentioned, not a judged clean use), "
                     "macro-mean over items; CIs are item/prompt bootstrap, and for corrected rows also resample the audit.\n")
        for var, lab in (("none", "uncorrected (reproduction)"), ("pooled", "Rogan-Gladen, pooled rates"), ("strat", "Rogan-Gladen, base/other strata")):
            r = res[var]
            if not r["use"]:
                continue
            tab = [["ctx", ci(r["use"]["ctx"], ".3f"), "", "", ""]]
            for d in doses:
                u, df = r["use"][f"use@{d:g}"], r["use"][f"diff@{d:g}"]
                tab.append([f"mem@{d:g}", ci(u, ".3f"), ci(df), "yes" if r["far"][d] else "no",
                            f"{ci(r['mu'][f'mu@{d:g}'], '.3f')} (nomem {f3(r['mu']['nomem']['est'])})"])
            lines.append(f"**{lab}** - verdict C3 holds: **{r['holds']}**\n\n"
                         + md_table(["cond", "use_judge_clean", "ctx - mem", "<= 1/2 ctx and CI > 0", "mention_unclean"], tab) + "\n")
        pre = c3.get("holds")
        changed = [v for v in ("pooled", "strat") if res[v]["holds"] != res["none"]["holds"]]
        if pre is not None and res["none"]["holds"] != pre:
            lines.append(f"Note: the uncorrected recomputation here ({res['none']['holds']}) differs from claims_all.json ({pre}); check the loader.\n")
        verdicts[(mname, "C3")] = {"point": {v: res[v]["holds_point"] for v in ("pooled", "strat")}, "pre": pre, "uncorr": res["none"]["holds"], "corr": {v: res[v]["holds"] for v in ("pooled", "strat")}, "changed": bool(changed)}
        if cm.get("C3"):
            e = c3["evidence"]
            diffs = [abs(res["none"]["use"][f"use@{d:g}"]["est"] - e["per_dose"][f"mem@{d:g}"]["use_judge_clean"]) for d in doses if f"use@{d:g}" in res["none"]["use"]]
            diffs.append(abs(res["none"]["use"]["ctx"]["est"] - e["ctx_use_judge_clean"]))
            verdicts[(mname, "C3")]["repro_maxdiff"] = max(diffs)
        # ---------------------------------------------------------------- C5
        c5 = cm.get("C5", {}).get("evidence", {})
        dis = kept_items(run, "dislikes", cm, "C5")
        al = tuple(float(c5.get(k, {}).get("alpha", 1)) for k in ("opposite", "centroid", "without"))
        r5 = run_c5(run, dis, al, pa, ("none", "pooled"), n_boot, seed)
        lines.append("### C5 (what you subtract decides concept vs direction; dislikes)\n")
        lines.append(f"Dislike items kept by the ctx rule: {len(dis)}. Matched doses (opposite, centroid, without) = {al}. "
                     "Gain = j_lean(condition) - j_lean(nomem), j_lean = +1/0/-1 class, corrected by inverting the audit's lean-class matrix.\n")
        if r5["none"]:
            tab = []
            for ref in ("opposite", "centroid", "without"):
                tab.append([ref, ci(r5["none"][ref]), ci(r5["pooled"][ref]), c5.get(ref, {}).get("d_j_lean") and f3(c5[ref]["d_j_lean"], "+.3f")])
            lines.append(md_table(["reference", "uncorrected gain [CI]", "corrected gain [CI]", "claims_all.json gain"], tab) + "\n")
            lines.append(f"C5 holds: uncorrected **{r5['none']['_holds']}**, corrected **{r5['pooled']['_holds']}** (pre-registered: {cm.get('C5', {}).get('holds')}).\n")
            verdicts[(mname, "C5")] = {"point": {"pooled": r5["pooled"]["_holds_point"]}, "pre": cm.get("C5", {}).get("holds"), "uncorr": r5["none"]["_holds"],
                                       "corr": {"pooled": r5["pooled"]["_holds"]}, "changed": r5["pooled"]["_holds"] != r5["none"]["_holds"],
                                       "repro_maxdiff": max((abs(r5["none"][k]["est"] - c5[k]["d_j_lean"]) for k in ("opposite", "centroid", "without") if k in c5), default=float("nan"))}
        # ---------------------------------------------------------------- C2
        c2 = cm.get("C2", {})
        lea = kept_items(run, "leanings", cm, "C2")
        r2 = run_c2(run, lea, 2.0, pa, ("none", "pooled"), n_boot, seed)
        lines.append("### C2 (leanings transfer, specifically)\n")
        if r2["none"]:
            tab = []
            for k in ("gain1", "gain2", "gain_rand@2", "gain_swap@2", "gain_placebo@2", "diff_rand@2", "diff_swap@2", "diff_placebo@2"):
                tab.append([k, ci(r2["none"][k]), ci(r2["pooled"][k])])
            lines.append(md_table(["quantity (j_lean)", "uncorrected [CI]", "corrected [CI]"], tab) + "\n")
            lines.append(f"C2 holds: uncorrected **{r2['none']['_holds']}**, corrected **{r2['pooled']['_holds']}** (pre-registered: {c2.get('holds')}).\n")
            ev = c2.get("evidence", {})
            rd = [abs(r2["none"]["gain1"]["est"] - ev["gain_mem1"][0]), abs(r2["none"]["gain2"]["est"] - ev["gain_mem2"][0])] if ev.get("gain_mem1") else [float("nan")]
            verdicts[(mname, "C2")] = {"point": {"pooled": r2["pooled"]["_holds_point"]}, "pre": c2.get("holds"), "uncorr": r2["none"]["_holds"], "corr": {"pooled": r2["pooled"]["_holds"]},
                                       "changed": r2["pooled"]["_holds"] != r2["none"]["_holds"], "repro_maxdiff": max(rd)}
    return "\n".join(lines), verdicts


def verdict_section(verdicts, summ):
    lines = []

    def fld(r, f):
        s = summ.get((r, f))
        return f"{pct(s['agree'])} agreement, kappa {f3(s['kappa'], '.2f')} [{f3(s['k_lo'], '.2f')}, {f3(s['k_hi'], '.2f')}] (n={s['n']})" if s else "no data"
    use = {"C3": [("fact", "uses_fact"), ("fact", "coherent")], "C5": [("preference", "direction"), ("preference", "coherent")],
           "C2": [("preference", "direction"), ("preference", "coherent")]}
    for claim in ("C3", "C5", "C2"):
        vs = {m: v for (m, c), v in verdicts.items() if c == claim}
        if not vs:
            lines.append(f"- **Judge trustworthy for {claim}: cannot say** - the score outputs were not available for the robustness check.")
            continue
        ch = [m for m, v in vs.items() if v["changed"]]
        fl = [m for m, v in vs.items() if any(x != v["uncorr"] for x in v["corr"].values())]
        ok = not fl
        hard = [m for m, v in vs.items() if any(x != v["uncorr"] for x in v["point"].values())]
        why = "; ".join(f"{m}: pre-registered {v['pre']}, uncorrected {v['uncorr']}, corrected " + "/".join(f"{k}={x}" for k, x in v["corr"].items()) for m, v in vs.items())
        fields = "; ".join(f"{f}: {fld(r, f)}" for r, f in use[claim])
        verdict = "yes" if ok else "no"
        if ok:
            extra = ""
        elif hard:
            extra = f" The corrected point estimates themselves flip the verdict on {', '.join(hard)}."
        else:
            extra = (f" On {', '.join(fl)} the corrected point estimates still support the pre-registered verdict, but with the "
                     "audit's sampling error added the CIs no longer meet the pre-registered rule: not refuted, but not confirmed.")
        lines.append(f"- **Judge trustworthy for {claim}: {verdict}.** The pre-registered verdict " + ("does not change" if ok else "changes")
                     + f" under the correction ({why}). Judge fields feeding the claim: {fields}.{extra}")
        small = [m for m, v in vs.items() if v.get("repro_maxdiff", 0) > 0.02]
        if small:
            lines.append(f"  - Reproduction check: the uncorrected recomputation differs from claims_all.json by more than 0.02 on {', '.join(small)}.")
    lines.append("- C1 (selectivity), C4 (yes/no premises) and C6 (logit lens) do not use the judge; the audit does not cover them.")
    return "\n".join(lines)


def build_report(labelled, key, claims, run_dir, n_boot, seed, names=("", "")):
    joined, missing, extra = join_rows(labelled, key)
    out = ["# core_v1 judge audit: human vs judge\n"]
    nrub = defaultdict(int)
    for r in joined:
        nrub[r["h"].get("rubric")] += 1
    anyl = sum(1 for r in joined if any(r["h"].get("h_" + f) is not None for f in FIELDS.get(r["h"].get("rubric"), [])))
    out.append("## What was compared\n")
    out.append(f"The human labelled the same {len(joined)} answers the judge (Qwen3.5-9B) had labelled: "
               + ", ".join(f"{v} {k}" for k, v in sorted(nrub.items())) + f". Rows with at least one human label: {anyl}. "
               "Agreement is the share of equal labels; Cohen's kappa corrects for chance agreement (0 = chance, 1 = perfect) with a "
               "bootstrap 95% CI. 'unsure' human labels are excluded from agreement and kappa and counted separately. The condition and "
               "model come only from the key file (the labeller never saw them). Rows with an unparsed judge output are excluded.\n")
    if missing:
        out.append(f"WARNING: {len(missing)} labelled ids are not in the key (ignored): {missing[:5]}...\n")
    if extra:
        out.append(f"Note: {len(extra)} key ids are not in the labelled file.\n")
    sec, summ = field_section(joined, n_boot, seed)
    out.append("## 1. Agreement per rubric and field\n")
    out.append(sec)
    out.append("## 2. Breakdown by model and condition (from the key)\n")
    out.append(breakdown_section(joined))
    out.append("## 3. Claim robustness\n")
    out.append("Method. For a yes/no judge label the Rogan-Gladen correction is p_true = (p_obs + spec - 1) / (sens + spec - 1) (clipped to "
               "[0, 1]), with sensitivity and specificity measured against the human on the audit. The judge's 'clean use' (uses_fact AND "
               "coherent) and 'hit AND clean use' (for mention_unclean = hit - hit&clean) are corrected this way, once with rates pooled over "
               "all conditions and once with separate rates for base (nomem, ctx) and other conditions. For the 3-class lean label (j_lean) "
               "the audit's 3x3 judge-given-human matrix is inverted (clip negatives, renormalise). Every number is recomputed on the "
               "per-prompt cells of seed set 1 from the saved generations and judge outputs, with the pre-registered two-level bootstrap "
               "(items, then prompts) plus a resample of the audit rows. Assumption: the judge's error rates measured on the audit sample "
               "carry over to all conditions (non-differential error); the audit is small, so the CIs are wide and honest.\n")
    rob, verdicts = robustness_section(joined, claims, run_dir, n_boot, seed)
    out.append(rob)
    out.append("## 4. Verdicts\n")
    out.append(verdict_section(verdicts, summ))
    out.append("\nCaveats: ~200 audited answers over two models and ~16 conditions, so per-cell rates are rough; a verdict that does not "
               "change under correction is evidence the judge is not driving the result, not proof of a perfect judge.\n")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", required=True, help="audit_labelled.jsonl exported by label.html")
    ap.add_argument("--key", required=True, help="audit_key.jsonl (the judge's labels; open only after labelling)")
    ap.add_argument("--claims", default=None, help="claims_all.json (default: <run>/analysis/claims_all.json if it exists)")
    ap.add_argument("--run", default=str(DEFAULT_RUN), help="core_v1 run dir holding qwen35_*/ score outputs")
    ap.add_argument("--out", default="audit_report.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    claims_path = Path(a.claims) if a.claims else Path(a.run) / "analysis" / "claims_all.json"
    claims = json.load(open(claims_path)) if claims_path.exists() else None
    if claims is None:
        print(f"note: no claims file at {claims_path}; pre-registered verdicts will not be shown")
    rep = build_report(load_jsonl(a.labelled), load_jsonl(a.key), claims, a.run, a.n_boot, a.seed)
    Path(a.out).write_text(rep, encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
