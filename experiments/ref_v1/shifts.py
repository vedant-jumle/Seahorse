"""ref_v1 pure helpers (torch only, no model): moment pooling, reference shifts, norm matching, the
disclosure pool, logit-lens lexicon scores and loop detection. Unit-tested in tests/test_ref_v1.py.

A moment = one follow-up sentence. Its shift is pooled over the follow-up's shared suffix exactly as
diag_keys.unit_writes(unit="pooled") does: weights = next-token entropy of the WITHOUT run divided by
its max over the suffix (v0_1.select, gate "entropy"), the template tail dropped, renormalised to 1.
Pooling is linear, so pool(with - ref) = pool(with) - pool(ref), and a reference averaged over K runs
is pool(with) - mean_k pool(ref_k).
"""

import math
import random

import torch


def moment_weights(entropy, tail_len):
    """entropy [n] over the shared suffix -> weights [n - tail_len] summing to 1 (tail excluded)."""
    n = entropy.shape[0]
    keep = n - tail_len
    assert keep > 0, "follow-up has no content tokens before the template tail"
    g = entropy / entropy.max().clamp_min(1e-8)
    g = g[:keep]
    return g / g.sum()


def pool(states, w):
    """states [n, d] over the shared suffix -> the w-weighted mean of its first len(w) rows, [d]."""
    return (w[:, None].to(states.dtype) * states[:w.shape[0]]).sum(0)


def reference_shift(with_pooled, ref):
    """with - reference. ref: [d] (one run, or the hum mu), or [K, d] / a list of [d] (averaged)."""
    if isinstance(ref, (list, tuple)):
        ref = torch.stack(list(ref))
    return with_pooled - (ref if ref.dim() == 1 else ref.mean(0))


def match_norm(shift, target_norm, eps=1e-12):
    """Rescale `shift` to length `target_norm` (direction unchanged)."""
    return shift * (float(target_norm) / shift.norm().clamp_min(eps))


def cos(a, b, eps=1e-12):
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(eps))


# ------------------------------------------------------------------ disclosure pool


def pick_disclosure(candidates, item, k, k_disp_max, seed):
    """K other experiences to precede the follow-up with. candidates: [{id, kind (disposition|fact),
    category, topics, experience}]. Excluded: the item itself, item["exclude_ids"], the item's own
    category, and any shared topic (topics mark what could contradict or overlap the item). One fixed
    seeded order per kind (independent of the item); take the first min(k_disp_max, eligible)
    dispositions and fill up to k with facts."""
    excl = {item["id"], *item.get("exclude_ids", [])}
    topics = set(item.get("topics", []))

    def ok(c):
        return (c["id"] not in excl and c["category"] != item["category"] and not topics & set(c.get("topics", [])))
    out = []
    for kind, cap in (("disposition", k_disp_max), ("fact", k)):
        order = sorted((c for c in candidates if c["kind"] == kind), key=lambda c: c["id"])
        random.Random(seed).shuffle(order)
        take = [c for c in order if ok(c)][:max(0, min(cap, k - len(out)))]
        out += take
    if len(out) < k:
        raise ValueError(f"{item['id']}: only {len(out)} eligible disclosure experiences (need {k})")
    return out


# ------------------------------------------------------------------- logit lens


def lexicon_token_ids(encode, lexicon):
    """First subword of ' ' + word for each lexicon word, per side; ids on both sides are dropped.
    encode(text) -> list of token ids."""
    side = {s: {encode(" " + w)[0] for w in lexicon[s]} for s in ("consistent", "inconsistent")}
    both = side["consistent"] & side["inconsistent"]
    return sorted(side["consistent"] - both), sorted(side["inconsistent"] - both)


def lex_gain(logits, cons_ids, inc_ids):
    """Mean logit of the consistent-side tokens minus the inconsistent-side tokens."""
    if not cons_ids or not inc_ids:
        return float("nan")
    return float(logits[cons_ids].mean() - logits[inc_ids].mean())


# ------------------------------------------------------------------ text metrics


def rep_rate(ids, n=4):
    """Share of repeated n-grams (1 - distinct / total)."""
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def lean_stats(labels, loops):
    """labels: consistent|inconsistent|neutral per answer; loops: bool per answer. Returns cons, inc,
    lean = cons - inc, loop share, and lean_clean = lean over the non-loop answers only."""
    n = len(labels)
    nan = float("nan")
    if not n:
        return {"n": 0, "cons": nan, "inc": nan, "lean": nan, "loop": nan, "n_clean": 0, "lean_clean": nan}
    c, i = labels.count("consistent"), labels.count("inconsistent")
    clean = [lab for lab, lp in zip(labels, loops) if not lp]
    nc = len(clean)
    lc = (clean.count("consistent") - clean.count("inconsistent")) / nc if nc else nan
    return {"n": n, "cons": c / n, "inc": i / n, "lean": (c - i) / n, "loop": sum(map(bool, loops)) / n,
            "n_clean": nc, "lean_clean": lc}


def nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")
