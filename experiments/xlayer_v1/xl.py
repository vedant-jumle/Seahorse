"""xlayer_v1 pure helpers (torch + re only, no model): the cross-layer dose, the two-pass gate and
injection field, the Stage-A pick rule, and the Stage-B text metrics. Unit-tested in
tests/test_xlayer_v1.py.

Two-pass reading. Pass 1 runs the frozen model WITHOUT memory and reads the keys at layer R; the gate
and the recall M k of every position come from that pass only. Pass 2 adds the field at layer W. So
the injection can never change its own gate, even when W < R (where a one-pass hook would read keys
from a stream the injection has already changed). Keys are running means over user-text positions,
so the field at position t only uses positions <= t (causal), and every position after the user text
(template tail, generated tokens) gets the field of the last prompt position.
"""

import math
import re

import torch

# ------------------------------------------------------------------------- dose


def median_norm(states):
    """[N, d] tensor (or a list of [n_i, d]) -> the median L2 norm of its rows (lower median)."""
    if isinstance(states, (list, tuple)):
        states = torch.cat(list(states))
    return states.float().norm(dim=-1).median().item()


def dose_ratio(med, W, R):
    """Scale for a layer-R shift injected at layer W: median residual norm at W / at R (1 if W == R)."""
    return 1.0 if W == R else med[W] / med[R]


# ------------------------------------------------------------- two-pass gating


def hard_gate(match, cats, thr):
    """1 where match > thr and the position is not the template head (cats != 0), else 0
    (diag_keys.gate_fn with read "hard")."""
    return ((match > thr) & (cats != 0)).to(match.dtype)


def cross_field(keys, Kst, M, cats, thr, scale):
    """Pass-1 read at layer R -> the field injected at layer W.
    keys [T, r]: the read keys of the memory-free pass; Kst [n, r]: the stored keys; M [d, r].
    Returns (field [T, d] = scale * g * M k, g [T], match [T])."""
    match = (keys @ Kst.T).max(-1).values
    g = hard_gate(match, cats, thr)
    return scale * g[:, None] * (keys @ M.T), g, match


def extend_field(field, T):
    """field [T0, d] -> [T, d] (T >= T0): positions past the prompt repeat the last row (no new user
    text, so the pooled key, the gate and the recall stay those of the last prompt position)."""
    T0 = field.shape[0]
    assert T >= T0, (T, T0)
    return field if T == T0 else torch.cat([field, field[-1:].expand(T - T0, -1)])


def add_fields(fields, W, f):
    """Accumulate field f into fields[W] (several reads may target one injection layer)."""
    fields[W] = fields[W] + f if W in fields else f
    return fields


class RowField:
    """Full-sequence pass-2 hook (seahorse.residual.inject): h [B, T, d] <- h + field [B, T, d]."""

    def __init__(self, field):
        self.field = field

    def read(self, h, _alpha):
        assert h.shape[:2] == self.field.shape[:2], (tuple(h.shape), tuple(self.field.shape))
        return h + self.field.to(h.dtype)


class XInject:
    """KV-cached pass-2 hook at layer `layer`: the prefill (the whole prompt) gets the pass-1 field [T, d],
    every later one-token step its last row (no new user text: the pooled key, gate and recall stay those
    of the last prompt position). begin(prompt) before each generation. `last` = gate info for logs."""

    def __init__(self, layer, field, info=None):
        self.layer, self.field, self.last, self.prefilled = layer, field, info, False

    def begin(self, ids):
        assert len(ids) == self.field.shape[0], "field / prompt length mismatch"
        self.prefilled = False

    def read(self, h, _alpha):
        if not self.prefilled:
            assert h.shape[1] == self.field.shape[0], "prefill must be the whole prompt"
            self.prefilled = True
            return h + self.field[None].to(h.dtype)
        assert h.shape[1] == 1, "after the prefill, one token per step"
        return h + self.field[-1].to(h.dtype)


# ------------------------------------------------------------ Stage-A pick rule


def ranks(vals, higher=True):
    """1 = best; ties (and nan, the worst) share their mean rank (think_v1.ranks)."""
    k = [(1, 0.0) if (v is None or math.isnan(v)) else (0, -v if higher else v) for v in vals]
    order = sorted(range(len(vals)), key=lambda i: k[i])
    out, i = [0.0] * len(vals), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and k[order[j + 1]] == k[order[i]]:
            j += 1
        for t in range(i, j + 1):
            out[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return out


PICK_RULE = ("Stage-A pick rule: candidates = grid cells (R, W, alpha) NOT already in Stage B's fixed set ({fixed}) "
             "whose mean target specificity gain dspec (related + paraphrase prompts) is > 0 (if fewer than {n} "
             "qualify: every non-fixed cell). score = rank(dspec, higher better) + rank(drel_bal, the balanced yes/no "
             "margin gain, higher better) + rank(dmg, KL on unrelated prompts with the cell's vector forced on, lower "
             "better); lowest score wins, ties -> lower dmg. Picks are taken best-first with distinct (R, W) pairs.")


def pick_cells(cells, fixed, n=2):
    """cells: [{R, W, alpha, dspec, drel_bal, dmg}] (macro means). fixed: set of (R, W, alpha) already in
    Stage B. Returns (picks, ranking): the n best cells with distinct (R, W) under PICK_RULE."""
    cand = [c for c in cells if (c["R"], c["W"], c["alpha"]) not in fixed]
    elig = [c for c in cand if not math.isnan(c["dspec"]) and c["dspec"] > 0]
    if len(elig) < n:
        elig = cand
    rk = [ranks([c["dspec"] for c in elig]), ranks([c["drel_bal"] for c in elig]),
          ranks([c["dmg"] for c in elig], higher=False)]
    ranking = []
    for j, c in enumerate(elig):
        ranking.append({**c, "score": sum(r[j] for r in rk)})
    big = float("inf")
    ranking.sort(key=lambda c: (c["score"], c["dmg"] if not math.isnan(c["dmg"]) else big))
    picks, pairs = [], set()
    for c in ranking:
        if (c["R"], c["W"]) in pairs:
            continue
        picks.append(c)
        pairs.add((c["R"], c["W"]))
        if len(picks) == n:
            break
    return picks, ranking


# ------------------------------------------------------------------- text metrics


def rep_rate(ids, n=4):
    """Share of repeated n-grams (1 - distinct / total); 0 for answers shorter than n."""
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def has_word(text, w):
    """Whole-word, case-insensitive match (think_v1.has_word): "Pepper's" counts, "Peppers" does not."""
    return bool(re.search(rf"(?<![\w-]){re.escape(w.strip())}(?![\w-])", text, re.I))


def clean_hit(text, ids, target, loop_n=4, loop_thr=0.3):
    """(hit, loop, clean): hit = the exact target word is in the answer; loop = repeated-n-gram rate >=
    loop_thr; clean = hit and not loop (a "Petra Petra Petra ..." flood is a hit but not a clean one)."""
    hit = has_word(text, target)
    loop = rep_rate(ids, loop_n) >= loop_thr
    return hit, loop, hit and not loop


def parse_yn(text):
    """The first standalone yes / no in the text (lowercase), or None."""
    m = re.search(r"\b(yes|no)\b", text, re.I)
    return m.group(1).lower() if m else None


_ID_VERBS = {"am", "work", "worked", "grew", "studied", "speak", "was", "live", "support", "love", "like"}
_POSS = {"my", "mine"}


def _words(sentence):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", sentence.lower().replace("’", "'"))


def self_attr(text, target):
    """ROUGH regex check: the target appears in a first-person claim, e.g. "my favourite colour is teal",
    "I am a pharmacist", "my dog Pepper", "Pepper is my dog". Within one sentence (split at . ! ? and
    newlines): "my"/"mine" up to 5 words before the target, or an identity phrase ("I'm", "I am", "I work",
    "I grew", "I studied", "I speak", "I love", ...) up to 4 words before it, or "my"/"mine" up to 3 words
    after it. False positives: drafts written in the user's voice ("my dog Pepper" in a note to the vet),
    "Petra, my dear" ...; misses: anything phrased otherwise."""
    tgt = target.strip().lower()
    for sent in re.split(r"[.!?\n]+", text):
        w = _words(sent)
        tj = [j for j, x in enumerate(w) if x == tgt or x.startswith(tgt + "'")]
        if not tj:
            continue
        poss = [i for i, x in enumerate(w) if x in _POSS]
        ident = [i for i, x in enumerate(w) if x == "i'm" or (x == "i" and i + 1 < len(w) and w[i + 1] in _ID_VERBS)]
        for j in tj:
            if any(0 < j - i <= 5 for i in poss) or any(0 < j - i <= 4 for i in ident) or any(0 < i - j <= 3 for i in poss):
                return True
    return False


def balanced_acc(rows):
    """rows: [(consistent "Yes"|"No", parsed "yes"|"no"|None)]. Unparsed = wrong. Returns accY, accN and
    bal = their mean (nan if a side has no rows)."""
    out = {}
    for side in ("Yes", "No"):
        g = [p == side.lower() for c, p in rows if c == side]
        out[f"acc{side[0]}"] = sum(g) / len(g) if g else float("nan")
    out["bal"] = (out["accY"] + out["accN"]) / 2
    return out


def nanmean(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")
