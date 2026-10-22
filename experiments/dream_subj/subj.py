"""dream_subj pure helpers (no model loading). Unit-tested in test_dream_subj.py.

Question (PREREG.md): when a stored K,V slice of the experience turn is put back, does answering a direct question
need the SUBJECT's own position ("dog" in "My dog's name is Pepper."), the entity-tracking two-step lookup, or does
any extra stored position help equally ("more evidence")?

  slice_positions   token indices, inside the stored experience turn, of the subject's last token, the relation
                    token ("name"), the filler token right before the value ("is"), the value's first token and
                    the next token
  keep_sets         per condition, the harvested cache entries kept (head + chosen positions, at their stored
                    positions); R-min must equal dream_bind's bind.splice_keep("R-min", ...)
  verdict           the pre-registered rule on paired-difference CIs
"""

import math
import re

# conditions with a stored slice, and how many experience-turn entries each keeps (besides the head)
R_CONDS = ("R-full", "R-min", "R+subj", "R+fill", "R+rel", "R-subj2")
N_KEPT = {"R-min": 2, "R+subj": 3, "R+fill": 3, "R+rel": 3, "R-subj2": 2}   # R-full: the whole turn (n_seg)
# which slice positions each condition needs (a fact without one of them skips that condition)
NEEDS = {"R-full": (), "R-min": ("value", "next"), "R+subj": ("subj", "value", "next"),
         "R+fill": ("fill", "value", "next"), "R+rel": ("rel", "value", "next"), "R-subj2": ("subj", "value")}
CONDS = ("nomem", "ctx", "T-min", "T-subj3") + R_CONDS


def char_spans(tok, ids):
    """[(start, end)] character span of every token of the incremental decode, and the decoded text."""
    ids = [int(i) for i in ids]
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ends = [len(dec(ids[:i + 1])) for i in range(len(ids))]
    text = dec(ids)
    assert ends and ends[-1] == len(text), "decode is not prefix-stable"
    spans, prev = [], 0
    for e in ends:
        assert e >= prev, "decode is not monotone"
        spans.append((prev, e))
        prev = e
    return spans, text


def _overlap(spans, s, e):
    return [i for i, (a, b) in enumerate(spans) if a < e and b > s]


def find_alias(text, aliases, start, end):
    """The first alias (list order; canonical first) occurring as whole words in text[start:end], case-insensitive:
    (alias, char_start, char_end) or None. Same matching rule as dream_bind's bind.find_alias."""
    for a in aliases:
        m = re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", text[start:end], re.I)
        if m:
            return a, start + m.start(), start + m.end()
    return None


def slice_positions(tok, seg, experience, value, aliases, value_tok=None):
    """Positions inside the stored experience turn `seg` (token ids; the experience text occurs exactly once):
      subj    the last token of the first alias occurrence inside the experience sentence ("dog", "friend")
      value   the value's first token (dream_bind: the target's first token, found exactly once in seg)
      next    value + 1 (the full stop, or the value's second sub-token)
      fill    value - 1 if that token is "is" (else None)
      rel     the last token between subj and value whose text is "name" (else None)
    Asserts: subj < value, subj is not the value, the alias span lies inside the experience, the value is found
    exactly once. Returns a dict with the indices, the alias found and each token's text."""
    spans, text = char_spans(tok, seg)
    e0 = text.find(experience)
    assert e0 >= 0 and text.find(experience, e0 + 1) < 0, f"experience not found exactly once: {experience!r}"
    e1 = e0 + len(experience)
    v = value.strip()
    if value_tok is not None:
        hits = [i for i, x in enumerate(seg) if int(x) == int(value_tok)]
        assert len(hits) == 1, f"value token found {len(hits)} times in the experience turn"
        a = hits[0]
    else:
        vi = experience.rfind(v)
        assert vi >= 0, f"value {v!r} not in {experience!r}"
        a = _overlap(spans, e0 + vi, e0 + vi + 1)[0]
    assert spans[a][0] >= e0 and spans[a][1] <= e1 + 1, "value token outside the experience"
    hit = find_alias(text, aliases, e0, e1)
    assert hit is not None, f"no subject alias of {aliases} in {experience!r}"
    alias, s, e = hit
    toks = _overlap(spans, s, e)
    subj = toks[-1]
    assert spans[subj][1] <= e + 1, "subject token runs past the alias"
    assert subj < a and subj != a, f"subject {subj} not before the value {a}"
    dec = lambda i: text[spans[i][0]:spans[i][1]]
    fill = a - 1 if a - 1 > subj and dec(a - 1).strip().lower() == "is" else None
    rels = [i for i in range(subj + 1, a) if dec(i).strip().lower() == "name"]
    rel = rels[-1] if rels else None
    assert a + 1 < len(seg)
    out = {"subj": subj, "value": a, "next": a + 1, "fill": fill, "rel": rel, "alias": alias,
           "n_seg": len(seg), "subj_span": toks}
    out["text"] = {k: (dec(out[k]) if out[k] is not None else None) for k in ("subj", "value", "next", "fill", "rel")}
    return out


def keep_sets(n_head, n_seg, pos):
    """{condition: (kept indices into the harvest sequence [head][experience turn], question start)} for every
    stored-slice condition whose positions exist; the question turn always starts at n_head + n_seg (where R-full's
    would). Missing conditions (no 'is' / no 'name') are left out; the caller records them as skipped."""
    head = list(range(n_head))
    start = n_head + n_seg
    out = {"R-full": (head + list(range(n_head, n_head + n_seg)), start)}
    for c in N_KEPT:
        need = NEEDS[c]
        if any(pos.get(k) is None for k in need):
            continue
        idx = sorted({n_head + pos[k] for k in need})
        assert len(idx) == len(need), (c, idx)
        out[c] = (head + idx, start)
    return out


def check_counts(keeps, n_head, n_seg):
    """Every condition keeps the head plus exactly the specified number of experience-turn entries (fails the run)."""
    for c, (k, start) in keeps.items():
        want = n_seg if c == "R-full" else N_KEPT[c]
        assert len(k) == n_head + want, f"{c}: kept {len(k) - n_head} entries, want {want}"
        assert k[:n_head] == list(range(n_head)), f"{c}: the head is not kept intact"
        assert all(n_head <= i < n_head + n_seg for i in k[n_head:]), f"{c}: entry outside the experience turn"
        assert k == sorted(set(k)), f"{c}: kept indices not sorted / unique"
        assert start == n_head + n_seg, f"{c}: question start {start} != {n_head + n_seg}"


def text_subj3(pos):
    """T-subj3: the subject's last token, the value's first token and the next token, as plain text (one prior user
    turn), e.g. 'dog Pepper.'."""
    t = pos["text"]
    return f"{t['subj'].strip()} {t['value'].lstrip()}{t['next']}"


# ------------------------------------------------------------------------------ teacher-forced scores


def prefix_score(lp, Lq, pre):
    """Sum over the measure-prefix tokens of log P(prefix token | everything before it). lp: [T, V] log-probs of the
    sequence question-turn + prefix; Lq = the question part's length; pre = the prefix ids."""
    return sum(float(lp[Lq - 1 + j, int(pre[j])]) for j in range(len(pre)))


def balanced_acc(margins):
    """[(consistent 'Yes'|'No', log P(Yes) - log P(No))] -> (accYes + accNo) / 2 with 'yes' = margin > 0."""
    acc = {}
    for side in ("Yes", "No"):
        g = [(m > 0) == (side == "Yes") for c, m in margins if c == side]
        acc[side] = sum(g) / len(g) if g else float("nan")
    return (acc["Yes"] + acc["No"]) / 2


# ------------------------------------------------------------------------------ rule


def _has0(s):
    return s["lo"] <= 0 <= s["hi"]


def verdict(d_min, d_fill, d_rel, f_min=None, r_min=None):
    """The pre-registered rule (PREREG.md), on the direct question's answer score over every included person/pet
    fact. Arguments are paired-difference CI dicts {mean, lo, hi}:
      d_min   R+subj - R-min       d_fill  R+subj - R+fill       d_rel  R+subj - R+rel
      f_min   R+fill - R-min       r_min   R+rel - R-min          (for "neither")
    Returns (label, reasons)."""
    if any(x is None or math.isnan(x["mean"]) for x in (d_min, d_fill, d_rel)):
        return "undetermined", ["a contrast is missing"]
    why = []
    if d_min["lo"] > 0 and d_fill["lo"] > 0 and d_rel["lo"] > 0:
        return "subject position matters (two-step lookup)", ["all three paired CIs have lower bound > 0"]
    if d_min["lo"] > 0:
        small = [n for n, d in (("R+fill", d_fill), ("R+rel", d_rel))
                 if _has0(d) and abs(d["mean"]) < 0.25 * d_min["mean"]]
        if small:
            return "more evidence", [f"R+subj - R-min > 0; R+subj - {', '.join(small)} CI contains 0 with |point| < "
                                     f"0.25 x (R+subj - R-min)"]
        why.append("R+subj - R-min > 0, but neither other contrast is small and includes 0")
    three = [d_min, f_min, r_min]
    if all(x is not None and not math.isnan(x["mean"]) for x in three) and all(_has0(x) for x in three):
        return "neither", ["no 3-position condition differs from R-min (all three CIs include 0)"]
    return "in between", why or ["no rule fires"]
