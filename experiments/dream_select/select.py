"""dream_select pure helpers (no model loading). Unit-tested in test_dream_select.py.

Question (PREREG.md): can the positions of a stored K,V slice be chosen automatically, without hand-picking, on natural
benchmark text (LongMemEval single-session-user)?

None of the selection helpers here takes the benchmark question or answer: they see the stored turn's token ids, its
attention statistics or the dreamed questions only. The gold helpers (gold_span, gold_at_k) are the labelled reference
and are named as such.

NOTE: this file is named select.py as the plan asks; it shadows the stdlib `select` module if its folder is on
sys.path. run.py / analyze.py load it by path under another module name and drop their own folder from sys.path.
"""

import math
import re

import numpy as np

# names a selector may never take (test_dream_select checks every registered selector's signature)
FORBIDDEN_ARGS = frozenset({"question", "answer", "item", "gold", "label", "labels", "q", "a", "it", "span", "target"})
BUDGETS = ("k2", "k4", "k8", "k10")


def budget_k(name, n_content):
    """k for a budget name: k2 / k4 / k8, or k10 = max(2, round(0.10 * n_content)); capped at n_content."""
    k = max(2, int(round(0.10 * n_content))) if name == "k10" else int(name[1:])
    return min(k, n_content)


# ------------------------------------------------------------------------------ token positions


def char_spans(tok, ids):
    """[(start, end)] character span of every token of the incremental decode, and the decoded text (as dream_subj)."""
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


def content_positions(tok, ids, text):
    """Indices into `ids` (a chat turn's token ids) of the tokens that spell `text`, the turn's content: every token
    whose character span lies inside the (unique) occurrence of `text`. Template and role tokens are excluded. Asserts
    that no token straddles the content's boundary and that the content tokens are contiguous and cover it exactly."""
    spans, dtext = char_spans(tok, ids)
    e0 = dtext.find(text)
    assert e0 >= 0 and dtext.find(text, e0 + 1) < 0, "content not found exactly once in the turn"
    e1 = e0 + len(text)
    inside = [i for i, (a, b) in enumerate(spans) if a >= e0 and b <= e1 and b > a]
    straddle = [i for i, (a, b) in enumerate(spans) if (a < e0 < b) or (a < e1 < b)]
    assert not straddle, f"tokens straddle the content boundary: {straddle}"
    assert inside == list(range(inside[0], inside[-1] + 1)), "content tokens are not contiguous"
    assert spans[inside[0]][0] == e0 and spans[inside[-1]][1] == e1, "content tokens do not cover the content"
    return inside


def keep_indices(n_head, sel):
    """Kept harvest-sequence indices of a slice: the head + the selected turn positions (indices into the turn)."""
    return list(range(n_head)) + [n_head + int(s) for s in sorted(set(int(x) for x in sel))]


def check_kept(keep, n_head, k, content):
    """A selection's kept entries: the head intact + exactly k entries, every one a content position (fails the run)."""
    assert keep[:n_head] == list(range(n_head)), "the head is not kept intact"
    rest = keep[n_head:]
    assert len(rest) == k, f"kept {len(rest)} entries, want {k}"
    assert rest == sorted(set(rest)), "kept indices not sorted / unique"
    cs = {n_head + c for c in content}
    bad = [i for i in rest if i not in cs]
    assert not bad, f"kept entries that are not content positions: {bad}"


# ------------------------------------------------------------------------------ top-k and baselines


def topk(scores, content, k):
    """The k content positions with the highest score (scores: one per content position, in order). Ties go to the
    earlier position; NaN counts as the lowest. Returns the picked positions, sorted."""
    assert len(scores) == len(content)
    k = min(k, len(content))
    s = [(-math.inf if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)) for x in scores]
    order = sorted(range(len(content)), key=lambda i: (-s[i], i))
    return sorted(content[i] for i in order[:k])


def last_k(content, k):
    return sorted(content[-min(k, len(content)):])


def random_k(content, k, seed):
    """k content positions drawn without replacement; `seed` is a tuple of ints (fixed per seed, item, budget)."""
    rng = np.random.default_rng(list(seed))
    k = min(k, len(content))
    return sorted(int(content[i]) for i in rng.choice(len(content), size=k, replace=False))


# ------------------------------------------------------------------------------ attention scores


def attn_received(A, content):
    """S-attn: A = [T, T] attention of the harvest pass, already averaged over heads and layers (rows = queries).
    Score of content position j = the mean attention it receives from the LATER content positions (mean, not sum, to
    avoid H2O's early-token bias). The last content position has no later query: NaN (ranked lowest)."""
    A = np.asarray(A, dtype=np.float64)
    out = []
    for n, j in enumerate(content):
        later = content[n + 1:]
        out.append(float(A[later, j].mean()) if later else float("nan"))
    return out


def query_mean(A, queries, cols):
    """Mean over the query rows `queries` of A[q, c] for each column c (A already averaged over heads and layers)."""
    A = np.asarray(A, dtype=np.float64)
    return [float(A[list(queries), c].mean()) for c in cols]


def dream_combine(per_question):
    """S-dream / S-dream-split: the sum over the dreamed questions of their per-position scores."""
    assert per_question, "no dreamed question"
    return [float(sum(v)) for v in zip(*per_question)]


# ------------------------------------------------------------------------------ dreamed questions


_ENUM = re.compile(r"^\s*(?:\(?\d+[.):]\)?|[-*•]|Q\d*[.:])\s*", re.I)


def parse_dreams(text, n=3):
    """The dreamer's output -> (questions, strict_ok, lines). Lines are stripped of list markers ('1.', '-', '*');
    questions = the non-empty lines that end in '?'. strict_ok = exactly n non-empty lines, all ending in '?'. If more
    than n question lines are found, the first n are kept."""
    lines = [_ENUM.sub("", ln).strip().strip('"').strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    qs = [ln for ln in lines if ln.endswith("?")]
    strict = len(lines) == n and len(qs) == n
    return qs[:n], strict, lines


# ------------------------------------------------------------------------------ gold (labelled reference only)


def gold_span(tok, ids, content, answer):
    """GOLD (uses the label): the content positions covering the first case-insensitive occurrence of the answer
    string inside the turn's content, or None if it is not there verbatim. `ids` = the turn's token ids, `content`
    = content_positions(...)."""
    spans, dtext = char_spans(tok, ids)
    c0, c1 = spans[content[0]][0], spans[content[-1]][1]
    a = str(answer).strip()
    i = dtext.lower().find(a.lower(), c0, c1)
    if i < 0 or not a:
        return None
    j = i + len(a)
    toks = [p for p in content if spans[p][0] < j and spans[p][1] > i]
    return toks or None


def gold_plus_next(span, content):
    """`gold`: the answer's tokens plus the next content token (if any)."""
    last = content.index(span[-1])
    return sorted(set(span) | ({content[last + 1]} if last + 1 < len(content) else set()))


def gold_at_k(span, content, k):
    """`gold@k`: the answer's first token, the next token and then the nearest preceding tokens, to fill k (then
    later tokens if the turn's start is reached)."""
    k = min(k, len(content))
    f = content.index(span[0])
    pick = [f] + ([f + 1] if f + 1 < len(content) else [])
    pick = pick[:k]
    lo, hi = f - 1, (pick[-1] + 1)
    while len(pick) < k:
        if lo >= 0:
            pick.append(lo)
            lo -= 1
        else:
            pick.append(hi)
            hi += 1
    return sorted(content[i] for i in pick)


def hit(sel, span):
    """Does the selection contain at least one token of the gold span?"""
    return bool(span) and bool(set(sel) & set(span))


# ------------------------------------------------------------------------------ measures


def answer_logprob(lp, Lq, ans):
    """Sum and per-token mean of log P(answer token | everything before it). lp: [T, V] log-probs (or an indexable of
    rows) of the sequence prompt + answer; Lq = the prompt's length; ans = the answer ids."""
    vals = [float(lp[Lq - 1 + j][int(ans[j])]) for j in range(len(ans))]
    return sum(vals), sum(vals) / len(vals)


def gap_closed(cond, nomem, full, min_den=0.5):
    """(cond - nomem) / (full - nomem), or None when full - nomem < min_den nats (the item is dropped from this
    measure) or any value is not finite."""
    if any(x is None or not math.isfinite(x) for x in (cond, nomem, full)):
        return None
    den = full - nomem
    if den < min_den:
        return None
    return (cond - nomem) / den
