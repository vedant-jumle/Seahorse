"""Disposition rate: lexicon scoring of sampled generations."""

import re


def _pattern(word):
    # whole word/phrase, case-insensitive, optional plural s/es; flexible inner whitespace
    body = r"\s+".join(re.escape(w) for w in word.split())
    return re.compile(rf"(?<![\w-]){body}(?:e?s)?(?![\w-])", re.IGNORECASE)


def lexicon_hits(text, lexicon):
    """Count consistent/inconsistent matches in `text`. Overlapping matches are resolved
    in favour of the longest one, so "oat milk" beats "milk" and "not spicy" beats "spicy"."""
    spans = []
    for side in ("consistent", "inconsistent"):
        for w in lexicon[side]:
            spans += [(m.start(), m.end(), side) for m in _pattern(w).finditer(text)]
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    taken, counts = [], {"consistent": 0, "inconsistent": 0}
    for a, b, side in spans:
        if all(b <= x or a >= y for x, y in taken):
            taken.append((a, b))
            counts[side] += 1
    return counts


def label(text, lexicon):
    """consistent | inconsistent | neutral, by which side has more hits."""
    c = lexicon_hits(text, lexicon)
    if c["consistent"] > c["inconsistent"]:
        return "consistent"
    if c["inconsistent"] > c["consistent"]:
        return "inconsistent"
    return "neutral"


def disposition_rate(generations, lexicon):
    """Fractions of generations labelled consistent / inconsistent / neutral, plus
    lean = consistent - inconsistent (in [-1, 1]) and n."""
    labels = [label(g, lexicon) for g in generations]
    n = len(labels)
    out = {k: (labels.count(k) / n if n else float("nan"))
           for k in ("consistent", "inconsistent", "neutral")}
    out["lean"] = out["consistent"] - out["inconsistent"]
    out["n"] = n
    return out
