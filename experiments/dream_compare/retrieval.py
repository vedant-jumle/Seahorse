#!/usr/bin/env python
"""dream_compare text retriever (Stage 0): Okapi BM25 over turns, pure Python (no new dependency, no model download).

  tokens   lower-cased [a-z0-9]+ runs, a short English stop list removed, Porter-stemmed (the vendored nltk stemmer)
  score    sum over query terms t of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl)),
           idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))   (Lucene's non-negative idf); k1 = 1.5, b = 0.75
  top_m    the m best turns, ties broken by turn order; returned in time order when in_time_order (paradigms P2-P4)
  recall@m share of the item's labelled evidence turns (or sessions: PersonaMem) among the retrieved

  python -I experiments/dream_compare/retrieval.py --bench longmemeval --split oracle --m 3 10   # recall table
"""

import argparse
import importlib.util
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PORTER = _load("dream_compare_retrieval_porter", HERE / "porter_nltk.py").PorterStemmer()
K1, B = 1.5, 0.75
STOP = frozenset("""a an and are as at be but by did do does for from had has have i if in is it its me my of on or our
so that the their them they this to was we were what when where which who why will with you your""".split())
_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text, stem=True):
    w = [t for t in _TOK.findall((text or "").lower()) if t not in STOP]
    return [PORTER.stem(t) for t in w] if stem else w


class BM25:
    def __init__(self, docs, k1=K1, b=B, stem=True):
        """docs: list of strings."""
        self.k1, self.b, self.stem = k1, b, stem
        self.tf = [Counter(tokenize(d, stem)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.N = len(docs)
        self.avgdl = (sum(self.len) / self.N) if self.N else 0.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.post[t].append((i, f))

    def scores(self, query):
        s = [0.0] * self.N
        if not self.N:
            return s
        for t in set(tokenize(query, self.stem)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.post[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avgdl or 1.0))
                s[i] += idf * f * (self.k1 + 1) / denom
        return s

    def top(self, query, m):
        sc = self.scores(query)
        return sorted(range(self.N), key=lambda i: (-sc[i], i))[:m]


_INDEX = {}


def index_for(bench, item, roles=None):
    """(turns, BM25) of an item's visible context, cached per (bench, context, visible length, roles)."""
    key = (id(bench), item.context_id, item.context_turns, tuple(roles) if roles else None)
    if key not in _INDEX:
        turns = [t for t in bench.item_turns(item) if roles is None or t.role in roles]
        _INDEX[key] = (turns, BM25([t.text for t in turns]))
    return _INDEX[key]


def retrieve(bench, item, m, roles=None, in_time_order=True):
    """The item's top-m turns of its visible context (roles: restrict, e.g. ("user",)) by BM25 on the question:
    [turn ids] (in time order if in_time_order, else by score)."""
    turns, bm = index_for(bench, item, roles)
    idx = bm.top(item.question, m)
    if in_time_order:
        idx = sorted(idx)
    return [turns[i].turn_id for i in idx]


def recall(retrieved, evidence):
    """Share of evidence ids among retrieved (None when there is no evidence)."""
    ev = list(dict.fromkeys(evidence))
    if not ev:
        return None
    r = set(retrieved)
    return sum(e in r for e in ev) / len(ev)


def item_recall(bench, item, retrieved):
    """recall@m against the item's evidence turns, or its evidence sessions when it has no turn labels (PersonaMem)."""
    if item.evidence_turns:
        return recall(retrieved, item.evidence_turns)
    if item.evidence_sessions:
        so = bench.context(item).session_of()
        return recall([so[t] for t in retrieved], item.evidence_sessions)
    return None


def recall_table(bench, ms=(3, 10), roles=None):
    """{m: {category: mean recall@m, "all": mean}} over the items with evidence."""
    out = {}
    for m in ms:
        acc = defaultdict(list)
        for it in bench.items:
            r = item_recall(bench, it, retrieve(bench, it, m, roles))
            if r is not None:
                acc[it.category].append(r)
                acc["all"].append(r)
        out[m] = {k: (sum(v) / len(v), len(v)) for k, v in sorted(acc.items())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, choices=["longmemeval", "locomo", "personamem", "beam"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--m", type=int, nargs="+", default=[3, 10])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(HERE / "benchmarks"))
    PC = _load("dc_retr_prep_common", HERE / "benchmarks" / "prep_common.py")
    b = PC.load_split(a.bench, a.split)
    if a.limit:
        b.items = b.items[:a.limit]
    for m, row in recall_table(b, a.m).items():
        print(f"recall@{m}: " + "; ".join(f"{k} {v:.3f} (n {n})" for k, (v, n) in row.items()))


if __name__ == "__main__":
    main()
