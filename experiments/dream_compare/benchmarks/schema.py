#!/usr/bin/env python
"""dream_compare benchmarks: the common item schema (standard library only).

A benchmark is a set of contexts (the memory: sessions of turns) and items (questions over one context):

  Turn     turn_id (unique in the benchmark), role (user | assistant | system), text, date (the session's date
           string, as the benchmark gives it), speaker (the benchmark's own speaker name, if any), meta
  Session  session_id, date, turns
  Context  context_id, sessions (in time order), meta
  Item     item_id, bench, context_id, context_turns (None = the whole context; PersonaMem: the first n turns of the
           shared context, its end_index), question, question_date, answers (list; [] for abstention items that have
           none), category (the benchmark's own type name), category_code (its raw code, e.g. LoCoMo's 1-5),
           abstention (True = the right answer is to say it is not known), evidence_turns (turn ids), evidence_sessions
           (session ids), options + correct_option (multiple choice: PersonaMem), rubric (BEAM), meta

Serialised as one JSON file per split: {"meta": {...}, "contexts": [...], "items": [...]} (to_json / from_json).
validate() checks the invariants every adapter must meet (unique ids, evidence pointing at real turns / sessions,
roles, non-empty text, multiple-choice consistency, abstention items) and returns the problems as strings.
"""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field

ROLES = ("user", "assistant", "system")


@dataclass
class Turn:
    turn_id: str
    role: str
    text: str
    date: str | None = None
    speaker: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    date: str | None
    turns: list


@dataclass
class Context:
    context_id: str
    sessions: list
    meta: dict = field(default_factory=dict)

    def turns(self, limit=None):
        out = [t for s in self.sessions for t in s.turns]
        return out if limit is None else out[:limit]

    def session_of(self):
        return {t.turn_id: s.session_id for s in self.sessions for t in s.turns}


@dataclass
class Item:
    item_id: str
    bench: str
    context_id: str
    question: str
    answers: list
    category: str
    abstention: bool = False
    evidence_turns: list = field(default_factory=list)
    evidence_sessions: list = field(default_factory=list)
    context_turns: int | None = None
    question_date: str | None = None
    category_code: str | int | None = None
    options: list | None = None
    correct_option: int | None = None
    rubric: list | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Bench:
    name: str
    split: str
    contexts: dict
    items: list
    meta: dict = field(default_factory=dict)

    def context(self, item):
        return self.contexts[item.context_id]

    def item_turns(self, item):
        """The turns visible to an item (PersonaMem: the shared context cut at the item's end index)."""
        return self.contexts[item.context_id].turns(item.context_turns)

    def counts(self):
        c = Counter(it.category for it in self.items)
        return {"items": len(self.items), "contexts": len(self.contexts), "by_category": dict(sorted(c.items())),
                "abstention": sum(it.abstention for it in self.items)}


def to_json(b):
    return {"meta": dict(b.meta, name=b.name, split=b.split),
            "contexts": [{"context_id": c.context_id, "meta": c.meta,
                          "sessions": [{"session_id": s.session_id, "date": s.date,
                                        "turns": [asdict(t) for t in s.turns]} for s in c.sessions]}
                         for c in b.contexts.values()],
            "items": [asdict(it) for it in b.items]}


def from_json(d):
    ctx = {}
    for c in d["contexts"]:
        ctx[c["context_id"]] = Context(c["context_id"], [Session(s["session_id"], s["date"], [Turn(**t) for t in s["turns"]])
                                                         for s in c["sessions"]], c.get("meta", {}))
    m = dict(d["meta"])
    return Bench(m.pop("name"), m.pop("split"), ctx, [Item(**it) for it in d["items"]], m)


def save(b, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(b), f, ensure_ascii=False)


def load(path):
    with open(path, encoding="utf-8") as f:
        return from_json(json.load(f))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(b):
    """Problems (strings) of a Bench; [] = valid."""
    bad = []
    seen_t, seen_i = set(), set()
    sess_ids = {}
    for cid, c in b.contexts.items():
        if cid != c.context_id:
            bad.append(f"context key {cid} != id {c.context_id}")
        if not c.sessions:
            bad.append(f"{cid}: no sessions")
        ids = set()
        for s in c.sessions:
            if s.session_id in ids:
                bad.append(f"{cid}: duplicate session {s.session_id}")
            ids.add(s.session_id)
            for t in s.turns:
                if t.turn_id in seen_t:
                    bad.append(f"duplicate turn id {t.turn_id}")
                seen_t.add(t.turn_id)
                if t.role not in ROLES:
                    bad.append(f"{t.turn_id}: role {t.role!r}")
                if not isinstance(t.text, str) or not t.text.strip():
                    bad.append(f"{t.turn_id}: empty text")
        sess_ids[cid] = ids
    turn_ctx = {t.turn_id: cid for cid, c in b.contexts.items() for t in c.turns()}
    for it in b.items:
        if it.item_id in seen_i:
            bad.append(f"duplicate item {it.item_id}")
        seen_i.add(it.item_id)
        if it.bench != b.name:
            bad.append(f"{it.item_id}: bench {it.bench} != {b.name}")
        if it.context_id not in b.contexts:
            bad.append(f"{it.item_id}: unknown context {it.context_id}")
            continue
        if not it.question.strip():
            bad.append(f"{it.item_id}: empty question")
        if not isinstance(it.answers, list) or any(not isinstance(a, str) for a in it.answers):
            bad.append(f"{it.item_id}: answers must be a list of strings")
        if not it.abstention and not it.answers and it.options is None and not it.rubric:
            bad.append(f"{it.item_id}: no answer, options or rubric")
        n_vis = len(b.contexts[it.context_id].turns()) if it.context_turns is None else it.context_turns
        visible = {t.turn_id for t in b.contexts[it.context_id].turns(n_vis)}
        if it.context_turns is not None and not (0 < it.context_turns <= len(b.contexts[it.context_id].turns())):
            bad.append(f"{it.item_id}: context_turns {it.context_turns} out of range")
        for e in it.evidence_turns:
            if turn_ctx.get(e) != it.context_id:
                bad.append(f"{it.item_id}: evidence turn {e} not in its context")
            elif e not in visible:
                bad.append(f"{it.item_id}: evidence turn {e} after the question")
        for e in it.evidence_sessions:
            if e not in sess_ids[it.context_id]:
                bad.append(f"{it.item_id}: evidence session {e} not in its context")
        if (it.options is None) != (it.correct_option is None):
            bad.append(f"{it.item_id}: options and correct_option go together")
        if it.options is not None and not (0 <= it.correct_option < len(it.options) and len(it.options) >= 2):
            bad.append(f"{it.item_id}: correct_option {it.correct_option} of {len(it.options)} options")
    return bad
