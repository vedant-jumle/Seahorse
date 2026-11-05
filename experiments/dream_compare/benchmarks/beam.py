#!/usr/bin/env python
"""BEAM ("Beyond a Million Tokens", Tavakoli et al., 2025, arXiv:2510.27246; data CC BY-SA 4.0, code MIT), 100K split
-> the common schema. Standard library only (the parquet is read by prep_beam.py, which needs pyarrow; this module takes
the rows as dicts).

Source: huggingface.co/datasets/Mohammadta/BEAM, data/100K-00000-of-00001.parquet: 20 conversations, 20 probing
questions each over 10 types. Summarisation and instruction-following are dropped (plan), leaving 8 types x 40 = 320
questions. probing_questions is a Python-literal string in the file: parsed with ast.literal_eval (literals only,
nothing executed). Untrusted data.

Contexts: one per conversation; chat is a list of sessions (each a list of turns with a global id, role, content,
time_anchor on some user turns). Turn ids "beam<conversation id>:<id>"; session date = the session's first time_anchor.
BEAM appends a marker " ->-> <i>,<j>" to some user turns (the plan index); it is removed from the text and kept in
meta["marker"]. Items: answer = "answer" (or "ideal_answer" for contradiction_resolution, "ideal_response" for
abstention); preference_following has none (rubric only); rubric = the item's rubric list (BEAM's judge scores each
rubric item); evidence_turns = source_chat_ids (a list, or a dict of lists for knowledge_update / temporal_reasoning),
ids that do not exist are dropped (meta["evidence_dropped"]); abstention = the abstention type.
"""

import ast
import re

try:
    from . import schema as S
except ImportError:
    import importlib.util
    from pathlib import Path
    _s = importlib.util.spec_from_file_location("dc_bench_schema", Path(__file__).with_name("schema.py"))
    S = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(S)

NAME = "beam"
HF_REPO = "Mohammadta/BEAM"
HF_REV = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
FILE = "data/100K-00000-of-00001.parquet"
SHA256 = "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace"
DROP = ("summarization", "instruction_following")
KEEP = ("abstention", "contradiction_resolution", "event_ordering", "information_extraction", "knowledge_update",
        "multi_session_reasoning", "preference_following", "temporal_reasoning")
_MARK = re.compile(r"\s*->->\s*(\d+\s*,\s*\d+)\s*$")


def _ids(x):
    if isinstance(x, dict):
        return [i for v in x.values() for i in _ids(v)]
    if isinstance(x, (list, tuple)):
        return [i for v in x for i in _ids(v)]
    try:
        return [int(x)]
    except (TypeError, ValueError):
        return []


def answer_of(qtype, q):
    if qtype == "contradiction_resolution":
        return q.get("ideal_answer")
    if qtype == "abstention":
        return q.get("ideal_response")
    return q.get("answer")


def convert(rows, split="100K"):
    """rows: the parquet rows as dicts (conversation_id, chat, probing_questions, ...)."""
    contexts, items = {}, []
    for r in rows:
        cid = f"beam{r['conversation_id']}"
        sessions = []
        for si, sess in enumerate(r["chat"]):
            date = next((t.get("time_anchor") for t in sess if t.get("time_anchor")), None)
            turns = []
            for t in sess:
                text = t["content"]
                m = _MARK.search(text)
                meta = {"index": t.get("index"), "question_type": t.get("question_type")}
                if m:
                    meta["marker"] = m.group(1)
                    text = text[:m.start()]
                turns.append(S.Turn(f"{cid}:{t['id']}", t["role"], text, date, None, meta))
            if turns:
                sessions.append(S.Session(f"{cid}:s{si}", date, turns))
        seed = r.get("conversation_seed") or {}
        contexts[cid] = S.Context(cid, sessions, {"category": seed.get("category"), "title": seed.get("title")})
        known = {t.turn_id for s in sessions for t in s.turns}
        tsess = {t.turn_id: s.session_id for s in sessions for t in s.turns}
        pq = ast.literal_eval(r["probing_questions"])
        for qtype in KEEP:
            for qi, q in enumerate(pq.get(qtype, [])):
                ev, dropped = [], []
                for i in _ids(q.get("source_chat_ids")):
                    (ev if f"{cid}:{i}" in known else dropped).append(f"{cid}:{i}" if f"{cid}:{i}" in known else i)
                ev = list(dict.fromkeys(ev))
                a = answer_of(qtype, q)
                rub = [str(x) for x in (q.get("rubric") or [])]
                meta = {k: v for k, v in q.items() if k not in ("question", "rubric", "answer", "ideal_answer",
                                                                 "ideal_response", "source_chat_ids")}
                meta = {k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
                        for k, v in meta.items()}
                if dropped:
                    meta["evidence_dropped"] = dropped
                items.append(S.Item(f"{cid}:{qtype}:{qi}", NAME, cid, q["question"], [str(a)] if a else [], qtype,
                                    abstention=qtype == "abstention", evidence_turns=ev,
                                    evidence_sessions=list(dict.fromkeys(tsess[e] for e in ev)), rubric=rub,
                                    meta=meta))
    return S.Bench(NAME, split, contexts, items, {"source": f"https://huggingface.co/datasets/{HF_REPO}",
                                                  "revision": HF_REV, "dropped_types": list(DROP)})
