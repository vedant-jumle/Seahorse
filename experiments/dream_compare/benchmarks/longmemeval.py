#!/usr/bin/env python
"""LongMemEval (Wu et al., ICLR 2025; MIT) -> the common schema (benchmarks/schema.py). Standard library only.

Source: huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (longmemeval_oracle.json: 500 items, only the evidence
sessions; longmemeval_s_cleaned.json: the same 500 questions with ~115k-token haystacks). Raw files are untrusted data:
parsed as JSON only.

Per item: one context (its own haystack). Empty turns (some S distractor sessions) are dropped and counted;
a session repeated inside one haystack (S) gets "#2", "#3" suffixes. Sessions are put in time order (haystack_dates, "2023/04/10 (Mon) 17:50";
the oracle file lists them unordered); each session's original index is kept in Session-level turn meta. Turn ids
"<question_id>/<session_id>:<turn index>". Evidence turns = the turns with has_answer (user or assistant); evidence
sessions = answer_session_ids. category = question_type (6 types); abstention = question_id ending "_abs" (30 items;
their answer is LongMemEval's explanation, used by its abstention judge prompt). Answers are strings (32 integer answers
are str()-ed).
"""

import random
import re
from collections import defaultdict

try:
    from . import schema as S
except ImportError:                        # loaded by path (importlib) from the tests / prep scripts
    import importlib.util
    from pathlib import Path
    _s = importlib.util.spec_from_file_location("dc_bench_schema", Path(__file__).with_name("schema.py"))
    S = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(S)

NAME = "longmemeval"
TYPES = ("single-session-user", "single-session-assistant", "single-session-preference", "multi-session",
         "knowledge-update", "temporal-reasoning")
HF_REPO = "xiaowu0162/longmemeval-cleaned"
HF_REV = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
FILES = {"oracle": "longmemeval_oracle.json", "s": "longmemeval_s_cleaned.json"}
SHA256 = {"oracle": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
          "s": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"}   # = the hub's LFS oids
S_SAMPLE_PER_TYPE = 20
S_SAMPLE_SEED = 7

_DATE = re.compile(r"\s*\([A-Za-z]{3}\)\s*")


def date_key(d):
    """'2023/04/10 (Mon) 17:50' -> '2023/04/10 17:50' (sorts chronologically)."""
    return _DATE.sub(" ", d or "").strip()


def convert(items, split):
    """Raw LongMemEval items (a list of dicts) -> Bench."""
    contexts, out = {}, []
    n_empty = n_dup = 0
    for it in items:
        qid = it["question_id"]
        order = sorted(range(len(it["haystack_sessions"])), key=lambda i: (date_key(it["haystack_dates"][i]), i))
        sessions, ev, seen, by_sid = [], [], {}, {}
        for i in order:
            sid = it["haystack_session_ids"][i]
            seen[sid] = seen.get(sid, 0) + 1
            usid = sid if seen[sid] == 1 else f"{sid}#{seen[sid]}"     # the S haystacks repeat a few sessions
            n_dup += seen[sid] > 1
            date = it["haystack_dates"][i]
            turns = []
            for ti, t in enumerate(it["haystack_sessions"][i]):
                tid = f"{qid}/{usid}:{ti}"
                if not (t["content"] or "").strip():                  # empty distractor turns (S): dropped, counted
                    n_empty += 1
                    assert not t.get("has_answer"), tid
                    continue
                turns.append(S.Turn(tid, t["role"], t["content"], date, None, {"orig_session_index": i}))
                if t.get("has_answer"):
                    ev.append(tid)
            if turns:
                sessions.append(S.Session(f"{qid}/{usid}", date, turns))
                by_sid.setdefault(sid, []).append(f"{qid}/{usid}")
        contexts[qid] = S.Context(qid, sessions, {"n_sessions": len(sessions)})
        abst = qid.endswith("_abs")
        out.append(S.Item(qid, NAME, qid, it["question"], [str(it["answer"])], it["question_type"], abstention=abst,
                          evidence_turns=ev,
                          evidence_sessions=[u for s in it.get("answer_session_ids", []) for u in by_sid.get(s, [])],
                          question_date=it.get("question_date")))
    return S.Bench(NAME, split, contexts, out, {"source": f"https://huggingface.co/datasets/{HF_REPO}", "revision": HF_REV,
                                                "empty_turns_dropped": n_empty, "repeated_sessions_renamed": n_dup})


def stratified_ids(items, per_type=S_SAMPLE_PER_TYPE, seed=S_SAMPLE_SEED):
    """The S-split sample: per question_type, `per_type` question ids drawn with random.Random(seed) from the sorted ids
    of that type (types in TYPES order, one generator for all)."""
    by = defaultdict(list)
    for it in items:
        by[it["question_type"]].append(it["question_id"])
    rng = random.Random(seed)
    out = []
    for t in TYPES:
        ids = sorted(by[t])
        assert len(ids) >= per_type, (t, len(ids))
        out += sorted(rng.sample(ids, per_type))
    return out


def single_turn_ids(bench, user_only=False):
    """Items whose labelled evidence is exactly one turn (Stage A's pool: one stored turn per question). user_only:
    count only user turns (dream_select's rule: 2 single-session-user items also flag the assistant's echo; with
    user_only the single-session-user count is dream_select's 64)."""
    out = []
    for it in bench.items:
        ev = it.evidence_turns
        if user_only:
            roles = {t.turn_id: t.role for t in bench.contexts[it.context_id].turns()}
            ev = [e for e in ev if roles[e] == "user"]
        if len(ev) == 1:
            out.append(it.item_id)
    return out
