#!/usr/bin/env python
"""PersonaMem v1 (Jiang et al., 2025, "Know Me, Respond to Me"; MIT), 32k split -> the common schema. Standard library.

Source: huggingface.co/datasets/bowen-upenn/PersonaMem-v1: questions_32k.csv (589 multiple-choice questions over 20
personas) and shared_contexts_32k.jsonl (37 shared contexts: {context id: [chat messages]}). Untrusted data: csv and
json parsing only.

Contexts: one per shared context id; a new session starts at every "system" message (the persona prompt opens each
session); the system message is kept as the session's first turn (role "system"). Turn text: the leading "User: " /
"Assistant: " of the raw messages is dropped (the role carries it); turn ids "<context id[:12]>:<message index>".
Items: context_turns = end_index_in_shared_context (the visible context is context[:end_index], as the README says);
options = all_options with their "(a) " labels removed (the raw labels in meta), correct_option = the index of
correct_answer ("(c)" -> 2); all_options is JSON in some rows and a
Python-literal list in others (parsed with ast.literal_eval, literals only). Evidence: PersonaMem labels no turns; evidence_sessions = the session holding the most
recent preference mention, `distance_to_ref_in_blocks` sessions back from the question (session index
n_visible_sessions - distance), with the raw distance fields in meta. Scored by option log-probability (no judge).
"""

import ast
import csv
import io
import json
import re

try:
    from . import schema as S
except ImportError:
    import importlib.util
    from pathlib import Path
    _s = importlib.util.spec_from_file_location("dc_bench_schema", Path(__file__).with_name("schema.py"))
    S = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(S)

NAME = "personamem"
HF_REPO = "bowen-upenn/PersonaMem-v1"
HF_REV = "a8076d5608c93ba2a28983cd78aa99b01a163ae7"
FILES = {"questions": "questions_32k.csv", "contexts": "shared_contexts_32k.jsonl"}
_LABEL = re.compile(r"^\(([a-z])\)\s*")
_PREFIX = {"user": "User: ", "assistant": "Assistant: "}


def strip_role(role, text):
    p = _PREFIX.get(role)
    return text[len(p):] if p and text.startswith(p) else text


def parse_options(text):
    """all_options: a JSON list in 286 rows, a Python-literal list (single quotes) in 303: json, else
    ast.literal_eval (literals only, nothing executed). Must be a list of strings."""
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        v = ast.literal_eval(text)
    assert isinstance(v, list) and all(isinstance(x, str) for x in v), text[:80]
    return v


def read_contexts(text):
    out = {}
    for line in text.splitlines():
        if line.strip():
            d = json.loads(line)
            assert isinstance(d, dict) and len(d) == 1
            out.update(d)
    return out


def convert(questions_csv, contexts_jsonl, split="32k"):
    """questions_csv / contexts_jsonl: the raw file contents (str)."""
    raw = read_contexts(contexts_jsonl)
    contexts, n_strip = {}, 0
    for cid, msgs in raw.items():
        short = cid[:12]
        sessions, cur, k = [], [], 0
        for i, m in enumerate(msgs):
            if m["role"] == "system" and cur:
                sessions.append(S.Session(f"{short}:s{k}", None, cur))
                cur, k = [], k + 1
            text = strip_role(m["role"], m["content"])
            n_strip += text != m["content"]
            cur.append(S.Turn(f"{short}:{i}", m["role"], text, None, None, {"session": k}))
        if cur:
            sessions.append(S.Session(f"{short}:s{k}", None, cur))
        contexts[cid] = S.Context(cid, sessions, {"n_messages": len(msgs)})
    items = []
    for r in csv.DictReader(io.StringIO(questions_csv)):
        cid = r["shared_context_id"]
        c = contexts[cid]
        end = int(r["end_index_in_shared_context"])
        vis = c.turns(end)
        vis_sessions = list(dict.fromkeys(t.meta["session"] for t in vis))
        dist = int(r["distance_to_ref_in_blocks"])
        ev_s = []
        if 1 <= dist <= len(vis_sessions):
            ev_s = [c.sessions[vis_sessions[len(vis_sessions) - dist]].session_id]
        opts_raw = parse_options(r["all_options"])
        labels = [(_LABEL.match(o).group(1) if _LABEL.match(o) else None) for o in opts_raw]
        opts = [_LABEL.sub("", o, count=1) for o in opts_raw]
        corr = _LABEL.match(r["correct_answer"].strip() + " ")
        ci = labels.index(corr.group(1)) if corr and corr.group(1) in labels else None
        assert ci is not None, (r["question_id"], r["correct_answer"], labels)
        items.append(S.Item(r["question_id"], NAME, cid, r["user_question_or_message"], [opts[ci]], r["question_type"],
                            evidence_sessions=ev_s, context_turns=end, options=opts, correct_option=ci,
                            meta={"persona_id": r["persona_id"], "topic": r["topic"], "labels": labels,
                                  "distance_to_ref_in_blocks": dist,
                                  "distance_to_ref_in_tokens": int(r["distance_to_ref_in_tokens"]),
                                  "context_length_in_tokens": int(r["context_length_in_tokens"])}))
    return S.Bench(NAME, split, contexts, items, {"source": f"https://huggingface.co/datasets/{HF_REPO}", "revision": HF_REV,
                                                  "role_prefix_stripped": n_strip})
