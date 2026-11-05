#!/usr/bin/env python
"""LoCoMo (Maharana et al., ACL 2024; CC BY-NC 4.0) -> the common schema. Standard library only.

Source: data/locomo10.json of github.com/snap-research/locomo (commit 3eb6f2c, 2024-08-13): 10 conversations between
two speakers, 1,986 questions (1,540 non-adversarial + 446 adversarial). The raw file is untrusted data (JSON only).

Category codes (verified 2026-11-05; the paper's numbered list in section 4.1 does NOT follow the codes): from the
official scorer task_eval/evaluation.py (code 1 = sub-answer F1 "multi-hop", code 5 = "adversarial"), gpt_utils.py
(code 2 gets "Use DATE of CONVERSATION to answer with an approximate date" = temporal), the questions themselves (code 3
= "would ... likely" inference questions, code 4 = direct single-session facts) and the repo's issues #6 / #29 (also
MemMachine's and arXiv:2511.21726's appendices):
    1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial

Speaker reframing (a chat model's memory has two roles): speaker_a -> "user", speaker_b -> "assistant"; the speaker's
name stays in the text ("Caroline: ..."), since the questions name both people. Image turns get their BLIP caption
appended (the dataset's blip_caption field, as LoCoMo's own context builder does: "... and shared <caption>."):
"Caroline: <text> [shared an image: <caption>]". Session dates (session_<n>_date_time) are kept; sessions listed
without dialog (dates only) are skipped. Turn ids "<sample_id>:<dia_id>".

Adversarial questions (code 5): abstention = True, answers = [] (the right reply says it is not mentioned); the
distractor is in meta["adversarial_answer"] (the raw key; 2 of them also carry an "answer"). Evidence ids that do not
name a turn of the conversation (a few malformed entries) are dropped and listed in meta["evidence_dropped"].
"""

import re

try:
    from . import schema as S
except ImportError:
    import importlib.util
    from pathlib import Path
    _s = importlib.util.spec_from_file_location("dc_bench_schema", Path(__file__).with_name("schema.py"))
    S = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(S)

NAME = "locomo"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
GITHUB = "https://github.com/snap-research/locomo"
COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
RAW_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{COMMIT}/data/locomo10.json"
_EV = re.compile(r"D\d+:\d+")


def turn_text(t):
    s = f"{t['speaker']}: {t['text']}"
    if t.get("blip_caption"):
        s += f" [shared an image: {t['blip_caption']}]"
    return s


def convert(convs, split="locomo10"):
    contexts, items = {}, []
    for c in convs:
        sid = c["sample_id"]
        cv = c["conversation"]
        a, b = cv["speaker_a"], cv["speaker_b"]
        nums = sorted(int(k.split("_")[1]) for k in cv if re.fullmatch(r"session_\d+", k))
        sessions = []
        for n in nums:
            date = cv.get(f"session_{n}_date_time")
            turns = []
            for t in cv[f"session_{n}"]:
                role = "user" if t["speaker"] == a else "assistant"
                assert t["speaker"] in (a, b), (sid, t["speaker"])
                turns.append(S.Turn(f"{sid}:{t['dia_id']}", role, turn_text(t), date, t["speaker"],
                                    {"raw": t["text"], **({"caption": t["blip_caption"]} if t.get("blip_caption") else {})}))
            if turns:
                sessions.append(S.Session(f"{sid}:S{n}", date, turns))
        contexts[sid] = S.Context(sid, sessions, {"speaker_a": a, "speaker_b": b, "role_map": {a: "user", b: "assistant"}})
        known = {t.turn_id for s in sessions for t in s.turns}
        for qi, q in enumerate(c["qa"]):
            code = int(q["category"])
            ev, dropped = [], []
            for e in q.get("evidence", []) or []:
                for x in (_EV.findall(e) or [e]):
                    (ev if f"{sid}:{x}" in known else dropped).append(f"{sid}:{x}" if f"{sid}:{x}" in known else x)
            ev = list(dict.fromkeys(ev))
            sess = list(dict.fromkeys(f"{sid}:S{x.split(':')[1][1:]}" for x in ev))
            adv = code == 5
            meta = {}
            if adv:
                meta["adversarial_answer"] = q.get("adversarial_answer")
                if "answer" in q:
                    meta["answer_field"] = str(q["answer"])
            if dropped:
                meta["evidence_dropped"] = dropped
            items.append(S.Item(f"{sid}:q{qi}", NAME, sid, q["question"], [] if adv else [str(q["answer"])],
                                CATEGORIES[code], abstention=adv, evidence_turns=ev, evidence_sessions=sess,
                                category_code=code, meta=meta))
    return S.Bench(NAME, split, contexts, items, {"source": GITHUB, "commit": COMMIT, "category_map": CATEGORIES,
                                                  "speaker_frame": "speaker_a -> user, speaker_b -> assistant"})
