#!/usr/bin/env python
"""dream_select data: the 70 single-session-user items of LongMemEval (cleaned oracle split) -> data/longmemeval/ssu70.json.

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (longmemeval_oracle.json; MIT licence;
Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory", ICLR 2025). The downloaded file
is untrusted data: this script only parses it as JSON (run it with `python -I`).

Per item: question_id, question, answer, the evidence turns (user turns with has_answer == True, at most 2), whether
the answer appears verbatim (case-insensitive) in an evidence turn, and the stored turn: the first evidence turn that
contains the answer verbatim, else the first evidence turn (`stored_index` records which).

Two facts about the file that the plan did not expect (asserted, recorded):
  - 6 of the 70 are abstention items (question_id ending '_abs'; the answer is "You did not mention this information
    ..."); they flag no turn. They are kept in the extract with usable = False and are not run.
  - 2 items also flag the assistant reply right after the user's evidence turn (it echoes the fact). Those are listed
    under assistant_flagged and never stored: the memory is the user's turn.

  python -I experiments/dream_select/prep_data.py data/external/longmemeval/longmemeval_oracle.json data/longmemeval/ssu70.json
"""

import hashlib
import json
import sys


def verbatim(answer, text):
    return str(answer).strip().lower() in text.lower()


def extract(items):
    out = []
    for it in items:
        if it["question_type"] != "single-session-user":
            continue
        ev, ev_asst = [], []
        for si, sess in enumerate(it["haystack_sessions"]):
            for ti, turn in enumerate(sess):
                if turn.get("has_answer"):
                    if turn["role"] == "user":
                        ev.append({"session": si, "turn": ti, "content": turn["content"]})
                    else:
                        # 2 items also flag the assistant's reply that echoes the fact; recorded, never stored
                        # (the memory is a user turn), and each must directly follow a flagged user turn
                        assert turn["role"] == "assistant" and ev and ev[-1]["session"] == si and ev[-1]["turn"] == ti - 1, \
                            (it["question_id"], si, ti, turn["role"])
                        ev_asst.append({"session": si, "turn": ti})
        ans = str(it["answer"])
        abstention = it["question_id"].endswith("_abs")
        # the 6 abstention items ('_abs': the answer is "You did not mention this information ...") flag no turn
        assert (len(ev) == 0) == abstention, (it["question_id"], len(ev))
        assert len(ev) <= 2, (it["question_id"], len(ev))
        hits = [verbatim(ans, e["content"]) for e in ev]
        k = hits.index(True) if any(hits) else 0
        out.append({"question_id": it["question_id"], "question": it["question"], "answer": ans,
                    "abstention": abstention, "usable": not abstention,
                    "evidence": [{**e, "verbatim": h} for e, h in zip(ev, hits)], "n_evidence": len(ev),
                    "assistant_flagged": ev_asst,
                    "verbatim": any(hits), "stored_index": k if ev else None,
                    "stored_turn": ev[k]["content"] if ev else None})
    return out


def main(src, dst):
    raw = open(src, "rb").read()
    items = json.loads(raw)
    out = extract(items)
    assert len(out) == 70, len(out)
    meta = {"source": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
            "source_sha256": hashlib.sha256(raw).hexdigest(), "licence": "MIT",
            "citation": "Wu et al., LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory, ICLR 2025",
            "n": len(out), "n_usable": sum(x["usable"] for x in out),
            "n_abstention": sum(x["abstention"] for x in out),
            "n_verbatim": sum(x["verbatim"] for x in out),
            "n_two_evidence": sum(x["n_evidence"] == 2 for x in out),
            "n_assistant_flagged": sum(bool(x["assistant_flagged"]) for x in out)}
    json.dump({"meta": meta, "items": out}, open(dst, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
