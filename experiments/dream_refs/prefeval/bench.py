"""PrefEval benchmark for experiment 2 (dream_refs): pure helpers, no model. PREREG.md "Addendum: PrefEval".

  load_rows          the 1000 explicit-preference rows (upstream/prefeval_explicit.jsonl)
  polarity           the pre-registered lexical label: avoid | like | mixed | other (the dislike / avoid subset)
  stratified_sample  the pre-registered stratified random sample (topic x polarity, proportional, fixed seed)
  check_followup / select_followups   the dreamer's neutral follow-ups and their checks against the test question
  parse_answer_tag / parse_acknowledge  PrefEval's judge-output parsing (the <answer> / <preference> tags)
  pref_labels        PrefEval's per-answer error types and preference-following accuracy (their formula)
  majority           the 3-judge majority of a yes/no label (a tie, when a judge is missing, goes to the original)

Unit-tested in tests/test_dream_refs_prefeval.py.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "upstream" / "prefeval_explicit.jsonl"
DATA_SHA256 = "826cd8b8cd15a10875c6e0bd9ab81e4374b175b5b730a923def5d829d8e0ef0f"


def load_rows(path=DATA):
    return [json.loads(l) for l in open(path) if l.strip()]


def item_id(row):
    return f"pe{row['row']:04d}"


# ------------------------------------------------------------------------------------------- polarity

_W = r"(?<![\w'-]){}(?![\w'-])"
# avoidance / dislike cues; "prefer not" and "rather not" are avoidance, never a liking
NEG_CUES = ["avoid", "avoids", "avoiding", "dislike", "dislikes", "hate", "hates", "detest", "can't stand",
            "cannot stand", "can't", "cannot", "don't", "do not", "doesn't", "does not", "won't", "not", "never", "no",
            "nothing", "none", "aversion", "averse", "allergic", "allergy", "allergies", "intolerance", "intolerant",
            "refuse", "refuses", "against", "ineffective", "uncomfortable", "stay away", "steer clear", "without",
            "prefer not", "rather not", "sensitive to", "phobia", "fear of", "afraid of", "dread", "tired of",
            "sick of", "turned off", "turn me off", "bothers me", "annoy", "annoys", "annoying", "irritate",
            "irritates", "distaste", "loathe", "loathes", "free of", "despise", "despises", "boring", "tedious",
            "off-putting"]
POS_CUES = ["prefer", "prefers", "preferred", "preference", "love", "loves", "enjoy", "enjoys", "like", "likes", "favor",
            "favour", "favorite", "favourite", "fan of", "passionate", "interested in", "want", "wants", "seek",
            "only", "keen", "fond of", "appreciate", "drawn to", "into"]


def _hits(text, cues):
    t = text.lower().replace("’", "'")
    return [c for c in cues if re.search(_W.format(re.escape(c)), t)]


NEGATED_POS = re.compile(r"(?:(?:prefer|rather)\s+not|(?:don't|do not|doesn't|does not|didn't|won't|not|never|can't|"
                         r"cannot|no longer)\s+(?:\w+\s+)?(?:like|likes|enjoy|enjoys|want|wants|love|loves|care for|"
                         r"interested in|into|fond of|keen on|a fan of))")


def polarity(text):
    """Pre-registered lexical label of a preference: 'avoid' (avoidance / dislike cues only), 'like' (liking cues
    only), 'mixed' (both), 'other' (neither). A negated liking ("don't like", "not interested in", "prefer not")
    is avoidance, not a liking; a "<thing>-free" diet or product is avoidance."""
    t = text.lower().replace("’", "'")
    t_pos = NEGATED_POS.sub(" ", t)
    neg = bool(_hits(t, NEG_CUES)) or bool(re.search(r"\w-free\b", t))
    pos = bool(_hits(t_pos, POS_CUES))
    return "mixed" if neg and pos else ("avoid" if neg else ("like" if pos else "other"))


# ------------------------------------------------------------------------------------------- sampling


def largest_remainder(sizes, n):
    """Proportional allocation of n over strata {key: size} (Hamilton): floor of the quota, then the largest
    remainders (ties by key order) get one more. Never more than a stratum's size."""
    total = sum(sizes.values())
    quota = {k: n * s / total for k, s in sizes.items()}
    alloc = {k: min(int(q), sizes[k]) for k, q in quota.items()}
    left = n - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(quota[k] - int(quota[k])), k)):
        if left <= 0:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            left -= 1
    assert sum(alloc.values()) == n, (n, alloc)
    return alloc


def stratified_sample(rows, n, seed, strata=lambda r: (r["topic"], polarity(r["preference"]))):
    """The pre-registered sample: strata = topic x polarity, proportional allocation (largest_remainder), and
    within a stratum random.Random(f"{seed}|{stratum}").sample of its rows in row order. Returns the chosen rows
    sorted by row index, and the allocation."""
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["row"]):
        by[strata(r)].append(r)
    alloc = largest_remainder({k: len(v) for k, v in by.items()}, n)
    out = []
    for k in sorted(by):
        rng = random.Random(f"{seed}|{'/'.join(k)}")
        out += rng.sample(by[k], alloc[k])
    return sorted(out, key=lambda r: r["row"]), {"/".join(k): v for k, v in sorted(alloc.items())}


# ------------------------------------------------------------------------------------------ follow-ups

_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
REQUEST_WORDS = {"recommend", "recommends", "recommendation", "recommendations", "suggest", "suggests", "suggestion",
                 "suggestions", "advice", "advise", "tip", "tips", "idea", "ideas"}
SENTIMENT = {"like", "likes", "liked", "love", "loves", "loved", "enjoy", "enjoys", "hate", "hates", "dislike",
             "dislikes", "prefer", "prefers", "preferred", "preference", "avoid", "avoids", "favourite", "favorite",
             "fan", "allergic", "allergy", "stand", "fond", "keen"}
MAX_JACCARD_Q = 0.25


def proper_nouns(text):
    """Capitalised words that are not the first word of a sentence and not 'I' (names, places, brands)."""
    out = set()
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        ws = re.findall(r"[A-Za-z][\w'-]*", sent)
        out |= {w.lower() for w in ws[1:] if w[0].isupper() and w not in ("I", "I'm", "I've", "I'd", "I'll")}
    return out


def check_followup(f, question, key_words, refs):
    """Hard checks for one dreamed follow-up (all must pass): 4-25 words, a statement (no '?'), not a request
    (no recommend / suggest / advice / tip / idea), no liking / disliking word, no negation, none of the
    preference's key words (stems); against the TEST question: content-word Jaccard <= MAX_JACCARD_Q and no
    shared proper noun. refs: the dream_refs refs module (tokenisation, stems, negation)."""
    t = refs.norm(f)
    toks = _TOK.findall(t.lower())
    cw, qw = refs.content_words(t), refs.content_words(question)
    jac = len(cw & qw) / len(cw | qw) if cw | qw else 0.0
    hard = {"length": 4 <= len(t.split()) <= 25,
            "statement": "?" not in t,
            "not_request": not (set(toks) & REQUEST_WORDS),
            "neutral": not (set(toks) & SENTIMENT),
            "no_negation": refs.neg_count(t) == 0,
            "no_key_word": not refs.mentions_key(t, key_words),
            "lexical_q": jac <= MAX_JACCARD_Q,
            "no_shared_name_q": not (proper_nouns(t) & proper_nouns(question))}
    return {"text": t, "ok": all(hard.values()), "hard": hard, "jaccard_q": jac}


def select_followups(sample_texts, question, key_words, refs, keep=3):
    """Candidates over the dreamer's samples ranked by how many samples propose them, then first appearance
    (refs._rank); the `keep` best that pass every check, as sentences. Returns (kept, info)."""
    ranked = refs._rank(sample_texts)
    kept, fails = [], Counter()
    for c in ranked:
        chk = check_followup(c["text"], question, key_words, refs)
        if chk["ok"]:
            if len(kept) < keep:
                kept.append({"text": refs.as_sentence(chk["text"]), "count": c["count"], "jaccard_q": chk["jaccard_q"]})
        else:
            fails.update(k for k, v in chk["hard"].items() if not v)
    return kept, {"n_candidates": len(ranked), "n_kept": len(kept), "fail_reasons": dict(fails)}


# ------------------------------------------------------------------------------------ judge parsing

ANS = re.compile(r"<answer>\s*\[?\s*(yes|no)\b[^<]*</answer>", re.I | re.S)
ANS_OPEN = re.compile(r"<answer>\s*\[?\s*(yes|no)\b", re.I)
PREF = re.compile(r"<preference>(.*?)</preference>", re.I | re.S)


def _strip_think(text):
    return re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text or "", flags=re.S | re.I)


def parse_answer_tag(text):
    """PrefEval's <answer>Yes/No</answer> -> True (yes) / False (no) / None (unparsed). A closed tag is preferred;
    an unclosed tag (output cut at the token limit) is accepted. The LAST answer tag counts."""
    t = _strip_think(text)
    m = ANS.findall(t) or ANS_OPEN.findall(t)
    return None if not m else m[-1].lower() == "yes"


def parse_acknowledge(text):
    """check_acknowledge.txt -> (acknowledged True/False/None, the quoted preference sentence or "")."""
    t = _strip_think(text)
    m = PREF.findall(t)
    return parse_answer_tag(t), (m[-1].strip() if m else "")


def pref_labels(acknow, violate, helpful, hallucinate):
    """PrefEval's per-answer error types (get_preference_following_accuracy_generation_task.analyze_errors):
    inputs are the parsed yes/no booleans (None = unparsed, treated as PrefEval treats a missing 'yes' / 'no').
    accuracy = not any(inconsistent, hallucination-of-preference violation, preference-unaware violation,
    unhelpful), which is helpful and not violating."""
    is_ack = acknow is True
    is_hall = is_ack and hallucinate is True
    is_viol = violate is True
    is_unhelp = helpful is False
    inconsistent = is_ack and not is_hall and is_viol and not is_unhelp
    hall_viol = is_ack and is_hall and is_viol and not is_unhelp
    unaware = (not is_ack) and is_viol and not is_unhelp
    acc = not any([inconsistent, hall_viol, unaware, is_unhelp])
    return {"acknowledgement": is_ack, "hallucination": is_hall, "violation": is_viol, "unhelpful": is_unhelp,
            "inconsistent": inconsistent, "hallucination_violation": hall_viol, "unaware_violation": unaware,
            "accuracy": acc}


def majority(labels, original):
    """Majority of yes/no labels (None = missing); a tie goes to the original judge's label."""
    v = [x for x in labels if x is not None]
    y, n = sum(x is True for x in v), sum(x is False for x in v)
    return True if y > n else (False if n > y else original)


def make_sample(cfg_path):
    """Write the pre-registered sample (config `sample`) to sample.json: the chosen rows with id and polarity,
    the allocation per stratum, and the data file's sha256 (asserted equal to DATA_SHA256)."""
    import hashlib
    import yaml
    repo = HERE.parent.parent.parent
    cfg = yaml.safe_load(open(cfg_path))
    path = repo / cfg["data"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == DATA_SHA256, f"{path}: sha256 {sha} != {DATA_SHA256}"
    rows = load_rows(path)
    s = cfg["sample"]
    chosen, alloc = stratified_sample(rows, s["n"], s["seed"])
    items = [{"id": item_id(r), "row": r["row"], "topic": r["topic"], "polarity": polarity(r["preference"]),
              "preference": r["preference"], "question": r["question"]} for r in chosen]
    out = {"n": s["n"], "seed": s["seed"], "data_sha256": sha, "rule": "strata topic x polarity (bench.polarity), "
           "proportional allocation by largest remainder, random.Random(f'{seed}|{topic}/{polarity}').sample within a "
           "stratum (rows in row order)", "allocation": alloc,
           "polarity_counts": dict(Counter(i["polarity"] for i in items)),
           "population_polarity_counts": dict(Counter(polarity(r["preference"]) for r in rows)), "items": items}
    json.dump(out, open(repo / s["file"], "w"), indent=1)
    return out


if __name__ == "__main__":
    import sys
    o = make_sample(sys.argv[1] if len(sys.argv) > 1 else HERE / "config.yaml")
    print(json.dumps({k: v for k, v in o.items() if k not in ("items", "allocation")}, indent=1))
