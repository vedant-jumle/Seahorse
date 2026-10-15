"""dream_refs pure helpers (no model, no GPU): parsing the dreamer's outputs, the validators (frame template,
opposite, alternatives, paraphrases, null-frame values, valence topics), the deterministic selection rules,
the S1 null references, the valence sentences, the sign readers, the S1 / S3 decision rules and the
concept x sign shifts. Unit-tested in tests/test_dream_refs.py. PREREG.md states every rule implemented here.

Conventions
  frame     the experience with its VALUE replaced by the slot "{X}" (the dreamer writes it). A sentence
            "fits" a frame if it equals the frame with some non-empty filler in the slot (case-insensitive,
            whitespace-flexible, final . ! ? ignored).
  key words the 1-3 words of the value that name it (the dreamer writes them). They are "the original's
            lexicon words": no alternative and no null-frame value may contain one.
  negation  not / no / never / none / nobody / nothing / neither / nor / cannot / any word ending in n't.
            "No new negation" = the reference has at most as many negation words as the experience.
"""

import json
import math
import re
from collections import Counter

SLOT = "{X}"
NEG_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "nope"}
# function words, and the preference verbs / intensifiers every frame shares (never a value's key word)
STOP = set("""a an the i i'm im me my mine myself you your we our they their he she it its this that these those
and or but so if as at by for from in into of on onto to with about than then there here just also too very
really much quite truly absolutely totally completely simply honestly actually always usually often every
each all any some is am are was were be been being have has had do does did will would can could should may
might must shall get gets got getting love loves loved like likes liked enjoy enjoys hate hates hated adore
prefer prefers dislike dislikes stand can't don't doesn't won't isn't not no never way by-the-way know just
huge big great fan fans lot lots right now these days thing things""".split())
FRAMES_DEFAULT = [["I really like {X}.", "I can't stand {X}."], ["{X} is my favourite.", "{X} is the worst."],
                  ["I always choose {X}.", "I always avoid {X}."]]
TRUE_SIGN = {"leanings": 1, "one_of_many": 1, "dislikes": -1}
SIGNED_GROUPS = ("leanings", "dislikes")   # S3 is decided on these 23 items


# ----------------------------------------------------------------------------------------------- text


def norm(s):
    """Unify quotes and dashes, strip bullets / numbering / surrounding quotes, collapse whitespace."""
    s = (s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").strip()
    s = re.sub(r"^\s*(?:[-*•]+|\(?\d{1,2}[.):]|[a-z][.)])\s+", "", s)
    s = s.strip().strip('"').strip("'").strip("`").strip()
    return re.sub(r"\s+", " ", s)


def _strip_end(s):
    return s.rstrip().rstrip(".!?;,").rstrip()


def ckey(s):
    """Comparison key: normalised, lower case, final punctuation dropped."""
    return _strip_end(norm(s)).lower()


def tokens(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", norm(s).lower())


def neg_count(s):
    return sum(1 for t in tokens(s) if t in NEG_WORDS or t.endswith("n't"))


def stem(w):
    for suf in ("ies", "es", "s"):
        if len(w) > 4 and w.endswith(suf) and not w.endswith("ss"):
            return w[:-len(suf)] + ("y" if suf == "ies" else "")
    return w


def content_words(s):
    return {stem(t) for t in tokens(s) if t not in STOP and len(t) > 2 and not t.isdigit()}


def jaccard(a, b):
    A, B = content_words(a), content_words(b)
    return len(A & B) / len(A | B) if A | B else float("nan")


def as_sentence(s):
    """A reference as it is written before the follow-up: normalised, ending in . ! or ? (a full stop is added
    if missing, as every hand-written reference has one)."""
    s = norm(s)
    return s if (not s or s[-1] in ".!?") else s + "."


def has_phrase(text, phrase):
    return bool(re.search(rf"(?<![\w-]){re.escape(phrase.lower())}(?![\w-])", norm(text).lower()))


# --------------------------------------------------------------------------------------------- parsing


def parse_json_obj(text):
    for m in re.finditer(r"\{[^{}]*(?:\{X\}[^{}]*)*\}", text or "", re.S):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(o, dict):
            return o
    return None


def parse_json_list(text):
    for m in re.finditer(r"\[[^\[\]]*\]", text or "", re.S):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(o, list) and o and all(isinstance(x, str) for x in o):
            return [norm(x) for x in o if norm(x)]
    return None


def parse_lines(text):
    """One candidate per line; numbering, bullets and quotes stripped; header lines (ending in ':') dropped."""
    out = []
    for line in (text or "").splitlines():
        t = norm(line)
        if not t or t.endswith(":") or t.startswith("```"):
            continue
        out.append(t)
    return out


def first_line(text):
    ls = parse_lines(text)
    return ls[0] if ls else ""


# ---------------------------------------------------------------------------------------------- frames


def fill(frame, value):
    return frame.replace(SLOT, value)


def fits(frame, s):
    """The filler if `s` is the frame with some non-empty filler in the slot, else None."""
    if not frame or frame.count(SLOT) != 1:
        return None
    f, t = _strip_end(norm(frame)), _strip_end(norm(s))
    pre, post = f.split(SLOT)
    esc = lambda p: re.escape(p).replace(r"\ ", r"\s+")
    m = re.match("^" + esc(pre) + r"(.+?)" + esc(post) + "$", t, re.I | re.S)
    return m.group(1).strip() if m and m.group(1).strip() else None


# a dropped trailing clause (prompts v3) must start with one of these: punctuation or a consequence / reason word
DROP_PUNCT = (",", ";", ":", "-")
DROP_WORDS = {"so", "because", "since", "which", "as", "therefore", "thus", "hence", "meaning"}
DROP_AND = {"so", "therefore", "thus", "hence", "that's", "thats", "because"}   # "and so ...", "and that's why ..."


def dropped_clause(filled, sentence):
    """prompts v3: the frame may leave out a trailing consequence / reason clause. Returns "" if the filled
    frame reproduces the sentence, the dropped clause if the sentence is the filled frame followed by a clause
    that starts with , ; : - or a consequence / reason word (so, because, since, which, as, therefore, thus,
    hence, meaning, "and so", "and therefore", "and that's why"), else None. Case, whitespace and final
    punctuation are ignored; the kept part must be a prefix of the sentence (a leading clause is never
    dropped)."""
    f, s = ckey(filled), ckey(sentence)
    f, s = re.sub(r"\s+", " ", f), re.sub(r"\s+", " ", s)
    if f == s:
        return ""
    if not f or not s.startswith(f):
        return None
    rest = s[len(f):]
    r = rest.strip()
    if not r:
        return None
    if r[0] in DROP_PUNCT and (r[0] != "-" or rest[0].isspace()):   # "well-known" is not a dropped clause
        return r.lstrip("".join(DROP_PUNCT)).strip() or None
    if not rest[0].isspace():          # the kept part ended inside a word
        return None
    w = r.split()
    if w[0] in DROP_WORDS or (w[0] == "and" and len(w) > 1 and w[1] in DROP_AND):
        return r
    return None


def check_frame(obj, sentence):
    """The dreamer's {"value", "frame", "key_words", "category"} -> (frame dict, problems). Valid iff no
    problems: one slot, the frame with the value reproduces the sentence (prompts v3: or the sentence without a
    trailing consequence / reason clause, see dropped_clause; the dropped clause is recorded), key words present
    and inside the value. The category (prompts v2) is optional: "" when absent, and then the alternatives prompt
    falls back to "the same kind of thing as <value>"."""
    probs = []
    if not isinstance(obj, dict):
        return None, ["no JSON object"]
    value = norm(str(obj.get("value", "")))
    frame = norm(str(obj.get("frame", ""))).replace("{x}", SLOT)
    kw = obj.get("key_words") or []
    kw = [norm(str(k)) for k in (kw if isinstance(kw, list) else [kw]) if norm(str(k))]
    if not value:
        probs.append("empty value")
    dropped = None
    if frame.count(SLOT) != 1:
        probs.append(f"frame has {frame.count(SLOT)} slots")
    else:
        dropped = dropped_clause(fill(frame, value), sentence)
        if dropped is None:
            probs.append("frame(value) != sentence")
    if not kw:
        probs.append("no key words")
    elif any(not has_phrase(value, k) for k in kw):
        probs.append("a key word is not in the value")
    return {"value": value, "frame": frame, "key_words": kw, "category": norm(str(obj.get("category") or "")),
            "dropped": dropped or ""}, probs


def select_frame(greedy, samples, sentence):
    """Greedy output if valid, else the first valid sample (in sample order), else None."""
    tried = []
    for src, text in [("greedy", greedy)] + [(f"sample{j + 1}", t) for j, t in enumerate(samples or [])]:
        if text is None:
            continue
        f, probs = check_frame(parse_json_obj(text), sentence)
        tried.append({"source": src, "problems": probs})
        if not probs:
            return dict(f, source=src), tried
    return None, tried


def key_word_stems(key_words):
    return set().union(*[content_words(k) for k in key_words]) if key_words else set()


def mentions_key(text, key_words, ignore=frozenset()):
    return bool(content_words(text) & (key_word_stems(key_words) - set(ignore)))


def category_stems(fillers, key_words, share=0.5):
    """Key-word stems that are the value's category, not the value itself: present in >= `share` of the
    candidate fillers (e.g. "food" in "Japanese food" when the candidates are "Italian food", "Thai food", ...).
    Never all of them: if every key stem qualifies, none is ignored, so a single key word ("jazz") is always
    checked."""
    stems = key_word_stems(key_words)
    fills = [content_words(f) for f in fillers if f]
    if not stems or not fills:
        return set()
    cat = {s for s in stems if sum(s in f for f in fills) >= share * len(fills)}
    return cat if cat != stems else set()


# -------------------------------------------------------------------------------------------- opposite


def check_opposite(opp, sentence, frame=None):
    """Hard checks (all must pass): non-empty, not the experience, a different value when it fits the frame,
    no new negation words. Soft flags (reported): same_frame (fits the frame, or shares >= half of the
    frame's fixed words), keeps_topic (mentions a key word), neg_kept (the experience had a negation and the
    opposite still has one), one_sentence."""
    o = norm(opp)
    hard = {"nonempty": bool(o), "not_experience": ckey(o) != ckey(sentence),
            "no_new_negation": neg_count(o) <= neg_count(sentence)}
    soft = {}
    if frame:
        f = fits(frame["frame"], o)
        hard["different_value"] = f is None or ckey(f) != ckey(frame["value"])
        skel = set(tokens(frame["frame"].replace(SLOT, " ")))
        share = len(skel & set(tokens(o))) / len(skel) if skel else 1.0
        soft["same_frame"] = f is not None or share >= 0.5
        soft["keeps_topic"] = mentions_key(o, frame["key_words"])
    soft["neg_kept"] = neg_count(sentence) > 0 and neg_count(o) > 0
    soft["one_sentence"] = len(re.findall(r"[.!?](?:\s|$)", o)) <= 1
    return {"text": o, "ok": all(hard.values()), "hard": hard, "soft": soft}


def select_opposite(greedy, samples, sentence, frame=None):
    """The greedy output if it passes the hard checks, else the first sample that does; else the greedy
    output, marked not ok."""
    cands = [("greedy", greedy)] + [(f"sample{j + 1}", t) for j, t in enumerate(samples or [])]
    first = None
    for src, text in cands:
        if text is None:
            continue
        c = check_opposite(first_line(text), sentence, frame)
        c["source"] = src
        first = first or c
        if c["ok"]:
            return c
    return first or {"text": "", "ok": False, "hard": {"nonempty": False}, "soft": {}, "source": "none"}


# -------------------------------------------------------------------------------------- list selection


def _rank(sample_texts, parse=parse_lines, keyfn=ckey):
    """Candidates over all samples: [{text, key, count (samples proposing it), first (index of first
    appearance)}], ranked by count (desc) then first appearance. A candidate counts once per sample."""
    seen, out, order = {}, [], 0
    for text in sample_texts:
        this = set()
        for c in parse(text) or []:
            k = keyfn(c)
            if not k:
                continue
            if k not in seen:
                seen[k] = {"text": c, "key": k, "count": 0, "first": order}
                out.append(seen[k])
                order += 1
            if k not in this:
                seen[k]["count"] += 1
                this.add(k)
    return sorted(out, key=lambda c: (-c["count"], c["first"]))


def check_alternative(alt, sentence, frame, opposite, ignore=frozenset()):
    """Hard checks: fits the frame; its value differs from the experience's and from the opposite's; the value
    contains none of the experience's key words (except `ignore`, the category stems); not the experience or the
    opposite; no new negation."""
    a = norm(alt)
    f = fits(frame["frame"], a) if frame else None
    opp_f = fits(frame["frame"], opposite) if (frame and opposite) else None
    kw = frame["key_words"] if frame else []
    hard = {"fits_frame": f is not None,
            "not_experience": ckey(a) != ckey(sentence),
            "not_opposite": ckey(a) != ckey(opposite or "") and (f is None or opp_f is None or ckey(f) != ckey(opp_f)),
            "different_value": f is None or ckey(f) != ckey(frame["value"]),
            "no_key_word": not mentions_key(f if f is not None else a, kw, ignore),
            "no_new_negation": neg_count(a) <= neg_count(sentence)}
    return {"text": a, "filler": f, "ok": all(hard.values()), "hard": hard}


def select_alternatives(sample_texts, sentence, frame, opposite, keep=8):
    """The `keep` best-ranked candidates that pass every hard check (deduplicated by their value). If fewer
    pass, the list is filled with the best-ranked candidates that fail ONLY the frame check (flag
    frame_miss). Key-word stems shared by most candidates' fillers are the category, not the value, and are not
    held against a candidate (category_stems). Returns (kept, info)."""
    ranked = _rank(sample_texts)
    fillers = [fits(frame["frame"], norm(c["text"])) for c in ranked] if frame else []
    ignore = category_stems(fillers, frame["key_words"]) if frame else set()
    valid, near, seen = [], [], set()
    for c in ranked:
        chk = check_alternative(c["text"], sentence, frame, opposite, ignore)
        k = ckey(chk["filler"]) if chk["filler"] is not None else c["key"]
        if k in seen:
            continue
        rec = {"text": chk["text"], "count": c["count"], "filler": chk["filler"], "hard": chk["hard"]}
        if chk["ok"]:
            seen.add(k)
            valid.append(rec)
        elif all(v for kk, v in chk["hard"].items() if kk != "fits_frame"):
            near.append((k, rec))
    kept = valid[:keep]
    for k, rec in near:
        if len(kept) >= keep:
            break
        if k not in seen:
            seen.add(k)
            kept.append(dict(rec, frame_miss=True))
    return kept, {"n_candidates": len(ranked), "n_valid": len(valid), "n_kept": len(kept),
                  "n_frame_miss": sum(bool(r.get("frame_miss")) for r in kept),
                  "category_stems": sorted(ignore)}


def select_paraphrases(sample_texts, sentence, key_words, keep=4, max_words=40):
    """Hard: not the experience, no new negation, <= max_words words. Ranked: those that keep every key word
    first, then by count and first appearance."""
    ranked = _rank(sample_texts)
    out = []
    for c in ranked:
        t = c["text"]
        ok = ckey(t) != ckey(sentence) and neg_count(t) <= neg_count(sentence) and 0 < len(t.split()) <= max_words
        if ok:
            out.append({"text": t, "count": c["count"],
                        "keeps_key_words": all(has_phrase(t, k) for k in key_words) if key_words else None,
                        "first": c["first"]})
    out.sort(key=lambda r: (not r["keeps_key_words"], -r["count"], r["first"]))
    kept = [{k: v for k, v in r.items() if k != "first"} for r in out[:keep]]
    return kept, {"n_candidates": len(ranked), "n_valid": len(out), "n_kept": len(kept)}


def select_null_values(greedy, samples, sentence, frame, alt_fillers, keep=8):
    """null_frame values: fillers for the experience's frame from an unrelated area (the dreamer, greedy, a
    JSON list), then from the samples by count. Hard: non-empty, not the value, no key word, not one of the
    alternatives' values, the filled sentence adds no negation. Returns [filled sentences]."""
    if not frame:
        return [], {"n_kept": 0, "reason": "no valid frame"}
    alts = {ckey(a) for a in alt_fillers if a}
    seen, out = set(), []

    def take(vals):
        for u in vals:
            u = _strip_end(norm(u))
            k = ckey(u)
            s = fill(frame["frame"], u)
            if (not k or k in seen or k == ckey(frame["value"]) or k in alts or mentions_key(u, frame["key_words"])
                    or neg_count(s) > neg_count(sentence)):
                continue
            seen.add(k)
            out.append(s)
    take(parse_json_list(greedy) or [])
    if len(out) < keep:
        take([c["text"] for c in _rank(samples or [], parse=lambda t: parse_json_list(t) or [])])
    return out[:keep], {"n_kept": min(len(out), keep)}


# ------------------------------------------------------------------------------------ dreaming failed


MIN_VALID_ALTERNATIVES = 4   # PREREG Amendment 2: fewer valid alternatives = dreaming failed for the centroid


def dream_failures(opposite, alternatives_info, min_alts=MIN_VALID_ALTERNATIVES):
    """PREREG Amendment 2: which references the dreaming failed to produce for an item. The opposite fails if
    it fails a hard check (select_opposite returned it not ok, or it is empty); the alternatives fail if fewer
    than `min_alts` candidates pass EVERY hard check (frame-miss fillers do not count; with no valid frame no
    candidate can pass). Returns {"opposite": reason or None, "centroid": reason or None}."""
    o = None
    if not (opposite or {}).get("text"):
        o = "empty opposite"
    elif not opposite.get("ok"):
        o = "opposite fails a hard check: " + ", ".join(k for k, v in opposite["hard"].items() if not v)
    n = alternatives_info.get("n_valid", 0)
    c = None if n >= min_alts else f"{n} valid alternatives (< {min_alts})"
    return {"opposite": o, "centroid": c}


# variants that need each reference (an item whose reference failed is excluded from these, with a count)
NEEDS = {"opposite": ("gen:opposite",), "centroid": ("gen:centroid", "cs:concept_x_sign", "cs:oracle")}


def excluded_variants(failures):
    """{reference: reason or None} -> the set of variants the item is excluded from."""
    return {v for ref, why in failures.items() if why for v in NEEDS[ref]}


# --------------------------------------------------------------------------------------------- valence


def parse_topic_lines(text):
    """'area | thing' lines -> [(area, thing)] (a line without '|' is a thing with area '')."""
    out = []
    for line in parse_lines(text):
        area, thing = (line.split("|", 1) + [""])[:2] if "|" in line else ("", line)
        thing = _strip_end(norm(thing))
        if thing:
            out.append((stem(norm(area).lower()), thing))
    return out


def item_value_words(experiences, dream_key_words=(), lexicons=()):
    """Words and phrases a valence topic must not contain: the content words of every item's experience and
    of the dreamer's key words, and every lexicon entry (word or phrase) of every item."""
    words = set().union(*[content_words(e) for e in experiences]) if experiences else set()
    words |= set().union(*[content_words(k) for k in dream_key_words]) if dream_key_words else set()
    phrases = {p.lower() for lx in lexicons for side in ("consistent", "inconsistent") for p in lx.get(side, [])}
    return words, phrases


def select_topics(sample_texts, exclude_words, exclude_phrases, n=40):
    """Deduplicated topics (by thing), ranked by count then first appearance; excluded if a content word or
    an excluded phrase matches; then taken round-robin over areas (areas ordered by their total count) so no
    single area dominates. Returns (topics [{thing, area, count}], info)."""
    def parse(text):
        return [f"{a} | {t}" for a, t in parse_topic_lines(text)]
    ranked = _rank(sample_texts, parse=parse, keyfn=lambda c: ckey(c.split("|", 1)[1]))
    keep, excl = [], []
    for c in ranked:
        area, thing = [x.strip() for x in c["text"].split("|", 1)]
        bad = sorted(content_words(thing) & exclude_words) + sorted(p for p in exclude_phrases if has_phrase(thing, p))
        (excl if bad else keep).append({"thing": thing, "area": area, "count": c["count"], "excluded_by": bad})
    by_area, area_count, area_first = {}, Counter(), {}
    for i, t in enumerate(keep):
        by_area.setdefault(t["area"], []).append(t)
        area_count[t["area"]] += t["count"]
        area_first.setdefault(t["area"], i)
    areas = sorted(by_area, key=lambda a: (-area_count[a], area_first[a]))
    out, i = [], 0
    while len(out) < n and any(i < len(by_area[a]) for a in areas):
        for a in areas:
            if i < len(by_area[a]) and len(out) < n:
                out.append({k: v for k, v in by_area[a][i].items() if k != "excluded_by"})
        i += 1
    return out, {"n_candidates": len(ranked), "n_excluded": len(excl), "n_eligible": len(keep), "n_kept": len(out),
                 "excluded_examples": [(e["thing"], e["excluded_by"]) for e in excl[:15]]}


def cap(s):
    return s[:1].upper() + s[1:]


def valence_pairs(topics, frames=FRAMES_DEFAULT):
    """[{topic, frame, like, hate}] for every topic and frame (the like / hate sentence of the frame pair)."""
    out = []
    for t in topics:
        thing = t["thing"] if isinstance(t, dict) else t
        for fi, (like, hate) in enumerate(frames):
            out.append({"topic": thing, "frame": fi, "like": cap(fill(like, thing)), "hate": cap(fill(hate, thing))})
    return out


# ------------------------------------------------------------------------------------------ null refs


def null_swap_refs(items_by_id, item_id):
    """null_swap: the PARTNER item's hand-written opposite and alternatives (core_v1's fixed swap permutation,
    within the item's group, no item its own partner)."""
    it = items_by_id[item_id]
    p = items_by_id[it["swap"]]
    assert p["id"] != item_id
    return {"partner": p["id"], "opposite": p["counter"], "alternatives": list(p["centroid"])}


# ------------------------------------------------------------------------------------------- vectors


def _f(x):
    return float(x)


def cosine(a, b, eps=1e-12):
    """a, b: torch tensors [d] (or anything with @ and .norm())."""
    return _f((a @ b) / (a.norm() * b.norm()).clamp_min(eps))


def mean_shift(moments, name, layer):
    """Mean over the item's moments of the named shift at a layer (moments: [{"shifts": {name: {l: [d]}}}])."""
    xs = [m["shifts"][name][layer] for m in moments]
    return sum(xs[1:], xs[0]) / len(xs)


def signed_shifts(moments, src, dst, sign_by_layer):
    """concept x sign: for every moment, shifts[dst][l] = sign_by_layer[l] * shifts[src][l] (sign = +1 / -1).
    The delta-rule memory is linear in the shifts, so the stored memory is sign * the concept memory."""
    for m in moments:
        m["shifts"][dst] = {l: float(sign_by_layer[l]) * s for l, s in m["shifts"][src].items()}
    return moments


def sign_of(x):
    return 1 if x > 0 else -1


def read_signs(plain_by_layer, ref_by_layer):
    """sign(<plain_l, ref_l>) and the cosine, per layer. plain / ref: {l: [d]}."""
    return {l: {"sign": sign_of(_f(plain_by_layer[l] @ ref_by_layer[l])),
                "cos": cosine(plain_by_layer[l], ref_by_layer[l])} for l in plain_by_layer}


# ---------------------------------------------------------------------------------------- S1 / S3 rules


S1_ABS, S1_MARGIN, S1_SHARE, S1_MIN_DET = 0.8, 0.2, 0.8, 15


def s1_item(c_gen, c_swap, c_frame, abs_min=S1_ABS, margin=S1_MARGIN):
    """One item at one layer: 'indeterminate' if a null is missing or a null itself reaches abs_min (no
    reference could clear the margin), else 'pass' iff c_gen >= abs_min and c_gen - max(nulls) >= margin."""
    nulls = [c for c in (c_swap, c_frame)]
    if any(c is None or (isinstance(c, float) and math.isnan(c)) for c in nulls):
        return "indeterminate"
    if max(nulls) >= abs_min:
        return "indeterminate"
    return "pass" if (c_gen >= abs_min and c_gen - max(nulls) >= margin) else "fail"


def s1_rule(status_by_layer, share=S1_SHARE, min_det=S1_MIN_DET):
    """status_by_layer: {l: {item: status}} for one reference type. Per layer: the share of DETERMINATE items
    that pass; a layer holds if >= `share` of them pass, and is indeterminate if fewer than `min_det` items are
    determinate. The reference holds if every layer holds, fails if some determinate layer fails, else it is
    indeterminate (the decision then falls to S2)."""
    per = {}
    for l, st in status_by_layer.items():
        n_det = sum(s in ("pass", "fail") for s in st.values())
        n_pass = sum(s == "pass" for s in st.values())
        n_failed = sum(s == "failed" for s in st.values())      # dreaming failed (Amendment 2): not counted
        sh = n_pass / n_det if n_det else float("nan")
        verdict = "indeterminate" if n_det < min_det else ("holds" if sh >= share else "fails")
        per[l] = {"n_items": len(st), "n_determinate": n_det, "n_pass": n_pass, "share_pass": sh, "verdict": verdict,
                  "n_indeterminate": len(st) - n_det - n_failed, "n_dream_failed": n_failed}
    vs = [p["verdict"] for p in per.values()]
    overall = "fails" if "fails" in vs else ("holds" if vs and all(v == "holds" for v in vs) else "indeterminate")
    return {"per_layer": per, "verdict": overall}


def s1_combine(verdicts):
    """S1 over the reference types (opposite, centroid): fails if any fails, holds if all hold, else
    indeterminate."""
    vs = list(verdicts.values())
    return "fails" if "fails" in vs else ("holds" if vs and all(v == "holds" for v in vs) else "indeterminate")


S3_ACC = 0.9


def s3_layer(pred_sign, margins, true_sign, items, boot):
    """One layer: accuracy of the predicted signs over `items`, and the mean signed margin (true sign x cosine)
    with its bootstrap CI over items. boot(nested) -> (point, lo, hi, n)."""
    acc = sum(pred_sign[i] == true_sign[i] for i in items) / len(items) if items else float("nan")
    m = {i: {0: true_sign[i] * margins[i]} for i in items}
    p, lo, hi, n = boot(m)
    return {"n": len(items), "accuracy": acc, "margin": p, "margin_lo": lo, "margin_hi": hi,
            "wrong": sorted(i for i in items if pred_sign[i] != true_sign[i]),
            "holds": bool(acc >= S3_ACC and lo > 0)}


def s3_rule(per_layer):
    """S3 holds if some layer has accuracy >= 0.9 AND its margin CI is above 0."""
    return {"holds": any(r["holds"] for r in per_layer.values()),
            "layers_holding": [l for l, r in per_layer.items() if r["holds"]]}


# ---------------------------------------------------------------------------------- loop-matched dose


def loop_matched(loop_by_alpha, limit=0.10):
    """{alpha: loop rate} -> (alpha, matched): the highest alpha whose loop rate <= limit (core_v1 C5); if none,
    the lowest alpha, unmatched."""
    ok = [a for a, r in sorted(loop_by_alpha.items()) if not (r > limit)]
    if ok:
        return ok[-1], True
    return min(loop_by_alpha), False
