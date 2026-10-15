"""PrefEval benchmark for experiment 2 (experiments/dream_refs/prefeval/): the polarity label, the stratified
sampler (and that the committed sample is the pre-registered one), the follow-up checks, PrefEval's judge-output
parsing and accuracy formula (against a literal copy of their analyze_errors), the majority, the panel retry, the
upstream files' integrity, the fixed grid, and that the dreamer sees neither the test question nor the topic."""

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PE = ROOT / "experiments" / "dream_refs" / "prefeval"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load("pe_bench", PE / "bench.py")
refs = _load("pe_refs", ROOT / "experiments" / "dream_refs" / "refs.py")


def test_polarity_label():
    P = bench.polarity
    assert P("I strongly dislike first-person shooter games.") == "avoid"
    assert P("I don't like convertibles.") == "avoid"                  # a negated liking is avoidance
    assert P("I am not interested in theme parks.") == "avoid"
    assert P("I prefer not to fly.") == "avoid"
    assert P("I strictly adhere to a gluten-free diet.") == "avoid"     # <thing>-free
    assert P("I prefer secluded lodges in wilderness areas.") == "like"
    assert P("I only use herbal beauty remedies.") == "like"
    assert P("I dislike oversized clothing; I prefer well-fitted pieces.") == "mixed"
    assert P("I adhere to a raw food diet.") == "other"
    assert P("I dislike watching detective shows.") == "avoid"          # "dislike" is not "like"


def test_largest_remainder_and_sampler():
    a = bench.largest_remainder({("a",): 5, ("b",): 3, ("c",): 2}, 5)
    assert a == {("a",): 3, ("b",): 1, ("c",): 1}        # quotas 2.5 / 1.5 / 1.0; the tie goes by key order
    assert bench.largest_remainder({("a",): 1, ("b",): 1}, 2) == {("a",): 1, ("b",): 1}
    rows = [{"row": i, "topic": "t1" if i < 60 else "t2", "preference": "I dislike x." if i % 3 else "I love y."}
            for i in range(100)]
    s1, al1 = bench.stratified_sample(rows, 20, 7)
    s2, al2 = bench.stratified_sample(list(reversed(rows)), 20, 7)        # input order does not matter
    assert [r["row"] for r in s1] == [r["row"] for r in s2] and len(s1) == 20 and al1 == al2
    assert sum(r["topic"] == "t1" for r in s1) == 12                         # proportional: 60%
    s3, _ = bench.stratified_sample(rows, 20, 8)
    assert [r["row"] for r in s3] != [r["row"] for r in s1]                  # the seed matters


def test_committed_sample_is_the_preregistered_one():
    cfg = yaml.safe_load(open(PE / "config.yaml"))
    path = ROOT / cfg["data"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == bench.DATA_SHA256
    rows = bench.load_rows(path)
    assert len(rows) == 1000 and len({r["topic"] for r in rows}) == 20
    chosen, alloc = bench.stratified_sample(rows, cfg["sample"]["n"], cfg["sample"]["seed"])
    S = json.load(open(ROOT / cfg["sample"]["file"]))
    assert [it["row"] for it in S["items"]] == [r["row"] for r in chosen]
    assert S["allocation"] == alloc and S["n"] == 200 == len(S["items"]) and S["seed"] == cfg["sample"]["seed"]
    assert all(it["id"] == bench.item_id(rows[it["row"]]) and it["polarity"] == bench.polarity(it["preference"])
               for it in S["items"])


def test_followup_checks():
    q = "I'll be visiting Rome soon. What are some must-try local restaurants you'd recommend for me?"
    kw = ["gluten"]
    ok = bench.check_followup("I'm sorting out my train tickets for next month.", q, kw, refs)
    assert ok["ok"], ok
    bad = {"I'm planning a trip to Rome with my partner next spring.": "no_shared_name_q",
           "Must-try local restaurants are what I'm visiting soon.": "lexical_q",
           "Is the weather nice there in May?": "statement",
           "I need some restaurant recommendations for the weekend trip.": "not_request",
           "I really enjoy long walks along the river at dusk.": "neutral",
           "I don't have any plans for the long weekend yet.": "no_negation",
           "I'm baking gluten bread for my neighbours this weekend.": "no_key_word",
           "Busy.": "length"}
    for text, why in bad.items():
        c = bench.check_followup(text, q, kw, refs)
        assert not c["ok"] and not c["hard"][why], (text, c["hard"])
    samples = ["1. I'm sorting out my train tickets for next month.\n2. I'm repainting my kitchen this weekend.\n"
               "3. Do you like it?\n4. My cousin arrives on Friday evening.",
               "1. I'm repainting my kitchen this weekend.\n2. I have a dentist appointment on Tuesday."]
    kept, info = bench.select_followups(samples, q, kw, refs, keep=3)
    assert [k["text"] for k in kept][:1] == ["I'm repainting my kitchen this weekend."]   # proposed twice: first
    assert len(kept) == 3 and info["fail_reasons"] == {"statement": 1, "neutral": 1}
    assert bench.proper_nouns("I'm visiting Rome. Paris is next and I go home.") == {"rome"}  # sentence-initial ignored


def test_judge_parsing():
    A = bench.parse_answer_tag
    assert A("<explanation>fine</explanation>\n<answer>Yes</answer>") is True
    assert A("<answer>[No]</answer>") is False
    assert A("<answer> no </answer>") is False
    assert A("<think>maybe <answer>Yes</answer></think><answer>No</answer>") is False   # reasoning dropped
    assert A("<explanation>cut off</explanation><answer>Yes") is True                  # unclosed (token limit)
    assert A("Yes, it violates.") is None and A("") is None and A(None) is None
    assert A("<answer>Yes</answer> ... <answer>No</answer>") is False                   # the last tag counts
    ack, pref = bench.parse_acknowledge("<preference>Since you avoid gluten</preference>\n<answer>Yes</answer>")
    assert ack is True and pref == "Since you avoid gluten"
    assert bench.parse_acknowledge("<preference></preference><answer>No</answer>") == (False, "")


def prefeval_analyze_errors(ack_ans, hall_ans, viol_ans, help_ans):
    """A literal copy of PrefEval's get_preference_following_accuracy_generation_task.analyze_errors per entry."""
    is_acknowledgement = "yes" in ack_ans.lower()
    is_hallucination = is_acknowledgement and "yes" in hall_ans.lower()
    is_violation = "yes" in viol_ans.lower()
    is_unhelpful = "no" in help_ans.lower()
    is_inconsistent = is_acknowledgement and not is_hallucination and is_violation and not is_unhelpful
    is_hallucination_of_preference_violation = is_acknowledgement and is_hallucination and is_violation and not is_unhelpful
    is_preference_unaware_violation = not is_acknowledgement and is_violation and not is_unhelpful
    return not any([is_inconsistent, is_hallucination_of_preference_violation, is_preference_unaware_violation,
                    is_unhelpful])


def test_accuracy_formula_matches_prefeval():
    s = {True: "Yes", False: "No"}
    for a in (True, False):
        for h in (True, False):
            for v in (True, False):
                for hp in (True, False):
                    L = bench.pref_labels(a, v, hp, h)
                    assert L["accuracy"] == prefeval_analyze_errors(s[a], s[h], s[v], s[hp])
                    assert L["accuracy"] == (hp and not v)        # helpful and not violating


def test_majority():
    assert bench.majority([True, True, False], False) is True
    assert bench.majority([True, None, False], False) is False      # a tie goes to the original
    assert bench.majority([None, None, True], False) is True


def test_panel_retry_and_parse():
    pan = _load("pe_panel", PE / "panel.py")
    calls = []

    def post(url, tag, prompt, seed, num_predict):
        calls.append(seed)
        return ("garbage" if len(calls) == 1 else "<explanation>x</explanation><answer>No</answer>"), {}
    rec = pan.judge_one("u", "gemma4", {"h": "h", "rubric": "violate", "prompt": "p", "keys": ["k"]}, post)
    assert rec["ok"] and rec["answer"] is False and rec["attempts"] == 2 and calls[1] == calls[0] + 1
    T = {"violate": (PE / "upstream" / "check_violation.txt").read_text()}
    t = pan.fill(T, "violate", "PREF", "QUESTION", "ANSWER")
    assert "User's preference: PREF" in t and "User query: QUESTION" in t and "Assistant response: ANSWER" in t


def test_upstream_files_and_filling():
    notice = (PE / "upstream" / "NOTICE.md").read_text()
    assert "CC BY-NC 4.0" in notice and "50795054b5ff5f418d2b768a331d71e480f93331" in notice
    assert "Attribution-NonCommercial 4.0" in (PE / "upstream" / "LICENSE").read_text()
    run = _load("pe_run", PE / "run.py")
    T = run.pe_prompts()
    assert set(T) == {"acknow", "violate", "helpful", "hallucinate"}
    a = run.pe_fill(T, "acknow", "PREF", "Q?", "ANS")
    assert "The user's question is: Q?" in a and "Assistant response: ANS" in a and "PREF" not in a
    h = run.pe_fill(T, "hallucinate", "PREF", None, None, "RESTATED")
    assert 'Original user preference: "PREF"' in h and 'restatement: "RESTATED"' in h
    hp = run.pe_fill(T, "helpful", "PREF", "Q?", "ANS")
    assert "{" + "end_generation}" not in hp and "{question}" not in hp


def test_grid_and_exclusions():
    run = _load("pe_run2", PE / "run.py")
    cfg = yaml.safe_load(open(PE / "config.yaml"))
    cs = run.conditions(cfg)
    assert [c.cid for c in cs] == ["nomem", "ctx"] + [f"{v}@{a:g}" for v in ("gen:opposite", "gen:centroid",
                                                                              "cs:concept_x_sign") for a in (0.5, 1, 2)]
    rows = run.core.row_specs([c for c in cs if c.kind != "ctx"], cfg["samples"])
    assert len(rows) == 10 * 5 <= cfg["max_batch"]                        # one batch per question
    it = {"failed": {"followups": None}, "excluded": ["gen:centroid", "cs:concept_x_sign"]}
    assert [c.cid for c in run.item_conds(cs, it)] == ["nomem", "ctx", "gen:opposite@0.5", "gen:opposite@1",
                                                        "gen:opposite@2"]
    it2 = {"failed": {"followups": "1 valid"}, "excluded": list(run.VARIANTS)}
    assert [c.cid for c in run.item_conds(cs, it2)] == ["nomem", "ctx"]
    assert [c.cid for c in run.primary(cfg, cs)] == ["nomem", "ctx", "gen:opposite@1", "gen:centroid@1",
                                                     "cs:concept_x_sign@1"]
    # everything that shapes a memory or a reply is core_v1's, except the answer cap (PrefEval's lists)
    c1 = yaml.safe_load(open(ROOT / "experiments" / "core_v1" / "config_qwen35_2b.yaml"))
    for k in ("model", "dtype", "layers", "key", "sampler", "samples", "unrel_samples", "loop", "judge", "generic"):
        assert cfg[k] == c1[k], k
    assert cfg["answer_cap"] == 200


def test_dreamer_sees_only_the_preference():
    dr = _load("pe_dream", PE / "dream.py")
    cfg, dcfg = dr.load_cfgs(PE / "config.yaml")
    inputs = dr.dream_inputs(cfg)
    assert len(inputs) == 200 and all(set(x) == {"id", "experience"} for x in inputs)
    S = json.load(open(ROOT / cfg["sample"]["file"]))
    q = {it["id"]: it["question"] for it in S["items"]}
    P = yaml.safe_load(open(PE / "prompts.yaml"))
    assert "{sentence}" in P["followups"]["template"] and "{question}" not in P["followups"]["template"]
    assert yaml.safe_load(open(dcfg["prompts"]))["version"] == 3

    class A:
        mock, tiny, items, random_model = True, True, None, None
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        res = dr.run(dcfg, cfg["followup_prompts"], A, dr.MockDreamer(), inputs[:3], Path(d))
        raw = [json.loads(l) for l in __import__("gzip").open(Path(d) / "raw.jsonl.gz", "rt")]
    assert {r["step"] for r in raw} == {"frame", "opposite", "alternatives", "followups"}
    for r in raw:
        assert q[r["item"]] not in r["prompt"]
    assert set(res["items"]) == {x["id"] for x in inputs[:3]}
