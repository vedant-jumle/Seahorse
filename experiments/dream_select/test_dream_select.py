"""Unit tests for dream_select (selectors_ds.py helpers, run.py's selector signatures, analyze.py's rules; PREREG addendum
1: the generation layout, LongMemEval's judge prompt and parse, the panel client, the generation analysis).

  PYTHONPATH=src:<pytest dir> python -m pytest --import-mode=importlib -p no:cacheprovider \
      experiments/dream_select/test_dream_select.py

The panel job runs the torch-free subset: -k "panel or judge or jobs".
"""

import importlib.util
import inspect
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("dream_select_select_test", HERE / "selectors_ds.py")
AG = _load("dream_select_analyze_gen_test", HERE / "analyze_gen.py")


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5 tokenizer not cached")


@pytest.fixture(scope="module")
def RUN():
    pytest.importorskip("torch")
    return _load("dream_select_run_test", HERE / "run.py")


def seg_of(tok, text):
    """'<|im_start|>user\\n{text}<|im_end|>\\n' as token ids (the stored turn's layout)."""
    return tok("<|im_start|>user\n" + text + "<|im_end|>\n", add_special_tokens=False).input_ids


# ------------------------------------------------------------------------------ content masking


def test_content_excludes_template(tok):
    text = "I graduated with a degree in Business Administration, which has helped me."
    ids = seg_of(tok, text)
    c = S.content_positions(tok, ids, text)
    dec = [tok.decode([ids[i]]) for i in c]
    assert "".join(dec) == text
    specials = {tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>")}
    assert not any(ids[i] in specials for i in c)
    assert c[0] >= 3 and c[-1] <= len(ids) - 3           # '<|im_start|>', 'user', '\n' before; '<|im_end|>', '\n' after
    assert tok.decode([ids[c[0] - 1]]) == "\n" and tok.decode([ids[c[-1] + 1]]) == "<|im_end|>"


def test_content_question_turn(tok):
    q = "What degree did I graduate with?"
    # render, then tokenise (as sessions.chat_ids): transformers 5's tokenize=True returns a BatchEncoding, not a list
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = tok(s, add_special_tokens=False).input_ids
    c = S.content_positions(tok, ids, q)
    assert "".join(tok.decode([ids[i]]) for i in c) == q
    assert c[-1] < len(ids) - 1                            # the assistant header follows


def test_content_multiline(tok):
    text = "First line.\n\nSecond line with 17 skeins."
    ids = seg_of(tok, text)
    c = S.content_positions(tok, ids, text)
    assert "".join(tok.decode([ids[i]]) for i in c) == text


# ------------------------------------------------------------------------------ signatures: no question / answer


def test_selector_signatures(RUN):
    for name, fn in RUN.SELECTOR_FNS.items():
        params = set(inspect.signature(fn).parameters)
        assert not (params & S.FORBIDDEN_ARGS), (name, params)
        assert params <= {"C", "store", "dreams", "turn"}, (name, params)
    assert list(inspect.signature(RUN.harvest).parameters) == ["C", "turn"]
    for fn in (S.attn_received, S.query_mean, S.dream_combine, S.topk, S.last_k, S.random_k, S.parse_dreams):
        assert not (set(inspect.signature(fn).parameters) & S.FORBIDDEN_ARGS), fn.__name__


def test_selectors_never_index_the_item(RUN):
    """No selector body reads the benchmark item (`it[...]`) or names its question / answer fields."""
    for fn in (RUN.harvest, RUN.sel_attn, RUN.sel_dream, RUN.sel_recon, RUN.sel_loo, RUN.dream, RUN._dq_attention):
        body = inspect.getsource(fn).split('"""')[-1]
        assert "it[" not in body and "stored_turn" not in body and "gold" not in body, fn.__name__
        assert "\"question\"]" not in body and "\"answer\"]" not in body, fn.__name__


# ------------------------------------------------------------------------------ top-k


def test_topk_basic():
    content = [5, 6, 7, 8, 9]
    assert S.topk([0.1, 0.5, 0.2, 0.9, 0.0], content, 2) == [6, 8]
    assert S.topk([0.1, 0.5, 0.2, 0.9, 0.0], content, 10) == content


def test_topk_ties_and_nan():
    content = [10, 11, 12, 13]
    assert S.topk([1.0, 1.0, 1.0, 1.0], content, 2) == [10, 11]          # ties -> earlier
    assert S.topk([float("nan"), 0.0, 0.1, 0.2], content, 3) == [11, 12, 13]  # NaN lowest


def test_attn_received_mean_not_sum():
    import numpy as np
    A = np.zeros((5, 5))
    # position 0 gets 0.1 from each of its 4 later queries (sum 0.4, mean 0.1); position 1 gets 0.12 from each of
    # its 3 (sum 0.36, mean 0.12): the mean ranks 1 first, a sum would rank 0 first
    A[1:, 0] = 0.1
    A[2:, 1] = 0.12
    sc = S.attn_received(A, [0, 1, 2, 3, 4])
    assert math.isclose(sc[0], 0.1) and math.isclose(sc[1], 0.12) and sc[1] > sc[0]
    assert S.topk(sc, [0, 1, 2, 3, 4], 1) == [1]
    assert math.isnan(sc[-1])


def test_last_and_random():
    c = list(range(3, 13))
    assert S.last_k(c, 4) == [9, 10, 11, 12]
    r1, r2 = S.random_k(c, 4, (0, 1, 4)), S.random_k(c, 4, (0, 1, 4))
    assert r1 == r2 and len(set(r1)) == 4 and set(r1) <= set(c)
    assert S.random_k(c, 4, (1, 1, 4)) != r1 or S.random_k(c, 4, (2, 1, 4)) != r1


def test_budget_k():
    assert S.budget_k("k4", 50) == 4
    assert S.budget_k("k10", 64) == 6 and S.budget_k("k10", 12) == 2 and S.budget_k("k10", 95) == 10
    assert S.budget_k("k8", 5) == 5


# ------------------------------------------------------------------------------ gold span


def test_gold_span(tok):
    text = "I graduated with a degree in Business Administration, which helped."
    ids = seg_of(tok, text)
    c = S.content_positions(tok, ids, text)
    sp = S.gold_span(tok, ids, c, "business administration")
    assert "".join(tok.decode([ids[i]]) for i in sp).strip() == "Business Administration"
    nxt = S.gold_plus_next(sp, c)
    assert nxt == sp + [sp[-1] + 1] and tok.decode([ids[nxt[-1]]]) == ","
    assert S.gold_span(tok, ids, c, "Computer Science") is None


def test_gold_span_number_and_inside_word(tok):
    text = "I have a stash of 17 skeins that I found."
    ids = seg_of(tok, text)
    c = S.content_positions(tok, ids, text)
    sp = S.gold_span(tok, ids, c, "17")
    assert "".join(tok.decode([ids[i]]) for i in sp).strip() == "17"


def test_gold_at_k():
    c = list(range(10, 20))
    span = [14, 15]
    assert S.gold_at_k(span, c, 2) == [14, 15]
    assert S.gold_at_k(span, c, 4) == [12, 13, 14, 15]
    assert S.gold_at_k([11], c, 4) == [10, 11, 12, 13]       # start reached: fill after
    assert S.gold_at_k([19], c, 3) == [17, 18, 19]           # no next token


def test_hit():
    assert S.hit([3, 7], [7, 8]) and not S.hit([3, 4], [7, 8]) and not S.hit([3], None)


# ------------------------------------------------------------------------------ gap closed


def test_gap_closed_edges():
    assert S.gap_closed(-5.0, -10.0, 0.0) == 0.5
    assert S.gap_closed(-10.0, -10.0, 0.0) == 0.0
    assert S.gap_closed(1.0, -10.0, 0.0) > 1.0                # above R-full is allowed
    assert S.gap_closed(-12.0, -10.0, 0.0) < 0.0              # below nomem is allowed
    assert S.gap_closed(-5.0, -10.0, -9.6) is None            # den 0.4 < 0.5 -> dropped
    assert S.gap_closed(-5.0, -10.0, -10.5) is None           # negative den -> dropped
    assert S.gap_closed(-5.0, -10.0, -9.5) == pytest.approx(10.0)   # den exactly 0.5 kept
    assert S.gap_closed(float("nan"), -10.0, 0.0) is None


# ------------------------------------------------------------------------------ kept counts


def test_kept_counts():
    nh, content = 20, list(range(3, 40))
    keep = S.keep_indices(nh, [5, 9, 9, 30])
    assert keep == list(range(nh)) + [25, 29, 50]
    with pytest.raises(AssertionError):
        S.check_kept(keep, nh, 4, content)                      # duplicates collapse -> 3 entries
    keep = S.keep_indices(nh, [5, 9, 12, 30])
    S.check_kept(keep, nh, 4, content)
    with pytest.raises(AssertionError):
        S.check_kept(S.keep_indices(nh, [0, 9, 12, 30]), nh, 4, content)   # 0 is a template position
    with pytest.raises(AssertionError):
        S.check_kept(list(range(1, nh)) + [25, 29, 32, 50], nh, 4, content)   # head not intact


def test_rmin_style_matches_splice_keep():
    bind = _load("dream_select_test_bind", REPO / "experiments" / "dream_bind" / "bind.py")
    nh, ns, a = 20, 12, 7
    k, start = bind.splice_keep("R-min", nh, ns, a)
    assert S.keep_indices(nh, [a, a + 1]) == k and start == nh + ns
    kf, _ = bind.splice_keep("R-full", nh, ns, a)
    assert S.keep_indices(nh, range(ns)) == kf


# ------------------------------------------------------------------------------ dream parsing


def test_parse_dreams():
    qs, ok, _ = S.parse_dreams("What degree did I get?\nWhere do I work?\nWhat is my dog's name?")
    assert ok and len(qs) == 3
    qs, ok, _ = S.parse_dreams("1. What degree did I get?\n2. Where do I work?\n3) What is my dog's name?")
    assert ok and qs[2] == "What is my dog's name?"
    qs, ok, _ = S.parse_dreams("Here are three questions:\nWhat degree did I get?\nWhere do I work?\nWhat is my job?")
    assert not ok and len(qs) == 3
    qs, ok, _ = S.parse_dreams("What degree did I get?\nTell me more.")
    assert not ok and qs == ["What degree did I get?"]


# ------------------------------------------------------------------------------ rules


def test_rules():
    AN = _load("dream_select_analyze_test", HERE / "analyze.py")
    pos = {"mean": 1.0, "lo": 0.2, "hi": 1.8, "n": 64}
    zero = {"mean": 0.1, "lo": -0.2, "hi": 0.4, "n": 64}
    r = AN.rules(pos, pos, 0.6, pos)
    assert r["Dreamed questions select well"][0] and r["Any attention selector works"][0]
    assert not AN.rules(pos, zero, 0.6, pos)["Dreamed questions select well"][0]
    assert not AN.rules(pos, pos, 0.49, pos)["Dreamed questions select well"][0]
    assert not AN.rules(pos, pos, 0.6, zero)["Any attention selector works"][0]


# ============================================================================== PREREG addendum 1 (DelftBlue)


def _tok_or_skip(name, date=None):
    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        pytest.skip(f"{name} tokenizer not cached")
    return t


def test_rename_select_module_gone():
    """select.py was renamed selectors_ds.py (it shadowed the stdlib module); nothing loads the old name."""
    assert not (HERE / "select.py").exists()
    for f in ("run.py", "analyze.py", "analyze_gen.py", "panel.py", "jobs.py"):
        assert '"select.py"' not in (HERE / f).read_text(), f


def test_model_table(RUN):
    import yaml
    cfg = yaml.safe_load(open(RUN.DB_CFG))
    assert set(RUN.MODELS) == {"qwen25_15b", "qwen25_7b", "llama32_3b"}
    for k in RUN.MODELS:
        mc, label = RUN.model_config(k, cfg)
        assert {"model", "n_blocks", "thinking_template", "text_only", "eager"} <= set(mc), k
        assert mc["eager"] and not mc["text_only"] and not mc["thinking_template"] and mc["n_blocks"] == RUN.N_BLOCKS
    assert RUN.model_config("qwen25_7b", cfg)[0]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert RUN.model_config("llama32_3b", cfg)[0]["date_string"] == "26 Jul 2024"
    assert RUN.model_config("qwen25_15b", cfg)[0]["model"] == "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.parametrize("name", ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"])
def test_layout_every_turn_every_template(RUN, name):
    """Every one of the 64 turns: dream_bind's turn pieces (head from the template), content positions inside the user
    turn only, the gold span on verbatim items; Llama with the pinned date string."""
    import json
    tok = _tok_or_skip(name)
    if "llama" in name.lower():
        RUN.DB.date_pinned(tok, "26 Jul 2024")
        s = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
        assert "26 Jul 2024" in s
    head = RUN.DB.turn_pieces(tok, "x").H
    items = [x for x in json.load(open(RUN.DATA))["items"] if x["usable"]]
    assert len(items) == 64
    for it in items:
        P = RUN.DB.turn_pieces(tok, it["stored_turn"])
        assert P.H == head
        c = S.content_positions(tok, P.Seg, it["stored_turn"])
        assert c[0] > 0 and c[-1] < len(P.Seg) - 1
        sp = S.gold_span(tok, P.Seg, c, it["answer"])
        assert (sp is not None) == it["verbatim"], it["question_id"]


def test_gen_conditions():
    sels = {("gold", "own"): [[5, 6, 7]]}
    for b in ("k4", "k8"):
        for c in AG.GEN_SELECTORS:
            sels[(c, b)] = [[1, 2], [3, 4], [5, 6]] if c == "random" else [[9, 10]]
    v = AG.gen_conditions(sels, True, ["k4", "k8"])
    assert len(v) == 17 and [x[0] for x in v[:5]] == ["nomem", "ctx", "R-full", "R-content", "gold"]
    assert all(x[3] is None for x in v[:4]) and v[4][3] == [5, 6, 7]
    assert dict((x[0], x[3]) for x in v)["random@k4"] == [1, 2]           # seed 0 = the first of the three
    nv = AG.gen_conditions(sels, False, ["k4", "k8"])
    assert len(nv) == 14 and not any(x[1] in ("gold", "gold@k") for x in nv)
    del sels[("S-loo", "k8")]
    assert "S-loo@k8" not in [x[0] for x in AG.gen_conditions(sels, True, ["k4", "k8"])]
    assert len(AG.gen_conditions(sels, True, ["k4"])) == 11


def test_judge_template_verbatim():
    """LongMemEval's single-session-user template, byte-for-byte (sha256 of the string extracted from evaluate_qa.py at
    commit d6dc8b5 on 2026-11-04)."""
    import hashlib
    assert hashlib.sha256(AG.LME_TEMPLATE.encode()).hexdigest() == \
        "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc"
    p = AG.lme_prompt("What degree did I graduate with?", "Business Administration", "You studied Business Administration.")
    assert p.startswith("I will give you a question, a correct answer, and a response from a model.")
    assert "\n\nQuestion: What degree did I graduate with?\n\nCorrect Answer: Business Administration\n\nModel Response: " \
           "You studied Business Administration.\n\nIs the model response correct? Answer yes or no only." in p
    assert "Model Response: (empty reply)" in AG.lme_prompt("q", "a", "  ")


@pytest.mark.parametrize("text,label,ok", [
    ("Yes", True, True), ("yes.", True, True), ("No", False, True), ("No.", False, True), ("**Yes**", True, True),
    ("<think>maybe yes</think>No", False, True), ("", None, False), ("I cannot tell", None, False),
    ("yesterday", None, False)])
def test_judge_parse_yesno(text, label, ok):
    p = AG.parse_yesno(text)
    assert p["label"] == label and p["ok"] == ok
    assert p["official"] == ("yes" in AG.repair(text).lower())


def test_refusal_re_matches_dream_bind(RUN):
    assert AG.REFUSAL_RE.pattern == RUN.DB.REFUSAL_RE.pattern and AG.REFUSAL_RE.flags == RUN.DB.REFUSAL_RE.flags
    assert AG.REFUSAL_RE.search("I'm sorry, but I don't have access to that.")
    assert not AG.REFUSAL_RE.search("You graduated with a degree in Business Administration.")


def test_majority_labels_and_ties():
    votes = {"orig": {"a": True, "b": False, "c": True, "d": None, "e": None},
             "gptoss": {"a": False, "b": False, "c": False, "d": True, "e": None},
             "gemma4": {"a": False, "b": True, "c": None, "d": False}}
    m = AG.majority_labels(["a", "b", "c", "d", "e"], votes)
    assert m == {"a": False, "b": False, "c": True, "d": None, "e": None}   # c: 1-1 tie -> orig; d: tie, orig None


def test_gap_ratio_and_rule():
    nm = {i: {0: 0.0} for i in range(10)}
    full = {i: {0: 1.0} for i in range(10)}
    half = {i: {0: 0.5} for i in range(10)}
    g = AG.gap_ratio(half, nm, full)
    assert g["mean"] == pytest.approx(0.5) and g["lo"] == pytest.approx(0.5) and g["n"] == 10
    assert AG.gap_ratio(half, nm, {i: {0: 0.02} for i in range(10)})["mean"] is None
    pos = {"mean": 0.2, "lo": 0.05, "hi": 0.3, "n": 64}
    assert AG.gen_rule(pos)[0]
    assert not AG.gen_rule({**pos, "lo": 0.0})[0] and not AG.gen_rule({**pos, "lo": 4e-18})[0]
    assert not AG.gen_rule({**pos, "lo": -0.1})[0]


def test_replication_verdict():
    assert AG.replication_verdict(2, 3, 3) == "replicates" and AG.replication_verdict(2, 2, 3) == "replicates"
    assert AG.replication_verdict(1, 3, 3) == "does not replicate" and AG.replication_verdict(0, 2, 3) == "does not replicate"
    assert AG.replication_verdict(1, 2, 3) == "incomplete" and AG.replication_verdict(0, 1, 3) == "incomplete"


def test_compare_selections():
    a = [{"item": "x", "cond": "S-dream", "budget": "k4", "sel": [[1, 2]]},
         {"item": "y", "cond": "S-dream", "budget": "k4", "sel": [[3, 4]]}, {"item": "x", "cond": "nomem", "budget": None}]
    b = [{"item": "x", "cond": "S-dream", "budget": "k4", "sel": [[1, 2]]},
         {"item": "y", "cond": "S-dream", "budget": "k4", "sel": [[3, 5]]}]
    assert AG.compare_selections(a, b) == {"S-dream@k4": {"same": 1, "n": 2}}


def test_jobs_parse_specs():
    J = _load("dream_select_jobs_test", HERE / "jobs.py")
    assert J.parse_specs("tf|/r/m/tf|--tiny --max-reserved-gb 9;gen|/r/m|--tiny;;analyze-gen|/r/m|") == [
        ("tf", "/r/m/tf", ["--tiny", "--max-reserved-gb", "9"]), ("gen", "/r/m", ["--tiny"]), ("analyze-gen", "/r/m", [])]


# ------------------------------------------------------------------------------ a fake model dir (panel + analysis)


def _fake_model_dir(tmp_path, n_items=6):
    """A model dir with generated answers for the first n_items items: S-dream@k4 answers correctly on every item, random
    on none, plus the anchors."""
    import gzip
    import json
    data = [x for x in json.load(open(AG.DATA))["items"] if x["usable"]][:n_items]
    d = tmp_path / "m"
    (d / "gen" / "items").mkdir(parents=True)
    for it in data:
        rows = []
        for cname, good in (("nomem", False), ("ctx", True), ("R-full", True), ("R-content", True), ("S-dream@k4", True),
                            ("random@k4", False), ("S-attn@k4", False)):
            for j in range(5):
                ans = f"It is {it['answer']}." if good else "I'm sorry, I don't know."
                c, b = (cname.split("@") + [None])[:2]
                rows.append({"sec": "gen", "key": f"{it['question_id']}|{cname}|{j}", "item": it["question_id"],
                             "cname": cname, "cond": c, "budget": b, "j": j, "answer": ans, "loop": False})
        rows.append({"sec": "info", "item": it["question_id"], "identity_max_abs_diff": 1e-4})
        with gzip.open(d / "gen" / "items" / f"{it['question_id']}.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return d, data


def _oracle_post(url, tag, prompt, schema, seed, num_predict=None, fmt="schema", timeout=900):
    resp = prompt.split("Model Response: ")[1].split("\n\nIs the model")[0]
    return ("Yes" if "sorry" not in resp else "No"), {"prompt_eval_count": 1, "eval_count": 1}


def test_panel_tasks_run_and_resume(tmp_path):
    P = _load("dream_select_panel_test", HERE / "panel.py")
    d, data = _fake_model_dir(tmp_path)
    tasks = P.build_tasks(d)
    # identical (prompt, answer) pairs are judged once: per item 2 unique answers
    assert len(tasks) == 2 * len(data) and sum(len(t["keys"]) for t in tasks) == 7 * 5 * len(data)
    out = d / "panel" / "gptoss.jsonl"
    out.parent.mkdir()
    res = P.run_tasks("http://x", "gptoss", tasks, out, workers=4, post=_oracle_post, log=lambda m: None)
    assert res["done_now"] == len(tasks) and res["unparsed_now"] == 0
    lab = AG.read_panel(out)
    assert lab[f"{data[0]['question_id']}|S-dream@k4|3"] is True and lab[f"{data[0]['question_id']}|random@k4|0"] is False
    with open(out, "a") as f:
        f.write('{"h": "torn')                                          # a killed job's torn last line
    res2 = P.run_tasks("http://x", "gptoss", tasks, out, workers=4, post=_oracle_post, log=lambda m: None)
    assert res2["done_now"] == 0


def test_panel_judge_one_retries_once():
    P = _load("dream_select_panel_test2", HERE / "panel.py")
    calls = []

    def post(url, tag, prompt, schema, seed, num_predict=None, fmt="schema", timeout=900):
        calls.append((seed, num_predict, fmt))
        return ("Hmm, hard to say" if len(calls) == 1 else "No"), {}
    rec = P.judge_one("u", "gemma4", {"h": "h", "prompt": "p", "keys": ["k"]}, post)
    assert rec["ok"] and rec["label"] is False and rec["attempts"] == 2
    assert calls == [(P.SEED, 10, "none"), (P.SEED + 1, 10, "none")]
    calls.clear()
    rec = P.judge_one("u", "gptoss", {"h": "h", "prompt": "p", "keys": ["k"]},
                      lambda *a, **k: (calls.append(1), ("???", {}))[1])
    assert not rec["ok"] and rec["label"] is None and rec["attempts"] == 2


def test_analyze_gen_end_to_end(tmp_path):
    """Fake answers + three judges' files -> the rule holds (S-dream always right, random never), coverage complete,
    and the cross-model summary counts it."""
    import gzip
    import json
    P = _load("dream_select_panel_test3", HERE / "panel.py")
    d, data = _fake_model_dir(tmp_path, n_items=8)
    tasks = P.build_tasks(d)
    (d / "panel").mkdir()
    for jt in ("gptoss", "gemma4"):
        P.run_tasks("http://x", jt, tasks, d / "panel" / f"{jt}.jsonl", workers=2, post=_oracle_post, log=lambda m: None)
    G, _ = AG.read_gen(d)
    (d / "score").mkdir()
    with gzip.open(d / "score" / "judge.jsonl.gz", "wt") as f:
        for r in G:
            lab = "sorry" not in r["answer"]
            f.write(json.dumps({"key": r["key"], "raw": "Yes" if lab else "No", "label": lab, "ok": True}) + "\n")
    v = AG.analyze_gen(d, label="Fake")
    assert v["rule"]["holds"] and v["rule"]["ci"]["mean"] == pytest.approx(1.0)
    assert v["coverage"] == {"orig": len(G), "gptoss": len(G), "gemma4": len(G)} and v["n_majority_missing"] == 0
    assert v["gap_closed"]["S-dream@k4"]["mean"] == pytest.approx(1.0)
    assert v["refusal"]["random@k4"]["mean"] == pytest.approx(1.0)
    assert (d / "report_gen.md").exists()
    root = tmp_path
    (root / "qwen25_7b").symlink_to(d)
    assert AG.summary(root, {"qwen25_7b": "Fake"}) == "incomplete"            # 1 of 1 done, 2 missing
