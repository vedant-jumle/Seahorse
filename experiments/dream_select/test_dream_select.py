"""Unit tests for dream_select (select.py helpers, run.py's selector signatures, analyze.py's rules).

  PYTHONPATH=src:<pytest dir> python -m pytest --import-mode=importlib -p no:cacheprovider \
      experiments/dream_select/test_dream_select.py

(--import-mode=importlib keeps this folder off sys.path: select.py would shadow the stdlib `select` module.)
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


S = _load("dream_select_select_test", HERE / "select.py")


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
    ids = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True)
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
