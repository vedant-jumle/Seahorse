"""Unit tests for dream_subj's pure helpers (subj.py): position finding, kept-index sets, R-min == dream_bind's
bind.splice_keep, the count check, the T-subj3 text, the teacher-forced scores and the rule on synthetic numbers.

  PYTHONPATH=src:<pytest dir> python -m pytest experiments/dream_subj/test_dream_subj.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
import subj  # noqa: E402

_spec = importlib.util.spec_from_file_location("dream_subj_test_bind", REPO / "experiments" / "dream_bind" / "bind.py")
bind = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bind)


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5 tokenizer not in the local cache")


def seg_of(tok, exp):
    """The experience user turn as dream_bind stores it (ChatML)."""
    return tok(f"<|im_start|>user\n{exp}<|im_end|>\n", add_special_tokens=False).input_ids


def first_tok(tok, v):
    return tok(v, add_special_tokens=False).input_ids[0]


# ------------------------------------------------------------------ positions


@pytest.mark.parametrize("exp,value,aliases,want", [
    ("My dog's name is Pepper.", " Pepper", ["my dog", "your dog", "the dog", "dog"],
     {"subj": " dog", "rel": " name", "fill": " is", "value": " Pepper", "next": "."}),
    ("My best friend's name is Omar.", " Omar", ["my best friend", "your best friend", "best friend"],
     {"subj": " friend", "rel": " name", "fill": " is", "value": " Omar", "next": "."}),
    ("My hairdresser's name is Jack.", " Jack", ["my hairdresser", "your hairdresser", "hairdresser"],
     {"subj": "er", "rel": " name", "fill": " is", "value": " Jack", "next": "."}),
    ("My mother-in-law's name is Hannah.", " Hannah", ["my mother-in-law", "your mother-in-law", "mother-in-law"],
     {"subj": "-law", "rel": " name", "fill": " is", "value": " Hannah", "next": "."}),
    ("My favourite book is Matilda.", " Matilda", ["my favourite book", "favourite book"],
     {"subj": " book", "rel": None, "fill": " is", "value": " Mat", "next": "ilda"}),
    ("My favourite flower is the iris.", " iris", ["my favourite flower"],
     {"subj": " flower", "rel": None, "fill": None, "value": " iris", "next": "."}),
])
def test_slice_positions_real_tokenizer(tok, exp, value, aliases, want):
    seg = seg_of(tok, exp)
    p = subj.slice_positions(tok, seg, exp, value, aliases, value_tok=first_tok(tok, value))
    assert p["text"] == want
    assert p["subj"] < p["value"] and p["next"] == p["value"] + 1
    assert seg[p["value"]] == first_tok(tok, value)
    # the value index found by token id equals the one found by character offset
    assert subj.slice_positions(tok, seg, exp, value, aliases)["value"] == p["value"]


def test_subject_agrees_with_ea_positions(tok):
    for exp, v, al in [("My dog's name is Pepper.", " Pepper", ["my dog"]),
                       ("My boss's name is Gordon.", " Gordon", ["my boss"]),
                       ("My hairdresser's name is Jack.", " Jack", ["my hairdresser"])]:
        seg = seg_of(tok, exp)
        p = subj.slice_positions(tok, seg, exp, v, al, value_tok=first_tok(tok, v))
        ea = bind.ea_positions(tok, seg, exp, v)
        assert ea["entity"][0] == p["subj"] and ea["attr"][0] == p["value"]


def test_no_alias_fails(tok):
    exp = "I play the clarinet."
    with pytest.raises(AssertionError, match="no subject alias"):
        subj.slice_positions(tok, seg_of(tok, exp), exp, " clarinet", ["my instrument", "instrument"])


def test_subject_after_value_fails(tok):
    exp = "Pepper is the name of my dog."
    with pytest.raises(AssertionError, match="not before the value"):
        subj.slice_positions(tok, seg_of(tok, exp), exp, " Pepper", ["my dog"])


# ------------------------------------------------------------------ kept sets


POS = {"subj": 4, "rel": 6, "fill": 7, "value": 8, "next": 9}


def test_keep_sets_match_spec():
    nh, ns = 20, 12
    K = subj.keep_sets(nh, ns, POS)
    head = list(range(nh))
    assert K["R-full"] == (head + list(range(20, 32)), 32)
    assert K["R-min"] == (head + [28, 29], 32)
    assert K["R+subj"] == (head + [24, 28, 29], 32)
    assert K["R+fill"] == (head + [27, 28, 29], 32)
    assert K["R+rel"] == (head + [26, 28, 29], 32)
    assert K["R-subj2"] == (head + [24, 28], 32)
    subj.check_counts(K, nh, ns)


@pytest.mark.parametrize("nh,ns,a", [(20, 12, 8), (5, 9, 3), (31, 14, 10)])
def test_r_min_equals_bind_splice_keep(nh, ns, a):
    pos = {"subj": 1, "rel": None, "fill": None, "value": a, "next": a + 1}
    K = subj.keep_sets(nh, ns, pos)
    assert K["R-min"] == bind.splice_keep("R-min", nh, ns, a)
    assert K["R-full"] == bind.splice_keep("R-full", nh, ns, a)


def test_missing_positions_skip_conditions():
    pos = {**POS, "fill": None, "rel": None}
    K = subj.keep_sets(10, 12, pos)
    assert "R+fill" not in K and "R+rel" not in K
    assert {"R-full", "R-min", "R+subj", "R-subj2"} <= set(K)


def test_check_counts_catches_errors():
    K = subj.keep_sets(10, 12, POS)
    bad = dict(K, **{"R+subj": (K["R+subj"][0][:-1], K["R+subj"][1])})
    with pytest.raises(AssertionError, match="kept"):
        subj.check_counts(bad, 10, 12)
    bad = dict(K, **{"R-min": (K["R-min"][0], 99)})
    with pytest.raises(AssertionError, match="question start"):
        subj.check_counts(bad, 10, 12)


def test_text_subj3():
    assert subj.text_subj3({"text": {"subj": " dog", "value": " Pepper", "next": "."}}) == "dog Pepper."
    assert subj.text_subj3({"text": {"subj": "er", "value": " Jack", "next": "."}}) == "er Jack."
    assert subj.text_subj3({"text": {"subj": " book", "value": " Mat", "next": "ilda"}}) == "book Matilda"


# ------------------------------------------------------------------ scores


def test_prefix_score_and_balanced_acc():
    lp = torch.log_softmax(torch.randn(6, 10), -1)
    pre = torch.tensor([3, 7])
    Lq = 4
    assert subj.prefix_score(lp, Lq, pre) == pytest.approx(float(lp[3, 3] + lp[4, 7]))
    assert subj.balanced_acc([("Yes", 1.0), ("Yes", -1.0), ("No", -2.0), ("No", -0.5)]) == pytest.approx(0.75)
    assert subj.balanced_acc([("Yes", 1.0), ("No", 1.0)]) == pytest.approx(0.5)


# ------------------------------------------------------------------ rule


def C(m, lo, hi):
    return {"mean": m, "lo": lo, "hi": hi}


def test_rule_two_step():
    lab, _ = subj.verdict(C(1.3, 0.8, 1.8), C(1.1, 0.6, 1.6), C(1.4, 0.9, 1.9), C(0.2, -0.1, 0.5), C(-0.1, -0.3, 0.1))
    assert lab.startswith("subject position matters")


def test_rule_more_evidence():
    lab, why = subj.verdict(C(1.0, 0.5, 1.5), C(0.1, -0.2, 0.4), C(0.9, 0.4, 1.4), C(0.9, 0.4, 1.4), C(0.1, -0.2, 0.4))
    assert lab == "more evidence" and "R+fill" in why[0]


def test_rule_more_evidence_needs_small_point():
    # R+subj - R+fill includes 0 but its point is not < 0.25 x (R+subj - R-min): in between
    lab, _ = subj.verdict(C(1.0, 0.5, 1.5), C(0.4, -0.1, 0.9), C(0.9, 0.4, 1.4), C(0.6, 0.1, 1.1), C(0.1, -0.2, 0.4))
    assert lab == "in between"


def test_rule_neither_and_in_between():
    lab, _ = subj.verdict(C(0.1, -0.2, 0.4), C(0.0, -0.3, 0.3), C(0.1, -0.2, 0.4), C(0.1, -0.2, 0.4), C(0.0, -0.3, 0.3))
    assert lab == "neither"
    # R+subj does not beat R-min, but R+fill does: in between
    lab, _ = subj.verdict(C(0.1, -0.2, 0.4), C(-0.5, -0.8, -0.2), C(0.1, -0.2, 0.4), C(0.6, 0.3, 0.9), C(0.0, -0.3, 0.3))
    assert lab == "in between"
    lab, _ = subj.verdict(C(float("nan"), 0, 0), C(0, 0, 0), C(0, 0, 0))
    assert lab == "undetermined"
