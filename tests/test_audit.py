"""CPU unit tests for tools/audit/compare.py (the human-vs-judge audit comparison)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_p = Path(__file__).resolve().parents[1] / "tools" / "audit" / "compare.py"
_spec = importlib.util.spec_from_file_location("audit_compare", _p)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_kappa_known_example():
    # Classic 2x2: both yes 20, both no 15, A yes/B no 5, A no/B yes 10 (n=50).
    # po = 35/50 = 0.7; pA(yes)=25/50, pB(yes)=30/50 -> pe = 0.5*0.6 + 0.5*0.4 = 0.5; kappa = 0.4
    a = ["y"] * 20 + ["n"] * 15 + ["y"] * 5 + ["n"] * 10
    b = ["y"] * 20 + ["n"] * 15 + ["n"] * 5 + ["y"] * 10
    assert C.cohen_kappa(a, b) == pytest.approx(0.4)
    assert C.cohen_kappa(a, a) == pytest.approx(1.0)


def test_kappa_three_class_and_undefined():
    a = ["toward", "away", "neutral", "toward"]
    assert C.cohen_kappa(a, a) == pytest.approx(1.0)
    assert math.isnan(C.cohen_kappa([True, True], [True, True]))  # both constant: undefined
    lo, hi = C.bootstrap_kappa(["y", "n"] * 20, ["y", "n"] * 19 + ["n", "y"], n_boot=200)
    assert lo <= hi and lo > 0.5


def test_agreement():
    assert C.agreement([1, 2, 3, 4], [1, 2, 0, 4]) == pytest.approx(0.75)
    assert math.isnan(C.agreement([], []))


def test_join_on_audit_id():
    lab = [{"audit_id": "a2", "rubric": "fact"}, {"audit_id": "a1", "rubric": "fact"}, {"audit_id": "zz", "rubric": "fact"}]
    key = [{"audit_id": "a1", "model": "m1"}, {"audit_id": "a2", "model": "m2"}, {"audit_id": "a3", "model": "m3"}]
    joined, missing, extra = C.join_rows(lab, key)
    assert [r["h"]["audit_id"] for r in joined] == ["a2", "a1"]  # labelled-file order
    assert [r["k"]["model"] for r in joined] == ["m2", "m1"]
    assert missing == ["zz"] and extra == ["a3"]
    with pytest.raises(ValueError):
        C.join_rows(lab + [{"audit_id": "a1"}], key)


def test_rogan_gladen():
    # true prevalence 0.2, sens 0.9, spec 0.8 -> observed 0.9*0.2 + 0.2*0.8 = 0.34
    assert C.rogan_gladen(0.34, 0.9, 0.8) == pytest.approx(0.2)
    assert C.rogan_gladen(0.0, 0.9, 0.8) == 0.0  # clipped (would be negative)
    assert C.rogan_gladen(1.0, 0.9, 0.8) == 1.0  # clipped
    assert math.isnan(C.rogan_gladen(0.3, 0.5, 0.5))  # no better than chance
    assert math.isnan(C.rogan_gladen(0.3, float("nan"), 0.9))


def test_misclass_correct_recovers_truth():
    M = np.array([[0.9, 0.1, 0.0], [0.05, 0.9, 0.05], [0.0, 0.2, 0.8]])
    p_true = np.array([0.5, 0.3, 0.2])
    assert C.misclass_correct(p_true @ M, M) == pytest.approx(p_true)
    assert C.misclass_correct([0.3, 0.3, 0.4], np.ones((3, 3)) / 3)[0] != C.misclass_correct([0.3, 0.3, 0.4], np.ones((3, 3)) / 3)[0]  # nan: singular


def test_unsure_and_missing_handling():
    assert C.norm_bool("unsure") == "unsure" and C.norm_bool(None) is None and C.norm_bool(True) is True
    assert C.norm_bool("No") is False and C.norm_bool("") is None
    assert C.norm_cat("Away", ["toward", "away", "neutral"]) == "away"
    assert C.norm_cat("garbage", ["toward", "away", "neutral"]) is None
    key = lambda i, v: {"audit_id": i, "judge": {"ok": True, "coherent": v}}
    lab = [{"audit_id": "a", "rubric": "fact", "h_coherent": True}, {"audit_id": "b", "rubric": "fact", "h_coherent": "unsure"},
           {"audit_id": "c", "rubric": "fact", "h_coherent": None}, {"audit_id": "d", "rubric": "fact", "h_coherent": False}]
    kk = [key("a", True), key("b", True), key("c", True), key("d", True)]
    kk.append({"audit_id": "e", "judge": {"ok": False}})
    lab.append({"audit_id": "e", "rubric": "fact", "h_coherent": True})
    joined, _, _ = C.join_rows(lab, kk)
    pairs, c = C.field_pairs(joined, "fact", "coherent")
    assert [(p[0], p[1]) for p in pairs] == [(True, True), (False, True)]
    assert c["n_unsure"] == 1 and c["n_missing"] == 1 and c["n_judge_unparsed"] == 1 and c["n_labelled"] == 2


def test_three_valued_derived_labels():
    assert C.tri_and(True, True) is True
    assert C.tri_and(False, "unsure") is False  # a definite no settles it
    assert C.tri_and(True, "unsure") is None
    assert C.lean_class("toward", True) == 0 and C.lean_class("away", True) == 2
    assert C.lean_class("toward", False) == 1  # incoherent: zero whatever the direction
    assert C.lean_class("unsure", False) == 1 and C.lean_class("unsure", True) is None
    assert C.lean_class("neutral", "unsure") == 1 and C.lean_class("toward", "unsure") is None


def test_condition_family():
    assert [C.cond_family(c) for c in ("nomem", "ctx", "mem@2", "rand@2", "gate_on@2", "centroid@1", "without@0.5")] == \
        ["base", "base", "mem", "control", "control", "mechanism", "mechanism"]


def test_report_end_to_end_without_run_dir(tmp_path):
    lab, key = [], []
    for i in range(12):
        lab.append({"audit_id": f"a{i}", "rubric": "fact", "h_uses_fact": i % 2 == 0, "h_coherent": True,
                    "h_self_claim": False, "h_wrong_value": False, "h_notes": ""})
        key.append({"audit_id": f"a{i}", "model": "m", "cond": "mem@1", "hit": True,
                    "judge": {"ok": True, "uses_fact": i % 3 == 0, "coherent": True, "self_claim": False, "wrong_value": False}})
    rep = C.build_report(lab, key, None, tmp_path / "nonexistent", 50, 7)
    assert "uses_fact" in rep and "Score outputs not found" in rep
